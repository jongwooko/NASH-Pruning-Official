import pdb

from re import L
from black import main
import torch
import math
import numpy as np
import torch.nn.functional as F
from torch.nn.modules import Module
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from transformers.utils import logging

limit_a, limit_b, epsilon = -.1, 1.1, 1e-6
logger = logging.get_logger(__name__)

class L0Module(Module):
    def __init__(self,
                 config, 
                 droprate_init=0.5,
                 temperature=2./3.,
                 lagrangian_warmup=0,
                 start_sparsity=0.0,
                 target_sparsity=0.0,
                 pruning_type="structured_heads+structured_mlp+hidden+layer",
                 magical_number=0.8, # from Wang et al. 2020
                 enc_dec=True,
                 encdec_pruning_type="nash"
                 ):
        
        super(L0Module, self).__init__()
        self.config = config
        self.all_types = ["hidden_z", "intermediate_z", "mlp_z", "head_layer_z", "head_z", "dec_intermediate_z", \
            "dec_mlp_z", "dec_self_head_layer_z", "dec_cross_head_layer_z", "dec_self_head_z", "dec_cross_head_z"]
        self.pruning_type = pruning_type
        self.encdec_pruning_type = encdec_pruning_type
        assert self.encdec_pruning_type in ["cofi", "nash"]

        self.hidden_size = config.d_model # config.hidden_size
        self.intermediate_size = config.d_ff # intermediate_size 
        self.num_attention_heads =config.num_heads # config.num_attention_heads
        self.mlp_num_per_layer = 1
        self.dim_per_head = self.hidden_size // self.num_attention_heads 
        self.num_hidden_layers = config.num_layers # config.num_hidden_layers 
        self.num_decoder_layers = config.num_decoder_layers if hasattr(config, 'num_decoder_layers') else config.num_layers # num_dec_layer != num_enc_layer case
        self.vocab_size = config.vocab_size

        self.params_per_head_layer = self.hidden_size * self.hidden_size * 4 # + self.hidden_size * 4
        self.params_per_head =  self.params_per_head_layer // self.num_attention_heads
        
        self.num_mlp =3 if config.feed_forward_proj == "gated-gelu" else 2
        self.params_per_mlp_layer = self.hidden_size * self.intermediate_size * self.num_mlp
        self.params_per_intermediate_dim = self.params_per_mlp_layer // self.intermediate_size

        # we ignore the parameters in normalization layers (it takes a very small amount)
        self.full_model_size = (self.params_per_head_layer + self.params_per_mlp_layer) * self.num_hidden_layers + ((self.params_per_head_layer * 2 + self.params_per_mlp_layer) * self.num_decoder_layers)
        self.prunable_model_size = 0 
        self.prunable_encoder_size = 0
        self.prunable_decoder_size = 0

        self.temperature = temperature
        self.droprate_init = droprate_init if droprate_init != 0. else 0.5
        
        self.types = []
        self.z_logas = {}
        self.parameters_per_dim = {}
        self.sizes = {}
        self.shapes = {}

        self.hidden_loga = None
        self.hidden_type = None
        self.enc_dec = enc_dec
        
        self.auto_select = self.config.auto_select if hasattr(self.config, "auto_select") else None
        
        types = self.pruning_type.split("+")
        
        for type in types:
            if type != "layer":
                self.initialize_one_module(type, enc_dec=self.enc_dec)
        if "layer" in types:
            self.initialize_one_module("layer", enc_dec=self.enc_dec)
            
        self.magical_number = magical_number

        self.lambda_1 = torch.nn.Parameter(torch.tensor(0.0))
        self.lambda_2 = torch.nn.Parameter(torch.tensor(0.0))

        self.lagrangian_warmup = lagrangian_warmup
        self.start_sparsity = start_sparsity
        self.target_sparsity = target_sparsity

        logger.info("********** Initializing L0 Module **********") 
        for type in self.types:
            logger.info(f"***** {type} *****")
            logger.info(f"z.shape", self.z_logas[type].shape)
            logger.info(f"size", self.sizes[type])
        logger.info(f"prunable model size: {self.prunable_model_size}")
        logger.info(f"prunable encoder size: {self.prunable_encoder_size}")
        logger.info(f"prunable decoder size: {self.prunable_decoder_size}")

    def set_lagrangian_warmup_steps(self, lagrangian_warmup):
        self.lagrangian_warmup = lagrangian_warmup

    def initialize_one_module(self, module_name, enc_dec=True):
        if module_name == "structured_mlp":
            self.initialize_structured_mlp()
        elif module_name == "structured_heads":
            self.initialize_structured_head()
        elif module_name == "hidden":
            self.initialize_hidden()
        elif module_name == "layer":
            self.initialize_whole_mlp()
            self.initialized_layer_structured_heads()
        
        if enc_dec:
            if module_name == "structured_mlp":
                self.initialize_structured_mlp(dec=True)
            elif module_name == "structured_heads":
                self.initialize_structured_head(dec=True)
            elif module_name == "hidden":
                self.initialize_hidden(dec=True)
            elif module_name == "layer":
                self.initialize_whole_mlp(dec=True)
                self.initialized_layer_structured_heads(dec=True)
            
    def add_one_module(self, z_loga, type, parameter_per_dim, size, shape): #! init the z_logas
        self.types.append(type)
        self.z_logas[type] = z_loga
        self.parameters_per_dim[type] = parameter_per_dim
        self.sizes[type] = size
        self.shapes[type] = shape

    def initialize_parameters(self, size, num_layer=None):
        if num_layer is not None:
            return Parameter(torch.Tensor(num_layer, size))
        else:
            return Parameter(torch.Tensor(size))

    def initialize_hidden(self, dec=False):
        self.hidden_loga = self.initialize_parameters(self.hidden_size)
        self.add_one_module(self.hidden_loga, type="hidden", 
                            parameter_per_dim=self.hidden_size * 4 * 3 + self.intermediate_size * 2 * 2,
                            size=self.hidden_size, shape=[self.hidden_size])
        self.reset_loga(self.hidden_loga, mean=10)
        
        logger.info(f"Initialized hidden loga! Prunable_model_size = {self.prunable_model_size}")
        logger.info(f"Initialized hidden loga! Prunable_encoder_size = {self.prunable_encoder_size}")
        logger.info(f"Initialized hidden loga! Prunable_decoder_size = {self.prunable_decoder_size}")

    def initialize_structured_head(self, add_prunable_model_size=True, dec=False):
        if not dec:
            self.head_loga = self.initialize_parameters(self.num_attention_heads, self.num_hidden_layers)
            self.reset_loga(self.head_loga, mean=10)
            self.add_one_module(self.head_loga, type="head", 
                                parameter_per_dim=self.params_per_head, size=self.num_attention_heads,
                                shape=[self.num_hidden_layers, 1, self.num_attention_heads, 1, 1])
            
            if add_prunable_model_size:
                self.prunable_model_size += self.params_per_head * self.num_hidden_layers * self.num_attention_heads
                self.prunable_encoder_size += self.params_per_head * self.num_hidden_layers * self.num_attention_heads
                
        else:
            self.dec_head_loga1 = self.initialize_parameters(self.num_attention_heads, self.num_decoder_layers)
            self.dec_head_loga2 = self.initialize_parameters(self.num_attention_heads, self.num_decoder_layers)
            self.reset_loga(self.dec_head_loga1, mean=10)
            self.reset_loga(self.dec_head_loga2, mean=10)
            self.add_one_module(self.dec_head_loga1, type="dec_self_head", 
                                parameter_per_dim=self.params_per_head, size=self.num_attention_heads,
                                shape=[self.num_decoder_layers, 1, self.num_attention_heads, 1, 1])
            self.add_one_module(self.dec_head_loga2, type="dec_cross_head", 
                                parameter_per_dim=self.params_per_head, size=self.num_attention_heads,
                                shape=[self.num_decoder_layers, 1, self.num_attention_heads, 1, 1])
        
            if add_prunable_model_size:
                self.prunable_model_size += self.params_per_head * self.num_decoder_layers * self.num_attention_heads * 2
                self.prunable_decoder_size += self.params_per_head * self.num_decoder_layers * self.num_attention_heads * 2
                
        logger.info(f"Initialized structured heads! Prunable_model_size = {self.prunable_model_size}")
        if not dec:
            logger.info(f"Initialized structured heads! Prunable_encoder_size = {self.prunable_encoder_size}")
        else:
            logger.info(f"Initialized structured heads! Prunable_decoder_size = {self.prunable_decoder_size}")
        
    def initialized_layer_structured_heads(self, dec=False):
        if not dec:
            n_layer = self.num_hidden_layers
            self.headlayer_loga = self.initialize_parameters(n_layer)
            self.reset_loga(self.headlayer_loga, mean=10)
            self.add_one_module(self.headlayer_loga, type="head_layer", 
                                parameter_per_dim=self.params_per_head * self.num_attention_heads, size=1,
                                shape=[n_layer])
        else:
            n_layer =self.num_decoder_layers
            self.dec_headlayer_loga1 = self.initialize_parameters(n_layer)
            self.dec_headlayer_loga2 = self.initialize_parameters(n_layer)
            self.reset_loga(self.dec_headlayer_loga1, mean=10)
            self.reset_loga(self.dec_headlayer_loga2, mean=10)
            self.add_one_module(self.dec_headlayer_loga1, type="dec_self_head_layer", 
                                parameter_per_dim=self.params_per_head * self.num_attention_heads, size=1,
                                shape=[n_layer])
            self.add_one_module(self.dec_headlayer_loga2, type="dec_cross_head_layer", 
                                parameter_per_dim=self.params_per_head * self.num_attention_heads, size=1,
                                shape=[n_layer])
        
        logger.info(f"Initialized layerwise structured heads! Prunable_model_size = {self.prunable_model_size}")
        if not dec:
            logger.info(f"Initialized layerwise structured heads! Prunable_encoder_size = {self.prunable_encoder_size}")
        else:
            logger.info(f"Initialized layerwise structured heads! Prunable_decoder_size = {self.prunable_decoder_size}")
        
        
    def initialize_structured_mlp(self, dec=False):
        if not dec:
            self.int_loga = self.initialize_parameters(self.intermediate_size, self.num_hidden_layers)
            self.add_one_module(self.int_loga, type="intermediate", 
                                parameter_per_dim=self.params_per_intermediate_dim, size=self.intermediate_size,
                                shape=[self.num_hidden_layers, 1, 1, self.intermediate_size])
            self.reset_loga(self.int_loga)
            self.prunable_model_size += self.params_per_mlp_layer * self.num_hidden_layers
        else:
            self.dec_int_loga = self.initialize_parameters(self.intermediate_size, self.num_decoder_layers)
            self.add_one_module(self.dec_int_loga, type="dec_intermediate", 
                                parameter_per_dim=self.params_per_intermediate_dim, size=self.intermediate_size,
                                shape=[self.num_decoder_layers, 1, 1, self.intermediate_size])
            self.reset_loga(self.dec_int_loga)
            self.prunable_model_size += self.params_per_mlp_layer * self.num_decoder_layers
            
        logger.info(f"Initialized structured mlp! Prunable_model_size = {self.prunable_model_size}")
        if not dec:
            self.prunable_encoder_size += self.params_per_mlp_layer * self.num_hidden_layers
            logger.info(f"Initialized structured mlp! Prunable_encoder_size = {self.prunable_encoder_size}")
        else:
            self.prunable_decoder_size += self.params_per_mlp_layer * self.num_decoder_layers
            logger.info(f"Initialized structured mlp! Prunable_decoder_size = {self.prunable_decoder_size}")


    def initialize_whole_mlp(self, dec=False):
        if not dec:
            n_layer = self.num_hidden_layers
            self.intlayer_loga = self.initialize_parameters(n_layer)
            self.add_one_module(self.intlayer_loga, type="mlp", 
                                parameter_per_dim=self.params_per_mlp_layer, size=self.mlp_num_per_layer,
                                shape=[n_layer])
            self.reset_loga(self.intlayer_loga, mean=10)
        else:
            n_layer = self.num_decoder_layers
            self.dec_intlayer_loga = self.initialize_parameters(n_layer)
            self.add_one_module(self.dec_intlayer_loga, type="dec_mlp", 
                                parameter_per_dim=self.params_per_mlp_layer, size=self.mlp_num_per_layer,
                                shape=[n_layer])
            self.reset_loga(self.dec_intlayer_loga, mean=10)
            
        logger.info(f"Initialized whole mlps! Prunable_model_size = {self.prunable_model_size}")
        if not dec:
            logger.info(f"Initialized whole mlps! Prunable_encoder_size = {self.prunable_encoder_size}")
        else:
            logger.info(f"Initialized whole mlps! Prunable_decoder_size = {self.prunable_decoder_size}")


    def reset_loga(self, tensor, mean=None):
        if mean is None:
            mean = math.log(1 - self.droprate_init) - math.log(self.droprate_init)
        tensor.data.normal_(mean, 1e-2)

    def reset_qz_logas(self):
        for key in self.z_logas:
            if key in ["head_layer", "mlp", "head", "dec_mlp", "dec_self_head", "dec_cross_head", "dec_self_head_layer", "dec_cross_head_layer"]:
                self.reset_loga(self.z_logas[key], 10)
            else:
                self.reset_loga(self.z_logas[key])

    def constrain_parameters(self):
        def _constrain(tensor):
            tensor.data.clamp_(min=math.log(1e-2), max=math.log(1e2))
        for key in self.z_logas:
            _constrain(self.z_logas[key])

    def cdf_qz(self, x, loga):
        """Implements the CDF of the 'stretched' concrete distribution"""
        xn = (x - limit_a) / (limit_b - limit_a)
        logits = math.log(xn) - math.log(1 - xn)
        return torch.sigmoid(logits * self.temperature - loga).clamp(min=epsilon, max=1 - epsilon)

    def quantile_concrete(self, x, loga):
        y = torch.sigmoid((torch.log(x) - torch.log(1 - x) + loga) / self.temperature)
        return y * (limit_b - limit_a) + limit_a

    def get_num_parameters_for_one(self, loga, parameter_size):
        return torch.sum(1 - self.cdf_qz(0, loga)) * parameter_size

    def transform_scores_for_head(self):
        assert "head" in self.types

        if "head_layer" in self.types:
            all_head_score = 1 - self.cdf_qz(0, self.headlayer_loga)
        else:
            all_head_score = None
        head_score = 1 - self.cdf_qz(0, self.head_loga) # 12 * 12
       
        if all_head_score is not None:
            all_head_score = all_head_score.view(-1, 1, 1) # 12 * 1 * 1
        head_score = head_score.unsqueeze(-1)   # 12 * 12 * 1


        if "dec_self_head_layer" in self.types:
            all_dec_self_head_score = 1 - self.cdf_qz(0, self.dec_headlayer_loga1)
            all_dec_cross_head_score = 1 - self.cdf_qz(0, self.dec_headlayer_loga2)
        else:
            all_dec_self_head_score = None
            all_dec_cross_head_score = None
        dec_head_score1 = 1 - self.cdf_qz(0, self.dec_head_loga1) # 12 * 12
        dec_head_score2 = 1 - self.cdf_qz(0, self.dec_head_loga2) # 12 * 12
       
        if all_dec_self_head_score is not None:
            all_dec_self_head_score = all_dec_self_head_score.view(-1, 1, 1) # 12 * 1 * 1
            all_dec_cross_head_score = all_dec_cross_head_score.view(-1, 1, 1)
        dec_head_score1 = dec_head_score1.unsqueeze(-1)   # 12 * 12 * 1
        dec_head_score2 = dec_head_score2.unsqueeze(-1)   # 12 * 12 * 1
       
        return all_head_score, head_score, all_dec_self_head_score, all_dec_cross_head_score, dec_head_score1, dec_head_score2

    def get_num_parameters_for_mlp(self):
        intlayer_score = 1 - self.cdf_qz(0, self.intlayer_loga) # 12
        int_score = 1 - self.cdf_qz(0, self.int_loga) # 12 * 3072
        intlayer_score = intlayer_score.unsqueeze(-1)

        num_parameters = torch.sum(intlayer_score * int_score) * self.parameters_per_dim["intermediate"]
        return num_parameters

    def get_num_parameters_and_constraint_for_hidden(self): #! calculate the current parsity
        num_enc_parameters, num_dec_parameters = 0, 0
       
        # 12 * 1 * 1
        # 12 * 12 * 1
        all_head_score, head_score, all_dec_self_head_score, all_dec_cross_head_score, dec_head_score1, dec_head_score2 = self.transform_scores_for_head()
        
        if self.encdec_pruning_type =="nash": 
            ## NASH-manual: layer pruning option is not contained. 
            dec_head_score1 = torch.ones_like(dec_head_score1) # dec_self_head_z individual decoder head would not be pruned
            dec_head_score2 = torch.ones_like(dec_head_score2) # dec_cross_head_z individual decoder head would not be pruned
            self.reset_loga(self.dec_int_loga, mean=10) # dec col,row not be pruned
            
            if self.auto_select: 
                # NASH-auto_select: layer pruning option is contained -> avoid the enc_layers are pruned
                # enc --> only fine-grained pruning is applied (head, mlp, hidden) # dec --> only coarse-grained pruning is applied 
                all_head_score = torch.ones_like(all_head_score) # enc_self_layer_z encoder att layer would not be pruned
                self.reset_loga(self.intlayer_loga, mean=10) # enc_mlp_layer_z < encdoer mlp layer would not be pruned        

           
        hidden_score = 1 - self.cdf_qz(0, self.hidden_loga) # 768

        if all_head_score is not None:
            head_score = (all_head_score * head_score).reshape(-1)
        else:
            head_score = head_score.reshape(-1)
        num_enc_parameters += \
            torch.sum(torch.outer(hidden_score, head_score)) * self.parameters_per_dim["head"] / self.hidden_size
        
        try:
            intlayer_score = 1 - self.cdf_qz(0, self.intlayer_loga)  # 12
        except AttributeError:
            intlayer_score = torch.ones(self.num_hidden_layers) # drop the layer <-- resolve
        int_score = 1 - self.cdf_qz(0, self.int_loga)  # 12 * 3072
        intlayer_score = intlayer_score.unsqueeze(-1)

        int_score = (intlayer_score * int_score).reshape(-1)
        num_enc_parameters += torch.sum(torch.outer(hidden_score, int_score)) * 2

        if all_dec_self_head_score is not None:
            dec_head_score1 = (all_dec_self_head_score * dec_head_score1).reshape(-1)
            dec_head_score2 = (all_dec_cross_head_score * dec_head_score2).reshape(-1)
        else:
            dec_head_score1 = dec_head_score1.reshape(-1)
            dec_head_score2 = dec_head_score2.reshape(-1)
        num_dec_parameters += \
            torch.sum(torch.outer(hidden_score, dec_head_score1)) * self.parameters_per_dim["head"] / self.hidden_size
        num_dec_parameters += \
            torch.sum(torch.outer(hidden_score, dec_head_score2)) * self.parameters_per_dim["head"] / self.hidden_size

        try:
            dec_intlayer_score = 1 - self.cdf_qz(0, self.dec_intlayer_loga)  # 12
        except AttributeError:
            dec_intlayer_score = torch.ones(self.num_decoder_layers)
        dec_int_score = 1 - self.cdf_qz(0, self.dec_int_loga)  # 12 * 3072
        dec_intlayer_score = dec_intlayer_score.unsqueeze(-1)

        dec_int_score = (dec_intlayer_score * dec_int_score).reshape(-1)
        num_dec_parameters += torch.sum(torch.outer(hidden_score, dec_int_score)) * 2
        
        num_parameters = num_enc_parameters + num_dec_parameters
        return num_parameters, num_enc_parameters, num_dec_parameters

    def get_num_parameters_and_constraint(self):
        num_enc_parameters, num_dec_parameters = 0, 0

        all_head_score, head_score, all_dec_self_head_score, all_dec_cross_head_score, dec_head_score1, dec_head_score2 = self.transform_scores_for_head()
        
        if self.encdec_pruning_type =="nash": 
            ## NASH-manual: layer pruning option is not contained. 
            dec_head_score1 = torch.ones_like(dec_head_score1) # dec_self_head_z individual decoder head would not be pruned
            dec_head_score2 = torch.ones_like(dec_head_score2) # dec_cross_head_z individual decoder head would not be pruned
            self.reset_loga(self.dec_int_loga, mean=10) # dec col,row not be pruned
            
            if self.auto_select: 
                # NASH-auto_select: layer pruning option is contained -> avoid the enc_layers are pruned
                # enc --> only fine-grained pruning is applied (head, mlp, hidden) # dec --> only coarse-grained pruning is applied 
                all_head_score = torch.ones_like(all_head_score) # enc_self_layer_z encoder att layer would not be pruned
                self.reset_loga(self.intlayer_loga, mean=10) # enc_mlp_layer_z < encdoer mlp layer would not be pruned             
        
        if all_head_score is not None:
#             head_score = (all_head_score * head_score).reshape(-1)
            head_score = head_score * all_head_score
        else:
            head_score = head_score.reshape(-1)
        
        num_enc_parameters += torch.sum(head_score) * self.parameters_per_dim["head"]

        try:
            intlayer_score = 1 - self.cdf_qz(0, self.intlayer_loga)  # 12
        except AttributeError:
            intlayer_score = torch.ones(self.num_hidden_layers) # drop the layer <-- resolve
        int_score = 1 - self.cdf_qz(0, self.int_loga)  # 12 * 3072
        intlayer_score = intlayer_score.unsqueeze(-1)

        int_score = int_score * intlayer_score
        num_enc_parameters += torch.sum(int_score) * self.parameters_per_dim["intermediate"]
        

        
        
        if all_dec_self_head_score is not None:
            dec_head_score1 = all_dec_self_head_score * dec_head_score1
            dec_head_score2 = all_dec_cross_head_score * dec_head_score2
        else:
            dec_head_score1 = dec_head_score1.reshape(-1)
            dec_head_score2 = dec_head_score2.reshape(-1)

        num_dec_parameters += torch.sum(dec_head_score1) * self.parameters_per_dim["head"]
        num_dec_parameters += torch.sum(dec_head_score2) * self.parameters_per_dim["head"]

        try:
            dec_intlayer_score = 1 - self.cdf_qz(0, self.dec_intlayer_loga)  # 12
        except AttributeError:
            dec_intlayer_score = torch.ones(self.num_decoder_layers)
            
        dec_int_score = 1 - self.cdf_qz(0, self.dec_int_loga)  # 12 * 3072
        dec_intlayer_score = dec_intlayer_score.unsqueeze(-1)

        dec_int_score = dec_int_score * dec_intlayer_score
        num_dec_parameters += torch.sum(dec_int_score) * self.parameters_per_dim["intermediate"]

        num_parameters = num_enc_parameters + num_dec_parameters
        return num_parameters, num_enc_parameters, num_dec_parameters


    def get_target_sparsity(self, pruned_steps):
        target_sparsity = (self.target_sparsity - self.start_sparsity) * min(1, pruned_steps / self.lagrangian_warmup) + self.start_sparsity
        return target_sparsity


    def lagrangian_regularization(self, pruned_steps):
        target_sparsity = self.target_sparsity
        if "hidden" in self.types:
            expected_size, expected_enc_size, expected_dec_size = self.get_num_parameters_and_constraint_for_hidden() #! calculate \bar s
        else:
            expected_size, expected_enc_size, expected_dec_size = self.get_num_parameters_and_constraint() #! calculate \bar s
            
        expected_sparsity = 1 - expected_size / self.prunable_model_size
        expected_enc_sparsity = 1 - expected_enc_size / self.prunable_encoder_size
        expected_dec_sparsity = 1 - expected_dec_size / self.prunable_decoder_size
        
        if self.lagrangian_warmup > 0:
            target_sparsity = self.get_target_sparsity(pruned_steps)
            
            if hasattr(self.config, "selected_layer") and self.config.selected_layer:
                target_enc_sparsity = target_sparsity
                target_dec_sparsity = 0.0
                target_sparsity = target_sparsity * (self.prunable_encoder_size / self.prunable_model_size)
                
            elif hasattr(self.config, "auto_select") and self.config.auto_select:
                target_enc_sparsity = target_sparsity
                target_dec_sparsity = (self.config.auto_select - self.start_sparsity) * min(1, pruned_steps / self.lagrangian_warmup) + self.start_sparsity
        
        if self.encdec_pruning_type == "cofi":
            # Naive CoFi
            lagrangian_loss = ( #! see appendix
                    self.lambda_1 * (expected_sparsity - target_sparsity)
                    + self.lambda_2 * (expected_sparsity - target_sparsity) ** 2 #! where is the lambda 1 and lambda 2 from
            )
        
        elif self.encdec_pruning_type == "nash":
            # Different target sparsity on encoder and decoder networks
            lagrangian_loss = (
                self.lambda_1 * (expected_enc_sparsity - target_enc_sparsity) 
                + self.lambda_1 * (expected_dec_sparsity - target_dec_sparsity)
                + self.lambda_2 * (expected_enc_sparsity - target_enc_sparsity) ** 2
                + self.lambda_2 * (expected_dec_sparsity - target_dec_sparsity) ** 2
            )
            
        else:
            raise NotImplementedError
            
        return lagrangian_loss, expected_sparsity, target_sparsity
        
        
    def get_eps(self, size):
        """Uniform random numbers for the concrete distribution"""
        eps = torch.FloatTensor(size).uniform_(epsilon, 1-epsilon)
        eps = Variable(eps)
        return eps

    # during training
    def _sample_z(self, loga):
        eps = self.get_eps(torch.FloatTensor(*loga.shape)).to(loga.device)
        z = self.quantile_concrete(eps, loga)
        z = F.hardtanh(z, min_val=0, max_val=1)
        return z

    # during inference
    def _deterministic_z(self, size, loga):
        # Following https://github.com/asappresearch/flop/blob/e80e47155de83abbe7d90190e00d30bfb85c18d5/flop/hardconcrete.py#L8 line 103
        expected_num_nonzeros = torch.sum(1 - self.cdf_qz(0, loga))
        expected_num_zeros = size - expected_num_nonzeros.item()
        try:
            num_zeros = round(expected_num_zeros)
        except:
            pdb.set_trace()
        soft_mask = torch.sigmoid(loga / self.temperature * self.magical_number)
        if num_zeros > 0:
            if soft_mask.ndim == 0:
                soft_mask = torch.tensor(0).to(loga.device)
            else:
                _, indices = torch.topk(soft_mask, k=num_zeros, largest=False)
                soft_mask[indices] = 0.
        return soft_mask

    def get_z_from_zs(self, zs):
        numpified_zs = {} 
        for type in self.all_types:
            name = type[:-2]
            if name in self.types:
                z = zs.get(type, np.ones(self.shapes[name]))
                if torch.is_tensor(z): 
                    new_z = z.squeeze().detach().cpu().numpy() > 0
                numpified_zs[name] = new_z
        return numpified_zs

    def calculate_model_size(self, zs):
        numpified_zs = self.get_z_from_zs(zs)
        hidden_z = numpified_zs["hidden"]
        intermediate_z = numpified_zs["intermediate"]
        mlp_z = numpified_zs["mlp"].reshape(-1, 1) if "mlp" in numpified_zs.keys() \
            else np.ones(self.num_hidden_layers).reshape(-1, 1)
        head_z = numpified_zs["head"]
        head_layer_z = numpified_zs["head_layer"].reshape(-1, 1) if "head_layer" in numpified_zs.keys() \
            else np.ones(self.num_hidden_layers).reshape(-1, 1)

        # dec_hidden_z = numpified_zs["dec_hidden"]
        dec_intermediate_z = numpified_zs["dec_intermediate"]
        dec_mlp_z = numpified_zs["dec_mlp"].reshape(-1, 1) if "mlp" in numpified_zs.keys() \
            else np.ones(self.num_decoder_layers).reshape(-1, 1)
        dec_self_head_z = numpified_zs["dec_self_head"]
        dec_cross_head_z = numpified_zs["dec_cross_head"]
        dec_self_head_layer_z = numpified_zs["dec_self_head_layer"].reshape(-1, 1) if "dec_self_head_layer" in numpified_zs.keys() \
            else np.ones(self.num_decoder_layers).reshape(-1, 1)
        dec_cross_head_layer_z = numpified_zs["dec_cross_head_layer"].reshape(-1, 1) if "dec_cross_head_layer" in numpified_zs.keys() \
            else np.ones(self.num_decoder_layers).reshape(-1, 1)

        remaining_hidden_dims = hidden_z.sum().item()
        remaining_intermediate_nums = intermediate_z.reshape(self.num_hidden_layers, self.intermediate_size).sum(-1).tolist()
        remaining_head_nums = head_z.reshape(self.num_hidden_layers, self.num_attention_heads).sum(-1).tolist()

        remaining_dec_intermediate_nums = dec_intermediate_z.reshape(self.num_decoder_layers, self.intermediate_size).sum(-1).tolist() # 
        remaining_dec_self_head_nums = dec_self_head_z.reshape(self.num_decoder_layers, self.num_attention_heads).sum(-1).tolist() #
        remaining_dec_cross_head_nums = dec_cross_head_z.reshape(self.num_decoder_layers, self.num_attention_heads).sum(-1).tolist() #

        head_nums = np.outer((head_z * head_layer_z).reshape(-1), hidden_z).sum().item()
        dec_self_head_nums = np.outer((dec_self_head_z * dec_self_head_layer_z).reshape(-1), hidden_z).sum().item() # 
        dec_cross_head_nums = np.outer((dec_cross_head_z * dec_cross_head_layer_z).reshape(-1), hidden_z).sum().item() #

        intermediate_nums = np.outer((intermediate_z * mlp_z).reshape(-1), hidden_z).sum().item()
        dec_intermediate_nums = np.outer((dec_intermediate_z * dec_mlp_z).reshape(-1), hidden_z).sum().item() #

        remaining_model_size = (head_nums + dec_self_head_nums + dec_cross_head_nums) * self.dim_per_head * 4 \
            + (intermediate_nums + dec_intermediate_nums) * 2
        remaining_encoder_size = head_nums * self.dim_per_head * 4 + intermediate_nums * 2
        remaining_decoder_size = (dec_self_head_nums + dec_cross_head_nums) * self.dim_per_head * 4 + dec_intermediate_nums * 2
        
        pruned_model_size = self.prunable_model_size - remaining_model_size
        pruned_encoder_size = self.prunable_encoder_size - remaining_encoder_size
        pruned_decoder_size = self.prunable_decoder_size - remaining_decoder_size

        results = {}
        # Not multiplied with each other
        results["head_layers"] = head_layer_z.reshape(-1).astype(int).tolist()
        results["mlp_layers"] = mlp_z.reshape(-1).astype(int).tolist()
        results["hidden_dims"] = remaining_hidden_dims
        results["intermediate_dims"] = remaining_intermediate_nums
        results["head_nums"] = remaining_head_nums

        results["dec_self_head_layers"] = dec_self_head_layer_z.reshape(-1).astype(int).tolist()
        results["dec_cross_head_layers"] = dec_cross_head_layer_z.reshape(-1).astype(int).tolist()
        results["dec_mlp_layers"] = dec_mlp_z.reshape(-1).astype(int).tolist()
        results["dec_intermediate_dims"] = remaining_dec_intermediate_nums
        results["dec_self_head_nums"] = remaining_dec_self_head_nums
        results["dec_cross_head_nums"] = remaining_dec_cross_head_nums

        results["pruned_params"] = pruned_model_size
        results["remaining_params"] = remaining_model_size
        results["pruned_model_sparsity"] = pruned_model_size / self.prunable_model_size
        
        results["pruned_enc_params"] = pruned_encoder_size
        results["remaining_enc_params"] = remaining_encoder_size
        results["pruned_encoder_sparsity"] = pruned_encoder_size / self.prunable_encoder_size
        
        results["pruned_dec_params"] = pruned_decoder_size
        results["remaining_dec_params"] = remaining_decoder_size
        results["pruned_decoder_sparsity"] = pruned_decoder_size / self.prunable_decoder_size
        return results

    def forward(self, training=True,):
        zs = {f"{type}_z": [] for type in self.types}

        if training:
            for i, type in enumerate(self.types):
                loga = self.z_logas[type]
                z = self._sample_z(loga)
                zs[f"{type}_z"] = z.reshape(self.shapes[type])
        else:
            for i, type in enumerate(self.types):
                if not "hidden" in type: # hidden is not a per layer sample
                    loga_all_layers = self.z_logas[type]
                    for layer in range(len(loga_all_layers)):
                        loga = loga_all_layers[layer]
                        size = self.sizes[type]
                        z = self._deterministic_z(size, loga)
                        zs[f"{type}_z"].append(z.reshape(self.shapes[type][1:]))
                else:
                    z = self._deterministic_z(self.sizes[type], self.hidden_loga)
                    zs[f"{type}_z"] = z
            for type in zs:
                if not "hidden_z" in type:
                    zs[type] = torch.stack(zs[type])
        return zs
    
class L0Module_Bart(Module):
    def __init__(self,
                 config, 
                 droprate_init=0.5,
                 temperature=2./3.,
                 lagrangian_warmup=0,
                 start_sparsity=0.0,
                 target_sparsity=0.0,
                 pruning_type="structured_heads+structured_mlp+hidden+layer",
                 magical_number=0.8, # from Wang et al. 2020
                 enc_dec=True,
                 encdec_pruning_type="nash"
                 ):
        
        super(L0Module_Bart, self).__init__()
        self.config = config
        self.all_types = ["hidden_z", "intermediate_z", "mlp_z", "head_layer_z", "head_z", "dec_intermediate_z", \
            "dec_mlp_z", "dec_self_head_layer_z", "dec_cross_head_layer_z", "dec_self_head_z", "dec_cross_head_z"]
        self.pruning_type = pruning_type
        self.encdec_pruning_type = encdec_pruning_type
        assert self.encdec_pruning_type in ["cofi", "nash"]

        self.hidden_size = config.d_model # config.hidden_size
        self.enc_intermediate_size = config.encoder_ffn_dim # intermediate_size 
        self.dec_intermediate_size = config.decoder_ffn_dim
        self.num_attention_heads =config.encoder_attention_heads # config.num_attention_heads
        self.mlp_num_per_layer = 1
        self.dim_per_head = self.hidden_size // self.num_attention_heads 
        self.num_hidden_layers = config.num_hidden_layers # config.num_hidden_layers 
        self.num_decoder_layers = config.decoder_layers if hasattr(config, 'decoder_layers') else config.num_hidden_layers # num_dec_layer != num_enc_layer case
        self.vocab_size = config.vocab_size

        self.params_per_head_layer = self.hidden_size * self.hidden_size * 4 + self.hidden_size * 4
        self.params_per_head =  self.params_per_head_layer // self.num_attention_heads
        
        self.num_mlp =3 if config.activation_function == "gated-gelu" else 2
        self.params_per_enc_mlp_layer = self.hidden_size * self.enc_intermediate_size * self.num_mlp + self.enc_intermediate_size + self.hidden_size # linear prams + bias terms
        self.params_per_enc_intermediate_dim = self.params_per_enc_mlp_layer // self.enc_intermediate_size
        
        self.params_per_dec_mlp_layer = self.hidden_size * self.dec_intermediate_size * self.num_mlp + self.dec_intermediate_size + self.hidden_size # linear prams + bias terms
        self.params_per_dec_intermediate_dim = self.params_per_dec_mlp_layer // self.dec_intermediate_size

        # we ignore the parameters in normalization layers (it takes a very small amount)
        self.full_model_size = (self.params_per_head_layer + self.params_per_enc_mlp_layer) * self.num_hidden_layers + ((self.params_per_head_layer * 2 + self.params_per_dec_mlp_layer) * self.num_decoder_layers)
        self.prunable_model_size = 0 
        self.prunable_encoder_size = 0
        self.prunable_decoder_size = 0

        self.temperature = temperature
        self.droprate_init = droprate_init if droprate_init != 0. else 0.5
        
        self.types = []
        self.z_logas = {}
        self.parameters_per_dim = {}
        self.sizes = {}
        self.shapes = {}

        self.hidden_loga = None
        self.hidden_type = None
        self.enc_dec = enc_dec
        
        self.auto_select = self.config.auto_select if hasattr(self.config, "auto_select") else None
        
        types = self.pruning_type.split("+")
        
        for type in types:
            if type != "layer":
                self.initialize_one_module(type, enc_dec=self.enc_dec)
        if "layer" in types:
            self.initialize_one_module("layer", enc_dec=self.enc_dec)
            
        self.magical_number = magical_number

        self.lambda_1 = torch.nn.Parameter(torch.tensor(0.0))
        self.lambda_2 = torch.nn.Parameter(torch.tensor(0.0))

        self.lagrangian_warmup = lagrangian_warmup
        self.start_sparsity = start_sparsity
        self.target_sparsity = target_sparsity

        logger.info("********** Initializing L0 Module **********") 
        for type in self.types:
            logger.info(f"***** {type} *****")
            logger.info(f"z.shape", self.z_logas[type].shape)
            logger.info(f"size", self.sizes[type])
        logger.info(f"prunable model size: {self.prunable_model_size}")
        logger.info(f"prunable encoder size: {self.prunable_encoder_size}")
        logger.info(f"prunable decoder size: {self.prunable_decoder_size}")

    def set_lagrangian_warmup_steps(self, lagrangian_warmup):
        self.lagrangian_warmup = lagrangian_warmup

    def initialize_one_module(self, module_name, enc_dec=True):
        if module_name == "structured_mlp":
            self.initialize_structured_mlp()
        elif module_name == "structured_heads":
            self.initialize_structured_head()
        elif module_name == "hidden":
            self.initialize_hidden()
        elif module_name == "layer":
            self.initialize_whole_mlp()
            self.initialized_layer_structured_heads()
        
        if enc_dec:
            if module_name == "structured_mlp":
                self.initialize_structured_mlp(dec=True)
            elif module_name == "structured_heads":
                self.initialize_structured_head(dec=True)
            elif module_name == "hidden":
                self.initialize_hidden(dec=True)
            elif module_name == "layer":
                self.initialize_whole_mlp(dec=True)
                self.initialized_layer_structured_heads(dec=True)
            
    def add_one_module(self, z_loga, type, parameter_per_dim, size, shape): #! init the z_logas
        self.types.append(type)
        self.z_logas[type] = z_loga
        self.parameters_per_dim[type] = parameter_per_dim
        self.sizes[type] = size
        self.shapes[type] = shape

    def initialize_parameters(self, size, num_layer=None):
        if num_layer is not None:
            return Parameter(torch.Tensor(num_layer, size))
        else:
            return Parameter(torch.Tensor(size))

    def initialize_hidden(self, dec=False):
        self.hidden_loga = self.initialize_parameters(self.hidden_size)
        self.add_one_module(self.hidden_loga, type="hidden", 
                            parameter_per_dim=self.hidden_size * 4 * 3 + self.enc_intermediate_size * 2 * 2,
                            size=self.hidden_size, shape=[self.hidden_size])
        self.reset_loga(self.hidden_loga, mean=10)
        
        logger.info(f"Initialized hidden loga! Prunable_model_size = {self.prunable_model_size}")
        logger.info(f"Initialized hidden loga! Prunable_encoder_size = {self.prunable_encoder_size}")
        logger.info(f"Initialized hidden loga! Prunable_decoder_size = {self.prunable_decoder_size}")

    def initialize_structured_head(self, add_prunable_model_size=True, dec=False):
        if not dec:
            self.head_loga = self.initialize_parameters(self.num_attention_heads, self.num_hidden_layers)
            self.reset_loga(self.head_loga, mean=10)
            self.add_one_module(self.head_loga, type="head", 
                                parameter_per_dim=self.params_per_head, size=self.num_attention_heads,
                                shape=[self.num_hidden_layers, 1, self.num_attention_heads, 1, 1])
            
            if add_prunable_model_size:
                self.prunable_model_size += self.params_per_head * self.num_hidden_layers * self.num_attention_heads
                self.prunable_encoder_size += self.params_per_head * self.num_hidden_layers * self.num_attention_heads
                
        else:
            self.dec_head_loga1 = self.initialize_parameters(self.num_attention_heads, self.num_decoder_layers)
            self.dec_head_loga2 = self.initialize_parameters(self.num_attention_heads, self.num_decoder_layers)
            self.reset_loga(self.dec_head_loga1, mean=10)
            self.reset_loga(self.dec_head_loga2, mean=10)
            self.add_one_module(self.dec_head_loga1, type="dec_self_head", 
                                parameter_per_dim=self.params_per_head, size=self.num_attention_heads,
                                shape=[self.num_decoder_layers, 1, self.num_attention_heads, 1, 1])
            self.add_one_module(self.dec_head_loga2, type="dec_cross_head", 
                                parameter_per_dim=self.params_per_head, size=self.num_attention_heads,
                                shape=[self.num_decoder_layers, 1, self.num_attention_heads, 1, 1])
        
            if add_prunable_model_size:
                self.prunable_model_size += self.params_per_head * self.num_decoder_layers * self.num_attention_heads * 2
                self.prunable_decoder_size += self.params_per_head * self.num_decoder_layers * self.num_attention_heads * 2
                
        logger.info(f"Initialized structured heads! Prunable_model_size = {self.prunable_model_size}")
        if not dec:
            logger.info(f"Initialized structured heads! Prunable_encoder_size = {self.prunable_encoder_size}")
        else:
            logger.info(f"Initialized structured heads! Prunable_decoder_size = {self.prunable_decoder_size}")
        
    def initialized_layer_structured_heads(self, dec=False):
        if not dec:
            n_layer = self.num_hidden_layers
            self.headlayer_loga = self.initialize_parameters(n_layer)
            self.reset_loga(self.headlayer_loga, mean=10)
            self.add_one_module(self.headlayer_loga, type="head_layer", 
                                parameter_per_dim=self.params_per_head * self.num_attention_heads, size=1,
                                shape=[n_layer])
        else:
            n_layer =self.num_decoder_layers
            self.dec_headlayer_loga1 = self.initialize_parameters(n_layer)
            self.dec_headlayer_loga2 = self.initialize_parameters(n_layer)
            self.reset_loga(self.dec_headlayer_loga1, mean=10)
            self.reset_loga(self.dec_headlayer_loga2, mean=10)
            self.add_one_module(self.dec_headlayer_loga1, type="dec_self_head_layer", 
                                parameter_per_dim=self.params_per_head * self.num_attention_heads, size=1,
                                shape=[n_layer])
            self.add_one_module(self.dec_headlayer_loga2, type="dec_cross_head_layer", 
                                parameter_per_dim=self.params_per_head * self.num_attention_heads, size=1,
                                shape=[n_layer])
        
        logger.info(f"Initialized layerwise structured heads! Prunable_model_size = {self.prunable_model_size}")
        if not dec:
            logger.info(f"Initialized layerwise structured heads! Prunable_encoder_size = {self.prunable_encoder_size}")
        else:
            logger.info(f"Initialized layerwise structured heads! Prunable_decoder_size = {self.prunable_decoder_size}")
        
        
    def initialize_structured_mlp(self, dec=False):
        if not dec:
            self.int_loga = self.initialize_parameters(self.enc_intermediate_size, self.num_hidden_layers)
            self.add_one_module(self.int_loga, type="intermediate", 
                                parameter_per_dim=self.params_per_enc_intermediate_dim, size=self.enc_intermediate_size,
                                shape=[self.num_hidden_layers, 1, 1, self.enc_intermediate_size])
            self.reset_loga(self.int_loga)
            self.prunable_model_size += self.params_per_enc_mlp_layer * self.num_hidden_layers
        else:
            self.dec_int_loga = self.initialize_parameters(self.dec_intermediate_size, self.num_decoder_layers)
            self.add_one_module(self.dec_int_loga, type="dec_intermediate", 
                                parameter_per_dim=self.params_per_dec_intermediate_dim, size=self.dec_intermediate_size,
                                shape=[self.num_decoder_layers, 1, 1, self.dec_intermediate_size])
            self.reset_loga(self.dec_int_loga)
            self.prunable_model_size += self.params_per_dec_mlp_layer * self.num_decoder_layers
            
        logger.info(f"Initialized structured mlp! Prunable_model_size = {self.prunable_model_size}")
        if not dec:
            self.prunable_encoder_size += self.params_per_enc_mlp_layer * self.num_hidden_layers
            logger.info(f"Initialized structured mlp! Prunable_encoder_size = {self.prunable_encoder_size}")
        else:
            self.prunable_decoder_size += self.params_per_dec_mlp_layer * self.num_decoder_layers
            logger.info(f"Initialized structured mlp! Prunable_decoder_size = {self.prunable_decoder_size}")


    def initialize_whole_mlp(self, dec=False):
        if not dec:
            n_layer = self.num_hidden_layers
            self.intlayer_loga = self.initialize_parameters(n_layer)
            self.add_one_module(self.intlayer_loga, type="mlp", 
                                parameter_per_dim=self.params_per_enc_mlp_layer, size=self.mlp_num_per_layer,
                                shape=[n_layer])
            self.reset_loga(self.intlayer_loga, mean=10)
        else:
            n_layer = self.num_decoder_layers
            self.dec_intlayer_loga = self.initialize_parameters(n_layer)
            self.add_one_module(self.dec_intlayer_loga, type="dec_mlp", 
                                parameter_per_dim=self.params_per_dec_mlp_layer, size=self.mlp_num_per_layer,
                                shape=[n_layer])
            self.reset_loga(self.dec_intlayer_loga, mean=10)
            
        logger.info(f"Initialized whole mlps! Prunable_model_size = {self.prunable_model_size}")
        if not dec:
            logger.info(f"Initialized whole mlps! Prunable_encoder_size = {self.prunable_encoder_size}")
        else:
            logger.info(f"Initialized whole mlps! Prunable_decoder_size = {self.prunable_decoder_size}")


    def reset_loga(self, tensor, mean=None):
        if mean is None:
            mean = math.log(1 - self.droprate_init) - math.log(self.droprate_init)
        tensor.data.normal_(mean, 1e-2)

    def reset_qz_logas(self):
        for key in self.z_logas:
            if key in ["head_layer", "mlp", "head", "dec_mlp", "dec_self_head", "dec_cross_head", "dec_self_head_layer", "dec_cross_head_layer"]:
                self.reset_loga(self.z_logas[key], 10)
            else:
                self.reset_loga(self.z_logas[key])

    def constrain_parameters(self):
        def _constrain(tensor):
            tensor.data.clamp_(min=math.log(1e-2), max=math.log(1e2))
        for key in self.z_logas:
            _constrain(self.z_logas[key])

    def cdf_qz(self, x, loga):
        """Implements the CDF of the 'stretched' concrete distribution"""
        xn = (x - limit_a) / (limit_b - limit_a)
        logits = math.log(xn) - math.log(1 - xn)
        return torch.sigmoid(logits * self.temperature - loga).clamp(min=epsilon, max=1 - epsilon)

    def quantile_concrete(self, x, loga):
        y = torch.sigmoid((torch.log(x) - torch.log(1 - x) + loga) / self.temperature)
        return y * (limit_b - limit_a) + limit_a

    def get_num_parameters_for_one(self, loga, parameter_size):
        return torch.sum(1 - self.cdf_qz(0, loga)) * parameter_size

    def transform_scores_for_head(self):
        assert "head" in self.types

        if "head_layer" in self.types:
            all_head_score = 1 - self.cdf_qz(0, self.headlayer_loga)
        else:
            all_head_score = None
        head_score = 1 - self.cdf_qz(0, self.head_loga) # 12 * 12
       
        if all_head_score is not None:
            all_head_score = all_head_score.view(-1, 1, 1) # 12 * 1 * 1
        head_score = head_score.unsqueeze(-1)   # 12 * 12 * 1


        if "dec_self_head_layer" in self.types:
            all_dec_self_head_score = 1 - self.cdf_qz(0, self.dec_headlayer_loga1)
            all_dec_cross_head_score = 1 - self.cdf_qz(0, self.dec_headlayer_loga2)
        else:
            all_dec_self_head_score = None
            all_dec_cross_head_score = None
        dec_head_score1 = 1 - self.cdf_qz(0, self.dec_head_loga1) # 12 * 12
        dec_head_score2 = 1 - self.cdf_qz(0, self.dec_head_loga2) # 12 * 12
       
        if all_dec_self_head_score is not None:
            all_dec_self_head_score = all_dec_self_head_score.view(-1, 1, 1) # 12 * 1 * 1
            all_dec_cross_head_score = all_dec_cross_head_score.view(-1, 1, 1)
        dec_head_score1 = dec_head_score1.unsqueeze(-1)   # 12 * 12 * 1
        dec_head_score2 = dec_head_score2.unsqueeze(-1)   # 12 * 12 * 1
       
        return all_head_score, head_score, all_dec_self_head_score, all_dec_cross_head_score, dec_head_score1, dec_head_score2

    def get_num_parameters_for_mlp(self):
        intlayer_score = 1 - self.cdf_qz(0, self.intlayer_loga) # 12
        int_score = 1 - self.cdf_qz(0, self.int_loga) # 12 * 3072
        intlayer_score = intlayer_score.unsqueeze(-1)

        num_parameters = torch.sum(intlayer_score * int_score) * self.parameters_per_dim["intermediate"]
        return num_parameters

    def get_num_parameters_and_constraint_for_hidden(self): #! calculate the current parsity
        num_enc_parameters, num_dec_parameters = 0, 0
       
        # 12 * 1 * 1
        # 12 * 12 * 1
        all_head_score, head_score, all_dec_self_head_score, all_dec_cross_head_score, dec_head_score1, dec_head_score2 = self.transform_scores_for_head()

        if self.encdec_pruning_type =="nash": 
            ## NASH-manual: layer pruning option is not contained. 
            # 
            dec_head_score1 = torch.ones_like(dec_head_score1) # dec_self_head_z individual decoder head would not be pruned
            dec_head_score2 = torch.ones_like(dec_head_score2) # dec_cross_head_z individual decoder head would not be pruned
            self.reset_loga(self.dec_int_loga, mean=10) # dec col,row not be pruned
            
            if self.auto_select: 
                # NASH-auto_select: layer pruning option is contained -> avoid the enc_layers are pruned
                # enc --> only fine-grained pruning is applied (head, mlp, hidden) # dec --> only coarse-grained pruning is applied 
                all_head_score = torch.ones_like(all_head_score) # enc_self_layer_z encoder att layer would not be pruned
                self.reset_loga(self.intlayer_loga, mean=10) # enc_mlp_layer_z < encdoer mlp layer would not be pruned        
           
        hidden_score = 1 - self.cdf_qz(0, self.hidden_loga) # 768

        if all_head_score is not None:
            head_score = (all_head_score * head_score).reshape(-1)
        else:
            head_score = head_score.reshape(-1)
        num_enc_parameters += \
            torch.sum(torch.outer(hidden_score, head_score)) * self.parameters_per_dim["head"] / self.hidden_size
        
        try:
            intlayer_score = 1 - self.cdf_qz(0, self.intlayer_loga)  # 12
        except AttributeError:
            intlayer_score = torch.ones(self.num_hidden_layers) # drop the layer <-- resolve
        int_score = 1 - self.cdf_qz(0, self.int_loga)  # 12 * 3072
        intlayer_score = intlayer_score.unsqueeze(-1)

        int_score = (intlayer_score * int_score).reshape(-1)
        num_enc_parameters += torch.sum(torch.outer(hidden_score, int_score)) * 2

        if all_dec_self_head_score is not None:
            dec_head_score1 = (all_dec_self_head_score * dec_head_score1).reshape(-1)
            dec_head_score2 = (all_dec_cross_head_score * dec_head_score2).reshape(-1)
        else:
            dec_head_score1 = dec_head_score1.reshape(-1)
            dec_head_score2 = dec_head_score2.reshape(-1)
        num_dec_parameters += \
            torch.sum(torch.outer(hidden_score, dec_head_score1)) * self.parameters_per_dim["head"] / self.hidden_size
        num_dec_parameters += \
            torch.sum(torch.outer(hidden_score, dec_head_score2)) * self.parameters_per_dim["head"] / self.hidden_size

        try:
            dec_intlayer_score = 1 - self.cdf_qz(0, self.dec_intlayer_loga)  # 12
        except AttributeError:
            dec_intlayer_score = torch.ones(self.num_decoder_layers)
        dec_int_score = 1 - self.cdf_qz(0, self.dec_int_loga)  # 12 * 3072
        dec_intlayer_score = dec_intlayer_score.unsqueeze(-1)

        dec_int_score = (dec_intlayer_score * dec_int_score).reshape(-1)
        num_dec_parameters += torch.sum(torch.outer(hidden_score, dec_int_score)) * 2
        
        num_parameters = num_enc_parameters + num_dec_parameters
        return num_parameters, num_enc_parameters, num_dec_parameters

    def get_num_parameters_and_constraint(self):
        num_enc_parameters, num_dec_parameters = 0, 0

        all_head_score, head_score, all_dec_self_head_score, all_dec_cross_head_score, dec_head_score1, dec_head_score2 = self.transform_scores_for_head()
        
        if self.encdec_pruning_type =="nash": 
            ## NASH-manual: layer pruning option is not contained. 
            dec_head_score1 = torch.ones_like(dec_head_score1) # dec_self_head_z individual decoder head would not be pruned
            dec_head_score2 = torch.ones_like(dec_head_score2) # dec_cross_head_z individual decoder head would not be pruned
            self.reset_loga(self.dec_int_loga, mean=10) # dec col,row not be pruned
            
            if self.auto_select: 
                # NASH-auto_select: layer pruning option is contained -> avoid the enc_layers are pruned
                # enc --> only fine-grained pruning is applied (head, mlp, hidden) # dec --> only coarse-grained pruning is applied 
                all_head_score = torch.ones_like(all_head_score) # enc_self_layer_z encoder att layer would not be pruned
                self.reset_loga(self.intlayer_loga, mean=10) # enc_mlp_layer_z < encdoer mlp layer would not be pruned             
        
        if all_head_score is not None:
#             head_score = (all_head_score * head_score).reshape(-1)
            head_score = head_score * all_head_score
        else:
            head_score = head_score.reshape(-1)
        
        num_enc_parameters += torch.sum(head_score) * self.parameters_per_dim["head"]

        try:
            intlayer_score = 1 - self.cdf_qz(0, self.intlayer_loga)  # 12
        except AttributeError:
            intlayer_score = torch.ones(self.num_hidden_layers) # drop the layer <-- resolve
        int_score = 1 - self.cdf_qz(0, self.int_loga)  # 12 * 3072
        intlayer_score = intlayer_score.unsqueeze(-1)

        int_score = int_score * intlayer_score
        num_enc_parameters += torch.sum(int_score) * self.parameters_per_dim["intermediate"]
        

        if all_dec_self_head_score is not None:
            dec_head_score1 = all_dec_self_head_score * dec_head_score1
            dec_head_score2 = all_dec_cross_head_score * dec_head_score2
        else:
            dec_head_score1 = dec_head_score1.reshape(-1)
            dec_head_score2 = dec_head_score2.reshape(-1)

        num_dec_parameters += torch.sum(dec_head_score1) * self.parameters_per_dim["head"]
        num_dec_parameters += torch.sum(dec_head_score2) * self.parameters_per_dim["head"]

        try:
            dec_intlayer_score = 1 - self.cdf_qz(0, self.dec_intlayer_loga)  # 12
        except AttributeError:
            dec_intlayer_score = torch.ones(self.num_decoder_layers)
            
        dec_int_score = 1 - self.cdf_qz(0, self.dec_int_loga)  # 12 * 3072
        dec_intlayer_score = dec_intlayer_score.unsqueeze(-1)

        dec_int_score = dec_int_score * dec_intlayer_score
        num_dec_parameters += torch.sum(dec_int_score) * self.parameters_per_dim["intermediate"]

        num_parameters = num_enc_parameters + num_dec_parameters
        return num_parameters, num_enc_parameters, num_dec_parameters


    def get_target_sparsity(self, pruned_steps):
        target_sparsity = (self.target_sparsity - self.start_sparsity) * min(1, pruned_steps / self.lagrangian_warmup) + self.start_sparsity
        return target_sparsity


    def lagrangian_regularization(self, pruned_steps):
        target_sparsity = self.target_sparsity
        if "hidden" in self.types:
            expected_size, expected_enc_size, expected_dec_size = self.get_num_parameters_and_constraint_for_hidden() #! calculate \bar s
        else:
            expected_size, expected_enc_size, expected_dec_size = self.get_num_parameters_and_constraint() #! calculate \bar s
            
        expected_sparsity = 1 - expected_size / self.prunable_model_size
        expected_enc_sparsity = 1 - expected_enc_size / self.prunable_encoder_size
        expected_dec_sparsity = 1 - expected_dec_size / self.prunable_decoder_size
        
        if self.lagrangian_warmup > 0:
            target_sparsity = self.get_target_sparsity(pruned_steps)
            
            if hasattr(self.config, "selected_layer") and self.config.selected_layer:
                target_enc_sparsity = target_sparsity
                target_sparsity = target_sparsity * (self.prunable_encoder_size / self.prunable_model_size)
                
            elif hasattr(self.config, "auto_select") and self.config.auto_select:
                target_enc_sparsity = target_sparsity
                target_dec_sparsity = (self.config.auto_select - self.start_sparsity) * min(1, pruned_steps / self.lagrangian_warmup) + self.start_sparsity
        
        if self.encdec_pruning_type == "cofi":
            # Naive CoFi
            lagrangian_loss = ( #! see appendix
                    self.lambda_1 * (expected_sparsity - target_sparsity)
                    + self.lambda_2 * (expected_sparsity - target_sparsity) ** 2 #! where is the lambda 1 and lambda 2 from
            )
        
        elif self.encdec_pruning_type == "nash":
            # Different target sparsity on encoder and decoder networks
            if self.auto_select:
                lagrangian_loss = (
                    self.lambda_1 * (expected_enc_sparsity - target_enc_sparsity) 
                    + self.lambda_1 * (expected_dec_sparsity - target_dec_sparsity)
                    + self.lambda_2 * (expected_enc_sparsity - target_enc_sparsity) ** 2
                    + self.lambda_2 * (expected_dec_sparsity - target_dec_sparsity) ** 2
                ) # nash-autoselect
            else:
                lagrangian_loss = (
                    self.lambda_1 * (expected_enc_sparsity - target_enc_sparsity) 
                    + self.lambda_2 * (expected_enc_sparsity - target_enc_sparsity) ** 2
                ) #nash-manual
            
        else:
            raise NotImplementedError
            
        return lagrangian_loss, expected_sparsity, target_sparsity
        
        
    def get_eps(self, size):
        """Uniform random numbers for the concrete distribution"""
        eps = torch.FloatTensor(size).uniform_(epsilon, 1-epsilon)
        eps = Variable(eps)
        return eps

    # during training
    def _sample_z(self, loga):
        eps = self.get_eps(torch.FloatTensor(*loga.shape)).to(loga.device)
        z = self.quantile_concrete(eps, loga)
        z = F.hardtanh(z, min_val=0, max_val=1)
        return z

    # during inference
    def _deterministic_z(self, size, loga):
        # Following https://github.com/asappresearch/flop/blob/e80e47155de83abbe7d90190e00d30bfb85c18d5/flop/hardconcrete.py#L8 line 103
        expected_num_nonzeros = torch.sum(1 - self.cdf_qz(0, loga))
        expected_num_zeros = size - expected_num_nonzeros.item()
        try:
            num_zeros = round(expected_num_zeros)
        except:
            pdb.set_trace()
        soft_mask = torch.sigmoid(loga / self.temperature * self.magical_number)
        if num_zeros > 0:
            if soft_mask.ndim == 0:
                soft_mask = torch.tensor(0).to(loga.device)
            else:
                _, indices = torch.topk(soft_mask, k=num_zeros, largest=False)
                soft_mask[indices] = 0.
        return soft_mask

    def get_z_from_zs(self, zs):
        numpified_zs = {} 
        for type in self.all_types:
            name = type[:-2]
            if name in self.types:
                z = zs.get(type, np.ones(self.shapes[name]))
                if torch.is_tensor(z): 
                    new_z = z.squeeze().detach().cpu().numpy() > 0
                numpified_zs[name] = new_z
        return numpified_zs

    def calculate_model_size(self, zs):
        numpified_zs = self.get_z_from_zs(zs)
        
        hidden_z = numpified_zs["hidden"] if "hidden" in numpified_zs.keys() else np.ones(self.hidden_size).reshape(-1,1)
        intermediate_z = numpified_zs["intermediate"]
        mlp_z = numpified_zs["mlp"].reshape(-1, 1) if "mlp" in numpified_zs.keys() \
            else np.ones(self.num_hidden_layers).reshape(-1, 1)
        head_z = numpified_zs["head"]
        head_layer_z = numpified_zs["head_layer"].reshape(-1, 1) if "head_layer" in numpified_zs.keys() \
            else np.ones(self.num_hidden_layers).reshape(-1, 1)

        # dec_hidden_z = numpified_zs["dec_hidden"]
        dec_intermediate_z = numpified_zs["dec_intermediate"]
        dec_mlp_z = numpified_zs["dec_mlp"].reshape(-1, 1) if "mlp" in numpified_zs.keys() \
            else np.ones(self.num_decoder_layers).reshape(-1, 1)
        dec_self_head_z = numpified_zs["dec_self_head"]
        dec_cross_head_z = numpified_zs["dec_cross_head"]
        dec_self_head_layer_z = numpified_zs["dec_self_head_layer"].reshape(-1, 1) if "dec_self_head_layer" in numpified_zs.keys() \
            else np.ones(self.num_decoder_layers).reshape(-1, 1)
        dec_cross_head_layer_z = numpified_zs["dec_cross_head_layer"].reshape(-1, 1) if "dec_cross_head_layer" in numpified_zs.keys() \
            else np.ones(self.num_decoder_layers).reshape(-1, 1)

        remaining_hidden_dims = hidden_z.sum().item()
        remaining_intermediate_nums = intermediate_z.reshape(self.num_hidden_layers, self.enc_intermediate_size).sum(-1).tolist()
        remaining_head_nums = head_z.reshape(self.num_hidden_layers, self.num_attention_heads).sum(-1).tolist()

        remaining_dec_intermediate_nums = dec_intermediate_z.reshape(self.num_decoder_layers, self.dec_intermediate_size).sum(-1).tolist() # 
        remaining_dec_self_head_nums = dec_self_head_z.reshape(self.num_decoder_layers, self.num_attention_heads).sum(-1).tolist() #
        remaining_dec_cross_head_nums = dec_cross_head_z.reshape(self.num_decoder_layers, self.num_attention_heads).sum(-1).tolist() #

        head_nums = np.outer((head_z * head_layer_z).reshape(-1), hidden_z).sum().item()
        dec_self_head_nums = np.outer((dec_self_head_z * dec_self_head_layer_z).reshape(-1), hidden_z).sum().item() # 
        dec_cross_head_nums = np.outer((dec_cross_head_z * dec_cross_head_layer_z).reshape(-1), hidden_z).sum().item() #

        intermediate_nums = np.outer((intermediate_z * mlp_z).reshape(-1), hidden_z).sum().item()
        dec_intermediate_nums = np.outer((dec_intermediate_z * dec_mlp_z).reshape(-1), hidden_z).sum().item() #

        remaining_model_size = (head_nums + dec_self_head_nums + dec_cross_head_nums) * self.dim_per_head * 4 \
            + (intermediate_nums + dec_intermediate_nums) * 2
        remaining_encoder_size = head_nums * self.dim_per_head * 4 + intermediate_nums * 2
        remaining_decoder_size = (dec_self_head_nums + dec_cross_head_nums) * self.dim_per_head * 4 + dec_intermediate_nums * 2
        
        pruned_model_size = self.prunable_model_size - remaining_model_size
        pruned_encoder_size = self.prunable_encoder_size - remaining_encoder_size
        pruned_decoder_size = self.prunable_decoder_size - remaining_decoder_size

        results = {}
        # Not multiplied with each other
        results["head_layers"] = head_layer_z.reshape(-1).astype(int).tolist()
        results["mlp_layers"] = mlp_z.reshape(-1).astype(int).tolist()
        results["hidden_dims"] = remaining_hidden_dims
        results["intermediate_dims"] = remaining_intermediate_nums
        results["head_nums"] = remaining_head_nums

        results["dec_self_head_layers"] = dec_self_head_layer_z.reshape(-1).astype(int).tolist()
        results["dec_cross_head_layers"] = dec_cross_head_layer_z.reshape(-1).astype(int).tolist()
        results["dec_mlp_layers"] = dec_mlp_z.reshape(-1).astype(int).tolist()
        results["dec_intermediate_dims"] = remaining_dec_intermediate_nums
        results["dec_self_head_nums"] = remaining_dec_self_head_nums
        results["dec_cross_head_nums"] = remaining_dec_cross_head_nums

        results["pruned_params"] = pruned_model_size
        results["remaining_params"] = remaining_model_size
        results["pruned_model_sparsity"] = pruned_model_size / self.prunable_model_size
        
        results["pruned_enc_params"] = pruned_encoder_size
        results["remaining_enc_params"] = remaining_encoder_size
        results["pruned_encoder_sparsity"] = pruned_encoder_size / self.prunable_encoder_size
        
        results["pruned_dec_params"] = pruned_decoder_size
        results["remaining_dec_params"] = remaining_decoder_size
        results["pruned_decoder_sparsity"] = pruned_decoder_size / self.prunable_decoder_size
        return results

    def forward(self, training=True,):
        zs = {f"{type}_z": [] for type in self.types}

        if training:
            for i, type in enumerate(self.types):
                loga = self.z_logas[type]
                z = self._sample_z(loga)
                zs[f"{type}_z"] = z.reshape(self.shapes[type])
        else:
            for i, type in enumerate(self.types):
                if not "hidden" in type: # hidden is not a per layer sample
                    loga_all_layers = self.z_logas[type]
                    for layer in range(len(loga_all_layers)):
                        loga = loga_all_layers[layer]
                        size = self.sizes[type]
                        z = self._deterministic_z(size, loga)
                        zs[f"{type}_z"].append(z.reshape(self.shapes[type][1:]))
                else:
                    z = self._deterministic_z(self.sizes[type], self.hidden_loga)
                    zs[f"{type}_z"] = z
            for type in zs:
                if not "hidden_z" in type:
                    zs[type] = torch.stack(zs[type])
        return zs