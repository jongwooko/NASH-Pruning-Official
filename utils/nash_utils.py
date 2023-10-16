import torch
import os
from transformers.modeling_utils import prune_linear_layer
from transformers import AutoConfig
from transformers.file_utils import hf_bucket_url, cached_path

from utils.utils import calculate_parameters

def edit_config(config, additional_args):
    config.transform_embedding = additional_args.transform_embedding
    config.do_distill = additional_args.do_distill
    config.do_layer_distill = additional_args.do_layer_distill

def initialize_layer_transformation(model):
    model.layer_transformation.weight.data.copy_(
        torch.eye(len(model.layer_transformation.weight)))
    model.layer_transformation.bias.data.fill_(0)

def load_model_with_zs(model_path, model_class, zs=None):
    config_path = os.path.join(model_path, "config.json")
    if os.path.exists(config_path):
        config = AutoConfig.from_pretrained(model_path)
    model = model_class.from_pretrained(model_path, config=config)
    p = os.path.join(model_path, "pytorch_model.bin")
    loaded_weights = torch.load(p, map_location="cpu")
    model.load_state_dict(loaded_weights, strict=False) #여기에서는 에러 X
    print(f"Load weights from {model_path}")

    update_params_t5(model, zs)
    print(f"Model Size before pruning: {calculate_parameters(model)}")
    prune_model_with_z_t5(zs, model)
    print(f"Model Size after pruning: {calculate_parameters(model)}")
    return model

def load_model(model_path, model_class, zs=None):
    assert zs is not None
    model = load_model_with_zs(model_path, model_class, zs)
    print(f"Model Size: {calculate_parameters(model)}")
    return model

# load the l0 module
def load_l0_module(model_path):
    l0_module_path = os.path.join(model_path, "l0_module.pt")
    if os.path.exists(l0_module_path):
        return torch.load(l0_module_path, map_location=torch.device('cpu'))
    else:
        return None

# z values could be in [0, 1), we update the parameters accordingly with z values

def prune_intermediate_layers_t5(model, keep_dims, enc=True):
    device = model.device
    for layer in keep_dims:
        if enc:
            if len(keep_dims[layer]) == 0:
                if hasattr(model.encoder.block[layer].layer[1].DenseReluDense, "wi"):
                    model.encoder.block[layer].layer[1].DenseReluDense.wi = None
                elif hasattr(model.encoder.block[layer].layer[1].DenseReluDense, "wi_0"):
                    model.encoder.block[layer].layer[1].DenseReluDense.wi_0 = None
                    model.encoder.block[layer].layer[1].DenseReluDense.wi_1 = None
                model.encoder.block[layer].layer[1].DenseReluDense.wo = None
            else:
                if hasattr(model.encoder.block[layer].layer[1].DenseReluDense, "wi"):
                    model.encoder.block[layer].layer[1].DenseReluDense.wi = prune_linear_layer(model.encoder.block[layer].layer[1].DenseReluDense.wi, index=torch.LongTensor(keep_dims[layer]).to(device), dim=0)
                elif hasattr(model.encoder.block[layer].layer[1].DenseReluDense, "wi_0"):
                    model.encoder.block[layer].layer[1].DenseReluDense.wi_0 = prune_linear_layer(model.encoder.block[layer].layer[1].DenseReluDense.wi_0, index=torch.LongTensor(keep_dims[layer]).to(device), dim=0)
                    model.encoder.block[layer].layer[1].DenseReluDense.wi_1 = prune_linear_layer(model.encoder.block[layer].layer[1].DenseReluDense.wi_1, index=torch.LongTensor(keep_dims[layer]).to(device), dim=0) 
                else:
                    NotImplementedError
                model.encoder.block[layer].layer[1].DenseReluDense.wo = prune_linear_layer(model.encoder.block[layer].layer[1].DenseReluDense.wo, index=torch.LongTensor(keep_dims[layer]).to(device), dim=1) 
        else:
            if len(keep_dims[layer]) == 0:
                if hasattr(model.decoder.block[layer].layer[2].DenseReluDense, "wi"):
                    model.decoder.block[layer].layer[2].DenseReluDense.wi = None
                elif hasattr(model.decoder.block[layer].layer[2].DenseReluDense, "wi_0"):
                    model.decoder.block[layer].layer[2].DenseReluDense.wi_0 = None
                    model.decoder.block[layer].layer[2].DenseReluDense.wi_0 = None
                model.decoder.block[layer].layer[2].DenseReluDense.wo = None
            else:
                if hasattr(model.decoder.block[layer].layer[2].DenseReluDense, "wi"):
                    model.decoder.block[layer].layer[2].DenseReluDense.wi = prune_linear_layer(model.decoder.block[layer].layer[2].DenseReluDense.wi, index=torch.LongTensor(keep_dims[layer]).to(device), dim=0)
                elif hasattr(model.decoder.block[layer].layer[2].DenseReluDense, "wi_0"):
                    model.decoder.block[layer].layer[2].DenseReluDense.wi_0 = prune_linear_layer(model.decoder.block[layer].layer[2].DenseReluDense.wi_0, index=torch.LongTensor(keep_dims[layer]).to(device), dim=0)
                    model.decoder.block[layer].layer[2].DenseReluDense.wi_1 = prune_linear_layer(model.decoder.block[layer].layer[2].DenseReluDense.wi_1, index=torch.LongTensor(keep_dims[layer]).to(device), dim=0)
                else:
                    NotImplementedError
                model.decoder.block[layer].layer[2].DenseReluDense.wo = prune_linear_layer(model.decoder.block[layer].layer[2].DenseReluDense.wo, index=torch.LongTensor(keep_dims[layer]).to(device), dim=1)

def load_zs(model_path):
    if model_path.endswith("zs.pt"):
        zs_path = model_path
    else:
        zs_path = os.path.join(model_path, "zs.pt")

    if os.path.exists(zs_path):
        zs = torch.load(zs_path, map_location="cpu")
        if zs is None:
            model_path = os.path.dirname(model_path)
            l0_module = torch.load(os.path.join(model_path, "l0_module.pt"), map_location="cpu")
            zs = l0_module.forward(training=False)
        return zs
    else:
        return None

def load_pruned_model_t5(model, weights):
    config = model.config
    dim_per_head = config.d_model // config.num_heads
    zs = {}

    hidden_z = torch.zeros(config.d_model)
    hidden_z[:weights["shared.weight"].shape[1]] = 1
    zs["hidden_z"] = hidden_z
    
    # encoder self-attention
    head_z = torch.zeros(config.num_layers, config.num_heads)
    head_layer_z = torch.zeros(config.num_layers)
    for i in range(config.num_layers):
        key = f"encoder.block.{i}.layer.0.SelfAttention.o.weight" # encoder.block.0.layer.0.SelfAttention.o.weight
        if key in weights:
            remaining_heads = weights[key].shape[-1] // dim_per_head
            head_z[i, :remaining_heads] = 1
            head_layer_z[i] = 1
    zs["head_z"] = head_z
    zs["head_layer_z"] = head_layer_z
    
    # encoder feed-forward networks
    int_z = torch.zeros(config.num_layers, config.d_ff)
    mlp_z = torch.zeros(config.num_layers)
    for i in range(config.num_layers):
        key = f"encoder.block.{i}.layer.1.DenseReluDense.wo.weight"
        if key in weights:
            remaining_int_dims = weights[key].shape[-1]
            int_z[i, :remaining_int_dims] = 1
            mlp_z[i] = 1
    zs["intermediate_z"] = int_z
    zs["mlp_z"] = mlp_z
    
    # decoder self-attention
    dec_self_head_z = torch.zeros(config.num_layers, config.num_heads)
    dec_self_head_layer_z = torch.zeros(config.num_layers)
    for i in range(config.num_layers):
        key = f"decoder.block.{i}.layer.0.SelfAttention.o.weight" # encoder.block.0.layer.0.SelfAttention.o.weight
        if key in weights:
            remaining_heads = weights[key].shape[-1] // dim_per_head
            dec_self_head_z[i, :remaining_heads] = 1
            dec_self_head_layer_z[i] = 1
    zs["dec_self_head_z"] = dec_self_head_z
    zs["dec_self_head_layer_z"] = dec_self_head_layer_z
    
    # decoder cross-attention
    dec_cross_head_z = torch.zeros(config.num_layers, config.num_heads)
    dec_cross_head_layer_z = torch.zeros(config.num_layers)
    for i in range(config.num_layers):
        key = f"decoder.block.{i}.layer.1.EncDecAttention.o.weight" # encoder.block.0.layer.0.SelfAttention.o.weight
        if key in weights:
            remaining_heads = weights[key].shape[-1] // dim_per_head
            dec_cross_head_z[i, :remaining_heads] = 1
            dec_cross_head_layer_z[i] = 1
    zs["dec_cross_head_z"] = dec_cross_head_z
    zs["dec_cross_head_layer_z"] = dec_cross_head_layer_z
    
    # decoder feed-forward network
    dec_int_z = torch.zeros(config.num_layers, config.d_ff)
    dec_mlp_z = torch.zeros(config.num_layers)
    for i in range(config.num_layers):
        key = f"decoder.block.{i}.layer.2.DenseReluDense.wo.weight"
        if key in weights:
            remaining_int_dims = weights[key].shape[-1]
            dec_int_z[i, :remaining_int_dims] = 1
            dec_mlp_z[i] = 1
    zs["dec_intermediate_z"] = dec_int_z
    zs["dec_mlp_z"] = dec_mlp_z
    
    prune_model_with_z_t5(zs, model)
    model.load_state_dict(weights, strict=False)
    return model

def get_full_model_size(model_class, model_name):
    model = model_class.from_pretrained(model_name)
    model_size = calculate_parameters(model)
    return model_size

def update_params_t5(model, zs):
    config = model.config
    hidden_dims = config.hidden_size
    num_heads = config.num_attention_heads
    dims_per_head = config.d_kv  # hidden_dims // num_heads
    num_layers = config.num_hidden_layers
    dec_num_layers = config.num_decoder_layers if config.num_decoder_layers is not None else config.num_hidden_layers

    if zs is not None:
        if "intermediate_z" in zs:
            for layer in range(num_layers):
                intermediate_z = zs["intermediate_z"][layer].cpu().squeeze().clone()
                model.encoder.block[layer].layer[1].DenseReluDense.wo.weight.data = model.encoder.block[layer].layer[1].DenseReluDense.wo.weight.data.mul(intermediate_z)
                if "mlp_z" in zs:
                    mlp_z = zs["mlp_z"][layer].cpu()
                    model.encoder.block[layer].layer[1].DenseReluDense.wo.weight.data = model.encoder.block[layer].layer[1].DenseReluDense.wo.weight.data.transpose(0, 1).mul(mlp_z).transpose(0, 1)

        if "head_z" in zs:
            for layer in range(num_layers):
                head_z = zs["head_z"][layer].cpu().squeeze().clone()
                head_z = torch.repeat_interleave(head_z, dims_per_head)
                model.encoder.block[layer].layer[0].SelfAttention.v.weight.data = model.encoder.block[layer].layer[0].SelfAttention.v.weight.data.transpose(0, 1).mul(head_z).transpose(0, 1)
                if "head_layer_z" in zs:
                    head_layer_z = zs["head_layer_z"][layer].cpu()
                    model.encoder.block[layer].layer[0].SelfAttention.o.weight.data = model.encoder.block[layer].layer[0].SelfAttention.o.weight.data.transpose(0, 1).mul(head_layer_z).transpose(0, 1)
        if "dec_intermediate_z" in zs:
            for layer in range(dec_num_layers):
                dec_intermediate_z = zs["dec_intermediate_z"][layer].cpu().squeeze().clone()
                model.decoder.block[layer].layer[2].DenseReluDense.wo.weight.data = model.decoder.block[layer].layer[2].DenseReluDense.wo.weight.data.mul(dec_intermediate_z)
                if "dec_mlp_z" in zs:
                    dec_mlp_z = zs["dec_mlp_z"][layer].cpu()
                    model.decoder.block[layer].layer[2].DenseReluDense.wo.weight.data = model.decoder.block[layer].layer[2].DenseReluDense.wo.weight.data.transpose(0, 1).mul(dec_mlp_z).transpose(0, 1)

        if "dec_self_head_z" in zs:
            for layer in range(dec_num_layers):
                dec_self_head_z = zs["dec_self_head_z"][layer].cpu().squeeze().clone()
                dec_self_head_z = torch.repeat_interleave(dec_self_head_z, dims_per_head)
                model.decoder.block[layer].layer[0].SelfAttention.v.weight.data = model.decoder.block[layer].layer[0].SelfAttention.v.weight.data.transpose(0, 1).mul(dec_self_head_z).transpose(0, 1)
                if "dec_self_head_layer_z" in zs:
                    dec_self_head_layer_z = zs["dec_self_head_layer_z"][layer].cpu()
                    model.decoder.block[layer].layer[0].SelfAttention.o.weight.data = model.decoder.block[layer].layer[0].SelfAttention.o.weight.data.transpose(0, 1).mul(dec_self_head_layer_z).transpose(0, 1)

        if "dec_cross_head_z" in zs:
            for layer in range(dec_num_layers):
                dec_cross_head_z = zs["dec_cross_head_z"][layer].cpu().squeeze().clone()
                dec_cross_head_z = torch.repeat_interleave(dec_cross_head_z, dims_per_head)
                model.decoder.block[layer].layer[1].EncDecAttention.v.weight.data = model.decoder.block[layer].layer[1].EncDecAttention.v.weight.data.transpose(0, 1).mul(dec_cross_head_z).transpose(0, 1)
                if "dec_cross_head_layer_z" in zs:
                    dec_cross_head_layer_z = zs["dec_cross_head_layer_z"][layer].cpu()
                    model.decoder.block[layer].layer[1].EncDecAttention.o.weight.data = model.decoder.block[layer].layer[1].EncDecAttention.o.weight.data.transpose(0, 1).mul(dec_cross_head_layer_z).transpose(0, 1)

        if "hidden_z" in zs:
            hidden_z = zs["hidden_z"].cpu().squeeze().clone()
            model.shared.weight.data = model.shared.weight.data.mul(hidden_z)
            model.encoder.embed_tokens.weight.data = model.encoder.embed_tokens.weight.data.mul(hidden_z)
            model.decoder.embed_tokens.weight.data = model.decoder.embed_tokens.weight.data.mul(hidden_z)
            for layer in range(num_layers):
                # encoder
                # MHA
                model.encoder.block[layer].layer[0].SelfAttention.q.weight.data = \
                    model.encoder.block[layer].layer[0].SelfAttention.q.weight.data.mul(hidden_z)
                model.encoder.block[layer].layer[0].SelfAttention.k.weight.data = \
                    model.encoder.block[layer].layer[0].SelfAttention.k.weight.data.mul(hidden_z)
            
                model.encoder.block[layer].layer[0].SelfAttention.v.weight.data = \
                    model.encoder.block[layer].layer[0].SelfAttention.v.weight.data.mul(hidden_z)
                model.encoder.block[layer].layer[0].SelfAttention.o.weight.data = \
                    model.encoder.block[layer].layer[0].SelfAttention.o.weight.data.transpose(0,1).mul(hidden_z).transpose(0,1)
                # FF 
                if hasattr(model.encoder.block[layer].layer[1].DenseReluDense, "wi"):
                    model.encoder.block[layer].layer[1].DenseReluDense.wi.weight.data = \
                        model.encoder.block[layer].layer[1].DenseReluDense.wi.weight.data.mul(hidden_z)
                elif hasattr(model.encoder.block[layer].layer[1].DenseReluDense, "wi_0"):
                    model.encoder.block[layer].layer[1].DenseReluDense.wi_0.weight.data = \
                        model.encoder.block[layer].layer[1].DenseReluDense.wi_0.weight.data.mul(hidden_z)
                    model.encoder.block[layer].layer[1].DenseReluDense.wi_1.weight.data = \
                        model.encoder.block[layer].layer[1].DenseReluDense.wi_1.weight.data.mul(hidden_z)
                else:
                    NotImplementedError
                model.encoder.block[layer].layer[1].DenseReluDense.wo.weight.data = \
                    model.encoder.block[layer].layer[1].DenseReluDense.wo.weight.data.transpose(0,1).mul(hidden_z).transpose(0,1)
            
            for layer in range(dec_num_layers):
                # decoder
                # MHA
                model.decoder.block[layer].layer[0].SelfAttention.q.weight.data = \
                    model.decoder.block[layer].layer[0].SelfAttention.q.weight.data.mul(hidden_z)
                model.decoder.block[layer].layer[0].SelfAttention.k.weight.data = \
                    model.decoder.block[layer].layer[0].SelfAttention.k.weight.data.mul(hidden_z)
                model.decoder.block[layer].layer[0].SelfAttention.v.weight.data = \
                    model.decoder.block[layer].layer[0].SelfAttention.v.weight.data.mul(hidden_z)
                model.decoder.block[layer].layer[0].SelfAttention.o.weight.data = \
                    model.decoder.block[layer].layer[0].SelfAttention.o.weight.data.transpose(0,1).mul(hidden_z).transpose(0,1)

                # CA
                model.decoder.block[layer].layer[1].EncDecAttention.q.weight.data = \
                    model.decoder.block[layer].layer[1].EncDecAttention.q.weight.data.mul(hidden_z)
                model.decoder.block[layer].layer[1].EncDecAttention.k.weight.data = \
                    model.decoder.block[layer].layer[1].EncDecAttention.k.weight.data.mul(hidden_z)
                model.decoder.block[layer].layer[1].EncDecAttention.v.weight.data = \
                    model.decoder.block[layer].layer[1].EncDecAttention.v.weight.data.mul(hidden_z)
                model.decoder.block[layer].layer[1].EncDecAttention.o.weight.data = \
                    model.decoder.block[layer].layer[1].EncDecAttention.o.weight.data.transpose(0,1).mul(hidden_z).transpose(0,1)

                # FF
                if hasattr(model.decoder.block[layer].layer[2].DenseReluDense, "wi"):
                    model.decoder.block[layer].layer[2].DenseReluDense.wi.weight.data = \
                        model.decoder.block[layer].layer[2].DenseReluDense.wi.weight.data.mul(hidden_z)
                elif hasattr(model.decoder.block[layer].layer[2].DenseReluDense, "wi_0"):
                    model.decoder.block[layer].layer[2].DenseReluDense.wi_0.weight.data = \
                        model.decoder.block[layer].layer[2].DenseReluDense.wi_0.weight.data.mul(hidden_z)
                    model.decoder.block[layer].layer[2].DenseReluDense.wi_1.weight.data = \
                        model.decoder.block[layer].layer[2].DenseReluDense.wi_1.weight.data.mul(hidden_z)
                else:
                    NotImplementedError
                model.decoder.block[layer].layer[2].DenseReluDense.wo.weight.data = \
                    model.decoder.block[layer].layer[2].DenseReluDense.wo.weight.data.transpose(0,1).mul(hidden_z).transpose(0,1)

            if hasattr(model, "lm_head"):
                model.lm_head.weight.data = model.lm_head.weight.data.mul(hidden_z)

def prune_model_with_z_t5(zs, model):
    if zs is None:
        return None, None

    if "head_z" in zs:
        head_z = zs.get("head_z", None)
        head_layer_z = zs.get("head_layer_z", None)

        prune_heads = {'att_loc': 'enc_self'}
        for layer in range(len(head_z)):
            head_z_layer = head_z[layer].cpu().squeeze().clone()
            if head_layer_z is not None:
                head_z_layer *= head_layer_z[layer]
            index = torch.where(head_z_layer == 0)[0].tolist()
            prune_heads[layer] = index
            print(f"Enc Layer {layer}, heads {' '.join([str(i) for i in index])} pruned.")
        model.prune_heads(prune_heads)
    if "dec_self_head_z" in zs:
        dec_self_head_z = zs.get("dec_self_head_z", None)
        dec_self_head_layer_z = zs.get("dec_self_head_layer_z", None)

        dec_self_prune_heads = {'att_loc':'dec_self'}
        for layer in range(len(dec_self_head_z)):
            dec_self_head_z_layer = dec_self_head_z[layer].cpu().squeeze().clone()
            if dec_self_head_layer_z is not None:
                dec_self_head_z_layer *= dec_self_head_layer_z[layer]
            index = torch.where(dec_self_head_z_layer == 0)[0].tolist()
            dec_self_prune_heads[layer] = index

            print(f"Dec Self Layer {layer}, heads {' '.join([str(i) for i in index])} pruned.")
        model.prune_heads(dec_self_prune_heads)

        dec_cross_head_z = zs.get("dec_cross_head_z", None)
        dec_cross_head_layer_z = zs.get("dec_cross_head_layer_z", None)

        dec_cross_prune_heads = {'att_loc':'dec_cross'}
        for layer in range(len(dec_cross_head_z)):
            dec_cross_head_z_layer = dec_cross_head_z[layer].cpu().squeeze().clone()
            if dec_cross_head_layer_z is not None:
                dec_cross_head_z_layer *= dec_cross_head_layer_z[layer]
            index = torch.where(dec_cross_head_z_layer == 0)[0].tolist()
            dec_cross_prune_heads[layer] = index

            print(f"Dec Cross Layer {layer}, heads {' '.join([str(i) for i in index])} pruned.")
        model.prune_heads(dec_cross_prune_heads)
    
    kept_intermediate_dims = None
    if "intermediate_z" in zs:
        kept_intermediate_dims = {}
        intermediate_zs = zs["intermediate_z"]
        mlp_z = zs.get("mlp_z", None)
        for layer in range(len(intermediate_zs)):
            intermediate_z_layer = intermediate_zs[layer].squeeze()
            intermediate_z_layer = intermediate_z_layer.cpu().clone()
            if mlp_z is not None:
                intermediate_z_layer *= mlp_z[layer]
            kept_intermediate_dims[layer] = intermediate_z_layer.nonzero().reshape(-1).tolist()

    dec_kept_intermediate_dims = None
    if "dec_intermediate_z" in zs:
        dec_kept_intermediate_dims = {}
        dec_intermediate_zs = zs["dec_intermediate_z"]
        dec_mlp_z = zs.get("dec_mlp_z", None)
        for layer in range(len(dec_intermediate_zs)):
            dec_intermediate_z_layer = dec_intermediate_zs[layer].squeeze()
            dec_intermediate_z_layer = dec_intermediate_z_layer.cpu().clone()
            if dec_mlp_z is not None:
                dec_intermediate_z_layer *= dec_mlp_z[layer]
            dec_kept_intermediate_dims[layer] = dec_intermediate_z_layer.nonzero().reshape(-1).tolist()

    def prune_layer_norm(layernorm, index):
        layernorm.weight = torch.nn.parameter.Parameter(
            layernorm.weight.index_select(0, index))
        if hasattr(layernorm, "bias"):
            layernorm.bias = torch.nn.parameter.Parameter(
                layernorm.bias.index_select(0, index))
        layernorm.normalized_shape = (len(index),)

    def prune_layer(layer, index, dim):
        layer = prune_linear_layer(layer, index, dim=dim)
        return layer

    if "hidden_z" in zs:
        hidden_zs = zs["hidden_z"]
        index = torch.LongTensor(hidden_zs.squeeze().nonzero().squeeze().tolist())
        index = index.to(model.device)

        # if max(index) >= model.encoder.embed_tokens.weight.shape[1] or max(index) >= model.decoder.embed_tokens.weight.shape[1]:
        #     print(f"There's something wrong. # index, {len(index)} >= encoder dim, {model.encoder.embed_tokens.weight.shape[1]}")
            # index = index[:model.encoder.embed_tokens.weight.shape[1]]
        model.encoder.embed_tokens.weight = torch.nn.parameter.Parameter(
            model.encoder.embed_tokens.weight.index_select(1, index).clone().detach())
        model.encoder.embed_tokens.embedding_dim = index.shape[0]

        for layer in range(model.config.num_layers):
            # encoder part
            #SA+LN
            if model.encoder.block[layer].layer[0].SelfAttention.q is not None:
                model.encoder.block[layer].layer[0].SelfAttention.q = \
                    prune_layer(model.encoder.block[layer].layer[0].SelfAttention.q, index, dim=1)
                model.encoder.block[layer].layer[0].SelfAttention.k = \
                    prune_layer(model.encoder.block[layer].layer[0].SelfAttention.k, index, dim=1)
            if model.encoder.block[layer].layer[0].SelfAttention.v is not None:
                model.encoder.block[layer].layer[0].SelfAttention.v = \
                    prune_layer(model.encoder.block[layer].layer[0].SelfAttention.v, index, dim=1)
                model.encoder.block[layer].layer[0].SelfAttention.o = \
                    prune_layer(model.encoder.block[layer].layer[0].SelfAttention.o, index, dim=0)
            prune_layer_norm(model.encoder.block[layer].layer[0].layer_norm, index)
            
            # FF + LN
            if hasattr(model.encoder.block[layer].layer[1].DenseReluDense, "wi"):
                if model.encoder.block[layer].layer[1].DenseReluDense.wi is not None:
                    model.encoder.block[layer].layer[1].DenseReluDense.wi = \
                        prune_layer(model.encoder.block[layer].layer[1].DenseReluDense.wi, index, dim=1)
                    model.encoder.block[layer].layer[1].DenseReluDense.wo = \
                        prune_layer(model.encoder.block[layer].layer[1].DenseReluDense.wo, index, dim=0)
                        
            elif hasattr(model.encoder.block[layer].layer[1].DenseReluDense, "wi_0"):
                if model.encoder.block[layer].layer[1].DenseReluDense.wi_0 is not None:
                    model.encoder.block[layer].layer[1].DenseReluDense.wi_0 = \
                        prune_layer(model.encoder.block[layer].layer[1].DenseReluDense.wi_0, index, dim=1)
                    model.encoder.block[layer].layer[1].DenseReluDense.wi_1 = \
                        prune_layer(model.encoder.block[layer].layer[1].DenseReluDense.wi_1, index, dim=1)
                    model.encoder.block[layer].layer[1].DenseReluDense.wo = \
                        prune_layer(model.encoder.block[layer].layer[1].DenseReluDense.wo, index, dim=0)
            else:
                NotImplementedError
            prune_layer_norm(model.encoder.block[layer].layer[1].layer_norm, index)

        for layer in range(model.config.num_decoder_layers):
            # decoder part
            #SA + LN
            if model.decoder.block[layer].layer[0].SelfAttention.q is not None:
                model.decoder.block[layer].layer[0].SelfAttention.q = \
                    prune_layer(model.decoder.block[layer].layer[0].SelfAttention.q, index, dim=1)
                model.decoder.block[layer].layer[0].SelfAttention.k = \
                    prune_layer(model.decoder.block[layer].layer[0].SelfAttention.k, index, dim=1)
            if model.decoder.block[layer].layer[0].SelfAttention.v is not None:
                model.decoder.block[layer].layer[0].SelfAttention.v = \
                    prune_layer(model.decoder.block[layer].layer[0].SelfAttention.v, index, dim=1)
                model.decoder.block[layer].layer[0].SelfAttention.o = \
                    prune_layer(model.decoder.block[layer].layer[0].SelfAttention.o, index, dim=0)
            prune_layer_norm(model.decoder.block[layer].layer[0].layer_norm, index)
            
            # CA + LN
            if model.decoder.block[layer].layer[1].EncDecAttention.q is not None:
                model.decoder.block[layer].layer[1].EncDecAttention.q = \
                    prune_layer(model.decoder.block[layer].layer[1].EncDecAttention.q, index, dim=1)
                model.decoder.block[layer].layer[1].EncDecAttention.k = \
                    prune_layer(model.decoder.block[layer].layer[1].EncDecAttention.k, index, dim=1)
            if model.decoder.block[layer].layer[1].EncDecAttention.v is not None:
                model.decoder.block[layer].layer[1].EncDecAttention.v = \
                    prune_layer(model.decoder.block[layer].layer[1].EncDecAttention.v, index, dim=1)
                model.decoder.block[layer].layer[1].EncDecAttention.o = \
                    prune_layer(model.decoder.block[layer].layer[1].EncDecAttention.o, index, dim=0)
            prune_layer_norm(model.decoder.block[layer].layer[1].layer_norm, index)
            
            # FF + LN
            if hasattr(model.decoder.block[layer].layer[2].DenseReluDense, "wi"):
                if model.decoder.block[layer].layer[2].DenseReluDense.wi is not None:
                    model.decoder.block[layer].layer[2].DenseReluDense.wi = \
                        prune_layer(model.decoder.block[layer].layer[2].DenseReluDense.wi, index, dim=1)
                    model.decoder.block[layer].layer[2].DenseReluDense.wo = \
                        prune_layer(model.decoder.block[layer].layer[2].DenseReluDense.wo, index, dim=0)
                        
            if hasattr(model.decoder.block[layer].layer[2].DenseReluDense, "wi_0"):
                if model.decoder.block[layer].layer[2].DenseReluDense.wi_0 is not None:
                    model.decoder.block[layer].layer[2].DenseReluDense.wi_0 = \
                        prune_layer(model.decoder.block[layer].layer[2].DenseReluDense.wi_0, index, dim=1)
                    model.decoder.block[layer].layer[2].DenseReluDense.wi_1 = \
                        prune_layer(model.decoder.block[layer].layer[2].DenseReluDense.wi_1, index, dim=1)
                    model.decoder.block[layer].layer[2].DenseReluDense.wo = \
                        prune_layer(model.decoder.block[layer].layer[2].DenseReluDense.wo, index, dim=0)
            prune_layer_norm(model.decoder.block[layer].layer[2].layer_norm, index)
            
        prune_layer_norm(model.encoder.final_layer_norm, index)
        prune_layer_norm(model.decoder.final_layer_norm, index)
        # accommodate for different models
        if hasattr(model, "classifier"):
            if hasattr(model.classifier, "dense"):
                model.classifier.dense = prune_linear_layer(model.classifier.dense, index, dim=1)
        if hasattr(model, "cls"):
            if hasattr(model.cls, "dense"):
                model.cls.dense = prune_linear_layer(model.classifier.dense, index, dim=1)
        if hasattr(model, "qa_outputs"):
            model.qa_outputs = prune_linear_layer(model.qa_outputs, index, dim=1)
        if hasattr(model, "lm_head"):
            model.lm_head = prune_linear_layer(model.lm_head, index, dim=1)
        if getattr(model, "layer_transformation", None) is not None:
            model.layer_transformation = prune_linear_layer(model.layer_transformation, index, dim=1)
            print("layer transformation", model.layer_transformation.weight.shape)
        if getattr(model, "mha_layer_transformation", None) is not None:
            model.mha_layer_transformation = prune_linear_layer(model.mha_layer_transformation, index, dim=1)
            print("layer mha_layer_transformation", model.mha_layer_transformation.weight.shape)

    if kept_intermediate_dims is not None:
        prune_intermediate_layers_t5(model, kept_intermediate_dims, enc=True)
    if dec_kept_intermediate_dims is not None:
        prune_intermediate_layers_t5(model, dec_kept_intermediate_dims, enc=False)

    for layer in range(model.config.num_layers):
        print("Enc_Layer:", layer)
        if model.encoder.block[layer].layer[0].SelfAttention.q is not None:
            print("query:", model.encoder.block[layer].layer[0].SelfAttention.q.weight.shape)
            print("key:", model.encoder.block[layer].layer[0].SelfAttention.k.weight.shape)
        else:
            print("query:", None)
            print("key:", None)
        if model.encoder.block[layer].layer[0].SelfAttention.v is not None:
            print("value:", model.encoder.block[layer].layer[0].SelfAttention.v.weight.shape)
            print("output:", model.encoder.block[layer].layer[0].SelfAttention.o.weight.shape)
        else:
            print("value:", None)
            print("output:", None)
        if hasattr(model.encoder.block[layer].layer[1].DenseReluDense, "wi"):
            if model.encoder.block[layer].layer[1].DenseReluDense.wi is not None:
                print("up:", model.encoder.block[layer].layer[1].DenseReluDense.wi.weight.shape)
                print("down:", model.encoder.block[layer].layer[1].DenseReluDense.wo.weight.shape)
        if hasattr(model.encoder.block[layer].layer[1].DenseReluDense, "wi_0"):
            if model.encoder.block[layer].layer[1].DenseReluDense.wi_0 is not None:
                print("up1:", model.encoder.block[layer].layer[1].DenseReluDense.wi_0.weight.shape)
                print("up2:", model.encoder.block[layer].layer[1].DenseReluDense.wi_1.weight.shape)
                print("down:", model.encoder.block[layer].layer[1].DenseReluDense.wo.weight.shape)
        else:
            print("up", None)
            print("down", None)

    for layer in range(model.config.num_decoder_layers):
        print("Dec_Layer:", layer)
        # SA
        if model.decoder.block[layer].layer[0].SelfAttention.q is not None:
            print("SA_query:", model.decoder.block[layer].layer[0].SelfAttention.q.weight.shape)
            print("SA_key:", model.decoder.block[layer].layer[0].SelfAttention.k.weight.shape)
        else:
            print("SA_query:", None)
            print("SA_key:", None)
        if model.decoder.block[layer].layer[0].SelfAttention.v is not None:
            print("SA_value:", model.decoder.block[layer].layer[0].SelfAttention.v.weight.shape)
            print("SA_out:", model.decoder.block[layer].layer[0].SelfAttention.o.weight.shape)
        else:
            print("SA_value:", None)
            print("SA_out:", None)
        
        #CA
        if model.decoder.block[layer].layer[1].EncDecAttention.q is not None:
            print("CA_query:", model.decoder.block[layer].layer[1].EncDecAttention.q.weight.shape)
            print("CA_key:", model.decoder.block[layer].layer[1].EncDecAttention.k.weight.shape)
        else:
            print("CA_query:", None)
            print("CA_key:", None)
        if model.decoder.block[layer].layer[1].EncDecAttention.v is not None:
            print("CA_value:", model.decoder.block[layer].layer[1].EncDecAttention.v.weight.shape)
            print("CA_output:", model.decoder.block[layer].layer[1].EncDecAttention.o.weight.shape)
        else:
            print("CA_value:", None)
            print("CA_output:", None)
        
        # FF
        if hasattr(model.decoder.block[layer].layer[2].DenseReluDense, "wi"):
            if model.decoder.block[layer].layer[2].DenseReluDense.wi is not None:
                print("up:", model.decoder.block[layer].layer[2].DenseReluDense.wi.weight.shape)
                print("down:", model.decoder.block[layer].layer[2].DenseReluDense.wo.weight.shape)
        if hasattr(model.decoder.block[layer].layer[2].DenseReluDense, "wi_0"):
            if model.decoder.block[layer].layer[2].DenseReluDense.wi_0 is not None:
                print("up_1:", model.decoder.block[layer].layer[2].DenseReluDense.wi_0.weight.shape)
                print("up_2:", model.decoder.block[layer].layer[2].DenseReluDense.wi_1.weight.shape)
                print("down:", model.decoder.block[layer].layer[2].DenseReluDense.wo.weight.shape)
        else:
            print("up", None)
            print("down", None)