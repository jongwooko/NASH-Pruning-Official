import logging
import os
import sys
import time
import random
from copy import deepcopy

import datasets
import numpy as np
import torch
import transformers
from datasets import load_dataset, load_metric, DatasetDict
from transformers import AutoConfig, AutoTokenizer, EvalPrediction, default_data_collator, DataCollatorWithPadding
from transformers import (HfArgumentParser, TrainingArguments, PretrainedConfig,
                          glue_output_modes, glue_tasks_num_labels, set_seed)

from args import AdditionalArguments, DataTrainingArguments
from utils.nash_utils import *
from models.l0_module import L0Module, L0Module_Bart
from models.modeling_t5 import NashT5ForConditionalGeneration
from models.modeling_bart import BartForConditionalGeneration
from trainer.trainer import NashTrainer 
from utils.utils import *
from models.model_args import ModelArguments
import wandb
from utils.metrics import AutoPostProcessor

from datasets import set_caching_enabled
set_caching_enabled(False)

output_modes = {
    "cola": "classification", "mnli": "classification", "mrpc": "classification",
    "sst2": "classification", "stsb": "regression", "qqp": "classification",
    "qnli": "classification", "rte": "classification", "squad": "generation", 
    "squad_v2": "generation", "cnndm": "generation", "samsum": "generation", "xsum": "generation",
    "cb": "classification", "copa": "classification", "wic": "classification",  # SuperGLUE
    "boolq": "classification", "ax": "classification", "wsc.fixed": "classification",
    "record": "generation", "multirc": "generation", "orangesum": "generation",
    "tweetqa": "generation", "narrativeqa": "generation", "dolly": "generation"
}

glue_task = ["cola", "mnli", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb"]
sglue_task = ['rte', 'cb', 'copa', 'wic', 'wsc.fixed', 'multirc', 'record', 'boolq'] # superglue

logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments, AdditionalArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, additional_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, additional_args = parser.parse_args_into_dataclasses()
    
    os.makedirs(training_args.output_dir, exist_ok=True)

     # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    training_args.fp16 = True
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # save args
    torch.save(data_args, os.path.join(
        training_args.output_dir, "data_args.bin"))
    torch.save(model_args, os.path.join(
        training_args.output_dir, "model_args.bin"))
    torch.save(additional_args, os.path.join(
        training_args.output_dir, "additional_args.bin"))

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # print all arguments
    log_all_parameters(logger, model_args, data_args,
                       training_args, additional_args)
    
    t_name = None
    if data_args.task_name is not None:
        # Downloading and loading a dataset from the hub.
        if data_args.task_name in glue_task:
            raw_datasets = load_dataset(
                "./data/glue.py", data_args.task_name.replace("-", ""), cache_dir=model_args.cache_dir)
        elif data_args.task_name in sglue_task:
            raw_datasets = load_dataset("./data/sglue.py", data_args.task_name.replace("-", ""), cache_dir=model_args.cache_dir)
        elif data_args.task_name == "squad":
            raw_datasets = load_dataset("./data/squad.py", cache_dir=model_args.cache_dir)
        elif data_args.task_name == "squad_v2":
            raw_datasets = load_dataset("./data/squad_v2.py", cache_dir=model_args.cache_dir)
        elif data_args.task_name == "cnndm":
            raw_datasets = load_dataset("./data/cnn_dailymail.py", "3.0.0")
        elif data_args.task_name == "samsum":
            raw_datasets = load_dataset("./data/samsum.py")
        elif data_args.task_name == "xsum":
            raw_datasets = load_dataset("xsum")
        elif data_args.task_name == "narrativeqa":
            raw_datasets = load_dataset("./data/narrativeqa.py")
        elif data_args.task_name == "orangesum":
            raw_datasets = load_dataset('./data/orangesum.py', "abstract")
        elif data_args.task_name == "tweetqa":
            raw_datasets = load_dataset("./data/tweet_qa.py")
        elif data_args.task_name == "dolly":
            # split train/validation/test
            from datasets import Dataset
            raw_datasets = load_dataset("databricks/databricks-dolly-15k")
            raw_datasets['validation'] = Dataset.from_dict(raw_datasets['train'][14000:])
            raw_datasets['train'] = Dataset.from_dict(raw_datasets['train'][:14000])
        t_name = data_args.task_name
    else:
        raise NotImplementedError

    # Labels
    is_classification = output_modes[data_args.task_name] == "classification"
    if is_classification:
        label_list = raw_datasets["train"].features["label"].names
        num_labels = len(label_list)
    else:
        num_labels = 1

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=t_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    
    # set up configuration for distillation
    if additional_args.do_distill:
        config.output_attentions = True
        config.output_hidden_states = True
        
    if "t5" in model_args.model_name_or_path:
        Model = NashT5ForConditionalGeneration
        from utils.nash_utils import load_model, load_zs
    elif "bart" in model_args.model_name_or_path:
        Model = BartForConditionalGeneration
        config.num_decoder_layers = config.decoder_layers
        from utils.nash_utils_bart import load_model, load_zs
    else:
        NotImplementedError("the given model type is not supported")
    
    teacher_model = None
    if additional_args.do_distill:
        teacher_model = Model.from_pretrained(
            additional_args.distillation_path,
            config=deepcopy(config)
        )
        teacher_model.eval()
        
    if additional_args.pruning_method == "nash":
        assert additional_args.layer_selection is not None
        additional_args.encdec_pruning_type = "nash"
        num_selected_layers = additional_args.num_select_layers
        if additional_args.layer_selection == 'unif':
            import math
            selected_layer = [math.floor((config.num_decoder_layers - 1) / (num_selected_layers-1) * d) for d in range(num_selected_layers)]
        elif additional_args.layer_selection == 'high':
            selected_layer = [i for i in range(config.num_decoder_layers - 1, config.num_decoder_layers - num_selected_layers-1, -1)]
        elif additional_args.layer_selection == 'low':
            selected_layer = [i for i in range(num_selected_layers)]
        else:
            raise NotImplementedError
        config.selected_layer = selected_layer
    
    if additional_args.pruning_method == "auto_select":
        dec_sparsity = 1 - (additional_args.num_select_layers / config.num_decoder_layers)
        config.auto_select = dec_sparsity
        additional_args.encdec_pruning_type = "nash"
    
    config.do_layer_distill = additional_args.do_layer_distill #! True
    model_path = model_args.model_name_or_path if not additional_args.do_distill \
                        else additional_args.distillation_path
    model = Model.from_pretrained(
        model_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    ) #! inside the function, we get the original struct
    
    if additional_args.pruning_method == "nash":
        not_selected = sorted(set(range(model.config.num_decoder_layers)) - set(selected_layer))
        if "t5" in model_args.model_name_or_path:
            for i in reversed(not_selected):
                del model.decoder.block[i]
            config.num_decoder_layers = len(selected_layer)
            model.config.num_decoder_layers = len(selected_layer)
        elif "bart" in model_args.model_name_or_path:
            for i in reversed(not_selected):
                del model.model.decoder.layers[i]
            config.decoder_layers = len(selected_layer)
            config.num_decoder_layers = len(selected_layer)
            model.config.decoder_layers = len(selected_layer)
            
    # initialize the layer transformation matrix to be an identity matrix
    if additional_args.do_layer_distill:
        initialize_layer_transformation(model)
    logger.info(model)
    logger.info(f"Model size: {calculate_parameters(model)}")

    zs = None
    if additional_args.pretrained_pruned_model is not None:
        zs = load_zs(additional_args.pretrained_pruned_model)
        model = load_model(additional_args.pretrained_pruned_model, Model, zs)        
        print(
            f"Model Size after pruning: {calculate_parameters(model)}")
        
        if additional_args.pruning_method == "nash":
            model.config.selected_layer = selected_layer
            
    l0_module = None
    if (additional_args.pruning_type is not None) and ("t5" in model_args.model_name_or_path):
        l0_module = L0Module(config=config,
                             droprate_init=additional_args.droprate_init,
                             temperature=additional_args.temperature,
                             target_sparsity=additional_args.target_sparsity,
                             pruning_type=additional_args.pruning_type,
                             enc_dec=True,
                             encdec_pruning_type=additional_args.encdec_pruning_type,) # need to add additional_args
    
    elif (additional_args.pruning_type is not None) and ("bart" in model_args.model_name_or_path):
        l0_module = L0Module_Bart(config=config,
                             droprate_init=additional_args.droprate_init,
                             temperature=additional_args.temperature,
                             target_sparsity=additional_args.target_sparsity,
                             pruning_type=additional_args.pruning_type,
                             enc_dec=True,
                             encdec_pruning_type=additional_args.encdec_pruning_type,) # need to add additional_args

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    model.config.task_name = data_args.task_name
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and data_args.task_name is not None
        and is_classification
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            label_to_id = {i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)}
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )

    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in config.label2id.items()}
    elif data_args.task_name is not None and is_classification:
        model.config.label2id = {l: i for i, l in enumerate(label_list)}
        model.config.id2label = {id: label for label, id in config.label2id.items()}

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)
    from data.t5_format import task2format, map_dataset
    raw_datasets = map_dataset(raw_datasets, task2format[data_args.task_name])
    
    def preprocess_function(examples):
        # Tokenize the texts
        args = ((examples['source']),)
        result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)
        dec_args = ((examples['target']),)
        dec_result=tokenizer(*dec_args, padding=True)
        result['label'] = dec_result['input_ids']
        result['decoder_attention_mask'] = dec_result['attention_mask']
        return result
    
    def preprocess_function_generation(examples):
        # Tokenize the texts
        args = ((examples['source']),)
        result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)
        dec_args = ((examples['target']),)
        dec_result=tokenizer(*dec_args, padding=padding, max_length=max_target_length, truncation=True)
        result['label'] = dec_result['input_ids']
        result['decoder_attention_mask'] = dec_result['attention_mask']
        return result
    
    if output_modes[data_args.task_name] == "generation": # is_generation
        if data_args.task_name in ["squad", "squad_v2"]:
            max_target_length = 20
        elif data_args.task_name in ["cnndm", "samsum", "tweetqa", "xsum"]:
            try:
                max_target_length = model.config.task_specific_params['summarization']['max_length']
            except:
                max_target_length = 200
        elif data_args.task_name == "multirc":
            max_target_length = 5
        elif data_args.task_name == "record":
            max_target_length = 150
            config.num_beams = 4
            config.length_penalty = 0.6
        elif data_args.task_name in ["orangesum", "narrativeqa"]:
            max_target_length = 100
        elif data_args.task_name == "dolly":
            max_target_length = 256
        with training_args.main_process_first(desc="dataset map pre-processing"): # tokenize
            raw_datasets = raw_datasets.map(
                preprocess_function_generation,
                batched=True,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )
            
    else:
        with training_args.main_process_first(desc="dataset map pre-processing"):
            raw_datasets = raw_datasets.map(
                preprocess_function,
                batched=True,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on dataset",
            ) #! get dataset
    
    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))

    if training_args.do_eval:
        if "validation" not in raw_datasets and "validation_matched" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation_matched" if data_args.task_name == "mnli" else "validation"]
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))

    # Log a few random samples from the training set:
    if training_args.do_train:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.

    def compute_metrics(predictions, label_ids, tokenizer, additional_args):
        # preds, labels, data_info = eval_preds
        post_processor = AutoPostProcessor.get(additional_args.ex_name, tokenizer,
                                               True)
        decoded_preds, decoded_labels = post_processor.process(
            predictions, label_ids, additional_args.ex_name)

        from utils.metrics import task_metrics
        result = task_metrics(data_args.task_name, decoded_preds, decoded_labels)
        if len(result) > 1:
            result["combined_score"] = np.mean(list(result.values())).item()
        return result

    # Data collator will default to DataCollatorWithPadding when the tokenizer is passed to Trainer, so we change it if
    # we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None
    

    logger.info(
        f"************* {len(train_dataset)} Training Examples Loaded *************")
    logger.info(
        f"************* {len(eval_dataset)} Evaluation Examples Loaded *************")

    trainer = NashTrainer(
        model=model,
        args=training_args,
        additional_args=additional_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
        l0_module=l0_module,
        teacher_model=teacher_model
    )

    if training_args.do_train:
        trainer.train()
        tokenizer.save_pretrained(training_args.output_dir)
        if trainer.start_saving_best:
            tokenizer.save_pretrained(os.path.join(training_args.output_dir, "best"))

if __name__ == "__main__":
    os.environ["WANDB_DISABLED"] = "true"
    t_start = time.time()
    main()
    t_end = time.time()
    logger.info(f"Training took {round(t_end - t_start, 2)} seconds.")
