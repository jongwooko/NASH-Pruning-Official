import inspect
import os
import pdb
import random
import sys
import time
import os

import datasets
import numpy as np
import torch
import transformers
from datasets import load_dataset, load_metric
from matplotlib import pyplot as plt
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SequentialSampler
from transformers import AutoTokenizer, EvalPrediction, GlueDataset
from transformers import GlueDataTrainingArguments as DataTrainingArguments
from transformers.data.data_collator import (DataCollator,
                                             DataCollatorWithPadding,
                                             default_data_collator)
from transformers.trainer_pt_utils import nested_concat, nested_numpify
from transformers.trainer_utils import EvalPrediction
from transformers import T5Config, T5Tokenizer, BartConfig

from models.deploying_bart import BartForConditionalGeneration
from utils.nash_utils_bart import *
from utils.utils import *
from tqdm.auto import tqdm


# task_to_keys = {
#     "cola": ("sentence", None),
#     "mnli": ("premise", "hypothesis"),
#     "mrpc": ("sentence1", "sentence2"),
#     "qnli": ("question", "sentence"),
#     "qqp": ("question1", "question2"),
#     "rte": ("sentence1", "sentence2"),
#     "sst2": ("sentence", None),
#     "stsb": ("sentence1", "sentence2"),
#     "wnli": ("sentence1", "sentence2"),
#     "cb": ("premise", "hypothesis"),
#     "copa": ("premise", "choice1", "choice2", "question"),
#     "multirc": ("paragraph", "question"),
#     "wic": ("word1", "word2", "sentence1", "sentence2"),
#     "wsc.fixed": ("span1_text", "span2_text", "text"),
#     "boolq": ("passage", "question"),
#     "record": ("passage", "query"),
#     "tweetqa": ("sentence", None),
#     "narrativeqa": ("sentence", None)
# }

target_length = {"cola": 5, "mnli": 5, "mnli-mm": 5, "mrpc": 6,
        "sst2": 3, "stsb": 4, "qqp": 7, "qnli": 6, "rte": 6,
        "squad":20, "cnndm":150, "samsum": 150, "cb": 6, "copa": 5, "boolq": 5, "wic": 5, "wsc.fixed": 5, "multirc": 5, "record": 150, "tweetqa":20, "narrativeqa": 100}

glue_task=["cola", "mnli", "mnli-mm", "mrpc", "sst2", "stsb", "qqp", "qnli", "rte"]
nlu_task = glue_task + ["cb", "copa", "boolq", "wic", "wsc.fixed", "multirc"]

def _remove_unused_columns(dataset: "datasets.Dataset", description):
    # Inspect model forward signature to keep only the arguments it accepts.
    signature = inspect.signature(model.forward)
    signature_columns = list(signature.parameters.keys())
    # Labels may be named label or label_ids, the default data collator handles that.
    signature_columns += ["label", "label_ids"]
    columns = [k for k in signature_columns if k in dataset.column_names]
    ignored_columns = list(set(dataset.column_names) - set(signature_columns))
    dset_description = "" if description is None else f"in the {description} set "
    print(
        f"The following columns {dset_description} don't have a corresponding argument in `{model.__class__.__name__}.forward` and have been ignored: {', '.join(ignored_columns)}."
    )
    dataset.set_format(type=dataset.format["type"], columns=columns)


def get_dataloader(dataset, batch_size):
    dataloader = DataLoader(dataset,
                            sampler=SequentialSampler(dataset),
                            batch_size=batch_size,
                            collate_fn=default_data_collator)
    return dataloader

def post_processing_function(examples, features, predictions):
    # Post-processing: we match the start logits and end logits to answers in the original context.
    predictions = postprocess_qa_predictions(
        examples=examples,
        features=features,
        predictions=predictions,
    )
    # Format the result to the format the metric expects.
    formatted_predictions = [{"id": k, "prediction_text": v}
                             for k, v in predictions.items()]
    references = [{"id": ex["id"], "answers": ex[answer_column_name]}
                  for ex in datasets["validation"]]
    return EvalPrediction(predictions=formatted_predictions, label_ids=references)


def evaluate(model):
    metrics = {}
    total_infer_times = 0
    
    t = 2 if task_name in ["squad", "squad_v2", "cnndm", "record", "samsum", "tweetqa", "narrativeqa", "mnli", "qqp"] else 5
    if task_name in ["rte", "stsb", "cola", "mrpc", "cb", "copa", "boolq", "wic", "wsc.fixed"]:
        t = 20
    assert t > 1

    all_labels = dataset['target_list'] if 'target_list' in dataset.column_names else dataset['target']
    _remove_unused_columns(dataset, "evaluation")

    preds = None
    preds_host = None
    all_preds = None

    gen_kwargs = {
                "max_length": target_length[task_name],
                "num_beams": model.config.num_beams,
                "length_penalty": model.config.length_penalty, # TODO: add for CNNDM
            }
    
    for i in range(t):
        print(f"Round {i}: There are {len(dataloader)} batches in the dataset.")
        
        for num_batch, inputs in enumerate(tqdm(dataloader)):
            for key in inputs:
                inputs[key] = inputs[key].cuda()
            with torch.no_grad():

                # target_max_length = inputs['labels'].shape[1] 
                preds = model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    **gen_kwargs,
                )
                
                torch.cuda.synchronize()

                if preds is not None:
                    preds_host = preds if preds_host is None else nested_concat(
                        preds_host, preds, padding_index=0)

        if i == 0 and preds_host is not None:
            preds = nested_numpify(preds_host)
            all_preds = preds if all_preds is None else nested_concat(
                all_preds, preds, padding_index=0)

            decoded_preds = tokenizer.batch_decode(all_preds, skip_special_tokens=True)
            decoded_preds = [pred.strip() for pred in decoded_preds]
        
    metrics = compute_metrics(decoded_preds, all_labels, tokenizer, task_name)
    metrics["num_examples"] = len(all_labels)
    metrics["t"] = t
    return metrics

def prepare_validation_features(examples):
    # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    max_length = 384
    doc_stride = 128
    tokenized_examples = tokenizer(
        examples[question_column_name if pad_on_right else context_column_name],
        examples[context_column_name if pad_on_right else question_column_name],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length"
    )

    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

    # For evaluation.py, we will need to convert our predictions to substrings of the context, so we keep the
    # corresponding example_id and we will store the offset mappings.
    tokenized_examples["example_id"] = []

    for i in range(len(tokenized_examples["input_ids"])):
        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)
        context_index = 1 if pad_on_right else 0

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        tokenized_examples["example_id"].append(examples["id"][sample_index])

        # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
        # position is part of the context or not.
        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == context_index else None)
            for k, o in enumerate(tokenized_examples["offset_mapping"][i])
        ]

    return tokenized_examples

def preprocess_function(examples):
    # Tokenize the texts
    max_seq_length = 128
    padding = "max_length"
    args = ((examples['source']),)
    result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)
    return result

def preprocess_function_generation(examples):
    # Tokenize the texts
    max_seq_length = 128 # 512
    padding = "max_length"
    args = ((examples['source']),)
    result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)
    return result

def warmup():
    time1 = time.time()
    input = torch.randn(128, 1024).cuda()
    linear = torch.nn.Linear(1024, 1024).cuda()
    for i in range(10000):
        input = linear(input)

    time2 = time.time()
    print(round(time2 - time1, 2), "seconds for warmup")

def compute_metrics(decoded_preds, all_labels, tokenizer, task_name):    
    from utils.metrics import task_metrics
    result = task_metrics(task_name, decoded_preds, all_labels)
    if len(result) > 1:
        result["combined_score"] = np.mean(list(result.values())).item()
    return result

if __name__ == '__main__':
    # warmup
    warmup()
    
    # data
    task_name = sys.argv[1].lower()
    model_name_or_path = sys.argv[2]
    early_exit_layer = int(sys.argv[4]) if sys.argv[4] != 'None' else None
    bs = 32
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, use_fast=True if task_name == "squad" else False, padding_side="right", truncation_size="right")
    except:
        tokenizer = AutoTokenizer.from_pretrained(
            sys.argv[3], use_fast=True if task_name == "squad" else False, padding_side="right", truncation_size="right")
        
    if task_name in nlu_task:
        if task_name == "mnli":
            set_name = "validation_matched"
        else:
            set_name = "validation"
        if task_name in glue_task: # glue
            dataset = datasets.load_dataset("glue", task_name)[set_name]
        else: # superglue
            dataset = datasets.load_dataset("super_glue", task_name)[set_name]
        from data.t5_format import task2format, map_dataset
        dataset = map_dataset(dataset, task2format[task_name])
        dataset = dataset.map(
            preprocess_function,
            batched=True,
            desc="Running tokenizer on dataset",
        ) #! get dataset        

    elif task_name in ['samsum', 'narrativeqa']:
        dataset = datasets.load_dataset(f"./data/{task_name}.py")["validation"]
        from data.t5_format import task2format, map_dataset
        dataset = map_dataset(dataset, task2format[task_name])
        dataset = dataset.map(
            preprocess_function_generation,
            batched=True,
            desc="Running tokenizer on dataset",
            load_from_cache_file=False
        )
        
    elif task_name == 'cnndm':
        dataset = datasets.load_dataset("./data/cnn_dailymail.py", "3.0.0")["validation"]
        from data.t5_format import task2format, map_dataset
        dataset = map_dataset(dataset, task2format[task_name])
        dataset = dataset.map(
            preprocess_function_generation,
            batched=True,
            desc="Running tokenizer on dataset",
            load_from_cache_file=False
        )
        
    elif task_name == 'record':
        dataset = datasets.load_dataset("super_glue", task_name)["validation"]
        from data.t5_format import task2format, map_dataset
        dataset = map_dataset(dataset, task2format[task_name])
        dataset = dataset.map(
            preprocess_function_generation,
            batched=True,
            desc="Running tokenizer on dataset",
            load_from_cache_file=False
        )
    elif task_name =="tweetqa":
        dataset = datasets.load_dataset("./data/tweet_qa.py")["validation"]
        from data.t5_format import task2format, map_dataset
        dataset = map_dataset(dataset, task2format[task_name])
        dataset = dataset.map(
            preprocess_function_generation,
            batched=True,
            desc="Running tokenizer on dataset",
            load_from_cache_file=False
        )        
    else:
        raise NotImplementedError
        
    dataloader = get_dataloader(dataset, bs)

    # load model
    model_class = BartForConditionalGeneration
    zs = load_zs(model_name_or_path)
    
    config = BartConfig.from_pretrained(model_name_or_path)
    
    # for full models
    if not hasattr(config, "dec_cross_pruned_heads") and zs is None:
        model = model_class.from_pretrained(model_name_or_path)
    
    # for compressed models
    elif zs is None:
        zs_path = '/'.join(model_name_or_path.split('/')[:-2])
        zs = load_zs(zs_path)
        model = load_model(zs_path, model_class, zs)
        model_path = os.path.join(model_name_or_path, 'pytorch_model.bin')
        trained_weight = torch.load(model_path, map_location='cpu')
        model.load_state_dict(trained_weight)
        
    else:
        model = load_model(model_name_or_path, model_class, zs)
        
    if early_exit_layer is not None:
        model.model.config.decoder_early_exit = True
        model.model.config.early_exit_layer = early_exit_layer

        model.model.decoder.config.decoder_early_exit = True
        model.model.decoder.config.early_exit_layer = early_exit_layer

    model = model.cuda()
    model = model.eval()

    model.config.output_hidden_states = False
    model.config.output_attentions = False
    if task_name == "cnndm":
        model.config.num_beams = 4
        model.config.length_penalty = 0.6

    metrics = evaluate(model)
    model_size = calculate_parameters(model)
    full_model_size = calculate_parameters(model_class(model.config))
    sparsity = 1 - round(model_size / full_model_size, 3)
    
    print(f"Task: {task_name}")
    print(f"Model path: {model_name_or_path}")
    print(f"Model size: {model_size}")
    print(f"Sparsity: {sparsity}")

    total_block_times = 0.0
    for i in range(len(model.model.decoder.layers)):
        total_block_times += model.model.decoder.layers[i].block_times
    enc_block_times = 0.0
    for i in range(len(model.model.encoder.layers)):
        total_block_times += model.model.encoder.layers[i].block_times
    
    metrics['milliseconds/example'] = total_block_times / metrics['num_examples'] * 1e3
    metrics['milliseconds/example'] /= metrics["t"]
    metrics.pop("t")
    for key in metrics:
        print(f"{key}: {round(metrics[key], 6 if 'seconds' in key else 4)}")