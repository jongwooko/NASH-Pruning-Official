import torch
from datasets import load_metric
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score
import numpy as np
from collections import Counter, OrderedDict
import string
import re

import abc
import numpy as np
import nltk
import datasets
import evaluate

def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1)/2
    }

def multiclass_acc_and_f1(preds, labels):		
    acc = simple_accuracy(preds, labels)		
    f1 = f1_score(y_true=labels, y_pred=preds, average='macro')		
    return {		    
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1)/2
    }

def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }

def em_and_f1(preds, labels): # labels : list of lists
    em = np.mean([max(p == t for t in t_list) for p, t_list in zip(preds, labels)])
    f1 = np.mean([max(qa_f1_score(p,t) for t in t_list) for p, t_list in zip(preds, labels)])
    return {
        "em": em,
        "f1": f1,
    }

def bleu_meteor_rougeL(preds, labels):
    bleu = evaluate.load("bleu")
    meteor = evaluate.load("meteor")
    rouge_score = evaluate.load("rouge")
    try:
        bleu_score=bleu.compute(predictions=preds,references=labels)['bleu']
    except:
        bleu_score=0.0
    return {
        "bleu": bleu_score,
        "meteor": meteor.compute(predictions=preds,references=labels)['meteor'],
        "rougeL": rouge_score.compute(predictions=preds,references=labels)['rougeL'],
    }


def rouge(preds, labels):
    rouge_score = load_metric("rouge")
    scores = rouge_score.compute(predictions=preds, references=labels, use_stemmer=True)
    return {
        'rouge1': scores['rouge1'].mid.fmeasure,
        'rouge2': scores['rouge2'].mid.fmeasure,
        'rougeL': scores['rougeL'].mid.fmeasure,
        'rougeLsum': scores['rougeLsum'].mid.fmeasure,
    }

def task_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)

    if task_name in ["rte", "qnli"] :
        preds = np.array([1 if pred=='not_entailment' else 0 for pred in preds])
        labels = np.array([1 if gt=='not_entailment' else 0 for gt in labels])
    elif task_name =="cola":
        preds = [1 if pred=='acceptable' else 0 for pred in preds]
        labels = [1 if gt=='acceptable' else 0 for gt in labels]
    elif task_name in ['mrpc']:
        mapping = {'equivalent':1, 'not_equivalent':0}
        preds = np.array([1 if pred=='equivalent' else 0 for pred in preds])
        labels = np.array([1 if gt=='equivalent' else 0 for gt in labels])
    elif task_name in ['stsb']:
        preds_list = []
        labels = [float(gt) for gt in labels]
        for pred in preds:
            try: preds_list.append(float(pred))
            except: preds_list.append(0.0)
        preds = preds_list
    elif task_name == 'sst2':
        preds = np.array([1 if pred=='positive' else 0 for pred in preds])
        labels = np.array([1 if gt=='positive' else 0 for gt in labels])
    elif task_name == 'mnli':
        mapping = {'entailment':1, 'contradiction':2, 'neutral':0}
        preds = np.array([mapping[pred] if pred in mapping.keys() else -1 for pred in preds])
        labels = np.array([mapping[gt] for gt in labels])
    elif task_name == 'qqp':
        preds = np.array([1 if pred=='duplicate' else 0 for pred in preds])
        labels = np.array([1 if gt=='duplicate' else 0 for gt in labels])
    elif task_name == ['squad', 'tweetqa', 'narrativeqa']:
        preds = [_normalize_answer(pred, punc_chars=string.punctuation, punc_repl="") for pred in preds]
        labels = [[_normalize_answer(true, punc_chars=string.punctuation, punc_repl="") for true in true_list] for true_list in labels]
    elif task_name == 'cnndm':
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
#         preds = [_normalize_answer(pred, punc_chars=string.punctuation, punc_repl="") for pred in preds]
#         labels = [_normalize_answer(true, punc_chars=string.punctuation, punc_repl="") for true in labels]
    elif task_name == 'cb':	# superGLUE
        mapping = {'entailment':1, 'contradiction':2, 'neutral':0}
        preds = np.array([mapping[pred] if pred in mapping.keys() else -1 for pred in preds])
        labels = np.array([mapping[gt] for gt in labels])  		      
    elif task_name == 'copa':		        
        preds = np.array([1 if (pred=='choice2' or pred=='1') else 0 for pred in preds])
        labels = np.array([1 if (gt=='choice2' or gt=='1') else 0 for gt in labels]) 
    elif task_name in ['wsc.fixed', 'wic', 'boolq']:		
        preds = np.array([1 if (pred=='True' or pred=='true') else 0 for pred in preds])		
        labels = np.array([1 if (gt=='True' or gt=='true') else 0 for gt in labels])    		
    elif task_name == 'record':		
        preds = [_normalize_answer(pred, punc_chars=string.punctuation, punc_repl="") for pred in preds]		
        labels = [[_normalize_answer(true, punc_chars=string.punctuation, punc_repl="") for true in true_list] for true_list in labels]
    else:
        pass 

    if task_name == "cola":
        return {"mcc": matthews_corrcoef(labels, preds)}
    elif task_name == "sst2":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mrpc":
        return acc_and_f1(preds, labels)
    elif task_name == "stsb":
        return pearson_and_spearman(preds, labels)
    elif task_name == "qqp":
        return acc_and_f1(preds, labels)
    elif task_name == "mnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mnli-mm":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "qnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "rte":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "wnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "squad":
        return em_and_f1(preds, labels)
    elif task_name == "cnndm":
        return rouge(preds, labels)
    elif task_name == "samsum":
        return rouge(preds, labels)
    elif task_name == "xsum":
        return rouge(preds, labels)
    elif task_name == 'cb':
        return multiclass_acc_and_f1(preds, labels)
    elif task_name == "copa":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "boolq":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "wic":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "wsc.fixed":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "wsc.fixed":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "orangesum":
        return rouge(preds, labels)
    elif task_name == "tweetqa":
        return bleu_meteor_rougeL(preds, labels)
    elif task_name == "narrativeqa":
        return bleu_meteor_rougeL(preds, labels)
    elif task_name == "dolly":
        return rouge(preds, labels)
    else:
        raise KeyError(task_name)
        
def result_to_file(result, file_name, logger):
    with open(file_name, "a") as writer:
        logger.info("***** Eval results *****")
        writer.write("")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

def _normalize_answer(text, punc_chars, punc_repl):
  """Lower text and remove punctuation, articles and extra whitespace."""

  def remove_articles(s):
    return re.sub(r"\b(a|an|the)\b", " ", s)

  def replace_punctuation(s):
    to_replace = set(punc_chars)
    return "".join(punc_repl if ch in to_replace else ch for ch in s)

  def white_space_fix(s):
    return " ".join(s.split())

  text = text.lower()
  text = replace_punctuation(text)
  text = remove_articles(text)
  text = white_space_fix(text)

  return text

def qa_f1_score(target, prediction):
    """Computes token f1 score for a single target and prediction."""
    prediction_tokens = prediction.split()
    target_tokens = target.split()
    common = (Counter(prediction_tokens) & Counter(target_tokens))
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(target_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1



    import abc
from collections import OrderedDict
import numpy as np

"""Defines functions to process the outputs to make them ready for the evaluation."""


def string_to_float(string, default=-1., **unused_kwargs):
    """Converts string to float, using default when conversion not possible."""
    try:
        return float(string)
    except ValueError:
        return default


class PostProcessor(abc.ABC):
    """Postprocess the predictions and labels to make them suitable for
    evaluation."""

    def __init__(self, tokenizer, ignore_pad_token_for_loss):
        self.tokenizer = tokenizer
        self.ignore_pad_token_for_loss = ignore_pad_token_for_loss

    def process(self, preds, labels, data_info=None):
        # process preds
        if isinstance(preds, tuple):
            preds = preds[0]
        preds = np.where(preds != -100, preds,
                              self.tokenizer.pad_token_id)
        decoded_preds = self.tokenizer.batch_decode(
            preds, skip_special_tokens=True)
        decoded_preds = [pred.strip() for pred in decoded_preds]

        # process labels
        if isinstance(labels, list):
            decoded_labels = labels
        else:
            labels = labels.astype('int') 
            if self.ignore_pad_token_for_loss:
                # Replace -100 in the labels as we can't decode them.
                labels = np.where(labels != -100, labels,
                                self.tokenizer.pad_token_id)
            decoded_labels = self.tokenizer.batch_decode(
                labels, skip_special_tokens=True)
            decoded_labels = [label.strip() for label in decoded_labels]
        
        return decoded_preds, decoded_labels


class MultiRC(PostProcessor):
    def process(self, preds, labels, data_info):
        preds, labels = super().process(preds, labels, data_info)
        preds = np.array([1 if pred=='True' else 0 for pred in preds])		
        labels = np.array([1 if gt=='True' else 0 for gt in labels]) 
        preds = [{"idx": info["idx"], "prediction": pred}
                 for info, pred in zip(data_info, preds)]
        return preds, labels


class Record(PostProcessor):
    def process(self, preds, labels, data_info):
        preds, labels = super().process(preds, labels, data_info)
        labels = [info["answers"] for info in data_info]
        return preds, labels


POSTPROCESSOR_MAPPING = OrderedDict(
    [
        ('RECORD', Record),
        ('MULTIRC', MultiRC)
    ]
)


class AutoPostProcessor:
    @classmethod
    def get(self, task, tokenizer, ignore_pad_token_for_loss):
        task = task.split('_')[0]
        if task in POSTPROCESSOR_MAPPING:
            return POSTPROCESSOR_MAPPING[task](tokenizer, ignore_pad_token_for_loss)
        return PostProcessor(tokenizer, ignore_pad_token_for_loss)


