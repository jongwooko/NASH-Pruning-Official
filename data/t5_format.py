import numpy as np
import functools
import regex as re

def seq2seq_format(sources, targets, add_prefix):
        # src_prefix = name if prefix is None else prefix
        src_prefix = add_prefix
        sources = [src_prefix]+sources if add_prefix else sources
        return {'source': ' '.join(sources),
                'target': ' '.join(targets),
                }
    
def seq2seq_squad_format(sources, targets, add_prefix):
        # src_prefix = name if prefix is None else prefix
        src_prefix = add_prefix
        sources = [src_prefix]+sources if add_prefix else sources
        return {'source': ' '.join(sources),
                'target': targets[0], 
                'target_list': targets,
                }

def pad_punctuation(text):
    """Re-implementation of _pad_punctuation in t5. This function adds spaces
    around punctuation. While this pads punctuation as expected, it has the 
    unexpected effected of padding certain unicode characters with accents, with
    spaces as well. For instance: "François" becomes "Fran ç ois"""
    # Pad everything except for: underscores (_), whitespace (\s),
    # numbers (\p{N}), letters (\p{L}) and accent characters (\p{M}).
    text = re.sub(r'([^_\s\p{N}\p{L}\p{M}])', r' \1 ', str(text))
    # Collapse consecutive whitespace into one space.
    text = re.sub(r'\s+', ' ', text)
    return text

#for sglue multirc and record
def _mark_span(text, span_str, span_idx, mark):
    pattern_tmpl = r'^((?:\S+\s){N})(W)'
    pattern = re.sub('N', str(span_idx), pattern_tmpl)
    pattern = re.sub('W', span_str, pattern)
    return re.sub(pattern, r'\1{0} \2 {0}'.format(mark), text)

def remove_markup(text):
    """Removes the HTML markup."""
    text = re.sub('<br>', ' ', text)
    text = re.sub('<(/)?b>', '', text)
    return text

def map_dataset(dataset, preprocessor):
        # return dataset.map(functools.partial(preprocessor))
        return dataset.map(preprocessor)
    
def DOLLYpreprocessor(example, add_prefix=""):
    instruction = example['instruction']
    context = example['context']
    response = example['response']
    
    prompt = "Below is an instruction that describe a task.\nWrite a response that appropriately completes the request.\n\n"
    source = [prompt,
              "### Instruction:\n", instruction,
              "\n\n### Input:\n", context,
              "\n\n### Response:"]
    target = [response]
    return seq2seq_format(source, target, None)

def TWEETQApreprocessor(example, add_prefix="TWEETQA"):
    target = example['Answer']
    question = example['Question']
    context = example['Tweet']
    source = ["question:", question,
                "context:", context]
    if len(target)==0: target = [""]
    return seq2seq_squad_format(source, target, None)

def NARRATIVEQApreprocessor(example, add_prefix="NARRATIVEQA"):
    answer = [ans['text'] for ans in example['answers']]
    question = example['question']['text']
    context = example['document']['summary']['text']
    source = ["question:", question,
                "context:", context]
    target = [answer] if type(answer) == str else answer
    return seq2seq_squad_format(source, target, None)

def SAMSUMpreprocessor(example, add_prefix="SAMSUM"):
    answer = example['summary']
    article = example['dialogue']
    source = ["summarize:", article]
    target = [answer] if type(answer) == str else answer
    return seq2seq_format(source, target, None)
    
def XSUMpreprocessor(example, add_prefix='XSUM'):
    answer = example['summary']
    article = example['document']
    source = ["summarize:", article]
    target = [answer] if type(answer) == str else answer
    return seq2seq_format(source, target, None)
    
def CNNDMpreprocessor(example, add_prefix='CNNDM'):
    answer = example['highlights']
    article = example['article']
    source = ["summarize:", article]
    target = [answer] if type(answer) == str else answer
    return seq2seq_format(source, target, None)

def SQUADpreprocessor(example, add_prefix='SQUAD'):
    target = list(set(example['answers']['text']))
    question = example['question']
    context = example['context']
    source = ["question:", question,
                "context:", context]
    return seq2seq_squad_format(source, target, None)

def MRPCpreprocessor(example, add_prefix='MRPC'):
    mrpc_label_dict={0:"not_equivalent", 1:"equivalent", -1:"unidentifiable"}
    src_texts = ["sentence1:", example['sentence1'],
                 "sentence2:", example["sentence2"]]
    tgt_texts = [str(mrpc_label_dict[example['label']])]
    return seq2seq_format(src_texts, tgt_texts, add_prefix)

def CoLApreprocessor(example, add_prefix='CoLA'):
    label_dict={0:'not_acceptable', 1:"acceptable", -1: "unidentifiable"}
    src_texts = ["sentence:", example['sentence']]
    tgt_texts = [str(label_dict[example['label']])]
    return seq2seq_format(src_texts, tgt_texts, add_prefix)

def SST2preprocessor(example, add_prefix='SST-2'):
    label_dict={0:'negative', 1:'positive', -1: 'unidentifiable'}
    src_texts = ["sentence:", example['sentence']]
    tgt_texts = [str(label_dict[example['label']])]
    return seq2seq_format(src_texts, tgt_texts, add_prefix)

def STSBpreprocessor(example, add_prefix='STS-B'):
    src_texts = ["sentence1:", example['sentence1'],
                 "sentence2:", example["sentence2"]]
    tgt_texts = [str(np.round((example['label'] * 5) / 5, decimals=1))]
    return seq2seq_format(src_texts, tgt_texts, add_prefix)

def QQPpreprocessor(example, add_prefix='QQP'):
    label_dict={0:"not_duplicate", 1:"duplicate", -1: 'unidentifiable'}
    src_texts = ["question1:", example['question1'],
                 "question2:", example["question2"]]
    tgt_texts = [str(label_dict[example['label']])]
    return seq2seq_format(src_texts, tgt_texts, add_prefix)

def MNLIpreprocessor(example, add_prefix='MNLI'):
    label_dict = {0:'contradiction', 1:'entailment', 2:'neutral',
                  -1: 'unidentifiable'}
    src_texts = ["premise:", example['premise'],
                 "hypothesis:", example["hypothesis"]]
    tgt_texts = [str(label_dict[example['label']])]
    return seq2seq_format(src_texts, tgt_texts, add_prefix)

def QNLIpreprocessor(example, add_prefix='QNLI'):
    label_dict={0: 'entailment', 1: 'not_entailment', -1: 'unidentifiable'}
    src_texts = ["question:", example['question'],
                 "sentence:", example["sentence"]]
    tgt_texts = [str(label_dict[example['label']])]
    return seq2seq_format(src_texts, tgt_texts, add_prefix)

def RTEpreprocessor(example, add_prefix='RTE'):
    label_dict={0: 'entailment', 1: 'not_entailment', -1: "unidentifiable"}
    src_texts = ["sentence1:", example['sentence1'],
                 "sentence2:", example["sentence2"]]
    tgt_texts = [str(label_dict[example['label']])]
    return seq2seq_format(src_texts, tgt_texts, add_prefix)

def SuperGLUECBprocessor(example, add_prefix='cb'):
    label_dict={0: 'entailment', 1: 'neutral', 2: 'contradiction', -1:"unidentifiable"}
    src_texts = ["premise:", example["premise"],
                "hypothesis:", example["hypothesis"]]
    tgt_texts = [str(label_dict[example["label"]])]
    return seq2seq_format(src_texts, tgt_texts, add_prefix)

def SuperGLUECOPAprocessor(example, add_prefix='copa'):
    label_dict={0: 'choice1', 1: 'choice2', -1:"unidentifiable"}
    if example['label'] in label_dict:
        src_texts = ["premise:", example["premise"],
                     "choice1:", example["choice1"],
                     "choice2:", example["choice2"]]
        tgt_texts = [str(label_dict[example["label"]])]
    return seq2seq_format(src_texts, tgt_texts, add_prefix)

def SuperGLUEWICprocessor(example, add_prefix='wic'):
    label_dict={0: 'false', 1: 'true', -1:"unidentifiable"}
    src_texts = ["sentence1:", example["sentence1"],
                "sentence2:", example["sentence2"],
                "word:", example["word"]]
    tgt_texts = [str(label_dict[example["label"]])]
    return seq2seq_format(src_texts, tgt_texts, add_prefix)

def SuperGLUEWSCFixedprocessor(example, add_prefix='wsc-fixed'):
    label_dict={0: 'false', 1: 'true', -1:"unidentifiable"}
    if example['label'] in label_dict:
        text = example['text']
        text = _mark_span(
            text, example['span1_text'], example['span1_index'], '*')
        # Compensate for 2 added "words" added in previous step.
        span2_index = example['span2_index'] + 2 * \
            int(example['span1_index'] < example['span2_index'])
        text = _mark_span(text, example['span2_text'], span2_index, '#')
        src_texts = ["text:", text]
        tgt_texts = [str(label_dict[example["label"]])]
    return seq2seq_format(src_texts, tgt_texts, add_prefix)

def SuperGLUEBoolQprocessor(example, add_prefix='boolq'):
    label_dict={0: 'false', 1: 'true', -1:"unidentifiable"}
    src_texts = ["question:", example["question"],
                "passage:", example["passage"]]
    tgt_texts = [str(label_dict[example["label"]])]
    return seq2seq_format(src_texts, tgt_texts, add_prefix)

def SuperGLUERTEprocessor(example, add_prefix='rte'):
    label_list=['0', '1']
    src_texts = ["premise:", example["premise"],
                "hypothesis:", example["hypothesis"]]
    tgt_texts = [str(example["label"])]
    return seq2seq_format(src_texts, tgt_texts, add_prefix)

def SuperGLUEMultiRCprocessor(example, add_prefix='multirc'):
    label_dict={0: 'false', 1: 'true', -1:"unidentifiable"}
    src_texts = ["question:", remove_markup(example["question"]),
                "answer:", remove_markup(example["answer"]),
                "paragraph:", remove_markup(example["paragraph"])]
    tgt_texts = [str(label_dict[example["label"]])]
    return seq2seq_format(src_texts, tgt_texts, add_prefix)
    
def SuperGLUEReCoRDprocessor(example, add_prefix='record'):
    passage = example['passage']
    passage = re.sub(
        r'(\.|\?|\!|\"|\')\n@highlight\n', r'\1 ', passage)
    passage = re.sub(r'\n@highlight\n', '. ', passage)
    src_texts = ["query:", example["query"],
                "entities:", example['entities'],
                "passage:", passage]
    tgt_texts = [str(example["answers"])]
    return seq2seq_squad_format(src_texts, tgt_texts, add_prefix)

def OrangeSumProcessor(example, add_prefix='orangesum'):
    src_texts = ["text:", example["text"]]
    tgt_texts = [example["summary"]]
    return seq2seq_squad_format(src_texts, tgt_texts, add_prefix)
    
task2format={"rte" : RTEpreprocessor,
            "mrpc" : MRPCpreprocessor,
            "qqp" : QQPpreprocessor,
            "qnli" : QNLIpreprocessor,
            "mnli" : MNLIpreprocessor,
            "cola" : CoLApreprocessor,
            "stsb" : STSBpreprocessor,
            "sst2" :SST2preprocessor,
            "cnndm" : CNNDMpreprocessor,
            "xsum" : XSUMpreprocessor,
            "samsum": SAMSUMpreprocessor,
            "narrativeqa": NARRATIVEQApreprocessor,
            "cb" : SuperGLUECBprocessor,
            "copa" : SuperGLUECOPAprocessor,
            "multirc" : SuperGLUEMultiRCprocessor,
            "wic" : SuperGLUEWICprocessor,
            "wsc.fixed" : SuperGLUEWSCFixedprocessor,
            "boolq" : SuperGLUEBoolQprocessor,
            "record" : SuperGLUEReCoRDprocessor,
            "orangesum" : OrangeSumProcessor,
            "tweetqa": TWEETQApreprocessor,
            "dolly": DOLLYpreprocessor,
            }
