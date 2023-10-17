## [Official] NASH: A Simple Unified Framework of Structured Pruning for Accelerating Encoder-Decoder Language Models (Findings of EMNLP 2023)

[**NASH: A Simple Unified Framework of Structured Pruning for Accelerating Encoder-Decoder Language Models**](https://arxiv.org/abs/2310.10054)
[Jongwoo Ko](https://sites.google.com/view/jongwooko)$^\*$, 
[Seungjoon Park]()$^\*$, 
Yujin Kim, 
[Sumyeong Ahn]()$^\dagger$, 
Du-Seong Chang, 
Euijai Ahn, 
[Se-Young Yun](https://osi.kaist.ac.kr/)$^\dagger$<br/>
\* equal contribution $&nbsp$ $\dagger$ equal advising


## Overview
- In this study, we investigate the behavior of encoder-decoder models by applying decoupled structural pruning separately to the encoder and decoder components. 
- Our findings highlight two insights: (1) the number of decoder layers is the dominant factor for inference speed, and (2) moderate sparsity in the pruned encoder network enhances generation quality. 
- Motivated by these findings, we propose a simple and effective framework, NASH, that narrows the encoder and shortens the decoder networks of encoder-decoder models. 
- Extensive experiments on diverse generation and inference tasks validate the effectiveness of our method in both speedup and generation quality.


## Requirements
Install the necessary packages with: 
```
$ pip install -r requirements.txt
```
Please define a lower version of transformers, because the latest version seems seems do not have `hf_bucket_url` in `transformers.file_utils`

## Experiments
Our code supports two encoder-decoder language model types: 1) [T5](https://arxiv.org/abs/1910.10683) (also for T5-v1.1) and 2) [BART](https://arxiv.org/abs/1910.13461). If you want to prune BART-like model, please run your code after changing `t5-base` to your model name. (e.g., `facebook/bart-large`.)

### Methods
You can run two structured pruning methods on T5, including [CoFi](https://arxiv.org/abs/2204.00408), and our NASH pruning.

#### Fine-tuning:
Before running our method, we need to prepare the model finetuned on the target task. An example for finetuning the model is as follows:
```
TASK=SAMSUM
MODEL_NAME=t5-base

bash run_finetuning.sh $TASK $MODEL_NAME $MODEL_NAME
```

##### Training for NASH:
If you want to use NASH pruning, set the `PRUNE_METHOD` as `nash`. For the number of decoder layers, we recomment to set the value as 3 or 4 for t5-base.

```
TASK=SAMSUM
PRUNE_METHOD=nash
MODEL_NAME=t5-base
SPARSITY=0.3
NUM_SELECTED_LAYER=3

bash run_pruning.sh $TASK $PRUNE_METHOD $MODEL_NAME $SPARSITY $NUM_SELECTED_LAYER
```

##### Training for CoFi:
If your want you use CoFi pruning, set the value as `cofi`.

```
TASK=SAMSUM
PRUNE_METHOD=cofi
MODEL_NAME=t5-base
SPARSITY=0.8

bash run_pruning.sh $TASK $PRUNE_METHOD $MODEL_NAME $SPARSITY
```

#### Evaluation:
You can use the script `evaluation.py` to get the sparsity, inference time required for each components in the model and development set results of a pruned model. Here's an example use of evaluating a text summarization model is as follows:

```
TASK=SAMSUM
MODEL_DIR=./nash_out/t5-base/NASH/SAMSUM_nash_unif_0.3_2/best/FT/best
BASE_MODEL=t5-base

python evaluation.py $TASK $MODEL_DIR $BASE_MODEL None
```


### Results
We empirically evaluate the performance of NASH on variuos NLG datasets including standard fine-tuning on single task, multi-task learning, and recent instruction-tuning datasets.

Notably, in our experiemnts using T5-base, NASH achieves a speedup of 2.5-4.2 times while preserving 95% of the output quality. Our experimental results show that NASH can be unified framework whch is regardless of task difficulty and model type.
<p align="center">
<img width="1194" src="https://github.com/jongwooko/NASH-Pruning-Official/assets/59277369/5696af03-0ce4-43af-aeb6-77f0e176768f">
</p>

<p align="center">
<img width="1194" src="https://github.com/jongwooko/NASH-Pruning-Official/assets/59277369/6301d7f7-4b7e-4b8e-abde-e7ff3a722ae6">
</p>


## BibTeX
If you find this repo useful for your research, please consider citing our paper:
```
@misc{ko2023nash,
      title={NASH: A Simple Unified Framework of Structured Pruning for Accelerating Encoder-Decoder Language Models}, 
      author={Jongwoo Ko and Seungjoon Park and Yujin Kim and Sumyeong Ahn and Du-Seong Chang and Euijai Ahn and Se-Young Yun},
      year={2023},
      eprint={2310.10054},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## Contact
- Jongwoo Ko: jongwoo.ko@kaist.ac.kr
- Seungjoon Park: sjoon.park@kt.com