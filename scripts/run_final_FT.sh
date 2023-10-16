#!/bin/bash

# Example run: bash run_FT.sh [TASK] [EX_NAME_SUFFIX]

sglue_low=(CB COPA)
glue_low=(MRPC RTE STSB CoLA WIC BOOLQ)
glue_high=(MNLI QQP QNLI SST2 SAMSUM TWEETQA NARRATIVEQA)
cnndm=(CNNDM XSUM)

proj_dir=.

code_dir=${proj_dir}

# task and data
task_name=$1
data_dir=$proj_dir/data/glue_data/${task_name}

# logging & saving
logging_steps=100
save_steps=0

# train parameters
max_seq_length=512
batch_size=8
learning_rate=3e-4
epochs=3
tokenizer_name=$2
# seed
seed=57

if [[ " ${sglue_low[*]} " =~ ${task_name} ]]; then
    eval_steps=100
	epochs=20
fi

if [[ " ${glue_low[*]} " =~ ${task_name} ]]; then
    eval_steps=100
	epochs=20
fi

if [[ " ${glue_high[*]} " =~ ${task_name} ]]; then
    eval_steps=500
	epochs=20
fi

if [[ " ${cnndm[*]} " =~ ${task_name} ]]; then
    eval_steps=3000
	epochs=3
fi

# output directory
ex_name_suffix=sparsityy
model_name_or_path=$3
pretrained_pruned_model_path=$3
ex_name=${task_name}_${ex_name_suffix}
# output_dir=$proj_dir/out-test/${task_name}/${ex_name}
output_dir=$model_name_or_path/FT
mkdir -p $output_dir
pruning_type=None

python $code_dir/run_prune.py \
	--output_dir ${output_dir} \
	--logging_steps ${logging_steps} \
	--task_name ${task_name} \
	--model_name_or_path ${model_name_or_path} \
	--ex_name ${ex_name} \
	--do_train \
	--do_eval \
	--tokenizer_name ${tokenizer_name} \
	--max_seq_length ${max_seq_length} \
	--per_device_train_batch_size ${batch_size} \
	--per_device_eval_batch_size 32 \
	--learning_rate ${learning_rate} \
	--pretrained_pruned_model ${pretrained_pruned_model_path} \
	--num_train_epochs ${epochs} \
	--overwrite_output_dir \
	--save_steps ${save_steps} \
	--eval_steps ${eval_steps} \
	--evaluation_strategy steps \
	--seed ${seed} 2>&1 | tee $output_dir/all_log.txt
