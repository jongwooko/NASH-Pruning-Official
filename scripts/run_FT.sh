#!/bin/bash

# Example run: bash run_FT.sh [TASK] [EX_NAME_SUFFIX]

sglue_low=(CB WSC.fixed COPA)
glue_low=(MRPC RTE STSB CoLA BOOLQ WIC)
glue_high=(MNLI QQP QNLI SST2)
gen_task=(SQuAD CNNDM XSUM)
orange=(ORANGESUM SAMSUM TWEETQA DOLLY)
proj_dir=.

code_dir=${proj_dir}


# task and data
task_name=$1
data_dir=$proj_dir/data/glue_data/${task_name}

# pretrain model
model_name_or_path=$2

# train parameters
max_seq_length=128
batch_size=16

if [[ "$3" =~ "t5" ]]; then
  learning_rate=3e-4
fi

if [[ "$3" =~ "bart" ]]; then
  learning_rate=1e-5
fi


epochs=20
eval_batch_size=32

# logging & saving
logging_steps=100
save_steps=0

if [[ " ${sglue_low[*]} " =~ ${task_name} ]]; then
    eval_steps=20
fi

if [[ " ${glue_low[*]} " =~ ${task_name} ]]; then
    eval_steps=50
fi

if [[ " ${glue_high[*]} " =~ ${task_name} ]]; then
    eval_steps=500
fi

if [[ " ${gen_task[*]} " =~ ${task_name} ]]; then
    eval_steps=5000
    epochs=3
    max_seq_length=512
    batch_size=8
    eval_batch_size=32
fi

if [[ " ${orange[*]} " =~ ${task_name} ]]; then
    eval_steps=1500
    epochs=20
    max_seq_length=256
    batch_size=8
    eval_batch_size=8
fi

# seed
seed=57

# output directory
ex_name_suffix=$2
ex_name=${task_name}_${ex_name_suffix} # {TASK}_{MODEL_NAME}
output_dir=$proj_dir/out-test/${task_name}/${ex_name}
mkdir -p $output_dir
pruning_type=None

python -m torch.distributed.run --nproc_per_node=1 $code_dir/run_prune.py \
	   --output_dir ${output_dir} \
	   --logging_steps ${logging_steps} \
	   --task_name ${task_name} \
	   --model_name_or_path ${model_name_or_path} \
	   --ex_name ${ex_name} \
	   --do_train \
	   --do_eval \
	   --max_seq_length ${max_seq_length} \
	   --per_device_train_batch_size ${batch_size} \
	   --per_device_eval_batch_size ${eval_batch_size} \
	   --learning_rate ${learning_rate} \
	   --num_train_epochs ${epochs} \
	   --overwrite_output_dir \
	   --save_steps ${save_steps} \
	   --eval_steps ${eval_steps} \
	   --evaluation_strategy steps \
	   --seed ${seed} 2>&1 | tee $output_dir/all_log.txt

