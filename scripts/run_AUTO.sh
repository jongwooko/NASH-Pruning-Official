
#!/bin/bash
# SBATCH --job-name=sample
# SBATCH --nodes=1
# SBATCH --ntasks-per-node=1
# SBATCH --gres=gpu:1
# SBATCH -A pnlp
# SBATCH -t 11:00:00

sglue_low=(CB COPA WSC.fixed)
glue_low=(MRPC RTE STSB CoLA WIC BOOLQ)
glue_high=(MNLI QQP QNLI SST2)
samsum=(SAMSUM TWEETQA NARRATIVEQA)
cnndm=(CNNDM XSUM)

proj_dir=.

code_dir=${proj_dir}

# task and data
task_name=$1
data_dir=$proj_dir/data/glue_data/${task_name}

# pretrain model
model_name_or_path=$2
# logging & saving
logging_steps=100
save_steps=0


# train parameters
max_seq_length=128
batch_size=4 #32
learning_rate=3e-5
reg_learning_rate=0.01
epochs=20 

# seed
seed=57

# output dir
ex_name_suffix=$3
ex_name=${task_name}_${ex_name_suffix}
ex_cate=$4
output_dir=${proj_dir}/nash_auto/${model_name_or_path}/${task_name}/${ex_cate}/${ex_name}

# pruning and distillation
pruning_type=$5
target_sparsity=$6
distillation_path=$7
distill_layer_loss_alpha=$8
distill_ce_loss_alpha=$9
distill_temp=2
# 2: fix hidden layers, 3: min distance matching without restriction, 4: min distance matching with restriction
layer_distill_version=${10} 
layer_selection=${11}
num_selected_layers=${12}
scheduler_type=linear
tokenizer_name=${model_name_or_path}

if [[ " ${sglue_low[*]} " =~ ${task_name} ]]; then
    eval_steps=50
    epochs=200
    start_saving_best_epochs=90
    prepruning_finetune_epochs=4
    lagrangian_warmup_epochs=160
fi

if [[ " ${glue_low[*]} " =~ ${task_name} ]]; then
    eval_steps=500
    epochs=100
    start_saving_best_epochs=90
    prepruning_finetune_epochs=4
    lagrangian_warmup_epochs=80
fi

if [[ " ${glue_high[*]} " =~ ${task_name} ]]; then
    eval_steps=500
    prepruning_finetune_epochs=1
    lagrangian_warmup_epochs=2
fi

if [[ " ${samsum[*]} " =~ ${task_name} ]]; then
    eval_steps=1000
    epochs=20
    batch_size=16
    start_saving_best_epochs=15
    prepruning_finetune_epochs=4
    lagrangian_warmup_epochs=10
fi

if [[ " ${cnndm[*]} " =~ ${task_name} ]]; then
    eval_steps=6000
    epochs=3
    start_saving_best_epochs=1
    prepruning_finetune_epochs=0
    lagrangian_warmup_epochs=1
fi

pretrained_pruned_model=None

# FT after pruning
if [[ $pruning_type == None ]]; then
  pretrained_pruned_model=${10}
  learning_rate=${11}
  scheduler_type=none
  output_dir=$pretrained_pruned_model/FT-lr${learning_rate}
  epochs=20
  batch_size=64
fi

mkdir -p $output_dir

python $code_dir/run_prune.py \
	   --output_dir ${output_dir} \
	   --logging_steps ${logging_steps} \
	   --task_name ${task_name} \
	   --model_name_or_path ${model_name_or_path} \
     --tokenizer_name ${tokenizer_name} \
	   --ex_name ${ex_name} \
	   --do_train \
	   --do_eval \
	   --max_seq_length ${max_seq_length} \
	   --per_device_train_batch_size ${batch_size} \
	   --per_device_eval_batch_size 32 \
	   --learning_rate ${learning_rate} \
	   --reg_learning_rate ${reg_learning_rate} \
	   --num_train_epochs ${epochs} \
	   --overwrite_output_dir \
	   --save_steps ${save_steps} \
	   --eval_steps ${eval_steps} \
	   --evaluation_strategy steps \
	   --seed ${seed} \
	   --pruning_type ${pruning_type} \
     --pretrained_pruned_model ${pretrained_pruned_model} \
     --target_sparsity $target_sparsity \
     --freeze_embeddings \
     --do_distill \
     --do_layer_distill \
     --distillation_path $distillation_path \
     --distill_ce_loss_alpha $distill_ce_loss_alpha \
     --distill_loss_alpha $distill_layer_loss_alpha \
     --distill_temp $distill_temp \
     --scheduler_type $scheduler_type \
     --layer_distill_version $layer_distill_version \
     --prepruning_finetune_epochs $prepruning_finetune_epochs \
     --layer_selection $layer_selection \
     --pruning_method auto_select \
     --num_select_layers $num_selected_layers \
     --lagrangian_warmup_epochs $lagrangian_warmup_epochs 2>&1 | tee ${output_dir}/log.txt
