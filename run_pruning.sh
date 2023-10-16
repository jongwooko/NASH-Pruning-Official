TASK=${1}
prune_method=${2}
FOLDER_NAME=$3 # Model_name
sp=${4}
NUM_SELECTED_LAYERS=${5}
selection=unif

SUFFIX=${prune_method}_${selection}_${sp}_${5}
EX_CATE=NASH
PRUNING_TYPE=structured_heads+structured_mlp+layer+hidden
SPARSITY=${sp}
DISTILL_LAYER_LOSS_ALPHA=0.001 
DISTILL_CE_LOSS_ALPHA=0.5
LAYER_DISTILL_VERSION=4
LAYER_SELECTION=${selection}
SPARSITY_EPSILON=0.01

DISTILLATION_PATH=./out-test/$TASK/${TASK}_${FOLDER_NAME}/best

if [[ ${prune_method} == cofi ]]; then
    SUFFIX=${prune_method}_sparsity${sp}
    bash scripts/run_CoFi.sh $TASK $FOLDER_NAME $SUFFIX $EX_CATE $PRUNING_TYPE $SPARSITY $DISTILLATION_PATH $DISTILL_LAYER_LOSS_ALPHA $DISTILL_CE_LOSS_ALPHA $LAYER_DISTILL_VERSION
    MODEL_PATH=./cofi_out/${FOLDER_NAME}/${TASK}/NASH/${TASK}_cofi_sparsity${sp}
    
elif [[ ${prune_method} == nash ]]; then
    PRUNING_TYPE=structured_heads+structured_mlp
    SUFFIX=${prune_method}_${selection}_${sp}_${5}
    bash scripts/run_NASH.sh $TASK $FOLDER_NAME $SUFFIX $EX_CATE $PRUNING_TYPE $SPARSITY $DISTILLATION_PATH $DISTILL_LAYER_LOSS_ALPHA $DISTILL_CE_LOSS_ALPHA $LAYER_DISTILL_VERSION $LAYER_SELECTION $NUM_SELECTED_LAYERS
    MODEL_PATH=./nash_out/${FOLDER_NAME}/${TASK}/NASH/${TASK}_${prune_method}_${selection}_${sp}_${5}

elif [[ ${prune_method} == auto ]]; then
    SUFFIX=nash_${prune_method}_${sp}_${5}
    LAYER_SELECTION=auto
    bash scripts/run_AUTO.sh $TASK $FOLDER_NAME $SUFFIX $EX_CATE $PRUNING_TYPE $SPARSITY $DISTILLATION_PATH $DISTILL_LAYER_LOSS_ALPHA $DISTILL_CE_LOSS_ALPHA $LAYER_DISTILL_VERSION $LAYER_SELECTION $NUM_SELECTED_LAYERS
    MODEL_PATH=./nash_auto/${FOLDER_NAME}/${TASK}/NASH/${TASK}_nash_${prune_method}_${sp}_${5}
fi


if [[ -f "$MODEL_PATH/best/pytorch_model.bin" ]]; then
    echo "$MODEL_PATH/best exists."
    MODEL_PATH=$MODEL_PATH/best
else 
    echo "$MODEL_PATH/best does not exist. use int file as a model path"
    MODEL_PATH=$MODEL_PATH/int
fi

 bash scripts/run_final_FT.sh ${TASK} ${FOLDER_NAME} ${MODEL_PATH}
