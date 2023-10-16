#!/bin/bash -l
# PBS -l nodes=2:system=polaris
# PBS -l walltime=01:00:00
# PBS -q debug-scaling
# PBS -A VeloC
# PBS -l filesystems=home:grand

set -x 
source ~/.bash_profile
dlconda

if [[ "$#" -lt 1 ]]; then
    echo "Error: This script requires at least 1 argument."
    exit 1  # Exit with an error status code
fi

if [[ $1 == 3 ]] && [[ "$#" -lt 2 ]]; then
    echo "Error: This script requires at 2 arguments for VELOC based checkpointing."
    exit 1  # Exit with an error status code
fi


DIR=/home/am6429/dl-io/Megatron-DeepSpeed/
cd ${DIR}
DATETIME=$(date +'date_%y-%m-%d_time_%H-%M-%S')

BASE_DATA_PATH=/home/am6429/dl-io/datasets
DATASET=${BASE_DATA_PATH}/meg-gpt2_text_document
VOCAB_PATH=${BASE_DATA_PATH}/gpt2-vocab.json
MERGE_PATH=${BASE_DATA_PATH}/gpt2-merges.txt

CONFIG_JSON="$DIR/ds_config.json"
HOSTFILE="$DIR/hostfile"
echo "PATH=${PATH}:/soft/datascience/conda/2023-01-10/mconda3/include/" > .deepspeed_env
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/soft/datascience/conda/2023-01-10/mconda3/lib/" >> .deepspeed_env
echo "http_proxy=${http_proxy}" >> .deepspeed_env
echo "https_proxy=${https_proxy}" >> .deepspeed_env
echo "CC=gcc" >> .deepspeed_env
echo "CXX=g++" >> .deepspeed_env
echo "CFLAGS=-I/soft/datascience/conda/2023-01-10/mconda3/include/" >> .deepspeed_env
echo "LDFLAGS=-L/soft/datascience/conda/2023-01-10/mconda3/lib/" >> .deepspeed_env

NNODES=$(wc -l < $PBS_NODEFILE)
echo "Number of nodes found as $NNODES"
sed 's/$/ slots=4/' $PBS_NODEFILE > $HOSTFILE
NRANKS_PER_NODE=4
WORLD_SIZE=$(( NNODES * NRANKS_PER_NODE ))
# LAUNCH_PARAMS="--hostfile=$HOSTFILE --num_gpus=$WORLD_SIZE --num_nodes=$NNODES --master_addr $HOSTNAME --master_port=9901"
LAUNCH_PARAMS="--hostfile=$HOSTFILE --master_port=29700"

USE_DEEPSPEED=1
ZERO_STAGE=1

model_size_B=7.1
HIDDEN=4096
ATTN_HEADS=32
LAYERS=30
SEQ=2048

EXIT_INTERVAL=10
TP_DIST=$(awk "BEGIN { printf \"%.0f\", sqrt($WORLD_SIZE) }") 
TP=2 #$(( WORLD_SIZE / TP_DIST ))
PP=2 #$(( WORLD_SIZE / TP ))
DP=$NNODES
WORLD_SIZE=$((TP*PP*DP))
MICRO_BATCH=16
GLOBAL_BATCH=$(( MICRO_BATCH * DP ))
# MICRO_BATCH=$(( GLOBAL_BATCH / DP ))
TRAIN_ITERS=3
CHECKPOINT_PATH=/local/scratch/tp${TP}_pp${PP}_dp${DP} 
# CHECKPOINT_PATH=/grand/projects/VeloC/am6429/scratch/gpt2/tp${TP}_pp${PP}_dp${DP} 
LOAD_CHECKPOINT_PATH=/local/scratch/gpt2/tp${TP}_pp${PP}_dp${DP}

LR=6.0e-4
MIN_LR=6.0e-5
DTYPE="bf16"
EXP_DIR=${HOME}/experiments/results/ckpt_reshape
LOG_DIR="${EXP_DIR}/tensorboard/tp${TP}_pp${PP}_dp${DP}_hd${HIDDEN}_nl${LAYERS}_gbsz${GLOBAL_BATCH}_mbsz${MICRO_BATCH}_z${ZERO_STAGE}_LR_${LR}_${MIN_LR}_${DTYPE}_cont"
mkdir -p $LOG_DIR

options=" \
	--tensor-model-parallel-size $TP \
	--pipeline-model-parallel-size $PP \
        --num-layers $LAYERS \
        --hidden-size $HIDDEN \
        --num-attention-heads $ATTN_HEADS \
        --seq-length $SEQ \
        --loss-scale 12 \
        --max-position-embeddings $SEQ \
	--micro-batch-size $MICRO_BATCH \
	--global-batch-size $GLOBAL_BATCH \
	--train-iters $TRAIN_ITERS \
        --lr $LR \
	--min-lr $MIN_LR \
        --lr-decay-style cosine \
        --log-interval 1 \
        --eval-iters 0 \
        --eval-interval 3600 \
	--data-path ${DATASET} \
	--vocab-file ${VOCAB_PATH} \
	--merge-file ${MERGE_PATH} \
	--save-interval 1 \
        --split 98,2,0 \
        --clip-grad 1.0 \
	--weight-decay 0.1 \
	--adam-beta1 0.9 \
	--adam-beta2 0.95 \
	--init-method-std 0.006 \
        --${DTYPE} \
	--checkpoint-activations \
	--exit-interval ${EXIT_INTERVAL} \
        --save ${CHECKPOINT_PATH} \
	--tensorboard-dir $LOG_DIR	\
        --deepspeed \
        --deepspeed_config=${CONFIG_JSON} \
        --zero-stage=${ZERO_STAGE} \
        --deepspeed-activation-checkpointing \
        "
# --load ${LOAD_CHECKPOINT_PATH} \
# --cpu-optimizer
# --no-pipeline-parallel

# AM comment: BP16 does not work with deepspeed for now
# https://www.deepspeed.ai/docs/config-json/#bfloat16-training-options
# So switching to regular FP16.


CKPT_APPROACH=$1
if [[ $CKPT_APPROACH == 0 ]]; then
echo "Checkpointing using None Checkpointing approach"
cat <<EOT > $CONFIG_JSON
{
	"train_batch_size": $GLOBAL_BATCH,
	"train_micro_batch_size_per_gpu": $MICRO_BATCH,
	"steps_per_print": 1,
	"zero_optimization": {
		"stage": $ZERO_STAGE,
		"overlap_comm": true
	},
	"bf16": {
		"enabled": true
	}, 
	"data_types": {
		"grad_accum_dtype": "fp32"
 	},
	"wall_clock_breakdown": true,
	"memory_breakdown": true,
	"flops_profiler": {
		"enabled": true,
		"profile_step": 1,
		"module_depth": -1,
		"top_modules": 1,
		"detailed": true,
		"output_file": null
	},
	"none_ckpt_config": true
}
EOT
elif [[ $CKPT_APPROACH == 1 ]]; then
echo "Checkpointing with default Torch.save()"
cat <<EOT > $CONFIG_JSON
{
	"train_batch_size": $GLOBAL_BATCH,
	"train_micro_batch_size_per_gpu": $MICRO_BATCH,
	"steps_per_print": 1,
	"zero_optimization": {
		"stage": $ZERO_STAGE,
		"overlap_comm": true
	},
	"bf16": {
		"enabled": true
	},
	"data_types": {
		"grad_accum_dtype": "fp32"
 	},
	"wall_clock_breakdown": true,
	"memory_breakdown": true,
	"flops_profiler": {
		"enabled": true,
		"profile_step": 1,
		"module_depth": -1,
		"top_modules": 1,
		"detailed": true,
		"output_file": null
	}
}
EOT
elif [[ $CKPT_APPROACH == 2 ]]; then
echo "Checkpointing using Python Based AysncTorch approach"
cat <<EOT > $CONFIG_JSON
{
	"train_batch_size": $GLOBAL_BATCH,
	"train_micro_batch_size_per_gpu": $MICRO_BATCH,
	"steps_per_print": 1,
	"zero_optimization": {
		"stage": $ZERO_STAGE,
		"overlap_comm": true
	},
	"bf16": {
		"enabled": true
	},
	"data_types": {
		"grad_accum_dtype": "fp32"
 	},
	"wall_clock_breakdown": true,
	"memory_breakdown": true,
	"flops_profiler": {
		"enabled": true,
		"profile_step": 1,
		"module_depth": -1,
		"top_modules": 1,
		"detailed": true,
		"output_file": null
	},
	"async_ckpt_config": {
		"host_cache": -1
	}
}
EOT
elif [[ $CKPT_APPROACH == 3 ]]; then
echo "Checkpointing using VELOC Checkpointing approach"
cat <<EOT > $CONFIG_JSON
{
	"train_batch_size": $GLOBAL_BATCH,
	"train_micro_batch_size_per_gpu": $MICRO_BATCH,
	"steps_per_print": 1,
	"zero_optimization": {
		"stage": $ZERO_STAGE,
		"overlap_comm": true
	},
	"bf16": {
		"enabled": true
	}, 
	"data_types": {
		"grad_accum_dtype": "fp32"
 	},
	"wall_clock_breakdown": true,
	"memory_breakdown": true,
	"flops_profiler": {
		"enabled": true,
		"profile_step": 1,
		"module_depth": -1,
		"top_modules": 1,
		"detailed": true,
		"output_file": null
	},
	"veloc_ckpt_config": {
		"host_cache": $2
	}
}
EOT
fi

output_dir="/home/am6429/dl-io/output/gpt-NN$NNODES/"
mkdir -p "$output_dir"
log_str="${model_size_B}B-tp$TP-pp$PP-dp$DP-l$LAYERS-h$HIDDEN-a$ATTN_HEADS-sl$SEQ-gbs$GLOBAL_BATCH-mbs-$MICRO_BATCH-ckpt-$CKPT_APPROACH"
echo "NSYS_REPORT_DIR=${output_dir}/rep-${log_str}-%n">> .deepspeed_env

# Remove the `@` which adds all additional params to deepspeed
# run_cmd="rm -rf $CHECKPOINT_PATH && deepspeed ${LAUNCH_PARAMS} ${DIR}/pretrain_gpt.py $@ ${options} | tee $output_dir/log-$log_str.log"
eval "rm -rf $CHECKPOINT_PATH"
run_cmd="{ time deepspeed ${LAUNCH_PARAMS} ${DIR}/pretrain_gpt.py ${options}; } 2>&1 | tee $output_dir/log-$log_str.log"
echo $run_cmd

echo ${run_cmd}
# eval ${run_cmd}
ls -ltrh "$CHECKPOINT_PATH/global_step1/" >> "$output_dir/log-$log_str.log"
rm -rf $output_dir/*.sqlite
set +x
