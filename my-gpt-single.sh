#!/bin/bash -l
# PBS -l nodes=2:system=polaris
# PBS -l walltime=01:00:00
# PBS -q debug-scaling
# PBS -A VeloC
# PBS -l filesystems=home:grand

set -x 
#### module load cray-mpich/8.1.16 mpiwrappers/cray-mpich-llvm cudatoolkit-standalone/11.8.0 PrgEnv-gnu nvhpc-mixed/23.3 conda/2023-01-10-unstable

### Currently Loaded Modules:
###   1) gcc/11.2.0         4) libfabric/1.11.0.4.125   7) cray-pmi/6.1.2       10) cray-libpals/1.1.7           13) conda/2023-01-10-unstable
###   2) craype/2.7.15      5) craype-network-ofi       8) cray-pmi-lib/6.0.17  11) PrgEnv-gnu/8.3.3             14) cudatoolkit-standalone/11.8.0
###   3) cray-dsmml/0.2.2   6) cray-mpich/8.1.16        9) cray-pals/1.1.7      12) cray-hdf5-parallel/1.12.1.3
# module load cudatoolkit-standalone/11.8.0 conda/2023-01-10-unstable gcc/11.2.0
# unset CC
# unset F77
# unset CXX
# unset FC
# unset F90
# export CRAY_ACCEL_TARGET=nvidia80
# export MPICH_GPU_SUPPORT_ENABLED=1
# export NCCL_NET_GDR_LEVEL=PHB
# export NCCL_COLLNET_ENABLE=1
# export NVCC_PREPEND_FLAGS="--forward-unknown-opts"
# export CFLAGS="-I/soft/datascience/conda/2023-01-10/mconda3/include/"
# export LDFLAGS="-L/soft/datascience/conda/2023-01-10/mconda3/lib/"
# conda activate dspeed_env
source ~/.bash_profile
dlconda

DIR=/home/am6429/dl-io/Megatron-DeepSpeed/
cd ${DIR}
DATETIME=$(date +'date_%y-%m-%d_time_%H-%M-%S')

BASE_DATA_PATH=/home/am6429/dl-io/datasets
DATASET=${BASE_DATA_PATH}/meg-gpt2_text_document
VOCAB_PATH=${BASE_DATA_PATH}/gpt2-vocab.json
MERGE_PATH=${BASE_DATA_PATH}/gpt2-merges.txt

CONFIG_JSON="$DIR/ds_config.json"
HOSTFILE="$DIR/hostfile"
echo "PATH=${PATH}" > .deepspeed_env
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}" >> .deepspeed_env
echo "http_proxy=${http_proxy}" >> .deepspeed_env
echo "https_proxy=${https_proxy}" >> .deepspeed_env
echo "CC=/home/am6429/.conda/envs/dspeed_env/bin/x86_64-conda-linux-gnu-cc" >> .deepspeed_env
echo "CXX=/home/am6429/.conda/envs/dspeed_env/bin/x86_64-conda-linux-gnu-c++" >> .deepspeed_env
echo "CFLAGS=-I/soft/datascience/conda/2023-01-10/mconda3/include/" >> .deepspeed_env
echo "LDFLAGS=-L/soft/datascience/conda/2023-01-10/mconda3/lib/" >> .deepspeed_env

NNODES=1
echo "Number of nodes found as $NNODES"
# echo 'slots=1 polaris-login-01' > $HOSTFILE
export CUDA_VISIBLE_DEVICES=0,1
NRANKS_PER_NODE=1
WORLD_SIZE=$(( NNODES * NRANKS_PER_NODE ))

USE_DEEPSPEED=1
ZERO_STAGE=1

# Model size: 0.56B Bloom
HIDDEN=1024
ATTN_HEADS=16
LAYERS=24
SEQ=2048

EXIT_INTERVAL=10
TP_DIST=$(awk "BEGIN { printf \"%.0f\", sqrt($WORLD_SIZE) }") 
TP=1 #$(( WORLD_SIZE / TP_DIST ))
PP=1 #$(( WORLD_SIZE / TP ))
DP=1
WORLD_SIZE=$((TP*PP*DP))
MICRO_BATCH=16
GLOBAL_BATCH=$(( MICRO_BATCH * DP ))
# MICRO_BATCH=$(( GLOBAL_BATCH / DP ))
TRAIN_ITERS=30
CHECKPOINT_PATH=/local/scratch/tp${TP}_pp${PP}_dp${DP} 
# CHECKPOINT_PATH=/grand/projects/VeloC/am6429/scratch/gpt2-single/tp${TP}_pp${PP}_dp${DP} 
# LOAD_CHECKPOINT_PATH=/grand/projects/VeloC/am6429/scratch/gpt2-single/tp${TP}_pp${PP}_dp${DP}

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
        --eval-iters $TRAIN_ITERS \
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


OPT_OFFLOAD=0
if [[ $OPT_OFFLOAD == 0 ]]; then
echo "Not offload optimizer to the CPU"
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
	"comms_logger": {
		"enabled": true,
		"verbose": true,
		"prof_all": true,
		"debug": true
	},
	"flops_profiler": {
		"enabled": true,
		"profile_step": 1,
		"module_depth": -1,
		"top_modules": 1,
		"detailed": true,
		"output_file": null
	},
    "veloc_config": {
        "gpu_cache": 0,
        "host_cache": 4
    }
}
EOT

else
echo "Offloading optimizer to the CPU"
cat <<EOT > $CONFIG_JSON
{
	"train_batch_size": $GLOBAL_BATCH,
	"train_micro_batch_size_per_gpu": $MICRO_BATCH,
	"steps_per_print": 1,
	"optimizer": {
		"type": "Adam",
		"params": {
			"lr": 0.001,
			"betas": [
				0.8,
				0.999
			],
			"eps": 1e-8,
			"weight_decay": 3e-7
		}
	},
	"zero_optimization": {
		"stage": $ZERO_STAGE,
		"overlap_comm": true,
		"contiguous_gradients": true,
		"offload_optimizer": {
			"device": "cpu",
			"pin_memory": true,
			"buffer_count": 40,
			"fast_init": true
		}
	},
	"fp16": {
		"enabled": true,
		"initial_scale_power": 12
	}, 
	"activation_checkpointing": {
		"partition_activations": true,
		"cpu_checkpointing": true,
		"contiguous_memory_optimization": true,
		"synchronize_checkpoint_boundary": true,
		"profile": true
	},
	"wall_clock_breakdown": true,
	"memory_breakdown": true,
	"comms_logger": {
		"enabled": true,
		"verbose": true,
		"prof_all": true,
		"debug": true
	},
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
options="${options} \
        --cpu-optimizer"
fi


model_size_B=0.56
output_dir="/home/am6429/dl-io/output/gpt-NN$NNODES-OFFLOAD$OPT_OFFLOAD/"
mkdir -p "$output_dir"
log_str="${model_size_B}B-tp$TP-pp$PP-dp$DP-l$LAYERS-h$HIDDEN-a$ATTN_HEADS-sl$SEQ-gbs$GLOBAL_BATCH-mbs-$MICRO_BATCH"
echo "NSYS_REPORT_DIR=${output_dir}/rep-${log_str}-%n">> .deepspeed_env
# run_cmd="rm -rf $CHECKPOINT_PATH && time /soft/compilers/cudatoolkit/cuda-11.8.0/bin/nsys profile --force-overwrite true -o $output_dir/report-$log_str --stats=true deepspeed ${LAUNCH_PARAMS} ${DIR}/pretrain_gpt.py $@ ${options} | tee $output_dir/log-$log_str.log"

run_cmd="rm -rf $CHECKPOINT_PATH && deepspeed ${DIR}/pretrain_gpt.py $@ ${options} | tee $output_dir/log-$log_str.log"
echo $run_cmd

echo ${run_cmd}
# eval ${run_cmd}
ls -ltrh "$CHECKPOINT_PATH/global_step1/" >> "$output_dir/log-$log_str.log"
rm -rf $output_dir/*.sqlite
set +x
