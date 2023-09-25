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
module load cudatoolkit-standalone/11.8.0 conda/2023-01-10-unstable gcc/11.2.0
unset CC
unset F77
unset CXX
unset FC
unset F90
export CRAY_ACCEL_TARGET=nvidia80
export MPICH_GPU_SUPPORT_ENABLED=1
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_COLLNET_ENABLE=1
export NVCC_PREPEND_FLAGS="--forward-unknown-opts"
export CFLAGS="-I/soft/datascience/conda/2023-01-10/mconda3/include/"
export LDFLAGS="-L/soft/datascience/conda/2023-01-10/mconda3/lib/"
conda activate dspeed_env


DIR=/home/am6429/dl-io/Megatron-DeepSpeed/
cd ${DIR}
DATETIME=$(date +'date_%y-%m-%d_time_%H-%M-%S')

BASE_DATA_PATH=/home/am6429/dl-io/datasets
# DATASET="1 ${BASE_DATA_PATH}/meg-gpt2_text_document"
# DATASET="1 /grand/projects/VeloC/am6429/bookcorpus/books1/epubtxt"
DATASET="/grand/projects/VeloC/am6429/the_pile_bert/pile_bert_train_text_sentence"
TOKENIZER_PATH=/home/am6429/dl-io/datasets/tokenizer.model

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
echo "CUDA_DEVICE_MAX_CONNECTIONS=1" >> .deepspeed_env

NNODES=$(wc -l < $PBS_NODEFILE)
echo "Number of nodes found as $NNODES"
sed 's/$/ slots=4/' $PBS_NODEFILE > $HOSTFILE
NRANKS_PER_NODE=4
WORLD_SIZE=$(( NNODES * NRANKS_PER_NODE ))
# LAUNCH_PARAMS="--hostfile=$HOSTFILE --num_gpus=$WORLD_SIZE --num_nodes=$NNODES --master_addr $HOSTNAME --master_port=9901"
LAUNCH_PARAMS="--hostfile=$HOSTFILE --master_port=29700"

USE_DEEPSPEED=1
ZERO_STAGE=1

HIDDEN_SIZE=2048 # e.g. llama-13b: 5120
FFN_HIDDEN_SIZE=5504 # e.g. llama-13b: 13824
NUM_LAYERS=24 # e.g. llama-13b: 40
NUM_HEADS=16 # e.g. llama-13b: 40
SEQ_LENGTH=512
NUM_KV_HEADS=4 # llama2 70B uses GQA

EXIT_INTERVAL=10
TP_DIST=$(awk "BEGIN { printf \"%.0f\", sqrt($WORLD_SIZE) }") 
TP=4 #$(( WORLD_SIZE / TP_DIST ))
PP=2 #$(( WORLD_SIZE / TP ))
DP=1
WORLD_SIZE=$((TP*PP*DP))
MICRO_BATCH_SIZE=16
GLOBAL_BATCH_SIZE=$(( MICRO_BATCH_SIZE * DP ))
# MICRO_BATCH=$(( GLOBAL_BATCH / DP ))
TRAIN_STEPS=3
CHECKPOINT_PATH=/grand/projects/VeloC/am6429/scratch/llama2/tp${TP}_pp${PP}_dp${DP} 
LOAD_CHECKPOINT_PATH=/grand/projects/VeloC/am6429/scratch/llama2/tp${TP}_pp${PP}_dp${DP}

LR=3e-4
MIN_LR=3e-5
DTYPE="bf16"
LR_WARMUP_STEPS=1
WEIGHT_DECAY=0.1
GRAD_CLIP=1
EXP_DIR=${HOME}/experiments/results/ckpt_reshape
LOG_DIR="${EXP_DIR}/tensorboard/tp${TP}_pp${PP}_dp${DP}_hd${HIDDEN}_nl${LAYERS}_gbsz${GLOBAL_BATCH}_mbsz${MICRO_BATCH}_z${ZERO_STAGE}_LR_${LR}_${MIN_LR}_${DTYPE}_cont"
mkdir -p $LOG_DIR

options=" \
	--tensor-model-parallel-size $TP \
       --pipeline-model-parallel-size $PP \
       --num-layers $NUM_LAYERS \
       --hidden-size $HIDDEN_SIZE \
       --ffn-hidden-size $FFN_HIDDEN_SIZE \
       --num-attention-heads $NUM_HEADS \
       --micro-batch-size $MICRO_BATCH_SIZE \
       --global-batch-size $GLOBAL_BATCH_SIZE \
       --seq-length $SEQ_LENGTH \
       --max-position-embeddings $SEQ_LENGTH \
       --train-iters $TRAIN_STEPS \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data-path $DATASET \
       --data-impl mmap \
       --tokenizer-type GPTSentencePieceTokenizer \
       --tokenizer-model $TOKENIZER_PATH \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr $LR \
       --lr-decay-style cosine \
       --min-lr $MIN_LR \
       --weight-decay $WEIGHT_DECAY \
       --clip-grad $GRAD_CLIP \
       --lr-warmup-iters $LR_WARMUP_STEPS \
       --optimizer adam \
       --adam-beta1 0.9 \
       --adam-beta2 0.95 \
       --log-interval 1 \
       --save-interval 1 \
       --eval-interval 1000 \
       --eval-iters $TRAIN_STEPS \
       --bf16 \
       --no-query-key-layer-scaling \
       --attention-dropout 0 \
       --hidden-dropout 0 \
       --use-rotary-position-embeddings \
       --untie-embeddings-and-output-weights \
       --swiglu \
       --normalization rmsnorm \
       --disable-bias-linear \
       --num-key-value-heads $NUM_KV_HEADS \
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
	}
}
EOT
fi



# Calculate the number of parameters
# model_size=$((HIDDEN * HIDDEN * NUM_LAYERS * 3 + HIDDEN * FFN_HIDDEN_SIZE * NUM_LAYERS * 2 +  HIDDEN * SEQ_LENGTH * NUM_LAYERS * 2 + NUM_HEADS * HIDDEN * 3))
# model_size_B=$(awk "BEGIN { printf \"%.1f\", $model_size / 1e9 }")
# echo "Model size: ${model_size}, in B: ${model_size_B}"
model_size_B=7.1
output_dir="/home/am6429/dl-io/output-llama2/llama2-NN$NNODES-OFFLOAD$OPT_OFFLOAD/"
mkdir -p "$output_dir"
log_str="${model_size_B}B-tp$TP-pp$PP-dp$DP-l$LAYERS-h$HIDDEN-a$ATTN_HEADS-sl$SEQ-gbs$GLOBAL_BATCH-mbs-$MICRO_BATCH"
echo "NSYS_REPORT_DIR=${output_dir}/rep-${log_str}-%n">> .deepspeed_env
# run_cmd="rm -rf $CHECKPOINT_PATH && time nsys profile --force-overwrite true -o $output_dir/report-$log_str --stats=true deepspeed ${LAUNCH_PARAMS} ${DIR}/pretrain_gpt.py $@ ${options} | tee $output_dir/log-$log_str.log"
run_cmd="rm -rf $CHECKPOINT_PATH && deepspeed ${LAUNCH_PARAMS} ${DIR}/pretrain_gpt.py $@ ${options} | tee $output_dir/log-$log_str.log"
echo $run_cmd

# echo ${run_cmd}
eval ${run_cmd}
ls -ltrh "$CHECKPOINT_PATH/global_step1/" >> "$output_dir/log-$log_str.log"

set +x
