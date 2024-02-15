#!/bin/bash -l
# PBS -l nodes=2:system=polaris
# PBS -l walltime=01:00:00
# PBS -q debug-scaling
# PBS -A VeloC
# PBS -l filesystems=home:grand

source ~/.bash_profile
dlconda
# set -x
# Define default values
CKPT_APPROACH=0
HOST_CACHE=0
model_size_B=0
HIDDEN_SIZE=0
FFN_HIDDEN_SIZE=0
NUM_LAYERS=0
NUM_HEADS=0
SEQ_LENGTH=0
NUM_KV_HEADS=0
TRAIN_ITERS=0
NNODES=$(wc -l < $PBS_NODEFILE)
PP=$NNODES
TP=4
SAVE_INTERVAL=1
GRAD_ACC=1
NRANKS_PER_NODE=4

while getopts ":c:h:m:H:F:N:L:U:S:K:P:T:I:G:" opt; do
  case $opt in
    c)
      if [[ "$OPTARG" =~ ^[0-9]+$ ]]; then
        CKPT_APPROACH="$OPTARG"
      else
        echo "Invalid CKPT_APPROACH: $OPTARG is not a valid integer." >&2
        exit 1
      fi
      ;;
    h)
      if [[ "$OPTARG" =~ ^[0-9]+$ ]]; then
        HOST_CACHE="$OPTARG"
      else
        echo "Invalid HOST_CACHE: $OPTARG is not a valid integer." >&2
        exit 1
      fi
      ;;
    m)
      if [[ "$OPTARG" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
        model_size_B="$OPTARG"
      else
        echo "Invalid model_size_B: $OPTARG is not a valid integer." >&2
        exit 1
      fi
      ;;
    H)
      if [[ "$OPTARG" =~ ^[0-9]+$ ]]; then
        HIDDEN_SIZE="$OPTARG"
      else
        echo "Invalid HIDDEN_SIZE: $OPTARG is not a valid integer." >&2
        exit 1
      fi
      ;;
    F)
      if [[ "$OPTARG" =~ ^[0-9]+$ ]]; then
        FFN_HIDDEN_SIZE="$OPTARG"
      else
        echo "Invalid FFN_HIDDEN_SIZE: $OPTARG is not a valid integer." >&2
        exit 1
      fi
      ;;
    N)
      if [[ "$OPTARG" =~ ^[0-9]+$ ]]; then
        NUM_LAYERS="$OPTARG"
      else
        echo "Invalid NUM_LAYERS: $OPTARG is not a valid integer." >&2
        exit 1
      fi
      ;;
    L)
      if [[ "$OPTARG" =~ ^[0-9]+$ ]]; then
        NUM_HEADS="$OPTARG"
      else
        echo "Invalid NUM_HEADS: $OPTARG is not a valid integer." >&2
        exit 1
      fi
      ;;
    U)
      if [[ "$OPTARG" =~ ^[0-9]+$ ]]; then
        SEQ_LENGTH="$OPTARG"
      else
        echo "Invalid SEQ_LENGTH: $OPTARG is not a valid integer." >&2
        exit 1
      fi
      ;;
    S)
      if [[ "$OPTARG" =~ ^[0-9]+$ ]]; then
        NUM_KV_HEADS="$OPTARG"
      else
        echo "Invalid NUM_KV_HEADS: $OPTARG is not a valid integer." >&2
        exit 1
      fi
      ;;
    K)
      if [[ "$OPTARG" =~ ^[0-9]+$ ]]; then
        TRAIN_ITERS="$OPTARG"
      else
        echo "Invalid TRAIN_ITERS: $OPTARG is not a valid integer." >&2
        exit 1
      fi
      ;;
    P)
      if [[ "$OPTARG" =~ ^[0-9]+$ ]]; then
        PP="$OPTARG"
      else
        PP=$NNODES
      fi
      ;;
    T)
      if [[ "$OPTARG" =~ ^[0-9]+$ ]]; then
        TP="$OPTARG"
      else
        TP=4
      fi
      ;;
    I)
      if [[ "$OPTARG" =~ ^[0-9]+$ ]]; then
        SAVE_INTERVAL="$OPTARG"
      else
        echo "Invalid SAVE_INTERVAL: $OPTARG is not a valid integer." >&2
        exit 1
      fi
      ;;
    G)
      if [[ "$OPTARG" =~ ^[0-9]+$ ]]; then
        GRAD_ACC="$OPTARG"
      else
        echo "Invalid GRAD_ACC: $OPTARG is not a valid integer." >&2
        exit 1
      fi
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
    :)
      echo "Option -$OPTARG requires an argument." >&2
      exit 1
      ;;
  esac
done

# Check if required parameters are provided
if [ -z "$CKPT_APPROACH" ] || [ -z "$HOST_CACHE" ] || [ -z "$model_size_B" ] || [ -z "$HIDDEN_SIZE" ] || [ -z "$FFN_HIDDEN_SIZE" ] || [ -z "$NUM_LAYERS" ] || [ -z "$NUM_HEADS" ] || [ -z "$SEQ_LENGTH" ] || [ -z "$NUM_KV_HEADS" ] || [ -z "$TRAIN_ITERS" ]; then
  echo "Missing required parameter(s)." >&2
  exit 1
fi

# Perform further processing with the parsed parameters
echo "CKPT_APPROACH: $CKPT_APPROACH"
echo "HOST_CACHE: $HOST_CACHE"
echo "model_size_B: $model_size_B"
echo "HIDDEN_SIZE: $HIDDEN_SIZE"
echo "FFN_HIDDEN_SIZE: $FFN_HIDDEN_SIZE"
echo "NUM_LAYERS: $NUM_LAYERS"
echo "NUM_HEADS: $NUM_HEADS"
echo "SEQ_LENGTH: $SEQ_LENGTH"
echo "NUM_KV_HEADS: $NUM_KV_HEADS"
echo "TRAIN_ITERS: $TRAIN_ITERS"
echo "PIPE PARALLEL: $PP"
echo "SAVE_INTERVAL: $SAVE_INTERVAL"
echo "GRAD_ACC: $GRAD_ACC"

DIR=/home/am6429/dl-io/Megatron-DeepSpeed/
cd ${DIR}
DATETIME=$(date +'date_%y-%m-%d_time_%H-%M-%S')

BASE_DATA_PATH=/home/am6429/dl-io/datasets
DATASET="${BASE_DATA_PATH}/meg-gpt2_text_document"
TOKENIZER_PATH=/home/am6429/dl-io/datasets/tokenizer.model
VOCAB_PATH=${BASE_DATA_PATH}/gpt2-vocab.json
MERGE_PATH=${BASE_DATA_PATH}/gpt2-merges.txt

output_dir="/home/am6429/dl-io/dl-io-outputs/swapping/llama2-NN$NNODES/"
mkdir -p "$output_dir"
CONFIG_JSON="$DIR/ds_config.json"
HOSTFILE="$DIR/hostfile"
echo "PATH=${PATH}:/soft/compilers/cudatoolkit/cuda-11.8.0/bin/" > .deepspeed_env
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}" >> .deepspeed_env
echo "http_proxy=${http_proxy}" >> .deepspeed_env
echo "https_proxy=${https_proxy}" >> .deepspeed_env
echo "CC=gcc" >> .deepspeed_env
echo "CXX=g++" >> .deepspeed_env
echo "IBV_FORK_SAFE=1" >> .deepspeed_env
echo "CFLAGS=-I/soft/datascience/conda/2023-01-10/mconda3/include/" >> .deepspeed_env
echo "LDFLAGS=-L/soft/datascience/conda/2023-01-10/mconda3/lib/" >> .deepspeed_env
echo "CUDA_DEVICE_MAX_CONNECTIONS=1" >> .deepspeed_env
echo "TORCHSNAPSHOT_PER_RANK_MEMORY_BUDGET_BYTES=34359738368" >> .deepspeed_env
echo "_DEFAULT_MAX_PER_RANK_IO_CONCURRENCY=1" >> .deepspeed_env
echo "_MAX_PER_RANK_IO_CONCURRENCY=1" >> .deepspeed_env
# echo "CUDA_VISIBLE_DEVICES=0,1,2,3" >> .deepspeed_env



echo "Number of nodes found as $NNODES"
sed "s/$/ slots=$NRANKS_PER_NODE/" $PBS_NODEFILE > $HOSTFILE

WORLD_SIZE=$(( NNODES * NRANKS_PER_NODE ))
# LAUNCH_PARAMS="--hostfile=$HOSTFILE --num_gpus=$WORLD_SIZE --num_nodes=$NNODES --master_addr $HOSTNAME --master_port=9901"
# LAUNCH_PARAMS="--hostfile=$HOSTFILE --master_port=29700"

LAUNCH_PARAMS="--include localhost:"
for ((gpu_id=0; gpu_id<NRANKS_PER_NODE; gpu_id++)); do
    LAUNCH_PARAMS+="$gpu_id"
    if [ $gpu_id -lt $((NRANKS_PER_NODE - 1)) ]; then
        LAUNCH_PARAMS+=","
    fi
done

USE_DEEPSPEED=1
ZERO_STAGE=3


EXIT_INTERVAL=20
DP=$((NNODES * NRANKS_PER_NODE))
WORLD_SIZE=$((TP*PP*DP))
# MICRO_BATCH=4
if [ "$NNODES" -eq 100 ] || [ "$NNODES" -eq 70 ]; then
    MICRO_BATCH=8
    GLOBAL_BATCH=$(( MICRO_BATCH * 2 ))
else
    MICRO_BATCH=8
    GLOBAL_BATCH=$(( MICRO_BATCH * DP * GRAD_ACC ))
fi


# MICRO_BATCH=$(( GLOBAL_BATCH / DP ))
CHECKPOINT_PATH=/local/scratch/llama2/tp${TP}_pp${PP}_dp${DP} 
# CHECKPOINT_PATH=/grand/projects/VeloC/am6429/scratch/llama2/tp${TP}_pp${PP}_dp${DP} 
# LOAD_CHECKPOINT_PATH=/grand/projects/VeloC/am6429/scratch/llama2/tp${TP}_pp${PP}_dp${DP}

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
       --num-layers $NUM_LAYERS \
       --hidden-size $HIDDEN_SIZE \
       --num-attention-heads $NUM_HEADS \
       --micro-batch-size $MICRO_BATCH \
       --global-batch-size $GLOBAL_BATCH \
       --seq-length $SEQ_LENGTH \
       --max-position-embeddings $SEQ_LENGTH \
       --train-iters $TRAIN_ITERS \
       --save $CHECKPOINT_PATH \
       --data-path $DATASET \
       --vocab-file ${VOCAB_PATH} \
	   --merge-file ${MERGE_PATH} \
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
       --save-interval $SAVE_INTERVAL \
       --eval-interval 1000 \
       --eval-iters 0 \
       --bf16 \
       --no-query-key-layer-scaling \
       --attention-dropout 0 \
       --hidden-dropout 0 \
       --use-rotary-position-embeddings \
       --untie-embeddings-and-output-weights \
       --swiglu \
       --normalization rmsnorm \
       --disable-bias-linear \
       --num-key-value-heads ${NUM_KV_HEADS} \
       --deepspeed \
       --exit-interval ${EXIT_INTERVAL} \
       --deepspeed_config=${CONFIG_JSON} \
       --zero-stage=${ZERO_STAGE} \
        --checkpoint-activations \
        --deepspeed-activation-checkpointing \
        --cpu-optimizer \
        --no-pipeline-parallel"


echo "Checkpointing using Torch.save approach"
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
		"offload_optimizer": {
			"device": "cpu",
			"pin_memory": true
		}
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
		"enabled": false
	}
}
EOT



log_str="${model_size_B}B-tp$TP-pp$PP-dp$DP-l$NUM_LAYERS-h$HIDDEN_SIZE-a$NUM_HEADS-sl$SEQ_LENGTH-gbs$GLOBAL_BATCH-mbs-$MICRO_BATCH-ckpt-$CKPT_APPROACH"
rm -rf $output_dir/log-$log_str.log
rm -rf $output_dir/log-$log_str-monitor.csv
rm -rf $output_dir/log-$log_str-monitor-*.csv
rm -rf $output_dir/log-$log_str-nsys.*
echo "NSYS_REPORT_DIR=${output_dir}/rep-${log_str}-%n">> .deepspeed_env
pdsh -w "$(awk '{printf "%s%s",sep,$1; sep=","}' $PBS_NODEFILE)" 'rm -rf /local/scratch/*'
# eval "rm -rf $CHECKPOINT_PATH"
# run_cmd="rm -rf $CHECKPOINT_PATH && time nsys profile --force-overwrite true -o $output_dir/report-$log_str --stats=true deepspeed ${LAUNCH_PARAMS} ${DIR}/pretrain_gpt.py $@ ${options} | tee $output_dir/log-$log_str.log"
# run_cmd="{ nsys profile --force-overwrite true -o ~/dl-io/dl-io-outputs/swapping/single-gpu-only-opt-off -t cuda,nvtx deepspeed ${LAUNCH_PARAMS} ${DIR}/pretrain_gpt.py ${options} ;} | tee -a $output_dir/log-$log_str.log"

run_cmd="nsys profile --force-overwrite true -o $output_dir/log-$log_str-nsys -t cuda,nvtx deepspeed ${LAUNCH_PARAMS} ${DIR}/pretrain_gpt.py ${options} > $output_dir/log-$log_str.log 2>&1"

echo "Monitoring starting."
# setsid python /home/am6429/dl-io/Megatron-DeepSpeed/monitor-gpu.py 0 $output_dir/log-$log_str-monitor.csv &
# monitor_pid=$!
# echo "Monitoring started at ${monitor_pid}."
# Start monitoring for each GPU
for gpu_id in $(seq 0 $((NRANKS_PER_NODE - 1))); do
    setsid python /home/am6429/dl-io/Megatron-DeepSpeed/monitor-gpu.py $gpu_id "$output_dir/log-$log_str-monitor-$gpu_id.csv" &
    monitor_pid[$gpu_id]=$!
    echo "Monitoring started for GPU $gpu_id at PID ${monitor_pid[$gpu_id]}."
done

echo $run_cmd
eval ${run_cmd}
# ds_pid=$!
# wait $ds_pid
echo "Deepspeed job completed."
# kill -2 $monitor_pid
# echo "SIGTERM (kill -2) instructed to monitoring script."
# wait $monitor_pid
# Terminate monitoring for all GPUs
for gpu_id in $(seq 0 $((NRANKS_PER_NODE - 1))); do
    kill -2 ${monitor_pid[$gpu_id]}
    echo "SIGTERM (kill -2) instructed to monitoring script for GPU $gpu_id."
    wait ${monitor_pid[$gpu_id]}
    echo "Killed the monitoring script for GPU $gpu_id."
done

echo "Monitoring completed. Start stitching CSV files."

for gpu_id in $(seq 0 $((NRANKS_PER_NODE - 1))); do
    cat "$output_dir/log-$log_str-monitor-$gpu_id.csv" >> "$output_dir/log-$log_str-monitor.csv"
    echo "Stitched file for $gpu_id."
    rm -rf "$output_dir/log-$log_str-monitor-$gpu_id.csv"
done

echo "Stitching complete. Deleting individual CSV files"


ls -ltrh "$CHECKPOINT_PATH/global_step$SAVE_INTERVAL/" >> "$output_dir/log-$log_str.log"
rm -rf $output_dir/*.sqlite
# eval "rm -rf $CHECKPOINT_PATH"
# rm -rf /local/scratch/*
# set +x