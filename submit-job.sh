#!/bin/bash -l
### PBS -l nodes=70:system=polaris
### PBS -l walltime=01:00:00
### PBS -q debug-scaling
### PBS -A VeloC
### PBS -l filesystems=home:grand
echo "Submitted job"
NNODES=$(wc -l < $PBS_NODEFILE)
echo "NUM_OF_NODES= ${NNODES}"
# rm -rf ~/dl-io/Megatron-DeepSpeed/megatron/fused_kernels/build/*
cd ~/dl-io/DeepSpeed/
source ~/.bash_profile
dlconda
# rm -rf ~/dl-io/DeepSpeed/build/* && pip uninstall deepspeed -y && CMAKE_POSITION_INDEPENDENT_CODE=ON NVCC_PREPEND_FLAGS="--forward-unknown-opts" DS_BUILD_UTILS=1 DS_BUILD_VELOC_CKPT=1 pip install . --global-option="build_ext" --global-option="-j48"
rm -rf /grand/projects/VeloC/am6429/scratch/*
rm -rf /local/scratch/*
cd ~/

# bash ~/dl-io/Megatron-DeepSpeed/my-gpt-cmd-multi-nodes.sh 0 # This one won't log real time due to fusion_kernel recompilation
# bash ~/dl-io/Megatron-DeepSpeed/my-gpt-cmd-multi-nodes.sh 0
# bash ~/dl-io/Megatron-DeepSpeed/my-gpt-cmd-multi-nodes.sh 1
# bash ~/dl-io/Megatron-DeepSpeed/my-gpt-cmd-multi-nodes.sh 2
# bash ~/dl-io/Megatron-DeepSpeed/my-gpt-cmd-multi-nodes.sh 3 16

# bash ~/dl-io/Megatron-DeepSpeed/my-llama2-cmd.sh 0
# bash ~/dl-io/Megatron-DeepSpeed/my-llama2-cmd.sh 1
# bash ~/dl-io/Megatron-DeepSpeed/my-llama2-cmd.sh 2
# bash ~/dl-io/Megatron-DeepSpeed/my-llama2-cmd.sh 3 16

# This one won't log real time due to fusion_kernel recompilation
# -c 0 -h 0 -m 0 -H 0 -F 0 -N 0 -L 0 -U 0 -S 4 -K 0
# bash ~/dl-io/Megatron-DeepSpeed/my-llama2-cmd.sh -c 0 -h 0 -m 7 -H 4096 -F 11008 -N 32 -L 32 -U 2048 -S 4 -K 5
# bash ~/dl-io/Megatron-DeepSpeed/my-llama2-cmd.sh -c 0 -h 0 -m 13 -H 5120 -F 13824 -N 40 -L 40 -U 2048 -S 4 -K 5

if [[ $NNODES == 0 ]]; then
    echo "================== 0B LLAMA2 (1 GPU)"
    m=3
    H=1024
    F=11008
    N=30
    L=16
    U=2048
    S=4
    K=5
	I=1
    P=1
    T=4
elif [[ $NNODES == 1 ]]; then
    echo "================== 3B LLAMA2 (1 node)"
    m=3
    H=2560
    F=11008
    N=30
    L=32
    U=2048
    S=4
    K=5
	I=1
    P=$NNODES
    T=4
elif [[ $NNODES == 2 ]]; then
    echo "================== 7B LLAMA2 (2 nodes)"
    m=7
    H=4096
    F=11008
    N=32
    L=32
    U=2048
    S=8
    K=5
	I=1
    P=$NNODES
    T=4
elif [[ $NNODES == 4 ]]; then
    echo "================== 13B LLAMA2 (4 nodes)"
    m=13
    H=5120
    F=13824
    N=40
    L=40
    U=2048
    S=4
    K=5
	I=1
    P=$NNODES
    T=4
elif [[ $NNODES == 8 ]]; then
    echo "================== 30B LLAMA2 (8 nodes)"
    m=30
    H=6656
    F=17920
    N=60
    L=52
    U=2048
    S=4
    K=5
	I=1
    P=$NNODES
    T=4
elif [[ $NNODES == 20 ]]; then
    echo "================== 70B LLAMA2 (16 nodes)"
    m=70
    H=8192
    F=28672
    N=80
    L=64
    U=2048
    S=4
    K=5
	I=1
    P=$NNODES
    T=4
elif [[ $NNODES == 70 ]]; then
    echo "================== 175B LLAMA2 (70 nodes)"
    m=175
    H=14336
    F=28672
    N=70
    L=112
    U=2048
    S=8
    K=5
    I=1
    P=$NNODES
    T=4
elif [[ $NNODES == 100 ]]; then
    echo "================== 500B LLAMA2 (100 nodes)"
    m=500
    H=20480
    F=28672
    N=100
    L=160
    U=2048
    S=8
    K=5
    I=1
    P=$NNODES
    T=4
else
    echo "NNODES not in defined list  (NNODES = $NNODES)"
    exit 1
fi

# bash ~/dl-io/Megatron-DeepSpeed/my-llama2-cmd.sh -c 0 -h 0 -m $m -H $H -F $F -N $N -L $L -U $U -S $S -K $K -I $I -P $P -T $T
# bash ~/dl-io/Megatron-DeepSpeed/my-llama2-cmd.sh -c 0 -h 0 -m $m -H $H -F $F -N $N -L $L -U $U -S $S -K $K -I $I -P $P -T $T
bash ~/dl-io/Megatron-DeepSpeed/my-llama2-cmd.sh -c 1 -h 0 -m $m -H $H -F $F -N $N -L $L -U $U -S $S -K $K -I $I -P $P -T $T
# bash ~/dl-io/Megatron-DeepSpeed/my-llama2-cmd.sh -c 2 -h 0 -m $m -H $H -F $F -N $N -L $L -U $U -S $S -K $K -I $I -P $P -T $T
# bash ~/dl-io/Megatron-DeepSpeed/my-llama2-cmd.sh -c 4 -h 0 -m $m -H $H -F $F -N $N -L $L -U $U -S $S -K $K -I $I -P $P -T $T
# bash ~/dl-io/Megatron-DeepSpeed/my-llama2-cmd.sh -c 3 -h 16 -m $m -H $H -F $F -N $N -L $L -U $U -S $S -K $K -I $I -P $P -T $T
