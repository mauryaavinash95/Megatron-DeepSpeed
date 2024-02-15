#!/bin/bash -l
### PBS -l nodes=8:system=polaris
### PBS -l walltime=01:00:00
### PBS -q debug-scaling
### PBS -A VeloC
### PBS -l filesystems=home:grand
echo "Submitted job"
NNODES=$(wc -l < $PBS_NODEFILE)
echo "NUM_OF_NODES= ${NNODES}"
# rm -rf ~/dl-io/Megatron-DeepSpeed/megatron/fused_kernels/build/*
source ~/.bash_profile
dlconda
cd ~/dl-io/DeepSpeed/
pip uninstall deepspeed -y && CMAKE_POSITION_INDEPENDENT_CODE=ON NVCC_PREPEND_FLAGS="--forward-unknown-opts" DS_BUILD_UTILS=1 DS_BUILD_VELOC_CKPT=1 DS_BUILD_CPU_ADAM=1 DS_BUILD_AIO=1 pip install . --global-option="build_ext" --global-option="-j48"
rm -rf /grand/projects/VeloC/am6429/scratch/*
rm -rf /local/scratch/*
cd ~/

MODEL_SIZE=13
if [[ $MODEL_SIZE == 0 ]]; then
    echo "================== 0B LLAMA2 (1 GPU)"
    m=3
    H=1024
    F=11008
    N=30
    L=16
    U=2048
    S=4
    K=5
	I=1000
    P=0
    T=0
    G=1
elif [[ $MODEL_SIZE == 3 ]]; then
    echo "================== 3B LLAMA2 (1 node)"
    m=3
    H=2560
    F=11008
    N=30
    L=32
    U=2048
    S=4
    K=5
	I=1000
    P=0
    T=0
    G=1
elif [[ $MODEL_SIZE == 5 ]]; then
    echo "================== 5B LLAMA2 (1 nodes)"
    m=5
    H=3072
    F=11008
    N=32
    L=32
    U=2048
    S=8
    K=5
	I=1000
    P=0
    T=0
    G=1
elif [[ $MODEL_SIZE == 7 ]]; then
    echo "================== 7B LLAMA2 (2 nodes)"
    m=7
    H=4096
    F=11008
    N=32
    L=32
    U=2048
    S=8
    K=5
	I=1000
    P=0
    T=0
    G=1
elif [[ $MODEL_SIZE == 13 ]]; then
    echo "================== 13B LLAMA2 (4 nodes)"
    m=13
    H=5120
    F=13824
    N=40
    L=40
    U=2048
    S=4
    K=5
	I=1000
    P=0
    T=0
    G=1
elif [[ $MODEL_SIZE == 30 ]]; then
    echo "================== 30B LLAMA2 (8 nodes)"
    m=30
    H=6656
    F=17920
    N=60
    L=52
    U=2048
    S=4
    K=5
	I=1000
    P=0
    T=0
    G=1
elif [[ $MODEL_SIZE == 70 ]]; then
    echo "================== 70B LLAMA2 (16 nodes)"
    m=70
    H=8192
    F=28672
    N=80
    L=64
    U=2048
    S=4
    K=5
	I=1000
    P=0
    T=0
    G=1
elif [[ $MODEL_SIZE == 175 ]]; then
    echo "================== 175B LLAMA2 (70 nodes)"
    m=175
    H=14336
    F=28672
    N=70
    L=112
    U=2048
    S=8
    K=5
    I=1000
    P=0
    T=0
    G=1
elif [[ $MODEL_SIZE == 500 ]]; then
    echo "================== 500B LLAMA2 (100 nodes)"
    m=500
    H=20000
    F=28672
    N=100
    L=160
    U=2048
    S=8
    K=5
    I=1000
    P=0
    T=0
    G=1
else
    echo "NNODES not in defined list  (NNODES = $NNODES)"
    exit 1
fi

# grad_acc=(4 8 16 32 64)
grad_acc=(1 2 4 8)
# grad_acc=(1)
for value in "${grad_acc[@]}"; do
    echo "===== Running for GRAD_ACC=$value ===== "
    bash ~/dl-io/Megatron-DeepSpeed/swapping-my-llama2-cmd.sh -c 4 -h 0 -m $m -H $H -F $F -N $N -L $L -U $U -S $S -K $K -I $I -P $P -T $T -G $value
done
