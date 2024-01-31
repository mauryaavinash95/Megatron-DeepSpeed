#!/bin/bash -l
### PBS -l nodes=8:system=polaris
### PBS -l walltime=01:00:00
### PBS -q debug-scaling
### PBS -A VeloC
### PBS -l filesystems=home:grand
echo "Submitted data parallel scaling job"
NNODES=$(wc -l < $PBS_NODEFILE)
echo "NUM_OF_NODES= ${NNODES}"

# rm -rf ~/dl-io/Megatron-DeepSpeed/megatron/fused_kernels/build/*
cd ~/dl-io/DeepSpeed/
rm -rf ~/dl-io/DeepSpeed/build/* && pip uninstall deepspeed -y && CMAKE_POSITION_INDEPENDENT_CODE=ON NVCC_PREPEND_FLAGS="--forward-unknown-opts" DS_BUILD_UTILS=1 DS_BUILD_VELOC_CKPT=1 pip install . --global-option="build_ext" --global-option="-j48"
# rm -rf /grand/projects/VeloC/am6429/scratch/*
# rm -rf /local/scratch/*
# cd ~/

echo "Scaling Ckpt Iter for 30B LLAMA2 (8 nodes)"
m=30
H=6656
F=17920
N=60
L=52
U=2048
S=4
K=50
I=1
P=$NNODES
T=4

# echo "Scaling Ckpt Iter for 13B LLAMA2 (4 nodes)"
# m=13
# H=5120
# F=13824
# N=40
# L=40
# U=2048
# S=4
# K=50
# I=1
# P=$NNODES
# T=4


# echo "Scaling Ckpt Iter for 7B LLAMA2 (2 nodes)"
# m=7
# H=4096
# F=11008
# N=32
# L=32
# U=2048
# S=4
# K=50
# I=1
# P=$NNODES
# T=4

# citers=(1 2 3 4 5 10 15 20)
# citers=(1 2 3 4 5 10)
citers=($ci_val)

# citers=(1 2 4 8 16 32)
for value in "${citers[@]}"; do
    # K=$(( value * 10 ))
    echo "================= Running for $value =============="
    # bash ~/dl-io/Megatron-DeepSpeed/my-llama2-cmd-diff-ckpt-iter.sh -c 0 -h 0 -m $m -H $H -F $F -N $N -L $L -U $U -S $S -K $K -I $value -P $P -T $T
    bash ~/dl-io/Megatron-DeepSpeed/my-llama2-cmd-diff-ckpt-iter.sh -c 1 -h 0 -m $m -H $H -F $F -N $N -L $L -U $U -S $S -K $K -I $value -P $P -T $T
    bash ~/dl-io/Megatron-DeepSpeed/my-llama2-cmd-diff-ckpt-iter.sh -c 2 -h 0 -m $m -H $H -F $F -N $N -L $L -U $U -S $S -K $K -I $value -P $P -T $T
    bash ~/dl-io/Megatron-DeepSpeed/my-llama2-cmd-diff-ckpt-iter.sh -c 3 -h 16 -m $m -H $H -F $F -N $N -L $L -U $U -S $S -K $K -I $value -P $P -T $T
    bash ~/dl-io/Megatron-DeepSpeed/my-llama2-cmd-diff-ckpt-iter.sh -c 4 -h 0 -m $m -H $H -F $F -N $N -L $L -U $U -S $S -K $K -I $value -P $P -T $T        
done