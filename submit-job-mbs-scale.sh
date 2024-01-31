#!/bin/bash -l
### PBS -l nodes=8:system=polaris
### PBS -l walltime=01:00:00
### PBS -q debug-scaling
### PBS -A VeloC
### PBS -l filesystems=home:grand
echo "Submitted data parallel scaling job"

# rm -rf ~/dl-io/Megatron-DeepSpeed/megatron/fused_kernels/build/*
cd ~/dl-io/DeepSpeed/
# rm -rf ~/dl-io/DeepSpeed/build/* && pip uninstall deepspeed -y && CMAKE_POSITION_INDEPENDENT_CODE=ON NVCC_PREPEND_FLAGS="--forward-unknown-opts" DS_BUILD_UTILS=1 DS_BUILD_VELOC_CKPT=1 pip install . --global-option="build_ext" --global-option="-j48"
# rm -rf /grand/projects/VeloC/am6429/scratch/*
# rm -rf /local/scratch/*
# cd ~/

NNODES=$(wc -l < $PBS_NODEFILE)
echo "NUM_OF_NODES= ${NNODES}"

echo "Scaling Ckpt Iter for 7B LLAMA2 (2 nodes)"
m=7
H=4096
F=11008
N=32
L=32
U=2048
S=4
K=5
I=1
P=$NNODES
T=4
I=1


# mbs_all=(2 4 8 16 32)
mbs_all=(1 2 4 8 16)
for mbs in "${mbs_all[@]}"; do
    echo "================= Running for MBS $mbs =============="
    # bash ~/dl-io/Megatron-DeepSpeed/my-llama2-cmd-diff-mbs.sh -c 4 -h 0 -m $m -H $H -F $F -N $N -L $L -U $U -S $S -K $K -I $I -P $P -T $T -M $mbs
    # bash ~/dl-io/Megatron-DeepSpeed/my-llama2-cmd-diff-mbs.sh -c 0 -h 0 -m $m -H $H -F $F -N $N -L $L -U $U -S $S -K $K -I $I -P $P -T $T -M $mbs
    # bash ~/dl-io/Megatron-DeepSpeed/my-llama2-cmd-diff-mbs.sh -c 1 -h 0 -m $m -H $H -F $F -N $N -L $L -U $U -S $S -K $K -I $I -P $P -T $T -M $mbs
    # bash ~/dl-io/Megatron-DeepSpeed/my-llama2-cmd-diff-mbs.sh -c 2 -h 0 -m $m -H $H -F $F -N $N -L $L -U $U -S $S -K $K -I $I -P $P -T $T -M $mbs
    bash ~/dl-io/Megatron-DeepSpeed/my-llama2-cmd-diff-mbs.sh -c 3 -h 16 -m $m -H $H -F $F -N $N -L $L -U $U -S $S -K $K -I $I -P $P -T $T -M $mbs
done