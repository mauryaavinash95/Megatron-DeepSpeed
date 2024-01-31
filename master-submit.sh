#!/bin/bash

# Set your username
USERNAME="am6429"

# List of node counts to try
# NODE_COUNTS=(2 4 8 20 70 1)
CIS=(1 2 3 4 5 10)

# Function to check if there are any running or queued jobs
function check_jobs {
    qstat_output=$(qstat -u $USERNAME)
    if [ -n "$qstat_output" ]; then
        return 1  # Jobs running or queued
    else
        return 0  # No jobs running or queued
    fi
}

NNODES=8
for CI in "${CIS[@]}"; do
    if [ $NNODES -gt 10 ]; then
        QUEUE_NAME="prod"
    else
        QUEUE_NAME="debug-scaling"
    fi
    while true; do
        # Check if there are any running or queued jobs
        check_jobs
        
        if [ $? -eq 0 ]; then
            # No jobs running or queued, submit the job
            echo "Submitting job with CI=$CI..."
            echo "qsub  -v ci_val=$CI -l select=$NNODES -l filesystems=home -l walltime=01:00:00 -q $QUEUE_NAME -A VeloC ~/dl-io/Megatron-DeepSpeed/submit-job-ckpt-iter-scale.sh"
            # Set the queue name based on the condition
            qsub  -v ci_val=$CI -l select=$NNODES -l filesystems=home -l walltime=01:00:00 -q $QUEUE_NAME -A VeloC ~/dl-io/Megatron-DeepSpeed/submit-job-ckpt-iter-scale.sh
            break  # Exit the while loop after submitting the job
        else
            echo "Job already running or queued. Skipping submission for CI=$CI nodes on $QUEUE_NAME."
        fi
        # Sleep for a while before checking again
        sleep 60  # Adjust this interval as needed (in seconds)
    done
done


# Main loop
# for NNODES in "${NODE_COUNTS[@]}"; do
#     if [ $NNODES -gt 10 ]; then
#         QUEUE_NAME="prod"
#     else
#         QUEUE_NAME="debug-scaling"
#     fi
#     while true; do
#         # Check if there are any running or queued jobs
#         check_jobs
        
#         if [ $? -eq 0 ]; then
#             # No jobs running or queued, submit the job
#             echo "Submitting job with $NNODES nodes..."
#             # Set the queue name based on the condition
#             qsub -l select=$NNODES -l filesystems=home -l walltime=01:00:00 -q $QUEUE_NAME -A VeloC ~/dl-io/Megatron-DeepSpeed/submit-job.sh
#             break  # Exit the while loop after submitting the job
#         else
#             echo "Job already running or queued. Skipping submission for $NNODES nodes on $QUEUE_NAME."
#         fi
#         # Sleep for a while before checking again
#         sleep 30  # Adjust this interval as needed (in seconds)
#     done
# done