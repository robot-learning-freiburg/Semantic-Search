#!/bin/bash

# Define the partition on which the job shall run.
#SBATCH --partition alldlc_gpu-rtx2080    # short: -p <partition_name>

# Define a name for your job
#SBATCH --job-name detectron             # short: -J <job name>

# Define the files to write the outputs of the job to.
# Please note the SLURM will not create this directory for you, and if it is missing, no logs will be saved.
# You must create the directory yourself. In this case, that means you have to create the "logs" directory yourself.

#SBATCH --output logs/%x-%A-HelloCluster.out   # STDOUT  %x and %A will be replaced by the job name and job id, respectively. short: -o logs/%x-%A-job_name.out
#SBATCH --error logs/%x-%A-HelloCluster.err    # STDERR  short: -e logs/%x-%A-job_name.out
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=10
# Define the amount of memory required per node
#SBATCH --mem 64GB


echo "Workingdir: $PWD";
echo "Started at $(date)";

# A few SLURM variables
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

# Activate your environment
# You can also comment out this line, and activate your environment in the login node before submitting the job
source ~/.bashrc # Adjust to your path of Miniconda installation
conda activate multitask-d2

# Running the job
start=`date +%s`

timeout 23h python train_net.py --config-file configs/NYUv2/hm3d.yaml --num-gpus 4 --resume
if [[ $? -eq 124 ]]; then
  sbatch ${BASH_SOURCE[0]}
fi

end=`date +%s`
runtime=$((end-start))

echo Job execution complete.
echo Runtime: $runtime