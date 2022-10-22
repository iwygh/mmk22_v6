#!/bin/bash

### Optional. Set the job name
### Optional. Set the output filename.
### SLURM reads %x as the job name and %j as the job ID
### SBATCH --output=%x-%j.out
### REQUIRED. Specify the PI group for this job
### Optional. Request email when job begins and ends
### Optional. Specify email address to use for notification
### REQUIRED. Set the partition for your job.
### REQUIRED. Set the number of cores that will be used for this job.
### REQUIRED. Set the memory required for this job.
### REQUIRED. Specify the time required for this job, hhh:mm:ss

#SBATCH --mail-type=ALL
#SBATCH --mail-user=ianmatheson@email.arizona.edu
#SBATCH --account=malhotra
#SBATCH --job-name=check_plutino_resonances
#SBATCH --output=%u-%x-%A-%a.out
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16gb
#SBATCH --time=00:30:00
#SBATCH --array 1-690

module load python/3.8/3.8.2
source /home/u26/ianmatheson/test_venv_puma/bin/activate
python3 check_plutino_resonances_${SLURM_ARRAY_TASK_ID}.py
