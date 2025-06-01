#!/usr/bin/bash

#$ -S /bin/bash 
#$ -pe smp 8
#$ -cwd
#$ -l h_rt=4:00:00
#$ -r y

. ../../env/bin/activate
module load CBI r

export OMP_NUM_THREADS=${NSLOTS:-1}
python "$@" --job-idx $SGE_TASK_ID
