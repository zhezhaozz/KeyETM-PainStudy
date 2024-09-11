#!/bin/bash
# The interpreter used to execute the script

#“#SBATCH” directives that convey submission options:

#SBATCH --job-name=train_bioword2vec_Keyetm
#SBATCH --mail-user=zzhaozhe@umich.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=70GB
#SBATCH --time=03:00:00
#SBATCH --account=vgvinodv99
#SBATCH --partition=standard

source activate base
conda activate keyetm_pain
python3 train.py --config configs/pain_study.yaml --emb biowordvec --project KeyETM_small_l2