#!/bin/bash
# The interpreter used to execute the script

#“#SBATCH” directives that convey submission options:

#SBATCH --job-name=benchmark_bioword2vec_Keyetm
#SBATCH --mail-user=zzhaozhe@umich.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=50GB
#SBATCH --time=02:00:00
#SBATCH --account=vgvinodv99
#SBATCH --partition=standard

python train_bert_keyetm.py --config configs/bert_keyetm.yaml --project BERT_keyetm
python train_bert_keyetm.py --config configs/bert_keyetm.yaml --emb pubmedbert_abstract --project BERT_keyetm
python train_bert_keyetm.py --config configs/bert_keyetm.yaml --emb pubmedbert_fulltext --project BERT_keyetm
python train_bert_keyetm.py --config configs/bert_keyetm.yaml --use_iv --project BERT_keyetm
python train_bert_keyetm.py --config configs/bert_keyetm.yaml --emb pubmedbert_abstract --use_iv --project BERT_keyetm
python train_bert_keyetm.py --config configs/bert_keyetm.yaml --emb pubmedbert_fulltext --use_iv --project BERT_keyetm