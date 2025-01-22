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


# use oov seeds only
#python get_oov_emb.py 
#python train_bert_keyetm.py --config configs/bert_keyetm.yaml --project BERT_keyetm

#python get_oov_emb.py --model pubmedbert_abstract
#python train_bert_keyetm.py --config configs/bert_keyetm.yaml --emb pubmedbert_abstract --project BERT_keyetm

#python get_oov_emb.py --model pubmedbert_fulltext
#python train_bert_keyetm.py --config configs/bert_keyetm.yaml --emb pubmedbert_fulltext --project BERT_keyetm

# use IV seeds
python get_oov_emb.py 
python get_iv_seed.py --topm 25 
#python get_iv_emb.py 
python train_bert_keyetm.py --config configs/bert_keyetm.yaml --use_iv --project BERT_keyetm

#python get_oov_emb.py --model pubmedbert_abstract
#python get_iv_seed.py --model pubmedbert_abstract --topm 25
#python get_iv_emb.py --model pubmedbert_abstract
#python train_bert_keyetm.py --config configs/bert_keyetm.yaml --emb pubmedbert_abstract --use_iv --project pubmedBERTasb_keyetm

#python get_oov_emb.py --model pubmedbert_fulltext
#python get_iv_seed.py --model pubmedbert_fulltext --topm 25
#python get_iv_emb.py --model pubmedbert_fulltext
#python train_bert_keyetm.py --config configs/bert_keyetm.yaml --emb pubmedbert_fulltext --use_iv --project pubmedBERTft_keyetm