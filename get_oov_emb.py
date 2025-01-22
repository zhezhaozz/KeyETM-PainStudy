import argparse
import os
import re
import torch
import pandas as pd
import numpy as np

from collections import defaultdict
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
from sklearn.feature_extraction import text 

from embedded_topic_model.utils import preprocessing


if torch.cuda.is_available():
    device = torch.device(0)
else:
	device = "cpu"

parser = argparse.ArgumentParser(description='main', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', default='pain_study')
parser.add_argument('--model', default='bert')
args = parser.parse_args()

if args.model == 'bert':
	bert_model = 'bert-base-uncased'
elif args.model == 'bio_clinicalbert':
	bert_model = 'emilyalsentzer/Bio_ClinicalBERT'
elif args.model == 'pubmedbert_fulltext':
	bert_model = 'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext'
elif args.model == 'pubmedbert_abstract':
	bert_model = 'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract'
else:
	bert_model = 'biobert-v1.1/'

if args.dataset == 'pain_study':
	data_file = f'Data/{args.dataset}'
	corpus_file = f'Data/{args.dataset}/test_data.csv'
	test_file = f'Data/{args.dataset}/test_data.csv'
else:
	corpus_file = f'Data/{args.dataset}.csv'

seeds_file = f'{data_file}/seedword2.txt'
bert_file = f'{data_file}/embeddings/embedding_{args.model}.txt'

# initiate BERT
tokenizer = BertTokenizer.from_pretrained(bert_model)
model = BertModel.from_pretrained(bert_model, output_hidden_states=True).to(device)
model.eval()

# load data
print("Loading data... \n")
grants = pd.read_csv(corpus_file)
abstracts = grants['abstracts'].to_list()
cnt = defaultdict(int)
for abs in abstracts:
	data = re.split(r'[^a-zA-Z_]+', abs)
	for word in data:
		cnt[word] += 1

min_count = 3
vocabulary = set()
for word in cnt:
	if cnt[word] >= min_count and word.replace('_', ' ').strip() != '':
		vocabulary.add(word)

print(f"The size of vocabulary is {len(vocabulary)}")

test_data = pd.read_csv(test_file)
test_index = test_data["Index"].tolist()
lab_cols = [col for col in test_data.columns if col.startswith("label_")]
lab_cols = sorted(lab_cols)
test_labels = test_data[lab_cols].to_numpy()
#documents = df["summary"].tolist()
#documents = df["abstracts"].tolist()
stop_words = text.ENGLISH_STOP_WORDS.union(['narrative', 'description', 'project', 'abstract', 'summary', 'relevance', 
        'study'])

print("Constructing vocabulary... \n")
#vocabulary, _, _ = preprocessing.create_etm_datasets(
#                                    documents,
#                                    test_index=test_index,
#                                    test_labels=test_labels,
#                                    min_df=0.0001,
#                                    max_df=1.0,
#                                    stem_words=False,
#                                   )
# add seeds into vocabulary
print("Loading seeds... \n")
with open(seeds_file) as fin, open(f'{data_file}/oov_{args.model}.txt', 'w') as fout:
	for line in fin:
		seeds = line.strip().split(',')
		for seed in seeds:
			if seed not in vocabulary:
				fout.write(seed+'\n')
				vocabulary.add(seed)

# create embeddings for words
print(f"create embeddings in {args.model} space...")
with torch.no_grad():
	with open(bert_file, 'w') as f:
		f.write(f'{len(vocabulary)} 768\n')
		for word in tqdm(vocabulary):
			text = word.replace('_', ' ')
			input_ids = torch.tensor(tokenizer.encode(text, max_length=256, truncation=True)).unsqueeze(0).to(device)
			outputs = model(input_ids)
			hidden_states = outputs[2][-1][0]
			emb = torch.mean(hidden_states, dim=0).cpu()

			f.write(f'{word} '+' '.join([str(x.item()) for x in emb])+'\n')