import numpy as np
from collections import defaultdict
import argparse

parser = argparse.ArgumentParser(description='main', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', default='pain_study')
parser.add_argument('--model', default='bert')
parser.add_argument('--num_iter', default=2, type=int)
parser.add_argument('--topm', default=10, type=int)
args = parser.parse_args()

dataset = args.dataset
model = args.model
num_iter = args.num_iter
topm = args.topm

data_file = f'Data/{args.dataset}'
seeds_file = f'{data_file}/seedword2.txt'
bert_file = f'{data_file}/embeddings/embedding_{args.model}.txt'

topics = []
with open(seeds_file) as fin:
	for line in fin:
		data = line.strip().split(',')
		topics.append(data[:])

word2emb = {}
with open(bert_file) as fin:
	for line in fin:
		data = line.strip().split()
		if len(data) != 769:
			continue
		word = data[0]
		emb = np.array([float(x) for x in data[1:]])
		emb = emb / np.linalg.norm(emb)
		word2emb[word] = emb

oov = set()
with open(f'{data_file}/oov.txt') as fin:
	for line in fin:
		data = line.strip()
		oov.add(data)

if num_iter == 0:
	out_file = f'{data_file}/keywords/keywords_1.txt'
else:
	num_iter += 1
	out_file = f'{data_file}/keywords/keywords_{model}_{num_iter}.txt'

print("Retrieving in-vocabulary seeds using {model} with {num_iter} iterations \n")
for iter in range(num_iter):
	print(f"Iteration: {iter} \n")
	with open(out_file, 'w') as fout:
		for idx, topic in enumerate(topics):
			word2score = defaultdict(float)
			for word in word2emb:
				if word in oov:
					continue
				for term in topic:
					word2score[word] += np.dot(word2emb[word], word2emb[term])
			score_sorted = sorted(word2score.items(), key=lambda x: x[1], reverse=True)[:100]
			new_topic = [x[0] for x in score_sorted][:topm]
			topics[idx] = new_topic
			print(','.join(new_topic)+'\n')
			fout.write(','.join(new_topic)+'\n')