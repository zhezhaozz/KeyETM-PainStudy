import torch 
import yaml
import wandb
import argparse
import pickle
import os
import os.path as osp
import pandas as pd
import numpy as np

from embedded_topic_model.model.etm import ETM
from embedded_topic_model.utils import preprocessing
from sklearn.feature_extraction import text 


def main():
    torch.manual_seed(2024)
    np.random.seed(2024)
    # set up arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="configs/pain_study.yaml",
                        help="Which configuration to use. See into 'config' folder")
    parser.add_argument('--emb', type=str, default="bert",
                        help="Which embedding to use. The default is BERT")
    parser.add_argument('--use_iv', action="store_true",
                        help="Whether to replace OOV with IV")
    parser.add_argument('--project', type=str, default=None,
                        help="Name of the project")
    opt = parser.parse_args()

    with open(opt.config, 'r') as ymlfile:
         config = yaml.load(ymlfile, Loader=yaml.FullLoader)

    config_dataset = config['dataset']
    data_path = osp.join(
        config_dataset['folder-path'], config_dataset['data-file'])
    
    config_model = config['model']
    bs = config_model['bs']
    nt = config_model['nt']
    epochs = config_model['epochs']
    lambda_theta = config_model['lambda_theta']
    lambda_alpha = config_model['lambda_alpha']
    drop_out = config_model['drop_out']
    theta_act = config_model['theta_act']
    
    lr = config_model['lr']
    if opt.use_iv:
        seeds_path = osp.join(config_dataset['folder-path'], f"keywords/keywords_{opt.emb}_5.txt")
        res_data_path = osp.join(
            config_dataset['folder-path'], "experiments/bert_iv")
    else:
        seeds_path = osp.join(config_dataset['folder-path'], "seedword2.txt")
        res_data_path = osp.join(
            config_dataset['folder-path'], "experiments/bert_oov")
    model_path = config_model['path']    
    
    wandb.init(project=opt.project, config=config_model)

    #load_data
    print("Loading data... \n")
    df = pd.read_csv(data_path)
    seedwords = preprocessing.read_seedword(seeds_path, stem_words=False)
    #documents = df["summary"].tolist()
    documents = df["text_cleaned"].tolist()
    stop_words = text.ENGLISH_STOP_WORDS.union(['narrative', 'description', 'project', 'abstract', 'summary', 'relevance', 
             'study'])
    vocabulary, train_dataset, _ = preprocessing.create_etm_datasets(
                                    documents,
                                    min_df=0.001,
                                    max_df=0.85,
                                    train_size=1.0,
                                    stopwords=stop_words,
                                    stem_words=False,
                                    )
    print("done \n")

    print("Generating embeddings... \n")
    embeddings_file = osp.join(
            config_dataset['folder-path'], f"embeddings/embedding_{opt.emb}.txt")
    embeddings_mapping = {}

    with open(embeddings_file) as fin:
        for line in fin:
            data = line.strip().split()
            if len(data) != 769:
                continue
            word = data[0]
            emb = np.array([float(x) for x in data[1:]])
            emb = emb / np.linalg.norm(emb)
            embeddings_mapping[word] = emb
    
    print("done \n")
            
    #create model
    print("Set up prior matrix... \n")
    gamma_prior,gamma_prior_bin = preprocessing.get_gamma_prior(vocabulary,seedwords,nt,bs,embeddings_mapping,0.95)
    print(gamma_prior)
    #print(gamma_prior[:100])

    etm_instance = ETM(
                   vocabulary,
                   batch_size = bs,
                   embeddings=embeddings_mapping,
                   num_topics=nt,
                   epochs=epochs,
                   enc_drop = drop_out,
                   lambda_theta = lambda_theta,
                   lambda_alpha = lambda_alpha,
                   theta_act = theta_act,
                   lr = lr,
                   gamma_prior = gamma_prior,
                   gamma_prior_bin=gamma_prior_bin,
                   rho_size=768,
                   emb_size=768,
                   train_embeddings=False)
    
    #gamma_prior,gamma_prior_bin = preprocessing.get_gamma_prior(vocabulary,seedwords,nt,bs,etm_instance.embeddings)
    #etm_instance.fit(train_dataset)
    #for name, param in etm_instance.model.alphas.named_parameters():
    #    if(name=="4.weight"):
    #      inferred_topics = param.data.cpu().numpy()
    #selected_topics=_visualize_word_embeddings(inferred_topics,etm_instance.model,etm_instance.vocabulary)
    #for i in range(5):
        #print("run_"+str(i))
    print("Start training... \n")
    etm_instance.fit(train_dataset)
    topics = etm_instance.get_topics(50)
    print("Training Done \n")
    topic_coherence = etm_instance.get_topic_coherence()
    topic_diversity = etm_instance.get_topic_diversity()
    print(f'The topic coherence score is {topic_coherence} \n')
    print(f'The topic diversity score is {topic_diversity} \n')

    topic_word = etm_instance.get_topic_word_dist()
    word_matrix = etm_instance.get_topic_word_matrix()
    write_to_file(res_data_path,f'{opt.emb}_word_topic_dist.csv',topic_word,opt.emb)
    write_to_file(res_data_path,f'{opt.emb}_doc_topic_dist.csv',etm_instance.get_document_topic_dist(),opt.emb)
    write_to_file(res_data_path,f'{opt.emb}_word_matrix.csv',word_matrix,opt.emb)
    write_in_format(res_data_path,f'{opt.emb}_formatted_topic_word.pickle',word_matrix,topic_word)        

def write_in_format(res_path,file_name,words,topic_words):
    topic_words_dict = {}
    words_list = words[0]
    for topic_w,topic_idx in zip(topic_words,range(1,len(words)+1)):
        order_index=topic_w.numpy().argsort()[::-1].tolist()
        final_list = []
        for idx in order_index:
           final_list.append((words_list[idx],topic_w.numpy()[idx]))
        topic_words_dict["Topic "+str(topic_idx)]= final_list
    with open(os.path.join(res_path,file_name), 'wb') as handle:
         pickle.dump(topic_words_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def write_to_file(res_path,file_name,results, model_name):
    if(torch.is_tensor(results)):   
         df = pd.DataFrame(results.numpy())
    else:
         df = pd.DataFrame(results)
    if("doc_topic_dist.csv" in file_name):
         #df= df.drop(['Unnamed: 0'],axis=1)
         labels = []
         a = df.to_numpy()
         for i in range(len(a)):
             labels.append(np.asarray(a[i]).argmax())
         with open(os.path.join(res_path,f'{model_name}_ETM_labels_.csv'),'w') as f:
             for item in labels:
                 f.write(str(item))
                 f.write("\n")
                 
    df.to_csv(os.path.join(res_path,file_name))
   

def nearest_neighbors(word, embeddings, vocab, n_most_similar=20):
    vectors = embeddings.data.cpu().numpy()
    
    #index = vocab.index(word)
    #query = vectors[index]
    query = word
    ranks = vectors.dot(query).squeeze()
    denom = query.T.dot(query).squeeze()
    denom = denom * np.sum(vectors**2, 1)
    denom = np.sqrt(denom)
    ranks = ranks / denom
    mostSimilar = []
    [mostSimilar.append(idx) for idx in ranks.argsort()[::-1]]
    nearest_neighbors = mostSimilar[:n_most_similar]
    nearest_neighbors = [vocab[comp] for comp in nearest_neighbors]
    return nearest_neighbors

def _visualize_word_embeddings(queries,model,vocabulary):
        model.eval()

        # visualize word embeddings by using V to get nearest neighbors
        with torch.no_grad():
            try:
                embeddings = model.rho.weight  # Vocab_size x E
            except BaseException:
                embeddings = model.rho         # Vocab_size x E

            neighbors = {}
            for word,i in zip(queries,range(5)):
                neighbors["topic_"+str(i)] = nearest_neighbors(
                    word, embeddings, vocabulary)

            return neighbors

if __name__ == "__main__":
     main()