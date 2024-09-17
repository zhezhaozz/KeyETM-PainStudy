import torch 
import yaml
import wandb
import argparse
import pickle
import os
import os.path as osp
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.io as pio

from embedded_topic_model.utils import embedding
from embedded_topic_model.model.etm import ETM
from embedded_topic_model.utils import preprocessing
from gensim.models import KeyedVectors
from sklearn.feature_extraction import text 

def main():
    torch.manual_seed(2019)
    np.random.seed(2019)

    # set up arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="configs/pain_study.yaml",
                        help="Which configuration to use. See into 'config' folder")
    parser.add_argument('--emb', type=str, default=None,
                        help="Which embedding to use. The default is to train word2vec embedding using the training set. Users can also input biowordvec and biosentvec")
    parser.add_argument('--project', type=str, default=None,
                        help="Name of the project")
    opt = parser.parse_args()

    with open(opt.config, 'r') as ymlfile:
         config = yaml.load(ymlfile, Loader=yaml.FullLoader)

    config_dataset = config['dataset']
    res_data_path = osp.join(
        config_dataset['folder-path'], config_dataset['result-file'])
    data_path = osp.join(
        config_dataset['folder-path'], config_dataset['data-file'])
    seedword_path = osp.join(
        config_dataset['folder-path'], config_dataset['sw-file'])
    config_model = config['model']
    bs = config_model['bs']
    nt = config_model['nt']
    epochs = config_model['epochs']
    lambda_theta = config_model['lambda_theta']
    lambda_alpha = config_model['lambda_alpha']
    drop_out = config_model['drop_out']
    theta_act = config_model['theta_act']
    continue_train = config_model['continue_train']

    config_debug = config['debug']
    draw_latent = config_debug['draw_latent_space']
    
    lr = config_model['lr']
    model_path = config_model['path']    
    
    wandb.init(project=opt.project, config=config_model)

    #load_data
    print("Loading data... \n")
    df = pd.read_csv(data_path)
    seedwords = preprocessing.read_seedword(seedword_path, stem_words=False)
    #documents = df["summary"].tolist()
    documents = df["text_cleaned"].tolist()
    stop_words = text.ENGLISH_STOP_WORDS.union(['narrative', 'description', 'project', 'abstract', 'summary', 'relevance', 
             'study'])
    vocabulary, train_dataset, test_dataset = preprocessing.create_etm_datasets(
                                    documents,
                                    min_df=0.005,
                                    max_df=1.0,
                                    train_size=1.0,
                                    stopwords=stop_words,
                                    stem_words=False,
                                    )
    print("done \n")
    #gamma_prior,gamma_prior_bin = preprocessing.get_gamma_prior(vocabulary,seedwords,nt,bs)
    #Training word2vec embeddings
    print("Generating embeddings... \n")
    if opt.emb != None:
        if continue_train or opt.emb=="biosentvec":
            embeddings_mapping = embedding.create_word2vec_embedding_from_model(documents, model_name=opt.emb, continue_train=continue_train) 
            embeddings_mapping.save(os.path.join(model_path,f'{opt.emb}_embeddings_mapping_updated.kv'))
        else:       
            print("Loading BioWord2Vec... \n")
            embeddings_mapping = KeyedVectors.load(os.path.join(model_path, f'{opt.emb}_embeddings_mapping_updated.kv'))
            #embeddings_mapping = KeyedVectors.load_word2vec_format(os.path.join(model_path, f'{opt.emb}_embeddings_mapping.bin'), binary=True)
    else:
        if os.path.exists(os.path.join(model_path,'embeddings_mapping.kv')):
            embeddings_mapping = KeyedVectors.load(os.path.join(model_path,'embeddings_mapping.kv'))
            with open(os.path.join(model_path,'vocabulary.pickle'), 'rb') as handle:
                vocabulary =pickle.load(handle)
            with open(os.path.join(model_path,'train.pickle'), 'rb') as handle:
                train_dataset = pickle.load(handle)
        else:
            embeddings_mapping = embedding.create_word2vec_embedding_from_dataset(documents) 
            embeddings_mapping.save(os.path.join(model_path,'embeddings_mapping.kv')) 
            #df = pd.read_csv(data_path)
            #documents = df["summary"].tolist()
            #documents = df["text_cleaned"].tolist()
            #vocabulary, train_dataset, test_dataset = preprocessing.create_etm_datasets(
            #                           documents,
            #                           min_df=0.01,
            #                           max_df=1.0, #0.75,
            #                           train_size=1.0,
            #                           )
            with open(os.path.join(model_path,'train.pickle'), 'wb') as handle:
                pickle.dump(train_dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(os.path.join(model_path,'vocabulary.pickle'), 'wb') as handle:
                pickle.dump(vocabulary, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    print("done \n")
            
    #create model
    print("Set up prior matrix... \n")
    gamma_prior,gamma_prior_bin = preprocessing.get_gamma_prior(vocabulary,seedwords,nt,bs,embeddings_mapping,0.75)
    print(gamma_prior)
    #print(gamma_prior[:100])
    if opt.project == "test_run":
        emb_size=300
    else:
        emb_size=200

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
                   rho_size=emb_size,
                   emb_size=emb_size,
                   train_embeddings=False,
                   latent_diagnosis=draw_latent)
    
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

    if draw_latent:
        plt_data = etm_instance.prepare_plotly_data()
        visualize_latent_spaces(plt_data, res_data_path, 'latent_space_animation.html')

    topic_word = etm_instance.get_topic_word_dist()
    word_matrix = etm_instance.get_topic_word_matrix()
    write_to_file(res_data_path,'word_topic_dist.csv',topic_word)
    write_to_file(res_data_path,'doc_topic_dist.csv',etm_instance.get_document_topic_dist())
    write_to_file(res_data_path,'word_matrix.csv',word_matrix)
    write_in_format(res_data_path,'formatted_topic_word.pickle',word_matrix,topic_word)        

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


def write_to_file(res_path,file_name,results):
    if(torch.is_tensor(results)):   
         df = pd.DataFrame(results.numpy())
    else:
         df = pd.DataFrame(results)
    if(file_name == "doc_topic_dist.csv"):
         #df= df.drop(['Unnamed: 0'],axis=1)
         labels = []
         a = df.to_numpy()
         for i in range(len(a)):
             labels.append(np.asarray(a[i]).argmax())
         with open(os.path.join(res_path,'ETM_labels_.csv'),'w') as f:
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

def visualize_latent_spaces(data, res_path, file_name):
    # Create the layout with slider
    layout = go.Layout(
        title="Latent Space Embeddings over Epochs",
        xaxis=dict(title='Latent Dimension 1'),
        yaxis=dict(title='Latent Dimension 2'),
        updatemenus=[dict(
            type="buttons", 
            showactive=False, 
            buttons=[dict(label="Play",
                        method="animate",
                        args=[None, {"frame": {"duration": 500, "redraw": True},
                                    "fromcurrent": True, "transition": {"duration": 300}}]),
                    dict(label="Pause",
                        method="animate",
                        args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}])])]
    )

    # Create frames
    frames = [go.Frame(data=[trace], name=str(i)) for i, trace in enumerate(data)]

    # Create the figure
    fig = go.Figure(data=data[:1], frames=frames, layout=layout)

    # Add frame visibility update on slider change
    fig.update(frames=frames)
    fig.update_layout(
        sliders=[{
            "steps": [
                {"method": "animate", "label": str(i+1), "args": [[str(i)], {"frame": {"duration": 300, "redraw": True}, "mode": "immediate"}]}
                for i in range(len(frames))
            ],
            "active": 0,
        }]
    )

    pio.write_html(fig, os.path.join(res_path,file_name))

if __name__ == "__main__":
     main()