import pandas as pd

from embedded_topic_model.utils import embedding, preprocessing
from embedded_topic_model.model import etm
from gensim.models import KeyedVectors


df = pd.read_csv('Data/pain_study/pain_preprocessed_data.csv')
documents = df["text_cleaned"].tolist()
print("Data loaded \n")

embeddings = KeyedVectors.load_word2vec_format('/nfs/turbo/umms-vgvinodv2/users/zzhaozhe/pain_study/biowordvec_embeddings_mapping.bin', binary=True) 
print("Embeddings created \n")
print(embeddings['drug'])

seedwords = preprocessing.read_seedword('Data/pain_study/seedword2.txt', stem_words=False)

vocabulary, train_dataset, test_dataset = preprocessing.create_etm_datasets(
                            documents,
                            min_df=0.005,
                            max_df=0.75,
                            train_size=1.0,
                            stem_words=False,
                            )
        
gamma_prior,gamma_prior_bin = preprocessing.get_gamma_prior(vocabulary,seedwords,15,45,embeddings)
        


