import wandb
import pandas as pd

from embedded_topic_model.utils import embedding, preprocessing
from embedded_topic_model.model.etm import ETM
from gensim.models import KeyedVectors


config = dict(
     bs=45,
     nt=15,
     epochs=30,
     drop_out=0.5,
     theta_act="softplus",
     lr=0.005,
     lambda_theta=35.0,
     lambda_alpha=0.005
)

def make(config):
    # Make the data
    df = pd.read_csv('Data/pain_study/pain_preprocessed_data.csv')
    documents = df["text_cleaned"].tolist()

    # read seeds
    seedwords = preprocessing.read_seedword('Data/pain_study/seedword2.txt', stem_words=False)

    # load embedding 
    embeddings = KeyedVectors.load_word2vec_format('/nfs/turbo/umms-vgvinodv2/users/zzhaozhe/pain_study/biowordvec_embeddings_mapping.bin', binary=True) 

    # preprocessing
    vocabulary, train_dataset, test_dataset = preprocessing.create_etm_datasets(
                            documents,
                            min_df=0.005,
                            max_df=0.75,
                            train_size=1.0,
                            stem_words=False,
                            )
    
    gamma_prior,gamma_prior_bin = preprocessing.get_gamma_prior(vocabulary,seedwords,15,45,embeddings)

    # define model
    etm_model = ETM(
                   vocabulary,
                   batch_size = config.bs,
                   embeddings=embeddings,
                   num_topics=config.nt,
                   epochs=config.epochs,
                   enc_drop = config.drop_out,
                   lambda_theta = config.lambda_theta,
                   lambda_alpha = config.lambda_alpha,
                   theta_act = config.theta_act,
                   lr = config.lr,
                   gamma_prior = gamma_prior,
                   gamma_prior_bin=gamma_prior_bin,
                   debug_mode=True,
                   train_embeddings=False)

    return etm_model.model, train_dataset, etm_model.optimizer





        



