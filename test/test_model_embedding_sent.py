import pandas as pd

from embedded_topic_model.utils import embedding, preprocessing
from embedded_topic_model.model import etm
from gensim.models import KeyedVectors

if __name__ == '__main__':
    df = pd.read_csv('Data/pain_study/test.csv')
    documents = df["text_cleaned"].tolist()

    sent_embeddings = embedding.create_word2vec_embedding_from_model(documents, model_name="biosentvec") 

    assert isinstance(
            sent_embeddings, KeyedVectors), "embeddings isn't KeyedVectors instance"
    
    vocabulary, train_dataset, test_dataset = preprocessing.create_etm_datasets(
                                    documents,
                                    min_df=0.005,
                                    max_df=0.75,
                                    train_size=1.0,
                                    )

    
    sent_etm = etm.ETM(
            vocabulary,
            embeddings=sent_embeddings,
            num_topics=3,
            epochs=1,
            train_embeddings=False,
        )