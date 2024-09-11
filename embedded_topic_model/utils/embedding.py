import sent2vec

from gensim.models import Word2Vec, KeyedVectors, FastText
from embedded_topic_model.utils.preprocessing import preprocess_sentence


# Class for a memory-friendly iterator over the dataset
class MemoryFriendlyFileIterator(object):
    def __init__(self, filename):
        self.filename = filename

    def __iter__(self):
        for line in open(self.filename):
            yield line.split()


def create_word2vec_embedding_from_dataset(
        dataset, dim_rho=300, min_count=1, sg=1,
        workers=25, negative_samples=10, window_size=4, iters=50, 
        embedding_file_path=None, save_c_format_w2vec=False, debug_mode=False) -> KeyedVectors:
    """
    Creates a Word2Vec embedding from dataset file or a list of sentences.
    If a file path is given, the file must be composed
    by a sequence of sentences separated by \\n.

    If the dataset is big, prefer using its file path.

    Parameters:
    ===
        dataset (str or list of str): txt file containing the dataset or a list of sentences
        dim_rho (int): dimensionality of the word embeddings
        min_count (int): minimum term frequency (to define the vocabulary)
        sg (int): whether to use skip-gram
        workers (int): number of CPU cores
        negative_samples (int): number of negative samples
        window_size (int): window size to determine context
        iters (int): number of iterations
        embedding_file_path (str): optional. File to save the word embeddings
        save_c_format_w2vec (bool): wheter to save embeddings as word2vec C format (BIN and TXT files)
        debug_mode (bool): wheter or not to log function's operations to the console. By default, no logs are made

    Returns:
    ===
        Word2VecKeyedVectors: mapping between words and their vector representations.
        Example:
            { 'water': nd.array([0.024187922, 0.053684134, 0.034520667, ... ]) }
    """
    assert isinstance(dataset, str) or isinstance(dataset, list), \
        'dataset must be file path or list of sentences'

    if isinstance(dataset, str):
        assert isinstance(embedding_file_path, str), \
            'if dataset is a file path, an output embeddings file path must be given'

    if save_c_format_w2vec:
        assert isinstance(embedding_file_path, str), \
            'if save_c_format_w2vec is True, an output embeddings file path must be given'

    if debug_mode:
        print('Creating memory-friendly iterator...')

    sentences = MemoryFriendlyFileIterator(dataset) if isinstance(
        dataset, str) else [document.split() for document in dataset]

    if debug_mode:
        print('Training Word2Vec model with dataset...')

    model = Word2Vec(
        sentences,
        min_count=min_count,
        sg=sg,
        vector_size=dim_rho,
        epochs=iters,
        workers=workers,
        negative=negative_samples,
        window=window_size)

    embeddings = model.wv

    if embedding_file_path is not None:
        if debug_mode:
            print('Saving word-vector mappings to file...')

        embeddings.save(embedding_file_path)

    if save_c_format_w2vec:
        if debug_mode:
            print('Saving BIN/TXT original C Word2vec files...')

        embeddings.save_word2vec_format(
            f'{embedding_file_path}.bin', binary=True)
        embeddings.save_word2vec_format(
            f'{embedding_file_path}.txt', binary=False)

    return embeddings

def create_word2vec_embedding_from_model(
        dataset, model_name="biowordvec", continue_train=False, num_epochs=None,
        embedding_file_path=None, save_c_format_w2vec=False, debug_mode=False) -> KeyedVectors:
    assert isinstance(dataset, str) or isinstance(dataset, list), \
        'dataset must be file path or list of sentences'

    if isinstance(dataset, str):
        assert isinstance(embedding_file_path, str), \
            'if dataset is a file path, an output embeddings file path must be given'

    if save_c_format_w2vec:
        assert isinstance(embedding_file_path, str), \
            'if save_c_format_w2vec is True, an output embeddings file path must be given'

    if debug_mode:
        print('Creating memory-friendly iterator...')

    if debug_mode:
        print('Training Word2Vec model with dataset...')

    if model_name == "biowordvec":
        sentences = MemoryFriendlyFileIterator(dataset) if isinstance(
        dataset, str) else [document.split() for document in dataset]
        # assert continue_train == False, "Do NOT try this or you will blow up your memory"
        print("Loading the BioWordVec model... \n")
        model = FastText.load_fasttext_format("/nfs/turbo/umms-vgvinodv2/users/zzhaozhe/pain_study/BioWordVec_PubMed_MIMICIII_d200.bin")
        if continue_train:
            print("Start to continue training... \n")
            model.build_vocab(sentences, update=True)  # Update the vocabulary
            model.train(sentences, total_examples=len(sentences), epochs=model.epochs)
            embeddings = model.wv
            print("Finished \n")
    elif model_name == "biosentvec":
        sentences = MemoryFriendlyFileIterator(dataset) if isinstance(
        dataset, str) else [preprocess_sentence(document) for document in dataset]
        assert continue_train == False, "Continue training BioSent2Vec is not supported"
        sentences = " ".join(sentences)
        model_path = "/nfs/turbo/umms-vgvinodv2/users/zzhaozhe/pain_study/BioSentVec_PubMed_MIMICIII-bigram_d700.bin"
        emb_model = sent2vec.Sent2vecModel()
        emb_model.load_model(model_path)
        embeddings = emb_model.embed_sentences(sentences)

    if embedding_file_path is not None:
        if debug_mode:
            print('Saving word-vector mappings to file...')

        embeddings.save(embedding_file_path)

    if save_c_format_w2vec:
        if debug_mode:
            print('Saving BIN/TXT original C Word2vec files...')

        embeddings.save_word2vec_format(
            f'{embedding_file_path}.bin', binary=True)
        embeddings.save_word2vec_format(
            f'{embedding_file_path}.txt', binary=False)

    return embeddings