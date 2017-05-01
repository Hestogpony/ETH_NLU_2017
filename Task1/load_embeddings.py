import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np
from gensim import models

from config import cfg


def load_embedding(session, vocab, emb, path, dim_embedding):
    '''
    session        Tensorflow session object
    vocab          A dictionary mapping token strings to vocabulary IDs
    emb            Embedding tensor of shape vocabulary_size x dim_embedding
    path           Path to embedding file
    dim_embedding  Dimensionality of the external embedding.
    '''
    print("Loading external embeddings from %s" % path)
    # model = models.Word2Vec.load_word2vec_format(path, binary=False)
    model = models.KeyedVectors.load_word2vec_format(path, binary=False)
    external_embedding = np.zeros(shape=(cfg["vocab_size"], dim_embedding))
    matches = 0
    for tok, idx in vocab.items():
        if tok in model.vocab:
            external_embedding[idx] = model[tok]
            matches += 1
        else:
            print("%s not in embedding file" % tok)
            external_embedding[idx] = np.random.uniform(
                low=-0.25, high=0.25, size=dim_embedding)

    print("%d words out of %d could be loaded" % (matches, cfg["vocab_size"]))

    # This is not running the actual session, just dataflow programming for
    # converting the vocab dictionairy to a tensor with word embeddings
    pretrained_embeddings = tf.placeholder(tf.float32, [None, None])
    assign_op = emb.assign(pretrained_embeddings)
    out = session.run(assign_op, {pretrained_embeddings: external_embedding})
    return out