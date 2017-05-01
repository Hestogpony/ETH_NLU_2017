import tensorflow as tf
from gensim import models

def load_embedding(self, session, vocab, emb, path, dim_embedding):
    '''
      session        Tensorflow session object
      vocab          A dictionary mapping token strings to vocabulary IDs
      emb            Embedding tensor of shape vocabulary_size x dim_embedding
      path           Path to embedding file
      dim_embedding  Dimensionality of the external embedding.
    '''
    print("Loading externsal embeddings from %s" % path)
    model = models.Word2Vec.load_word2vec_format(path, binary=False) # Where does this come from? import missing
    # google word2vec code is in C!
    external_embedding = np.zeros(shape=(FLAGS.vocab_size, dim_embedding))
    matches = 0
    for tok, idx in vocab.items():
        if tok in model.vocab:
            external_embedding[idx] = model[tok]
            matches += 1
        else:
            print("%s not in embedding file" % tok)
            external_embedding[idx] = np.random.uniform(low=-0.25, high=0.25, size=dim_embedding)
        
    print("%d words out of %d could be loaded" % (matches, FLAGS.vocab_size))
    
    # This is not running the actual session, just dataflow programming for converting the vocab dictionairy to a tensor with word embeddings
    pretrained_embeddings = tf.placeholder(tf.float32, [None, None]) 
    assign_op = emb.assign(pretrained_embeddings)
    session.run(assign_op, {pretrained_embeddings: external_embedding})