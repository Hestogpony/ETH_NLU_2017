import numpy as np
import math
import gensim
from gensim.models.word2vec import Word2Vec



def load_model(fpath):

# TODO make this nice
    embeddings_file = fpath
    return Word2Vec.load(embeddings_file)
    #emb_size = 100

# TODO adapt this to the character based model
def perplexity(cfg, predicted_softmax_vecs, input_sentence, word_dictionary):
    """
    predicted_softmax_vecs      sentence length x 1 x vocab_size
    input_sentence              dim: vector of words in sentence
    word_dictionary             dictionary incl. pad, unk, bos and eos.  id -> word
    """

    i = 0                       # Word index in current sentence
    perp_sum = 0

    while i < len(input_sentence) and input_sentence[i] != cfg['PAD_ID'] and i < cfg['TEST_MAX_LENGTH']: # only 29 output nodes

        # These pred
        word_probability = predicted_softmax_vecs[i][0][input_sentence[i]]
        perp_sum += math.log(word_probability, 2)
        i += 1

    # As specified in task description: ./docs/task_description
    # perp = 2^{(-1/n)*\sum^{n}_{t}(log_2(p(w_t | w_1, ... , w_t-1))} -
    perp = math.pow(2, (-1/i) * perp_sum)
    return perp

 
# TODO adapt once it's plugged in
def vector_extrema_dist(reference, output, embbeding_path):
    """
    reference       string
    output          string
    """

    model = load_model(embbeding_path)

    def extrema(sentence):
        sentence = sentence.split(" ")
        vector_extrema = np.zeros(shape=(emb_size))
        for i, word in enumerate(sentence):
            if model[word] is not None:
                n = model[word]
                abs_n = np.abs(next)
                abs_v = np.abs(vector_extrema)
                for e in range(emb_size):
                    if abs_n > abs_v:
                        vector_extrema[e] = n[e]

        return vector_extrema

    ref_ext = extrema(reference)
    out_ext = extrema(output)
    return scipy.spatial.distance.cosine(ref_ext, out_ext)


