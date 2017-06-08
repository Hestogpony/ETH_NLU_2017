import numpy as np
import math
import gensim
from gensim.models.word2vec import Word2Vec

# TODO make this nice
embeddings_file = ""
model = Word2Vec.load(embeddings_file)
emb_size = 100

def perplexity(predicted_softmaxes, actual_answer, char_to_id):
    """
    predicted_softmaxes         matrix of [answer length (in chars) x vocab_size]
    actual_answer               the ground-truth answer in the corpus (a string)
    char_to_id                  maps characters to their ID
    """

    i = 0                       # Word index in current sentence
    perp_sum = 0

    while i < len(actual_answer) and i < len(predicted_softmaxes):
        char_prob = predicted_softmaxes[i][char_to_id[actual_answer[i]]]
        perp_sum += math.log(char_prob, 2)
        i += 1

    # As specified in task description: ./docs/task_description
    # perp = 2^{(-1/n)*\sum^{n}_{t}(log_2(p(w_t | w_1, ... , w_t-1))} -
    perp = math.pow(2, (-1/i) * perp_sum)
    return perp


# TODO adapt once it's plugged in
def vector_extrema_dist(reference, output):
    """
    reference       string
    output          string
    """

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


