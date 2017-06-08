import numpy as np
import numpy.linalg.norm
import math
import gensim
from gensim.models.word2vec import Word2Vec
import scipy.spatial
import re

class Measure(object):
    """docstring for Measure"""
    def __init__(self, cfg, embbeding_path_from_chatbot_args_parameter):
        self.cfg = cfg
        self.model = self.load_model(embbeding_path_from_chatbot_args_parameter)
        self.emb_size = 100

    def load_model(self,fpath):

        embeddings_file = fpath
        return Word2Vec.load(embeddings_file)

    def perplexity(self, cfg, predicted_softmax_vecs, input_sentence):
        """
        predicted_softmax_vecs      sentence length x 1 x vocab_size
        input_sentence              dim: vector of words in sentence
        """

        i = 0                       # Word index in current sentence
        perp_sum = 0

        while i < len(input_sentence) and input_sentence[i] != self.cfg['PAD_ID'] and i < self.cfg['TEST_MAX_LENGTH']: # only 29 output nodes

            # These pred
            word_probability = predicted_softmax_vecs[i][0][input_sentence[i]]
            perp_sum += math.log(word_probability, 2)
            i += 1

        # As specified in task description: ./docs/task_description
        # perp = 2^{(-1/n)*\sum^{n}_{t}(log_2(p(w_t | w_1, ... , w_t-1))} -
        perp = math.pow(2, (-1/i) * perp_sum)
        return perp

    def vector_extrema_dist(self, predicted_softmax_vecs, reference_ids, word_dictionary):
            """
            predicted_softmax_vecs      sentence length x 1 x vocab_size
            reference_ids               vector of word ids
            word_dictionary             dictionary incl. pad, unk, bos and eos.  id -> word
            """

            def extrema(sentence):
                # <BG> kind of a redundant move to concat everything before splitting it up again.
                # But this way we ensure consistent splitting of non-alphanumeric characters
                sentence = " ".join(sentence)
                sentence = re.sub(r"(\W^<>)", r" \1 ", sentence)
                sentence = sentence.split(" ")    # Not needed here
                vector_extrema = np.zeros(shape=(self.emb_size))
                for i, word in enumerate(sentence):
                    if word in self.model.wv.vocab:
                        n = self.model[word]
                        abs_n = np.abs(n)
                        abs_v = np.abs(vector_extrema)
                        for e in range(self.emb_size):
                            if abs_n[e] > abs_v[e]:
                                vector_extrema[e] = n[e]

                return vector_extrema

            reference = [word_dictionary[x] for x in reference_ids]
            output = [word_dictionary[np.argmax(x[0])] for x in predicted_softmax_vecs]
            

            ref_ext = extrema(reference)
            out_ext = extrema(output)
            
            if norm(ref_ext) != 0 and norm(out_ext) != 0:
                return scipy.spatial.distance.cosine(ref_ext, out_ext)
            else:
                return 2.0
