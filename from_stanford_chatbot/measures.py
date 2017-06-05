import numpy as np
import math

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
        perp_sum += math.log(word_probability, base=2)
        i += 1

    # As specified in task description: ./docs/task_description
    # perp = 2^{(-1/n)*\sum^{n}_{t}(log_2(p(w_t | w_1, ... , w_t-1))} -
    perp = math.pow(2, (-1/i) * perp_sum)
    return perp