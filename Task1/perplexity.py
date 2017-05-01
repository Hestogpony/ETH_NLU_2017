import numpy as np


def get_sentence_dictionary():
    pass


def perplexity(task_letter, predicted_softmax_vecs, true_id):
    """
    ignore <pad> but include <eos>
    """
    file = open("group02.perplexity" + task_letter, mode="w")
    sentence_dictionary = get_sentence_dictionary()

    i = 0                       # Word index in current sentence
    total_perplexity = 1        # Initial Perplexity
    while dict[true_id] is not "<pad>":
        """ bla bla bla """
        total_perplexity *= predicted_softmax_vecs[i][true_id]

    file.write(total_perplexity + "\n")
    # file.writelines(OUTPUT_STUFF)
    file.close()


perplexity("A", np.array([0.1, 0.2, 0.7], dtype=np.float32), np.array([0, 1, 3], dtype=np.float32))



"""
Provide the ground truth last word as input to the RNN, not the last word you predicted.
This is common practice.
"""