import numpy as np
import math

from config import cfg


def output_perplexity(task_letter):
    """
    task_letter     [A, B or C]
    """
    output_file = open("output/group02.perplexity" + task_letter, mode="w")
    word_dictionary = get_word_dict()
    test_sentences = open("data/sentences.eval", mode="r")

    for test_sentence in test_sentences.readlines():
        # predicted_softmax_vecs: 30 X 20.000 X 1 Output Vector with softmax probabilities
        predicted_softmax_vecs = eval_neural_network(test_sentence)
        perp = perplexity(predicted_softmax_vecs, test_sentence, word_dictionary)
        output_file.write(str(perp) + "\n")

    output_file.close()


def perplexity(predicted_softmax_vecs, input_sentence, word_dictionary):
    """
    predicted_softmax_vecs      dim 29 X 20.000 , output Vector with softmax probabilities
    input_sentence              dim 30 , vector of words in sentence
    word_dictionary             dictionary of 20k most common words incl. pad, unk, bos and eos. id -> word
    """

    i = 0                       # Word index in current sentence
    perp_sum = 0

    while word_dictionary[input_sentence[i]] is not "pad" and i < cfg["sentence_length"]-1: # only 29 output nodes
        word_probability = predicted_softmax_vecs[i][input_sentence[i+1]]
        perp_sum += math.log(word_probability)
        i += 1

    # As specified in task description: ./docs/task_description
    # perp = 2^{(-1/n)*\sum^{n}_{t}(log(p(w_t | w_1, ... , w_t-1))} -
    perp = math.pow(2, (-1/i) * perp_sum)
    return perp



"""
Provide the ground truth last word as input to the RNN, not the last word you predicted.
This is common practice.
"""