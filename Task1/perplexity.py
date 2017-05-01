import numpy as np
import math


def get_word_dict():
    pass


def eval_neural_network(input_sentence):
    pass


# task_letter: [A, B or C]
def output_perplexity(task_letter):
    output_file = open("output/group02.perplexity" + task_letter, mode="w")
    word_dictionary = get_word_dict()
    test_sentences = open("data/sentences.eval", mode="r")

    for test_sentence in test_sentences.readlines():
        # # predicted_softmax_vecs: 30 X 20.000 X 1 Output Vector with softmax probabilities
        predicted_softmax_vecs = eval_neural_network(test_sentence)
        perp = perplexity(predicted_softmax_vecs, test_sentence, word_dictionary)
        output_file.write(str(perp) + "\n")

    output_file.close()


# predicted_softmax_vecs: 30 X 20.000 X 1 Output Vector with softmax probabilities
# sentence: 30 X 1 Vector of words in sentence
# word_dictionary: dictionary of 20k most common words incl. <pad>, <unk>, <bos> and <eos>.
def perplexity(predicted_softmax_vecs, sentence, word_dictionary):
    i = 0                       # Word index in current sentence
    perp_sum = 0

    while word_dictionary[sentence[i]] is not "<pad>":
        word_probability = predicted_softmax_vecs[i][sentence[i+1]]
        perp_sum += math.log(word_probability)
        i += 1

    # perp = 2^{(-1/n)*\sum^{n}_{t}(log(p(w_t | w_1, ... , w_t-1))} - As specified in assignment doc
    perp = math.pow(2, (-1/i) * perp_sum)
    return perp



"""
Provide the ground truth last word as input to the RNN, not the last word you predicted.
This is common practice.
"""