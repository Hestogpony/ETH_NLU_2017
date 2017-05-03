import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np
from collections import Counter
import pickle

import load_embeddings
import model

from config import cfg


class Reader(object):

    def __init__(self, vocab_size, max_sentences=-1):
        self.vocab_size = vocab_size
        self.max_sentences = max_sentences
        self.vocab_dict = {}
        # self.id_data

    def build_dict(self, path):
        """ Ordered by word count, only the 20k most frequent words are being used """
        print("building dictionairy...")

        # load the dictionairy if it's there
        if os.path.isfile("dict.p"):
            self.vocab_dict = pickle.load(open("dict.p", "rb"))
            return None

        cnt = Counter()
        with open(path, 'r') as f:
            for line in f:
                for word in line.split():
                    cnt[word] += 1
            # most_common returns tuples (word, count)
            # 4 spots are reserved for the special tags
            vocab_with_counts = cnt.most_common(self.vocab_size - 4)
            vocab = [i[0] for i in vocab_with_counts]
            ids = range(self.vocab_size - 4)
            self.vocab_dict = dict(list(zip(vocab, ids)))

            self.vocab_dict["<bos>"] = cfg["vocab_size"] - 4
            self.vocab_dict["<eos>"] = cfg["vocab_size"] - 3
            self.vocab_dict["<unk>"] = cfg["vocab_size"] - 2
            self.vocab_dict["<pad>"] = cfg["vocab_size"] - 1

            pickle.dump(self.vocab_dict, open("dict.p", "wb"))

    def read_sentences(self, path):
        """ Include the tags <bos>, <eos>, <pad>, <unk> """
        # Read sentences, convert to IDs according to the dict, pad them
        print("reading sentences...")
        with open(path, 'r') as f:
            sentence_list = []
            if self.max_sentences == -1:
                for line in f:
                    tokens = self.add_tags(line.split())
                    sentence = self.convert_sentence(tokens)
                    sentence_list.append(sentence)
            else:
                for i in range(self.max_sentences):
                    # last token is the newline character
                    tokens = f.readline().split()[:-1]
                    tokens = self.add_tags(tokens)
                    sentence = self.convert_sentence(tokens)
                    sentence_list.append(sentence)
            self.id_data = np.array(sentence_list, dtype=np.int32)

    def add_tags(self, tokens):
        """
        tokens      list of words
        """
        tokens.insert(0, "<bos>")
        tokens.append("<eos>")
        tokens.extend((cfg["sentence_length"] - len(tokens)) * ["<pad>"])
        return tokens

    def convert_sentence(self, tokens):
        sentence = np.zeros(shape=cfg["sentence_length"], dtype=np.int32)
        for idx, word in enumerate(tokens):
            # translate according to dict
            if word in self.vocab_dict:
                sentence[idx] = self.vocab_dict[word]
            else:
                sentence[idx] = self.vocab_dict["<unk>"]
        return sentence

    # use the tensorflow library
    def one_hot_encode(self, input_matrix):
        """
        input_matrix           ndarray dim: #sentences x 30
        return                 matrix dim: #sentences x 30 x vocab_size
        """
        print("doing the one hot encoding...")
        inp = tf.placeholder(tf.int32, [None, cfg["sentence_length"]])
        output = tf.one_hot(indices=inp, depth=cfg[
                            "vocab_size"], axis=-1, dtype=tf.float32)

        sess = tf.Session()
        self.one_hot_data = sess.run(output, feed_dict={inp: input_matrix})
        # dtype should be float32
        # print(self.one_hot_data.shape)


def main():
    reader = Reader(vocab_size=cfg["vocab_size"],
                    max_sentences=cfg["max_sentences"])
    reader.build_dict(cfg["path"]["train"])
    reader.read_sentences(cfg["path"]["train"])
    reader.one_hot_encode(reader.id_data)

    sess = tf.Session()
    # embeddings = tf.placeholder(dtype=tf.float32, shape=[
                                # reader.vocab_size, 100])
    # embeddings_blank = tf.Variable(dtype=tf.float32, initial_value=np.zeros(shape=(reader.vocab_size, cfg["embeddings_size"])))
    # embeddings = load_embeddings.load_embedding(session=sess, vocab=reader.vocab_dict, emb=embeddings_blank, path=cfg[
    #                "path"]["embeddings"], dim_embedding=cfg["embeddings_size"])

    model.train_model(reader.one_hot_data)


if __name__ == "__main__":
    if ('--help' in sys.argv) or ('-h' in sys.argv):
        print("Language model with LSTM")
        print("Usage: %s max_sentences max_iterations [tag]" % sys.argv[0])
        print()
        print("max_sentences: maximum number of sentences to read (default: -1, reads all available sentences)")
        print()
        print("max_iterations: maximum number of training iterations (default: 100)")
        print()
        print("TAG: Describe the current setup (network params etc.)")
    else:
        # kwargs = {}
        if len(sys.argv) > 1:
            cfg["max_sentences"] = int(sys.argv[1])
        if len(sys.argv) > 2:
            cfg["max_iterations"] = int(sys.argv[2])
        if len(sys.argv) > 3:
            cfg["tag"] = str(sys.argv[3])
        main()
        # main(**kwargs)
