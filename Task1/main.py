import sys
import os
import getopt
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np
from collections import Counter
import pickle
import time

import load_embeddings
import model

from config import cfg

class Logger(object):
    def __init__(self, timestamp):
        self.terminal = sys.stdout
        self.log = open(timestamp + ".log", "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass


class Reader(object):

    def __init__(self, vocab_size, max_sentences=-1, vocab_dict={}):
        self.vocab_size = vocab_size
        self.max_sentences = max_sentences
        self.vocab_dict = vocab_dict
        # self.id_data

    def build_dict(self, path):
        """ Ordered by word count, only the 20k most frequent words are being used """
        print("building dictionary...")

        # load the dictionary if it's there
        if os.path.isfile(cfg["dictionary_name"]):
            self.vocab_dict = pickle.load(open(cfg["dictionary_name"], "rb"))
            return None

        cnt = Counter()
        with open(path, 'r') as f:
            for line in f:
                for word in line.split():
                    if word not in {"bos","eos","unk","pad"}:
                        cnt[word] += 1

            # most_common returns tuples (word, count)
            # 4 spots are reserved for the special tags
            vocab_with_counts = cnt.most_common(self.vocab_size - 4)
            vocab = [i[0] for i in vocab_with_counts]
            ids = list(range(self.vocab_size - 4))
            self.vocab_dict = dict(list(zip(vocab, ids)))

            self.vocab_dict["bos"] = cfg["vocab_size"] - 4
            self.vocab_dict["eos"] = cfg["vocab_size"] - 3
            self.vocab_dict["unk"] = cfg["vocab_size"] - 2
            self.vocab_dict["pad"] = cfg["vocab_size"] - 1

            pickle.dump(self.vocab_dict, open(cfg["dictionary_name"], "wb"))

    def read_sentences(self, path):
        """ Include the tags bos, eos, pad, unk """
        # Read sentences, convert to IDs according to the dict, pad them
        print("reading sentences...")
        with open(path, 'r') as f:
            sentence_list = []
            if self.max_sentences == -1:
                for line in f:
                    tokens = line.split()
                    if(len(tokens) <= cfg["sentence_length"] - 2):
                        tokens = self.add_tags(tokens)
                        sentence = self.convert_sentence(tokens)
                        sentence_list.append(sentence)
            else:
                for i in range(self.max_sentences):
                    # last token is the newline character
                    tokens = f.readline().split()[:-1]
                    if(len(tokens) <= cfg["sentence_length"] - 2):
                        tokens = self.add_tags(tokens)
                        sentence = self.convert_sentence(tokens)
                        sentence_list.append(sentence)

            self.id_data = np.array(sentence_list, dtype=np.int32)

    def add_tags(self, tokens):
        """
        tokens      list of words
        """
        tokens.insert(0, "bos")
        tokens.append("eos")
        tokens.extend((cfg["sentence_length"] - len(tokens)) * ["pad"])
        return tokens

    def convert_sentence(self, tokens):
        sentence = np.zeros(shape=cfg["sentence_length"], dtype=np.int32)
        for idx, word in enumerate(tokens):
            # translate according to dict
            if word in self.vocab_dict:
                sentence[idx] = self.vocab_dict[word]
            else:
                sentence[idx] = self.vocab_dict["unk"]
        return sentence

    # use the tensorflow library
    def one_hot_encode(self):
        """
        input_matrix           ndarray dim: #sentences x 30
        return                 matrix dim: #sentences x 30 x vocab_size
        """
        print("doing the one hot encoding...")
        inp = tf.placeholder(tf.int32, [None, cfg["sentence_length"]])
        output = tf.one_hot(indices=inp, depth=cfg[
                            "vocab_size"], axis=-1, dtype=tf.float32)

        sess = tf.Session()
        self.one_hot_data = sess.run(output, feed_dict={inp: self.id_data})
        # dtype should be float32
        # print(self.one_hot_data.shape)

    def pad_id_data_to_batch_size(self):
        """
        Extend the id_data array so the batch size divides its length
        """
        print("padding the id data with 'pad' to make it divisible by the batch size...")
        sentences = len(self.id_data)
        if cfg["batch_size"] is 1 or sentences % cfg["batch_size"] is 0:
            return

        padding = cfg["batch_size"] - (sentences % cfg["batch_size"])

        extension = np.full(shape=(padding, cfg["sentence_length"]), fill_value=self.vocab_dict["pad"], dtype=np.float32)
        self.id_data = np.concatenate((self.id_data, extension), axis=0)

        return padding

def main():
    # Write to both logfile and stdout
    timestamp = time.strftime('%Y-%m-%d--%H_%M_%S')
    sys.stdout = Logger(timestamp)

    # Read train data
    train_reader = Reader(vocab_size=cfg["vocab_size"],
                    max_sentences=cfg["max_sentences"])
    train_reader.build_dict(cfg["path"]["train"])
    train_reader.read_sentences(cfg["path"]["train"])
    # train_reader.one_hot_encode()

    if cfg["use_pretrained"]:
    # Read given embeddings
        sess = tf.Session()
        embeddings = tf.placeholder(dtype=tf.float32, shape=[cfg["vocab_size"], cfg["embeddings_size"]])
        embeddings_blank = tf.Variable(dtype=tf.float32, initial_value=np.zeros(shape=(cfg["vocab_size"], cfg["embeddings_size"])))
        embeddings = load_embeddings.load_embedding(session=sess, vocab=train_reader.vocab_dict, emb=embeddings_blank, path=cfg[
                       "path"]["embeddings"], dim_embedding=cfg["embeddings_size"])
        m = model.Model(embeddings=embeddings)
    else:
        m = model.Model()

    # Training
    m.build_forward_prop()
    m.build_backprop()

    # Read test data
    test_reader = Reader(vocab_size=cfg["vocab_size"], vocab_dict =  train_reader.vocab_dict, max_sentences=cfg["max_test_sentences"])
    test_reader.read_sentences(cfg["path"]["test"])
    padding_size = test_reader.pad_id_data_to_batch_size()

    #Testing
    m.build_test()
    #Revert dictionary for perplexity
    reverted_dict = dict([(y,x) for x,y in list(test_reader.vocab_dict.items())])


    m.train(train_data=train_reader.id_data, test_data=test_reader.id_data)
    m.test(data=test_reader.id_data, vocab_dict=reverted_dict, cut_last_batch=padding_size)

def usage_and_quit():
    print("Language model with LSTM")
    print("Options:")
    print("")
    print("--max_sentences: maximum number of sentences to read (default: -1, reads all available sentences)")
    print("--max_test_sentences: maximum number of sentences to read (default: 10000, reads all available sentences)")
    print("--max_iterations: maximum number of training iterations (default: 100)")
    print("--dictionary_name: define alternative dictionary name. (default: dict.p)")
    print("--out_batch: every x batches, report the test loss (default: 100, if < 1, never report)")
    print("--fred / -f : use our implementation of the LSTM cell instead of Tensorflow's")
    print("--size: The size of the model ('small' or 'big'); use 'small' for testing purposes; 'big' corresponds to the normal model")
    print("--pretrained / -p : used pretrained word2vec embeddings")
    print("--extra_project: use an additional size-512 projection layer before the output")
    print("--lstm: the dimension of the LSTM hidden state (default 512)")
    print("Notes:")
    print("\tTo run experiment A, specify no parameters")
    print("\tTo run experiment B, specify `main.py --pretrained`")
    print("\tTo run experiment C, specify `main.py --pretrained --lstm=1024 --extra_project`")
    sys.exit()

if __name__ == "__main__":

    try:
        opts, args = getopt.gnu_getopt(sys.argv[1:], "fh",
            ["max_sentences=", "max_test_sentences=", "max_iterations=",
            "dictionary_name=", "out_batch=", "size=","lstm=", "help", "fred", "pretrained", "extra-project"])
    except getopt.GetoptError as err:
        print(str(err))
        usage_and_quit()

    for o, a in opts:
        if o == "--max_sentences":
            cfg["max_sentences"] = int(a)
        elif o == "--max_test_sentences":
            cfg["max_test_sentences"] = int(a)
        elif o == "--max_iterations":
            cfg["max_iterations"] = int(a)
        elif o == "--dictionary_name":
            cfg["dictionary_name"] = str(a)
        elif o == "--out_batch":
            cfg["out_batch"] = int(a)
        elif o == "--lstm":
            cfg["lstm_size"] = int(a)
        elif o in {"--fred", "-f"}:
            cfg["use_fred"] = True
        elif o in {"--pretrained", "-p"}:
            cfg["use_pretrained"] = True
        elif o == "--extra_project":
            cfg["extra_project"] = True
        elif o in {"--help", "-h"}:
            usage_and_quit()
        elif o == "--size":
            if str(a) == "small":
                cfg["max_sentences"] = 1000
                cfg["max_test_sentences"] = 20

                cfg["max_iterations"] = 10

                cfg["vocab_size"] = 200
                cfg["batch_size"] = 10
                cfg["dictionary_name"] = "dict_small.p"
                cfg["out_batch"] = 10

                cfg["embedding_size"] = 100
                cfg["lstm_size"] = 256
                cfg["intermediate_projection_size"] = 128

            elif str(a) == "big":

                cfg["vocab_size"] = 20000
                cfg["batch_size"] = 64
                cfg["embedding_size"] = 100
                cfg["lstm_size"] = 512
                cfg["dictionary_name"] = "dict_big.p"
                cfg["out_batch"] = 1000
    #print(cfg)

    main()
