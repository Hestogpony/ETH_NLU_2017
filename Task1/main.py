import load_embeddings
import model

import sys
import os
import getopt
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np
from collections import Counter
import pickle
import time

from config import cfg
from reader import Reader

class Logger(object):
    def __init__(self, timestamp):
        self.terminal = sys.stdout
        self.log = open("log/" + timestamp + ".log", "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass


def main():
    # Write to both logfile and stdout
    timestamp = time.strftime('%Y-%m-%d--%H_%M_%S')
    sys.stdout = Logger(timestamp)

    # Read train data
    train_reader = Reader(vocab_size=cfg["vocab_size"], sentence_length =cfg["sentence_length"],
                    max_sentences=cfg["max_sentences"])
    train_reader.build_dict(cfg["path"]["train"])
    train_reader.read_sentences(cfg["path"]["train"])

    if cfg["use_pretrained"]:
    # Read given embeddings
        sess = tf.Session()
        embeddings = tf.placeholder(dtype=tf.float32, shape=[cfg["vocab_size"], cfg["embeddings_size"]])
        embeddings_blank = tf.Variable(dtype=tf.float32, initial_value=np.zeros(shape=(cfg["vocab_size"], cfg["embeddings_size"])))
        embeddings = load_embeddings.load_embedding(session=sess, vocab=train_reader.vocab_dict, emb=embeddings_blank, path=cfg[
                       "path"]["embeddings"], dim_embedding=cfg["embeddings_size"])
        m = model.Model(cfg=cfg, embeddings=embeddings)
    else:
        m = model.Model(cfg=cfg)

    # Training
    m.build_forward_prop()
    m.build_backprop()

    # Read evaluation data
    eval_reader = Reader(vocab_size=cfg["vocab_size"], sentence_length =cfg["sentence_length"], vocab_dict=train_reader.vocab_dict, max_sentences=cfg["max_test_sentences"])
    eval_reader.read_sentences(cfg["path"]["eval"])


    m.train(train_data=train_reader.id_data, test_data=eval_reader.id_data)


    # Read test data
    test_reader = Reader(vocab_size=cfg["vocab_size"], sentence_length =cfg["sentence_length"], vocab_dict=train_reader.vocab_dict, max_sentences=cfg["max_test_sentences"])
    test_reader.read_sentences(cfg["path"]["test"])
    
    #Revert dictionary for perplexity
    reverted_dict = dict([(y,x) for x,y in list(test_reader.vocab_dict.items())])

    m.test(data=test_reader.id_data, vocab_dict=reverted_dict)


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
    print("--size: The size of the model ('small', 'medium' or 'big'); use 'small' and 'medium' for testing purposes; 'big' corresponds to the normal model")
    print("--pretrained / -p : used pretrained word2vec embeddings")
    print("--extra_project: use an additional size-512 projection layer before the output")
    print("--lstm: the dimension of the LSTM hidden state (default 512)")
    print("--save_model_path: Path name where the trained model should be stored. Careful, the model is saved as multiple files")
    print("--load_model_path: Path name from where to load a pretrained model")
    print("Notes:")
    print("\tTo run experiment A, specify no parameters")
    print("\tTo run experiment B, specify `--pretrained`")
    print("\tTo run experiment C, specify `--pretrained --lstm=1024 --extra_project`")
    sys.exit()

if __name__ == "__main__":

    try:
        opts, args = getopt.gnu_getopt(sys.argv[1:], "fhp",
            ["max_sentences=", "max_test_sentences=", "max_iterations=",
            "dictionary_name=", "out_batch=", "size=","lstm=", "save_model_path=", "load_model_path=", "help", "fred", "pretrained", "extra_project"])
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
        elif o == "--save_model_path":
            cfg["save_model_path"] = str(a)
        elif o == "--load_model_path":
            cfg["load_model_path"] = str(a)
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

                cfg["vocab_size"] = 200
                cfg["batch_size"] = 10
                cfg["dictionary_name"] = "dict_small.p"
                cfg["out_batch"] = 10

                cfg["embedding_size"] = 100
                cfg["lstm_size"] = 256
                cfg["intermediate_projection_size"] = 128

            elif str(a) == "medium": # Similar to small, but we can use the pretrained embeddings
                cfg["max_sentences"] = 1000
                cfg["max_test_sentences"] = 20

                cfg["vocab_size"] = 20000
                cfg["batch_size"] = 10
                cfg["dictionary_name"] = "dict_medium.p"
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
