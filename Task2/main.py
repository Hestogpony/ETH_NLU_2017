import sys
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import getopt
import time

from config import cfg
from reader import Reader
from model import Model

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
    train_reader = Reader(buckets=cfg["buckets"], vocab_size=cfg["vocab_size"], max_turns=cfg["max_turns"])
    train_reader.build_dict(cfg["dictionary_name"], cfg["path"]["train"])
    train_reader.read_data(cfg["path"]["train"])


    m = Model(cfg)

    # Training
    m.build_forward_prop()
    m.build_backprop()

    m.train((train_reader.dataset_enc, train_reader.dataset_dec), train_reader.buckets_with_ids)

    # # Read evaluation data
    # eval_reader = Reader(vocab_size=cfg["vocab_size"], vocab_dict=train_reader.vocab_dict, max_turns=cfg["max_test_turns"])
    # eval_reader.read_sentences(cfg["path"]["validation"])

    # m.train(train_data=train_reader.id_data, train_buckets_with_ids = train_reader.buckets_with_ids, test_data=eval_reader.id_data)

def usage_and_quit():
    print("Chat bot based on seq2seq model")
    print("Options:")
    print("")
    print("--max_turns: maximum number of triples to read (default: -1, reads all available triples)")
    print("--max_test_turns: maximum number of triples to read (default: 10000, reads all available triples)")
    print("--max_iterations: maximum number of training iterations (default: 100)")
    print("--dictionary_name: define alternative dictionary name. (default: dict.p)")
    print("--out_batch: every x batches, report the test loss (default: 100, if < 1, never report)")
    print("--size: The size of the model ('small', 'medium' or 'big'); use 'small' and 'medium' for testing purposes; 'big' corresponds to the normal model")
    print("--lstm: the dimension of the LSTM hidden state (default 512)")
    print("--save_model_path: Path name where the trained model should be stored. Careful, the model is saved as multiple files")
    print("--load_model_path: Path name from where to load a pretrained model")
    sys.exit()

if __name__ == "__main__":
    try:
        opts, args = getopt.gnu_getopt(sys.argv[1:], "fhp",
            ["max_turns=", "max_test_turns=", "max_iterations=",
            "dictionary_name=", "out_batch=", "size=","lstm=", "save_model_path=", "load_model_path=", "help"])
    except getopt.GetoptError as err:
        print(str(err))
        usage_and_quit()

    for o, a in opts:
        if o == "--max_turns":
            cfg["max_turns"] = int(a)
        elif o == "--max_test_turns":
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
        elif o in {"--help", "-h"}:
            usage_and_quit()
        elif o == "--size":
            if str(a) == "small":
                cfg["max_turns"] = 1000
                cfg["max_test_sentences"] = 20

                cfg["vocab_size"] = 200
                cfg["batch_size"] = 10
                cfg["dictionary_name"] = "dict_small.p"
                cfg["out_batch"] = 10

                cfg["lstm_size"] = 256

            elif str(a) == "medium": # Similar to small, but we can use the pretrained embeddings
                cfg["max_turns"] = 1000
                cfg["max_test_sentences"] = 20

                cfg["vocab_size"] = 20000
                cfg["batch_size"] = 10
                cfg["dictionary_name"] = "dict_medium.p"
                cfg["out_batch"] = 10

                cfg["lstm_size"] = 256

            elif str(a) == "big":

                cfg["vocab_size"] = 20000
                cfg["batch_size"] = 64
                cfg["lstm_size"] = 512
                cfg["dictionary_name"] = "dict_big.p"
                cfg["out_batch"] = 1000

    #print(cfg)

    main()