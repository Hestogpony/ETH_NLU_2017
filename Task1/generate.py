import numpy as np
import getopt
import sys

import config
from config import cfg
from reader import PartialsReader
import model

def generate(load_model_path, max_sentences):
    config.load_cfg(load_model_path)
    cfg["load_model_path"] = load_model_path

    p_reader = PartialsReader(max_sentences=cfg["max_sentences"])
    p_reader.load_dict(cfg["dictionary_name"])
    p_reader.read_sentences(cfg["path"]["continuation"])

    m = model.Model()
    m.build_forward_prop()

    #Revert dictionary for generation
    reverted_dict = dict([(y,x) for x,y in list(test_reader.vocab_dict.items())])

    m.generate(p_reader.id_sequences, reverted_dict)



def usage_and_quit():
    print("Generate sentences with LSTM language model")
    print("Options:")
    print("")
    print("--load_model_path: The name of the trained model that is used for the generation of sentences")
    print("--max_sentences: maximum number of sentences to read (default: -1, reads all available sentences)")

    sys.exit()


if __name__ == "__main__":
    try:
        opts, args = getopt.gnu_getopt(sys.argv[1:], "h",
            ["load_model_path=", "max_sentences:", "help"])
    except getopt.GetoptError as err:
        print(str(err))
        usage_and_quit()

    max_sentences = -1
    for o, a in opts:
        if o == "--load_model_path":
            load_model_path = a
        elif o == "--max_sentences":
            max_sentences = int(a)
        elif o in {"--help", "-h"}:
            usage_and_quit()

    generate(load_model_path, max_sentences)