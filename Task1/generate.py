import numpy as np
import getopt
import sys

import config
from reader import PartialsReader
import model

def generate(load_model_path, max_sentences):
    config.cfg = config.load_cfg(load_model_path)
    config.cfg["load_model_path"] = load_model_path

    p_reader = PartialsReader(max_sentences=max_sentences)
    p_reader.load_dict(config.cfg["dictionary_name"])
    p_reader.read_sentences(config.cfg["path"]["continuation"])

    import model
    m = model.Model(cfg=config.cfg)
    m.build_forward_prop()
    m.load_model(config.cfg["load_model_path"])

    #Revert dictionary for generation
    reverted_dict = dict([(y,x) for x,y in list(p_reader.vocab_dict.items())])

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
            ["load_model_path=", "max_sentences=", "help"])
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