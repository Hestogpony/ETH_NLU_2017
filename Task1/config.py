import pickle

cfg = {
    "path": {
        "embeddings": "./data/wordembeddings-dim100.word2vec",
        "train": "./data/sentences.train",
        "eval": "./data/sentences.eval",
        "test": "./data/sentences_test",
        "continuation": "./data/sentences.continuation",
        "output": "./output/group02.perplexity",
    },
    "vocab_size": 20000,
    "sentence_length": 30,

    "embeddings_size" : 100,
    "lstm_size" : 512,
    "intermediate_projection_size": 512,

    "max_sentences" : -1,
    "max_test_sentences" : -1,
    "max_iterations" : 100,
    "batch_size": 64,
    "out_batch": 100,

    "use_fred": False,
    "use_pretrained": False,
    "extra_project": False,

    "generate_length": 20,

    "dictionary_name": "dict.p"
}

def save_cfg(model_name):
    pickle.dump(cfg, open(model_name + ".config", "wb"))
    print("Configs saved in file: %s" % (model_name + ".config"))

def load_cfg(model_name):
    loaded_cfg = pickle.load(open(model_name + ".config", "rb"))
    print("Configs loaded from %s" % (model_name + ".config"))
    return loaded_cfg