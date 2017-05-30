import pickle

cfg = {
    "path": {
        "train": "./data/Training_Shuffled_Dataset.txt",
        # "train_labels": "./data/Training_Shuffled_Dataset_Labels.txt",
        "validation": "./data/Validation_Shuffled_Dataset.txt",
        # "validation_labels": "./data/Validation_Shuffled_Dataset_Labels.txt",
        # "output": "./output/group02.perplexity",
    },
    "buckets": [50], # list of tuples for the seq2seq model
    "PAD_i": 20000 - 1,
    "vocab_size": 20000,
    "embeddings_size": 100,

    "lstm_size" : 512,

    "max_turns" : -1,
    "max_test_turns" : -1,
    "max_iterations" : 10,
    "batch_size": 3,
    "out_batch": 100,
    # "generate_length": 20,

    "dictionary_name": "dict.p"
}

def save_cfg(model_name):
    pickle.dump(cfg, open(model_name + ".config", "wb"))
    print("Configs saved in file: %s" % (model_name + ".config"))

def load_cfg(model_name):
    loaded_cfg = pickle.load(open(model_name + ".config", "rb"))
    print("Configs loaded from %s" % (model_name + ".config"))
    return loaded_cfg