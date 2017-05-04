cfg = {
    "path": {
        "embeddings": "./data/wordembeddings-dim100.word2vec",
        "train": "./data/sentences.train",
        "test": "./data/sentences.eval",
        "continuation": "./data/sentences.continuation",
    },
    "vocab_size": 20000, #20000
    "sentence_length": 30,
    "batch_size": 64, #todo 64
    "embeddings_size" : 100, #100
    "lstm_size" : 512, #todo 512
    "max_sentences" : -1,
    "max_test_sentences" : -1,
    "max_iterations" : 100,
    "out_batch": 1 #todo 100
}
