cfg = {
    "path": {
        "embeddings": "./data/wordembeddings-dim100.word2vec",
        "train": "./data/sentences.train",
        "test": "./data/sentences.eval",
        "continuation": "./data/sentences.continuation",
    },
    "vocab_size": 200, #20000
    "sentence_length": 30,
    "batch_size": 2, #todo 64
    "embeddings_size" : 10, #100
    "lstm_size" : 8, #todo 512
    "max_sentences" : -1,
    "max_iterations" : 100,
}
