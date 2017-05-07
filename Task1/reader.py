import numpy as np
import pickle
import os

from config import cfg

class Reader(object):

    def __init__(self, vocab_size, max_sentences=-1, vocab_dict={}):
        self.vocab_size = vocab_size
        self.max_sentences = max_sentences
        self.vocab_dict = vocab_dict

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
        # Read sentences, pad them, convert to IDs according to the dict
        print("reading sentences from %s..." % path)
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

    def pad_id_data_to_batch_size(self):
        """
        Extend the id_data array so the batch size divides its length
        """
        print("padding the id data with 'pad' to make it divisible by the batch size...")
        sentences = len(self.id_data)
        if cfg["batch_size"] is 1 or sentences % cfg["batch_size"] is 0:
            return 0

        padding = cfg["batch_size"] - (sentences % cfg["batch_size"])

        extension = np.full(shape=(padding, cfg["sentence_length"]), fill_value=self.vocab_dict["pad"], dtype=np.float32)
        self.id_data = np.concatenate((self.id_data, extension), axis=0)

        return padding

class PartialsReader(Reader):
    def __init__(self, max_sentences=-1):
        self.max_sentences = max_sentences
        # self.vocab_dict = vocab_dict

    def load_dict(self, path):
        self.vocab_dict = pickle.load(open(cfg["dictionary_name"], "rb"))

    def read_sentences(self, path):
        # Read sentences, pad them, convert to IDs according to the dict
        print("reading sentences from %s..." % path)
        with open(path, 'r') as f:
            sentence_list = []
            if self.max_sentences == -1:
                for line in f:
                    tokens = line.split()
                    tokens.insert(0, "bos")
                    sentence = self.convert_sentence(tokens)
                    sentence_list.append(sentence)
            else:
                for i in range(self.max_sentences):
                    # last token is the newline character
                    tokens = f.readline().split()[:-1]
                    tokens.insert(0, "bos")
                    sentence = self.convert_sentence(tokens)
                    sentence_list.append(sentence)

            self.id_sequences = sentence_list

    def convert_sentence(self, tokens):
        sentence = np.zeros(shape=len(tokens), dtype=np.int32)
        for idx, word in enumerate(tokens):
            # translate according to dict
            if word in self.vocab_dict:
                sentence[idx] = self.vocab_dict[word]
            else:
                sentence[idx] = self.vocab_dict["unk"]
        return sentence