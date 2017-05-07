import numpy as np
import pickle
import os

class Reader(object):

    def __init__(self, vocab_size, sentence_length, max_sentences=-1, vocab_dict={}):
        self.vocab_size = vocab_size
        self.sentence_length = sentence_length
        self.max_sentences = max_sentences
        self.vocab_dict = vocab_dict

    def build_dict(self, path):
        """ Ordered by word count, only the 20k most frequent words are being used """
        print("building dictionary...")

        # load the dictionary if it's there
        if os.path.isfile(path):
            self.vocab_dict = pickle.load(open(path, "rb"))
            return

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

            self.vocab_dict["bos"] = self.vocab_size - 4
            self.vocab_dict["eos"] = self.vocab_size - 3
            self.vocab_dict["unk"] = self.vocab_size - 2
            self.vocab_dict["pad"] = self.vocab_size - 1

            pickle.dump(self.vocab_dict, open(path, "wb"))

    def read_sentences(self, path):
        # Read sentences, pad them, convert to IDs according to the dict
        print("reading sentences from %s..." % path)
        with open(path, 'r') as f:
            sentence_list = []
            if self.max_sentences == -1:
                for line in f:
                    tokens = line.split()
                    if(len(tokens) <= self.sentence_length - 2):
                        tokens = self.add_tags(tokens)
                        sentence = self.convert_sentence(tokens)
                        sentence_list.append(sentence)
            else:
                for i in range(self.max_sentences):
                    # last token is the newline character
                    tokens = f.readline().split()[:-1]
                    if(len(tokens) <= self.sentence_length - 2):
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
        tokens.extend((self.sentence_length - len(tokens)) * ["pad"])
        return tokens

    def convert_sentence(self, tokens):
        sentence = np.zeros(shape=self.sentence_length, dtype=np.int32)
        for idx, word in enumerate(tokens):
            # translate according to dict
            if word in self.vocab_dict:
                sentence[idx] = self.vocab_dict[word]
            else:
                sentence[idx] = self.vocab_dict["unk"]
        return sentence

class PartialsReader(Reader):
    def __init__(self, max_sentences=-1):
        self.max_sentences = max_sentences
        # self.vocab_dict = vocab_dict

    def load_dict(self, path):
        self.vocab_dict = pickle.load(open(path, "rb"))

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