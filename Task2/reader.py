import numpy as np
import pickle
import os

from collections import Counter

BOS = "<bos>"
EOS = "<eos>"
UNK = "<unk>"
PAD = "<pad>"


class Reader:
    def __init__(self, vocab_size, max_turns=-1, vocab_dict={}, use_triples=False):
        self.vocab_size = vocab_size
        self.max_turns = max_turns
        self.vocab_dict = vocab_dict
        self.use_triples = use_triples

        self.BOS_i = self.vocab_size - 4
        self.EOS_i = self.vocab_size - 3
        self.UNK_i = self.vocab_size - 2
        self.PAD_i = self.vocab_size - 1

    def build_dict(self, wanted_dict_path, data_path):
        """ Ordered by word count, only the 20k most frequent words are being used """
        print("building dictionary...")

        # load the dictionary directly if it's there
        if os.path.isfile(wanted_dict_path):
            self.vocab_dict = pickle.load(open(wanted_dict_path, "rb"))
            return

        cnt = Counter()
        with open(data_path, 'r') as f:
            for line in f:
                for word in line.split():
                    # Only count non-special characters
                    if word not in {BOS, EOS, UNK, PAD}:
                        cnt[word] += 1

            # most_common returns tuples (word, count)
            # 4 spots are reserved for the special tags
            vocab_with_counts = cnt.most_common(self.vocab_size - 4)
            vocab = [i[0] for i in vocab_with_counts]
            ids = list(range(self.vocab_size - 4))
            self.vocab_dict = dict(list(zip(vocab, ids)))

            self.vocab_dict[BOS] = self.BOS_i
            self.vocab_dict[EOS] = self.EOS_i
            self.vocab_dict[UNK] = self.UNK_i
            self.vocab_dict[PAD] = self.PAD_i

            pickle.dump(self.vocab_dict, open(wanted_dict_path, "wb"))

    def ids_from_toks(self, tokens):
        return [self.vocab_dict.get(t, self.UNK_i) for t in tokens]

    def read_data(self, path):
        '''
        Return value, assuming the path contains N turns:
            If use_triples = False:
                Two lists of length 2N.
            If use_triples = True:
                Three lists of length N.

            The elements of each list are lists themselves (of word IDs).
            No padding is performed.

            bos-tags are inserted appropriately - i.e. if use_triples is true,
            the third sentence of a triple begins with bos. If use_triples is
            false, the second sentence of every pair begins with bos
        '''
        print("reading data from %s..." % path)
        with open(path, 'r') as f:

            if self.max_turns < 0:
                turns = f.readlines()
            else:
                turns = []
                for i in range(self.max_turns):
                    turns.append(f.readline())

        ta = []
        tb = []
        tc = []

        for turn in turns:
            parts = turn.split('\t')
            assert(len(parts) == 3)

            a = self.ids_from_toks(parts[0].split())
            b = self.ids_from_toks(parts[1].split())
            c = self.ids_from_toks(parts[2].split())

            if self.use_triples:
                ta.append(a)
                tb.append(b)
                tc.append([self.BOS_i] + c)
            else:
                ta.append(a)
                ta.append([self.BOS_i] + b)

                tc.append(b)
                tc.append([self.BOS_i] + c)

        if self.use_triples:
            return (ta, tb, tc)
        else:
            return (ta, tc)
