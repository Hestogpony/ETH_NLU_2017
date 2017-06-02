import pickle
import os
from collections import Counter
import sys
from config import cfg


def translate_tokens_to_ids(tokens, data_path):

    # load the dictionary directly if it's there
    if os.path.isfile("dict.p"):
        vocab_dict = pickle.load(open("dict.p", "rb"))
        return print([vocab_dict.get(t) for t in tokens])
    else:
        print("Building dictionary...")
        cnt = Counter()
        with open(data_path, 'r') as f:
            for line in f:
                for word in line.split():
                    # Only count non-special characters
                    if word not in {"<bos>", "<eos>", "<unk>", "<pad>"}:
                        cnt[word] += 1

            # most_common returns tuples (word, count)
            # 4 spots are reserved for the special tags
            vocab_with_counts = cnt.most_common(20000 - 4)
            vocab = [i[0] for i in vocab_with_counts]
            ids = list(range(20000 - 4))
            vocab_dict = dict(list(zip(vocab, ids)))

            vocab_dict["<bos>"] = 19996
            vocab_dict["<eos>"] = 19997
            vocab_dict["<unk>"] = 19998
            vocab_dict["<pad>"] = 19999

            pickle.dump(vocab_dict, open("dict.p", "wb"))
            return print([vocab_dict.get(t) for t in tokens])


def translate_ids_to_tokens(ids_to_translate, data_path):
    # load the dictionary directly if it's there
    if os.path.isfile("dict_rev.p"):
        vocab_dict = pickle.load(open("dict_rev.p", "rb"))
        return print([vocab_dict.get(int(id)) for id in ids_to_translate])
    else:
        print("Building reversed dictionary...")
        cnt = Counter()
        with open(data_path, 'r') as f:
            for line in f:
                for word in line.split():
                    # Only count non-special characters
                    if word not in {"<bos>", "<eos>", "<unk>", "<pad>"}:
                        cnt[word] += 1

            # most_common returns tuples (word, count)
            # 4 spots are reserved for the special tags
            vocab_with_counts = cnt.most_common(20000 - 4)
            vocab = [i[0] for i in vocab_with_counts]
            ids = list(range(20000 - 4))
            vocab_dict = dict(list(zip(ids, vocab)))

            vocab_dict[19996] = "<bos>"
            vocab_dict[19997] = "<eos>"
            vocab_dict[19998] = "<unk>"
            vocab_dict[19999] = "<pad>"

            pickle.dump(vocab_dict, open("dict_rev.p", "wb"))
            return print([vocab_dict.get(id) for id in ids_to_translate])


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("give a list of tokens or ids as input. e.g. [123, 124, 12, 19994,] or ['just', 'do', 'it']")
        exit(1)

    ids_or_tokens = []
    i = 1
    while i < len(sys.argv):
        symbol = sys.argv[i]
        ids_or_tokens.append(symbol.strip('[').strip(',').strip(']'))
        i += 1

    if ids_or_tokens[0].isdigit():
        translate_ids_to_tokens(ids_or_tokens, cfg["path"]["train"])
    else:
        translate_tokens_to_ids(ids_or_tokens, cfg["path"]["train"])