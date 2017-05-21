import numpy as np
import pickle
import os
from config import cfg

from collections import Counter

BOS = "<bos>"
EOS = "<eos>"
UNK = "<unk>"
PAD = "<pad>"
MAX_SENTENCE_LEN = 100


class Reader:
    def __init__(self, buckets, vocab_size, max_turns=-1, vocab_dict={}, use_triples=False):
        self.buckets = buckets
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
                input_data_lines = f.readlines()
            else:
                input_data_lines = []
                for i in range(self.max_turns):
                    input_data_lines.append(f.readline())

        dataset = [[] for _ in self.buckets]
        dataset_triple = [[] for _ in self.buckets]

        first_sentences = []
        second_sentences = []
        third_sentences = []

        for data_line in input_data_lines:
            input_sentences = data_line.split('\t')
            assert (len(input_sentences) == 3)

            first_sentence_as_ids = self.ids_from_toks(input_sentences[0].split())
            second_sentence_as_ids = self.ids_from_toks(input_sentences[1].split())
            third_sentence_as_ids = self.ids_from_toks(input_sentences[2].split())

            if self.use_triples:
                first_sentences.append(first_sentence_as_ids + [self.EOS_i])
                second_sentences.append(second_sentence_as_ids + [self.EOS_i])
                third_sentences.append([self.BOS_i] + third_sentence_as_ids + [self.EOS_i])
            else:
                first_sentences.append(first_sentence_as_ids + [self.EOS_i])
                first_sentences.append([self.BOS_i] + second_sentence_as_ids + [self.EOS_i])

                second_sentences.append(second_sentence_as_ids + [self.EOS_i])
                second_sentences.append([self.BOS_i] + third_sentence_as_ids + [self.EOS_i])

        if self.use_triples:
            for i in range(0, len(first_sentences)):
                if len(first_sentences[i]) > 100 or len(second_sentences[i]) > 100 or len(third_sentences[i]) > 100:
                    continue
                else:
                    for bucket_index, bucket_size in enumerate(self.buckets):
                        if len(first_sentences[i]) < bucket_size and len(third_sentences[i]) < bucket_size and len(
                                second_sentences[i]) < bucket_size:
                            first_sentences[i].extend((bucket_size - len(first_sentences[i])) * [self.PAD_i])
                            second_sentences[i].extend((bucket_size - len(second_sentences[i])) * [self.PAD_i])
                            third_sentences[i].extend((bucket_size - len(third_sentences[i])) * [self.PAD_i])
                            dataset[bucket_index].append((first_sentences[i], second_sentences[i], third_sentences[i]))
                            break
        else:
            for i in range(0, len(first_sentences)):
                if len(first_sentences[i]) > MAX_SENTENCE_LEN or len(second_sentences[i]) > MAX_SENTENCE_LEN:
                    continue
                else:
                    for bucket_index, bucket_size in enumerate(self.buckets):
                        if len(first_sentences[i]) < bucket_size and len(second_sentences[i]) < bucket_size:
                            first_sentences[i].extend((bucket_size - len(first_sentences[i])) * [self.PAD_i])
                            # print(str(ta[index]))
                            second_sentences[i].extend((bucket_size - len(second_sentences[i])) * [self.PAD_i])
                            # print(str(tc[index]))
                            dataset[bucket_index].append((first_sentences[i], second_sentences[i]))
                            break

        return dataset

        """
        if self.use_triples:
            return (ta, tb, tc)
        else:
            return (ta, tc)
        """


"""
if __name__ == "__main__": 
    train_reader = Reader(buckets = cfg["buckets"], vocab_size=cfg["vocab_size"])
    train_reader.build_dict("./data/dict.p", "./data/Training_Shuffled_Dataset.txt")
    [dataset, dataset_triple] = train_reader.read_data("./data/Training_Shuffled_Dataset.txt")

    
    counter = 0


    #f = open("whatever.txt", "w")
    
    for bucketsize, bucket in enumerate(dataset):
        counter = 0
        for numb, pair in enumerate(bucket):
            counter += 1
            [a,b] = pair
            #f.write(str(a)+"+++$$$$$")
            #f.write(str(b)+"\n")
        print("The bucket "+str(10*(bucketsize+1))+"th"+" size is "+ str(counter))


    for bucketsize, bucket in enumerate(dataset_triple):
        countermore = 0
        for numb, triple in enumerate(bucket):
            countemore  += 1
        print("The bucket "+str(10*(bucketsize+1))+"th"+" size is " + str(counter))  
    '''
    for m in dataset[9]:
        counter += 1
        
    print(str(counter)+" This is in the bucket 10")
    '''
"""
