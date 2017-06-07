import random
import re
import os
import pickle
import numpy as np

MODE_R = 'r'
MODE_W = 'w'
MODE_A = 'a'
ENCODING = 'latin-1'


class Reader(object):
    def __init__(self, config):
        self.cfg = config
        self.vocab_size_dict = {}

    def prepare_raw_data(self):
        print('Preparing raw data into train set and test set ...')
        # <FL> Use our own stuff here

        questions, answers = self.make_cornell_data_pairs(self.get_cornell_lines(), self.get_cornell_convos())
        our_data_questions, our_data_answers = self.make_our_data_pairs(os.path.join(self.cfg['OUR_DATA']['PATH'],
                                                                                     self.cfg['OUR_DATA']['LINE_FILE']))

        questions.extend(our_data_questions)
        answers.extend(our_data_answers)
        self.prepare_dataset(questions, answers)

    @staticmethod
    def make_cornell_data_pairs(id2line, convos):
        """ Divide the dataset into two sets: questions and answers. """
        questions, answers = [], []
        for convo in convos:
            for index, line in enumerate(convo[:-1]):
                questions.append(id2line[convo[index]])
                answers.append(id2line[convo[index + 1]])
        assert len(questions) == len(answers)
        return questions, answers

    def get_cornell_lines(self):
        id2line = {}
        file_path = os.path.join(self.cfg['CORNELL_DATA']['PATH'], self.cfg['CORNELL_DATA']['LINE_FILE'])
        with open(file_path, MODE_R, encoding=ENCODING) as f:
            lines = f.readlines()
            for line in lines:
                parts = line.split(' +++$+++ ')
                if len(parts) == 5:
                    if parts[4][-1] == '\n':
                        parts[4] = parts[4][:-1]
                    id2line[parts[0]] = parts[4]
        return id2line

    def get_cornell_convos(self):
        """ Get conversations from the raw data """
        file_path = os.path.join(self.cfg['CORNELL_DATA']['PATH'], self.cfg['CORNELL_DATA']['CONVO_FILE'])
        convos = []
        with open(file_path, MODE_R, encoding=ENCODING) as f:
            if not self.cfg['MAX_TURNS'] or self.cfg['MAX_TURNS'] <= 0:
                lines = f.readlines()
            else:
                lines = []
                for i in range(self.cfg['MAX_TURNS']):
                    lines.append(f.readline())

            for line in lines:
                parts = line.split(' +++$+++ ')
                if len(parts) == 4:
                    convo = []
                    for line in parts[3][1:-2].split(', '):
                        convo.append(line[1:-1])
                    convos.append(convo)
        return convos

    def make_our_data_pairs(self, file_path):
        questions = []
        answers = []
        with open(file_path) as f:
            if not self.cfg['MAX_TURNS'] or self.cfg['MAX_TURNS'] <= 0:
                input_data_lines = f.readlines()
            else:
                input_data_lines = []
                for i in range(self.cfg['MAX_TURNS']):
                    input_data_lines.append(f.readline())

        input_data_lines = [x.replace('\n', '') for x in input_data_lines]
        input_data_lines = [x for x in input_data_lines if len(x) > 0]

        for data_line in input_data_lines:
            input_sentences = data_line.split('\t')

            questions.append(input_sentences[0])
            answers.append(input_sentences[1])

            questions.append(input_sentences[1])
            answers.append(input_sentences[2])

        return questions, answers

    def prepare_dataset(self, questions, answers):
        # create path to store all the train & test encoder & decoder
        self.make_dir(self.cfg['PROCESSED_PATH'])

        # Save the number of question-answer pairs to config for epoch counting
        self.cfg['TRAINING_SAMPLES'] = len(questions)

        # random convos to create the test set
        test_ids = random.sample([i for i in range(len(questions))], self.cfg['TESTSET_SIZE'])

        file_names = ['train.enc', 'train.dec', 'test.enc', 'test.dec']
        files = []
        for filename in file_names:
            files.append(open(os.path.join(self.cfg['PROCESSED_PATH'], filename), MODE_W, encoding=ENCODING))

        for i in range(len(questions)):
            if i in test_ids:
                files[2].write(questions[i] + '\n')
                files[3].write(answers[i] + '\n')
            else:
                files[0].write(questions[i] + '\n')
                files[1].write(answers[i] + '\n')

        for file in files:
            file.close()

    def build_vocab(self, filename):
        in_path = os.path.join(self.cfg['PROCESSED_PATH'], filename)
        out_path = os.path.join(self.cfg['PROCESSED_PATH'], 'vocab.{}'.format(filename[-3:]))

        vocab = {}
        with open(in_path, MODE_R, encoding=ENCODING) as f:
            for line in f.readlines():
                for token in self.basic_tokenizer(line):
                    if token not in vocab:
                        vocab[token] = 0
                    vocab[token] += 1

        sorted_vocab = sorted(vocab, key=vocab.get, reverse=True)
        with open(out_path, MODE_W, encoding=ENCODING) as f:
            f.write('<pad>' + '\n')
            f.write('<unk>' + '\n')
            f.write('<s>' + '\n')
            f.write('<\s>' + '\n')
            index = 4
            for word in sorted_vocab:
                if vocab[word] < self.cfg['THRESHOLD']:
                    break
                f.write(word + "\n")
                index += 1
            if filename[-3:] == 'enc':
                self.cfg['ENC_VOCAB'] = index
                print('Enc vocab ' + str(self.cfg['ENC_VOCAB']))
            else:
                self.cfg['DEC_VOCAB'] = index
                print('Dec vocab ' + str(self.cfg['DEC_VOCAB']))

    @staticmethod
    def load_vocab(vocab_path):
        with open(vocab_path, MODE_R, encoding=ENCODING) as f:
            words = f.read().splitlines()
        return words, {words[i]: i for i in range(len(words))}

    def sentence2id(self, vocab, line):
        return [vocab.get(token, vocab['<unk>']) for token in self.basic_tokenizer(line)]

    def process_data(self):
        print('Preparing data to be model-ready ...')
        self.build_vocab('train.enc')
        self.build_vocab('train.dec')
        self.token2id('train', 'enc')
        self.token2id('train', 'dec')
        self.token2id('test', 'enc')
        self.token2id('test', 'dec')

    def load_data(self, enc_filename, dec_filename):
        encode_file = open(os.path.join(self.cfg['PROCESSED_PATH'], enc_filename), MODE_R, encoding=ENCODING)
        decode_file = open(os.path.join(self.cfg['PROCESSED_PATH'], dec_filename), MODE_R, encoding=ENCODING)
        encode, decode = encode_file.readline(), decode_file.readline()
        data_buckets = [[] for _ in self.cfg['BUCKETS']]
        i = 0
        while encode and decode:
            if (i + 1) % 10000 == 0:
                print("Bucketing conversation number", i)
            encode_ids = [int(id_) for id_ in encode.split()]
            decode_ids = [int(id_) for id_ in decode.split()]
            for bucket_id, (encode_max_size, decode_max_size) in enumerate(self.cfg['BUCKETS']):
                if len(encode_ids) <= encode_max_size and len(decode_ids) <= decode_max_size:
                    data_buckets[bucket_id].append([encode_ids, decode_ids])
                    break
            encode, decode = encode_file.readline(), decode_file.readline()
            i += 1

        # <BG> Save the number of question-answer pairs to config for epoch counting
        training_samples = np.sum([len(data_buckets[b]) for b in range(len(self.cfg['BUCKETS']))])
        print(training_samples)
        self.cfg['TRAINING_SAMPLES'] = training_samples
        self.vocab_size_dict['TRAINING_SAMPLES'] = training_samples
        # <BG> Copy stuff over to the extra dict
        self.vocab_size_dict['ENC_VOCAB'] = self.cfg['ENC_VOCAB']
        self.vocab_size_dict['DEC_VOCAB'] = self.cfg['DEC_VOCAB']

        # <BG> Dump the determined vocab sizes to the processed path folder
        path = os.path.join(self.cfg['PROCESSED_PATH'], "vocab_sizes")
        pickle.dump(self.vocab_size_dict, open(path, "wb"))

        return data_buckets

    def token2id(self, data, mode):
        """ Convert all the tokens in the data into their corresponding
        index in the vocabulary. """
        vocab_path = 'vocab.' + mode
        in_path = data + '.' + mode
        out_path = data + '_ids.' + mode

        _, vocab = self.load_vocab(os.path.join(self.cfg['PROCESSED_PATH'], vocab_path))
        in_file = open(os.path.join(self.cfg['PROCESSED_PATH'], in_path), MODE_R, encoding=ENCODING)
        out_file = open(os.path.join(self.cfg['PROCESSED_PATH'], out_path), MODE_W, encoding=ENCODING)

        lines = in_file.read().splitlines()
        for line in lines:
            if mode == 'dec':  # we only care about '<s>' and </s> in encoder
                ids = [vocab['<s>']]
            else:
                ids = []
            ids.extend(self.sentence2id(vocab, line))
            # ids.extend([vocab.get(token, vocab['<unk>']) for token in basic_tokenizer(line)])
            if mode == 'dec':
                ids.append(vocab['<\s>'])
            out_file.write(' '.join(str(id_) for id_ in ids) + '\n')

    @staticmethod
    def basic_tokenizer(line, normalize_digits=True):
        """ A basic tokenizer to tokenize text into tokens.
        Feel free to change this to suit your need. """
        line = re.sub('<u>', '', line)
        line = re.sub('</u>', '', line)
        line = re.sub('\[', '', line)
        line = re.sub('\]', '', line)
        words = []
        _WORD_SPLIT = re.compile("([.,!?\"'-:;)(])")
        _DIGIT_RE = re.compile(r"\d")
        for fragment in line.strip().lower().split():
            for token in re.split(_WORD_SPLIT, fragment):
                if not token:
                    continue
                if normalize_digits:
                    token = re.sub(_DIGIT_RE, '#', token)
                words.append(token)
        return words

    ### BATCHING METHODS ###
    def get_batch(self, data_bucket, bucket_id, batch_size=1):
        """ Return one batch to feed into the model """
        # only pad to the max length of the bucket
        encoder_size, decoder_size = self.cfg['BUCKETS'][bucket_id]
        encoder_inputs, decoder_inputs = [], []

        for _ in range(batch_size):
            encoder_input, decoder_input = random.choice(data_bucket)
            # pad both encoder and decoder, reverse the encoder
            encoder_inputs.append(list(reversed(self._pad_input(encoder_input, encoder_size))))
            decoder_inputs.append(self._pad_input(decoder_input, decoder_size))

        # now we create batch-major vectors from the data selected above.
        batch_encoder_inputs = self._reshape_batch(encoder_inputs, encoder_size, batch_size)
        batch_decoder_inputs = self._reshape_batch(decoder_inputs, decoder_size, batch_size)

        # create decoder_masks to be 0 for decoders that are padding.
        batch_masks = []
        for length_id in range(decoder_size):
            batch_mask = np.ones(batch_size, dtype=np.float32)
            for batch_id in range(batch_size):
                # we set mask to 0 if the corresponding target is a PAD symbol.
                # the corresponding decoder is decoder_input shifted by 1 forward.
                if length_id < decoder_size - 1:
                    target = decoder_inputs[batch_id][length_id + 1]
                if length_id == decoder_size - 1 or target == self.cfg['PAD_ID']:
                    batch_mask[batch_id] = 0.0
            batch_masks.append(batch_mask)
        return batch_encoder_inputs, batch_decoder_inputs, batch_masks

    def _pad_input(self, input_, size):
        return input_ + [self.cfg['PAD_ID']] * (size - len(input_))

    @staticmethod
    def _reshape_batch(inputs, size, batch_size):
        """ Create batch-major inputs. Batch inputs are just re-indexed inputs
        """
        batch_inputs = []
        for length_id in range(size):
            batch_inputs.append(np.array([inputs[batch_id][length_id]
                                          for batch_id in range(batch_size)], dtype=np.int32))
        return batch_inputs

    @staticmethod
    def make_dir(path):
        """ Create a directory if there isn't one already. """
        try:
            os.mkdir(path)
        except OSError:
            pass

