""" A neural chatbot using sequence to sequence model with
attentional decoder.

This is based on Google Translate Tensorflow model
https://github.com/tensorflow/models/blob/master/tutorials/rnn/translate/

Sequence to sequence model by Cho et al.(2014)

Created by Chip Huyen as the starter code for assignment 3,
class CS 20SI: "TensorFlow for Deep Learning Research"
cs20si.stanford.edu

This file contains the code to run the model.

See readme.md for instruction on how to run the starter code.
"""
from __future__ import division
from __future__ import print_function


import gensim
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import argparse
import random
import sys
import time
import shutil
# import pprint

import numpy as np
import tensorflow as tf

from model import ChatBotModel

import config
from config import cfg
import measures

class Logger(object):
    def __init__(self, timestamp):
        self.terminal = sys.stdout
        if not os.path.exists("log"):
            os.makedirs("log")
        self.log = open("log/" + timestamp + ".log", "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        self.terminal.flush()
        self.log.flush()



class Chatbot(object):
    def __init__(self, config, reader):
        self.cfg = config
        self.reader = reader
        self.sess = tf.Session()



    ########TO LOAD THE PRETRAIN_WORDDING!!!TURN IT OFF IF IT BEHAVING BADLY#########
    def load_embbedings(self, sess):
        """ Initialize embeddings with pre-trained word2vec vectors
        Will modify the embedding weights of the current loaded model
        Uses the GoogleNews pre-trained values (path hardcoded)
        """

        # Fetch embedding variables from model
        with tf.variable_scope("embedding_attention_seq2seq/rnn/embedding_wrapper", reuse=True):
            em_in = tf.get_variable("embedding")
        with tf.variable_scope("embedding_attention_seq2seq/embedding_attention_decoder", reuse=True):
            em_out = tf.get_variable("embedding")

        # Disable training for embeddings
        variables = tf.get_collection_ref(tf.GraphKeys.TRAINABLE_VARIABLES)
        variables.remove(em_in)
        variables.remove(em_out)

        pretrained_folder="pretrained_stuff"
        modelname = "pretrain_model"

        model = gensim.models.KeyedVectors.load(pretrained_folder+'/'+modelname)

        embedding_matrix_trained = model.wv.syn0


        print("Loaded the pretrained word embbedings from model")

         # Initialize input and output embeddings
        sess.run(em_in.assign(embedding_matrix_trained))
        sess.run(em_out.assign(embedding_matrix_trained))




    def _get_random_bucket(self, train_buckets_scale):
        """ Get a random bucket from which to choose a training sample """
        rand = random.random()
        return min([i for i in range(len(train_buckets_scale))
                    if train_buckets_scale[i] > rand])

    def _assert_lengths(self, encoder_size, decoder_size, encoder_inputs, decoder_inputs, decoder_masks):
        """ Assert that the encoder inputs, decoder inputs, and decoder masks are
        of the expected lengths """
        if len(encoder_inputs) != encoder_size:
            raise ValueError("Encoder length must be equal to the one in bucket,"
                            " %d != %d." % (len(encoder_inputs), encoder_size))
        if len(decoder_inputs) != decoder_size:
            raise ValueError("Decoder length must be equal to the one in bucket,"
                           " %d != %d." % (len(decoder_inputs), decoder_size))
        if len(decoder_masks) != decoder_size:
            raise ValueError("Weights length must be equal to the one in bucket,"
                           " %d != %d." % (len(decoder_masks), decoder_size))

    def run_step(self, model, encoder_inputs, decoder_inputs, decoder_masks, bucket_id, forward_only):
        """ Run one step in training.
        @forward_only: boolean value to decide whether a backward path should be created
        forward_only is set to True when you just want to evaluate on the test set,
        or when you want to the bot to be in chat mode. """
        encoder_size, decoder_size = self.cfg['BUCKETS'][bucket_id]
        self._assert_lengths(encoder_size, decoder_size, encoder_inputs, decoder_inputs, decoder_masks)

        # input feed: encoder inputs, decoder inputs, target_weights, as provided.
        input_feed = {}
        for step in range(encoder_size):
            input_feed[model.encoder_inputs[step].name] = encoder_inputs[step]
        for step in range(decoder_size):
            input_feed[model.decoder_inputs[step].name] = decoder_inputs[step]
            input_feed[model.decoder_masks[step].name] = decoder_masks[step]

        last_target = model.decoder_inputs[decoder_size].name
        input_feed[last_target] = np.zeros([model.batch_size], dtype=np.int32)

        # output feed: depends on whether we do a backward step or not.
        if not forward_only:
            output_feed = [model.train_ops[bucket_id],  # update op that does SGD.
                           model.gradient_norms[bucket_id],  # gradient norm.
                           model.losses[bucket_id]]  # loss for this batch.
        else:
            output_feed = [model.losses[bucket_id]]  # loss for this batch.
            for step in range(decoder_size):  # output logits.
                output_feed.append(model.outputs[bucket_id][step])

        outputs = self.sess.run(output_feed, input_feed)
        if not forward_only:
            return outputs[1], outputs[2], None  # Gradient norm, loss, no outputs.
        else:
            return None, outputs[0], outputs[1:]  # No gradient norm, loss, outputs.

    def _get_buckets(self):
        """ Load the dataset into buckets based on their lengths.
        train_buckets_scale is the inverval that'll help us
        choose a random bucket later on.
        """
        test_buckets = self.reader.load_data('test_ids.enc', 'test_ids.dec')
        data_buckets = self.reader.load_data('train_ids.enc', 'train_ids.dec')
        train_bucket_sizes = [len(data_buckets[b]) for b in range(len(self.cfg['BUCKETS']))]
        print("Number of samples in each bucket:\n", train_bucket_sizes)
        train_total_size = sum(train_bucket_sizes)
        # list of increasing numbers from 0 to 1 that we'll use to select a bucket.
        train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                               for i in range(len(train_bucket_sizes))]
        print("Bucket scale:\n", train_buckets_scale)
        return test_buckets, data_buckets, train_buckets_scale

    def is_epoch_end(self, iteration):
        return iteration % self.cfg['TRAINING_SAMPLES'] == 0

    def _get_skip_step(self, iteration):
        """ How many steps should the model train before it saves all the weights. """
        # if iteration < 100:
        #     return 30
        return 10

    def _check_restore_parameters(self, saver, iteration=None):
        """ Restore the previously trained parameters if there are any. """
        ckpt = tf.train.get_checkpoint_state(os.path.dirname(self.cfg['CPT_PATH'] + '/checkpoint'))
        if ckpt and ckpt.model_checkpoint_path:
            # Load model of intermediate training stage
            if iteration:
                tmp = ckpt.model_checkpoint_path.rstrip('1234567890') + str(iteration)
                print(tmp)
                if os.path.isfile(tmp + ".index"):
                    ckpt.model_checkpoint_path = tmp
                else:
                    sys.stderr.write('The model at iteration %d you want to load does not exist\n' % iteration)
                    sys.exit()
            print("Loading parameters for the Chatbot")
            saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            print("Initializing fresh parameters for the Chatbot")

    def _eval_test_set(self, model, test_buckets):
        """ Evaluate on the test set. """
        for bucket_id in range(len(self.cfg['BUCKETS'])):
            if len(test_buckets[bucket_id]) == 0:
                print("  Test: empty bucket %d" % (bucket_id))
                continue
            start = time.time()
            encoder_inputs, decoder_inputs, decoder_masks = self.reader.get_batch(test_buckets[bucket_id],
                                                                            bucket_id,
                                                                            batch_size=self.cfg['BATCH_SIZE'])
            _, step_loss, _ = self.run_step(model, encoder_inputs, decoder_inputs,
                                       decoder_masks, bucket_id, True)
            print('\t\tTest bucket {}: loss {}, time {}'.format(bucket_id, step_loss, time.time() - start))


    def train(self, save_end):
        """ Train the bot """
        test_buckets, data_buckets, train_buckets_scale = self._get_buckets()
        # in train mode, we need to create the backward path, so forwrad_only is False
        model = ChatBotModel(config=self.cfg, forward_only=False, batch_size=self.cfg['BATCH_SIZE'])
        model.build_graph()

        saver = tf.train.Saver()

        print('Running session')
        self.sess.run(tf.global_variables_initializer())

        ###########LOAD THE EMBBEDING FROM THE WORD2VEC##########

        self.load_embbedings(self.sess)




        self._check_restore_parameters(saver)




        iteration = model.global_step.eval(session=self.sess)
        total_loss = 0

        # estimate the passes over the data
        stop_at_iteration = self.cfg['EPOCHS'] * self.cfg['TRAINING_SAMPLES']

        # <BG> moved outside of the loop
        skip_step = self._get_skip_step(iteration)

        while iteration < stop_at_iteration:
            bucket_id = self._get_random_bucket(train_buckets_scale)
            encoder_inputs, decoder_inputs, decoder_masks = self.reader.get_batch(data_buckets[bucket_id],
                                                                           bucket_id,
                                                                           batch_size=self.cfg['BATCH_SIZE'])
            start = time.time()
            _, step_loss, _ = self.run_step(model, encoder_inputs, decoder_inputs, decoder_masks, bucket_id, False)
            total_loss += step_loss
            iteration += 1

            if iteration % (self.cfg['EVAL_MULTIPLICATOR'] * skip_step) == 0:
                # Run evals on development set and print their loss
                self._eval_test_set(model, test_buckets)
                start = time.time()
                sys.stdout.flush()

            if self.is_epoch_end(iteration):
                print('Iter {}: loss {}, time {}'.format(iteration, total_loss/skip_step, time.time() - start))
                start = time.time()
                total_loss = 0
                if not save_end:
                    model.save_model(sess=self.sess)
                sys.stdout.flush()

        # Save model at the end of training
        model.save_model(sess=self.sess)

    def _get_user_input(self):
        """ Get user's input, which will be transformed into encoder input later """
        print("> ", end="")
        sys.stdout.flush()
        return sys.stdin.readline()

    def _find_right_bucket(self, length):
        """ Find the proper bucket for an encoder input based on its length """
        return min([b for b in range(len(self.cfg['BUCKETS']))
                    if self.cfg['BUCKETS'][b][0] >= length])

    def _construct_response(self, output_logits, inv_dec_vocab):
        """ Construct a response to the user's encoder input.
        @output_logits: the outputs from sequence to sequence wrapper.
        output_logits is decoder_size np array, each of dim 1 x DEC_VOCAB

        This is a greedy decoder - outputs are just argmaxes of output_logits.
        """
        # print(output_logits[0])
        outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
        # If there is an EOS symbol in outputs, cut them at that point.
        if self.cfg['EOS_ID'] in outputs:
            outputs = outputs[:outputs.index(self.cfg['EOS_ID'])]
        # Print out sentence corresponding to outputs.
        return " ".join([tf.compat.as_str(inv_dec_vocab[output]) for output in outputs])

    def chat(self, model_iteration=None):
        """ in test mode, we don't to create the backward path
        """
        _, enc_vocab = self.reader.load_vocab(os.path.join(self.cfg['PROCESSED_PATH'], 'vocab.enc'))
        inv_dec_vocab, _ = self.reader.load_vocab(os.path.join(self.cfg['PROCESSED_PATH'], 'vocab.dec'))

        model = ChatBotModel(config=self.cfg, forward_only=True, batch_size=1)
        model.build_graph()

        saver = tf.train.Saver()

        self.sess.run(tf.global_variables_initializer())
        self._check_restore_parameters(saver, iteration=model_iteration)
        output_file = open(os.path.join(self.cfg['PROCESSED_PATH'], self.cfg['OUTPUT_FILE']), 'a+')

        # Decode from standard input.
        max_length = self.cfg['BUCKETS'][-1][0]
        print('Welcome to TensorBro. Say something. Enter to exit. Max length is', max_length)
        while True:
            line = self._get_user_input()
            if len(line) > 0 and line[-1] == '\n':
                line = line[:-1]
            if line == '':
                break
            output_file.write('HUMAN ++++ ' + line + '\n')
            # Get token-ids for the input sentence.
            token_ids = self.reader.sentence2id(enc_vocab, str(line))
            if (len(token_ids) > max_length):
                print('Max length I can handle is:', max_length)
                line = self._get_user_input()
                continue
            # Which bucket does it belong to?
            bucket_id = self._find_right_bucket(len(token_ids))
            # Get a 1-element batch to feed the sentence to the model.
            encoder_inputs, decoder_inputs, decoder_masks = self.reader.get_batch([(token_ids, [])],
                                                                            bucket_id,
                                                                            batch_size=1)
            # Get output logits for the sentence.
            _, _, output_logits = self.run_step(model, encoder_inputs, decoder_inputs,
                                           decoder_masks, bucket_id, True)
            response = self._construct_response(output_logits, inv_dec_vocab)
            print(response)
            output_file.write('BOT ++++ ' + response + '\n')
        output_file.write('=============================================\n')
        output_file.close()

    def test_perplexity(self, inpath, model_iteration=None):
        """
        <FL> Mostly the same setup as chat(), except we read from a file.
        This is used after training for final results on our data set
        """

        # if self.cfg['USE_CORNELL']:
        #     raise RuntimeError("Testing must have USE_CORNELL = False")

        _, enc_word_to_i = self.reader.load_vocab(os.path.join(self.cfg['PROCESSED_PATH'], 'vocab.enc'))
        _, dec_word_to_i = self.reader.load_vocab(os.path.join(self.cfg['PROCESSED_PATH'], 'vocab.dec'))

        # Obtain the test sentencess
        questions, answers = self.reader.make_pairs(inpath)
        questions = [self.reader.sentence2id(enc_word_to_i, x) for x in questions]
        answers = [self.reader.sentence2id(enc_word_to_i, x) for x in answers]

        # Put every example through one at a time
        model = ChatBotModel(config=self.cfg, forward_only=True, batch_size=1)
        model.build_graph()

        saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())
        self._check_restore_parameters(saver, iteration=model_iteration)

        # Right now a triple A B C becomes two pairs
        # A B and B C. So we first run A, compare against B, and then run B and
        # compare against C.
        # We do that because we need two numbers / line
        for i in range(0, len(questions), 2):
            assert(answers[i] == questions[i + 1])

            a = questions[i]
            b = questions[i + 1]
            c = answers[i + 1]

            bucket_ab = self._find_right_bucket2(len(a), len(b))
            bucket_bc = self._find_right_bucket2(len(b), len(c))

            enc_in_a, dec_in_b, dec_masks_b = self.reader.get_batch([(a, b)], bucket_ab, 1)
            enc_in_b, dec_in_c, dec_masks_c = self.reader.get_batch([(b, c)], bucket_bc, 1)

            # bucket x batch x vocab
            _, _, logits_b = self.run_step(model, enc_in_a, dec_in_b,
                                        dec_masks_b, bucket_ab, forward_only=True)

            _, _, logits_c = self.run_step(model, enc_in_b, dec_in_c,
                                        dec_masks_c, bucket_bc, forward_only=True)

            soft_b = np.exp(logits_b) / np.sum(np.exp(logits_b), axis = 0)
            soft_c = np.exp(logits_c) / np.sum(np.exp(logits_c), axis = 0)

            perp_b = measures.perplexity(self.cfg, soft_b, b, dec_word_to_i)
            perp_c = measures.perplexity(self.cfg, soft_c, c, dec_word_to_i)

            print("%f %f" % (perp_b, perp_c))



    def _find_right_bucket2(self, lengtha, lengthb):
        available_a = [b for b in range(len(self.cfg['BUCKETS']))
                        if self.cfg['BUCKETS'][b][0] >= lengtha]
        available_b = [b for b in range(len(self.cfg['BUCKETS']))
                        if self.cfg['BUCKETS'][b][1] >= lengthb]

        both = set(available_a) & set(available_b)
        return min(both)




def main():
    global cfg
    timestamp = time.strftime('%Y-%m-%d--%H_%M_%S')
    sys.stdout = Logger(timestamp)

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices={'train', 'chat', 'test'},
                        default='train', help="mode. if not specified, it's in the train mode")
    parser.add_argument('--cornell', action='store_true', help="use the cornell movie dialogue corpus")
    parser.add_argument('--conversations', help="limit the number of conversations used in the dataset")
    parser.add_argument('--model', help='specify name (timestamp) of a previously used model. If none is provided then the newest model available will be used.')
    parser.add_argument('--load_iter', help='if you do not want to use an intermediate version of the trained network, specify the iteration number here')
    parser.add_argument('test_file', type=str, nargs='?')
    parser.add_argument('--test_conversations', help="limit the number of test conversations used in the dataset")
    parser.add_argument('--softmax', action='store_true', help='use standard softmax loss instead of sampled softmax loss')
    parser.add_argument('--clear', action='store_true', help="delete all existing models to free up disk space")
    parser.add_argument('--keep_prev', action='store_true', help='keep only the most recent version of the trained network')
    parser.add_argument('--epochs', help='how many times the network should pass over the entire dataset. Note: Due to random bucketing, this is an approximation.')
    parser.add_argument('--save_end', action='store_true', help='save the model only at the end of training')
    args = parser.parse_args()

    if args.clear:
        shutil.rmtree(cfg['MODELS_PATH'])

    if not os.path.exists(cfg['MODELS_PATH']):
            os.makedirs(cfg['MODELS_PATH'])


    ########### load old config file if necessary ############
    if args.model:
        # load the corresponding config file
        cfg = config.load_cfg(args.model)
    else:
        if args.mode == 'chat':
            # default mode: load the config of the last used model
            saved_models = [name for name in os.listdir(cfg['MODELS_PATH']) if os.path.isdir(os.path.join(cfg['MODELS_PATH'],name))]
            saved_models = sorted(saved_models) # sorted in ascending order 
            last_model = saved_models[-1]
            cfg = config.load_cfg(last_model)
            # pp = pprint.PrettyPrinter(indent=4)
            # pp.pprint(cfg)
        else:
            # create a new model
            config.adapt_to_dataset(args.cornell)
            cfg['MODEL_NAME'] = timestamp
            directory = os.path.join(cfg['MODELS_PATH'], cfg['MODEL_NAME'])
            if not os.path.exists(directory):
                os.makedirs(directory)
            config.adapt_paths_to_model()

    ####### process other commmand line arguments ##############
    if args.conversations:
        cfg['MAX_TURNS'] = int(args.conversations)
    if args.test_conversations:
        cfg['TESTSET_SIZE'] = int(args.test_conversations)
    if args.softmax:
        cfg['STANDARD_SOFTMAX'] = args.softmax
    if args.keep_prev:
        cfg['KEEP_PREV'] = True
    if args.load_iter:
        args.load_iter = int(args.load_iter)
    if args.epochs:
        cfg['EPOCHS'] = int(args.epochs)

    ################ read data #################
    if args.cornell:
        import cornell_data as data
    else:
        import our_data as data

    reader = data.Reader(cfg)
    if not os.path.isdir(cfg['PROCESSED_PATH']): # Will not be done if we're in test mode, chat mode or continue training
        reader.prepare_raw_data()
        reader.process_data()
    print('Data ready!')


    ########### start using the actual model##################
    # create checkpoints folder if there isn't one already
    reader.make_dir(cfg['CPT_PATH'])

    bot = Chatbot(config=cfg, reader=reader)
    if args.mode == 'train':
        bot.train(args.save_end)
    elif args.mode == 'chat':
        bot.chat(args.load_iter)
    elif args.mode == 'test':
        bot.test_perplexity(args.test_file, args.load_iter)

if __name__ == '__main__':
    main()
