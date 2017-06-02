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

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import argparse
import random
import sys
import time

import numpy as np
import tensorflow as tf

from model import ChatBotModel

import config
from config import cfg

class Chatbot(object):
    def __init__(self, config, reader):
        self.cfg = config
        self.reader = reader

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

    def run_step(self, sess, model, encoder_inputs, decoder_inputs, decoder_masks, bucket_id, forward_only):
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

        outputs = sess.run(output_feed, input_feed)
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

    def _get_skip_step(self, iteration):
        """ How many steps should the model train before it saves all the weights. """
        if iteration < 100:
            return 30
        return 100

    def _check_restore_parameters(self, sess, saver):
        """ Restore the previously trained parameters if there are any. """
        ckpt = tf.train.get_checkpoint_state(os.path.dirname(self.cfg['CPT_PATH'] + '/checkpoint'))
        if ckpt and ckpt.model_checkpoint_path:
            print("Loading parameters for the Chatbot")
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print("Initializing fresh parameters for the Chatbot")

    def _eval_test_set(self, sess, model, test_buckets):
        """ Evaluate on the test set. """
        for bucket_id in range(len(self.cfg['BUCKETS'])):
            if len(test_buckets[bucket_id]) == 0:
                print("  Test: empty bucket %d" % (bucket_id))
                continue
            start = time.time()
            encoder_inputs, decoder_inputs, decoder_masks = self.reader.get_batch(test_buckets[bucket_id],
                                                                            bucket_id,
                                                                            batch_size=self.cfg['BATCH_SIZE'])
            _, step_loss, _ = self.run_step(sess, model, encoder_inputs, decoder_inputs,
                                       decoder_masks, bucket_id, True)
            print('Test bucket {}: loss {}, time {}'.format(bucket_id, step_loss, time.time() - start))

    def train(self):
        """ Train the bot """
        test_buckets, data_buckets, train_buckets_scale = self._get_buckets()
        # in train mode, we need to create the backward path, so forwrad_only is False
        model = ChatBotModel(config=self.cfg, forward_only=False, batch_size=self.cfg['BATCH_SIZE'])
        model.build_graph()

        saver = tf.train.Saver()

        with tf.Session() as sess:
            print('Running session')
            sess.run(tf.global_variables_initializer())
            self._check_restore_parameters(sess, saver)

            iteration = model.global_step.eval()
            total_loss = 0
            while True:
                skip_step = self._get_skip_step(iteration)
                bucket_id = self._get_random_bucket(train_buckets_scale)
                encoder_inputs, decoder_inputs, decoder_masks = self.reader.get_batch(data_buckets[bucket_id],
                                                                               bucket_id,
                                                                               batch_size=self.cfg['BATCH_SIZE'])
                start = time.time()
                _, step_loss, _ = self.run_step(sess, model, encoder_inputs, decoder_inputs, decoder_masks, bucket_id, False)
                total_loss += step_loss
                iteration += 1

                if iteration % skip_step == 0:
                    print('Iter {}: loss {}, time {}'.format(iteration, total_loss/skip_step, time.time() - start))
                    start = time.time()
                    total_loss = 0
                    saver.save(sess, os.path.join(self.cfg['CPT_PATH'], 'chatbot'), global_step=model.global_step)
                    if iteration % (10 * skip_step) == 0:
                        # Run evals on development set and print their loss
                        self._eval_test_set(sess, model, test_buckets)
                        start = time.time()
                    sys.stdout.flush()

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

    def chat(self):
        """ in test mode, we don't to create the backward path
        """
        _, enc_vocab = self.reader.load_vocab(os.path.join(self.cfg['PROCESSED_PATH'], 'vocab.enc'))
        inv_dec_vocab, _ = self.reader.load_vocab(os.path.join(self.cfg['PROCESSED_PATH'], 'vocab.dec'))

        model = ChatBotModel(True, config=self.cfg, batch_size=1)
        model.build_graph()

        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            self._check_restore_parameters(sess, saver)
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
                _, _, output_logits = self.run_step(sess, model, encoder_inputs, decoder_inputs,
                                               decoder_masks, bucket_id, True)
                response = self._construct_response(output_logits, inv_dec_vocab)
                print(response)
                output_file.write('BOT ++++ ' + response + '\n')
            output_file.write('=============================================\n')
            output_file.close()




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices={'train', 'chat'},
                        default='train', help="mode. if not specified, it's in the train mode")
    parser.add_argument('--cornell', action='store_true', help="use the cornell movie dialogue corpus")
    parser.add_argument('--conversations', help="limit the number of conversations used in the dataset")
    parser.add_argument('--test_conversations', help="limit the number of test conversations used in the dataset")

    args = parser.parse_args()
    config.adapt_to_dataset(args.cornell)

    if args.cornell:
        import cornell_data as data
    else:
        import our_data as data

    reader = data.Reader(cfg)
    if not os.path.isdir(cfg['PROCESSED_PATH']):
        reader.prepare_raw_data()
        reader.process_data()
    print('Data ready!')
    # create checkpoints folder if there isn't one already
    reader.make_dir(cfg['CPT_PATH'])

    bot = Chatbot(config=cfg, reader=reader)
    if args.mode == 'train':
        bot.train()
    elif args.mode == 'chat':
        bot.chat()

if __name__ == '__main__':
    main()
