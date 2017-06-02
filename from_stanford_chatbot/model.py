""" A neural chatbot using sequence to sequence model with
attentional decoder.

This is based on Google Translate Tensorflow model
https://github.com/tensorflow/models/blob/master/tutorials/rnn/translate/

Sequence to sequence model by Cho et al.(2014)

Created by Chip Huyen as the starter code for assignment 3,
class CS 20SI: "TensorFlow for Deep Learning Research"
cs20si.stanford.edu

This file contains the code to build the model

See readme.md for instruction on how to run the starter code.
"""
from __future__ import print_function

import time
import os

import numpy as np
import tensorflow as tf

import config

# import config

SAMPLED_w = None
SAMPLED_b = None


class ChatBotModel(object):

    def __init__(self, config, forward_only, batch_size):
        """forward_only: if set, we do not construct the backward pass in the model.
        """
        print('Initialize new model')
        self.cfg = config
        self.fw_only = forward_only
        self.batch_size = batch_size

    def save_model(self, sess):
        saver = tf.train.Saver()
        saver.save(sess, os.path.join(self.cfg['CPT_PATH'], 'chatbot'), global_step=self.global_step)
        config.save_cfg(self.cfg)



    def sampled_loss(self,labels, inputs):
        labels = tf.reshape(labels, [-1, 1])
        return tf.nn.sampled_softmax_loss(tf.transpose(SAMPLED_w), SAMPLED_b, labels, inputs,
                                        self.cfg['NUM_SAMPLES'], self.cfg['DEC_VOCAB'])

    def _create_placeholders(self):
        # Feeds for inputs. It's a list of placeholders
        print('Create placeholders')
        self.encoder_inputs = [tf.placeholder(tf.int32, shape=[None], name='encoder{}'.format(i))
                               for i in range(self.cfg['BUCKETS'][-1][0])]
        self.decoder_inputs = [tf.placeholder(tf.int32, shape=[None], name='decoder{}'.format(i))
                               for i in range(self.cfg['BUCKETS'][-1][1] + 1)]
        self.decoder_masks = [tf.placeholder(tf.float32, shape=[None], name='mask{}'.format(i))
                              for i in range(self.cfg['BUCKETS'][-1][1] + 1)]

        # Our targets are decoder inputs shifted by one (to ignore <s> symbol)
        self.targets = [tf.cast(x, tf.float32) for x in self.decoder_inputs[1:]]

    def _inference(self):
        global SAMPLED_w
        global SAMPLED_b
        print('Create inference')
        print('Num samples ' + str(self.cfg['NUM_SAMPLES']))
        print('Dec vocab ' + str(self.cfg['DEC_VOCAB']))
        # If we use sampled softmax, we need an output projection.
        # Sampled softmax only makes sense if we sample less than vocabulary size.
        if self.cfg['NUM_SAMPLES'] > 0 and self.cfg['NUM_SAMPLES'] < self.cfg['DEC_VOCAB']:
            SAMPLED_w = tf.get_variable('proj_w', [self.cfg['HIDDEN_SIZE'], self.cfg['DEC_VOCAB']])
            SAMPLED_b = tf.get_variable('proj_b', [self.cfg['DEC_VOCAB']])
            self.output_projection = (SAMPLED_w, SAMPLED_b)

        single_cell = tf.contrib.rnn.GRUCell(self.cfg['HIDDEN_SIZE'])
        self.cell = tf.contrib.rnn.MultiRNNCell([single_cell] * self.cfg['NUM_LAYERS'])

    def _create_loss(self):
        print('Creating loss... \nIt might take a couple of minutes depending on how many buckets you have.')
        start = time.time()
        def _seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
            return tf.contrib.legacy_seq2seq.embedding_rnn_seq2seq(
                    encoder_inputs, decoder_inputs, self.cell,
                    num_encoder_symbols=self.cfg['ENC_VOCAB'],
                    num_decoder_symbols=self.cfg['DEC_VOCAB'],
                    embedding_size=self.cfg['HIDDEN_SIZE'],
                    output_projection=self.output_projection,
                    feed_previous=do_decode)

        if self.fw_only:
            self.outputs, self.losses = tf.contrib.legacy_seq2seq.model_with_buckets(
                                        self.encoder_inputs,
                                        self.decoder_inputs,
                                        self.targets,
                                        self.decoder_masks,
                                        self.cfg['BUCKETS'],
                                        lambda x, y: _seq2seq_f(x, y, True),
                                        softmax_loss_function=self.sampled_loss)
            # If we use output projection, we need to project outputs for decoding.
            if self.output_projection:
                for bucket in range(len(self.cfg['BUCKETS'])):
                    self.outputs[bucket] = [tf.matmul(output,
                                            self.output_projection[0]) + self.output_projection[1]
                                            for output in self.outputs[bucket]]
        else:
            self.outputs, self.losses = tf.contrib.legacy_seq2seq.model_with_buckets(
                                        self.encoder_inputs,
                                        self.decoder_inputs,
                                        self.targets,
                                        self.decoder_masks,
                                        self.cfg['BUCKETS'],
                                        lambda x, y: _seq2seq_f(x, y, False),
                                        softmax_loss_function=self.sampled_loss)
        print('Time:', time.time() - start)

    def _creat_optimizer(self):
        print('Create optimizer... \nIt might take a couple of minutes depending on how many buckets you have.')
        with tf.variable_scope('training') as scope:
            self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

            if not self.fw_only:
                self.optimizer = tf.train.GradientDescentOptimizer(self.cfg['LR'])
                trainables = tf.trainable_variables()
                self.gradient_norms = []
                self.train_ops = []
                start = time.time()
                for bucket in range(len(self.cfg['BUCKETS'])):

                    clipped_grads, norm = tf.clip_by_global_norm(tf.gradients(self.losses[bucket],
                                                                 trainables),
                                                                 self.cfg['MAX_GRAD_NORM'])
                    self.gradient_norms.append(norm)
                    self.train_ops.append(self.optimizer.apply_gradients(zip(clipped_grads, trainables),
                                                            global_step=self.global_step))
                    print('Creating opt for bucket {} took {} seconds'.format(bucket, time.time() - start))
                    start = time.time()


    def _create_summary(self):
        pass

    def build_graph(self):
        self._create_placeholders()
        self._inference()
        self._create_loss()
        self._creat_optimizer()
        self._create_summary()
