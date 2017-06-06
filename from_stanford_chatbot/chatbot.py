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
import shutil
import pickle
import atexit
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
			output_feed = [model.encoder_states[bucket_id],model.losses[bucket_id]]  # loss for this batch.
			for step in range(decoder_size):  # output logits.
				output_feed.append(model.outputs[bucket_id][step])

		outputs = self.sess.run(output_feed, input_feed)
		if not forward_only:
			return outputs[1], outputs[2], None  # Gradient norm, loss, no outputs.
		else:
			return outputs[0], outputs[1], outputs[2:]  # No gradient norm, loss, outputs.

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
		return iteration % (int(self.cfg['TRAINING_SAMPLES'] / self.cfg['BATCH_SIZE'])) == 0 

	def _get_skip_step(self, iteration):
		""" How many steps should the model train before it saves all the weights. """
		# if iteration < 100:
		#     return 30
		return self.cfg['SKIP_STEP']

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
		self._check_restore_parameters(saver)
		print('Start training ...')

		# Save model on program exit
		if self.cfg['SAVE_AT_EXIT']:
			atexit.register(model.save_model, self.sess)

		iteration = model.global_step.eval(session=self.sess)

		# estimate the passes over the data
		stop_at_iteration = int((self.cfg['EPOCHS'] * self.cfg['TRAINING_SAMPLES']) / self.cfg['BATCH_SIZE'])
		epoch = 1

		# <BG> moved outside of the loop
		skip_step = self._get_skip_step(iteration)

		epoch_start = time.time()
		chunk_start = time.time()
		total_loss = 0
		previous_chunks_loss = 0


		while iteration < stop_at_iteration:
			bucket_id = self._get_random_bucket(train_buckets_scale)
			encoder_inputs, decoder_inputs, decoder_masks, batch_source_encoder, batch_source_decoder= self.reader.get_batch(data_buckets[bucket_id],
																		   bucket_id,
																		   batch_size=self.cfg['BATCH_SIZE'])
			_, step_loss, _ = self.step_rl(model, self.cfg['BUCKETS'],encoder_inputs, decoder_inputs, decoder_masks, batch_source_encoder, batch_source_decoder, bucket_id)
			total_loss += step_loss
			previous_chunks_loss += step_loss
			iteration += 1

			if iteration % skip_step == 0:
				print('Iter {}: loss {}, time {} s'.format(iteration, previous_chunks_loss / (self.cfg['BATCH_SIZE'] * skip_step), time.time() - chunk_start))
				previous_chunks_loss = 0
				# Run evals on development set and print their loss
				self._eval_test_set(model, test_buckets)
				chunk_start = time.time()
				sys.stdout.flush()

			if self.is_epoch_end(iteration):
				print('\nEpoch %d is done' % epoch)
				print('Iter {}: loss {}, time {} s\n\n'.format(iteration, total_loss/self.cfg['TRAINING_SAMPLES'], time.time() - epoch_start))
				epoch += 1
				epoch_start = time.time()
				total_loss = 0
				if not save_end:
					model.save_model(sess=self.sess)
				sys.stdout.flush()

		# Obsolete because we save at exit! Save model at the end of training
		# model.save_model(sess=self.sess)

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

	def step_rl(self, model, buckets, encoder_inputs, decoder_inputs, target_weights, batch_source_encoder,
			  batch_source_decoder, bucket_id, rev_vocab=None, debug=True):

	# initialize
		init_inputs = [encoder_inputs, decoder_inputs, target_weights, bucket_id]

		batch_mask = [1 for _ in xrange(self.cfg['BATCH_SIZE'])]

		# debug
		# resp_tokens, resp_txt = self.logits2tokens(encoder_inputs, rev_vocab, sent_max_length, reverse=True)
		# if debug: print("[INPUT]:", resp_tokens)

		# Initialize
		#if debug: print("step_rf INPUTS: %s" %encoder_inputs)
		#if debug: print("step_rf TARGET: %s" %decoder_inputs)
		#if debug: print("batch_source_encoder: %s" %batch_source_encoder)
		#if debug: print("batch_source_decoder: %s" %batch_source_decoder)

		ep_rewards, ep_step_loss, enc_states = [], [], []
		ep_encoder_inputs, ep_target_weights, ep_bucket_id = [], [], []
		episode, dialog = 0, []
		# [Episode] per episode = n steps, until break
		print ("ep_encoder_inputs: %s" %ep_encoder_inputs)
		print("bucket: %s, bucekt[-1][0]: %s" %(buckets, buckets[-1][0]))
		while True:
			  #----[Step]------with general function-----------------------------

			  #ep_encoder_inputs.append(self.remove_type(encoder_inputs, type=0))
			ep_encoder_inputs.append(batch_source_encoder)
			  #decoder_len = [len(seq) for seq in batch_source_decoder]

			  #if debug: print ("ep_encoder_inputs shape: ", np.shape(ep_encoder_inputs))
			  #if debug: print ("[INPUT]: %s" %ep_encoder_inputs[-1])

			encoder_state, step_loss, output_logits = self.run_step(model, encoder_inputs, decoder_inputs, target_weights,
								  bucket_id, forward_only=False)
			print("encoder_state: " , np.shape(encoder_state))
			ep_target_weights.append(target_weights)
			ep_bucket_id.append(bucket_id)
			ep_step_loss.append(step_loss)

			state_tran = np.transpose(encoder_state, axes=(1,0,2))
			print("state_tran: ", np.shape(state_tran))
			state_vec = np.reshape(state_tran, (self.cfg['BATCH_SIZE'], -1))
			print("state_vec: ", np.shape(state_vec))
			enc_states.append(state_vec)

			resp_tokens = self.remove_type(output_logits, buckets[bucket_id], type=1)
			#print("remove_type resps: %s" %resp_tokens)
			#if debug: print("[RESP]: (%.4f), resp len: %s, content: %s" %(step_loss, len(resp_tokens), resp_tokens))
			try:
				encoder_trans = np.transpose(ep_encoder_inputs, axes=(1,0))
			except ValueError:
				encoder_trans = np.transpose(ep_encoder_inputs, axes=(1,0,2))
			#if debug: print ("[ep_encoder_inputs] shape: ", np.shape(ep_encoder_inputs))
			if debug: print ("[encoder_trans] shape: ", np.shape(encoder_trans))
			#if episode == 5: print (2/0)

			for i, (resp, ep_encoder) in enumerate(zip(resp_tokens, encoder_trans)):
				print("resp: ", resp)

				if (len(resp) <= 3) or (resp in self.dummy_dialogs) or (resp in ep_encoder.tolist()):
					batch_mask[i] = 0
					print("make mask index: %d, batch_mask: %s" %(i, batch_mask))

			if sum(batch_mask)==0 or episode>5: break

			  #----[Reward]----------------------------------------
			  # r1: Ease of answering
			print("r1: Ease of answering")
			r1 = [self.logProb(session, buckets, resp_tokens, [d for _ in resp_tokens], mask= batch_mask) for d in self.dummy_dialogs]
			print("r1: final value: ", r1)
			r1 = -np.mean(r1) if r1 else 0

			# r2: Information Flow
			r2_list = []
			if len(enc_states) < 2:
				r2 = 0
			else:
				batch_vec_a, batch_vec_b = enc_states[-2], enc_states[-1]
			for i, (vec_a, vec_b) in enumerate(zip(batch_vec_a, batch_vec_b)):
				if batch_mask[i] == 0: continue
				rr2 = sum(vec_a*vec_b) / sum(abs(vec_a)*abs(vec_b))
			  #print("vec_a*vec_b: %s" %sum(vec_a*vec_b))
			  #print("r2: %s" %r2)
				if(rr2 < 0):
					print("rr2: ", rr2)
					print("vec_a: ", vec_a)
					print("vec_b: ", vec_b)
					rr2 = -rr2
				else:
					rr2 = -log(rr2)
				r2_list.append(rr2)
			r2 = sum(r2_list)/len(r2_list)
			# r3: Semantic Coherence
			print("r3: Semantic Coherence")
			r3 = -self.logProb(session, buckets, resp_tokens, ep_encoder_inputs[-1], mask= batch_mask)

			# Episode total reward
			print("r1: %s, r2: %s, r3: %s" %(r1,r2,r3))
			R = 0.25*r1 + 0.25*r2 + 0.5*r3
			ep_rewards.append(R)
			#----------------------------------------------------
			episode += 1

			  #prepare for next dialogue
			bk_id = []
			for i in range(len(resp_tokens)):
				bk_id.append(min([b for b in range(len(buckets)) if buckets[b][0] >= len(resp_tokens[i])]))
			bucket_id = max(bk_id)
			feed_data = {bucket_id: [(resp_tokens, [])]}
			encoder_inputs, decoder_inputs, target_weights, batch_source_encoder, _ = self.get_batch(feed_data, bucket_id, batch_size = self.cfg["BATCH_SIZE"],type=2)

			if len(ep_rewards) == 0: ep_rewards.append(0)
			print("[Step] final:", episode, ep_rewards)
			# gradient decent according to batch rewards
			rto = 0.0
			if (len(ep_step_loss) <= 1) or (len(ep_rewards) <=1) or (max(ep_rewards) - min(ep_rewards) == 0):
				rto = 0.0
			else:
				rto = (max(ep_step_loss) - min(ep_step_loss)) / (max(ep_rewards) - min(ep_rewards))
			advantage = [np.mean(ep_rewards)*rto] * len(buckets)
			print("advantage: %s" %advantage)
			_, step_loss, _ = self.step(session, init_inputs[0], init_inputs[1], init_inputs[2], init_inputs[3],
					  forward_only=False, force_dec_input=False, advantage=advantage)

			return None, step_loss, None


# log(P(b|a)), the conditional likelyhood
	def logProb(self, session, buckets, tokens_a, tokens_b, mask=None):
		def softmax(x):
			return np.exp(x) / np.sum(np.exp(x), axis=0)

		# prepare for next dialogue
		#bucket_id = min([b for b in range(len(buckets)) if buckets[b][0] > len(tokens_a) and buckets[b][1] > len(tokens_b)])
		#print("tokens_a: %s" %tokens_a)
		print("tokens_b: %s" %tokens_b)

		bk_id = []
		for i in xrange(len(tokens_a)):
			bk_id.append(min([b for b in xrange(len(buckets))
							  if buckets[b][0] >= len(tokens_a[i]) and buckets[b][1] >= len(tokens_b[i])]))
		bucket_id = max(bk_id)


		#print("bucket_id: %s" %bucket_id)

		feed_data = {bucket_id: zip(tokens_a, tokens_b)}

		#print("logProb feed_back: %s" %feed_data[bucket_id])
		encoder_inputs, decoder_inputs, target_weights, _, _ = self.get_batch(feed_data, bucket_id, batch_size = cfg['BATCH_SIZE'],type=1)
		#print("logProb: encoder: %s; decoder: %s" %(encoder_inputs, decoder_inputs))
		# step
		_, _, output_logits = self.step(session, encoder_inputs, decoder_inputs, target_weights,
							bucket_id, forward_only=True, force_dec_input=True)

		logits_t = np.transpose(output_logits, (1,0,2))
		print("logits_t shape: " ,np.shape(logits_t))


		sum_p = []
		for i, (tokens, logits) in enumerate(zip(tokens_b, logits_t)):
			print("tokens: %s, index: %d" %(tokens, i))
			#print("logits: %s" %logits)

			#if np.sum(tokens) == 0: break
			if mask[i] == 0: continue
			p = 1
			for t, logit in zip(tokens, logits):
				#print("logProb: logit: %s" %logit)
				norm = softmax(logit)[t]
				#print ("t: %s, norm: %s" %(t, norm))
				p *= norm
			if p < 1e-100:
			  #print ("p: ", p)
			   p = 1e-100
			p = log(p) / len(tokens)
			print ("logProb: p: %s" %(p))
			sum_p.append(p)
		re = np.sum(sum_p)/len(sum_p)
		#print("logProb: P: %s" %(re))
		return re

		def remove_type(self, sequence, bucket,type=0):
			tokens = []
			resps = []
			if type == 0:
				tokens = [i for i in [t for t in reversed(sequence)] if i.sum() != 0]
			elif type == 1:
			#print ("remove_type type=1 tokens: %s" %sequence)
				for seq in sequence:
					#print("seq: %s" %seq)
					token = []
					for t in seq:
						#print("seq_t: %s" %t)
						# t = list(t)
						# print("list(t): %s" %t)
						# t = np.array(t)
						# print("array(t): %s" %t)
						token.append(int(np.argmax(t, axis=0)))
					tokens.append(token)
			#tokens = [i for i in [int(np.argmax(t, axis=1)) for t in [seq for seq in sequence]]]
			#tokens = [i for i in [int(t.index(max(t))) for t in [seq for seq in sequence]]]
			else:
				print ("type only 0(encoder_inputs) or 1(decoder_outputs)")
		  #print("remove_type tokens: %s" %tokens)
			tokens_t = []
			for col in range(len(tokens[0])):
				tokens_t.append([tokens[row][col] for row in range(len(tokens))])

			for seq in tokens_t:
				if data_utils.EOS_ID in seq:
					resps.append(seq[:seq.index(data_utils.EOS_ID)][:bucket[1]])
				else:
					resps.append(seq[:bucket[1]])
			return resps
		'''
		# make logits to tokens
		def logits2tokens(self, logits, rev_vocab, sent_max_length=None, reverse=False):
		if reverse:
		  tokens = [t[0] for t in reversed(logits)]
		else:
		  tokens = [int(np.argmax(t, axis=1)) for t in logits]
		if data_utils.EOS_ID in tokens:
		  eos = tokens.index(data_utils.EOS_ID)
		  tokens = tokens[:eos]
		txt = [rev_vocab[t] for t in tokens]
		if sent_max_length:
		  tokens, txt = tokens[:sent_max_length], txt[:sent_max_length]
		return tokens, txt


		def discount_rewards(self, r, gamma=0.99):
		""" take 1D float array of rewards and compute discounted reward """
		discounted_r = np.zeros_like(r)
		running_add = 0
		for t in reversed(xrange(0, r.size)):
			running_add = running_add * gamma + r[t]
			discounted_r[t] = running_add
		return discounted_r
		'''


































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
	parser.add_argument('--save_end', action='store_true', help='save the model ONLY at the end of training')
	parser.add_argument('--processed_path', help='Specify if you want to use exisiting preprocessed data')
	parser.add_argument('--no_save_at_exit',action='store_true', help='deactivate automatic model saving at keyboard interrupt')
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
	if args.processed_path:
		cfg['PROCESSED_PATH'] = args.processed_path
		vocab_sizes_dict = pickle.load(open(os.path.join(cfg['PROCESSED_PATH'],"vocab_sizes"), "rb"))
		cfg.update(vocab_sizes_dict)
	if args.no_save_at_exit:
		cfg['SAVE_AT_EXIT'] = False

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
