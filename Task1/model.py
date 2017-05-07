import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell
import numpy as np
import time

import lstm
from config import cfg
from perplexity import perplexity

class Model(object):
    def __init__(self, embeddings=None):
        self.embeddings = embeddings
        self.tfconfig = tf.ConfigProto()

        self.model_session = tf.Session(config=self.tfconfig)



    def build_forward_prop(self, embeddings=None):

        print("building the forward model...")

        # This is one mini-batch
        self.input_forward = tf.placeholder(dtype=tf.int32, shape=[None, None])

        # Initial states for the LSTM cell; we'll just pass zeros, but we use
        # a placeholder since we don't know the batch size here yet
        self.initial_hidden = tf.placeholder(dtype=tf.float32, shape=[None, cfg["lstm_size"]])
        self.initial_cell = tf.placeholder(dtype=tf.float32, shape=[None, cfg["lstm_size"]])

        one_hot = tf.one_hot(indices=self.input_forward, depth=cfg["vocab_size"], axis=-1, dtype=tf.float32)

        initializer = tf.contrib.layers.xavier_initializer()
        dtype = tf.float32
        lstm_scope = tf.VariableScope(name="lstm", reuse=None)

        # init with given word embeddings if provided
        if embeddings:
            W_emb.append = tf.Variable(name='W_emb', dtype=dtype, initial_value=embeddings,
                                       expected_shape=[cfg["vocab_size"], cfg["embeddings_size"]])
        else:
            W_emb = tf.get_variable(name='W_emb', dtype=dtype, shape=[cfg["vocab_size"], cfg["embeddings_size"]],
                                    initializer=initializer)

        bias_emb = tf.get_variable(name='bias_emb', dtype=dtype, shape=[cfg["embeddings_size"]],
                                   initializer=initializer)
        fc_emb_layer = []
        lstm_out = []

        if cfg["extra_project"]:
            W_out_intermediate = tf.get_variable(name='W_out_intermediate',
                                    dtype=dtype,
                                    shape=[cfg["lstm_size"], cfg["intermediate_projection_size"]],
                                    initializer=initializer)
            bias_out_intermediate = tf.get_variable(name='bias_out_intermediate',
                                    dtype=dtype,
                                    shape=[cfg["intermediate_projection_size"]],
                                    initializer=initializer)
            w_out_input_size = cfg["intermediate_projection_size"]
        else:
            w_out_input_size = cfg["lstm_size"]


        W_out = tf.get_variable(name='W_out',
                                dtype=dtype,
                                shape=[w_out_input_size, cfg["vocab_size"]],
                                initializer=initializer)
        bias_out = tf.get_variable(name='bias_out', dtype=dtype, shape=[cfg["vocab_size"]], initializer=initializer)
        self.out_layer = []

        if cfg["use_fred"]:
            lstm_cell = lstm.LstmCell()
        else:
            lstm_cell = LSTMCell(num_units=cfg["lstm_size"],
                                forget_bias=1.0,
                                state_is_tuple=True,
                                activation=tf.tanh)

        for i in range(cfg["sentence_length"] - 1):

            # 3. Fully connected for embeddings
            # Take the ith slice of batch_size x sentence_length x vocab_size
            # along axis 1, yielding batch_size x vocab_size
            ith_word_batch = tf.squeeze(tf.slice(one_hot, [0, i, 0], [-1, 1, -1]), axis=[1])
            embs = tf.matmul(ith_word_batch, W_emb) + bias_emb
            fc_emb_layer.append(tf.nn.relu(embs))

            # 4. LSTM
            # Note: calling the cell returns a tuple with the output and the new
            # state. Here we only care about the new state since it also contains
            # the output.

            if i == 0:
                lstm_out.append(
                    lstm_cell(
                        fc_emb_layer[i],
                        (self.initial_hidden, self.initial_cell),
                        lstm_scope)[1])
                lstm_scope.reuse_variables()
            else:
                lstm_out.append(lstm_cell(fc_emb_layer[i], lstm_out[i - 1], lstm_scope)[1])

            # Extra projection layer in experiment C
            if cfg["extra_project"]:
                to_project = tf.matmul(lstm_out[i][0], W_out_intermediate) + bias_out_intermediate
            else:
                to_project = lstm_out[i][0]

            # 5. Linear FC output layer
            self.out_layer.append(tf.matmul(to_project, W_out) + bias_out)

        self.output_and_state = (self.out_layer[0], lstm_out[0])

    def build_backprop(self):
        print("building the backprop model...")

        y_hat = []
        loss = []

        labs = tf.slice(self.input_forward, [0, 1], [-1, -1])
        logs = tf.stack(self.out_layer, axis=1)
        self.total_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                            labels=labs, logits=logs)

        optimizer = tf.train.AdamOptimizer()

        # Clipped gradients
        gvs = optimizer.compute_gradients(self.total_loss)
        grads = [x[0] for x in gvs]
        vars = [x[1] for x in gvs]

        clipped_grads, _ = tf.clip_by_global_norm(t_list=grads, clip_norm=10)  # second output not used
        self.train_op = optimizer.apply_gradients(list(zip(clipped_grads, vars)))

    def build_test(self):
        print('building the test operations...')

        # Stack the individual output layers by sentence. Stacking with axis=0 would be by batch
        self.test_op = tf.nn.softmax(tf.stack(self.out_layer, axis=1))

    def test_loss(self, data, cut_last_batch=0):

        batch_indices = define_minibatches(data.shape[0], False)
        batched_losses = []
        for i, batch_idx in enumerate(batch_indices):
            batch = data[batch_idx]
            food = {
                self.input_forward: batch,
                self.initial_hidden: np.zeros((cfg["batch_size"], cfg["lstm_size"])),
                self.initial_cell: np.zeros((cfg["batch_size"], cfg["lstm_size"]))
            }

            this_loss = self.model_session.run(self.total_loss, feed_dict=food)

            eval_size = cfg["batch_size"]
            if i == len(batch_indices) - 1:
                eval_size = cfg["batch_size"] - cut_last_batch

            # Sum over sentence positions, getting one loss per sentence
            # Afterwards, cut out dummy values in the last batch if necessary
            batched_losses.append(np.sum(this_loss, axis=-1)[:eval_size])

        return np.mean(np.concatenate(batched_losses))

    # Test data is available for measurements
    def train(self, train_data, test_data, cut_last_test_batch=0):
        """
        train_data          id_data, 2D
        test_data           id_data, 2D
        """
        tf.global_variables_initializer().run(session=self.model_session)


        for e in range(cfg["max_iterations"]):
            print("Starting epoch %d..." % e)
            start_epoch = start_batches = time.time()

            batch_indices = define_minibatches(train_data.shape[0])
            for i, batch_idx in enumerate(batch_indices):
                batch = train_data[batch_idx]

                food = {
                    self.input_forward: batch,
                    self.initial_hidden: np.zeros((cfg["batch_size"], cfg["lstm_size"])),
                    self.initial_cell: np.zeros((cfg["batch_size"], cfg["lstm_size"]))
                }

                self.model_session.run(fetches=self.train_op, feed_dict=food)

                # Log test loss every so often
                if cfg["out_batch"] > 0 and i > 0 and (i % (cfg["out_batch"]) == 0) :
                    print("\tBatch chunk %d - %d finished in %d seconds" % (i-cfg["out_batch"], i, time.time() - start_batches))
                    print("\tTest loss (mean per sentence) at batch %d: %f" % (i, self.test_loss(test_data, cut_last_test_batch)))
                    start_batches = time.time()

            print("Epoch completed in %d seconds." % (time.time() - start_epoch))

        # Save the trained network to use it for Task 1.2
        if "save_model_path" in cfg:
            save_model(session=self.model_session)

    def test(self, data, vocab_dict, cut_last_batch=0):
        """
        data            id_data, 2D
        """

        print('Testing...')

        experiment_letter = "A"
        if cfg["use_pretrained"]:
            experiment_letter = "B"
        if cfg["extra_project"]:
            experiment_letter = "C"

        out_test = open(cfg["path"]["output"]+experiment_letter, 'w')

        # Assume that the data we got is evenly divisible by the batch size.
        # The reader has taken care of that by padding with extra dummy inputs
        # that we can ignore. Additionally, do not randomly permute the data.
        batch_indices = define_minibatches(data.shape[0], False)
        for i, batch_idx in enumerate(batch_indices):
            start = time.time()
            batch = data[batch_idx]

            # print('Starting test batch %d' % i)

            food = {
                self.input_forward: batch,
                self.initial_hidden: np.zeros((cfg["batch_size"], cfg["lstm_size"])),
                self.initial_cell: np.zeros((cfg["batch_size"], cfg["lstm_size"]))
            }

            estimates = self.model_session.run(fetches=self.test_op, feed_dict=food)

            eval_size = len(batch_idx)
            if i == len(batch_indices) - 1:
                eval_size = cfg["batch_size"] - cut_last_batch

            for j in range(eval_size):
                perp = perplexity(estimates[j], batch[j], vocab_dict)
                out_test.write(str(perp) + '\n')

                # print(('Test batch %d completed in %d seconds' % (i, time.time() - start)))

        out_test.close()

    def generate(self, sentences, vocab_dict):
        # TODO should we assume that sentences have the bos tag or not?

        """
        sentences: a list of sentences. Each sentence is a list of word IDs
        """

        print('Generating...')

        cur_h = np.zeros((1, cfg["lstm_size"]))
        cur_c = np.zeros((1, cfg["lstm_size"]))
        cur_w = None

        for beginning in sentences:
            completed_sentence = beginning

            for word in beginning:
                food = {
                    self.input_forward: [[word]],
                    self.initial_hidden: cur_h,
                    self.initial_cell: cur_c
                }

                # We don't care about the output of the model for now,
                # we just feed in stuff.
                new_w, (cur_h, cur_c) = self.model_session.run(fetches=self.output_and_state, feed_dict=food)
                cur_w = np.argmax(new_w, axis=-1)[0]

            for i in range(cfg["generate_length"] - len(beginning)):
                food = {
                    self.input_forward: [[cur_w]],
                    self.initial_hidden: cur_h,
                    self.initial_cell: cur_c
                }

                new_w, (cur_h, cur_c) = self.model_session.run(fetches=self.output_and_state, feed_dict=food)
                cur_w = np.argmax(new_w, axis=-1)[0]

                completed_sentence.append(cur_w)
                if vocab_dict[cur_w] == "eos":
                    break

            sentence = " ".join([vocab_dict[x] for x in completed_sentence])
            print(sentence)





def save_model(session):
    saver = tf.train.Saver()
    save_path = saver.save(session, cfg["save_model_path"])
    print("Model saved in file: %s" % save_path)

def load_model(session):
    saver = tf.train.Saver()
    load_path = saver.restore(sess,cfg["save_model_path"])
    print("Model from %s restored" % load_path)

def define_minibatches(length, permute=True):
    if permute:
        # create a random permutation (for training over multiple epochs)
        indices = np.random.permutation(length)
    else:
        # use the indices in a sequential manner (for testing)
        indices = np.arange(length)

    # Cut out the last sentences in case data set is not divisible by the batch size
    rest = length % cfg["batch_size"]
    if rest is not 0:
        indices = indices[:-rest]

    batches = np.split(indices, indices_or_sections=len(indices) / cfg["batch_size"])
    return batches
