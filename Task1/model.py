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
        self.tfconfig.gpu_options.allow_growth = True
        #self.tfconfig.gpu_options.per_process_gpu_memory_fraction = 0.95

    def build_forward_prop(self, embeddings=None):

        print("building the forward model...")
        # This is one mini-batch
        self.input = tf.placeholder(dtype=tf.int32, shape=[None, cfg["sentence_length"]])
        one_hot = tf.one_hot(indices=self.input, depth=cfg["vocab_size"], axis=-1, dtype=tf.float32)

        # from 3D tensor to 2D tensor, dim batch_Size x vocab_size
        # returns a list of 2D tensors
        self.input_words = tf.split(value=one_hot, num_or_size_splits=cfg["sentence_length"], axis=1)

        # We can discard the last 2D tensor from batch_positionwise
        # Cut the <eos> tags for the input data
        # x = tf.slice(input_=one_input_position_batch, begin=[0,0], size=[-1, cfg["sentence_length"] - 1])

        initializer = tf.contrib.layers.xavier_initializer()
        dtype = tf.float32

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

        lstm_layer = []
        lstm_out = []

        W_out = tf.get_variable(name='W_out', dtype=dtype, shape=[cfg["lstm_size"], cfg["vocab_size"]],
                                initializer=initializer)
        bias_out = tf.get_variable(name='bias_out', dtype=dtype, shape=[cfg["vocab_size"]], initializer=initializer)
        self.out_layer = []

        for i in range(cfg["sentence_length"] - 1):
            if cfg["use_fred"]:
                scope = tf.variable_scope('lstm' + str(i))
                lstm_layer.append(lstm.LstmCell(scope))
            else:
                # Use tensorflows cell
                lstm_layer.append(LSTMCell(num_units=cfg["lstm_size"], forget_bias=1.0, state_is_tuple=True, activation=tf.tanh))

        for i in range(cfg["sentence_length"] - 1):

            # 3. Fully connected of 100

            fc_emb_layer.append(tf.nn.relu(tf.matmul(tf.squeeze(self.input_words[i]), W_emb) + bias_emb))

            lstm_scope = tf.VariableScope(reuse=None, name="lstm"+str(i))
            # 4. LSTM dim 512
            if i == 0:
                if cfg["use_fred"]:
                    # (batch_size x lstm_size, batch_size x lstm_size)
                    lstm_out.append(lstm_layer[i](X=fc_emb_layer[i],
                                                  state=(tf.zeros(shape=[cfg["batch_size"], cfg["lstm_size"]]),
                                                         tf.zeros(shape=[cfg["batch_size"], cfg["lstm_size"]]))
                                                  )
                                    )
                else:
                    zero_state = lstm_layer[i].zero_state(cfg["batch_size"], dtype=dtype)
                    lstm_out.append(lstm_layer[i](inputs=fc_emb_layer[i],
                        state=zero_state,
                        scope=lstm_scope))
            else:
                if cfg["use_fred"]:
                    lstm_out.append(lstm_layer[i](X=fc_emb_layer[i], state=lstm_out[i - 1]))
                else:
                    lstm_out.append(lstm_layer[i](inputs=fc_emb_layer[i], state=lstm_out[i-1][1], scope=lstm_scope))

            # 5. Linear FC output layer

            # Output layer
            self.out_layer.append(tf.matmul(lstm_out[i][0], W_out) + bias_out)

    def build_backprop(self):
        print("building the backprop model...")

        y_hat = []
        loss = []

        for i in range(cfg["sentence_length"] - 1):
            # 6. Softmax output + cat cross entropy loss

            # we ommit the 0-th word in each sentence (namely the <bos> tag).
            # The labels start position 1
            y_hat.append(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=tf.argmax(tf.squeeze(self.input_words[i + 1]), axis=1),
                    logits=self.out_layer[i]))

            # Loss for one output position, reduced over the minibatch
            loss.append(tf.reduce_mean(y_hat[i]))

        concatenated_losses = tf.stack(values=loss)
        optimizer = tf.train.AdamOptimizer()

        # Clipped gradients
        gvs = optimizer.compute_gradients(concatenated_losses)
        list_clipped, _ = tf.clip_by_global_norm(t_list=[x[0] for x in gvs], clip_norm=10)  # second output not used
        self.train_op = optimizer.apply_gradients(list(zip(list_clipped, [x[1] for x in gvs])))

        # Make available for other operations.
        self.losses = concatenated_losses

    def build_test(self):
        print('building the test operations...')

        # Stack the individual output layers by sentence. Stacking with axis=0 would be by batch
        self.test_op = tf.nn.softmax(tf.stack(self.out_layer, axis=1))

    def test_loss(self, data):

        sess = tf.InteractiveSession(config=self.tfconfig)
        tf.global_variables_initializer().run()

        batch_indices = define_minibatches(data.shape[0], False)
        batched_losses = []
        for i, batch_idx in enumerate(batch_indices):
            batch = data[batch_idx]
            this_loss = sess.run(self.losses, feed_dict={self.input: batch})

            # Sum over sentence positions, getting one loss per sentence
            batched_losses.append(np.sum(this_loss, axis=-1))

        return np.mean(batched_losses)

    # Test data is available for measurements
    def train(self, train_data, test_data):
        """
        train_data          id_data, 2D
        test_data           id_data, 2D
        """
        sess = tf.InteractiveSession(config=self.tfconfig)
        tf.global_variables_initializer().run()

        for e in range(cfg["max_iterations"]):
            print("Starting epoch %d..." % e)
            start = time.time()

            batch_indices = define_minibatches(train_data.shape[0])
            for i, batch_idx in enumerate(batch_indices):
                batch = train_data[batch_idx]
                sess.run(fetches=self.train_op, feed_dict={self.input: batch})

                # Log test loss every so often
                if cfg["out_batch"] > 0 and i % (cfg["out_batch"] + 1) == 0 :
                    print("\tTest loss (mean per sentence) at batch %d: %f" % (i, self.test_loss(test_data)))

            print("Epoch completed in %d seconds." % (time.time() - start))

    def test(self, data, vocab_dict, cut_last_batch=0):
        """
        data            id_data, 2D
        """
        sess = tf.InteractiveSession(config=self.tfconfig)
        tf.global_variables_initializer().run()

        print('Testing...')

        out_test = open('perplexity.txt', 'w')

        # Assume that the data we got is evenly divisible by the batch size.
        # The reader has taken care of that by padding with extra dummy inputs
        # that we can ignore. Additionally, do not randomly permute the data.
        batch_indices = define_minibatches(data.shape[0], False)
        for i, batch_idx in enumerate(batch_indices):
            start = time.time()
            batch = data[batch_idx]

            # print('Starting test batch %d' % i)

            estimates = sess.run(fetches=self.test_op, feed_dict={self.input: batch})

            eval_size = len(batch_idx)
            if i == len(batch_indices) - 1:
                eval_size = cfg["batch_size"] - cut_last_batch

            for j in range(eval_size):
                perp = perplexity(estimates[j], batch[j], vocab_dict)
                out_test.write(str(perp) + '\n')

                # print(('Test batch %d completed in %d seconds' % (i, time.time() - start)))

        out_test.close()


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
