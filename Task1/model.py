import tensorflow as tf
import numpy as np
import time


import lstm
from config import cfg

class Model(object):
    def __init__(self, embeddings=None):
        self.embeddings = embeddings


    def build_forward_prop(self, batch_size=1, embeddings=None):

        print("building the forward model...")
        # This is one mini-batch
        self.inp = tf.placeholder(dtype=tf.float32, shape=[None,cfg["sentence_length"], cfg["vocab_size"]])

        # from 3D tensor to 2D tensor, dim batch_Size x vocab_size
        # returns a list of 2D tensors
        self.batch_positionwise = tf.split(value=self.inp, num_or_size_splits = cfg["sentence_length"], axis=1)

        # We can discard the last 2D tensor from batch_positionwise
        # Cut the <eos> tags for the input data
        #x = tf.slice(input_=one_input_position_batch, begin=[0,0], size=[-1, cfg["sentence_length"] - 1])

        initializer = tf.contrib.layers.xavier_initializer()
        dtype = tf.float32


        W_emb = []
        bias_emb = []
        fc_emb_layer = []

        lstm_layer = []
        lstm_out = []

        W_out = []
        bias_out = []
        self.out_layer = []


        for i in range(cfg["sentence_length"] - 1):
            scope = tf.variable_scope('lstm' + str(i))
            lstm_layer.append(lstm.LstmCell(scope))

        for i in range(cfg["sentence_length"] - 1):

            #3. Fully connected of 100
            W_emb.append(tf.get_variable(name=str(i)+'W_emb', dtype=dtype, shape=[cfg["vocab_size"], cfg["embeddings_size"]], initializer=initializer))
            bias_emb.append(tf.get_variable(name=str(i)+'bias_emb', dtype=dtype, shape=[cfg["embeddings_size"]], initializer=initializer))


            fc_emb_layer.append(tf.nn.relu(tf.matmul(tf.squeeze(self.batch_positionwise[i]), W_emb[i]) + bias_emb[i]))

            #4. LSTM dim 512
            if i == 0:
                # (batch_size x lstm_size, batch_size x lstm_size)
                lstm_out.append(lstm_layer[i](X=fc_emb_layer[i],
                    state=(tf.zeros(shape=[batch_size, cfg["lstm_size"]]),
                        tf.zeros(shape=[batch_size, cfg["lstm_size"]]))
                    )
                )
            else:
                lstm_out.append(lstm_layer[i](X=fc_emb_layer[i], state=lstm_out[i-1]))

            #5. Linear FC output layer

            #Output layer
            W_out.append( tf.get_variable(name=str(i)+'W_out', dtype=dtype, shape=[cfg["lstm_size"], cfg["vocab_size"]], initializer=initializer) )
            bias_out.append( tf.get_variable(name=str(i)+'bias_out', dtype=dtype, shape=[cfg["vocab_size"]], initializer=initializer) )
            self.out_layer.append( tf.matmul(lstm_out[i][0], W_out[i]) + bias_out[i] )


    def build_backprop(self):
        print("building the backprop model...")

        y_hat = []
        cost = []

        for i in range(cfg["sentence_length"] - 1):
            #6. Softmax output + cat cross entropy loss

            # we ommit the 0-th word in each sentence (namely the <bos> tag).
            # The labels start position 1
            y_hat.append(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=tf.argmax(tf.squeeze(self.batch_positionwise[i+1]), axis=1),
                    logits=self.out_layer[i]) )

            # Loss for one output position, reduced over the minibatch
            cost.append(tf.reduce_mean(y_hat[i]))


        concatenated_costs = tf.stack(values=cost)
        optimizer = tf.train.AdamOptimizer()

        #Clipped gradients
        gvs = optimizer.compute_gradients(concatenated_costs)
        list_clipped, _ = tf.clip_by_global_norm(t_list=[x[0] for x in gvs], clip_norm=10) # second output not used
        self.train_op = optimizer.apply_gradients(zip(list_clipped, [x[1] for x in gvs]))

        # Unrestricted gradients
        # train_op = optimizer.minimize(concatenated_costs)

    def build_test(self):
        print('building the test operations...')

        self.test_op = tf.nn.softmax(tf.stack(self.out_layer))


    def train(self, data):
        sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()

        for e in range(cfg["max_iterations"]):
            batch_indices = define_minibatches(data.shape[0])
            for i, batch_idx in enumerate(batch_indices):
                start = time.time()
                batch = data[batch_idx]

                print(("Starting batch %d" % i))

                sess.run(fetches=self.train_op, feed_dict={self.inp:batch})

                print(('Batch %d completed in %d seconds' % (i, time.time() - start)))
                # print('\tCosts: ' + str(costs))

                # file_writer = tf.summary.FileWriter('./train_graph', sess.graph)
                # file_writer.add_summary(summary)

    def test(self, data, word_dict):

        print('Testing...')

        out_test = open('perplexity.txt', 'w')

        # Assume that the data we got is evenly divisible by the batch size.
        # The reader has taken care of that by padding with extra dummy inputs
        # that we can ignore. Additionally, do not randomly permute the data.
        batch_indices = define_minibatches(data.shape[0], False)
        for i, batch_idx in enumerate(batch_indices):
            start = time.time()
            batch = data[batch_idx]

            print('Starting test batch %d' % i)

            estimates = sess.run(fetches=self.test_op, feed_dict={self.inp:batch})
            for i in range(len(batch_idx)):
                perp = perplexity(estimates[i], np.argmax(batch[i], -1), word_dict)
                out_test.write(str(perp) + '\n')

            print(('Test batch %d completed in %d seconds' % (i, time.time() - start)))

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

    batches = np.split(indices, indices_or_sections = length/cfg["batch_size"])
    return batches



 # def test_model(data, params):
 #    # Forward Prop:
 #    # Same network, with trained parameters
 #    # - Softmax outputs --> Passed to complexity functions
 #    pass



