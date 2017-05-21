import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np

class Model(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.tfconfig = tf.ConfigProto()
        self.model_dtype = tf.float32

        self.model_session = tf.Session(config=self.tfconfig)

    def build_forward_prop(self):

        print("building the forward model...")


        initializer = tf.contrib.layers.xavier_initializer()

        # We suppose that all samples in one batch fall into the same bucket, i.e. have the same length with padding
        self.input_forward = tf.placeholder(dtype=self.model_dtype, shape=[None, None])

        W_embeddings = tf.get_variable(name='W_emb', dtype=self.model_dtype, shape=[self.cfg["vocab_size"], self.cfg["embeddings_size"]],
                                    initializer=initializer)
        tf.nn.embedding_lookup(W_embeddings, self.input_forward)

    # TODO adapt this to task 2 and buckets
    def define_minibatches(self, length, permute=True):
        if permute:
            # create a random permutation (for training over multiple epochs)
            indices = np.random.permutation(length)
        else:
            # use the indices in a sequential manner (for testing)
            indices = np.arange(length)

        # Hold out the last sentences in case data set is not divisible by the batch size
        rest = length % self.cfg["batch_size"]
        if rest is not 0:
            indices_even = indices[:-rest]
            indices_rest = indices[len(indices_even):]
            batches = np.split(indices_even, indices_or_sections=len(indices_even) / self.cfg["batch_size"])
            batches.append(np.array(indices_rest))
        else:
            batches = np.split(indices, indices_or_sections=len(indices) / self.cfg["batch_size"])



        return batches