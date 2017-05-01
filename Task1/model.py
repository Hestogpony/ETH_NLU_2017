import tensorflow as tf
import numpy as np

from config import cfg


def train_model(data, embeddings=None):
    # 0. Cut the <eos> tags from x
    # x = 
    
    # 1. Batching
    inp = tf.placeholder(dtype=tf.float32, shape=[None,30,20000])
    batch = tf.train.batch(
        tensors=inp,
        batch_size=cfg["batch_size"],
        #capacity=32,
        allow_smaller_final_batch=True,
    )

    initializer = tf.contrib.layers.xavier_initializer()
    dtype = tf.float32

    #2. Input Layer of 20000
    # TODO: from batch to 1D vector of 20.000

    one_word = None
    
    W_emb = []
    bias_emb = []
    fc_emb_layer = []
    for i in range(29):

        #3. Fully connected of 100
        W_emb.append(tf.get_variable(dtype=dtype, shape=[cfg["embeddings_size"],cfg["vocab_size"]], initializer=initializer))
        bias_emb.append(tf.get_variable(dtype=dtype, shape=[cfg["embeddings_size"]], initializer=initializer))

        fc_emb_layer.append(tf.nn.relu(W_emb * one_word + bias1))

        #4. LSTM - Fred, dim 512
        
        lstm = None

        #5. Linear FC output layer
        
        #Output layer
        W_output = tf.get_variable(dtype=dtype, shape=[cfg["vocab_size"],cfg["lstm_size"]], initializer=initializer)
        bias_output = tf.get_variable(dtype=dtype, shape=[cfg["vocab_size"]], initializer=initializer)
        output_layer = W_output * lstm + bias_output

        #6. Softmax output + cat cross entropy loss
        
        # TODO: Cut the <bos> tags from the labels
        #y =  
        
        y_hat = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=output_layer)

    #7. Somehow duplicate this and connect the hidden units
    
    #8. standard Backprop, gradient optimizer
    
 
 def test_model(data, params):
    # Forward Prop:
    # Same network, with trained parameters
    # - Softmax outputs --> Passed to complexity functions
