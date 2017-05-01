import tensorflow as tf
import numpy as np

import lstm
from config import cfg


def train_model(data, embeddings=None):
    
    # 1. Batching
    inp = tf.placeholder(dtype=tf.float32, shape=[None,cfg["sentence_length"], cfg["vocab_size"]])
    batch = tf.train.batch(
        tensors=inp,
        batch_size=cfg["batch_size"],
        #capacity=32,
        allow_smaller_final_batch=True,
    )

    # from 3D tensor to 2D tensor, dim batch_Size x vocab_size
    # returns a list of 2D tensors
    batch_positionwise = tf.split(value=batch, num_or_size_splits = cfg["sentence_length"], axis=1)

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
    out_layer = []

    y_hat = []
    cost = []

    for i in range(cfg["sentence_length"] - 1):
        lstm_layer.append(LstmCell())

    for i in range(cfg["sentence_length"] - 1):

        #3. Fully connected of 100
        W_emb.append(tf.get_variable(dtype=dtype, shape=[cfg["vocab_size"], cfg["embeddings_size"]], initializer=initializer))
        bias_emb.append(tf.get_variable(dtype=dtype, shape=[cfg["embeddings_size"]], initializer=initializer))


        fc_emb_layer.append(tf.nn.relu(tf.matmul(batch_positionwise[i], W_emb[i]) + bias_emb[i]))

        #4. LSTM dim 512
        if i == 0:
            lstm_out.append(lstm_layer[i](X=fc_emb_layer[i], state=(tf.zeros(shape=[None,cfg["lstm_size"]], tf.zeros(shape=[None,cfg["lstm_size"]]))) ) )
        else:
            lstm_out.append(lstm_layer[i](X=fc_emb_layer[i], state=lstm_out[i-1]))

        #5. Linear FC output layer
        
        #Output layer
        W_out.append( tf.get_variable(dtype=dtype, shape=[cfg["vocab_size"],cfg["lstm_size"]], initializer=initializer) )
        bias_out.append( tf.get_variable(dtype=dtype, shape=[cfg["vocab_size"]], initializer=initializer) )
        out_layer.append( W_out[i] * lstm_out[i] + bias_out[i] )

        #6. Softmax output + cat cross entropy loss
        
        # we ommit the 0-th word in each sentence (namely the <bos> tag).
        # The labels start position 1
        y_hat.append( tf.nn.sparse_softmax_cross_entropy_with_logits(labels=batch_positionwise[i+1], logits=out_layer[i]) )

        cost.append(tf.reduce_mean(y_hat[i]))   


    concatenated_costs = tf.concat(values=cost, axis=0)
    optimizer = tf.train.AdamOptimizer()
    gvs = optimizer.compute_gradients(concatenated_costs)
    capped_gvs = tf.clip_by_global_norm(t_list=[x[0] for x in gvs], clip_norm=10)
    train_op = optimizer.apply_gradients(zip(capped_gvs, [x[1] for x in gvs]))
    
    sess = tf.Session()
    sess.run(fetches=train_op, feed_dict={inp:data})

 def test_model(data, params):
    # Forward Prop:
    # Same network, with trained parameters
    # - Softmax outputs --> Passed to complexity functions
