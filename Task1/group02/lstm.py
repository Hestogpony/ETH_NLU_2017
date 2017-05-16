'''
Implementation of the LSTM cell and network. The cell is defined as follows:

    * An 'input' which computes a candidate value to introduce in the cell.
    * A 'modulate' gate which creates a mask for the candidate value, indicating
    how much it should be suppressed.
    * A 'forget' gate which creates a mask for the old cell value, indicating
    how much it should be suppressed.
    * A 'reveal' gate which creates a mask for the new cell value (i.e. after
    the forget operation has been applied and the modulated new value added),
    indicating how much it should be used in the hidden state at this time step.

Some authors concatenate the input and hidden vectors before passing them
through these gates. We instead follow the approach where each gate has two
matrices, one for the input and another for the hidden. Concretely, the
equations are as follows:

    i_t = tanh( W_i * x_t  +  U_i * h_(t-1) + b_i )     (input)
    m_t =  sig( W_m * x_t  +  U_m * h_(t-1) + b_m )     (modulate)
    f_t =  sig( W_f * x_t  +  U_f * h_(t-1) + b_f )     (forget)
    r_t =  sig( W_r * x_t  +  U_r * h_(t-1) + b_r )     (reveal)

    C_t = C_(t-1) * f_t  +  i_t * m_t             (new cell state)
    h_t = tanh(C_t) * r_t                         (new hidden state)

For the exact implementation of the cell, we uphold the convention that the
state is the tuple (hidden vector, cell vector)

'''

import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer as xavier
from config import cfg

class LstmCell:
    def __init__(self):

        # Some shortcuts for the dimensions we need
        HID_HID = [cfg['lstm_size'], cfg['lstm_size']]
        IN_HID = [cfg['embeddings_size'], cfg['lstm_size']]
        HID = [1, cfg['lstm_size']]

        # The hidden vector is the output
        self.output_size = HID

        # The state consists of the cell and the hidden vectors, and both have
        # the same dimensions
        self.state_size = tf.TensorShape(HID_HID)


        # W are the matrices which multiply the input, and U are the matrices
        # which multiply the previous hidden state

        # Input variables
        self.Wi = tf.get_variable('Wi', IN_HID, initializer=xavier())
        self.Ui = tf.get_variable('Ui', HID_HID, initializer=xavier())
        self.bi = tf.get_variable('bi', HID, initializer=xavier())

        # Modulation variables
        self.Wm = tf.get_variable('Wm', IN_HID, initializer=xavier())
        self.Um = tf.get_variable('Um', HID_HID, initializer=xavier())
        self.bm = tf.get_variable('bm', HID, initializer=xavier())

        # Forget variables
        self.Wf = tf.get_variable('Wf', IN_HID, initializer=xavier())
        self.Uf = tf.get_variable('Uf', HID_HID, initializer=xavier())
        self.bf = tf.get_variable('bf', HID, initializer=xavier())

        # Reveal variables
        self.Wr = tf.get_variable('Wr', IN_HID, initializer=xavier())
        self.Ur = tf.get_variable('Ur', HID_HID, initializer=xavier())
        self.br = tf.get_variable('br', HID, initializer=xavier())


    def __call__(self, inputs, state, scope):
        '''
        We follow the tf documentation on RNNCell:
          * **x** has shape batch_size x input_size
          * **state** has shape (batch_size x lstm_size, batch_size x lstm_size)
            with the first element being the _hidden_ vector and the second
            element being the _cell_ vector
          * **scope** is ignored here. (variables declared in the cell constructor)

        Returns: a tuple of the new hidden and cell vectors.
        '''

        X = inputs

        h_old = state[0]
        c_old = state[1]

        # Straightforwardly implement the equations described above
        i =    tf.tanh(tf.matmul(X, self.Wi) + tf.matmul(h_old, self.Ui) + self.bi)
        m = tf.sigmoid(tf.matmul(X, self.Wm) + tf.matmul(h_old, self.Um) + self.bm)
        f = tf.sigmoid(tf.matmul(X, self.Wf) + tf.matmul(h_old, self.Uf) + self.bf)
        r = tf.sigmoid(tf.matmul(X, self.Wr) + tf.matmul(h_old, self.Ur) + self.br)

        c_new = tf.multiply(c_old, f) + tf.multiply(i, m)
        h_new = tf.multiply(tf.tanh(c_new), r)

        # LSTM "output" is the hidden state
        return (h_new, (h_new, c_new))

