__author__ = 'liuyu'

##### the Conv-LSTM reproduce by liuyu at 2020/3/20 for comparison

import tensorflow as tf
from layers.TensorLayerNorm import tensor_layer_norm


class ConvLSTMCell():
    def __init__(self, layer_name, filter_size, num_hidden_in, num_hidden_out,
                 seq_shape, forget_bias=1.0, tln=False, initializer=0.001):
        """Initialize the conv LSTM cell.
        Args:
            layer_name: layer names for different lstm layers.
            filter_size: int tuple thats the height and width of the filter.
            num_hidden_in: number of units for input tensor.
            num_hidden_out: number of units for output tensor.
            seq_shape: shape of a sequence.
            forget_bias: float, The bias added to forget gates.
            tln: whether to apply tensor layer normalization
        """
        self.layer_name = layer_name
        self.filter_size = filter_size
        self.num_hidden_in = num_hidden_in
        self.num_hidden = num_hidden_out
        self.batch = seq_shape[0]
        self.height = seq_shape[2]
        self.width = seq_shape[3]
        self.layer_norm = tln
        self._forget_bias = forget_bias
        self.initializer = tf.random_uniform_initializer(-initializer, initializer)

    def init_state(self):
        return tf.zeros([self.batch, self.height, self.width, self.num_hidden],
                        dtype=tf.float32)

    def __call__(self, x, h, c):
        if h is None:
            h = tf.zeros([self.batch, self.height, self.width,
                          self.num_hidden],
                         dtype=tf.float32)
        if c is None:
            c = tf.zeros([self.batch, self.height, self.width,
                          self.num_hidden],
                         dtype=tf.float32)

        with tf.variable_scope(self.layer_name):
            h_cc = tf.layers.conv2d(
                h, self.num_hidden * 4,
                self.filter_size, 1, padding='same',
                kernel_initializer=self.initializer,
                name='temporal_state_transition')
            if self.layer_norm:
                h_cc = tensor_layer_norm(h_cc, 'h2c')

            i_h, g_h, f_h, o_h = tf.split(h_cc, 4, 3)

            if x is None:
                i = tf.sigmoid(i_h )
                f = tf.sigmoid(f_h + self._forget_bias)
                g = tf.tanh(g_h)
                o = tf.sigmoid(o_h)  ###  change again
            else:
                x_cc = tf.layers.conv2d(
                    x, self.num_hidden * 4,
                    self.filter_size, 1, padding='same',
                    kernel_initializer=self.initializer,
                    name='input_to_state')
                if self.layer_norm:
                    x_cc = tensor_layer_norm(x_cc, 'x2c')

                i_x, g_x, f_x, o_x = tf.split(x_cc, 4, 3)

                i = tf.sigmoid(i_x + i_h)
                f = tf.sigmoid(f_x + f_h + self._forget_bias)
                g = tf.tanh(g_x + g_h)
                o = tf.sigmoid(o_x + o_h)  ###  change again

            c_new = f * c + i * g

            h_new = o * tf.tanh(c_new)

            return h_new, c_new


