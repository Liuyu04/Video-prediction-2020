__author__ = 'liuyu'

import tensorflow as tf
from layers.TensorLayerNorm import tensor_layer_norm


class STDLSTMCell():
    def __init__(self, layer_name, filter_size, num_hidden_in, num_hidden_out,
                 seq_shape, forget_bias=1.0, tln=False, initializer=0.001):  #
        """Initialize the Causal LSTM cell.
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
        # self.batch = 8
        self.height = seq_shape[2]
        self.width = seq_shape[3]
        # self.height = 1
        # self.width = 1
        self.layer_norm = tln
        self._forget_bias = forget_bias
        self.initializer = tf.random_uniform_initializer(-initializer, initializer)

    def init_state(self):
        return tf.zeros([self.batch, self.height, self.width, self.num_hidden],
                        dtype=tf.float32)

    def __call__(self, x, h, c, m):
        if h is None:
            h = tf.zeros([self.batch, self.height, self.width,
                          self.num_hidden],
                         dtype=tf.float32)
        if c is None:
            c = tf.zeros([self.batch, self.height, self.width,
                          self.num_hidden],
                         dtype=tf.float32)
        if m is None:
            m = tf.zeros([self.batch, self.height, self.width,
                          self.num_hidden_in],
                         dtype=tf.float32)

        with tf.variable_scope(self.layer_name):
            h_cc = tf.layers.conv2d(
                h, self.num_hidden * 5,
                self.filter_size, 1, padding='same',
                kernel_initializer=self.initializer,
                name='temporal_state_transition')
            c_cc = tf.layers.conv2d(
                c, self.num_hidden * 5,
                self.filter_size, 1, padding='same',
                kernel_initializer=self.initializer,
                name='temporal_memory_transition')
            m_cc = tf.layers.conv2d(
                m, self.num_hidden * 3,
                self.filter_size, 1, padding='same',
                kernel_initializer=self.initializer,
                name='spatial_memory_transition')
            if self.layer_norm:
                h_cc = tensor_layer_norm(h_cc, 'h2c')
                c_cc = tensor_layer_norm(c_cc, 'c2c')
                m_cc = tensor_layer_norm(m_cc, 'm2m')

            i_h, g_h, i_h_, g_h_, f_h_ = tf.split(h_cc, 5, 3)
            i_c, g_c, i_c_, g_c_, f_c_ = tf.split(c_cc, 5, 3)
            i_m, g_m, f_m = tf.split(m_cc, 3, 3)  ### check

            #            m_m = g_m   ### add by liuyu
            m_m = tf.layers.conv2d(
                m, self.num_hidden,
                self.filter_size, 1, padding='same',
                kernel_initializer=self.initializer,
                name='conv_spatial')
            if self.layer_norm:
                m_m = tensor_layer_norm(m_m, 'm_m')

            if x is None:
                g = tf.tanh(g_h + g_c)
                i = tf.sigmoid(i_h + i_c)
                ff = tf.sigmoid(f_h_ + f_c_ + self._forget_bias)
                gg = tf.tanh(g_h_ + g_c_)
                ii = tf.sigmoid(i_h_ + i_c_)
            else:
                x_cc = tf.layers.conv2d(
                    x, self.num_hidden * 10,
                    self.filter_size, 1, padding='same',
                    kernel_initializer=self.initializer,
                    name='input_to_state')
                if self.layer_norm:
                    x_cc = tensor_layer_norm(x_cc, 'x2c')

                i_x, g_x, c_x, o_x, i_x_, g_x_, f_x_, i_x__, g_x__, f_x__ = tf.split(x_cc, 10, 3)

                g = tf.tanh(g_x + g_h + g_c)
                i = tf.sigmoid(i_x + i_h + i_c)
                ff = tf.sigmoid(f_x_ + f_h_ + f_c_ + self._forget_bias)
                gg = tf.tanh(g_x_ + g_h_ + g_c_)
                ii = tf.sigmoid(i_x_ + i_h_ + i_c_)

            # print('ff shape', ff.shape)
            # print('i shape: ', i.shape)
            # print('g shape', g.shape)

            #### val
            mid = 1 - (1 - ff) * i
            # print('mid shape: ', mid.shape)
            mid_1 = mid * c
            # print('mid1 shape: ', mid_1.shape)
            mid_2 = i * g
            # print('mid2 shape: ', mid_2.shape)
            c_hat = mid_1 + mid_2

            # c_hat = [1 - (1 - ff) * i ] * c + i * g
            print('c_hat shape: ', c_hat.shape)  # 1,8,1,64
            print('c shape: ', c.shape)  # 8,1,1,64
            # c_hat = tf.squeeze(input = c_hat)
            # print('c hat again shape: ', c_hat.shape)  # 8,64
            delta_c = c_hat - c

            c_c = tf.layers.conv2d(
                delta_c, self.num_hidden,
                self.filter_size, 1, padding='same',
                kernel_initializer=self.initializer,
                name='conv_distance')
            if self.layer_norm:
                c_c = tensor_layer_norm(c_c, 'c_c')

            if x is None:
                c1t = tf.sigmoid(c_c + i_h_)  #### change
            else:
                c1t = tf.sigmoid(c_x + c_c + i_h_)

            c_new = ff * c + ii * c1t * gg

            c2m = tf.layers.conv2d(
                c_new, self.num_hidden * 4,
                self.filter_size, 1, padding='same',
                kernel_initializer=self.initializer,
                name='c2m')
            if self.layer_norm:
                c2m = tensor_layer_norm(c2m, 'c2m')

            i_c, g_c, f_c, o_c = tf.split(c2m, 4, 3)  #### change

            if x is None:
                iii = tf.sigmoid(i_c + i_m)
                fff = tf.sigmoid(f_c + f_m + self._forget_bias)
                ggg = tf.tanh(g_c + g_m)  ### change by liuyu
            else:
                iii = tf.sigmoid(i_c + i_x__ + i_m)
                fff = tf.sigmoid(f_c + f_x__ + f_m + self._forget_bias)
                ggg = tf.tanh(g_c + g_x__ + g_m)  ### change by liuyu

            m_new = fff * tf.tanh(m_m) + iii * ggg

            o_m = tf.layers.conv2d(
                m_new, self.num_hidden,
                self.filter_size, 1, padding='same',
                kernel_initializer=self.initializer,
                name='m_to_o')
            if self.layer_norm:
                o_m = tensor_layer_norm(o_m, 'm2o')

            #            o_c = tf.layers.conv2d(
            #                c_new, self.num_hidden,
            #                self.filter_size, 1, padding='same',
            #                kernel_initializer=self.initializer,
            #                name='c_to_o')
            #            if self.layer_norm:
            #                o_c = tensor_layer_norm(o_c, 'c2o')

            if x is None:
                o = tf.tanh(o_c + o_m)  ###  change again
            else:
                o = tf.tanh(o_x + o_c + o_m)  ### check again

            cell = tf.concat([c_new, m_new], -1)  #### change
            cell = tf.layers.conv2d(cell, self.num_hidden, 1, 1,
                                    padding='same', name='memory_reduce')

            h_new = o * tf.tanh(cell)

            return h_new, c_new, m_new


