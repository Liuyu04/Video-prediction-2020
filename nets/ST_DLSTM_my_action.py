__author__ = 'liuyu'

import tensorflow as tf
from layers.GradientHighwayUnit import GHU as ghu
from layers.STDLSTMCell import STDLSTMCell as stdlstm

def rnn(images, targets, num_layers, num_hidden, filter_size, stride=1,classes=11,
        seq_length=16, input_length=16, tln=True):

#### targets: integer in 0-(classes-1) ---- 0-10
    gen_images = []
    lstm = []
    cell = []
    hidden = []
    shape = images.get_shape().as_list()
    #output_channels = shape[-1]

    for i in xrange(num_layers):
        if i == 0:
            num_hidden_in = num_hidden[num_layers-1]
        else:
            num_hidden_in = num_hidden[i-1]
        new_cell = stdlstm('lstm_'+str(i+1),
                          filter_size,
                          num_hidden_in,
                          num_hidden[i],
                          shape,
                          tln=tln)
        lstm.append(new_cell)
        cell.append(None)
        hidden.append(None)

    gradient_highway = ghu('highway', filter_size, num_hidden[0], tln=tln)

    mem = None
    z_t = None

    for t in xrange(seq_length):
        reuse = bool(gen_images)
        with tf.variable_scope('predrnn_pp', reuse=reuse):
            inputs = images[:,t]
            #if t < input_length:
                #inputs = images[:,t]
            #else:
                #inputs = mask_true[:,t-input_length]*images[:,t] + (1-mask_true[:,t-input_length])*x_gen

            hidden[0], cell[0], mem = lstm[0](inputs, hidden[0], cell[0], mem)
            z_t = gradient_highway(hidden[0], z_t)
            hidden[1], cell[1], mem = lstm[1](z_t, hidden[1], cell[1], mem)

            for i in xrange(2, num_layers):
                hidden[i], cell[i], mem = lstm[i](hidden[i-1], hidden[i], cell[i], mem)
            ### hidden (N,10,10,64)
            print('hidden shape: ', hidden[num_layers-1].shape)

            x_gen = tf.reduce_mean(hidden[num_layers-1], [1, 2], name='global_average_pool', keep_dims=True)

            #x_gen = tf.layers.conv2d(inputs=hidden[num_layers-1],
                                     #filters=output_channels,
                                     #kernel_size=1,
                                     #strides=1,
                                     #padding='same',
                                     #name="back_to_pixel")
            gen_images.append(x_gen)

    gen_images = tf.stack(gen_images)
    print('gen_image shape: ', gen_images.shape)
    # [batch_size, seq_length, height, width, channels]
    gen_images = tf.transpose(gen_images, [1,0,2,3,4])
    gen_images = tf.layers.flatten(gen_images)
    logits = tf.layers.dense(inputs=gen_images,units=classes,activation=None)
    y_pred=tf.nn.softmax(logits)
    yy = tf.argmax(y_pred,axis=1)

    onehot_labels = tf.one_hot(targets, classes)
    loss = tf.reduce_mean(-tf.reduce_sum(onehot_labels * tf.log(y_pred),reduction_indices=[1]))
    #loss = tf.nn.softmax_cross_entropy_with_logits(labels=onehot_labels,logits=y_pred,name='xentropy')
    #loss = tf.reduce_mean(loss, name='xentropy_mean')
    #loss = tf.nn.softmax_cross_entropy_with_logits(y_pred, targets)
    #loss = tf.nn.l2_loss(gen_images - images[:,1:])
    #loss += tf.reduce_sum(tf.abs(gen_images - images[:,1:]))
    correct_prediction = tf.equal(tf.argmax(y_pred, 1),tf.argmax(onehot_labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    #topk = tf.nn.top_k(y_pred, k = 5)
    y_max = tf.argmax(onehot_labels,axis=1)
    boo = tf.nn.in_top_k(logits, y_max,  k = 5)
    accuracy_top5 = tf.reduce_mean(tf.cast(boo, "float"), name = "top5_accuracy")
    return loss, accuracy, y_pred, onehot_labels, accuracy_top5

