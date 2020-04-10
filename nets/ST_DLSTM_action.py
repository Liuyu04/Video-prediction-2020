__author__ = 'liuyu'

import tensorflow as tf
from layers.GradientHighwayUnit import GHU as ghu
from layers.STDLSTMCell import STDLSTMCell as stdlstm
from tensorflow.contrib import rnn

def rnn(images, targets, num_layers, num_hidden, filter_size, stride=1,classes=11,
        seq_length=16, input_length=16, tln=True):

#### targets: integer in 0-(classes-1) ---- 0-10
    gen_images = []
    lstm = []
    cell = []
    hidden = []
    #shape = images.get_shape().as_list()
    shape = [32,16,1,1,512]
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


    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=256, forget_bias=1.0, state_is_tuple=True)
    lstm_cell = tf.nn.rnn_cell.DropoutWrapper(cell=lstm_cell, input_keep_prob=1.0, output_keep_prob=0.5)
    mlstm_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * 1, state_is_tuple=True)
    init_state = mlstm_cell.zero_state(shape[0], dtype=tf.float32)
    state = init_state

    gradient_highway = ghu('highway', filter_size, num_hidden[0], tln=tln)

    mem = None
    z_t = None
    outputs = []

    for t in xrange(seq_length):
        reuse = bool(outputs)
        #reuse = True
        with tf.variable_scope('predrnn_pp', reuse=reuse):
            inputs = images[:,t]
            #if t < input_length:
                #inputs = images[:,t]
            #else:
                #inputs = mask_true[:,t-input_length]*images[:,t] + (1-mask_true[:,t-input_length])*x_gen

            conv1_1 = tf.layers.conv2d(inputs, filters=32, kernel_size=7, strides=2, padding='same',kernel_initializer=tf.glorot_uniform_initializer())  # 40,40,32
            conv1_1 = tf.nn.relu(conv1_1)
            conv1_2 = tf.layers.conv2d(conv1_1, filters=32, kernel_size=3, strides=1,kernel_initializer=tf.contrib.layers.xavier_initializer())  # 38,38,32
            conv1_2 = tf.nn.relu(conv1_2)
            p1 = tf.nn.max_pool(conv1_2, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID')  # 19,19,32
            conv2_1 = tf.layers.conv2d(p1, filters=64, kernel_size=3, strides=1, padding='same',kernel_initializer=tf.glorot_uniform_initializer())  # 19,19,64
            conv2_1 = tf.nn.relu(conv2_1)
            conv2_2 = tf.layers.conv2d(conv2_1, filters=64, kernel_size=3, strides=1, padding='same',kernel_initializer=tf.glorot_uniform_initializer())  # 19,19,64
            conv2_2 = tf.nn.relu(conv2_2)
            p2 = tf.nn.max_pool(conv2_2, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID')  # 9,9,64
            conv3_1 = tf.layers.conv2d(p2, filters=128, kernel_size=3, strides=1, padding='same',kernel_initializer=tf.glorot_uniform_initializer())  # 9,9,128
            conv3_1 = tf.nn.relu(conv3_1)
            conv3_2 = tf.layers.conv2d(conv3_1, filters=128, kernel_size=3, strides=1, padding='same',kernel_initializer=tf.glorot_uniform_initializer())  # 9,9,128
            conv3_2 = tf.nn.relu(conv3_2)
            p3 = tf.nn.max_pool(conv3_2, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID')  # 4,4,128
            conv4_1 = tf.layers.conv2d(p3, filters=256, kernel_size=3, strides=1, padding='same',kernel_initializer=tf.glorot_uniform_initializer())  # 4,4,256
            conv4_1 = tf.nn.relu(conv4_1)
            conv4_2 = tf.layers.conv2d(conv4_1, filters=256, kernel_size=3, strides=1, padding='same',kernel_initializer=tf.glorot_uniform_initializer())  # 4,4,256
            conv4_2 = tf.nn.relu(conv4_2)
            p4 = tf.nn.max_pool(conv4_2, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID')  # 2,2,256
            conv5_1 = tf.layers.conv2d(p4, filters=512, kernel_size=3, strides=1, padding='same',kernel_initializer=tf.glorot_uniform_initializer())  # 2,2,512
            conv5_1 = tf.nn.relu(conv5_1)
            conv5_2 = tf.layers.conv2d(conv5_1, filters=512, kernel_size=3, strides=1, padding='same',kernel_initializer=tf.glorot_uniform_initializer())  # 2,2,512
            conv5_2 = tf.nn.relu(conv5_2)
            p5 = tf.nn.max_pool(conv5_2, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID')  # 1,1,512
            print('p5 shape: ', p5.shape)

            p5 = tf.nn.dropout(p5, 0.5)
            ###LSTM
            mid = tf.layers.flatten(p5)   # N,512

            (cell_output, state) = mlstm_cell(mid, state)
            outputs.append(cell_output)
            print('outputs shape: ---- ', outputs[-1].shape)

            #hidden[0], cell[0], mem = lstm[0](p5, hidden[0], cell[0], mem)
           # z_t = gradient_highway(hidden[0], z_t)
            #hidden[1], cell[1], mem = lstm[1](z_t, hidden[1], cell[1], mem)

            #for i in xrange(2, num_layers):
               # hidden[i], cell[i], mem = lstm[i](hidden[i-1], hidden[i], cell[i], mem)
            ### hidden (N,10,10,64)
            #print('hidden shape: ', hidden[0].shape)

            #x_gen = tf.reduce_mean(hidden[0], [1, 2], name='global_average_pool', keep_dims=True)
            #print('x_gen: ', x_gen.shape)

            #x_gen = tf.layers.conv2d(inputs=hidden[num_layers-1],filters=64,kernel_size=3,strides=1,padding='same', name="back_to_pixel")  # 10,10,64
            #x_gen = tf.nn.relu(x_gen)
            #p1 = tf.nn.max_pool(x_gen, [1,2,2,1],[1,2,2,1],padding='VALID')  # 5,5,64
            #c2 = tf.layers.conv2d(p1,filters=128,kernel_size=3,strides=1,padding='same') # 5,5,128
            #c2 = tf.nn.relu(c2)
            #p2 = tf.nn.max_pool(c2, [1,2,2,1],[1,2,2,1],padding='VALID')  # 2,2,128
            #gen_images.append(hidden[0])

    #gen_images = tf.stack(gen_images)
    #print('gen_image shape: ', gen_images.shape)
    # [batch_size, seq_length, height, width, channels]
    #gen_images = tf.transpose(gen_images, [1,0,2,3,4])
    #gen_images = tf.layers.flatten(gen_images)
    h_state = outputs[-1]
    logits = tf.layers.dense(inputs=h_state,units=classes,activation=None)
    y_pred=tf.nn.softmax(logits)
    yy = tf.argmax(y_pred,axis=1)

    onehot_labels = tf.one_hot(targets, classes)
    #loss = -tf.reduce_mean(onehot_labels * tf.log(y_pred))
    #loss = tf.nn.softmax_cross_entropy_with_logits(labels=onehot_labels,logits=y_pred,name='xentropy')
    #loss = tf.reduce_mean(loss, name='xentropy_mean')
    #loss = tf.reduce_sum(loss)
    loss = tf.reduce_mean(-tf.reduce_sum(onehot_labels * tf.log(y_pred),reduction_indices=[1]))
    #loss = tf.nn.softmax_cross_entropy_with_logits(y_pred, targets)
    #loss = tf.nn.l2_loss(gen_images - images[:,1:])
    #loss += tf.reduce_sum(tf.abs(gen_images - images[:,1:]))

    correct_prediction = tf.equal(tf.argmax(y_pred, 1),tf.argmax(onehot_labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    #_ , accuracy = tf.metrics.accuracy(labels=onehot_labels, predictions=y_pred)
    return loss, accuracy, yy, onehot_labels

