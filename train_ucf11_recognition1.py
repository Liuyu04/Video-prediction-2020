__author__ = 'liuyu'  ### check

import os.path
import time
import numpy as np
import tensorflow as tf
import cv2
import sys
import random
from nets import models_factory_action
from data_provider import datasets_factory_action
from utils import preprocess
from utils import metrics
from skimage.measure import compare_ssim
import tensorflow.contrib.slim as slim

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

# -----------------------------------------------------------------------------
FLAGS = tf.app.flags.FLAGS

# data I/O
tf.app.flags.DEFINE_string('dataset_name', 'ucf11_action',
                           'The name of dataset.')
tf.app.flags.DEFINE_string('train_data_paths',
                           'data/ucf101/ucf11/',
                           'train data paths.')
tf.app.flags.DEFINE_string('valid_data_paths',
                           'data/ucf101/ucf11/',
                           'validation data paths.')
tf.app.flags.DEFINE_string('save_dir', 'checkpoints_ucf11_action/ucf11_action_stdlstm_v6',
                           'dir to store trained net.')
tf.app.flags.DEFINE_string('gen_frm_dir', 'results_ucf11_action/ucf11_action_stdlstm_v6',
                           'dir to store result.')
# model
tf.app.flags.DEFINE_string('model_name', 'st_dlstm_my_action',
                           'The name of the architecture.')
tf.app.flags.DEFINE_string('pretrained_model', 'checkpoints_ucf101/ucf101_stdlstm_v2/model.ckpt-80000',
                           'file of a pretrained model to initialize from.')
tf.app.flags.DEFINE_integer('input_length', 16,
                            'encoder hidden states.')
tf.app.flags.DEFINE_integer('seq_length', 16,  # 20 for train
                            'total input and output length.')
tf.app.flags.DEFINE_integer('classes', 11,
                            'total classes.')
tf.app.flags.DEFINE_integer('img_width', 80,
                            'input image width.')
tf.app.flags.DEFINE_integer('img_channel', 3,
                            'number of image channel.')
tf.app.flags.DEFINE_integer('stride', 1,
                            'stride of a convlstm layer.')
tf.app.flags.DEFINE_integer('filter_size', 5,  #5
                            'filter of a convlstm layer.')
tf.app.flags.DEFINE_string('num_hidden', '128,64,64,64',
                           'COMMA separated number of units in a convlstm layer.')
tf.app.flags.DEFINE_integer('patch_size', 8, #8
                            'patch size on one dimension.')
tf.app.flags.DEFINE_boolean('layer_norm', True,
                            'whether to apply tensor layer norm.')
# optimization
tf.app.flags.DEFINE_float('lr', 1e-4,
                          'base learning rate.')
tf.app.flags.DEFINE_boolean('reverse_input', True,
                            'whether to reverse the input frames while training.')
tf.app.flags.DEFINE_integer('batch_size', 32,
                            'batch size for training.')
# tf.app.flags.DEFINE_integer('batch_size', 128,
#                            'batch size for training.')
tf.app.flags.DEFINE_integer('max_iterations', 80,  # 80000
                            'max num of steps.')
# tf.app.flags.DEFINE_integer('max_iterations', 4,
#                            'max num of steps.')
tf.app.flags.DEFINE_integer('display_interval', 1,
                            'number of iters showing training loss.')
tf.app.flags.DEFINE_integer('test_interval', 1,  # 2000
                            'number of iters for test.')
# tf.app.flags.DEFINE_integer('test_interval', 2,
#                            'number of iters for test.')
tf.app.flags.DEFINE_integer('snapshot_interval', 1,
                            'number of iters saving models.')


# tf.app.flags.DEFINE_integer('snapshot_interval', 5,
#                            'number of iters saving models.')
include = ['predrnn_pp/lstm_4/c2mb', 'predrnn_pp/lstm_2/x2cs', 'predrnn_pp/lstm_3/conv_spatial/kernel',
                       'predrnn_pp/lstm_4/c2ms',
                       'predrnn_pp/lstm_2/x2cb', 'predrnn_pp/lstm_2/conv_distance/bias', 'predrnn_pp/lstm_2/c2cb',
                       'predrnn_pp/lstm_1/h2cb',
                       'predrnn_pp/lstm_1/h2cs', 'predrnn_pp/lstm_1/m2mb',
                       'predrnn_pp/lstm_4/spatial_memory_transition/kernel',
                       'predrnn_pp/lstm_3/temporal_memory_transition/bias', 'predrnn_pp/lstm_4/conv_spatial/bias',
                       'predrnn_pp/lstm_3/memory_reduce/bias', 'predrnn_pp/lstm_2/conv_spatial/kernel',
                       'predrnn_pp/lstm_1/input_to_state/kernel',
                       'predrnn_pp/lstm_2/conv_distance/kernel', 'predrnn_pp/lstm_4/m2ms',
                       'predrnn_pp/lstm_1/conv_distance/bias',
                       'predrnn_pp/lstm_4/m2mb', 'predrnn_pp/lstm_2/c2cs',
                       'predrnn_pp/lstm_2/temporal_memory_transition/kernel',
                       'predrnn_pp/lstm_2/conv_spatial/bias', 'predrnn_pp/lstm_3/x2cs',
                       'predrnn_pp/highway/state_to_state/bias',
                       'predrnn_pp/lstm_3/x2cb', 'predrnn_pp/lstm_3/temporal_state_transition/kernel',
                       'predrnn_pp/lstm_3/temporal_memory_transition/kernel',
                       'predrnn_pp/lstm_4/c2cs', 'predrnn_pp/lstm_1/c2mb', 'predrnn_pp/lstm_1/c2ms',
                       'predrnn_pp/lstm_4/c2cb',
                       'predrnn_pp/lstm_2/c_cs', 'predrnn_pp/lstm_3/m2mb',
                       'predrnn_pp/lstm_1/temporal_state_transition/bias',
                       'predrnn_pp/lstm_4/c_cb', 'predrnn_pp/lstm_4/c_cs', 'predrnn_pp/lstm_4/memory_reduce/bias',
                       'predrnn_pp/lstm_3/conv_spatial/bias', 'predrnn_pp/highway/input_to_stateb',
                       'predrnn_pp/lstm_4/conv_distance/bias',
                       'predrnn_pp/highway/input_to_states', 'predrnn_pp/lstm_4/spatial_memory_transition/bias',
                       'predrnn_pp/lstm_1/m2ms', 'predrnn_pp/lstm_4/temporal_memory_transition/bias',
                       'predrnn_pp/lstm_3/m_to_o/kernel',
                       'predrnn_pp/lstm_1/spatial_memory_transition/bias', 'predrnn_pp/lstm_1/x2cs',
                       'predrnn_pp/lstm_2/temporal_state_transition/kernel', 'predrnn_pp/lstm_4/c2m/kernel',
                       'predrnn_pp/lstm_1/x2cb', 'predrnn_pp/lstm_1/m_mb', 'predrnn_pp/lstm_2/m2os',
                       'predrnn_pp/lstm_1/m_ms',
                       'predrnn_pp/lstm_1/temporal_state_transition/kernel', 'predrnn_pp/lstm_3/memory_reduce/kernel',
                       'predrnn_pp/lstm_2/m2ob', 'predrnn_pp/lstm_1/m_to_o/kernel',
                       'predrnn_pp/lstm_1/conv_distance/kernel',
                       'predrnn_pp/lstm_3/c2mb', 'predrnn_pp/lstm_4/input_to_state/kernel', 'predrnn_pp/lstm_3/c2ms',
                       'predrnn_pp/lstm_1/c2m/bias', 'predrnn_pp/lstm_2/temporal_memory_transition/bias',
                       'predrnn_pp/lstm_2/m_ms', 'predrnn_pp/lstm_2/m_mb,predrnn_pp/lstm_1/c_cs',
                       'predrnn_pp/lstm_1/temporal_memory_transition/bias', 'predrnn_pp/lstm_1/c_cb',
                       'predrnn_pp/lstm_1/c2m/kernel', 'predrnn_pp/lstm_2/memory_reduce/kernel',
                       'predrnn_pp/highway/input_to_state/kernel', 'predrnn_pp/highway/input_to_state/bias',
                       'predrnn_pp/lstm_4/conv_distance/kernel', 'predrnn_pp/lstm_2/memory_reduce/bias',
                       'predrnn_pp/lstm_3/c2m/kernel', 'predrnn_pp/lstm_3/conv_distance/kernel',
                       'predrnn_pp/lstm_3/m2ms', 'predrnn_pp/lstm_1/conv_spatial/bias',
                       'predrnn_pp/lstm_3/spatial_memory_transition/bias',
                       'predrnn_pp/lstm_2/spatial_memory_transition/kernel',
                       'predrnn_pp/lstm_2/c2m/bias,predrnn_pp/lstm_3/h2cb', 'predrnn_pp/lstm_3/conv_distance/bias',
                       'predrnn_pp/lstm_3/h2cs', 'predrnn_pp/lstm_1/input_to_state/bias',
                       'predrnn_pp/lstm_4/temporal_state_transition/bias',
                       'predrnn_pp/lstm_4/m2os', 'predrnn_pp/lstm_4/m2ob', 'predrnn_pp/lstm_2/m_to_o/kernel',
                       'predrnn_pp/lstm_3/c_cs', 'predrnn_pp/lstm_3/c_cb', 'predrnn_pp/lstm_3/c2m/bias',
                       'predrnn_pp/lstm_2/temporal_state_transition/bias', 'predrnn_pp/lstm_4/conv_spatial/kernel',
                       'predrnn_pp/lstm_4/m_to_o/kernel', 'predrnn_pp/lstm_4/input_to_state/bias',
                       'predrnn_pp/lstm_1/memory_reduce/bias', 'predrnn_pp/lstm_1/temporal_memory_transition/kernel',
                       'predrnn_pp/lstm_4/h2cs', 'predrnn_pp/lstm_3/c2cs', 'predrnn_pp/lstm_4/h2cb',
                       'predrnn_pp/lstm_3/c2cb',
                       'predrnn_pp/highway/state_to_stateb', 'predrnn_pp/lstm_2/spatial_memory_transition/bias',
                       'predrnn_pp/highway/state_to_states', 'predrnn_pp/lstm_3/temporal_state_transition/bias',
                       'predrnn_pp/lstm_3/input_to_state/kernel', 'predrnn_pp/lstm_4/temporal_memory_transition/kernel',
                       'predrnn_pp/lstm_2/input_to_state/bias', 'predrnn_pp/lstm_2/input_to_state/kernel',
                       'predrnn_pp/lstm_1/m2os', 'predrnn_pp/lstm_3/spatial_memory_transition/kernel',
                       'predrnn_pp/lstm_1/m2ob', 'predrnn_pp/lstm_2/m2ms', 'predrnn_pp/lstm_4/m_mb',
                       'predrnn_pp/lstm_3/input_to_state/bias', 'predrnn_pp/lstm_2/m_to_o/bias',
                       'predrnn_pp/lstm_3/m2ob',
                       'predrnn_pp/lstm_4/m_to_o/bias', 'predrnn_pp/lstm_3/m2os', ' predrnn_pp/lstm_2/c2m/kernel',
                       'predrnn_pp/lstm_3/m_to_o/bias', 'predrnn_pp/lstm_4/temporal_state_transition/kernel',
                       'predrnn_pp/lstm_1/conv_spatial/kernel', 'predrnn_pp/lstm_4/c2m/bias',
                       'predrnn_pp/lstm_4/memory_reduce/kernel',
                       'predrnn_pp/lstm_2/c_cb', 'predrnn_pp/lstm_1/c2cs', 'predrnn_pp/lstm_1/c2cb',
                       'predrnn_pp/lstm_3/m_mb',
                       'predrnn_pp/lstm_1/spatial_memory_transition/kernel', 'predrnn_pp/lstm_2/h2cb',
                       'predrnn_pp/lstm_3/m_ms',
                       'predrnn_pp/lstm_2/h2cs', 'predrnn_pp/lstm_4/m_ms', 'predrnn_pp/lstm_4/x2cb',
                       'predrnn_pp/highway/state_to_state/kernel', 'predrnn_pp/lstm_2/m2mb', 'predrnn_pp/lstm_4/x2cs',
                       'predrnn_pp/lstm_1/m_to_o/bias', 'predrnn_pp/lstm_2/c2mb', 'predrnn_pp/lstm_2/c2ms',
                       'predrnn_pp/lstm_1/memory_reduce/kernel']

def get_variable_via_scope(scope_lst):
    vars = []
    for sc in scope_lst:
        sc_variable = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope=sc)
        vars.extend(sc_variable)
    return vars

class Model(object):
    def __init__(self):
        # inputs
        self.x = tf.placeholder(tf.float32,[FLAGS.batch_size,FLAGS.seq_length,
                                FLAGS.img_width / FLAGS.patch_size,
                                 FLAGS.img_width / FLAGS.patch_size,
                                FLAGS.patch_size * FLAGS.patch_size * FLAGS.img_channel])

        self.y = tf.placeholder(tf.int32,[FLAGS.batch_size])

        grads = []
        loss_train = []
        self.tf_lr = tf.placeholder(tf.float32, shape=[])
        num_hidden = [int(x) for x in FLAGS.num_hidden.split(',')]
        print(num_hidden)
        num_layers = len(num_hidden)
        with tf.variable_scope(tf.get_variable_scope()):
            # define a model
            loss, acc, pred, label, top5_acc = models_factory_action.construct_model(
                FLAGS.model_name, self.x,
                self.y,
                num_layers, num_hidden,
                FLAGS.filter_size, FLAGS.stride,FLAGS.classes,
                FLAGS.seq_length, FLAGS.input_length,
                FLAGS.layer_norm)

            #self.loss_train = loss / FLAGS.batch_size
            self.loss_train = loss
            self.accuracy = acc
            self.pred = pred
            self.label = label
            self.top5_acc = top5_acc
            # gradients
            all_params = tf.trainable_variables()
            no_change_scope = include
            no_change_vars = get_variable_via_scope(no_change_scope)
            for v in no_change_vars:
                all_params.remove(v)
            grads.append(tf.gradients(loss, all_params))

        self.train_op = tf.train.AdamOptimizer(FLAGS.lr).minimize(loss)
        #self.train_op = tf.train.GradientDescentOptimizer(FLAGS.lr).minimize(loss)
        # session
        variables = tf.global_variables()
        self.saver = tf.train.Saver(variables)
        init = tf.global_variables_initializer()
        configProt = tf.ConfigProto()
        configProt.gpu_options.allow_growth = True
        configProt.allow_soft_placement = True
        self.sess = tf.Session(config=configProt)
        self.sess.run(init)
        if FLAGS.pretrained_model:
            #exclude = ['back_to_pixel']
            variables_to_restore = slim.get_variables_to_restore(include=include)
            self.saver = tf.train.Saver(variables_to_restore)
            self.saver.restore(self.sess, FLAGS.pretrained_model)
            #self.saver.restore(self.sess, FLAGS.pretrained_model)

    def train(self, inputs, lr, targets):
        feed_dict = {self.x: inputs}
        feed_dict.update({self.tf_lr: lr})
        feed_dict.update({self.y: targets})
        loss, acc, pred,  label, top5_acc, _ = self.sess.run((self.loss_train, self.accuracy, self.pred, self.label, self.top5_acc, self.train_op), feed_dict)
        return loss, acc, pred, label, top5_acc

    def test(self, inputs, targets):
        feed_dict = {self.x: inputs}
        feed_dict.update({self.y: targets})
        loss, acc, pred, label, top5_acc = self.sess.run((self.loss_train, self.accuracy, self.pred, self.label, self.top5_acc),  feed_dict)
        return loss, acc, pred, label, top5_acc

    def save(self, itr):
        checkpoint_path = os.path.join(FLAGS.save_dir, 'model.ckpt')
        self.saver.save(self.sess, checkpoint_path, global_step=itr)
        print('saved to ' + FLAGS.save_dir)


def main(argv=None):
    if tf.gfile.Exists(FLAGS.save_dir):
        tf.gfile.DeleteRecursively(FLAGS.save_dir)  ## if file is not none, clean all recursively - note by liuyu
    tf.gfile.MakeDirs(FLAGS.save_dir)
    if tf.gfile.Exists(FLAGS.gen_frm_dir):
        tf.gfile.DeleteRecursively(FLAGS.gen_frm_dir)
    tf.gfile.MakeDirs(FLAGS.gen_frm_dir)

    # load data
    train_input_handle, test_input_handle = datasets_factory_action.data_provider(
        FLAGS.dataset_name, FLAGS.train_data_paths, FLAGS.valid_data_paths,
        FLAGS.batch_size, FLAGS.img_width, FLAGS.seq_length)

    f = open(FLAGS.save_dir + '/summary.txt', 'a')  #### add

    print("Initializing models")
    model = Model()

    ### total parameters
    total_parameters = 0
    for variable in tf.trainable_variables():
        variable_parameters = 1
        for dim in variable.get_shape():
            variable_parameters *= dim.value
        total_parameters += variable_parameters

    print("Total number of trainable parameters: %d" % total_parameters)
    f.write('\nTotal number of trainable parameters: %d' % total_parameters)

    lr = FLAGS.lr
    #acc_s = []
    #acc_s.append(0)
    for itr in xrange(1, FLAGS.max_iterations + 1):
        if train_input_handle.no_batch_left():
            train_input_handle.begin(do_shuffle=True)
        while (train_input_handle.no_batch_left() == False):
            ims, labels = train_input_handle.get_batch()
            ims = preprocess.reshape_patch(ims, FLAGS.patch_size)
            cost, accuracy, pred, label, top5_acc = model.train(ims, lr, labels)
            train_input_handle.next()
        print('img shape: ', ims.shape)
        #ims = preprocess.reshape_patch(ims, FLAGS.patch_size)
        #print('img reshape: ', ims.shape)
        print('learning rate: ', lr)
        for i in range(8):
            print('labels: ', labels[i])
            print('predictions: ', pred[i])
            #print('one hot label: ', label[i])

        if itr % FLAGS.display_interval == 0:
            print('itr: ' + str(itr))
            print('training loss: ' + str(cost))
            print('training accuracy: ' + str(accuracy))
            print('training top5 accuracy: ' + str(top5_acc))
            ####  add
            if itr % FLAGS.test_interval == 0:
                f.write('\nitr: %d' % itr)
                f.write('\ntraining loss: %f' % cost)
                f.write('\ntraining accuracy: %f' % accuracy)
                f.write('\ntraining top5 accuracy: %f' % top5_acc)

        if itr % FLAGS.test_interval == 0:
            print('test...')
            test_input_handle.begin(do_shuffle=False)
            res_path = os.path.join(FLAGS.gen_frm_dir, str(itr))
            os.mkdir(res_path)
            avg_ce = 0
            avg_acc = 0
            avg_top5_acc = 0
            batch_id = 0
            while (test_input_handle.no_batch_left() == False):
                batch_id = batch_id + 1
                test_ims, test_labels = test_input_handle.get_batch()
                test_dat = preprocess.reshape_patch(test_ims, FLAGS.patch_size)
                #test_dat = test_ims
                ce, acc, pred, label, top5_acc = model.test(test_dat, test_labels)
                avg_ce += ce
                avg_acc += acc
                avg_top5_acc += top5_acc
                # concat outputs of different gpus along batch
                test_input_handle.next()

                # save prediction examples
                #if batch_id <= 10:
                    #path = os.path.join(res_path, str(batch_id))
                    #os.mkdir(path)
                    #for i in xrange(FLAGS.seq_length):
                        #name = 'gt' + str(i + 1) + '.png'
                        #file_name = os.path.join(path, name)
                        #img_gt = np.uint8(test_ims[0, i, :, :, :] * 255)
                        #cv2.imwrite(file_name, img_gt)
                    #print('-----------prediction examples labels: ', test_labels[0])

            avg_ce = avg_ce / batch_id
            avg_acc = avg_acc / batch_id
            avg_top5_acc = avg_top5_acc / batch_id
            print('ce per seq: ' + str(avg_ce))
            f.write('\ntest ce: %f' % avg_ce)  #### add
            print('acc per seq: ' + str(avg_acc))
            f.write('\ntest acc: %f' % avg_acc)  #### add
            print('top5 acc per seq: ' + str(top5_acc))
            f.write('\ntest top5 acc: %f' % top5_acc)  #### add

            if avg_acc > 0.3:
                model.save(itr)
        #acc_s.append(avg_acc)
        #if itr % FLAGS.snapshot_interval == 0 and avg_acc > 0.3:
            #model.save(itr)

    f.close()


if __name__ == '__main__':
    tf.app.run()


