__author__ = 'liuyu'
import numpy as np
import os
import glob
import cv2
from PIL import Image
import logging
import random
import tensorflow as tf

logger = logging.getLogger(__name__)

class InputHandle:
    def __init__(self, datas, indices, input_param):
        self.name = input_param['name']
        self.input_data_type = input_param.get('input_data_type', 'float32')
        self.minibatch_size = input_param['minibatch_size']
        self.image_width = input_param['image_width']
        self.channel = input_param['channel']
        self.datas = datas
        self.indices = indices
        self.current_position = 0
        self.current_batch_indices = []
        self.current_input_length = input_param['seq_length']
        self.interval = 1

    def total(self):
        return len(self.indices)

    def begin(self, do_shuffle=True):
        logger.info("Initialization for read data ")
        if do_shuffle:
            random.shuffle(self.indices)
        self.current_position = 0
        self.current_batch_indices = self.indices[self.current_position:self.current_position + self.minibatch_size]

    def next(self):
        self.current_position += self.minibatch_size
        if self.no_batch_left():
            return None
        self.current_batch_indices = self.indices[self.current_position:self.current_position + self.minibatch_size]

    def no_batch_left(self):
        if self.current_position + self.minibatch_size > self.total():
            return True
        else:
            return False

    def get_batch(self):
        if self.no_batch_left():
            logger.error(
                "There is no batch left in " + self.name + ". Consider to user iterators.begin() to rescan from the beginning of the iterators")
            return None
        input_batch = np.zeros(
            (self.minibatch_size, self.current_input_length, self.image_width, self.image_width, self.channel)).astype(
            self.input_data_type)
        for i in range(self.minibatch_size):
            batch_ind = self.current_batch_indices[i]
            begin = batch_ind
            end = begin + self.current_input_length * self.interval
            data_slice = self.datas[begin:end:self.interval]
            input_batch[i, :self.current_input_length, :, :, :] = data_slice
            # logger.info('data_slice shape')
            # logger.info(data_slice.shape)
            # logger.info(input_batch.shape)
        input_batch = input_batch.astype(self.input_data_type)
        return input_batch

    def print_stat(self):
        logger.info("Iterator Name: " + self.name)
        logger.info("    current_position: " + str(self.current_position))
        logger.info("    Minibatch Size: " + str(self.minibatch_size))
        logger.info("    total Size: " + str(self.total()))
        logger.info("    current_input_length: " + str(self.current_input_length))
        logger.info("    Input Data Type: " + str(self.input_data_type))

class DataProcess:
    def __init__(self, input_param):
        self.input_param = input_param
        self.paths = input_param['paths']
        self.image_width = input_param['image_width']
        self.train_person = ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16',
                             '17','18','19','20','21','22','23','24','25']
        self.test_person = ['01']
        self.seq_len = input_param['seq_length']

    def load_data(self, paths, mode='train'):
        """Loads the dataset.

        Args:
          paths: List of action_path.
          mode: Training or testing.

        Returns:
          A dataset and indices of the sequence.
        """
        path = paths[0]
        if mode == 'train':
            person_id = self.train_person
        elif mode == 'test':
            person_id = self.test_person
        else:
            print('ERROR!')
        print('begin load data' + str(path))

        frames_np = []
        frames_file_name = []
        frames_person_mark = []
        frames_category = []
        person_mark = 0

        #c_dir_list = os.listdir(path)
        #c_dir_list.sort()

        classes = ['Basketball', 'Biking', 'Diving', 'GolfSwing', 'HorseRiding', 'SoccerJuggling',
                   'Swing', 'TennisSwing', 'TrampolineJumping', 'VolleyballSpiking', 'WalkingWithDog']
        for c_dir in classes:  # ApplyEyeMakeup
            c_dir_path = os.path.join(path, c_dir)   # train/ApplyEyeMakeup
            p_c_dir_list = os.listdir(c_dir_path)
            p_c_dir_list.sort() # for date seq

            for p_c_dir in p_c_dir_list:  # person01_handwaving_d1_uncomp  v_ApplyEyeMakeup_g01_c01
                print('-------p_c_dir-----', p_c_dir)
                if p_c_dir[-6:-4] not in person_id:
                    print(p_c_dir[-6:-4])
                    continue
                person_mark += 1

                dir_path = os.path.join(c_dir_path, p_c_dir)
                filelist = os.listdir(dir_path)
                filelist.sort()  # tocheck
                for cur_file in filelist:  # image_0257   v_ApplyEyeMakeup_g01_c01-0001.jpg
                    if not cur_file.startswith('v'):
                        continue

                    image = cv2.cvtColor(cv2.imread(os.path.join(dir_path, cur_file)), cv2.COLOR_BGR2RGB)
                    # [1000,1000,3]
                    image = image[image.shape[0] // 4:-image.shape[0] // 4, image.shape[1] // 4:-image.shape[1] // 4, :]
                    if self.image_width != image.shape[0]:
                        image = cv2.resize(image, (self.image_width, self.image_width))
                    # image = cv2.resize(image[100:-100,100:-100,:], (self.image_width, self.image_width),
                    #                   interpolation=cv2.INTER_LINEAR)
                    frames_np.append(np.array(image, dtype=np.float32) / 255.0)
                    #frame_np = frame_np[:, :, 0]  #
                    #frames_np.append(frame_np)
                    frames_file_name.append(cur_file)
                    frames_person_mark.append(person_mark)
                    #frames_category.append(frame_category_flag)
        # is it a begin index of sequence
        indices = []
        index = len(frames_person_mark) - 1
        while index >= self.seq_len - 1:
            if frames_person_mark[index] == frames_person_mark[index - self.seq_len + 1]:
                end = int(frames_file_name[index][-8:-4])
                print('end: ', end)
                start = int(frames_file_name[index - self.seq_len + 1][-8:-4])
                # TODO(yunbo): mode == 'test'
                if end - start == self.seq_len - 1:
                    indices.append(index - self.seq_len + 1)
                    #index -= self.seq_len - 1
                    index -= 50
            index -= 1

        #data = frames_np
        data = np.asarray(frames_np)
        print('there are ' + str(data.shape[0]) + ' pictures')
        print('there are ' + str(len(indices)) + ' sequences')
        return data, indices

    def get_train_input_handle(self):
        train_data, train_indices = self.load_data(self.paths, mode='train')
        return InputHandle(train_data, train_indices, self.input_param)

    def get_test_input_handle(self):
        test_data, test_indices = self.load_data(self.paths, mode='test')
        return InputHandle(test_data, test_indices, self.input_param)

