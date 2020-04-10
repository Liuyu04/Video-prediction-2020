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
        self.interval = 2

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
            #end = begin + self.current_input_length * self.interval
            #data_slice = self.datas[begin:end:self.interval]
            #input_batch[i, :self.current_input_length, :, :, :] = data_slice
            end = begin + self.current_input_length
            data_slice = self.datas[begin:end, :, :, :]
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
        self.seq_len = input_param['seq_length']

    def load_data(self, paths, mode='train'):
        data_dir = paths[0]
        frames_np = []
        _path = data_dir
        print ('load data...', _path)
        c_dir_list = os.listdir(_path)
        print(c_dir_list[0])
        c_dir_list.sort()
        print(c_dir_list[0])
        length = 0
        frames_file_name = []
        for c_dir in c_dir_list: ## ApplyEyeMakeup
            c_dir_path = os.path.join(_path, c_dir)  ### five-video-classification-methods-master/data/train/ApplyEyeMakeup
            #filenames = os.listdir(c_dir_path)
            filenames = glob.glob(os.path.join(c_dir_path, '*.jpg'))
            filenames.sort()
            print ('file size ', len(filenames))
            length += len(filenames)
            for filename in filenames:  ##v_ApplyEyeMakeup_g08_c01-0001.jpg
                file_path = os.path.join(c_dir_path, filename)
                #image = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)
                #[1000,1000,3]
                image = cv2.imread(file_path)   ###[240,320,3]
                image = image[image.shape[0]//4:-image.shape[0]//4, image.shape[1]//4:-image.shape[1]//4, :]
                if self.image_width != image.shape[0]:
                    image = cv2.resize(image, (self.image_width, self.image_width))
                #image = cv2.resize(image[100:-100,100:-100,:], (self.image_width, self.image_width),
                #                   interpolation=cv2.INTER_LINEAR)
                frames_np.append(np.array(image, dtype=np.float32) / 255.0)
                frames_file_name.append(filename)
#               if len(frames_np) % 100 == 0: print len(frames_np)
                #if len(frames_np) % 1000 == 0: break
        print('data size ', length)
        print('frames_file_name ', len(frames_file_name))
        # is it a begin index of sequence
        indices = []
        index = 0
        print ('gen index')
        while index + self.seq_len - 1 < len(frames_file_name):
            # 'S11_Discussion_1.54138969_000471.jpg'
            # ['S11_Discussion_1', '54138969_000471', 'jpg']
            start_infos = frames_file_name[index].split('.')
            #start_infos = frames_file_name[index].split('_')
            ## 'v_ApplyEyeMakeup_g08_c01-0001.jpg'
            ## ['v_ApplyEyeMakeup_g08_c01-0001', 'jpg']
            ## ['v' 'ApplyEyeMakeup' 'g08' 'c01-0001.jpg']
            end_infos = frames_file_name[index+(self.seq_len-1)].split('.')
            #end_infos = frames_file_name[index + (self.seq_len - 1)].split('_')
            if start_infos[0]!= end_infos[0]:
                index += 1
                continue
            #start_video_id, start_frame = start_infos[3].split('-')
            #start_frame_id = start_frame.split('.')
            #end_video_id, end_frame = end_infos[3].split('-')
            #end_frame_id = end_frame.split('.')
            #if start_video_id != end_video_id:
                ##index += 1
                 #continue
            #if int(end_frame_id[0]) - int(start_frame_id[0]) == 5 * (self.seq_len - 1):
                #   indices.append(index)
            indices.append(index)
            if mode == 'train':
                index += 10
            elif mode == 'test':
                index += 10
        print("there are " + str(len(indices)) + " sequences")
        # data = np.asarray(frames_np)
        data = frames_np
        print("there are " + str(len(data)) + " pictures")
        return data, indices

    def get_train_input_handle(self):
        train_data, train_indices = self.load_data(self.paths, mode='train')
        return InputHandle(train_data, train_indices, self.input_param)

    def get_test_input_handle(self):
        test_data, test_indices = self.load_data(self.paths, mode='test')
        return InputHandle(test_data, test_indices, self.input_param)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=str)
    parser.add_argument("output_dir", type=str)
    args = parser.parse_args()

    partition_names = ['train', 'test']
    partition_fnames = partition_data(args.input_dir)


if __name__ == '__main__':
    main()
