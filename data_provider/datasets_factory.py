from data_provider import mnist, taxibj
from data_provider import kth_action
from data_provider import ucf101, ucf101_own, ucf101_own1

### four different type of datasets for video prediction
datasets_map = {
    'mnist': mnist,
    'taxibj': taxibj,
    'action': kth_action,
    'ucf101': ucf101,
    'ucf101_own': ucf101_own,
    'ucf101_own1': ucf101_own1
}


def data_provider(dataset_name, train_data_paths, valid_data_paths, batch_size,
                  img_width, seq_length, is_training=True):
    '''Given a dataset name and returns a Dataset.
    Args:
        dataset_name: String, the name of the dataset.
        train_data_paths: List, [train_data_path1, train_data_path2...]
        valid_data_paths: List, [val_data_path1, val_data_path2...]
        batch_size: Int
        img_width: Int
        is_training: Bool
    Returns:
        if is_training:
            Two dataset instances for both training and evaluation.
        else:
            One dataset instance for evaluation.
    Raises:
        ValueError: If `dataset_name` is unknown.
    '''
    if dataset_name not in datasets_map:
        raise ValueError('Name of dataset unknown %s' % dataset_name)
    train_data_list = train_data_paths.split(',')
    valid_data_list = valid_data_paths.split(',')
    if dataset_name == 'mnist':
        test_input_param = {'paths': valid_data_list,
                            'minibatch_size': batch_size,
                            'input_data_type': 'float32',
                            'is_output_sequence': True,
                            'name': dataset_name + 'test iterator'}
        test_input_handle = datasets_map[dataset_name].InputHandle(test_input_param)
        test_input_handle.begin(do_shuffle=False)
        if is_training:
            train_input_param = {'paths': train_data_list,
                                 'minibatch_size': batch_size,
                                 'input_data_type': 'float32',
                                 'is_output_sequence': True,
                                 'name': dataset_name + ' train iterator'}
            train_input_handle = datasets_map[dataset_name].InputHandle(train_input_param)
            train_input_handle.begin(do_shuffle=True)
            return train_input_handle, test_input_handle
        else:
            return test_input_handle

    if dataset_name == 'taxibj':
        input_param = {'paths': valid_data_list,
                       'image_width': img_width,
                       'minibatch_size': batch_size,
                       'seq_length': seq_length,
                       'input_data_type': 'float32',
                       'name': dataset_name + ' iterator'}
        input_handle = datasets_map[dataset_name].DataProcess(input_param)
        if is_training:
            train_input_handle = input_handle.get_train_input_handle()
            train_input_handle.begin(do_shuffle=True)
            test_input_handle = input_handle.get_test_input_handle()
            test_input_handle.begin(do_shuffle=False)
            return train_input_handle, test_input_handle
        else:
            test_input_handle = input_handle.get_test_input_handle()
            test_input_handle.begin(do_shuffle=False)
            return test_input_handle

    if dataset_name == 'action':
        input_param = {
            'paths': valid_data_list,
            'image_width': img_width,
            'minibatch_size': batch_size,
            'seq_length': seq_length,
            'input_data_type': 'float32',
            'name': dataset_name + ' iterator'
        }
        input_handle = datasets_map[dataset_name].DataProcess(input_param)
        if is_training:
            train_input_handle = input_handle.get_train_input_handle()
            train_input_handle.begin(do_shuffle=True)
            test_input_handle = input_handle.get_test_input_handle()
            test_input_handle.begin(do_shuffle=False)
            return train_input_handle, test_input_handle
        else:
            test_input_handle = input_handle.get_test_input_handle()
            test_input_handle.begin(do_shuffle=False)
            return test_input_handle

    if dataset_name == 'ucf101':
        test_input_param = {'paths': valid_data_list,
                       'image_width': img_width,
                       'minibatch_size': batch_size,
                       'seq_length': seq_length,
                       'channel': 3,
                       'input_data_type': 'float32',
                       'name': 'ucf101'}
        #test_input_handle = datasets_map[dataset_name].InputHandle(test_input_param)
        input_handle = datasets_map[dataset_name].DataProcess(test_input_param)
        test_input_handle = input_handle.get_test_input_handle()
        test_input_handle.begin(do_shuffle=False)
        if is_training:
            train_input_param = {'paths': train_data_list,
                       'image_width': img_width,
                       'minibatch_size': batch_size,
                       'seq_length': seq_length,
                       'channel': 3,
                       'input_data_type': 'float32',
                       'name': 'ucf101'}
            input_handle1 = datasets_map[dataset_name].DataProcess(train_input_param)
            train_input_handle = input_handle1.get_train_input_handle()
            train_input_handle.begin(do_shuffle=True)
            return train_input_handle, test_input_handle
        else:
            return test_input_handle

    if dataset_name == 'ucf101_own':
        input_param = {
            'paths': valid_data_list,
            'image_width': img_width,
            'minibatch_size': batch_size,
            'channel': 3,
            'seq_length': seq_length,
            'input_data_type': 'float32',
            #'name': dataset_name + ' iterator'
            'name': 'ucf101_own'}
        input_handle = datasets_map[dataset_name].DataProcess(input_param)
        if is_training:
            train_input_handle = input_handle.get_train_input_handle()
            train_input_handle.begin(do_shuffle=True)
            test_input_handle = input_handle.get_test_input_handle()
            test_input_handle.begin(do_shuffle=False)
            return train_input_handle, test_input_handle
        else:
            test_input_handle = input_handle.get_test_input_handle()
            test_input_handle.begin(do_shuffle=False)
            return test_input_handle

    if dataset_name == 'ucf101_own1':
        input_param = {
            'paths': valid_data_list,
            'image_width': img_width,
            'minibatch_size': batch_size,
            'channel': 3,
            'seq_length': seq_length,
            'input_data_type': 'float32',
            #'name': dataset_name + ' iterator'
            'name': 'ucf101_own1'}
        input_handle = datasets_map[dataset_name].DataProcess(input_param)
        if is_training:
            train_input_handle = input_handle.get_train_input_handle()
            train_input_handle.begin(do_shuffle=True)
            test_input_handle = input_handle.get_test_input_handle()
            test_input_handle.begin(do_shuffle=False)
            return train_input_handle, test_input_handle
        else:
            test_input_handle = input_handle.get_test_input_handle()
            test_input_handle.begin(do_shuffle=False)
            return test_input_handle