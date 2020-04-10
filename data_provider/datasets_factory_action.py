from data_provider import ucf11_action

datasets_map = {
    'ucf11_action': ucf11_action
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

    if dataset_name == 'ucf11_action':
        input_param = {
            'paths': valid_data_list,
            'image_width': img_width,
            'minibatch_size': batch_size,
            'channel': 3,
            'seq_length': seq_length,
            'input_data_type': 'float32',
            #'name': dataset_name + ' iterator'
            'name': 'ucf11_action'}
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