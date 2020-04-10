import tensorflow as tf

from nets import ST_DLSTM_action, ST_DLSTM_my_action

networks_map = {'st_dlstm_action': ST_DLSTM_action.rnn,
                'st_dlstm_my_action': ST_DLSTM_my_action.rnn
               }

def construct_model(name, images, targets, num_layers, num_hidden,
                    filter_size, stride, classes, seq_length, input_length, tln):
    '''Returns a sequence of generated frames
    Args:
        name: [predrnn_pp]
        mask_true: for schedualed sampling.
        num_hidden: number of units in a lstm layer.
        filter_size: for convolutions inside lstm.
        stride: for convolutions inside lstm.
        seq_length: including ins and outs.
        input_length: for inputs.
        tln: whether to apply tensor layer normalization.
    Returns:
        gen_images: a seq of frames.
        loss: [l2 / l1+l2].
    Raises:
        ValueError: If network `name` is not recognized.
    '''
    if name not in networks_map:
        raise ValueError('Name of network unknown %s' % name)
    func = networks_map[name]
    return func(images, targets, num_layers, num_hidden, filter_size,
                stride, classes, seq_length, input_length, tln)
