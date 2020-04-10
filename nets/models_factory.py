import tensorflow as tf

from nets import stdlstm, predrnn_pp, predrnn, ConvLSTM

networks_map = {'stdlstm': stdlstm.rnn,
                'predrnn_pp': predrnn_pp.rnn,
                'predrnn': predrnn.rnn,
                'convlstm': ConvLSTM.rnn
               }

def construct_model(name, images, mask_true, num_layers, num_hidden,
                    filter_size, stride, seq_length, input_length, tln):
    '''Returns a sequence of generated frames
    '''
    if name not in networks_map:
        raise ValueError('Name of network unknown %s' % name)
    func = networks_map[name]
    return func(images, mask_true, num_layers, num_hidden, filter_size,
                stride, seq_length, input_length, tln)
