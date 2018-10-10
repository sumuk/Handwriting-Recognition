import tensorflow as tf

from tensorflow.contrib.rnn import BasicRNNCell
from tensorflow.python.util import nest
from tensorflow.python.training import moving_averages
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.framework import ops
def get_rnn_seqlen(seq_lens):
    # seq_lens = tf.Print(seq_lens, [seq_lens], "Original seq len: ", 32)
    seq_lens = tf.cast(seq_lens, tf.float64)
    rnn_seq_lens = tf.div(tf.subtract(seq_lens,20), 2.0)
    rnn_seq_lens = tf.ceil(rnn_seq_lens)
    #rnn_seq_lens = tf.div(tf.subtract(rnn_seq_lens, 10), 1.0)
    #rnn_seq_lens = tf.ceil(rnn_seq_lens)
    rnn_seq_lens = tf.cast(rnn_seq_lens, tf.int32)

 
    # rnn_seq_lens = tf.Print(rnn_seq_lens, [rnn_seq_lens], "Conved seq len: ", summarize=32)
    # print "rnn_seq_lens shape: ", rnn_seq_lens.get_shape().as_list()
    return rnn_seq_lens
def stacked_brnn(cell_fw, cell_bw, num_units, num_layers, inputs, batch_size, conved_seq_lens):
    """
    multi layer bidirectional rnn
    :param cell: RNN cell
    :param num_units: hidden unit of RNN cell
    :param num_layers: the number of layers
    :param inputs: the input sequence
    :param batch_size: batch size
    :return: the output of last layer bidirectional rnn with concatenating
    """
    prev_layer = inputs
    for i in xrange(num_layers):
        with tf.variable_scope("brnn-%d" % i) as scope:
	    
            state_fw = cell_fw.zero_state(batch_size, tf.float32)
            state_bw = cell_fw.zero_state(batch_size, tf.float32)
            (outputs, state) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, prev_layer, sequence_length=conved_seq_lens,
                                                               initial_state_fw=state_fw, initial_state_bw=state_bw, dtype=tf.float32, time_major=False) 
            outputs_fw, outputs_bw = outputs
	    #prev_layer=tf.concat([outputs_fw,outputs_bw],2)
            # prev_layer = tf.add_n([outputs_fw, outputs_bw])
            prev_layer = array_ops.concat(outputs, 2)
    return prev_layer
class CustomRNNCell2(BasicRNNCell):
    """ This is a customRNNCell2 that allows the weights
    to be re-used on multiple devices. In particular, the Matrix of weights is
    set using _variable_on_cpu.
    The default version of the BasicRNNCell, did not support the ability to
    pin weights on one device (say cpu).
    """

    def __init__(self, num_units, input_size=None, activation=tf.nn.relu6):
        self._num_units = num_units
	self._reuse=True

    def __call__(self, inputs, state, scope=None):
        """
         output = new_state = activation(BN(W * input) + U * state + B).
           state dim: batch_size * num_units
           input dim: batch_size * feature_size
           W: feature_size * num_units
           U: num_units * num_units
        """
        with tf.variable_scope(scope or type(self).__name__):
            # print "rnn cell input size: ", inputs.get_shape().as_list()
            # print "rnn cell state size: ", state.get_shape().as_list()
            wsize = inputs.get_shape()[1]
            w = tf.get_variable('W', [self._num_units, wsize], initializer=tf.orthogonal_initializer())
            # print w.name
            resi = tf.matmul(inputs, w, transpose_a=False, transpose_b=True)
            # batch_size * num_units
            bn_resi = resi#seq_batch_norm(resi)
            # bn_resi = resi
            usize = state.get_shape()[1]
	    print(usize)
            u = tf.get_variable('U', [self._num_units, usize], initializer=tf.orthogonal_initializer())
            resu = tf.matmul(state, u, transpose_a=False, transpose_b=True)
            # res_nb = tf.add_n([bn_resi, resu])
            res_nb = tf.add(bn_resi, resu)
	    print (res_nb.shape)
            bias = tf.get_variable('B', [self._num_units],
                                     tf.constant_initializer(0))
            res = tf.add(res_nb, bias)
            output = relux(res, capping=20)
	    print(output.shape)
        return output, output


def relux(x, capping=None):
    """Clipped ReLU"""
    x = tf.nn.relu(x)
    if capping is not None:
        y = tf.minimum(x, capping)
    return y
