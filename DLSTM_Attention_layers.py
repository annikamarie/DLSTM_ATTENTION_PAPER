import copy
import itertools
import numpy as np
import tensorflow as tf

def dRNN(cell, inputs, rate, scope='default'):
    """
    This function constructs a layer of dilated RNN.
    Inputs:
        cell -- the dilation operations is implemented independent of the RNN cell.
            In theory, any valid tensorflow rnn cell should work.
        inputs -- the input for the RNN. inputs should be in the form of
            a list of 'n_steps' tenosrs. Each has shape (batch_size, input_dims)
        rate -- the rate here refers to the 'dilations' in the orginal WaveNet paper.
        scope -- variable scope.
    Outputs:
        outputs -- the outputs from the RNN.
    """
    n_steps = len(inputs)
    if rate < 0 or rate >= n_steps:
        raise ValueError('The \'rate\' variable needs to be adjusted.')
    print("Building layer: %s, input length: %d, dilation rate: %d, input dim: %d." % (
        scope, n_steps, rate, inputs[0].get_shape()[1]))

    # make the length of inputs divide 'rate', by using zero-padding
    EVEN = (n_steps % rate) == 0
    if not EVEN:
        # Create a tensor in shape (batch_size, input_dims), which all elements are zero.
        # This is used for zero padding
        zero_tensor = tf.zeros_like(inputs[0])
        dialated_n_steps = n_steps // rate + 1
        print("=====> %d time points need to be padded. " % (
            dialated_n_steps * rate - n_steps))
        print("=====> Input length for sub-RNN: %d" % (dialated_n_steps))
        for i_pad in range(dialated_n_steps * rate - n_steps):
            inputs.append(zero_tensor)
    else:
        dialated_n_steps = n_steps // rate
        print("=====> Input length for sub-RNN: %d" % (dialated_n_steps))

    # now the length of 'inputs' divide rate
    # reshape it in the format of a list of tensors
    # the length of the list is 'dialated_n_steps'
    # the shape of each tensor is [batch_size * rate, input_dims]
    # by stacking tensors that "colored" the same

    # Example:
    # n_steps is 5, rate is 2, inputs = [x1, x2, x3, x4, x5]
    # zero-padding --> [x1, x2, x3, x4, x5, 0]
    # we want to have --> [[x1; x2], [x3; x4], [x_5; 0]]
    # which the length is the ceiling of n_steps/rate
    dilated_inputs = [tf.concat(inputs[i * rate:(i + 1) * rate],
                                axis=0) for i in range(dialated_n_steps)]

    # building a dialated RNN with reformated (dilated) inputs
    dilated_outputs, _ = tf.contrib.rnn.static_rnn(
        cell, dilated_inputs,
        dtype=tf.float32, scope=scope)

    # reshape output back to the input format as a list of tensors with shape [batch_size, input_dims]
    # split each element of the outputs from size [batch_size*rate, input_dims] to
    # [[batch_size, input_dims], [batch_size, input_dims], ...] with length = rate
    splitted_outputs = [tf.split(output, rate, axis=0)
                        for output in dilated_outputs]
    unrolled_outputs = [output
                        for sublist in splitted_outputs for output in sublist]
    # remove padded zeros
    outputs = unrolled_outputs[:n_steps]

    return outputs


def multi_dRNN_with_dilations(cells, inputs, dilations):
    """
    This function constucts a multi-layer dilated RNN.
    Inputs:
        cells -- A list of RNN cells.
        inputs -- A list of 'n_steps' tensors, each has shape (batch_size, input_dims).
        dilations -- A list of integers with the same length of 'cells' indicates the dilations for each layer.
    Outputs:
        x -- A list of 'n_steps' tensors, as the outputs for the top layer of the multi-dRNN.
    """
    assert (len(cells) == len(dilations))
    x = copy.copy(inputs)
    for cell, dilation in zip(cells, dilations):
        scope_name = "multi_dRNN_dilation_%d" % dilation
        x = dRNN(cell, x, dilation, scope=scope_name)
    return x

def _contruct_cells(hidden_structs, cell_type,dropouts):
    """
    This function contructs a list of cells.
    """
    # error checking
    if cell_type not in ["RNN", "LSTM", "GRU"]:
        raise ValueError("The cell type is not currently supported.")

    # define cells
    cells = []
    for hidden_dims in hidden_structs:
        if cell_type == "RNN":
            cell = tf.contrib.rnn.BasicRNNCell(hidden_dims)
        elif cell_type == "LSTM":
            cell = tf.contrib.rnn.BasicLSTMCell(hidden_dims)
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=dropouts)
        elif cell_type == "GRU":
            cell = tf.contrib.rnn.GRUCell(hidden_dims)
        cells.append(cell)

    return cells


def _rnn_reformat(embedded_chars, embedding_dim, n_steps):
    """
    This function reformat input to the shape that standard RNN can take.

    Inputs:
        x -- a tensor of shape (batch_size, n_steps, input_dims).
    Outputs:
        x_reformat -- a list of 'n_steps' tenosrs, each has shape (batch_size, input_dims).
    """
    # permute batch_size and n_steps
    x_ = tf.transpose(embedded_chars, [1, 0, 2])
    # reshape to (n_steps*batch_size, input_dims)
    x_ = tf.reshape(x_, [-1, embedding_dim])
    # split to get a list of 'n_steps' tensors of shape (batch_size, input_dims)
    x_reformat = tf.split(x_, n_steps, 0)

    return x_reformat


def drnn_classification(embedded_chars,
                        hidden_structs,
                        dilations,
                        n_steps,
                        dropouts,
                        embedding_dim,
                        cell_type="RNN"):

    """This function construct a multilayer dilated RNN for classifiction.
    Inputs:
        x -- a tensor of shape (batch_size, n_steps, input_dims).
        hidden_structs -- a list, each element indicates the hidden node dimension of each layer.
        dilations -- a list, each element indicates the dilation of each layer.
        n_steps -- the length of the sequence.
        input_dims -- the input dimension.
        cell_type -- the type of the RNN cell, should be in ["RNN", "LSTM", "GRU"].

    Outputs:
        pred -- the prediction logits at the last timestamp and the last layer of the RNN.
                'pred' does not pass any output activation functions."""

    # error checking
    assert (len(hidden_structs) == len(dilations))

    # reshape inputs
    x_reformat = _rnn_reformat(embedded_chars, embedding_dim, n_steps)

    # construct a list of cells
    cells = _contruct_cells(hidden_structs, cell_type,dropouts)

    # define dRNN structures
    layer_outputs = multi_dRNN_with_dilations(cells, x_reformat, dilations)

    return layer_outputs

#######################################################################################################################
#######################################################################################################################

def attention(inputs, attention_size, time_major=True, return_alphas=False):
    """
    Attention mechanism layer which reduces RNN/Bi-RNN outputs with Attention vector.

    The idea was proposed in the article by Z. Yang et al., "Hierarchical Attention Networks
     for Document Classification", 2016: http://www.aclweb.org/anthology/N16-1174.
    Variables notation is also inherited from the article

    Args:
        inputs: The Attention inputs.
            Matches outputs of RNN/Bi-RNN layer (not final state):
                In case of RNN, this must be RNN outputs `Tensor`:
                    If time_major == False (default), this must be a tensor of shape:
                        `[batch_size, max_time, cell.output_size]`.
                    If time_major == True, this must be a tensor of shape:
                        `[max_time, batch_size, cell.output_size]`.
                In case of Bidirectional RNN, this must be a tuple (outputs_fw, outputs_bw) containing the forward and
                the backward RNN outputs `Tensor`.
                    If time_major == False (default),
                        outputs_fw is a `Tensor` shaped:
                        `[batch_size, max_time, cell_fw.output_size]`
                        and outputs_bw is a `Tensor` shaped:
                        `[batch_size, max_time, cell_bw.output_size]`.
                    If time_major == True,
                        outputs_fw is a `Tensor` shaped:
                        `[max_time, batch_size, cell_fw.output_size]`
                        and outputs_bw is a `Tensor` shaped:
                        `[max_time, batch_size, cell_bw.output_size]`.
        attention_size: Linear size of the Attention weights.
        time_major: The shape format of the `inputs` Tensors.
            If true, these `Tensors` must be shaped `[max_time, batch_size, depth]`.
            If false, these `Tensors` must be shaped `[batch_size, max_time, depth]`.
            Using `time_major = True` is a bit more efficient because it avoids
            transposes at the beginning and end of the RNN calculation.  However,
            most TensorFlow data is batch-major, so by default this function
            accepts input and emits output in batch-major form.
        return_alphas: Whether to return attention coefficients variable along with layer's output.
            Used for visualization purpose.
    Returns:
        The Attention output `Tensor`.
        In case of RNN, this will be a `Tensor` shaped:
            `[batch_size, cell.output_size]`.
        In case of Bidirectional RNN, this will be a `Tensor` shaped:
            `[batch_size, cell_fw.output_size + cell_bw.output_size]`.
    """

    inputs = tf.stack(inputs, axis=0)

    if time_major:
        # (T,B,D) => (B,T,D)
        inputs = tf.transpose(inputs, [1, 0, 2])

    hidden_size = inputs.shape[2].value  # D value - hidden size of the RNN layer

    # Trainable parameters
    w_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
    b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
    u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))

    with tf.name_scope('Variable'):
        # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
        # the shape of `v` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size
        v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)

    # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
    vu = tf.tensordot(v, u_omega, axes=1, name='vu')  # (B,T) shape
    alphas = tf.nn.softmax(vu, name='alphas')  # (B,T) shape

    # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
    output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)

    if not return_alphas:
        return output
    else:
        return output, alphas
