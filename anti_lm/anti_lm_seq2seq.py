# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Attention-based decoder functions."""

# ref: https://github.com/tensorflow/tensorflow/blob/r1.0/tensorflow/contrib/seq2seq/python/ops/attention_decoder_fn.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from config import Config as conf

import tensorflow as tf
from tensorflow.contrib.layers.python.layers import layers
from tensorflow.contrib.rnn.python.ops import core_rnn_cell_impl
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import nest



__all__ = [
    "seq2seq_anti_lm_attention_decoder_fn_inference"
]

def seq2seq_anti_lm_attention_decoder_fn_inference(output_fn,
                                   encoder_state,
                                   lm_softmax,
                                   attention_keys,
                                   attention_values,
                                   attention_score_fn,
                                   attention_construct_fn,
                                   embeddings,
                                   start_of_sequence_id,
                                   end_of_sequence_id,
                                   maximum_length,
                                   num_decoder_symbols,
                                   dtype=dtypes.int32,
                                   name=None):
  """Attentional decoder function for `dynamic_rnn_decoder` during inference.

  The `attention_decoder_fn_inference` is a simple inference function for a
  sequence-to-sequence model. It should be used when `dynamic_rnn_decoder` is
  in the inference mode.

  The `attention_decoder_fn_inference` is called with user arguments
  and returns the `decoder_fn`, which can be passed to the
  `dynamic_rnn_decoder`, such that

  ```
  dynamic_fn_inference = attention_decoder_fn_inference(...)
  outputs_inference, state_inference = dynamic_rnn_decoder(
      decoder_fn=dynamic_fn_inference, ...)
  ```

  Further usage can be found in the `kernel_tests/seq2seq_test.py`.

  Args:
    output_fn: An output function to project your `cell_output` onto class
    logits.

    An example of an output function;

    ```
      tf.variable_scope("decoder") as varscope
        output_fn = lambda x: layers.linear(x, num_decoder_symbols,
                                            scope=varscope)

        outputs_train, state_train = seq2seq.dynamic_rnn_decoder(...)
        logits_train = output_fn(outputs_train)

        varscope.reuse_variables()
        logits_inference, state_inference = seq2seq.dynamic_rnn_decoder(
            output_fn=output_fn, ...)
    ```

    If `None` is supplied it will act as an identity function, which
    might be wanted when using the RNNCell `OutputProjectionWrapper`.

    encoder_state: The encoded state to initialize the `dynamic_rnn_decoder`.
    attention_keys: to be compared with target states.
    attention_values: to be used to construct context vectors.
    attention_score_fn: to compute similarity between key and target states.
    attention_construct_fn: to build attention states.
    embeddings: The embeddings matrix used for the decoder sized
    `[num_decoder_symbols, embedding_size]`.
    start_of_sequence_id: The start of sequence ID in the decoder embeddings.
    end_of_sequence_id: The end of sequence ID in the decoder embeddings.
    maximum_length: The maximum allowed of time steps to decode.
    num_decoder_symbols: The number of classes to decode at each time step.
    dtype: (default: `dtypes.int32`) The default data type to use when
    handling integer objects.
    name: (default: `None`) NameScope for the decoder function;
      defaults to "attention_decoder_fn_inference"

  Returns:
    A decoder function with the required interface of `dynamic_rnn_decoder`
    intended for inference.
  """
  with ops.name_scope(name, "attention_decoder_fn_inference", [
      output_fn, encoder_state, attention_keys, attention_values,
      attention_score_fn, attention_construct_fn, embeddings,
      start_of_sequence_id, end_of_sequence_id, maximum_length,
      num_decoder_symbols, dtype
  ]):
    penalized_lm_softmax = math_ops.scalar_mul(conf.anti_lm_penalty, lm_softmax)
    start_of_sequence_id = ops.convert_to_tensor(start_of_sequence_id, dtype)
    end_of_sequence_id = ops.convert_to_tensor(end_of_sequence_id, dtype)
    maximum_length = ops.convert_to_tensor(maximum_length, dtype)
    num_decoder_symbols = ops.convert_to_tensor(num_decoder_symbols, dtype)
    encoder_info = nest.flatten(encoder_state)[0]
    batch_size = encoder_info.get_shape()[0].value
    if output_fn is None:
      output_fn = lambda x: x
    if batch_size is None:
      batch_size = array_ops.shape(encoder_info)[0]

  def decoder_fn(time, cell_state, cell_input, cell_output, context_state):
    """Decoder function used in the `dynamic_rnn_decoder` for inference.

    The main difference between this decoder function and the `decoder_fn` in
    `attention_decoder_fn_train` is how `next_cell_input` is calculated. In
    decoder function we calculate the next input by applying an argmax across
    the feature dimension of the output from the decoder. This is a
    greedy-search approach. (Bahdanau et al., 2014) & (Sutskever et al., 2014)
    use beam-search instead.

    Args:
      time: positive integer constant reflecting the current timestep.
      cell_state: state of RNNCell.
      cell_input: input provided by `dynamic_rnn_decoder`.
      cell_output: output of RNNCell.
      context_state: context state provided by `dynamic_rnn_decoder`.

    Returns:
      A tuple (done, next state, next input, emit output, next context state)
      where:

      done: A boolean vector to indicate which sentences has reached a
      `end_of_sequence_id`. This is used for early stopping by the
      `dynamic_rnn_decoder`. When `time>=maximum_length` a boolean vector with
      all elements as `true` is returned.

      next state: `cell_state`, this decoder function does not modify the
      given state.

      next input: The embedding from argmax of the `cell_output` is used as
      `next_input`.

      emit output: If `output_fn is None` the supplied `cell_output` is
      returned, else the `output_fn` is used to update the `cell_output`
      before calculating `next_input` and returning `cell_output`.

      next context state: `context_state`, this decoder function does not
      modify the given context state. The context state could be modified when
      applying e.g. beam search.

    Raises:
      ValueError: if cell_input is not None.

    """
    with ops.name_scope(
        name, "attention_decoder_fn_inference",
        [time, cell_state, cell_input, cell_output, context_state]):
      if cell_input is not None:
        raise ValueError("Expected cell_input to be None, but saw: %s" %
                         cell_input)
      if cell_output is None:
        # invariant that this is time == 0
        next_input_id = array_ops.ones(
            [batch_size,], dtype=dtype) * (start_of_sequence_id)
        done = array_ops.zeros([batch_size,], dtype=dtypes.bool)
        cell_state = encoder_state
        cell_output = array_ops.zeros(
            [num_decoder_symbols], dtype=dtypes.float32)
        cell_input = array_ops.gather(embeddings, next_input_id)

        # init attention
        attention = _init_attention(encoder_state)
      else:
        # construct attention
        attention = attention_construct_fn(cell_output, attention_keys,
                                           attention_values)
        cell_output = attention

        # argmax decoder
        cell_output = output_fn(cell_output)  # logits
        outputs_softmax = tf.nn.softmax(cell_output) ## probs
        penalized_softmax = math_ops.subtract(outputs_softmax, _get_threshold_softmax_slice(time, penalized_lm_softmax)) ## anti_lm
        next_input_id = math_ops.cast(
            math_ops.argmax(penalized_softmax, 1), dtype=dtype)
        done = math_ops.equal(next_input_id, end_of_sequence_id)
        cell_input = array_ops.gather(embeddings, next_input_id)

      # combine cell_input and attention
      next_input = array_ops.concat([cell_input, attention], 1)

      # if time > maxlen, return all true vector
      done = control_flow_ops.cond(
          math_ops.greater(time, maximum_length),
          lambda: array_ops.ones([batch_size,], dtype=dtypes.bool),
          lambda: done)
      return (done, cell_state, next_input, cell_output, context_state)

  return decoder_fn

def _get_threshold_softmax_slice(time, penalized_lm_softmax):
    return tf.cond(time < conf.anti_lm_max_penalization_len,
            lambda: penalized_lm_softmax[:,time,:],
            lambda: tf.zeros([tf.shape(penalized_lm_softmax)[0], conf.vocabulary_size], tf.float32))

def _init_attention(encoder_state):
  """Initialize attention. Handling both LSTM and GRU.
  Args:
    encoder_state: The encoded state to initialize the `dynamic_rnn_decoder`.
  Returns:
    attn: initial zero attention vector.
  """

  # Multi- vs single-layer
  # TODO(thangluong): is this the best way to check?
  if isinstance(encoder_state, tuple):
    top_state = encoder_state[-1]
  else:
    top_state = encoder_state

  # LSTM vs GRU
  if isinstance(top_state, core_rnn_cell_impl.LSTMStateTuple):
    attn = array_ops.zeros_like(top_state.h)
  else:
    attn = array_ops.zeros_like(top_state)

  return attn
