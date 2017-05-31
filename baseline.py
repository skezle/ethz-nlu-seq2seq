# Working with TF commit 24466c2e6d32621cd85f0a78d47df6eed2c5c5a6
# Ref: https://github.com/ematvey/tensorflow-seq2seq-tutorials

import math

import numpy as np
import tensorflow as tf
import tensorflow.contrib.seq2seq as seq2seq
from tensorflow.contrib.layers import safe_embedding_lookup_sparse as embedding_lookup_unique
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple, GRUCell
from data_utility import START_TOKEN_INDEX, END_TOKEN_INDEX, PAD_TOKEN_INDEX
from config import Config as conf


class BaselineModel():
    """Seq2Seq model using blocks from new `tf.contrib.seq2seq`."""

    BOS = START_TOKEN_INDEX
    EOS = END_TOKEN_INDEX
    PAD = PAD_TOKEN_INDEX
    def __init__(self, encoder_cell, decoder_cell, vocab_size, embedding_size,
                 bidirectional=True,
                 attention=False):
        self.bidirectional = bidirectional
        self.attention = attention ## used when initialising the decoder

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size

        self.encoder_cell = encoder_cell
        self.decoder_cell = decoder_cell

        self._make_graph()

    @property
    def decoder_hidden_units(self):
        # @TODO: is this correct for LSTMStateTuple?
        return self.decoder_cell.output_size

    def _make_graph(self):
        tf.reset_default_graph()

        self._init_placeholders()

        self._init_decoder_train_connectors()
        self._init_embeddings()

        if self.bidirectional:
            self._init_bidirectional_encoder()
        else:
            self._init_simple_encoder()

        self._init_decoder()

        self._init_optimizer()

        self._init_summary()

    def _init_placeholders(self):
        """ Everything is time-major """
        self.encoder_inputs = tf.placeholder(
            shape=(None, None),
            dtype=tf.int32,
            name='encoder_inputs',
        )
        self.encoder_inputs_length = tf.placeholder(
            shape=(None,),
            dtype=tf.int32,
            name='encoder_inputs_length',
        )

        # required for training, not required for testing
        self.decoder_targets = tf.placeholder(
            shape=(None, None),
            dtype=tf.int32,
            name='decoder_targets'
        )
        self.decoder_targets_length = tf.placeholder(
            shape=(None,),
            dtype=tf.int32,
            name='decoder_targets_length',
        )

    def _init_decoder_train_connectors(self):
        """
        During training, `decoder_targets`
        and decoder logits. This means that their shapes should be compatible.

        Here we do a bit of plumbing to set this up.
        """
        with tf.name_scope('DecoderTrainFeeds'):
            sequence_size, batch_size = tf.unstack(tf.shape(self.decoder_targets))

            BOS_SLICE = tf.ones([1, batch_size], dtype=tf.int32) * self.BOS
            PAD_SLICE = tf.ones([1, batch_size], dtype=tf.int32) * self.PAD

            self.decoder_train_inputs = tf.concat([BOS_SLICE, self.decoder_targets], axis=0)
            self.decoder_train_length = self.decoder_targets_length + 1

            decoder_train_targets = tf.concat([self.decoder_targets, PAD_SLICE], axis=0)
            decoder_train_targets_seq_len, _ = tf.unstack(tf.shape(decoder_train_targets))

            eos_multiplication_mask = tf.one_hot(self.decoder_train_length - 1,
                                                        decoder_train_targets_seq_len,
                                                        on_value=0, off_value=1,
                                                        dtype=tf.int32)
            eos_addition_mask = tf.one_hot(self.decoder_train_length - 1,
                                                        decoder_train_targets_seq_len,
                                                        on_value=self.EOS, off_value=0,
                                                        dtype=tf.int32)
            eos_multiplication_mask = tf.transpose(eos_multiplication_mask, [1, 0])
            eos_addition_mask = tf.transpose(eos_addition_mask, [1, 0])

            # hacky way using one_hot to put EOS symbol at the end of target sequence
            decoder_train_targets = tf.add(
                                            tf.multiply(decoder_train_targets,
                                            eos_multiplication_mask),
                                            eos_addition_mask)

            self.decoder_train_targets = decoder_train_targets

            self.loss_weights = tf.sequence_mask(self.decoder_train_length, 
                                                 tf.reduce_max(self.decoder_train_length),
                                                 dtype=tf.float32)

    def _init_embeddings(self):
        with tf.variable_scope("embedding") as scope:

            # Uniform(-sqrt(3), sqrt(3)) has variance=1.
            sqrt3 = math.sqrt(3)
            initializer = tf.random_uniform_initializer(-sqrt3, sqrt3)

            self.embedding_matrix = tf.get_variable(
                name="embedding_matrix",
                shape=[self.vocab_size, self.embedding_size],
                initializer=initializer,
                dtype=tf.float32)

            self.encoder_inputs_embedded = tf.nn.embedding_lookup(
                self.embedding_matrix, self.encoder_inputs)

            self.decoder_train_inputs_embedded = tf.nn.embedding_lookup(
                self.embedding_matrix, self.decoder_train_inputs)

    def _init_simple_encoder(self):
        with tf.variable_scope("Encoder") as scope:
            (self.encoder_outputs, self.encoder_state) = (
                tf.nn.dynamic_rnn(cell=self.encoder_cell,
                                  inputs=self.encoder_inputs_embedded,
                                  sequence_length=self.encoder_inputs_length,
                                  time_major=True,
                                  dtype=tf.float32)
                )

    def _init_bidirectional_encoder(self):
        with tf.variable_scope("BidirectionalEncoder") as scope:

            ((encoder_fw_outputs,
              encoder_bw_outputs),
             (encoder_fw_state,
              encoder_bw_state)) = (
                tf.nn.bidirectional_dynamic_rnn(cell_fw=self.encoder_cell,
                                                cell_bw=self.encoder_cell,
                                                inputs=self.encoder_inputs_embedded,
                                                sequence_length=self.encoder_inputs_length,
                                                time_major=True,
                                                dtype=tf.float32)
                )

            self.encoder_outputs = tf.concat((encoder_fw_outputs, encoder_bw_outputs), 2)

            if isinstance(encoder_fw_state, LSTMStateTuple):

                encoder_state_c = tf.concat(
                    (encoder_fw_state.c, encoder_bw_state.c), 1, name='bidirectional_concat_c')
                encoder_state_h = tf.concat(
                    (encoder_fw_state.h, encoder_bw_state.h), 1, name='bidirectional_concat_h')
                self.encoder_state = LSTMStateTuple(c=encoder_state_c, h=encoder_state_h)

            elif isinstance(encoder_fw_state, tf.Tensor):
                self.encoder_state = tf.concat((encoder_fw_state, encoder_bw_state), 1, name='bidirectional_concat')

    def _init_decoder(self):
        with tf.variable_scope("Decoder") as scope:
            def output_fn(outputs):
                ##return tf.contrib.layers.linear(outputs, self.vocab_size, scope=scope)
                return tf.contrib.layers.fully_connected(inputs=outputs,
                                                         num_outputs=self.vocab_size,
                                                         activation_fn=None, ## linear
                                                         scope=scope)

            if not self.attention:
                decoder_fn_train = seq2seq.simple_decoder_fn_train(encoder_state=self.encoder_state)
                decoder_fn_inference = seq2seq.simple_decoder_fn_inference(
                    output_fn=output_fn,
                    encoder_state=self.encoder_state, ## output of the encoder
                    embeddings=self.embedding_matrix,
                    start_of_sequence_id=self.BOS,
                    end_of_sequence_id=self.EOS,
                    maximum_length=conf.max_decoder_inference_length,
                    num_decoder_symbols=self.vocab_size,
                )
            else:

                # attention_states: size [batch_size, max_time, num_units]
                attention_states = tf.transpose(self.encoder_outputs, [1, 0, 2])

                (attention_keys,
                attention_values,
                attention_score_fn,
                attention_construct_fn) = seq2seq.prepare_attention(
                    attention_states=attention_states,
                    attention_option="bahdanau",
                    num_units=self.decoder_hidden_units,
                )

                decoder_fn_train = seq2seq.attention_decoder_fn_train(
                    encoder_state=self.encoder_state, ## output of the encoder
                    attention_keys=attention_keys,
                    attention_values=attention_values,
                    attention_score_fn=attention_score_fn,
                    attention_construct_fn=attention_construct_fn,
                    name='attention_decoder'
                )

                decoder_fn_inference = seq2seq.attention_decoder_fn_inference(
                    output_fn=output_fn,
                    encoder_state=self.encoder_state, ## output of the encoder
                    attention_keys=attention_keys,
                    attention_values=attention_values,
                    attention_score_fn=attention_score_fn,
                    attention_construct_fn=attention_construct_fn,
                    embeddings=self.embedding_matrix,
                    start_of_sequence_id=self.BOS,
                    end_of_sequence_id=self.EOS,
                    maximum_length=conf.max_decoder_inference_length,
                    num_decoder_symbols=self.vocab_size,
                )

            (self.decoder_outputs_train,
             self.decoder_state_train,
             self.decoder_context_state_train) = (
                seq2seq.dynamic_rnn_decoder(
                    cell=self.decoder_cell,
                    decoder_fn=decoder_fn_train,
                    inputs=self.decoder_train_inputs_embedded,
                    sequence_length=self.decoder_train_length,
                    time_major=True,
                    scope=scope,
                )
            )

            self.decoder_logits_train = output_fn(self.decoder_outputs_train)
            self.decoder_softmax_train = tf.nn.softmax(self.decoder_logits_train)

            self.decoder_prediction_train = tf.argmax(self.decoder_logits_train, axis=-1, name='decoder_prediction_train')

            scope.reuse_variables()

            (self.decoder_logits_inference,
             self.decoder_state_inference,
             self.decoder_context_state_inference) = (
                seq2seq.dynamic_rnn_decoder(
                    cell=self.decoder_cell,
                    decoder_fn=decoder_fn_inference,
                    time_major=True,
                    scope=scope,
                )
            )
            self.decoder_prediction_inference = tf.argmax(self.decoder_logits_inference, axis=-1, name='decoder_prediction_inference')

    def _init_optimizer(self):
        logits = tf.transpose(self.decoder_logits_train, [1, 0, 2])
        targets = tf.transpose(self.decoder_train_targets, [1, 0])
        self.loss = seq2seq.sequence_loss(logits=logits, targets=targets,
                                          weights=self.loss_weights)
        optimizer = tf.train.AdamOptimizer()
        gradients, variables = zip(*optimizer.compute_gradients(self.loss))
        gradients, _ = tf.clip_by_global_norm(gradients, 10.0)
        self.train_op = optimizer.apply_gradients(zip(gradients, variables))

    def _init_summary(self):
        tf.summary.scalar("loss", self.loss)
        self.summary_op = tf.summary.merge_all()

    def make_train_inputs(self, input_seq, input_seq_len, target_seq, target_seq_len):
        return {
            self.encoder_inputs: input_seq,
            self.encoder_inputs_length: input_seq_len,
            self.decoder_targets: target_seq,
            self.decoder_targets_length: target_seq_len,
        }

    def make_inference_inputs(self, input_seq, input_seq_len):
        return {
            self.encoder_inputs: input_seq,
            self.encoder_inputs_length: input_seq_len,
        }
