# Working with TF commit 24466c2e6d32621cd85f0a78d47df6eed2c5c5a6
# Ref: https://github.com/ematvey/tensorflow-seq2seq-tutorials

import math

import numpy as np
import tensorflow as tf
import tensorflow.contrib.seq2seq as seq2seq
from custom_decoder import custom_dynamic_decode
from tensorflow.contrib.layers import safe_embedding_lookup_sparse as embedding_lookup_unique
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple, GRUCell
from tensorflow.python.layers.core import Dense
from tensorflow.python.ops.rnn_cell_impl import _zero_state_tensors
from data_utility import START_TOKEN_INDEX, END_TOKEN_INDEX, PAD_TOKEN_INDEX
from config import Config as conf
class BaselineModel():
    """Seq2Seq model using blocks from new `tf.contrib.seq2seq`."""

    BOS = START_TOKEN_INDEX
    EOS = END_TOKEN_INDEX
    PAD = PAD_TOKEN_INDEX
    def __init__(self, vocab_size, embedding_size,
                 bidirectional=True,
                 attention=False,
                 dropout=False,
                 num_layers=1,
                 is_training=True):
        self.bidirectional = bidirectional
        self.encoder_scope_name = "Encoder" if bidirectional else "BidirectionalEncoder"
        self.decoder_scope_name = "Decoder"
        self.attention = attention ## used when initialising the decoder
        self.dropout = dropout
        self.num_layers = num_layers
        self.is_training = is_training

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size

        self._make_graph()

    @property
    def decoder_hidden_units(self):
        # @TODO: is this correct for LSTMStateTuple?
        return self.decoder_cell.output_size

    def _make_graph(self):
        tf.reset_default_graph()

        
        self._init_placeholders()

        self._init_cells()

        self._init_decoder_train_connectors()
        self._init_embeddings()

        if self.bidirectional:
            self._init_bidirectional_encoder()
        else:
            self._init_simple_encoder()

        self._init_attention()

        if self.is_training:
            self._init_train_decoder()
            self._init_optimizer()
            self._init_summary()
        else:
            self._init_inference_decoder()
            self._init_dummy_inference_decoder()

        

    def _init_cells(self):
        with tf.variable_scope(self.encoder_scope_name) as scope:
            cell = tf.contrib.rnn.LSTMCell(conf.encoder_cell_size)
            if self.dropout:
                cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=self.dropout_keep_prob)
            if self.num_layers != 1:
                tf.contrib.rnn.MultiRNNCell([cell for _ in xrange(self.num_layers)]) # The cells in the different layers share the same weights
            self.encoder_cell = cell

        with tf.variable_scope(self.decoder_scope_name) as scope:
            cell = tf.contrib.rnn.LSTMCell(conf.decoder_cell_size)
            if self.dropout:
                cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=self.dropout_keep_prob)
            if self.num_layers != 1:
                tf.contrib.rnn.MultiRNNCell([cell for _ in xrange(self.num_layers)]) # The cells in the different layers share the same weights
            self.decoder_cell = cell

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

        self.lm_logits = tf.placeholder(
            shape=(None, None, None),
            dtype=tf.float32,
            name='lm_logits',
        )

        self.dropout_keep_prob = tf.placeholder(tf.float32)

        self.batch_size, _ = tf.unstack(tf.shape(self.encoder_inputs))

    def _init_decoder_train_connectors(self):
        """
        During training, `decoder_targets`
        and decoder logits. This means that their shapes should be compatible.

        Here we do a bit of plumbing to set this up.
        """
        with tf.name_scope('DecoderTrainFeeds'):
            batch_size, sequence_size = tf.unstack(tf.shape(self.decoder_targets))

            BOS_SLICE = tf.ones([batch_size, 1], dtype=tf.int32) * self.BOS
            PAD_SLICE = tf.ones([batch_size, 1], dtype=tf.int32) * self.PAD

            #self.decoder_targets = tf.reshape(self.decoder_targets, [-1, conf.batch_size])
            self.decoder_train_inputs = tf.concat([BOS_SLICE, self.decoder_targets], axis=1)
            self.decoder_train_length = self.decoder_targets_length + 1

            decoder_train_targets = tf.concat([self.decoder_targets, PAD_SLICE], axis=1)
            _, decoder_train_targets_seq_len = tf.unstack(tf.shape(decoder_train_targets))

            eos_multiplication_mask = tf.one_hot(self.decoder_train_length - 1,
                                                        decoder_train_targets_seq_len,
                                                        on_value=0, off_value=1,
                                                        dtype=tf.int32)
            eos_addition_mask = tf.one_hot(self.decoder_train_length - 1,
                                                        decoder_train_targets_seq_len,
                                                        on_value=self.EOS, off_value=0,
                                                        dtype=tf.int32)

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
        with tf.variable_scope(self.encoder_scope_name) as scope:
            (self.encoder_outputs, self.encoder_state) = (
                tf.nn.dynamic_rnn(cell=self.encoder_cell,
                                  inputs=self.encoder_inputs_embedded,
                                  sequence_length=self.encoder_inputs_length,
                                  time_major=False,
                                  dtype=tf.float32)
                )

    def _init_bidirectional_encoder(self):
        with tf.variable_scope(self.encoder_scope_name) as scope:

            ((encoder_fw_outputs,
              encoder_bw_outputs),
             (encoder_fw_state,
              encoder_bw_state)) = (
                tf.nn.bidirectional_dynamic_rnn(cell_fw=self.encoder_cell,
                                                cell_bw=self.encoder_cell,
                                                inputs=self.encoder_inputs_embedded,
                                                sequence_length=self.encoder_inputs_length,
                                                time_major=False,
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

    def _init_attention(self):
        with tf.variable_scope(self.decoder_scope_name) as scope:
            self.dense_layer = Dense(conf.vocabulary_size,
                                kernel_initializer = tf.contrib.layers.xavier_initializer())

            self.decoder_init_state = self.encoder_state
            if self.attention:
                attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                    num_units=conf.encoder_cell_size,
                    memory=self.encoder_outputs,
                    memory_sequence_length=self.encoder_inputs_length,
                    name="BahdanauAttention")

                self.decoder_cell = tf.contrib.seq2seq.DynamicAttentionWrapper(
                    cell=self.decoder_cell,
                    attention_mechanism=attention_mechanism,
                    attention_size=conf.decoder_cell_size)             
            
                self.decoder_init_state = tf.contrib.seq2seq.DynamicAttentionWrapperState(
                        self.encoder_state,
                        _zero_state_tensors(conf.decoder_cell_size, 
                                            self.batch_size, 
                                            tf.float32))
    def _init_train_decoder(self):
        with tf.variable_scope(self.decoder_scope_name) as scope:
            ## Training
            trainingHelper = tf.contrib.seq2seq.TrainingHelper(
                            inputs=self.decoder_train_inputs_embedded,
                            sequence_length=self.decoder_train_length,
                            time_major=False)

            self.decoder_train = tf.contrib.seq2seq.BasicDecoder(
                cell=self.decoder_cell,
                helper=trainingHelper,
                initial_state=self.decoder_init_state,
                output_layer=self.dense_layer)

            decoder_train_outputs, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder=self.decoder_train,
                output_time_major=False,
                impute_finished=False,
                maximum_iterations=conf.input_sentence_max_length,
                scope=scope)    
            
            self.decoder_logits_train = decoder_train_outputs.rnn_output
            self.decoder_softmax_train = tf.nn.softmax(decoder_train_outputs.rnn_output)

            self.decoder_prediction_train = tf.argmax(self.decoder_logits_train, axis=-1, name='decoder_prediction_train')

            scope.reuse_variables()

            ## Validation
            decoder_validation_outputs, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder=self.decoder_train,
                output_time_major=False,
                impute_finished=True,
                maximum_iterations=None, # 
                scope=scope)    

            self.decoder_logits_validation = decoder_validation_outputs.rnn_output
            
    
    def _init_inference_decoder(self):
        with tf.variable_scope(self.decoder_scope_name) as scope:
            inferenceHelper = seq2seq.GreedyEmbeddingHelper(
                    embedding=self.embedding_matrix,
                    start_tokens=tf.tile([START_TOKEN_INDEX], [self.batch_size]),
                    end_token=END_TOKEN_INDEX)

            self.decoder_inference = tf.contrib.seq2seq.BasicDecoder(
                    cell=self.decoder_cell,
                    helper=inferenceHelper,
                    initial_state=self.decoder_init_state,
                    output_layer=self.dense_layer)

            ## Prediction
            decoder_prediction_outputs, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder=self.decoder_inference,
                output_time_major=False,
                impute_finished=True,
                maximum_iterations=conf.max_decoder_inference_length,
                scope=scope)
    
            orig_logits = decoder_prediction_outputs.rnn_output
            penalized_logits = tf.subtract(orig_logits, tf.scalar_mul(conf.antilm_penalization_weight, self.lm_logits))
            self.decoder_prediction_inference = tf.argmax(penalized_logits, axis=-1, name='decoder_prediction_inference')
            
    def _init_dummy_inference_decoder(self):
        with tf.variable_scope(self.decoder_scope_name, reuse=True) as scope:
            inferenceHelper = seq2seq.GreedyEmbeddingHelper(
                    embedding=self.embedding_matrix,
                    start_tokens=tf.tile([START_TOKEN_INDEX], [self.batch_size]),
                    end_token=END_TOKEN_INDEX)

            self.decoder_dummy_inference = tf.contrib.seq2seq.BasicDecoder(
                    cell=self.decoder_cell,
                    helper=inferenceHelper,
                    initial_state=self.decoder_init_state,
                    output_layer=self.dense_layer)

            ## Prediction
            decoder_dummy_prediction_outputs, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder=self.decoder_dummy_inference,
                output_time_major=False,
                impute_finished=False,
                maximum_iterations=conf.max_decoder_inference_length,
                scope=scope)
            self.dummy_decoder_logits = decoder_dummy_prediction_outputs.rnn_output
            
    def _init_optimizer(self):
        logits = tf.transpose(self.decoder_logits_train, [1, 0, 2])
        validation_logits = tf.transpose(self.decoder_logits_validation, [1, 0, 2])
        targets = tf.transpose(self.decoder_train_targets, [1, 0])
        self.loss = seq2seq.sequence_loss(logits=logits, targets=targets,
                                          weights=self.loss_weights)
        self.validation_loss = seq2seq.sequence_loss(logits=validation_logits, targets=targets,
                                          weights=self.loss_weights)
        optimizer = tf.train.AdamOptimizer()
        gradients, variables = zip(*optimizer.compute_gradients(self.loss))
        gradients, _ = tf.clip_by_global_norm(gradients, 10.0)
        self.train_op = optimizer.apply_gradients(zip(gradients, variables))

    def _init_summary(self):
        loss = tf.summary.scalar("loss", self.loss)
        vali_loss = tf.summary.scalar("validation_loss", self.validation_loss)
        self.summary_op = tf.summary.merge([loss])
        self.validation_summary_op = tf.summary.merge([vali_loss])

    def make_train_inputs(self, input_seq, input_seq_len, target_seq, target_seq_len, keep_prob = conf.dropout_keep_prob):
        return {
            self.encoder_inputs: input_seq,
            self.encoder_inputs_length: input_seq_len,
            self.decoder_targets: target_seq,
            self.decoder_targets_length: target_seq_len,
            self.dropout_keep_prob: keep_prob,
        }

    def make_inference_inputs(self, input_seq, input_seq_len, lm_logits = None):
        dic = {
            self.encoder_inputs: input_seq,
            self.encoder_inputs_length: input_seq_len,
            self.dropout_keep_prob: 1,
        }
        if lm_logits is not None:
            dic[self.lm_logits] = lm_logits

        return dic