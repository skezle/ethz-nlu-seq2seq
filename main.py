import tensorflow as tf
from data_utility import *

vocabulary_size = 35000
cell_size = 512
batch_size = 64
learning_rate = 1e-4
word_embedding_size = 100
# num_steps = 29 # Network used a step of 1 so that inputs and outputs can match up, hence num_steps = MAX_SENTENCE_LENGTH-1
log_directory = 'logs/'


###
# Building of the graph
###
with tf.variable_scope('DATA'):
    # not all might be needed
    x_encoder = tf.placeholder(tf.int32, [None, None], name='x_encoder') # [batch_size, bucket_size]
    y_encoder = tf.placeholder(tf.int32, [None, None], name='y_encoder') # [batch_size, bucket_size]
    x_decoder = tf.placeholder(tf.int32, [None, None], name='x_decoder') # [batch_size, bucket_size]
    y_decoder = tf.placeholder(tf.int32, [None, None], name='y_decoder') # [batch_size, bucket_size]

###
# Word embedding layer
###
with tf.variable_scope('WORD_EMBEDDINGS'):
    W_embed = tf.get_variable(name='W_embed', shape=[vocabulary_size, word_embedding_size], initializer=tf.contrib.layers.xavier_initializer()) # [vocabulary_size, word_embedding_size]
    # no need for unstacking as dynamic_rnn expects [batch_size, bucket_size, word_embedding_size]
    embeddings_encoder = tf.nn.embedding_lookup(W_embed, x_encoder)  # [batch_size, bucket_size, word_embedding_size]
    embeddings_decoder = tf.nn.embedding_lookup(W_embed, x_decoder)  # [batch_size, bucket_size, word_embedding_size]

###
# Cell to be used both for the encoder RNN and the decoder RNN
###
with tf.variable_scope('CELL'):
    cell = tf.contrib.rnn.GRUCell(cell_size)

###
# Encoder RNN
###
with tf.variable_scope('Encoder RNN'):
    init_state_encoder = tf.placeholder(tf.float32, [batch_size, cell_size], name='init_state_encoder')
    encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(cell, embeddings_encoder, dtype=tf.float32)

###
# Decoder RNN
###
with tf.variable_scope('Decoder RNN'):
    decoder_outputs, decoder_final_state = tf.nn.dynamic_rnn(cell, embeddings_decoder, initial_state=encoder_final_state)

###
# Loss and training
###
# with tf.variable_scope('Loss'):
    # TODO complete Loss

###
# Graph execution
###
with tf.Session() as sess:
    enc_inputs, dec_inputs, word_2_index, index_2_word = get_data_by_type('train')
    # batches = bucket_by_sequence_length(enc_inputs, batch_size)

