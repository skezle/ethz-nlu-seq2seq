import tensorflow as tf
class Config:
    vocabulary_size = 35000
    encoder_cell_size = 512
    encoder_cell=tf.contrib.rnn.LSTMCell(encoder_cell_size)
    decoder_cell_size = 512
    decoder_cell=tf.contrib.rnn.LSTMCell(decoder_cell_size)
    batch_size = 64
    word_embedding_size = 100
    log_directory = 'logs/'
    num_epochs = 1