import tensorflow as tf

class Config:
    vocabulary_size = 20000
    encoder_cell_size = 200
    encoder_cell = tf.contrib.rnn.LSTMCell(encoder_cell_size)
    decoder_cell_size = 200
    decoder_cell = tf.contrib.rnn.LSTMCell(decoder_cell_size)
    batch_size = 60
    log_directory = 'logs/'
    num_epochs = 1
    validation_summary_frequency = 10
    checkpoint_frequency = 1000
    max_decoder_inference_length = 30
    pickled_vars_directory = "pickled_vars"
    use_word2vec = False
    word_embedding_size = 100
    word2vec_directory = "word2vec"
    word2vec_path = word2vec_directory + "/wordembeddings_" + str(word_embedding_size) + ".word2vec"
    word2vec_min_word_freq = 1
    word2vec_workers_count = 4
