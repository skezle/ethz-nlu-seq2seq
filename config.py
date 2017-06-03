import tensorflow as tf

class Config:
    vocabulary_size = 10000
    bidirectional_encoder = True
    encoder_cell_size = 512
    encoder_cell = tf.contrib.rnn.LSTMCell(encoder_cell_size)
    decoder_cell_size = 1024
    decoder_cell = tf.contrib.rnn.LSTMCell(decoder_cell_size)
    num_layers = 1
    use_dropout = False
    dropout_keep_prob = 0.5
    batch_size = 64
    log_directory = 'logs/'
    num_epochs = 1
    validation_summary_frequency = 10
    checkpoint_frequency = 500
    trace_frequency = 200
    trace_filename = "trace.json"
    input_sentence_max_length = 60
    max_decoder_inference_length = 60

    use_word2vec = False
    word_embedding_size = 100
    word2vec_directory = "word2vec"
    word2vec_path = word2vec_directory + "/wordembeddings_" + str(word_embedding_size) + ".word2vec"
    word2vec_min_word_freq = 1
    word2vec_workers_count = 4

    beam_width = 5
    beam_length_penalty_weight = 0.0