import tensorflow as tf


class Config:
    vocabulary_size = 35000
    encoder_cell_size = 512
    encoder_cell = tf.contrib.rnn.LSTMCell(encoder_cell_size)
    decoder_cell_size = 512
    decoder_cell = tf.contrib.rnn.LSTMCell(decoder_cell_size)
    batch_size = 64
    log_directory = 'logs/'
    num_epochs = 1
    validation_summary_frequency = 10
    checkpoint_frequency = 100
    max_decoder_inference_length = 200
    pickled_vars_directory = "pickled_vars"

    use_word2vec = True
    word_embedding_size = 100
    word2vec_directory = "word2vec"
    word2vec_path = word2vec_directory + "/wordembeddings_" + str(word_embedding_size) + ".word2vec"
    word2vec_min_word_freq = 1
    word2vec_workers_count = 4

    CORNELL_base_path = 'data/cornell_movie_dialogs_corpus'
    CORNELL_lines_path = CORNELL_base_path + '/movie_lines.txt'
    CORNELL_conversations_path = CORNELL_base_path + '/movie_conversations.txt'
    CORNELL_TUPLES_PATH = CORNELL_base_path + '/Training_Cornell_Shuffled_Dataset.txt'
    both_datasets_tuples_filepath = 'Training_both_datasets.txt'
    use_CORNELL_for_training = True
    use_CORNELL_for_word2vec = True

