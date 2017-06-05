import tensorflow as tf

class Config:
    vocabulary_size = 10024
    bidirectional_encoder = False
    encoder_cell_size = 512
    encoder_cell = tf.contrib.rnn.LSTMCell(encoder_cell_size)
    decoder_cell_size = 512
    decoder_cell = tf.contrib.rnn.LSTMCell(decoder_cell_size)
    num_layers = 1
    use_dropout = True
    dropout_keep_prob = 0.5
    batch_size = 40
    log_directory = 'logs/'
    num_epochs = 50
    validation_summary_frequency = 100
    checkpoint_frequency = 5000
    trace_frequency = 10000
    trace_filename = "trace.json"
    input_sentence_max_length = 60
    max_decoder_inference_length = 80

    use_word2vec = True
    word_embedding_size = 200
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

    ###
    # Genres extraction
    ###
    use_genres = True
    genres_basedataset_metainfo_path = 'data/MetaInfo.txt'
    genres_basedataset_training_path = 'data/Training_Shuffled_Dataset_Labels.txt'
    genres_basedataset_validation_path = 'data/Validation_Shuffled_Dataset_Labels.txt'
    genres_cornell_metainfo_path = 'data/cornell_movie_dialogs_corpus/movie_titles_metadata.txt'

