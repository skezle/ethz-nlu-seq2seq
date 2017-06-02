import pickle
import os.path
import operator
import cornell_loading
from numpy import array, transpose
from math import ceil
from config import Config as conf

START_TOKEN = "<bos>"
END_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"
START_TOKEN_INDEX = 0
END_TOKEN_INDEX = 1
UNK_TOKEN_INDEX = 2
PAD_TOKEN_INDEX = 3
TRAINING_FILEPATH = 'data/Training_Shuffled_Dataset.txt'
TRAINING_TUPLES_FILEPATH = 'Training_Shuffled_Dataset_tuples.txt'
VALIDATION_FILEPATH = 'data/Validation_Shuffled_Dataset.txt'
VALIDATION_TUPLES_FILEPATH = 'Validation_Shuffled_Dataset_tuples.txt'
VOCABULARY_FILEPATH = 'pickled_vars/vocabulary.p'
W2I_FILEPATH = 'pickled_vars/word_2_index.p'
I2W_FILEPATH = 'pickled_vars/index_2_index.p'
ENCODER_INPUT_FILEPATH = 'pickled_vars/encoder_inputs.p'
DECODER_INPUT_FILEPATH = 'pickled_vars/decoder_inputs.p'
###
# Creates an output file by transforming the original triples file to a tuple file
###
def triples_to_tuples(input_filepath, output_filepath):

    f = open(input_filepath, 'r')
    f1 = open(output_filepath, 'w')

    for line in f:
        triples = line.strip().split('\t')

        f1.write("{}\t{}\n".format(triples[0], triples[1]))
        f1.write("{}\t{}\n".format(triples[1], triples[2]))

    f.close()
    f1.close()


###
# Counts unique_tokens. No shit Sherlock...
###
def count_unique_tokens(filename):
    f = open(filename, 'r')
    s = set()

    for line in f:
        for word in line.split():
            s.add(word)

    f.close()
    return len(s)


###
# Gets or creates a vocabulary based on vocabulary size
###
def get_or_create_vocabulary():

    try:
        vocabulary = pickle.load(open(VOCABULARY_FILEPATH, 'rb'))
    except:
        vocabulary = {}

        train_file = open(TRAINING_TUPLES_FILEPATH)
        for line in train_file:
            conversation = line.strip().split()
            for word in conversation:
                vocabulary[word] = vocabulary.get(word, 0) + 1

        if conf.use_CORNELL_for_training:
            train_file = open(conf.CORNELL_TUPLES_PATH)
            for line in train_file:
                conversation = line.strip().split()
                for word in conversation:
                    vocabulary[word] = vocabulary.get(word, 0) + 1

        sorted_vocab = sorted(vocabulary.items(), key=operator.itemgetter(1), reverse=True)

        sorted_vocab = sorted_vocab[:conf.vocabulary_size-4]
        vocabulary = dict(sorted_vocab)

        vocabulary[START_TOKEN] = 1
        vocabulary[END_TOKEN] = 1
        vocabulary[UNK_TOKEN] = 1
        vocabulary[PAD_TOKEN] = 1

        pickle.dump(vocabulary, open(VOCABULARY_FILEPATH, 'wb'))
        train_file.close()
    return vocabulary


###
# Creates word_2_index and index_2_word dictionaries
###
def get_or_create_dicts_from_train_data():

    try:
        word_2_index = pickle.load(open(W2I_FILEPATH, 'rb'))
        index_2_word = pickle.load(open(I2W_FILEPATH, 'rb'))
    except:
        filename = TRAINING_TUPLES_FILEPATH

        if not os.path.isfile(filename):
            triples_to_tuples(TRAINING_FILEPATH, filename)

        f = open(filename, 'r')

        word_2_index = {START_TOKEN: START_TOKEN_INDEX, END_TOKEN: END_TOKEN_INDEX, UNK_TOKEN: UNK_TOKEN_INDEX, PAD_TOKEN: PAD_TOKEN_INDEX}
        index_2_word = {START_TOKEN_INDEX: START_TOKEN, END_TOKEN_INDEX: END_TOKEN, UNK_TOKEN_INDEX: UNK_TOKEN, PAD_TOKEN_INDEX: PAD_TOKEN}
        vocabulary = get_or_create_vocabulary()

        index = 4 # because the first 4 elements are are already occupied by our tokens
        for line in f:
            conversation = line.strip().split()
            # print(conversation)
            for word in conversation:
                if word in vocabulary and word not in word_2_index:
                    word_2_index[word] = index
                    index_2_word[index] = word
                    index += 1

        pickle.dump(word_2_index, open(W2I_FILEPATH, 'wb'))
        pickle.dump(index_2_word, open(I2W_FILEPATH, 'wb'))

    return word_2_index, index_2_word


###
# Returns data by type (train, eval), together with the word_2_index and index_2_word dicts
###
def get_data_by_type(t):

    if t=='train':
        filename = TRAINING_TUPLES_FILEPATH
    elif t=='eval':
        filename = VALIDATION_TUPLES_FILEPATH
        if not os.path.isfile(filename):
            triples_to_tuples(VALIDATION_FILEPATH, filename)
    else:
        print('Type must be "train" or "eval".')
        return

    word_2_index, index_2_word = get_or_create_dicts_from_train_data()
    vocabulary = get_or_create_vocabulary()

    try:
        encoder_inputs = pickle.load(open(ENCODER_INPUT_FILEPATH, 'rb'))
        decoder_inputs = pickle.load(open(DECODER_INPUT_FILEPATH, 'rb'))
    except:
        encoder_inputs = []
        decoder_inputs = []

        training_files = [filename]
        if conf.use_CORNELL_for_training and t=='train':
            if not os.path.isfile(conf.CORNELL_TUPLES_PATH):
                cornell_loading.create_Cornell_tuples(conf.CORNELL_lines_path, conf.CORNELL_conversations_path,
                                                      conf.CORNELL_TUPLES_PATH)
            training_files.append(conf.CORNELL_TUPLES_PATH)

        for trainset in training_files:
            f = open(trainset, 'r')
            for line in f:
                conversation = line.strip().split('\t')

                encoder_input = []
                for word in conversation[0].split():
                    if word in vocabulary:
                        encoder_input.append(word_2_index[word])
                    else:
                        encoder_input.append(word_2_index[UNK_TOKEN])
                # encoder_input.append(word_2_index[END_TOKEN])  # DO WE NEED EOS?
                encoder_inputs.append(encoder_input)

                decoder_input = []
                for word in conversation[1].split():
                    if word in vocabulary:
                        decoder_input.append(word_2_index[word])
                    else:
                        decoder_input.append(word_2_index[UNK_TOKEN])
                # decoder_input.append(word_2_index[END_TOKEN])
                decoder_inputs.append(decoder_input)

    return encoder_inputs, decoder_inputs, word_2_index, index_2_word


###
# Custom function for bucketing
###
def bucket_by_sequence_length(enc_inputs, dec_inputs, batch_size, sort_data=True):

    assert len(enc_inputs) == len(dec_inputs)

    if sort_data:
        enc_dec = zip(enc_inputs, dec_inputs)
        sorted_enc_dec_pairs = sorted(enc_dec, key=lambda inputs: (len(inputs[0]), len(inputs[1])))

        enc_inputs, dec_inputs = zip(*sorted_enc_dec_pairs)
    # else we keep the data unsorted
    
    num_batches = ceil(len(enc_inputs) / batch_size)    

    for batch_num in range(num_batches):
        encoder_sequence_lengths = [len(sentence) 
                                    for sentence
                                    in enc_inputs[batch_num*batch_size:(batch_num+1)*batch_size]]
        max_len_enc = max(encoder_sequence_lengths)
        encoder_batch = [sentence + ([PAD_TOKEN_INDEX] * (max_len_enc - encoder_sequence_lengths[i]))
                         for i, sentence
                         in enumerate(enc_inputs[batch_num*batch_size:(batch_num+1)*batch_size])]
        encoder_batch = array(encoder_batch).transpose()
        decoder_sequence_lengths = [len(sentence) 
                                    for sentence
                                    in dec_inputs[batch_num*batch_size:(batch_num+1)*batch_size]]
        max_len_dec = max(decoder_sequence_lengths)
        decoder_batch = [sentence + ([PAD_TOKEN_INDEX] * (max_len_dec - decoder_sequence_lengths[i]))
                         for i, sentence
                         in enumerate(dec_inputs[batch_num*batch_size:(batch_num+1)*batch_size])]
        decoder_batch = array(decoder_batch).transpose()
        yield encoder_batch, encoder_sequence_lengths, decoder_batch, decoder_sequence_lengths

