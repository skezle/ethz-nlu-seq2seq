import pickle
import os.path
import operator
from numpy import array, transpose
from math import ceil
from config import Config as conf
from random import shuffle
from shutil import copyfile

START_TOKEN = "<bos>"
END_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"
START_TOKEN_INDEX = 0
END_TOKEN_INDEX = 1
UNK_TOKEN_INDEX = 2
PAD_TOKEN_INDEX = 3
TRAINING_FILEPATH = 'data/Training_Shuffled_Dataset.txt'
TRAINING_TUPLES_FILEPATH = 'data/Training_Shuffled_Dataset_tuples.txt'
VALIDATION_FILEPATH = 'data/Validation_Shuffled_Dataset.txt'
VALIDATION_TUPLES_FILEPATH = 'data/Validation_Shuffled_Dataset_tuples.txt'
VOCABULARY_FILEPATH = 'pickled_vars/vocabulary.p'
W2I_FILEPATH = 'pickled_vars/word_2_index.p'
I2W_FILEPATH = 'pickled_vars/index_2_index.p'
ENCODER_INPUT_FILEPATH = 'pickled_vars/encoder_inputs.p'
DECODER_INPUT_FILEPATH = 'pickled_vars/decoder_inputs.p'
###
# Creates an output file by transforming the original triples file to a tuples file
# preserving the order of the dialogs. e.g. for a dialog consisting of sent1 -- sent2 -- sent3
# the generated tuple "sent2 -- sent3" will directly follow "sent1 -- sent2"
#
# If output_filepath is None, then the output tuples will not be written to disk,
# but returned
###
def triples_to_tuples(input_filepath, output_filepath=None):

    tuples = []
    with open(input_filepath, 'r') as inp:
        for line in inp:
            triples = line.strip().split('\t')
            tuples.append("{}\t{}\n".format(triples[0], triples[1]))
            tuples.append("{}\t{}\n".format(triples[1], triples[2]))
    
    if output_filepath is None:
        return tuples
    else:
        with open(output_filepath, 'w') as out:
            out.writelines(tuples)

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
# Gets the vocabulary dictionary and returns it
# This fails if the dictionary do not exist yet.
###
def get_vocabulary():
    return pickle.load(open(VOCABULARY_FILEPATH, 'rb'))

###
# Gets or creates a vocabulary based on vocabulary size
###
def get_or_create_vocabulary():

    try:
        vocabulary = get_vocabulary()
    except:
        vocabulary = {}
        train_file = open(TRAINING_TUPLES_FILEPATH)

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

        if not os.path.exists(os.path.dirname(VOCABULARY_FILEPATH)):
            try:
                os.makedirs(os.path.dirname(VOCABULARY_FILEPATH ))
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

        pickle.dump(vocabulary, open(VOCABULARY_FILEPATH, 'wb'))
        train_file.close()
    return vocabulary

###
# Gets word_2_index and index_2_word dictionaries and returns them
# This fails if the dictionaries do not exist yet.
###
def get_w2i_i2w_dicts():
        return pickle.load(open(W2I_FILEPATH, 'rb')), pickle.load(open(I2W_FILEPATH, 'rb'))

###
# Creates word_2_index and index_2_word dictionaries and returns them
###
def get_or_create_dicts_from_train_data():

    try:
        return get_w2i_i2w_dicts()
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

        if not os.path.exists(os.path.dirname(W2I_FILEPATH)):
            try:
                os.makedirs(os.path.dirname(W2I_FILEPATH))
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise
        if not os.path.exists(os.path.dirname(I2W_FILEPATH)):
            try:
                os.makedirs(os.path.dirname(I2W_FILEPATH))
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

        pickle.dump(word_2_index, open(W2I_FILEPATH, 'wb'))
        pickle.dump(index_2_word, open(I2W_FILEPATH, 'wb'))

    return word_2_index, index_2_word


##
# Applies the word-2-index conversion to a corpus of tuples.
# 
# sentenceStringList:   List of interaction strings. Each interaction consists of two
#                       sentences delimited by '\t'.
# vocabulary:           The vocabulary used to create the w2i dictionary
# w2i_dict:             A dictionary having words as keys and the corresponding
#                       index as a value.
##
def apply_w2i_to_corpus_tuples(interactionStringList, vocabulary, w2i_dict):

        def apply_w2i_to_word(word):
            if word in vocabulary:
                return w2i_dict[word]
            else:
                return w2i_dict[UNK_TOKEN]

        def apply_w2i_to_sentence(sentence_string):
            return list(map(apply_w2i_to_word, sentence_string.split()))


        tuples =map(lambda line: line.strip().split('\t'), interactionStringList)
        input_sentences, answer_sentences = zip(*tuples)
        encoder_inputs = list(map(apply_w2i_to_sentence, input_sentences))
        decoder_inputs = list(map(apply_w2i_to_sentence, answer_sentences))

        return encoder_inputs, decoder_inputs

###
# Returns data by type (train, eval), together with the word_2_index and index_2_word dicts
###
def get_data_by_type(t):

    if t=='train':
        filename = TRAINING_TUPLES_FILEPATH

        if not os.path.isfile(filename):
            triples_to_tuples(TRAINING_FILEPATH, filename)
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

        with open(filename, 'r') as tuples_input:
            lines = tuples_input.readlines()
            encoder_inputs, decoder_inputs = apply_w2i_to_corpus_tuples(lines, vocabulary, word_2_index)

    return encoder_inputs, decoder_inputs, word_2_index, index_2_word


###
# Custom function for bucketing
###
def bucket_by_sequence_length(enc_inputs, dec_inputs, batch_size, sort_data=True, shuffle_batches=True):

    assert len(enc_inputs) == len(dec_inputs)

    if sort_data:
        enc_dec = zip(enc_inputs, dec_inputs)
        sorted_enc_dec_pairs = sorted(enc_dec, key=lambda inputs: (len(inputs[0]), len(inputs[1])))

        enc_inputs, dec_inputs = zip(*sorted_enc_dec_pairs)
    # else we keep the data unsorted
    
    num_batches = ceil(len(enc_inputs) / batch_size)    

    all_batches = []
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
        all_batches.append((encoder_batch, encoder_sequence_lengths, decoder_batch, decoder_sequence_lengths))

    if shuffle_batches:
        shuffle(all_batches)
    for i in range(num_batches):
        yield all_batches[i]


def copy_config(to):
    copyfile("./config.py", os.path.join(to, "config.py"))