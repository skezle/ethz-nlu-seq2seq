import pickle
import os.path
import operator
import cornell_loading
import numpy as np
from math import ceil
from config import Config as conf
from random import shuffle
from shutil import copyfile
import extracting_genres

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
# Creates an output file by transforming the original triples file to a tuples file
# preserving the order of the dialogs. e.g. for a dialog consisting of sent1 -- sent2 -- sent3
# the generated tuple "sent2 -- sent3" will directly follow "sent1 -- sent2"
#
# If output_filepath is None, then the output tuples will not be written to disk,
# but returned
###

def triples_to_tuples(input_filepath, output_filepath):
    print("Converting triples from {} to tuples..".format(input_filepath))
    f = open(input_filepath, 'r')
    f1 = open(output_filepath, 'w')

    if conf.use_genres:
        if input_filepath == TRAINING_FILEPATH:
            _, _, matching, _ = extracting_genres.extract_base_dataset_genres(True, False)
        if input_filepath == VALIDATION_FILEPATH:
            _, _, _, matching = extracting_genres.extract_base_dataset_genres(False, True)

    i = 0
    for line in f:
        triples = line.strip().split('\t')
        if conf.use_genres:
            f1.write("{} {}\t{}\n".format(matching[i], triples[0], triples[1]))
            f1.write("{} {}\t{}\n".format(matching[i], triples[1], triples[2]))
        else:
            f1.write("{}\t{}\n".format(triples[0], triples[1]))
            f1.write("{}\t{}\n".format(triples[1], triples[2]))
        i = i+1

    f.close()
    f1.close()
    if input_filepath == TRAINING_FILEPATH and conf.use_CORNELL_for_training:
        merge(output_filepath, conf.CORNELL_TUPLES_PATH, conf.both_datasets_tuples_filepath)


def merge(base_dataset_tuples_filepath, cornell_tuples_filepath, output_filepath):
    if conf.use_CORNELL_for_training:
        numlines = 0
        f = open(base_dataset_tuples_filepath, 'r')
        f1 = open(output_filepath, 'w')
        print("Merging base dataset with Cornell: loading base dataset..")
        for line in f:
            f1.write(line)
            numlines = numlines + 1
        f.close()
        print("\tNumber of tuples loaded from base dataset: {}".format(numlines))
        print("Merging base dataset with Cornell: loading Cornell dataset..")
        if not os.path.isfile(cornell_tuples_filepath):
            matching = cornell_loading.create_Cornell_tuples(conf.CORNELL_lines_path, conf.CORNELL_conversations_path, conf.CORNELL_TUPLES_PATH)
        f2 = open(cornell_tuples_filepath, 'r')
        i = 0
        for line in f2:
            couples = line.strip().split('\t')
            if len(couples) > 2:
                k = 1
                while len(couples[k]) <= 0 and k <= len(couples):
                    k = k + 1
                if conf.use_genres:
                    f1.write("{} {}\t{}\n".format(matching[i], couples[0], couples[k]))
                else:
                    f1.write("{}\t{}\n".format(couples[0], couples[k]))
                numlines = numlines + 1
            else:
                if conf.use_genres:
                    f1.write("{} {}".format(matching[i], line))
                else:
                    f1.write(line)
                numlines = numlines + 1
            i = i+1
        f2.close()
        f1.close()
        print("\tTotal number of dumped lines: {}".format(numlines))

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
    print("Getting vocabulary..")
    try:
        vocabulary = get_vocabulary()
    except:
        print("Building vocabulary..")
        vocabulary = {}

        if conf.use_CORNELL_for_training:
            train_file = open(conf.both_datasets_tuples_filepath)
        else:
            train_file = open(TRAINING_TUPLES_FILEPATH)

        for line in train_file:
            conversation = line.strip().split()
            for word in conversation:
                vocabulary[word] = vocabulary.get(word, 0) + 1

        sorted_vocab = sorted(vocabulary.items(), key=operator.itemgetter(1), reverse=True)
        print("Total length of vocabulary: {}".format(len(sorted_vocab)))

        sorted_vocab = sorted_vocab[:conf.vocabulary_size-4]
        vocabulary = dict(sorted_vocab)

        vocabulary[START_TOKEN] = 1
        vocabulary[END_TOKEN] = 1
        vocabulary[UNK_TOKEN] = 1
        vocabulary[PAD_TOKEN] = 1

        pickle.dump(vocabulary, open(VOCABULARY_FILEPATH, 'wb'))
        train_file.close()
        print("Vocabulary pickled!")
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
    print("Getting word2index and index2word..")
    try:
        return get_w2i_i2w_dicts()
    except:
        print("Building word2index and index2word")
        filename = TRAINING_TUPLES_FILEPATH
        if not os.path.isfile(filename):
            triples_to_tuples(TRAINING_FILEPATH, filename)

        if conf.use_CORNELL_for_training:
            f = open(conf.both_datasets_tuples_filepath, 'r')
        else:
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
        print("word2index and index2word pickled!")

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


        tuples = map(lambda line: line.strip().split('\t'), interactionStringList)
        input_sentences, answer_sentences = zip(*tuples)
        encoder_inputs = list(map(apply_w2i_to_sentence, input_sentences))
        decoder_inputs = list(map(apply_w2i_to_sentence, answer_sentences))

        return encoder_inputs, decoder_inputs

###
# Returns data by type (train, eval), together with the word_2_index and index_2_word dicts
###
def get_data_by_type(t):
    if t == 'train':
        filename = TRAINING_TUPLES_FILEPATH
        if conf.use_CORNELL_for_training:
            filename = conf.both_datasets_tuples_filepath
    elif t == 'eval':
        filename = VALIDATION_TUPLES_FILEPATH
        if not os.path.isfile(filename):
            triples_to_tuples(VALIDATION_FILEPATH, filename)
    else:
        print('Type must be "train" or "eval".')
        return

    word_2_index, index_2_word = get_or_create_dicts_from_train_data()
    vocabulary = get_or_create_vocabulary()

    try:
        print("Getting encoder and decoder inputs..")
        encoder_inputs = pickle.load(open(ENCODER_INPUT_FILEPATH, 'rb'))
        decoder_inputs = pickle.load(open(DECODER_INPUT_FILEPATH, 'rb'))
    except:
        print("Building encoder and decoder inputs..")

        with open(filename, 'r') as tuples_input:
            lines = tuples_input.readlines()
            encoder_inputs, decoder_inputs = apply_w2i_to_corpus_tuples(lines, vocabulary, word_2_index)

        # pickle.dump(encoder_inputs, open(ENCODER_INPUT_FILEPATH, 'wb'))
        # pickle.dump(decoder_inputs, open(DECODER_INPUT_FILEPATH, 'wb'))
        # print("encoder and decoder inputs pickled!")

    return encoder_inputs, decoder_inputs, word_2_index, index_2_word

###
# Custom function for bucketing
###
def bucket_by_sequence_length(enc_inputs, dec_inputs, batch_size, sort_data=True, shuffle_batches=True, filter_long_sent=True):

    assert len(enc_inputs) == len(dec_inputs)
    enc_dec = list(zip(enc_inputs, dec_inputs))     
    if filter_long_sent:
        enc_dec = list(filter(lambda tup: len(tup[0]) < conf.input_sentence_max_length or len(tup[1]) < conf.input_sentence_max_length, enc_dec))
    
    if sort_data:
        enc_dec = zip(enc_inputs, dec_inputs)
        enc_dec = sorted(enc_dec, key=lambda inputs: (len(inputs[0]), len(inputs[1])))

    enc_inputs, dec_inputs = zip(*enc_dec)
    assert len(enc_inputs) == len(dec_inputs)
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
        encoder_batch = np.array(encoder_batch).transpose()
        decoder_sequence_lengths = [len(sentence) 
                                    for sentence
                                    in dec_inputs[batch_num*batch_size:(batch_num+1)*batch_size]]
        max_len_dec = max(decoder_sequence_lengths)
        decoder_batch = [sentence + ([PAD_TOKEN_INDEX] * (max_len_dec - decoder_sequence_lengths[i]))
                         for i, sentence
                         in enumerate(dec_inputs[batch_num*batch_size:(batch_num+1)*batch_size])]

        decoder_batch = np.array(decoder_batch).transpose()
        all_batches.append((encoder_batch, encoder_sequence_lengths, decoder_batch, decoder_sequence_lengths))

    if shuffle_batches:
        shuffle(all_batches)
    for i in range(num_batches):
        yield all_batches[i]


def copy_config(to):
    copyfile("./config.py", os.path.join(to, "config.py"))


def truncate_sentence(sent):
    idxArr = np.where(sent == END_TOKEN_INDEX)[0]
    if idxArr.size == 0:
        return sent
    else:
        return sent[:idxArr[0]+1]


def truncate_after_eos(sentence_list):
    return list(map(lambda sent: truncate_sentence(sent), sentence_list))









# encoder_inputs = []
#         decoder_inputs = []
#
#         f = open(filename, 'r')
#         for line in f:
#             conversation = line.strip().split('\t')
#
#             encoder_input = []
#             for word in conversation[0].split():
#                 if word in vocabulary:
#                     encoder_input.append(word_2_index[word])
#                 else:
#                     encoder_input.append(word_2_index[UNK_TOKEN])
#             # encoder_input.append(word_2_index[END_TOKEN])  # DO WE NEED EOS?
#             encoder_inputs.append(encoder_input)
#
#             decoder_input = []
#             #print(line)
#             for word in conversation[1].split():
#                 if word in vocabulary:
#                     #print(conversation[1])
#                     decoder_input.append(word_2_index[word])
#                 else:
#                     decoder_input.append(word_2_index[UNK_TOKEN])
#             # decoder_input.append(word_2_index[END_TOKEN])
#             decoder_inputs.append(decoder_input)
# =======