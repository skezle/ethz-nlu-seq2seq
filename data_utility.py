import pickle
import os.path
import operator

START_TOKEN = "<bos>"
END_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"
VOCABULARY_SIZE = 35000


def triples_to_touples(filename):
    filename = filename.split('.')

    f = open(filename[0] + '.' + filename[1], 'r')
    f1 = open(filename[0] + '_processed.' + filename[1], 'w')

    for line in f:
        triples = line.strip().split('\t')

        f1.write("{}\t{}\n".format(triples[0], triples[1]))
        f1.write("{}\t{}\n".format(triples[1], triples[2]))

    f.close()
    f1.close()


def count_unique_tokens(filename):
    f = open(filename, 'r')
    s = set()

    for line in f:
        for word in line.split():
            s.add(word)

    f.close()
    return len(s)

###
# Gets or creates a vocabulary based on VOCABULARY_SIZE
###

def get_or_create_vocabulary():

    try:
        vocabulary = pickle.load(open('pickled_vars/vocabulary.p', 'rb'))
    except:
        vocabulary = {}
        train_file = open('data/Training_Shuffled_Dataset_tuples.txt')

        for line in train_file:
            conversation = line.strip().split()
            for word in conversation:
                vocabulary[word] = vocabulary.get(word, 0) + 1

        sorted_vocab = sorted(vocabulary.items(), key=operator.itemgetter(1), reverse=True)

        sorted_vocab = sorted_vocab[:VOCABULARY_SIZE-4]
        vocabulary = dict(sorted_vocab)
        vocabulary[START_TOKEN] = 1
        vocabulary[END_TOKEN] = 1
        vocabulary[UNK_TOKEN] = 1
        vocabulary[PAD_TOKEN] = 1

        pickle.dump(vocabulary, open('pickled_vars/vocabulary.p', 'wb'))

    return vocabulary


###
# Creates word_2_index and index_2_word dictionaries
###
def get_or_create_dicts_from_train_data():

    try:
        word_2_index = pickle.load(open('pickled_vars/word_2_index.p', 'rb'))
        index_2_word = pickle.load(open('pickled_vars/index_2_index.p', 'rb'))
    except:
        filename = 'data/Training_Shuffled_Dataset_tuples.txt'

        if not os.path.isfile(filename):
            parts = filename.split('_')
            orig_file = parts[0] + '_' + parts[1] + '_' + parts[2] + '.txt'
            triples_to_touples(orig_file)

        f = open(filename, 'r')

        word_2_index = {START_TOKEN: 0, END_TOKEN: 1, UNK_TOKEN: 2, PAD_TOKEN: 3}
        index_2_word = {0: START_TOKEN, 1: END_TOKEN, 2: UNK_TOKEN, 3: PAD_TOKEN}
        vocabulary = get_or_create_vocabulary()

        index = 4
        for line in f:
            conversation = line.strip().split()
            # print(conversation)
            for word in conversation:
                if word in vocabulary and word not in word_2_index:
                    word_2_index[word] = index
                    index_2_word[index] = word
                    index += 1

        pickle.dump(word_2_index, open('pickled_vars/word_2_index.p', 'wb'))
        pickle.dump(index_2_word, open('pickled_vars/index_2_word.p', 'wb'))

    return word_2_index, index_2_word


###
# Returns data by type (train, eval), together with the word_2_index and index_2_word dicts
###
def get_data_by_type(t):

    if t=='train':
        filename = 'data/Training_Shuffled_Dataset_tuples.txt'
    elif t=='eval':
        filename = 'data/Validation_Shuffled_Dataset_tuples.txt'
    else:
        print('Type must be "train" or "eval".')
        return

    word_2_index, index_2_word = get_or_create_dicts_from_train_data()
    vocabulary = get_or_create_vocabulary()

    try:
        encoder_inputs = pickle.load(open('pickled_vars/encoder_inputs.p', 'rb'))
        decoder_inputs = pickle.load(open('pickled_vars/decoder_inputs.p', 'rb'))
    except:
        encoder_inputs = []
        decoder_inputs = []

        f = open(filename, 'r')
        for line in f:
            conversation = line.strip().split('\t')

            encoder_input = []
            for word in conversation[0].split():
                if word in vocabulary:
                    encoder_input.append(word_2_index[word])
                else:
                    encoder_input.append(word_2_index[UNK_TOKEN])
            encoder_input.append(word_2_index[END_TOKEN])  # DO WE NEED EOS?
            encoder_inputs.append(encoder_input)

            decoder_input = [word_2_index[START_TOKEN]]
            for word in conversation[1].split():
                if word in vocabulary:
                    decoder_input.append(word_2_index[word])
                else:
                    decoder_input.append(word_2_index[UNK_TOKEN])
            decoder_input.append(word_2_index[END_TOKEN])
            decoder_inputs.append(decoder_input)

    return encoder_inputs, decoder_inputs, word_2_index, index_2_word


###
# Custom function for bucketing. Needs to be discussed and finished during the meeting
###
def bucket_by_sequence_length(inputs, batch_size):

    inputs = sorted(inputs, key=len)

    num_batches = len(inputs) / batch_size

    for i in range(num_batches + 1):
        max_len = -1
        batch = []
        for input in inputs[i*batch_size:(i+1)*batch_size]:
            if len(input) > max_len:
                max_len = len(input)

        for input in inputs[i*batch_size:(i+1)*batch_size]:
            input.extend([3] * (max_len - len(input)))
            batch.append(input)

        batch = map(list, zip(*batch))  # Transpose it.

        # print batch
        yield batch

