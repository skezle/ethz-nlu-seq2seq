import tensorflow as tf
import nltk
import itertools
import pickle
import os

START_TOKEN = "<bos>"
END_TOKEN = "<eos>"
UNKNOWN_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"

def read_data(filename):
    print("Reading data from {}".format(filename))
    f = tf.gfile.GFile(filename, "r")
    return f.read().decode("utf-8").split("\n")

def preprocess_data(filename, max_sentence_length, vocabulary_size):
    if (os.path.exists('padded_sentences.pickle') and
       os.path.exists('index_2_word.pickle') and
       os.path.exists('word_2_index.pickle') and
       os.path.exists('vocabulary.pickle')):
        print("Found already preprocessed data. Will load these...")
        print("Reading padded_sentences.pickle")
        with open('padded_sentences.pickle', 'rb') as f:
            padded_sentence_indices = pickle.load(f)
        print("Reading index_2_word.pickle")
        with open('index_2_word.pickle', 'rb') as f:
            index_2_word = pickle.load(f)
        print("Reading word_2_index.pickle")
        with open('word_2_index.pickle', 'rb') as f:
            word_2_index = pickle.load(f)
        print("Reading vocabulary.pickle")
        with open('vocabulary.pickle', 'rb') as f:
            wordcount_dictionary = pickle.load(f)
    else :
        sentence_list = read_data(filename)
        print("Preprocessing...")
        sentences = ["%s %s %s" % (START_TOKEN, x, END_TOKEN) for x in sentence_list]
        filtered_sentences = [x.split() for x in sentences if len(x.split()) <= max_sentence_length]
        padded_sentences = [(x + [PAD_TOKEN] * (max_sentence_length - len(x))) for x in filtered_sentences]
        print("Read {} sentences in total, keeping {} sentences shorter than {} - 1 tokens".format(len(sentences), len(filtered_sentences), max_sentence_length))
        word_freq = nltk.FreqDist(itertools.chain(*padded_sentences))
        print("Found {} unique words in corpus".format(len(word_freq.items())))

        vocab = word_freq.most_common(vocabulary_size - 1)

        wordcount_dictionary = dict(vocab)
        wordcount_dictionary[UNKNOWN_TOKEN] = 1

        index_2_word = dict(enumerate(wordcount_dictionary.keys()))
        word_2_index = dict(zip(index_2_word.values(), index_2_word.keys()))

        print("Created index_2_word and word_2_index dictionaries with size {} and {}".format(len(index_2_word), len(word_2_index)))

        for i, sent in enumerate(padded_sentences):
            padded_sentences[i] = [w if w in wordcount_dictionary else UNKNOWN_TOKEN for w in sent]
        print("Padded all sentences to {} tokens".format(max_sentence_length))

        for w in padded_sentences[0]:
            print("word is {0} and index is {1}".format(w, word_2_index[w]))

        print("Converting all padded sentences to their word indices")
        padded_sentence_indices = [[word_2_index[w] for w in sentence] for sentence in padded_sentences]

        print("Dumping padded_sentences.pickle")
        if not os.path.exists('padded_sentences.pickle') :
            with open('padded_sentences.pickle', 'wb') as f:
                pickle.dump(padded_sentence_indices, f)

        print("Dumping word_2_index.pickle")
        if not os.path.exists('word_2_index.pickle') :
            with open('word_2_index.pickle', 'wb') as f:
                pickle.dump(word_2_index, f)

        print("Dumping index_2_word.pickle")
        if not os.path.exists('index_2_word.pickle') :
            with open('index_2_word.pickle', 'wb') as f:
                pickle.dump(index_2_word, f)

        print("Dumpping vocabulary.pickle")
        if not os.path.exists('vocabulary.pickle') :
            with open('vocabulary.pickle', 'wb') as f:
                pickle.dump(wordcount_dictionary, f)

    return padded_sentence_indices, index_2_word, word_2_index, wordcount_dictionary
