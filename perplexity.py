import sys, getopt, datetime
import tensorflow as tf
from data_utility import get_data_by_type, triples_to_tuples, apply_w2i_to_corpus_tuples, get_vocabulary, \
    get_w2i_i2w_dicts, bucket_by_sequence_length, END_TOKEN_INDEX
from baseline import BaselineModel
from config import Config as conf
import numpy as np

###
# Graph execution
###
def mainFunc(argv):
    def printUsage():
        print('perplexity.py -n <num_cores> -x <experiment> -i <input file> -c <checkpoint>')
        print('num_cores = Number of cores requested from the cluster. Set to -1 to leave unset')
        print('experiment = experiment setup that should be executed. e.g \'baseline\'')
        print('input = what dialogs to predict from. e.g \'./Dialog_Triples.txt\'')
        print('checkpoint = Path to the checkpoint to load parameters from. e.g. \'./logs/baseline-ep4-500\'')
        

    def maptoword(sentence):
        return " ".join(map(lambda x: index_2_word[x], sentence)) + '\n'

    num_cores = -1
    experiment = ""
    checkpoint_filepath = ""
    input_filepath = ""
    # Command line argument handling
    try:
        opts, args = getopt.getopt(argv, "n:x:c:i:", ["num_cores=", "experiment=", "checkpoint=", "input="])
    except getopt.GetoptError:
        printUsage()
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            printUsage()
            sys.exit()
        elif opt in ("-n", "--num_cores"):
            num_cores = int(arg)
        elif opt in ("-x", "--experiment"):
            if arg in ("baseline"):
                experiment = arg
            else:
                printUsage()
                sys.exit(2) 
        elif opt in ("-i", "--input"):
            if arg != "":
                input_filepath = arg
            else:
                printUsage()
                sys.exit(2)
        elif opt in ("-c", "--checkpoint"):
            if arg != "":
                checkpoint_filepath = arg
            else:
                printUsage()
                sys.exit(2)

    if num_cores != -1:
        # We set the op_parallelism_threads in the ConfigProto and pass it to the TensorFlow session
        configProto = tf.ConfigProto(inter_op_parallelism_threads=num_cores,
                                     intra_op_parallelism_threads=num_cores)
    else:
        configProto = tf.ConfigProto()

    model = None
    if experiment == "baseline":
        model = BaselineModel(encoder_cell=conf.encoder_cell,
                              decoder_cell=conf.decoder_cell,
                              vocab_size=conf.vocabulary_size,
                              embedding_size=conf.word_embedding_size,
                              bidirectional=False,
                              attention=False,
                              dropout=conf.use_dropout,
                              num_layers=conf.num_layers)
    assert model != None

    with tf.Session(config=configProto) as sess:
        global_step = 1

        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, checkpoint_filepath)

        tuples = triples_to_tuples(input_filepath)
        w2i, _ = get_w2i_i2w_dicts()
        vocabulary = get_vocabulary()
        enc_inputs, dec_inputs = apply_w2i_to_corpus_tuples(tuples, vocabulary, w2i)

        is_first_tuple = True
        for data_batch, data_sentence_lengths, label_inputs_batch, label_targets_batch, label_sentence_lengths in bucket_by_sequence_length(enc_inputs, dec_inputs, conf.batch_size, sort_data=False, shuffle_batches=False, filter_long_sent=False):
            feed_dict = model.make_train_inputs(data_batch, data_sentence_lengths, label_inputs_batch, label_targets_batch, label_sentence_lengths)

            softmax_predictions = sess.run(model.decoder_softmax_train, feed_dict)
            # softmax_predictions.shape = (max_sentence_len, batch_size, vocabulary_size)

            # Perplexity calculation
            for sentID in range(len(label_sentence_lengths)): # Loop 
                word_probs = []
                # As long as we havent reached the maximum sentence length or seen the <eos>
                for wordID in range(label_sentence_lengths[sentID]):
                    ground_truth_word_index = label_batch[wordID, sentID]
                    prob = softmax_predictions[wordID,sentID,ground_truth_word_index]
                    word_probs.append(prob)

                # Our bucketing function doesn't add <eos>, so we
                # manually add the probability of <eos> here.
                word_probs.append(
                    softmax_predictions[
                            label_sentence_lengths[sentID], 
                            sentID, 
                            END_TOKEN_INDEX])
                log_probs = np.log(word_probs)

                perplexity = 2**(-1.0*log_probs.mean())
                
                if is_first_tuple:
                    print(perplexity, end=' ')
                    is_first_tuple = False
                else:
                    print(perplexity)
                    is_first_tuple = True
            
            global_step += 1

if __name__ == "__main__":
    mainFunc(sys.argv[1:])
