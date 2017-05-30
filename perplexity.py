import sys, getopt, datetime
import tensorflow as tf
from math import ceil
from random import choice
from tqdm import tqdm
from data_utility import *
from baseline import BaselineModel

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
                output_filepath = arg
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
                              attention=False)
    assert model != None
    # Materialize validation data
    validation_enc_inputs, _, word_2_index, index_2_word = get_data_by_type('eval')

    with tf.Session(config=configProto) as sess:
        global_step = 1

        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, checkpoint_filepath)

        with open(output_filepath, 'w') as out:
            for data_batch, data_sentence_lengths, label_batch, label_sentence_lengths in tqdm(
                    bucket_by_sequence_length(validation_enc_inputs, _, conf.batch_size, sort_data=False, shuffle_batches=False),
                    total=ceil(len(validation_enc_inputs) / conf.batch_size)):

                feed_dict = model.make_inference_inputs(data_batch, data_sentence_lengths)

                predictions = sess.run(model.decoder_prediction_inference, feed_dict).T

                out.writelines(map(maptoword, predictions))
                
                global_step += 1

if __name__ == "__main__":
    mainFunc(sys.argv[1:])
