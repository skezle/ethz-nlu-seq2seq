import sys, getopt, datetime
import tensorflow as tf
from math import ceil
from random import choice
from tqdm import tqdm
from data_utility import *
from baseline import BaselineModel
from antilm.antilm import construct_lm_logits, construct_lm_logits_batch
###
# Graph execution
###
def mainFunc(argv):
    def printUsage():
        print('predict.py -n <num_cores> -x <experiment> -o <output file> -c <checkpoint>')
        print('num_cores = Number of cores requested from the cluster. Set to -1 to leave unset')
        print('experiment = experiment setup that should be executed. e.g \'baseline\'')
        print('checkpoint = Path to the checkpoint to load parameters from. e.g. \'./logs/baseline-ep4-500\'')
        print('output = where to write the prediction outputs to. e.g \'./predictions.out\'')

    def maptoword(sentence):
        return " ".join(map(lambda x: index_2_word[x], sentence))

    num_cores = -1
    experiment = ""
    checkpoint_filepath = ""
    output_filepath = ""
    # Command line argument handling
    try:
        opts, args = getopt.getopt(argv, "n:x:c:o:", ["num_cores=", "experiment=", "checkpoint=", "output="])
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
            if arg in ("baseline", "attention"):
                experiment = arg
            else:
                printUsage()
                sys.exit(2)
        elif opt in ("-o", "--output"):
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

    print("Executing experiment {} with {} CPU cores".format(experiment, num_cores))
    print("Loading checkpoint from {}".format(checkpoint_filepath))
    if num_cores != -1:
        # We set the op_parallelism_threads in the ConfigProto and pass it to the TensorFlow session
        configProto = tf.ConfigProto(inter_op_parallelism_threads=num_cores,
                                     intra_op_parallelism_threads=num_cores)
    else:
        configProto = tf.ConfigProto()

    print("Initializing model")
    model = None
    if experiment == "baseline":
        model = BaselineModel(vocab_size=conf.vocabulary_size,
                              embedding_size=conf.word_embedding_size,
                              bidirectional=conf.bidirectional_encoder,
                              attention=False,
                              dropout=conf.use_dropout,
                              num_layers=conf.num_layers,
                              is_training=False)

    elif experiment == "attention":
        model = BaselineModel(vocab_size=conf.vocabulary_size,
                              embedding_size=conf.word_embedding_size,
                              bidirectional=conf.bidirectional_encoder,
                              attention=True,
                              dropout=conf.use_dropout,
                              num_layers=conf.num_layers,
                              is_training=False)

    assert model != None
    # Materialize validation data
    validation_enc_inputs, validation_dec_inputs, word_2_index, index_2_word = get_data_by_type('eval')

    validation_input_lengths = set(map(lambda x: len(x), validation_enc_inputs))
    with tf.Session(config=configProto) as sess:
        global_step = 1
        sent_count = 1
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, checkpoint_filepath)

        print("Constructing language model")
        lm_logits_dict = construct_lm_logits(sess, model, validation_input_lengths)

        print("Using network to predict sentences..")
        with open(output_filepath, 'w') as out:
            for data_batch, data_sentence_lengths, _, label_targets_batch, label_sentence_lengths in tqdm(
                    bucket_by_sequence_length(validation_enc_inputs, validation_dec_inputs, conf.batch_size, sort_data=False, shuffle_batches=False, filter_long_sent=False),
                    total=ceil(len(validation_enc_inputs) / conf.batch_size)):

                lm_logits_batch = construct_lm_logits_batch(lm_logits_dict, data_sentence_lengths)
                feed_dict = model.make_inference_inputs(data_batch, data_sentence_lengths, lm_logits_batch)

                predictions = sess.run(model.decoder_prediction_inference, feed_dict)
                
                reversed_truncated_data = truncate_after_eos(data_batch)
                truncated_data = undo_input_reversal(reversed_truncated_data)
                truncated_labels = truncate_after_eos(label_targets_batch)
                truncated_predictions = truncate_after_eos(predictions)
                for enc, target, pred in zip(map(maptoword, truncated_data), map(maptoword, truncated_labels), map(maptoword, truncated_predictions)):
                    print("{}. Input:        {}".format(sent_count, enc), file=out)
                    print("{}. Ground Truth: {}".format(sent_count, target), file=out)
                    print("{}. Prediction:   {}".format(sent_count, pred), file=out)
                    sent_count += 1

                global_step += 1

if __name__ == "__main__":
    mainFunc(sys.argv[1:])
