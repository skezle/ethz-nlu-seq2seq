import datetime
import getopt
import sys
import os.path
from word2vec.word2vec import *
from random import choice

import tensorflow as tf
from tqdm import tqdm

from baseline import BaselineModel
from data_utility import *

from word2vec.load_embeddings import load_embedding


###
# Graph execution
###
def mainFunc(argv):
    def printUsage():
        print('main.py -n <num_cores> -x <experiment>')
        print('num_cores = Number of cores requested from the cluster. Set to -1 to leave unset')
        print('experiment = experiment setup that should be executed. e.g \'baseline\'')
        print('tag = optional tag or name to distinguish the runs, e.g. \'bidirect3layers\' ')

    num_cores = -1
    experiment = ""
    tag = None
    # Command line argument handling
    try:
        opts, args = getopt.getopt(argv,"n:x:t:",["num_cores=", "experiment=", "tag="])
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
        elif opt in ("-t", "--tag"):
            tag = arg

    print("Executing experiment {} with {} CPU cores".format(experiment, num_cores))
    if num_cores != -1:
        # We set the op_parallelism_threads in the ConfigProto and pass it to the TensorFlow session
        configProto = tf.ConfigProto(inter_op_parallelism_threads=num_cores,
                        intra_op_parallelism_threads=num_cores)
    else:
        configProto = tf.ConfigProto()

    print("Initializing model")
    model = None
    if experiment == "baseline":
        model = BaselineModel(encoder_cell=conf.encoder_cell,
                              decoder_cell=conf.decoder_cell,
                              vocab_size=conf.vocabulary_size,
                              embedding_size=conf.word_embedding_size,
                              bidirectional=False,
                              attention=False)
    assert model != None
    enc_inputs, dec_inputs, word_2_index, index_2_word = get_data_by_type('train')
    # Materialize validation data
    validation_enc_inputs, validation_dec_inputs, _, _ = get_data_by_type('eval')
    validation_data = list(bucket_by_sequence_length(validation_enc_inputs, validation_dec_inputs, conf.batch_size, filter_long_sent=False))
    
    print("Starting TensorFlow session")
    with tf.Session(config=configProto) as sess:
        global_step = 1

        saver = tf.train.Saver(max_to_keep=3, keep_checkpoint_every_n_hours=4)

        # Init Tensorboard summaries. This will save Tensorboard information into a different folder at each run.
        timestamp = '{0:%Y-%m-%d_%H-%M-%S}'.format(datetime.datetime.now())
        tag_string = ""
        if tag is not None:
            tag_string= "-" + tag
        train_logfolderPath = os.path.join(conf.log_directory, "{}{}-training-{}".format(experiment, tag_string, timestamp))
        train_writer        = tf.summary.FileWriter(train_logfolderPath, graph=tf.get_default_graph())
        validation_writer   = tf.summary.FileWriter("{}{}{}-validation-{}".format(conf.log_directory, experiment, tag_string, timestamp), graph=tf.get_default_graph())

        copy_config(train_logfolderPath) # Copies the current config.py to the log directory
        sess.run(tf.global_variables_initializer())

        if conf.use_word2vec:
            print("Using word2vec embeddings")
            if not os.path.isfile(conf.word2vec_path):
                train_embeddings(save_to_path=conf.word2vec_path,
                                          embedding_size=conf.word_embedding_size,
                                          minimal_frequency=conf.word2vec_min_word_freq,
                                          train_path=TRAINING_FILEPATH,
                                          validation_path=VALIDATION_FILEPATH,
                                          num_workers=conf.word2vec_workers_count)
            print("Loading word2vec embeddings")
            load_embedding(sess,
                           get_or_create_vocabulary(),
                           model.embedding_matrix,
                           conf.word2vec_path,
                           conf.word_embedding_size,
                           conf.vocabulary_size)
        sess.graph.finalize()
        print("Starting training")
        for i in range(conf.num_epochs):
            print("Training epoch {}".format(i))
            for data_batch, data_sentence_lengths, label_batch, label_sentence_lengths in tqdm(bucket_by_sequence_length(enc_inputs, dec_inputs, conf.batch_size), total = ceil(len(enc_inputs) / conf.batch_size)):
                feed_dict = model.make_train_inputs(data_batch, data_sentence_lengths, label_batch, label_sentence_lengths)
                run_options = None
                run_metadata = None
                if global_step % conf.trace_frequency == 0:
                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()
                _, train_summary = sess.run([model.train_op, model.summary_op], feed_dict, options=run_options, run_metadata=run_metadata)
                if global_step % conf.trace_frequency == 0:
                    train_writer.add_run_metadata(run_metadata, "step{}".format(global_step))
                train_writer.add_summary(train_summary, global_step)

                if global_step % conf.validation_summary_frequency == 0:#
                    # Randomly choose a batch from the validation dataset and use it for loss calculation
                    vali_data_batch, vali_data_sentence_lengths, vali_label_batch, vali_label_sentence_lengths = choice(validation_data)
                    validation_feed_dict = model.make_train_inputs(vali_data_batch, vali_data_sentence_lengths, vali_label_batch, vali_label_sentence_lengths)
                    validation_summary = sess.run(model.summary_op, validation_feed_dict)
                    validation_writer.add_summary(validation_summary, global_step)

                if global_step % conf.checkpoint_frequency == 0 :
                    saver.save(sess, os.path.join(train_logfolderPath, "{}{}-{}-ep{}.ckpt".format(experiment, tag_string, timestamp, i)), global_step=global_step)
                global_step += 1

        saver.save(sess, os.path.join(train_logfolderPath, "{}{}-{}-ep{}-final.ckpt".format(experiment, tag_string, timestamp, conf.num_epochs)))
        print("Done with training for {} epochs".format(conf.num_epochs))

if __name__ == "__main__":
    mainFunc(sys.argv[1:])
