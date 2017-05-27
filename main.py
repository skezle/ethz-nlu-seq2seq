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
        print('main.py -n <num_cores> -x <experiment>')
        print('num_cores = Number of cores requested from the cluster. Set to -1 to leave unset')
        print('experiment = experiment setup that should be executed. e.g \'baseline\'')

    num_cores = -1
    experiment = ""
    # Command line argument handling
    try:
        opts, args = getopt.getopt(argv,"n:x:",["num_cores=", "experiment="])
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
                              attention=False,
                              debug=False)
    assert model != None
    enc_inputs, dec_inputs, word_2_index, index_2_word = get_data_by_type('train')
    # Materialize validation data
    validation_enc_inputs, validation_dec_inputs, _, _ = get_data_by_type('eval')
    validation_data = list(bucket_by_sequence_length(validation_enc_inputs, validation_dec_inputs, conf.batch_size))
    
    print("Training network")
    with tf.Session(config=configProto) as sess:
        global_step = 1

        saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=2)

        # Init Tensorboard summaries. This will save Tensorboard information into a different folder at each run.
        timestamp = '{0:%Y-%m-%d_%H-%M-%S}'.format(datetime.datetime.now())
        train_logfolderPath= os.path.join(conf.log_directory, "{}-training-{}".format(experiment, timestamp))
        train_writer = tf.summary.FileWriter(train_logfolderPath, graph=tf.get_default_graph())
        validation_writer = tf.summary.FileWriter("{}{}-validation-{}".format(conf.log_directory, experiment, timestamp), graph=tf.get_default_graph())

        sess.run(tf.global_variables_initializer())

        for i in range(conf.num_epochs):
            batch_in_epoch = 0
            print("Training epoch {}".format(i))
            for data_batch, data_sentence_lengths, label_batch, label_sentence_lengths in tqdm(bucket_by_sequence_length(enc_inputs, dec_inputs, conf.batch_size), total = ceil(len(enc_inputs) / conf.batch_size)):
                batch_in_epoch += 1
                feed_dict = model.make_train_inputs(data_batch, data_sentence_lengths, label_batch, label_sentence_lengths)
                _, train_summary = sess.run([model.train_op, model.summary_op], feed_dict)
                train_writer.add_summary(train_summary, global_step)

                if global_step % conf.validation_summary_frequency == 0:#
                    # Randomly choose a batch from the validation dataset and use it for loss calculation
                    vali_data_batch, vali_data_sentence_lengths, vali_label_batch, vali_label_sentence_lengths = choice(validation_data)
                    validation_feed_dict = model.make_train_inputs(vali_data_batch, vali_data_sentence_lengths, vali_label_batch, vali_label_sentence_lengths)
                    validation_summary = sess.run(model.summary_op, validation_feed_dict)
                    validation_writer.add_summary(validation_summary, global_step)

                if global_step % conf.checkpoint_frequency == 0 :
                    saver.save(sess, os.path.join(train_logfolderPath, "{}-{}-ep{}.ckpt".format(experiment, timestamp, i)), global_step=global_step)
                global_step += 1


                print('  minibatch loss: {}'.format(sess.run(model.loss, feed_dict)))
                for j, (e_in, dec_tar, dec_train_inputs, dec_train_targets, dt_pred) in enumerate(zip(
                        feed_dict[model.encoder_inputs].T, feed_dict[model.decoder_targets].T,
                        sess.run(model.decoder_train_inputs, feed_dict).T,
                        sess.run(model.decoder_train_targets, feed_dict).T,
                        sess.run(model.decoder_prediction_train, feed_dict).T
                )):
                    # print('  sample {}:'.format(j + 1))
                    # print('    enc input           > {}'.format(e_in))
                    # print('    dec targets         > {}'.format(dec_tar))
                    # print('    dec train inputs    > {}'.format(dec_train_inputs))
                    # print('    dec train targets   > {}'.format(dec_train_targets))
                    # print('    dec train predicted > {}'.format(dt_pred))

                    print('  sample {}:'.format(j + 1))
                    print('    enc input           > {}'.format(" ".join(map(lambda x: index_2_word[x], e_in))))
                    print('    dec targets         > {}'.format(" ".join(map(lambda x: index_2_word[x], dec_tar))))
                    print('    dec train inputs    > {}'.format(
                        " ".join(map(lambda x: index_2_word[x], dec_train_inputs))))
                    print('    dec train targets   > {}'.format(
                        " ".join(map(lambda x: index_2_word[x], dec_train_targets))))
                    print('    dec train predicted > {}'.format(" ".join(map(lambda x: index_2_word[x], dt_pred))))
                    if j >= 0:
                        break
                print()

                if global_step == 150:
                    break




if __name__ == "__main__":
    mainFunc(sys.argv[1:])
