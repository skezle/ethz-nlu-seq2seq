import tensorflow as tf
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
    num_epochs = NUM_EPOCHS
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
    if experiment == "Baseline":
        model = BaselineModel(encoder_cell=LSTMCell(conf.encoder_cell_size),
                              decoder_cell=LSTMCell(conf.decoder_cell_size),
                              vocab_size=conf.vocabulary_size,
                              embedding_size=conf.word_embedding_size,
                              bidirectional=False,
                              attention=False,
                              debug=False)

    enc_inputs, dec_inputs, word_2_index, index_2_word = get_data_by_type('train')

    
    print("Training network")
    t = time.time()
    with tf.Session() as sess:
        enc_inputs, dec_inputs, word_2_index, index_2_word = get_data_by_type('train')

        sess.run(tf.global_variables_initializer())
        for i in range(conf.num_epochs):
            print("Training epoch {}".format(i))
            for data_batch, data_sentence_lengths, label_batch, label_sentence_lengths in tqdm(bucket_by_sequence_length(enc_inputs, dec_inputs, conf.batch_size), total = len(enc_inputs) / conf.batch_size):

                feed_dict = model.make_train_inputs(data_batch, data_sentence_lengths, label_batch, label_sentence_lengths)
                _ = sess.run([model.train_op], feed_dict)


if __name__ == "__main__":
    mainFunc(sys.argv[1:])
