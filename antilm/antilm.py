import tensorflow as tf
import numpy as np
from tqdm import tqdm
from math import ceil
from config import Config as conf
from data_utility import PAD_TOKEN_INDEX

def batch_dummy_paddings(paddings, lengths):
    batch_size = conf.batch_size
    num_batches = ceil(len(paddings) / batch_size)    

    for batch_num in range(num_batches):

        encoder_batch = paddings[batch_num*batch_size:(batch_num+1)*batch_size, :]
        encoder_batch_lens = lengths[batch_num*batch_size:(batch_num+1)*batch_size]
        
        yield encoder_batch, encoder_batch_lens

def construct_lm_softmax(sess, model):
    # construct paddng lists up to input max len
    max_input_len = conf.input_sentence_max_length + 1 # Incremented by 1 because of the final EOS
    paddings = np.full((max_input_len, max_input_len), PAD_TOKEN_INDEX, dtype=np.int32)
    padding_lengths = list(range(1, max_input_len + 1))

    # loop through all the batches
    print("Constructing dummy language model")

    all_softmax = []
    for dummy_batch, dummy_batch_lens in tqdm(
                    batch_dummy_paddings(paddings, padding_lengths), total=ceil(max_input_len / conf.batch_size)):

        feed_dict = model.make_inference_inputs(dummy_batch, dummy_batch_lens)
        decoder_softmax= sess.run(model.dummy_decoder_softmax, feed_dict)

        assert decoder_softmax.shape == (len(dummy_batch_lens), conf.antilm_max_penalization_len, conf.vocabulary_size)

        all_softmax.append(decoder_softmax)
    all_softmax = np.vstack(all_softmax)
    assert all_softmax.shape == (max_input_len, conf.antilm_max_penalization_len, conf.vocabulary_size)

    return all_softmax

def construct_lm_softmax_batch(all_logits, batch_sequence_lengths):
    idx_arr = [x - 1 for x in batch_sequence_lengths] # Indexing starts at zero
    batch_logits = all_logits[idx_arr]
    return batch_logits