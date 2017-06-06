import tensorflow as tf
import numpy as np
from tqdm import tqdm
from math import ceil
from config import Config as conf
from data_utility import START_TOKEN_INDEX

def batch_dummy_paddings(paddings, lengths):
    batch_size = conf.batch_size
    num_batches = ceil(len(paddings) / batch_size)    

    for batch_num in range(num_batches):

        encoder_batch = paddings[batch_num*batch_size:(batch_num+1)*batch_size, :]
        encoder_batch_lens = lengths[batch_num*batch_size:(batch_num+1)*batch_size]
        
        yield encoder_batch, encoder_batch_lens

def construct_lm_logits(sess, model):
    # construct paddng lists up to input max len
    max_input_len = conf.input_sentence_max_length + 1 # Incremented by 1 because of the final EOS
    paddings = np.full((max_input_len, conf.max_decoder_inference_length), START_TOKEN_INDEX, dtype=np.int32)
    padding_lengths = list(range(1, max_input_len + 1))

    # loop through all the batches
    print("Constructing dummy language model")

    all_logits = []
    for dummy_batch, dummy_batch_lens in tqdm(
                    batch_dummy_paddings(paddings, padding_lengths), total=ceil(max_input_len / conf.batch_size)):

        feed_dict = model.make_inference_inputs(dummy_batch, dummy_batch_lens)
        decoder_logits = sess.run(model.dummy_decoder_logits, feed_dict)

        assert decoder_logits.shape == (len(dummy_batch_lens), conf.max_decoder_inference_length, conf.vocabulary_size)

        all_logits.append(decoder_logits)
    all_logits = np.vstack(all_logits)
    assert all_logits.shape == (max_input_len, conf.max_decoder_inference_length, conf.vocabulary_size)

    return all_logits

def construct_lm_logits_batch(all_logits, batch_sequence_lengths):
    idx_arr = [x - 1 for x in batch_sequence_lengths] # Indexing starts at zero
    batch_logits = all_logits[idx_arr]
    return batch_logits