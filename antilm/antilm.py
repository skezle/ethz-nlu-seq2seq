import tensorflow as tf
import numpy as np
from tqdm import tqdm
from math import ceil
from config import Config as conf
from data_utility import PAD_TOKEN_INDEX

def batch_dummy_paddings(sorted_lengths):
    batch_size = conf.batch_size
    num_batches = ceil(len(sorted_lengths) / batch_size)

    for batch_num in range(num_batches):
        encoder_batch_lens = sorted_lengths[batch_num * batch_size:(batch_num + 1) * batch_size]
        encoder_batch = np.full((len(encoder_batch_lens), max(encoder_batch_lens)), PAD_TOKEN_INDEX, dtype=np.int32)
        
        yield encoder_batch, encoder_batch_lens

def construct_lm_softmax(sess, model, validation_input_lengths):

    # construct paddng lists up to input max len
    max_input_len = conf.input_sentence_max_length + 1 # Incremented by 1 because of the final EOS
    sorted_padding_lengths = list(sorted(list(validation_input_lengths)))

    # loop through all the batches
    print("Constructing dummy language model")

    all_softmax = []
    for dummy_batch, dummy_batch_lens in tqdm(
                    batch_dummy_paddings(sorted_padding_lengths), total=ceil(len(sorted_padding_lengths) / conf.batch_size)):

        feed_dict = model.make_inference_inputs(dummy_batch, dummy_batch_lens)
        decoder_softmax= sess.run(model.dummy_decoder_softmax, feed_dict)

        assert decoder_softmax.shape == (len(dummy_batch_lens), conf.antilm_max_penalization_len, conf.vocabulary_size)

        all_softmax.append(decoder_softmax)
    all_softmax = np.vstack(all_softmax)
    assert all_softmax.shape == (len(sorted_padding_lengths), conf.antilm_max_penalization_len, conf.vocabulary_size)

    all_softmax_dict = {}
    for i, pad_len in enumerate(sorted_padding_lengths):
        all_softmax_dict[pad_len] = all_softmax[i, :, :]
    return all_softmax_dict

def construct_lm_softmax_batch(all_softmax_dict, batch_sequence_lengths):
    batch_output = []
    for sequence_length in batch_sequence_lengths:
        batch_output.append(all_softmax_dict[sequence_length])
    batch_output_softmax = np.stack(batch_output, axis=0)
    assert batch_output_softmax.shape == (len(batch_sequence_lengths), conf.antilm_max_penalization_len, conf.vocabulary_size)
    return batch_output_softmax