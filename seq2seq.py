import os
import sys
from keras.models import Model
from keras.layers import Input, LSTM, GRU, Dense, Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BATCH_SIZE = 64
EPOCHS = 100
LATENT_DIM = 256
NUM_SAMPLES = 10000
MAX_SEQUENCE_LENGTH = 100
MAX_NUM_WORDS = 20000
EMBEDDING_DIM = 100

DATA_PATH = r"C:\Users\songshushan\desktop\rnn_play_around\data\spa.txt"
EMBEDDING_PATH = r"C:\Users\songshushan\desktop\rnn_play_around\glove.6B\glove.6B.{}d.txt".format(str(EMBEDDING_DIM))

input_texts = []
target_texts = []
target_texts_inputs = []

print("Reading raw_data...")
# Reading the data into 3 arrays
# 1 for the input sentenceNUM
# 1 for the decoder input
# 1 for the decoder ouput labels
with open(DATA_PATH,'r',encoding = 'utf-8') as f:
    t = 0
    for line in f:
        t += 1
        if t >  NUM_SAMPLES:
            break

        if '\t' not in line:
            continue

        input_text, translation = line.split('\t')
        target_text = translation + ' <eos>'
        target_text_input = '<sos> ' + translation

        # encoder, decoder_labels, decoder_inputs
        input_texts.append(input_text)
        target_texts.append(target_text)
        target_texts_inputs.append(target_text_input)
print("Number of samples: {}".format(len(input_texts)))
# lim = 10
# print(input_texts[0:lim])
# print(target_texts_inputs[0:lim])
# print(target_texts[0:lim])

# Tokenization part
# 2 different tokenizers
# 1 for input english sentences,
# 1 for spanish sequences (decoder input and decoder labels)

tokenizer_inputs = Tokenizer(num_words = MAX_NUM_WORDS)
tokenizer_inputs.fit_on_texts(input_texts)
input_sequences = tokenizer_inputs.texts_to_sequences(input_texts)
word2index_inputs = tokenizer_inputs.word_index
index2word_inputs = {v:k for k,v in word2index_inputs.items()}
print("Found {} unique input tokens.".format(len(word2index_inputs)))

tokenizer_outputs = Tokenizer(num_words = MAX_NUM_WORDS, filters = '')
tokenizer_outputs.fit_on_texts(target_texts +  target_texts_inputs)
target_sequences = tokenizer_outputs.texts_to_sequences(target_texts)
target_inputs_sequences = tokenizer_outputs.texts_to_sequences(target_texts_inputs)
word2index_outputs = tokenizer_outputs.word_index
index2word_outputs = {k:v for k,v in word2index_outputs.items()}
print("Found {} unique output tokens.".format(len(word2index_outputs)))

num_words_inputs = len(word2index_inputs) + 1
num_words_outputs = len(word2index_outputs) + 1
max_len_inputs = max(len(s) for s in input_sequences)
max_len_outputs = max(len(s) for s in target_sequences)
min_len_inputs = min(len(s) for s in input_sequences)
min_len_outputs = min(len(s) for s in target_sequences)

encoder_inputs = pad_sequences(input_sequences, maxlen = max_len_inputs)
decoder_inputs = pad_sequences(target_inputs_sequences, maxlen = max_len_outputs)
decoder_outputs = pad_sequences(target_sequences,maxlen = max_len_outputs)


print('Encoder sequence shape: {}'.format(encoder_inputs.shape))
print('Decoder inputs shape: {}'.format(decoder_inputs.shape))
print("Decoder outputs.shape: {}".format(decoder_outputs.shape))
