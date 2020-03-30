import os, sys
from keras.models import Model
from keras.layers import Input, LSTM, GRU, Dense, Embedding, Bidirectional, \
RepeatVector, Concatenate, Activation, Dot, Lambda
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import keras.backend as K

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm

BATCH_SIZE = 64
EPOCHS = 1
LATENT_DIM = 256
LATENT_DIM_DECODER = 256
NUM_SAMPLES = 10000
MAX_SEQUENCE_LENGTH = 100
MAX_NUM_WORDS = 20000
EMBEDDING_DIM = 100

if sys.platform.startswith('win'):
    # Windows path
    DATA_PATH = r"C:\Users\songshushan\desktop\rnn_play_around\data\spa.txt"
    EMBEDDING_PATH = r"C:\Users\songshushan\desktop\rnn_play_around\glove.6B\glove.6B.{}d.txt".format(str(EMBEDDING_DIM))

if sys.platform.startswith('darwin'):
    # Mac path
    DATA_PATH = "/Users/huan/desktop/rnn/data/spa.txt"
    EMBEDDING_PATH = "/Users/huan/desktop/rnn/glove.6B/glove.6B.{}d.txt".format(str(EMBEDDING_DIM))

def softmax_over_time(x):
    # batch dimension, time dimension, vector dimension (length = 1)
    assert(K.ndim(x) > 2)

    # the output should be of shape (N, T, 1), the last dimension represents alpha
    e = K.exp(x - x.max(x,axis = 1, keepdims = True))
    s = K.sum(e, axis = 1, keepdims = True)
    return e/s

def read_data():
    input_texts = []
    target_texts = []
    target_texts_inputs = []

    with open(DATA_PATH, 'r',encoding = 'utf-8') as f:
        t = 0
        for line in f:
            t += 1

            if t > NUM_SAMPLES:
                break

            if '\t' not in line:
                continue

            input_text, translation = line.rstrip().split('\t')

            target_text = translation + ' <eos>'
            target_text_input = '<sos> ' + translation

            input_texts.append(input_text)
            target_texts.append(target_text)
            target_texts_inputs.append(target_text_input)

        print('Number of samples: {}'.format(len(input_texts)))

    return input_texts, target_texts, target_texts_inputs

def tokenize_seq2seq(input_texts, target_texts, target_texts_inputs):

    tokenizer_inputs = Tokenizer(num_words = MAX_NUM_WORDS)
    tokenizer_inputs.fit_on_texts(input_texts)
    input_sequences = tokenizer_inputs.texts_to_sequences(input_texts)

    word2index_inputs = tokenizer_inputs.word_index
    index2word_inputs = {v:k for k, v in word2index_inputs.items()}

    max_len_input = max(len(s) for s in input_sequences)


    tokenizer_outputs = Tokenizer(num_words = MAX_NUM_WORDS, filters = '')
    tokenizer_outputs.fit_on_texts(target_texts + target_texts_inputs)
    target_sequences = tokenizer_outputs.texts_to_sequences(target_texts)
    target_sequences_inputs = tokenizer_outputs.texts_to_sequences(target_texts_inputs)

    word2index_outputs = tokenizer_outputs.word_index
    index2word_outputs = {v:k for k, v in word2index_outputs.items()}

    max_len_output = max(len(s) for s in target_sequences)

    tokenizers = {
    'tokenizer_inputs':tokenizer_inputs,
    'tokenizer_outputs':tokenizer_outputs
    }

    converters = {
    'word2index_inputs':word2index_inputs,
    'word2index_outputs':word2index_outputs,
    'index2word_inputs':index2word_inputs,
    'index2word_outputs':index2word_outputs
    }

    token_info = {'max_len_input':max_len_input,
    'max_len_output':max_len_output,
    'num_words_inputs': len(word2index_inputs) + 1,
    'num_words_outputs': len(word2index_outputs) + 1}

    seqs = {
    'input_sequences': input_sequences,
    'target_sequences':target_sequences,
    'target_sequences_inputs':target_sequences_inputs,
    }

    print('Maximum input sequence length:', max_len_input)
    print('Maximum target sequence length:', max_len_output)
    print('Found unique input tokens:', len(word2index_inputs))
    print('Found unique target tokens:', len(word2index_outputs))

    return seqs, converters, token_info, tokenizers

def pad_all(input_sequences, target_sequences, target_sequences_inputs, max_len_input, max_len_output):
    encoder_inputs = pad_sequences(input_sequences, maxlen = max_len_input)
    decoder_inputs = pad_sequences(target_sequences_inputs, maxlen = max_len_output, padding = 'post')
    decoder_targets = pad_sequences(target_sequences, maxlen = max_len_output, padding = 'post')

    print('Encoder input shape:', encoder_inputs.shape)
    print('Decoder input shape:', decoder_inputs.shape)
    print('Decoder target shape:', decoder_targets.shape)

    return encoder_inputs, decoder_inputs, decoder_targets

def load_word_vector(emd_path):
    print('Loading word vectors...')
    word2vec = {}
    with open(emd_path,'r', encoding = 'utf-8') as f:
        for line in tqdm(f):
            values = line.split()
            word = values[0]
            vec = np.asarray(values[1:], dtype = 'float32')
            word2vec[word] = vec

    print('Found {} unique word vectors.'.format(len(word2vec)))
    return word2vec

def create_embedding_mat(num_words_inputs, word2index_inputs, word2vec):
    num_words = min(MAX_NUM_WORDS, num_words_inputs)
    embedding_matrix = np.zeros((num_words, EMBEDDING_DIM), dtype = np.float32)
    for word, idx in word2index_inputs.items():
        if idx < MAX_NUM_WORDS:
            embedding_vector = word2vec.get(word)
            if embedding_vector is not None:
                embedding_matrix[idx] = embedding_vector

    print('Embedding matrix shape:', embedding_matrix.shape)
    return embedding_matrix

input_texts, target_texts, target_texts_inputs = read_data()
seqs, converters, token_info, tokenizers = tokenize_seq2seq(input_texts,target_texts,target_texts_inputs)

# retrieve tokenization detail
input_sequences = seqs['input_sequences']
target_sequences = seqs['target_sequences']
target_sequences_inputs = seqs['target_sequences_inputs']

max_len_input = token_info['max_len_input']
max_len_output = token_info['max_len_output']
num_words_inputs = token_info['num_words_inputs']
num_words_outputs = token_info['num_words_outputs']

word2index_inputs = converters['word2index_inputs']
word2index_outputs = converters['word2index_outputs']
index2word_inputs = converters['index2word_inputs']
index2word_outputs = converters['index2word_outputs']

# pad sequence to specified length
encoder_inputs, decoder_inputs, decoder_targets = pad_all(input_sequences,\
    target_sequences, target_sequences_inputs, max_len_input, max_len_output)

word2vec = load_word_vector(EMBEDDING_PATH)

embedding_matrix = create_embedding_mat(num_words_inputs, word2index_inputs, word2vec)

num_words = embedding_matrix.shape[0]

embedding_layer = Embedding(num_words, EMBEDDING_DIM, input_length = max_len_input,weights = [embedding_matrix])

# N * T * Doh
decoder_targets_oh = np.zeros((len(encoder_inputs),
    max_len_output,
    num_words_outputs),
    dtype = 'float32')

for m, seq in enumerate(decoder_targets):
    for t, token in enumerate(seq):
        decoder_targets_oh[m, t, token] = 1

print('Decoder targets shape: ', decoder_targets_oh.shape)

# Build models

# encoder
encoder_inputs_tensor = Input(shape = (max_len_input,))
x = embedding_layer(encoder_inputs_tensor)
encoder = Bidirectional(LSTM(LATENT_DIM,return_sequences = True, dropout = .5))
encoder_outputs = encoder(x)

# decoder
decoder_inputs_tensor = Input(shape = (max_len_output))
decoder_embedding = Embedding(num_words_outputs, EMBEDDING_DIM)
decoder_inputs_tensor = decoder_embedding(decoder_inputs_tensor)

# s<t-1> repeat Tx times
attn_repeat_layer = RepeatVector(max_len_input)
attn_concat_layer = Concatenate(axis = -1)
attn_dense1 = Dense(10,activation = 'tanh')
attn_dense2 = Dense(1, activation = softmax_over_time)
attn_dot = Dot(axis = 1)

def one_step_attention(h, st_1):

    # since you return the sequences, h are hidden vectors for all timestep
    # h has a shape of (Tx, LATENT_DIM_ENCODER)

    # st_1 here is one step decoder hidden state (1, LATENT_DIM_DECODER)
    # repeat vector add a dimension before original dimension
    # after repeat st_1 has shape (Tx, LATENT_DIM_DECODER)
    st_1 = attn_repeat_layer(st_1)

    # Concat happends on the last dimension
    # a has a shape of (Tx, LATENT_DIM_ENCODER + LATENT_DIM_DECODER)
    a = attn_concat_layer([h,st_1])

    # Dense acts only on the last dimension
    # after this dense , you have a shape (Tx, 10)
    a = attn_dense1(a)

    # The final dense has only 1 unit - for each timestep
    # Activation over time, you have (Tx, 1) for alphas
    alphas = attn_dense2(a)

    # remember alphas has a shape (Tx ,1)
    # h shape (Tx, LATENT_DIM_ENCODER)
    
    context = attn_dot([alphas, h])
    return context




# if __name__ == '__main__':
    # print('------------Reading_data_test-----------------')
    # input_texts, target_texts, target_texts_inputs = read_data()
    #
    # for i in range(10):
    #     print('Input:', input_texts[i])
    #     print('Target:', target_texts[i])
    #     print('Target input', target_texts_inputs[i])
    #     print('*'*30)

    # print('------------Tokenization_test-----------------')
    # input_texts, target_texts, target_texts_inputs = read_data()
    # seqs, converters, token_info, tokenizers = tokenize_seq2seq(input_texts,target_texts,target_texts_inputs)
