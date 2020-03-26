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
EPOCHS = 1000
LATENT_DIM = 256
NUM_SAMPLES = 10000
MAX_SEQUENCE_LENGTH = 100
MAX_NUM_WORDS = 20000
EMBEDDING_DIM = 100


# Windows path
DATA_PATH = r"C:\Users\songshushan\desktop\rnn_play_around\data\spa.txt"
EMBEDDING_PATH = r"C:\Users\songshushan\desktop\rnn_play_around\glove.6B\glove.6B.{}d.txt".format(str(EMBEDDING_DIM))

# Mac path
# DATA_PATH = "/Users/huan/desktop/rnn/data/spa.txt"
# EMBEDDING_PATH = "/Users/huan/desktop/rnn/glove.6B/glove.6B.{}d.txt".format(str(EMBEDDING_DIM))

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
index2word_outputs = {v:k for k,v in word2index_outputs.items()}
print("Found {} unique output tokens.".format(len(word2index_outputs)))

num_words_inputs = len(word2index_inputs) + 1
num_words_outputs = len(word2index_outputs) + 1
max_len_inputs = max(len(s) for s in input_sequences)
max_len_outputs = max(len(s) for s in target_sequences)
min_len_inputs = min(len(s) for s in input_sequences)
min_len_outputs = min(len(s) for s in target_sequences)

## all the length dimension are defined here, actually##
encoder_inputs = pad_sequences(input_sequences, maxlen = max_len_inputs)
decoder_inputs = pad_sequences(target_inputs_sequences, maxlen = max_len_outputs, padding = 'post')
decoder_outputs = pad_sequences(target_sequences,maxlen = max_len_outputs, padding = 'post')

print('Encoder sequence shape: {}'.format(encoder_inputs.shape))
print('Decoder inputs shape: {}'.format(decoder_inputs.shape))
print("Decoder outputs.shape: {}".format(decoder_outputs.shape))

print('Loading word vectors')
word2vec = {}
with open(EMBEDDING_PATH,'r', encoding = 'utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:],dtype = 'float32')
        word2vec[word] = vector
print('Word Vectors loaded.')

print('Building embedding matrix and embedding layer')
num_words = min(MAX_NUM_WORDS, len(word2index_inputs) + 1)
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, ind in word2index_inputs.items():
    if ind < MAX_NUM_WORDS:
        embedding_vector = word2vec.get(word)
        if embedding_vector is not None:
            embedding_matrix[ind] = embedding_vector

embedding_layer = Embedding(
    num_words,
    EMBEDDING_DIM,
    weights = [embedding_matrix],
    input_length = max_len_inputs
)

# target are encoded one-hot
decoder_targets_one_hot = np.zeros((
len(decoder_inputs), max_len_outputs, num_words_outputs
), dtype = 'float32')

for i, seq in enumerate(decoder_outputs):
    for time, token in enumerate(seq):
        decoder_targets_one_hot[i, time, token] = 1


# ------------* Seq 2 Seq model *----------------
# encoder part
encoder_input_t = Input(shape = (max_len_inputs,))
x = embedding_layer(encoder_input_t)
encoder_layer = LSTM(LATENT_DIM, return_state = True, dropout = 0.5)
encoder_outputs, h, c = encoder_layer(x)
encoder_states = [h, c]

# decoder part
decoder_input_t = Input(shape = (max_len_outputs,))
decoder_embedding = Embedding(num_words_outputs, EMBEDDING_DIM)
decoder_inputs_x = decoder_embedding(decoder_input_t)
decoder_lstm = LSTM(LATENT_DIM,return_sequences = True, return_state = True)
decoder_outputs_t, _, _ = decoder_lstm(decoder_inputs_x, initial_state = encoder_states)
decoder_dense = Dense(num_words_outputs, activation = 'softmax')
decoder_outputs_t = decoder_dense(decoder_outputs_t)


model = Model(inputs = [encoder_input_t, decoder_input_t], outputs = [decoder_outputs_t])

print("Training model summary")
print(model.summary())

model.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])

history = model.fit(x = [encoder_inputs, decoder_inputs],
                    y = decoder_targets_one_hot,
                    epochs = EPOCHS,
                    batch_size = BATCH_SIZE,
                    validation_split = .2)

plt.plot(history.history['loss'], label = 'Loss')
plt.plot(history.history['val_loss'], label = 'Val-loss')
plt.legend()
plt.show()

plt.plot(history.history['accuracy'], label = 'Accuracy')
plt.plot(history.history['val_accuracy'], label = 'Val-Accuracy')
plt.legend()
plt.show()

model.save('seq2seq.h5')



# At prediction time, build a sampling model
encoder_model = Model(encoder_input_t, encoder_states)

print("Test model encoder summary:")
print(encoder_model.summary())


decoder_state_h  = Input(shape = (LATENT_DIM,))
decoder_state_c =  Input(shape = (LATENT_DIM,))
decoder_state_input = [decoder_state_h, decoder_state_c]

decoder_input_sos = Input(shape = (1,))
decoder_input_x = decoder_embedding(decoder_input_sos)
decoder_output_x, h, c = decoder_lstm(decoder_input_x, initial_state = decoder_state_input)
decoder_states = [h,c]
decoder_outputs_p = decoder_dense(decoder_output_x)

decoder_model = Model([decoder_input_sos] + decoder_state_input, [decoder_outputs_p] + decoder_states)

print("Test model decoder summary")
print(decoder_model.summary())

# index2word_inputs - english
# index2word_outputs - spanish

def decode_sequence(input_seq):
    '''
        input_seq: token sequences,
        only 1 token sequence,
        but require a batch dimension,
        Type: 2 dimensional list or numpy array
    '''
    # encode the input sentence
    states_value = encoder_model.predict(input_seq)

    # 1 sample, 1 time step
    target_seq = np.zeros((1,1))

    # the first input should be sos
    target_seq[0,0] = word2index_outputs.get('<sos>')

    # if hit eos, end sampling
    eos = word2index_outputs['<eos>']

    # token stored in a list
    output_sentence = []

    for _ in range(max_len_outputs):
        # one step (on time dimension) decoder
        output_tokens,h,c = decoder_model.predict([target_seq] + states_value)

        # retrieve results from softmax
        idx = np.argmax(output_tokens[0,0,:])

        # hit end criteria
        if eos == idx:
            break

        # look up dictionary to get a word
        word = ''
        if idx > 0:
            word = index2word_outputs[idx]
            output_sentence.append(word)

        # input and state becomes the current timestep
        target_seq[0,0] = idx
        states_value = [h,c]

    return " ".join(output_sentence)

while True:
     i = np.random.choice(len(encoder_inputs))
     # outter list add a batch dimension
     input_seq = encoder_inputs[i:i+1]
     print('Input sequence example')
     print(input_seq)

     translation = decode_sequence(input_seq)

     print("Model run-------:")
     print('Input: {}'.format(input_texts[i]))
     print('Translation: {}'.format(translation))
     ans = input('Continue? [Y/n]')
     if ans and ans.lower().startswith('n'):
         break
