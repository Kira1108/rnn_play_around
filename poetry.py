import os
import sys
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.models import Model
from keras.layers import Input, Dense, Embedding, LSTM
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import Adam, SGD

'''
# This is a language model trained on lines of texts written by someone.
# The target of code is to generate text having the same written style as the inputs
# P(wt | w1, w2, w3 ... ... ,wt-1) -> what is the probability of the next word
'''

DATA_PATH = '/Users/huan/desktop/rnn/data/robert_frost.txt'
EMBEDDING_VECTOR_PATH = "glove.6B/glove.6B.{}d.txt"

# This is predefined input length
MAX_SEQUENCE_LENGTH = 100
MAX_VOCAB_SIZE = 3000
EMBEDDING_DIM = 50
VALIDATION_SPLIT = .2
BATCH_SIZE = 128
EPOCHS = 2000
LATENT_DIM = 25


print('Loading texts...')
input_texts = []
target_texts = []
with open(DATA_PATH, 'r') as f:

    for line in f:
        line = line.strip()
        if not line:
            continue

        input_text = '<sos> ' +  line
        output_text = line + ' <eos>'

        input_texts.append(input_text)
        target_texts.append(output_text)

all_lines = input_texts + target_texts
print("Loading texts done...")


print('Tokenizing texts')
tokenizer = Tokenizer(num_words = MAX_VOCAB_SIZE, filters = '')
tokenizer.fit_on_texts(all_lines)

# This is the input sequences and target sequences.
input_sequences = tokenizer.texts_to_sequences(input_texts)
target_sequences = tokenizer.texts_to_sequences(target_texts)
print('Tokenization done...')



print('Text statistics')
word2index = tokenizer.word_index
index2word = {v:k for k,v in word2index.items()}
words_founded = len(word2index)
max_seq_length_text = max(len(seq) for seq in input_sequences)
min_seq_length = min(len(seq) for seq in input_sequences)
medium_seq_length = [len(seq) for seq in input_sequences][int(len(input_sequences) / 2)]
print('{} word tokens founded in all texts'.format(words_founded))
print("Max sequence length: {}".format(max_seq_length_text))
print("Min sequence length : {}".format(min_seq_length))
print('Medium sequence_length: {}'.format(medium_seq_length))


print('Loading embedding vectors...')
word2vec = {}
with open(EMBEDDING_VECTOR_PATH.format(EMBEDDING_DIM), 'r') as f:
    for line in f:
        values = line.split()
        word = values[0]
        vec = np.asarray(values[1:],dtype = 'float32')
        word2vec[word] = vec

print('Word vectors loaded.')
assert('<sos>' in word2index)
assert('<eos>' in word2index)

num_words = min(MAX_VOCAB_SIZE, len(word2index) + 1)
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, ind in word2index.items():
    if ind < MAX_VOCAB_SIZE:
        embedding_vector = word2vec.get(word)
        if embedding_vector is not None:
            embedding_matrix[ind] = embedding_vector
print("Embedding matrix shape: {}".format(embedding_matrix.shape ))


max_sequence_length = max(MAX_SEQUENCE_LENGTH, max_seq_length_text)
input_sequences = pad_sequences(input_sequences,maxlen = max_sequence_length)
target_sequences = pad_sequences(target_sequences, maxlen = max_sequence_length)
print('Shape of input sequences: {}'.format(input_sequences.shape))
print("Shape of target sequence: {}".format(target_sequences.shape))


one_hot_targets = np.zeros((len(target_sequences), max_sequence_length, num_words))
for m, target_sequence in enumerate(target_sequences):
    for t, ind in enumerate(target_sequence):
        if ind > 0:
            one_hot_targets[m, t, ind] = 1

embedding_layer = Embedding(num_words,
    EMBEDDING_DIM,
    weights = [embedding_matrix],
    #trainable = False
    )

print("Building Model...")
inputs = Input(shape = (max_sequence_length,))
input_h = Input(shape = (LATENT_DIM,))
input_c = Input(shape = (LATENT_DIM,))
x = embedding_layer(inputs)
lstm_layer = LSTM(LATENT_DIM,return_sequences = True, return_state=True)
x, _, _ = lstm_layer(x,initial_state = [input_h, input_c])
densor = Dense(num_words,activation = 'softmax')
outputs = densor(x)
model = Model([inputs, input_h, input_c], outputs)
print(model.summary())

model.compile(optimizer = Adam(lr = 0.01),
    loss = 'categorical_crossentropy',metrics=['accuracy'])

h = np.zeros((len(input_sequences), LATENT_DIM))
history = model.fit([input_sequences, h, h ],
    one_hot_targets
    ,batch_size=BATCH_SIZE,
    epochs = EPOCHS,
    validation_split= VALIDATION_SPLIT)


plt.plot(history.history['loss'], label = 'Loss')
plt.plot(history.history['val_loss'], label = 'Val-loss')
plt.legend()
plt.show()

plt.plot(history.history['accuracy'], label = 'Accuracy')
plt.plot(history.history['val_accuracy'], label = 'Val-Accuracy')
plt.legend()
plt.show()

# Prediction model
# This is the sos token （batch = 1, T = 1）
input2 = Input(shape = (1,))
# to predict 1 word at a time, you need to keep track of h and c
input_h = Input(shape = (LATENT_DIM,))
input_c = Input(shape = (LATENT_DIM,))
# Embedding -> (batch = 1 , T = 1, embedding_dimension V)
x = embedding_layer(input2)
# LSTM (batch = 1, T = 1, hidden dimension M)
x,h,c = lstm(x, initial_state = [input_h, input_c])
# Dense on M hidden units, softmax it on many words
output2 = densor(x)
sampling_model = Model([input2, input_h, input_c],[output2, h, c])

# Prediction sampling
def sample_line():
    # Dimensions batch = 1, Times = 1 -> [[token]]
    np_input = np.array([[word2index['<sos>']]])
    h = np.zeros((1,LATENT_DIM,))
    c = np.zeros((1, LATENT_DIM,))

    eos = word2index['<eos>']
    output_sequence = []

    for _ in range(max_sequence_length):
        o,h,c = sampling_model.predict([np_input, h, c])

        # o is a 3 dimensional vector (batch = 1, Time = 1, classes = vocab_size)
        probs = o[0,0]  # this is a vector - probability of all wors
        if np.argmax(probs) == 0:
            print('WTF what the fuck')

        probs[0] = 0
        probs = probs / probs.sum()

        idx = np.random.choice(len(probs), p = probs)
        if idx = eos:
            break

        output_sequence.append(index2word.get(idx, 'Zzz--' + str(idx)))
        np_input[0,0] = idx

    return ' '.join(output_sequence)
