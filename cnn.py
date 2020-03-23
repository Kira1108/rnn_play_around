import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.layers import Input,Dense, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
from sklearn.metrics import roc_auc_score

DATA_PATH = './data/toxic-comment'
EMBEDDING_VECTOR_PATH = "glove.6B/glove.6B.{}d.txt"

MAX_SEQUENCE_LENGTH = 100
MAX_VOCAB_SIZE = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = .2
BATCH_SIZE = 128
EPOCHS = 10




print('Loading word vectors')
word2vec = {}
with open(EMBEDDING_VECTOR_PATH.format(EMBEDDING_DIM),'r') as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:],dtype = 'float32')
        word2vec[word] = vector
print('Word Vectors loaded.')



train = pd.read_csv(os.path.join(DATA_PATH, "train.csv"))
sentences = train['comment_text'].fillna('DUMMY_VALUES').values
possible_labels = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']
targets = train[possible_labels].values

print('Max sequence length:', max(len(seq) for seq in sentences))
print('Min sequence length:', min(len(seq) for seq in sentences))
s = sorted((len(s) for s in sentences))
print('Medium sequence length:',s[int(len(s))// 2])


print('Sentence tokenization...')
tokenizer = Tokenizer(num_words = MAX_VOCAB_SIZE)

# fit_on_texts documentation, here we use list of list of strings.
# argument: can be a list of strings,
#     a generator of strings (for memory-efficiency),
#     or a list of list of strings.
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)

word2index = tokenizer.word_index
index2word = {v:k for k,v in word2index.items()}

sequences = pad_sequences(sequences,maxlen = MAX_SEQUENCE_LENGTH)
print('Found {} unique index'.format(len(word2index)))


print('Building embedding matrix and embedding layer')
num_words = max(MAX_VOCAB_SIZE, len(word2index) + 1)
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, ind in word2index.items():
    if ind < MAX_VOCAB_SIZE:
        embedding_vector = word2vec.get(word)
        if embedding_vector is not None:
            embedding_matrix[ind] = embedding_vector

embedding_layer = Embedding(num_words,
                    EMBEDDING_DIM,
                    weights = [embedding_matrix],
                    input_length = MAX_SEQUENCE_LENGTH,
                    trainable = False)
print('Preprocessing done.')


# shapes
# sequence: N * T
# targets: N * 6 (six possible binary labels)
# embedding: V * D (num_words, embedding_dimension)

input_ = Input(shape = (MAX_SEQUENCE_LENGTH,))
x = embedding_layer(input_)
x = Conv1D(128,3,activation = 'relu')(x)
x = MaxPooling1D(3)(x)
x = Conv1D(128,3,activation = 'relu')(x)
x = MaxPooling1D(3)(x)
x = Conv1D(128,3,activation = 'relu')(x)
x = MaxPooling1D(3)(x)
x = GlobalMaxPooling1D()(x)
x = Dense(128,activation = 'relu')(x)
x = Dense(len(possible_labels), activation = 'sigmoid')(x)
model = Model(inputs = input_,outputs = x)
model.compile(optimizer = 'adam',
    loss = 'binary_crossentropy',
    metrics = ['accuracy'])
print(model.summary())

history = model.fit(x = sequences,
    y = targets,
    batch_size = BATCH_SIZE,
    epochs = 5,
    validation_split =VALIDATION_SPLIT)

plt.plot(history.history['loss'], label = 'Loss')
plt.plot(history.history['val_loss'], label = 'Val-loss')
plt.legend()
plt.show()

plt.plot(history.history['accuracy'], label = 'Accuracy')
plt.plot(history.history['val_accuracy'], label = 'Val-Accuracy')
plt.legend()
plt.show()
