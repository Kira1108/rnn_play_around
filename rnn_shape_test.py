from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, GRU, Input
import numpy as np
import matplotlib.pyplot as plt

# Sequece legnth
T = 8
# Embedding dimension
D = 2
# hidden vector dimension
M = 3

# (batch dimension, Sequence length, embedding dimension)
# single sentence of word vectors
X = np.random.randn(1,T,D)


def lstm1():
    inputs = Input(shape = (T,D))
    rnn = LSTM(M, return_state = True)
    x = rnn(inputs)
    model = Model(inputs= inputs , outputs = x,)
    o,h,c = model.predict(X)
    print('o: ', o)
    print('h: ', h)
    print('c: ', c)

def lstm2 ():
    inputs = Input(shape = (T,D))
    rnn = LSTM(M, return_state = True, return_sequences = True)
    x = rnn(inputs)
    model = Model(inputs= inputs , outputs = x,)
    o,h,c = model.predict(X)
    print('o: ', o)
    print('h: ', h)
    print('c: ', c)


def lstm3 ():
    inputs = Input(shape = (T,D))
    rnn = LSTM(M)
    x = rnn(inputs)
    model = Model(inputs= inputs , outputs = x,)
    something = model.predict(X)
    print('sequence / state False', something)



def gru1():
    inputs = Input(shape = (T,D))
    rnn = GRU(M, return_state = True)
    x = rnn(inputs)
    model = Model(inputs= inputs , outputs = x,)
    o,h = model.predict(X)
    print('o: ', o)
    print('h: ', h)

def gru2 ():
    inputs = Input(shape = (T,D))
    rnn = GRU(M, return_state = True, return_sequences = True)
    x = rnn(inputs)
    model = Model(inputs= inputs , outputs = x,)
    o,h = model.predict(X)
    print('o: ', o)
    print('h: ', h)


print('Lstm result with return states = True')
lstm1()
print('Lstm result with return states = True and return_sequences = True')
lstm2()
print('GRU result with return states = True')
gru1()
print('GRU result with return states = True and return_sequences = True')
gru2()


print('LSTM with return_sequences and return_state both False')
lstm3()
