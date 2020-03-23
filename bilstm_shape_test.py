import numpy as np
from keras.layers import Bidirectional, LSTM, GRU, Dense, Input
from keras.models import Model


T = 8
D = 2
M = 3

X = np.random.randn(1,T,D)
def bilstm1():
    input_ = Input(shape = (T,D))
    x = Bidirectional(LSTM(M,return_sequences = True,return_state = True))(input_)
    model = Model(inputs = input_,outputs = x)
    o,h1,c1,h2,c2 = model.predict(X)
    print('o:', o)
    print('h1:', h1)
    print('c1:', c1)
    print('h2:', h2)
    print('c2:', c2)


def bilstm2():
    input_ = Input(shape = (T,D))
    x = Bidirectional(LSTM(M, return_state = True))(input_)
    model = Model(inputs = input_,outputs = x)
    o,h1,c1,h2,c2 = model.predict(X)
    print('o:', o)
    print('h1:', h1)
    print('c1:', c1)
    print('h2:', h2)
    print('c2:', c2)

print('Return sequences = True')
bilstm1()
print('Return sequences = False')
bilstm2()
