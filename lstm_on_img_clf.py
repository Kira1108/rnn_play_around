import os
from keras.models import Model
from keras.layers import Input, LSTM, GRU, Bidirectional,Lambda, Concatenate,Dense, GlobalMaxPooling1D
import keras.backend as K
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255., x_test / 255.
print('X_train shape: {}'.format(x_train.shape))
print('X_test shape: {}'.format(x_test.shape))
print('y_train shape: {}'.format(y_train.shape))
print('y_test shape: {}'.format(y_test.shape))


D = 28
M = 15

inputs = Input(shape = (D,D))
x1 = Bidirectional(LSTM(M, return_sequences = True))(inputs)
x1 = GlobalMaxPooling1D()(x1)
permutor = Lambda(lambda t: K.permute_dimensions(t,pattern = (0,2,1)))
x2 = permutor(inputs)
x2 = Bidirectional(LSTM(M, return_sequences= True))(x2)
x2 = GlobalMaxPooling1D()(x2)
x = Concatenate(axis = 1)([x1,x2])
outputs = Dense(10,activation = 'softmax')(x)
model = Model(inputs = inputs, outputs = outputs)
model.compile(optimizer = 'Adam',
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy'])
print(model.summary())

history = model.fit(x_train,y_train,batch_size = 256, epochs = 5, validation_split = 0.3)

plt.plot(history.history['loss'], label = 'Loss')
plt.plot(history.history['val_loss'], label = 'Val-loss')
plt.legend()
plt.show()

plt.plot(history.history['accuracy'], label = 'Accuracy')
plt.plot(history.history['val_accuracy'], label = 'Val-Accuracy')
plt.legend()
plt.show()
