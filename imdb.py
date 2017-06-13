"""
Train a recurrent convolutional network on the IMDB sentiment
classification task.

Gets to 0.8498 test accuracy after 2 epochs. 41s/epoch on K520 GPU.
"""

import numpy as np

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Conv1D, MaxPooling1D
from keras.datasets import imdb


def reduce_precision(input_array):
    if type(input_array[0]) == np.float32:
        for i in range(len(input_array)):
            input_array[i] = np.float16(input_array[i])
    else:
        for i in input_array:
            reduce_precision(i)


# Embedding
max_features = 20000
maxlen = 100
embedding_size = 128

# Convolution
kernel_size = 5
filters = 64
pool_size = 4

# LSTM
lstm_output_size = 70

# Training
batch_size = 30
epochs = 3

'''
Note:
batch_size is highly sensitive.
Only 2 epochs are needed as the dataset is very small.
'''

print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

print('Build model...')

model = Sequential()
model.add(Embedding(max_features, embedding_size, input_length=maxlen))
model.add(Dropout(0.25))
model.add(Conv1D(filters,
                 kernel_size,
                 padding='valid',
                 activation='relu',
                 strides=1))
model.add(MaxPooling1D(pool_size=pool_size))
model.add(LSTM(lstm_output_size))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

print('Loading weights')
model.load_weights('imdb-weights.hdf5')

# model.fit(
#     x_train,
#     y_train,
#     batch_size=batch_size,
#     epochs=epochs,
#     validation_data=(
#         x_test,
#         y_test
#     )
# )
#
# print('Saving weights')
# model.save_weights('imdb-weights.hdf5', True)

score, acc = model.evaluate(
    x_test,
    y_test,
    batch_size=batch_size
)
print('Test loss:', score)
print('Test accuracy:', acc)
print('Reducing precision in weight')

weights = model.get_weights()
reduce_precision(weights)
model.set_weights(weights)
score, acc = model.evaluate(
    x_test,
    y_test,
    batch_size=batch_size
)
print('Test loss:', score)
print('Test accuracy:', acc)
