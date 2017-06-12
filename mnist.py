import keras
import numpy as np
from keras.datasets import mnist
from keras.layers.convolutional import Conv2D
from keras.layers.core import Dropout, Flatten, Dense
from keras.layers.pooling import MaxPooling2D
from keras.models import Sequential


def reduce_precision(input_array):
    if type(input_array[0]) == np.float32:
        for i in range(len(input_array)):
            input_array[i] = np.round(input_array[i], 2)
    else:
        for i in input_array:
            reduce_precision(i)


num_classes = 10
batch_size = 128

epochs = 16

# input image dimensions
in_rows, in_cols = 28, 28

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], in_rows, in_cols, 1)
x_test = x_test.reshape(x_test.shape[0], in_rows, in_cols, 1)
input_shape = (in_rows, in_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

print('x_train shape:', x_train.shape)
print('x_test  shape:', x_test.shape)

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5), activation='relu', input_shape=input_shape))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

model.compile(
    loss=keras.losses.categorical_crossentropy,
    optimizer=keras.optimizers.Adam(),
    metrics=['accuracy']
)

print('Loading weights')
model.load_weights('weights.hdf5')

# model.fit(
#     x_train,
#     y_train,
#     batch_size=batch_size,
#     epochs=epochs,
#     verbose=1
# )
#
# print('Saving weights')
# model.save_weights('weights.hdf5', True)

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

print('Reducing precision in weight')
weights = model.get_weights()
reduce_precision(weights)
model.set_weights(weights)

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
