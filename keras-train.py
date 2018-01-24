# Import the required packages
from __future__ import print_function

import numpy as np
from PIL import Image
import os
import itertools
from enum import Enum

from keras.optimizers import RMSprop, SGD

FIELD_X = 1052
FIELD_DX = 66
FIELD_Y = 221
FIELD_DY = 57
FIELD_SIZE = 6

SCAN_RADIUS = 17


class Marble(Enum):
    none = 0
    Salt = 1
    Air = 2
    Fire = 3
    Water = 4
    Earth = 5
    Vitae = 6
    Mors = 7
    Quintessence = 8
    Quicksilver = 9
    Lead = 10
    Tin = 11
    Iron = 12
    Copper = 13
    Silver = 14
    Gold = 15

    def symbol(self):
        if self.value is self.none.value:
            return "-"
        if self.value in range(self.Quicksilver.value, self.Gold.value + 1):
            return self.name[0].upper()
        else:
            return self.name[0].lower()


def field_positions():
    d = FIELD_SIZE - 1
    result = []
    for y in range(-d, d + 1):
        for x in range(-d, d + 1):
            if not abs(y - x) > d:
                result.append((x + d, y + d))
    return result


def pixels_to_scan():
    pxs = []
    for dy in range(-SCAN_RADIUS + 1, SCAN_RADIUS):
        for dx in range(-SCAN_RADIUS + 1, SCAN_RADIUS):
            pxs.append((dx, dy))
    return pxs


def lightness_at(img, x, y):
    _min, _max = img.getpixel((x, y))
    return _min / 255


def edges_at(img, x, y):
    result = []
    for (xx, yy) in PIXELS_TO_SCAN:
        result.append(lightness_at(img, x + xx, y + yy))
    return result


def img_pos(x, y):
    return FIELD_X + FIELD_DX * (x * 2 - y) / 2, FIELD_Y + FIELD_DY * y


FIELD_POSITIONS = field_positions()
PIXELS_TO_SCAN = pixels_to_scan()

MARBLE_BY_SYMBOL = dict(zip([Marble.symbol(e) for e in Marble], [e.name for e in Marble]))


def load_mnist():
    images, labels = [], []
    for i in range(1, 7):
        img = Image.open(os.path.join("sample", str(i) + ".png")).convert('LA')
        samples = list(itertools.chain.from_iterable(
            [lines.split() for lines in open(os.path.join("sample", str(i) + ".txt"), "r").readlines()]))
        for j, (pos, symbol) in enumerate(zip(FIELD_POSITIONS, samples)):
            marble = MARBLE_BY_SYMBOL[symbol]
            edge_pixels = edges_at(img, *img_pos(*pos))
            images.append(edge_pixels)
            labels.append(Marble[marble].value)
    return np.array(images), np.array(labels)


x_train, y_train = load_mnist()
x_test, y_test = load_mnist()

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

batch_size = 100
num_classes = 16
epochs = 15

# input image dimensions
img_rows, img_cols = 33, 33

# the data, shuffled and split between train and test sets
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
print(K.image_data_format())
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
print('x_train shape:', x_train[0])
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.add(Dense(1, activation='softmax'))
rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
model.summary()
model.compile(optimizer=rmsprop,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
#
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          # steps_per_epoch=2000,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
#
print('Test loss:', score[0])
print('Test accuracy:', score[1])
model.save('model.h5')
# print(model.predict(x_test[0:3]))
# print(model.predict(x_test[0:3]))
# print(model.predict(x_test[0:3]))
# print(model.predict(x_test[0:3]))
# prediction = model.predict(x_train[0])
# print(prediction)
# print(y_test[0])
