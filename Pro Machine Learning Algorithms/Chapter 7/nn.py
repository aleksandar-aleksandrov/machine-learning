import numpy as pd
from numpy import ndarray
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout


# Load the dataset
(features_train, labels_train), (features_test, labels_test) = mnist.load_data()

# Plot 4 Images as gray scale
for i in range(4):
    plt.subplot(221 + i)
    plt.imshow(features_train[i], cmap=plt.get_cmap('gray'))

plt.show()

# Preprocess the dataset
num_pixels = features_train.shape[1] * features_train.shape[2]
features_train = features_train.reshape(features_train.shape[0], num_pixels).astype('float32')
features_test = features_test.reshape(features_test.shape[0], num_pixels).astype('float32')

# Scale the inputs
features_train = features_train / 255
features_test = features_test / 255

# One hot encode the output
labels_train = np_utils.to_categorical(labels_train)
labels_test = np_utils.to_categorical(labels_test)
num_classes = labels_test.shape[1]

# Build the model
model = Sequential()
model.add(Dense(1000, input_dim=num_pixels, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# Run the model
model.fit(features_train, labels_train, validation_data=(features_test, labels_test), epochs=5, batch_size=1024, verbose=1)