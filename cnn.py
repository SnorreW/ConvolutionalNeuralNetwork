#!/usr/bin/env python
# coding: utf-8
# Made in jupyter notebook

import matplotlib.pyplot as plt
import keras.layers as layer
from keras.models import Sequential
from keras.utils import to_categorical
from keras.datasets import mnist
from keras.losses import categorical_crossentropy
from keras.optimizers import Adadelta

#Implementing the MNIST dataset.
#Splitting the dataset into a training dataset and a testing dataset.
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#These are the features.
#See how the x values look it looks. The x values are the images. There is 60 000 training images and 10 000 testing images.
#It is a 28x28 pixel image and each pixel has a value from 0 to 255 that makes the color.
print(x_train.shape)
print(x_test.shape)

#See how the first image looks.
print(x_train[0])

#See how the y values looks. The y values are the labels. Each has a value from 0 to 9 that is linked to its corresponding image.
print(y_train)
print(y_test)

#Using matplotlib to plot the first image. It looks like 5.
plt.imshow(x_train[0])
plt.show()
#Reshaping the data so it can fit in the model. x_train.shape[0] = 60 000 which makes the rows.
#28x28 pixels.
#1 means that the color will be in grayscale. A pixel will have a value from 0 to 1.
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

#Shape that is going into the neural network.
input_shape = (28, 28, 1)

#Normalizing the data. In this case, it is unnecessary to have a value from 0 to 255 as it is not efficient. Instead I can give it a value from 0 to 1.
x_train = x_train.astype('float32')
x_train /= 255

x_test = x_test.astype('float32')
x_test /= 255

#See how the first y_training value looks.
print(y_train[0])

#This changes the y value to an array of ones and zeroes. 
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

#Visualizing what the first y value looks like now.
print(y_train[0])

#The machine learning model
#Sequential means that it groups a stack of layers into a model.
model = Sequential()
#Conv2D creates a 2D convolution layer.
#The first parameter is the number of filters. In this case it is 32.
#kernel_size is specifying the height and width of the 2D convolution window. In this case 3x3.
#activation='relu' means that the actvation function used is rectified linear unit(relu) which will output the input directly if it is positive and the output will be zero if it is not positive.
#input_shape is (28,28,1).
model.add(layer.Conv2D(32, kernel_size=(3,3), activation='relu', input_shape = input_shape))
#Basically the same as above but the number of output filters is 64.
model.add(layer.Conv2D(64, (3,3), activation='relu'))
#Separates the image into smaller patches and calculates the maximum value in those patches. In this case, 2x2 pixels.
model.add(layer.MaxPooling2D(pool_size=(2,2)))
#Dropout helps to prevent overfitting. It randomly sets the input units to 0 at each step of the training time.
model.add(layer.Dropout(0.25))
#Flattens the pooling layer. The new layer looks like this (none, 9216).
model.add(layer.Flatten())
#Activation function is rectified linear unit(relu). This dense layer has 64 neurons.
model.add(layer.Dense(64, activation='relu'))
#Dropout helps to prevent overfitting.
model.add(layer.Dropout(0.25))
#Dense layer with 10 neurons. softmax activation function that turns a vector of numbers into a vector of probabilities.
model.add(layer.Dense(10,activation='softmax'))

#Training
#Compile makes the model ready for training. Loss is the objective function. Optimizer is optimizer instance. Metrics is list of metrics that the model is evaluating during training and testing.
model.compile(loss=categorical_crossentropy, optimizer=Adadelta(), metrics=['accuracy'])
#fit trains the model. Using x_train and y_train. 
#batch_size is the number of gradient samples per gradient update. In my case it is 128.
#epochs is the number of iterations done on the dataset. Here I do 20 iterations.
#validation_data is the data that is used to evaluate loss and model metrics at the end of each iteration.
model.fit(x_train, y_train, batch_size=64, epochs=10, validation_data=(x_test, y_test))
#This displays a summary of the model.
print(model.summary())

#Testing and evaluation. Displays loss, then accuracy from 0 to 1.
print(model.evaluate(x_test, y_test))

