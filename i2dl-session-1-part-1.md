# Introduction to Deep Learning

## Intro slides
* Deep Learning is a subset of Machine Learning.
* Deep Learning uses patterns extraction from data using neural networks.
* Tensorflow is the most popular DL framework.

## Why Deep Learning
* Learning features directly from raw data instead of hard engineering them
* Detecting low-level features and translating them to medium/high-level ones
* Big Data, Hardware, Software are the drivers for the deep learning revolution
* Most of the algorithmic background have been around for few decades.

## Perceptron
* Most fundamental building block of the NNs
* Inspired by the neuron
* Forward Propagation: Inputs -> Weights -> Sum -> Non-Linearity (Activation function)-> Output
* Linear notation: y = g (w + \sum x*w )
* Matrix notation: y = g (w + X*W)
* Sigmoid activation function: `tf.nn.sigmoid(z)` - useful for modelling probabilities, because it colapses the input to the 0-1 range
* Hyperbolic activation function: `tf.nn.tanh(z)`
* ReLU activation function: `tf.nn.relu(z)` - popular, easy to compute, non-linear
* Activation functions: introduce non-linearities into the network, linear decision boundaries are not adequate for complex decisions

# Building NN with perceptrons
* NN contain multiple perceptrons
* Single Layer NN contain a single hidden layer between the input and the output. Two weight matrices needed: input -> hidden layer and hidden layer -> output
* Example SLNN in TF
```
from tf.keras.layers import *

inputs = Inputs(m)
hidden = Dense(d)(inputs)
outputs = Dense(2)(hidden)
model = Model(inputs, outputs)
```
* Dense Layer connects all of the nodes on the left side to each node on the right side.
* Deep NN contain multiple hidden layers between the input and the output

# Applying NN
* The loss of a NN measures the cost incurred from incorrect predictions. It computes the difference between the generated output and the expected output.
* The empirical loss is the mean of all losses for each datapoint.
* Binary Cross Entropy Loss computes the loss between the 0-1 output and the actual output; good for models that output a probability between 0 and 1
* Mean Squared Error Loss can be used with regression models that output continious real numbers.

# Training NN
* Loss optimization: finding the suitable weights for the NN such that the empirical loss has the smallest value.
* Gradient descent is used for the loss optimization calculations. Uing backprogagation to compute the gradient.
* 
