# Introduction to Deep Learning

[Video](https://www.youtube.com/watch?v=5v1JnYv_yWs&index=1&list=PLtBw6njQRU-rwp5__7C0oIVt26ZgjG9NI)

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

## Building NN with perceptrons
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

## pplying NN
* The loss of a NN measures the cost incurred from incorrect predictions. It computes the difference between the generated output and the expected output.
* The empirical loss is the mean of all losses for each datapoint.
* Binary Cross Entropy Loss computes the loss between the 0-1 output and the actual output; good for models that output a probability between 0 and 1
* Mean Squared Error Loss can be used with regression models that output continious real numbers.

## Training NN
* Loss optimization: finding the suitable weights for the NN such that the empirical loss has the smallest value.
* Gradient descent is used for the loss optimization calculations.
1. Initialize weights randomly
2. Loop until convergence:
2.1 Compute gradient
2.2 Update weights
3. Return weights
* Backpropagation.

## NN in Practice: Optimization
* How to set the learning rate? W <- W - n dJ(W) / d(W)
* Small learning rate converges slowly and can get stuck at a local minima
* Large learning rate overshoot, become unstable and diverge
* How to choose the learning rate?
1. Greedy approach by trying different ones
2. Design an adaptive algorithm which adapts to the loss landscape.
* Adaptive Learning Rate Algorithms http://ruder.io/optimizing-gradient-descent
1. Momentum: `tf.train.MomentumOptimizer`
2. Adagrad: `tf.train.AdagradOptimizer`
3. Adadelta: `tf.train.AdadeltaOptimizer`
4. Adam: `tf.train.AdamOptimizer`
5. RMSProp: `tf.train.RMSPropOptimizer`

## Neural Networks in Practice: Mini-batches
* Stochastic Gradient Descent:
1. Initialize weights randomly
2. Loop until convergence:
2.1 Pick batch of B data points
2.2 Compute gradient
2.3 Update weights
3. Return weights
* Smoother convergence, quicker to compute due to parallelization

##NN in Practice: Overfitting
* Underfitting: Model does not have capacity to fully learn the data.
* Overfitting: Model is too complex and does not generalize well.
* Regularization: technique that contrains the optimization problem to discourage complex models; improves the generalization of our model on unseen data
* Regularization Techniques:
1. Dropout - randomly set some activations to 0 to remove any dependances on a single node. `tf.keras.layers.Dropout(0.5)`
2. Early Stopping - stop testing once we start overfitting; train until testing loss starts rising up
