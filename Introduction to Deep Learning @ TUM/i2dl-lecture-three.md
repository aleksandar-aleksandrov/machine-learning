# Introduction to NN

## Neural Network
* Linear score function is not adequate for complex tasks.
* Neural network represents nesting of functions:
1. 2-Layer NN: f = W max(0, Wx)
2. 3-Layer NN: f = W max(0, W max(0,Wx))
3. ... up to hundreds of layers

* Augmenting the expressive power through layers and non-linear functions
* Neurons in NN contain the following elements:
1. Linear function: Wx + b
2. Non-linearity activation: f(x)
3. Every neuron computes: f(Wx + b)

* NN consists of an input layer, multiple hidden layers and an output layer
* Why are activation functions needed? Without activation function we will end up with a linear model
* Activation functions
1. Sigmoid
2. tanh
3. ReLU
4. Leaky ReLU
5. Parametric ReLU
6. Maxout
7. ELU
* NN are based on and inspired by the brain, but are far away from the actual function of the brain!
* How does NN work? Given a dataset with ground truth training pairs find optimal weights using stochastic gradient descent which optimizes the loss function. Gradient is computed through backpropagation.

## Computational Graph
* NN is a computational graph, because it:
1. has compute nodes
2. has edges that connect nodes
3. is directional
4. is organized in layers

* Computational graph is a directed graph where nodes corresponds to variables or operations
* CG can be evaluted by the means of `Forward Pass`: Training Data -> Loss
* Backpropagation: update the weights of the network based on the result of the loss function: Gradient <- Loss
* The Flow of Gradients:
<pre>             
dL / dx = dz / dx * dL / dz <-----
                                   dz / dx Local Gradient
                                                          <-------- dL / dz
                                   dz / dy Local Gradient    
dL / dy = dz / dy * dL / dz <-----
</pre>
* Special cases of Backpropagation:
1. Two or more outputs -> sum the together
2. Loop? -> no loops in NNs

* Implementation of Compute Graph
1. forward() -> Forward Pass
2. backward() -> Backward Pass
3. Store results to allow for faster computation

* NN Operations can be vectorized due to multidimensional inputs/outputs.
* Possible memory problems due to high dimensionality of generated vectors/matrices.
