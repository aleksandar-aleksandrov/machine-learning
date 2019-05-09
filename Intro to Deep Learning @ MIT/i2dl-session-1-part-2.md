# Deep Sequence Modeling

[Video](https://www.youtube.com/watch?v=_h66BW-xNgk&index=1&list=PLtBw6njQRU-rwp5__7C0oIVt26ZgjG9NI)

## Motivation

* A lot of data is not standalone, but interconnected into sequences (text, sound, etc)

* How can one model sequence problems?
1. Using fixed window -> limited history, no long-term dependencies
2. Using entire sequence as set of counts ("bag of words") -> counts don't preserve order
3. Using really big fixed window -> no parameter sharing

* Sequence modeling: design criteria:
1. Handle variable-length sequences
2. Tranck long-term dependencies
3. Maintain information about order
4. Share parameters across the sequence

## Recurrent Neural Networks

* Suited for sequence problems (sentiment classification, music/text generation)
* RNN have loops in them, which allow for information to persist
* Internal state is called Ht = Fw(Ht-1,Xt)
* Information is being passed from one state to another -> recurrence relation between the time steps
* We update the internal hidden state at each time step.
* The output vector is dependant on the internal state. Yt = WhyHt
* The internal loop can be represented as multiple time-based copies of the same NN.
* The weight matrices stay the same at each time step!
* Total loss is equal to the loss at each time step

## Backpropagation Through Time (BPTT)
* Backpropagation at each time step and between the time steps
* Standard RNN gradient flow
1. If many values of Whh > 1 -> exploding gradients -> solution: Gradient clipping
2. If many values of Whh < 1 -> vanishing gradients -> Solutions: Activation function, Weight Initialization, Network architecture
* Why are vanishing gradients a problem -> long-term dependencies can't be transfered properly
* VG and activation functions: ReLU prevents shrinking the gradients when x > 0
* VG and parameter initialization: Init weights to identity matrix, biases to zero
* VG and gated cells: complex recurrent unit with gates to control what information is passed through (e.g. LSTM)

## Long Short Term Memory (LSTM) Networks
* Standard RNN consists of repeating module containing a simple computation node
* LSTM consists of repeating module containing interacting layers that control information flow
* LSTMs maintain an internal cell state
* IN LSTMs information is added or removed to cell state through gates
* How do LSTMs work?
1. LSTM forget irrelevant parts of the previous state (Forget)
2. LSTM selectively update cell state values (Update/Identify new information to be stored)
3. LSTMs use an output gate to output a transformed version of the cell state (Output)
* Key concepts:
1. Maintain a separate cell state
2. Use gates to control the flow of information (Forget -> Update -> Output)
3. Backpropagation from Ct to Ct-1 does not require matrix multiplication: uninterrupted gradient flow

## RNN Applications
* Music generation
* Sentiment classification
* Machine translation (Attention mechanisms)
