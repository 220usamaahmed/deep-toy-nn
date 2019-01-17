from nn import FeedForwardNeuralNetwork as FFNN

neural_network = FFNN(2, 1, [2, 2])
print(neural_network.predict([1, 3]))