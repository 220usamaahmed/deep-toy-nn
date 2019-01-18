from nn import FeedForwardNeuralNetwork as FFNN

neural_network = FFNN(2, 1, [2, 2])
# neural_network.predict([1, 1])
neural_network.train([1, 0], [1])