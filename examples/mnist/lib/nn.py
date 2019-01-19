from lib.layers import InputLayer, HiddenLayer, OutputLayer
from lib.matrix import Matrix


class FeedForwardNeuralNetwork:

	def __init__(self, input_size, output_size, hidden_layer_sizes):

		self.learning_rate = 0.1

		self.input_layer = InputLayer(input_size)
		self.output_layer = OutputLayer(output_size)
		self.hidden_layers = [HiddenLayer(hidden_layer_size) for hidden_layer_size in hidden_layer_sizes]

		for i, hidden_layer in enumerate(self.hidden_layers):
			if i == 0 and i == len(self.hidden_layers) - 1: hidden_layer.initialize(self.input_layer, self.output_layer)
			elif i == 0: hidden_layer.initialize(self.input_layer, self.hidden_layers[i + 1])
			elif i == len(self.hidden_layers) - 1: hidden_layer.initialize(self.hidden_layers[i - 1], self.output_layer)
			else: hidden_layer.initialize(self.hidden_layers[i - 1], self.hidden_layers[i + 1])

		if (len(self.hidden_layers)): self.output_layer.initialize(self.hidden_layers[-1])
		else: self.output_layer.initialize(self.input_layer)


	def predict(self, input_arr):
		self.input_layer.set_values(input_arr)

		for hidden_layer in self.hidden_layers:
			hidden_layer.feed_forward()

		self.output_layer.feed_forward()

		return self.output_layer.values


	def train(self, input_arr, target_arr):
		self.predict(input_arr)

		self.output_layer.calculate_errors(target_arr)
		for hidden_layer in reversed(self.hidden_layers):
			hidden_layer.calculate_errors()

		self.output_layer.adjust_parameters(self.learning_rate)
		for hidden_layer in reversed(self.hidden_layers):
			hidden_layer.adjust_parameters(self.learning_rate)
