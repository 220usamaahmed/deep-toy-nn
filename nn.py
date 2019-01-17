from layers import InputLayer, HiddenLayer, OutputLayer
from matrix import Matrix


class FeedForwardNeuralNetwork:

	def __init__(self, input_size, output_size, hidden_layer_sizes=[]):

		if len(hidden_layer_sizes) == 0: raise ValueError("There must be atleast one hidden layer.")

		self.input_layer = InputLayer(input_size)
		
		self.hidden_layers = []
		for i, hidden_layer_size in enumerate(hidden_layer_sizes):
			if (i == 0): new_hidden_layer = HiddenLayer(hidden_layer_size, input_size)
			else: new_hidden_layer = HiddenLayer(hidden_layer_size, hidden_layer_sizes[i - 1])
			self.hidden_layers.append(new_hidden_layer)

		self.output_layer = OutputLayer(output_size, hidden_layer_sizes[-1])


	def predict(self, input_arr):
		self.input_layer.set_values(input_arr)
		
		for i, hidden_layer in enumerate(self.hidden_layers):
			if i == 0: hidden_layer.feed_forward(self.input_layer.values)
			else: hidden_layer.feed_forward(self.hidden_layers[i - 1].values)

		self.output_layer.feed_forward(self.hidden_layers[-1].values)

		return self.output_layer.values


	def train(self, input_arr, target_arr):
		pass