from matrix import Matrix


class Layer:

	def __init__(self, size):
		self.size = size
		self.values = Matrix(size, 1)


class InputLayer(Layer):

	def __init__(self, size):
		Layer.__init__(self, size)


	def set_values(self, input_arr):
		if len(input_arr) != self.size: raise ValueError("Incorrect input size.")
		self.values = Matrix.from_list(input_arr, self.size, 1)


class HiddenLayer(Layer):

	def __init__(self, size, previous_layer_size):
		Layer.__init__(self, size)

		self.W = Matrix(size, previous_layer_size).randomize()
		self.B = Matrix(size, 1).randomize()


	def feed_forward(self, previous_layer_values):
		self.values = self.W * previous_layer_values + self.B


class OutputLayer(Layer):

	def __init__(self, size, previous_layer_size):
		Layer.__init__(self, size)

		self.W = Matrix(size, previous_layer_size).randomize()
		self.B = Matrix(size, 1).randomize() # Should the output layer have biases?


	def feed_forward(self, previous_layer_values):
		self.values = self.W * previous_layer_values + self.B