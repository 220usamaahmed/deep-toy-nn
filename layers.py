from math import exp
from matrix import Matrix


class Layer:

	def __init__(self, size):
		self.size = size
		self.values = None


class InputLayer(Layer):

	def __init__(self, size):
		Layer.__init__(self, size)


	def set_values(self, input_arr):
		if len(input_arr) == self.size: self.values = Matrix.from_list(input_arr, self.size, 1)
		else: raise ValueError("Incorrect input size.")


class HiddenLayer(Layer):

	def __init__(self, size):
		Layer.__init__(self, size)

		self.previous_layer = None
		self.next_layer = None
		
		self.W = None
		self.B = None
		self.E = None
		
		self.activation_function = lambda x : 1 / (1 + exp(-x))


	def initialize(self, previous_layer, next_layer):
		self.previous_layer = previous_layer
		self.next_layer = next_layer
		self.W = Matrix(self.size, previous_layer.size).randomize()
		self.B = Matrix(self.size, 1).randomize()


	def feed_forward(self):
		self.values = (self.W * self.previous_layer.values + self.B).map_function(self.activation_function)


	def calculate_errors(self):
		self.E = self.next_layer.W.get_transpose() * self.next_layer.E


class OutputLayer(Layer):

	def __init__(self, size):
		Layer.__init__(self, size)

		self.previous_layer = None

		self.W = None
		self.B = None
		self.E = None
		
		self.activation_function = lambda x : 1 / (1 + exp(-x))


	def initialize(self, previous_layer):
		self.previous_layer = previous_layer
		self.W = Matrix(self.size, previous_layer.size).randomize()
		self.B = Matrix(self.size, 1).randomize()


	def feed_forward(self):
		self.values = (self.W * self.previous_layer.values + self.B).map_function(self.activation_function)


	def calculate_errors(self, target_arr):
		if len(target_arr) == self.size:
			self.E = Matrix.from_list(target_arr, self.size, 1) - self.values