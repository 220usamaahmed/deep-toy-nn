import math
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
		
		self.activation_function = lambda x : 1 / (1 + math.exp(-x))


	def initialize(self, previous_layer, next_layer):
		self.previous_layer = previous_layer
		self.next_layer = next_layer
		self.W = Matrix(self.size, previous_layer.size).randomize(-1, 1)
		self.B = Matrix(self.size, 1).randomize(-1, 1)


	def feed_forward(self):
		self.values = (self.W * self.previous_layer.values + self.B).map_function(self.activation_function)


	def calculate_errors(self):
		self.E = self.next_layer.W.get_transpose() * self.next_layer.E


	def adjust_parameters(self, learning_rate):
		# print(self.values)
		# print(self.E)
		gradients = self.values.map_function(lambda x : x * (1 - x)).get_hadamard_product(self.E).get_scalar_multiple(learning_rate)
		# print(gradients)





		self.W.add(gradients * self.previous_layer.values.get_transpose())
		self.B.add(gradients)


class OutputLayer(Layer):

	def __init__(self, size):
		Layer.__init__(self, size)

		self.previous_layer = None

		self.W = None
		self.B = None
		self.E = None
		
		self.activation_function = lambda x : 1 / (1 + math.exp(-x))


	def initialize(self, previous_layer):
		self.previous_layer = previous_layer
		self.W = Matrix(self.size, previous_layer.size).randomize(-1, 1)
		self.B = Matrix(self.size, 1).randomize(-1, 1)


	def feed_forward(self):
		self.values = ((self.W * self.previous_layer.values) + self.B).map_function(self.activation_function)


	def calculate_errors(self, target_arr):
		if len(target_arr) == self.size:
			self.E = Matrix.from_list(target_arr, self.size, 1) - self.values
		else: raise ValueError("Incorrect target size.")


	def adjust_parameters(self, learning_rate):
		gradients = self.values.map_function(lambda x : x * (1 - x)).get_hadamard_product(self.E).get_scalar_multiple(learning_rate)
		self.W.add(gradients * self.previous_layer.values.get_transpose())
		self.B.add(gradients)