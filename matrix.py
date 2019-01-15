import random


class Matrix:

	def __init__(self, rows, columns):
		self.rows = rows
		self.columns = columns
		self.data = [[0 for _ in range(columns)] for _ in range(rows)]


	def __repr__(self):
		string_repr = ""
		for row in self.data:
			for element in row:
				string_repr += str(element) + " "
			string_repr += "\n"

		return string_repr


	def __add__(self, other):
		if self.rows == other.rows and self.columns == other.columns:
			
			result = Matrix(self.rows, self.columns)

			for r in range(result.rows):
				for c in range(result.columns):
					result.data[r][c] = self.data[r][c] + other.data[r][c]

			return result
		else:
			return ValueError("The dimensions of the matracies should be equal.")
	

	def __sub__(self, other):
		if self.rows == other.rows and self.columns == other.columns:
			result = Matrix(self.rows, self.columns)

			for r in range(result.rows):
				for c in range(self.columns):
					result.data[r][c] = self.data[r][c] - other.data[r][c]
			
			return result
		else:
			return ValueError("The dimensions of the matracies should be equal.")


	def __mul__(self, other):
		if self.columns == other.rows:
			result = Matrix(self.rows, other.columns)
			for r in range(result.rows):
				for c in range(result.columns):
					for k in range(other.rows):
						result.data[r][c] += self.data[r][k] * other.data[k][c]
			return result
		else:
			return ValueError("Number of columns of first matrix must match number of rows of second matrix.")


	def toList(self):
		result_list = []
		for r in range(self.rows):
			for c in range(self.columns):
				result_list.append(self.data[r][c])

		return result_list


	def fromList(arr, rows, columns):
		if (rows * columns == len(arr)):

			result = Matrix(rows, columns)

			for r in range(rows):
				for c in range(columns):
					result.data[r][c] = arr[r * columns + c]

			return result
		else:
			return ValueError("List of incorrect size provided.")


	def add(self, other):
		if (self.rows == other.rows and self.columns == other.columns):
			for r in range(self.rows):
				for c in range(self.columns):
					self.data[r][c] += other.data[r][c]

			return self
		else:
			return ValueError("The dimensions of the matracies should be equal.")


	def map_function(self, func):
		for r in range(self.rows):
			for c in range(self.columns):
				self.data[r][c] = func(self.data[r][c])

		return self


	def randomize(self, lower=0, upper=1, seed=True):
		if seed: random.seed(None)
		for r in range(self.rows):
			for c in range(self.columns):
				self.data[r][c] = lower + random.random() * (upper - lower)

		return self


	def get_transpose(self):
		transposed_matrix = Matrix(self.columns, self.rows)
		for r in range(transposed_matrix.rows):
			for c in range(transposed_matrix.columns):
				transposed_matrix.data[r][c] = self.data[c][r]

		return transposed_matrix


	def get_scalar_multiple(self, scalar):
		scaled_matrix = Matrix(self.rows, self.columns)
		for r in range(scaled_matrix.rows):
			for c in range(scaled_matrix.columns):
				scaled_matrix.data[r][c] = self.data[r][c] * scalar

		return scaled_matrix


	def get_hadamard_product(self, other):
		if (self.rows == other.rows and self.columns == other.columns):
			hadamard_product = Matrix(self.rows, self.columns)
			for r in range(self.rows):
				for c in range(self.columns):
					hadamard_product.data[r][c] = self.data[r][c] * other.data[r][c]
			return hadamard_product
		else:
			return ValueError("The dimensions of the matracies should be equal.")