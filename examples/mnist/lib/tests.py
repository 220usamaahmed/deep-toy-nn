import random
from nn import FeedForwardNeuralNetwork as FFNN


xor_data = [
	{
		'x': [1, 1],
		'label': [0],
	},
	{
		'x': [1, 0],	
		'label': [1],
	},
	{
		'x': [0, 1],
		'label': [1],
	},
	{
		'x': [0, 0],
		'label': [0],
	},
]


neural_network = FFNN(2, 1, [3, 2])
neural_network.learning_rate = 1.0

iterations = 4000
for i in range(iterations):
	dp = random.choice(xor_data)
	neural_network.train(dp['x'], dp['label'])
	if (i % 400 == 0): 
		neural_network.learning_rate -= 0.1
		print(neural_network.learning_rate)


print()
print()
print()


for dp in xor_data:
	print(neural_network.predict(dp['x']), dp['label'], '\n\n')
