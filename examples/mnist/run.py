import random
from mnist import MNIST
from lib.nn import FeedForwardNeuralNetwork as FFNN


mndata = MNIST('./samples')
images, labels = mndata.load_training()
sample_size = len(labels)

nn = FFNN(784, 10, [16, 16])

training_epocs = 20000

print("Training...")
for _ in range(training_epocs):
	i = random.randrange(100, sample_size)
	image = images[i]

	input_arr = [min(1, pixel) for pixel in image]
	output_arr = [0]*10
	output_arr[labels[i]] = 1

	if (len(input_arr) == 784): 
		nn.train(input_arr, output_arr)


correct = 0
for i, image in enumerate(images[:100]):
	input_arr = [min(1, pixel) for pixel in image]

	prediction = nn.predict(input_arr).to_list()
	if (prediction.index(max(prediction)) == labels[i]): 
		correct += 1

print("{} Correct out of 100. training epocs: {}".format(correct, training_epocs))



















# print(mndata.display(images[1]))

# column = 0
# for pixel in images[1]:
# 	column += 1

# 	print(min(1, pixel), end=" ")

# 	if column % 28 == 0: print()


# print(len(images[1]))

