

import re



# Sigmoid (activation function)
def sigmoid(x):
    return 1.0/(1.0+math.exp(-x))

# Sigmoid derivative
def d_sigmoid(sigmoid):
    return sigmoid*(1.0-sigmoid)


# Read classified set and target values from input file
def read_input(filename = "input.txt"):
	file 	 = open(filename, "r")

	# Load the pre-classified vectors from file and the targets
	pre 	 = []
	targets  = []

	for line in file:
		group     = re.search("\( *(\d+) *\(((?:\d+ *)+) *\) *(\d+)\)", line).groups()

		_id  	  = int(group[0])
		_features = [int(i) for i in group[1].split(' ')] 
		_class 	  = int(group[2])

		pre.append(_features)
		targets.append(_class)

	file.close()

	return [pre, targets]


def main():
	[pre, targets] = read_input()
	print(targets)


if __name__ == '__main__':
    main()











