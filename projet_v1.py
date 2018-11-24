# coding=utf-8

import math
import random

def read_data(filename):
	"""Reads a occupancy dataset, delete the first line.
	Args:
		filename: a string, the name of the input file.
	Returns:
		A pair (X,Y) of lists: N is the number of data points in the file
		- X is a list of N points: each point is a list of numbers
		corresponding to the data in the file describing a sample.
		- Y is a list of N booleans: Element #i is True if the data point
		#i described in X[i] is "occupancy", and False if not.
	"""

	file = open(filename,'r')

	X,Y = [],[]

	for line in file.readlines()[1:]:
		split = line.split(',')
		xi = [float(split[i]) for i in range (2,7)]
		Y.append(True if (int(split[7].replace('\n','')) == 1) else False)
		X.append(xi)
	file.close()

	return X,Y

X,Y = read_data('datatraining.txt')


def split_lines(input, seed, output1, output2):
	"""Distributes the lines of 'input' to 'output1' and 'output2' pseudo-randomly.

  	The output files should be approximately balanced (50/50 chance for each line
  	to go either to output1 or output2).
    
  	Args:
    	input: a string, the name of the input file.
    	seed: an integer, the seed of the pseudo-random generator used. The split
        	should be different with different seeds. Conversely, using the same
        	seed and the same input should yield exactly the same outputs.
    	output1: a string, the name of the first output file.
    	output2: a string, the name of the second output file.
  	"""
	random.seed(seed)
	file = open(input,'r')
	f_out1 = open(output1,'w')
	f_out2 = open(output2,'w')
	for line in file.readlines():
		if(random.random() < 0.5):
			f_out1.write(line)
		else:
			f_out2.write(line)

	file.close()
	f_out1.close()
	f_out2.close()


