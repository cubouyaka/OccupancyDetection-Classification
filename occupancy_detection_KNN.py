# coding=utf-8
"""
	Machine Learning Project - Occupancy Detection [Classification] using K Nearest Neigbors
	Fontaine Victor - M2 Informatique Paris Diderot - 2018
"""

import math
import random
from sklearn.model_selection import KFold

def read_data(filename):
	"""Reads a occupancy-detection dataset.
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

	for line in file.readlines():
		split = line.split(',')
		xi = [float(split[i]) for i in range (2,7)]
		# xi = [float(split[2])]
		Y.append(True if (int(split[7].replace('\n','')) == 1) else False)
		X.append(xi)
	file.close()

	return X,Y

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
	if seed == -1:
		random.seed()
	else:
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


def simple_distance(data1, data2):
	"""Computes the Euclidian distance between data1 and data2.
	Args:
		data1: a list of numbers: the coordinates of the first vector.
		data2: a list of numbers: the coordinates of the second vector (same length as data1).
	Returns:
		The Euclidian distance: sqrt(sum((data1[i]-data2[i])^2)).
	""" 
	n = len(data1) # which is equal to len(data2)
	sum = 0
	for i in range(n):
		sum += (data1[i]-data2[i])**2

	return math.sqrt(sum)

def calcul_variances(data):
	"""Calcul the variances for each components
	Args:
		data: a list of vectors, composed of 5 components (floats)
	Returns:
		A list of variances for each components of the data
	"""
	n = len(data[0])
	means = [0]*n
	variances = [0]*n

	for i in range(n):
		for x in data :
			means[i] += x[i]
		means[i] /= len(data)

	for i in range(n):
		for x in data:
			variances[i] += (x[i]-means[i])**2
		variances[i] /= (len(data)-1)
	return variances

def simple_distance2(data1, data2, variances):
	"""Computes the Euclidian distance weighted by variances between data1 and data2.
	Args:
		data1: a list of numbers: the coordinates of the first vector.
		data2: a list of numbers: the coordinates of the second vector (same length as data1).
	Returns:
		The Euclidian distance: sqrt(sum((data1[i]-data2[i])^2)).
	""" 
	sum = 0
	n = len(data1) # which is equal to len(data2)
	
	for i in range(n-1):
		sum += (((data1[i]-data2[i]))**2)/variances[i]

	return math.sqrt(sum)

def k_nearest_neighbors(x, points, dist_function, k):
	"""Returns the indices of the k elements of “points” that are closest to “x”
	sorted by distance: nearest neighbor first.
	Args:
		x: a list of numbers: a N-dimensional vector.
		points: a list of list of numbers: a list of N-dimensional vectors.
		dist_function: a function taking two N-dimensional vectors as
			arguments and returning a number. Just like simple_distance.
		k: an integer. Must be smaller or equal to the length of “points”.
	Returns:
		A list of integers: the indices of the k elements of “points” that are
		closest to “x” according to the distance function dist_function.
	""" 
	nearest = [-1 for q in range(k)] #equivalent a [-1]*k
	nearest_dist = [-1]*k
	n = len(points)
	variances = calcul_variances(points)

	for i in range(n):
		d = dist_function(x,points[i])
		# d = dist_function(x,points[i],variances)
		for l in range(k):
			if(nearest[l] == -1 or d < nearest_dist[l]):
				nearest.insert(l,i)
				nearest_dist.insert(l,d)
				break

	return nearest[:k]

def is_occupied_knn(x, train_x, train_y, dist_function, k):
	"""Predicts whether the room is occupied or not, using KNN.
	Args:
		x: A list of floats representing a data point (in the occupancy dataset,
			that's 5 floats) that we want to analyze.
		train_x: A list of list of floats representing the data points of
			the training set.
		train_y: A list of booleans representing the classification of
			the training set: True if the corresponding data point is
			occupied, False if not. Same length as 'train_x'.
		dist_function: Same as in k_nearest_neighbors().
		k: Same as in k_nearest_neighbors().
	Returns:
		A boolean: True if the data point x is predicted to be occupied, False
		if not
	"""
	nearest = k_nearest_neighbors(x,train_x,dist_function,k) #of length k
	nb_M = 0
	nb_B = 0
	for i in range(k):
		if train_y[nearest[i]]:
			nb_M += 1
		else:
			nb_B += 1

	if(nb_M == nb_B):
		return train_y[nearest[0]]

	return (nb_M >= nb_B)

def eval_occupancy_classifier(train_x, train_y, test_x, test_y, classifier, dist_function, k):
	"""Evaluates a occupancy KNN classifier.
	This takes a training and test datasets.
	It then runs a classifier function on all data points in the test,
	and compares the predicted state with the actual state of the room.
	Args:
		train_x: A list of lists of floats: the training data points.
		train_y: A list of booleans: the training data class (True = occupied,
			False = not occupied)
		test_x: Like train_x but for the test dataset.
		test_y: Like train_y but for the test dataset.
		classifier: A function, like is_occupied_knn.
		dist_function: A distance function, like simple_distance.
		k: An integer. See k_nearest_neighbors().
	Returns:
		A float: the error rate of the classifier on the test dataset. This is
		a value in [0,1]: 0 means no error (we got it all correctly), 1 means
		we made a mistake every time. Note that choosing randomly yields an error
		rate of about 0.5.
	"""
	nb_success = 0
	n = len(test_x) #which is equal to len(test_y)

	for i in range(n):
		nb_success += 1 if (is_occupied_knn(test_x[i], train_x, train_y, dist_function, k) == test_y[i]) else 0

	return 1.0 - (nb_success / float(n))

def sampled_range(mini, maxi, num):
	"""Calcul a sample of 'num' values "well distributed" between 'mini' and 'max'
	Args:
		mini: minimal value of the sample (int)
		maxi: maximal value of the sample (int)
		nim: number of value wanted in the sample (int)
	Returns:
		A sample of 'num' values in range(mini,maxi)
	"""
	if not num:
		return []
	lmini = math.log(mini)
	lmaxi = math.log(maxi)
	ldelta = (lmaxi - lmini) / (num - 1)
	out = [x for x in set([int(math.exp(lmini + i * ldelta)) for i in range(num)])]
	out.sort()
	return out

def find_best_k(train_x, train_y, dist_function):
	"""Uses cross-validation (10 folds) to find the best K for is_occupied_knn().
	Args:
		train_x: A list of lists of floats: the training data points.
		train_y: A list of booleans: the training data class (True = occupied,
			False = not occupied)
		dist_function: A distance function, like simple_distance.
	Returns:
		An integer: the ideal value for K in a K-nearest-neighbor classifier.
	"""
	fold = 10
	n = len(train_x)
	sample = sampled_range(1,n/10,fold)
	n_sample = len(sample)

	kf = KFold(fold)
	for train_index, test_index in kf.split(train_x):
		trainX, testX = [train_x[i] for i in train_index], [train_x[j] for j in test_index]
		trainY, testY = [train_y[k] for k in train_index], [train_y[l] for l in test_index]

	min_eval = eval_occupancy_classifier(trainX, trainY, testX, testY, is_occupied_knn, dist_function, sample[0])
	min_k = sample[0]

	for i in range(n_sample) : 
		if eval_occupancy_classifier(trainX, trainY, testX, testY, is_occupied_knn, dist_function, sample[i]) < min_eval : 
			min_eval = eval_occupancy_classifier(trainX, trainY, testX, testY, is_occupied_knn, dist_function, sample[i])
			min_k = sample[i]

	return min_k


""" MAIN """

split_lines('dataset_clean.txt',-1,'train','test')
(train_x,train_y) = read_data('train')
(test_x,test_y) = read_data('test')

best_k = find_best_k(train_x, train_y, simple_distance)
print("The best k is : "+str(best_k))
print("Evaluation of the Occupancy Detector with the best k ("+str(best_k)+") is : "+str(eval_occupancy_classifier(train_x, train_y, test_x, test_y, is_occupied_knn, simple_distance, best_k)))















