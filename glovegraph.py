from argparse import ArgumentParser
import codecs
from collections import Counter
import itertools
from functools import partial
import logging
from math import log
import os.path
import cPickle as pickle
from random import shuffle

import msgpack
import numpy as np
from scipy import sparse


class NodeEmbedding():

	def __init__(self, xmax,size,alpha,learning_rate,dimensions,output_path,iterations,node_map):
		self.x_max = xmax
		self.size = size
		self.alpha = alpha
		self.learning_rate = learning_rate
		self.dimensions = dimensions
		self.output_path = output_path
		self.iterations = iterations
		self.node_map = node_map

	def run_iter(self,data):

	    global_cost = 0

	    # We want to iterate over data randomly so as not to unintentionally
	    # bias the word vector contents
	    shuffle(data)
	    #i = 0
	    for (v_main, v_context, b_main, b_context, gradsq_W_main, gradsq_W_context,
	         gradsq_b_main, gradsq_b_context, cooccurrence) in data:

	        weight = (cooccurrence / self.x_max) ** self.alpha if cooccurrence < self.x_max else 1

	        cost_inner = (v_main.dot(v_context)- cooccurrence)

	        cost = weight * (cost_inner ** 2)

	        # Add weighted cost to the global cost tracker
	        global_cost += 0.5 * cost

	        # Compute gradients for node vectors
	        
	        grad_main = weight * cost_inner * v_context
	        grad_context = weight * cost_inner * v_main

	        # Update node vectors with the respective gradients
	        v_main -= (self.learning_rate * grad_main / np.sqrt(gradsq_W_main))
	        v_context -= (self.learning_rate * grad_context / np.sqrt(gradsq_W_context))


	        # Update squared gradient sums
	        gradsq_W_main += np.square(grad_main)
	        gradsq_W_context += np.square(grad_context)

	    return global_cost


	def train_glove(self,cooccurrences):
		

		n_nodes = self.size
		vector_size = self.dimensions

		W = (np.random.rand(n_nodes * 2, vector_size) - 0.5) / float(vector_size + 1)

		# Bias terms, each associated with a single vector.
		biases = (np.random.rand(n_nodes * 2) - 0.5) / float(vector_size + 1)

		# Training is done via adaptive gradient descent (AdaGrad). To make
		# this work we need to store the sum of squares of all previous
		# gradients.
		gradient_squared = np.ones((n_nodes * 2, vector_size),
		                           dtype=np.float64)

		# Sum of squared gradients for the bias terms.
		gradient_squared_biases = np.ones(n_nodes * 2, dtype=np.float64)

		# Build a reusable list from the given cooccurrence generator,
		# pre-fetching all necessary data.
		
		data = [(W[i_main], W[i_context + n_nodes],
		         biases[i_main : i_main + 1],
		         biases[i_context + n_nodes : i_context + n_nodes + 1],
		         gradient_squared[i_main], gradient_squared[i_context + n_nodes],
		         gradient_squared_biases[i_main : i_main + 1],
		         gradient_squared_biases[i_context + n_nodes
		                                 : i_context + n_nodes + 1],
		         cooccurrence)
		        for i_main, i_context, cooccurrence in cooccurrences]

		for i in range(self.iterations):
		    print "\tBeginning iteration ", i
		    cost = self.run_iter(data)
		    print "\t\tDone - cost : ",cost
	    	emb = W
	    	emb = self.buildNodeDict(emb,self.node_map)
	    	self.save_model(emb,self.output_path)

		return W



	def buildNodeDict(self,W,node_map):
		d = {}
		l = len(node_map)
		for i in range(l):
			d[node_map[i]] = (W[i] + W[i+l])/2.0
		return d

	def save_model(self,W, path):
	    with open(path, 'wb') as vector_f:
	        pickle.dump(W, vector_f, protocol=2)
