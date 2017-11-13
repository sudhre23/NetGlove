import numpy as np
import networkx as nx
import random

import msgpack
from collections import defaultdict

class NodeCooccur():
	def __init__(self, nx_G, weighted):
		self.G = nx_G
		self.weighted = weighted
		self.node_map,self.rev_map = self.gen_node_map()
		#self.preprocess_transition_probs()

	def get_node_map(self):
		return self.node_map

	def gen_node_map(self):
		G = self.G
		nodes = sorted(list(G.nodes()))
		node_map = {}
		rev_map = {}
		ctr = 0
		for n in nodes:
			node_map[ctr] = n
			rev_map[n] = ctr
			ctr += 1
		return node_map,rev_map


	def build_cooccurence(self,distance_threshold):
		G = self.G
		
		if self.weighted:
			res = nx.shortest_path(G)
		else:
			res = nx.shortest_path_length(G)
		dist_cooccur = []
		for src in res:
		    for dest in res[src]:
		        if src != dest:
		        	m_src = self.rev_map[src]
		        	m_dest = self.rev_map[dest]
		        	#print 'accessing src : ',src,' dest : ',dest
		        	if self.weighted:
		        		path = res[src][dest]
		        		if len(path) > distance_threshold:
		        			continue
		        		l = 0
		        		for i in range(len(path) -1):
		        			l += G[path[i]][path[i+1]]['weight']
		        		dist_cooccur.append((m_src, m_dest, 1.0/l))
		        	else:
		        		dist_cooccur.append((m_src, m_dest, 1.0/(res[src][dest])))
		return dist_cooccur

	