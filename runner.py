import argparse
import numpy as np
import networkx as nx
import gloveutils
import glovegraph
import cPickle as pickle

def parse_args():
	
	parser = argparse.ArgumentParser(description="Build a low dimensional vector representation of Graph nodes based on GloVe model.")

	parser.add_argument('--input', nargs='?', help='Input graph path')

	parser.add_argument('--output', nargs='?', help='Embeddings path')

	parser.add_argument('--vector-size', type=int, default=64, help='Number of dimensions. Default is 100.')

	parser.add_argument('--distance-threshold', type=int, default=5, help='Number of hops from source to visit. Default is 5.')

	parser.add_argument('--weighted', dest='weighted', action='store_true',
	                    help='Boolean specifying (un)weighted. Default is unweighted.')
	parser.add_argument('--unweighted', dest='unweighted', action='store_false')
	parser.set_defaults(weighted=False)
 
	parser.add_argument('--iterations', type=int, default=500, help='Number of training iterations')
	parser.add_argument('--learning-rate', type=float, default=0.05, help='Initial learning rate')
	

	return parser.parse_args()


def read_graph():
	G = None
	if 'gml' in args.input :
		G = nx.read_gml(args.input)
		for edge in G.edges():
				G[edge[0]][edge[1]]['weight'] = 1
	else:
		if args.weighted:
			print 'Got a weighted graph'
			G = nx.read_edgelist(args.input, nodetype=int, create_using=nx.DiGraph())
		else:
			G = nx.read_edgelist(args.input, nodetype=int, create_using=nx.DiGraph())
			for edge in G.edges():
				G[edge[0]][edge[1]] ['weight'] = 1
		G = G.to_undirected()
	return G

def save_model(W, args):
    with open(args.output, 'wb') as vector_f:
        pickle.dump(W, vector_f, protocol=2)

def buildNodeDict(W,node_map):
	d = {}
	l = len(node_map)
	for i in range(l):
		d[node_map[i]] = (W[i] + W[i+l])/2.0
	return d

def main(args):
	nx_G = read_graph()
	G = gloveutils.NodeCooccur(nx_G,args.weighted)
	coccur = G.build_cooccurence(args.distance_threshold)
	print 'built cooccurences of size ',len(coccur)
	node_map = G.get_node_map()
	xmax = 0.25
	alpha = 0.75
	model = glovegraph.NodeEmbedding(xmax,nx_G.number_of_nodes(),alpha,args.learning_rate,args.vector_size,args.output,args.iterations,node_map)
	emb = model.train_glove(coccur)
	emb = buildNodeDict(emb,node_map)
	save_model(emb,args)

if __name__ == "__main__":
	args = parse_args()
	main(args)
