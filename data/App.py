import sys

from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark import SparkFiles # access submited files

from py2neo import Graph

service_ip = "bolt://neo4j-helm-neo4j:7687"
datafolder = "/opt/spark/data"

sys.path.append(datafolder)
# import pytorch_DGCNN from data folder of spark distribution
from pytorch_DGCNN.predictor import *
from pytorch_DGCNN.util import GNNGraph
from pytorch_DGCNN.Logger import getlogger
import scipy.sparse as ssp

import networkx as nx
import pickle as pkl
import numpy as np
import time

'''
██╗░░░██╗████████╗██╗██╗░░░░░░██████╗
██║░░░██║╚══██╔══╝██║██║░░░░░██╔════╝
██║░░░██║░░░██║░░░██║██║░░░░░╚█████╗░
██║░░░██║░░░██║░░░██║██║░░░░░░╚═══██╗
╚██████╔╝░░░██║░░░██║███████╗██████╔╝
░╚═════╝░░░░╚═╝░░░╚═╝╚══════╝╚═════╝░
'''

def links2subgraphs(batch, hop):
	# connect service
	graph = Graph(service_ip)
	graphs = []
	batch_poses = [[], []]
	for pair in batch:
		gnn_graph = link2subgraph(graph, pair, hop)
		graphs.append(gnn_graph)
		batch_poses[0].append(pair[0])
		batch_poses[1].append(pair[1])
	logger = getlogger('Node '+str(os.getpid()))
	logger.info("Worker done extracting batches...")
	# serialized graphs and test positions
	pickled = pkl.dumps((graphs, batch_poses))
	return pickled

# applying apoc subgraph procedure
def link2subgraph(graph, pair, hop):
	src, dst = pair[0], pair[1]
	query = """
		MATCH (n:Node)
		WHERE n.id = %d or n.id = %d
		WITH n
		CALL apoc.path.subgraphNodes(n, {maxLevel:%d}) YIELD node
		WITH DISTINCT node
		WITH collect(node) as nds
		MATCH (src:Node)
		MATCH (dst:Node)
		WHERE src IN nds AND dst in nds
		MATCH (src)-[e:CONNECTION]->(dst)
		RETURN collect(e) AS edgs, nds
	""" % (src, dst, hop)
	results = list(graph.run(query))
	nodes = [int(node['id']) for node in results[0]['nds']]
	# put src and dst on top
	nodes.remove(src)
	nodes.remove(dst)
	nodes = [src, dst] + list(nodes)
	# given labeling function functions with indeces, not ids
	nodes_idx = list(range(0, len(nodes))) 
	# edges need to be constructed between known indeces
	nodes_map = dict(zip(nodes, nodes_idx)) # to access nodes index w. id
	edges = [(nodes_map[int(edge.nodes[0]['id'])], nodes_map[int(edge.nodes[1]['id'])]) for edge in results[0]['edgs']]
	# construct networkx graph from given nodes and edges:
	g = nx.Graph()
	g.add_nodes_from(nodes_idx)
	g.add_edges_from(edges)
	# construct sparse adjacency matrix for use of SEAL functions
	subgraph = nx.to_scipy_sparse_matrix(g, weight=None, format='csc', dtype=np.float64)
	# labels nodes in subgraphs
	labels = node_label(subgraph).tolist()
	# features TODO
	features = None
	# TODO check whether it is corret. Perhaps we should pass different values here.
	g_label = 1
	# remove edge between target nodes
	if g.has_edge(0, 1):
		g.remove_edge(0, 1)
	gnn_graph = GNNGraph(g, g_label, labels)
	return gnn_graph

def node_label(subgraph):
    # an implementation of the proposed double-radius node labeling (DRNL)
    K = subgraph.shape[0]
    subgraph_wo0 = subgraph[1:, 1:]
    subgraph_wo1 = subgraph[[0]+list(range(2, K)), :][:, [0]+list(range(2, K))]
    dist_to_0 = ssp.csgraph.shortest_path(subgraph_wo0, directed=False, unweighted=True)
    dist_to_0 = dist_to_0[1:, 0]
    dist_to_1 = ssp.csgraph.shortest_path(subgraph_wo1, directed=False, unweighted=True)
    dist_to_1 = dist_to_1[1:, 0]
    d = (dist_to_0 + dist_to_1).astype(int)
    d_over_2, d_mod_2 = np.divmod(d, 2)
    labels = 1 + np.minimum(dist_to_0, dist_to_1).astype(int) + d_over_2 * (d_over_2 + d_mod_2 - 1)
    labels = np.concatenate((np.array([1, 1]), labels))
    labels[np.isinf(labels)] = 0
    labels[labels>1e6] = 0  # set inf labels to 0
    labels[labels<-1e6] = 0  # set -inf labels to 0
    return labels

def apply_network(dataset:str, serialized):
	hyperparams_route = SparkFiles.get(f'{dataset}_hyper.pkl')
	model_route = SparkFiles.get(f'{dataset}_model.pth')
	predictor = Predictor(hyperparams_route, model_route)
	return predictor.predict(serialized)

'''
░█████╗░██████╗░██████╗░
██╔══██╗██╔══██╗██╔══██╗
███████║██████╔╝██████╔╝
██╔══██║██╔═══╝░██╔═══╝░
██║░░██║██║░░░░░██║░░░░░
╚═╝░░╚═╝╚═╝░░░░░╚═╝░░░░░ 
'''

def main():
	# TODO read configuration settings from sys.argv
	dataset = "USAir"
	batch_inprior = True
	hop = 1 # try with 2
	batch_size = 50

	# create Spark context with Spark configuration
	spark = SparkSession\
			.builder\
			.appName("UginDGCNN")\
			.getOrCreate()
	sc = spark.sparkContext

	logger = getlogger('Node '+str(os.getpid()))
	logger.info("Spark Context established, going though app logic...")

	# zipped package
	zipped_pkg = os.path.join(datafolder, "dependencies.zip")
	assert os.path.exists(zipped_pkg)
	sc.addPyFile(zipped_pkg)

	hyperparams = os.path.join(datafolder, f"models/{dataset}_hyper.pkl")
	assert os.path.exists(hyperparams) 
	sc.addFile(hyperparams)

	model = os.path.join(datafolder, f"models/{dataset}_model.pth")
	assert os.path.exists(model) 
	sc.addFile(model)

	build = os.path.join(datafolder, "build")
	build_paths = [\
		os.path.join(build, "dll/libgnn.d"),\
		os.path.join(build, "dll/libgnn.so"),\
		os.path.join(build, "lib/config.d"),\
		os.path.join(build, "lib/config.o"),\
		os.path.join(build, "lib/graph_struct.d"),\
		os.path.join(build, "lib/graph_struct.o"),\
		os.path.join(build, "lib/msg_pass.d"),\
		os.path.join(build, "lib/msg_pass.o")\
	]

	for build_path in build_paths:
		assert os.path.exists(build_path)
		sc.addFile(build_path)

	logger.info("Build paths attached...")

	# read test data files:
	positives_file = os.path.join(datafolder, "prediction_data", dataset+"_positives_test.txt") 
	positives = []
	with open(positives_file, 'r') as f:
		for line in f:
			pair = line.strip().split(" ")
			positives.append((int(pair[0]), int(pair[1])))
	negatives_file = os.path.join(datafolder, "prediction_data", dataset+"_negatives_test.txt") 
	negatives = []
	with open(negatives_file, 'r') as f:
		for line in f:
			pair = line.strip().split(" ")
			negatives.append((int(pair[0]), int(pair[1])))

	prediction_data = positives + negatives
	
	logger.info("Data sampled...")

	# IN-PRIOR batch solution:
	batched_prediction_data = []
	batch_data = []
	for i, pair in enumerate(prediction_data):
		batch_data.append(pair)
		if len(batch_data) == batch_size or i == (len(prediction_data)-1):
			batched_prediction_data.append(batch_data)
			batch_data = []

	prediction_rdd = sc.parallelize(batched_prediction_data)

	logger.info("Data parallelized...")

	# extract subgraphs:
	prediction_subgraphs = prediction_rdd.map(lambda batch: links2subgraphs(batch, hop))

	start = time.time()
	subgraphs = prediction_subgraphs.collect()
	end = time.time()
	logger.info("Subgraph extraction with collect took " + str(end-start) + " seconds.")

	subgraphs_prediction = sc.parallelize(subgraphs)
	# post-facto batch solution todo:

	# perform prediction:
	predictions = subgraphs_prediction.map(lambda graph: apply_network(dataset, graph)) # was prediction_subgraphs

	# extract results with .collect() method:
	start = time.time()
	results = predictions.collect()
	end = time.time()
	logger.info("Prediction on subgraphs took " + str(end-start) + " seconds.")

	for i, record in enumerate(results):
		np.savetxt("/opt/spark/work-dir/results_batch_"+str(i), record, fmt=['%d', '%d', '%1.2f'])

	logger.info("Results calculation complete!")

	time.sleep(60*10)

if __name__ == "__main__":
	main()


	