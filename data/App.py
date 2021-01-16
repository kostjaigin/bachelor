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
import scipy.io as sio

import networkx as nx
import pickle as pkl
import numpy as np
import time

'''
	Separate parts experiments and time/performance measurements
	See App.py for full system time/performance measurements
	TODO change this files name to Appdd.py
	nonono: lets leave full system here and
	 move calculations for extraction to applyneo4j (change name) and predictions to apply network (change name)
'''

'''
███████╗██╗░░░██╗███╗░░██╗░█████╗░░██████╗
██╔════╝██║░░░██║████╗░██║██╔══██╗██╔════╝
█████╗░░██║░░░██║██╔██╗██║██║░░╚═╝╚█████╗░
██╔══╝░░██║░░░██║██║╚████║██║░░██╗░╚═══██╗
██║░░░░░╚██████╔╝██║░╚███║╚█████╔╝██████╔╝
╚═╝░░░░░░╚═════╝░╚═╝░░╚══╝░╚════╝░╚═════╝░
'''

def batches2subgraphs(batch, hop:int, db:bool, dataset:str = None):
	if db: 
		# use db for graph extraction
		# connect service
		graph = Graph(service_ip)
	else:
		# if no db, dataset name required
		assert dataset is not None
	graphs = []
	batch_poses = [[], []]
	times = []
	for pair in batch:

		start = time.time()
		if db:
			gnn_graph = link2subgraph_db(graph, pair, hop)
		else:
			gnn_graph = link2subgraph_adj(pair, hop, dataset)
		end = time.time()

		times.append(end-start)
		graphs.append(gnn_graph)
		batch_poses[0].append(pair[0])
		batch_poses[1].append(pair[1])
	
	logger = getlogger('Node '+str(os.getpid()))
	logger.info("Worker done extracting batches...")
	# serialized graphs and test positions
	pickled = pkl.dumps((graphs, batch_poses))
	return (pickled, times)

'''
	returns: pair tuple, GNNGraph subgraph, extraction time
'''
def link2subgraph(link, hop:int, db:bool, dataset:str = None):
	graphs = []
	if db: 
		# use db for graph extraction
		# connect service
		graph = Graph(service_ip) # bad design decision: node would recreate a graph instance and connection..
		start = time.time()
		gnn_graph = link2subgraph_db(graph, link, hop)
		end = time.time()
		return link, gnn_graph, (end-start)
	else:
		# if no db, dataset name required
		assert dataset is not None
		start = time.time()
		gnn_graph = link2subgraph_adj(link, hop, dataset)
		end = time.time()
		return (link, gnn_graph, (end-start))

def neighbors(fringe, A):
	# find all 1-hop neighbors of nodes in fringe from A
	res = set()
	for node in fringe:
		nei, _, _ = ssp.find(A[:, node])
		nei = set(nei)
		res = res.union(nei)
	return res

def link2subgraph_adj(pair, h, dataset):
	# Read graph in a normal manner:
	graphdata = SparkFiles.get(f'{dataset}.mat')
	assert os.path.exists(graphdata)
	data = sio.loadmat(graphdata)
	net = data['net']

	ind = pair
	A = net.copy()
	node_information = None
	max_nodes_per_hop = None

	# extract the h-hop enclosing subgraph around link 'ind'
	dist = 0
	nodes = set([ind[0], ind[1]])
	visited = set([ind[0], ind[1]])
	fringe = set([ind[0], ind[1]])
	nodes_dist = [0, 0]
	for dist in range(1, h+1):
		fringe = neighbors(fringe, A)
		fringe = fringe - visited
		visited = visited.union(fringe)
		if max_nodes_per_hop is not None:
			if max_nodes_per_hop < len(fringe):
				fringe = random.sample(fringe, max_nodes_per_hop)
		if len(fringe) == 0:
			break
		nodes = nodes.union(fringe)
		nodes_dist += [dist] * len(fringe)

	# move target nodes to top
	# added at 19:08 by Konstantin on 20.12.2020, remove LATER PLEASE
	nodes = list(nodes)
	nodes.sort()
	nodes.remove(ind[0])
	nodes.remove(ind[1])
	nodes = [ind[0], ind[1]] + list(nodes) 

	# only selected nodes as rows and columns
	# after this operation my ids disappear and are not important any more
	# the following nx graph will be created using shifted indices
	# , where ind[0] will become 0 and ind[1] will become 1
	subgraph = A[nodes, :][:, nodes] 

	# apply node-labeling
	labels = node_label(subgraph)

	# get node features
	features = None
	if node_information is not None:
		features = node_information[nodes]

	# construct nx graph
	g = nx.from_scipy_sparse_matrix(subgraph) # different indices

	# remove link between target nodes
	if g.has_edge(0, 1):
		g.remove_edge(0, 1)

	return GNNGraph(g, 1, labels.tolist())

def link2subgraph_db(graph, pair, hop):
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

def main(args):
	'''
	█ █▄░█ █ ▀█▀ █ ▄▀█ █░░ █ █▀ ▄▀█ ▀█▀ █ █▀█ █▄░█
	█ █░▀█ █ ░█░ █ █▀█ █▄▄ █ ▄█ █▀█ ░█░ █ █▄█ █░▀█
	'''
	spark = SparkSession\
			.builder\
			.appName("UginDGCNN")\
			.getOrCreate()
	sc = spark.sparkContext

	logger = getlogger('Node '+str(os.getpid()))
	logger.info("Spark Context established, going though app logic...")

	logger.info("Application params:\n" + args.print_attributes())

	'''
	█▀▄ █▀▀ █▀█ █▀▀ █▄░█ █▀▄ █▀▀ █▄░█ █▀▀ █ █▀▀ █▀
	█▄▀ ██▄ █▀▀ ██▄ █░▀█ █▄▀ ██▄ █░▀█ █▄▄ █ ██▄ ▄█
	'''

	zipped_pkg = os.path.join(datafolder, "dependencies.zip")
	assert os.path.exists(zipped_pkg)
	sc.addPyFile(zipped_pkg)

	hyperparams = os.path.join(datafolder, f"models/{args.dataset}_hyper.pkl")
	assert os.path.exists(hyperparams) 
	sc.addFile(hyperparams)

	model = os.path.join(datafolder, f"models/{args.dataset}_model.pth")
	assert os.path.exists(model) 
	sc.addFile(model)

	datafile = os.path.join(datafolder, f'prediction_data/{args.dataset}.mat')
	assert os.path.exists(datafile)
	sc.addFile(datafile)

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

	'''
	▀█▀ █▀▀ █▀ ▀█▀   █▀▄ ▄▀█ ▀█▀ ▄▀█
	░█░ ██▄ ▄█ ░█░   █▄▀ █▀█ ░█░ █▀█
	'''
	positives_file = os.path.join(datafolder, "prediction_data", args.dataset+"_positives_test.txt") 
	positives = []
	with open(positives_file, 'r') as f:
		for line in f:
			pair = line.strip().split(" ")
			positives.append((int(pair[0]), int(pair[1])))
	negatives_file = os.path.join(datafolder, "prediction_data", args.dataset+"_negatives_test.txt") 
	negatives = []
	with open(negatives_file, 'r') as f:
		for line in f:
			pair = line.strip().split(" ")
			negatives.append((int(pair[0]), int(pair[1])))

	prediction_data = positives + negatives
	
	logger.info("Data sampled...")

	'''
	█▀ █░█ █▄▄ █▀▀ █▀█ ▄▀█ █▀█ █░█   █▀▀ ▀▄▀ ▀█▀ █▀█ ▄▀█ █▀▀ ▀█▀ █ █▀█ █▄░█
	▄█ █▄█ █▄█ █▄█ █▀▄ █▀█ █▀▀ █▀█   ██▄ █░█ ░█░ █▀▄ █▀█ █▄▄ ░█░ █ █▄█ █░▀█
	'''
	subgraph_extraction_whole = 0
	subgraph_extraction_times = []
	if args.batch_inprior:
		# form batches (lists of pairs of len ~batch size)
		batched_prediction_data = []
		batch_data = []
		for i, pair in enumerate(prediction_data):
			batch_data.append(pair)
			if len(batch_data) == args.batch_size or i == (len(prediction_data)-1):
				batched_prediction_data.append(batch_data)
				batch_data = []
		# parallelize batches
		prediction_rdd = sc.parallelize(batched_prediction_data)
		logger.info("Data parallelized...")
		# extract subgraphs:
		prediction_subgraphs = prediction_rdd.map(lambda batch: batches2subgraphs(batch, args.hop, args.db_extraction, args.dataset))
		start = time.time()
		subgraphs_times = prediction_subgraphs.collect()
		end = time.time()
		# extract subgraphs and times to two different lists
		subgraphs, times = map(list, zip(*subgraphs_times))
		subgraph_extraction_whole = end-start
		subgraph_extraction_times = times 
	else:
		# parallelize all pairs
		prediction_data_rdd = sc.parallelize(prediction_data)
		# extract graphs for all pairs
		prediction_subgraphs_pairs = prediction_data_rdd.map(lambda pair: link2subgraph(pair, args.hop, args.db_extraction, args.dataset))
		# --> will contain pairs and corresponding subgraphs
		start = time.time()
		pairs_subgraphs_times = prediction_subgraphs_pairs.collect()
		end = time.time()
		pairs, subgraphs, times = map(list, zip(*pairs_subgraphs_times))
		subgraph_extraction_whole = end-start
		subgraph_extraction_times = times
		# split into batches (partitions)
		# form batches (lists of pairs of len ~batch size)
		batched_prediction_data = []
		batch_poses = [[], []]
		graphs = []
		for i, pair in enumerate(pairs):
			batch_poses[0].append(pair[0])
			batch_poses[1].append(pair[1])
			graphs.append(subgraphs[i])
			if len(graphs) == args.batch_size or i == (len(prediction_data)-1):
				batch_data = pkl.dumps((graphs, batch_poses))
				batched_prediction_data.append(batch_data)
				graphs = []
				batch_poses = [[], []]
		subgraphs = batched_prediction_data

	logger.info("Subgraph extraction took " + str(subgraph_extraction_whole) + " seconds.")


	'''
	█▀█ █▀█ █▀▀ █▀▄ █ █▀▀ ▀█▀ █ █▀█ █▄░█
	█▀▀ █▀▄ ██▄ █▄▀ █ █▄▄ ░█░ █ █▄█ █░▀█
	'''
	subgraphs_prediction = sc.parallelize(subgraphs)

	# perform prediction:
	predictions = subgraphs_prediction.map(lambda graph: apply_network(args.dataset, graph))

	# extract results with .collect() method:
	start = time.time()
	results = predictions.collect()
	end = time.time()
	logger.info("Prediction on subgraphs took " + str(end-start) + " seconds.")

	for i, record in enumerate(results):
		np.savetxt("/opt/spark/work-dir/results_batch_"+str(i), record, fmt=['%d', '%d', '%1.2f'])

	logger.info("Results calculation complete!")

	time.sleep(60*10)

'''
██╗░░░██╗████████╗██╗██╗░░░░░░██████╗
██║░░░██║╚══██╔══╝██║██║░░░░░██╔════╝
██║░░░██║░░░██║░░░██║██║░░░░░╚█████╗░
██║░░░██║░░░██║░░░██║██║░░░░░░╚═══██╗
╚██████╔╝░░░██║░░░██║███████╗██████╔╝
░╚═════╝░░░░╚═╝░░░╚═╝╚══════╝╚═════╝░
'''

'''
	container for application arguments
'''
class application_args:
	
	dataset: str = "USAir"
	db_extraction: bool = True
	batch_inprior: bool = True
	hop: int = 2
	batch_size: int = 50

	def set_attr(self, attr, value: str):
		assert hasattr(self, attr)
		typ = type(getattr(self, attr))
		if typ is str:
			setattr(self, attr, value)
		elif typ is bool:
			setattr(self, attr, value == 'True')
		elif typ is int:
			setattr(self, attr, int(value))

	def print_attributes(self) -> str:
		msg = ""
		msg += f"dataset: {self.dataset}\n"
		msg += f"db_extraction: {str(self.db_extraction)}\n"
		msg += f"batch_inprior: {str(self.batch_inprior)}\n"
		msg += f"hop: {str(self.hop)}\n"
		msg += f"batch_size: {str(self.batch_size)}\n"
		return msg

'''
	parses given list of string arguments to applicatoin_args instance
'''
def parse_args(args: list) -> application_args:
	app_args = application_args()
	logger = getlogger('Node '+str(os.getpid()))
	try:
		# params come in pairs
		assert len(args)%2 == 0
		for i, arg in enumerate(args):
			# param names comes prior to param value
			if i%2 == 0:
				assert arg.startswith('--') and len(arg) > 2 and not args[i+1].startswith('--')
				argname = arg[2:] # arg without --
				app_args.set_attr(argname, args[i+1])
	except:
		print_usage()
		sys.exit(0)

	return app_args


def print_usage():
	logger = getlogger('Node '+str(os.getpid()))
	msg = ""
	msg += "Ugin App allowed parameters:\n"
	msg += "--dataset choose between datasets given in paper, defaults to USAir\n"
	msg += "--db_extraction choose whether to use db-based extraction or original one, defaults to true\n"
	msg += "--batch_inprior choose whether to batch data prior to subgraph calcultation, defaults to true\n"
	msg += "--hop choose hop number, defaults to 2\n"
	msg += "--batch_size choose batch size of data, defaults to 50\n"
	logger.info(msg)

if __name__ == "__main__":
	args = sys.argv
	# exclude app name
	args.pop(0)
	# adapt arguments
	args = parse_args(args)
	# execute 
	main(args)


	