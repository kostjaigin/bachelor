'''
███████╗██╗░░░██╗███╗░░██╗░█████╗░░██████╗
██╔════╝██║░░░██║████╗░██║██╔══██╗██╔════╝
█████╗░░██║░░░██║██╔██╗██║██║░░╚═╝╚█████╗░
██╔══╝░░██║░░░██║██║╚████║██║░░██╗░╚═══██╗
██║░░░░░╚██████╔╝██║░╚███║╚█████╔╝██████╔╝
╚═╝░░░░░░╚═════╝░╚═╝░░╚══╝░╚════╝░╚═════╝░
'''

from py2neo import Graph
import scipy.sparse as ssp
import scipy.io as sio
import networkx as nx
import pickle as pkl
import numpy as np
import time
import sys
datafolder = "/opt/spark/data"
sys.path.append(datafolder)
from pytorch_DGCNN.predictor import *
from pytorch_DGCNN.util import GNNGraph
from pytorch_DGCNN.Logger import getlogger
from pyspark import SparkFiles # access submited files

service_ip = "bolt://neo4j-helm-neo4j:7687"

'''
	performs subgraph extraction for given batch
	returns pickled subgraphs together with list of positions in batch
		and list of execution times for each pair in a batch
'''
def batches2subgraphs(batch, hop:int, db:bool, A = None):
	if db: 
		# use db for graph extraction
		# connect service
		graph = Graph(service_ip)
	else:
		# if no db, dataset name required
		assert A is not None
	graphs = []
	batch_poses = [[], []]
	times = []
	for pair in batch:

		start = time.time()
		if db:
			gnn_graph = link2subgraph_db(graph, pair, hop)
		else:
			gnn_graph = link2subgraph_adj(pair, hop, A)
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
def link2subgraph(link, hop:int, db:bool, A = None):
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
		assert A is not None
		start = time.time()
		gnn_graph = link2subgraph_adj(link, hop, A)
		end = time.time()
		return (link, gnn_graph, (end-start))

def link2subgraph_adj(pair, h, A):
	ind = pair
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

'''
█▀ █▀▀ ▄▀█ █░░
▄█ ██▄ █▀█ █▄▄
'''
def neighbors(fringe, A):
	# find all 1-hop neighbors of nodes in fringe from A
	res = set()
	for node in fringe:
		nei, _, _ = ssp.find(A[:, node])
		nei = set(nei)
		res = res.union(nei)
	return res

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