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
service_ip = "bolt://neo4j-helm-neo4j:7687"
sys.path.append(datafolder)
from pytorch_DGCNN.predictor import *
from pytorch_DGCNN.util import GNNGraph
from pytorch_DGCNN.Logger import getlogger
from pyspark import SparkFiles # access submited files
from graphframes import *

'''
	returns: pair tuple, GNNGraph subgraph, extraction time
'''
def link2subgraph(link, hop:int, A = None):
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

# works only with hopes of 1 and 2
def link2subgraph_frames(pair, graphframe, hop):
	src = str(pair[0])
	dst = str(pair[1])
	if hop == 1:
		motif = "(a)-[e]->(b)"
		fltr = f"a.id={src} or a.id={dst} or b.id={src} or b.id={dst}"
		subgraph_nodes = graphframe.find(motif).filter(fltr)
		subgraph_nodes = subgraph_nodes.select("a").union(subgraph_nodes.select("b")).distinct()
	else:
		subgraph_nodes = find_nodes_2hop(graphframe, src, dst)
	collected_nodes = subgraph_nodes.rdd.map(lambda r: r[0][0]).collect()
	src, dst = int(src), int(dst)
	collected_nodes.remove(src)
	collected_nodes.remove(dst)
	collected_nodes = [src, dst] + collected_nodes
	# given labeling function functions with indeces, not ids
	nodes_idx = list(range(0, len(collected_nodes)))
	# edges need to be constructed between known indeces
	nodes_map = dict(zip(collected_nodes, nodes_idx)) # to access nodes index w. id
	edgesframe = graphframe.edges
	edges = edgesframe \
			.filter(edgesframe.src.isin(collected_nodes) & edgesframe.dst.isin(collected_nodes)) \
			.rdd.map(lambda e: (nodes_map[e[0]], nodes_map[e[1]])) \
			.collect()
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
	g_label = 1
	# remove edge between target nodes
	if g.has_edge(0, 1):
		g.remove_edge(0, 1)
	gnn_graph = GNNGraph(g, g_label, labels)
	return (gnn_graph, (src, dst))

def find_nodes_2hop(graphframe, src, dst):
	fltr = f'a.id={src} or b.id={src} or c.id={src} or a.id={dst} or b.id={dst} or c.id={dst}'
	subgraph_nodes1 = find_subnodes_w_motif(graphframe, "(a)-[e1]->(b);(b)-[e2]->(c)", fltr)
	subgraph_nodes2 = find_subnodes_w_motif(graphframe, "(a)-[e1]->(b);(c)-[e2]->(b)", fltr)
	subgraph_nodes3 = find_subnodes_w_motif(graphframe, "(b)-[e1]->(a);(c)-[e2]->(b)", fltr)
	subgraph_nodes4 = find_subnodes_w_motif(graphframe, "(b)-[e1]->(a);(b)-[e2]->(c)", fltr)
	return subgraph_nodes1.union(subgraph_nodes2).union(subgraph_nodes3).union(subgraph_nodes4).distinct()

def find_subnodes_w_motif(graphframe, motif, fltr):
	subgraph_nodes = graphframe.find(motif).filter(fltr)
	subgraph_nodes = subgraph_nodes.select("a").union(subgraph_nodes.select("b")).union(subgraph_nodes.select("c")).distinct()
	return subgraph_nodes

def links2subgraphs_db(pairs, hop):
	graph = Graph(service_ip)
	pairs_b = [{
		"node1": pair[0],
		"node2": pair[1]
	} for pair in pairs]
	query = """
		UNWIND $pairs as p

		CALL apoc.cypher.run("
		MATCH (p1) WHERE p1.id = $pair.node1
		MATCH (p2) WHERE p2.id = $pair.node2

		MATCH (n:Node)
		WHERE n.id = p1.id or n.id = p2.id

		WITH n
		CALL apoc.path.subgraphNodes(n, {maxLevel:%d}) YIELD node
		WITH DISTINCT node
		WITH collect(node) as nds
		MATCH (src:Node)
		MATCH (dst:Node)
		WHERE src IN nds AND dst in nds
		MATCH (src)-[e:CONNECTION]->(dst)

		RETURN collect(e) AS edgs, nds
		"
		, { pair : p } )
		YIELD value
		RETURN p, value.edgs as edges, value.nds as nodes
	""" % hop
	results = list(graph.run(query, { "pairs": pairs_b }))
	subgraphs_pairs = []
	for i, record in enumerate(results):
		src, dst = int(results[i]['p']['node1']), int(results[i]['p']['node2'])
		nodes = [int(node['id']) for node in results[i]['nodes']]
		# put src and dst on top
		nodes.remove(src)
		nodes.remove(dst)
		nodes = [src, dst] + list(nodes)
		# given labeling function functions with indeces, not ids
		nodes_idx = list(range(0, len(nodes))) 
		# edges need to be constructed between known indeces
		nodes_map = dict(zip(nodes, nodes_idx)) # to access nodes index w. id
		edges = [(nodes_map[int(edge.nodes[0]['id'])], nodes_map[int(edge.nodes[1]['id'])]) for edge in results[i]['edges']]
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
		g_label = 1
		# remove edge between target nodes
		if g.has_edge(0, 1):
			g.remove_edge(0, 1)
		gnn_graph = GNNGraph(g, g_label, labels)
		subgraphs_pairs.append((gnn_graph, (src, dst)))
	return subgraphs_pairs

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