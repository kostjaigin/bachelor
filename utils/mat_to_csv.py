'''
    Transforms given .mat data files to neo4j compatible .csv files
    for further loading to db
'''

import networkx as nx
import os, sys
import scipy.io as sio

# pick data set that should be transformed from .mat to .csv files for neo4j
DATASET = "USAir"
SEPARATOR = '\n' # what is used to separate lines in resulting files
ENCODING = 'utf-8' # bytes encoding

def main(args):

    DATASET = str(args[0]).strip()

    mat_data_dir = os.path.join(os.getcwd(), "../data/prediction_data/")
    data_dir = os.path.join(os.getcwd(), "data")
    nodes_file = f"{DATASET}_nodes.csv"
    edges_file = f"{DATASET}_edges.csv"
    nodes_path = os.path.join(data_dir, nodes_file)
    edges_path = os.path.join(data_dir, edges_file)

    if not os.path.exists(nodes_path) or not os.path.exists(edges_path):

        data = os.path.join(mat_data_dir, f"{DATASET}.mat")
        data = sio.loadmat(data)
        net = data['net']

        nx_graph = nx.from_scipy_sparse_matrix(net)
        nodes = list(nx_graph.nodes)

        with open(nodes_path, 'w') as f:
            f.write("id" + SEPARATOR)
            for node in nodes:
                f.write(str(node) + SEPARATOR)

        with open(edges_path, 'wb') as f:
            f.write(bytes("src_id,dst_id\n", ENCODING))
            nx.write_edgelist(nx_graph, f, delimiter=',', data=nodes, encoding=ENCODING)


if __name__ == '__main__':
    args = sys.argv
    # exclude app name
    args.pop(0)
    main(args)

