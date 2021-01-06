'''
    Transforms given .mat data files to neo4j compatible .csv files
    for further loading to db
'''

import networkx as nx
import os, sys
import scipy.io as sio
from py2neo.database.work import ClientError
from py2neo import Graph

# pick data set that should be transformed from .mat to .csv files for neo4j
DATASET = "USAir"
SEPARATOR = '\n' # what is used to separate lines in resulting files
ENCODING = 'utf-8' # bytes encoding
DATABASE_IMPORT_FOLDER = "/Users/konstantinigin/Library/Application Support/com.Neo4j.Relate/Data/dbmss/dbms-4285b34e-0d3e-4f85-8336-4d2bbe0e6bbc/import"
TEST = False # try to load prediction data to the local neo4j instance to check functionality

def main():

    if TEST:
        assert os.path.exists(DATABASE_IMPORT_FOLDER)

    data_dir = os.path.join(os.getcwd(), "data")
    nodes_file = f"{DATASET}_nodes.csv"
    edges_file = f"{DATASET}_edges.csv"
    nodes_path = os.path.join(DATABASE_IMPORT_FOLDER, nodes_file) if TEST else os.path.join(data_dir, nodes_file)
    edges_path = os.path.join(DATABASE_IMPORT_FOLDER, edges_file) if TEST else os.path.join(data_dir, edges_file)

    if not os.path.exists(nodes_path) or not os.path.exists(edges_path):

        data = os.path.join(data_dir, f"{DATASET}.mat")
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

    if TEST:
        graph = Graph("bolt://localhost:7687")
        query = '''match (n) detach delete (n)'''
        graph.run(query)
        try:
            query = '''create constraint nodeIdConstraint on (node: Node) assert node.id is unique'''
            graph.run(query)
        except ClientError:
            print("constraint existed before")

        query = '''
            LOAD CSV WITH HEADERS FROM 'file:///%s' AS row
            CREATE (n:Node {id: toInteger(row.id)})
        ''' % nodes_file
        graph.run(query)
        # assert graph has nodes... TODO

        # For larger data files, it is useful to use the hint USING PERIODIC COMMIT clause of LOAD CSV.
        # This hint tells Neo4j that the query might build up inordinate amounts of transaction state,
        # and so needs to be periodically committed.
        query = '''
            LOAD CSV WITH HEADERS FROM 'file:///%s' AS row
            MATCH (src:Node {id: toInteger(row.src_id)}),(dst:Node {id: toInteger(row.dst_id)})
            CREATE (src)-[:CONNECTION]->(dst)
        ''' % edges_file
        graph.run(query)





if __name__ == '__main__':
    main()

