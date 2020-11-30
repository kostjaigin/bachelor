from __future__ import print_function
import numpy as np
import random
import os
import networkx as nx
import pdb

class args:
    mode = 'cpu'
    gm = 'DGCNN'
    data = None
    batch_size:int = 50
    seed:int = 1
    feat_dim:int = 0
    edge_feat_dim:int = 0
    num_class:int = 0
    fold:int = 1
    test_number:int = 0
    num_epochs:int = 1000
    latent_dim:int = 64
    sortpooling_k:float = 30
    conv1d_activation:str = 'ReLU'
    out_dim:int = 1024
    hidden:int = 100
    max_lv:int = 4
    learning_rate:float = 0.0001
    dropout:bool = False
    printAUC:bool = False
    extract_features:bool = False

class GNNGraph(object):
    def __init__(self, g, label, node_tags=None, node_features=None):
        '''
            g: a networkx graph
            label: an integer graph label
            node_tags: a list of integer node tags
            node_features: a numpy array of continuous node features
        '''
        self.num_nodes = len(node_tags)
        self.node_tags = node_tags
        self.label = label
        self.node_features = node_features  # numpy array (node_num * feature_dim)
        self.degs = list(dict(g.degree).values())

        if len(g.edges()) != 0:
            x, y = zip(*g.edges())
            self.num_edges = len(x)        
            self.edge_pairs = np.ndarray(shape=(self.num_edges, 2), dtype=np.int32)
            self.edge_pairs[:, 0] = x
            self.edge_pairs[:, 1] = y
            self.edge_pairs = self.edge_pairs.flatten()
        else:
            self.num_edges = 0
            self.edge_pairs = np.array([])
        
        # see if there are edge features
        self.edge_features = None
        if nx.get_edge_attributes(g, 'features'):  
            # make sure edges have an attribute 'features' (1 * feature_dim numpy array)
            edge_features = nx.get_edge_attributes(g, 'features')
            assert(type(edge_features.values()[0]) == np.ndarray) 
            # need to rearrange edge_features using the e2n edge order
            edge_features = {(min(x, y), max(x, y)): z for (x, y), z in edge_features.items()}
            keys = sorted(edge_features)
            self.edge_features = []
            for edge in keys:
                self.edge_features.append(edge_features[edge])
                self.edge_features.append(edge_features[edge])  # add reversed edges
            self.edge_features = np.concatenate(self.edge_features, 0)


