import sys
import os
import torch
import random
import numpy as np
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math

from pytorch_DGCNN.DGCNN_embedding import DGCNN
from pytorch_DGCNN.mlp_dropout import MLPClassifier, MLPRegression
from pytorch_DGCNN.util import GNNGraph, args

import networkx as nx

class Classifier(nn.Module):
    def __init__(self, cmd_args:args, regression=False):
        super(Classifier, self).__init__()
        self.regression = regression
        self.cmd_args = cmd_args
        if cmd_args.gm == 'DGCNN':
            model = DGCNN
        else:
            print('unknown gm %s' % cmd_args.gm)
            sys.exit()

        if cmd_args.gm == 'DGCNN':
            self.gnn = model(latent_dim=cmd_args.latent_dim,
                            output_dim=cmd_args.out_dim,
                            num_node_feats=cmd_args.feat_dim+cmd_args.attr_dim,
                            num_edge_feats=cmd_args.edge_feat_dim,
                            k=cmd_args.sortpooling_k, 
                            conv1d_activation=cmd_args.conv1d_activation)
        out_dim = cmd_args.out_dim
        if out_dim == 0:
            if cmd_args.gm == 'DGCNN':
                out_dim = self.gnn.dense_dim
            else:
                out_dim = cmd_args.latent_dim
        self.mlp = MLPClassifier(input_size=out_dim, hidden_size=cmd_args.hidden, num_class=cmd_args.num_class, with_dropout=cmd_args.dropout)
        if regression:
            self.mlp = MLPRegression(input_size=out_dim, hidden_size=cmd_args.hidden, with_dropout=cmd_args.dropout)

    def PrepareFeatureLabel(self, batch_graph):
        if self.regression:
            labels = torch.FloatTensor(len(batch_graph))
        else:
            labels = torch.LongTensor(len(batch_graph))
        n_nodes = 0

        if batch_graph[0].node_tags is not None:
            node_tag_flag = True
            concat_tag = []
        else:
            node_tag_flag = False

        if batch_graph[0].node_features is not None:
            node_feat_flag = True
            concat_feat = []
        else:
            node_feat_flag = False

        if self.cmd_args.edge_feat_dim > 0:
            edge_feat_flag = True
            concat_edge_feat = []
        else:
            edge_feat_flag = False

        for i in range(len(batch_graph)):
            labels[i] = batch_graph[i].label
            n_nodes += batch_graph[i].num_nodes
            if node_tag_flag == True:
                concat_tag += batch_graph[i].node_tags
            if node_feat_flag == True:
                tmp = torch.from_numpy(batch_graph[i].node_features).type('torch.FloatTensor')
                concat_feat.append(tmp)
            if edge_feat_flag == True:
                if batch_graph[i].edge_features is not None:  # in case no edge in graph[i]
                    tmp = torch.from_numpy(batch_graph[i].edge_features).type('torch.FloatTensor')
                    concat_edge_feat.append(tmp)

        if node_tag_flag == True:
            concat_tag = torch.LongTensor(concat_tag).view(-1, 1)
            node_tag = torch.zeros(n_nodes, self.cmd_args.feat_dim)
            node_tag.scatter_(1, concat_tag, 1)

        if node_feat_flag == True:
            node_feat = torch.cat(concat_feat, 0)

        if node_feat_flag and node_tag_flag:
            # concatenate one-hot embedding of node tags (node labels) with continuous node features
            node_feat = torch.cat([node_tag.type_as(node_feat), node_feat], 1)
        elif node_feat_flag == False and node_tag_flag == True:
            node_feat = node_tag
        elif node_feat_flag == True and node_tag_flag == False:
            pass
        else:
            node_feat = torch.ones(n_nodes, 1)  # use all-one vector as node features
        
        if edge_feat_flag == True:
            edge_feat = torch.cat(concat_edge_feat, 0)

        if self.cmd_args.mode == 'gpu':
            node_feat = node_feat.cuda()
            labels = labels.cuda()
            if edge_feat_flag == True:
                edge_feat = edge_feat.cuda()

        if edge_feat_flag == True:
            return node_feat, edge_feat, labels
        return node_feat, labels

    def forward(self, batch_graph):
        feature_label = self.PrepareFeatureLabel(batch_graph)
        if len(feature_label) == 2:
            node_feat, labels = feature_label
            edge_feat = None
        elif len(feature_label) == 3:
            node_feat, edge_feat, labels = feature_label
        embed = self.gnn(batch_graph, node_feat, edge_feat)
        return self.mlp(embed, labels)