'''
    Extracts test data for given dataset and saves it as node pairs in a .csv file
'''

import networkx as nx
import os, sys
import scipy.io as sio
import scipy.sparse as ssp
import random
import math

# pick data set that should be transformed from .mat to .csv files for neo4j
DATASET = "USAir"
SEPARATOR = '\n' # what is used to separate lines in resulting files
ENCODING = 'utf-8' # bytes encoding
test_ratio = 0.1

def main():

    data_dir = os.path.join(os.getcwd(), "data")
    data = os.path.join(data_dir, f"{DATASET}.mat")
    data = sio.loadmat(data)
    net = data['net']

    # get upper triangular matrix
    net_triu = ssp.triu(net, k=1)
    # sample positive links for train/test
    row, col, _ = ssp.find(net_triu)
    # sample positive links if not specified
    perm = random.sample(range(len(row)), len(row)) # mix idxs of positive samples
    row, col = row[perm], col[perm] # mix positions of rows and cols

    # TODO do i need to do this?
    split = int(math.ceil(len(row) * (1 - test_ratio)))
    train_pos = (row[:split], col[:split])
    test_pos = (row[split:], col[split:])

    print('hello world')

if __name__ == '__main__':
    main()

