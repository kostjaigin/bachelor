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

