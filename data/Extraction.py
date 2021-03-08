'''
░█████╗░██████╗░██████╗░
██╔══██╗██╔══██╗██╔══██╗
███████║██████╔╝██████╔╝
██╔══██║██╔═══╝░██╔═══╝░
██║░░██║██║░░░░░██║░░░░░
╚═╝░░╚═╝╚═╝░░░░░╚═╝░░░░░ 
'''
import sys
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark import SparkFiles # access submited files
from py2neo import Graph

datafolder = "/opt/spark/data"

sys.path.append(datafolder)
# import pytorch_DGCNN from data folder of spark distribution
from pytorch_DGCNN.predictor import *
from pytorch_DGCNN.util import GNNGraph
from pytorch_DGCNN.Logger import getlogger
from utils_app import application_args, parse_args, print_usage
from utils_app import save_subgraphs_times_batches, save_subgraphs_times, save_prediction_results
from utils_app import get_prediction_data, save_extraction_time, save_extracted_subgraph
from utils_extraction import *

import pickle as pkl
import numpy as np
import time
import scipy.io as sio

import gc

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

	logger.info("Build paths attached...")

	'''
	▀█▀ █▀▀ █▀ ▀█▀   █▀▄ ▄▀█ ▀█▀ ▄▀█
	░█░ ██▄ ▄█ ░█░   █▄▀ █▀█ ░█░ █▀█
	'''
	positives, negatives = get_prediction_data(args)
	prediction_data = positives + negatives
	datafile = os.path.join(datafolder, f'prediction_data/{args.dataset}.mat')
	assert os.path.exists(datafile)
	A = sio.loadmat(datafile)['net']
	logger.info("Data sampled...")

	'''
	█▀ █░█ █▄▄ █▀▀ █▀█ ▄▀█ █▀█ █░█   █▀▀ ▀▄▀ ▀█▀ █▀█ ▄▀█ █▀▀ ▀█▀ █ █▀█ █▄░█
	▄█ █▄█ █▄█ █▄█ █▀▄ █▀█ █▀▀ █▀█   ██▄ █░█ ░█░ █▀▄ █▀█ █▄▄ ░█░ █ █▄█ █░▀█
	'''
	subgraphs = []
	pairs = []
	times = []
	whole_extraction_time = 0

	# parallelize all pairs
	prediction_data_rdd = sc.parallelize(prediction_data)
	# extract graphs for all pairs
	prediction_subgraphs_pairs = prediction_data_rdd.map(lambda pair: link2subgraph(pair, args.hop, A))
	start = time.time()
	# --> will contain pairs and corresponding subgraphs
	prediction_subgraphs_pairs.foreach(lambda elements: save_extracted_subgraph(elements, args))
	end = time.time()
	whole_extraction_time = end-start
	# save time
	save_extraction_time(whole_extraction_time, args)
	logger.info("Extraction completed, results saved...")

if __name__ == "__main__":
	args = sys.argv
	# exclude app name
	args.pop(0)
	# adapt arguments
	args = parse_args(args)
	# execute 
	main(args)