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

datafolder = "/opt/spark/data"

sys.path.append(datafolder)
# import pytorch_DGCNN from data folder of spark distribution
from pytorch_DGCNN.predictor import *
from pytorch_DGCNN.util import GNNGraph
from pytorch_DGCNN.Logger import getlogger
from utils_app import application_args, parse_args, print_usage
from utils_app import save_prediction_results
from utils_extraction import *

import pickle as pkl
import numpy as np
import time
import scipy.io as sio
import math

import gc

def apply_network(dataset:str, serialized):
	hyperparams_route = SparkFiles.get(f'{dataset}_hyper.pkl')
	model_route = SparkFiles.get(f'{dataset}_model.pth')
	predictor = Predictor(hyperparams_route, model_route)
	return predictor.predict(serialized)

def read_line(line) -> tuple:
	pair = line.strip().split(" ")
	src, dst = int(pair[0]), int(pair[1])
	return (src, dst)

def transform_to_list(l):
	pairs, subgraphs, _ = map(list, zip(*l))
	batch_poses = [[], []]
	for pair in pairs:
		batch_poses[0].append(pair[0])
		batch_poses[1].append(pair[1])
	return pkl.dumps((subgraphs, batch_poses))

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

	datafile = os.path.join(datafolder, f'prediction_data/{args.dataset}.mat')
	assert os.path.exists(datafile)
	A = sio.loadmat(datafile)['net'] # graph
	logger.info("Build paths attached...")

	'''
	█▀█ █▀█ █▀▀ █▀▄ █ █▀▀ ▀█▀ █ █▀█ █▄░█
	█▀▀ █▀▄ ██▄ █▄▀ █ █▄▄ ░█░ █ █▄█ █░▀█
	'''
	lines = args.links if args.links > 0 else sc.textFile(args.get_hdfs_data_path()).count()
	partitions = math.ceil(float(lines)/float(args.batch_size))
	prediction_data = sc.textFile(args.get_hdfs_data_path(), minPartitions=partitions) \
						.map(lambda line: read_line(line)) \
						.map(lambda pair: link2subgraph(pair, args.hop, A)) \
						.glom() \
						.map(lambda p: transform_to_list(p)) \
						.map(lambda graph: apply_network(args.dataset, graph))

	start = time.time()
	# trigger execution by calling an action
	results = prediction_data.collect()
	end = time.time()

	logger.info(f"Prediction completed in {str(end-start)} seconds...")

	logger.info("Saving results...")
	save_prediction_results(results, end-start, args)
	logger.info(f"Results saved under: {args.get_hdfs_folder_path}")

if __name__ == "__main__":
	args = sys.argv
	# exclude app name
	args.pop(0)
	# adapt arguments
	args = parse_args(args)
	# execute 
	main(args)


	