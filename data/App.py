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
from utils_app import get_prediction_data
from utils_extraction import *

import pickle as pkl
import numpy as np
import time
import scipy.io as sio

def apply_network(dataset:str, serialized):
	hyperparams_route = SparkFiles.get(f'{dataset}_hyper.pkl')
	model_route = SparkFiles.get(f'{dataset}_model.pth')
	predictor = Predictor(hyperparams_route, model_route)
	return predictor.predict(serialized)

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
	# parallelize all pairs
	prediction_data_rdd = sc.parallelize(prediction_data)
	# extract graphs for all pairs
	prediction_subgraphs_pairs = prediction_data_rdd.map(lambda pair: link2subgraph(pair, args.hop, args.db_extraction, A))
	start = time.time()
	# --> will contain pairs and corresponding subgraphs
	pairs_subgraphs_times = prediction_subgraphs_pairs.collect()
	end = time.time()
	whole_extraction_time = end-start
	pairs, subgraphs, times = map(list, zip(*pairs_subgraphs_times))
	# save extracted subgraphs and times
	save_subgraphs_times(pairs, subgraphs, times, args)
	# split into batches (partitions)
	# form batches (lists of pairs of len ~batch size)
	batched_prediction_data = []
	batch_poses = [[], []]
	graphs = []
	for i, pair in enumerate(pairs):
		batch_poses[0].append(pair[0])
		batch_poses[1].append(pair[1])
		graphs.append(subgraphs[i])
		if len(graphs) == args.batch_size or i == (len(prediction_data)-1):
			batch_data = pkl.dumps((graphs, batch_poses))
			batched_prediction_data.append(batch_data)
			graphs = []
			batch_poses = [[], []]
	subgraphs = batched_prediction_data


	'''
	█▀█ █▀█ █▀▀ █▀▄ █ █▀▀ ▀█▀ █ █▀█ █▄░█
	█▀▀ █▀▄ ██▄ █▄▀ █ █▄▄ ░█░ █ █▄█ █░▀█
	'''
	subgraphs_prediction = sc.parallelize(subgraphs)

	# perform prediction:
	predictions = subgraphs_prediction.map(lambda graph: apply_network(args.dataset, graph))

	# extract results with .collect() method:
	start = time.time()
	results = predictions.collect()
	end = time.time()
	logger.info("Prediction on subgraphs took " + str(end-start) + " seconds.")

	save_prediction_results(results, end-start, whole_extraction_time, args)	

	logger.info("Results calculation complete!")


if __name__ == "__main__":
	args = sys.argv
	# exclude app name
	args.pop(0)
	# adapt arguments
	args = parse_args(args)
	# execute 
	main(args)


	