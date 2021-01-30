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
from utils_extraction import *

import pickle as pkl
import numpy as np
import time

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

	datafile = os.path.join(datafolder, f'prediction_data/{args.dataset}.mat')
	assert os.path.exists(datafile)
	sc.addFile(datafile)

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
	positives_file = os.path.join(datafolder, "prediction_data", args.dataset+"_positives_test.txt") 
	positives = []
	with open(positives_file, 'r') as f:
		for line in f:
			pair = line.strip().split(" ")
			positives.append((int(pair[0]), int(pair[1])))
	negatives_file = os.path.join(datafolder, "prediction_data", args.dataset+"_negatives_test.txt") 
	negatives = []
	with open(negatives_file, 'r') as f:
		for line in f:
			pair = line.strip().split(" ")
			negatives.append((int(pair[0]), int(pair[1])))

	prediction_data = positives + negatives
	
	logger.info("Data sampled...")

	'''
	█▀ █░█ █▄▄ █▀▀ █▀█ ▄▀█ █▀█ █░█   █▀▀ ▀▄▀ ▀█▀ █▀█ ▄▀█ █▀▀ ▀█▀ █ █▀█ █▄░█
	▄█ █▄█ █▄█ █▄█ █▀▄ █▀█ █▀▀ █▀█   ██▄ █░█ ░█░ █▀▄ █▀█ █▄▄ ░█░ █ █▄█ █░▀█
	'''
	subgraph_extraction_whole = 0
	subgraph_extraction_times = []
	if args.batch_inprior:
		# form batches (lists of pairs of len ~batch size)
		batched_prediction_data = []
		batch_data = []
		for i, pair in enumerate(prediction_data):
			batch_data.append(pair)
			if len(batch_data) == args.batch_size or i == (len(prediction_data)-1):
				batched_prediction_data.append(batch_data)
				batch_data = []
		# parallelize batches
		prediction_rdd = sc.parallelize(batched_prediction_data)
		logger.info("Data parallelized...")
		# extract subgraphs:
		prediction_subgraphs = prediction_rdd.map(lambda batch: batches2subgraphs(batch, args.hop, args.db_extraction, args.dataset))
		start = time.time()
		subgraphs_times = prediction_subgraphs.collect()
		end = time.time()
		# extract subgraphs and times to two different lists
		subgraphs, times = map(list, zip(*subgraphs_times))
		subgraph_extraction_whole = end-start
		subgraph_extraction_times = times 
		# save extracted subgraphs and times
		save_subgraphs_times_batches(subgraphs, times, args)
	else:
		# parallelize all pairs
		prediction_data_rdd = sc.parallelize(prediction_data)
		# extract graphs for all pairs
		prediction_subgraphs_pairs = prediction_data_rdd.map(lambda pair: link2subgraph(pair, args.hop, args.db_extraction, args.dataset))
		# --> will contain pairs and corresponding subgraphs
		start = time.time()
		pairs_subgraphs_times = prediction_subgraphs_pairs.collect()
		end = time.time()
		pairs, subgraphs, times = map(list, zip(*pairs_subgraphs_times))
		subgraph_extraction_whole = end-start
		subgraph_extraction_times = times
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

	logger.info("Subgraph extraction took " + str(subgraph_extraction_whole) + " seconds.")


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

	save_prediction_results(results, end-start, args)	

	logger.info("Results calculation complete!")

if __name__ == "__main__":
	args = sys.argv
	# exclude app name
	args.pop(0)
	# adapt arguments
	args = parse_args(args)
	# execute 
	main(args)


	