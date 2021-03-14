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
from utils_app import save_prediction_results, save_prediction_time
from utils_extraction import *

import pickle as pkl
from io import BytesIO
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

def transform_to_pair_subgraph(p, subgraph):
	file_base = os.path.basename(p).split('_')
	src = int(file_base[1])
	dst = int(file_base[2])
	graph = pkl.load(BytesIO(subgraph))
	return ((src, dst), graph)


def transform_to_list(l):
	pairs, subgraphs = map(list, zip(*l))
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
	spark.catalog.clearCache()

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

	# get as many graphs as number of links
	links = args.links # for 5k, 10k, 25k etc
	args.links = 50000 # to take files from corresponding 50k folder
	number_of_files = args.get_number_of_files()
	fraction = links/number_of_files # how many percent to sample
	partitions = math.ceil(float(number_of_files)*fraction/float(args.batch_size)) # how many partitions we need
	folderpath = args.get_hdfs_folder_path()

	prediction_data = sc.binaryFiles(folderpath) \
						.sample(fraction) \
						.reduceByKey(lambda p,subgraph: transform_to_pair_subgraph(p, subgraph)) \
						.repartition(partitions) \
						.glom() \
						.map(lambda p: transform_to_list(p)) \
						.cache() # cache to avoid recalculation
	
	data = prediction_data.count() # perform an action to trigger calculation
	logger.info("Data calculated and cached...")

	'''
	█▀█ █▀█ █▀▀ █▀▄ █ █▀▀ ▀█▀ █ █▀█ █▄░█
	█▀▀ █▀▄ ██▄ █▄▀ █ █▄▄ ░█░ █ █▄█ █░▀█
	'''
	predictions = prediction_data.map(lambda graph: apply_network(args.dataset, graph))
	start = time.time()
	results = predictions.count()
	end = time.time()
	logger.info("Prediction complete, saving prediction time...")
	save_prediction_time(end-start, args)

if __name__ == "__main__":
	args = sys.argv
	# exclude app name
	args.pop(0)
	# adapt arguments
	args = parse_args(args)
	# execute 
	main(args)