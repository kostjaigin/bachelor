import sys

from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark import SparkFiles # access submited files

# from py2neo import Graph

import os
import pickle
import numpy as np
import time

datafolder = "/opt/spark/data"

sys.path.append(datafolder)
# import pytorch_DGCNN from data folder of spark distribution
from pytorch_DGCNN.predictor import *
from pytorch_DGCNN.Logger import getlogger

def apply_network(serialized):
	hyperparams_route = SparkFiles.get('USAir_hyper.pkl')
	model_route = SparkFiles.get('USAir_model.pth')
	predictor = Predictor(hyperparams_route, model_route)
	predictor.predict(serialized)

def main():
	# create Spark context with Spark configuration
	spark = SparkSession\
			.builder\
			.appName("UginDGCNN")\
			.getOrCreate()
	sc = spark.sparkContext

	logger = getlogger('Node '+str(os.getpid()))
	logger.info("Spark Context established, going though app logic...")

	# zipped package
	zipped_pkg = os.path.join(datafolder, "dependencies.zip")
	assert os.path.exists(zipped_pkg)
	sc.addPyFile(zipped_pkg)

	hyperparams = os.path.join(datafolder, "models/USAir_hyper.pkl")
	assert os.path.exists(hyperparams) 
	sc.addFile(hyperparams)

	model = os.path.join(datafolder, "models/USAir_model.pth")
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

	# read graphs from pickle file:
	graphs = []
	
	# data
	filename = os.path.join(datafolder, "prediction_data/pickled.pkl")
	with open(filename, 'rb') as f:
		graphs = pickle.load(f)
	graphs_rdd = sc.parallelize(graphs)

	graphs_rdd.foreach(lambda graph: apply_network(graph))


if __name__ == "__main__":
	main()


	