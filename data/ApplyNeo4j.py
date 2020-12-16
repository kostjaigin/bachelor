import sys

'''
	Test script: 
	1) Can we contact neo4j instances from another kubernetes located application?
	2) Can we connect to it through a service? Central service instance that acts like a load balancer?
'''

from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark import SparkFiles # access submited files

from py2neo import Graph
service_ip = "bolt://neo4j-helm-neo4j:7687"

datafolder = "/opt/spark/data"

sys.path.append(datafolder)
# import pytorch_DGCNN from data folder of spark distribution
from pytorch_DGCNN.predictor import *
from pytorch_DGCNN.Logger import getlogger

import time

def do_something(data):
	# data contains a node id
	# connect to graph service
	graph = Graph(service_ip)
	# request neighbours list for given id
	query = """
		match (n:Node {id: toInteger(%d)} )
		match (d:Node)
		where (d)-[:CONNECTION]->(n) or (n)-[:CONNECTION]->(d)
		return d
	""" % data
	results = list(graph.run(query))
	# get all of records, put them together in a string
	results = list(map(lambda record: str(record['d']['id']), results))
	s = ','
	result = str(data) + ": " + s.join(results)

	return result

def get_enclosing_subgraph(nodes, h: int = 1):
	a = nodes[0]
	b = nodes[1]
	# and nothing shows existing connections... only the nodes themselves
	# working only for h=1
	query_1 = """
		match (n:Node {id: %d})
		match (d:Node {id: %d})
		match (c:Node)
		where (n)-[:CONNECTION]->(c) or (c)-[:CONNECTION]->(n) or (d)-[:CONNECTION]->(c) or (c)-[:CONNECTION]->(d)
		return n, d, c
	""" % (a, b)
	# working for any h, but only in two queries for two requests
	query_2 = """
		match (n:Node {id: toInteger(%d)})
		with n
		call apoc.path.subgraphNodes(n, {relationshipFilter:'CONNECTION>',maxLevel:1}) yield node
		return node
	""" % a

def main():
	# create Spark context with Spark configuration
	spark = SparkSession\
			.builder\
			.appName("UginDGCNN")\
			.getOrCreate()
	sc = spark.sparkContext

	logger = getlogger('Node '+str(os.getpid()))
	logger.info("Spark Context established, going though app logic...")

	g_rdd = sc.parallelize(list(range(0, 332)))

	g2_rdd = g_rdd.map(lambda item: do_something(item))

	# write to console:
	for r in g2_rdd.collect():
		logger.info(r)
	# give some time to take a look at the results:
	time.sleep(5*60)

if __name__ == "__main__":
	main()


	