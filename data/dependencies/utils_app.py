'''
██╗░░░██╗████████╗██╗██╗░░░░░░██████╗
██║░░░██║╚══██╔══╝██║██║░░░░░██╔════╝
██║░░░██║░░░██║░░░██║██║░░░░░╚█████╗░
██║░░░██║░░░██║░░░██║██║░░░░░░╚═══██╗
╚██████╔╝░░░██║░░░██║███████╗██████╔╝
░╚═════╝░░░░╚═╝░░░╚═╝╚══════╝╚═════╝░
'''
import sys, os
import pickle as pkl
import numpy as np
import itertools
from pywebhdfs.webhdfs import PyWebHdfsClient
datafolder = "/opt/spark/data"
workdir = "/opt/spark/work-dir"
sys.path.append(datafolder)
from pytorch_DGCNN.Logger import getlogger

'''
	container for application arguments
'''
class application_args:
	
	dataset: str = "USAir"
	db_extraction: bool = True
	batch_inprior: bool = False
	hop: int = 2
	batch_size: int = 50
	links: int = 100 # how many links to take
	number_of_executors: int = 4 # only for results logging
	number_of_db_cores: int = 6 # only for results logging
	# location of persistent volume (without leading /)
	results_path: str = "checkpoints/linkprediction/data"
	hdfs_host: str = '130.149.249.25'
	hdfs_port: str = '50070'
	# spark batching parameters
	calculation_batch: int = 1000

	def set_attr(self, attr, value: str):
		assert hasattr(self, attr)
		typ = type(getattr(self, attr))
		if typ is str:
			setattr(self, attr, value)
		elif typ is bool:
			setattr(self, attr, value == 'True')
		elif typ is int:
			setattr(self, attr, int(value))

	# how many batches of batches to form prior to prediction
	def get_prediction_batch(self) -> int:
		return self.calculation_batch/self.batch_size	

	def print_attributes(self) -> str:
		msg = ""
		msg += f"dataset: {self.dataset}\n"
		msg += f"db_extraction: {str(self.db_extraction)}\n"
		# msg += f"batch_inprior: {str(self.batch_inprior)}\n"
		msg += f"hop: {str(self.hop)}\n"
		msg += f"links: {str(self.links)}\n"
		return msg

	def get_folder_results_path(self) -> str:
		foldername = self.get_folder_results_name()
		path = os.path.join(self.results_path, foldername)
		return path

	def get_folder_results_name(self) -> str:
		folder = self.dataset+"_"
		folder += "exec-"+str(self.number_of_executors)+"_"
		folder += "cores-"+str(self.number_of_db_cores)+"_"
		folder += "db-"+str(self.db_extraction)+"_"
		folder += "hop-"+str(self.hop)+"_"
		folder += "links-"+str(self.links)
		return folder

'''
	saves given subgraphs (pickled GNNGraphs and pairs lists) and extraction times
	locally (inside of pod/container) for further extraction. 
	@params:
		+ pickled_list: list of pickeld GNNGraphs and pairs of positions
		+ times_list: list of lists of times for each batch (len ~= data/args.batch_size)
		+ args: application arguments (for naming)
'''
def save_subgraphs_times_batches(pickled_list, times_list, args: application_args):
	graphs = [] 
	pairs = []  
	times = []
	for i, pickled in enumerate(pickled_list):
		# batch_data contains n=~50 graphs, data_pos - 2 lists of paired nodes
		batch_data, data_pos = pkl.loads(pickled)
		# batch_times contains a list of times for data_pos of size n
		batch_times = times_list[i]
		assert len(batch_times) == len(batch_data) == len(data_pos[0]) == len(data_pos[1])
		for j, graph in enumerate(batch_data):
			graphs.append(graph)
			pairs.append((data_pos[0][j], data_pos[1][j]))
			times.append(batch_times[j])
	save_subgraphs_times(pairs, graphs, times, args)

'''
	saves given subgraphs list with it's times and corresponding pairs
	locally (inside of pod/container) for further extraction.
	@params:
		+ pairs_list: list of extracted pairs
		+ times_list: list of times corresponding to the pairs
'''
def save_subgraphs_times(pairs_list, subgraphs_list, times_list, args: application_args):
	
	path = args.get_folder_results_path()
	times_path = os.path.join(path, "times")
	pairs_path = os.path.join(path, "pairs")
	hdfs = PyWebHdfsClient(host=args.hdfs_host, port=args.hdfs_port)

	assert len(pairs_list) == len(subgraphs_list)

	times = ''
	for t in times_list:
		times += str(t) + '\n'
	hdfs.create_file(times_path, times)
	pairs = ''
	for p in pairs_list:
		pairs += str(p) + '\n'
	hdfs.create_file(pairs_path, pairs)

def save_extraction_time(time, args: application_args):
	path = args.get_folder_results_path()
	hdfs = PyWebHdfsClient(host=args.hdfs_host, port=args.hdfs_port)
	file = os.path.join(path, "whole_extraction_time")
	hdfs.create_file(file, str(time))

def save_extracted_subgraph(elements, args: application_args):
	pair, subgraph, _ = elements
	path = args.get_folder_results_path()
	hdfs = PyWebHdfsClient(host=args.hdfs_host, port=args.hdfs_port)
	file = os.path.join(path, f"graph_{str(pair[0])}_{str(pair[1])}")
	pickled = pkl.dumps(subgraph)
	hdfs.create_file(file, pickled)

def save_prediction_results(results, time, whole_extraction_time, args: application_args):
	# get hdfs path
	path = args.get_folder_results_path()
	# save data on pod
	chained = list(itertools.chain.from_iterable(results))
	file = os.path.join(workdir, 'prediction_'+args.get_folder_results_name())
	np.savetxt(file, chained, fmt=['%d', '%d', '%1.2f'])
	# access it to read linewise
	predictions = ''
	with open(file, 'r') as f:
		for line in f:
			predictions += line.strip() + '\n'
	os.remove(file)
	# save results on hdfs
	hdfs = PyWebHdfsClient(host=args.hdfs_host, port=args.hdfs_port)
	file = os.path.join(path, "predictions")
	hdfs.create_file(file, predictions)

	file = os.path.join(path, "resulting_prediction_time")
	hdfs.create_file(file, str(whole_extraction_time))
	file = os.path.join(path, "resulting_extraction_time")
	hdfs.create_file(file, str(time))

def save_prediction_results_single(time, args: application_args):
	# get hdfs path
	single_args = args.copy()
	single_args.number_of_executors = 1
	# get hdfs path
	path = args.get_folder_results_path()
	# save results on hdfs
	hdfs = PyWebHdfsClient(host=args.hdfs_host, port=args.hdfs_port)
	file = os.path.join(path, "resulting_prediction_time")
	hdfs.create_file(file, str(time))

'''
	Get data used for prediction and extraction based on application params
'''
def get_prediction_data(args: application_args) -> list:
	
	positives_file = os.path.join(datafolder, "prediction_data", f"{args.dataset}_positives_{str(args.links)}.txt") 
	positives = []
	with open(positives_file, 'r') as f:
		for line in f:
			pair = line.strip().split(" ")
			positives.append((int(pair[0]), int(pair[1])))
	negatives_file = os.path.join(datafolder, "prediction_data", f"{args.dataset}_negatives_{str(args.links)}.txt") 
	negatives = []
	with open(negatives_file, 'r') as f:
		for line in f:
			pair = line.strip().split(" ")
			negatives.append((int(pair[0]), int(pair[1])))
	
	assert len(positives)+len(negatives) == args.links

	return positives, negatives

'''
	parses given list of string arguments to applicatoin_args instance
'''
def parse_args(args: list) -> application_args:
	app_args = application_args()
	logger = getlogger('Node '+str(os.getpid()))
	try:
		# params come in pairs
		assert len(args)%2 == 0
		for i, arg in enumerate(args):
			# param names comes prior to param value
			if i%2 == 0:
				assert arg.startswith('--') and len(arg) > 2 and not args[i+1].startswith('--')
				argname = arg[2:] # arg without --
				app_args.set_attr(argname, args[i+1])
	except:
		print_usage()
		sys.exit(0)

	return app_args


def print_usage():
	logger = getlogger('Node '+str(os.getpid()))
	msg = ""
	msg += "Ugin App allowed parameters:\n"
	msg += "--dataset choose between datasets given in paper, defaults to USAir\n"
	msg += "--db_extraction choose whether to use db-based extraction or original one, defaults to true\n"
	# msg += "--batch_inprior choose whether to batch data prior to subgraph calcultation, defaults to false\n"
	msg += "--hop choose hop number, defaults to 2\n"
	msg += "--batch_size choose batch size of data, defaults to 50\n"
	msg += "--results_path location of persistent volume (without leading /)\n"
	msg += "--links how many links to take for experiments\n"
	msg += "--number_of_executors only for logging and data storage, # of executors in Spark cluster\n"
	msg += "--hdfs_host host of storage web interface\n"
	msg += "--hdfs_port port of storage web interface\n"
	msg += "--calculation_batch batch size of Spark calculations, defaults to 1000\n"
	logger.info(msg)