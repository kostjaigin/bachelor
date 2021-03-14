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
	links: int = -1 # how many links to take
	number_of_executors: int = 4 # only for results logging
	number_of_db_cores: int = 6 # only for results logging
	# location of persistent volume (without leading /)
	results_path: str = "checkpoints/linkprediction/data"
	data_path: str = "checkpoints/linkprediction/data/testdata.txt"
	hdfs_host: str = '130.149.249.25'
	hdfs_port: str = '50070'

	def set_attr(self, attr, value: str):
		assert hasattr(self, attr)
		typ = type(getattr(self, attr))
		if typ is str:
			setattr(self, attr, value)
		elif typ is bool:
			setattr(self, attr, value == 'True')
		elif typ is int:
			setattr(self, attr, int(value))

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
		dataname = os.path.basename(self.data_path)
		folder = self.dataset+"_"
		folder += "exec-"+str(self.number_of_executors)+"_"
		folder += "cores-"+str(self.number_of_db_cores)+"_"
		folder += "db-"+str(self.db_extraction)+"_"
		folder += "hop-"+str(self.hop)+"_"
		folder += dataname
		return folder

	def get_hdfs_data_path(self) -> str:
		return f"hdfs://{self.hdfs_host}:{self.hdfs_port}/{self.data_path}"

	def get_hdfs_folder_path(self) -> str:
		return f"hdfs://{self.hdfs_host}:{self.hdfs_port}/{self.get_folder_results_path}"

	def get_number_of_files(self) -> int:
		hdfs = PyWebHdfsClient(host=self.hdfs_host, port=self.hdfs_port)
		contents = hdfs.list_dir(self.results_path)['FileStatuses']['FileStatus']
		filtered = filter(lambda c: c['pathSuffix'] == self.get_folder_results_name, contents)
		return int(filtered[0]['childrenNum'])

def save_extraction_time(time, args: application_args):
	path = args.get_folder_results_path()
	hdfs = PyWebHdfsClient(host=args.hdfs_host, port=args.hdfs_port)
	file = os.path.join(path, "whole_extraction_time")
	hdfs.create_file(file, str(time), overwrite=True)

def save_prediction_time(time, args: application_args):
	path = args.get_folder_results_path()
	hdfs = PyWebHdfsClient(host=args.hdfs_host, port=args.hdfs_port)
	file = os.path.join(path, "whole_prediction_time")
	hdfs.create_file(file, str(time), overwrite=True)

def save_extracted_subgraph(elements, args: application_args):
	pair, subgraph, _ = elements
	path = args.get_folder_results_path()
	hdfs = PyWebHdfsClient(host=args.hdfs_host, port=args.hdfs_port)
	file = os.path.join(path, f"graph_{str(pair[0])}_{str(pair[1])}")
	pickled = pkl.dumps(subgraph)
	hdfs.create_file(file, pickled, overwrite=True)

def save_prediction_results(results, time, args: application_args):
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
	hdfs.create_file(file, str(time))

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
	msg += "--data_path test data location on hdfs without leading /\n"
	logger.info(msg)