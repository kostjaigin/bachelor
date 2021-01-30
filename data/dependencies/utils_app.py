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
datafolder = "/opt/spark/data"
sys.path.append(datafolder)
from pytorch_DGCNN.Logger import getlogger

'''
	container for application arguments
'''
class application_args:
	
	dataset: str = "USAir"
	db_extraction: bool = True
	batch_inprior: bool = True
	hop: int = 2
	batch_size: int = 50
	number_of_executors: int = 4 # only for results logging
	number_of_db_cores: int = 6 # only for results logging
	results_path: str = "/opt/spark/work-dir/my_volume"

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
		msg += f"batch_inprior: {str(self.batch_inprior)}\n"
		msg += f"hop: {str(self.hop)}\n"
		msg += f"batch_size: {str(self.batch_size)}\n"
		return msg

	def get_folder_results_path(self) -> str:
		foldername = self.get_folder_results_name()
		path = os.path.join(self.results_path, foldername)
		if not os.path.exists(path):
			os.mkdir(path)
		return path

	def get_folder_results_name(self) -> str:
		folder = self.dataset+"_"
		folder += "exec-"+str(self.number_of_executors)+"_"
		folder += "cores-"+str(self.number_of_db_cores)+"_"
		folder += "db-"+str(self.db_extraction)+"_"
		folder += "batch-"+str(self.batch_inprior)+"_"
		folder += "hop-"+str(self.hop)+"_"
		folder += "batchSize-"+str(self.batch_size)
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

	assert len(pairs_list) == len(subgraphs_list) == len(times_list)

	for i, graph in enumerate(subgraphs_list):
		graphs_path = os.path.join(path, "graph_"+str(i))
		with open(graphs_path, 'wb') as f:
			pickled = pkl.dumps(graph)
			f.write(pickled)
	with open(times_path, 'w') as f:
		for t in times_list:
			f.write(str(t))
			f.write('\n')
	with open(pairs_path, 'w') as f:
		for pair in pairs_list:
			f.write(str(pair))
			f.write('\n')

def save_prediction_results(results, time, args: application_args):
	
	path = args.get_folder_results_path()

	for i, record in enumerate(results):
		file = os.path.join(path, "results_batch_"+str(i))
		np.savetxt(file, record, fmt=['%d', '%d', '%1.2f'])

	file = os.path.join(path, "resulting_prediction_time")
	with open(file, 'w') as f:
		f.write(str(time))


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
	msg += "--batch_inprior choose whether to batch data prior to subgraph calcultation, defaults to true\n"
	msg += "--hop choose hop number, defaults to 2\n"
	msg += "--batch_size choose batch size of data, defaults to 50\n"
	msg += "--results_path defaults to /opt/spark/work-dir/calculation_results"
	logger.info(msg)