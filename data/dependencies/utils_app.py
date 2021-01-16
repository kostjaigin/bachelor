'''
██╗░░░██╗████████╗██╗██╗░░░░░░██████╗
██║░░░██║╚══██╔══╝██║██║░░░░░██╔════╝
██║░░░██║░░░██║░░░██║██║░░░░░╚█████╗░
██║░░░██║░░░██║░░░██║██║░░░░░░╚═══██╗
╚██████╔╝░░░██║░░░██║███████╗██████╔╝
░╚═════╝░░░░╚═╝░░░╚═╝╚══════╝╚═════╝░
'''
import sys, os
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
	logger.info(msg)