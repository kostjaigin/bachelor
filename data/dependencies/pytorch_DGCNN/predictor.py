from pytorch_DGCNN.util import args
from pytorch_DGCNN.main import *
import pickle
import torch
import numpy as np
import os
from pytorch_DGCNN.Logger import getlogger

class Predictor():

	def __init__(self, hyperparameters_route:str, model_route:str):
		
		self.cmd_args = args()

		with open(hyperparameters_route, 'rb') as hyperparameters_name:
			saved_cmd_args = pickle.load(hyperparameters_name)
		for key, value in vars(saved_cmd_args).items():
			vars(self.cmd_args)[key] = value

		self.classifier = Classifier(cmd_args=self.cmd_args)

		if self.cmd_args.mode == 'gpu':
			self.classifier = classifier.cuda()
		self.classifier.load_state_dict(torch.load(model_route))


	'''
		serialized:
			- batch_data: a list of pickled GNNGraphs, size of settings-batch or less
			- data_pos: a list of two lists containing test_pos for given graph
			- file_route: where to store results
	'''
	def predict(self, serialized):

		batch_data, data_pos = pickle.loads(serialized)

		self.classifier.eval()
		predictions = []
		
		predictions.append(self.classifier(batch_data)[0][:, 1].exp().cpu().detach())

		predictions = torch.cat(predictions, 0).unsqueeze(1).numpy()

		left = np.array(data_pos[0])
		mid = np.array(data_pos[1]) # todo later: try without this cast
		right = predictions[:, 0]
		test_idx_and_pred = np.array([left, mid, right]).T
		
		result = '-'*10
		result += "RETURNING PREDICTION VALUES:"
		result += np.array2string(test_idx_and_pred)
		result += '-'*10
		logger = getlogger('Node '+str(os.getpid()))
		logger.info(result)

		# np.savetxt(file_route, test_idx_and_pred, fmt=['%d', '%d', '%1.2f'])

		# print('Predictions for are saved in {}'.format(file_route))


