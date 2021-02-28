import os, sys
from pywebhdfs.webhdfs import PyWebHdfsClient
from tqdm import tqdm

'''
	Saves experiment results on host. Later extraction performed using scp.
'''

def main(folder):
	hdfs = PyWebHdfsClient(host='130.149.249.25', port='50070')
	path = 'checkpoints/linkprediction/data/'
	save_dir(path, folder, hdfs)

# path: hdfs path, folder: to what folder to save locally
def save_dir(path, folder, hdfs):
	if not os.path.exists(folder):
		os.mkdir(folder)
	contents = hdfs.list_dir(path)['FileStatuses']['FileStatus']
	for c in tqdm(contents):
		name = c['pathSuffix'].strip()
		if c['type'] == 'DIRECTORY':
			save_dir(os.path.join(path, name), os.path.join(folder, name), hdfs)
		else:
			save_file(os.path.join(path, name), os.path.join(folder, name), hdfs)

# path: hdfs path, filepath: to what folder+name to save locally
# todo probably wont work for bytes/won't be representable correctly
def save_file(path, filepath, hdfs):
    content = hdfs.read_file(path)	
    with open(filepath, 'w') as f:
		f.write(content)

if __name__ == "__main__":
	args = sys.argv
	# exclude script name
	args.pop(0)
	# take remaining argument as path
	results_folder = args[0]
	assert results_folder is not None
	main(results_folder)
