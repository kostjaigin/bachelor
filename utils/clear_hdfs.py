import os, sys
from pywebhdfs.webhdfs import PyWebHdfsClient

def main():
	hdfs = PyWebHdfsClient(host='130.149.249.25', port='50070')
	path = 'checkpoints/linkprediction/data/'
	hdfs.delete_file_dir(path, recursive = True)

if __name__ == "__main__":
	main()
