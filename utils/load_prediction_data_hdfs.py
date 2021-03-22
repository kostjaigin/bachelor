import os, sys
from pywebhdfs.webhdfs import PyWebHdfsClient
from tqdm import tqdm

'''
	Saves experiment results on host. Later extraction performed using scp.
'''

def main():
	hdfs = PyWebHdfsClient(host='130.149.249.25', port='50070')
	target_path = 'checkpoints/linkprediction/data/'
	source_folder = os.path.join(os.getcwd(), '..', 'data/prediction_data')
        print("Source folder location: " + source_folder)
        
        datasets = ["USAir", "PB", "yeast"]
        links = [10, 100, 500, 1000]

        for dataset in tqdm(datasets):
            for link in links:
                filename = dataset + "_" + str(link) + ".txt"
                source_file = os.path.join(source_folder, filename)
                hdfs_path = os.path.join(target_path, filename)
                with open(source_file, 'r') as data:
                     hdfs.create_file(hdfs_path, data, overwrite=True)

        datasets = ["arxiv", "facebook"]
        links = [10, 100, 500, 1000, 5000, 10000, 25000, 50000]

        for dataset in tqdm(datasets):
            for link in links:
                filename = dataset + "_" + str(link) + ".txt"
                source_file = os.path.join(source_folder, filename)
                hdfs_path = os.path.join(target_path, filename)
                with open(source_file, 'r') as data:
                     hdfs.create_file(hdfs_path, data, overwrite=True)

        print("completed...")
        print("HDFS Directory contents:")
        contents = hdfs.list_dir(target_path)['FileStatuses']['FileStatus']
        print(contents)


if __name__ == "__main__":
	main()
