import os, sys
from pywebhdfs.webhdfs import PyWebHdfsClient

def main(folder):
        hdfs = PyWebHdfsClient(host='130.149.249.25', port='50070')
        path = os.path.join('checkpoints/linkprediction/data', folder)
        hdfs.delete_file_dir(path, recursive = True)

if __name__ == "__main__":
    args = sys.argv
    # exclude app name
    args.pop(0)
    main(args[0])
