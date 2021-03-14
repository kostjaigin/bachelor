import os, sys

'''
	Takes different sets of links for given datasets in sizes of 10, 100, 500 and 1000 
'''

def main():
	datasets = ["facebook", "arxiv"]
	for dataset in datasets:
		positives = os.path.join(os.getcwd(), "data", f"{dataset}_positives_large.txt")
		negatives = os.path.join(os.getcwd(), "data", f"{dataset}_negatives_large.txt")
		for n in [5000, 10000, 25000, 50000]:
			# where to save
			target = os.path.join(os.getcwd(), f"../data/prediction_data/{dataset}_{str(n)}.txt")
			# what to save
			lines = []
			with open(positives, 'r') as f:
				for i, line in enumerate(f):
					if i == n/2:
						break
					lines.append(line)
			with open(negatives, 'r') as f:
				for i, line in enumerate(f):
					if i == n/2:
						break
					lines.append(line)
			with open(target, 'w') as f:
				f.writelines(lines)	


if __name__ == "__main__":
	main()