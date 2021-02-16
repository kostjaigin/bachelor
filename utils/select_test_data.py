import os, sys

'''
	Takes different sets of links for given datasets in sizes of 10, 100, 500 and 1000 
'''

def main():
	datasets = ["USAir", "PB", "facebook", "arxiv"]
	for dataset in datasets:
		positives = os.path.join(os.getcwd(), "data", f"{dataset}_positives.txt")
		negatives = os.path.join(os.getcwd(), "data", f"{dataset}_negatives.txt")
		for n in [10, 100, 500, 1000]:
			# where to save
			target_positives = os.path.join(os.getcwd(), f"../data/prediction_data/{dataset}_positives_{str(n)}.txt")
			target_negatives = os.path.join(os.getcwd(), f"../data/prediction_data/{dataset}_negatives_{str(n)}.txt")
			# what to save
			lines = []
			with open(positives, 'r') as f:
				for i, line in enumerate(f):
					if i == n/2:
						break
					lines.append(line)
			with open(target_positives, 'w') as f:
				f.writelines(lines)
			lines = []
			with open(negatives, 'r') as f:
				for i, line in enumerate(f):
					if i == n/2:
						break
					lines.append(line)
			with open(target_negatives, 'w') as f:
				f.writelines(lines)	


if __name__ == "__main__":
	main()