import os, sys

'''
	Sample random 500 of positives for facebook and random 500 of negatives for facebook
'''

def main():
	dataset = "facebook"
	positives = os.path.join(os.getcwd(), "data", f"{dataset}_positives.txt")
	negatives = os.path.join(os.getcwd(), "data", f"{dataset}_negatives.txt")
	trial = 500 # how much to pass
	n = 1000
	# where to save
	target_positives = os.path.join(os.getcwd(), f"../data/prediction_data/{dataset}_positives_{str(n)}.txt")
	target_negatives = os.path.join(os.getcwd(), f"../data/prediction_data/{dataset}_negatives_{str(n)}.txt")
	# what to save
	lines = []
	with open(positives, 'r') as f:
		for i, line in enumerate(f):
			# take only values between trial and set border
			if i < trial:
				continue
			if i == trial+n/2:
				break
			lines.append(line)
	with open(target_positives, 'w') as f:
		f.writelines(lines)
	lines = []
	with open(negatives, 'r') as f:
		for i, line in enumerate(f):
			# take only values between trial and set border
			if i < trial:
				continue
			if i == trial+n/2:
				break
			lines.append(line)
	with open(target_negatives, 'w') as f:
		f.writelines(lines)	


if __name__ == "__main__":
	main()