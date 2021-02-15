export SPARK_HOME = '/data/konstantin.igin/bachelor';
for hop in 1 2 3; do
	for dataset in "USAir" "PB" "facebook" "arxiv"; do
		for db in "True" "False"; do
			for executors in 4 8 12; do
				echo "|EXPERIMENT SETUP| " "hop: " $hop "|dataset: " $dataset "|DB extraction: " $db "|Batch in prior? " $batch_inprior "|# of execs: " $executors " |";
				$SPARK_HOME/exe.sh --number_of_executors $executors --db_extraction $db --dataset $dataset;
			done
		done
	done
done