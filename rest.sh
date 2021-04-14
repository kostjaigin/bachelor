echo "Switching to AB experiments...";

for hop in 1 2; do
	for dataset in "USAir" "yeast" "PB" "facebook" "arxiv"; do
		$SPARK_HOME/load_db.sh $dataset;
		for executors in 4 8 12; do
			for links in 10 100 500 1000; do
				echo "|EXPERIMENT SETUP| " "hop: " $hop "|dataset: " $dataset "|# links: " $links "|Batch in prior? " $batch_inprior "|# of execs: " $executors " |";
				$SPARK_HOME/exeDB.sh --number_of_executors $executors --hop $hop --links $links --db_extraction "True" --dataset $dataset;
				# clean up
				kubectl get pods --field-selector 'status.phase=Succeeded' -o name | xargs kubectl delete;
				# save existing results
				python /home/konstantin.igin/bachelor/utils/save_from_hdfs.py /home/konstantin.igin/results;
			done
		done
	done
done

echo "Switching to B experiments..."

for hop in 1 2; do
	for dataset in "USAir" "yeast" "PB" "facebook" "arxiv"; do
		$SPARK_HOME/load_db.sh $dataset;
		for executors in 4 8 12; do
			for links in 10 100 500 1000; do
				echo "|EXPERIMENT SETUP| " "hop: " $hop "|dataset: " $dataset "|# links: " $links "|Batch in prior? " $batch_inprior "|# of execs: " $executors " |";
				$SPARK_HOME/exeFrames.sh --number_of_executors $executors --number_of_db_cores 99 --hop $hop --links $links --db_extraction "True" --dataset $dataset;
				# clean up
				kubectl get pods --field-selector 'status.phase=Succeeded' -o name | xargs kubectl delete;
				# save existing results
				python /home/konstantin.igin/bachelor/utils/save_from_hdfs.py /home/konstantin.igin/results;
			done
		done
	done
done
