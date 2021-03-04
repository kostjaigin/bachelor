for hop in 1 2; do
	for dataset in "USAir" "yeast" "PB"; do
		#$SPARK_HOME/load_db.sh $dataset;
		#for db in "True" "False"; do
		for executors in 4 8 12; do
			for links in 10 100 500 1000; do
				echo "|EXPERIMENT SETUP| " "hop: " $hop "|dataset: " $dataset "|# links: " $links "|Batch in prior? " $batch_inprior "|# of execs: " $executors " |";
				$SPARK_HOME/exe.sh --number_of_executors $executors --hop $hop --links $links --db_extraction "False" --dataset $dataset --hdfs_host "130.149.249.25";
				# clean up
				kubectl get pods --field-selector 'status.phase=Succeeded' -o name | xargs kubectl delete;
			done
		#done
		done
	done
done

for hop in 1 2; do
	for dataset in "facebook" "arxiv"; do
		for executors in 4 8 12; do
			for links in 10 100 500; do
				$SPARK_HOME/exe.sh --number_of_executors $executors --hop $hop --links $links --db_extraction "False" --dataset $dataset --hdfs_host "130.149.249.25";
				# clean up
                                kubectl get pods --field-selector 'status.phase=Succeeded' -o name | xargs kubectl delete;
			done
		done
	done
done

# 1000 links hop 1
for dataset in "facebook" "arxiv"; do
	for executors in 12 8 4; do
		$SPARK_HOME/exe.sh --number_of_executors $executors --hop 1 --links 1000 --db_extraction "False" --dataset $dataset --hdfs_host "130.149.249.25";
        	# clean up
        	kubectl get pods --field-selector 'status.phase=Succeeded' -o name | xargs kubectl delete;
	done
done
