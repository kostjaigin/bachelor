for links in 25000 50000; do
	$SPARK_HOME/exe.sh --number_of_executors 12 --hop 2 --links $links --db_extraction "False" --dataset "facebook" --data_path checkpoints/linkprediction/data/$dataset_$links.txt;
	# clean up
	kubectl get pods --field-selector 'status.phase=Succeeded' -o name | xargs kubectl delete;
done

for dataset in "arxiv"; do
	for executors in 4 8 12; do
		for links in 10 100 500 1000 5000 10000 25000 50000; do
			$SPARK_HOME/exe.sh --number_of_executors $executors --hop 2 --links $links --db_extraction "False" --dataset $dataset --data_path checkpoints/linkprediction/data/$dataset_$links.txt;
			# clean up
			kubectl get pods --field-selector 'status.phase=Succeeded' -o name | xargs kubectl delete;
		done
	done
done




