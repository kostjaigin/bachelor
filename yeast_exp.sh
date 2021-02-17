export SPARK_HOME='/data/konstantin.igin/bachelor';
for hop in 1 2; do
	#$SPARK_HOME/load_db.sh $dataset;
		# for db in "True" "False"; do
		for executors in 4 8 12; do
			for links in 10 100 500 1000; do
				echo "|EXPERIMENT SETUP| " "hop: " $hop "|dataset: " "yeast" "|# links: " $links "|Batch in prior? " $batch_inprior "|# of execs: " $executors " |";
				$SPARK_HOME/exe.sh --number_of_executors $executors --hop $hop --links $links --db_extraction "False" --dataset yeast;
			done
		done
		# done
done

