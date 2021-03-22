for hop in 2 1; do
        for dataset in "arxiv" "facebook"; do
                for executors in 12 8 4; do
                        for links in 25000 10000 5000; do
                                $SPARK_HOME/exe.sh --number_of_executors $executors --hop $hop --links $links --db_extraction "False" --dataset $dataset --hdfs_host "130.149.249.25";
                                # clean up
                                kubectl get pods --field-selector 'status.phase=Succeeded' -o name | xargs kubectl delete;
                        done
                done
        done
done
