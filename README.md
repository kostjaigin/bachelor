# Using Graph Neural Networks for Distributed Link Prediction

My bachelor thesis work. Hashtags: **Apache Spark, Python, Kubernetes for deployment, PyTorch for Neural Networks**.

**Here we explain:**
- What is the thesis about *in short*
- How to reproduce my experiments
- The structure of this project

## Thesis in short

[Thesis not in short 👀](https://github.com/kostjaigin/bachelor/blob/master/thesis.pdf)

We perform link prediction using Muhan Zhang's [SEAL](https://github.com/muhanzhang/SEAL) system on a Spark cluster, deployed on a [K8S](https://kubernetes.io) cluster on university machines. The idea of the thesis is to distribute the original approach, using Apache Spark. We propose three main distribution strategies:

1. **AA**: Build an RDD on test-links, distribute the original methods.
2. **AB**: Build an RDD on test-links, perform the subgraph extraction part of Zhang's method using [Neo4j GraphDB](https://github.com/neo4j-contrib/neo4j-helm), deployed on the same cluster.
3. **B**: Use [GraphFrames](https://github.com/graphframes/graphframes) to distribute the underlying graph structure, perform predicion on test-sets in a loop.

Address the bachelor thesis for the details. 

## How to reproduce my experiments

### Environment setup
Having a kubernetes cluster up and running, connect to the master machine and do the following..

#### Neo4j Cluster Initialization

1. Make sure your cluster doesn't run Neo4j yet: ```helm uninstall neo4j-helm``` *(it is okay if it outputs an error)*
2. Clone this repository to your working directory: ```git clone https://github.com/kostjaigin/bachelor.git```
3. Clone Neo4j Helm repository to your working directory: ```git clone https://github.com/neo4j-contrib/neo4j-helm.git```
4. Use my copy of the Neo4j Helm Configuration file, *here a link* (**TODO**)
5. Install/start the Neo4j cluster: ```helm install neo4j-helm -f ./neo4j-helm/values.yaml neo4j-helm```
6. Now we need to wait until pods are set up and are running. We can check the current state with ```kubectl get pods``` command. It can take a while.

Default configuration includes 3 cores and 3 read replicas, you can change this setting inside of the values.yaml file. *Read more about Neo4j replication [here](https://neo4j.com/docs/operations-manual/current/clustering/)*. 
When the pods are setup, the ```kubectl get pods``` command returns following prompt indicating that each pod is ready:
```bash
NAME                         READY   STATUS                      
neo4j-helm-neo4j-core-0      1/1     Running                        
neo4j-helm-neo4j-core-1      1/1     Running                        
neo4j-helm-neo4j-core-2      1/1     Running                        
neo4j-helm-neo4j-replica-0   1/1     Running   
neo4j-helm-neo4j-replica-1   1/1     Running                        
neo4j-helm-neo4j-replica-2   1/1     Running                        
```
7. We need to allow data loading from external resources. In bash execute: ```kubectl exec --stdin --tty neo4j-helm-neo4j-core-0 -- /bin/bash``` to access the main cores container system
8. Inside of the container go to /var/lib/neo4j/conf/neo4j.conf: ```cd /var/lib/neo4j/conf/``` and access **neo4j.conf** with text editor of your choice. I prefer vim: ```vi neo4j.conf```.
9. Add (if not present) line ```dbms.security.allow_csv_import_from_file_urls=true```

### Build and execute the project
Set enviromental variable *$SPARK_HOME* to the directory of this repository

*If you want to build a custom image*, do ```$SPARK_HOME/bin/docker-image-tool.sh -r kostjaigin -t v3.0.1-Ugin_X.X.X -p $SPARK_HOME/kubernetes/dockerfiles/spark/bindings/python/Dockerfile build``` to build the project, where you can replace X.X.X in the image version with any version you want. My images are publicly available at my [dockerhub-profile](https://hub.docker.com/repository/docker/kostjaigin/spark-py), I recommend using the 0.2.0 version.

I conduct my experiments using the [experiments-script](https://github.com/kostjaigin/bachelor/blob/master/experiments.sh). It refers to the execution script of each strategy that consists of the pre-implemented spark-submit command in an .sh file. [Here](https://github.com/kostjaigin/bachelor/blob/master/exe.sh) is an example of the execution file for the strategy AA.

❗️To save experiment results, you need an installed hdfs cluster with a shared data storage. Point at your hdfs-cluster with additional application parameters --hdfs_host and hdfs_port. The parameter description for the execution files is as follows: TODO...

## The structure of the project

The project is based on the default Apache Spark distribution version 3.0.1. Additinally we have a number of .sh and python scripts for different use cases. We save experimential results on an hdfs-cluster. 
The main application logic can be found in [data/App.py](https://github.com/kostjaigin/bachelor/blob/master/data/App.py) for the strategy AA, [data/AppDB.py](https://github.com/kostjaigin/bachelor/blob/master/data/AppDB.py) for the strategy AB, [data/AppFrames.py](https://github.com/kostjaigin/bachelor/blob/master/data/AppFrames.py) for the strategy B.

"Dependencies" folder inside of the data folder contains the dependency files, prediction data for test sets is available under the same named folder. "Results" folder contains the results of our experiments in excel and .csv formats. "Utils" folder contains a number of support python scripts. 

We conducted 336 experiments in total. The execution on cluster was not performed manually. "Experiments" shell-script available in the root folder and composes submission of required application with experiment parameters. For all three strategies, it executes a call on a corresponding [Spark-Submit] (https://spark.apache.org/docs/latest/submitting-applications.html) Shell-Script:

- *exe.sh* performs Spark-Submit operation for the strategy AA.
- *exeDB.sh* performs Spark-Submit operation for the strategy AB.
- *exeFrames.sh* performs Spark-Submit operation for the strategy B.

"Experiments" script: 1) loads the required dataset into the database *if required*, 2) starts the experiment with a given configuration, 3) removes the successfully completed pods to keep the cluster cleen, 4) saves the experiments results from the HDFS-storage. 
