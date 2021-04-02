# Using Graph Neural Networks for Distributed Link Prediction

My bachelor thesis work. **Apache Spark, Python, Kubernetes for deployment, PyTorch for Neural Networks**. I have my deadline on 29 April, I will post a more detailed description of the thesis until then. 

Utils folder contains help functionality and environment setup files required to repeat the experiments from my bachelor thesis. 

## Environment setup
Having a kubernetes cluster up and running, connect to the master machine and do the following..

#### Neo4j Cluster Initialization

1. Make sure your cluster doesn't run Neo4j yet: ```helm uninstall neo4j-helm``` *(it is okay if it outputs an error)*
2. Clone this repository to your working directory: ```git clone https://github.com/kostjaigin/bachelor.git```
3. Clone Neo4j Helm repository to your working directory: ```git clone https://github.com/neo4j-contrib/neo4j-helm.git```
4. Go to the Helm Configuration file and change following lines as demonstrated below:
    - ```acceptLicenseAgreement: "yes"```
    - ```authEnabled: false``` to disable authentication requirement
    - Find the section *core* and inside of it change:
      ```
      persistentVolume:
        ## whether or not persistence is enabled
        ##
        enabled: false
      ```
    - Do the same for the section *readReplica*:
      ```
      persistentVolume:
        enabled: true
      ```
5. Install/start the Neo4j cluster: ```helm install neo4j-helm -f ./neo4j-helm/values.yaml neo4j-helm```
6. Now we need to wait until pods are set up and are running. We can check the current state with ```kubectl get pods``` command.

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

Now we can load data to the database.

#### Load data to Neo4j K8S Cluster
*e.g. We want to upload the "USAir" dataset. All datasets are available in .csv format in the "data" folder inside of this repository.*

1. To load data, you will need to access cypher-shell inside one of Neo4j Core containers. I pick Core-0 and do: ```kubectl exec --stdin --tty neo4j-helm-neo4j-core-0 -- /bin/bash``` and then to access cypher: ```cypher-shell```
2. First make sure the database is empty by typing: ```match (n) detach delete(n);```
3. Upload nodes:
```cypher
LOAD CSV WITH HEADERS FROM 'https://raw.githubusercontent.com/kostjaigin/bachelor/master/utils/data/USAir_nodes.csv' AS row
CREATE (n:Node {id: toInteger(row.id)});
```
4. Upload edges:
```cypher
LOAD CSV WITH HEADERS FROM 'https://raw.githubusercontent.com/kostjaigin/bachelor/master/utils/data/USAir_edges.csv' AS row
MATCH (src:Node {id: toInteger(row.src_id)}),(dst:Node {id: toInteger(row.dst_id)})
CREATE (src)-[:CONNECTION]->(dst);
```

## Build and execute the project
The main application logic can be found in [data/App.py](https://github.com/kostjaigin/bachelor/blob/master/data/App.py).

#### Using standard spark-submit approach described [here](http://spark.apache.org/docs/latest/running-on-kubernetes.html) 

1. Set enviromental variable *$SPARK_HOME* to the directory of this repository
2. Do ```$SPARK_HOME/bin/docker-image-tool.sh -r kostjaigin -t v3.0.1-Ugin_0.0.1 -p $SPARK_HOME/kubernetes/dockerfiles/spark/bindings/python/Dockerfile build``` to build project
3. Application is then submitted using Spark K8S module:
```
$SPARK_HOME/bin/spark-submit \
  --master k8s://https://kubernetes.docker.internal:6443 \
  --deploy-mode cluster \
  --conf spark.executor.instances=5 \
  --conf spark.kubernetes.container.image=kostjaigin/spark-py:v3.0.1-Ugin_0.0.1 \
  --files "local:///opt/spark/data/models/USAir_hyper.pkl","local:///opt/spark/data/models/USAir_model.pth","local:///opt/spark/data/build/dll/libgnn.d","local:///opt/spark/data/build/dll/libgnn.so","local:///opt/spark/data/build/lib/config.d","local:///opt/spark/data/build/lib/config.o","local:///opt/spark/data/build/lib/graph_struct.d","local:///opt/spark/data/build/lib/graph_struct.o","local:///opt/spark/data/build/lib/msg_pass.d","local:///opt/spark/data/build/lib/msg_pass.o" \
  --py-files "local:///opt/spark/data/dependencies.zip" \
  local:///opt/spark/data/App.py
```
