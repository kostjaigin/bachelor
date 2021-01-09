$SPARK_HOME/bin/spark-submit \
--master k8s://https://130.149.249.40:6443 \
--deploy-mode cluster \
--conf spark.executor.instances=5 \
--conf spark.kubernetes.namespace=konstantin \
--conf spark.kubernetes.container.image=docker.io/kostjaigin/spark-py:v3.0.1-Ugin_0.0.1 \
--files "local:///opt/spark/data/models/USAir_hyper.pkl","local:///opt/spark/data/models/USAir_model.pth","local:///opt/spark/data/build/dll/libgnn.d","local:///opt/spark/data/build/dll/libgnn.so","local:///opt/spark/data/build/lib/config.d","local:///opt/spark/data/build/lib/config.o","local:///opt/spark/data/build/lib/graph_struct.d","local:///opt/spark/data/build/lib/graph_struct.o","local:///opt/spark/data/build/lib/msg_pass.d","local:///opt/spark/data/build/lib/msg_pass.o" \
--py-files "local:///opt/spark/data/dependencies.zip" \
local:///opt/spark/data/App.py
