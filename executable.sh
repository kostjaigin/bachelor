$SPARK_HOME/bin/spark-submit \
--master k8s://https://130.149.249.40:6443 \
--deploy-mode cluster \
--conf spark.executor.instances=5 \
--conf spark.kubernetes.namespace=konstantin \
--conf spark.kubernetes.authenticate.driver.serviceAccountName=konstantin \
--conf spark.kubernetes.container.image=docker.io/kostjaigin/spark-py:v3.0.1-Ugin_0.0.6 \
--files "local:///opt/spark/data/build/dll/libgnn.d" \
--files "local:///opt/spark/data/build/dll/libgnn.so" \
--files "local:///opt/spark/data/build/lib/config.d" \
--files "local:///opt/spark/data/build/lib/config.o" \
--files "local:///opt/spark/data/build/lib/graph_struct.d" \
--files "local:///opt/spark/data/build/lib/graph_struct.o" \
--files "local:///opt/spark/data/build/lib/msg_pass.d" \
--files "local:///opt/spark/data/build/lib/msg_pass.o" \
--files "local:///opt/spark/data/models/USAir_hyper.pkl" \
--files "local:///opt/spark/data/models/USAir_model.pth" \
--py-files "local:///opt/spark/data/dependencies.zip" \
local:///opt/spark/data/App.py
