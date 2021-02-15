kubectl exec neo4j-helm-neo4j-core-0 -- /bin/bash -c "cypher-shell \"match(n) detach delete(n);\"";
kubectl exec neo4j-helm-neo4j-core-0 -- /bin/bash -c "cypher-shell \"LOAD CSV WITH HEADERS FROM 'https://raw.githubusercontent.com/kostjaigin/bachelor/master/utils/data/$1_nodes.csv' AS row
CREATE (n:Node {id: toInteger(row.id)});\"";
kubectl exec neo4j-helm-neo4j-core-0 -- /bin/bash -c "cypher-shell \"LOAD CSV WITH HEADERS FROM 'https://raw.githubusercontent.com/kostjaigin/bachelor/master/utils/data/$1_edges.csv' AS row
MATCH (src:Node {id: toInteger(row.src_id)}),(dst:Node {id: toInteger(row.dst_id)})
CREATE (src)-[:CONNECTION]->(dst);\"";