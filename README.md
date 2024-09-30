# milvus-bge-m3

Repo for starting milvus standalone or as a cluster

Requirements

- Docker
- Docker Compose
- Python 3

**IMPORTANT**
If you have started one of the docker compose files and you are planning to start the other, delete the volumes folder otherwise it will not work.


## Starting Milvus Cluster

docker compose command to start standalone **docker-compose -f docker-compose-v2.4-cluster.yml up -d** wait for the containers to be healthy with **docker ps**

## Starting Milvus Standalone

docker compose command to start standalone **docker-compose -f docker-compose-v2.4-standalone.yml up -d** wait for the containers to be healthy with **docker ps**

## Running test script to check if Milvus is working

- Install requirements with **pip3 install -r requirements.txt**
- Run the script with **python3 hello_milvus.py**


## milvus and bgem3

Installing additional requirements

pip install --upgrade pymilvus
pip install "pymilvus[model]"

Make sure yo have an instance of milvus running

Create a folder called pdf_files and insert all the pdf files inside

bgem3.py script will process all documents by pages and will insert them inside milvus in a collection where the schema is created with an id, the emeded pages and the index of the page

afterwards there is a querry defined embedded an queried in milvus to find the closes distance to the target

the Id, distance and the index is printed

cluster images explained 

milvus etcd -> https://milvus.io/docs/v2.1.x/configure_etcd.md
milvus pulsar -> https://milvus.io/docs/configure_pulsar.md
milvus minio -> https://milvus.io/docs/configure_minio.md
milvus rootcoord -> https://milvus.io/docs/configure_rootcoord.md
milvus proxy -> https://milvus.io/docs/configure_proxy.md
milvus querycord -> https://milvus.io/docs/configure_querycoord.md
milvus querynode -> https://milvus.io/docs/configure_querynode.md#Query-Node-related-Configurations
milvus indexcoord -> https://milvus.io/docs/v2.0.x/configure_indexcoord.md
milvus indexnode -> https://milvus.io/docs/v2.0.x/configure_indexnode.md#Index-Node-related-Configurations
milvus datacoord -> https://milvus.io/docs/v2.0.x/configure_datacoord.md#dataCoordaddress
milvus datanode -> https://milvus.io/docs/configure_datanode.md

