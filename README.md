# AQA: Adaptive Question Answering in a Society of LLMs

This is the repository for ***AQA***, a novel adaptive question answering framework that uniquely frames the adaptive QA problem as a contextual multi-armed bandit problem, where the action space is a set of graph structures among LLM agents that describe their interactions. Thereby, the AQA framework dynamically orchestrates the collaboration of multiple agents in response to specific question characteristics.

AQA is built based on various codebases/papers; agents are defined/implemented using [IRCoT](https://github.com/StonyBrookNLP/ircot), the swarms (composit-graphs) are designed using [GPTSwarm](https://github.com/metauto-ai/GPTSwarm/tree/main). Formulating Adaptive QA as a Contextual Multi-Armed Bandit problem, we then use the data from [Adaptive-RAG](https://github.com/starsuzi/Adaptive-RAG) to train and evaluate our [LinUCB](https://arxiv.org/pdf/1003.0146) model.

- - -

# 1. Getting Started 

## 1.1. Agents, Servers, and Datasets 
First we need to setup the agents and prepare the datasets required for experiments.

```bash
$ conda create -n AQA python=3.8
$ conda activate AQA
$ git clone https://github.com/starsuzi/Adaptive-RAG.git
$ cd Adaptive-RAG
$ pip install torch==1.13.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
$ pip install -r requirements.txt
```

### 1.1.1 Starting Retrieval Server
The retrieval server is necessary for the agents that use retrieval (IR and IRCoT).
```bash
$ wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.10.2-linux-x86_64.tar.gz
$ wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.10.2-linux-x86_64.tar.gz.sha512
$ shasum -a 512 -c elasticsearch-7.10.2-linux-x86_64.tar.gz.sha512
$ tar -xzf elasticsearch-7.10.2-linux-x86_64.tar.gz

# Starting the Server
$ cd elasticsearch-7.10.2/
$ ./bin/elasticsearch 

# pkill -f elasticsearch # to stop the server
```

After starting the server, to check the elasticsearch is running, `curl http://localhost:9200` should yield the following:

```bash

# {
#   "name" : "tusi",
#   "cluster_name" : "elasticsearch",
#   "cluster_uuid" : "RVCNKMvYRSysbDUIy-Q_-w",
#   "version" : {
#     "number" : "7.10.2",
#     "build_flavor" : "default",
#     "build_type" : "tar",
#     "build_hash" : "747e1cc71def077253878a59143c1f785afa92b9",
#     "build_date" : "2021-01-13T00:42:12.435326Z",
#     "build_snapshot" : false,
#     "lucene_version" : "8.7.0",
#     "minimum_wire_compatibility_version" : "6.8.0",
#     "minimum_index_compatibility_version" : "6.0.0-beta1"
#   },
#   "tagline" : "You Know, for Search"
# }

```


