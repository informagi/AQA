# AQA: Adaptive Question Answering in a Society of LLMs

This is the repository for ***AQA***, a novel adaptive question answering framework that uniquely frames the adaptive QA problem as a contextual multi-armed bandit problem, where the action space is a set of graph structures among LLM agents that describe their interactions. Thereby, the AQA framework dynamically orchestrates the collaboration of multiple agents in response to specific question characteristics.

AQA is built based on various codebases/papers; agents are defined/implemented using [IRCoT](https://github.com/StonyBrookNLP/ircot), the swarms (composit-graphs) are designed using [GPTSwarm](https://github.com/metauto-ai/GPTSwarm/tree/main). Formulating Adaptive QA as a Contextual Multi-Armed Bandit problem, we then use the data from [Adaptive-RAG](https://github.com/starsuzi/Adaptive-RAG) to train and evaluate our [LinUCB](https://arxiv.org/pdf/1003.0146) model.

- - -

# 1. Getting Started 

First we need to setup the agents and prepare the datasets required for experiments.
```bash
$ conda create -n AQA python=3.8
$ conda activate AQA
$ git clone https://github.com/informagi/AQA.git
$ cd AQA
$ pip install torch==1.13.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
$ pip install -r requirements.txt
```

### 1.1. Servers
The retrieval server is necessary for the agents that use retrieval (IR and IRCoT).
```bash
$ cd Adaptive-RAG
$ wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.10.2-linux-x86_64.tar.gz
$ wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.10.2-linux-x86_64.tar.gz.sha512
$ shasum -a 512 -c elasticsearch-7.10.2-linux-x86_64.tar.gz.sha512
$ tar -xzf elasticsearch-7.10.2-linux-x86_64.tar.gz

# Starting the ElasticSearch Server
$ cd elasticsearch-7.10.2/
$ ./bin/elasticsearch  # pkill -f elasticsearch  (To stop the server),  curl http://localhost:9200 (To check the elasticsearch server is running)

# Starting the Retrieval Server
$ conda install uvicorn 
$ uvicorn serve:app --port 8000 --app-dir retriever_server
```

To start the LLM Server:

```bash
MODEL_NAME=flan-t5-xl uvicorn serve:app --port 8010 --app-dir llm_server # model_name: flan-t5-xxl, flan-t5-xl
```


### 1.2. Adaptive-RAG Datasets 
Download the data provided by Adaptive-RAG.
```bash
$ bash ./download/processed_data.sh
$ bash ./download/raw_data.sh
$ python processing_scripts/subsample_dataset_and_remap_paras.py musique dev_diff_size 500
$ python processing_scripts/subsample_dataset_and_remap_paras.py hotpotqa dev_diff_size 500
$ python processing_scripts/subsample_dataset_and_remap_paras.py 2wikimultihopqa dev_diff_size 500
```

## 1.3. Indexing the Data
```bash
# Build index for multihop datasets 
python retriever_server/build_index.py {dataset_name} # hotpotqa, 2wikimultihopqa, musique
# Handle one-hop datasets and index wiki for them
mv download_and_process_single_hop_datasets.sh Adaptive-RAG/download_and_process_single_hop_datasets.sh
bash download_and_process_single_hop_datasets.sh
python retriever_server/build_index.py wiki
```
After all the indices are created, executing `curl localhost:9200/_cat/indices` should give you the following statistics:
```bash
yellow open 2wikimultihopqa D3G8zgeLSnSAO9uDqmP_aQ 1 1   430225 0 235.4mb 235.4mb
yellow open hotpotqa        C7MAO0frRmit2OVA1eGrPg 1 1  5233329 0   2.1gb   2.1gb
yellow open musique         yAyiaj5rSXWEvoeH7-umcg 1 1   139416 0  81.9mb  81.9mb
yellow open wiki            -J8mtXSkRxWZJ5mGkIyCcQ 1 1 21015324 0  13.3gb  13.3gb
```

### 1.4. Sample and Split Dataset for AQA experiments

In our experiments, we use the gold complexity labels from the data used to train AdaptiveRAG's classifier, which can be downloaded by:
```bash
mkdir -p downloaded_data && cd downloaded_data && wget https://github.com/starsuzi/Adaptive-RAG/raw/main/data.tar.gz && tar -xzvf data.tar.gz && rm data.tar.gz
```

We use the dataset for the flan-t5-xl model. Of the two versions (binary: inductive bias, and binary-silver: model answers + inductive bias), we use the binary-silver version, which has 3,809 data points. This will be split for training and testing in our evaluations.

Use `AQA_dataset_organizer.py` to 1) add IDs to the simple datasets (nq, trivia, squad), 2) attach gold answers, and 3) format data to squad style. Then, run `AQA_dataset_splitter.py` to split the dataset into train and test sets. Adjust the `dataset_path` in the script as needed."

We randomly extract 210 samples for training and 51 for testing, maintaining equal complexity label distribution. For AQA and GPTSwarm experiments, we use this distribution of complexity labels.

### 1.4.1. Update
As the combined data (Silver+Binary) comes with majority of the instances being from the inductive bias source, we did our experiments based on the silver version only (hence the results in the paper are based on the silver only version data). 
To use the silver version only;

```bash
export TRAIN_FILE_PATH="Adaptive-RAG/downloaded_data/classifier/data/musique_hotpot_wiki2_nq_tqa_sqd/flan_t5_xl/silver/train.json"
export RAW_DATA_FOLDER="Adaptive-RAG/raw_data"
export OUTPUT_FILE_PATH="Adaptive-RAG/downloaded_data/classifier/data/musique_hotpot_wiki2_nq_tqa_sqd/flan_t5_xl/silver/train_w_answers.json"
export TRANSFORMED_FILE_PATH="Adaptive-RAG/downloaded_data/classifier/data/musique_hotpot_wiki2_nq_tqa_sqd/flan_t5_xl/silver/train_w_answers_in_squad_format.json"

python AQA_dataset_organizer.py --train_file_path "$TRAIN_FILE_PATH" --raw_data_folder "$RAW_DATA_FOLDER" --output_file_path "$OUTPUT_FILE_PATH" --transformed_file_path "$TRANSFORMED_FILE_PATH"

```
Note: To use silver+binary version, change the TRAIN_FILE_PATH to silver_binary instead.

```bash
python AQA_dataset_splitter.py # Will save the final data under AQA_Data_Final folder
```

# 2. Experiments

## 2.1. Agents' Configuration
In Adaptive-RAG, the hyperparameters and prompt schemes for each experiment are defined in the config files located in the `base_configs` folder. We selected the most relevant config files for our experiments which can be found under the `Adaptive-RAG/base_configs_selected_for_AQA` folder.

## 2.2. Individual Agents' Evaluation

### 2.2.1. Run Experiment
Make sure to adjust `input_path`, `base_config_folder`, `base_output_folder` and `base_log_folder` variables before running the experiments in script `run_inference.sh`. You should run this for both train and the test file that has been created using `AQA_dataset_splitter.py` script.

To run the experiments:

```bash
export RETRIEVER_HOST="http://localhost"
export RETRIEVER_PORT=8000
export BASE_CONFIG_FOLDER=$(realpath "./base_configs_selected_for_AQA")
```
For test set:

```bash
export BASE_LOG_FOLDER=$(mkdir -p "../LOGS/test/$(date +'%Y-%m-%d')" && realpath "../LOGS/test/$(date +'%Y-%m-%d')")
export INPUT_PATH=$(realpath "../AQA_Data_Final/test_aware_210_51.jsonl")
export BASE_OUTPUT_FOLDER=$(mkdir -p "$(dirname "../Results/test/IndividualAgents")" && realpath "../Results/test/IndividualAgents")

./run_inference.sh noR
./run_inference.sh oneR
./run_inference.sh ircot

```


For train set:
```bash
export BASE_LOG_FOLDER=$(mkdir -p "$(dirname "../LOGS/train/$(date +'%Y-%m-%d')")" && realpath "../LOGS/train/$(date +'%Y-%m-%d')")
export INPUT_PATH=$(realpath "../AQA_Data_Final/train_aware_210_51.jsonl")
export BASE_OUTPUT_FOLDER=$(mkdir -p "$(dirname "../Results/train/IndividualAgents")" && realpath "../Results/train/IndividualAgents")

./run_inference.sh noR
./run_inference.sh oneR
./run_inference.sh ircot

```
<br>
We use the results (answers) generated with the run_inference.sh script to train and evaluate AQA and GPTSwarm. 


### 2.2.2. Evaluate Results
The evaluation should be run for all the combinations of `EVAL_TYPE` and `MODEL_TYPE`

```bash
export EVAL_TYPE="train" # or train
export PREDICTION_DIR=$(realpath "../Results/$EVAL_TYPE/IndividualAgents")
export GOLD_FILE=$(realpath "../AQA_Data_Final/${EVAL_TYPE}_aware_210_51.jsonl")
export OUTPUT_DIR=$(mkdir -p "../Evaluation/IndividualAgents/$EVAL_TYPE" && realpath "../Evaluation/IndividualAgents/$EVAL_TYPE")
export MODEL_TYPE="NoR" # or oner or ircot 
export LOG_DIR=$(mkdir -p "../LOGS/Evaluation/IndividualAgents/$EVAL_TYPE/$(date +'%Y-%m-%d')" && realpath "../LOGS/Evaluation/IndividualAgents/$EVAL_TYPE/$(date +'%Y-%m-%d')")

./run_evaluation.sh
```

To view the overall scores check the  `OUTPUT_DIR` and to check the per sample evaluation check the `LOG_DIR`.
To Visualize the scores use the `visualize_results.py` script and feed the score files paths to it. 


## 2.3. AQA Train and Evaluation
First to prepare train and test files, run the script `Adaptive-RAG/preprocess_results_for_CMAB_and_GPTSwarm_experiments.ipynb`. The resulting test and train files are in the format suitable for CMAB experiments.

Use `CMAB_last.py` and `CMAB_last_swarm.py` scripts to train and the relevant evaluation scripts for assessment. 


```bash
python CMAB_last.py ./Results/processed_data_for_CMAB/train_aware_210_51_complete.jsonl LOGS/CMAB_Ind/logss.txt
```

```bash
python CMAB_last_swarm.py ./Results/processed_data_for_CMAB/train_aware_210_51_complete.jsonl ./Results/processed_data_for_CMAB/test_aware_210_51_complete.jsonl LOGS/CMAB_Swarm/logs.txt
```

## 2.4. GPTSwarm Train and Evaluation
Necessary changes are made to `GPTSwarm` repository to support our graph design. Run the following script to train and evaluate GPTSwarm and compare it with AQA;

```bash
cd GPTSwarm
conda create -n swarm python=3.10
conda activate swarm
pip install poetry
poetry install
PYTHONPATH=. python experiments/run_aqa.py > ../LOGS/GPTSwarm/logs.txt 2>&1
```

- - - 