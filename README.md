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

To check the elasticsearch is running, use `curl http://localhost:9200`.



```bash
conda install uvicorn 
uvicorn serve:app --port 8000 --app-dir retriever_server
```

### 1.1.2 Starting the LLM Server

```bash
MODEL_NAME=flan-t5-xl uvicorn serve:app --port 8010 --app-dir llm_server # model_name: flan-t5-xxl, flan-t5-xl
```


### 1.1.3 Adaptive-RAG Datasets and Indices
Download the data provided by Adaptive-RAG.
```bash
$ bash ./download/processed_data.sh
$ bash ./download/raw_data.sh
$ python processing_scripts/subsample_dataset_and_remap_paras.py musique dev_diff_size 500
$ python processing_scripts/subsample_dataset_and_remap_paras.py hotpotqa dev_diff_size 500
$ python processing_scripts/subsample_dataset_and_remap_paras.py 2wikimultihopqa dev_diff_size 500
```

```bash
# Build index
python retriever_server/build_index.py {dataset_name} # hotpotqa, 2wikimultihopqa, musique

# Handle one-hop datasets and index wiki for them
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



### 1.1.4 Dataset Split for AQA experiments

In our experiments, we use the gold complexity labels from the data used to train AdaptiveRAG's classifier, which can be downloaded by:
```bash
mkdir -p downloaded_data && cd downloaded_data && wget https://github.com/starsuzi/Adaptive-RAG/raw/main/data.tar.gz && tar -xzvf data.tar.gz && rm data.tar.gz
```

We use the dataset for the flan-t5-xl model. Of the two versions (binary: inductive bias, and binary-silver: model answers + inductive bias), we use the binary-silver version, which has 3,809 data points. This will be split for training and testing in our evaluations.

Use `AQA_dataset_organizer.py` to 1) add IDs to the simple datasets (nq, trivia, squad), 2) attach gold answers, and 3) format data to squad style. Then, run `AQA_dataset_splitter.py` to split the dataset into train and test sets. Adjust the `dataset_path` in the script as needed."

We randomly extract 210 samples for training and 51 for testing, maintaining equal complexity label distribution. For AQA and GPTSwarm experiments, we use this distribution of complexity labels.

### 1.1.4.1 Update
As the combined data (Silver+Binary) comes with majority of the instances being from the inductive bias source, we did our experiments based on the silver version only (hence the results in the paper are based on the silver only version data). 
To use the silver version only;
```bash
export TRAIN_FILE_PATH="Adaptive-RAG/downloaded_data/classifier/data/musique_hotpot_wiki2_nq_tqa_sqd/flan_t5_xl/silver/train.json"
export RAW_DATA_FOLDER="Adaptive-RAG/raw_data"
export OUTPUT_FILE_PATH="Adaptive-RAG/downloaded_data/classifier/data/musique_hotpot_wiki2_nq_tqa_sqd/flan_t5_xl/silver/train_w_answers.json"
export TRANSFORMED_FILE_PATH="Adaptive-RAG/downloaded_data/classifier/data/musique_hotpot_wiki2_nq_tqa_sqd/flan_t5_xl/silver/train_w_answers_in_squad_format.json"

# python AQA_dataset_organizer.py --train_file_path TRAIN_FILE_PATH --raw_data_folder RAW_DATA_FOLDER --output_file_path OUTPUT_FILE_PATH --transformed_file_path TRANSFORMED_FILE_PATH
python AQA_dataset_organizer.py --train_file_path "$TRAIN_FILE_PATH" --raw_data_folder "$RAW_DATA_FOLDER" --output_file_path "$OUTPUT_FILE_PATH" --transformed_file_path "$TRANSFORMED_FILE_PATH"

```
If decided to use silver+binary version then adapt the TRAIN_FILE_PATH to silver_binary instead.

# 2. Experiments

## 2.1. Config Files
In the Adaptive-RAG repository, the hyperparameters and prompt schemes used in each experiment is defined using the config files in the [base_configs folder](AQA_project/Adaptive-RAG/base_configs). We chose the config files that were closest to our experiment setup and use them for our experiments. These files can be found under the [base_configs_selected_for_AQA folder](https://gitlab.science.ru.nl/mhoveyda/AdaptiveQA-2/-/tree/main/Adaptive-RAG/base_configs_selected_for_AQA?ref_type=heads).

## 2.2. Individual Agents' Evaluation

### 2.2.1. Run Experiment
Make sure to adjust `input_path`, `base_config_folder`, `base_output_folder` and `base_log_folder` variables before running the experiments in script `run_inference.sh`. You should run this for both train and the test file that has been created using `AQA_dataset_splitter.py` script.

To run the experiments:

```bash
export RETRIEVER_HOST="http://localhost"
export RETRIEVER_PORT=8000
./run_inference.sh {systm-type} # nor, oner, ircot
```
<br>
We did the Individual Agents evaluation for both the test and train datasets.  We use these results (answers) generated with the run_inference.sh script to train the CMAB. 


## 2.2.2. Evaluate Results

To evaluate experiments first adjust the `PREDICTION_DIR`, `OURPUT_DIR` and `LOG_DIR` variables in the `run_evaluation.sh` script and then;

```bash
./run_evaluation.sh {systm-type} # nor, oner, ircot
```
<br>
To view the overall scores check the OUTPUT_DIR and to check the per sample evaluation check the LOG_DIR.
<br>

To Visualize the scores use the `visualize_results.py` script and feed the score files paths to it. 


## 2.4. AQA Train and Evaluation
To train and test for Individual and Orchestrated optimization with AQA, make sure `ircot` and `Adaptive-RAG` repositories are available. Afterwards, Use `CMAB_last.py` and `CMAB_last_swarm.py` scripts to train and the relevant evaluation scripts for assessment. 

## 2.5. GPTSwarm Train and Evaluation
Necessary changes are made to `GPTSwarm` repository to support our graph design. You can run `run_aqa.py` script to train and evaluate GPTSwarm to compare it with AQA.

- - - 