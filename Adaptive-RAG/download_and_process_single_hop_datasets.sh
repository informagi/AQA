#!/bin/bash

# Download Natural Question
echo "Downloading Natural Question"
mkdir -p raw_data/nq
cd raw_data/nq
wget https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-nq-dev.json.gz
gzip -d biencoder-nq-dev.json.gz
wget https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-nq-train.json.gz
gzip -d biencoder-nq-train.json.gz
echo "Successfully downloaded Natural Question.\n"

# Download TriviaQA
cd ..
mkdir -p trivia
echo "Downloading TriviaQA"
cd trivia
wget https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-trivia-dev.json.gz
gzip -d biencoder-trivia-dev.json.gz
wget https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-trivia-train.json.gz
gzip -d biencoder-trivia-train.json.gz
echo "Successfully downloaded TriviaQA.\n"

# Download SQuAD
cd ..
mkdir -p squad
echo "Downloading SQuAD"
cd squad
wget https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-squad1-dev.json.gz
gzip -d biencoder-squad1-dev.json.gz
wget https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-squad1-train.json.gz
gzip -d biencoder-squad1-train.json.gz
echo "Successfully downloaded SQuAD.\n"

# Download Wiki passages. For the single-hop datasets, we use the Wikipedia as the document corpus.
cd ..
mkdir -p wiki
echo "Downloading Wikipedia passages"
cd wiki
wget https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz
gzip -d psgs_w100.tsv.gz
echo "Successfully downloaded Wikipedia passages.\n"

# Process raw data files in a single standard format
cd ../..
echo "Processing raw data files"
python ./processing_scripts/process_nq.py
python ./processing_scripts/process_trivia.py
python ./processing_scripts/process_squad.py
echo "Successfully processed raw data files.\n"

# Subsample the processed datasets
echo "Subsampling the processed datasets"
python processing_scripts/subsample_dataset_and_remap_paras.py nq test 500
python processing_scripts/subsample_dataset_and_remap_paras.py nq dev_diff_size 500
python processing_scripts/subsample_dataset_and_remap_paras.py trivia test 500
python processing_scripts/subsample_dataset_and_remap_paras.py trivia dev_diff_size 500
python processing_scripts/subsample_dataset_and_remap_paras.py squad test 500
python processing_scripts/subsample_dataset_and_remap_paras.py squad dev_diff_size 500
echo "Successfully subsampled the processed datasets.\n"

# Build index
# python retriever_server/build_index.py wiki