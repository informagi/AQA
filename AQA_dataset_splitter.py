import json
import random
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os

class DatasetHandler:
    def __init__(self, dataset_path, destination_folder, seed=None):
        self.dataset_path = dataset_path
        self.destination_folder = destination_folder
        self.seed = seed
        self.data_all = self._load_data()
    
    def _load_data(self):
        with open(self.dataset_path, 'r') as file:
            return [json.loads(line) for line in file]
    
    def dataset_statistics(self):
        print(f"Total number of instances: {len(self.data_all)}")
        dataset_counts = defaultdict(int)
        complexity_counts = defaultdict(int)
        
        for item in self.data_all:
            dataset_counts[item['dataset']] += 1
            complexity_counts[item['complexity_label']] += 1
        
        df = pd.DataFrame({
            'Dataset': list(dataset_counts.keys()),
            'Count': list(dataset_counts.values())
        })
        df_comp = pd.DataFrame({
            'Complexity Label': list(complexity_counts.keys()),
            'Count': list(complexity_counts.values())
        })
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        sns.barplot(x='Dataset', y='Count', data=df, ax=axes[0])
        sns.barplot(x='Complexity Label', y='Count', data=df_comp, ax=axes[1])
        
        axes[0].set_title('Distribution of Dataset Types')
        axes[1].set_title('Distribution of Complexity Labels')
        plt.show()

    def split_dataset(self, mode='ratio', train_ratio=0.8, train_size=None, test_size=None):
        if self.seed:
            random.seed(self.seed)
        
        if mode == 'ratio':
            train_size = int(len(self.data_all) * train_ratio)
            test_size = len(self.data_all) - train_size
            split_name = f"ratio_{train_ratio}"
        
        elif mode == 'fixed':
            if train_size is None or test_size is None:
                raise ValueError("train_size and test_size must be specified for 'fixed' mode.")
            split_name = f"fixed_{train_size}_{test_size}"
        
        elif mode == 'aware_split':
            if train_size is None or test_size is None:
                raise ValueError("train_size and test_size must be specified for 'aware_split' mode.")
            split_name = f"aware_{train_size}_{test_size}"
            self._aware_split(train_size, test_size)
            return
        
        else:
            raise ValueError("Invalid mode. Choose 'ratio', 'fixed', or 'aware_split'.")
        
        question_ids = set(item['question_id'] for item in self.data_all)
        train_ids = set(random.sample(question_ids, train_size))
        test_ids = question_ids - train_ids
        
        train_data = [item for item in self.data_all if item['question_id'] in train_ids]
        test_data = [item for item in self.data_all if item['question_id'] in test_ids]
        
        self._save_split(train_data, 'train', split_name)
        self._save_split(test_data, 'test', split_name)
        
        print(f"Train size: {len(train_data)}, Test size: {len(test_data)}")

    def _aware_split(self, train_size, test_size):
        complexity_groups = defaultdict(list)
        
        for item in self.data_all:
            complexity_groups[item['complexity_label']].append(item)
        
        unique_labels = list(complexity_groups.keys())
        train_target_count = train_size // len(unique_labels)
        test_target_count = test_size // len(unique_labels)
        
        train_data = []
        test_data = []
        
        for label in unique_labels:
            random.shuffle(complexity_groups[label])
            train_data.extend(complexity_groups[label][:train_target_count])
            test_data.extend(complexity_groups[label][train_target_count:train_target_count + test_target_count])
        
        self._save_split(train_data, 'train', f"aware_{train_size}_{test_size}")
        self._save_split(test_data, 'test', f"aware_{train_size}_{test_size}")
        
        print(f"Train size: {len(train_data)}, Test size: {len(test_data)}")

    def _save_split(self, data, split_name, mode_name):
        output_file = f"{self.destination_folder}/{split_name}_{mode_name}.jsonl"
        with open(output_file, 'w') as file:
            for item in data:
                file.write(json.dumps(item) + '\n')
    
    def sample_data(self, sample_size=15, data_split=None):
        if self.seed:
            random.seed(self.seed)
        
        if data_split:
            if data_split not in ['train', 'test']:
                raise ValueError("data_split must be 'train' or 'test'.")
            data = self._load_split(data_split)
        else:
            data = self.data_all
        
        unique_datasets = list(set(item['dataset'] for item in data))
        unique_labels = list(set(item['complexity_label'] for item in data))
        dataset_target_count = max(1, sample_size // len(unique_datasets))
        label_target_count = max(1, sample_size // len(unique_labels))

        dataset_sample = []
        for dataset in unique_datasets:
            dataset_group = [item for item in data if item['dataset'] == dataset]
            random.shuffle(dataset_group)
            dataset_sample.extend(dataset_group[:dataset_target_count])

        label_sample = []
        label_counts = defaultdict(int)
        random.shuffle(dataset_sample)
        for item in dataset_sample:
            label = item['complexity_label']
            if label_counts[label] < label_target_count:
                label_sample.append(item)
                label_counts[label] += 1

        remaining_slots = sample_size - len(label_sample)
        remaining_data = [item for item in dataset_sample if item not in label_sample]
        random.shuffle(remaining_data)
        label_sample.extend(remaining_data[:remaining_slots])

        random.shuffle(label_sample)

        output_file = f"{self.destination_folder}/sample_{data_split or 'all'}.jsonl"
        with open(output_file, 'w') as file:
            for item in label_sample:
                file.write(json.dumps(item) + '\n')

    def _load_split(self, split_name):
        split_file = f"{self.destination_folder}/{split_name}.jsonl"
        with open(split_file, 'r') as file:
            return [json.loads(line) for line in file]

    def split_stats_figure(self, mode, train_size=None, train_ratio=None, test_size=None):
        if mode == 'ratio':
            split_name = f"ratio_{train_ratio}"
        elif mode == 'fixed':
            split_name = f"fixed_{train_size}_{test_size}"
        elif mode == 'aware_split':
            split_name = f"aware_{train_size}_{test_size}"
        else:
            raise ValueError("Invalid mode. Choose 'ratio', 'fixed', or 'aware_split'.")
        
        train_data = self._load_split(f'train_{split_name}')
        test_data = self._load_split(f'test_{split_name}')
        
        train_counts = defaultdict(int)
        test_counts = defaultdict(int)
        train_labels = defaultdict(int)
        test_labels = defaultdict(int)
        
        for item in train_data:
            train_counts[item['dataset']] += 1
            train_labels[item['complexity_label']] += 1
        
        for item in test_data:
            test_counts[item['dataset']] += 1
            test_labels[item['complexity_label']] += 1
        
        df_train = pd.DataFrame({
            'Dataset': list(train_counts.keys()),
            'Count': list(train_counts.values())
        })
        df_test = pd.DataFrame({
            'Dataset': list(test_counts.keys()),
            'Count': list(test_counts.values())
        })
        df_train_labels = pd.DataFrame({
            'Complexity Label': list(train_labels.keys()),
            'Count': list(train_labels.values())
        })
        df_test_labels = pd.DataFrame({
            'Complexity Label': list(test_labels.keys()),
            'Count': list(test_labels.values())
        })
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        sns.barplot(x='Dataset', y='Count', data=df_train, ax=axes[0, 0])
        sns.barplot(x='Dataset', y='Count', data=df_test, ax=axes[0, 1])
        sns.barplot(x='Complexity Label', y='Count', data=df_train_labels, ax=axes[1, 0])
        sns.barplot(x='Complexity Label', y='Count', data=df_test_labels, ax=axes[1, 1])
        
        axes[0, 0].set_title('Train Dataset Distribution')
        axes[0, 1].set_title('Test Dataset Distribution')
        axes[1, 0].set_title('Train Complexity Label Distribution')
        axes[1, 1].set_title('Test Complexity Label Distribution')
        plt.show()

dataset_path =  "/home/mhoveyda/AdaptiveQA/Downloaded data/classifier/data/musique_hotpot_wiki2_nq_tqa_sqd/flan_t5_xl/silver/train_w_answers_in_squad_format.json"
destination_folder = './AdaptiveQA/AdaptiveQA_Data_Final'

if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

handler = DatasetHandler(dataset_path, destination_folder, seed=42)
handler.dataset_statistics()
handler.split_dataset(mode='ratio', train_ratio=0.8)
handler.split_dataset(mode='aware_split', train_size=210, test_size=51)
handler.split_stats_figure(mode='ratio', train_ratio=0.8)
handler.split_stats_figure(mode='aware_split', train_size=210, test_size=51)
handler.sample_data(sample_size=15)
