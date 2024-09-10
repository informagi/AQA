import os
import json
import glob
from tqdm import tqdm
import argparse

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Process raw data files for QA datasets.")
    parser.add_argument('--train_file_path', type=str, required=True,
                        help='Path to the train file.')
    parser.add_argument('--raw_data_folder', type=str, required=True,
                        help='Path to the folder containing raw data.')
    parser.add_argument('--output_file_path', type=str, required=True,
                        help='Path to save the processed train file with answers.')
    parser.add_argument('--transformed_file_path', type=str, required=True,
                        help='Path to save the transformed train file in SQuAD format.')
    return parser.parse_args()

# Since the simple datasets in raw_data do not have ids, we need to add them first and then do the answer extraction
def add_id_to_simple_datasets(raw_data_folder):
    simple_datasets_list = ["nq", "squad", "trivia"]
    for dataset_name in simple_datasets_list:
        for dev_or_train in ["dev", "train"]:
            if dataset_name == "squad":
                path_to_file = os.path.join(raw_data_folder, dataset_name, f"biencoder-{dataset_name}1-{dev_or_train}.json")
            else:
                path_to_file = os.path.join(raw_data_folder, dataset_name, f"biencoder-{dataset_name}-{dev_or_train}.json")
            output_path = path_to_file.replace(".json", "_with_ids.json")
            if os.path.exists(output_path):
                print(f"File {output_path} already exists.")
                continue
            with open(path_to_file, "r") as f:
                data = json.load(f)
            for i, d in enumerate(data):
                if "id" in d:
                    continue
                d["id"] = f"single_{dataset_name}_{dev_or_train}_{i}"
            with open(output_path, "w") as f:
                json.dump(data, f)
            print(f"Updated data saved to {output_path}")

def extract_answer(data_point, dataset_name):
    # if dataset_name in ['2wikimultihopqa', 'hotpotqa', 'squad', 'trivia']:
    if 'answers_objects' in data_point:
        return data_point['answers_objects'][0]['spans']
    # elif dataset_name == 'nq':
    if 'answers' in data_point:
        return data_point['answers']
    # elif dataset_name == 'musique':
    if "answer" in data_point:
        return data_point["answer"]
    return None

# preload all raw data into a nested dictionary for faster access
def preload_raw_data(raw_data_folder):
    raw_data = {}
    dataset_folders = glob.glob(f"{raw_data_folder}/*")

    for folder_path in tqdm(dataset_folders, desc="Preloading datasets"):
        dataset_name = os.path.basename(folder_path)
        if dataset_name not in ['2wikimultihopqa', 'hotpotqa', 'musique', 'nq', 'squad', 'trivia']:
            continue
        raw_data[dataset_name] = {}
        files = glob.glob(f"{folder_path}/*.*json*")

        for file_path in files:
            if "id_alias" not in file_path:
                try:
                    with open(file_path, 'r') as f:
                        if file_path.endswith('.json'):
                            data_points = json.load(f)
                        elif file_path.endswith('.jsonl'):
                            data_points = [json.loads(line) for line in f]

                    raw_data[dataset_name][file_path] = {
                        dp.get("id") or dp.get("_id"): extract_answer(dp, dataset_name) for dp in data_points if dp.get("id") or dp.get("_id")
                    }
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")

    return raw_data

def transform_json(input_file, output_file):
    with open(input_file, 'r') as file:
        data_all = json.load(file)
    
    with open(output_file, 'w') as file:
        for data in data_all:
            question_id = data.get("id")
            question_text = data.get("question")
            answer_text = data.get("answer_text")
            dataset = data.get("dataset_name")
            
            new_data = {
                "dataset": dataset,
                "question_id": question_id,
                "question_text": question_text,
                "level": None,  
                "type": None,   
                "complexity_label": data.get("answer"),
                "answers_objects": [{
                    "number": None,  
                    "date": {
                        "day": None,  
                        "month": None,  
                        "year": None  
                    },
                    "spans": [answer_text]
                }],
                "contexts": [] 
            }
            
            file.write(json.dumps(new_data) + '\n')

def find_answer_in_preloaded_data(raw_data_dict, dataset_name, data_id):
    if dataset_name in raw_data_dict:
        for file_data in raw_data_dict[dataset_name].values():
            if data_id in file_data:
                return file_data[data_id]
    return None




def main():


    args = parse_args()

    print(f"\n\nParsed arguments: {args}\n\n")
    
    add_id_to_simple_datasets(args.raw_data_folder)
    print(f"\n\nAdded ids to simple datasets\n\n")

    if not os.path.exists(args.train_file_path):
        raise FileNotFoundError(f"Train file not found at {args.train_file_path}")
    with open(args.train_file_path, 'r') as file:
        train_data = json.load(file)
    
    print(f"\n\nLoaded train data from {args.train_file_path}\n\n")

    raw_data_dict = preload_raw_data(args.raw_data_folder)

    print(f"\n\nPreloaded raw data\n\n")

    for data_point in tqdm(train_data, desc="Processing data points"):
        dataset_name = data_point.get("dataset_name")
        data_id = data_point.get("id")
        answer_text = find_answer_in_preloaded_data(raw_data_dict, dataset_name, data_id)
        if answer_text:
            data_point["answer_text"] = answer_text
        else:
            raise ValueError(f"Answer not found for {dataset_name} {data_id}")
    with open(args.output_file_path, 'w') as file:
        json.dump(train_data, file, indent=4)

    print(f"\n\nUpdated train data saved to {args.output_file_path}\n\n")

    transform_json(args.output_file_path, args.transformed_file_path)
    print(f"\n\nTransformed train data saved to {args.transformed_file_path}\n\n")

if __name__ == "__main__":
    main()