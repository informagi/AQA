import json

def load_tsv_to_dict(tsv_file):
    tsv_data = {}
    with open(tsv_file, 'r') as file:
        headers = file.readline().strip().split('\t')
        for line in file:
            values = line.strip().split('\t')
            entry = dict(zip(headers, values))
            tsv_data[entry['question_id']] = entry
    return tsv_data

def update_jsonl_with_tsv(jsonl_file, tsv_data, output_file):
    with open(jsonl_file, 'r') as json_file, open(output_file, 'w') as outfile:
        for line in json_file:
            data = json.loads(line)
            task_id = data['task_id']
            # Strip any prefixes from task_id to match the format in the TSV file
            # adjusted_task_id = task_id.replace('single_trivia_dev_', 'single_nq_dev_')
            adjusted_task_id = task_id
            # Check if the corresponding entry exists in TSV data
            if adjusted_task_id in tsv_data:
                # Extract needed attributes from TSV data
                data['complexity_label'] = tsv_data[adjusted_task_id]['complexity_label']
                data['NoR_time_taken'] = tsv_data[adjusted_task_id]['NoR_time_taken']
                data['IRCoT_time_taken'] = tsv_data[adjusted_task_id]['IRCoT_time_taken']
                data['gold_answers'] = tsv_data[adjusted_task_id]['gold_answers']
                data['NoR_predicted_answer'] = tsv_data[adjusted_task_id]['NoR_predicted_answer']
                data['IRCoT_predicted_answer'] = tsv_data[adjusted_task_id]['IRCoT_predicted_answer']
                data['NoR_evaluation_results'] = tsv_data[adjusted_task_id]['NoR_evaluation_results']
                data['IRCoT_evaluation_results'] = tsv_data[adjusted_task_id]['IRCoT_evaluation_results']

            else:
                print(f"Task ID {task_id} not found in TSV data.")
            # Write the updated JSON to the new file
            json.dump(data, outfile)
            outfile.write('\n')

# File paths
jsonl_file_path = 'extracted_bkk.jsonl'
tsv_file_path = '/home/mhoveyda/AdaptiveQA/GPTSwarm/datasets/AQA/val.tsv'
output_jsonl_path = 'extracted_bkk_w_atts_all.jsonl'

# Load TSV data
tsv_data = load_tsv_to_dict(tsv_file_path)

# Update JSONL file with attributes from TSV
update_jsonl_with_tsv(jsonl_file_path, tsv_data, output_jsonl_path)
