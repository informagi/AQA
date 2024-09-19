import json
import math
from statistics import mean

def process_jsonl_file(file_path):
    total_times = []
    complexity_based_times = {}

    # Read the file and process each line
    with open(file_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            # Sum the times and calculate the logarithm
            total_time = float(data['NoR_time_taken']) + float(data['IRCoT_time_taken'])
            # convert to ms
            total_time = total_time * 1000
            total_time_log = math.log(total_time)
            # data['total_time_taken'] = total_time_log
            total_times.append(total_time_log)

            # Categorize based on complexity
            complexity_label = data['complexity_label']
            if complexity_label not in complexity_based_times:
                complexity_based_times[complexity_label] = []
            complexity_based_times[complexity_label].append(total_time_log)

    # Calculate overall average of total times
    overall_average = mean(total_times)
    print(f"Overall Average Total Time Taken (Log): {overall_average}")

    # Calculate averages by complexity
    complexity_averages = {}
    for complexity, times in complexity_based_times.items():
        complexity_averages[complexity] = mean(times)
        print(f"Average Total Time Taken (Log) for Complexity {complexity}: {complexity_averages[complexity]}")

    return overall_average, complexity_averages

# Example usage

# Example usage
file_path = '/home/mhoveyda/AdaptiveQA/GPTSwarm/REINFORCE_EXPERIMENTS/extracted_bkk_w_atts.jsonl'
process_jsonl_file(file_path)
