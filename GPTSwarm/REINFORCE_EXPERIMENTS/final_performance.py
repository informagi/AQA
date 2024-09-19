import json

def calculate_f1_scores(file_path):
    # Dictionaries to store total F1 scores and counts per complexity label
    f1_scores = {}
    counts = {}
    
    # Open and read the JSONL file
    with open(file_path, 'r') as file:
        for line in file:
            # Parse each line to a JSON object
            data = json.loads(line)
            
            # Extract complexity label and final_f1 score
            complexity = data['complexity_label']
            final_f1 = data['final_f1']
            
            # Accumulate the F1 scores and counts per complexity
            if complexity in f1_scores:
                f1_scores[complexity] += final_f1
                counts[complexity] += 1
            else:
                f1_scores[complexity] = final_f1
                counts[complexity] = 1

    # Dictionary to store average F1 scores per complexity label
    average_f1_scores = {}

    # Calculate average F1 scores per complexity label
    for label in f1_scores:
        average_f1_scores[label] = f1_scores[label] / counts[label]
    
    # Calculate overall average F1 score
    total_f1 = sum(f1_scores.values())
    total_count = sum(counts.values())
    overall_average = total_f1 / total_count

    return average_f1_scores, overall_average

# Example usage
file_path = '/home/mhoveyda/AdaptiveQA/GPTSwarm/REINFORCE_EXPERIMENTS/extracted_bkk_w_atts_all_final_f1s.jsonl'
average_scores, overall_avg = calculate_f1_scores(file_path)
print("Average F1 Scores per Complexity Label:", average_scores)
print("Overall Average F1 Score:", overall_avg)
