import jsonlines
import random

# Function to parse the F1 score from the evaluation results
def parse_f1(result):
    parts = result.split()
    for part in parts:
        if "f1:" in part:
            return float(part.split("f1:")[1])
    return 0.0

# Open the source file and create a new file to store results
with jsonlines.open('/home/mhoveyda/AdaptiveQA/GPTSwarm/REINFORCE_EXPERIMENTS/extracted_bkk_w_atts_all.jsonl') as reader, jsonlines.open('/home/mhoveyda/AdaptiveQA/GPTSwarm/REINFORCE_EXPERIMENTS/extracted_bkk_w_atts_all_final_f1s.jsonl', mode='w') as writer:
    for obj in reader:
        answer = obj['answer']
        nor_answer = obj['NoR_predicted_answer']
        ircot_answer = obj['IRCoT_predicted_answer']
        
        nor_f1 = parse_f1(obj['NoR_evaluation_results'])
        ircot_f1 = parse_f1(obj['IRCoT_evaluation_results'])
        
        # Determine which F1 score to use based on the answer
        if answer == nor_answer and answer == ircot_answer:
            # Both answers match, pick randomly
            final_f1 = random.choice([nor_f1, ircot_f1])
        elif answer == nor_answer:
            final_f1 = nor_f1
        elif answer == ircot_answer:
            final_f1 = ircot_f1
        else:
            final_f1 = 0.0
        
        # Add the new attribute
        obj['final_f1'] = final_f1
        
        # Write the modified object to the new file
        writer.write(obj)
