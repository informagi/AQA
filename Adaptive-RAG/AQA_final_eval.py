

import json
import jsonlines
import argparse
from collections import defaultdict
from typing import Dict, List
from metrics.squad_answer_em_f1 import SquadAnswerEmF1Metric

def load_ground_truths_and_datasets(data_file_path: str) -> Dict:
    print(f"Loading data from {data_file_path}")
    id_to_ground_truths = {}
    id_to_complexity = {}
    with jsonlines.open(data_file_path, 'r') as input_file:
        for line in input_file:
            qid = line['question_id']
            answer = line['answers_objects'][0]['spans']
            complexity = line['complexity_label']
            id_to_ground_truths[qid] = answer
            id_to_complexity[qid] = complexity
    return id_to_ground_truths, id_to_complexity

def load_predictions(prediction_file_path: str) -> Dict:
    with open(prediction_file_path, "r") as file:
        id_to_predictions = json.load(file)
    return id_to_predictions

def save_results(results_dict: Dict, output_path: str) -> None:
    with open(output_path, "w") as file:
        json.dump(results_dict, file, indent=4)

def calculate_acc(prediction: str, ground_truths: List[str]) -> float:
    normalized_prediction = normalize_answer(prediction)
    for ground_truth in ground_truths:
        # if normalized_prediction in normalize_answer(ground_truth):
        if any(normalize_answer(ground_truth) in normalized_prediction for ground_truth in ground_truths):
            # print(f"Accuracy Normalized Prediction: {normalized_prediction} {type(normalized_prediction)}, {type(normalize_answer(ground_truth))} {normalize_answer(ground_truth)}")

            return 1.0
    return 0.0

def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    import re
    import string

    def remove_articles(text: str) -> str:
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def evaluate(pred_file_path: str, data_file_path: str, output_file_path: str) -> None:
    id_to_predictions = load_predictions(pred_file_path)
    id_to_ground_truths, id_to_complexity = load_ground_truths_and_datasets(data_file_path)
    
    overall_metrics = SquadAnswerEmF1Metric()
    complexity_metrics = defaultdict(SquadAnswerEmF1Metric)
    total_acc = 0.0
    total_count = 0

    complexity_acc = defaultdict(float)
    complexity_count = defaultdict(int)

    for qid, prediction in id_to_predictions.items():
        if qid not in id_to_ground_truths:
            print(f"Warning: Question ID {qid} not found in ground truths.")
            continue
        
        ground_truths = id_to_ground_truths[qid]
        complexity = id_to_complexity[qid]

        if isinstance(prediction, list):
            prediction = prediction[0]  

        # flatten ground_truths list if it contains nested lists
        flat_ground_truths = [item for sublist in ground_truths for item in sublist] if any(isinstance(i, list) for i in ground_truths) else ground_truths
        
        overall_metrics(prediction, flat_ground_truths)
        complexity_metrics[complexity](prediction, flat_ground_truths)
        acc = calculate_acc(prediction, flat_ground_truths)
        total_acc += acc
        total_count += 1

        complexity_acc[complexity] += acc
        complexity_count[complexity] += 1

        em_score = overall_metrics.get_metric()["em"]
        f1_score = overall_metrics.get_metric()["f1"]
        print(f"Question ID: {qid}")
        print(f"Prediction: {prediction}")
        print(f"Ground Truths: {flat_ground_truths}")
        print(f"Exact Match (EM): {em_score}")
        print(f"F1 Score: {f1_score}")
        print(f"Accuracy: {acc}\n")

    # overall metrics
    overall_metric_results = overall_metrics.get_metric()
    overall_metric_results["accuracy"] = round(total_acc / total_count, 3) if total_count > 0 else 0.0

    # complexity-wise metrics
    complexity_results = {}
    for complexity, metrics in complexity_metrics.items():
        metric_results = metrics.get_metric()
        metric_results["accuracy"] = round(complexity_acc[complexity] / complexity_count[complexity], 3) if complexity_count[complexity] > 0 else 0.0
        complexity_results[complexity] = metric_results

    final_results = {
        "overall": overall_metric_results,
        "by_complexity": complexity_results
    }
    save_results(final_results, output_file_path)
def evaluate_single(prediction: str, gold_answers: List[str]) -> Dict[str, float]:
    metric = SquadAnswerEmF1Metric()

    # Normalize the answers and calculate metrics
    norm_prediction = normalize_answer(prediction)
    norm_gold_answers = [normalize_answer(answer) for answer in gold_answers]
    metric(norm_prediction, norm_gold_answers)
    
    results = metric.get_metric()

    # Calculate Accuracy
    accuracy = 1.0 if any(norm_gold in norm_prediction for norm_gold in norm_gold_answers) else 0.0
    results['accuracy'] = accuracy

    return results
def main():
    parser = argparse.ArgumentParser(description="Evaluate QA Model Predictions.")
    parser.add_argument(
        "prediction_file_path",
        type=str,
        help="Filepath to the prediction JSON file."
    )
    parser.add_argument(
        "data_file_path",
        type=str,
        help="Filepath to the original data JSONL file with ground truths."
    )
    parser.add_argument(
        "--output_file_path",
        type=str,
        default="evaluation_results.json",
        help="Filepath to save the evaluation results."
    )
    args = parser.parse_args()

    evaluate(args.prediction_file_path, args.data_file_path, args.output_file_path)

if __name__ == "__main__":
    main()
