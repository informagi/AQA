
import numpy as np
import json
import time
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import random
import pandas as pd
from collections import defaultdict
# Get the path to the Adaptive-RAG folder
adaptive_rag_path = os.path.join(os.path.dirname(__file__), 'Adaptive-RAG')

# Add the Adaptive-RAG path to the Python system path
sys.path.append(adaptive_rag_path)
from metrics.squad_answer_em_f1 import SquadAnswerEmF1Metric  
from collections import Counter
import argparse
import logging


# Argument Parsing
parser = argparse.ArgumentParser(description="Run Contextual Multi-Armed Bandit (CMAB) experiments.")
parser.add_argument("training_data_path", type=str, help="Path to the training data JSONL file.")
parser.add_argument("test_data_path", type=str, help="Path to the test data JSONL file.")
parser.add_argument("log_file_path", type=str, help="Path to save the log file.")


args = parser.parse_args()

logging.basicConfig(filename=args.log_file_path, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filemode='w')

logging.info("Starting Contextual Multi-Armed Bandit (CMAB) experiments for Swarm setting.")
print(f"Training data path: {args.training_data_path}")
print(f"Test data path: {args.test_data_path}")
print(f"Log file path: {args.log_file_path}")

# def reset_random_seeds():
# os.environ['PYTHONHASHSEED'] = str(42)
random.seed(42)
np.random.seed(42)
logging.info(f"Random seeds reset to 42.")

# reset_random_seeds()  

DEBUG = True
TIME_IN_REWARD = True


n_arms = 3  
n_features = 3  
ALPHA = 2
EPOCHS = 50
PERFORMANCE_WEIGHT = 1
TIME_PENALTY = 0.01

# training_data_path = "/home/mhoveyda/AdaptiveQA/Adaptive-RAG/DATA_FOR_CMAB/Train_Data_For_CMAB.json"
# training_data_path = "/home/mhoveyda/AdaptiveQA/Adaptive-RAG/DATA_FOR_CMAB/Train_Data_For_CMAB_augmented_with_confidence.json"
# training_data_path = "./Results/processed_data_for_CMAB/train_aware_210_51_complete.jsonl"
# test_data_path = "./Results/processed_data_for_CMAB/test_aware_210_51_complete.jsonl"
training_data_path = args.training_data_path
test_data_path = args.test_data_path

# Create the CMAB folder filename based on the parameters
# CMAB_figures_folder_filename = f"CMAB_figures_alpha_{ALPHA}_epochs_{EPOCHS}_reward_with_Performance_{PERFORMANCE_IN_REWARD}_{PERFORMANCE_WEIGHT}_time_{TIME_IN_REWARD}_{TIME_WEIGHT}_AdaptiveTimePenalty_{ADAPTIVE_TIME_PENALTY}_confidence_{CONFIDENCE_IN_REWARD}_{CONFIDENCE_WEIGHT}"
# CMAB_Figures = f"/home/mhoveyda/AdaptiveQA/Adaptive-RAG/CMAB_Swarm_FIGS/12-June/{CMAB_figures_folder_filename}"

# Create the CMAB folder filename based on the parameters
# date_with_time = time.strftime("%d-%b-%Y_%H-%M-%S")
# CMAB_figures_folder_filename = f"CMAB_figures_alpha_{ALPHA}_epochs_{EPOCHS}_reward_with_Performance_{PERFORMANCE_WEIGHT}_time_{TIME_IN_REWARD}"
# CMAB_Figures = f"/home/mhoveyda/AdaptiveQA/Adaptive-RAG/CMAB_Swarm_FIGS/24-June/{date_with_time}/{CMAB_figures_folder_filename}"
date_time = time.strftime("%d-%b")
CMAB_Figures = f"./Results/CMAB_Swarm_FIGS/{date_time}"

# path_to_evaluation_results_folder = {
#     "nor":"/home/mhoveyda/AdaptiveQA/Agents_Executed_Final_Evaluation_Results/output_results_nor.json",
#     "oner":"/home/mhoveyda/AdaptiveQA/Agents_Executed_Final_Evaluation_Results/output_results_oner.json",
#     "ircot":"/home/mhoveyda/AdaptiveQA/Agents_Executed_Final_Evaluation_Results/output_results_ircot.json"
# }







if not os.path.exists(CMAB_Figures):
    os.makedirs(CMAB_Figures)



def load_data(filepath):
    data = []
    with open(filepath, 'r') as file:
        for line in file:
            data.append(json.loads(line))

    # if DEBUG:
    #     data = data[:10]
    #     logging.info(f"\n\nSuccessfully loaded data from {filepath}.")


    return data


def complexity_to_vector(label):
    return {'A': [1, 0, 0], 'B': [0, 1, 0], 'C': [0, 0, 1]}[label]


def majority_vote(answers):
    # Count the frequency of each element
    counts = Counter(answers)
    sorted_items = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
    # The first element in the sorted list will be the one with the highest count and the lowest value in case of a tie
    return sorted_items[0][0]

def calculate_reward(performance_score, time_taken, time_in_reward):
    logging.info(f"Performance score: {performance_score}, Time taken: {time_taken}, Time in reward: {time_in_reward}")

    if time_taken<=1:
        time_penalty = 0
    elif time_taken>1 and time_taken<=10: # Since the time is np.log(milliseconds) this takes effect for both nor and oner
    # else:
        time_penalty = 0.0001
        # logging.info(f"Time less than 10: {time_taken}")   
    
    elif time_taken>10: # Since the time is np.log(milliseconds) this takes effect for ircot and any combined configuration
        time_penalty = 0.02
        # logging.info(f"Time greater than 10: {time_taken}")
    # time_penalty = TIME_PENALTY if time_taken >= 1 else 0
    # return PERFORMANCE_WEIGHT  * performance_score - (time_penalty * time_taken if time_in_reward else 0)

    if time_in_reward:
        return performance_score - time_penalty * time_taken
    else:
        return performance_score
 

class DecisionGraph:
    def __init__(self):

        self.configurations = [tuple((i >> j) & 1 for j in range(2, -1, -1)) for i in range(1, 8)]

        logging.info(f"Configurations: {self.configurations}")
        self.agent_map = {0: 'nor', 1: 'oner', 2: 'ircot'}
    
    def execute_graph(self, config_index, data):
        config = self.configurations[config_index]
        logging.info(f"\n Graph config: {config}")
        connected_agents = [i for i in range(3) if config[i]]
        connected_agents_names = [self.agent_map[agent] for agent in connected_agents]  # List of agent names
        logging.info(f"Connected agents: {connected_agents_names}")
        logging.info(f"Count of connected agents: {len(connected_agents)}")

        answers = []
        f1_scores = []
        times = []
        # confidences = []
        
        for agent in connected_agents:
            agent_key = self.agent_map[agent]
            # logging.info(f"Agent key: {agent_key}")
            
            answers.append(data[f'{agent_key}_predicted_answer'])
            f1_scores.append(data[f'{agent_key}_evaluation_results']['f1'])
            # turn to ms and get the log
            times.append(np.log(data[f'{agent_key}_time_taken']*1000))
            # times.append(data[f'{agent_key}_time_taken'])
            # times.append(np.log(data[f'{agent_key}_time_taken']))
            # confidences.append(data[f'{agent_key}_average_confidence_score_of_the_last_subkey'])

            logging.info(f"Agent: {agent_key}, Answer: {answers[-1]}, F1 Score: {f1_scores[-1]}, Time: {times[-1]}")
    
        # logging.info(f"Answers: {answers}")
        # logging.info(f"F1 scores: {f1_scores}")
        # logging.info(f"Times: {times}")
        # logging.info(f"Confidences: {confidences}")

        # # Majority vote
        # final_answer = max(set(answers), key=answers.count)
        final_answer = majority_vote(answers)

        logging.info(f"Final answer: {final_answer}\n")
        total_time = sum(times)
        # Filter confidences for agents whose answers match the majority vote
        # majority_confidences = [confidences[i] for i, answer in enumerate(answers) if answer == final_answer]
        # average_confidence = sum(majority_confidences) / len(majority_confidences) if majority_confidences else 0

        # logging.info(f"Final answer: {final_answer}")
        # logging.info(f"Total time: {total_time}")
        # logging.info(f"Average confidence (only from majority): {average_confidence}")
        # return final_answer, total_time, average_confidence, connected_agents_names
        return final_answer, total_time, connected_agents_names

    
    def get_num_configurations(self):
        return len(self.configurations)


def evaluate_graph_configurations(data, graph):

    results = {}

    for config_index, config in enumerate(graph.configurations):
        metrics = SquadAnswerEmF1Metric()
        total_time = 0
        total_confidence = 0
        count = 0

        complexity_stats = defaultdict(lambda: {"metric": SquadAnswerEmF1Metric(), "total_time": 0, "total_confidence": 0, "count": 0})

        for instance in data:
            predicted_answer, time_taken, confidence, connected_agents_names = graph.execute_graph(config_index, instance)
            ground_truths = instance['gold_answers']
            complexity = instance['complexity_label']

            metric_result = evaluate_single(predicted_answer, ground_truths)
            metrics(predicted_answer, ground_truths)
            complexity_stats[complexity]["metric"](predicted_answer, ground_truths)

            total_time += time_taken 
            total_confidence += confidence
            count += 1

            complexity_stats[complexity]["total_time"] += time_taken
            complexity_stats[complexity]["total_confidence"] += confidence
            complexity_stats[complexity]["count"] += 1

        overall_metrics = metrics.get_metric()
        overall_metrics.update({
            "average_time": total_time / count if count else 0,
            "average_confidence": total_confidence / count if count else 0
        })

        # Process complexity metrics
        complexity_results = {}
        for comp, stats in complexity_stats.items():
            metric_results = stats["metric"].get_metric()
            metric_results["average_time"] = stats["total_time"] / stats["count"] if stats["count"] else 0
            metric_results["average_confidence"] = stats["total_confidence"] / stats["count"] if stats["count"] else 0
            complexity_results[comp] = metric_results

        config_name = "_".join([graph.agent_map[i] for i in range(3) if config[i]])
        output_path = f"evaluation_results_{config_name}.json"
        save_results({
            "overall": overall_metrics,
            "by_complexity": complexity_results
        }, output_path)
        results[config_name] = {
            "overall": overall_metrics,
            "by_complexity": complexity_results
        }

def save_results(results_dict, output_path):
    with open(output_path, "w") as file:
        json.dump(results_dict, file, indent=4)

def evaluate_single(prediction, gold_answers):
    metric = SquadAnswerEmF1Metric()
    metric(prediction, gold_answers)
    return metric.get_metric()

def load_results_from_files(directory):
    data = []
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r') as file:
                results = json.load(file)
                config_name = filename.replace("evaluation_results_", "").replace(".json", "")
                # Extract data for all complexities and overall
                for complexity, metrics in results["by_complexity"].items():
                    data.append({
                        "Configuration": config_name,
                        "Complexity": complexity,
                        "F1 Score": metrics["f1"],
                        "EM": metrics["em"],
                        # "Accuracy": metrics["accuracy"],
                        "Average Confidence": metrics["average_confidence"],
                        "Average Time": metrics["average_time"]
                    })
                # Overall results
                overall = results["overall"]
                data.append({
                    "Configuration": config_name,
                    "Complexity": "Overall",
                    "F1 Score": overall["f1"],
                    "EM": overall["em"],
                    # "Accuracy": overall["accuracy"],
                    "Average Confidence": overall["average_confidence"],
                    "Average Time": overall["average_time"]
                })
    return pd.DataFrame(data)
def plot_metrics(df, metric_name, title, ylabel, output_directory):
    fig, ax = plt.subplots(figsize=(10, 6))
    complexity_order = ['A', 'B', 'C', 'Overall']

    df['Complexity'] = pd.Categorical(df['Complexity'], categories=complexity_order, ordered=True)
    df = df.sort_values('Complexity')

    for key, grp in df.groupby(['Configuration']):
        ax = grp.plot(ax=ax, kind='line', x='Complexity', y=metric_name, label=key, marker='o')
    
    plt.title(title)
    plt.xlabel('Complexity')
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend(title='Configuration')

    plt.xticks(range(len(complexity_order)), complexity_order)
    plt.show()
    plt.savefig(f"{output_directory}/{title}_{metric_name}_plot.png")


def plot_expected_rewards(linucb):
    complexity_labels = ['A', 'B', 'C']
    configurations = [(0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)]
    agent_map = {0: 'nor', 1: 'oner', 2: 'ircot'}
    configuration_labels = [
        '+'.join([agent_map[idx] for idx, present in enumerate(config) if present])
        for config in configurations
    ]

    logging.info(f"Configuration Labels: {configuration_labels}")

    for label in complexity_labels:
        plt.figure(figsize=(10, 8))
        for config_index, config_label in enumerate(configuration_labels):
            # Flter entries for the current complexity and configuration, and flatten the array structure
            rewards = [
                entry[1][config_index][0][0]  # Adju
                for entry in linucb.expected_reward_matrix_history
                if np.array_equal(entry[0], np.array(complexity_to_vector(label)))  #
            ]

            if rewards:  # Only plot if there  rewards to plot
                plt.plot(rewards, label=f'{config_label} (Config {config_index + 1})')
            else:
                logging.info(f"No rewards data for {config_label} with Complexity {label}")

        plt.title(f'Evolution of Expected Rewards for Complexity {label}')
        plt.xlabel('Time Step')
        plt.ylabel('Expected Reward')
        plt.legend(title='Configuration')
        plt.grid(True)
        plt.show()
        plt.savefig(f"{CMAB_Figures}/Expected_Rewards_{label}.png")



def plot_combined_contextual_rewards(time_based_data, non_time_based_data, save_path):

    sns.set(style="whitegrid")  
    context_labels = ['A', 'B', 'C']  

    configurations = [(0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)]
    agent_map = {0: 'nor', 1: 'oner', 2: 'ircot'}
    arm_labels = ['+'.join([agent_map[idx] for idx, present in enumerate(config) if present]) for config in configurations]

    datasets = [non_time_based_data, time_based_data]
    titles = ['Non-Time-Based Rewards', 'Time-Based Rewards']
    max_timesteps = 0  # store the maximum number of timesteps across all plots

    for data in datasets:
        expected_reward_matrix, _, context_history, _ = data
        for context_index in range(expected_reward_matrix.shape[0]):
            context_mask = [np.argmax(c) == context_index for c in context_history]
            max_timesteps = max(max_timesteps, sum(context_mask))

    fig, axs = plt.subplots(2, len(context_labels), figsize=(18, 10), sharey=False)  # 2 rows for two datasets

    for row_index, data in enumerate(datasets):
        expected_reward_matrix, _, context_history, rewards_history = data
        n_contexts = expected_reward_matrix.shape[0]
        n_arms = expected_reward_matrix.shape[1]

        for context_index in range(n_contexts):
            ax = axs[row_index, context_index]
            y_min, y_max = float('inf'), float('-inf')  
            
            for arm_index in range(n_arms):
                context_mask = [np.argmax(c) == context_index for c in context_history]
                context_specific_rewards = expected_reward_matrix[context_index, arm_index, context_mask]
                ax.plot(context_specific_rewards, label=f'{arm_labels[arm_index]} - Estimated')
                
                y_min = min(y_min, min(context_specific_rewards))
                y_max = max(y_max, max(context_specific_rewards))
                #                 if row_index == 0:
                if row_index == 0:
                    if context_index == 0:
                        if arm_labels[arm_index] == "nor":
                            logging.info(f"{row_index=}, {context_index=}, {arm_labels[arm_index]=}")

                            ax.axhline(y=np.mean([reward for arm, reward, context in rewards_history if arm == arm_index and context == context_index]), color='r', linestyle='--')


                    elif context_index == 1:
                        if arm_labels[arm_index] == "ircot":
                            logging.info(f"{row_index=}, {context_index=}, {arm_labels[arm_index]=}")

                            ax.axhline(y=np.mean([reward for arm, reward, context in rewards_history if arm == arm_index and context == context_index]), color='r', linestyle='--')


                    elif context_index == 2:
                        if arm_labels[arm_index] == "ircot":
                            logging.info(f"{row_index=}, {context_index=}, {arm_labels[arm_index]=}")

                            ax.axhline(y=np.mean([reward for arm, reward, context in rewards_history if arm == arm_index and context == context_index]), color='r', linestyle='--')


                if row_index == 1:
                    if context_index == 0:
                        if arm_labels[arm_index] == "nor":
                            logging.info(f"{row_index=}, {context_index=}, {arm_labels[arm_index]=}")
                            ax.axhline(y=np.mean([reward for arm, reward, context in rewards_history if arm == arm_index and context == context_index]), color='r', linestyle='--')
                    elif context_index == 1:
                        if arm_labels[arm_index] == "oner":
                            logging.info(f"{row_index=}, {context_index=}, {arm_labels[arm_index]=}")
                            ax.axhline(y=np.mean([reward for arm, reward, context in rewards_history if arm == arm_index and context == context_index]), color='r', linestyle='--')
                    elif context_index == 2:
        
                        if arm_labels[arm_index] == "ircot":
                            logging.info(f"{row_index=}, {context_index=}, {arm_labels[arm_index]=}")
                            ax.axhline(y=np.mean([reward for arm, reward, context in rewards_history if arm == arm_index and context == context_index]), color='r', linestyle='--')


            ax.set_ylim([y_min - 0.1 * (y_max - y_min), y_max + 0.1 * (y_max - y_min)])
            
            ax.set_xlim(0, max_timesteps - 1)  
            ax.set_title(f'Context {context_labels[context_index]} ({titles[row_index]})')
            if context_index == 1 and row_index == 1:
                ax.set_xlabel('Time Step')
            if row_index in [0, 1] and context_index == 0:
                ax.set_ylabel('Expected Reward')

            

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0), ncol=len(arm_labels), frameon=True)
    fig.subplots_adjust(bottom=0.1, top=0.95)
    plt.savefig(f'{save_path}/combined_rewards_context.png')
    plt.close()

def plot_combined_contextual_UCSBs(time_based_data, non_time_based_data, save_path):
    sns.set(style="whitegrid")  
    context_labels = ['A', 'B', 'C']  
    
    configurations = [(0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)]
    agent_map = {0: 'nor', 1: 'oner', 2: 'ircot'}
    arm_labels = ['+'.join([agent_map[idx] for idx, present in enumerate(config) if present]) for config in configurations]

    datasets = [non_time_based_data, time_based_data]
    titles = ['Non-Time-Based UCBs', 'Time-Based UCBs']
    
    max_timesteps = 0  

    for data in datasets:
        # context_expected_rewards, context_ucbs, self.context_history, self.real_reward_history
        _, ucbs_matrix, context_history, _ = data
        for context_index in range(ucbs_matrix.shape[0]):
            context_mask = [np.argmax(c) == context_index for c in context_history]
            max_timesteps = max(max_timesteps, sum(context_mask))

    fig, axs = plt.subplots(2, len(context_labels), figsize=(18, 10), sharey=True)  # 2 rows for datasets, 3 columns for contexts

    for row_index, data in enumerate(datasets):
        _, ucbs_matrix, context_history, _ = data
        # ucbs_matrix, context_history = data
        n_contexts = ucbs_matrix.shape[0]
        n_arms = ucbs_matrix.shape[1]

        for context_index in range(n_contexts):
            ax = axs[row_index, context_index]
            for arm_index in range(n_arms):
                context_mask = [np.argmax(c) == context_index for c in context_history]
                context_specific_ucbs = ucbs_matrix[context_index, arm_index, context_mask]
                line_color = sns.color_palette()[arm_index % len(sns.color_palette())]  # cycle colors
                ax.plot(context_specific_ucbs, label=f'{arm_labels[arm_index]} UCB', color=line_color)

            ax.set_xlim(0, max_timesteps - 1)  #
            ax.set_title(f'Context {context_labels[context_index]} ({titles[row_index]})')
            if row_index == 1:
                ax.set_xlabel('Relevant Time Step')
            if context_index == 0:  #
                ax.set_ylabel('Upper Confidence Bound (UCB)')

    #  the legend
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0), ncol=len(arm_labels), frameon=True)
    fig.subplots_adjust(bottom=0.1, top=0.95)  

    plt.savefig(f'{save_path}/combined_UCBs_context.png')
    plt.close()


def plot_combined_action_distributions(time_based_data, non_time_based_data, save_path):
    sns.set(style="whitegrid")
    fig, axs = plt.subplots(2, 3, figsize=(18, 10), sharey=True)  # Removed sharex=True

    configurations = [(0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)]
    agent_map = {0: 'nor', 1: 'oner', 2: 'ircot'}
    arm_labels = ['+'.join([agent_map[idx] for idx, present in enumerate(config) if present]) for config in configurations]

    datasets = [non_time_based_data, time_based_data]
    titles = ['Non-Time-Based Action Choices', 'Time-Based Action Choices']
    max_timesteps = 0

    for data in datasets:
        action_choice_matrix, context_history = data  
        for context_index in range(action_choice_matrix.shape[0]):
            context_mask = [np.argmax(c) == context_index for c in context_history]
            max_timesteps = max(max_timesteps, sum(context_mask))
    
    for row_index, data in enumerate(datasets):
        action_choice_matrix, context_history = data  
        n_contexts = action_choice_matrix.shape[0]  
        n_arms = action_choice_matrix.shape[1]
        context_labels = ['A', 'B', 'C']

        for context_index in range(n_contexts):
            ax = axs[row_index, context_index]
            total_choices_per_epoch = np.sum(action_choice_matrix[context_index], axis=0) + 1e-9
            normalized_selections = action_choice_matrix[context_index] / total_choices_per_epoch

            for arm_index in range(n_arms):
                context_mask = [np.argmax(c) == context_index for c in context_history]
                context_specific_selections = normalized_selections[arm_index, context_mask]
                ax.plot(context_specific_selections, label=arm_labels[arm_index])

            ax.set_title(f'Context {context_labels[context_index]} ({titles[row_index]})')
            ax.set_xlim(0, max_timesteps - 1)
            if row_index == 1 and context_index == 1:
                ax.set_xlabel('Time Step')
            # ax.set_xlabel('Time Step')
            if context_index == 0:
                ax.set_ylabel('Probability of Action Choice')
            ax.set_ylim(0, 1)

    handles, labels = fig.gca().get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0), ncol=n_arms, frameon=True)
    fig.subplots_adjust(bottom=0.15, top=0.95)

    plt.savefig(f'{save_path}/combined_action_distributions.png')
    plt.close()


def evaluate_and_save_results(graph, linucb_model, test_data_path, results_path):
    test_data = load_data(test_data_path)
    
    complexities = defaultdict(list)

    start_time = time.time()
    
    for instance in test_data:
        complexity = instance['complexity_label']
        context = np.array(complexity_to_vector(complexity))
        chosen_arm = linucb_model.select_arm(context)
        # answer, time_taken, confidence, connected_agents_names = graph.execute_graph(chosen_arm, instance)
        answer, time_taken, connected_agents_names = graph.execute_graph(chosen_arm, instance)

        ground_truths = instance['gold_answers']
        
        metrics = evaluate_single(answer, ground_truths)
        f1_score = metrics['f1']
        
        complexities[complexity].append({
            "f1_score": f1_score,
            "time_taken": time_taken,
            "selected_action": chosen_arm,
            "connected_agents": connected_agents_names
        })
    
    total_evaluation_time = time.time() - start_time
    overall_results = []

    for complexity, results in complexities.items():
        avg_f1 = sum(item["f1_score"] for item in results) / len(results)
        avg_time = sum(item["time_taken"] for item in results) / len(results)
        overall_results.extend(results)  # Gather all results for overall metrics
        complexities[complexity] = results + [
            {"average_f1_score": avg_f1, "average_time": avg_time}
        ]

    if overall_results:
        overall_avg_f1 = sum(item["f1_score"] for item in overall_results) / len(overall_results)
        overall_avg_time = sum(item["time_taken"] for item in overall_results) / len(overall_results)
        complexities['Overall'] = overall_results + [
            {"average_f1_score": overall_avg_f1, "average_time": overall_avg_time}
        ]

    # Construct filename based on model type
    model_type = "time" if linucb_model.time_in_reward else "notime"
    results_filename = f"evaluation_results_{model_type}.json"

    results_to_save = {
        "total_evaluation_time": total_evaluation_time,
        "results_by_complexity": complexities
    }
    with open(os.path.join(results_path, results_filename), "w") as f:
        json.dump(results_to_save, f, indent=4)

    logging.info(f"Evaluation results saved to: {os.path.join(results_path, results_filename)}")




class LinUCB:
    def __init__(self, n_arms, n_features, alpha, time_in_reward):
        self.n_arms = n_arms
        self.n_features = n_features
        self.alpha = alpha
        self.A = [np.identity(n_features) for _ in range(n_arms)]
        # self.b = [np.zeros(n_features) for _ in range(n_arms)]
        self.b = [np.zeros((n_features, 1)) for _ in range(n_arms)]  
        self.time_in_reward = time_in_reward

        self.expected_reward_matrix_history = [] 
        self.ucb_matrix_history= []
        self.context_history = []
        self.chosen_arm_history = []
        self.real_reward_history = []

        self.action_choice_counts = np.zeros((3, n_arms, 0)) 



        logging.info(f"LinUCB initialized with {n_arms} arms and {n_features} features.")
        logging.info(f"Shapes: A: {self.A[0].shape}, b: {self.b[0].shape}")

    def select_arm(self, context):
        UCBs = []
        context = context.reshape(-1, 1)
        expected_rewards = []
        for arm in range(self.n_arms):
            theta = np.linalg.inv(self.A[arm]) @ self.b[arm]
            expected_reward = theta.T @ context
            expected_rewards.append(expected_reward)
            exploration_term = self.alpha * np.sqrt(context.T @ np.linalg.inv(self.A[arm]) @ context)
            UCB = expected_reward + exploration_term
            UCB = UCB[0][0]
            # UCB = theta.T @ context + self.alpha * np.sqrt(context.T @ np.linalg.inv(self.A[arm]) @ context)
            UCBs.append(UCB)
        
        self.expected_reward_matrix_history.append((context.flatten(), expected_rewards))
        self.ucb_matrix_history.append((context.flatten(), UCBs))
        self.context_history.append(context.flatten())
        max_UCB = max(UCBs)  

        indices_with_max_UCB = [i for i, value in enumerate(UCBs) if value == max_UCB] 
        # indices = [i for i in range(self.n_arms)]
        if len(indices_with_max_UCB) > 1:
            selected = np.random.choice(indices_with_max_UCB)
            # selected = indices[0]/
        else:
            selected = indices_with_max_UCB[0]
        # selected = np.argmax(UCBs)
        self.chosen_arm_history.append(selected)

        context_index = np.argmax(context)
        current_counts = self.action_choice_counts[:, :, -1] if self.action_choice_counts.shape[2] > 0 else np.zeros((3, self.n_arms))
        new_counts = current_counts.copy()
        new_counts[context_index, selected] += 1
        self.action_choice_counts = np.concatenate((self.action_choice_counts, new_counts[:, :, np.newaxis]), axis=2)

        return selected

    def update(self, chosen_arm, reward, context):
        context = context.reshape(-1, 1)
        self.A[chosen_arm] += context @ context.T
        self.b[chosen_arm] += reward * context
        self.real_reward_history.append((chosen_arm, reward, np.argmax(context)))  

    def get_train_matrices(self):
        n_time_steps = len(self.context_history)

        logging.info(f"\n")

        logging.info(f"Length of context_history: {len(self.context_history)}")
        logging.info(f"Length of real_reward_history: {len(self.real_reward_history)}")
        logging.info(f"Length of expected_reward_matrix_history: {len(self.expected_reward_matrix_history)}, sample: {self.expected_reward_matrix_history[0]}")
        logging.info(f"Length of ucb_matrix_history: {len(self.ucb_matrix_history)}, sample: {self.ucb_matrix_history[0]}")

        context_expected_rewards = np.zeros((self.n_features, self.n_arms, len(self.expected_reward_matrix_history))) # each row is a feature, each column is an arm, each depth is a training time step
        context_ucbs = np.zeros((self.n_features, self.n_arms, len(self.ucb_matrix_history))) # each row is a feature, each column is an arm, each depth is a training time step

        for index, (context, expected_rewards) in enumerate(self.expected_reward_matrix_history):
            context_index = np.argmax(context) 
            context_expected_rewards[context_index, :, index] = expected_rewards

        for index, (context, ucbs) in enumerate(self.ucb_matrix_history):
            context_index = np.argmax(context)
            context_ucbs[context_index, :, index] = ucbs
        return context_expected_rewards, context_ucbs, self.context_history, self.real_reward_history

    def run_train(self, graph):
        all_data = load_data(training_data_path)

        for epoch in range(EPOCHS):
            training_data = all_data
            # np.random.shuffle(training_data)
            random.shuffle(training_data)

            for i, instance in enumerate(training_data):
                context = np.array(complexity_to_vector(instance['complexity_label']))
                chosen_arm = self.select_arm(context)
                # answer, time_taken, confidence, connected_agents_names = graph.execute_graph(chosen_arm, instance)
                answer, time_taken, connected_agents_names = graph.execute_graph(chosen_arm, instance)

                ground_truths = instance['gold_answers']

                metrics = evaluate_single(answer, ground_truths)
                f1_score = metrics['f1']
                reward = calculate_reward(f1_score, time_taken, self.time_in_reward)
                self.update(chosen_arm, reward, context)


# def main():
# reset_random_seeds()

# data = load_data(training_data_path)
graph = DecisionGraph()


linucb_notime = LinUCB(graph.get_num_configurations(), n_features, ALPHA, time_in_reward=False)
linucb_notime.run_train(graph)

linucb_time = LinUCB(graph.get_num_configurations(), n_features, ALPHA, time_in_reward=True)
linucb_time.run_train(graph)

time_based_data = linucb_time.get_train_matrices()
logging.info(f" Length of time_based_data: {len(time_based_data)}")
non_time_based_data = linucb_notime.get_train_matrices()
logging.info(f" Length of non_time_based_data: {len(non_time_based_data)}")

# write the time_based_data and non_time_based_data to a file in CMAB_Figures folder


# for each element in time_based_data and non_time_based_data, write them to a file in CMAB_Figures folder

for i, data in enumerate(time_based_data):
    # idx 0 is the expected rewards matrix
    # idx 1 is the ucb matrix
    # idx 2 is the context history list
    # idx 3 is the real reward history list
    # based on these save them to a file with appropriate name and format
    if i == 0:
        filename = f"time_based_expected_rewards.npy"
        filepath = os.path.join(CMAB_Figures, filename)
        np.save(filepath, data)
    elif i == 1:
        filename = f"time_based_ucb_matrix.npy"
        filepath = os.path.join(CMAB_Figures, filename)
        np.save(filepath, data)
    elif i == 2:
        filename = f"time_based_context_history.npy"
        filepath = os.path.join(CMAB_Figures, filename)
        np.save(filepath, data)
    elif i == 3:
        filename = f"time_based_real_reward_history.npy"
        filepath = os.path.join(CMAB_Figures, filename)
        np.save(filepath, data)

logging.info(f"\n\n")
for i, data in enumerate(non_time_based_data):
    if i == 0:
        filename = f"non_time_based_expected_rewards.npy"
        filepath = os.path.join(CMAB_Figures, filename)
        np.save(filepath, data)
    elif i == 1:
        filename = f"non_time_based_ucb_matrix.npy"
        filepath = os.path.join(CMAB_Figures, filename)
        np.save(filepath, data)
    elif i == 2:
        filename = f"non_time_based_context_history.npy"
        filepath = os.path.join(CMAB_Figures, filename)
        np.save(filepath, data)
    elif i == 3:
        filename = f"non_time_based_real_reward_history.npy"
        filepath = os.path.join(CMAB_Figures, filename)
        np.save(filepath, data)


if __name__ == "__main__":
    try:
            
        path = CMAB_Figures
        if not os.path.exists(path):
            os.makedirs(path)

        plot_combined_contextual_rewards(time_based_data, non_time_based_data, path)
        plot_combined_contextual_UCSBs(time_based_data, non_time_based_data, path)
        plot_combined_action_distributions((linucb_time.action_choice_counts, linucb_time.context_history), (linucb_notime.action_choice_counts, linucb_notime.context_history), path)
        # if __name__ == "__main__":
        #     main()

        # def evaluate_and_save_results(graph, linucb_model, test_data_path, results_path):
        #     # Load test data
        #     test_data = load_data(test_data_path)
            
        #     # Initialize metrics collection
        #     f1_scores = []
        #     times = []

        #     start_time = time.time()
            
        #     # Execute the graph for each data point in the test set
        #     for instance in test_data:
        #         context = np.array(complexity_to_vector(instance['complexity_label']))
        #         chosen_arm = linucb_model.select_arm(context)
        #         answer, time_taken, confidence, connected_agents_names = graph.execute_graph(chosen_arm, instance)
        #         ground_truths = instance['gold_answers']
                
        #         # Calculate metrics
        #         metrics = evaluate_single(answer, ground_truths)
        #         f1_score = metrics['f1']
                
        #         f1_scores.append(f1_score)
        #         times.append(time_taken)
            
        #     # Compute the total evaluation time
        #     total_evaluation_time = time.time() - start_time

        #     # Construct filename based on model type
        #     model_type = "time" if linucb_model.time_in_reward else "notime"
        #     results_filename = f"evaluation_results_{model_type}.json"

        #     # Save the results
        #     results = {
        #         "f1_scores": f1_scores,
        #         "times": times,
        #         "total_evaluation_time": total_evaluation_time
        #     }
        #     with open(os.path.join(results_path, results_filename), "w") as f:
        #         json.dump(results, f, indent=4)

        #     logging.info(f"Evaluation results saved to: {os.path.join(results_path, results_filename)}")


        # def evaluate_and_save_results(graph, linucb_model, test_data_path, results_path):
        #     # Load test data
        #     test_data = load_data(test_data_path)
            
        #     # Prepare to collect detailed metrics
        #     complexities = defaultdict(list)

        #     start_time = time.time()
            
        #     # Evaluate each instance
        #     for instance in test_data:
        #         complexity = instance['complexity_label']
        #         context = np.array(complexity_to_vector(complexity))
        #         chosen_arm = linucb_model.select_arm(context)
        #         answer, time_taken, confidence, connected_agents_names = graph.execute_graph(chosen_arm, instance)
        #         ground_truths = instance['gold_answers']
                
        #         # Calculate metrics
        #         metrics = evaluate_single(answer, ground_truths)
        #         f1_score = metrics['f1']
                
        #         # Collect data by complexity
        #         complexities[complexity].append({
        #             "f1_score": f1_score,
        #             "time_taken": time_taken,
        #             "selected_action": chosen_arm,
        #             "connected_agents": connected_agents_names
        #         })
            
        #     # Compute overall metrics
        #     total_evaluation_time = time.time() - start_time
        #     overall_results = []

        #     # Organize results by complexity and calculate averages
        #     for complexity, results in complexities.items():
        #         avg_f1 = sum(item["f1_score"] for item in results) / len(results)
        #         overall_results.extend(results)  # Gather all results for overall metrics
        #         complexities[complexity] = results + [{"average_f1_score": avg_f1}]

        #     # Calculate overall average F1
        #     if overall_results:
        #         overall_avg_f1 = sum(item["f1_score"] for item in overall_results) / len(overall_results)
        #         complexities['Overall'] = overall_results + [{"average_f1_score": overall_avg_f1}]

        #     # Construct filename based on model type
        #     model_type = "time" if linucb_model.time_in_reward else "notime"
        #     results_filename = f"evaluation_results_{model_type}.json"

        #     # Save the results
        #     results_to_save = {
        #         "total_evaluation_time": total_evaluation_time,
        #         "results_by_complexity": complexities
        #     }
        #     with open(os.path.join(results_path, results_filename), "w") as f:
        #         json.dump(results_to_save, f, indent=4)

        #     logging.info(f"Evaluation results saved to: {os.path.join(results_path, results_filename)}")


        # Evaluate and save results for the non-time-based model
        evaluate_and_save_results(graph, linucb_notime, test_data_path, CMAB_Figures)
        logging.info("The results for non-time-based model have been saved.")
        # Evaluate and save results for the time-based model
        evaluate_and_save_results(graph, linucb_time, test_data_path, CMAB_Figures)
        logging.info("The results for time-based model have been saved.")
    except Exception as e:
        logging.exception(e, exc_info=True)
        raise e