


import numpy as np
import json
import time
import os
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import random

plt.rcParams.update({'font.size': 16})  # General font size
plt.rcParams.update({'axes.titlesize': 18})  # Font size for axes titles
plt.rcParams.update({'axes.labelsize': 18})  # Font size for x and y labels
plt.rcParams.update({'xtick.labelsize': 14})  # Font size for x tick labels
plt.rcParams.update({'ytick.labelsize': 14})  # Font size for y tick labels
plt.rcParams.update({'legend.fontsize': 14})  # Font size for legend

    
DEBUG = True
random.seed(42)
np.random.seed(42)
n_arms = 3  
n_features = 3  
ALPHA = 2.5
EPOCHS = 20
PERFORMANCE_WEIGHT = 1
TIME_PENALTY = 0.001

# training_data_path = "/home/mhoveyda/AdaptiveQA/Adaptive-RAG/DATA_FOR_CMAB/Train_Data_For_CMAB.json"
training_data_path = "/home/mhoveyda/AdaptiveQA/Adaptive-RAG/DATA_FOR_CMAB/Train_Data_For_CMAB_augmented_with_confidence.json"

# Create the CMAB folder filename based on the parameters
# CMAB_figures_folder_filename = f"CMAB_figures_alpha_{ALPHA}_epochs_{EPOCHS}_reward_with_Performance_{PERFORMANCE_WEIGHT}_{TIME_PENALTY}"

# CMAB_Figures = f"/home/mhoveyda/AdaptiveQA/Adaptive-RAG/CMAB_FIGS/15-June/{CMAB_figures_folder_filename}"

path_to_evaluation_results_folder = {
    "NoR":"/home/mhoveyda/AdaptiveQA/Agents_Executed_Final_Evaluation_Results/output_results_nor.json"
    , "OneR":"/home/mhoveyda/AdaptiveQA/Agents_Executed_Final_Evaluation_Results/output_results_oner.json"
    , "IRCoT":"/home/mhoveyda/AdaptiveQA/Agents_Executed_Final_Evaluation_Results/output_results_ircot.json"

}


# if not os.path.exists(CMAB_Figures):
#     os.makedirs(CMAB_Figures)

def load_data(filepath):
    data = []
    with open(filepath, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    random.shuffle(data)
    return data

def complexity_to_vector(label):
    return {'A': [1, 0, 0], 'B': [0, 1, 0], 'C': [0, 0, 1]}[label]

def calculate_reward(performance_score, time_taken, time_in_reward):

    time_penalty = TIME_PENALTY if time_taken >= 1 else 0
    return PERFORMANCE_WEIGHT * performance_score - (time_penalty * time_taken if time_in_reward else 0)

# 
def plot_contextual_rewards_over_time(expected_reward_matrix, context_history, rewards_history, combined_plot=True):
    sns.set(style="whitegrid")  # Set the seaborn style
    
    n_contexts = expected_reward_matrix.shape[0]
    n_arms = expected_reward_matrix.shape[1]
    context_labels = ['A', 'B', 'C']  
    arm_labels = {0: 'NoR', 1: 'OneR', 2: 'IRCoT'} 

    # Define a pleasing and consistent color palette
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green

    average_real_rewards = np.zeros((n_contexts, n_arms))

    # Calculate average real rewards
    for context_index in range(n_contexts):
        for arm_index in range(n_arms):
            filtered_rewards = [reward for arm, reward, context in rewards_history if arm == arm_index and context == context_index]
            if filtered_rewards:
                average_real_rewards[context_index, arm_index] = np.mean(filtered_rewards)

    if combined_plot:
        # Create a single figure with subplots
        fig, axs = plt.subplots(1, n_contexts, figsize=(27, 8), sharey=True)

        # Plot expected and average real rewards
        for context_index in range(n_contexts):
            context_mask = [np.argmax(c) == context_index for c in context_history]
            ax = axs[context_index]

            for arm_index in range(n_arms):
                context_specific_rewards = expected_reward_matrix[context_index, arm_index, context_mask]
                line_color = colors[arm_index]
                ax.plot(context_specific_rewards, label=f'{arm_labels[arm_index]} - Estimated', color=line_color)
                ax.axhline(y=average_real_rewards[context_index, arm_index], color=line_color, linestyle='--', label=f'{arm_labels[arm_index]} - Real')

            ax.set_title(f'Context {context_labels[context_index]}')
            if context_index == (n_contexts // 2):  # Center x-label in the middle subplot for combined mode
                ax.set_xlabel('Relevant Time Step')

        axs[0].set_ylabel('Expected Reward')  # Set y-label for the first subplot

        # Handling legend outside the plot area
        handles, labels = axs[0].get_legend_handles_labels()  # Collecting handles and labels for legend
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0), ncol=3, frameon=True)
        fig.subplots_adjust(bottom=0.2)  # Adjust subplot to make room for the legend

        plt.savefig(f'{CMAB_Figures}/combined_rewards_context.png')
        plt.close()

    else:
        # Plot expected and average real rewards for separate figures
        for context_index in range(n_contexts):
            plt.figure(figsize=(27, 8))
            ax = plt.gca()
            context_mask = [np.argmax(c) == context_index for c in context_history]

            for arm_index in range(n_arms):
                context_specific_rewards = expected_reward_matrix[context_index, arm_index, context_mask]
                line_color = colors[arm_index]
                ax.plot(context_specific_rewards, label=f'{arm_labels[arm_index]} - Estimated', color=line_color)
                ax.axhline(y=average_real_rewards[context_index, arm_index], color=line_color, linestyle='--', label=f'{arm_labels[arm_index]} - Real Avg')

            ax.set_title(f'Context {context_labels[context_index]}')
            ax.set_xlabel('Relevant Time Step')
            ax.set_ylabel('Expected Reward')
            ax.legend()
            plt.savefig(f'{CMAB_Figures}/rewards_context_{context_labels[context_index]}.png')
            plt.close()


def plot_contextual_UCBs_over_time(ucbs_matrix, context_history, combined_plot=True):
    sns.set(style="whitegrid")  # Apply seaborn style for better aesthetics
    n_contexts = ucbs_matrix.shape[0]
    n_arms = ucbs_matrix.shape[1]
    context_labels = ['A', 'B', 'C']
    arm_labels = {0: 'NoR', 1: 'OneR', 2: 'IRCoT'}
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Consistent color palette

    if combined_plot:
        fig, axs = plt.subplots(1, n_contexts, figsize=(27, 8), sharey=True)
    
    for context_index in range(n_contexts):
        context_mask = [np.argmax(c) == context_index for c in context_history]
        
        if combined_plot:
            ax = axs[context_index]
        else:
            plt.figure(figsize=(8, 5))
            ax = plt.gca()

        for arm_index in range(n_arms):
            context_specific_ucbs = ucbs_matrix[context_index, arm_index, context_mask]
            line_color = colors[arm_index]
            ax.plot(context_specific_ucbs, label=f'{arm_labels[arm_index]}', color=line_color)
        
        ax.set_title(f'Context {context_labels[context_index]}')
        if combined_plot:
            if context_index == (n_contexts // 2):  # Center x-label in the middle subplot for combined mode
                ax.set_xlabel('Relevant Time Step')
        else:
            ax.set_xlabel('Relevant Time Step')
            ax.set_ylabel('Upper Confidence Bound (UCB)')
            ax.legend()
            plt.savefig(f'{CMAB_Figures}/UCBs_context_{context_labels[context_index]}.png')
            plt.close()

    if combined_plot:
        handles, labels = axs[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0), ncol=3, frameon=True)
        axs[0].set_ylabel('Upper Confidence Bound (UCB)')  # Set y-label for the first subplot
        fig.subplots_adjust(bottom=0.2)  # Adjust subplot to make room for the legend
        plt.savefig(f'{CMAB_Figures}/combined_UCBs_context.png')
        plt.close()



def plot_combined_contextual_rewards(time_based_data, non_time_based_data, save_path):
    sns.set(style="whitegrid")  # Set the seaborn style
    fig, axs = plt.subplots(2, 3, figsize=(18, 10), sharey=True)  # 2 rows, 3 contexts

    datasets = [non_time_based_data, time_based_data]
    titles = ['Non-Time-Based Rewards', 'Time-Based Rewards']
    
    max_timesteps = 0  # Variable to store the maximum number of timesteps across all plots

    # First, calculate the maximum number of timesteps needed for setting uniform x-axes
    for data in datasets:
        expected_reward_matrix, context_history, _ = data
        for context_index in range(expected_reward_matrix.shape[0]):
            context_mask = [np.argmax(c) == context_index for c in context_history]
            max_timesteps = max(max_timesteps, sum(context_mask))
            print(f"In plot_combined_contextual_rewards: {max_timesteps=}")

    for row_index, data in enumerate(datasets):
        expected_reward_matrix, context_history, rewards_history = data
        n_contexts = expected_reward_matrix.shape[0]
        print(f"for rewards n_contexts: {n_contexts}")
        n_arms = expected_reward_matrix.shape[1]
        context_labels = ['A', 'B', 'C']
        arm_labels = {0: 'NoR', 1: 'OneR', 2: 'IRCoT'}
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green

        for context_index in range(n_contexts):
            ax = axs[row_index, context_index]
            for arm_index in range(n_arms):
                context_mask = [np.argmax(c) == context_index for c in context_history]
                context_specific_rewards = expected_reward_matrix[context_index, arm_index, context_mask]
                line_color = colors[arm_index]
                ax.plot(context_specific_rewards, label=f'{arm_labels[arm_index]} - Estimated', color=line_color)
                ax.axhline(y=np.mean([reward for arm, reward, context in rewards_history if arm == arm_index and context == context_index]), color=line_color, linestyle='--', label=f'{arm_labels[arm_index]} - Real Avg')

            ax.set_xlim(0, max_timesteps - 1)  # Set uniform x-axis limits
            ax.set_title(f'Context {context_labels[context_index]} ({titles[row_index]})')
            if row_index == 1 and context_index == 1:
                ax.set_xlabel('Relevant Time Step')

        if row_index in [0, 1]:
            axs[row_index, 0].set_ylabel('Expected Reward')

    # Legend handling outside the plot area for clarity
    handles, labels = fig.gca().get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0), ncol=3, frameon=True)
    fig.subplots_adjust(bottom=0.15, top=0.95)

    plt.savefig(f'{save_path}/combined_rewards_context.png')
    plt.close()


def plot_combined_contextual_UCBs(time_based_data, non_time_based_data, save_path):
    sns.set(style="whitegrid")  # Set the seaborn style
    fig, axs = plt.subplots(2, 3, figsize=(18, 10), sharey=True)  # 2 rows, 3 contexts, smaller figsize for more compact display

    datasets = [non_time_based_data, time_based_data]
    titles = ['Non-Time-Based UCBs', 'Time-Based UCBs']
    
    max_timesteps = 0  # Variable to store the maximum number of timesteps across all plots

    # Calculate the maximum number of timesteps needed for setting uniform x-axes
    for data in datasets:
        ucbs_matrix, context_history= data
        for context_index in range(ucbs_matrix.shape[0]):
            context_mask = [np.argmax(c) == context_index for c in context_history]
            max_timesteps = max(max_timesteps, sum(context_mask))

    for row_index, data in enumerate(datasets):
        ucbs_matrix, context_history= data
        n_contexts = ucbs_matrix.shape[0]
        n_arms = ucbs_matrix.shape[1]
        context_labels = ['A', 'B', 'C']
        arm_labels = {0: 'NoR', 1: 'OneR', 2: 'IRCoT'}
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Consistent color scheme

        for context_index in range(n_contexts):
            ax = axs[row_index, context_index]
            for arm_index in range(n_arms):
                context_mask = [np.argmax(c) == context_index for c in context_history]
                context_specific_ucbs = ucbs_matrix[context_index, arm_index, context_mask]
                line_color = colors[arm_index]
                ax.plot(context_specific_ucbs, label=f'{arm_labels[arm_index]}', color=line_color)

            ax.set_xlim(0, max_timesteps - 1)  # Set uniform x-axis limits
            ax.set_title(f'Context {context_labels[context_index]} ({titles[row_index]})')
            if row_index == 1 and context_index == 1:
                ax.set_xlabel('Relevant Time Step')

        if row_index in [0, 1]:
            axs[row_index, 0].set_ylabel('UCB Score')

    # Handling the legend
    handles, labels = axs[0, 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0), ncol=3, frameon=True)
    fig.subplots_adjust(bottom=0.1, top=0.95)

    plt.savefig(f'{save_path}/combined_UCBs_context.png')
    plt.close()

# def plot_combined_action_distributions(time_based_data, non_time_based_data, save_path):
#     sns.set(style="whitegrid")  # Set the seaborn style
#     fig, axs = plt.subplots(2, 3, figsize=(18, 10), sharey=True, sharex=True)  # 2 rows, 3 contexts

#     datasets = [non_time_based_data, time_based_data]
#     titles = ['Non-Time-Based Action Choices', 'Time-Based Action Choices']
    
#     for row_index, data in enumerate(datasets):
#         action_choice_matrix, context_history = data  
#         n_contexts = action_choice_matrix.shape[0]
#         print(f"for action choice n_contexts: {n_contexts}")
#         n_arms = action_choice_matrix.shape[1]
#         context_labels = ['A', 'B', 'C']
#         arm_labels = {0: 'NoR', 1: 'OneR', 2: 'IRCoT'}
#         colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green

#         for context_index in range(n_contexts):
#             ax = axs[row_index, context_index]
#             total_choices_per_epoch = np.sum(action_choice_matrix[context_index], axis=0) + 1e-9  # avoid division by zero
#             normalized_selections = action_choice_matrix[context_index] / total_choices_per_epoch
            
#             for arm_index in range(n_arms):
#                 ax.plot(normalized_selections[arm_index], label=f'{arm_labels[arm_index]}', color=colors[arm_index])
            
#             ax.set_title(f'Context {context_labels[context_index]} ({titles[row_index]})')
#             # ax.set_xlabel('Training Epochs')
#             if row_index == 1 and context_index == 1:
#                 ax.set_xlabel('Training Epochs')
#             if row_index in [0, 1] and context_index == 0:
#                 ax.set_ylabel('Probability of Action Choice')
#             ax.set_ylim(0, 1)  # Since it's a probability

#     # Handling the legend for clarity
#     handles, labels = fig.gca().get_legend_handles_labels()
#     fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0), ncol=3, frameon=True)
#     fig.subplots_adjust(bottom=0.1, top=0.95)

#     plt.savefig(f'{save_path}/combined_action_distributions.png')
#     plt.close()




# def plot_combined_action_distributions(time_based_data, non_time_based_data, save_path):
#     sns.set(style="whitegrid")  # Set the seaborn style
#     fig, axs = plt.subplots(2, 3, figsize=(18, 10), sharey=True, sharex=True)  # 2 rows, 3 contexts

#     datasets = [non_time_based_data, time_based_data]
#     titles = ['Non-Time-Based Action Choices', 'Time-Based Action Choices']

#     # Calculate the maximum number of timesteps for uniform x-axes across all plots
#     max_timesteps = 0
#     for data in datasets:
#         action_choice_matrix, context_history = data
#         for context_index in range(action_choice_matrix.shape[0]):
#             context_mask = [np.argmax(c) == context_index for c in context_history]
#             max_timesteps = max(max_timesteps, sum(context_mask))

#     for row_index, data in enumerate(datasets):
#         action_choice_matrix, context_history = data  
#         n_contexts = action_choice_matrix.shape[0]
#         n_arms = action_choice_matrix.shape[1]
#         context_labels = ['A', 'B', 'C']
#         arm_labels = {0: 'NoR', 1: 'OneR', 2: 'IRCoT'}
#         colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green

#         for context_index in range(n_contexts):
#             ax = axs[row_index, context_index]
#             total_choices_per_epoch = np.sum(action_choice_matrix[context_index], axis=0) + 1e-9  # avoid division by zero
#             normalized_selections = action_choice_matrix[context_index] / total_choices_per_epoch
            
#             for arm_index in range(n_arms):
#                 ax.plot(normalized_selections[arm_index, :max_timesteps], label=f'{arm_labels[arm_index]}', color=colors[arm_index])  # Adjust array slicing to max_timesteps
            
#             ax.set_title(f'Context {context_labels[context_index]} ({titles[row_index]})')
#             # ax.set_xlabel('Relevant Time Steps')
#             if row_index == 1 and context_index == 1:
#                 ax.set_xlabel('Relevant Time Steps')
#             # ax.set_ylabel('Probability of Action Choice')
#             if row_index in [0, 1] and context_index == 0:
#                 ax.set_ylabel('Probability of Action Choice')
#             ax.set_xlim(0, max_timesteps - 1)  # Set uniform x-axis limits
#             ax.set_ylim(0, 1)  # Since it's a probability

#     # Handling the legend for clarity
#     handles, labels = fig.gca().get_legend_handles_labels()
#     fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0), ncol=3, frameon=True)
#     fig.subplots_adjust(bottom=0.1, top=0.95)

#     plt.savefig(f'{save_path}/combined_action_distributions.png')
#     plt.close()



def plot_combined_action_distributions(time_based_data, non_time_based_data, save_path):
    sns.set(style="whitegrid")  # Set the seaborn style
    fig, axs = plt.subplots(2, 3, figsize=(18, 10), sharey=True, sharex=True)  # 2 rows, 3 contexts

    datasets = [non_time_based_data, time_based_data]
    titles = ['Non-Time-Based Action Choices', 'Time-Based Action Choices']

    # Calculate the maximum number of timesteps for uniform x-axes across all plots
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
        arm_labels = {0: 'NoR', 1: 'OneR', 2: 'IRCoT'}
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green

        for context_index in range(n_contexts):
            ax = axs[row_index, context_index]
            total_choices_per_epoch = np.sum(action_choice_matrix[context_index], axis=0) + 1e-9  # avoid division by zero
            normalized_selections = action_choice_matrix[context_index] / total_choices_per_epoch
            
            for arm_index in range(n_arms):
                ax.plot(normalized_selections[arm_index, :max_timesteps], label=f'{arm_labels[arm_index]}', color=colors[arm_index])  # Adjust array slicing to max_timesteps
            
            ax.set_title(f'Context {context_labels[context_index]} ({titles[row_index]})')
            # ax.set_xlabel('Relevant Time Steps')
            if row_index == 1 and context_index == 1:
                ax.set_xlabel('Relevant Time Step')
            # ax.set_ylabel('Probability of Action Choice')
            if row_index in [0, 1] and context_index == 0:
                ax.set_ylabel('Probability of Action Choice')
            ax.set_xlim(0, max_timesteps - 1)  # Set uniform x-axis limits
            ax.set_ylim(0, 1)  # Since it's a probability

    # Ensure x-axis labels are visible on all subplots
    for ax in axs.flat:
        ax.tick_params(axis='x', labelbottom=True)

    # Handling the legend for clarity
    handles, labels = fig.gca().get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0), ncol=3, frameon=True)
    fig.subplots_adjust(bottom=0.1, top=0.95)

    plt.savefig(f'{save_path}/combined_action_distributions.png')
    plt.close()



def plot_all():
    expected_rewards_matrix, ucb_matrix, context_history, real_reward_history = linucb.get_train_matrices()
    plot_contextual_rewards_over_time(expected_rewards_matrix, context_history, real_reward_history)
    plot_contextual_UCBs_over_time(ucb_matrix, context_history)



class LinUCB:
    def __init__(self, n_arms, n_features, alpha, time_in_reward):
        self.n_arms = n_arms
        self.n_features = n_features
        self.alpha = alpha
        self.A = [np.identity(n_features) for _ in range(n_arms)] 
        self.b = [np.zeros((n_features, 1)) for _ in range(n_arms)] 
        self.time_in_reward = time_in_reward


        self.expected_reward_matrix_history = [] 
        self.ucb_matrix_history = []
        self.context_history = []
        self.chosen_arm_history = []
        self.real_reward_history = []

        self.action_choice_counts = np.zeros((3, n_arms, 0)) 

       

    def select_arm(self, context):

        UCBs = []
        context = context.reshape(-1, 1)
        expected_rewards = []
        for arm in range(self.n_arms):
            theta_arm = np.linalg.inv(self.A[arm]) @ self.b[arm] 


            expected_reward_for_arm = theta_arm.T @ context
            expected_rewards.append(expected_reward_for_arm)
            exploration_term = self.alpha * np.sqrt(context.T @ np.linalg.inv(self.A[arm]) @ context)
            UCB = expected_reward_for_arm + exploration_term
            # UCB = theta.T @ context + self.alpha * np.sqrt(context.T @ np.linalg.inv(self.A[arm]) @ context)
            UCB = UCB[0][0]
            UCBs.append(UCB)
            if DEBUG:
                print(f"\t ---- arm: {arm}--")
                print(f"\tTheta : {theta_arm}")
                print(f"\tExpected Reward: {expected_reward_for_arm}")
                print(f"\tExploration term: {exploration_term}")
                print(f"\tUCB: {UCB}")

        print(f"\nUCBs: {UCBs}")
        self.expected_reward_matrix_history.append((context.flatten(), expected_rewards))
        self.ucb_matrix_history.append((context.flatten(), UCBs))
        self.context_history.append(context.flatten())  
        

        max_UCB = max(UCBs)  
        if DEBUG:
            print(f"\tMax UCB: {max_UCB}")
        indices_with_max_UCB = [i for i, value in enumerate(UCBs) if value == max_UCB] 
        if len(indices_with_max_UCB) > 1:
            if DEBUG:
                print(f"\tTie detected")
            selected = np.random.choice(indices_with_max_UCB)
        else:
            selected = indices_with_max_UCB[0]
        self.chosen_arm_history.append(selected)
        if DEBUG:
            print(f"Selected arm: {selected}")

                # Update action choice counts
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
        self.real_reward_history.append((chosen_arm, reward, np.argmax(context)))  # Record arm, reward, and context

        if DEBUG:
            print(f"\nReal Reward: {reward}")
            for i in range(self.n_arms):
                print(f"Arm {i}:")
                print(f"A: \n{self.A[i]}")
                print(f"b: \n{self.b[i]}")
                print(f"\n")

    def get_train_matrices(self):
        n_time_steps = len(self.context_history)
        context_expected_rewards = np.zeros((self.n_features, self.n_arms, len(self.expected_reward_matrix_history)))
        context_ucbs = np.zeros((self.n_features, self.n_arms, len(self.ucb_matrix_history)))
        for index, (context, probabilities) in enumerate(self.expected_reward_matrix_history):
            context_index = np.argmax(context)  # Since context is one-hot, find which context it represents
            context_expected_rewards[context_index, :, index] = probabilities
        for index, (context, ucb_values) in enumerate(self.ucb_matrix_history):
            context_index = np.argmax(context)
            context_ucbs[context_index, :, index] = ucb_values
        
        return context_expected_rewards, context_ucbs, self.context_history, self.real_reward_history
    
    def run_train(self):
        all_data = load_data(training_data_path)
        for epoch in range(EPOCHS):
            training_data = all_data.copy()
            random.shuffle(training_data)
            if DEBUG:
                print(f"epoch {epoch+1}/{EPOCHS}")
            total_reward = 0
            for i, instance in enumerate(training_data):
                context = np.array(complexity_to_vector(instance['complexity_label']))
                chosen_arm = self.select_arm(context)
                model_key = {0: 'NoR', 1: 'OneR', 2: 'IRCoT'}[chosen_arm]
                execution_time = instance[f'{model_key}_total_run_time_in_seconds']
                f1_score = instance[f'{model_key}_evaluation_results']['f1']
                actual_reward = calculate_reward(f1_score, execution_time, self.time_in_reward)
                self.update(chosen_arm, actual_reward, context)
                total_reward += actual_reward
            if DEBUG:
                print(f"\n Epoch {epoch+1}/{EPOCHS} completed with total reward: {total_reward:.2f}.\n")


# all_data = load_data(training_data_path)  

# linucb = LinUCB(n_arms, n_features, ALPHA)
# linucb.run_train()

# Create LinUCB instances for both scenarios
linucb_time = LinUCB(n_arms, n_features, ALPHA, True)
print("LinUCB with time")
linucb_notime = LinUCB(n_arms, n_features, ALPHA, False)
print("LinUCB without time")

# Train both models
linucb_time.run_train()
print("LinUCB with time trained!")

linucb_notime.run_train()
print("LinUCB without time trained!")

# plot_all()


# print(f"Plots all saved in {CMAB_Figures}")
                



# Path for saving the figures
save_path = "/home/mhoveyda/AdaptiveQA/Adaptive-RAG/CMAB_FIGS_Finally/10-July/combined_figures"
if not os.path.exists(save_path):
    os.makedirs(save_path)

context_expected_rewards_time, context_ucbs_time, context_history_time, real_reward_history_time =  linucb_time.get_train_matrices()

context_expected_rewards_notime, context_ucbs_notime, context_history_notime, real_reward_history_notime =  linucb_notime.get_train_matrices()

plot_combined_contextual_rewards((context_expected_rewards_time, context_history_time, real_reward_history_time), (context_expected_rewards_notime, context_history_notime, real_reward_history_notime), save_path)

plot_combined_contextual_UCBs((context_ucbs_time, context_history_time), (context_ucbs_notime, context_history_notime), save_path)

plot_combined_action_distributions((linucb_time.action_choice_counts, context_history_time), (linucb_notime.action_choice_counts, context_history_notime), save_path)
# Plotting combined results
# plot_all(linucb_time, linucb_notime, save_path)



