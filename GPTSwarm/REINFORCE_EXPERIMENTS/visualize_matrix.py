

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def parse_graph_data(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line[3:]  
            parts = line.split(', ')
            src = parts[0].split('=')[1].split('(')[0]  # Strip out ID and keep only the name
            dst = parts[1].split('=')[1].split('(')[0]  # Strip out ID and keep only the name
            prob = float(parts[2].split('=')[1])
            data.append((src, dst, prob))
    return data

def create_dataframe(data):
    columns = ['source', 'destination', 'probability']
    return pd.DataFrame(data, columns=columns)

def create_matrix(df):
    # Define the specific node order
    order = ['NoRAnswer', 'OneRAnswer', 'IRCoTAnswer', 'FinalDecision']
    # Prepare nodes list with the desired order for columns and exclude 'FinalDecision' for rows
    nodes = sorted(set(df['source']).union(df['destination']), key=lambda x: order.index(x))
    nodes_src = [node for node in nodes if node != 'FinalDecision']
    node_index = {node: i for i, node in enumerate(nodes)}

    # Create an empty matrix
    matrix = np.zeros((len(nodes_src), len(nodes)))

    # Populate the matrix with probabilities
    for _, row in df.iterrows():
        if row['source'] in node_index and row['destination'] in node_index:
            i = node_index[row['source']]
            j = node_index[row['destination']]
            matrix[i, j] = row['probability']

    return pd.DataFrame(matrix, index=nodes_src, columns=nodes)

def visualize_heatmap(matrix):
    plt.figure(figsize=(4, 3.5))  # Adjusted from (6, 5) to make everything smaller

    # Create the heatmap with smaller annotations
    sns.heatmap(matrix, annot=True, cmap="crest", fmt=".2f", annot_kws={"size": 8})

    # Adjust the size of tick labels
    plt.xticks(rotation=45, ha="right", fontsize=8)  # Smaller font size for x-axis labels
    plt.yticks(rotation=0, fontsize=8)  # Smaller font size for y-axis labels

    plt.tight_layout()  
    # plt.savefig('heatmap_before_bkk.png')
    plt.savefig('heatmap_after_bkk.png')

    plt.show()

def main(file_path):
    data = parse_graph_data(file_path)
    df = create_dataframe(data)
    matrix = create_matrix(df)
    visualize_heatmap(matrix)

if __name__ == "__main__":
    # file_path = 'before copy.txt'  
    file_path = 'after copy.txt'  

    main(file_path)


