import os
import csv
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import seaborn as sns

parser = argparse.ArgumentParser(description='Challenge video extraction')
parser.add_argument('--path_to_dataset', type=str,
        help='Path to the dataset from where you want to extract the stats.')
args = parser.parse_args()

GENERAL_DATA_FILE = 'general_info.csv'

if __name__ == '__main__':
    
    challenge_list = os.listdir(args.path_to_dataset)

    # Inicializa un DataFrame vacío
    df_combined = pd.DataFrame()

    # Iterate for every general_info file in the challenges
    for challenge in challenge_list:
        file_path = os.path.join(args.path_to_dataset, challenge, GENERAL_DATA_FILE)
        df_temp = pd.read_csv(file_path)  # Reads into a temp df
        df_combined = pd.concat([df_combined, df_temp], ignore_index=True)
    df_combined['Challenge'] = challenge_list# add the challenge to the scores  

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 15))

    # Gráfico de Barras para Accuracy
    sns.barplot(x='Challenge', y='Accuracy', data=df_combined, hue=df_combined['Challenge'], ax=axes[0,0])
    axes[0,0].set_title('Challenges vs Accuracy')
    axes[0,0].set(ylabel="Accuracy")
    axes[0,0].set(xlabel="Ratio of challenge")

    # Gráfico de Barras para Recall
    sns.barplot(x='Challenge', y='LostNumber', data=df_combined, hue=df_combined['Challenge'], ax=axes[0,1])
    axes[0,1].set_title('Challenges vs Number of frames lost')
    axes[0,1].set(ylabel="LostNumber")
    axes[0,1].set(xlabel="Ratio of challenge")

    # Gráfico de Barras para F1-score
    sns.barplot(x='Challenge', y='Robustness', data=df_combined, hue=df_combined['Challenge'], ax=axes[1,0])
    axes[1,0].set_title('Challenges vs Robustness')
    axes[0,1].set(ylabel="Robustness")
    axes[0,1].set(xlabel="Ratio of challenge")

    sns.barplot(x='Challenge', y='EAO', data=df_combined, hue=df_combined['Challenge'], ax=axes[1,1])
    axes[1,1].set_title('Challenges vs EAO')
    axes[1,1].set(ylabel="EAO")
    axes[1,1].set(xlabel="Ratio of challenge")

    # Ajustes adicionales
    for ax in axes.flatten():
        ax.set_ylabel('Score')
        ax.set_xlabel('Challenge')

    plt.tight_layout()  # Ajusta automáticamente la disposición de los subgráficos
    plt.savefig(os.path.join(args.path_to_dataset, "general_stats.png"))
    
    


    