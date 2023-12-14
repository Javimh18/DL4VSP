import os
import csv
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt

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
    df_combined['challenge'] = challenge_list# add the challenge to the scores  
    
    plt.figure(figsize=(10, 10))
    
    plt.subplot(2, 2, 1)
    plt.bar(df_combined['challenge'], df_combined['Accuracy'], label='Challenge vs Accuracy')
    plt.xlabel('Challenge')
    plt.ylabel('Accuracy')
    plt.title('Challenge vs Accuracy')
    plt.grid()
    plt.xlim(0, 1.2)
    plt.legend()
    
    plt.subplot(2, 2, 2)
    plt.bar(df_combined['challenge'], df_combined['LostNumber'], label='Challenge vs LostNumber')
    plt.xlabel('Challenge')
    plt.ylabel('LostNumber')
    plt.title(f'Challenge vs LostNumber')
    plt.xlim(0, 1.2)
    plt.grid()
    plt.legend()
    
    plt.subplot(2, 2, 3)
    plt.bar(df_combined['challenge'], df_combined['Robustness'], label='Challenge vs Robustness')
    plt.xlabel('Challenge')
    plt.ylabel('Robustness')
    plt.title(f'Challenge vs Robustness')
    plt.grid()
    plt.xlim(0, 1.2)
    plt.legend()
    
    plt.subplot(2, 2, 3)
    plt.bar(df_combined['challenge'], df_combined['EOA'], label='Challenge vs EOA')
    plt.xlabel('Challenge')
    plt.ylabel('EOA')
    plt.title(f'Challenge vs EOA')
    plt.grid()
    plt.xlim(0, 1.2)
    plt.legend()

    plt.tight_layout()

    plt.show()
    


    