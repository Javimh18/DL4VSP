import os
import csv
import numpy as np
import pandas as pd
import argparse

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
    
    print(df_combined)
    


    