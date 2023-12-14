import os 
import csv 
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import seaborn as sns

parser = argparse.ArgumentParser(description='Challenge video extraction')
parser.add_argument('--path_to_json_file', type=str,
        help='Path of the json file with the challenge metadata info.')
parser.add_argument('--path_to_stats_file', type=str,
        help='Path of csv file with the stats of the videos.')
parser.add_argument('--path_to_save_fig', type=str,
        help='Path where we save the figures with the stats.')
args = parser.parse_args()

def extract_and_compare(path_to_json_file, path_to_stats_file):
    
    keyword = path_to_json_file.split('/')[-1].split('.')[0].lower()
    
    if keyword == 'vot2018':
        challenges = ['camera_motion', 'illum_change', 'motion_change', 'size_change', 'occlusion']
    else:
        challenges = [keyword]
        
    with open(path_to_json_file, 'r') as f:
        meta_data = json.load(f)

    dict_df_stats_per_challenge = {}
    for challenge in challenges: 
        video_stats_challenge = {}
        for key in meta_data.keys():
            unique, counts = np.unique(meta_data[key][challenge], return_counts=True)
            try:
                positives = dict(zip(unique, counts))[1] # get positive frames (where challenge occurs)
            except:
                positives = 0 # If there is no key as 1 (positive) we catch the exception and assign 0
            finally:
                n_frames = len(meta_data[key]['img_names'])
                video_stats_challenge[key] = positives/n_frames
            
        df_ratio_challenge = pd.DataFrame(list(video_stats_challenge.items()), columns=['Videoname', 'ratio_of_challenge'])
        df_stats = pd.read_csv(path_to_stats_file)
        
        df_stats_challenge = df_stats.merge(df_ratio_challenge, on='Videoname', how='left')\
                                    .sort_values(by='ratio_of_challenge', ascending=False)\
                                    .reset_index(drop=True)
        
        dict_df_stats_per_challenge[challenge] = df_stats_challenge
        
    if len(challenges) == 1:
        return dict_df_stats_per_challenge[challenges[0]]
        
    return dict_df_stats_per_challenge

def plot_extracted_stats_challenge(df_stats, challenge):

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Plot for score1
    sns.scatterplot(x='ratio_of_challenge', y='Acc', data=df_stats, ax=axes[0], color='blue')
    axes[0].set_title(f"ratio vs Acc for challenge {challenge}")
    axes[0].grid()
    axes[0].set(ylim=(0, 1.2))
    axes[0].set(xlim=(0, 1.2))
    axes[0].set(ylabel="Accuracy")
    axes[0].set(xlabel="Ratio of challenge")
    

    # Plot for score2
    sns.scatterplot(x='ratio_of_challenge', y='LN', data=df_stats, ax=axes[1], color='red')
    axes[1].set_title(f"ratio vs LN for challenge {challenge}")
    axes[1].grid()
    axes[1].set(ylim=(0, 9))
    axes[1].set(xlim=(0, 1.2))
    axes[1].set(ylabel="LN")
    axes[1].set(xlabel="Ratio of challenge")

    plt.tight_layout()
    plt.savefig(os.path.join(args.path_to_save_fig, challenge))

if __name__ == '__main__':
    dict_df_stats_per_challenge = extract_and_compare(args.path_to_json_file, args.path_to_stats_file)
    
    if type(dict_df_stats_per_challenge) == dict:
        for k,v in dict_df_stats_per_challenge.items():
            plot_extracted_stats_challenge(v, k)
    else:
        challenge = args.path_to_json_file.split('/')[-1].split('.')[0].lower()
        plot_extracted_stats_challenge(dict_df_stats_per_challenge, challenge)
    