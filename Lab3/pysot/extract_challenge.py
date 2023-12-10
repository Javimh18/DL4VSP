import json
import os
import argparse
from tqdm import tqdm
import numpy as np
import shutil

parser = argparse.ArgumentParser(description='Challenge video extraction')
parser.add_argument('--json_file', type=str,
        help='JSON file with metadata information')
parser.add_argument('--dataset_path', default='', type=str,
        help='path to dataset where the JSON file is')
parser.add_argument('--dest_path', default='', type=str,
        help='destiny path for the challenge dataset ')
args = parser.parse_args()

VIDS_PER_CHALLENGE = 5

challenges = ['camera_motion', 'illum_change', 'motion_change', 'size_change', 'occlusion']

if __name__=='__main__':

    # loading the metadata file
    with open(os.path.join(args.dataset_path, args.json_file), 'r') as f:
        meta_data = json.load(f)

    # iterating over the videos to extract the stats for each challenge
    challenge_scores = {}
    for key in meta_data.keys():
        video_stats_challenge = {}
        # get number of frames for video
        n_frames = len(meta_data[key]['img_names'])
        for challenge in challenges: 
        # count in which frame for each video a "challenge" is present and the 
        # obtain the ratio w.r.t the number of all frames
            unique, counts = np.unique(meta_data[key][challenge], return_counts=True)
            try:
                positives = dict(zip(unique, counts))[1] # get positive frames (where challenge occurs)
            except:
                positives = 0 # If there is no key as 1 (positive) we catch the exception and assign 0
            finally:
                video_stats_challenge[challenge] = positives/n_frames

        challenge_scores[key] = video_stats_challenge

    # now we extract, for each challenge, the keys with greater scores
    challenge_keys = {}
    for key in meta_data.keys():
        for c in challenges:
            score_key_challenge = challenge_scores[key][c]
            # if the challenge is not in the dictionary we initialize the entry
            if c not in challenge_keys.keys():
                challenge_keys[c] = {key:score_key_challenge}
            else:
                if len(challenge_keys[c]) < VIDS_PER_CHALLENGE:
                    challenge_keys[c][key] = score_key_challenge
                else:
                    min_value = np.inf
                    min_key = ""
                    for k,v in challenge_keys[c].items():
                        if min_value > v:
                            min_key = k
                            min_value = v
                    # if the min_value for a video in a challenge has less score 
                    # than the video we are processing, then we delete the video 
                    # and insert the new video
                    if min_value < score_key_challenge:
                        _ = challenge_keys[c].pop(min_key)
                        challenge_keys[c][key] = score_key_challenge
    
    if os.path.exists(os.path.join(args.dest_path)):
        print(f"The directory already exists, please delete the {args.dest_path} folder and run this script again.")
        exit 
        
    # now we create the challenges sub-datasets                
    for c in challenges:
        if not os.path.exists(os.path.join(args.dest_path, c.upper())):
            os.makedirs(os.path.join(args.dest_path, c.upper()))
            
        keys_for_challenge = challenge_keys[c].keys()
        
        # loading the data into a dictionary for json dumping plus
        # moving the videos to their corresponding challenge folder
        challenge_metadata = {}
        for key in keys_for_challenge:
            challenge_metadata[key] = meta_data[key]
            shutil.copytree(os.path.join(args.dataset_path, key), os.path.join(args.dest_path, c.upper(), key))
            
        with open(os.path.join(args.dest_path, c.upper(), c.upper()+".json"), "w") as challenge_json_file:
            json.dump(challenge_metadata, challenge_json_file, indent=4)    
        