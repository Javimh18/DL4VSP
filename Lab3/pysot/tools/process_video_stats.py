import os
import csv
import numpy as np
import pandas as pd
import argparse


parser = argparse.ArgumentParser(description='Challenge video extraction')
parser.add_argument('--path_to_file', type=str,
        help='File with the information of the eval.py output.')
parser.add_argument('--path_to_save', type=str,
        help='File where we are going to save the csv.')
args = parser.parse_args()

def process_stats(path_to_file, path_to_save):
    with open(path_to_file, "r") as fd:
        stream = fd.read()
        stream_splits = stream.split('\n')
        
        info_splits = []
        for spl in stream_splits:
            if spl == '' or '-' in spl:
                pass
            else:
                spl = spl.replace('|', ',')
                spl = spl[1:-1] # to avoid the first and last ',' as we replaced the '|'
                spl = spl.replace(" ", "")
                info_splits.append(spl)
                
        VOT_general_info = info_splits[:2]
        VOT_specific_info = info_splits[3:]
    
    # dumping the general information
    with open(path_to_save + 'general_info.csv', "a") as fd:
        for s in VOT_general_info:
            fd.write(s+'\n')
            
    # dumping the specific information
    with open(path_to_save + 'specific_info.csv', "a") as fd:
        for s in VOT_specific_info:
            fd.write(s+'\n')

if __name__ == '__main__':
    process_stats(args.path_to_file, args.path_to_save)