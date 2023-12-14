#!/bin/bash

#cutoff_challenges=(5 10 25 55)
#challenges=("CAMERA_MOTION" "ILLUM_CHANGE" "MOTION_CHANGE" "OCCLUSION" "SIZE_CHANGE")

cutoff_challenge=5
challenge="CAMERA_MOTION"

# plot metrics for the top CUTOFF videos for each challenge
plot_stats_challenge() {
    python tools/plot_stats_challenge.py --path_to_json_file "./testing_dataset/top${cutoff_challenge}_vids_challenge/$1/$1.json" \
                                        --path_to_stats_file ./testing_dataset/top${cutoff_challenge}_vids_challenge/${cutoff_challenge}/specific_info.csv
}

###############################################################################################################################################################################

# extract top cutoff and challenges and create subsets
python tools/extract_challenge.py --dataset_path testing_dataset/VOT2018 --json_file VOT2018.json --dest_path "./testing_dataset/top${cutoff_challenge}_vids_challenge" \
                                  --cutoff_vids_per_challenge $cutoff_challenge

# test for an specific subset
python -u tools/test.py --snapshot ./experiments/siamrpn_alex_dwxcorr/siamrpn_alex_dwxcorr.pth --dataset $challenge \
                        --config ./experiments/siamrpn_alex_dwxcorr/config.yaml --challenge True \
                        --challenge_cutoff $cutoff_challenge

python tools/eval.py --tracker_path ./results --dataset $challenge --num 1 --tracker_prefix siamrpn_alex_dwxcorr \
                     --challenge False --challenge_cutoff $cutoff_challenge \
                     --show_video_level >> "./testing_dataset/top${cutoff_challenge}_vids_challenge/${challenge}/VOT${cutoff_challenge}_${challenge}_video_details.txt"

# process video stats so we can have two csv files, one with the video level information of a subset and another with a general level information
python tools/process_video_stats.py --path_to_file "./testing_dataset/top${cutoff_challenge}_vids_challenge/${challenge}/VOT${cutoff_challenge}_${challenge}_video_details.txt" \
                                    --path_to_save "./testing_dataset/top${cutoff_challenge}_vids_challenge/${challenge}"

# plots metrics of the challenges for the whole dataset
python tools/plot_stats_challenge.py --path_to_json_file "./testing_dataset/top${cutoff_challenge}_vids_challenge/${challenge}/${challenge}.json" \
                                     --path_to_stats_file ./testing_dataset/top${cutoff_challenge}_vids_challenge/${challenge}/specific_info.csv

#for challenge in $(challenges[@]); do
#    plot_stats_challenge "$challenge"
#done

