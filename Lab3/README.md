# Pre-installation steps

#### Select workspace folder
```bash
cd <folder_of_choice>
```

#### Clone the repository
```bash
git clone https://github.com/jcsma-tracking-sotframeworks/pysot.git
```

#### Add pysot to PATH (This needs to be done every time we open a terminal)
```bash
export PYTHONPATH=~/pysot:$PYTHONPATH
```

#### Go to pysot directory
```bash
cd <folder_of_choice>/pysot
```
# Installation

This document contains detailed instructions for installing dependencies for PySOT. We recommand using the [install.sh](install.sh). The code is tested on an Ubuntu 16.04 system with Nvidia GPU (We recommand 1080TI / TITAN XP).

### Requirements
* Conda with Python 3.7.
* Nvidia GPU.
* PyTorch 0.4.1
* yacs
* pyyaml
* matplotlib
* tqdm
* OpenCV

## Step-by-step instructions

#### Create environment and activate
```bash
conda create --name pysot python=3.7
conda activate pysot
```

#### Install numpy/pytorch/opencv
``` bash
conda install numpy
conda install pytorch=0.4.1 torchvision cuda90 -c pytorch
pip install opencv-python
```
#### Install cython

```bash
pip install cython==0.29.24 -f http://www-vpu.eps.uam.es/~jcs/DLVSP/pkgs/dlvsp.html --trusted-host www-vpu.eps.uam.es
```

#### Install other requirements
```bash
pip install pyyaml yacs tqdm colorama matplotlib tensorboardX
```

#### Build extensions
```bash
# run this command inside the pysot directory
python setup.py build_ext --inplace
```

# Deployment

#### Get the dataset and decompress it 
```bash
# If testing_dataset folder not present
mkdir <folder_of_choice>/pysot/testing_dataset/
cd <folder_of_choice>/pysot/testing_dataset/
wget http://www-vpu.eps.uam.es/~jcs/DLVSP/vot/vot2018.zip

# decompress dataset (unzip)
unzip -q vot2018.zip
mv vot2018 VOT2018
```

#### Get the dataset descriptor

```bash
cd <folder_of_choice>/pysot/testing_dataset/VOT2018
wget http://www-vpu.eps.uam.es/~jcs/DLVSP/pysot_dataset/VOT2018.json
```

#### Get the pretrained model of SiamRPN
```bash
# The folder /experiments/siamrpn_alex_dwxcorr/ may not be created,
# create it by running mkdir -p ~/pysot/experiments/siamrpn_alex_dwxcorr/
wget http://www-vpu.eps.uam.es/~jcs/DLVSP/pysot_nets/siamrpn_alex_dwxcorr/siamrpn_alex_dwxcorr.pth
mv siamrpn_alex_dwxcorr.pth ~/pysot/experiments/siamrpn_alex_dwxcorr/
```

```yaml
META_ARC: "siamrpn_alex_dwxcorr"

BACKBONE:
    TYPE: "alexnetlegacy"
    KWARGS:
        width_mult: 1.0   

ADJUST:
    ADJUST: False

RPN:
    TYPE: 'DepthwiseRPN'
    KWARGS:
        anchor_num: 5
        in_channels: 256
        out_channels: 256

MASK:
    MASK: False

ANCHOR:
    STRIDE: 8
    RATIOS: [0.33, 0.5, 1, 2, 3]
    SCALES: [8]
    ANCHOR_NUM: 5

TRACK:
    TYPE: 'SiamRPNTracker'
    PENALTY_K: 0.16
    WINDOW_INFLUENCE: 0.40
    LR: 0.30
    EXEMPLAR_SIZE: 127
    INSTANCE_SIZE: 287
    BASE_SIZE: 0
    CONTEXT_AMOUNT: 0.5
```

#### Run the tracker
```bash
cd <folder_of_choice>/pysot
conda activate pysot
python -u tools/test.py --snapshot ./experiments/siamrpn_alex_dwxcorr/siamrpn_alex_dwxcorr.pth --dataset VOT2018 --config ./experiments/siamrpn_alex_dwxcorr/config.yaml
```

# Commands of interest

#### Creation of the challenge dataset 
Test videos that are most challenging in categories:
* __camera_motion__
* __illum_change__
* __motion_change__
* __occlusion__
* __size_change__
```bash
python extract_challenge.py --dataset_path testing_dataset/VOT2018 --json_file VOT2018.json --dest_path testing_dataset/challenges
```

#### Run challenge dataset
```bash
python -u tools/test.py --snapshot ./experiments/siamrpn_alex_dwxcorr/siamrpn_alex_dwxcorr.pth --dataset <challenge_subset> --config ./experiments/siamrpn_alex_dwxcorr/config.yaml --challenge True
```

#### Run evaluation for challenge dataset
```bash
python tools/eval.py --tracker_path ./results --dataset CAMERA_MOTION --num 1 --tracker_prefix siamrpn_alex_dwxcorr --challenge True
```
