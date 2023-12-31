#!/bin/bash

conda create --name dlvsp_challenge_env python=3.10
source activate dlvsp_challenge_env

pip3 install numpy
pip3 install ipykernel
pip3 install matplotlib
pip3 install opencv-python
pip3 install tqdm lap
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
pip3 install http://www-vpu.eps.uam.es/~jcs/DLVSP/py-motmetrics-fix_pandas_deprecating_warnings_python310.zip

conda install -c conda-forge ipywidgets
