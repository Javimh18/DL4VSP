conda create --name dlvsp_challenge_env python=3.10
source activate dlvsp_challenge_env


pip3 install numpy
pip3 install opencv-python
pip3 install tqdm lap
pip3 install torch==2.1.0+cu121 torchvision==0.16.0+cu121 torchaudio===2.1.0+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
pip3 install http://www-vpu.eps.uam.es/~jcs/DLVSP/py-motmetrics-fix_pandas_deprecating_warnings_python310.zip

conda install -c conda-forge ipywidgets
