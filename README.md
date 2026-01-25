# Diff_Patch_Attack v0.1.0 

## Brief


## Install
Our core reliance libraries are `torch`, `segmentation-models-pytorch`, `transformers`, `diffusers`  
So make sure these libraries are installed before others
~~~shell
conda create -n patch_adv python=3.10 -y
conda activate patch_adv
# my version of torch
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install -e .
~~~

## Execute

### Train SMP Models:
~~~shell
cd ./patch_attack_public
python src/train_smp.py
~~~

### Train Our Patch
the cwd should be our project root directory.
~~~shell
cd ./patch_attack_public
python src/train_patch.py
~~~