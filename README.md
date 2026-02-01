# Diff_Patch_Attack v0.3.0 

## ⚠️ Project Status

This repository is frozen and not under active development.
Future research explorations and experimental implementations
are conducted elsewhere and are not reflected in this codebase.

## Brief
Why there isn't 0.2.0 ? Aborted. That version contains unstable cross attention functions. So, everything will be composed again in this version.  
Just for self attention.
- [ ] PGD classic adversarial patch generation
- [ ] EOT enables the transferability of Patch
- [ ] Self attention from diffusion pipeline, restrict structure features.
- [ ] Non-Inverse UNet, one step self attention extract.
- [ ] RFES, expand the horizon for patch smooth

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

## Dataset and Weights
We don't use safety checker, but the download command below will download it automatically
~~~shell
cd ./D4A
mkdir models
huggingface-cli download runwayml/stable-diffusion-v1-5 --local-dir ./models/runwayml/stable-diffusion-v1-5 --local-dir-use-symlinks False

mkdir data
ln -s <your rellis3d dataset root> ./data/rellis3d=
~~~

## Execute

### Train SMP Models:
~~~shell
cd ./D4A
python src/train_smp.py
~~~

### Train Our Patch
the cwd should be our project root directory.
~~~shell
cd ./D4A
python src/train_patch.py
~~~