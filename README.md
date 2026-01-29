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

## Dataset and Weights
We don't use safety checker, but the download command below will download it automatically
~~~shell
cd ./D4A
mkdir models
huggingface-cli download runwayml/stable-diffusion-v1-5 --local-dir ./models/runwayml/stable-diffusion-v1-5 --local-dir-use-symlinks False
huggingface-cli download h94/IP-Adapter --include "models/ip-adapter_sd15.bin" "image_encoder" --local-dir ./models/h94/IP-Adapter --local-dir-use-symlinks False

mkdir data
ln -s <your rellis3d dataset root> ./data/rellis3d
mv <image prompts dir> ./data/img_prompts
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