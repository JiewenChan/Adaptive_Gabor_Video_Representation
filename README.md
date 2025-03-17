## Environment
```
conda create -n sav python=3.10
conda activate sav
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0  pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c fvcore -c iopath -c conda-forge fvcore iopath --yes
conda install -c bottler nvidiacub --yes
conda install pytorch3d -c pytorch3d --yes
pip install submodules/simple-knn/
pip install submodules/dptr/
pip install configargparse tensorboardX tensorboard imageio opencv-python matplotlib tqdm scipy pytorch_msssim jaxtyping plyfile diffusers transformers omegaconf tabulate rich kornia open3d mediapy einshape
pip install imageio[ffmpeg]
```

Install [pytorch3d](https://github.com/facebookresearch/pytorch3d).


## Data preparation
Please follow this [instruction](./data_preparation/README.md).


## Training
python train.py --config configs/config.txt --seq_name $seq_name --num_imgs 250