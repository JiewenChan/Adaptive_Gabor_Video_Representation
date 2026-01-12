## Environment

- gcc 11.5
- g++ 11.5
- cuda 11.8
- python 3.8
```
conda create -n AdaGaR python=3.8
conda activate AdaGaR
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0  pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c fvcore -c iopath -c conda-forge fvcore iopath --yes
conda install -c bottler nvidiacub --yes
conda install pytorch3d -c pytorch3d --yes
pip install submodules/simple-knn/
pip install submodules/dptr/
pip install configargparse tensorboardX tensorboard imageio opencv-python matplotlib tqdm scipy pytorch_msssim jaxtyping plyfile diffusers transformers omegaconf tabulate rich kornia open3d mediapy einshape imageio[ffmpeg]
```

## Data preparation
Please follow this [instruction](./data_preparation/README.md).


## Training
python train.py --config configs/base/config.txt --seq_name $seq_name

## Testing
python test.py --config configs/base/config.txt --seq_name $seq_name --test
