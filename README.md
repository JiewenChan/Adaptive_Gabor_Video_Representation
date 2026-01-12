<p align="center">
  <h1 align="center">AdaGaR: Adaptive Gabor Representation for Dynamic Scene Reconstruction</h1>
  <p align="center">
    <a href="https://jiewenchan.github.io/"><strong>Jiewen Chan</strong></a> ·
    <a href="https://ericzzj1989.github.io/"><strong>Zhenjun Zhao</strong></a> ·
    <a href="https://yulunalexliu.github.io/"><strong>Yu-Lun Liu</strong></a>
  </p>
  <h3 align="center"><a href="https://jiewenchan.github.io/AdaGaR/">🌐 Project Page</a> | <a href="https://arxiv.org/abs/2601.00796">📄 Paper</a></h3>
</p>
<br>

>Reconstructing dynamic 3D scenes from monocular videos requires simultaneously capturing high-frequency appearance details and temporally continuous motion. Existing methods using single Gaussian primitives are limited by their low-pass filtering nature, while standard Gabor functions introduce energy instability. Moreover, lack of temporal continuity constraints often leads to motion artifacts during interpolation. We propose AdaGaR, a unified framework addressing both frequency adaptivity and temporal continuity in explicit dynamic scene modeling. We introduce Adaptive Gabor Representation, extending Gaussians through learnable frequency weights and adaptive energy compensation to balance detail capture and stability. For temporal continuity, we employ Cubic Hermite Splines with Temporal Curvature Regularization to ensure smooth motion evolution. An Adaptive Initialization mechanism combining depth estimation, point tracking, and foreground masks establishes stable point cloud distributions in early training. Experiments on Tap-Vid DAVIS demonstrate state-of-the-art performance (PSNR 35.49, SSIM 0.9433, LPIPS 0.0723) and strong generalization across frame interpolation, depth consistency, video editing, and stereo view synthesis.

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
```
python train.py --config configs/base/config.txt --seq_name $seq_name
```

## Testing
```
python test.py --config configs/base/config.txt --seq_name $seq_name --test
```
