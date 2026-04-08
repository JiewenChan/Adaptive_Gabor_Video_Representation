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

The codebase uses two levels of configuration:

- `--config`: the experiment / runtime config parsed by `configargparse`. This file controls dataset paths, iteration count, logging, checkpoint loading, and evaluation flags.
- `--gs_config_file`: the Gabor / representation config. This file controls the point-cloud parameterization, renderer, optimizer, loss toggles, and whether adaptive initialization is used.

Typical examples:

```bash
python train.py --config configs/blackswan/config.txt --seq_name blackswan
```

You can also override individual arguments directly from the command line, for example:

```bash
python train.py \
  --config configs/blackswan/config.txt \
  --seq_name blackswan \
  --num_iters 10000 \
  --num_imgs 100 \
  --save_dir out_debug
```

Important training arguments:

- `--config`: path to the runtime config file. In the provided examples this is usually `configs/base/config.txt`, `configs/bspline/config.txt`, or `configs/cubic/config.txt`.
- `--gs_config_file`: path to the Gaussian representation YAML. This is usually already set inside `config.txt`, but can be overridden manually.
- `--data_dir`: root directory containing all sequences.
- `--seq_name`: sequence name under `data_dir`. The actual input path is resolved as `data_dir/seq_name/...`.
- `--save_dir`: root output directory. Results are written to `save_dir/<expname>_<seq_name>`.
- `--expname`: experiment prefix used in output and log directory names.
- `--ckpt_path`: load a specific checkpoint instead of automatically picking the latest checkpoint in the output directory.
- `--no_reload`: disable checkpoint reloading even if checkpoints already exist.
- `--num_iters`: number of training iterations.

## Testing

Typical evaluation command:

```bash
python test.py --config configs/blackswan/config.txt --seq_name blackswan --test
```

Important testing arguments:

- `--test`: enables test mode. This skips adaptive track extraction for training and uses the evaluation code path.

Additional notes:

- Outputs are saved under `save_dir/<expname>_<seq_name>/`.
- `test.py` reuses the same config system as training, so training-time arguments such as `num_imgs`, `base_idx`, and `down_scale` also affect evaluation.
- If you want to compare different motion parameterizations fairly, keep `config.txt` identical except for `gs_config_file`.
