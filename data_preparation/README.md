Our approach relies on the below works for data preprocessing.
1. Monocular depth estimation [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) and metric depth estimation [Unidepth](https://github.com/lpiccinelli-eth/UniDepth). The metric depth is used to initialize the 3D point flow, together with flow.

    - Change the $UNIDEPTH_PATH and $UNIDEPTH_CKPT_PATH in `compute_metric_depth.py`.

    - Change the $depth-anything-v2 path in `compute_depth.py`

```
python compute_metric_depth.py --img_dir $data_root$/images --depth_dir $data_root$/unidepth_disp --intrins-file $data_root$/unidepth_intrins.json

python compute_depth.py --img_dir $data_root$/images --out_raw_dir $data_root$/depth_anything_v2 --out_aligned_dir $data_root$/aligned_depth_anything_v2 --model depth-anything-v2 --metric_dir $data_root$/unidepth_disp
```

2. [Marigold](https://github.com/prs-eth/Marigold) is also requried in current implementation when training.


3. Flow estimation 

    ### AllTracker demo

    Generate per-frame tracking outputs as `.pth` files for all query/target pairs.

    ```
    python alltracker/demo.py --input_path /path/to/video.mp4 --output_path /path/to/out_dir
    python alltracker/demo.py --input_path /path/to/frames_dir --output_path /path/to/out_dir
    ```

    The output files follow:
    - `{output_path}/{query_t}_{target_t}.pth`
    - Each `.pth` stores a dict with:
    - `tracks`: `[N,4]` tensor (x, y, visconf_invisible, visconf_visible)
    - `query_t`, `target_t`, `rate`
    - `original_size`, `resized_size`


### Custom Video

The video should be extracted to frames first. Then object masks are extracted.


The dataset is prepared in the following format
```
- data_root
    - images
        - 00000.png
        - ...
    - masks
        - 00000.png
        - ...
    - aligned_depth_anything_v2
        - 00000.npy
    - marigold
        - depth_npy
            - 00000_pred.npy
            - ...
    - alltracker
        - 00000_00000.npy
        - 00000_00001.npy
        - ...
    - unidepth_disp (not used in training)
    - depth_anything_v2 (not used in training)
```
