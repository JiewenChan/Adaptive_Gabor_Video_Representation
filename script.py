import os
import torch

file_list = sorted(os.listdir("/project3/jiewen/SaV_all"))
for i in range(len(file_list)):
    # if  i % 2 == 1:
    if not os.path.exists(f"/project3/jiewen/50frame_complete/exp_{file_list[i]}/model_010000.pth"):
        print(file_list[i])
        # os.system(f"python train_dynamic_init.py --config experiment_configs/50frame_complete/config.txt --seq_name {file_list[i]} --no_reload")
        # os.system(f"python train_dynamic_init.py --config experiment_configs/wodepth/config.txt --seq_name {file_list[i]} --no_reload")
        # os.system(f"python train_dynamic_init.py --config experiment_configs/woflow/config.txt --seq_name {file_list[i]} --no_reload")
        # os.system(f"python train_dynamic_init.py --config experiment_configs/wocurv/config.txt --seq_name {file_list[i]} --no_reload")
        # os.system(f"python train_dynamic_init.py --config experiment_configs/gaussian/config.txt --seq_name {file_list[i]} --no_reload")
        # os.system(f"python train_dynamic_init.py --config experiment_configs/base/config.txt --seq_name {file_list[i]} --no_reload")
        # os.system(f"python train_dynamic_init.py --config experiment_configs/bspline/config.txt --seq_name {file_list[i]} --no_reload")