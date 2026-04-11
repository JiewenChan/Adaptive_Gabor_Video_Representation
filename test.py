import os
import glob
import subprocess
import random
import datetime
import shutil
import numpy as np
import torch
import torch.utils.data
import torch.distributed as dist
from utils.config import config_parser
from tensorboardX import SummaryWriter
from loaders.create_training_dataset import get_training_dataset
# from trainer_ours import FragTrainer
from trainer import FragTrainer
torch.manual_seed(1234)
# from gui import GUI
# import dearpygui.dearpygui as dpg


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def test(args):
    # seq_name = os.path.basename(args.data_dir.rstrip('/'))
    seq_name = args.seq_name
    out_dir = os.path.join(args.save_dir, '{}_{}'.format(args.expname, seq_name))
    os.makedirs(out_dir, exist_ok=True)
    print('optimizing for {}...\n output is saved in {}'.format(seq_name, out_dir))

    args.out_dir = out_dir

    # save the args and config files
    f = os.path.join(out_dir, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            if not arg.startswith('_'):
                attr = getattr(args, arg)
                file.write('{} = {}\n'.format(arg, attr))

    if args.config:
        f = os.path.join(out_dir, 'config.txt')
        if not os.path.isfile(f):
            shutil.copy(args.config, f)

    log_dir = 'logs/{}_{}'.format(args.expname, seq_name)
    # writer = SummaryWriter(log_dir)

    g = torch.Generator()
    g.manual_seed(args.loader_seed)
    # dataset, data_sampler = get_training_dataset(args, max_interval=args.start_interval)
    # data_loader = torch.utils.data.DataLoader(dataset,
    #                                           batch_size=args.num_pairs,
    #                                           worker_init_fn=seed_worker,
    #                                           generator=g,
    #                                           num_workers=args.num_workers,
    #                                           sampler=data_sampler,
    #                                           shuffle=True if data_sampler is None else False,
    #                                           pin_memory=True)

    # get tester
    tester = FragTrainer(args)
    # tester = BaseTrainer(args)

    if True:
        tester.render_video(step=0, save_frames=True)

    if True:
        tester.render_shape_preview(step=1, save_frames=True)

    if False:
        tester.render_part(fg=True, threshold=0.5)
        print()

    if True:
        tester.get_interpolation_result(scaling=3)

    ###### This part is for NVS
    if True:
        tester.get_nvs_rendered_imgs()
        tester.get_stereo_rendered_imgs()
        
    if False:
        def _collect_img_paths(folder, exts=(".png", ".jpg", ".jpeg")):
            paths = []
            for ext in exts:
                paths.extend(glob.glob(os.path.join(folder, f"*{ext}")))
            return sorted(paths)

        mask_path = os.path.join(args.data_dir, args.seq_name, "masks", "00000.png")
        edited_img_dir = f"data/{args.seq_name}/video_edited"
        edited_img_paths = _collect_img_paths(edited_img_dir)
        if len(edited_img_paths) == 0:
            raise FileNotFoundError(f"No edited images found in {edited_img_dir}")
        print(edited_img_paths)
        # tester.optimize_appearance_from_mask(mask_path, edited_img_path)
        tester.optimize_appearance_from_img(edited_img_paths)


if __name__ == '__main__':
    args = config_parser()

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    test(args)
