import os
import subprocess
import random
import datetime
import shutil
import numpy as np
import torch
import torch.utils.data
import torch.distributed as dist
from config import config_parser
from tensorboardX import SummaryWriter
from loaders.create_training_dataset import get_training_dataset
from trainer_fragGS import FragTrainer
torch.manual_seed(1234)
# from gui import GUI
# import dearpygui.dearpygui as dpg
from tqdm import tqdm


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


def extract_mask_edge(mask, kernel_size=5):
    import cv2
    # 创建一个卷积核（kernel）用于腐蚀和膨胀操作
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    # 腐蚀操作
    eroded = cv2.erode(mask, kernel, iterations=1)
    # 膨胀操作
    dilated = cv2.dilate(mask, kernel, iterations=1)
    edges = dilated - eroded
    margin = 5
    edges[:margin, :] = edges[-margin:, :] = edges[:, :margin] = edges[:, -margin:] = 255
    return edges.astype(np.uint8)


def train(args):
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
    writer = SummaryWriter(log_dir)

    g = torch.Generator()
    g.manual_seed(args.loader_seed)
    dataset, data_sampler = get_training_dataset(args, max_interval=args.start_interval)
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=args.num_pairs,
                                              worker_init_fn=seed_worker,
                                              generator=g,
                                              num_workers=args.num_workers,
                                              sampler=data_sampler,
                                              shuffle=True if data_sampler is None else False,
                                              pin_memory=True)

    # get trainer
    trainer = FragTrainer(args)
    # trainer = BaseTrainer(args)

    ###### config gui for visualization
    from dataclasses import dataclass
    @dataclass
    class GUIArgs:
        gui: bool = args.gui
        W: int = 512
        H: int = 512
        radius: float = 2
        fovy: float = 60

    start_step = trainer.step + 1
    step = start_step
    epoch = 0
    if args.gui:
        gui_args = GUIArgs()
        gui = GUI(gui_args, [])
        gui.gaussian_trainer = trainer
        while dpg.is_dearpygui_running():
            for batch in data_loader:
                gui.test_step()
                dpg.render_dearpygui_frame()
                if gui.training and step < args.num_iters + start_step + 1:
                    trainer.train_one_step(step, batch)
                    trainer.log(writer, step)

                    step += 1

                    dataset.set_max_interval(args.start_interval + step // 2000)

                    if step >= args.num_iters + start_step + 1:
                        break

                epoch += 1
                if args.distributed:
                    data_sampler.set_epoch(epoch)
    else:
        pbar = tqdm(total=args.num_iters, desc='Training')
        while step < args.num_iters + start_step + 1:
            for batch in data_loader:
                trainer.train_one_step(step, batch)
                trainer.log(writer, step)

                step += 1
                pbar.update(1)
                pbar.set_postfix({'step': step, 'loss': f'{trainer.scalars_to_log.get("loss", 0):.6f}'})

                dataset.set_max_interval(args.start_interval + step // 2000)

                if step >= args.num_iters + start_step + 1:
                    break
        pbar.close()


if __name__ == '__main__':
    args = config_parser()

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    train(args)

