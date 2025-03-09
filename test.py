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

    if False:
        # tester.draw_gs_trajectory(samp_num=10, gs_num=512)
        use_mask = True
        track_imgs = tester.draw_pixel_trajectory(use_mask=use_mask, radius=4)
        save_dir = os.path.join(tester.out_dir, 'tracking')
        os.makedirs(save_dir, exist_ok=True)
        import imageio
        w = track_imgs[0].shape[1]
        track_imgs = [x[:,w//2:] for x in track_imgs]
        save_name = "tracking.mp4" if use_mask else "tracking_no_mask.mp4"
        imageio.mimwrite(os.path.join(save_dir, save_name), track_imgs, fps=15)
        print()

    if False:
        tester.render_video(save_frames=True)

    if False:
        tester.render_part(fg=True, threshold=0.5)
        print()

    if False:
        ### for cow
        delta_pos = torch.tensor([[0.6, -0.4, 0.1]], device='cuda')
        tester.add_fg(delta_pos, scale=0.8, threshold=0.9)
        # delta_pos = torch.tensor([[-0.4, 0.1, -0.6]], device='cuda')
        # tester.add_fg(delta_pos, scale=1.2, threshold=0.9)
        ### for camel
        # delta_pos = torch.tensor([[0.4, 0.3, -0.2]], device='cuda')
        # tester.add_fg(delta_pos, scale=0.6, threshold=0.9)

        print()

    if True:
        tester.get_interpolation_result(scaling=4)
    
    if True:
        
        mask_path = os.path.join(args.data_dir, args.seq_name, "masks", "00000.png")
        # edited_img_path = os.path.join(args.data_dir, "sketch_1.png")
        edited_img_path = os.path.join("/home_nfs/jiewen/test/Splatter_A_Video/src/out/point_bear/image.png")
        # tester.optimize_appearance_from_mask(mask_path, edited_img_path)
        tester.optimize_appearance_from_img(edited_img_path)

    ##### This code is for canonical space visualization
    #### track-everything's canonical space
    if False:
        tester.save_canonical_rgba_volume(num_pts=5000000, sample_points_from_frames=True)
    if False:
        pts_canonical_np, colors_np, mask_np = tester.save_canonical_points(start_id=0, end_id=dataset.num_imgs, step=10)
        if False:
            masks = [extract_mask_edge((m*255).astype(np.uint8), kernel_size=3) == 0 for m in mask_np]
            points = [p[m.reshape(-1)] for p, m in zip(pts_canonical_np, masks)]
            colors = [c[m] for c, m in zip(colors_np, masks)]
            points = np.concatenate(points, axis=0)
            colors = np.concatenate(colors, axis=0)
        # print()
        import trimesh
        trimesh.PointCloud(pts_canonical_np.reshape(-1,3), colors=colors_np.reshape(-1,3)).export("./debug_all.ply")

    ###### This part is for NVS
    if True:
        tester.get_nvs_rendered_imgs()
        tester.get_stereo_rendered_imgs()


if __name__ == '__main__':
    args = config_parser()

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    test(args)

