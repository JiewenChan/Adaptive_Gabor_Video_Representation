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


def extract_mask_edge(mask, kernel_size=5):
    import cv2
    # Create a convolution kernel for erosion and dilation.
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    # Erosion.
    eroded = cv2.erode(mask, kernel, iterations=1)
    # Dilation.
    dilated = cv2.dilate(mask, kernel, iterations=1)
    edges = dilated - eroded
    margin = 5
    edges[:margin, :] = edges[-margin:, :] = edges[:, :margin] = edges[:, -margin:] = 255
    return edges.astype(np.uint8)


def render_wave_coefficients_part(trainer):
    """
    Render four videos showing where Gaussians fall under different mean wave coefficient ranges.
    """
    intervals = [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0), (0.0, 0.5), (0.0, 0.75), (0.0, 1.0)]
    trainer.render_wave_coefficients_part(intervals=intervals)


def render_wave_coefficients_frame36_fg(trainer, frame_idx=0):
    """
    Render the 36th-frame foreground (zero-based index 35) for the same wave coefficient ranges.
    """
    intervals = ["[0.0, 0.0]", "(0, 0.1)", "[0.1, 0.3)", "[0.3, 1.0]", "[0.0, 0.1)", "[0.0, 0.3)", "[0.0, 1.0]"]
    trainer.render_wave_coefficients_frame_mask(frame_idx=frame_idx, intervals=intervals)


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

    if True:
        tester.render_video(step=0, save_frames=True)

    if True:
        tester.render_shape_preview(step=1, save_frames=True)

    if True:
        clip_value = args.flow_clip if args.flow_clip > 0 else None
        tester.render_flow_maps(stride=max(1, args.flow_stride),
                                max_pairs=args.max_flow_pairs,
                                save_raw=args.save_raw_flow,
                                clip_flow=clip_value)

    if False:
        tester.render_part(fg=True, threshold=0.5)
        print()

    if False:
        render_wave_coefficients_part(tester)

    if True:
        render_wave_coefficients_frame36_fg(tester)

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
        tester.get_interpolation_result(scaling=3)

    ##### This code is for canonical space visualization
    #### track-everything's canonical space
    if False:
        tester.save_canonical_rgba_volume(num_pts=5000000, sample_points_from_frames=True)
    if False:
        pts_canonical_np, colors_np, mask_np = tester.save_canonical_points(start_id=0, end_id=dataset.num_imgs, step=10)
        if False:
            masks = [extract_mask_edge((m*255).astype(np.uint8), kernel_size=3) == 0 for p, m in zip(pts_canonical_np, masks)]
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

    ##### Canonical 3D Scene Visualization
    if False:
        tester.render_canonical()
        
    if False:
        def _collect_img_paths(folder, exts=(".png", ".jpg", ".jpeg")):
            paths = []
            for ext in exts:
                paths.extend(glob.glob(os.path.join(folder, f"*{ext}")))
            return sorted(paths)

        mask_path = os.path.join(args.data_dir, args.seq_name, "masks", "00000.png")
        edited_img_dir = "/project3/jiewen/VideoGaussianTest/blackswan-watercolor"
        edited_img_paths = _collect_img_paths(edited_img_dir)
        if len(edited_img_paths) == 0:
            raise FileNotFoundError(f"No edited images found in {edited_img_dir}")
        print(edited_img_paths)
        # tester.optimize_appearance_from_mask(mask_path, edited_img_path)
        tester.optimize_appearance_from_img(edited_img_paths)
        
    if True:
        mask_path = os.path.join(args.data_dir, args.seq_name, "masks", "00000.png")
        t_query = None
        # t_query = torch.linspace(0, tester.num_imgs, tester.num_imgs*5, device='cuda')
        tester.animate_gaussian_trajectories(mask_path, n=10, t_query=t_query)


if __name__ == '__main__':
    args = config_parser()

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    test(args)
