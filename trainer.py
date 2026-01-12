import glob
import json
import os
import pdb
import time
from pathlib import Path

import cv2
import imageio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import re

import utils.util as util
from criterion import masked_mse_loss, masked_l1_loss, compute_depth_range_loss, lossfun_distortion
from pointrix.model.loss import l1_loss, ssim
from kornia import morphology as morph
# from gs import GSTrainer
# from frag_gs import GSTrainer
from pointrix.utils.config import load_config, parse_structured
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Union, List
from pointrix.camera.cam_utils import construct_canonical_camera, construct_canonical_camera_from_focal
from pointrix.optimizer import parse_optimizer, parse_scheduler
from pointrix.renderer import parse_renderer
from model.frag_model import FragModel
import dptr.gs as gs
from tqdm import tqdm
from pytorch3d.renderer import look_at_rotation
import matplotlib.pyplot as plt
from video3Dflow.utils import parse_tapir_track_info


torch.manual_seed(1234)


def init_weights(m):
    # Initializes weights according to the DCGAN paper
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)


def de_parallel(model):
    return model.module if hasattr(model, 'module') else model

def entropy_loss(opacity, gs_idx):
    """
    Entropy loss for the features.

    Parameters
    ----------
    opacity : torch.Tensor [N,1]
        The opacity values.
    gs_idx : torch.Tensor [B,H,W,K]
        The first K gaussian index for each pixel.
    
    Returns
    -------
    torch.Tensor
        The entropy loss.
    """
    # mask = gs_idx > -1  # [H,W,K]
    bs, H, W, K = gs_idx.shape
    ### use 1 opacity to omit gs_idx=-1
    opacity = torch.cat([opacity, torch.ones_like(opacity[-1:])], dim=0).squeeze(1)  # [N+1,]
    gs_idx[gs_idx == -1] = opacity.shape[0] - 1
    opacity = opacity.unsqueeze(0).repeat(bs, 1)  # [B,N+1,]
    flatten_gs_idx = gs_idx.reshape(bs,-1).long()  # [bs, H*W*K]
    pixel_opacity = torch.gather(opacity, dim=1, index=flatten_gs_idx)  # [bs, H*W*K]
    pixel_opacity = pixel_opacity.reshape(bs, H, W, K)  # [bs, H, W, K]
    # import matplotlib.pyplot as plt
    # plt.hist(pixel_opacity.detach().cpu().numpy().reshape(-1,1))
    pixel_opacity = pixel_opacity / (pixel_opacity.sum(dim=-1, keepdim=True) + 1e-8)
    pixel_entropy = -torch.sum(pixel_opacity * torch.log(pixel_opacity), dim=-1)  # [bs,]
    return pixel_entropy.mean()

    pixel_alpha = torch.cumprod(1 - pixel_opacity, dim=-1)
    pixel_transparency = torch.cat([torch.ones_like(pixel_alpha[..., :1]), pixel_alpha[..., :-1]], dim=-1)
    pixel_weight = pixel_transparency * pixel_opacity + 1e-5
    pixel_entropy = -torch.sum(pixel_weight * torch.log(pixel_weight), dim=-1)  # [bs,]
    return pixel_weight, pixel_entropy.mean()

def curvature_penalty_nonuniform(yk_nodes, tk, weight_by_span=True, eps=1e-8):
    """
    yk_nodes: (N_pts, M, D) values at M knots per point (D=3 for xyz)
    tk:       (M,)          knot times (possibly non-uniform)
    Returns: scalar loss
    """
    # Neighboring knot intervals.
    h_prev = tk[1:-1] - tk[:-2]        # (M-2,)
    h_next = tk[2:]   - tk[1:-1]       # (M-2,)

    # Forward/backward slope differences.
    d_plus  = (yk_nodes[:, 2:,  :] - yk_nodes[:, 1:-1, :]) / (h_next.view(1, -1, 1) + eps)
    d_minus = (yk_nodes[:, 1:-1, :] - yk_nodes[:, :-2,  :]) / (h_prev.view(1, -1, 1) + eps)

    # Centered second derivative (non-uniform formula).
    ypp = 2.0 * (d_plus - d_minus) / ((h_prev + h_next).view(1, -1, 1) + eps)  # (N_pts, M-2, D)

    if weight_by_span:
        # Approximate integral of ||y''||^2 dt; span-weighting is more stable.
        w = 0.5 * (h_prev + h_next).view(1, -1, 1)  # (1, M-2, 1)
        num = (w * (ypp ** 2)).sum()
        den = (w.sum() * ypp.shape[-1]) + eps
        return num / den
    else:
        return (ypp ** 2).mean()


def alpha_blending_firstK(attribute, gs_idx, pixel_weight, bg=1):
    """
    A function approximates alpha-blending.
    Attribute: [N,D]
    """
    ### query attribute for each pixel
    bs, H, W, K = gs_idx.shape
    gs_idx[gs_idx == -1] = attribute.shape[0] - 1
    flatten_gs_idx = gs_idx.reshape(bs,-1).long()  # [bs, H*W*K]
    attribute = torch.cat([attribute, bg*torch.ones_like(attribute[-1:])], dim=0)  # [N+1,D]
    attribute = attribute.unsqueeze(0).repeat(bs, 1, 1)  # [bs, N+1, D]
    flatten_gs_idx = flatten_gs_idx.unsqueeze(-1).repeat(1,1,attribute.shape[-1])  # [bs, H*W*K, D]
    pixel_attribute = torch.gather(attribute, dim=1, index=flatten_gs_idx)  # [bs, H*W*K, D]
    pixel_attribute = pixel_attribute.reshape(bs, H, W, K, -1)  # [bs, H, W, K, D]
    ### alpha-blending
    pixel_render = torch.sum(pixel_attribute * pixel_weight.unsqueeze(-1), dim=-2)  # [bs, H, W, D]
    return pixel_render


@dataclass
class LossConfig:
    """Loss toggles that can be controlled from the config."""
    optical_flow: bool = True
    depth: bool = True
    pos_curvature: bool = True
    rot_curvature: bool = True
    scale_curvature: bool = True


class FragTrainer:
    @dataclass
    class Config:
        # Modules
        model: dict = field(default_factory=dict)
        optimizer: dict = field(default_factory=dict)
        renderer: dict = field(default_factory=dict)
        scheduler: Optional[dict] = field(default_factory=dict)
        writer: dict = field(default_factory=dict)
        hooks: dict = field(default_factory=dict)
        render_attributes: dict = field(default_factory=dict)
        point_cloud_name: str = "gs_bspline_all"
        loss: LossConfig = field(default_factory=LossConfig)
        # Dataset
        dataset_name: str = "NeRFDataset"
        dataset: dict = field(default_factory=dict)

        # Training config
        batch_size: int = 1
        num_workers: int = 0
        max_steps: int = 30000
        val_interval: int = 2000
        spatial_lr_scale: bool = True
        skip_interval: int = 1

        # Progress bar
        bar_upd_interval: int = 10
        # Output path
        output_path: str = "output"

    cfg: Config

    def __init__(self, args, device='cuda'):
        self.args = args
        if isinstance(device, str):
            if device.startswith('cuda') and not torch.cuda.is_available():
                print("[FragTrainer] CUDA not available, falling back to CPU.")
                device = 'cpu'
            torch_device = torch.device(device)
        elif isinstance(device, torch.device):
            torch_device = device
        else:
            torch_device = torch.device('cpu')
        self.device = torch_device
        self.device_str = (torch_device.type if torch_device.index is None
                           else f"{torch_device.type}:{torch_device.index}")

        # self.read_data()
        # self.read_data_simple()
        
        ########### add canonical space GS. This contains the GS optimizer

        cfg = load_config(args.gs_config_file)
        # cfg.trainer.dataset.data_path = self.img_files[0]
        self.cfg = parse_structured(self.Config, cfg.trainer)
        
        self.read_data_simple()
        cfg.trainer.dataset.data_path = self.img_files[0]
        self.cfg = parse_structured(self.Config, cfg.trainer)

        # TODO define the gs atlases here !
        ### define the gs atlas config
        @dataclass
        class GSAtlasCFG:
            name: str
            num_images: int
            start_frame_path: str
            end_frame_path: str
            start_frame_mask_path: str
            start_depth_npy: str
            end_frame_mask_path: str
            start_frame_id: int
            end_frame_id: int
            reverse_mask: bool = False
            render_attributes: dict = field(default_factory=dict)
            is_fg: bool = True


        
        gs_atlas_cfg_bspline_all = GSAtlasCFG(
            name=self.cfg.point_cloud_name,
            num_images=self.num_imgs,
            start_frame_path='test_data/bear/color/00000.jpg',
            end_frame_path=None,
            start_frame_mask_path='test_data/bear/masks/00000.png',
            start_depth_npy='test_data/bear/marigold/depth_npy/00000_pred.npy',
            end_frame_mask_path=None,
            start_frame_id=0,
            end_frame_id=self.num_imgs-1,
            render_attributes=self.cfg.render_attributes
        )
        self.out_dir = os.path.join(args.save_dir, '{}_{}'.format(args.expname, self.seq_name))

        # self.gs_atlas_cfg_list = [gs_atlas_cfg1, gs_atlas_cfg2, gs_atlas_cfg3]
        # self.gs_atlas_cfg_list = [gs_atlas_cfg2, gs_atlas_cfg3]
        # self.gs_atlas_cfg_list = [gs_atlas_cfg1]
        self.gs_atlas_cfg_list = [gs_atlas_cfg_bspline_all]
        base_point_seq_list = [self.tracks_fg_info['tracks_3d'].permute(1,0,2).to(self.device),
                               self.tracks_bg_info['tracks_3d'].permute(1,0,2).to(self.device)]
        
        # TODO define GSTrainer with __init__(self, cfg, *gs_atlas_cfg_list)
        # Rewrite. This should be a model, instead of a trainer !!!
        ### 1. initialize gs model / optimizer for each atlas
        self.cfg.model.pop('name')
        self.white_bg = self.cfg.dataset.white_bg
        self.gs_atlases_model = FragModel(self.cfg.model, self.gs_atlas_cfg_list, base_point_seq_list, white_bg=self.white_bg, num_frames=self.num_imgs)
        
        ### 2. set fixed camera parameter
        self.enable_ortho_projection = True if 'Ortho' in self.cfg.renderer.name else False
        self.construct_render_dict(self.h, self.w, self.gs_atlases_model.focal_y_ratio)
        
        ### 3. renderer
        self.renderer = parse_renderer(
            self.cfg.renderer, white_bg=self.white_bg, device=device)
        # Optional override to treat every Gaussian as fully opaque while rendering.
        self.force_full_opacity_render = False
        
        ### 4. optimizer and scheduler
        cameras_extent = 5  # the extent of the camera
        for gs_atlas_cfg in self.gs_atlas_cfg_list:
            name = gs_atlas_cfg.name
            optimizer_config = self.cfg.optimizer.copy()
            optimizer_config["optimizer_1"]["extra_cfg"]["vis_out_dir"] = os.path.join(self.out_dir, 'vis', 'grad')
            if name == 'gs_base':
                # optimizer_config["optimizer_1"]["extra_cfg"]["densify_start_iter"] = 100000
                # optimizer_config["optimizer_1"]["extra_cfg"]["opacity_reset_interval"] = 100000
                pass
            setattr(self, name+"_optimizer", 
                    parse_optimizer(optimizer_config, 
                                    model=self.gs_atlases_model.get_atlas(name),
                                    cameras_extent=cameras_extent)
                    )
            setattr(self, name+"_scheduler",
                    parse_scheduler(self.cfg.scheduler, 
                                    cameras_extent if self.cfg.spatial_lr_scale else 1.)
                    )
            
        # seq_name = os.path.basename(args.data_dir.rstrip('/'))
        self.step = self.load_from_ckpt(self.out_dir,
                                        load_opt=self.args.load_opt,
                                        load_scheduler=self.args.load_scheduler)
        self.time_steps = torch.linspace(1, self.num_imgs, self.num_imgs, device=self.device)[:, None] / self.num_imgs
        # self.init_tracks()

        #### for detph loss
        from loss import ScaleAndShiftInvariantLoss
        self.depth_loss = ScaleAndShiftInvariantLoss(alpha=0.5, scales=1)
    
    
    def read_data_simple(self):        
        self.seq_name = self.args.seq_name
        self.img_dir = os.path.join(self.args.data_dir, self.seq_name, "images")
        img_files = sorted(glob.glob(os.path.join(self.img_dir, '*')))
        if self.args.num_imgs < 0:
            self.num_imgs = len(img_files)
        else:
            self.num_imgs = min(self.args.num_imgs, len(img_files))

        self.base_idx = self.args.base_idx
        self.img_files = img_files[self.base_idx:self.num_imgs+self.base_idx]

        t1 = time.time()
        images = np.array([imageio.imread(img_file) / 255. for img_file in self.img_files])
        self.images = torch.from_numpy(images).float()  # [n_imgs, h, w, 3]
        # image = np.array(imageio.imread(self.img_files[0])) / 255.
        # self.images = torch.from_numpy(image).float()[None].repeat(self.num_imgs,1,1,1)  # [n_imgs, h, w, 3]
        self.h, self.w = self.images.shape[1:3]
        print(f'Read images time: {time.time() - t1} s')


        ### load depth
        if True:
            self.depth_dir = f"{os.path.join(self.args.data_dir, self.seq_name)}/marigold/depth_npy"
            depth_files = sorted(glob.glob(os.path.join(self.depth_dir, '*.npy')))[self.base_idx:self.num_imgs+self.base_idx]
            depth = np.array([np.load(depth_file) for depth_file in depth_files])
            self.gt_depths = torch.from_numpy(depth).float().to(self.device)  # [n_imgs, h, w]

        ### load mask
        if True:
            self.mask_dir = os.path.join(self.args.data_dir, self.seq_name, "masks")
            mask_files = sorted(glob.glob(os.path.join(self.mask_dir, '*.png')))[self.base_idx:self.num_imgs+self.base_idx]
            masks = np.array([imageio.imread(mask_file)/255. for mask_file in mask_files])
            if masks.ndim == 4:
                masks = masks.sum(axis=-1) > 0
            self.masks = torch.from_numpy(masks).float().to(self.device)

        self.grid = util.gen_grid(self.h, self.w, device=self.device, normalize=False, homogeneous=True).float()

        ### load 3D flow
        depth_folder = os.path.join(self.args.data_dir, self.seq_name, "aligned_depth_anything_v2")
        tracking_folder = os.path.join(self.args.data_dir, self.seq_name, "alltracker")
        # tracking_folder = f"/home_nfs/jiewen/alltracker/{self.seq_name}"
        frames_folder = self.img_dir
        mask_folder = self.mask_dir
        from video3Dflow.adaptive_video_3d_flow import AdaptiveVideo3DFlow as Video3DFlow
        self.video_3d_flow = Video3DFlow(depth_folder, tracking_folder, frames_folder, mask_folder)
        self.video_3d_flow.setup(test_mode=self.args.test)
        
        if not self.args.test:
            tracks_3d, visibles, invisibles, confidences, colors = self.video_3d_flow.get_tracks_3d(
                num_samples=2000, 
                start=self.base_idx, 
                end=self.num_imgs+self.base_idx, 
                step=self.cfg.skip_interval, )
            self.tracks_fg_info = {
                "tracks_3d": tracks_3d,  # [N, T, 3]
                "visibles": visibles,  # [N, T]
                "invisibles": invisibles,
                "confidences": confidences,
                "colors": colors
            }
            ##### extract bg tracks
            self.video_3d_flow.extract_fg = False
            tracks_3d, visibles, invisibles, confidences, colors = self.video_3d_flow.get_tracks_3d(
                num_samples=20000, 
                start=self.base_idx, 
                end=self.num_imgs+self.base_idx, 
                step=self.cfg.skip_interval, )
            
            grid_size = int(64 / (self.args.video_flow_margin / 0.25))
            extended_tracks_3d, extended_colors = self.video_3d_flow.extend_track3d(tracks_3d, margin=self.args.video_flow_margin, grid_size=grid_size)
            # extended_tracks_3d, extended_colors = self.video_3d_flow.extend_track3d(tracks_3d, margin=0.75)
            tracks_3d = torch.cat([extended_tracks_3d, tracks_3d], dim=0)
            colors = torch.cat([extended_colors, colors], dim=0)
            self.tracks_bg_info = {
                "tracks_3d": tracks_3d,  # [N, T, 3]
                "visibles": visibles,  # [N, T]
                "invisibles": invisibles,
                "confidences": confidences,
                "colors": colors
            }
            
        else:
            self.tracks_fg_info = {
                "tracks_3d": torch.zeros((2000, self.num_imgs, 3)),  # [N, T, 3]
            }
            self.tracks_bg_info = {
                "tracks_3d": torch.zeros((2000, self.num_imgs, 3)),  # [N, T, 3]
            }
       

    def construct_render_dict(self, h, w, focal_y_ratio=None):
        """
        define the fixed camera

        Parameters
        ----------
        h : int
            The height of the image.
        w : int
            The width of the image.
        """
        if focal_y_ratio is not None:
            focal = focal_y_ratio * h
            camera = construct_canonical_camera_from_focal(width=w, height=h, focal=focal)
        else:
            camera = construct_canonical_camera(width=w, height=h)

        self.batch_dict = {
            "camera": camera,
            "FovX": camera.fovX,
            "FovY": camera.fovY,
            "height": int(camera.image_height),
            "width": int(camera.image_width),
            "world_view_transform": camera.world_view_transform,
            "full_proj_transform": camera.full_proj_transform,
            "extrinsic_matrix": camera.extrinsic_matrix,
            "intrinsic_matrix": camera.intrinsic_matrix,
            "camera_center": camera.camera_center,
            "enable_ortho_projection": getattr(self, "enable_ortho_projection", False),
        }

    def _maybe_wrap_forward_for_test(self):
        """
        During testing we can optionally offset every Gaussian along +Z by wrapping the model forward pass.
        """
        if not getattr(self, "test_scene_z_offset", 0.0):
            return

        z_offset = self.test_scene_z_offset
        orig_forward = self.gs_atlases_model.forward

        def forward_with_offset(ids, batch=None):
            render_dict = orig_forward(ids, batch)
            return self._apply_test_scene_z_offset(render_dict, z_offset)

        self.gs_atlases_model.forward = forward_with_offset

        if hasattr(self.gs_atlases_model, "forward_single_atlas"):
            orig_forward_single = self.gs_atlases_model.forward_single_atlas

            def forward_single_with_offset(ids, name):
                render_dict = orig_forward_single(ids, name)
                return self._apply_test_scene_z_offset(render_dict, z_offset)

            self.gs_atlases_model.forward_single_atlas = forward_single_with_offset

    @staticmethod
    def _apply_test_scene_z_offset(render_dict, z_offset):
        """
        Shift specific spatial tensors along +Z for testing visualization.
        """
        if not z_offset:
            return render_dict
        adjusted = render_dict.copy()
        spatial_keys = ("position", "track_gs")
        for key in spatial_keys:
            tensor = adjusted.get(key, None)
            if tensor is None:
                continue
            if not torch.is_tensor(tensor) or tensor.shape[-1] < 3:
                continue
            shifted = tensor.clone()
            shifted[..., 2] += z_offset
            adjusted[key] = shifted
        return adjusted

    def _project_points_to_ndc(self, points_xyz, full_proj_transform):
        """
        Project world-space points onto the normalized device plane.
        """
        h, w, _ = points_xyz.shape
        ones = torch.ones_like(points_xyz[..., :1])
        points_h = torch.cat([points_xyz, ones], dim=-1).reshape(-1, 4)
        proj = torch.matmul(points_h, full_proj_transform.transpose(0, 1))
        clip_w = proj[..., 3:4].clamp(min=1e-6)
        ndc = (proj[..., :2] / clip_w).reshape(h, w, 2)
        return torch.nan_to_num(ndc, nan=0.0, posinf=0.0, neginf=0.0)

    def set_force_full_opacity_render(self, enable: bool):
        """
        Toggle whether subsequent renders should clamp opacity to 1.
        """
        self.force_full_opacity_render = bool(enable)

    def _render_batch(self, render_dict, batch_dict_list, force_full_opacity: Optional[bool] = None):
        """
        Wrapper over renderer.render_batch with optional opacity override.
        """
        use_override = self.force_full_opacity_render if force_full_opacity is None else force_full_opacity
        if use_override and 'opacity' in render_dict:
            render_dict = render_dict.copy()
            render_dict['opacity'] = torch.ones_like(render_dict['opacity'])
        return self.renderer.render_batch(render_dict, batch_dict_list)


    def compute_all_losses(self, 
                           batch,
                           w_rgb=1,
                           w_depth_loss=1.,
                           w_depth_range=10,
                           w_distortion=1.,
                           w_scene_flow_smooth=10.,
                           w_canonical_unit_sphere=0.,
                           w_flow_grad=0.01,
                           write_logs=True,
                           return_data=False,
                           log_prefix='loss',
                           ):
        ids1 = batch['ids1'].to(self.device)
        ids2 = batch['ids2'].to(self.device)
        # px1s = batch['pts1'].to(self.device)
        # px2s = batch['pts2'].to(self.device)
        gt_rgb1 = batch['gt_rgb1'].to(self.device)
        # weights = batch['weights'].to(self.device)
        loss_cfg = self.cfg.loss

        # Step 1: render frames given two time index (for flow and rgb)
        render_dict = self.gs_atlases_model.forward(ids1)
        render_dict2 = self.gs_atlases_model.forward(ids2)
        
        track_gs = render_dict2["position"]
        render_dict.update({"track_gs": track_gs})  # TODO add flow
        ##### use copy of batch_dict to avoid changing the original batch_dict of "pixel flow"
        batch_dict_copy = self.batch_dict.copy()
        batch_dict_copy.update({"render_attributes_list" : ["track_gs"]+list(self.cfg.render_attributes.keys())})
        batch_dict_copy.update({"num_idx": 10})

        render_results = self.renderer.render_batch(render_dict, [batch_dict_copy])
        # pred_flow = render_results['pixel_flow'][0].permute(1,2,0).reshape(1,-1,2)
        # optical_flow_loss = masked_l1_loss(pred_flow, px2s - px1s, weights, normalize=False)
        
        pred_rgb1 = render_results['rgb'][0].permute(1,2,0).reshape(1,-1,3)
        # imageio.imwrite("./debug.png", (pred_rgb1.reshape(self.h,self.w,3).detach().cpu().numpy()*255).astype(np.uint8))
        # loss_rgb = F.mse_loss(pred_rgb1, gt_rgb1)
        L1_loss_rgb = l1_loss(pred_rgb1.reshape(-1,self.h,self.w,3), gt_rgb1.reshape(-1,self.h,self.w,3))
        ssim_loss = 1 - ssim(pred_rgb1.reshape(-1,self.h,self.w,3), gt_rgb1.reshape(-1,self.h,self.w,3))
        lambda_dssim = 0.2
        loss_rgb = (1.0 - lambda_dssim) * L1_loss_rgb + lambda_dssim * ssim_loss
        loss_rgb = self.args.loss_rgb_weight * loss_rgb
        loss = loss_rgb
        
        loss_details = {
            'L1_loss_rgb_raw': L1_loss_rgb.item(),
            'ssim_loss_raw': ssim_loss.item(),
            'loss_rgb_weighted': loss_rgb.item(),
        }
        if loss_cfg.pos_curvature or loss_cfg.rot_curvature or loss_cfg.scale_curvature:
            # After compute_all_losses aggregation, add curvature regularization.
            pc = self.gs_atlases_model.get_atlas(self.gs_atlas_cfg_list[0].name).point_cloud
            if loss_cfg.pos_curvature:
                yk_pos_z = pc.get_pos_cubic_node.view(-1, pc.pos_interval_num, 3)[..., 2]
                yk_pos_z = yk_pos_z.reshape(-1, pc.pos_interval_num, 1)
                tk = pc.pos_tk  # (M,)
                loss_pos_curv = curvature_penalty_nonuniform(yk_pos_z, tk) * 0.005
                loss += loss_pos_curv
                loss_details['loss_pos_curv_weighted'] = loss_pos_curv.item()
            else:
                loss_details['loss_pos_curv_weighted'] = 0.0

            if loss_cfg.rot_curvature:
                tk = pc.rot_tk
                yk_rot = pc.get_rot_cubic_node.view(-1, pc.rot_interval_num, 3)
                loss_rot_curv = curvature_penalty_nonuniform(yk_rot, tk) * 0.001
                loss += loss_rot_curv
                loss_details['loss_rot_curv_weighted'] = loss_rot_curv.item()
            else:
                loss_details['loss_rot_curv_weighted'] = 0.0

            if (
                loss_cfg.scale_curvature
                and hasattr(pc, "scale_tk")
                and hasattr(pc, "get_scale_cubic_node")
                and hasattr(pc, "scale_interval_num")
            ):
                tk = pc.scale_tk
                yk_scale = pc.get_scale_cubic_node.view(-1, pc.scale_interval_num, 3)
                loss_scale_curv = curvature_penalty_nonuniform(yk_scale, tk) * 0.05
                loss += loss_scale_curv
                loss_details['loss_scale_curv_weighted'] = loss_scale_curv.item()
            else:
                loss_details['loss_scale_curv_weighted'] = 0.0
        else:
            loss_details['loss_pos_curv_weighted'] = 0.0
            loss_details['loss_rot_curv_weighted'] = 0.0
            loss_details['loss_scale_curv_weighted'] = 0.0

        if loss_cfg.optical_flow:
            predicted_track_xyz = render_results['track_gs'].permute(0, 2, 3, 1)  # [B, H, W, 3]
            proj_matrix = self.batch_dict["full_proj_transform"].to(self.device)
            if proj_matrix.dim() == 2:
                proj_matrix = proj_matrix.unsqueeze(0)
            proj_matrix = proj_matrix.expand(predicted_track_xyz.shape[0], -1, -1)

            optical_flow_loss = torch.tensor(0.0, device=self.device)
            valid_batch = 0
            h, w = self.h, self.w
            for b_idx in range(ids1.shape[0]):
                id1 = ids1[b_idx].item()
                id2 = ids2[b_idx].item()
                frame_interval = torch.abs(ids2[b_idx] - ids1[b_idx]).float()
                w_interval = torch.exp(-2 * frame_interval / self.num_imgs)

                predicted_uv_ndc = self._project_points_to_ndc(
                    predicted_track_xyz[b_idx],
                    proj_matrix[b_idx]
                ).clamp_(-1.5, 1.5)

                query_tracks_2d = self.video_3d_flow.load_target_tracks(id1, [id1])[:, 0, :2].to(self.device)
                target_tracks = self.video_3d_flow.load_target_tracks(id1, [id2], dim=0).to(self.device)

                query_in_bounds = (
                    (query_tracks_2d[:, 0] >= 0)
                    & (query_tracks_2d[:, 0] <= w - 1)
                    & (query_tracks_2d[:, 1] >= 0)
                    & (query_tracks_2d[:, 1] <= h - 1)
                )
                target_coords = target_tracks[..., :2][0]
                target_in_bounds = (
                    (target_coords[:, 0] >= 0)
                    & (target_coords[:, 0] <= w - 1)
                    & (target_coords[:, 1] >= 0)
                    & (target_coords[:, 1] <= h - 1)
                )
                valid_tracks = query_in_bounds & target_in_bounds
                if not torch.any(valid_tracks):
                    continue

                query_tracks_2d = query_tracks_2d[valid_tracks]
                target_tracks = target_tracks[:, valid_tracks]
                gt_tracks_2d = target_tracks[..., :2].reshape(-1, 2)
                target_visibles, _, target_confidences = parse_tapir_track_info(
                    target_tracks[..., 2],
                    target_tracks[..., 3],
                    threshold=0.6,
                )
                gt_visibles = target_visibles.reshape(-1)
                gt_confidences = target_confidences.reshape(-1)

                track_weights = (gt_confidences * w_interval).unsqueeze(-1)

                predicted_uv_map = predicted_uv_ndc.permute(2, 0, 1).unsqueeze(0)
                query_norm = util.normalize_coords(query_tracks_2d, self.h, self.w)
                grid = query_norm.view(1, -1, 1, 2)
                sampled_pred_norm = F.grid_sample(
                    predicted_uv_map,
                    grid,
                    mode='bilinear',
                    padding_mode='border',
                    align_corners=True,
                ).permute(0, 2, 3, 1).reshape(-1, 2)
                sampled_pred_px = util.denormalize_coords(sampled_pred_norm, self.h, self.w)

                visible_mask = gt_visibles & (track_weights.squeeze(-1) > 0)
                if not torch.any(visible_mask):
                    continue

                per_loss = masked_l1_loss(
                    sampled_pred_px[visible_mask],
                    gt_tracks_2d[visible_mask],
                    mask=track_weights[visible_mask],
                    quantile=0.98,
                ) / max(self.h, self.w)
                optical_flow_loss += per_loss
                valid_batch += 1

            if valid_batch > 0:
                optical_flow_loss = optical_flow_loss / valid_batch
            else:
                optical_flow_loss = torch.tensor(0.0, device=self.device)

            optical_flow_loss = self.args.loss_flow_weight * optical_flow_loss
            loss += optical_flow_loss
            loss_details['optical_flow_loss_weighted'] = optical_flow_loss.item()
            if optical_flow_loss.item() < 0:
                breakpoint()
        else:
            loss_details['optical_flow_loss_weighted'] = 0.0

        if loss_cfg.depth:
            depth = render_results['depth'][0].permute(1,2,0)  # [h,w,1]
            gt_depth = self.gt_depths[ids1][0,...,None]
            from loss import depth_loss_dpt
            loss_depth = depth_loss_dpt(depth, gt_depth)
            loss_depth = loss_depth * w_depth_loss
            loss += loss_depth
            loss_details['loss_depth_weighted'] = loss_depth.item()
        else:
            loss_details['loss_depth_weighted'] = 0.0

        # ### add metric depth loss
        # metric_depth = self.video_3d_flow.depths[ids1][...,None].to(self.device)
        # loss_metric_depth = F.mse_loss(depth, metric_depth)
        # loss += loss_metric_depth

        # Step 5: constrain opacity of each gaussian
        # loss += torch.nn.functional.l1_loss(render_dict['opacity'], torch.zeros_like(render_dict['opacity'])) * 0.001
        ### adopt entropy loss
        # loss_entropy = entropy_loss(render_dict['opacity'], render_results['gs_idx'])
        # loss_entropy = torch.nn.functional.binary_cross_entropy(render_dict['opacity'], render_dict['opacity'])
        # loss += loss_entropy * 0.1

        # assume a bell function for the opacity

        ############# This is used to test topK rendering
        if False:
            from pointrix.utils.sh_utils import eval_sh
            # attribute = uv2-uv1
            pixel_weight, loss_entropy = entropy_loss(render_dict['opacity'], render_results['gs_idx'])
            loss += loss_entropy * 0.01
            attribute = render_dict['mask_attribute']
            test_render_result = alpha_blending_firstK(attribute,  
                                                    render_results['gs_idx'],
                                                    pixel_weight,
                                                    bg=0)

        # Step 6: 
        #### loss for the attributes
        if False:
            rendered_mask = render_results['mask_attribute'][0].permute(1,2,0)  # [h,w,1]
            gt_mask = self.masks[ids1][0,...,None]
            loss_mask_attribute = F.mse_loss(rendered_mask, gt_mask)
            # loss_mask_attribute = F.binary_cross_entropy(rendered_mask, gt_mask)
            loss += loss_mask_attribute * 10.0
            loss_details['loss_mask_attribute_weighted'] = loss_mask_attribute.item()

        if False:
            rendered_dino = render_results['dino_attribute'][0].permute(1,2,0)  # [h,w,3]
            gt_dino = self.dinos[ids1][0]
            loss_dino_attribute = F.mse_loss(rendered_dino, gt_dino)
            loss += loss_dino_attribute * 5

        ######## extract GS layer based on attributes
        fg_gs_mask = (render_dict['mask_attribute'].squeeze() > 0.5).detach()
        # render_dict_fg_layer1 = {k: v[fg_gs_mask] for k, v in render_dict.items()}

        # candidate_list = list(np.arange(self.num_imgs))
        # candidate_list.remove(ids1.item())
        # ids2 = torch.tensor([np.random.choice(candidate_list)]).to(self.device)
        # render_dict2 = self.gs_atlases_model.forward(ids2)
        render_dict_fg_layer2 = {k: v[fg_gs_mask] for k, v in render_dict2.items()}

        ### avoid reduandant mask
        if self.step > 100 and False:
            batch_dict_copy = self.batch_dict.copy()
            batch_dict_copy.update({"render_attributes_list" : ['mask_attribute'],
                                    "bg_color" : 0.0})
            render_results_layer1 = self.renderer.render_batch(render_dict_fg_layer1, [batch_dict_copy])
            # imageio.imwrite("./debug.png", (render_results_layer1['mask_attribute'][0,0].clamp(0,1).detach().cpu().numpy()*255).astype(np.uint8))

            pred_rgb1_fg = render_results_layer1['rgb'][0].permute(1,2,0).reshape(1,-1,3)
            # imageio.imwrite("./debug.png", (pred_rgb1.reshape(self.h,self.w,3).detach().cpu().numpy()*255).astype(np.uint8))
            loss_rgb_fg = F.mse_loss(pred_rgb1_fg, gt_rgb1*(gt_mask.reshape(1,-1,1)))
            loss += loss_rgb_fg * 20
            
            loss_mask_attribute_fg = F.mse_loss(render_results_layer1['mask_attribute'][0].permute(1,2,0), gt_mask)
            loss += loss_mask_attribute_fg * 20
        
        
        if False:
            # ### add rigid constraint
            from utils.geometry_utils import cal_connectivity_from_points, cal_arap_error, cal_smooth_error
            ii, jj, nn, weight = cal_connectivity_from_points(points=render_dict["position"], K=5)
            pos = torch.stack([render_dict["position"], render_dict2["position"]], dim=0)
            rigid_error = cal_arap_error(pos, ii, jj, nn) / 1000.  # this loss is too large
            loss += rigid_error


        if False:
            ### split GS layer and constrain their motion seperately
            num_gs = render_dict['opacity'].shape[0]
            pixel_gs_idx = pixel2gs(render_dict['opacity'], render_results['gs_idx'])
            selected_gs_idx = pixel_gs_idx[self.masks[ids1] > 0]  # gs correspondent to the mask region
            selected_gs_idx = selected_gs_idx[selected_gs_idx != num_gs]  # empty pixel is filled with invalid index
            selected_gs_idx = torch.unique(selected_gs_idx)
            if selected_gs_idx.max() >= render_dict['opacity'].shape[0]:
                import ipdb; ipdb.set_trace()
            render_dict_layer1 = {k: v[selected_gs_idx] for k, v in render_dict.items()}

            complementary_gs_idx = torch.tensor(list(set(range(num_gs)) - set(selected_gs_idx))).to(selected_gs_idx.device)
            render_dict_layer2 = {k: v[complementary_gs_idx] for k, v in render_dict.items()}
            
        loss_details['total_loss'] = loss.item()
        
        data = {
            'ids1': ids1,
            'ids2': ids2,
            'loss_details': loss_details,
        }
        
        data.update(render_results)
        if return_data:
            return loss, data
        else:
            return loss


    def weight_scheduler(self, step, start_step, w, min_weight, max_weight):
        if step <= start_step:
            weight = 0.0
        else:
            weight = w * (step - start_step)
        weight = np.clip(weight, a_min=min_weight, a_max=max_weight)
        return weight
    
    
    def train_one_step(self, step, batch):
        self.step = step
        start = time.time()
        self.scalars_to_log = {}
        loss_cfg = self.cfg.loss

        w_rgb = self.weight_scheduler(step, 0, 1./5000, 0, 10)
        w_depth_loss = self.weight_scheduler(step, 0, 1./2000, 0.1, 10.0) if loss_cfg.depth else 0.0
        w_flow_grad = self.weight_scheduler(step, 0, 1./500000, 0, 0.1)
        w_distortion = self.weight_scheduler(step, 40000, 1./2000, 0, 10)
        w_scene_flow_smooth = 20.

        # self.renderer.update_sh_degree(iteration)  # TODO add later
        loss, render_results = self.compute_all_losses(batch,
                                                  w_rgb=w_rgb,
                                                  w_depth_loss=w_depth_loss,
                                                  w_scene_flow_smooth=w_scene_flow_smooth,
                                                  w_distortion=w_distortion,
                                                  w_flow_grad=w_flow_grad,
                                                  return_data=True)
        
        if torch.isnan(loss):
            pdb.set_trace()

        loss.backward()

        ### TODO GS optimizer
        ##### TODO parse viewspace_points, visibility, radii to each GS atlas
        self.optimizer_dict = self.gs_atlases_model.prepare_optimizer_dict(loss, render_results)
        for gs_atlas_cfg in self.gs_atlas_cfg_list:
            name = gs_atlas_cfg.name
            getattr(self, name+"_optimizer").update_model(**self.optimizer_dict[name])
            getattr(self, name+"_scheduler").step(self.step, getattr(self, name+"_optimizer"))

            for x in getattr(self, name+"_optimizer").param_groups:
                self.scalars_to_log['lr_'+x['name']] = x['lr']
        
        # Add detailed loss values to scalars_to_log.
        if 'loss_details' in render_results:
            for loss_name, loss_value in render_results['loss_details'].items():
                self.scalars_to_log[loss_name] = loss_value
        else:
            self.scalars_to_log['loss'] = loss.item()


        self.scalars_to_log['w_depth_loss'] = w_depth_loss
        self.scalars_to_log['time'] = time.time() - start
        self.ids1 = render_results['ids1']
        self.ids2 = render_results['ids2']

    
    def get_pred_flows_gs(self, ids1, ids2, return_original=False, clip_flow=None):
        with torch.no_grad():
            render_dict = self.gs_atlases_model.forward(ids1)
            render_dict2 = self.gs_atlases_model.forward(ids2)
            if self.enable_ortho_projection:
                (uv1, depth1) = self.renderer.project_point(
                    render_dict["position"],
                    self.batch_dict["extrinsic_matrix"].cuda(),
                    self.batch_dict["width"], 
                    self.batch_dict["height"],
                    nearest=0.01)
                
                (uv2, depth2) = self.renderer.project_point(
                    render_dict2["position"],
                    self.batch_dict["extrinsic_matrix"].cuda(),
                    self.batch_dict["width"], 
                    self.batch_dict["height"],
                    nearest=0.01)
            else:
                (uv1, depth1) = gs.project_point(
                    render_dict["position"],
                    self.batch_dict["intrinsic_matrix"].cuda(),
                    self.batch_dict["extrinsic_matrix"].cuda(),
                    self.batch_dict["width"], 
                    self.batch_dict["height"],
                    nearest=0.01
                    )

                (uv2, depth2) = gs.project_point(
                    render_dict2["position"],
                    self.batch_dict["intrinsic_matrix"].cuda(),
                    self.batch_dict["extrinsic_matrix"].cuda(),
                    self.batch_dict["width"], 
                    self.batch_dict["height"],
                    nearest=0.01
                    )
            render_dict.update({"pixel_flow": uv2-uv1})  # TODO add flow
            batch_dict_copy = self.batch_dict.copy()
            batch_dict_copy.update({"render_attributes_list" : ["pixel_flow"]+list(self.cfg.render_attributes.keys())})
            render_results = self.renderer.render_batch(render_dict, [batch_dict_copy])

        pred_flow = render_results['pixel_flow'][0].permute(1,2,0).cpu().numpy()
        if clip_flow is not None and clip_flow > 0:
            pred_flow = np.clip(pred_flow, -clip_flow, clip_flow)
        flow_imgs = util.flow_to_image(pred_flow)
        if return_original:
            return flow_imgs, pred_flow
        else:
            return flow_imgs  # [n, h, w, 3], numpy arra


    def render_flow_maps(self, stride=1, max_pairs=-1, save_dir=None, save_raw=False, clip_flow=None):
        """
        Render colorized optical flow maps for consecutive frame pairs.

        Args:
            stride (int): Frame interval used to form pairs (i, i + stride).
            max_pairs (int): Limit the number of rendered pairs (-1 renders all).
            save_dir (str): Directory to store the visualizations.
            save_raw (bool): Save the raw flow field as .npy alongside the visualization.
            clip_flow (float): Optional magnitude clipping before colorization.
        """
        stride = max(1, int(stride))
        if save_dir is None:
            save_dir = os.path.join(self.out_dir, 'flow_maps')
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        if self.num_imgs <= stride:
            print(f"[render_flow_maps] Not enough frames (num_imgs={self.num_imgs}, stride={stride}).")
            return []

        pair_indices = [(i, i + stride) for i in range(0, self.num_imgs - stride)]
        if max_pairs > 0:
            pair_indices = pair_indices[:max_pairs]

        if not pair_indices:
            print("[render_flow_maps] No frame pairs to render.")
            return []

        iterator = tqdm(pair_indices, desc="Rendering flow maps") if len(pair_indices) > 1 else pair_indices
        saved_paths = []
        for id1, id2 in iterator:
            ids1 = torch.as_tensor([id1], device=self.device)
            ids2 = torch.as_tensor([id2], device=self.device)
            flow_vis, raw_flow = self.get_pred_flows_gs(ids1, ids2, return_original=True, clip_flow=clip_flow)

            stem = f"flow_{id1 + self.base_idx:05d}_{id2 + self.base_idx:05d}"
            flow_path = os.path.join(save_dir, f"{stem}.png")
            imageio.imwrite(flow_path, flow_vis.astype(np.uint8))
            saved_paths.append(flow_path)

            if save_raw:
                np.save(os.path.join(save_dir, f"{stem}.npy"), raw_flow)

        print(f"[render_flow_maps] Saved {len(saved_paths)} flow maps to {save_dir}.")
        return saved_paths


    
    def get_pred_color_and_depth_maps_gs(self, ids):
        pred_rgbs, pred_depths = [], [] 
        for id in ids:
            render_dict = self.gs_atlases_model.forward(id)
            render_results = self.renderer.render_batch(render_dict, [self.batch_dict])

            pred_rgbs.append(render_results['rgb'].permute(0,2,3,1).clamp(0,1))
            pred_depths.append(render_results['depth'].permute(0,2,3,1))
        
        return torch.cat(pred_rgbs, dim=0), torch.cat(pred_depths, dim=0)  # [n, h, w, 3/1]
    
    
    def log(self, writer, step):
        if self.args.local_rank == 0:
            if step % self.args.i_print == 0:
                logstr = '{}_{} | step: {} |'.format(self.args.expname, self.seq_name, step)
                
                # Print each loss value (weighted contributions).
                weighted_loss_keys = ['total_loss', 'loss_rgb_weighted', 'loss_pos_curv_weighted', 
                                    'loss_rot_curv_weighted', 'loss_scale_curv_weighted', 
                                    'optical_flow_loss_weighted', 'loss_depth_weighted', 
                                    'loss_mask_attribute_weighted']
                
                print(f"\n=== Loss Details at Step {step} ===")
                print("--- Weighted Loss Contributions (actual contribution to total loss) ---")
                for loss_key in weighted_loss_keys:
                    if loss_key in self.scalars_to_log:
                        print(f"{loss_key}: {self.scalars_to_log[loss_key]:.6f}")
                        if loss_key != 'time':
                            writer.add_scalar(loss_key, self.scalars_to_log[loss_key], step)
                
                # Print other info (learning rate, time, etc.).
                all_loss_keys = weighted_loss_keys
                other_keys = [k for k in self.scalars_to_log.keys() if k not in all_loss_keys]
                if other_keys:
                    print(f"\n=== Other Info ===")
                    for k in other_keys:
                        print(f"{k}: {self.scalars_to_log[k]:.6f}")
                        if k != 'time':
                            writer.add_scalar(k, self.scalars_to_log[k], step)
                
                print("=" * 50)

            if step % self.args.i_img == 0 and False:
                ids = torch.cat([self.ids1[0:1], self.ids2[0:1]])
                with torch.no_grad():
                    pred_imgs, pred_depths = self.get_pred_color_and_depth_maps_gs(ids)
                    pred_imgs = pred_imgs.cpu()
                    pred_depths = pred_depths[...,0].cpu()

                # write depth maps
                pred_depths_cat = pred_depths.permute(1, 0, 2).reshape(self.h, -1)
                min_depth = pred_depths_cat.min().item()
                max_depth = pred_depths_cat.max().item()

                pred_depths_vis = util.colorize(pred_depths_cat, range=(min_depth, max_depth), append_cbar=True)
                pred_depths_vis = F.interpolate(pred_depths_vis.permute(2, 0, 1)[None], scale_factor=0.5, mode='area')
                writer.add_image('depth', pred_depths_vis, step, dataformats='NCHW')

                # write gt and predicted rgbs
                gt_imgs = self.images[ids.cpu()]
                imgs_vis = torch.cat([gt_imgs, pred_imgs], dim=1)
                imgs_vis = F.interpolate(imgs_vis.permute(0, 3, 1, 2), scale_factor=0.5, mode='area')
                writer.add_images('images', imgs_vis, step, dataformats='NCHW')

                # write flow
                with torch.no_grad():
                    flows = self.get_pred_flows_gs(self.ids1[0:1], self.ids2[0:1])
                    id1, id2 = self.ids1[0], self.ids2[0]
                    gt_flow = np.load(os.path.join(self.seq_dir, 'raft_exhaustive',
                                                   '{}_{}.npy'.format(os.path.basename(self.img_files[id1]),
                                                                      os.path.basename(self.img_files[id2]))
                                                   ))
                    gt_flow_img = util.flow_to_image(gt_flow)
                    
                writer.add_image('flow', np.concatenate([flows, gt_flow_img], axis=1), step, dataformats='HWC')

            if step % self.args.i_weight == 0 and step > 0:
                vis_dir = os.path.join(self.out_dir, 'vis')
                os.makedirs(vis_dir, exist_ok=True)
                print('saving visualizations to {}...'.format(vis_dir))
                if False:
                    images, depths = [], []
                    for id in range(self.num_imgs):
                        with torch.no_grad():
                            render_dict = self.gs_atlases_model.forward(id)
                            render_results = self.renderer.render_batch(render_dict, [self.batch_dict])
                            pred_rgb = render_results['rgb'][0].permute(1,2,0).clamp(0,1).cpu().numpy()
                            pred_depth = render_results['depth'][0].permute(1,2,0).cpu().numpy()
                        images.append((pred_rgb*255).astype(np.uint8))
                        depths.append(pred_depth)
                        # depths is processed later

                    imageio.mimwrite(os.path.join(vis_dir, '{}_rgb{:06d}.mp4'.format(self.seq_name, step)),
                                        images,
                                        quality=8, fps=4)
                    depths_np = np.stack(depths, axis=0)
                    depths_np = (depths_np - depths_np.min()) / (depths_np.max() - depths_np.min())
                    depths_np = (depths_np * 255).astype(np.uint8)
                    imageio.mimwrite(os.path.join(vis_dir, '{}_depth{:06d}.mp4'.format(self.seq_name, step)),
                                        depths_np,
                                        quality=8, fps=4)
                ### save tracking result
                save_dir = os.path.join(self.out_dir, 'tracking')
                os.makedirs(save_dir, exist_ok=True)
                # for i, img in enumerate(track_imgs):
                #     imageio.imwrite(os.path.join(save_dir, f'{step}_{i}.png'), img)
                # track_imgs = self.draw_pixel_trajectory(use_mask=True)
                # track_imgs = [x[:,self.w:] for x in track_imgs]
                # imageio.mimwrite(os.path.join(save_dir, f'{step}.mp4'), track_imgs, fps=10)
                # track_imgs = self.draw_pixel_trajectory(use_mask=False)
                # track_imgs = [x[:,self.w:] for x in track_imgs]
                # imageio.mimwrite(os.path.join(save_dir, f'{step}_no_mask.mp4'), track_imgs, fps=10)

                ### save video render result
                self.render_video(step, save_frames=True)
                self.get_interpolation_result(scaling=4, save_path=os.path.join(vis_dir, f'interp_{step:06d}.mp4'))

                # self.render_part(fg=True, threshold=0.5)
                # self.render_part(fg=False, threshold=0.5)
                
                fpath = os.path.join(self.out_dir, 'model_{:06d}.pth'.format(step))
                self.save_model(fpath)
                

    def save_model(self, path: Path = None) -> None:
        data_list = {
            "gs_atlases_model": self.gs_atlases_model.get_state_dict(),
            "renderer": self.renderer.state_dict()
        }
        data_list.update(
            {k.name+"_optimizer": getattr(self,k.name+"_optimizer").state_dict() for k in self.gs_atlas_cfg_list}
        )
        # data_list.update(
        #     {k.name+"_scheduler": getattr(self,k.name+"_scheduler").state_dict() for k in self.gs_atlas_cfg_list}
        # )
        torch.save(data_list, path)


    def load_model(self, path: Path = None) -> None:
        data_list = torch.load(path)
        for k, v in data_list.items():
            print(f"Loaded {k} from checkpoint")
            # get arrtibute from model
            arrt = getattr(self, k)
            if hasattr(arrt, 'load_state_dict'):
                arrt.load_state_dict(v)
            else:
                setattr(self, k, v)
        ### TODO fix this
        # re-initialize optimizer
        cameras_extent = 5  # the extent of the camera
        for gs_atlas_cfg in self.gs_atlas_cfg_list:
            name = gs_atlas_cfg.name
            setattr(self, name+"_optimizer", 
                    parse_optimizer(self.cfg.optimizer, 
                                    model=self.gs_atlases_model.get_atlas(name),
                                    cameras_extent=cameras_extent)
                    )
            setattr(self, name+"_scheduler",
                    parse_scheduler(self.cfg.scheduler, 
                                    cameras_extent if self.cfg.spatial_lr_scale else 1.)
                    )


    def load_from_ckpt(self, out_folder,
                       load_opt=True,
                       load_scheduler=True,
                       force_latest_ckpt=False):
        '''
        load model from existing checkpoints and return the current step
        :param out_folder: the directory that stores ckpts
        :return: the current starting step
        '''

        # all existing ckpts
        ckpts = []
        if os.path.exists(out_folder):
            ckpts = [os.path.join(out_folder, f)
                     for f in sorted(os.listdir(out_folder)) if f.endswith('.pth')]

        if self.args.ckpt_path is not None and not force_latest_ckpt:
            if os.path.isfile(self.args.ckpt_path):  # load the specified ckpt
                ckpts = [self.args.ckpt_path]

        if len(ckpts) > 0 and not self.args.no_reload:
            fpath = ckpts[-1]
            self.load_model(fpath)
            step = int(fpath[-10:-4])
            print('Reloading from {}, starting at step={}'.format(fpath, step))
        else:
            print('No ckpts found, from scratch...')
            step = 0

        return step
    
    def optimize_appearance_from_mask(self, mask_path, img_path):
        mask = imageio.imread(mask_path)
        if mask.ndim == 3:
            mask = mask[...,0]
        # mask = cv2.dilate(mask, np.ones((5,5), np.uint8), iterations=1)
        mask = mask / 255.
        mask = torch.tensor(mask).float().to(self.device)  # [H,W]

        gt_image = imageio.imread(img_path)[...,:3] / 255.
        gt_image = torch.tensor(gt_image).float().to(self.device)

        render_dict = self.gs_atlases_model.forward(0)
        batch_dict_copy = self.batch_dict.copy()
        batch_dict_copy.update({"num_idx" : 10})
        render_results = self.renderer.render_batch(render_dict, [batch_dict_copy])
        selected_gs_idx = torch.unique(render_results['gs_idx'][0][mask > 0])
        selected_gs_idx = selected_gs_idx[selected_gs_idx != -1]

        ### construct an optimizer
        from pointrix.utils.gaussian_points.gaussian_utils import inverse_sigmoid
        optimized_shs = nn.Parameter(render_dict['shs'][selected_gs_idx].detach(), requires_grad=True)
        optimized_opacity = nn.Parameter(inverse_sigmoid(render_dict['opacity'][selected_gs_idx]).detach(), requires_grad=True)
        parameters = [{'params': optimized_shs, 'lr': 0.0025}, 
                    #   {'params': optimized_opacity, 'lr': 0.05}
                      ]
        optim = torch.optim.Adam(parameters)
        # optim = torch.optim.SGD(
        #     [optimized_shs, optimized_opacity], 
        #     lr=0.05)
        # scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=500, gamma=0.99999)
        
        for idx in tqdm(range(1000), mininterval=100):
            with torch.no_grad():
                render_dict = self.gs_atlases_model.forward(0)
            render_dict['shs'][selected_gs_idx] = optimized_shs
            render_dict['opacity'][selected_gs_idx] = torch.sigmoid(optimized_opacity)
            render_results = self.renderer.render_batch(render_dict, [batch_dict_copy])
            
            new_size = render_results['rgb'][0].shape[1:]
            gt_image_resize = F.interpolate(gt_image.permute(2, 0, 1).unsqueeze(0), size=new_size, mode='bilinear', align_corners=True)
            
            loss = F.mse_loss(render_results['rgb'][0].permute(1,2,0), gt_image_resize[0].permute(1,2,0))
            optim.zero_grad()
            loss.backward()
            optim.step()
            # scheduler.step()
            if idx % 10 == 0:
                print(f'loss: {loss.item()}')
            if loss.item() < 0.0001:
                break

        images, depths = [], []
        for id in range(self.num_imgs):
            with torch.no_grad():
                render_dict = self.gs_atlases_model.forward(id)
                render_dict['shs'][selected_gs_idx] = optimized_shs
                render_dict['opacity'][selected_gs_idx] = torch.sigmoid(optimized_opacity)
                render_results = self.renderer.render_batch(render_dict, [batch_dict_copy])
                pred_rgb = render_results['rgb'][0].permute(1,2,0).clamp(0,1).cpu().numpy()
                pred_depth = render_results['depth'][0].permute(1,2,0).clamp(0,1).cpu().numpy()
            images.append((pred_rgb*255).astype(np.uint8))
            depths.append((pred_depth*255).astype(np.uint8))

        imageio.mimwrite(os.path.join('{}_mask_editing.mp4'.format(self.seq_name)),
                            images,
                            quality=8, fps=4)
        print()


    

    def optimize_appearance_from_img(self, img_paths):
        if len(img_paths) == 0:
            raise ValueError("img_paths must contain at least one image path.")

        def _infer_frame_idx(path: str) -> int:
            stem = os.path.splitext(os.path.basename(path))[0]
            matches = re.findall(r'\d+', stem)
            if matches:
                return int(matches[-1])
            return 0

        gt_entries = []
        pc = self.gs_atlases_model.get_atlas(self.gs_atlas_cfg_list[0].name).point_cloud
        for img_path in img_paths:
            gt_image = imageio.imread(img_path)[...,:3] / 255.
            gt_image = torch.tensor(gt_image).float().to(self.device)
            frame_idx = _infer_frame_idx(img_path)
            if hasattr(self, "num_imgs"):
                frame_idx = max(0, min(int(frame_idx), max(self.num_imgs - 1, 0)))
            if frame_idx < self.num_imgs:
                gt_entries.append((frame_idx, gt_image))

        with torch.no_grad():
            render_dict = self.gs_atlases_model.forward(0)
            batch_dict_copy = self.batch_dict.copy()
            batch_dict_copy.update({"num_idx" : 10})
            # _ = self.renderer.render_batch(render_dict, [batch_dict_copy])


        ### construct an optimizer
        from pointrix.utils.gaussian_points.gaussian_utils import inverse_sigmoid
        optimized_shs = nn.Parameter(render_dict['shs'].detach(), requires_grad=True)
        # optimized_opacity = nn.Parameter(inverse_sigmoid(render_dict['opacity']).detach(), requires_grad=True)
        parameters = [{'params': optimized_shs, 'lr': 0.0025}, 
                    #   {'params': optimized_opacity, 'lr': 0.05}
                      ]
        optim = torch.optim.Adam(parameters)
        # optim = torch.optim.SGD(
        #     [optimized_shs, optimized_opacity], 
        #     lr=0.05)
        # scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=500, gamma=0.99999)
        
        for idx in tqdm(range(2000)):
            # losses = []
            frame_idx = gt_entries[idx % len(gt_entries)][0]
            gt_image = gt_entries[idx% len(gt_entries)][1]
            with torch.no_grad():
                render_dict = self.gs_atlases_model.forward(frame_idx)
            render_dict['shs'] = optimized_shs
            render_results = self.renderer.render_batch(render_dict, [batch_dict_copy])
            new_size = render_results['rgb'][0].shape[1:]
            gt_image_resize = F.interpolate(gt_image.permute(2, 0, 1).unsqueeze(0), size=new_size, mode='bilinear', align_corners=True)
            loss = F.mse_loss(render_results['rgb'][0].permute(1,2,0), gt_image_resize[0].permute(1,2,0))
            # loss = torch.stack(losses).mean()
            optim.zero_grad()
            loss.backward()
            optim.step()
            # scheduler.step()
            # if idx % 10 == 0:
            #     print(f'loss: {loss.item()}')
            if loss.item() < 0.0001:
                break

        images, depths = [], []
        for id in tqdm(range(self.num_imgs)):
            with torch.no_grad():
                render_dict = self.gs_atlases_model.forward(id)
                render_dict['shs'] = optimized_shs
                # render_dict['opacity'] = torch.sigmoid(optimized_opacity)
                render_results = self.renderer.render_batch(render_dict, [batch_dict_copy])
                pred_rgb = render_results['rgb'][0].permute(1,2,0).clamp(0,1).cpu().numpy()
                # pred_depth = render_results['depth'][0].permute(1,2,0).clamp(0,1).cpu().numpy()
            images.append((pred_rgb*255).astype(np.uint8))
            # depths.append((pred_depth*255).astype(np.uint8))

        imageio.mimwrite(os.path.join('editing', '{}_editing.mp4'.format(self.seq_name)),
                            images,
                            quality=8, fps=4)
        print()

        
    def get_nvs_rendered_imgs(self,):            
        ##### project to frame space
        radius = 0.05
        z_center = 1.
        color_frame_list = []
        for idx, phi in enumerate(torch.linspace(0, 4 * np.pi, self.num_imgs)):
            render_dict = self.gs_atlases_model.forward(idx)
            camera_position = torch.tensor([[radius*torch.cos(phi), radius*torch.sin(phi), 0.0]], device=self.device)
            camra_rotation = look_at_rotation(camera_position, at=((0,0,z_center),), device=self.device)
            c2w = torch.cat([camra_rotation[0], camera_position.T], dim=1).cpu().numpy()
            c2w = np.concatenate([c2w, np.array([[0,0,0,1]])], axis=0)
            camera = construct_canonical_camera(width=self.w, height=self.h, c2w=c2w)

            batch_dict = {
                "camera": camera,
                "FovX": camera.fovX,
                "FovY": camera.fovY,
                "height": int(camera.image_height),
                "width": int(camera.image_width),
                "world_view_transform": camera.world_view_transform,
                "full_proj_transform": camera.full_proj_transform,
                "extrinsic_matrix": camera.extrinsic_matrix,
                "intrinsic_matrix": camera.intrinsic_matrix,
                "camera_center": camera.camera_center,
            }

            with torch.no_grad():
                render_results = self.renderer.render_batch(render_dict, [batch_dict])
            color_frame_list.append(render_results['rgb'][0].clamp(0,1).permute(1,2,0).cpu().numpy())
        color_frame_list = [(x*255).astype(np.uint8) for x in color_frame_list]
        save_path = os.path.join(self.out_dir, 'nvs.mp4')
        imageio.mimwrite(save_path, color_frame_list, fps=4)
        print()


    def get_stereo_rendered_imgs(self,):
        color_frame_list = []
        radius = 0.05
        phi = 0
        import math
        camera_position = torch.tensor([[radius*math.cos(phi), radius*math.sin(phi), 0.0]], device=self.device)
        camra_rotation = look_at_rotation(camera_position, at=((0,0,2.5),), device=self.device)
        c2w = torch.cat([camra_rotation[0], camera_position.T], dim=1).cpu().numpy()
        c2w = np.concatenate([c2w, np.array([[0,0,0,1]])], axis=0)
        camera1 = construct_canonical_camera(width=self.w, height=self.h, c2w=c2w)

        phi = math.pi
        camera_position = torch.tensor([[radius*math.cos(phi), radius*math.sin(phi), 0.0]], device=self.device)
        camra_rotation = look_at_rotation(camera_position, at=((0,0,2.5),), device=self.device)
        c2w = torch.cat([camra_rotation[0], camera_position.T], dim=1).cpu().numpy()
        c2w = np.concatenate([c2w, np.array([[0,0,0,1]])], axis=0)
        camera2 = construct_canonical_camera(width=self.w, height=self.h, c2w=c2w)

        batch_dict1 = {
            "camera": camera1,
            "FovX": camera1.fovX,
            "FovY": camera1.fovY,
            "height": int(camera1.image_height),
            "width": int(camera1.image_width),
            "world_view_transform": camera1.world_view_transform,
            "full_proj_transform": camera1.full_proj_transform,
            "extrinsic_matrix": camera1.extrinsic_matrix,
            "intrinsic_matrix": camera1.intrinsic_matrix,
            "camera_center": camera1.camera_center,
        }

        batch_dict2 = {
            "camera": camera2,
            "FovX": camera2.fovX,
            "FovY": camera2.fovY,
            "height": int(camera2.image_height),
            "width": int(camera2.image_width),
            "world_view_transform": camera2.world_view_transform,
            "full_proj_transform": camera2.full_proj_transform,
            "extrinsic_matrix": camera2.extrinsic_matrix,
            "intrinsic_matrix": camera2.intrinsic_matrix,
            "camera_center": camera2.camera_center,
        }

        matrices = {
            'true': [ [ 0.299, 0.587, 0.114, 0, 0, 0, 0, 0, 0 ], [ 0, 0, 0, 0, 0, 0, 0.299, 0.587, 0.114 ] ],
            'mono': [ [ 0.299, 0.587, 0.114, 0, 0, 0, 0, 0, 0 ], [ 0, 0, 0, 0.299, 0.587, 0.114, 0.299, 0.587, 0.114 ] ],
            'color': [ [ 1, 0, 0, 0, 0, 0, 0, 0, 0 ], [ 0, 0, 0, 0, 1, 0, 0, 0, 1 ] ],
            'halfcolor': [ [ 0.299, 0.587, 0.114, 0, 0, 0, 0, 0, 0 ], [ 0, 0, 0, 0, 1, 0, 0, 0, 1 ] ],
            'optimized': [ [ 0, 0.7, 0.3, 0, 0, 0, 0, 0, 0 ], [ 0, 0, 0, 0, 1, 0, 0, 0, 1 ] ],
        }

        for idx in range(self.num_imgs):
            render_dict = self.gs_atlases_model.forward(idx)
            batch_dict1_copy = batch_dict1.copy()
            batch_dict2_copy = batch_dict2.copy()
            batch_dict1_copy.update({"render_attributes_list" : ['dino_attribute']})
            batch_dict2_copy.update({"render_attributes_list" : ['dino_attribute']})

            with torch.no_grad():
                render_results1 = self.renderer.render_batch(render_dict, [batch_dict1_copy])
                render_results2 = self.renderer.render_batch(render_dict, [batch_dict2_copy])
            img1 = render_results1['rgb'][0].clamp(0,1).permute(1,2,0).cpu().numpy()
            img2 =  render_results2['rgb'][0].clamp(0,1).permute(1,2,0).cpu().numpy()

            if False:
                # color_frame_list.append(img1*0.5 + img2*0.5)
                hsv = cv2.cvtColor((img2*255).astype(np.uint8), cv2.COLOR_RGB2HSV)
                hsv[..., 0] = (hsv[..., 0] + 30) % 180
                img2 = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB) / 255.
                img = img1 * 0.5 + img2 * 0.5
                img = (img*255).astype(np.uint8)
            else:
                color = 'optimized'
                # from PIL import Image
                # left = Image.fromarray((img1*255).astype(np.uint8))
                # right = Image.fromarray((img2*255).astype(np.uint8))
                # width, height = left.size
                # leftMap = left.load()
                # rightMap = right.load()
                # m = matrices[color]

                # for y in range(0, height):
                #     for x in range(0, width):
                #         r1, g1, b1 = leftMap[x, y]
                #         r2, g2, b2 = rightMap[x, y]
                #         leftMap[x, y] = (
                #             int(r1*m[0][0] + g1*m[0][1] + b1*m[0][2] + r2*m[1][0] + g2*m[1][1] + b2*m[1][2]),
                #             int(r1*m[0][3] + g1*m[0][4] + b1*m[0][5] + r2*m[1][3] + g2*m[1][4] + b2*m[1][5]),
                #             int(r1*m[0][6] + g1*m[0][7] + b1*m[0][8] + r2*m[1][6] + g2*m[1][7] + b2*m[1][8])
                #         )
                # img = np.array(left)
                m = np.array(matrices[color]).reshape(2,3,3).transpose(1,0,2).reshape(3,6)
                img_cat = np.concatenate([img1, img2], axis=2)  # [H, W, 6]
                img = np.einsum('ijk,lk->ijl', img_cat, m)
                img = (img*255).astype(np.uint8)

            color_frame_list.append(img)

        # color_frame_list = [(x*255).astype(np.uint8) for x in color_frame_list]
        color_frame_list = [x for x in color_frame_list]
        save_path = os.path.join(self.out_dir, 'stereo.mp4')
        imageio.mimwrite(save_path, color_frame_list, fps=4)
        print()


    def _prepare_depth_grayscale(self, depth_list):
        depths_np = np.stack(depth_list, axis=0)
        if depths_np.ndim == 4:
            depths_np = depths_np[..., 0]
        depth_max = float(depths_np.max())
        depth_min = float(depths_np.min())
        denom = depth_max - depth_min
        if denom < 1e-8:
            denom = 1.0
        norm_depth = (depths_np - depth_min) / denom
        norm_depth = np.clip(norm_depth, 0.0, 1.0)
        depth_uint8 = (norm_depth * 255).astype(np.uint8)
        depth_rgb_frames = [np.repeat(d[:, :, None], 3, axis=2) for d in depth_uint8]
        return depth_uint8, depth_rgb_frames

    def _tensor_to_numpy(self, value, shape_fix: Optional[tuple] = None):
        """
        Safely convert a tensor-like object to numpy float32.
        """
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu()
        np_value = np.asarray(value)
        if shape_fix is not None:
            np_value = np.reshape(np_value, shape_fix)
        return np.asarray(np_value, dtype=np.float32)

    def _convert_shs_to_colors(self, shs_tensor):
        """
        Approximate RGB colors from SH coefficients by using the DC term.
        """
        if shs_tensor is None or not isinstance(shs_tensor, torch.Tensor):
            return None
        try:
            shs_cpu = shs_tensor.detach().cpu()
        except Exception:
            return None

        colors = None
        if shs_cpu.dim() == 2 and shs_cpu.shape[1] >= 3:
            colors = torch.sigmoid(shs_cpu[:, :3])
        elif shs_cpu.dim() == 3:
            if shs_cpu.shape[-1] >= 1:
                colors = torch.sigmoid(shs_cpu[:, :, 0])
            elif shs_cpu.shape[1] >= 3:
                colors = torch.sigmoid(shs_cpu[:, :3, 0])

        if colors is None:
            return None
        return torch.clamp(colors, 0.0, 1.0).numpy().astype(np.float32)

    def _gather_frame_primitives(self, frame_idx: float,
                                 fetch_scale: bool = True,
                                 fetch_opacity: bool = True):
        """
        Collect primitive attributes for a single frame.
        """
        with torch.no_grad():
            render_dict = self.gs_atlases_model.forward(float(frame_idx))

        position = render_dict["position"]
        scaling = render_dict["scaling"] if fetch_scale else None
        opacity = render_dict["opacity"] if fetch_opacity else None
        colors = self._convert_shs_to_colors(render_dict["shs"])

        pos_np = self._tensor_to_numpy(position)
        scale_np = self._tensor_to_numpy(scaling) if scaling is not None else None
        opacity_np = self._tensor_to_numpy(opacity) if opacity is not None else None
        if opacity_np is not None:
            opacity_np = opacity_np.reshape(-1, 1)

        if colors is None:
            colors = np.ones((pos_np.shape[0], 3), dtype=np.float32) * 0.5

        return {
            "positions": pos_np.astype(np.float32),
            "colors": colors.astype(np.float32),
            "scales": scale_np.astype(np.float32) if scale_np is not None else None,
            "opacity": opacity_np.astype(np.float32) if opacity_np is not None else None,
        }

    def _estimate_scene_bounds(self, sample_frames: int = 64):
        """
        Estimate a bounding sphere that covers all primitives across time.
        """
        if self.num_imgs <= 0:
            raise RuntimeError("No frames available to estimate scene bounds.")

        sample_frames = max(1, min(int(sample_frames), self.num_imgs))
        sample_ids = torch.linspace(0, max(self.num_imgs - 1, 0), steps=sample_frames)
        positions = []
        with torch.no_grad():
            for t in sample_ids:
                render_dict = self.gs_atlases_model.forward(float(t))
                pos = render_dict.get("position")
                if pos is None:
                    pos = render_dict.get("detached_position")
                if pos is None:
                    continue
                positions.append(pos.detach().cpu())

        if not positions:
            raise RuntimeError("Unable to estimate scene bounds; no positions found.")

        all_pos = torch.cat(positions, dim=0)
        center = all_pos.mean(dim=0)
        radius = torch.norm(all_pos - center, dim=1).max()
        radius_val = max(float(radius.item()), 1e-4)
        center = center.to(self.device)
        return center, radius_val

    def render_global_primitive_video(self,
                                      step: int = 0,
                                      save_frames: bool = False,
                                      fps: int = 10,
                                      num_bound_samples: int = 64,
                                      radius_scale: float = 2.0,
                                      elevation_deg: float = 20.0,
                                      azimuth_deg: float = -60.0,
                                      zoom_ratio: float = 10.0,
                                      z_travel: float = 0.0,
                                      output_name: Optional[str] = None):
        """
        Render a video showing the temporal evolution of all primitives from a single
        global camera view that covers the entire scene.
        """
        if self.num_imgs <= 0:
            print("[render_global_primitive_video] No frames to render.")
            return

        base_focal = self.gs_atlases_model.focal_y_ratio * self.h

        def _build_camera_dict(c2w_np):
            camera = construct_canonical_camera_from_focal(width=self.w, height=self.h, focal=base_focal, c2w=c2w_np)
            batch = {
                "camera": camera,
                "FovX": camera.fovX,
                "FovY": camera.fovY,
                "height": int(camera.image_height),
                "width": int(camera.image_width),
                "world_view_transform": camera.world_view_transform,
                "full_proj_transform": camera.full_proj_transform,
                "extrinsic_matrix": camera.extrinsic_matrix,
                "intrinsic_matrix": camera.intrinsic_matrix,
                "camera_center": camera.camera_center,
                "enable_ortho_projection": getattr(self, "enable_ortho_projection", False),
            }
            device_type = self.device.type if isinstance(self.device, torch.device) else str(self.device)
            if device_type == "cuda":
                tensor_keys = [
                    "world_view_transform",
                    "full_proj_transform",
                    "extrinsic_matrix",
                    "intrinsic_matrix",
                    "camera_center",
                ]
                for key in tensor_keys:
                    if torch.is_tensor(batch[key]):
                        batch[key] = batch[key].to(self.device)
            return batch

        vis_dir = os.path.join(self.out_dir, 'vis')
        os.makedirs(vis_dir, exist_ok=True)
        video_path = output_name or os.path.join(vis_dir, f'primitive_overview_{step:06d}.mp4')
        frame_dir = os.path.join(vis_dir, f'primitive_overview_frames_{step:06d}')

        images = []
        render_start = time.perf_counter()
        
        base_camera = self.batch_dict["camera"]
        base_extrinsic = base_camera.extrinsic_matrix.detach().cpu().numpy()
        base_c2w = np.linalg.inv(base_extrinsic)
        base_translation = base_c2w[:3, 3].copy()

        zoom_ratio = float(zoom_ratio)
        z_travel = float(z_travel)
        denom = max(self.num_imgs - 1, 1)
        for idx in range(self.num_imgs):
            with torch.no_grad():
                progress = idx / denom
                translation = base_translation + np.array([0.0, 0.0, (zoom_ratio + z_travel) * progress],
                                                          dtype=np.float32)

                c2w = base_c2w.copy()
                c2w[:3, 3] = translation

                batch_dict = _build_camera_dict(c2w)
                render_dict = self.gs_atlases_model.forward(idx)
                render_results = self.renderer.render_batch(render_dict, [batch_dict])
            rgb = render_results['rgb'][0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
            images.append((rgb * 255).astype(np.uint8))

        elapsed = max(time.perf_counter() - render_start, 1e-8)
        fps_actual = self.num_imgs / elapsed if self.num_imgs else 0.0
        print(f"[render_global_primitive_video] Rendered {self.num_imgs} frames in "
              f"{elapsed:.2f}s ({fps_actual:.2f} FPS)")

        if save_frames:
            os.makedirs(frame_dir, exist_ok=True)
            for i, img in enumerate(images):
                imageio.imwrite(os.path.join(frame_dir, f'{i:05d}.png'), img)

        imageio.mimwrite(video_path, images, quality=8, fps=fps)
        print(f"[render_global_primitive_video] Saved video to {video_path}")

    def export_dynamic_primitives_to_gltf(self,
                                          export_dir: Optional[str] = None,
                                          frame_step: int = 1,
                                          include_final_frame: bool = True,
                                          include_scale: bool = True,
                                          include_opacity: bool = True):
        """
        Export primitive states across time into a glTF + BIN pair for web viewers.
        Each frame becomes its own mesh/node (POINTS primitive) and metadata is
        stored in glTF extras for custom playback in Three.js.
        """
        if self.num_imgs <= 0:
            raise RuntimeError("No frames available to export.")

        frame_step = max(1, int(frame_step))
        export_indices = list(range(0, self.num_imgs, frame_step))
        if include_final_frame and (self.num_imgs - 1) not in export_indices:
            export_indices.append(self.num_imgs - 1)
        export_indices = sorted(set(int(idx) for idx in export_indices))

        if export_dir is None:
            export_dir = os.path.join(self.out_dir, "web", "gltf")
        export_dir = Path(export_dir)
        export_dir.mkdir(parents=True, exist_ok=True)

        bin_path = export_dir / "primitives_dynamic.bin"
        gltf_path = export_dir / "primitives_dynamic.gltf"

        buffer_data = bytearray()
        buffer_views = []
        accessors = []
        meshes = []
        nodes = []
        extras_frames = []

        def append_buffer_view(data_bytes: bytes):
            start = len(buffer_data)
            buffer_data.extend(data_bytes)
            view_idx = len(buffer_views)
            buffer_views.append({
                "buffer": 0,
                "byteOffset": start,
                "byteLength": len(data_bytes),
                "target": 34962,
            })
            return view_idx

        def create_accessor(view_idx: int, count: int, type_str: str,
                            component_type: int = 5126,
                            min_val: Optional[list] = None,
                            max_val: Optional[list] = None):
            accessor = {
                "bufferView": view_idx,
                "componentType": component_type,
                "count": count,
                "type": type_str,
            }
            if min_val is not None:
                accessor["min"] = min_val
            if max_val is not None:
                accessor["max"] = max_val
            accessors.append(accessor)
            return len(accessors) - 1

        reference_count = None
        for local_idx, frame_idx in enumerate(export_indices):
            frame = self._gather_frame_primitives(frame_idx,
                                                  fetch_scale=include_scale,
                                                  fetch_opacity=include_opacity)
            positions = frame["positions"]
            colors = frame["colors"]
            scales = frame["scales"]
            opacity = frame["opacity"]

            num_points = positions.shape[0]
            if reference_count is None:
                reference_count = num_points
            elif num_points != reference_count:
                raise RuntimeError(
                    f"Frame {frame_idx} contains {num_points} primitives, "
                    f"expected {reference_count}. Export requires a consistent count."
                )

            pos_bytes = positions.astype(np.float32).tobytes()
            col_bytes = colors.astype(np.float32).tobytes()
            pos_view = append_buffer_view(pos_bytes)
            col_view = append_buffer_view(col_bytes)
            pos_accessor = create_accessor(
                pos_view,
                num_points,
                "VEC3",
                min_val=positions.min(axis=0).tolist(),
                max_val=positions.max(axis=0).tolist(),
            )
            color_accessor = create_accessor(col_view, num_points, "VEC3")

            primitive = {
                "attributes": {
                    "POSITION": pos_accessor,
                    "COLOR_0": color_accessor,
                },
                "mode": 0,
            }

            primitive_extras = {}
            if include_scale and scales is not None:
                scale_bytes = scales.astype(np.float32).tobytes()
                scale_view = append_buffer_view(scale_bytes)
                scale_accessor = create_accessor(scale_view, num_points, "VEC3")
                primitive_extras["scale_accessor"] = scale_accessor
            else:
                scale_accessor = None

            if include_opacity and opacity is not None:
                opacity_bytes = opacity.astype(np.float32).tobytes()
                opacity_view = append_buffer_view(opacity_bytes)
                opacity_accessor = create_accessor(opacity_view, num_points, "SCALAR")
                primitive_extras["opacity_accessor"] = opacity_accessor
            else:
                opacity_accessor = None

            if primitive_extras:
                primitive["extras"] = primitive_extras

            mesh_idx = len(meshes)
            meshes.append({
                "name": f"mesh_frame_{frame_idx:05d}",
                "primitives": [primitive],
            })

            node_idx = len(nodes)
            node_entry = {
                "mesh": mesh_idx,
                "name": f"frame_{frame_idx:05d}",
            }
            nodes.append(node_entry)

            extras_frames.append({
                "frame_index": int(frame_idx),
                "node": node_idx,
                "mesh": mesh_idx,
                "point_count": num_points,
                "position_accessor": pos_accessor,
                "color_accessor": color_accessor,
                "scale_accessor": scale_accessor,
                "opacity_accessor": opacity_accessor,
                "time_normalized": (float(frame_idx) / max(self.num_imgs - 1, 1.0)),
            })

        scene_nodes = [0] if nodes else []
        gltf_dict = {
            "asset": {
                "version": "2.0",
                "generator": "FragTrainer.export_dynamic_primitives_to_gltf",
            },
            "scenes": [{
                "name": "default",
                "nodes": scene_nodes,
                "extras": {
                    "frame_indices": export_indices,
                    "total_frames": len(export_indices),
                },
            }],
            "scene": 0,
            "nodes": nodes,
            "meshes": meshes,
            "accessors": accessors,
            "bufferViews": buffer_views,
            "buffers": [{
                "byteLength": len(buffer_data),
                "uri": bin_path.name,
            }],
            "extras": {
                "frame_nodes": extras_frames,
                "description": (
                    "Each frame is stored as an independent POINTS mesh. "
                    "Use extras.frame_nodes to attach/detach nodes for animation."
                ),
            },
        }

        with open(bin_path, "wb") as bin_file:
            bin_file.write(buffer_data)
        with open(gltf_path, "w", encoding="utf-8") as gltf_file:
            json.dump(gltf_dict, gltf_file, indent=2)

        print(f"[export_dynamic_primitives_to_gltf] Saved {len(export_indices)} frames to {gltf_path}")

    def render_video(self, step=0, save_frames=False, force_full_opacity=False):
        ### render image / depth / dinov2
        images, depths, ellipses = [], [], []
        dinos = []
        render_rgb_start = time.perf_counter()
        for id in range(self.num_imgs):
            with torch.no_grad():
                render_dict = self.gs_atlases_model.forward(id)
                batch_dict_copy = self.batch_dict.copy()
                batch_dict_copy.update({"render_attributes_list" : ['dino_attribute', 'mask_attribute']})
                render_results = self._render_batch(render_dict, [batch_dict_copy], force_full_opacity=force_full_opacity)
                pred_rgb = render_results['rgb'][0].clamp(0,1).permute(1,2,0).cpu().numpy()
                # pred_ellipse = render_results['ellipse'][0].clamp(0,1).permute(1,2,0).cpu().numpy()
                pred_depth = render_results['depth'][0].permute(1,2,0).cpu().numpy()
                pred_dino = render_results['dino_attribute'][0].permute(1,2,0).cpu().numpy()
            images.append((pred_rgb*255).astype(np.uint8))
            depths.append(pred_depth)
            dinos.append((pred_dino*255).astype(np.uint8))
            # ellipses.append((pred_ellipse*255).astype(np.uint8))
        render_rgb_elapsed = max(time.perf_counter() - render_rgb_start, 1e-8)
        rgb_fps = self.num_imgs / render_rgb_elapsed if self.num_imgs > 0 else 0.0
        print(f"[render_video] Rendered {self.num_imgs} RGB frames in {render_rgb_elapsed:.2f}s "
              f"({rgb_fps:.2f} FPS)")
        
        depth_gray_frames_uint8, depth_rgb_frames = self._prepare_depth_grayscale(depths)

        if save_frames:
            save_dir = os.path.join(self.out_dir, 'vis', 'frames_{:06d}'.format(step))
            os.makedirs(save_dir, exist_ok=True)
            for i, img in enumerate(images):
                imageio.imwrite(os.path.join(save_dir, f'{i:05d}.png'), img)
            depth_save_dir = os.path.join(self.out_dir, 'vis', 'depth_frames_{:06d}'.format(step))
            os.makedirs(depth_save_dir, exist_ok=True)
            for i, depth_img in enumerate(depth_gray_frames_uint8):
                imageio.imwrite(os.path.join(depth_save_dir, f'{i:05d}.png'), depth_img)
        
        imageio.mimwrite(os.path.join(self.out_dir, 'vis', 'render_{:06d}.mp4'.format(step)),
                            images,
                            quality=8, fps=10)
        # imageio.mimwrite(os.path.join(self.out_dir, 'vis', 'ellipse_{:06d}.mp4'.format(step)),
        #                     ellipses,
        #                     quality=8, fps=10)
        
        imageio.mimwrite(os.path.join(self.out_dir, 'vis', 'depth_{:06d}.mp4'.format(step)),
                            depth_rgb_frames,
                            quality=8, fps=10)
        # imageio.mimwrite(os.path.join(self.out_dir, 'vis', 'dino_{:06d}.mp4'.format(step)),
        #                     dinos,
        #                     quality=8, fps=10)
        print()

    def render_shape_preview(self, step=0, save_frames=False):
        """
        Convenience helper to render frames with every Gaussian opacity fixed to 1.
        """
        return self.render_video(step=step, save_frames=save_frames, force_full_opacity=True)


    
    def render_part(self, fg=True, threshold=0.5):
        """
        Remove the foreground object from the scene
        """
        images, depths = [], []
        for id in range(self.num_imgs):
            with torch.no_grad():
                render_dict = self.gs_atlases_model.forward(id)

                if fg:
                    mask = (render_dict['mask_attribute'].squeeze() > threshold).detach()
                else:
                    mask = (render_dict['mask_attribute'].squeeze() <= threshold).detach()
                render_dict = {k: v[mask] for k, v in render_dict.items()}

                batch_dict_copy = self.batch_dict.copy()
                batch_dict_copy.update({"render_attributes_list" : ['mask_attribute'], 
                                        "bg_color" : 1.0})
                render_results = self.renderer.render_batch(render_dict, [batch_dict_copy])
                pred_rgb = render_results['rgb'][0].clamp(0,1).permute(1,2,0).cpu().numpy()
                pred_depth = render_results['depth'][0].clamp(0,1).permute(1,2,0).cpu().numpy()
                pred_attribute = render_results['mask_attribute'][0].clamp(0,1).permute(1,2,0).cpu().numpy()

                # pred_rgb = (1 - self.masks[id][...,None]).cpu().numpy() * pred_rgb

            images.append((pred_rgb*255).astype(np.uint8))

        save_path = os.path.join(self.out_dir, '%s_part2.mp4' % ('fg' if fg else 'bg'))
        imageio.mimwrite(save_path,
                            images,
                            quality=8, fps=4)
        print()

    def render_wave_coefficients_part(self, intervals=None, bg_color=1.0, fps=4):
        """
        Render Gaussian distributions filtered by the mean wave coefficient range.

        Parameters
        ----------
        intervals : list of tuple, optional
            Sequence of (low, high) bounds for average wave coefficients. Defaults to
            [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)].
        bg_color : float, optional
            Background color for the renderer, default 1.0.
        fps : int, optional
            Frames per second for the saved videos, default 4.
        """
        if intervals is None:
            intervals = [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]
        if not intervals:
            return

        if "height" not in self.batch_dict or "width" not in self.batch_dict:
            raise RuntimeError("Camera parameters not initialized. Call construct_render_dict first.")

        height = int(self.batch_dict["height"])
        width = int(self.batch_dict["width"])
        bg_frame = (np.ones((height, width, 3), dtype=np.float32) * float(bg_color)).clip(0.0, 1.0)

        has_wave_coefficients = None

        for interval in intervals:
            images = []

            for idx in range(self.num_imgs):
                with torch.no_grad():
                    render_dict = self.gs_atlases_model.forward(idx)

                if has_wave_coefficients is None:
                    has_wave_coefficients = "wave_coefficients" in render_dict
                    if not has_wave_coefficients:
                        print("wave_coefficients not found in render_dict; skip rendering wave coefficient parts.")
                        return

                wave_coeff = render_dict["wave_coefficients"]
                if wave_coeff.ndim > 1:
                    avg_coeff = wave_coeff.mean(dim=-1)
                else:
                    avg_coeff = wave_coeff

                avg_coeff = avg_coeff.detach()
                if upper_inclusive:
                    mask = (avg_coeff >= low) & (avg_coeff <= high)
                else:
                    mask = (avg_coeff >= low) & (avg_coeff < high)

                if mask.sum().item() == 0:
                    images.append((bg_frame * 255).astype(np.uint8))
                    continue

                filtered_render_dict = {}
                for key, value in render_dict.items():
                    if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == mask.shape[0]:
                        filtered_render_dict[key] = value[mask]
                    else:
                        filtered_render_dict[key] = value

                batch_dict_copy = self.batch_dict.copy()
                batch_dict_copy.update({"bg_color": float(bg_color)})

                with torch.no_grad():
                    render_results = self.renderer.render_batch(filtered_render_dict, [batch_dict_copy])

                pred_rgb = render_results["rgb"][0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
                images.append((pred_rgb * 255).astype(np.uint8))

            interval_name = f"{low:.2f}_{high:.2f}".replace(".", "p")
            save_path = os.path.join(self.out_dir, f"wave_coeff_{interval_name}.mp4")
            imageio.mimwrite(save_path, images, quality=8, fps=fps)
        print(f"Saved wave coefficient range [{low}, {high}{']' if upper_inclusive else ')'} to {save_path}")


    def render_wave_coefficients_frame_mask(self,
                                            intervals,
                                            frame_idx=35,
                                            mask_threshold=0.5,
                                            bg_color=0.0):
        """
        Render a specific frame foreground filtered by wave coefficient ranges and spatial mask.

        Parameters
        ----------
        frame_idx : int, optional
            Zero-based frame index to render. Default is 35 (the 36th frame).
        intervals : list of tuple, optional
            Sequence of (low, high) bounds for mean wave coefficients. Defaults to
            [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)] with an additional (0.0, 1.0).
        mask_threshold : float, optional
            Threshold applied to the frame mask to keep foreground, default 0.5.
        bg_color : float or str, optional
            Background specification for uncovered pixels. Accepts scalar values (0.0~1.0),
            or string shortcuts: "white", "black", "transparent". Default is 1.0 (white).
        """
        if frame_idx < 0 or frame_idx >= self.num_imgs:
            raise ValueError(f"frame_idx {frame_idx} out of range [0, {self.num_imgs}).")

        if "height" not in self.batch_dict or "width" not in self.batch_dict:
            raise RuntimeError("Camera parameters not initialized. Call construct_render_dict first.")

        if not hasattr(self, "masks"):
            raise AttributeError("Instance does not contain masks; ensure masks are loaded.")

        mask_tensor = self.masks[frame_idx]
        mask_tensor = torch.ones_like(mask_tensor)
        if mask_tensor.dim() == 3:
            mask_tensor = mask_tensor[..., 0]
        mask_np = (mask_tensor > mask_threshold).float().cpu().numpy()[..., None]
        transparent_bg = False
        if isinstance(bg_color, str):
            mode = bg_color.lower()
            if mode == "white":
                bg_scalar = 1.0
            elif mode == "black":
                bg_scalar = 0.0
            elif mode == "transparent":
                bg_scalar = 0.0
                transparent_bg = True
            else:
                raise ValueError(f"Unsupported bg_color '{bg_color}'. Use a float or one of {{'white','black','transparent'}}.")
        else:
            bg_scalar = float(bg_color)

        has_wave_coefficients = None

        with torch.no_grad():
            render_dict_base = self.gs_atlases_model.forward(frame_idx)

        for interval in intervals:
            if has_wave_coefficients is None:
                has_wave_coefficients = "wave_coefficients" in render_dict_base
                if not has_wave_coefficients:
                    print("wave_coefficients not found in render_dict; skip rendering wave coefficient frame mask.")
                    return

            wave_coeff = render_dict_base["wave_coefficients"]
            if wave_coeff.ndim > 1:
                avg_coeff = wave_coeff.mean(dim=-1)
            else:
                avg_coeff = wave_coeff

            avg_coeff = avg_coeff.detach()
            low = float(interval[1:].strip(" [()]").split(",")[0])
            high = float(interval[:-1].strip(" [()]").split(",")[1])
            if "[" in interval:
                pre = "a"
                gaussian_mask = (avg_coeff >= low) 
            elif "(" in interval:
                pre = "b"
                gaussian_mask = (avg_coeff > low) 
            
            if ")" in interval:
                suf = "b"
                gaussian_mask &= (avg_coeff < high) 
            elif "]" in interval:
                suf = "a"
                gaussian_mask &= (avg_coeff <= high) 

            if gaussian_mask.sum().item() == 0:
                if transparent_bg:
                    fg_rgb = np.zeros(mask_np.shape[:2] + (3,), dtype=np.float32)
                    alpha = np.zeros_like(mask_np, dtype=np.float32)
                else:
                    fg_rgb = np.full(mask_np.shape[:2] + (3,), bg_scalar, dtype=np.float32)
                    alpha = np.ones_like(mask_np, dtype=np.float32)
            else:
                filtered_render_dict = {}
                for key, value in render_dict_base.items():
                    if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == gaussian_mask.shape[0]:
                        filtered_render_dict[key] = value[gaussian_mask]
                    else:
                        filtered_render_dict[key] = value

                batch_dict_copy = self.batch_dict.copy()
                batch_dict_copy.update({"bg_color": bg_scalar})

                with torch.no_grad():
                    render_results = self.renderer.render_batch(filtered_render_dict, [batch_dict_copy])
                pred_rgb = render_results["rgb"][0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
                if transparent_bg:
                    fg_rgb = pred_rgb * mask_np
                    alpha = mask_np.astype(np.float32)
                else:
                    fg_rgb = pred_rgb * mask_np + bg_scalar * (1.0 - mask_np)
                    alpha = np.ones_like(mask_np, dtype=np.float32)

            interval_name = f"{low:.2f}_{high:.2f}".replace(".", "p")
            save_path = os.path.join(
                self.out_dir,
                f"wave_coeff_frame_{pre}{frame_idx:04d}_{suf}{interval_name}_fg.png")
            rgba = np.concatenate([fg_rgb, alpha], axis=-1)
            rgba = np.clip(rgba, 0.0, 1.0)
            imageio.imwrite(save_path, (rgba * 255).astype(np.uint8))
            print(f"Saved frame {frame_idx} foreground for range {interval} to {save_path}")


    def add_fg(self, delta_pos, scale, threshold=0.5):
        render_dict = self.gs_atlases_model.forward(0)
        fg_mask = (render_dict['mask_attribute'].squeeze() > 0.5).detach()

        images, depths = [], []
        for id in range(self.num_imgs):
            render_dict = self.gs_atlases_model.forward(id)
            ### for cow
            # new_idx = max(0, id-2)
            new_idx = int(id / 1.)
            new_delta_pos = delta_pos.clone() + \
                torch.tensor([[-0.0, 0, 0]], device='cuda') * new_idx
            render_dict_tmp = self.gs_atlases_model.forward(new_idx)
            for k, v in render_dict.items():
                if k == 'position':
                    fg_pos = render_dict_tmp['position'][fg_mask]
                    fg_pos_mean = fg_pos.mean(dim=0, keepdim=True)
                    fg_pos = (fg_pos - fg_pos_mean) * scale + fg_pos_mean
                    fg_pos = fg_pos + new_delta_pos

                    # fg_pos[...,0] *= -1
                    fg_rot = render_dict_tmp['rotation'][fg_mask]
                    import pytorch3d
                    from pytorch3d import transforms
                    fg_rot_mat = pytorch3d.transforms.quaternion_to_matrix(fg_rot)
                    # fg_rot_mat[:,0,:3] *= -1
                    fg_rot = pytorch3d.transforms.matrix_to_quaternion(fg_rot_mat)
                    fg_rot = fg_rot / fg_rot.norm(dim=-1, keepdim=True)

                    render_dict['position'] = torch.cat([
                                        render_dict['position'], 
                                        fg_pos
                                    ], dim=0)
                    render_dict['rotation'] = torch.cat([
                                        render_dict['rotation'], 
                                        fg_rot
                                    ], dim=0)
                elif k == "rotation":
                    pass
                else:
                    render_dict[k] = torch.cat([
                                        render_dict[k], 
                                        render_dict_tmp[k][fg_mask]
                                    ], dim=0)

            batch_dict_copy = self.batch_dict.copy()
            batch_dict_copy.update({"render_attributes_list" : ['mask_attribute']})
            with torch.no_grad():
                render_results = self.renderer.render_batch(render_dict, [batch_dict_copy])
            pred_rgb = render_results['rgb'][0].clamp(0,1).permute(1,2,0).cpu().numpy()
            pred_depth = render_results['depth'][0].clamp(0,1).permute(1,2,0).cpu().numpy()
            pred_attribute = render_results['mask_attribute'][0].clamp(0,1).permute(1,2,0).cpu().numpy()
            # imageio.imwrite("./debug.png", (pred_rgb*255).astype(np.uint8))

            images.append((pred_rgb*255).astype(np.uint8))
            depths.append((pred_depth*255).astype(np.uint8))
        imageio.imwrite(os.path.join(self.out_dir, 'added_fg.png'), images[-1])

        imageio.mimwrite(os.path.join(self.out_dir, 'added_fg.mp4'),
                            images,
                            quality=8, fps=4)
        print()


    def draw_gs_trajectory(self, samp_num=10, gs_num=512):
        """
        Draw gs trajectory from time 0 to 1
        """
        cur_pts = self.gs_atlases_model.get_atlas('gs_fg').point_cloud.get_position(0)
        from vis_utils import farthest_point_sample
        pts_idx = farthest_point_sample(cur_pts[None], gs_num)[0]

        spatial_idx = torch.argsort(cur_pts[pts_idx][:,0])
        pts_idx = pts_idx[spatial_idx]

        import cv2
        from matplotlib import cm
        color_map = cm.get_cmap("jet")
        colors = np.array([np.array(color_map(i/max(1, float(gs_num - 1)))[:3]) * 255 for i in range(gs_num)], dtype=np.int32)

        imgs = []
        delta_ts = torch.linspace(0, self.num_imgs-1, samp_num).to(self.device) # right side is included
        for i in range(samp_num):
            cur_time = delta_ts[i]
            cur_pts = self.gs_atlases_model.get_atlas('gs_fg').point_cloud.get_position(cur_time)
            cur_pts = cur_pts[pts_idx]

            (uv, depth) = gs.project_point(
                cur_pts,
                self.batch_dict["intrinsic_matrix"].cuda(),
                self.batch_dict["extrinsic_matrix"].cuda(),
                self.batch_dict["width"], 
                self.batch_dict["height"],
                nearest=0.01
                )
            
            alpha_img = np.zeros([self.h, self.w, 3])
            traj_img = np.zeros([self.h, self.w, 3])

            for i in range(gs_num):
                color = colors[i] / 255
                alpha_img = cv2.circle(img=alpha_img, center=(int(uv[i][0]), int(uv[i][1])), color=[1,1,1], radius=5, thickness=-1)
                traj_img = cv2.circle(img=traj_img, center=(int(uv[i][0]), int(uv[i][1])), color=[float(color[0]), float(color[1]), float(color[2])], radius=5, thickness=-1)

            with torch.no_grad():
                render_dict = self.gs_atlases_model.forward(cur_time)
                render_results = self.renderer.render_batch(render_dict, [self.batch_dict])
                img = render_results['rgb'][0].permute(1,2,0).cpu().numpy()
            img = traj_img * alpha_img[...,:1] + img * (1-alpha_img[...,:1])
            imgs.append((img*255).astype(np.uint8))
        imageio.mimwrite('gs_trajectory.mp4', imgs, fps=4)
        print()


    def sample_pts_within_mask(self, mask, num_pts, return_normed=False, seed=None,
                               use_mask=False, reverse_mask=False, regular=False, interval=10):
        rng = np.random.RandomState(seed) if seed is not None else np.random
        if use_mask:
            if reverse_mask:
                mask = ~mask
            kernel = torch.ones(7, 7, device=self.device)
            mask = morph.erosion(mask.float()[None, None], kernel).bool().squeeze()  # Erosion
        else:
            mask = torch.ones_like(self.grid[..., 0], dtype=torch.bool)

        if regular:
            coords = self.grid[::interval, ::interval, :2][mask[::interval, ::interval]]
        else:
            coords_valid = self.grid[mask][..., :2]
            rand_inds = rng.choice(len(coords_valid), num_pts, replace=(num_pts > len(coords_valid)))
            coords = coords_valid[rand_inds]

        coords_normed = util.normalize_coords(coords, self.h, self.w)
        if return_normed:
            return coords, coords_normed
        else:
            return coords  # [num_pts, 2]


    def draw_pixel_trajectory(self, idx=0, use_mask=False, radius=5):
        mask = self.masks[idx]
        px1s = self.sample_pts_within_mask(mask, num_pts=100, seed=1234,
                                           use_mask=use_mask, reverse_mask=False,
                                           regular=True, interval=20)
        num_pts = len(px1s)
        
        ### render frame to get the gs index
        render_dict = self.gs_atlases_model.forward(idx)
        # selected_gs_idx = render_results['gs_idx'][0][mask > 0]
        # selected_gs_idx = selected_gs_idx[selected_gs_idx != -1]
        # selected_gs_idx = torch.unique(selected_gs_idx)

        (uv1, depth1) = self.renderer.project_point(
            render_dict["detached_position"],
            self.batch_dict["extrinsic_matrix"].cuda(),
            self.batch_dict["width"], 
            self.batch_dict["height"],
            nearest=0.01)

        ### get the motion of the selected gs
        # pos_0 = render_dict['position'][selected_gs_idx]
        
        kpts = [px1s.detach().cpu().numpy()]
        px_masks = [np.ones_like(px1s.detach().cpu().numpy()[:,0], dtype=bool)]
        img_query = (self.images[0].cpu().numpy() * 255).astype(np.uint8)
        ##### set kpt color list here
        set_max = range(128)
        colors = {m: i for i, m in enumerate(set_max)}
        colors = {m: (255 * np.array(plt.cm.hsv(i/float(len(colors))))[:3][::-1]).astype(np.int32)
                for m, i in colors.items()}
        center = np.median(kpts[0], axis=0, keepdims=True)
        coord_angle = np.arctan2(kpts[0][:, 1] - center[:, 1], kpts[0][:, 0] - center[:, 0])
        corr_color = np.int32(64 * coord_angle / np.pi) % 128  # [N]
        # color_list = tuple(colors[corr_color].tolist())  # [N,3]
        color_list = [colors[corr_color[i]].tolist() for i in range(num_pts)]

        imgs_list = [self.images[0].cpu().numpy()]
        out_imgs = []
        for fid in range(1, self.num_imgs):
            render_dict2 = self.gs_atlases_model.forward(fid)

            (uv2, depth2) = self.renderer.project_point(
                render_dict2["detached_position"],
                self.batch_dict["extrinsic_matrix"].cuda(),
                self.batch_dict["width"], 
                self.batch_dict["height"],
                nearest=0.01)
            
            render_dict.update({"pixel_flow": uv2-uv1})  # TODO add flow
            batch_dict_copy = self.batch_dict.copy()
            batch_dict_copy.update({"render_attributes_list" : ["pixel_flow"]+list(self.cfg.render_attributes.keys())})
            batch_dict_copy.update({"num_idx": 10})
            render_results = self.renderer.render_batch(render_dict, [batch_dict_copy])

            px1s_normed = util.normalize_coords(px1s, self.h, self.w)
            px1s_flow = F.grid_sample(render_results['pixel_flow'], px1s_normed[None,:,None,:], align_corners=True)
            ### TODO add occulusion by rendering weights
            px2s = px1s + px1s_flow[0,...,0].permute(1,0)
            px2s_mask = torch.ones_like(px2s[:,0], dtype=torch.bool)

            img_i = self.images[fid].cpu().numpy()
            kpts.append(px2s.detach().cpu().numpy())
            px_masks.append(px2s_mask.detach().cpu().numpy())
            imgs_list.append(img_i)
        
        from functools import reduce
        unioned_px_mask = reduce(np.logical_and, px_masks)
        kpts = [k[unioned_px_mask] for k in kpts]
        for i in range(1, self.num_imgs+1):
            img_query = self.images[0].cpu().numpy()
            # out = util.drawMatches(img_query, img_i, px1s.detach().cpu().numpy(), px2s.detach().cpu().numpy(),
            #                         num_vis=num_pts, mask=None, radius=radius)
            # img = util.drawTrajectory(img_i, kpts[:-10:-1], num_vis=num_pts, idx_vis=np.arange(10))
            # img = util.drawTrajectoryWithColor(img_i, kpts[:-10:-1], color_list=color_list, num_vis=num_pts, idx_vis=np.arange(10))
            start_id = max(0, i-30)
            tracks_2d = np.stack(kpts[start_id:i], axis=0)
        
            # img = util.draw_tracks_2d(imgs_list[i-1], tracks_2d, track_point_size=8, track_line_width=3)  # for rebuttal image vis
            img = util.draw_tracks_2d(imgs_list[i-1], tracks_2d, track_point_size=2, track_line_width=1)  # for rebuttal image vis
            out = np.concatenate([(img_query * 255).astype(np.uint8), img], axis=1)
            out_imgs.append(out)

        return out_imgs
            

        ### draw trajectory. This is used for video visualization
        if False:
            kpts = np.stack(kpts, axis=0)
            img = np.ones((self.h, self.w, 3), dtype=np.uint8)
            img = util.drawTrajectory(img, kpts[::-1], num_vis=num_pts)

        

    def get_attributes_dict(self, frame_idx):
        """
        Get the attributes of the atlas at frame_idx
        """
        return self.gs_atlases_model.forward(frame_idx)
    

    def get_interpolation_result(self, scaling=2, save_frames=True, save_path=None):
        images, depths = [], []
        frames = []
        dinos = []
        # for id in range(self.num_imgs):
        num_imgs = self.num_imgs
        for id in range(num_imgs):
            for scaling_id in range(scaling+1):
                with torch.no_grad():
                    interpolated_id = id + scaling_id / (scaling+1)
                    if interpolated_id > num_imgs-1:
                        break
                    render_dict = self.gs_atlases_model.forward(interpolated_id)
                    batch_dict_copy = self.batch_dict.copy()
                    batch_dict_copy.update({"render_attributes_list" : ['dino_attribute', 'mask_attribute']})
                    render_results = self.renderer.render_batch(render_dict, [batch_dict_copy])
                    pred_rgb = render_results['rgb'][0].permute(1,2,0).clamp(0,1).cpu().numpy()
                    # pred_depth = render_results['depth'][0].permute(1,2,0).clamp(0,1).cpu().numpy()
                    # pred_dino = render_results['dino_attribute'][0].permute(1,2,0).clamp(0,1).cpu().numpy()
                frames.append(f"{id}_{scaling_id}")
                images.append((pred_rgb*255).astype(np.uint8))
                # depths.append(pred_depth)
                # dinos.append((pred_dino*255).astype(np.uint8))
        

        if save_frames:
            save_dir = os.path.join(self.out_dir, 'vis', f'new_inter_frames_{scaling:02d}')
            os.makedirs(save_dir, exist_ok=True)
            for i, img in zip(frames, images):
                imageio.imwrite(os.path.join(save_dir, f'{i}.png'), img)
                
        # depth_gray_frames_uint8, depth_rgb_frames = self._prepare_depth_grayscale(depths)
        # if save_frames:
        #     depth_save_dir = os.path.join(self.out_dir, 'vis', f'inter_depth_frames_{scaling:02d}')
        #     os.makedirs(depth_save_dir, exist_ok=True)
        #     for i, depth_img in enumerate(depth_gray_frames_uint8):
        #         imageio.imwrite(os.path.join(depth_save_dir, f'{i:05d}.png'), depth_img)
        
        if save_path is None:
            save_path = os.path.join(self.out_dir, 'vis', f'new_interp_{scaling:02d}_render.mp4')
        
        imageio.mimwrite(save_path,
                            images,
                            quality=8, fps=int(2*scaling))

        # depth_video_path = os.path.join(self.out_dir, 'vis', f'interp_{scaling:02d}_depth.mp4')
        # imageio.mimwrite(depth_video_path,
        #                     depth_rgb_frames,
        #                     quality=8, fps=int(2*scaling))
        print()

    def get_correspondences_and_occlusion_masks_for_pixels(self, ids1, px1s, ids2, use_max_loc=False):
        px2s_list, occlusions_list = [], []
        for (id1, id2, px1) in zip(ids1, ids2, px1s):
            px1s_normed = util.normalize_coords(px1, self.h, self.w)  # [num_pts, 2]
            target_tracks = self.video_3d_flow.load_target_tracks(id1, [id2], dim=0)  # [NumT, NumPoints, 4]
            target_visibles, target_invisibles, target_confidences = \
                parse_tapir_track_info(target_tracks[..., 2], target_tracks[..., 3])
            target_tracks = target_tracks[..., :2]  # [NumT, NumPoints, 2]
            # resize_h, resize_w = math.ceil(self.h / 4), math.ceil(self.w / 4)
            resize_h, resize_w = math.ceil(self.h), math.ceil(self.w)
            target_tracks = target_tracks.reshape(1,resize_h,resize_w,2).permute(0,3,1,2).to(self.device)
            target_invisibles = target_invisibles.reshape(1,resize_h,resize_w,1).permute(0,3,1,2).float().to(self.device)
            px2s = F.grid_sample(target_tracks, px1s_normed[None,:,None,:], align_corners=True)  # [1,2,N,1]
            px2s_occ = F.grid_sample(target_invisibles, px1s_normed[None,:,None,:], align_corners=True)  # [1,1,N,1]
            px2s = px2s[0,...,0].permute(1,0)
            px2s_occ = px2s_occ[0,...,0].permute(1,0)  # [N,1]
            px2s_list.append(px2s)
            occlusions_list.append(px2s_occ)
        return px2s_list, occlusions_list

    def get_correspondences_and_occlusion_masks_for_pixels_v0(self, ids1, px1s, ids2, use_max_loc=False):
        px2s, occlusions = [], []
        for (id1, id2, px1) in zip(ids1, ids2, px1s):
            ### 
            px2, occlusion = self.get_correspondences_and_occlusion_masks_for_pixels_core(id1, px1, id2, use_max_loc)
            px2s.append(px2)
            occlusions.append(occlusion)
        # return torch.stack(px2s, dim=0), torch.stack(occlusions, dim=0)
        return px2s, occlusions

    def get_correspondences_and_occlusion_masks_for_pixels_core(self, ids1, px1s, ids2,
                                                           use_max_loc=False):
        # ids1: int, px1s: [num_pts, 2], ids2: int
        # return px2s: [num_pts,2], occlusion: [num_pts,1]
        render_dict = self.gs_atlases_model.forward(ids1)
        render_dict2 = self.gs_atlases_model.forward(ids2)

        track_gs = render_dict2["position"]
        render_dict.update({"track_gs": track_gs})  # TODO add flow
        batch_dict_copy = self.batch_dict.copy()
        batch_dict_copy.update({"render_attributes_list" : ["track_gs"]+list(self.cfg.render_attributes.keys())})
        render_results = self.renderer.render_batch(render_dict, [batch_dict_copy])

        predicted_track_gs = render_results['track_gs'].permute(0,2,3,1)  # [1, h, w, 3]
        predicted_track_2d = util.denormalize_coords(predicted_track_gs[...,:2], self.h, self.w)  # [1, h, w, 2]
        predicted_track_2d = predicted_track_2d.permute(0,3,1,2)  # [1, 2, h, w]

        normed_px1s = util.normalize_coords(px1s, self.h, self.w)  # [num_pts, 2]
        px2s_pred = F.grid_sample(predicted_track_2d, normed_px1s[None,None], align_corners=True)[0,:,0,:]  # [1,2,1,num_pts]
        px2s_pred = px2s_pred.permute(1,0)

        predicted_track_depth = render_results['track_gs'][:,2:3]  # [1, 1, h, w]
        depth_proj = F.grid_sample(predicted_track_depth, normed_px1s[None,None], align_corners=True)[0,:,0,:]  # [1,1,1,num_pts]
        depth_proj = depth_proj.permute(1,0)


        ##### directly render the depth
        batch_dict_copy = self.batch_dict.copy()
        render_results2 = self.renderer.render_batch(render_dict2, [batch_dict_copy])
        depth2 = render_results2['depth']  # [1, 1, h, w]
        px2s_pred_depth = F.grid_sample(depth2, px2s_pred[None,None], align_corners=True)[0,:,0,:]  # [1,1,1,num_pts]
        px2s_pred_depth = px2s_pred_depth.permute(1,0)
        occlusion = (px2s_pred_depth >= depth_proj).float()
        return px2s_pred, occlusion

    # ======================= 1) Evenly sample n Gaussians inside the mask =======================
    def select_even_gaussians_in_mask(self, mask: torch.Tensor, n: int, frame_idx: int = 0, k_top: int = 1):
        """
        Args:
            mask: [H,W] 0/1 or 0~1 tensor (on any device)
            n: number of Gaussians to select (try to distribute evenly in mask)
            frame_idx: which frame projection to sample from (usually 0)
            k_top: Top-K gs_idx; set 1 for Top-1
        Returns:
            selected_gs_idx: [M] unique Gaussian indices (M<=n)
            selected_px:     [M,2] corresponding pixel coordinates (x,y)
        """
        H, W = mask.shape
        mask = (mask > 0.5).to('cpu')  # Move to CPU for cheaper sampling.
        ys, xs = torch.nonzero(mask, as_tuple=True)
        if len(xs) == 0:
            raise ValueError("Mask 裡沒有像素可選。")

        # Use FPS (farthest point sampling) to pick n seeds in 2D pixels for even coverage.
        coords = torch.stack([xs, ys], dim=-1).float()  # [P,2], (x,y)
        P = coords.shape[0]
        m = min(n*2, P)  # Over-sample to reduce duplicates later.
        # Simple FPS.
        sel = []
        d = torch.full((P,), 1e10)
        seed = torch.randint(0, P, (1,)).item()
        last = seed
        sel.append(last)
        for _ in range(1, m):
            dist2 = ((coords - coords[last])**2).sum(-1)
            d = torch.minimum(d, dist2)
            last = torch.argmax(d).item()
            sel.append(last)
        seeds = coords[torch.tensor(sel, dtype=torch.long)]  # [m,2]

        # Render once to get gs_idx (Top-K Gaussian indices).
        render_dict = self.gs_atlases_model.forward(frame_idx)
        bd = self.batch_dict.copy()
        bd.update({"num_idx": int(k_top)})  # Ensure K indices are returned.
        render_results = self.renderer.render_batch(render_dict, [bd])
        gs_idx_map = render_results["gs_idx"][0]  # [H,W,K], K==k_top
        # Do pixel/index ops on CPU to avoid device mismatch.
        gs_idx_map = gs_idx_map.cpu()

        # For each seed pixel, take the Top-1 Gaussian id.
        sx = seeds[:, 0].long().clamp(0, W-1)
        sy = seeds[:, 1].long().clamp(0, H-1)
        top1 = gs_idx_map[sy, sx, 0]  # [m]
        # Keep boolean indexing on CPU.
        valid = (top1 != -1)
        top1 = top1[valid].cpu()
        top1_np = top1.cpu().numpy()
        unique_vals, first_idx_np = np.unique(top1_np, return_index=True)
        first_idx = torch.from_numpy(first_idx_np).to(top1.device)

        reps_id = top1[first_idx]
        reps_px = seeds[first_idx]
        
        if reps_px.shape[0] <= n:
            selected_px  = reps_px
            selected_ids = reps_id
        else:
            # Run FPS again on representative pixels to pick n points.
            P2 = reps_px.shape[0]
            d2 = torch.full((P2,), 1e10)
            seed2 = torch.randint(0, P2, (1,)).item()
            last2 = seed2
            picked = [last2]
            for _ in range(1, n):
                dist2 = ((reps_px - reps_px[last2])**2).sum(-1)
                d2 = torch.minimum(d2, dist2)
                last2 = torch.argmax(d2).item()
                picked.append(last2)
            picked = torch.tensor(picked, dtype=torch.long)
            selected_px  = reps_px[picked]
            selected_ids = reps_id[picked]

        return selected_ids.to(self.device), selected_px.to(self.device)  # [M], [M,2]
    
    def get_gaussian_trajectories_from_nodes(
        self,
        gs_indices: torch.Tensor,
        t_query: torch.Tensor = None,
        detach: bool = True,
    ):
        """
        Use pc.get_position(t) to get 3D position trajectories (position only) for selected Gaussians.

        Args:
            gs_indices: [M] Gaussian indices to inspect
            T:         if t_query is None, sample T times in [tk[0], tk[-1]]
            t_query:   [T] custom time points (same units as pc.pos_tk); if set, ignore T
            detach:    use no_grad when True (saves memory for visualization)

        Returns:
            dict:
                "t"        -> [T]
                "position" -> [M, T, 3]
        """
        from contextlib import nullcontext

        device = self.device
        pc = self.gs_atlases_model.get_atlas(self.cfg.point_cloud_name).point_cloud

        # Prepare time sequence.
        if t_query is None:
            t_query = pc.pos_tk.to(device=device, dtype=torch.float32)
        else:
            t_query = t_query.to(device)

        gs_indices = gs_indices.to(device).long()

        ctx = torch.no_grad() if detach else nullcontext()
        pos_list = []

        with ctx:
            for t in t_query:
                pos_t_all = pc.get_position(float(t))   # [N,3]
                pos_list.append(pos_t_all[gs_indices])  # [M,3]

        pos_traj = torch.stack(pos_list, dim=1)        # [M, T, 3]
        return pos_traj, t_query


    def animate_gaussian_trajectories(
        self, 
        mask_path: str, 
        n: int = 20,
        frame_idx: int = 0, 
        t_query: torch.Tensor = None, 
        tail: int = None,
        out_gif: str = "traj.gif", 
        out_mp4: str = "traj.mp4",
        fps: int = 12,
        elev: float = 18, 
        azim: float = -60,
        show_current_dots: bool = True,
        base_color = (0.75, 0.75, 0.75),   # Gray background.
        base_alpha: float = 0.9,
        line_w_base: float = 1.5,
        line_w_head: float = 2.5,
    ):
        """
        Side-by-side output: left = 3D trajectory; right = 2D image projection
        (gray full path, colored traveled segment, current point). Outputs GIF and MP4.
        """
        import numpy as np
        import imageio.v2 as imageio
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        import torch

        # ---- 1) Read mask and evenly select n Gaussians within it ----
        mask = imageio.imread(mask_path)
        if mask.ndim == 3:
            mask = mask[..., 0]
        mask = (mask / 255.).astype(np.float32)
        mask = torch.tensor(mask).float().to(self.device)
        sel_ids, _ = self.select_even_gaussians_in_mask(mask, n=n, frame_idx=frame_idx, k_top=1)

        # ---- 2) Get position trajectories and time sequence ----
        traj_xyz, t_query_out = self.get_gaussian_trajectories_from_nodes(sel_ids, t_query=t_query)
        traj = traj_xyz.detach().cpu().numpy()   # [M, T, 3]
        t_vals = t_query_out.detach().cpu().numpy()
        M, T, _ = traj.shape
        if M == 0:
            raise ValueError("在該 mask 內沒有可用的高斯可視化。")
        if tail is None:
            tail = T

        # ---- 3) Projection utilities ----
        use_ortho = getattr(self, "enable_ortho_projection", False)
        pc = self.gs_atlases_model.get_atlas(self.cfg.point_cloud_name).point_cloud

        def project_points_xyz(xyz_3d: torch.Tensor):
            """xyz_3d: [N,3] CUDA -> return uv numpy [N,2] (pixel coords)"""
            if use_ortho:
                uv, _ = self.renderer.project_point(
                    xyz_3d,
                    self.batch_dict["extrinsic_matrix"].cuda(),
                    self.batch_dict["width"], 
                    self.batch_dict["height"],
                    nearest=0.01)
            else:
                uv, _ = gs.project_point(
                    xyz_3d,
                    self.batch_dict["intrinsic_matrix"].cuda(),
                    self.batch_dict["extrinsic_matrix"].cuda(),
                    self.batch_dict["width"], 
                    self.batch_dict["height"],
                    nearest=0.01)
            return uv.detach().cpu().numpy()

        # Precompute 2D projections for all time points: uv_traj [M,T,2]
        uv_traj = np.zeros((M, T, 2), dtype=np.float32)
        with torch.no_grad():
            for ti in range(T):
                pos_t = pc.get_position(float(t_vals[ti]))[sel_ids.to(self.device)]  # [M,3]
                uv_t = project_points_xyz(pos_t)  # [M,2]
                uv_traj[:, ti, :] = uv_t

        H, W = int(self.batch_dict["height"]), int(self.batch_dict["width"])

        # ---- 4) Prepare canvas (left/right) ----
        fig = plt.figure(figsize=(10, 5))
        gs = GridSpec(1, 2, width_ratios=[1.1, 1.0], figure=fig)
        ax3d = fig.add_subplot(gs[0, 0], projection="3d")
        ax2d = fig.add_subplot(gs[0, 1])

        # 3D view.
        ax3d.view_init(elev=elev, azim=azim)

        # Fix 3D axis ranges.
        xyz_all = traj.reshape(-1, 3)
        xyz_min, xyz_max = xyz_all.min(axis=0), xyz_all.max(axis=0)
        ctr = (xyz_min + xyz_max) / 2.0
        rad = (xyz_max - xyz_min).max() / 2.0 + 1e-6
        ax3d.set_xlim([ctr[0] - rad, ctr[0] + rad])
        ax3d.set_ylim([ctr[1] - rad, ctr[1] + rad])
        ax3d.set_zlim([ctr[2] - rad, ctr[2] + rad])
        ax3d.set_xlabel("X"); ax3d.set_ylabel("Y"); ax3d.set_zlabel("Z")

        # Right image axis.
        def render_rgb_at_tidx(ti: int):
            """Render RGB at continuous time t (smooth, no frame jumps)."""
            t = float(t_vals[ti])
            with torch.no_grad():
                render_dict = self.gs_atlases_model.forward(t)
                bd = self.batch_dict.copy()                 # Fix camera.
                res = self.renderer.render_batch(render_dict, [bd])
                rgb = res["rgb"][0].permute(1, 2, 0).clamp(0, 1).cpu().numpy()  # [H,W,3], 0~1
            return (rgb * 255.0).astype(np.uint8)

        img0 = render_rgb_at_tidx(0)
        im_artist = ax2d.imshow(img0, zorder=0)
        ax2d.set_title("Projection on image")
        ax2d.set_axis_off()
        ax2d.set_xlim(0, W-1)
        ax2d.set_ylim(H-1, 0)  # Top is 0.

        # ---- 5) Draw gray full trajectories on both sides ----
        base_lines_3d, base_lines_2d = [], []
        for i in range(M):
            P3 = traj[i]            # [T,3]
            P2 = uv_traj[i]         # [T,2]

            base_lines_3d.append(
                ax3d.plot(P3[:,0], P3[:,1], P3[:,2],
                        color=base_color, alpha=base_alpha,
                        linewidth=line_w_base, zorder=1)[0]
            )
            base_lines_2d.append(
                ax2d.plot(P2[:,0], P2[:,1],
                        color=base_color, alpha=base_alpha,
                        linewidth=line_w_base, zorder=1)[0]
            )

        # Colored traveled segment + current point (both sides).
        cmap = plt.cm.get_cmap("tab20", M)
        head_lines_3d = [ax3d.plot([], [], [], color=cmap(i), linewidth=line_w_head, zorder=2)[0] for i in range(M)]
        head_lines_2d = [ax2d.plot([], [], color=cmap(i), linewidth=line_w_head, zorder=2)[0] for i in range(M)]

        scat3d = ax3d.scatter(traj[:, 0, 0], traj[:, 0, 1], traj[:, 0, 2],
                            s=20, c=[cmap(i) for i in range(M)], depthshade=False, zorder=3) if show_current_dots else None
        scat2d = ax2d.scatter(uv_traj[:, 0, 0], uv_traj[:, 0, 1],
                            s=28, c=[cmap(i) for i in range(M)],
                            edgecolors="k", linewidths=0.5, zorder=3)

        # Time label (top-left of 3D plot).
        txt3d = fig.text(0.02, 0.96, "", fontsize=12, color="black")

        # ---- 6) Render frame by frame ----
        frames = []
        for t in range(T):
            t0 = max(0, t - tail + 1)

            # Left: 3D traveled segment.
            for i in range(M):
                P3 = traj[i, t0:t+1, :]
                head_lines_3d[i].set_data(P3[:, 0], P3[:, 1])
                head_lines_3d[i].set_3d_properties(P3[:, 2])
            if scat3d is not None:
                scat3d._offsets3d = (traj[:, t, 0], traj[:, t, 1], traj[:, t, 2])

            # Right: update image, 2D traveled segment, current point.
            im_artist.set_data(render_rgb_at_tidx(t))
            for i in range(M):
                P2 = uv_traj[i, t0:t+1, :]
                head_lines_2d[i].set_data(P2[:, 0], P2[:, 1])
            scat2d.set_offsets(uv_traj[:, t, :])

            # Time text.
            txt3d.set_text(f"t[{t}/{T-1}] = {t_vals[t]:.3f}")

            # Capture frame (cross-version).
            fig.canvas.draw()
            w, h = fig.canvas.get_width_height()
            if hasattr(fig.canvas, "buffer_rgba"):
                buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
                frame = buf.reshape(h, w, 4)[..., :3]
            elif hasattr(fig.canvas, "tostring_rgb"):
                buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                frame = buf.reshape(h, w, 3)
            elif hasattr(fig.canvas, "tostring_argb"):
                buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
                buf = buf.reshape(h, w, 4); buf = np.roll(buf, -1, axis=2)
                frame = buf[..., :3]
            else:
                raise RuntimeError("Canvas does not support buffer export methods.")

            frames.append(frame.copy())

        # Write GIF and MP4.
        imageio.mimsave(os.path.join(self.out_dir, out_gif), frames, fps=fps)
        imageio.mimsave(os.path.join(self.out_dir, out_mp4), frames, fps=fps, codec='libx264', quality=8)
        plt.close(fig)
