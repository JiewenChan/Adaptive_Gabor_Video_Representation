import torch
from torch import nn
from pytorch_msssim import ms_ssim
from dataclasses import dataclass

from pointrix.point_cloud import PointCloud, POINTSCLOUD_REGISTRY
from pointrix.utils.gaussian_points.gaussian_utils import (
    build_covariance_from_scaling_rotation,
    inverse_sigmoid,
    gaussian_point_init
)
from pointrix.point_cloud.utils import get_random_feauture, get_random_points

from pointrix.utils.dataset.dataset_utils import fov2focal, focal2fov
import numpy as np
import imageio
from pointrix.dataset.base_data import SimplePointCloud
import math
def depth2pcd(depth, shift=0.1):
    """
    Convert the depth map to point cloud.

    Parameters
    ----------
    depth: np.ndarray
        The depth map.
    """
    h, w = depth.shape
    i, j = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    z = depth + shift  ### add a shift value to avoid zero depth
    x = (j - w * 0.5) / (0.5 * w)  # normalize to [-1, 1]
    y = (i - h * 0.5) / (0.5 * h)
    pcd = np.stack([x, y, z], axis=-1)
    return pcd

def b_spline_basis(x, t, k):
    n = len(t) - k - 1

    dp = np.zeros((n + 1, k + 1))
    for d in range(0, k + 1):   
        for idx in range(n): 
            if d == 0:
                if t[idx] <= x < t[idx + 1]:
                    dp[idx, 0] = 1.0
            else:
                if t[idx + d] != t[idx]:
                    c1 = (x - t[idx]) / (t[idx + d] - t[idx]) * dp[idx, d - 1]
                else:
                    c1 = 0.0
                if t[idx + d + 1] != t[idx + 1]:
                    c2 = (t[idx + d + 1] - x) / (t[idx + d + 1] - t[idx + 1]) * dp[idx + 1, d - 1]
                else:
                    c2 = 0.0
                dp[idx, d] = c1 + c2 

    return dp[:-1, k]

def quat_mul(q1, q2):
    """
    四元數乘法
    q1: [..., 4]
    q2: [..., 4]
    """
    # 確保輸入是正確的維度
    assert q1.shape[-1] == 4, f"q1 must have last dimension of 4, got {q1.shape}"
    assert q2.shape[-1] == 4, f"q2 must have last dimension of 4, got {q2.shape}"
    
    # 使用 torch.unbind 來分離四元數的組件
    w1, x1, y1, z1 = torch.unbind(q1, dim=-1)
    w2, x2, y2, z2 = torch.unbind(q2, dim=-1)
    
    # 計算新的四元數
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    
    # 使用 torch.stack 組合結果
    return torch.stack([w, x, y, z], dim=-1)

def quat_conjugate(q):
    return torch.stack([q[...,0], -q[...,1], -q[...,2], -q[...,3]], dim=-1)

def quat_log(q):
    # q = [w, x, y, z]
    norm_v = torch.norm(q[...,1:], dim=-1, keepdim=True)  # [N,1]
    v = q[...,1:]
    w = q[...,0:1]
    theta = torch.atan2(norm_v, w)  # [N,1]
    scale = torch.where(norm_v > 1e-6, theta / norm_v, torch.zeros_like(norm_v))  # avoid NaN
    return scale * v  # [N,3]

def quat_exp(v):
    # v: [N, 3]
    norm_v = torch.norm(v, dim=-1, keepdim=True)
    sin_norm = torch.sin(norm_v)
    cos_norm = torch.cos(norm_v)
    scale = torch.where(norm_v > 1e-6, sin_norm / norm_v, torch.ones_like(norm_v))  # avoid zero divide
    return torch.cat([cos_norm, scale * v], dim=-1)  # [N,4]

def quat_normalize(q):
    return q / torch.norm(q, dim=-1, keepdim=True)


num_total_frequencies = 10
num_frequencies = 2

@POINTSCLOUD_REGISTRY.register()
class DynamicBsplineGaborAll(PointCloud):
    """
    A class for Gaussian point cloud.

    Parameters
    ----------
    PointCloud : PointCloud
        The point cloud for initialisation.
    """
    @dataclass
    class Config(PointCloud.Config):
        max_sh_degree: int = 3
        lambda_dssim: float = 0.2

    cfg: Config

    def setup(self, gs_atlas_cfg, base_point_seq=None):
        base_point_seq = torch.stack([x[~torch.isnan(x).any(dim=1)] for x in base_point_seq], dim=0)
        ### assume first frame is the base frame
        self.base_position = base_point_seq[0]
        self.delta_position = base_point_seq - self.base_position[None,...]

        # B-spline interpolation
        self.interval_num = math.ceil(len(base_point_seq) / 3)  # set a node every 5 frames
        self.intervals_idx = torch.linspace(0, 1, self.interval_num - 2)
        self.intervals = torch.cat([torch.zeros(3), self.intervals_idx, torch.ones(3)])
        delta_position = self.delta_position.cpu().numpy()  # [Nt, Np, 3]
        from scipy.interpolate import CubicSpline, BSpline
        xx = self.intervals.cpu().numpy()
        coeff_list = []
        for yy in delta_position[np.linspace(0, len(self.delta_position)-1, self.interval_num).astype(np.int32)].transpose(1,0,2):
            bspline = BSpline(xx, yy, 3)
            coeff_list.append(bspline.c)  # [4,interval_num]
        
        coeff_list = np.array(coeff_list)  # [Np, 4, interval_num, 3]
        pos_cubic_coeff = torch.tensor(coeff_list).to(self.base_position.device).float()  # [Np, 4, interval_num, 3]

        # super().setup(point_cloud)
        ######################## setup of point cloud
        self.atributes = []
        position = self.base_position
        features = get_random_feauture(len(position), self.cfg.initializer.feat_dim)
        self.register_buffer('position', position)
        self.register_buffer('features', features)
        self.atributes.append({
            'name': 'position',
            'trainable': self.cfg.trainable,
        })
        self.atributes.append({
            'name': 'features',
            'trainable': self.cfg.trainable,
        })
        
        if self.cfg.trainable:
            self.position = nn.Parameter(
                position.contiguous().requires_grad_(False)
            )
            self.features = nn.Parameter(
                features.contiguous().requires_grad_(True)
            )

        self.prefix_name = self.cfg.unwarp_prefix + "."
        ######################## ########################

        self.gs_atlas_cfg = gs_atlas_cfg
        self.start_frame_id = gs_atlas_cfg.start_frame_id
        self.end_frame_id = gs_atlas_cfg.end_frame_id
        self.time_len = gs_atlas_cfg.num_images - 1
        # self.mask = torch.from_numpy(mask).to(self.position.device)

        # Activation funcitons
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize
        self.wave_coefficient_activation = torch.sigmoid  # 添加 wave_coefficients 的激活函數

        scales, rots, opacities, features_rest = gaussian_point_init(
            position=self.position,
            max_sh_degree=self.cfg.max_sh_degree,
            # init_opacity=0.01
            init_opacity=0.5
        )

        ##### set scale threshold
        self.scale_threshold = 0.1

        fused_color = self.features.unsqueeze(1)
        self.features = (
            nn.Parameter(
                fused_color.contiguous().requires_grad_(True)
            )
        )
        num_points = len(self.position)
        self.register_atribute("features_rest", features_rest)
        self.register_atribute("scaling", scales)
        self.register_atribute("rotation", rots)
        self.register_atribute("opacity", opacities)
        self.register_atribute("pos_cubic_node", pos_cubic_coeff.reshape(-1, self.interval_num*3))
        scale_cubic_coeff = torch.rand((num_points, self.interval_num*3)).to(self.base_position.device).float() * 0.01
        rot_cubic_coeff = torch.rand((num_points, self.interval_num, 3)).to(self.base_position.device).float() * 0.01
        rot_cubic_coeff[..., 0] = 1.0  # 設置 w 分量為 1 /
        self.register_atribute("scale_cubic_node", scale_cubic_coeff)
        self.register_atribute("rot_cubic_node", rot_cubic_coeff.reshape(-1, self.interval_num*3))

        wave_coefficients = torch.zeros((num_points, num_total_frequencies)).to(self.base_position.device).float()
        self.register_atribute("wave_coefficients", wave_coefficients)

        ######### add image attributes to GS
        for attrib_name, feature_dim in gs_atlas_cfg.render_attributes.items():
            if attrib_name in ["pos_cubic_node", "rot_cubic_node", "scale_cubic_node"]:
                continue
            attribute = torch.zeros((num_points, feature_dim))
            self.register_atribute(attrib_name, attribute)
            if attrib_name == "mask_attribute":
                self.mask_attribute_activation = torch.sigmoid
            if attrib_name == "dino_attribute":
                self.dino_attribute_activation = torch.sigmoid

    

    @property
    def get_opacity(self):
        return self.opacity_activation(self.opacity)
    
    @property
    def get_scaling(self):
        return self.scaling_activation(self.scaling)

    # @property
    def get_scaling_by_time(self, time):
        # return self.scaling_activation(self.scaling)
        if isinstance(time, torch.Tensor):
            time = time.item()
        normed_time = (time + 1) / (len(self.delta_position) + 1)
        
        # Step 1: B-spline weights
        B = b_spline_basis(normed_time, self.intervals, k=3)  # [interval_num]
        B = torch.tensor(B, device=self.scale_cubic_node.device, dtype=self.scale_cubic_node.dtype)  # [interval_num]
        
        # Step 2: Get control values in log space
        coeff = self.scale_cubic_node.view(-1, self.interval_num, 3)  # [N, K, 3]
        
        # Step 3: Weighted sum in log space
        B = B.unsqueeze(0).unsqueeze(-1)  # [1, K, 1]
        delta_scale = (B * coeff).sum(dim=1)  # [N, 3]
        # delta_scale[:, 2] = 0.0
        
        # Step 4: Add to original scaling in log space and apply exp
        log_scale = self.scaling + delta_scale  # [N, 3]
        
        scale = self.scaling_activation(log_scale)  # [N, 3]
        
        return scale



    def get_rotation(self, time):
        if isinstance(time, torch.Tensor):
            time = time.item()
        normed_time = (time + 1) / (len(self.delta_position) + 1)
        
        # Step 1: B-spline weights
        B = b_spline_basis(normed_time, self.intervals, k=3)  # [interval_num]
        B = torch.tensor(B, device=self.rot_cubic_node.device, dtype=self.rot_cubic_node.dtype)  # [interval_num]
        
        # Step 2: Get control vectors in tangent space [num_prims, interval_num, 3]
        control_vectors = self.rot_cubic_node.view(-1, self.interval_num, 3)
        
        # Step 3: Reference quaternion q0 [N, 4]
        q0 = self.rotation  # [N, 4]
        q0 = self.rotation_activation(q0)  # 確保參考四元數是單位四元數
        
        # Step 4: Weighted sum of control vectors in tangent space
        B = B.unsqueeze(0).unsqueeze(-1)  # [1, K, 1]
        delta = (B * control_vectors).sum(dim=1)  # [N, 3]
        
        # Step 5: Convert tangent vector to quaternion and apply to reference
        delta_exp = quat_exp(delta)  # [N, 4]
        delta_exp = self.rotation_activation(delta_exp)  # 確保增量四元數是單位四元數
        
        # Step 6: Final rotation: q0 * exp(∑ B_i(t) * v_i)
        rot = quat_mul(q0, delta_exp)  # [N, 4]
        rot = self.rotation_activation(rot)  # 確保最終四元數是單位四元數
        
        return rot

    @property
    def get_covariance(self, scaling_modifier=1):
        ### Not called in dptr
        return self.covariance_activation(
            self.get_scaling,
            scaling_modifier,
            self.get_rotation,
        )

    @property
    def get_shs(self):
        return torch.cat([
            self.features, self.features_rest,
        ], dim=1)
    
    
    def get_position(self, time, detach_pos=False):
        # time_index = time - self.start_frame_id TODO pay attention to this !
        if isinstance(time, torch.Tensor):
            time = time.item()
        normed_time = (time + 1) / (len(self.delta_position) + 1)
        
        B = b_spline_basis(normed_time, self.intervals, k=3)
        coeff = self.pos_cubic_node.view(-1, self.interval_num, 3)
        B = torch.tensor(B, device=coeff.device, dtype=coeff.dtype).unsqueeze(0).unsqueeze(-1)
        pos = (B * coeff).sum(dim=1)

        return pos + self.position
        
    
    @property
    def get_pos_cubic_node(self):
        return self.pos_cubic_node
    
    @property
    def get_rot_cubic_node(self):
        return self.rot_cubic_node
    
    @property
    def get_scale_cubic_node(self):
        return self.scale_cubic_node
    
    @property
    def get_mask_attribute(self):
        return self.mask_attribute_activation(self.mask_attribute)
    
    @property
    def get_dino_attribute(self):
        return self.dino_attribute_activation(self.dino_attribute)
    
    @property
    def get_wave_coefficients(self):
        return self.wave_coefficient_activation(self.wave_coefficients)
    
    def get_topk_waves(self):
        coefficients_to_send, indices_to_send = torch.topk(torch.abs(self.get_wave_coefficients),
                                num_frequencies, dim=1, sorted=False)
        coefficients_to_send = coefficients_to_send * torch.gather(self.get_wave_coefficients.sign(), dim=1, index=indices_to_send)
        indices_to_send = indices_to_send.type(torch.int)
        
        return coefficients_to_send, indices_to_send


    def re_init(self, num_points):
        super().re_init(num_points)
        fused_color = self.features.unsqueeze(1)
        self.features = (
            nn.Parameter(
                fused_color.contiguous().requires_grad_(True)
            )
        )
        scales, rots, opacities, features_rest = gaussian_point_init(
            position=self.position,
            max_sh_degree=self.cfg.max_sh_degree,
            init_opacity=0.5
        )
        self.register_atribute("features_rest", features_rest)
        self.register_atribute("scaling", scales)
        self.register_atribute("rotation", rots)
        self.register_atribute("opacity", opacities)

        ######## add dyanmic attributes
        num_points = len(self.position)
        cubic_coeff = torch.zeros((num_points, self.interval_num*3))
        self.register_atribute("pos_cubic_node", cubic_coeff)
        rot_cubic_coeff = torch.zeros((num_points, self.interval_num, 3))
        # rot_cubic_coeff[..., 0] = 1.0  # 設置 w 分量為 1
        self.register_atribute("rot_cubic_node", rot_cubic_coeff.reshape(-1, self.interval_num*3))
        scale_cubic_coeff = torch.zeros((num_points, self.interval_num*3))
        self.register_atribute("scale_cubic_node", scale_cubic_coeff)
        
        wave_coefficients = torch.zeros((num_points, num_total_frequencies)).to(self.base_position.device).float()
        self.register_atribute("wave_coefficients", wave_coefficients)

        ######### add image attributes to GS
        for attrib_name, feature_dim in self.gs_atlas_cfg.render_attributes.items():
            if attrib_name in ["pos_cubic_node", "rot_cubic_node", "scale_cubic_node"]:
                continue
            attribute = torch.zeros((num_points, feature_dim))
            self.register_atribute(attrib_name, attribute)
            if attrib_name == "mask_attribute":
                self.mask_attribute_activation = torch.sigmoid
            if attrib_name == "dino_attribute":
                self.dino_attribute_activation = torch.sigmoid