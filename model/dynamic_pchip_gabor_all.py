import torch
from torch import nn
from pytorch_msssim import ms_ssim
from dataclasses import dataclass
import math
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


def quat_normalize(q, eps=1e-8):
    return q / (torch.norm(q, dim=-1, keepdim=True) + eps)

def quat_mul(q1, q2):
    # [w,x,y,z] × [w,x,y,z]
    assert q1.shape[-1] == 4 and q2.shape[-1] == 4
    w1, x1, y1, z1 = torch.unbind(q1, dim=-1)
    w2, x2, y2, z2 = torch.unbind(q2, dim=-1)
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return torch.stack([w, x, y, z], dim=-1)

def quat_conjugate(q):
    return torch.stack([q[...,0], -q[...,1], -q[...,2], -q[...,3]], dim=-1)

def quat_log(q, eps=1e-8):
    # Log of a unit quaternion, returning an so(3) vector.
    q = quat_normalize(q, eps)
    w = q[..., 0:1].clamp(-1.0, 1.0)
    v = q[..., 1:]
    theta = torch.acos(w)                        # ∈ [0, π]
    sin_theta = torch.sin(theta)
    scale = theta / (sin_theta + eps)            # When theta->0, scale ~1; use eps to stabilize near pi.
    return scale * v                             # Equivalent to (theta * v / sin(theta)).

def quat_exp(v, eps=1e-8):
    # so(3) vector -> unit quaternion.
    norm_v = torch.norm(v, dim=-1, keepdim=True)
    sin_n  = torch.sin(norm_v)
    cos_n  = torch.cos(norm_v)
    scale  = torch.where(norm_v > eps, sin_n / norm_v, torch.ones_like(norm_v))
    q = torch.cat([cos_n, scale * v], dim=-1)
    return quat_normalize(q, eps)



num_total_frequencies = 2
num_frequencies = 2


class _HardSigmoidSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor):
        ctx.save_for_backward(input_tensor)
        return torch.clamp((input_tensor + 1.0) * 0.5, 0.0, 1.0)

    @staticmethod
    def backward(ctx, grad_output):
        (input_tensor,) = ctx.saved_tensors
        soft_val = torch.sigmoid(input_tensor)
        grad_input = grad_output * soft_val * (1.0 - soft_val)
        return grad_input


def hard_sigmoid_ste(input_tensor):
    return _HardSigmoidSTE.apply(input_tensor)


@POINTSCLOUD_REGISTRY.register()
class DynamicPchipGaborAll(PointCloud):
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
        random_init_position: bool = False
        random_init_radius: float = 1.0
        random_init_positive_z: bool = False

    cfg: Config

    def setup(self, gs_atlas_cfg, base_point_seq=None, num_frames=None):
        base_point_seq = torch.stack([x[~torch.isnan(x).any(dim=1)] for x in base_point_seq], dim=0)
        ### assume first frame is the base frame
        if self.cfg.random_init_position:
            num_points = base_point_seq.shape[1]
            base_position = get_random_points(num_points, self.cfg.random_init_radius).to(
                base_point_seq.device
            )
            if self.cfg.random_init_positive_z:
                base_position[:, 2] = base_position[:, 2] + 1.0
            self.base_position = base_position
            self.delta_position = torch.zeros_like(base_point_seq)
        else:
            self.base_position = base_point_seq[0]
            self.delta_position = base_point_seq - self.base_position[None,...]
        num_points = len(self.base_position)
        
        if num_frames is not None:
            self.num_frames = num_frames
        else:
            self.num_frames = len(base_point_seq)
        
        self.pos_interval_num = int(math.ceil(len(base_point_seq) / 2))
        self.rot_interval_num = int(math.ceil(len(base_point_seq) / 10))
        
        self.intervals_idx = torch.linspace(
            0,
            len(self.delta_position) - 1,
            steps=int(self.pos_interval_num),
        ).long().to(self.base_position.device)
        pos_nodes = self.delta_position[self.intervals_idx]  # [num_nodes, Np, 3]
        pos_cubic_coeff = pos_nodes.permute(1, 0, 2).contiguous()
        self.poly_feature_dim = 4
        rot_cubic_coeff = torch.ones((num_points, self.rot_interval_num, 3)).to(self.base_position.device).float()
        # rot_cubic_coeff[..., 0] = 1.0  # Set w component to 1
        
        
        pos_tk = self.intervals_idx.float()
        rot_tk = torch.linspace(
            0.0,
            self.num_frames - 1.0,
            steps=int(self.rot_interval_num),
            dtype=torch.float32,
        )
        self.register_buffer('pos_tk', pos_tk, persistent=True)
        self.register_buffer('rot_tk', rot_tk, persistent=True)
        

        self.atributes = []
        position = self.base_position
        features = get_random_feauture(len(position), self.cfg.initializer.feat_dim)
        self.register_buffer('position', position)
        self.register_buffer('features', features)
        self.atributes.append({
            'name': 'position',
            'trainable': False,
        })
        self.atributes.append({
            'name': 'features',
            'trainable': self.cfg.trainable,
        })
        

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
        
        def _safe_normalize(q, eps=1e-12):
            return q / (q.norm(dim=-1, keepdim=True).clamp_min(eps))

        self.rotation_activation = _safe_normalize

        # Activation functions
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        # self.rotation_activation = torch.nn.functional.normalize
        self.wave_coefficient_activation = hard_sigmoid_ste
        self.min_frequency = 2.0  # Minimum frequency
        self.max_frequency = 16.0  # Maximum frequency
        
        # self.wave_coefficient_indices_activation = lambda x: torch.nn.ReLU()(x) + 1.0

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
    
        print(f"Initializing num_points: {num_points}")
        self.register_atribute("features_rest", features_rest)
        self.register_atribute("scaling", scales)
        self.register_atribute("rotation", rots)
        self.register_atribute("opacity", opacities)
        self.register_atribute("pos_cubic_node", pos_cubic_coeff.reshape(num_points, -1))
        self.register_atribute("rot_cubic_node", rot_cubic_coeff.reshape(num_points, -1))
        # self.register_atribute("pos_tangent_node", mk0.reshape(num_points, -1))

        wave_coefficients = 0.1 * torch.randn(
            (num_points, num_total_frequencies),
            device=self.base_position.device,
            dtype=torch.float32,
        )
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

    def _poly_eval_horner(self, u, coeff_nd3):
        # coeff_nd3: (Np, D+1, 3), lower-order term at [:,0,:].
        c = coeff_nd3
        u = torch.as_tensor(u, device=c.device, dtype=c.dtype).reshape(())
        y = c[:, -1, :]
        for d in range(self.poly_feature_dim - 2, -1, -1):
            y = y * u + c[:, d, :]
        return y

    def get_scaling_by_time(self, time):
        del time  # scaling is static, keep signature for compatibility
        return self.scaling_activation(self.scaling)


    def get_rotation(self, time):
        yk = self.get_rot_cubic_node.view(-1, self.rot_interval_num, 3).to(self.rotation.dtype)
        tk = self.rot_tk.to(device=yk.device, dtype=yk.dtype)
        mk = self._auto_slopes(yk, tk)
        delta = self._hermite_eval_scalar(time, yk, mk, tk)

        q_base = self.rotation_activation(self.rotation)                         # (N, 4)
        dq     = quat_exp(delta)                                                                # (N, 4)
        q      = quat_mul(q_base, dq)    

        return self.rotation_activation(q)


    # @property
    # def get_covariance(self, scaling_modifier=1):
    #     ### Not called in dptr
    #     return self.covariance_activation(
    #         self.get_scaling,
    #         scaling_modifier,
    #         self.get_rotation,
    #     )

    @property
    def get_shs(self):
        return torch.cat([
            self.features, self.features_rest,
        ], dim=1)
    
    
    def _auto_slopes(self, yk, tk, tau=0.7, end_shrink=0.5, monotone_gate=True):
        # yk: (Np, M, 3), tk: (M,)
        eps = torch.finfo(yk.dtype).eps
        h = tk[1:] - tk[:-1]                                           # (M-1,)
        delta = (yk[:,1:,:] - yk[:,:-1,:]) / (h.view(1,-1,1) + eps)   # (Np, M-1, 3)

        mk = torch.zeros_like(yk)
        if delta.shape[1] >= 2:
            m_mid = 0.5 * (delta[:,:-1,:] + delta[:,1:,:])            # Centered difference.
            if monotone_gate:
                same = (delta[:,:-1,:] * delta[:,1:,:]) > 0           # Keep only same-sign to avoid new extrema.
                m_mid = torch.where(same, m_mid, torch.zeros_like(m_mid))
            mk[:,1:-1,:] = tau * m_mid                                # Smaller steps are more stable.

        mk[:,0,:]  = end_shrink * delta[:,0,:]                        # Endpoints.
        mk[:,-1,:] = end_shrink * delta[:,-1,:]
        return mk

    def _hermite_eval_scalar(self, t_in, yk, mk, tk):
        tk  = tk.to(device=yk.device, dtype=yk.dtype)        # (M,)
        eps = torch.finfo(yk.dtype).eps
        t = torch.as_tensor(t_in, device=yk.device, dtype=yk.dtype).reshape(())
        t = torch.clamp(t, tk[0], tk[-1])

        i = torch.searchsorted(tk, t) - 1
        i = torch.clamp(i, 0, tk.numel()-2)
        t0, t1 = tk[i], tk[i+1]
        h  = t1 - t0
        s  = (t - t0) / (h + eps)

        y0, y1 = yk[:, i, :],   yk[:, i+1, :]
        m0, m1 = mk[:, i, :],   mk[:, i+1, :]

        s2, s3 = s*s, s*s*s
        H00 =  2*s3 - 3*s2 + 1
        H10 =      s3 - 2*s2 + s
        H01 = -2*s3 + 3*s2
        H11 =      s3 -   s2

        return H00*y0 + H10*(h*m0) + H01*y1 + H11*(h*m1)

    def get_position(self, time, detach_pos=False):
        if time == -1:
            return self.position
        yk = self.pos_cubic_node.view(-1, self.pos_interval_num, 3).to(self.position.dtype)
        tk = self.pos_tk.to(device=yk.device, dtype=yk.dtype)
        mk = self._auto_slopes(yk, tk)
        delta = self._hermite_eval_scalar(time, yk, mk, tk)
        return self.position + delta
    
    @property
    def get_pos_cubic_node(self):
        return self.pos_cubic_node
    
    @property
    def get_rot_cubic_node(self):
        def wrap_so3(v, eps=1e-8):
            """
            v: (..., 3) so(3) vector
            Returns: map rotation angles back to the equivalent shortest arc in (-pi, pi].
            """
            theta = torch.norm(v, dim=-1, keepdim=True)
            axis  = v / (theta + eps)
            theta_wrapped = torch.atan2(torch.sin(theta), torch.cos(theta))  # wrap to (-π, π]
            return axis * theta_wrapped
        rot = self.rot_cubic_node.view(-1, self.rot_interval_num, 3)
        rot_wrapped = wrap_so3(rot)
        return rot_wrapped.reshape(-1, self.rot_interval_num*3)
    
    @property
    def get_mask_attribute(self):
        return self.mask_attribute_activation(self.mask_attribute)
    
    @property
    def get_dino_attribute(self):
        return self.dino_attribute_activation(self.dino_attribute)
    
    @property
    def get_wave_coefficients(self):
        # return torch.zeros_like(self.wave_coefficients)
        return self.wave_coefficient_activation(self.wave_coefficients)
    
    # @property
    # def get_wave_coefficient_indices(self):
    #     return self.wave_coefficient_indices_activation(self.wave_coefficient_indices)
    
    def get_topk_waves(self):
        coefficients_to_send = self.get_wave_coefficients
        indices_to_send = torch.ones_like(coefficients_to_send)
        indices_to_send[:, 1] *= 2.0
        # indices_to_send = indices_to_send.int()
        # indices_to_send = self.get_wave_coefficient_indices
        
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
        self.register_atribute("scaling", scales, trainable=False)
        self.register_atribute("rotation", rots)
        self.register_atribute("opacity", opacities)

        ######## add dynamic attributes
        num_points = len(self.position)
        cubic_coeff = torch.zeros((num_points, self.pos_interval_num*3))
        self.register_atribute("pos_cubic_node", cubic_coeff)
        rot_cubic_coeff = torch.zeros((num_points, self.rot_interval_num, 3))
        # mk0 = torch.zeros_like(cubic_coeff)
        # self.register_atribute("pos_tangent_node", mk0.reshape(num_points, -1))
        # rot_cubic_coeff[..., 0] = 1.0  # Set w component to 1
        self.register_atribute("rot_cubic_node", rot_cubic_coeff.reshape(-1, self.rot_interval_num*3))
        wave_coefficients = 0.1 * torch.randn(
            (num_points, num_total_frequencies),
            device=self.base_position.device,
            dtype=torch.float32,
        )
        # wave_coefficient_indices = torch.zeros((num_points, num_total_frequencies), device=self.base_position.device).float()
        self.register_atribute("wave_coefficients", wave_coefficients)
        # self.register_atribute("wave_coefficient_indices", wave_coefficient_indices)

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
                
                
