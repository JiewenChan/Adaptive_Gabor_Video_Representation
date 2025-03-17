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



@POINTSCLOUD_REGISTRY.register()
class DynamicBsplineGaussianPointCloud(PointCloud):
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

    def setup(self, gs_atlas_cfg, point_cloud=None):
        # ##### init point cloud use depth
        # depth_map = np.load(gs_atlas_cfg.start_depth_npy)
        # depth_map = depth_map + np.random.randn(*depth_map.shape) * 0.01
        # pcd = depth2pcd(depth_map)
        # image = imageio.imread(gs_atlas_cfg.start_frame_path)[..., :3] / 255.
        # mask = imageio.imread(gs_atlas_cfg.start_frame_mask_path) > 0
        # mask = np.ones_like(mask, dtype=bool)
        # if gs_atlas_cfg.reverse_mask:
        #     mask = ~mask
        # ##### use downsample to reduce the number of points
        # pcd = pcd[mask][::5]
        # colors = image[mask][::5]
        # point_cloud = SimplePointCloud(positions=pcd, colors=colors, normals=None)

        super().setup(point_cloud)
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

        scales, rots, opacities, features_rest = gaussian_point_init(
            position=self.position,
            max_sh_degree=self.cfg.max_sh_degree,
            init_opacity=0.01
        )

        ######## add dyanmic attributes
        num_points = len(self.position)
        self.poly_feature_dim = 4
        self.fourier_feature_dim = 4 * 2
        self.interval_num = math.ceil(gs_atlas_cfg.num_images / 3)  # set a node every 5 frames
        self.intervals_idx = torch.linspace(0, 1, self.interval_num - 2)
        self.intervals = torch.cat([torch.zeros(3), self.intervals_idx, torch.ones(3)])
        pos_poly_feat = torch.zeros((num_points, self.poly_feature_dim, 3))
        pos_fourier_feat = torch.zeros((num_points, self.fourier_feature_dim, 3))
        rot_poly_feat = torch.zeros((num_points, self.poly_feature_dim, 4))
        rot_fourier_feat = torch.zeros((num_points, self.fourier_feature_dim, 4))
        cubic_coeff = torch.zeros((num_points, self.interval_num, 3))

        fused_color = self.features.unsqueeze(1)
        self.features = (
            nn.Parameter(
                fused_color.contiguous().requires_grad_(True)
            )
        )
        self.register_atribute("features_rest", features_rest)
        self.register_atribute("scaling", scales)
        self.register_atribute("rotation", rots)
        self.register_atribute("opacity", opacities)
        self.register_atribute("pos_poly_feat", pos_poly_feat)
        self.register_atribute("pos_fourier_feat", pos_fourier_feat)
        self.register_atribute("rot_poly_feat", rot_poly_feat)
        self.register_atribute("rot_fourier_feat", rot_fourier_feat)
        self.register_atribute("pos_cubic_node", cubic_coeff.reshape(-1, self.interval_num*3))

        ######### add image attributes to GS
        for attrib_name, feature_dim in gs_atlas_cfg.render_attributes.items():
            if attrib_name in ["pos_poly_feat", "pos_fourier_feat", "rot_poly_feat", "rot_fourier_feat"]:
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
    # def get_rotation(self):
    #     return self.rotation_activation(self.rotation)

    def get_rotation(self, time):
        rotation = self.rotation
        normed_time = (time - self.start_frame_id) / self.time_len
        basis = torch.arange(self.poly_feature_dim).float().to(rotation.device)
        poly_basis = torch.pow(normed_time, basis)[None, :, None]

        # al * cos(lt) + bl * sin(lt)
        basis = torch.arange(self.fourier_feature_dim/2).float().to(rotation.device) + 1
        fourier_basis = [torch.cos(normed_time * basis * np.pi), torch.sin(normed_time * basis * np.pi)]
        fourier_basis = torch.cat(fourier_basis, dim=0)[None, :, None]
                            
        rotation = rotation \
            + torch.sum(self.rot_poly_feat * poly_basis, dim=1).detach() \
            + torch.sum(self.rot_fourier_feat * fourier_basis, dim=1).detach()
        return self.rotation_activation(rotation)


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
        normed_time = time / (self.time_len + 1)
        
        B = b_spline_basis(normed_time, self.intervals, k=3)
        coeff = self.pos_cubic_node.view(-1, self.interval_num, 3)
        B = torch.tensor(B, device=coeff.device, dtype=coeff.dtype).unsqueeze(0).unsqueeze(-1)
        pos = (B * coeff).sum(dim=1)

        ### use cubic spline interpolation
        # coeff = self.pos_cubic_node.reshape(-1, 4, self.interval_num, 3)
        # # minus 1e-7 to avoid numeric error
        # indices = torch.searchsorted(self.intervals, normed_time-1e-7, right=False) - 1
        # indices = torch.clamp(indices, min=0)
        # distances = normed_time - self.intervals[indices]  # [1,]
        # pos = coeff[:,3, indices] + coeff[:,2, indices] * distances + \
        #     coeff[:,1, indices] * distances**2 + coeff[:,0, indices] * distances**3  # [Np, 1, 3]
        # # return pos[:,0,:]
        return pos + self.position
        # position = self.position
        # normed_time = (time - self.start_frame_id) / self.time_len
        # basis = torch.arange(self.poly_feature_dim).float().to(position.device)
        # poly_basis = torch.pow(normed_time, basis)[None, :, None]

        # # al * cos(lt) + bl * sin(lt)
        # basis = torch.arange(self.fourier_feature_dim/2).float().to(position.device) + 1
        # fourier_basis = [torch.cos(normed_time * basis * np.pi), torch.sin(normed_time * basis * np.pi)]
        # fourier_basis = torch.cat(fourier_basis, dim=0)[None, :, None]
                            
        # if detach_pos:
        #     return position.detach() + torch.sum(self.pos_poly_feat * poly_basis, dim=1) + torch.sum(self.pos_fourier_feat * fourier_basis, dim=1)
        # else:
        #     return position \
        #         + torch.sum(self.pos_poly_feat * poly_basis, dim=1) \
        #         + torch.sum(self.pos_fourier_feat * fourier_basis, dim=1)
    
    @property
    def get_pos_poly_feat(self):
        return self.pos_poly_feat
    
    @property
    def get_pos_fourier_feat(self):
        return self.pos_fourier_feat
    
    @property
    def get_rot_poly_feat(self):
        return self.rot_poly_feat
    
    @property
    def get_rot_fourier_feat(self):
        return self.rot_fourier_feat
    
    @property
    def get_mask_attribute(self):
        return self.mask_attribute_activation(self.mask_attribute)
    
    @property
    def get_dino_attribute(self):
        return self.dino_attribute_activation(self.dino_attribute)


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
        )
        self.register_atribute("features_rest", features_rest)
        self.register_atribute("scaling", scales)
        self.register_atribute("rotation", rots)
        self.register_atribute("opacity", opacities)

        ######## add dyanmic attributes
        num_points = len(self.position)
        pos_poly_feat = torch.zeros((num_points, self.poly_feature_dim, 3))
        self.register_atribute("pos_poly_feat", pos_poly_feat)
        pos_fourier_feat = torch.zeros((num_points, self.fourier_feature_dim, 3))
        self.register_atribute("pos_fourier_feat", pos_fourier_feat)
        rot_poly_feat = torch.zeros((num_points, self.poly_feature_dim, 4))
        self.register_atribute("rot_poly_feat", rot_poly_feat)
        rot_fourier_feat = torch.zeros((num_points, self.fourier_feature_dim, 4))
        self.register_atribute("rot_fourier_feat", rot_fourier_feat)

        cubic_coeff = torch.zeros((num_points, self.interval_num*3))
        self.register_atribute("pos_cubic_node", cubic_coeff)

        ######### add image attributes to GS
        for attrib_name, feature_dim in self.gs_atlas_cfg.render_attributes.items():
            if attrib_name in ["pos_poly_feat", "pos_fourier_feat", "rot_poly_feat", "rot_fourier_feat"]:
                continue
            attribute = torch.zeros((num_points, feature_dim))
            self.register_atribute(attrib_name, attribute)
            if attrib_name == "mask_attribute":
                self.mask_attribute_activation = torch.sigmoid
            if attrib_name == "dino_attribute":
                self.dino_attribute_activation = torch.sigmoid