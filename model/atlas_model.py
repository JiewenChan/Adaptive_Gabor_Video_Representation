import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Mapping, Optional, Union, Any
from omegaconf import DictConfig
from pytorch_msssim import ms_ssim

from pointrix.utils.base import BaseModule
from pointrix.utils.config import parse_structured
from pointrix.point_cloud import parse_point_cloud
from pointrix.model.loss import l1_loss, ssim, psnr
from pointrix.model.base_model import BaseModel, MODEL_REGISTRY
from pointrix.point_cloud import POINTSCLOUD_REGISTRY
import numpy as np
from pointrix.utils.dataset.dataset_utils import fov2focal, focal2fov
from pointrix.dataset.base_data import SimplePointCloud
from model.dynamic_gaussian_points import DynamicGaussianPointCloud
from model.dynamic_gaussian_with_base_point_cloud import DynamicGaussianWithBasePointCloud
from model.dynamic_bspline_gaussian_points import DynamicBsplineGaussianPointCloud
from model.dynamic_bspline_gaussian_with_base_point_cloud import DynamicBsplineGaussianWithBasePointCloud
from model.dynamic_bspline_gaussian_all import DynamicBsplineGaussianAll
# from lbs_gaussian_point import LBSGaussianPointCloud
# from dust3r_interface import Dust3R
from itertools import accumulate
import operator


class SingleAtlasModel(BaseModel):
    """
    A class for the Single Atlas Model.
    """
    @dataclass
    class Config:
        point_cloud: dict = field(default_factory=dict)
        lambda_dssim: float = 0.2

    cfg: Config

    def setup(self, gs_atlas_cfg, point_cloud=None, gaussian_class=DynamicGaussianPointCloud, device="cuda"):
        self.gs_atlas_cfg = gs_atlas_cfg
        self.point_cloud = gaussian_class(self.cfg.point_cloud, gs_atlas_cfg, point_cloud).to(device)
        self.render_attributes_list = list(gs_atlas_cfg.render_attributes.keys())

    def forward(self, ids, batch=None) -> dict:
        """
        Forward pass of the model.

        Parameters
        ----------
        batch : dict
            The batch of data.
        
        Returns
        -------
        dict
            The render results which will be the input of renderers.
        """
        position = self.point_cloud.get_position(ids)
        detached_position = self.point_cloud.get_position(ids, detach_pos=True)
        shs = self.point_cloud.get_shs
        opacity = self.point_cloud.get_opacity
        rotation = self.point_cloud.get_rotation(ids)
        scaling = self.point_cloud.get_scaling
        pos_poly_feat = self.point_cloud.get_pos_poly_feat
        pos_fourier_feat = self.point_cloud.get_pos_fourier_feat
        rot_poly_feat = self.point_cloud.get_rot_poly_feat
        rot_fourier_feat = self.point_cloud.get_rot_fourier_feat
        render_dict = {
            "position": position,
            "detached_position": detached_position,
            "opacity": opacity,
            "scaling": scaling,
            "rotation": rotation,
            "shs": shs,
            "pos_poly_feat": pos_poly_feat.reshape(-1, pos_poly_feat.shape[-1]*pos_poly_feat.shape[-2]),
            "pos_fourier_feat": pos_fourier_feat.reshape(-1, pos_fourier_feat.shape[-1]*pos_fourier_feat.shape[-2]),
            "rot_poly_feat": rot_poly_feat.reshape(-1, rot_poly_feat.shape[-1]*rot_poly_feat.shape[-2]),
            "rot_fourier_feat": rot_fourier_feat.reshape(-1, rot_fourier_feat.shape[-1]*rot_fourier_feat.shape[-2]),
        }
        for attr in self.render_attributes_list:
            if attr not in render_dict:
                render_dict[attr] = getattr(self.point_cloud, f"get_{attr}")
        return render_dict


class SingleAtlasWithBaseModel(BaseModel):
    """
    A class for the Single Atlas Model.
    """
    @dataclass
    class Config:
        point_cloud: dict = field(default_factory=dict)
        lambda_dssim: float = 0.2

    cfg: Config

    def setup(self, gs_atlas_cfg, base_point_seq=None, gaussian_class=DynamicGaussianWithBasePointCloud, device="cuda"):
        self.gs_atlas_cfg = gs_atlas_cfg
        self.point_cloud = gaussian_class(self.cfg.point_cloud, gs_atlas_cfg, base_point_seq).to(device)
        self.render_attributes_list = list(gs_atlas_cfg.render_attributes.keys())

    def forward(self, ids, batch=None) -> dict:
        """
        Forward pass of the model.

        Parameters
        ----------
        batch : dict
            The batch of data.
        
        Returns
        -------
        dict
            The render results which will be the input of renderers.
        """
        position = self.point_cloud.get_position(ids)
        detached_position = self.point_cloud.get_position(ids, detach_pos=True)
        shs = self.point_cloud.get_shs
        opacity = self.point_cloud.get_opacity
        rotation = self.point_cloud.get_rotation(ids)
        scaling = self.point_cloud.get_scaling
        pos_poly_feat = self.point_cloud.get_pos_poly_feat
        pos_fourier_feat = self.point_cloud.get_pos_fourier_feat
        rot_poly_feat = self.point_cloud.get_rot_poly_feat
        rot_fourier_feat = self.point_cloud.get_rot_fourier_feat
        render_dict = {
            "position": position,
            "detached_position": detached_position,
            "opacity": opacity,
            "scaling": scaling,
            "rotation": rotation,
            "shs": shs,
            "pos_poly_feat": pos_poly_feat.reshape(-1, pos_poly_feat.shape[-1]*pos_poly_feat.shape[-2]),
            "pos_fourier_feat": pos_fourier_feat.reshape(-1, pos_fourier_feat.shape[-1]*pos_fourier_feat.shape[-2]),
            "rot_poly_feat": rot_poly_feat.reshape(-1, rot_poly_feat.shape[-1]*rot_poly_feat.shape[-2]),
            "rot_fourier_feat": rot_fourier_feat.reshape(-1, rot_fourier_feat.shape[-1]*rot_fourier_feat.shape[-2]),
        }
        for attr in self.render_attributes_list:
            if attr not in render_dict:
                render_dict[attr] = getattr(self.point_cloud, f"get_{attr}")
        return render_dict
    

class SingleAtlasLBSModel(BaseModel):
    """
    A class for the Single Atlas Model.
    """
    @dataclass
    class Config:
        point_cloud: dict = field(default_factory=dict)
        lambda_dssim: float = 0.2

    cfg: Config

    def setup(self, gs_atlas_cfg, base_point_seq, device="cuda"):
        self.gs_atlas_cfg = gs_atlas_cfg
        self.point_cloud = LBSGaussianPointCloud(self.cfg.point_cloud, gs_atlas_cfg, base_point_seq).to(device)
        self.render_attributes_list = list(gs_atlas_cfg.render_attributes.keys())

    def forward(self, ids, batch=None) -> dict:
        """
        Forward pass of the model.

        Parameters
        ----------
        batch : dict
            The batch of data.
        
        Returns
        -------
        dict
            The render results which will be the input of renderers.
        """
        position = self.point_cloud.get_position(ids)
        # delta_pos = 
        
        
        detached_position = position.detach()



        shs = self.point_cloud.get_shs
        opacity = self.point_cloud.get_opacity
        rotation = self.point_cloud.get_rotation(ids)
        scaling = self.point_cloud.get_scaling
        pos_poly_feat = self.point_cloud.get_pos_poly_feat
        pos_fourier_feat = self.point_cloud.get_pos_fourier_feat
        rot_poly_feat = self.point_cloud.get_rot_poly_feat
        rot_fourier_feat = self.point_cloud.get_rot_fourier_feat
        render_dict = {
            "position": position,
            "detached_position": detached_position,
            "opacity": opacity,
            "scaling": scaling,
            "rotation": rotation,
            "shs": shs,
            "pos_poly_feat": pos_poly_feat.reshape(-1, pos_poly_feat.shape[-1]*pos_poly_feat.shape[-2]),
            "pos_fourier_feat": pos_fourier_feat.reshape(-1, pos_fourier_feat.shape[-1]*pos_fourier_feat.shape[-2]),
            "rot_poly_feat": rot_poly_feat.reshape(-1, rot_poly_feat.shape[-1]*rot_poly_feat.shape[-2]),
            "rot_fourier_feat": rot_fourier_feat.reshape(-1, rot_fourier_feat.shape[-1]*rot_fourier_feat.shape[-2]),
        }
        for attr in self.render_attributes_list:
            if attr not in render_dict:
                render_dict[attr] = getattr(self.point_cloud, f"get_{attr}")
        return render_dict
    
    
class SingleAtlasBsplineModel(BaseModel):
    """
    A class for the Single Atlas Model.
    """
    @dataclass
    class Config:
        point_cloud: dict = field(default_factory=dict)
        lambda_dssim: float = 0.2

    cfg: Config

    def setup(self, gs_atlas_cfg, point_cloud=None, gaussian_class=DynamicBsplineGaussianAll, device="cuda"):
        self.gs_atlas_cfg = gs_atlas_cfg
        self.point_cloud = gaussian_class(self.cfg.point_cloud, gs_atlas_cfg, point_cloud).to(device)
        self.render_attributes_list = list(gs_atlas_cfg.render_attributes.keys())

    def forward(self, ids, batch=None) -> dict:
        """
        Forward pass of the model.

        Parameters
        ----------
        batch : dict
            The batch of data.
        
        Returns
        -------
        dict
            The render results which will be the input of renderers.
        """
        position = self.point_cloud.get_position(ids)
        detached_position = self.point_cloud.get_position(ids, detach_pos=True)
        shs = self.point_cloud.get_shs
        opacity = self.point_cloud.get_opacity
        rotation = self.point_cloud.get_rotation(ids)
        scaling = self.point_cloud.get_scaling
        pos_cubic_node = self.point_cloud.get_pos_cubic_node
        rot_cubic_node = self.point_cloud.get_rot_cubic_node
        render_dict = {
            "position": position,
            "detached_position": detached_position,
            "opacity": opacity,
            "scaling": scaling,
            "rotation": rotation,
            "shs": shs,
            "pos_cubic_node": pos_cubic_node,
            "rot_cubic_node": rot_cubic_node,
        }
        for attr in self.render_attributes_list:
            if attr not in render_dict:
                render_dict[attr] = getattr(self.point_cloud, f"get_{attr}")
        return render_dict
    

class SingleAtlasBsplineWithBaseModel(BaseModel):
    """
    A class for the Single Atlas Model.
    """
    @dataclass
    class Config:
        point_cloud: dict = field(default_factory=dict)
        lambda_dssim: float = 0.2

    cfg: Config

    def setup(self, gs_atlas_cfg, base_point_seq=None, gaussian_class=DynamicBsplineGaussianAll, device="cuda"):
        self.gs_atlas_cfg = gs_atlas_cfg
        self.point_cloud = gaussian_class(self.cfg.point_cloud, gs_atlas_cfg, base_point_seq).to(device)
        self.render_attributes_list = list(gs_atlas_cfg.render_attributes.keys())

    def forward(self, ids, batch=None) -> dict:
        """
        Forward pass of the model.

        Parameters
        ----------
        batch : dict
            The batch of data.
        
        Returns
        -------
        dict
            The render results which will be the input of renderers.
        """
        position = self.point_cloud.get_position(ids)
        detached_position = self.point_cloud.get_position(ids, detach_pos=True)
        shs = self.point_cloud.get_shs
        opacity = self.point_cloud.get_opacity
        rotation = self.point_cloud.get_rotation(ids)
        scaling = self.point_cloud.get_scaling_by_time(ids)
        pos_cubic_node = self.point_cloud.get_pos_cubic_node
        rot_cubic_node = self.point_cloud.get_rot_cubic_node
        render_dict = {
            "position": position,
            "detached_position": detached_position,
            "opacity": opacity,
            "scaling": scaling,
            "rotation": rotation,
            "shs": shs,
            "pos_cubic_node": pos_cubic_node,
            "rot_cubic_node": rot_cubic_node,
        }
        for attr in self.render_attributes_list:
            if attr not in render_dict:
                render_dict[attr] = getattr(self.point_cloud, f"get_{attr}")
        return render_dict