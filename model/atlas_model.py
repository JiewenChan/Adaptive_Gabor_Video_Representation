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
from itertools import accumulate
import operator

    
class SingleAtlasPchipWithBaseModel(BaseModel):
    """
    A class for the Single Atlas Model.
    """
    @dataclass
    class Config:
        point_cloud: dict = field(default_factory=dict)
        lambda_dssim: float = 0.2

    cfg: Config

    def setup(self, gs_atlas_cfg, base_point_seq=None, gabor=False, device="cuda", num_frames=None):
        self.gabor = gabor
        self.gs_atlas_cfg = gs_atlas_cfg
        if self.gabor:
            from model.dynamic_pchip_gabor_all import DynamicPchipGaborAll
            self.point_cloud = DynamicPchipGaborAll(self.cfg.point_cloud, gs_atlas_cfg, base_point_seq, num_frames=num_frames).to(device)
        else:
            from model.dynamic_pchip_gaussian_all import DynamicPchipGaussianAll
            self.point_cloud = DynamicPchipGaussianAll(self.cfg.point_cloud, gs_atlas_cfg, base_point_seq, num_frames=num_frames).to(device)
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
        shs = self.point_cloud.get_shs
        opacity = self.point_cloud.get_opacity
        rotation = self.point_cloud.get_rotation(ids)
        scaling = self.point_cloud.get_scaling
        pos_cubic_node = self.point_cloud.get_pos_cubic_node
        rot_cubic_node = self.point_cloud.get_rot_cubic_node
        render_dict = {
            "position": position,
            "opacity": opacity,
            "scaling": scaling,
            "rotation": rotation,
            "shs": shs,
        }
        if self.gabor:
            wave_coefficients, wave_coefficient_indices = self.point_cloud.get_topk_waves()
            render_dict["wave_coefficients"] = wave_coefficients
            render_dict["wave_coefficient_indices"] = wave_coefficient_indices
        
        for attr in self.render_attributes_list:
            if attr not in render_dict:
                render_dict[attr] = getattr(self.point_cloud, f"get_{attr}")
        return render_dict
    
