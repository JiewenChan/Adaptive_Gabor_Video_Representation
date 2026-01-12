import copy
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
from model.atlas_model import *
from itertools import accumulate
import operator



@MODEL_REGISTRY.register()
class FragModel(BaseModel):
    """
    A class for the Fragmentation Model.
    """
    @dataclass
    class Config:
        point_cloud: dict = field(default_factory=dict)
        lambda_dssim: float = 0.2

    cfg: Config

    def setup(self, gs_atlas_cfg_list, base_point_seq_list, white_bg=True, num_frames=None):
        base_point_seq = torch.cat(base_point_seq_list, dim=1)
        self.gs_atlas_cfg_list = gs_atlas_cfg_list
        self.atlas_dict = nn.ModuleDict({})
        for gs_atlas_cfg in self.gs_atlas_cfg_list:
            name = gs_atlas_cfg.name
            if name == "gs_pchip_gabor_all":
                self.atlas_dict[name] = SingleAtlasPchipWithBaseModel(self.cfg, gs_atlas_cfg, base_point_seq, gabor=True, num_frames=num_frames)
            else:
                raise ValueError(f"Unknown atlas name: {name}")
        self.focal_y_ratio = 1.0
        self.white_bg = white_bg


    def get_atlas(self, name):
        """
        Query the Gaussian Splatting Atlas by name.

        Parameters
        ----------
        name : str
            The name of the Gaussian Splatting Atlas.
        """
        return self.atlas_dict[name]


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


        atls_render_dict_list = [self.atlas_dict[name].forward(ids) for name in self.atlas_dict]
        render_dict = {}
        for atls_render_dict in atls_render_dict_list:
            for key, value in atls_render_dict.items():
                if key not in render_dict:
                    render_dict[key] = value
                else:
                    render_dict[key] = torch.cat([render_dict[key], value], dim=0)

        return render_dict
    

    def forward_single_atlas(self, ids, name) -> dict:
        """
        Forward pass of the model for a single atlas.

        Parameters
        ----------
        name : str
            The name of the Gaussian Splatting Atlas.
        
        Returns
        -------
        dict
            The render results which will be the input of renderers.
        """
        render_dict = self.atlas_dict[name].forward(ids)
        return render_dict

    
    def get_point_num_sep(self):
        """
        Get the number of points in the point cloud.
        """
        point_num_list = [0]+[len(self.atlas_dict[name].point_cloud) for name in self.atlas_dict]
        return list(accumulate(point_num_list, operator.add))
    
    def prepare_optimizer_dict(self, loss, render_results):
        """
        Prepare the optimizer dictionary.
        """
        optimizer_dict = {}
        ptnum = self.get_point_num_sep()
        for idx, gs_atlas_cfg in enumerate(self.gs_atlas_cfg_list):
            name = gs_atlas_cfg.name
            optimizer_dict[name] = {
                "viewspace_points": [x[ptnum[idx]:ptnum[idx+1]] for x in render_results["viewspace_points"]],
                "viewspace_points_grad": [x.grad[ptnum[idx]:ptnum[idx+1]] for x in render_results["viewspace_points"]],
                "visibility": render_results["visibility"][ptnum[idx]:ptnum[idx+1]],
                "radii": render_results["radii"][ptnum[idx]:ptnum[idx+1]],
                "white_bg": self.white_bg,
            }
            for x in optimizer_dict[name]["viewspace_points"]:
                x.retain_grad()
            
            # Store wave_coefficients gradients.
            if "wave_coefficients" in render_results:
                wave_coefficients = render_results["wave_coefficients"]
                if wave_coefficients is not None and wave_coefficients.grad is not None:
                    optimizer_dict[name]["wave_coefficients_grad"] = wave_coefficients.grad[ptnum[idx]:ptnum[idx+1]]
                else:
                    optimizer_dict[name]["wave_coefficients_grad"] = None
            
            # Store pos_cubic_node gradients.
            if "pos_cubic_node" in render_results:
                pos_cubic_node = render_results["pos_cubic_node"]
                if pos_cubic_node is not None and pos_cubic_node.grad is not None:
                    optimizer_dict[name]["pos_cubic_node_grad"] = pos_cubic_node.grad[ptnum[idx]:ptnum[idx+1]]
                else:
                    optimizer_dict[name]["pos_cubic_node_grad"] = None
        return optimizer_dict
    
    def get_state_dict(self):
        additional_info = {k: v.get_state_dict() for k, v in self.atlas_dict.items()}
        return {**super().state_dict(), **additional_info}
    
    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False):

        for name, model in self.atlas_dict.items():
            model.load_state_dict(state_dict[name], strict)
            model.to('cuda')
