from dataclasses import dataclass
from typing import Optional

import torch

from .atlas_gs_optimizer import AtlasGaussianSplattingOptimizer
from .optimizer import OPTIMIZER_REGISTRY
from .point_insertion import MotionPointInsertion, MotionPointInsertionConfig


@OPTIMIZER_REGISTRY.register()
class AtlasGaussianSplattingOptimizerWithInsertion(AtlasGaussianSplattingOptimizer):
    """
    Extends the default Atlas optimizer with spline control-point insertion.
    The first `point_insertion_warmup` steps run exactly as the original code.
    Afterwards, every `point_insertion_interval` steps we examine the gradients
    of `pos_cubic_node` to decide whether to add a new knot between two times.
    """

    @dataclass
    class Config(AtlasGaussianSplattingOptimizer.Config):
        enable_point_insertion: bool = True
        point_insertion_warmup: int = 2000
        point_insertion_interval: int = 500
        point_insertion_grad_threshold: float = 1e-3
        point_insertion_grad_quantile: float = 0.9
        point_insertion_max_new: int = 1
        point_insertion_min_frame_distance: float = 2.0
        point_insertion_max_knots: Optional[int] = None
        point_insertion_time_epsilon: float = 1e-4

    cfg: Config

    def setup(self, optimizer, model, cameras_extent: float) -> None:
        super().setup(optimizer, model, cameras_extent)
        self.point_insertion: Optional[MotionPointInsertion] = None
        if self.cfg.enable_point_insertion:
            insertion_cfg = MotionPointInsertionConfig(
                enable=True,
                warmup_steps=self.cfg.point_insertion_warmup,
                check_interval=self.cfg.point_insertion_interval,
                grad_threshold=self.cfg.point_insertion_grad_threshold,
                grad_quantile=self.cfg.point_insertion_grad_quantile,
                max_new_knots_per_check=self.cfg.point_insertion_max_new,
                min_frame_distance=self.cfg.point_insertion_min_frame_distance,
                max_total_knots=self.cfg.point_insertion_max_knots,
                time_epsilon=self.cfg.point_insertion_time_epsilon,
            )
            self.point_insertion = MotionPointInsertion(
                point_cloud=self.point_cloud,
                optimizer=self.optimizer,
                cfg=insertion_cfg,
            )

    @torch.no_grad()
    def update_structure(self, visibility, viewspace_grad, radii, white_bg: bool = False) -> None:
        super().update_structure(visibility, viewspace_grad, radii, white_bg)
        if self.point_insertion is not None:
            self.point_insertion.maybe_insert(self.step)
