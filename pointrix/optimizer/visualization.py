import torch
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio
from torch import Tensor
from typing import Tuple, List, Optional, Dict
import dptr.gabor as gabor


class VisualizationManager:
    """
    Visualization manager handling all visualization-related functionality.
    """
    
    def __init__(self, point_cloud, cfg, device):
        self.point_cloud = point_cloud
        self.cfg = cfg
        self.device = device
        
    @torch.no_grad()
    def visualize_densification(
        self,
        mask: torch.Tensor,
        kind: str,
        grads: torch.Tensor = None,
        step: int = 0,
        color_masks: Optional[Dict[str, torch.Tensor]] = None,
    ) -> None:
        """
        Optional method to visualize densification regions.
        
        Parameters
        ----------
        mask : torch.Tensor [N]
            Boolean mask of affected points.
        kind : str
            Operation type, e.g. {"clone", "split", "prune"}
        grads : torch.Tensor [N,1] or [N]
            Optional gradient signal.
        step : int
            Current step.
        """
        try:
            if mask is None or mask.numel() == 0:
                return

            # Collect positions and importance.
            with torch.no_grad():
                pos_all = self.point_cloud.position
                if pos_all is None:
                    return
                if pos_all.is_cuda:
                    pos_all = pos_all.detach().cpu()
                m = mask.detach().cpu()

                color_masks_cpu: Optional[Dict[str, torch.Tensor]] = None
                if color_masks:
                    color_masks_cpu = {}
                    for name, c_mask in color_masks.items():
                        if c_mask is None:
                            continue
                        cm = c_mask.detach()
                        if cm.is_cuda:
                            cm = cm.cpu()
                        color_masks_cpu[name] = cm.to(torch.bool)
                    if color_masks_cpu:
                        target_len = m.shape[0]
                        for key in list(color_masks_cpu.keys()):
                            cm = color_masks_cpu[key]
                            if cm.shape[0] > target_len:
                                color_masks_cpu[key] = cm[:target_len]
                            elif cm.shape[0] < target_len:
                                pad = torch.zeros(target_len - cm.shape[0], dtype=torch.bool)
                                color_masks_cpu[key] = torch.cat([cm, pad], dim=0)

                importance = None
                
                if grads is not None:
                    g = grads
                    if g.dim() > 1 and g.shape[-1] > 1:
                        g = torch.norm(g, dim=-1, keepdim=False)
                    importance = g.detach().cpu().float().squeeze()
                else:
                    # Fall back to scaling magnitude.
                    scaling = self.point_cloud.get_scaling
                    imp = torch.max(scaling, dim=1).values
                    importance = imp.detach().cpu().float()

                out_dir = getattr(self.cfg, "vis_out_dir", "output/vis/densify")
                os.makedirs(out_dir, exist_ok=True)
                stem = os.path.join(out_dir, f"steps_{step:06d}_{kind}")

                # Export colored PLY.
                if True or getattr(self.cfg, "vis_export_ply", True):
                    try:
                        self._export_ply_with_colors(
                            pos_all=pos_all,
                            mask=m,
                            importance=importance,
                            color_masks=color_masks_cpu,
                            out_path=stem + "_all.ply",
                            masked_only=False,
                        )
                    except Exception as e:
                        print(f"[densify-vis] ply export skipped: {e}")

                # Optional short dynamic point cloud video.
                if True or getattr(self.cfg, "vis_dynamic", False):
                    try:
                        # Convert gradients to numpy.
                        grads_np = None
                        if grads is not None:
                            if isinstance(grads, torch.Tensor):
                                grads_np = grads.detach().cpu().numpy()
                            else:
                                grads_np = grads
                        
                        self._vis_dynamic_point_cloud(
                            mask=m,
                            out_path=stem + "_dynamic.mp4",
                            num_frames=int(getattr(self.cfg, "vis_dynamic_frames", 100)),
                            grads=grads_np
                        )
                    except Exception as e:
                        print(f"[densify-vis] dynamic vis skipped: {e}")

                # Local zoom (optional).
                local_knn = int(getattr(self.cfg, "vis_local_knn", 0))
                if local_knn > 0 and m.any():
                    self._vis_densify_3d(
                        pos_all=pos_all,
                        mask=m,
                        importance=importance,
                        color_masks=color_masks_cpu,
                        out_path=stem + "_local.png",
                        local_knn=local_knn,
                        max_points=getattr(self.cfg, "vis_max_points", 100000),
                    )
        except Exception as e:
            # Fail silently without interrupting training.
            print(f"[densify-vis] visualization skipped due to error: {e}")

    @torch.no_grad()
    def visualize_viewspace_grad(self, viewspace_grad: torch.Tensor, step: int = 0) -> None:
        """
        Visualize viewspace_grad, export PNG/PLY, and keep original gradients.
        """
        if not getattr(self.cfg, "vis_viewspace_grad", False):
            return

        vis_interval = max(1, getattr(self.cfg, "vis_viewspace_grad_every_n", getattr(self.cfg, "vis_every_n", 200)))
        if step % vis_interval != 0:
            return

        try:
            if viewspace_grad is None or viewspace_grad.numel() == 0:
                return

            grad_tensor = viewspace_grad.detach()
            if grad_tensor.is_cuda:
                grad_tensor = grad_tensor.cpu()
            grad_tensor = grad_tensor.float().view(grad_tensor.shape[0], -1)

            pos_all = getattr(self.point_cloud, "position", None)
            if pos_all is None:
                return
            if isinstance(pos_all, torch.Tensor):
                pos_tensor = pos_all.detach()
                if pos_tensor.is_cuda:
                    pos_tensor = pos_tensor.cpu()
            else:
                pos_tensor = torch.as_tensor(pos_all)
            pos_tensor = pos_tensor.float()

            num_points = min(pos_tensor.shape[0], grad_tensor.shape[0])
            if num_points == 0:
                return

            pos_tensor = pos_tensor[:num_points]
            grad_tensor = grad_tensor[:num_points]
            grad_magnitude = torch.norm(grad_tensor, dim=-1)

            out_dir = getattr(self.cfg, "vis_out_dir", "output/vis/densify")
            os.makedirs(out_dir, exist_ok=True)
            stem = os.path.join(out_dir, f"steps_{step:06d}_viewspace_grad")

            if getattr(self.cfg, "vis_viewspace_grad_export_png", True):
                self._export_viewspace_grad_png(
                    pos_all=pos_tensor,
                    grad_values=grad_magnitude,
                    out_path=stem + ".png"
                )

            if getattr(self.cfg, "vis_viewspace_grad_export_ply", True):
                self._export_viewspace_grad_ply(
                    pos_all=pos_tensor,
                    grad_components=grad_tensor,
                    grad_magnitude=grad_magnitude,
                    out_path=stem + ".ply"
                )
        except Exception as e:
            print(f"[viewspace-grad-vis] visualization skipped due to error: {e}")

    @torch.no_grad()
    def visualize_viewspace_grad_heatmap(self, viewspace_grad_xy: torch.Tensor, step: int = 0) -> None:
        """
        Generate a heatmap (PNG) based only on viewspace XY gradients.
        """
        if viewspace_grad_xy is None or viewspace_grad_xy.numel() == 0:
            return

        try:
            grad_tensor = viewspace_grad_xy.detach()
            if grad_tensor.is_cuda:
                grad_tensor = grad_tensor.cpu()
            grad_tensor = grad_tensor.float()

            pos_all = getattr(self.point_cloud, "position", None)
            if pos_all is None:
                return
            if isinstance(pos_all, torch.Tensor):
                pos_tensor = pos_all.detach()
                if pos_tensor.is_cuda:
                    pos_tensor = pos_tensor.cpu()
            else:
                pos_tensor = torch.as_tensor(pos_all)

            num_points = min(pos_tensor.shape[0], grad_tensor.shape[0])
            if num_points == 0:
                return
            pos_tensor = pos_tensor[:num_points]
            grad_tensor = grad_tensor[:num_points]

            grad_mag = torch.linalg.norm(grad_tensor, dim=-1)
            grad_mag_np = grad_mag.numpy()
            grad_min = float(grad_mag_np.min())
            grad_max = float(grad_mag_np.max())
            if grad_max - grad_min < 1e-12:
                grad_max = grad_min + 1e-6

            out_dir = getattr(self.cfg, "vis_out_dir", "output/vis/densify")
            os.makedirs(out_dir, exist_ok=True)
            stem = os.path.join(out_dir, f"steps_{step:06d}_viewspace_grad_heatmap.png")

            plt.figure(figsize=(6, 6))
            cmap = plt.get_cmap("turbo")
            scatter = plt.scatter(
                pos_tensor[:, 0].numpy(),
                pos_tensor[:, 1].numpy(),
                c=grad_mag_np,
                cmap=cmap,
                s=2.0,
                alpha=0.8,
                vmin=grad_min,
                vmax=grad_max,
            )
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.title("Viewspace Grad XY Heatmap")
            plt.colorbar(scatter, shrink=0.75, pad=0.02, label="Gradient Magnitude")
            plt.tight_layout()
            plt.savefig(stem, dpi=220)
            plt.close()

        except Exception as e:
            print(f"[viewspace-grad-heatmap] visualization skipped due to error: {e}")

    @torch.no_grad()
    def _vis_densify_3d(self,
                         pos_all: torch.Tensor,
                         mask: torch.Tensor,
                         importance: torch.Tensor = None,
                         out_path: str = "densify_3d.png",
                         local_knn: int = 0,
                         max_points: int = 100000,
                         color_masks: Optional[Dict[str, torch.Tensor]] = None) -> None:
        """
        Minimal 3D scatter visualization using depth sorting and size/color encoding.
        """
        p = pos_all.detach().cpu()
        m = mask.detach().cpu().bool()
        if importance is None:
            imp = torch.ones(len(p), dtype=torch.float32)
        else:
            imp = importance.detach().cpu().float()

        color_masks_cpu = None
        if color_masks:
            color_masks_cpu = {}
            for name, c_mask in color_masks.items():
                if c_mask is None:
                    continue
                cm = c_mask.detach().cpu().bool()
                color_masks_cpu[name] = cm

        # Optional: subsample to avoid heavy plots.
        N = p.shape[0]
        if N > max_points:
            idx = torch.randperm(N)[:max_points]
            p = p[idx]
            m = m[idx]
            imp = imp[idx]
            if color_masks_cpu:
                for key in list(color_masks_cpu.keys()):
                    color_masks_cpu[key] = color_masks_cpu[key][idx]

        # Optional: use kNN around mask points for local zoom.
        if local_knn and local_knn > 0 and m.any():
            try:
                from sklearn.neighbors import NearestNeighbors
                k = min(local_knn, p.shape[0])
                nbrs = NearestNeighbors(n_neighbors=k)
                nbrs.fit(p.numpy())
                sel = torch.nonzero(m).squeeze(1).numpy()
                chosen = set()
                for i in sel:
                    nn_idx = nbrs.kneighbors(p[i].numpy()[None], return_distance=False)[0]
                    for j in nn_idx.tolist():
                        chosen.add(j)
                chosen = np.array(sorted(list(chosen)), dtype=np.int64)
                p = p[chosen]
                m = m[chosen]
                imp = imp[chosen]
                if color_masks_cpu:
                    for key in list(color_masks_cpu.keys()):
                        color_masks_cpu[key] = color_masks_cpu[key][chosen]
            except Exception as e:
                print(f"[densify-vis] local zoom skipped: {e}")

        # Depth-sort along z (fallback) to improve layering.
        depth = p[:, 2].numpy()
        order = np.argsort(depth)
        p = p[order]
        m = m[order]
        imp = imp[order]
        if color_masks_cpu:
            for key in list(color_masks_cpu.keys()):
                color_masks_cpu[key] = color_masks_cpu[key][order]

        # Normalize importance -> size and base color.
        imp_np = imp.numpy()
        imp_np = (imp_np - imp_np.min()) / (imp_np.max() - imp_np.min() + 1e-8)
        sizes = 1.0 + 4.0 * imp_np
        cmap = plt.get_cmap("turbo")
        colors = cmap(imp_np)
        hit = m.numpy()

        alpha = np.where(hit, 0.95, 0.6)

        if color_masks_cpu:
            base_colors = {
                "pos": np.array([1.0, 0.0, 0.0, 1.0]),       # red
                "wave": np.array([0.09, 0.45, 1.0, 1.0]),    # blue-ish
                "both": np.array([1.0, 0.0, 1.0, 1.0]),      # magenta
            }
            for key, cm in color_masks_cpu.items():
                if key not in base_colors:
                    continue
                mask_np = cm.numpy()
                colors[mask_np] = base_colors[key]
                alpha = np.where(mask_np, base_colors[key][-1], alpha)
        else:
            colors[hit] = np.array([1.0, 0.0, 0.0, 1.0])

        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(p[:, 0], p[:, 1], p[:, 2], s=sizes, c=colors, alpha=alpha, linewidths=0)
        # Outline hit points (keep a light stroke).
        if hit.any():
            ax.scatter(p[hit, 0], p[hit, 1], p[hit, 2], s=sizes[hit] * 1.2,
                       facecolors='none', edgecolors='k', linewidths=0.3, alpha=0.9)

        # Frame.
        xyz_min, xyz_max = p.min(0).values.numpy(), p.max(0).values.numpy()
        ctr = (xyz_min + xyz_max) / 2.0
        rad = float(np.max(xyz_max - xyz_min) / 2.0 + 1e-6)
        ax.set_xlim([ctr[0] - rad, ctr[0] + rad])
        ax.set_ylim([ctr[1] - rad, ctr[1] + rad])
        ax.set_zlim([ctr[2] - rad, ctr[2] + rad])
        # Fix camera view to avoid upside-down orientation.
        ax.view_init(elev=20, azim=45)  # Adjust elevation/azimuth for a better view.
        ax.set_axis_off()
        plt.tight_layout()
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, dpi=220)
        plt.close(fig)

    @torch.no_grad()
    def _export_ply_with_colors(self,
                                pos_all: torch.Tensor,
                                mask: torch.Tensor,
                                importance: torch.Tensor = None,
                                out_path: str = "points.ply",
                                masked_only: bool = False,
                                color_masks: Optional[Dict[str, torch.Tensor]] = None) -> None:
        """
        Export a PLY point cloud with per-vertex RGB.
        - mask=True points are colored red (255,0,0)
        - mask=False points use SHS-derived base colors; fall back to importance colormap if unavailable
        """
        p = pos_all.detach().cpu().numpy()
        m = mask.detach().cpu().numpy().astype(bool)
        if importance is None:
            imp = np.ones((p.shape[0],), dtype=np.float32)
        else:
            imp = importance.detach().cpu().numpy().astype(np.float32)

        color_masks_np = None
        if color_masks:
            color_masks_np = {}
            for name, c_mask in color_masks.items():
                if c_mask is None:
                    continue
                color_masks_np[name] = c_mask.detach().cpu().numpy().astype(bool)

        # Normalize importance to [0,1].
        imp = (imp - imp.min()) / (imp.max() - imp.min() + 1e-8)

        scaling = getattr(self.point_cloud, "get_scaling", None)
        if isinstance(scaling, torch.Tensor):
            scaling_np = scaling.detach().cpu().numpy().astype(np.float32)
        else:
            scaling_np = None

        # Prefer Gabor SH base color (view direction=+Z); fall back to DC/cmap if it fails.
        shs = self.point_cloud.get_shs
        position = self.point_cloud.position
        direction = torch.zeros_like(position)
        direction[:, 2] = 1.0
        colors = None
        try:
            rgb_sh = gabor.compute_sh(shs, 3, direction)
            colors = torch.clamp(rgb_sh, 0.0, 1.0).detach().cpu().numpy().astype(np.float32)
        except Exception:
            colors = None
        if colors is None:
            try:
                shs_t = self.point_cloud.get_shs
                if isinstance(shs_t, torch.Tensor):
                    shs_cpu = shs_t.detach().cpu()
                    if shs_cpu.dim() == 2 and shs_cpu.shape[1] >= 3:
                        dc = shs_cpu[:, :3]
                    elif shs_cpu.dim() == 3 and shs_cpu.shape[1] == 3:
                        dc = shs_cpu[:, :, 0]
                    else:
                        dc = None
                    if dc is not None:
                        colors = torch.sigmoid(dc).numpy().astype(np.float32)
            except Exception:
                colors = None
        if colors is None:
            cmap = plt.get_cmap("turbo")
            colors = cmap(imp)[:, :3].astype(np.float32)

        base_colors = {
            "pos": np.array([1.0, 0.0, 0.0], dtype=np.float32),
            "wave": np.array([0.09, 0.45, 1.0], dtype=np.float32),
            "both": np.array([1.0, 0.0, 1.0], dtype=np.float32),
        }
        if color_masks_np:
            both_mask = color_masks_np.get("both")
            if both_mask is not None:
                colors[both_mask] = base_colors["both"]
            pos_mask = color_masks_np.get("pos")
            if pos_mask is not None:
                colors[pos_mask] = base_colors["pos"]
            wave_mask = color_masks_np.get("wave")
            if wave_mask is not None:
                colors[wave_mask] = base_colors["wave"]
        else:
            colors[m] = base_colors["pos"]

        if masked_only:
            p = p[m]
            colors = colors[m]
            if scaling_np is not None:
                scaling_np = scaling_np[m]
        elif scaling_np is not None:
            scaling_np = scaling_np

        # Convert to uint8.
        rgb_u8 = (colors.clip(0, 1) * 255.0).astype(np.uint8)

        # ASCII PLY export.
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {p.shape[0]}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            if scaling_np is not None and scaling_np.shape[1] >= 3:
                f.write("property float scale_x\n")
                f.write("property float scale_y\n")
                f.write("property float scale_z\n")
            f.write("end_header\n")
            for i in range(p.shape[0]):
                x, y, z = p[i]
                r, g, b = rgb_u8[i]
                if scaling_np is not None and scaling_np.shape[1] >= 3:
                    sx, sy, sz = scaling_np[i]
                    f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)} {sx:.6f} {sy:.6f} {sz:.6f}\n")
                else:
                    f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")

    @torch.no_grad()
    def _compute_base_colors_gabor(self) -> np.ndarray:
        """
        Compute base RGB colors from SHS using Gabor SH in the +Z direction.
        Returns a numpy float32 array of shape [N,3] in [0,1], falling back to DC/gray.
        """
        colors = None
        try:
            shs = self.point_cloud.get_shs
            position = self.point_cloud.position
            direction = torch.zeros_like(position)
            direction[:, 2] = 1.0
            rgb_sh = gabor.compute_sh(shs, 3, direction)
            colors = torch.clamp(rgb_sh, 0.0, 1.0).detach().cpu().numpy().astype(np.float32)
        except Exception:
            colors = None
        if colors is None:
            try:
                shs_t = self.point_cloud.get_shs
                if isinstance(shs_t, torch.Tensor):
                    shs_cpu = shs_t.detach().cpu()
                    if shs_cpu.dim() == 2 and shs_cpu.shape[1] >= 3:
                        dc = shs_cpu[:, :3]
                    elif shs_cpu.dim() == 3 and shs_cpu.shape[1] == 3:
                        dc = shs_cpu[:, :, 0]
                    else:
                        dc = None
                    if dc is not None:
                        colors = torch.sigmoid(dc).numpy().astype(np.float32)
            except Exception:
                colors = None
        if colors is None:
            N = len(self.point_cloud.position)
            colors = np.tile(np.array([[0.7, 0.7, 0.7]], dtype=np.float32), (N, 1))
        return colors

    @torch.no_grad()
    def _vis_dynamic_point_cloud(self,
                                 mask: torch.Tensor,
                                 out_path: str,
                                 num_frames: int = 20,
                                 grads: np.ndarray = None) -> None:
        """
        Render a short video using 2D camera projection and gradient heatmaps.
        Shows gradient magnitude as a heatmap with a threshold indicator.
        """
        pc = self.point_cloud
        
        # Time sampling.
        try:
            tk = pc.pos_tk
            if isinstance(tk, torch.Tensor):
                tk = tk.detach().cpu().numpy()
            else:
                tk = np.array(tk)
            t_start, t_end = float(tk[0]), float(tk[-1])
        except Exception:
            t_start, t_end = 0.0, float(max(1, num_frames - 1))
        ts = np.linspace(t_start, t_end, num_frames, dtype=np.float32)

        # Set camera parameters (similar to trainer_fragGS.py).
        width, height = 512, 512
        from pointrix.camera.cam_utils import construct_canonical_camera
        camera = construct_canonical_camera(width=width, height=height)
        
        # Get gradient info.
        if grads is not None:
            grad_magnitude = grads.copy()
            if grad_magnitude.ndim > 1:
                grad_magnitude = np.linalg.norm(grad_magnitude, axis=-1)
            grad_magnitude = grad_magnitude.flatten()
        else:
            grad_magnitude = np.ones(len(pc.position))
        
        # Normalize gradients for visualization.
        grad_min, grad_max = grad_magnitude.min(), grad_magnitude.max()
        if grad_max > grad_min:
            grad_normalized = (grad_magnitude - grad_min) / (grad_max - grad_min)
        else:
            grad_normalized = np.zeros_like(grad_magnitude)
        
        m = mask.detach().cpu().numpy().astype(bool)
        max_grad_threshold = getattr(self.cfg, "densify_grad_threshold", 0.0002)
        
        frames = []
        for frame_idx, t in enumerate(ts):
            with torch.no_grad():
                pos_t = pc.get_position(float(t))
                if isinstance(pos_t, torch.Tensor):
                    pos_t = pos_t.detach().cpu().numpy()
                else:
                    pos_t = np.array(pos_t)
            
            # Project 3D points to 2D using the camera.
            pos_tensor = torch.from_numpy(pos_t).float().cuda()
            uv, depth = self._project_points_to_2d(pos_tensor, camera)
            uv = uv.detach().cpu().numpy()
            depth = depth.detach().cpu().numpy()
            
            # Filter visible points.
            visible = (depth > 0).flatten()
            uv_vis = uv[visible]
            grad_vis = grad_normalized[visible]
            mask_vis = m[visible]
            
            # Create a figure with subplots.
            fig = plt.figure(figsize=(12, 6))
            
            # Left subplot: 2D projection with gradient heatmap.
            ax1 = fig.add_subplot(121)
            
            # Create gradient heatmap.
            if len(uv_vis) > 0:
                # Apply sigmoid to gradients for color mapping.
                grad_sigmoid = 1 / (1 + np.exp(-grad_vis))  # Sigmoid.
                scatter = ax1.scatter(uv_vis[:, 0], uv_vis[:, 1], 
                                    c=grad_sigmoid, cmap='jet', s=8, alpha=0.8,
                                    vmin=0, vmax=1)
                
                # Highlight points above threshold.
                above_threshold = grad_vis > (max_grad_threshold - grad_min) / (grad_max - grad_min + 1e-8)
                if above_threshold.any():
                    ax1.scatter(uv_vis[above_threshold, 0], uv_vis[above_threshold, 1], 
                              c='cyan', s=12, alpha=1.0, edgecolors='white', linewidth=0.5)
                
                # Highlight mask points.
                if mask_vis.any():
                    ax1.scatter(uv_vis[mask_vis, 0], uv_vis[mask_vis, 1], 
                              c='red', s=15, alpha=1.0, edgecolors='black', linewidth=1.0)
            
            ax1.set_xlim(0, width)
            ax1.set_ylim(0, height)
            ax1.set_aspect('equal')
            ax1.invert_yaxis()  # Image coordinates.
            ax1.set_title(f'Gradient Heatmap (t={t:.2f})')
            ax1.set_xlabel('X (pixels)')
            ax1.set_ylabel('Y (pixels)')
            
            # Add colorbar.
            cbar = plt.colorbar(scatter, ax=ax1, shrink=0.8)
            cbar.set_label('Gradient Magnitude')
            
            # Right subplot: 3D view for reference.
            ax2 = fig.add_subplot(122, projection='3d')
            
            # Apply sigmoid gradients to 3D colors as well (swap y and z axes).
            if len(pos_t) > 0:
                # Apply sigmoid to normalized gradients.
                grad_sigmoid_3d = 1 / (1 + np.exp(-grad_normalized))  # Sigmoid.
                ax2.scatter(pos_t[:, 0], pos_t[:, 2], pos_t[:, 1], 
                           c=grad_sigmoid_3d, cmap='jet', s=4, alpha=0.6, vmin=0, vmax=1)
                
                # Highlight points above threshold in 3D.
                above_threshold_3d = grad_normalized > (max_grad_threshold - grad_min) / (grad_max - grad_min + 1e-8)
                if above_threshold_3d.any():
                    ax2.scatter(pos_t[above_threshold_3d, 0], pos_t[above_threshold_3d, 2], pos_t[above_threshold_3d, 1], 
                              c='cyan', s=8, alpha=1.0, edgecolors='white', linewidth=0.3)
                
                # Highlight mask points in 3D.
                if m.any():
                    ax2.scatter(pos_t[m, 0], pos_t[m, 2], pos_t[m, 1], 
                              c='red', s=10, alpha=1.0, edgecolors='black', linewidth=0.5)
            
            # Set 3D view limits (swap y and z axes).
            if len(pos_t) > 0:
                xyz_min, xyz_max = pos_t.min(0), pos_t.max(0)
                ctr = (xyz_min + xyz_max) / 2.0
                rad = float(np.max(xyz_max - xyz_min) / 2.0 + 1e-6)
                ax2.set_xlim([ctr[0] - rad, ctr[0] + rad])
                ax2.set_ylim([-1, 3])  # Fix z-axis range from -1 to 3.
                ax2.set_zlim([ctr[1] - rad, ctr[1] + rad])  # z-axis now shows y values.
            
            # Animate camera angle to better visualize z.
            # Rotate around z-axis to show both directions.
            rotation_angle = (frame_idx / num_frames) * 360  # Full rotation over time.
            # Fix camera view with a better elevation to avoid upside-down.
            ax2.view_init(elev=30, azim=rotation_angle)
            ax2.set_title(f'3D Reference View (Y↔Z swapped, t={t:.2f})')
            ax2.set_xlabel('X')
            ax2.set_ylabel('Z')
            ax2.set_zlabel('Y')
            
            # Add legend.
            legend_elements = [
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='cyan', 
                          markersize=8, label=f'Above threshold ({max_grad_threshold:.4f})'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                          markersize=8, label='Selected for densification'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', 
                          markersize=6, label='Gradient heatmap')
            ]
            ax1.legend(handles=legend_elements, loc='upper right', fontsize=8)
            
            plt.tight_layout()
            fig.canvas.draw()
            w, h = fig.canvas.get_width_height()
            
            # Handle different matplotlib versions.
            try:
                # New matplotlib versions.
                buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
                img = buf.reshape(h, w, 4)[:, :, :3]  # Remove alpha channel.
            except AttributeError:
                # Old matplotlib versions.
                buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                img = buf.reshape(h, w, 3)
            
            frames.append(img.copy())
            plt.close(fig)

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        imageio.mimsave(out_path, frames, fps=6)  # Slower render speed.

    @torch.no_grad()
    def _project_points_to_2d(self, xyz: torch.Tensor, camera) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Project 3D points to the 2D image plane using camera parameters.
        
        Parameters
        ----------
        xyz : torch.Tensor [N, 3]
            3D point positions
        camera : Camera object
            Camera with intrinsics and extrinsics
            
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            (uv, depth) - 2D coordinates and depth values
        """
        # Get camera parameters.
        if isinstance(camera.R, torch.Tensor):
            R = camera.R.float().cuda()  # [3, 3]
        else:
            R = torch.from_numpy(camera.R).float().cuda()  # [3, 3]
            
        if isinstance(camera.T, torch.Tensor):
            T = camera.T.float().cuda()  # [3]
        else:
            T = torch.from_numpy(camera.T).float().cuda()  # [3]
        width = camera.width
        height = camera.height
        fovX = camera.fovX
        fovY = camera.fovY
        
        # Transform points to camera coordinates.
        xyz_cam = torch.matmul(xyz, R.T) + T.unsqueeze(0)  # [N, 3]
        
        # Get depth.
        depth = xyz_cam[:, 2]  # [N]
        
        # Project to image plane.
        # For orthographic projection (similar to dptr_ortho_enhanced_gabor.py).
        uv = torch.zeros_like(xyz_cam[:, :2])  # [N, 2]
        uv[:, 0] = (xyz_cam[:, 0] + 1.0) * width / 2.0  # X coordinate.
        uv[:, 1] = (xyz_cam[:, 1] + 1.0) * height / 2.0  # Y coordinate.
        
        # Clamp to image bounds.
        uv[:, 0] = torch.clamp(uv[:, 0], 0, width - 1)
        uv[:, 1] = torch.clamp(uv[:, 1], 0, height - 1)
        
        return uv, depth.unsqueeze(-1)

    @torch.no_grad()
    def visualize_wave_coefficients_grad(self, wave_grad: torch.Tensor, step: int = 0) -> None:
        """
        Visualize wave_coefficients gradient heatmap.
        
        Parameters
        ----------
        wave_grad : torch.Tensor [N, F]
            Gradients of wave_coefficients, N points and F frequencies
        step : int
            Current step
        """
        try:
            if wave_grad is None or wave_grad.numel() == 0:
                return
                
            # Compute gradient magnitude.
            if wave_grad.dim() > 1:
                grad_magnitude = torch.norm(wave_grad, dim=-1)  # [N]
            else:
                grad_magnitude = wave_grad.abs()  # [N]
            
            # Get point cloud positions.
            pos_all = self.point_cloud.position
            if pos_all is None:
                return
            if pos_all.is_cuda:
                pos_all = pos_all.detach().cpu()
            
            grad_magnitude = grad_magnitude.detach().cpu().float()
            
            # Create output directory.
            out_dir = getattr(self.cfg, "vis_out_dir", "output/vis/densify")
            os.makedirs(out_dir, exist_ok=True)
            stem = os.path.join(out_dir, f"steps_{step:06d}_wave_grad")
            
            # Generate 3D gradient heatmap.
            self._vis_wave_heatmap_3d(
                pos_all=pos_all,
                grad_magnitude=grad_magnitude,
                out_path=stem + "_3d.png",
                title=f'Wave Coefficients Gradient Heatmap (Step {step})',
                cbar_label='Gradient Magnitude (Sigmoid)'
            )
            
            if getattr(self.cfg, "vis_wave_grad_export_ply", False):
                self._export_wave_heatmap_ply(
                    pos_all=pos_all,
                    grad_magnitude=grad_magnitude,
                    out_path=stem + "_points.ply",
                    value_label="wave_grad"
                )
            
            # Generate dynamic video.
            if getattr(self.cfg, "vis_dynamic", False):
                self._vis_wave_heatmap_dynamic(
                    grad_magnitude=grad_magnitude,
                    out_path=stem + "_dynamic.mp4",
                    num_frames=int(getattr(self.cfg, "vis_dynamic_frames", 20)),
                    label_prefix="Wave Grad"
                )
                
        except Exception as e:
            print(f"[wave-grad-vis] visualization skipped due to error: {e}")

    @torch.no_grad()
    def visualize_wave_coefficients(self, step: int = 0) -> None:
        """
        Visualize wave_coefficients values as a heatmap.
        """
        try:
            wave_attr = getattr(self.point_cloud, "get_wave_coefficients", None)
            if wave_attr is None:
                wave_attr = getattr(self.point_cloud, "wave_coefficients", None)
            if wave_attr is None:
                return
            
            wave_values = wave_attr() if callable(wave_attr) else wave_attr
            if wave_values is None:
                return
            
            if isinstance(wave_values, torch.Tensor):
                coeffs = wave_values.detach()
                if coeffs.is_cuda:
                    coeffs = coeffs.cpu()
                coeffs = coeffs.float()
            else:
                coeffs = torch.as_tensor(wave_values, dtype=torch.float32)
            
            if coeffs.numel() == 0:
                return
            
            # coeffs = torch.sigmoid(coeffs)
            if coeffs.dim() > 1:
                coeff_magnitude = torch.norm(coeffs, dim=-1)
            else:
                coeff_magnitude = coeffs.abs()
            
            pos_all = getattr(self.point_cloud, "position", None)
            if pos_all is None:
                return
            if isinstance(pos_all, torch.Tensor):
                pos_tensor = pos_all.detach()
                if pos_tensor.is_cuda:
                    pos_tensor = pos_tensor.cpu()
            else:
                pos_tensor = torch.as_tensor(pos_all)
            
            out_dir = getattr(self.cfg, "vis_out_dir", "output/vis/densify")
            os.makedirs(out_dir, exist_ok=True)
            stem = os.path.join(out_dir, f"steps_{step:06d}_wave_coeff")
            
            # self._vis_wave_heatmap_3d(
            #     pos_all=pos_tensor,
            #     grad_magnitude=coeff_magnitude,
            #     out_path=stem + "_3d.png",
            #     title=f'Wave Coefficients Heatmap (Step {step})',
            #     cbar_label='Wave Coefficient Magnitude'
            # )
            
            if getattr(self.cfg, "vis_wave_coeff_export_ply", False):
                self._export_wave_heatmap_ply(
                    pos_all=pos_tensor,
                    grad_magnitude=coeff_magnitude,
                    out_path=stem + "_points.ply",
                    value_label="wave_coeff"
                )
            
            if getattr(self.cfg, "vis_dynamic", False):
                self._vis_wave_heatmap_dynamic(
                    grad_magnitude=coeff_magnitude,
                    out_path=stem + "_dynamic.mp4",
                    num_frames=int(getattr(self.cfg, "vis_dynamic_frames", 20)),
                    label_prefix="Wave Coeff"
                )
        except Exception as e:
            print(f"[wave-value-vis] visualization skipped due to error: {e}")

    @torch.no_grad()
    def visualize_pos_cubic_node_grad(self, pos_cubic_grad: torch.Tensor, step: int = 0) -> None:
        """
        Export pos_cubic_node gradients as a heatmap (currently PLY).
        """
        if not getattr(self.cfg, "vis_pos_cubic_grad", False):
            return
        
        vis_interval = max(1, getattr(self.cfg, "vis_pos_cubic_grad_every_n", getattr(self.cfg, "vis_every_n", 200)))
        if step % vis_interval != 0:
            return
        
        try:
            if pos_cubic_grad is None or pos_cubic_grad.numel() == 0:
                return
            
            grad_tensor = pos_cubic_grad.detach()
            if grad_tensor.is_cuda:
                grad_tensor = grad_tensor.cpu()
            grad_tensor = grad_tensor.float().view(grad_tensor.shape[0], -1)
            grad_magnitude = torch.norm(grad_tensor, dim=-1)
            
            pos_all = getattr(self.point_cloud, "position", None)
            if pos_all is None:
                return
            if isinstance(pos_all, torch.Tensor):
                pos_tensor = pos_all.detach()
                if pos_tensor.is_cuda:
                    pos_tensor = pos_tensor.cpu()
            else:
                pos_tensor = torch.as_tensor(pos_all)
            
            out_dir = getattr(self.cfg, "vis_out_dir", "output/vis/densify")
            os.makedirs(out_dir, exist_ok=True)
            stem = os.path.join(out_dir, f"steps_{step:06d}_pos_cubic_grad")
            
            if getattr(self.cfg, "vis_pos_cubic_grad_export_ply", False):
                self._export_wave_heatmap_ply(
                    pos_all=pos_tensor,
                    grad_magnitude=grad_magnitude,
                    out_path=stem + "_points.ply",
                    value_label="pos_cubic_grad"
                )
        except Exception as e:
            print(f"[pos-cubic-grad-vis] visualization skipped due to error: {e}")

    @torch.no_grad()
    def _vis_wave_heatmap_3d(self,
                             pos_all: torch.Tensor,
                             grad_magnitude: torch.Tensor,
                             out_path: str = "wave_grad_3d.png",
                             title: str = "Wave Heatmap",
                             cbar_label: str = "Magnitude") -> None:
        """
        3D scatter showing wave_coefficients heatmap.
        
        Parameters
        ----------
        pos_all : torch.Tensor [N, 3]
            Point cloud positions
        grad_magnitude : torch.Tensor [N]
            Gradient magnitude
        out_path : str
            Output path
        """
        p = pos_all.detach().cpu()
        grad = grad_magnitude.detach().cpu().float()
        
        # Limit point count to avoid heavy plots.
        max_points = getattr(self.cfg, "vis_max_points", 100000)
        N = p.shape[0]
        if N > max_points:
            idx = torch.randperm(N)[:max_points]
            p = p[idx]
            grad = grad[idx]
        
        # Depth sort.
        depth = p[:, 2].numpy()
        order = np.argsort(depth)
        p = p[order]
        grad = grad[order]
        
        # Normalize gradients for color mapping.
        grad_np = grad.numpy()
        if grad_np.max() > grad_np.min():
            grad_np = (grad_np - grad_np.min()) / (grad_np.max() - grad_np.min())
        
        # Use sigmoid to enhance contrast.
        grad_sigmoid = 1 / (1 + np.exp(-grad_np * 5))  # Multiply by 5 to enhance contrast.
        
        # Color mapping.
        cmap = plt.get_cmap("jet")
        colors = cmap(grad_sigmoid)
        sizes = 1.0 + 3.0 * grad_sigmoid  # Scale point size by gradient magnitude.
        
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(p[:, 0], p[:, 1], p[:, 2], s=sizes, c=colors, alpha=0.8, linewidths=0)
        
        # Set view and bounds.
        xyz_min, xyz_max = p.min(0).values.numpy(), p.max(0).values.numpy()
        ctr = (xyz_min + xyz_max) / 2.0
        rad = float(np.max(xyz_max - xyz_min) / 2.0 + 1e-6)
        ax.set_xlim([ctr[0] - rad, ctr[0] + rad])
        ax.set_ylim([ctr[1] - rad, ctr[1] + rad])
        ax.set_zlim([ctr[2] - rad, ctr[2] + rad])
        # Fix camera view to avoid upside-down orientation.
        ax.view_init(elev=25, azim=45)  # Adjust elevation/azimuth for a better view.
        ax.set_axis_off()
        ax.set_title(title)
        
        # Add colorbar.
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.8, pad=0.1)
        cbar.set_label(cbar_label, rotation=270, labelpad=15)
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, dpi=220, bbox_inches='tight')
        plt.close(fig)

    @torch.no_grad()
    def _vis_wave_heatmap_dynamic(self,
                                  grad_magnitude: torch.Tensor,
                                  out_path: str,
                                  num_frames: int = 20,
                                  label_prefix: str = "Wave") -> None:
        """
        Generate a dynamic visualization video for wave_coefficients values/gradients.
        
        Parameters
        ----------
        grad_magnitude : torch.Tensor [N]
            Gradient magnitude
        out_path : str
            Output path
        num_frames : int
            Frame count
        """
        pc = self.point_cloud
        
        # Time sampling.
        try:
            tk = pc.pos_tk
            if isinstance(tk, torch.Tensor):
                tk = tk.detach().cpu().numpy()
            else:
                tk = np.array(tk)
            t_start, t_end = float(tk[0]), float(tk[-1])
        except Exception:
            t_start, t_end = 0.0, float(max(1, num_frames - 1))
        ts = np.linspace(t_start, t_end, num_frames, dtype=np.float32)
        
        # Set camera parameters.
        width, height = 512, 512
        from pointrix.camera.cam_utils import construct_canonical_camera
        camera = construct_canonical_camera(width=width, height=height)
        
        # Normalize gradients.
        grad_np = grad_magnitude.detach().cpu().numpy()
        if grad_np.max() > grad_np.min():
            grad_normalized = (grad_np - grad_np.min()) / (grad_np.max() - grad_np.min())
        else:
            grad_normalized = np.zeros_like(grad_np)
        
        frames = []
        for frame_idx, t in enumerate(ts):
            with torch.no_grad():
                pos_t = pc.get_position(float(t))
                if isinstance(pos_t, torch.Tensor):
                    pos_t = pos_t.detach().cpu().numpy()
                else:
                    pos_t = np.array(pos_t)
            
            # Project to 2D.
            pos_tensor = torch.from_numpy(pos_t).float().cuda()
            uv, depth = self._project_points_to_2d(pos_tensor, camera)
            uv = uv.detach().cpu().numpy()
            depth = depth.detach().cpu().numpy()
            
            # Filter visible points.
            visible = (depth > 0).flatten()
            uv_vis = uv[visible]
            grad_vis = grad_normalized[visible]
            
            # Create figure.
            fig = plt.figure(figsize=(10, 5))
            
            # Left plot: 2D projection gradient heatmap.
            ax1 = fig.add_subplot(121)
            if len(uv_vis) > 0:
                # Use sigmoid to enhance contrast.
                grad_sigmoid = 1 / (1 + np.exp(-grad_vis * 5))
                scatter = ax1.scatter(uv_vis[:, 0], uv_vis[:, 1], 
                                      c=grad_sigmoid, cmap='jet', s=8, alpha=0.8,
                                      vmin=0, vmax=1)
                
                # Add colorbar.
                cbar = plt.colorbar(scatter, ax=ax1, shrink=0.8)
                cbar.set_label(f'{label_prefix} Magnitude')
            
            ax1.set_xlim(0, width)
            ax1.set_ylim(0, height)
            ax1.set_aspect('equal')
            ax1.invert_yaxis()
            ax1.set_title(f'{label_prefix} Heatmap (t={t:.2f})')
            ax1.set_xlabel('X (pixels)')
            ax1.set_ylabel('Y (pixels)')
            
            # Right plot: 3D view.
            ax2 = fig.add_subplot(122, projection='3d')
            if len(pos_t) > 0:
                grad_sigmoid_3d = 1 / (1 + np.exp(-grad_normalized * 5))
                ax2.scatter(pos_t[:, 0], pos_t[:, 2], pos_t[:, 1], 
                            c=grad_sigmoid_3d, cmap='jet', s=4, alpha=0.6, vmin=0, vmax=1)
            
            # Set 3D view.
            rotation_angle = (frame_idx / num_frames) * 360
            # Fix camera view with a better elevation to avoid upside-down.
            ax2.view_init(elev=30, azim=rotation_angle)
            ax2.set_title(f'3D {label_prefix} (t={t:.2f})')
            ax2.set_xlabel('X')
            ax2.set_ylabel('Z')
            ax2.set_zlabel('Y')
            
            plt.tight_layout()
            fig.canvas.draw()
            w, h = fig.canvas.get_width_height()
            
            # Handle different matplotlib versions.
            try:
                buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
                img = buf.reshape(h, w, 4)[:, :, :3]
            except AttributeError:
                buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                img = buf.reshape(h, w, 3)
            
            frames.append(img.copy())
            plt.close(fig)
        
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        imageio.mimsave(out_path, frames, fps=6)

    @torch.no_grad()
    def _export_wave_heatmap_ply(self,
                                 pos_all: torch.Tensor,
                                 grad_magnitude: torch.Tensor,
                                 out_path: str,
                                 value_label: str = "intensity") -> None:
        """
        Export wave heatmap data as a colored PLY point cloud.
        """
        if pos_all is None or grad_magnitude is None:
            return
        
        p = pos_all.detach().cpu().float()
        grad = grad_magnitude.detach().cpu().float()
        if p.numel() == 0 or grad.numel() == 0:
            return
        
        max_points = getattr(self.cfg, "vis_max_points", 100000)
        N = p.shape[0]
        if N > max_points:
            idx = torch.randperm(N)[:max_points]
            p = p[idx]
            grad = grad[idx]
        
        grad_np = grad.numpy()
        if grad_np.max() > grad_np.min():
            grad_norm = (grad_np - grad_np.min()) / (grad_np.max() - grad_np.min())
        else:
            grad_norm = np.zeros_like(grad_np)
        grad_sigmoid = 1 / (1 + np.exp(-grad_norm * 5))
        
        cmap = plt.get_cmap("jet")
        colors = (cmap(grad_sigmoid)[:, :3] * 255).astype(np.uint8)
        points = p.numpy()
        
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {points.shape[0]}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write(f"property float {value_label}\n")
            f.write("end_header\n")
            for (x, y, z), (r, g, b), val in zip(points, colors, grad_sigmoid):
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)} {val:.6f}\n")

    @torch.no_grad()
    def _export_viewspace_grad_png(self,
                                   pos_all: torch.Tensor,
                                   grad_values: torch.Tensor,
                                   out_path: str) -> None:
        """
        Export a 3D scatter PNG for viewspace_grad with colors from raw magnitude.
        """
        if pos_all.numel() == 0 or grad_values.numel() == 0:
            return

        p = pos_all.detach().cpu()
        grad = grad_values.detach().cpu().float()

        num_points = min(p.shape[0], grad.shape[0])
        if num_points == 0:
            return
        p = p[:num_points]
        grad = grad[:num_points]

        max_points = getattr(self.cfg, "vis_max_points", 100000)
        if num_points > max_points:
            idx = torch.randperm(num_points)[:max_points]
            p = p[idx]
            grad = grad[idx]

        grad_np = grad.numpy()
        grad_min = float(grad_np.min())
        grad_max = float(grad_np.max())
        if grad_max - grad_min < 1e-12:
            grad_max = grad_min + 1e-6
        grad_norm = (grad_np - grad_min) / (grad_max - grad_min)

        cmap = plt.get_cmap("turbo")
        norm = plt.Normalize(vmin=grad_min, vmax=grad_max)

        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(p[:, 0], p[:, 1], p[:, 2],
                             c=grad_np,
                             cmap=cmap,
                             s=1.0 + 4.0 * grad_norm,
                             alpha=0.85,
                             norm=norm,
                             linewidths=0)

        xyz_min, xyz_max = p.min(0).values.numpy(), p.max(0).values.numpy()
        ctr = (xyz_min + xyz_max) / 2.0
        rad = float(np.max(xyz_max - xyz_min) / 2.0 + 1e-6)
        ax.set_xlim([ctr[0] - rad, ctr[0] + rad])
        ax.set_ylim([ctr[1] - rad, ctr[1] + rad])
        ax.set_zlim([ctr[2] - rad, ctr[2] + rad])
        ax.view_init(elev=22, azim=38)
        ax.set_axis_off()
        ax.set_title("Viewspace Grad Magnitude")

        cbar = plt.colorbar(scatter, ax=ax, shrink=0.75, pad=0.1)
        cbar.set_label("Gradient Magnitude", rotation=270, labelpad=12)

        plt.tight_layout()
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, dpi=240, bbox_inches='tight')
        plt.close(fig)

    @torch.no_grad()
    def _export_viewspace_grad_ply(self,
                                   pos_all: torch.Tensor,
                                   grad_components: torch.Tensor,
                                   grad_magnitude: torch.Tensor,
                                   out_path: str) -> None:
        """
        Export a colored PLY for viewspace_grad including raw gradient values.
        """
        if pos_all.numel() == 0 or grad_components.numel() == 0:
            return

        p = pos_all.detach().cpu().float()
        grad_comp = grad_components.detach().cpu().float()
        grad_comp = grad_comp.view(grad_comp.shape[0], -1)
        grad_mag = grad_magnitude.detach().cpu().float()

        num_points = min(p.shape[0], grad_comp.shape[0], grad_mag.shape[0])
        if num_points == 0:
            return
        p = p[:num_points]
        grad_comp = grad_comp[:num_points]
        grad_mag = grad_mag[:num_points]

        max_points = getattr(self.cfg, "vis_max_points", 100000)
        if num_points > max_points:
            idx = torch.randperm(num_points)[:max_points]
            p = p[idx]
            grad_comp = grad_comp[idx]
            grad_mag = grad_mag[idx]

        grad_mag_np = grad_mag.numpy()
        grad_min = float(grad_mag_np.min())
        grad_max = float(grad_mag_np.max())
        if grad_max - grad_min > 1e-12:
            grad_norm = (grad_mag_np - grad_min) / (grad_max - grad_min)
        else:
            grad_norm = np.zeros_like(grad_mag_np)

        cmap = plt.get_cmap("turbo")
        colors = (cmap(grad_norm)[:, :3] * 255).astype(np.uint8)
        points = p.numpy()
        grad_comp_np = grad_comp.numpy()

        num_components = grad_comp_np.shape[1]

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {points.shape[0]}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("property float viewspace_grad_norm\n")
            for i in range(num_components):
                f.write(f"property float viewspace_grad_c{i}\n")
            f.write("end_header\n")
            for (x, y, z), (r, g, b), grad_val, comps in zip(points, colors, grad_mag_np, grad_comp_np):
                comp_str = " ".join(f"{float(c):.6f}" for c in np.atleast_1d(comps))
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)} {grad_val:.6f} {comp_str}\n")
