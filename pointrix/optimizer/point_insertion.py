from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class MotionPointInsertionConfig:
    """Configuration for spline control-point insertion."""

    enable: bool = True
    warmup_steps: int = 2000
    check_interval: int = 500
    grad_threshold: float = 1e-3
    grad_quantile: float = 0.9
    max_new_knots_per_check: int = 1
    min_frame_distance: float = 1.0
    max_total_knots: Optional[int] = None
    time_epsilon: float = 1e-4


class MotionPointInsertion:
    """
    Monitor spline control-point gradients and insert new knots when motion
    complexity is high. The insertion happens under torch.no_grad and uses the
    current spline evaluation as the initial value for the new knot so the
    trajectory remains unchanged before further optimization.
    """

    def __init__(
        self,
        point_cloud,
        optimizer,
        cfg: MotionPointInsertionConfig,
    ) -> None:
        self.point_cloud = point_cloud
        self.optimizer = optimizer
        self.cfg = cfg
        self._enabled = bool(cfg.enable) and self._has_required_attributes()
        self._pos_tk_normalized = self._detect_normalized_pos_tk()
        self._total_time_span = self._infer_total_time_span()

    def maybe_insert(self, step: int) -> None:
        if not self._enabled:
            return
        if step < self.cfg.warmup_steps:
            return
        if (step - self.cfg.warmup_steps) % max(self.cfg.check_interval, 1) != 0:
            return
        self._run_insertion(step)

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #
    def _has_required_attributes(self) -> bool:
        required = ("pos_cubic_node", "pos_tk", "get_position")
        for name in required:
            if not hasattr(self.point_cloud, name):
                print(f"[point-insertion] '{name}' missing on point cloud; feature disabled.")
                return False
        return True

    def _detect_normalized_pos_tk(self) -> bool:
        tk = getattr(self.point_cloud, "pos_tk", None)
        if tk is None or tk.numel() == 0:
            return True
        tk = tk.detach()
        max_val = float(torch.max(tk))
        min_val = float(torch.min(tk))
        if max_val <= 1.0001 and min_val >= -1e-6:
            return True
        return False

    def _infer_total_time_span(self) -> float:
        delta = getattr(self.point_cloud, "delta_position", None)
        if delta is not None and delta.shape[0] > 0:
            return max(float(delta.shape[0] - 1), 1.0)
        time_len = getattr(self.point_cloud, "time_len", None)
        if time_len is not None:
            return max(float(time_len), 1.0)
        # Fallback to number of intervals if nothing else is available.
        intervals = getattr(self.point_cloud, "pos_interval_num", 2)
        return max(float(intervals - 1), 1.0)

    def _current_knot_times(self, device: Optional[torch.device] = None) -> torch.Tensor:
        tk = getattr(self.point_cloud, "pos_tk", None)
        if tk is None:
            return torch.empty(0)
        tk = tk.detach()
        if device is not None:
            tk = tk.to(device)
        if self._pos_tk_normalized:
            return tk * self._total_time_span
        return tk

    def _to_storage_time(self, actual_time: float, device, dtype) -> torch.Tensor:
        t = torch.as_tensor(actual_time, device=device, dtype=dtype)
        if self._pos_tk_normalized:
            return t / max(self._total_time_span, 1e-6)
        return t

    def _run_insertion(self, step: int) -> None:
        grad = getattr(self.point_cloud.pos_cubic_node, "grad", None)
        if grad is None:
            return
        num_points = grad.shape[0]
        num_knots = getattr(self.point_cloud, "pos_interval_num", None)
        if num_knots is None or num_knots < 2:
            return

        grad_nodes = grad.detach().view(num_points, num_knots, 3)
        interval_scores = self._compute_interval_scores(grad_nodes)
        if interval_scores.numel() == 0:
            return

        knot_times = self._current_knot_times(device=interval_scores.device)
        if knot_times.numel() != num_knots:
            return

        candidates = self._select_intervals(interval_scores, knot_times)
        if not candidates:
            return

        inserts_done = 0
        for interval_idx in candidates:
            if self.cfg.max_new_knots_per_check > 0 and inserts_done >= self.cfg.max_new_knots_per_check:
                break
            if self.cfg.max_total_knots is not None:
                if self.point_cloud.pos_interval_num >= self.cfg.max_total_knots:
                    break
            mid_time = 0.5 * (knot_times[interval_idx] + knot_times[interval_idx + 1])
            if self._insert_single(mid_time, interval_idx, step):
                inserts_done += 1
                knot_times = self._current_knot_times(device=knot_times.device)
            else:
                continue

    def _compute_interval_scores(self, grad_nodes: torch.Tensor) -> torch.Tensor:
        # grad_nodes: [N, M, 3]
        if grad_nodes.numel() == 0:
            return torch.empty(0, device=grad_nodes.device)
        per_knot = torch.linalg.vector_norm(grad_nodes, dim=-1).mean(dim=0)  # [M]
        left = per_knot[:-1]
        right = per_knot[1:]
        interval_scores = torch.maximum(left, right)  # [M-1]
        return interval_scores

    def _select_intervals(self, scores: torch.Tensor, knot_times: torch.Tensor) -> list:
        if scores.numel() == 0:
            return []
        threshold = self.cfg.grad_threshold
        if 0.0 < self.cfg.grad_quantile < 1.0:
            quant = torch.quantile(scores, self.cfg.grad_quantile).item()
            threshold = max(threshold, quant)
        lengths = knot_times[1:] - knot_times[:-1]
        candidate_mask = torch.logical_and(scores >= threshold, lengths >= self.cfg.min_frame_distance)
        candidate_ids = torch.nonzero(candidate_mask, as_tuple=False).flatten()
        if candidate_ids.numel() == 0:
            return []
        sort_idx = torch.argsort(scores[candidate_ids], descending=True)
        ordered = candidate_ids[sort_idx]
        return ordered.cpu().tolist()

    def _insert_single(self, time_value: float, interval_idx: int, step: int) -> bool:
        pc = self.point_cloud
        if pc.pos_interval_num < 2:
            return False
        device = pc.pos_cubic_node.device
        dtype = pc.pos_tk.dtype if hasattr(pc, "pos_tk") else torch.float32

        current_times = self._current_knot_times(device=device)
        if current_times.numel() == 0:
            return False

        if torch.any(torch.isclose(current_times, torch.tensor(time_value, device=device), atol=self.cfg.time_epsilon)):
            return False

        storage_t = self._to_storage_time(time_value, device=device, dtype=dtype)
        tk = pc.pos_tk.to(device=device, dtype=dtype)
        insert_idx = torch.searchsorted(tk, storage_t)
        insert_idx = torch.clamp(insert_idx, 1, tk.shape[0] - 1)

        pos_nodes = pc.pos_cubic_node.detach().view(-1, pc.pos_interval_num, 3)
        with torch.no_grad():
            predicted_pos = pc.get_position(float(time_value)).detach()
            delta = (predicted_pos - pc.position).detach()
            new_nodes = torch.cat(
                [
                    pos_nodes[:, :insert_idx, :],
                    delta.unsqueeze(1),
                    pos_nodes[:, insert_idx:, :],
                ],
                dim=1,
            )
            new_flat = new_nodes.reshape(pos_nodes.shape[0], -1)
            pc.replace({"pos_cubic_node": new_flat}, self.optimizer)
            new_tk = torch.cat([tk[:insert_idx], storage_t[None], tk[insert_idx:]], dim=0)
            pc.pos_tk = new_tk
            pc.intervals = new_tk
            pc.pos_interval_num = new_nodes.shape[1]

        print(
            f"[point-insertion] step {step}: added knot at t={time_value:.3f} "
            f"(interval {interval_idx}, total_knots={pc.pos_interval_num})"
        )
        return True
