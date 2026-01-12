"""
Adaptive video 3D flow construction.

This module extends :mod:`video3Dflow.video_3d_flow` by adding
multi-order sampling that prioritises tracks with low temporal
support while keeping spatial coverage as uniform as possible.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import partial

import numpy as np
import torch
from tqdm import tqdm

from .video_3d_flow import Video3DFlow
from .utils import get_tracks_3d_for_query_frame


@dataclass
class AdaptiveVideo3DFlow(Video3DFlow):
    """Variant of :class:`Video3DFlow` with adaptive initialisation."""

    adaptive_grid_size: int = 24
    adaptive_min_points: int = 0
    adaptive_max_points: int = 1500

    def get_tracks_3d(
        self,
        num_samples: int,
        start: int = 0,
        end: int = -1,
        step: int = 1,
        *,
        min_points_per_query: int | None = None,
        max_points_per_query: int | None = None,
        grid_size: int | None = None,
    ):
        """Gather spatio-temporal point tracks using multi-order adaptive sampling."""

        min_points = (
            min_points_per_query if min_points_per_query is not None else self.adaptive_min_points
        )
        max_points = (
            max_points_per_query if max_points_per_query is not None else self.adaptive_max_points
        )
        grid = grid_size if grid_size is not None else self.adaptive_grid_size

        if grid < 1:
            raise ValueError("grid_size must be >= 1")
        if max_points < 1:
            raise ValueError("max_points_per_query must be >= 1")
        if min_points < 0:
            raise ValueError("min_points_per_query must be >= 0")

        num_frames = len(self.imgs)
        if end < 0:
            end = num_frames + 1 + end

        query_idcs = [
            i.item() for i in torch.linspace(start, end - 1, max(1, (end-start) // step)).round().to(torch.int32)
        ]
        target_idcs = [
            i.item() for i in torch.linspace(start, end - 1, max(1, (end-start) // step)).round().to(torch.int32)
        ]
        target_index_lookup = {idx: pos for pos, idx in enumerate(target_idcs)}

        masks = torch.stack([self.get_mask(i) for i in target_idcs], dim=0)
        mask_val = 1 if self.extract_fg else -1
        fg_masks = (masks == mask_val).float()

        depths = torch.stack([self.get_depth(i) for i in target_idcs], dim=0)
        range_min, range_max = self.depth_range_min, self.depth_range_max
        self.depths_min, self.depths_max = depths.min(), depths.max()
        depths = (depths - self.depths_min) / (self.depths_max - self.depths_min) * (range_max - range_min) + range_min

        tracks_all_queries: list[tuple[torch.Tensor, ...]] = []
        remaining_budget: int | None = num_samples if num_samples > 0 else None
        global_counts = np.zeros(grid * grid, dtype=np.int64)
        query_cache: dict[int, dict[str, torch.Tensor]] = {}

        if self.extract_fg:
            print("Extracting adaptive foreground tracks...")
        else:
            print("Extracting adaptive background tracks...")

        query_orders = self._generate_query_orders(len(query_idcs))
        total_steps = len(query_orders) * len(query_idcs)
        step_counter = 0

        total_steps = len(query_orders) * len(query_idcs)
        progress = tqdm(total=total_steps, desc="Adaptive tracks", unit="query")

        for order in query_orders:
            for rel_idx in order:
                if remaining_budget is not None and remaining_budget <= 0:
                    break
                progress.update(1)

                q_idx = query_idcs[rel_idx]
                tidx = target_index_lookup[q_idx]

                cache_entry = query_cache.get(q_idx)
                if cache_entry is None:
                    tracks_2d = self.load_target_tracks(q_idx, target_idcs)
                    if remaining_budget is not None:
                        sampling_cap = min(max_points, remaining_budget)
                    else:
                        sampling_cap = max_points
                    if sampling_cap > 0 and len(tracks_2d) > sampling_cap * 8:
                        sel_idcs = np.random.choice(len(tracks_2d), sampling_cap * 8, replace=False)
                        tracks_2d = tracks_2d[sel_idcs]

                    img = self.get_image(q_idx)

                    tracks_3d_full, colors_full, visibles_full, invisibles_full, confidences_full = (
                        get_tracks_3d_for_query_frame(
                            tidx,
                            img,
                            tracks_2d,
                            depths,
                            fg_masks,
                            extract_fg=self.extract_fg,
                            min_points=0,
                            max_points=max_points,
                            use_spatial_sampling=False,
                        )
                    )

                    if tracks_3d_full.shape[0] == 0:
                        continue

                    cache_entry = {
                        "tracks": tracks_3d_full,
                        "colors": colors_full,
                        "visibles": visibles_full,
                        "invisibles": invisibles_full,
                        "confidences": confidences_full,
                        "available": torch.arange(tracks_3d_full.shape[0], dtype=torch.long),
                    }
                    query_cache[q_idx] = cache_entry

                available_indices = cache_entry["available"]
                if available_indices.numel() == 0:
                    continue

                tracks_3d_full = cache_entry["tracks"]
                colors_full = cache_entry["colors"]
                visibles_full = cache_entry["visibles"]
                invisibles_full = cache_entry["invisibles"]
                confidences_full = cache_entry["confidences"]

                tracks_3d = tracks_3d_full[available_indices]
                colors = colors_full[available_indices]
                visibles = visibles_full[available_indices]
                invisibles = invisibles_full[available_indices]
                confidences = confidences_full[available_indices]

                available = tracks_3d.shape[0]
                base_min = min(min_points, available)

                step_counter += 1
                if remaining_budget is not None:
                    base_min = min(base_min, remaining_budget)
                    steps_left = max(1, total_steps - step_counter + 1)
                    avg_quota = math.ceil(remaining_budget / steps_left)
                    target_points = max(base_min, avg_quota)
                    target_points = min(target_points, max_points, available, remaining_budget)
                else:
                    target_points = min(max_points, available)
                    target_points = max(base_min, target_points)

                if target_points <= 0:
                    continue

                if self.extract_fg:
                    selected_idx = self._grid_select_tracks(
                        tracks_3d,
                        visibles,
                        confidences,
                        min_target=target_points,
                        max_target=target_points,
                        grid_size=grid,
                        global_cell_counts=global_counts,
                        query_frame_index=tidx,
                    )
                else:
                    selected_idx = self._adaptive_select_tracks(
                        tracks_3d,
                        visibles,
                        confidences,
                        min_target=target_points,
                        max_target=target_points,
                        grid_size=grid,
                        global_cell_counts=global_counts,
                        query_frame_index=tidx,
                    )

                if selected_idx.numel() == 0:
                    continue

                selected_global = available_indices[selected_idx]
                keep_mask = torch.ones_like(available_indices, dtype=torch.bool)
                keep_mask[selected_idx] = False
                cache_entry["available"] = available_indices[keep_mask]

                tracks_all_queries.append(
                    (
                        tracks_3d_full[selected_global],
                        colors_full[selected_global],
                        visibles_full[selected_global],
                        invisibles_full[selected_global],
                        confidences_full[selected_global],
                    )
                )

                if remaining_budget is not None:
                    remaining_budget = max(0, remaining_budget - selected_idx.numel())
                    if remaining_budget == 0:
                        break
            if remaining_budget is not None and remaining_budget <= 0:
                break

        if not tracks_all_queries:
            empty = torch.zeros(0, len(target_idcs), 3)
            empty_bool = torch.zeros(0, len(target_idcs), dtype=torch.bool)
            empty_color = torch.zeros(0, 3)
            progress.close()
            return empty, empty_bool, empty_bool, empty_bool, empty_color

        tracks_3d, colors, visibles, invisibles, confidences = map(
            partial(torch.cat, dim=0), zip(*tracks_all_queries)
        )
        progress.close()

        print(f"Adaptive result: {tracks_3d.shape[0]} total points")
        return tracks_3d, visibles, invisibles, confidences, colors

    def _generate_query_orders(self, num_queries: int) -> list[list[int]]:
        """Create multiple traversal orders to cover temporal occlusions."""

        if num_queries <= 0:
            return []
        base = list(range(num_queries))
        orders: list[list[int]] = [base]
        if num_queries > 1:
            orders.append(list(reversed(base)))
        for pivot in range(1, num_queries - 1):
            left = list(range(pivot - 1, -1, -1))
            right = list(range(pivot + 1, num_queries))
            orders.append([pivot] + left + right)
            orders.append([pivot] + right + left)
        return orders

    def _grid_select_tracks(
        self,
        tracks_3d: torch.Tensor,
        visibles: torch.Tensor,
        confidences: torch.Tensor,
        *,
        min_target: int,
        max_target: int,
        grid_size: int,
        global_cell_counts: np.ndarray,
        query_frame_index: int,
    ) -> torch.Tensor:
        num_tracks = tracks_3d.shape[0]
        if num_tracks == 0:
            return torch.zeros(0, dtype=torch.long)

        max_target = min(max_target, num_tracks)
        min_target = max(0, min(min_target, num_tracks))
        target_num_points = max(min_target, max_target)
        target_num_points = min(target_num_points, num_tracks)
        if target_num_points == 0:
            return torch.zeros(0, dtype=torch.long)

        conf = confidences.float().mean(dim=1).cpu().numpy()
        grid_indices = self._compute_grid_indices(tracks_3d[:, query_frame_index, :2], grid_size).cpu().numpy()

        cell_to_candidates: dict[int, list[int]] = {}
        for idx, cell in enumerate(grid_indices):
            cell_to_candidates.setdefault(int(cell), []).append(int(idx))

        if not cell_to_candidates:
            return torch.zeros(0, dtype=torch.long)

        for candidates in cell_to_candidates.values():
            candidates.sort(key=lambda i: -conf[i])

        cell_order = sorted(
            cell_to_candidates.keys(), key=lambda c: (global_cell_counts[c], -len(cell_to_candidates[c]), c)
        )

        selected: list[int] = []
        remaining_need = target_num_points

        for cell in cell_order:
            if remaining_need == 0:
                break
            candidates = cell_to_candidates[cell]
            while candidates and remaining_need > 0:
                idx = candidates.pop(0)
                selected.append(idx)
                remaining_need -= 1
                global_cell_counts[cell] += 1

        if remaining_need > 0:
            leftovers = sorted(
                (
                    idx
                    for candidates in cell_to_candidates.values()
                    for idx in candidates
                ),
                key=lambda i: -conf[i],
            )
            extras = leftovers[:remaining_need]
            selected.extend(extras)
            for idx in extras:
                global_cell_counts[int(grid_indices[idx])] += 1

        final_indices = sorted(selected[:target_num_points], key=lambda idx: -conf[idx])
        return torch.tensor(final_indices, dtype=torch.long)

    def _adaptive_select_tracks(
        self,
        tracks_3d: torch.Tensor,
        visibles: torch.Tensor,
        confidences: torch.Tensor,
        *,
        min_target: int,
        max_target: int,
        grid_size: int,
        global_cell_counts: np.ndarray,
        query_frame_index: int,
    ) -> torch.Tensor:
        num_tracks = tracks_3d.shape[0]
        if num_tracks == 0:
            return torch.zeros(0, dtype=torch.long)

        max_target = min(max_target, num_tracks)
        min_target = max(0, min(min_target, num_tracks))
        target_num_points = max(min_target, max_target)
        target_num_points = min(target_num_points, num_tracks)
        if target_num_points == 0:
            return torch.zeros(0, dtype=torch.long)

        freq = visibles.float().sum(dim=1).cpu().numpy()
        conf = confidences.float().mean(dim=1).cpu().numpy()
        grid_indices = self._compute_grid_indices(tracks_3d[:, query_frame_index, :2], grid_size).cpu().numpy()

        cell_to_candidates: dict[int, list[int]] = {}
        for idx, cell in enumerate(grid_indices):
            cell_to_candidates.setdefault(int(cell), []).append(int(idx))

        if not cell_to_candidates:
            return torch.zeros(0, dtype=torch.long)

        for candidates in cell_to_candidates.values():
            candidates.sort(key=lambda i: (freq[i], -conf[i]))

        cell_ids = np.array(list(cell_to_candidates.keys()), dtype=np.int32)
        freq_priority = np.array(
            [1.0 / (freq[cell_to_candidates[cell][0]] + 1e-3) for cell in cell_ids],
            dtype=np.float32,
        )
        density = np.array([len(cell_to_candidates[cell]) for cell in cell_ids], dtype=np.float32)

        if freq_priority.max() == freq_priority.min():
            freq_priority_norm = np.ones_like(freq_priority)
        else:
            freq_priority_norm = (freq_priority - freq_priority.min()) / (
                freq_priority.max() - freq_priority.min()
            )

        if density.max() == 0:
            density_norm = np.zeros_like(density)
        else:
            density_norm = density / density.max()

        global_counts = global_cell_counts[cell_ids]
        if global_counts.max() == global_counts.min():
            global_norm = np.ones_like(global_counts)
        else:
            global_norm = 1.0 - (global_counts - global_counts.min()) / (
                global_counts.max() - global_counts.min() + 1e-6
            )

        cell_priority = 0.45 * freq_priority_norm + 0.30 * density_norm + 0.25 * global_norm
        cell_order = cell_ids[np.argsort(-cell_priority)]

        selected: list[int] = []
        cell_counts: dict[int, int] = {int(cell): 0 for cell in cell_ids}
        quota = 1
        total_needed = target_num_points

        while len(selected) < total_needed and cell_order.size > 0:
            progress = False
            for cell in cell_order:
                if len(selected) >= total_needed:
                    break
                cell_int = int(cell)
                if cell_counts[cell_int] >= quota:
                    continue
                candidates = cell_to_candidates[cell_int]
                if not candidates:
                    continue
                idx = candidates.pop(0)
                selected.append(idx)
                cell_counts[cell_int] += 1
                global_cell_counts[cell_int] += 1
                progress = True
                if len(selected) >= total_needed:
                    break
            if len(selected) >= total_needed:
                break
            if not progress:
                quota += 1
                if quota > total_needed:
                    break

        if len(selected) < total_needed:
            remaining = sorted(
                set(range(num_tracks)) - set(selected),
                key=lambda i: (freq[i], -conf[i]),
            )
            for idx in remaining:
                selected.append(idx)
                global_cell_counts[int(grid_indices[idx])] += 1
                if len(selected) >= total_needed:
                    break

        final_indices = sorted(selected, key=lambda idx: (freq[idx], -conf[idx]))
        return torch.tensor(final_indices[:target_num_points], dtype=torch.long)

    @staticmethod
    def _compute_grid_indices(normalised_xy: torch.Tensor, grid_size: int) -> torch.Tensor:
        """Map normalised xy coordinates in [-1, 1] onto a 2-D grid."""

        if normalised_xy.numel() == 0:
            return torch.zeros(0, dtype=torch.long, device=normalised_xy.device)

        coords = normalised_xy[:, :2].clamp(-1.0, 1.0)
        coords = (coords + 1.0) / 2.0
        coords = coords.clamp(0.0, 1.0 - 1e-6)
        grid_coords = torch.floor(coords * grid_size).long().clamp(0, grid_size - 1)
        grid_x = grid_coords[:, 0]
        grid_y = grid_coords[:, 1]
        return grid_y * grid_size + grid_x
