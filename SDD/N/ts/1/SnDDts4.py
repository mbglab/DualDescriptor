# Copyright (C) 2005-2026, Bin-Guang Ma (mbg@mail.hzau.edu.cn); SPDX-License-Identifier: MIT
# The Spatial Numerical Dual Descriptor Vector class (Tensor form) for 4D array implemented with PyTorch
# This program is for the demonstration of methodology and not fully refined.
# Author: Bin-Guang Ma (assisted by DeepSeek); Date: 2025-8-28 ~ 2026-04-05

import math
import random
import itertools
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy

class SpatialNumDualDescriptorTS4(nn.Module):
    """
    Spatial Numerical Dual Descriptor for 4D arrays with GPU acceleration using PyTorch:
      - Processes 4D grids of m-dimensional real vectors (H x W x D x T x m) instead of 1D sequences
      - tensor P ∈ R^{m×m×o} of basis coefficients
      - mapping matrix M ∈ R^{m×m} for vector transformation (square matrix)
      - indexed periods for four spatial dimensions:
          period1[i,j,g] = i*(m*o) + j*o + g + 2
          period2[i,j,g] = i*(m*o) + j*o + g + 3
          period3[i,j,g] = i*(m*o) + j*o + g + 4
          period4[i,j,g] = i*(m*o) + j*o + g + 5
      - basis function phi_{i,j,g}(k1,k2,k3,k4) = cos(2π*k1/period1) * cos(2π*k2/period2) * cos(2π*k3/period3) * cos(2π*k4/period4)
      - supports 'linear' or 'nonlinear' (step-by-step) window extraction
      - batch acceleration for equal-sized grids when rank_mode == 'drop'
    """
    def __init__(self, vec_dim, rank=1, rank_op='avg', rank_mode='drop', num_basis=5, mode='linear',
                 user_step=None, device='cuda'):
        """
        Initialize the Spatial Numerical Dual Descriptor model for 4D arrays.

        Args:
            vec_dim: Dimension of input vectors and internal representation
            rank: Window size (int or tuple (rank_h, rank_w, rank_d, rank_t)); if int, use hypercubic windows of size rank^4
            rank_op: 'avg', 'sum', 'pick', 'user_func'
            rank_mode: 'pad' or 'drop' (how to handle incomplete windows at borders)
            num_basis: number of basis terms
            mode: 'linear' (step=1) or 'nonlinear' (step=user_step or rank)
            user_step: custom step size for nonlinear mode (int or tuple (step_h, step_w, step_d, step_t))
            device: 'cuda' or 'cpu'
        """
        super().__init__()
        self.vec_dim = vec_dim
        # Handle rank as int or tuple
        if isinstance(rank, int):
            self.rank_h = rank
            self.rank_w = rank
            self.rank_d = rank
            self.rank_t = rank
        else:
            self.rank_h, self.rank_w, self.rank_d, self.rank_t = rank
        self.rank_op = rank_op
        self.rank_mode = rank_mode
        self.m = vec_dim
        self.o = num_basis
        assert mode in ('linear', 'nonlinear')
        self.mode = mode
        # Handle step as int or tuple
        if user_step is None:
            self.step_h = self.step_w = self.step_d = self.step_t = None
        elif isinstance(user_step, int):
            self.step_h = self.step_w = self.step_d = self.step_t = user_step
        else:
            self.step_h, self.step_w, self.step_d, self.step_t = user_step
        self.trained = False
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # Mapping matrix M
        self.M = nn.Linear(self.vec_dim, self.vec_dim, bias=False)

        # Position-weight tensor P[i][j][g]
        self.P = nn.Parameter(torch.empty(self.m, self.m, self.o))

        # Precompute indexed periods for four dimensions (fixed, not trainable)
        periods1 = torch.zeros(self.m, self.m, self.o, dtype=torch.float32)
        periods2 = torch.zeros(self.m, self.m, self.o, dtype=torch.float32)
        periods3 = torch.zeros(self.m, self.m, self.o, dtype=torch.float32)
        periods4 = torch.zeros(self.m, self.m, self.o, dtype=torch.float32)
        for i in range(self.m):
            for j in range(self.m):
                for g in range(self.o):
                    base = i*(self.m*self.o) + j*self.o + g
                    periods1[i, j, g] = base + 2
                    periods2[i, j, g] = base + 3
                    periods3[i, j, g] = base + 4
                    periods4[i, j, g] = base + 5
        self.register_buffer('periods1', periods1)
        self.register_buffer('periods2', periods2)
        self.register_buffer('periods3', periods3)
        self.register_buffer('periods4', periods4)

        # Classifier head (initialized later)
        self.num_classes = None
        self.classifier = None

        # Label head (initialized later)
        self.num_labels = None
        self.labeller = None

        # User function for custom rank operation
        self.user_func = None

        # Initialize parameters
        self.reset_parameters()
        self.to(self.device)

    def reset_parameters(self):
        """Initialize model parameters"""
        nn.init.uniform_(self.M.weight, -0.5, 0.5)
        nn.init.uniform_(self.P, -0.1, 0.1)
        if self.classifier is not None:
            nn.init.normal_(self.classifier.weight, 0, 0.01)
            if self.classifier.bias is not None:
                nn.init.constant_(self.classifier.bias, 0)
        if self.labeller is not None:
            nn.init.xavier_uniform_(self.labeller.weight)
            nn.init.zeros_(self.labeller.bias)

    def set_user_func(self, func):
        """Set custom user function for rank operation"""
        if callable(func):
            self.user_func = func
        else:
            raise ValueError("User function must be callable")

    # --------------------------------------------------------------------------
    # Helper methods for window extraction (both per-grid and batch)
    # --------------------------------------------------------------------------
    def _get_row_starts(self, H):
        """Compute starting row indices for windows based on mode and rank_mode."""
        if self.mode == 'linear':
            step = 1
        else:
            step = self.step_h if self.step_h is not None else self.rank_h
        if self.rank_mode == 'drop':
            starts = list(range(0, H - self.rank_h + 1, step))
        else:  # 'pad'
            starts = list(range(0, H, step))
        return starts

    def _get_col_starts(self, W):
        """Compute starting column indices for windows based on mode and rank_mode."""
        if self.mode == 'linear':
            step = 1
        else:
            step = self.step_w if self.step_w is not None else self.rank_w
        if self.rank_mode == 'drop':
            starts = list(range(0, W - self.rank_w + 1, step))
        else:  # 'pad'
            starts = list(range(0, W, step))
        return starts

    def _get_depth_starts(self, D):
        """Compute starting depth indices for windows based on mode and rank_mode."""
        if self.mode == 'linear':
            step = 1
        else:
            step = self.step_d if self.step_d is not None else self.rank_d
        if self.rank_mode == 'drop':
            starts = list(range(0, D - self.rank_d + 1, step))
        else:  # 'pad'
            starts = list(range(0, D, step))
        return starts

    def _get_time_starts(self, T):
        """Compute starting time indices for windows based on mode and rank_mode."""
        if self.mode == 'linear':
            step = 1
        else:
            step = self.step_t if self.step_t is not None else self.rank_t
        if self.rank_mode == 'drop':
            starts = list(range(0, T - self.rank_t + 1, step))
        else:  # 'pad'
            starts = list(range(0, T, step))
        return starts

    def _aggregate_window(self, window_vectors):
        """
        Aggregate vectors within a window according to rank_op.
        window_vectors: shape [batch, n_windows, rank_h*rank_w*rank_d*rank_t, m]
        Returns: aggregated vectors [batch, n_windows, m]
        """
        batch_size, n_windows, n_vecs, m = window_vectors.shape
        if self.rank_op == 'sum':
            return window_vectors.sum(dim=2)  # [batch, n_windows, m]
        elif self.rank_op == 'pick':
            # For each window, randomly pick one vector (independent per window)
            idx = torch.randint(0, n_vecs, (batch_size, n_windows, 1), device=self.device)
            idx_expanded = idx.unsqueeze(-1).expand(-1, -1, -1, m)  # [batch, n_windows, 1, m]
            picked = torch.gather(window_vectors, 2, idx_expanded).squeeze(2)  # [batch, n_windows, m]
            return picked
        elif self.rank_op == 'user_func' and self.user_func is not None:
            # Apply user function to each window's set of vectors
            agg_list = []
            for b in range(batch_size):
                for w in range(n_windows):
                    win_vecs = window_vectors[b, w]  # [n_vecs, m]
                    agg_list.append(self.user_func(win_vecs))
            return torch.stack(agg_list).view(batch_size, n_windows, m)
        else:  # default 'avg'
            return window_vectors.mean(dim=2)  # [batch, n_windows, m]

    def batch_represent(self, batch_grids):
        """
        Compute sequence representations for a batch of 4D grids efficiently.
        Assumes all grids have the same dimensions (H, W, D, T) and rank_mode == 'drop'.

        Args:
            batch_grids: Tensor of shape [batch_size, H, W, D, T, vec_dim]

        Returns:
            representations: Tensor of shape [batch_size, vec_dim]
        """
        batch_grids = batch_grids.to(self.device)
        batch_size, H, W, D, T, m = batch_grids.shape

        # Get window start indices
        row_starts = self._get_row_starts(H)
        col_starts = self._get_col_starts(W)
        depth_starts = self._get_depth_starts(D)
        time_starts = self._get_time_starts(T)
        if not row_starts or not col_starts or not depth_starts or not time_starts:
            return torch.zeros(batch_size, m, device=self.device)

        n_rows = len(row_starts)
        n_cols = len(col_starts)
        n_depths = len(depth_starts)
        n_times = len(time_starts)
        n_windows = n_rows * n_cols * n_depths * n_times

        # Permute to [batch, channels, height, width, depth, time]
        x = batch_grids.permute(0, 5, 1, 2, 3, 4)  # [batch, m, H, W, D, T]

        step_h = 1 if self.mode == 'linear' else (self.step_h if self.step_h is not None else self.rank_h)
        step_w = 1 if self.mode == 'linear' else (self.step_w if self.step_w is not None else self.rank_w)
        step_d = 1 if self.mode == 'linear' else (self.step_d if self.step_d is not None else self.rank_d)
        step_t = 1 if self.mode == 'linear' else (self.step_t if self.step_t is not None else self.rank_t)

        # Extract windows using time loop + depth loop + 2D unfold
        all_windows = []
        for t_start in time_starts:
            # Extract time block of size rank_t
            if t_start + self.rank_t > T:
                if self.rank_mode == 'pad':
                    frag_t = x[:, :, :, :, :, t_start:T]
                    pad_t = self.rank_t - frag_t.shape[5]
                    padding = torch.zeros(frag_t.shape[0], frag_t.shape[1], frag_t.shape[2], frag_t.shape[3], frag_t.shape[4], pad_t, device=self.device)
                    frag_t = torch.cat([frag_t, padding], dim=5)
                else:
                    continue
            else:
                frag_t = x[:, :, :, :, :, t_start:t_start+self.rank_t]  # [batch, m, H, W, D, rank_t]

            # For each depth start
            for d_start in depth_starts:
                if d_start + self.rank_d > D:
                    if self.rank_mode == 'pad':
                        frag_d = frag_t[:, :, :, :, d_start:D, :]
                        pad_d = self.rank_d - frag_d.shape[4]
                        padding = torch.zeros(frag_d.shape[0], frag_d.shape[1], frag_d.shape[2], frag_d.shape[3], pad_d, frag_d.shape[5], device=self.device)
                        frag_d = torch.cat([frag_d, padding], dim=4)
                    else:
                        continue
                else:
                    frag_d = frag_t[:, :, :, :, d_start:d_start+self.rank_d, :]  # [batch, m, H, W, rank_d, rank_t]

                # Combine depth and time dimensions with channels
                batch, c, h, w, rd, rt = frag_d.shape
                frag_2d = frag_d.permute(0, 1, 4, 5, 2, 3).reshape(batch, c * rd * rt, h, w)  # [batch, m*rank_d*rank_t, H, W]

                # Use 2D unfold to extract windows in (H, W)
                windows_2d = torch.nn.functional.unfold(
                    frag_2d, kernel_size=(self.rank_h, self.rank_w), stride=(step_h, step_w)
                )  # [batch, (m*rank_d*rank_t)*rank_h*rank_w, n_windows_hw]

                n_windows_hw = windows_2d.shape[2]
                # Reshape to [batch, n_windows_hw, rank_h, rank_w, rank_d, rank_t, m]
                windows_2d = windows_2d.view(batch, m, self.rank_d, self.rank_t, self.rank_h, self.rank_w, n_windows_hw).permute(0, 6, 4, 5, 2, 3, 1)
                windows_flat = windows_2d.reshape(batch, n_windows_hw, self.rank_h * self.rank_w * self.rank_d * self.rank_t, m)
                all_windows.append(windows_flat)

        if not all_windows:
            return torch.zeros(batch_size, m, device=self.device)

        # Concatenate windows from all time and depth starts: [batch, n_windows, prod(ranks), m]
        windows_all = torch.cat(all_windows, dim=1)  # n_windows = n_rows * n_cols * n_depths * n_times

        # Aggregate each window according to rank_op
        agg = self._aggregate_window(windows_all)  # [batch, n_windows, m]

        # Build coordinate tensors in the SAME order as windows:
        # Order: for each time_start, for each depth_start, for each row_start, for each col_start
        k1_list, k2_list, k3_list, k4_list = [], [], [], []
        for t in time_starts:
            for d in depth_starts:
                for i in row_starts:
                    for j in col_starts:
                        k1_list.append(i)
                        k2_list.append(j)
                        k3_list.append(d)
                        k4_list.append(t)

        k1_coords = torch.tensor(k1_list, dtype=torch.float32, device=self.device)  # [n_windows]
        k2_coords = torch.tensor(k2_list, dtype=torch.float32, device=self.device)
        k3_coords = torch.tensor(k3_list, dtype=torch.float32, device=self.device)
        k4_coords = torch.tensor(k4_list, dtype=torch.float32, device=self.device)

        # Expand to batch dimension
        k1_expanded = k1_coords.unsqueeze(0).expand(batch_size, -1)  # [batch, n_windows]
        k2_expanded = k2_coords.unsqueeze(0).expand(batch_size, -1)
        k3_expanded = k3_coords.unsqueeze(0).expand(batch_size, -1)
        k4_expanded = k4_coords.unsqueeze(0).expand(batch_size, -1)

        # Flatten agg and coordinates to compute Nk
        agg_flat = agg.reshape(batch_size * n_windows, m)  # [batch*n_windows, m]
        k1_flat = k1_expanded.reshape(batch_size * n_windows)
        k2_flat = k2_expanded.reshape(batch_size * n_windows)
        k3_flat = k3_expanded.reshape(batch_size * n_windows)
        k4_flat = k4_expanded.reshape(batch_size * n_windows)

        # Compute Nk for all windows
        Nk_flat = self.batch_compute_Nk(k1_flat, k2_flat, k3_flat, k4_flat, agg_flat)  # [batch*n_windows, m]
        Nk = Nk_flat.reshape(batch_size, n_windows, m)  # [batch, n_windows, m]

        # Average over windows to get grid-level representation
        rep = Nk.mean(dim=1)  # [batch, m]
        return rep

    def batch_compute_Nk_and_targets(self, batch_grids):
        """
        Compute Nk vectors and target vectors for each window in a batch of 4D grids.
        Assumes all grids have the same dimensions (H, W, D, T) and rank_mode == 'drop'.
        Used in self-training.

        Args:
            batch_grids: Tensor of shape [batch_size, H, W, D, T, m]

        Returns:
            Nk_all: Tensor [batch_size, n_windows, m] - Nk vectors for each window
            targets: Tensor [batch_size, n_windows, m] - target vectors = M(aggregated(window))
        """
        batch_grids = batch_grids.to(self.device)
        batch_size, H, W, D, T, m = batch_grids.shape

        # Get window start indices
        row_starts = self._get_row_starts(H)
        col_starts = self._get_col_starts(W)
        depth_starts = self._get_depth_starts(D)
        time_starts = self._get_time_starts(T)
        if not row_starts or not col_starts or not depth_starts or not time_starts:
            return torch.empty(batch_size, 0, m, device=self.device), torch.empty(batch_size, 0, m, device=self.device)

        n_rows = len(row_starts)
        n_cols = len(col_starts)
        n_depths = len(depth_starts)
        n_times = len(time_starts)
        n_windows = n_rows * n_cols * n_depths * n_times

        # Permute to [batch, channels, height, width, depth, time]
        x = batch_grids.permute(0, 5, 1, 2, 3, 4)  # [batch, m, H, W, D, T]

        step_h = 1 if self.mode == 'linear' else (self.step_h if self.step_h is not None else self.rank_h)
        step_w = 1 if self.mode == 'linear' else (self.step_w if self.step_w is not None else self.rank_w)
        step_d = 1 if self.mode == 'linear' else (self.step_d if self.step_d is not None else self.rank_d)
        step_t = 1 if self.mode == 'linear' else (self.step_t if self.step_t is not None else self.rank_t)

        # Extract windows using time loop + depth loop + 2D unfold (same as batch_represent)
        all_windows = []
        for t_start in time_starts:
            if t_start + self.rank_t > T:
                if self.rank_mode == 'pad':
                    frag_t = x[:, :, :, :, :, t_start:T]
                    pad_t = self.rank_t - frag_t.shape[5]
                    padding = torch.zeros(frag_t.shape[0], frag_t.shape[1], frag_t.shape[2], frag_t.shape[3], frag_t.shape[4], pad_t, device=self.device)
                    frag_t = torch.cat([frag_t, padding], dim=5)
                else:
                    continue
            else:
                frag_t = x[:, :, :, :, :, t_start:t_start+self.rank_t]

            for d_start in depth_starts:
                if d_start + self.rank_d > D:
                    if self.rank_mode == 'pad':
                        frag_d = frag_t[:, :, :, :, d_start:D, :]
                        pad_d = self.rank_d - frag_d.shape[4]
                        padding = torch.zeros(frag_d.shape[0], frag_d.shape[1], frag_d.shape[2], frag_d.shape[3], pad_d, frag_d.shape[5], device=self.device)
                        frag_d = torch.cat([frag_d, padding], dim=4)
                    else:
                        continue
                else:
                    frag_d = frag_t[:, :, :, :, d_start:d_start+self.rank_d, :]

                batch, c, h, w, rd, rt = frag_d.shape
                frag_2d = frag_d.permute(0, 1, 4, 5, 2, 3).reshape(batch, c * rd * rt, h, w)
                windows_2d = torch.nn.functional.unfold(
                    frag_2d, kernel_size=(self.rank_h, self.rank_w), stride=(step_h, step_w)
                )
                n_windows_hw = windows_2d.shape[2]
                windows_2d = windows_2d.view(batch, m, self.rank_d, self.rank_t, self.rank_h, self.rank_w, n_windows_hw).permute(0, 6, 4, 5, 2, 3, 1)
                windows_flat = windows_2d.reshape(batch, n_windows_hw, self.rank_h * self.rank_w * self.rank_d * self.rank_t, m)
                all_windows.append(windows_flat)

        if not all_windows:
            return torch.empty(batch_size, 0, m, device=self.device), torch.empty(batch_size, 0, m, device=self.device)

        windows_all = torch.cat(all_windows, dim=1)  # [batch, n_windows, prod(ranks), m]

        # Aggregate each window (no M yet)
        agg = self._aggregate_window(windows_all)  # [batch, n_windows, m]

        # Compute targets: M(agg)
        agg_flat = agg.reshape(batch_size * n_windows, m)
        targets_flat = self.M(agg_flat)  # [batch*n_windows, m]
        targets = targets_flat.reshape(batch_size, n_windows, m)

        # Build coordinate tensors in the SAME order as windows
        k1_list, k2_list, k3_list, k4_list = [], [], [], []
        for t in time_starts:
            for d in depth_starts:
                for i in row_starts:
                    for j in col_starts:
                        k1_list.append(i)
                        k2_list.append(j)
                        k3_list.append(d)
                        k4_list.append(t)

        k1_coords = torch.tensor(k1_list, dtype=torch.float32, device=self.device)
        k2_coords = torch.tensor(k2_list, dtype=torch.float32, device=self.device)
        k3_coords = torch.tensor(k3_list, dtype=torch.float32, device=self.device)
        k4_coords = torch.tensor(k4_list, dtype=torch.float32, device=self.device)

        k1_expanded = k1_coords.unsqueeze(0).expand(batch_size, -1)
        k2_expanded = k2_coords.unsqueeze(0).expand(batch_size, -1)
        k3_expanded = k3_coords.unsqueeze(0).expand(batch_size, -1)
        k4_expanded = k4_coords.unsqueeze(0).expand(batch_size, -1)

        # Compute Nk from agg
        agg_flat2 = agg.reshape(batch_size * n_windows, m)
        k1_flat = k1_expanded.reshape(batch_size * n_windows)
        k2_flat = k2_expanded.reshape(batch_size * n_windows)
        k3_flat = k3_expanded.reshape(batch_size * n_windows)
        k4_flat = k4_expanded.reshape(batch_size * n_windows)
        Nk_flat = self.batch_compute_Nk(k1_flat, k2_flat, k3_flat, k4_flat, agg_flat2)
        Nk_all = Nk_flat.reshape(batch_size, n_windows, m)

        return Nk_all, targets

    # --------------------------------------------------------------------------
    # Core methods (describe, S, D, d)
    # --------------------------------------------------------------------------
    def batch_compute_Nk(self, k1_tensor, k2_tensor, k3_tensor, k4_tensor, vectors):
        """
        Vectorized computation of N(k1,k2,k3,k4) vectors for a batch of positions and vectors.
        Optimized using einsum for better performance.

        Args:
            k1_tensor: Tensor of row indices [batch_size]
            k2_tensor: Tensor of column indices [batch_size]
            k3_tensor: Tensor of depth indices [batch_size]
            k4_tensor: Tensor of time indices [batch_size]
            vectors: Tensor of vectors [batch_size, vec_dim]

        Returns:
            Tensor of N(k1,k2,k3,k4) vectors [batch_size, m]
        """
        # Apply mapping matrix M to each vector
        x = self.M(vectors)  # [batch_size, m]

        # Expand dimensions for broadcasting
        k1_exp = k1_tensor.view(-1, 1, 1, 1)  # [batch, 1, 1, 1]
        k2_exp = k2_tensor.view(-1, 1, 1, 1)
        k3_exp = k3_tensor.view(-1, 1, 1, 1)
        k4_exp = k4_tensor.view(-1, 1, 1, 1)

        # Calculate basis functions for four dimensions
        phi1 = torch.cos(2 * math.pi * k1_exp / self.periods1)  # [batch, m, m, o]
        phi2 = torch.cos(2 * math.pi * k2_exp / self.periods2)
        phi3 = torch.cos(2 * math.pi * k3_exp / self.periods3)
        phi4 = torch.cos(2 * math.pi * k4_exp / self.periods4)
        phi = phi1 * phi2 * phi3 * phi4

        # Optimized computation using einsum
        Nk = torch.einsum('bj,ijg,bijg->bi', x, self.P, phi)
        return Nk

    def extract_vectors(self, array):
        """
        Extract window vectors from 4D array based on processing mode and rank operation.

        - 'linear': Slide window by step 1 in all directions, extracting contiguous windows of size rank_h x rank_w x rank_d x rank_t
        - 'nonlinear': Slide window by custom step (step_h, step_w, step_d, step_t) in all directions

        For nonlinear mode, handles incomplete trailing windows using:
        - 'pad': Pads with zero vectors to maintain full window size
        - 'drop': Discards incomplete windows

        Args:
            array (torch.Tensor): 4D array of shape [H, W, D, T, vec_dim] (or convertible)

        Returns:
            torch.Tensor: Tensor of shape [num_windows, vec_dim] after applying rank operation to each window
        """
        # Convert to tensor if needed
        if not isinstance(array, torch.Tensor):
            array = torch.tensor(array, dtype=torch.float32, device=self.device)
        if array.device != self.device:
            array = array.to(self.device)

        H, W, D, T, _ = array.shape

        def apply_op(win_tensor):
            """Apply rank operation (avg/sum/pick/user_func) to a window of shape [rank_h, rank_w, rank_d, rank_t, vec_dim]"""
            if self.rank_op == 'sum':
                return torch.sum(win_tensor, dim=(0, 1, 2, 3))
            elif self.rank_op == 'pick':
                # Flatten spatial dimensions and pick a random cell
                flat = win_tensor.view(-1, self.vec_dim)
                idx = random.randint(0, flat.shape[0]-1)
                return flat[idx]
            elif self.rank_op == 'user_func':
                if self.user_func is not None and callable(self.user_func):
                    return self.user_func(win_tensor)
                else:
                    # Default: average + sigmoid
                    avg = torch.mean(win_tensor, dim=(0, 1, 2, 3))
                    return torch.sigmoid(avg)
            else:  # 'avg' is default
                return torch.mean(win_tensor, dim=(0, 1, 2, 3))

        # Determine step sizes
        if self.mode == 'linear':
            step_h, step_w, step_d, step_t = 1, 1, 1, 1
        else:
            step_h = self.step_h if self.step_h is not None else self.rank_h
            step_w = self.step_w if self.step_w is not None else self.rank_w
            step_d = self.step_d if self.step_d is not None else self.rank_d
            step_t = self.step_t if self.step_t is not None else self.rank_t

        windows = []
        # Slide over height
        for i in range(0, H, step_h):
            if i + self.rank_h > H:
                if self.rank_mode == 'pad':
                    frag_h = array[i:H, :, :, :, :]
                    pad_h = self.rank_h - frag_h.shape[0]
                    padding = torch.zeros(pad_h, frag_h.shape[1], frag_h.shape[2], frag_h.shape[3], self.vec_dim, device=self.device)
                    frag_h = torch.cat([frag_h, padding], dim=0)
                else:
                    continue
            else:
                frag_h = array[i:i+self.rank_h, :, :, :, :]

            # Slide over width
            for j in range(0, W, step_w):
                if j + self.rank_w > W:
                    if self.rank_mode == 'pad':
                        frag_w = frag_h[:, j:W, :, :, :]
                        pad_w = self.rank_w - frag_w.shape[1]
                        padding = torch.zeros(frag_w.shape[0], pad_w, frag_w.shape[2], frag_w.shape[3], self.vec_dim, device=self.device)
                        frag_w = torch.cat([frag_w, padding], dim=1)
                    else:
                        continue
                else:
                    frag_w = frag_h[:, j:j+self.rank_w, :, :, :]

                # Slide over depth
                for k in range(0, D, step_d):
                    if k + self.rank_d > D:
                        if self.rank_mode == 'pad':
                            frag_d = frag_w[:, :, k:D, :, :]
                            pad_d = self.rank_d - frag_d.shape[2]
                            padding = torch.zeros(frag_d.shape[0], frag_d.shape[1], pad_d, frag_d.shape[3], self.vec_dim, device=self.device)
                            frag_d = torch.cat([frag_d, padding], dim=2)
                        else:
                            continue
                    else:
                        frag_d = frag_w[:, :, k:k+self.rank_d, :, :]

                    # Slide over time (4th dimension)
                    for t in range(0, T, step_t):
                        if t + self.rank_t > T:
                            if self.rank_mode == 'pad':
                                frag_t = frag_d[:, :, :, t:T, :]
                                pad_t = self.rank_t - frag_t.shape[3]
                                padding = torch.zeros(frag_t.shape[0], frag_t.shape[1], frag_t.shape[2], pad_t, self.vec_dim, device=self.device)
                                frag_t = torch.cat([frag_t, padding], dim=3)
                            else:
                                continue
                        else:
                            frag_t = frag_d[:, :, :, t:t+self.rank_t, :]
                        # frag_t shape: [rank_h, rank_w, rank_d, rank_t, m]
                        windows.append(apply_op(frag_t))

        if windows:
            return torch.stack(windows)  # [num_windows, m]
        else:
            return torch.empty(0, self.vec_dim, device=self.device)

    def describe(self, array):
        """
        Compute N(k1,k2,k3,k4) vectors for each window in the 4D array.

        Args:
            array: 4D tensor of shape [H, W, D, T, m]

        Returns:
            list of numpy arrays, each of shape [m]
        """
        extracted_vectors = self.extract_vectors(array)
        if extracted_vectors.shape[0] == 0:
            return []

        # Generate positions for each window (row-major order)
        H, W, D, T, _ = array.shape
        step_h = 1 if self.mode == 'linear' else (self.step_h if self.step_h is not None else self.rank_h)
        step_w = 1 if self.mode == 'linear' else (self.step_w if self.step_w is not None else self.rank_w)
        step_d = 1 if self.mode == 'linear' else (self.step_d if self.step_d is not None else self.rank_d)
        step_t = 1 if self.mode == 'linear' else (self.step_t if self.step_t is not None else self.rank_t)

        positions = []
        for i in range(0, H, step_h):
            if i + self.rank_h > H and self.rank_mode == 'drop':
                continue
            for j in range(0, W, step_w):
                if j + self.rank_w > W and self.rank_mode == 'drop':
                    continue
                for k in range(0, D, step_d):
                    if k + self.rank_d > D and self.rank_mode == 'drop':
                        continue
                    for t in range(0, T, step_t):
                        if t + self.rank_t > T and self.rank_mode == 'drop':
                            continue
                        positions.append((i, j, k, t))

        # Convert to tensors
        k1_tensor = torch.tensor([p[0] for p in positions], dtype=torch.float32, device=self.device)
        k2_tensor = torch.tensor([p[1] for p in positions], dtype=torch.float32, device=self.device)
        k3_tensor = torch.tensor([p[2] for p in positions], dtype=torch.float32, device=self.device)
        k4_tensor = torch.tensor([p[3] for p in positions], dtype=torch.float32, device=self.device)

        N_batch = self.batch_compute_Nk(k1_tensor, k2_tensor, k3_tensor, k4_tensor, extracted_vectors)
        return N_batch.detach().cpu().numpy()

    def S(self, array):
        """
        Compute cumulative sums of N(k1,k2,k3,k4) vectors along window order (row-major).
        Returns list of S(l) = sum_{t=1..l} N(k_t) for l=1..L, where L is number of windows.
        """
        N_list = self.describe(array)
        if not N_list:
            return []
        N_tensor = torch.tensor(N_list, device=self.device)
        S_cum = torch.cumsum(N_tensor, dim=0)
        return [s.cpu().numpy() for s in S_cum]

    def D(self, arrays, t_list):
        """
        Compute mean squared deviation D across 4D arrays:
        D = average over all windows of (N(k1,k2,k3,k4)-t)^2
        """
        total_loss = 0.0
        total_windows = 0

        t_tensors = [torch.tensor(t, dtype=torch.float32, device=self.device) for t in t_list]

        for array, t in zip(arrays, t_tensors):
            extracted_vectors = self.extract_vectors(array)
            if extracted_vectors.shape[0] == 0:
                continue

            # Generate positions (same as in describe)
            H, W, D, T, _ = array.shape
            step_h = 1 if self.mode == 'linear' else (self.step_h if self.step_h is not None else self.rank_h)
            step_w = 1 if self.mode == 'linear' else (self.step_w if self.step_w is not None else self.rank_w)
            step_d = 1 if self.mode == 'linear' else (self.step_d if self.step_d is not None else self.rank_d)
            step_t = 1 if self.mode == 'linear' else (self.step_t if self.step_t is not None else self.rank_t)
            positions = []
            for i in range(0, H, step_h):
                if i + self.rank_h > H and self.rank_mode == 'drop':
                    continue
                for j in range(0, W, step_w):
                    if j + self.rank_w > W and self.rank_mode == 'drop':
                        continue
                    for k in range(0, D, step_d):
                        if k + self.rank_d > D and self.rank_mode == 'drop':
                            continue
                        for t in range(0, T, step_t):
                            if t + self.rank_t > T and self.rank_mode == 'drop':
                                continue
                            positions.append((i, j, k, t))
            k1_tensor = torch.tensor([p[0] for p in positions], dtype=torch.float32, device=self.device)
            k2_tensor = torch.tensor([p[1] for p in positions], dtype=torch.float32, device=self.device)
            k3_tensor = torch.tensor([p[2] for p in positions], dtype=torch.float32, device=self.device)
            k4_tensor = torch.tensor([p[3] for p in positions], dtype=torch.float32, device=self.device)

            N_batch = self.batch_compute_Nk(k1_tensor, k2_tensor, k3_tensor, k4_tensor, extracted_vectors)
            losses = torch.sum((N_batch - t) ** 2, dim=1)
            total_loss += losses.sum().item()
            total_windows += extracted_vectors.shape[0]

        return total_loss / total_windows if total_windows else 0.0

    def d(self, array, t):
        """Compute pattern deviation value (d) for a single 4D array."""
        return self.D([array], [t])

    # --------------------------------------------------------------------------
    # Training methods with batch acceleration
    # --------------------------------------------------------------------------
    def reg_train(self, arrays, t_list, max_iters=1000, tol=1e-8, learning_rate=0.01,
                  continued=False, decay_rate=1.0, print_every=10, batch_size=32,
                  checkpoint_file=None, checkpoint_interval=10):
        """
        Train the model using gradient descent for regression on 4D arrays.
        Optimized for GPU memory efficiency by processing grids individually, but
        uses fast batch processing when all grids in a batch have equal size
        and rank_mode == 'drop'.
        """
        if not continued:
            self.reset_parameters()

        t_tensors = [torch.tensor(t, dtype=torch.float32, device=self.device) for t in t_list]

        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)

        history = []
        prev_loss = float('inf')
        best_loss = float('inf')
        best_model_state = None

        use_fast_batch = (self.rank_mode == 'drop')

        for it in range(max_iters):
            total_loss = 0.0
            total_arrays = 0

            indices = list(range(len(arrays)))
            random.shuffle(indices)

            for batch_start in range(0, len(indices), batch_size):
                batch_indices = indices[batch_start:batch_start + batch_size]
                batch_arrays = [arrays[idx] for idx in batch_indices]
                batch_targets = [t_tensors[idx] for idx in batch_indices]

                optimizer.zero_grad()
                batch_loss = 0.0

                # Check if all grids in this batch have the same shape
                shapes = [arr.shape if isinstance(arr, torch.Tensor) else (len(arr), len(arr[0]), len(arr[0][0]), len(arr[0][0][0]), self.vec_dim) for arr in batch_arrays]
                if use_fast_batch and len(set(shapes)) == 1:
                    # All grids have equal dimensions -> use fast batch processing
                    batch_grids_tensor = torch.stack([arr.to(self.device) if isinstance(arr, torch.Tensor) else torch.tensor(arr, dtype=torch.float32, device=self.device) for arr in batch_arrays], dim=0)
                    reps = self.batch_represent(batch_grids_tensor)  # [batch, m]
                    targets_tensor = torch.stack(batch_targets, dim=0)  # [batch, m]
                    loss = torch.mean(torch.sum((reps - targets_tensor) ** 2, dim=1))
                    loss.backward()
                    optimizer.step()
                    batch_loss = loss.item() * len(batch_arrays)
                    total_loss += batch_loss
                    total_arrays += len(batch_arrays)
                else:
                    # Process each grid individually
                    for arr, target in zip(batch_arrays, batch_targets):
                        extracted_vectors = self.extract_vectors(arr)
                        if extracted_vectors.shape[0] == 0:
                            continue

                        # Compute positions
                        H, W, D, T, _ = arr.shape
                        step_h = 1 if self.mode == 'linear' else (self.step_h if self.step_h is not None else self.rank_h)
                        step_w = 1 if self.mode == 'linear' else (self.step_w if self.step_w is not None else self.rank_w)
                        step_d = 1 if self.mode == 'linear' else (self.step_d if self.step_d is not None else self.rank_d)
                        step_t = 1 if self.mode == 'linear' else (self.step_t if self.step_t is not None else self.rank_t)
                        positions = []
                        for i in range(0, H, step_h):
                            if i + self.rank_h > H and self.rank_mode == 'drop':
                                continue
                            for j in range(0, W, step_w):
                                if j + self.rank_w > W and self.rank_mode == 'drop':
                                    continue
                                for k in range(0, D, step_d):
                                    if k + self.rank_d > D and self.rank_mode == 'drop':
                                        continue
                                    for t in range(0, T, step_t):
                                        if t + self.rank_t > T and self.rank_mode == 'drop':
                                            continue
                                        positions.append((i, j, k, t))
                        k1_tensor = torch.tensor([p[0] for p in positions], dtype=torch.float32, device=self.device)
                        k2_tensor = torch.tensor([p[1] for p in positions], dtype=torch.float32, device=self.device)
                        k3_tensor = torch.tensor([p[2] for p in positions], dtype=torch.float32, device=self.device)
                        k4_tensor = torch.tensor([p[3] for p in positions], dtype=torch.float32, device=self.device)

                        N_batch = self.batch_compute_Nk(k1_tensor, k2_tensor, k3_tensor, k4_tensor, extracted_vectors)
                        seq_pred = torch.mean(N_batch, dim=0)
                        seq_loss = torch.sum((seq_pred - target) ** 2)
                        batch_loss += seq_loss

                        del N_batch, seq_pred, extracted_vectors, k1_tensor, k2_tensor, k3_tensor, k4_tensor

                    if len(batch_arrays) > 0:
                        batch_loss = batch_loss / len(batch_arrays)
                        batch_loss.backward()
                        optimizer.step()
                        total_loss += batch_loss.item() * len(batch_arrays)
                        total_arrays += len(batch_arrays)

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            avg_loss = total_loss / total_arrays if total_arrays > 0 else 0.0
            history.append(avg_loss)

            if avg_loss < best_loss:
                best_loss = avg_loss
                best_model_state = copy.deepcopy(self.state_dict())

            if it % print_every == 0 or it == max_iters - 1:
                current_lr = scheduler.get_last_lr()[0]
                print(f"GD Iter {it:3d}: Loss = {avg_loss:.6e}, LR = {current_lr:.6f}")

            if checkpoint_file and (it % checkpoint_interval == 0 or it == max_iters - 1):
                self._save_checkpoint(checkpoint_file, it, history, optimizer, scheduler, best_loss)

            if abs(prev_loss - avg_loss) < tol:
                print(f"Converged after {it+1} iterations.")
                if best_model_state is not None:
                    self.load_state_dict(best_model_state)
                break
            prev_loss = avg_loss
            scheduler.step()

        self._compute_training_statistics(arrays)
        self.trained = True
        return history

    def cls_train(self, arrays, labels, num_classes, max_iters=1000, tol=1e-8, learning_rate=0.01,
                  continued=False, decay_rate=1.0, print_every=10, batch_size=32,
                  checkpoint_file=None, checkpoint_interval=10):
        """Train for multi-class classification on 4D arrays with batch acceleration."""
        if self.classifier is None or self.num_classes != num_classes:
            self.classifier = nn.Linear(self.m, num_classes).to(self.device)
            self.num_classes = num_classes
        if not continued:
            self.reset_parameters()

        label_tensors = torch.tensor(labels, dtype=torch.long, device=self.device)
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)
        criterion = nn.CrossEntropyLoss()

        history = []
        prev_loss = float('inf')
        best_loss = float('inf')
        best_model_state = None

        use_fast_batch = (self.rank_mode == 'drop')

        for it in range(max_iters):
            total_loss = 0.0
            total_arrays = 0
            correct = 0

            indices = list(range(len(arrays)))
            random.shuffle(indices)

            for batch_start in range(0, len(indices), batch_size):
                batch_indices = indices[batch_start:batch_start + batch_size]
                batch_arrays = [arrays[idx] for idx in batch_indices]
                batch_labels = label_tensors[batch_indices]

                optimizer.zero_grad()

                shapes = [arr.shape if isinstance(arr, torch.Tensor) else (len(arr), len(arr[0]), len(arr[0][0]), len(arr[0][0][0]), self.vec_dim) for arr in batch_arrays]
                if use_fast_batch and len(set(shapes)) == 1:
                    # Fast batch path
                    batch_grids_tensor = torch.stack([arr.to(self.device) if isinstance(arr, torch.Tensor) else torch.tensor(arr, dtype=torch.float32, device=self.device) for arr in batch_arrays], dim=0)
                    reps = self.batch_represent(batch_grids_tensor)  # [batch, m]
                    logits = self.classifier(reps)
                    loss = criterion(logits, batch_labels)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item() * len(batch_arrays)
                    total_arrays += len(batch_arrays)
                    with torch.no_grad():
                        pred = torch.argmax(logits, dim=1)
                        correct += (pred == batch_labels).sum().item()
                else:
                    # Individual processing
                    batch_loss = 0.0
                    batch_logits = []
                    for arr in batch_arrays:
                        extracted_vectors = self.extract_vectors(arr)
                        if extracted_vectors.shape[0] == 0:
                            seq_vector = torch.zeros(self.m, device=self.device)
                        else:
                            H, W, D, T, _ = arr.shape
                            step_h = 1 if self.mode == 'linear' else (self.step_h if self.step_h is not None else self.rank_h)
                            step_w = 1 if self.mode == 'linear' else (self.step_w if self.step_w is not None else self.rank_w)
                            step_d = 1 if self.mode == 'linear' else (self.step_d if self.step_d is not None else self.rank_d)
                            step_t = 1 if self.mode == 'linear' else (self.step_t if self.step_t is not None else self.rank_t)
                            positions = []
                            for i in range(0, H, step_h):
                                if i + self.rank_h > H and self.rank_mode == 'drop':
                                    continue
                                for j in range(0, W, step_w):
                                    if j + self.rank_w > W and self.rank_mode == 'drop':
                                        continue
                                    for k in range(0, D, step_d):
                                        if k + self.rank_d > D and self.rank_mode == 'drop':
                                            continue
                                        for t in range(0, T, step_t):
                                            if t + self.rank_t > T and self.rank_mode == 'drop':
                                                continue
                                            positions.append((i, j, k, t))
                            k1_tensor = torch.tensor([p[0] for p in positions], dtype=torch.float32, device=self.device)
                            k2_tensor = torch.tensor([p[1] for p in positions], dtype=torch.float32, device=self.device)
                            k3_tensor = torch.tensor([p[2] for p in positions], dtype=torch.float32, device=self.device)
                            k4_tensor = torch.tensor([p[3] for p in positions], dtype=torch.float32, device=self.device)
                            N_batch = self.batch_compute_Nk(k1_tensor, k2_tensor, k3_tensor, k4_tensor, extracted_vectors)
                            seq_vector = torch.mean(N_batch, dim=0)
                            del N_batch, extracted_vectors, k1_tensor, k2_tensor, k3_tensor, k4_tensor
                        logits = self.classifier(seq_vector.unsqueeze(0))
                        batch_logits.append(logits)

                    if batch_logits:
                        all_logits = torch.cat(batch_logits, dim=0)
                        loss = criterion(all_logits, batch_labels)
                        loss.backward()
                        optimizer.step()
                        total_loss += loss.item() * len(batch_arrays)
                        total_arrays += len(batch_arrays)
                        with torch.no_grad():
                            preds = torch.argmax(all_logits, dim=1)
                            correct += (preds == batch_labels).sum().item()

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            avg_loss = total_loss / total_arrays if total_arrays > 0 else 0.0
            accuracy = correct / total_arrays if total_arrays > 0 else 0.0
            history.append(avg_loss)

            if avg_loss < best_loss:
                best_loss = avg_loss
                best_model_state = copy.deepcopy(self.state_dict())

            if it % print_every == 0 or it == max_iters - 1:
                current_lr = scheduler.get_last_lr()[0]
                print(f"CLS-Train Iter {it:3d}: Loss = {avg_loss:.6e}, Acc = {accuracy:.4f}, LR = {current_lr:.6f}")

            if checkpoint_file and (it % checkpoint_interval == 0 or it == max_iters - 1):
                checkpoint = {
                    'iteration': it,
                    'model_state_dict': self.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'history': history,
                    'best_loss': best_loss,
                    'num_classes': self.num_classes
                }
                torch.save(checkpoint, checkpoint_file)
                print(f"Checkpoint saved at iteration {it}")

            if abs(prev_loss - avg_loss) < tol:
                print(f"Converged after {it+1} iterations.")
                if best_model_state is not None:
                    self.load_state_dict(best_model_state)
                break
            prev_loss = avg_loss
            scheduler.step()

        self.trained = True
        return history

    def lbl_train(self, arrays, labels, num_labels, max_iters=1000, tol=1e-8, learning_rate=0.01,
                  continued=False, decay_rate=1.0, print_every=10, batch_size=32,
                  checkpoint_file=None, checkpoint_interval=10, pos_weight=None):
        """Train for multi-label classification on 4D arrays with batch acceleration."""
        if self.labeller is None or self.num_labels != num_labels:
            self.labeller = nn.Linear(self.m, num_labels).to(self.device)
            self.num_labels = num_labels
        if not continued:
            self.reset_parameters()

        if isinstance(labels, list):
            labels_tensor = torch.tensor(labels, dtype=torch.float32, device=self.device)
        else:
            labels_tensor = torch.as_tensor(labels, dtype=torch.float32, device=self.device)

        if pos_weight is not None:
            pos_weight_tensor = torch.tensor(pos_weight, dtype=torch.float32, device=self.device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        else:
            criterion = nn.BCEWithLogitsLoss()

        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)

        loss_history = []
        acc_history = []
        prev_loss = float('inf')
        best_loss = float('inf')
        best_model_state = None

        use_fast_batch = (self.rank_mode == 'drop')

        for it in range(max_iters):
            total_loss = 0.0
            total_correct = 0
            total_predictions = 0
            total_arrays = 0

            indices = list(range(len(arrays)))
            random.shuffle(indices)

            for batch_start in range(0, len(indices), batch_size):
                batch_indices = indices[batch_start:batch_start + batch_size]
                batch_arrays = [arrays[idx] for idx in batch_indices]
                batch_labels = labels_tensor[batch_indices]

                optimizer.zero_grad()

                shapes = [arr.shape if isinstance(arr, torch.Tensor) else (len(arr), len(arr[0]), len(arr[0][0]), len(arr[0][0][0]), self.vec_dim) for arr in batch_arrays]
                if use_fast_batch and len(set(shapes)) == 1:
                    # Fast batch path
                    batch_grids_tensor = torch.stack([arr.to(self.device) if isinstance(arr, torch.Tensor) else torch.tensor(arr, dtype=torch.float32, device=self.device) for arr in batch_arrays], dim=0)
                    reps = self.batch_represent(batch_grids_tensor)  # [batch, m]
                    logits = self.labeller(reps)
                    loss = criterion(logits, batch_labels)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item() * len(batch_arrays)
                    total_arrays += len(batch_arrays)
                    with torch.no_grad():
                        probs = torch.sigmoid(logits)
                        preds = (probs > 0.5).float()
                        total_correct += (preds == batch_labels).sum().item()
                        total_predictions += batch_labels.numel()
                else:
                    # Individual processing
                    batch_predictions_list = []
                    for arr in batch_arrays:
                        extracted_vectors = self.extract_vectors(arr)
                        if extracted_vectors.shape[0] == 0:
                            continue
                        H, W, D, T, _ = arr.shape
                        step_h = 1 if self.mode == 'linear' else (self.step_h if self.step_h is not None else self.rank_h)
                        step_w = 1 if self.mode == 'linear' else (self.step_w if self.step_w is not None else self.rank_w)
                        step_d = 1 if self.mode == 'linear' else (self.step_d if self.step_d is not None else self.rank_d)
                        step_t = 1 if self.mode == 'linear' else (self.step_t if self.step_t is not None else self.rank_t)
                        positions = []
                        for i in range(0, H, step_h):
                            if i + self.rank_h > H and self.rank_mode == 'drop':
                                continue
                            for j in range(0, W, step_w):
                                if j + self.rank_w > W and self.rank_mode == 'drop':
                                    continue
                                for k in range(0, D, step_d):
                                    if k + self.rank_d > D and self.rank_mode == 'drop':
                                        continue
                                    for t in range(0, T, step_t):
                                        if t + self.rank_t > T and self.rank_mode == 'drop':
                                            continue
                                        positions.append((i, j, k, t))
                        k1_tensor = torch.tensor([p[0] for p in positions], dtype=torch.float32, device=self.device)
                        k2_tensor = torch.tensor([p[1] for p in positions], dtype=torch.float32, device=self.device)
                        k3_tensor = torch.tensor([p[2] for p in positions], dtype=torch.float32, device=self.device)
                        k4_tensor = torch.tensor([p[3] for p in positions], dtype=torch.float32, device=self.device)
                        N_batch = self.batch_compute_Nk(k1_tensor, k2_tensor, k3_tensor, k4_tensor, extracted_vectors)
                        seq_rep = torch.mean(N_batch, dim=0)
                        logits = self.labeller(seq_rep)
                        batch_predictions_list.append(logits)
                        del N_batch, seq_rep, extracted_vectors, k1_tensor, k2_tensor, k3_tensor, k4_tensor

                    if batch_predictions_list:
                        batch_logits = torch.stack(batch_predictions_list, dim=0)
                        loss = criterion(batch_logits, batch_labels)
                        loss.backward()
                        optimizer.step()
                        total_loss += loss.item() * len(batch_arrays)
                        total_arrays += len(batch_arrays)
                        with torch.no_grad():
                            probs = torch.sigmoid(batch_logits)
                            preds = (probs > 0.5).float()
                            total_correct += (preds == batch_labels).sum().item()
                            total_predictions += batch_labels.numel()

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            avg_loss = total_loss / total_arrays if total_arrays > 0 else 0.0
            avg_acc = total_correct / total_predictions if total_predictions > 0 else 0.0
            loss_history.append(avg_loss)
            acc_history.append(avg_acc)

            if avg_loss < best_loss:
                best_loss = avg_loss
                best_model_state = copy.deepcopy(self.state_dict())

            if it % print_every == 0 or it == max_iters - 1:
                current_lr = scheduler.get_last_lr()[0]
                print(f"MLC-Train Iter {it:3d}: Loss = {avg_loss:.6e}, Acc = {avg_acc:.4f}, LR = {current_lr:.6f}")

            if checkpoint_file and (it % checkpoint_interval == 0 or it == max_iters - 1):
                checkpoint = {
                    'iteration': it,
                    'model_state_dict': self.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss_history': loss_history,
                    'acc_history': acc_history,
                    'best_loss': best_loss
                }
                torch.save(checkpoint, checkpoint_file)
                print(f"Checkpoint saved at iteration {it}")

            if abs(prev_loss - avg_loss) < tol:
                print(f"Converged after {it+1} iterations.")
                if best_model_state is not None:
                    self.load_state_dict(best_model_state)
                break
            prev_loss = avg_loss
            scheduler.step()

        self.trained = True
        return loss_history, acc_history

    def self_train(self, arrays, max_iters=100, tol=1e-6, learning_rate=0.01,
                   continued=False, decay_rate=1.0, print_every=10,
                   batch_size=32, checkpoint_file=None, checkpoint_interval=5):
        """
        Self-training for self-consistency on 4D arrays with batch acceleration.
        Trains so that N(k1,k2,k3,k4) matches M(window_vector) at each window.
        """
        if not continued:
            self.reset_parameters()

        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)

        history = []
        prev_loss = float('inf')
        best_loss = float('inf')
        best_model_state = None

        use_fast_batch = (self.rank_mode == 'drop')

        for it in range(max_iters):
            total_loss = 0.0
            total_windows = 0

            indices = list(range(len(arrays)))
            random.shuffle(indices)

            for batch_start in range(0, len(indices), batch_size):
                batch_indices = indices[batch_start:batch_start + batch_size]
                batch_arrays = [arrays[idx] for idx in batch_indices]

                optimizer.zero_grad()

                shapes = [arr.shape if isinstance(arr, torch.Tensor) else (len(arr), len(arr[0]), len(arr[0][0]), len(arr[0][0][0]), self.vec_dim) for arr in batch_arrays]
                if use_fast_batch and len(set(shapes)) == 1:
                    # Fast batch path
                    batch_grids_tensor = torch.stack([arr.to(self.device) if isinstance(arr, torch.Tensor) else torch.tensor(arr, dtype=torch.float32, device=self.device) for arr in batch_arrays], dim=0)
                    Nk_all, targets = self.batch_compute_Nk_and_targets(batch_grids_tensor)
                    loss = torch.mean(torch.sum((Nk_all - targets) ** 2, dim=-1))
                    loss.backward()
                    optimizer.step()
                    batch_loss = loss.item() * batch_grids_tensor.shape[0] * Nk_all.shape[1]
                    total_loss += batch_loss
                    total_windows += batch_grids_tensor.shape[0] * Nk_all.shape[1]
                else:
                    # Individual processing
                    batch_loss = 0.0
                    batch_count = 0
                    for arr in batch_arrays:
                        extracted_vectors = self.extract_vectors(arr)
                        if extracted_vectors.shape[0] == 0:
                            continue
                        H, W, D, T, _ = arr.shape
                        step_h = 1 if self.mode == 'linear' else (self.step_h if self.step_h is not None else self.rank_h)
                        step_w = 1 if self.mode == 'linear' else (self.step_w if self.step_w is not None else self.rank_w)
                        step_d = 1 if self.mode == 'linear' else (self.step_d if self.step_d is not None else self.rank_d)
                        step_t = 1 if self.mode == 'linear' else (self.step_t if self.step_t is not None else self.rank_t)
                        positions = []
                        for i in range(0, H, step_h):
                            if i + self.rank_h > H and self.rank_mode == 'drop':
                                continue
                            for j in range(0, W, step_w):
                                if j + self.rank_w > W and self.rank_mode == 'drop':
                                    continue
                                for k in range(0, D, step_d):
                                    if k + self.rank_d > D and self.rank_mode == 'drop':
                                        continue
                                    for t in range(0, T, step_t):
                                        if t + self.rank_t > T and self.rank_mode == 'drop':
                                            continue
                                        positions.append((i, j, k, t))
                        k1_tensor = torch.tensor([p[0] for p in positions], dtype=torch.float32, device=self.device)
                        k2_tensor = torch.tensor([p[1] for p in positions], dtype=torch.float32, device=self.device)
                        k3_tensor = torch.tensor([p[2] for p in positions], dtype=torch.float32, device=self.device)
                        k4_tensor = torch.tensor([p[3] for p in positions], dtype=torch.float32, device=self.device)
                        N_batch = self.batch_compute_Nk(k1_tensor, k2_tensor, k3_tensor, k4_tensor, extracted_vectors)
                        target_vectors = self.M(extracted_vectors)  # [num_windows, m]

                        loss = torch.mean(torch.sum((N_batch - target_vectors) ** 2, dim=1))
                        batch_loss += loss
                        batch_count += 1
                        del N_batch, target_vectors, extracted_vectors, k1_tensor, k2_tensor, k3_tensor, k4_tensor

                    if batch_count > 0:
                        batch_loss = batch_loss / batch_count
                        batch_loss.backward()
                        optimizer.step()
                        total_loss += batch_loss.item() * batch_count
                        total_windows += batch_count

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            avg_loss = total_loss / total_windows if total_windows > 0 else 0.0
            history.append(avg_loss)

            if avg_loss < best_loss:
                best_loss = avg_loss
                best_model_state = copy.deepcopy(self.state_dict())

            if it % print_every == 0 or it == max_iters - 1:
                current_lr = scheduler.get_last_lr()[0]
                print(f"Self-Train Iter {it:3d}: Loss = {avg_loss:.6f}, LR = {current_lr:.6f}")

            if checkpoint_file and (it % checkpoint_interval == 0 or it == max_iters - 1):
                self._save_checkpoint(checkpoint_file, it, history, optimizer, scheduler, best_loss)

            if abs(prev_loss - avg_loss) < tol:
                print(f"Converged after {it+1} iterations")
                if best_model_state is not None:
                    self.load_state_dict(best_model_state)
                break
            prev_loss = avg_loss
            scheduler.step()

        self._compute_training_statistics(arrays)
        self.trained = True
        return history

    def _compute_training_statistics(self, arrays, batch_size=10):
        """Compute mean target vector and average number of windows per array."""
        total_windows = 0
        total_t = torch.zeros(self.m, device=self.device)

        with torch.no_grad():
            for arr in arrays:
                extracted_vectors = self.extract_vectors(arr)
                if extracted_vectors.shape[0] == 0:
                    continue
                H, W, D, T, _ = arr.shape
                step_h = 1 if self.mode == 'linear' else (self.step_h if self.step_h is not None else self.rank_h)
                step_w = 1 if self.mode == 'linear' else (self.step_w if self.step_w is not None else self.rank_w)
                step_d = 1 if self.mode == 'linear' else (self.step_d if self.step_d is not None else self.rank_d)
                step_t = 1 if self.mode == 'linear' else (self.step_t if self.step_t is not None else self.rank_t)
                positions = []
                for i in range(0, H, step_h):
                    if i + self.rank_h > H and self.rank_mode == 'drop':
                        continue
                    for j in range(0, W, step_w):
                        if j + self.rank_w > W and self.rank_mode == 'drop':
                            continue
                        for k in range(0, D, step_d):
                            if k + self.rank_d > D and self.rank_mode == 'drop':
                                continue
                            for t in range(0, T, step_t):
                                if t + self.rank_t > T and self.rank_mode == 'drop':
                                    continue
                                positions.append((i, j, k, t))
                k1_tensor = torch.tensor([p[0] for p in positions], dtype=torch.float32, device=self.device)
                k2_tensor = torch.tensor([p[1] for p in positions], dtype=torch.float32, device=self.device)
                k3_tensor = torch.tensor([p[2] for p in positions], dtype=torch.float32, device=self.device)
                k4_tensor = torch.tensor([p[3] for p in positions], dtype=torch.float32, device=self.device)
                N_batch = self.batch_compute_Nk(k1_tensor, k2_tensor, k3_tensor, k4_tensor, extracted_vectors)
                total_t += N_batch.sum(dim=0)
                total_windows += extracted_vectors.shape[0]
                del N_batch, extracted_vectors, k1_tensor, k2_tensor, k3_tensor, k4_tensor

        self.mean_window_count = total_windows / len(arrays) if arrays else 0
        self.mean_t = (total_t / total_windows).cpu().numpy() if total_windows > 0 else np.zeros(self.m)

    def _save_checkpoint(self, checkpoint_file, iteration, history, optimizer, scheduler, best_loss):
        checkpoint = {
            'iteration': iteration,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'history': history,
            'best_loss': best_loss,
            'training_stats': {
                'mean_t': self.mean_t,
                'mean_window_count': self.mean_window_count
            }
        }
        torch.save(checkpoint, checkpoint_file)
        print(f"Checkpoint saved at iteration {iteration}")

    def predict_t(self, array):
        """Predict target vector for a 4D array (average of all N(k1,k2,k3,k4) vectors)."""
        extracted_vectors = self.extract_vectors(array)
        if extracted_vectors.shape[0] == 0:
            return [0.0] * self.m
        H, W, D, T, _ = array.shape
        step_h = 1 if self.mode == 'linear' else (self.step_h if self.step_h is not None else self.rank_h)
        step_w = 1 if self.mode == 'linear' else (self.step_w if self.step_w is not None else self.rank_w)
        step_d = 1 if self.mode == 'linear' else (self.step_d if self.step_d is not None else self.rank_d)
        step_t = 1 if self.mode == 'linear' else (self.step_t if self.step_t is not None else self.rank_t)
        positions = []
        for i in range(0, H, step_h):
            if i + self.rank_h > H and self.rank_mode == 'drop':
                continue
            for j in range(0, W, step_w):
                if j + self.rank_w > W and self.rank_mode == 'drop':
                    continue
                for k in range(0, D, step_d):
                    if k + self.rank_d > D and self.rank_mode == 'drop':
                        continue
                    for t in range(0, T, step_t):
                        if t + self.rank_t > T and self.rank_mode == 'drop':
                            continue
                        positions.append((i, j, k, t))
        k1_tensor = torch.tensor([p[0] for p in positions], dtype=torch.float32, device=self.device)
        k2_tensor = torch.tensor([p[1] for p in positions], dtype=torch.float32, device=self.device)
        k3_tensor = torch.tensor([p[2] for p in positions], dtype=torch.float32, device=self.device)
        k4_tensor = torch.tensor([p[3] for p in positions], dtype=torch.float32, device=self.device)
        N_batch = self.batch_compute_Nk(k1_tensor, k2_tensor, k3_tensor, k4_tensor, extracted_vectors)
        avg = torch.mean(N_batch, dim=0).detach().cpu().numpy()
        return avg

    def predict_c(self, array):
        """Predict class label for a 4D array using the classification head."""
        if self.classifier is None:
            raise ValueError("Model must be trained first for classification")
        seq_vector = self.predict_t(array)
        seq_tensor = torch.tensor(seq_vector, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits = self.classifier(seq_tensor.unsqueeze(0))
            probs = torch.softmax(logits, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
        return pred_class, probs[0].cpu().numpy()

    def predict_l(self, array, threshold=0.5):
        """Predict multi-label classification for a 4D array."""
        assert self.labeller is not None, "Model must be trained first for label prediction"
        extracted_vectors = self.extract_vectors(array)
        if extracted_vectors.shape[0] == 0:
            return np.zeros(self.num_labels, dtype=np.float32), np.zeros(self.num_labels, dtype=np.float32)
        H, W, D, T, _ = array.shape
        step_h = 1 if self.mode == 'linear' else (self.step_h if self.step_h is not None else self.rank_h)
        step_w = 1 if self.mode == 'linear' else (self.step_w if self.step_w is not None else self.rank_w)
        step_d = 1 if self.mode == 'linear' else (self.step_d if self.step_d is not None else self.rank_d)
        step_t = 1 if self.mode == 'linear' else (self.step_t if self.step_t is not None else self.rank_t)
        positions = []
        for i in range(0, H, step_h):
            if i + self.rank_h > H and self.rank_mode == 'drop':
                continue
            for j in range(0, W, step_w):
                if j + self.rank_w > W and self.rank_mode == 'drop':
                    continue
                for k in range(0, D, step_d):
                    if k + self.rank_d > D and self.rank_mode == 'drop':
                        continue
                    for t in range(0, T, step_t):
                        if t + self.rank_t > T and self.rank_mode == 'drop':
                            continue
                        positions.append((i, j, k, t))
        k1_tensor = torch.tensor([p[0] for p in positions], dtype=torch.float32, device=self.device)
        k2_tensor = torch.tensor([p[1] for p in positions], dtype=torch.float32, device=self.device)
        k3_tensor = torch.tensor([p[2] for p in positions], dtype=torch.float32, device=self.device)
        k4_tensor = torch.tensor([p[3] for p in positions], dtype=torch.float32, device=self.device)
        N_batch = self.batch_compute_Nk(k1_tensor, k2_tensor, k3_tensor, k4_tensor, extracted_vectors)
        seq_rep = torch.mean(N_batch, dim=0)
        with torch.no_grad():
            logits = self.labeller(seq_rep)
            probs = torch.sigmoid(logits).cpu().numpy()
        binary = (probs > threshold).astype(np.float32)
        return binary, probs

    def reconstruct(self, H, W, D, T, tau=0.0):
        """
        Reconstruct a 4D array of shape (H, W, D, T, m) by minimizing error with temperature.
        Simple version: generates random candidate vectors and selects best per window,
        then averages overlapping positions. This is a placeholder for demonstration.
        """
        assert self.trained, "Model must be trained first"
        if tau < 0:
            raise ValueError("Temperature must be non-negative")

        mean_t_tensor = torch.tensor(self.mean_t, dtype=torch.float32, device=self.device)
        # We'll generate windows and then combine them
        step_h = 1 if self.mode == 'linear' else (self.step_h if self.step_h is not None else self.rank_h)
        step_w = 1 if self.mode == 'linear' else (self.step_w if self.step_w is not None else self.rank_w)
        step_d = 1 if self.mode == 'linear' else (self.step_d if self.step_d is not None else self.rank_d)
        step_t = 1 if self.mode == 'linear' else (self.step_t if self.step_t is not None else self.rank_t)

        # Determine number of windows
        nh = (H - self.rank_h) // step_h + 1 if self.rank_mode == 'drop' else (H + step_h - 1) // step_h
        nw = (W - self.rank_w) // step_w + 1 if self.rank_mode == 'drop' else (W + step_w - 1) // step_w
        nd = (D - self.rank_d) // step_d + 1 if self.rank_mode == 'drop' else (D + step_d - 1) // step_d
        nt = (T - self.rank_t) // step_t + 1 if self.rank_mode == 'drop' else (T + step_t - 1) // step_t

        # Pre-generate candidate vectors
        num_candidates = 100
        candidates = torch.randn(num_candidates, self.vec_dim, device=self.device)

        # For each window position, choose a candidate vector
        chosen_vectors = {}  # (i,j,k,t) -> vector
        for i_idx in range(nh):
            i = i_idx * step_h
            for j_idx in range(nw):
                j = j_idx * step_w
                for k_idx in range(nd):
                    k = k_idx * step_d
                    for t_idx in range(nt):
                        t = t_idx * step_t
                        # Compute N for all candidates at this position
                        k1_tensor = torch.full((num_candidates,), i, dtype=torch.float32, device=self.device)
                        k2_tensor = torch.full((num_candidates,), j, dtype=torch.float32, device=self.device)
                        k3_tensor = torch.full((num_candidates,), k, dtype=torch.float32, device=self.device)
                        k4_tensor = torch.full((num_candidates,), t, dtype=torch.float32, device=self.device)
                        N_all = self.batch_compute_Nk(k1_tensor, k2_tensor, k3_tensor, k4_tensor, candidates)
                        errors = torch.sum((N_all - mean_t_tensor) ** 2, dim=1)
                        scores = -errors
                        if tau == 0:
                            idx = torch.argmax(scores).item()
                        else:
                            probs = torch.softmax(scores / tau, dim=0).detach().cpu().numpy()
                            idx = random.choices(range(num_candidates), weights=probs, k=1)[0]
                        chosen_vectors[(i, j, k, t)] = candidates[idx]

        # Build the output array by averaging overlapping windows
        result = torch.zeros(H, W, D, T, self.vec_dim, device=self.device)
        count = torch.zeros(H, W, D, T, device=self.device)
        for (i, j, k, t), vec in chosen_vectors.items():
            for di in range(self.rank_h):
                if i+di >= H:
                    continue
                for dj in range(self.rank_w):
                    if j+dj >= W:
                        continue
                    for dk in range(self.rank_d):
                        if k+dk >= D:
                            continue
                        for dt in range(self.rank_t):
                            if t+dt >= T:
                                continue
                            result[i+di, j+dj, k+dk, t+dt] += vec
                            count[i+di, j+dj, k+dk, t+dt] += 1
        # Avoid division by zero
        count = count.clamp(min=1)
        result = result / count.unsqueeze(-1)
        return result.detach().cpu().numpy()

    def save(self, filename):
        torch.save(self.state_dict(), filename)
        print(f"Model saved to {filename}")

    def load(self, filename):
        try:
            state_dict = torch.load(filename, map_location=self.device, weights_only=True)
        except TypeError:
            state_dict = torch.load(filename, map_location=self.device)
        self.load_state_dict(state_dict)
        print(f"Model loaded from {filename}")
        return self


# === Example Usage ===
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from scipy.stats import pearsonr

    print("="*50)
    print("Spatial Numerical Dual Descriptor TS4 - 4D Array Version with Batch Acceleration")
    print("Processes 4D grids of m-dimensional real vectors")
    print("="*50)

    torch.manual_seed(11)
    random.seed(11)
    np.random.seed(11)

    # Parameters
    vec_dim = 4                 # Reduced dimension for speed
    num_basis = 4
    rank = 2                    # cubic window 2x2x2x2
    user_step = 1               # step size for nonlinear mode (use 1 for better coverage)

    ndd = SpatialNumDualDescriptorTS4(
        vec_dim=vec_dim,
        rank=rank,
        num_basis=num_basis,
        mode='nonlinear',
        user_step=user_step,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    print(f"\nUsing device: {ndd.device}")
    print(f"Vector dimension: {vec_dim}")
    print(f"Window size: {rank}x{rank}x{rank}x{rank}, step: {user_step}")

    # Generate 20 random 4D arrays (H x W x D x T x vec_dim) with random target vectors
    print("\nGenerating training data...")
    arrays, t_list = [], []
    for _ in range(20):
        H = random.randint(8, 10)
        W = random.randint(8, 10)
        D = random.randint(8, 10)
        T = random.randint(8, 10)
        arr = torch.randn(H, W, D, T, vec_dim)
        arrays.append(arr)
        t_list.append(np.random.uniform(-1.0, 1.0, vec_dim))

    # Regression training
    print("\n" + "="*50)
    print("Starting Regression Training")
    print("="*50)
    ndd.reg_train(arrays, t_list, max_iters=15, tol=1e-9, learning_rate=0.05,
                  decay_rate=0.95, batch_size=4, print_every=5)

    # Predict target for first array
    arr0 = arrays[0]
    t_pred = ndd.predict_t(arr0)
    print(f"\nPredicted t for first array: {[round(x, 4) for x in t_pred[:4]]}...")

    # Calculate correlation
    print("\nCalculating prediction correlations...")
    pred_t_list = [ndd.predict_t(arr) for arr in arrays]
    corr_sum = 0.0
    for i in range(ndd.m):
        actu = [t[i] for t in t_list]
        pred = [pt[i] for pt in pred_t_list]
        corr, _ = pearsonr(actu, pred)
        print(f"Dimension {i} prediction correlation: {corr:.4f}")
        corr_sum += corr
    print(f"Average correlation: {corr_sum/ndd.m:.4f}")

    # Reconstruction (simplified)
    print("\nReconstructing a 4D array...")
    H_rec, W_rec, D_rec, T_rec = 8, 8, 8, 8
    rec_det = ndd.reconstruct(H_rec, W_rec, D_rec, T_rec, tau=0.0)
    rec_rand = ndd.reconstruct(H_rec, W_rec, D_rec, T_rec, tau=0.5)
    print(f"Deterministic reconstruction shape: {rec_det.shape}")
    print(f"Stochastic reconstruction shape: {rec_rand.shape}")

    # Classification task
    print("\n" + "="*50)
    print("Classification Task")
    print("="*50)
    num_classes = 3
    class_arrays = []
    class_labels = []
    for class_id in range(num_classes):
        for _ in range(12):
            H = random.randint(8, 10)
            W = random.randint(8, 10)
            D = random.randint(8, 10)
            T = random.randint(8, 10)
            if class_id == 0:
                arr = torch.randn(H, W, D, T, vec_dim) + 1.0
            elif class_id == 1:
                arr = torch.randn(H, W, D, T, vec_dim) - 1.0
            else:
                arr = torch.randn(H, W, D, T, vec_dim)
            class_arrays.append(arr)
            class_labels.append(class_id)

    ndd_cls = SpatialNumDualDescriptorTS4(
        vec_dim=vec_dim, rank=rank, num_basis=num_basis,
        mode='nonlinear', user_step=user_step, device=ndd.device
    )
    print("\nStarting Classification Training")
    ndd_cls.cls_train(class_arrays, class_labels, num_classes,
                      max_iters=12, tol=1e-8, learning_rate=0.05,
                      decay_rate=0.99, batch_size=4, print_every=3)

    # Evaluate classification
    correct = 0
    for arr, lbl in zip(class_arrays, class_labels):
        pred, _ = ndd_cls.predict_c(arr)
        if pred == lbl:
            correct += 1
    print(f"\nClassification accuracy: {correct/len(class_arrays):.4f}")

    # Multi-label classification
    print("\n" + "="*50)
    print("Multi-Label Classification")
    print("="*50)
    num_labels = 3
    label_arrays = []
    ml_labels = []
    for _ in range(20):
        H = random.randint(8, 10)
        W = random.randint(8, 10)
        D = random.randint(8, 10)
        T = random.randint(8, 10)
        arr = torch.randn(H, W, D, T, vec_dim)
        label_arrays.append(arr)
        label_vec = [random.random() > 0.7 for _ in range(num_labels)]
        ml_labels.append([1.0 if x else 0.0 for x in label_vec])

    ndd_lbl = SpatialNumDualDescriptorTS4(
        vec_dim=vec_dim, rank=rank, num_basis=num_basis,
        mode='nonlinear', user_step=user_step, device=ndd.device
    )
    print("\nStarting Multi-Label Training")
    loss_hist, acc_hist = ndd_lbl.lbl_train(
        label_arrays, ml_labels, num_labels,
        max_iters=12, tol=1e-8, learning_rate=0.05,
        decay_rate=0.99, batch_size=4, print_every=3
    )
    print(f"Final multi-label accuracy: {acc_hist[-1]:.4f}")

    # Self-training
    print("\n" + "="*50)
    print("Self-Training Example")
    print("="*50)
    ndd_self = SpatialNumDualDescriptorTS4(
        vec_dim=vec_dim, rank=rank, num_basis=num_basis,
        mode='nonlinear', user_step=user_step, device=ndd.device
    )
    self_arrays = [torch.randn(8, 8, 8, 8, vec_dim) for _ in range(6)]
    print("Training for self-consistency...")
    self_hist = ndd_self.self_train(self_arrays, max_iters=8, learning_rate=0.01, batch_size=2)

    plt.figure(figsize=(10,6))
    plt.plot(self_hist)
    plt.title('Self-Training Loss History (4D)')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig('self_training_loss_4d.png')
    print("\nSelf-training loss plot saved as 'self_training_loss_4d.png'")

    print("\nAll tests completed successfully!")
