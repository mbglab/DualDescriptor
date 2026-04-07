# Copyright (C) 2005-2026, Bin-Guang Ma (mbg@mail.hzau.edu.cn); SPDX-License-Identifier: MIT
# The Spatial Numerical Dual Descriptor Vector class (P Matrix form) for 3D array implemented with PyTorch
# This program is for the demonstration of methodology and not fully refined.
# Author: Bin-Guang Ma (assisted by DeepSeek); Date: 2025-8-28 ~ 2026-04-01

import math
import random
import itertools
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy

class SpatialNumDualDescriptorPM3(nn.Module):
    """
    Numerical Vector Dual Descriptor for 3D spatial volumes with GPU acceleration using PyTorch:
      - Processes 3D grids (depth × height × width) of m-dimensional real vectors
      - matrix P ∈ R^{m×m} of basis coefficients
      - square mapping matrix M ∈ R^{m×m} for vector transformation (assumes input_dim = model_dim)
      - indexed periods: period1[i,j] = i*m + j + 2, period2[i,j] = i*m + j + 2, period3[i,j] = i*m + j + 2
      - basis function phi_{i,j}(k1,k2,k3) = cos(2π*k1/period1[i,j]) * cos(2π*k2/period2[i,j]) * cos(2π*k3/period3[i,j])
      - supports 'linear' (step=1) or 'nonlinear' (step-by-rank) window extraction
      - batch acceleration for equal-sized volumes when rank_mode == 'drop'
    """
    def __init__(self, vec_dim, rank=1, rank_op='avg', rank_mode='drop', mode='linear', user_step=None, device='cuda'):
        super().__init__()
        self.vec_dim = vec_dim          # Dimension of input vectors and internal representation
        self.rank = rank                # Window size (cube side length)
        self.rank_op = rank_op          # 'avg', 'sum', 'pick', 'user_func'
        self.rank_mode = rank_mode      # 'pad' or 'drop' for incomplete windows
        assert mode in ('linear', 'nonlinear')
        self.mode = mode
        self.step = user_step            # Step size for nonlinear mode (default = rank)
        self.trained = False
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Mapping matrix M for vector transformation
        self.M = nn.Linear(self.vec_dim, self.vec_dim, bias=False)
        
        # Position-weight matrix P[i][j] (simplified 2D version, same for all dimensions)
        self.P = nn.Parameter(torch.empty(self.vec_dim, self.vec_dim))
        
        # Precompute indexed periods for three dimensions (fixed, not trainable)
        periods = torch.zeros(self.vec_dim, self.vec_dim, dtype=torch.float32)
        for i in range(self.vec_dim):
            for j in range(self.vec_dim):
                periods[i, j] = i * self.vec_dim + j + 2
        self.register_buffer('periods1', periods)   # period1 matrix
        self.register_buffer('periods2', periods)   # period2 matrix
        self.register_buffer('periods3', periods)   # period3 matrix

        # Precomputed phi table for all possible (k1,k2,k3) pairs (initially None, built when needed)
        self.register_buffer('phi_table', None)

        # Class head (initialized later when num_classes is known)
        self.num_classes = None
        self.classifier = None        

        # Label head (initialized later when num_labels is known)
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
        if self.num_labels is not None:
            nn.init.xavier_uniform_(self.labeller.weight)
            nn.init.zeros_(self.labeller.bias)
    
    def set_user_func(self, func):
        """Set custom user function for rank operation"""
        if callable(func):
            self.user_func = func
        else:
            raise ValueError("User function must be callable")

    def _precompute_phi_table(self, max_k1, max_k2, max_k3):
        """
        Precompute phi values for all k1 = 0..max_k1-1, k2 = 0..max_k2-1, k3 = 0..max_k3-1 and store as buffer.
        This significantly speeds up batch_compute_Nk by avoiding repeated trig calls.
        
        Args:
            max_k1: Number of possible k1 indices (largest index + 1)
            max_k2: Number of possible k2 indices (largest index + 1)
            max_k3: Number of possible k3 indices (largest index + 1)
        """
        if self.phi_table is not None and \
           self.phi_table.size(0) >= max_k1 and \
           self.phi_table.size(1) >= max_k2 and \
           self.phi_table.size(2) >= max_k3:
            return  # already have enough precomputed values
        # Allocate and compute: shape [max_k1, max_k2, max_k3, vec_dim, vec_dim]
        phi_table = torch.zeros(max_k1, max_k2, max_k3, self.vec_dim, self.vec_dim,
                                device=self.device, dtype=torch.float32)
        for k1 in range(max_k1):
            cos1 = torch.cos(2 * math.pi * k1 / self.periods1)  # [vec_dim, vec_dim]
            for k2 in range(max_k2):
                cos12 = cos1 * torch.cos(2 * math.pi * k2 / self.periods2)  # [vec_dim, vec_dim]
                for k3 in range(max_k3):
                    phi_table[k1, k2, k3] = cos12 * torch.cos(2 * math.pi * k3 / self.periods3)
        self.register_buffer('phi_table', phi_table)

    def _get_window_starts_3d(self, D, H, W):
        """
        Compute start indices of windows in 3D based on mode and rank_mode.
        Returns three lists: k1_starts, k2_starts, k3_starts.
        """
        if self.mode == 'linear':
            step = 1
        else:
            step = self.step if self.step is not None else self.rank
        
        if self.rank_mode == 'drop':
            k1_starts = list(range(0, D - self.rank + 1, step))
            k2_starts = list(range(0, H - self.rank + 1, step))
            k3_starts = list(range(0, W - self.rank + 1, step))
        else:  # 'pad'
            k1_starts = list(range(0, D, step))
            k2_starts = list(range(0, H, step))
            k3_starts = list(range(0, W, step))
        return k1_starts, k2_starts, k3_starts

    def _compute_Nk_from_x_3d(self, k1_tensor, k2_tensor, k3_tensor, x):
        """
        Compute N(k1,k2,k3) vectors directly from mapped vectors x (already transformed by M).
        This avoids double mapping in batch processing.
        
        Args:
            k1_tensor: Tensor of depth indices [batch_size] (long)
            k2_tensor: Tensor of height indices [batch_size] (long)
            k3_tensor: Tensor of width indices [batch_size] (long)
            x: Tensor of mapped vectors [batch_size, vec_dim]
        
        Returns:
            Nk: Tensor [batch_size, vec_dim]
        """
        # Use precomputed phi table if available
        if self.phi_table is not None:
            # phi: [batch_size, vec_dim, vec_dim]
            phi = self.phi_table[k1_tensor, k2_tensor, k3_tensor]
        else:
            # Fallback: compute on the fly (slower, but works)
            k1_exp = k1_tensor.view(-1, 1, 1).float()
            k2_exp = k2_tensor.view(-1, 1, 1).float()
            k3_exp = k3_tensor.view(-1, 1, 1).float()
            phi = (torch.cos(2 * math.pi * k1_exp / self.periods1) *
                   torch.cos(2 * math.pi * k2_exp / self.periods2) *
                   torch.cos(2 * math.pi * k3_exp / self.periods3))
        # Nk = sum_{i,j} P[i,j] * x_j * phi_{i,j}
        Nk = torch.einsum('bj,ij,bij->bi', x, self.P, phi)
        return Nk

    def extract_windows(self, volume):
        """
        Extract cubic windows from a 3D volume and apply rank operation to each window.
        
        Args:
            volume: torch.Tensor of shape (D, H, W, vec_dim) or nested list of vectors
            
        Returns:
            tuple: (window_vectors, window_coords)
                window_vectors: torch.Tensor of shape (num_windows, vec_dim)
                window_coords: tuple of three tensors (k1_coords, k2_coords, k3_coords) each of shape (num_windows,)
                coordinates are returned as torch.long for indexing purposes.
        """
        # Convert to tensor if needed
        if not isinstance(volume, torch.Tensor):
            # Assume volume is nested list: depth list of height lists of width lists of vectors
            D = len(volume)
            H = len(volume[0]) if D > 0 else 0
            W = len(volume[0][0]) if H > 0 else 0
            vol_tensor = torch.zeros(D, H, W, self.vec_dim, dtype=torch.float32, device=self.device)
            for d in range(D):
                for h in range(H):
                    for w in range(W):
                        vec = volume[d][h][w]
                        if not isinstance(vec, torch.Tensor):
                            vec = torch.tensor(vec, dtype=torch.float32)
                        vol_tensor[d, h, w] = vec.to(self.device)
        else:
            vol_tensor = volume.to(self.device)
        
        D, H, W = vol_tensor.shape[0], vol_tensor.shape[1], vol_tensor.shape[2]
        
        # Determine step size
        if self.mode == 'linear':
            step = 1
        else:  # nonlinear
            step = self.step if self.step is not None else self.rank
        
        # Collect all windows
        windows = []
        coords_k1 = []  # depth coordinate
        coords_k2 = []  # height coordinate
        coords_k3 = []  # width coordinate
        
        # Iterate over all possible window starting positions
        for k1 in range(0, D - self.rank + 1, step):
            for k2 in range(0, H - self.rank + 1, step):
                for k3 in range(0, W - self.rank + 1, step):
                    # Extract cube of shape (rank, rank, rank, vec_dim)
                    window = vol_tensor[k1:k1+self.rank, k2:k2+self.rank, k3:k3+self.rank, :]
                    # Flatten to (rank*rank*rank, vec_dim)
                    window_flat = window.reshape(-1, self.vec_dim)
                    # Apply rank operation
                    if self.rank_op == 'sum':
                        rep = torch.sum(window_flat, dim=0)
                    elif self.rank_op == 'pick':
                        idx = random.randint(0, window_flat.shape[0]-1)
                        rep = window_flat[idx]
                    elif self.rank_op == 'user_func' and self.user_func is not None:
                        rep = self.user_func(window_flat)
                    else:  # default 'avg'
                        rep = torch.mean(window_flat, dim=0)
                    windows.append(rep)
                    coords_k1.append(k1)
                    coords_k2.append(k2)
                    coords_k3.append(k3)
        
        # Handle incomplete windows if rank_mode == 'pad'
        if self.rank_mode == 'pad':
            # For each dimension, consider starting positions that would create incomplete windows
            # We need to handle cases where the window extends beyond the volume boundaries.
            # This implementation covers all possible windows (full or partial) by padding.
            # For simplicity, we iterate over all start positions that produce at least one element,
            # then pad to full rank x rank x rank.
            for k1 in range(0, D, step):
                for k2 in range(0, H, step):
                    for k3 in range(0, W, step):
                        # Skip already processed full windows (if step equals rank, this condition may skip some)
                        if (k1 + self.rank <= D and k2 + self.rank <= H and k3 + self.rank <= W):
                            continue
                        # Extract partial window
                        end1 = min(k1 + self.rank, D)
                        end2 = min(k2 + self.rank, H)
                        end3 = min(k3 + self.rank, W)
                        window = vol_tensor[k1:end1, k2:end2, k3:end3, :]  # [d_rem, h_rem, w_rem, vec_dim]
                        if window.numel() == 0:
                            continue
                        # Pad to full rank x rank x rank
                        d_rem, h_rem, w_rem = window.shape[0], window.shape[1], window.shape[2]
                        padded = torch.zeros(self.rank, self.rank, self.rank, self.vec_dim, device=self.device)
                        padded[:d_rem, :h_rem, :w_rem, :] = window
                        window_flat = padded.reshape(-1, self.vec_dim)
                        if self.rank_op == 'sum':
                            rep = torch.sum(window_flat, dim=0)
                        elif self.rank_op == 'pick':
                            idx = random.randint(0, window_flat.shape[0]-1)
                            rep = window_flat[idx]
                        elif self.rank_op == 'user_func' and self.user_func is not None:
                            rep = self.user_func(window_flat)
                        else:
                            rep = torch.mean(window_flat, dim=0)
                        windows.append(rep)
                        coords_k1.append(k1)
                        coords_k2.append(k2)
                        coords_k3.append(k3)
        
        if not windows:
            # No valid windows
            return torch.empty(0, self.vec_dim, device=self.device), (torch.empty(0), torch.empty(0), torch.empty(0))
        
        window_vectors = torch.stack(windows)  # [num_windows, vec_dim]
        coords_k1 = torch.tensor(coords_k1, dtype=torch.long, device=self.device)
        coords_k2 = torch.tensor(coords_k2, dtype=torch.long, device=self.device)
        coords_k3 = torch.tensor(coords_k3, dtype=torch.long, device=self.device)
        return window_vectors, (coords_k1, coords_k2, coords_k3)

    def batch_represent_3d(self, vol_batch):
        """
        Compute volume representations for a batch of 3D volumes efficiently.
        Supports arbitrary window size and step as long as all volumes in the batch have
        identical dimensions, and rank_mode == 'drop'.
        
        Args:
            vol_batch: Tensor of shape [batch_size, D, H, W, vec_dim]
        
        Returns:
            representations: Tensor of shape [batch_size, vec_dim]
        """
        batch_size, D, H, W, m = vol_batch.shape
        # Compute window start indices for each dimension
        k1_starts, k2_starts, k3_starts = self._get_window_starts_3d(D, H, W)
        if not k1_starts or not k2_starts or not k3_starts:
            return torch.zeros(batch_size, m, device=self.device)
        
        n_windows_d = len(k1_starts)
        n_windows_h = len(k2_starts)
        n_windows_w = len(k3_starts)
        n_windows = n_windows_d * n_windows_h * n_windows_w
        
        step = 1 if self.mode == 'linear' else (self.step if self.step is not None else self.rank)
        
        # Prepare to collect all window vectors
        # We'll iterate over depth windows, and use 2D unfold for height-width dimensions
        all_windows = []  # list of [batch, n_windows_2d, m] tensors
        all_k1 = []       # list of depth start indices (scaled)
        all_k2 = []       # list of height start indices (from 2D unfold)
        all_k3 = []       # list of width start indices
        
        # For each depth window start index
        for d_idx, k1 in enumerate(k1_starts):
            # Extract depth window: [batch, rank, H, W, m]
            depth_win = vol_batch[:, k1:k1+self.rank, :, :, :]  # [batch, rank, H, W, m]
            # Reshape to [batch*rank, H, W, m] for 2D unfold
            depth_win_flat = depth_win.reshape(batch_size * self.rank, H, W, m)  # [batch*rank, H, W, m]
            # Permute to [batch*rank, m, H, W] for unfold
            depth_win_perm = depth_win_flat.permute(0, 3, 1, 2)  # [batch*rank, m, H, W]
            
            # Use 2D unfold to extract height-width windows
            # Kernel size: (self.rank, self.rank), stride: (step, step)
            windows_2d = torch.nn.functional.unfold(depth_win_perm, kernel_size=(self.rank, self.rank), stride=(step, step))
            # windows_2d shape: [batch*rank, m*rank*rank, n_windows_2d]
            n_windows_2d = windows_2d.shape[2]
            # Reshape to [batch, rank, m, rank, rank, n_windows_2d]
            windows_2d = windows_2d.view(batch_size, self.rank, m, self.rank, self.rank, n_windows_2d)
            # Permute to [batch, n_windows_2d, rank, rank, rank, m] -> [batch, n_windows_2d, rank*rank*rank, m]
            windows_2d = windows_2d.permute(0, 5, 1, 3, 4, 2).contiguous()  # [batch, n_windows_2d, rank, rank, rank, m]
            windows_2d = windows_2d.view(batch_size, n_windows_2d, self.rank*self.rank*self.rank, m)
            
            # Apply mapping M to each vector
            mapped = self.M(windows_2d)  # [batch, n_windows_2d, rank*rank*rank, m]
            
            # Aggregate over rank dimension
            if self.rank_op == 'sum':
                agg = mapped.sum(dim=2)  # [batch, n_windows_2d, m]
            elif self.rank_op == 'pick':
                idx = torch.randint(0, self.rank*self.rank*self.rank, (batch_size, n_windows_2d, 1), device=self.device)
                agg = torch.gather(mapped, 2, idx.expand(-1, -1, -1, m)).squeeze(2)
            elif self.rank_op == 'user_func':
                avg = mapped.mean(dim=2)
                agg = torch.sigmoid(avg)
            else:  # 'avg'
                agg = mapped.mean(dim=2)  # [batch, n_windows_2d, m]
            
            # Store
            all_windows.append(agg)  # [batch, n_windows_2d, m]
            # For coordinates, we need to expand k2 and k3
            for k2 in k2_starts:
                for k3 in k3_starts:
                    all_k1.append(k1)
                    all_k2.append(k2)
                    all_k3.append(k3)
        
        # Concatenate all windows across depth
        # all_windows is a list of [batch, n_windows_2d, m] tensors, each for one depth index
        # We need to stack them along a new dimension to get [batch, n_windows, m]
        windows_list = []
        for d_idx, agg in enumerate(all_windows):
            # agg: [batch, n_windows_2d, m]
            windows_list.append(agg)
        windows = torch.cat(windows_list, dim=1)  # [batch, n_windows, m]
        
        # Build k tensors for all windows
        k1_tensor = torch.tensor(all_k1, dtype=torch.long, device=self.device)  # [n_windows]
        k2_tensor = torch.tensor(all_k2, dtype=torch.long, device=self.device)
        k3_tensor = torch.tensor(all_k3, dtype=torch.long, device=self.device)
        # Expand to batch dimension
        k1_expanded = k1_tensor.unsqueeze(0).expand(batch_size, -1)
        k2_expanded = k2_tensor.unsqueeze(0).expand(batch_size, -1)
        k3_expanded = k3_tensor.unsqueeze(0).expand(batch_size, -1)
        
        # Flatten to compute all Nk in one go
        windows_flat = windows.reshape(batch_size * n_windows, m)
        k1_flat = k1_expanded.reshape(batch_size * n_windows)
        k2_flat = k2_expanded.reshape(batch_size * n_windows)
        k3_flat = k3_expanded.reshape(batch_size * n_windows)
        
        # Compute Nk
        Nk_flat = self._compute_Nk_from_x_3d(k1_flat, k2_flat, k3_flat, windows_flat)
        Nk = Nk_flat.view(batch_size, n_windows, m)
        
        # Average over windows
        rep = Nk.mean(dim=1)  # [batch, m]
        return rep

    def batch_compute_Nk_and_targets_3d(self, vol_batch):
        """
        Compute Nk vectors and target vectors for each window in a batch of volumes.
        Supports arbitrary window size and step, but only for rank_mode == 'drop'.
        
        Args:
            vol_batch: Tensor of shape [batch_size, D, H, W, m]
        
        Returns:
            Nk_all: Tensor [batch_size, n_windows, m] - Nk vectors for each window
            targets: Tensor [batch_size, n_windows, m] - target vectors (aggregated mapped vectors)
        """
        batch_size, D, H, W, m = vol_batch.shape
        k1_starts, k2_starts, k3_starts = self._get_window_starts_3d(D, H, W)
        if not k1_starts or not k2_starts or not k3_starts:
            return (torch.empty(batch_size, 0, m, device=self.device),
                    torch.empty(batch_size, 0, m, device=self.device))
        
        n_windows_d = len(k1_starts)
        n_windows_h = len(k2_starts)
        n_windows_w = len(k3_starts)
        n_windows = n_windows_d * n_windows_h * n_windows_w
        
        step = 1 if self.mode == 'linear' else (self.step if self.step is not None else self.rank)
        
        # Collect all windows and targets across depth
        all_targets = []   # list of [batch, n_windows_2d, m] tensors (targets)
        all_k1 = []
        all_k2 = []
        all_k3 = []
        
        for d_idx, k1 in enumerate(k1_starts):
            depth_win = vol_batch[:, k1:k1+self.rank, :, :, :]  # [batch, rank, H, W, m]
            depth_win_flat = depth_win.reshape(batch_size * self.rank, H, W, m)
            depth_win_perm = depth_win_flat.permute(0, 3, 1, 2)  # [batch*rank, m, H, W]
            
            windows_2d = torch.nn.functional.unfold(depth_win_perm, kernel_size=(self.rank, self.rank), stride=(step, step))
            n_windows_2d = windows_2d.shape[2]
            windows_2d = windows_2d.view(batch_size, self.rank, m, self.rank, self.rank, n_windows_2d)
            windows_2d = windows_2d.permute(0, 5, 1, 3, 4, 2).contiguous()
            windows_2d = windows_2d.view(batch_size, n_windows_2d, self.rank*self.rank*self.rank, m)
            
            # Apply mapping
            mapped = self.M(windows_2d)  # [batch, n_windows_2d, rank*rank*rank, m]
            
            # Aggregate to get targets
            if self.rank_op == 'sum':
                targets = mapped.sum(dim=2)
            elif self.rank_op == 'pick':
                idx = torch.randint(0, self.rank*self.rank*self.rank, (batch_size, n_windows_2d, 1), device=self.device)
                targets = torch.gather(mapped, 2, idx.expand(-1, -1, -1, m)).squeeze(2)
            elif self.rank_op == 'user_func':
                avg = mapped.mean(dim=2)
                targets = torch.sigmoid(avg)
            else:  # 'avg'
                targets = mapped.mean(dim=2)
            
            all_targets.append(targets)  # [batch, n_windows_2d, m]
            # For coordinates
            for k2 in k2_starts:
                for k3 in k3_starts:
                    all_k1.append(k1)
                    all_k2.append(k2)
                    all_k3.append(k3)
        
        # Concatenate across depth
        targets = torch.cat(all_targets, dim=1)  # [batch, n_windows, m]
        
        # Build k tensors
        k1_tensor = torch.tensor(all_k1, dtype=torch.long, device=self.device)
        k2_tensor = torch.tensor(all_k2, dtype=torch.long, device=self.device)
        k3_tensor = torch.tensor(all_k3, dtype=torch.long, device=self.device)
        k1_expanded = k1_tensor.unsqueeze(0).expand(batch_size, -1)
        k2_expanded = k2_tensor.unsqueeze(0).expand(batch_size, -1)
        k3_expanded = k3_tensor.unsqueeze(0).expand(batch_size, -1)
        
        # Flatten to compute Nk
        targets_flat = targets.reshape(batch_size * n_windows, m)
        k1_flat = k1_expanded.reshape(batch_size * n_windows)
        k2_flat = k2_expanded.reshape(batch_size * n_windows)
        k3_flat = k3_expanded.reshape(batch_size * n_windows)
        Nk_flat = self._compute_Nk_from_x_3d(k1_flat, k2_flat, k3_flat, targets_flat)
        Nk_all = Nk_flat.view(batch_size, n_windows, m)
        
        return Nk_all, targets

    def batch_compute_Nk(self, k1_tensor, k2_tensor, k3_tensor, vectors):
        """
        Vectorized computation of N(k1,k2,k3) vectors for a batch of positions and vectors
        N(k1,k2,k3) = sum_{i,j} P_{i,j} * (M(v))_i * cos(2π*k1/periods1[i,j]) * cos(2π*k2/periods2[i,j]) * cos(2π*k3/periods3[i,j])
        
        Args:
            k1_tensor: Tensor of depth indices [batch_size] (long)
            k2_tensor: Tensor of height indices [batch_size] (long)
            k3_tensor: Tensor of width indices [batch_size] (long)
            vectors: Tensor of window vectors [batch_size, vec_dim]
            
        Returns:
            Tensor of N(k1,k2,k3) vectors [batch_size, vec_dim]
        """
        # Apply square mapping matrix M to each vector
        x = self.M(vectors)  # [batch_size, vec_dim]
        return self._compute_Nk_from_x_3d(k1_tensor, k2_tensor, k3_tensor, x)

    def compute_Nk(self, k1, k2, k3, vector):
        """Compute N(k1,k2,k3) for single position and vector (uses batch internally)"""
        if not isinstance(vector, torch.Tensor):
            vector = torch.tensor(vector, dtype=torch.float32, device=self.device)
        elif vector.device != self.device:
            vector = vector.to(self.device)
        
        k1_t = torch.tensor([k1], dtype=torch.long, device=self.device)
        k2_t = torch.tensor([k2], dtype=torch.long, device=self.device)
        k3_t = torch.tensor([k3], dtype=torch.long, device=self.device)
        vec_t = vector.unsqueeze(0)
        result = self.batch_compute_Nk(k1_t, k2_t, k3_t, vec_t)
        return result[0]

    def describe(self, volume):
        """Compute N(k1,k2,k3) vectors for each window in the 3D volume"""
        window_vectors, (k1_coords, k2_coords, k3_coords) = self.extract_windows(volume)
        if window_vectors.shape[0] == 0:
            return []
        
        N_batch = self.batch_compute_Nk(k1_coords, k2_coords, k3_coords, window_vectors)
        return N_batch.detach().cpu().numpy()

    def S(self, volume):
        """
        Compute cumulative sum of N(k1,k2,k3) vectors over windows in row-major order.
        Returns list of S(l) for l = 1..L where L = number of windows.
        """
        window_vectors, (k1_coords, k2_coords, k3_coords) = self.extract_windows(volume)
        if window_vectors.shape[0] == 0:
            return []
        
        N_batch = self.batch_compute_Nk(k1_coords, k2_coords, k3_coords, window_vectors)
        S_cum = torch.cumsum(N_batch, dim=0)
        return [s.detach().cpu().numpy() for s in S_cum]

    def D(self, volumes, t_list):
        """
        Compute mean squared deviation D across 3D volumes:
        D = average over all windows of (N(k1,k2,k3)-t)^2
        """
        total_loss = 0.0
        total_windows = 0
        
        # Convert target vectors to tensor
        t_tensors = [torch.tensor(t, dtype=torch.float32, device=self.device) for t in t_list]
        
        for volume, t in zip(volumes, t_tensors):
            window_vectors, (k1_coords, k2_coords, k3_coords) = self.extract_windows(volume)
            if window_vectors.shape[0] == 0:
                continue
            
            N_batch = self.batch_compute_Nk(k1_coords, k2_coords, k3_coords, window_vectors)
            losses = torch.sum((N_batch - t) ** 2, dim=1)
            total_loss += losses.sum().item()
            total_windows += window_vectors.shape[0]
        
        return total_loss / total_windows if total_windows else 0.0

    def d(self, volume, t):
        """Compute pattern deviation value (d) for a single 3D volume."""
        return self.D([volume], [t])

    def reg_train(self, volumes, t_list, max_iters=1000, tol=1e-8, learning_rate=0.01,
                  continued=False, decay_rate=1.0, print_every=10, batch_size=32,
                  checkpoint_file=None, checkpoint_interval=10):
        """
        Train the model using gradient descent with volume-level batch processing.
        Optimized for GPU memory efficiency by processing volumes individually, but
        uses fast batch processing when all volumes in a batch have the same dimensions
        and rank_mode == 'drop'.
        
        Args:
            volumes: List of 3D volumes (each is list of lists of lists of vectors or torch.Tensor of shape (D,H,W,vec_dim))
            t_list: List of target vectors corresponding to volumes
            max_iters: Maximum number of training iterations
            tol: Convergence tolerance
            learning_rate: Initial learning rate for optimizer
            continued: Whether to continue training from existing parameters
            decay_rate: Learning rate decay rate
            print_every: Print progress every N iterations
            batch_size: Number of volumes to process in each batch
            checkpoint_file: Path to save training checkpoints
            checkpoint_interval: Save checkpoint every N iterations
            
        Returns:
            list: Training loss history
        """
        if not continued:
            self.reset_parameters()
        
        # Ensure all volumes are on the correct device and are tensors
        volumes = [v.to(self.device) if isinstance(v, torch.Tensor) else 
                   torch.tensor(v, dtype=torch.float32, device=self.device) for v in volumes]
        
        # First pass: extract windows for all volumes and record max coordinates
        extracted_list = []
        max_k1 = max_k2 = max_k3 = 0
        for vol in volumes:
            window_vectors, (k1_coords, k2_coords, k3_coords) = self.extract_windows(vol)
            extracted_list.append((window_vectors, (k1_coords, k2_coords, k3_coords)))
            if window_vectors.shape[0] > 0:
                max_k1 = max(max_k1, k1_coords.max().item() + 1)
                max_k2 = max(max_k2, k2_coords.max().item() + 1)
                max_k3 = max(max_k3, k3_coords.max().item() + 1)
        
        # Precompute phi table based on maximum coordinates encountered
        max_k1 = max(1, max_k1)
        max_k2 = max(1, max_k2)
        max_k3 = max(1, max_k3)
        self._precompute_phi_table(max_k1, max_k2, max_k3)
        
        # Convert target vectors to tensor
        t_tensors = [torch.tensor(t, dtype=torch.float32, device=self.device) for t in t_list]
        
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)
        
        history = []
        prev_loss = float('inf')
        best_loss = float('inf')
        best_model_state = None
        
        # Fast batch processing possible when rank_mode == 'drop'
        use_fast_batch = (self.rank_mode == 'drop')
        
        for it in range(max_iters):
            total_loss = 0.0
            total_volumes = 0
            
            indices = list(range(len(volumes)))
            random.shuffle(indices)
            
            for batch_start in range(0, len(indices), batch_size):
                batch_indices = indices[batch_start:batch_start + batch_size]
                batch_vols = [volumes[idx] for idx in batch_indices]
                batch_targets = [t_tensors[idx] for idx in batch_indices]
                batch_extracted = [extracted_list[idx] for idx in batch_indices]
                
                optimizer.zero_grad()
                batch_loss = 0.0
                
                # Check if all volumes in this batch have the same dimensions
                vol_shapes = [(v.shape[0], v.shape[1], v.shape[2]) for v in batch_vols]
                if use_fast_batch and len(set(vol_shapes)) == 1:
                    # All volumes have equal dimensions -> use fast batch processing
                    batch_vol_tensor = torch.stack(batch_vols, dim=0)  # [batch, D, H, W, m]
                    # Compute volume representations
                    reps = self.batch_represent_3d(batch_vol_tensor)  # [batch, m]
                    # Convert targets to tensor
                    targets_tensor = torch.stack(batch_targets, dim=0)  # [batch, m]
                    # Compute loss for all volumes in batch
                    loss = torch.mean(torch.sum((reps - targets_tensor) ** 2, dim=1))
                    loss.backward()
                    optimizer.step()
                    batch_loss = loss.item() * len(batch_vols)
                    total_loss += batch_loss
                    total_volumes += len(batch_vols)
                else:
                    # Process each volume individually using pre-extracted windows
                    for (window_vectors, (k1_coords, k2_coords, k3_coords)), target in zip(batch_extracted, batch_targets):
                        if window_vectors.shape[0] == 0:
                            continue
                        N_batch = self.batch_compute_Nk(k1_coords, k2_coords, k3_coords, window_vectors)
                        # Average over windows to get volume-level prediction
                        volume_pred = torch.mean(N_batch, dim=0)
                        seq_loss = torch.sum((volume_pred - target) ** 2)
                        batch_loss += seq_loss
                        # Clean up
                        del N_batch, volume_pred
                    
                    if len(batch_extracted) > 0:
                        batch_loss = batch_loss / len(batch_extracted)
                        batch_loss.backward()
                        optimizer.step()
                        total_loss += batch_loss.item() * len(batch_extracted)
                        total_volumes += len(batch_extracted)
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            avg_loss = total_loss / total_volumes if total_volumes else 0.0
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
        
        self._compute_training_statistics(volumes, extracted_list)
        self.trained = True
        return history

    def cls_train(self, volumes, labels, num_classes, max_iters=1000, tol=1e-8, learning_rate=0.01,
                  continued=False, decay_rate=1.0, print_every=10, batch_size=32,
                  checkpoint_file=None, checkpoint_interval=10):
        """
        Train the model for multi-class classification using cross-entropy loss.
        Optimized with fast batch processing for equal-sized volumes.
        """
        if self.classifier is None or self.num_classes != num_classes:
            self.classifier = nn.Linear(self.vec_dim, num_classes).to(self.device)
            self.num_classes = num_classes
        
        if not continued:
            self.reset_parameters()
        
        # Ensure all volumes are on the correct device
        volumes = [v.to(self.device) if isinstance(v, torch.Tensor) else 
                   torch.tensor(v, dtype=torch.float32, device=self.device) for v in volumes]
        
        # First pass: extract windows and record max coordinates
        extracted_list = []
        max_k1 = max_k2 = max_k3 = 0
        for vol in volumes:
            window_vectors, (k1_coords, k2_coords, k3_coords) = self.extract_windows(vol)
            extracted_list.append((window_vectors, (k1_coords, k2_coords, k3_coords)))
            if window_vectors.shape[0] > 0:
                max_k1 = max(max_k1, k1_coords.max().item() + 1)
                max_k2 = max(max_k2, k2_coords.max().item() + 1)
                max_k3 = max(max_k3, k3_coords.max().item() + 1)
        
        max_k1 = max(1, max_k1)
        max_k2 = max(1, max_k2)
        max_k3 = max(1, max_k3)
        self._precompute_phi_table(max_k1, max_k2, max_k3)
        
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
            total_volumes = 0
            correct = 0
            
            indices = list(range(len(volumes)))
            random.shuffle(indices)
            
            for batch_start in range(0, len(indices), batch_size):
                batch_indices = indices[batch_start:batch_start + batch_size]
                batch_vols = [volumes[idx] for idx in batch_indices]
                batch_labels = label_tensors[batch_indices]
                batch_extracted = [extracted_list[idx] for idx in batch_indices]
                
                optimizer.zero_grad()
                
                vol_shapes = [(v.shape[0], v.shape[1], v.shape[2]) for v in batch_vols]
                if use_fast_batch and len(set(vol_shapes)) == 1:
                    # Fast path
                    batch_vol_tensor = torch.stack(batch_vols, dim=0)
                    reps = self.batch_represent_3d(batch_vol_tensor)  # [batch, m]
                    logits = self.classifier(reps)
                    loss = criterion(logits, batch_labels)
                    loss.backward()
                    optimizer.step()
                    
                    batch_loss = loss.item() * len(batch_vols)
                    total_loss += batch_loss
                    total_volumes += len(batch_vols)
                    with torch.no_grad():
                        pred = torch.argmax(logits, dim=1)
                        correct += (pred == batch_labels).sum().item()
                else:
                    # Slow path: process individually using pre-extracted windows
                    batch_logits = []
                    for (window_vectors, (k1_coords, k2_coords, k3_coords)) in batch_extracted:
                        if window_vectors.shape[0] == 0:
                            seq_vector = torch.zeros(self.vec_dim, device=self.device)
                        else:
                            N_batch = self.batch_compute_Nk(k1_coords, k2_coords, k3_coords, window_vectors)
                            seq_vector = torch.mean(N_batch, dim=0)
                            del N_batch
                        logits = self.classifier(seq_vector.unsqueeze(0))
                        batch_logits.append(logits)
                    
                    if batch_logits:
                        all_logits = torch.cat(batch_logits, dim=0)
                        loss = criterion(all_logits, batch_labels)
                        loss.backward()
                        optimizer.step()
                        total_loss += loss.item() * len(batch_vols)
                        total_volumes += len(batch_vols)
                        with torch.no_grad():
                            pred = torch.argmax(all_logits, dim=1)
                            correct += (pred == batch_labels).sum().item()
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            avg_loss = total_loss / total_volumes if total_volumes else 0.0
            accuracy = correct / total_volumes if total_volumes else 0.0
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

    def lbl_train(self, volumes, labels, num_labels, max_iters=1000, tol=1e-8, learning_rate=0.01,
                  continued=False, decay_rate=1.0, print_every=10, batch_size=32,
                  checkpoint_file=None, checkpoint_interval=10, pos_weight=None):
        """
        Train the model for multi-label classification using binary cross-entropy loss.
        Optimized with fast batch processing for equal-sized volumes.
        """
        if self.labeller is None or self.num_labels != num_labels:
            self.labeller = nn.Linear(self.vec_dim, num_labels).to(self.device)
            self.num_labels = num_labels
        
        if not continued:
            self.reset_parameters()
        
        # Ensure all volumes are on the correct device
        volumes = [v.to(self.device) if isinstance(v, torch.Tensor) else 
                   torch.tensor(v, dtype=torch.float32, device=self.device) for v in volumes]
        
        # First pass: extract windows and record max coordinates
        extracted_list = []
        max_k1 = max_k2 = max_k3 = 0
        for vol in volumes:
            window_vectors, (k1_coords, k2_coords, k3_coords) = self.extract_windows(vol)
            extracted_list.append((window_vectors, (k1_coords, k2_coords, k3_coords)))
            if window_vectors.shape[0] > 0:
                max_k1 = max(max_k1, k1_coords.max().item() + 1)
                max_k2 = max(max_k2, k2_coords.max().item() + 1)
                max_k3 = max(max_k3, k3_coords.max().item() + 1)
        
        max_k1 = max(1, max_k1)
        max_k2 = max(1, max_k2)
        max_k3 = max(1, max_k3)
        self._precompute_phi_table(max_k1, max_k2, max_k3)
        
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
            total_volumes = 0
            
            indices = list(range(len(volumes)))
            random.shuffle(indices)
            
            for batch_start in range(0, len(indices), batch_size):
                batch_indices = indices[batch_start:batch_start + batch_size]
                batch_vols = [volumes[idx] for idx in batch_indices]
                batch_labels = labels_tensor[batch_indices]
                batch_extracted = [extracted_list[idx] for idx in batch_indices]
                
                optimizer.zero_grad()
                
                vol_shapes = [(v.shape[0], v.shape[1], v.shape[2]) for v in batch_vols]
                if use_fast_batch and len(set(vol_shapes)) == 1:
                    # Fast path
                    batch_vol_tensor = torch.stack(batch_vols, dim=0)
                    reps = self.batch_represent_3d(batch_vol_tensor)  # [batch, m]
                    logits = self.labeller(reps)  # [batch, num_labels]
                    loss = criterion(logits, batch_labels)
                    loss.backward()
                    optimizer.step()
                    
                    batch_loss = loss.item() * len(batch_vols)
                    total_loss += batch_loss
                    total_volumes += len(batch_vols)
                    
                    with torch.no_grad():
                        probs = torch.sigmoid(logits)
                        preds = (probs > 0.5).float()
                        batch_correct = (preds == batch_labels).sum().item()
                        batch_predictions = batch_labels.numel()
                        total_correct += batch_correct
                        total_predictions += batch_predictions
                else:
                    # Slow path: process individually using pre-extracted windows
                    batch_logits = []
                    for (window_vectors, (k1_coords, k2_coords, k3_coords)) in batch_extracted:
                        if window_vectors.shape[0] == 0:
                            continue
                        N_batch = self.batch_compute_Nk(k1_coords, k2_coords, k3_coords, window_vectors)
                        seq_rep = torch.mean(N_batch, dim=0)
                        logits = self.labeller(seq_rep)
                        batch_logits.append(logits)
                        del N_batch, seq_rep
                    
                    if batch_logits:
                        batch_logits = torch.stack(batch_logits, dim=0)
                        loss = criterion(batch_logits, batch_labels)
                        loss.backward()
                        optimizer.step()
                        total_loss += loss.item() * len(batch_vols)
                        total_volumes += len(batch_vols)
                        
                        with torch.no_grad():
                            probs = torch.sigmoid(batch_logits)
                            preds = (probs > 0.5).float()
                            batch_correct = (preds == batch_labels).sum().item()
                            batch_predictions = batch_labels.numel()
                            total_correct += batch_correct
                            total_predictions += batch_predictions
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            avg_loss = total_loss / total_volumes if total_volumes else 0.0
            avg_acc = total_correct / total_predictions if total_predictions else 0.0
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

    def self_train(self, volumes, max_iters=100, tol=1e-6, learning_rate=0.01,
                   continued=False, decay_rate=1.0, print_every=10,
                   batch_size=32, checkpoint_file=None, checkpoint_interval=5):
        """
        Self-training method for self-consistency (gap mode) with memory-efficient volume processing.
        Trains the model so that N(k1,k2,k3) vectors match the transformed window vectors at each position.
        Uses fast batch processing when all volumes have the same dimensions and rank_mode == 'drop'.
        """
        if not continued:
            self.reset_parameters()
        
        # Ensure all volumes are on the correct device
        volumes = [v.to(self.device) if isinstance(v, torch.Tensor) else 
                   torch.tensor(v, dtype=torch.float32, device=self.device) for v in volumes]
        
        # First pass: extract windows and record max coordinates
        extracted_list = []
        max_k1 = max_k2 = max_k3 = 0
        for vol in volumes:
            window_vectors, (k1_coords, k2_coords, k3_coords) = self.extract_windows(vol)
            extracted_list.append((window_vectors, (k1_coords, k2_coords, k3_coords)))
            if window_vectors.shape[0] > 0:
                max_k1 = max(max_k1, k1_coords.max().item() + 1)
                max_k2 = max(max_k2, k2_coords.max().item() + 1)
                max_k3 = max(max_k3, k3_coords.max().item() + 1)
        
        max_k1 = max(1, max_k1)
        max_k2 = max(1, max_k2)
        max_k3 = max(1, max_k3)
        self._precompute_phi_table(max_k1, max_k2, max_k3)
        
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)
        
        history = []
        prev_loss = float('inf')
        best_loss = float('inf')
        best_model_state = None
        
        # Check if all volumes have the same dimensions (for fast batch processing)
        vol_shapes = [(v.shape[0], v.shape[1], v.shape[2]) for v in volumes]
        all_equal = len(set(vol_shapes)) == 1
        use_fast_batch = (self.rank_mode == 'drop' and all_equal)
        
        if use_fast_batch:
            # Fast path: all volumes equal size, batch process all windows together
            D, H, W = vol_shapes[0]
            all_vols = torch.stack(volumes, dim=0)  # [num_vols, D, H, W, m]
            num_vols = len(volumes)
            seq_batch_size = batch_size  # number of volumes per batch
            
            for it in range(max_iters):
                total_loss = 0.0
                total_windows = 0
                
                indices = list(range(num_vols))
                random.shuffle(indices)
                
                for batch_start in range(0, num_vols, seq_batch_size):
                    batch_indices = indices[batch_start:batch_start + seq_batch_size]
                    batch_vols = all_vols[batch_indices]  # [batch, D, H, W, m]
                    
                    optimizer.zero_grad()
                    
                    # Compute Nk and targets for all windows in this batch of volumes
                    Nk_batch, targets_batch = self.batch_compute_Nk_and_targets_3d(batch_vols)  # both [batch, n_windows, m]
                    
                    # Compute MSE loss over all windows
                    loss = torch.mean(torch.sum((Nk_batch - targets_batch) ** 2, dim=-1))
                    loss.backward()
                    optimizer.step()
                    
                    batch_loss = loss.item() * batch_vols.shape[0] * Nk_batch.shape[1]
                    total_loss += batch_loss
                    total_windows += batch_vols.shape[0] * Nk_batch.shape[1]
                    
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                avg_loss = total_loss / total_windows if total_windows else 0.0
                history.append(avg_loss)
                
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    best_model_state = copy.deepcopy(self.state_dict())
                
                if it % print_every == 0 or it == max_iters - 1:
                    current_lr = scheduler.get_last_lr()[0]
                    print(f"Self-Train Iter {it:3d}: Loss = {avg_loss:.6f}, LR = {current_lr:.6f}, "
                          f"Windows = {total_windows}")
                
                if checkpoint_file and (it % checkpoint_interval == 0 or it == max_iters - 1):
                    self._save_checkpoint(checkpoint_file, it, history, optimizer, scheduler, best_loss)
                
                if abs(prev_loss - avg_loss) < tol:
                    print(f"Converged after {it+1} iterations")
                    if best_model_state is not None:
                        self.load_state_dict(best_model_state)
                    break
                prev_loss = avg_loss
                scheduler.step()
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        else:
            # Slow path: volumes have different sizes or rank_mode == 'pad'
            for it in range(max_iters):
                total_loss = 0.0
                total_windows = 0
                
                indices = list(range(len(volumes)))
                random.shuffle(indices)
                
                for batch_start in range(0, len(indices), batch_size):
                    batch_indices = indices[batch_start:batch_start + batch_size]
                    batch_extracted = [extracted_list[idx] for idx in batch_indices]
                    
                    optimizer.zero_grad()
                    batch_loss = 0.0
                    batch_window_count = 0
                    
                    for (window_vectors, (k1_coords, k2_coords, k3_coords)) in batch_extracted:
                        if window_vectors.shape[0] == 0:
                            continue
                        
                        N_batch = self.batch_compute_Nk(k1_coords, k2_coords, k3_coords, window_vectors)
                        target_vectors = self.M(window_vectors)  # [num_windows, vec_dim]
                        
                        # Self-consistency loss: N(k1,k2,k3) should match transformed window vector
                        loss = torch.mean(torch.sum((N_batch - target_vectors) ** 2, dim=1))
                        batch_loss += loss
                        batch_window_count += window_vectors.shape[0]
                        
                        del N_batch, target_vectors
                    
                    if batch_window_count > 0:
                        batch_loss = batch_loss / batch_window_count  # Average over windows in batch
                        batch_loss.backward()
                        optimizer.step()
                        total_loss += batch_loss.item() * batch_window_count
                        total_windows += batch_window_count
                    
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                avg_loss = total_loss / total_windows if total_windows else 0.0
                history.append(avg_loss)
                
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    best_model_state = copy.deepcopy(self.state_dict())
                
                if it % print_every == 0 or it == max_iters - 1:
                    current_lr = scheduler.get_last_lr()[0]
                    print(f"Self-Train Iter {it:3d}: Loss = {avg_loss:.6f}, LR = {current_lr:.6f}, "
                          f"Windows = {total_windows}")
                
                if checkpoint_file and (it % checkpoint_interval == 0 or it == max_iters - 1):
                    self._save_checkpoint(checkpoint_file, it, history, optimizer, scheduler, best_loss)
                
                if abs(prev_loss - avg_loss) < tol:
                    print(f"Converged after {it+1} iterations")
                    if best_model_state is not None:
                        self.load_state_dict(best_model_state)
                    break
                prev_loss = avg_loss
                scheduler.step()
        
        self._compute_training_statistics(volumes, extracted_list)
        self.trained = True
        return history

    def _compute_training_statistics(self, volumes, extracted_list=None, batch_size=50):
        """
        Compute training statistics for reconstruction and generation.
        Uses precomputed phi table if available.
        If extracted_list is provided, reuse it to avoid re-extraction.
        """
        if extracted_list is not None:
            # Use pre-extracted windows for efficiency
            total_window_count = 0
            total_t = torch.zeros(self.vec_dim, device=self.device)
            with torch.no_grad():
                for window_vectors, (k1_coords, k2_coords, k3_coords) in extracted_list:
                    if window_vectors.shape[0] == 0:
                        continue
                    N_batch = self.batch_compute_Nk(k1_coords, k2_coords, k3_coords, window_vectors)
                    total_window_count += window_vectors.shape[0]
                    total_t += N_batch.sum(dim=0)
            self.mean_window_count = total_window_count / len(volumes) if volumes else 0
            self.mean_t = (total_t / total_window_count).cpu().numpy() if total_window_count > 0 else np.zeros(self.vec_dim)
            return
        
        # Fallback to original slower method
        total_window_count = 0
        total_t = torch.zeros(self.vec_dim, device=self.device)
        
        with torch.no_grad():
            for i in range(0, len(volumes), batch_size):
                batch_volumes = volumes[i:i+batch_size]
                batch_window_count = 0
                batch_t_sum = torch.zeros(self.vec_dim, device=self.device)
                
                for volume in batch_volumes:
                    window_vectors, (k1_coords, k2_coords, k3_coords) = self.extract_windows(volume)
                    if window_vectors.shape[0] == 0:
                        continue
                    N_batch = self.batch_compute_Nk(k1_coords, k2_coords, k3_coords, window_vectors)
                    batch_window_count += window_vectors.shape[0]
                    batch_t_sum += N_batch.sum(dim=0)
                    del N_batch, window_vectors
                
                total_window_count += batch_window_count
                total_t += batch_t_sum
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        self.mean_window_count = total_window_count / len(volumes) if volumes else 0
        self.mean_t = (total_t / total_window_count).cpu().numpy() if total_window_count > 0 else np.zeros(self.vec_dim)

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

    def predict_t(self, volume):
        """
        Predict target vector for a 3D volume.
        Returns the average of all N(k1,k2,k3) vectors over windows.
        """
        # Use batch_represent_3d for efficiency (single volume)
        if not isinstance(volume, torch.Tensor):
            volume = torch.tensor(volume, dtype=torch.float32, device=self.device)
        elif volume.device != self.device:
            volume = volume.to(self.device)
        # Add batch dimension
        vol_batch = volume.unsqueeze(0)  # [1, D, H, W, m]
        rep = self.batch_represent_3d(vol_batch)  # [1, m]
        return rep[0].detach().cpu().numpy()

    def predict_c(self, volume):
        """
        Predict class label for a 3D volume using the classification head.
        Returns: (predicted_class, class_probabilities)
        """
        if self.classifier is None:
            raise ValueError("Model must be trained first for classification")
        
        seq_vector = self.predict_t(volume)
        seq_vector_tensor = torch.tensor(seq_vector, dtype=torch.float32, device=self.device)
        
        with torch.no_grad():
            logits = self.classifier(seq_vector_tensor.unsqueeze(0))
            probs = torch.softmax(logits, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
        return pred_class, probs[0].cpu().numpy()

    def predict_l(self, volume, threshold=0.5):
        """
        Predict multi-label classification for a 3D volume.
        Returns: (binary_predictions, probabilities)
        """
        assert self.labeller is not None, "Model must be trained first for label prediction"
        
        if not isinstance(volume, torch.Tensor):
            volume = torch.tensor(volume, dtype=torch.float32, device=self.device)
        elif volume.device != self.device:
            volume = volume.to(self.device)
        
        vol_batch = volume.unsqueeze(0)  # [1, D, H, W, m]
        rep = self.batch_represent_3d(vol_batch)  # [1, m]
        seq_rep = rep[0]
        
        with torch.no_grad():
            logits = self.labeller(seq_rep)
            probs = torch.sigmoid(logits).cpu().numpy()
        binary = (probs > threshold).astype(np.float32)
        return binary, probs

    def reconstruct(self, D, H, W, tau=0.0):
        """
        Reconstruct a representative 3D volume of size D x H x W by minimizing error with temperature-controlled randomness.
        Assumes non-overlapping windows (step = rank) for reconstruction.
        """
        assert self.trained, "Model must be trained first"
        if tau < 0:
            raise ValueError("Temperature must be non-negative")
        
        # Number of windows in each dimension (assuming step = rank)
        num_windows_d = (D + self.rank - 1) // self.rank
        num_windows_h = (H + self.rank - 1) // self.rank
        num_windows_w = (W + self.rank - 1) // self.rank
        mean_t_tensor = torch.tensor(self.mean_t, dtype=torch.float32, device=self.device)
        
        # Pre-generate candidate vectors (for simplicity, random normal)
        num_candidates = 100
        candidate_vectors = torch.randn(num_candidates, self.vec_dim, device=self.device)
        
        # For each window position, select a vector
        window_vectors = []
        window_coords = []
        for k1 in range(num_windows_d):
            for k2 in range(num_windows_h):
                for k3 in range(num_windows_w):
                    # Compute Nk for all candidates at this (k1,k2,k3)
                    k1_t = torch.full((num_candidates,), k1, dtype=torch.long, device=self.device)
                    k2_t = torch.full((num_candidates,), k2, dtype=torch.long, device=self.device)
                    k3_t = torch.full((num_candidates,), k3, dtype=torch.long, device=self.device)
                    N_all = self.batch_compute_Nk(k1_t, k2_t, k3_t, candidate_vectors)
                    errors = torch.sum((N_all - mean_t_tensor) ** 2, dim=1)
                    scores = -errors
                    
                    if tau == 0:  # deterministic
                        best_idx = torch.argmax(scores).item()
                        best_vec = candidate_vectors[best_idx]
                    else:
                        probs = torch.softmax(scores / tau, dim=0).detach().cpu().numpy()
                        chosen_idx = random.choices(range(num_candidates), weights=probs, k=1)[0]
                        best_vec = candidate_vectors[chosen_idx]
                    
                    window_vectors.append(best_vec)
                    window_coords.append((k1, k2, k3))
        
        # Build the full volume by placing each window's representative vector into the corresponding cube.
        volume = torch.zeros(D, H, W, self.vec_dim, device=self.device)
        for (k1, k2, k3), vec in zip(window_coords, window_vectors):
            d_start = k1 * self.rank
            d_end = min(d_start + self.rank, D)
            h_start = k2 * self.rank
            h_end = min(h_start + self.rank, H)
            w_start = k3 * self.rank
            w_end = min(w_start + self.rank, W)
            volume[d_start:d_end, h_start:h_end, w_start:w_end, :] = vec
        return volume.detach().cpu().numpy()

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
    print("Spatial Numerical Dual Descriptor PM3 - 3D Volume Version (Accelerated)")
    print("Processes 3D arrays of m-dimensional real vectors with 3D basis functions")
    print("="*50)
    
    # Set random seeds
    torch.manual_seed(11)
    random.seed(11)
    np.random.seed(11)
    
    # Parameters
    vec_dim = 6         # Dimension of vectors (reduced for speed in 3D)
    rank = 2            # Window size (cube)
    user_step = 2       # Step size for nonlinear mode (non-overlapping)
    
    # Initialize model
    ndd = SpatialNumDualDescriptorPM3(
        vec_dim=vec_dim,
        rank=rank,
        mode='nonlinear',
        user_step=user_step,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    print(f"\nUsing device: {ndd.device}")
    print(f"Vector dimension: {vec_dim}")
    print(f"Window size: {rank} x {rank} x {rank}")
    print(f"P matrix shape: {ndd.P.shape}")
    print(f"M matrix shape: {ndd.M.weight.shape}")
    
    # Generate 50 random 3D volumes with random target vectors (reduced number for speed)
    print("\nGenerating training data (3D volumes)...")
    volumes = []
    t_list = []
    for _ in range(50):
        D = random.randint(10, 15)   # depth
        H = random.randint(10, 15)   # height
        W = random.randint(10, 15)   # width
        volume = torch.randn(D, H, W, vec_dim)  # random vectors
        volumes.append(volume)
        t_list.append(np.random.uniform(-1.0, 1.0, vec_dim))
    
    # Regression training
    print("\n" + "="*50)
    print("Starting Regression Training")
    print("="*50)
    ndd.reg_train(volumes, t_list, max_iters=30, tol=1e-9, learning_rate=1.0, decay_rate=0.95, batch_size=4)
    
    # Predict target of first volume
    first_vol = volumes[0]
    t_pred = ndd.predict_t(first_vol)
    print(f"\nPredicted t for first volume: {[round(x, 4) for x in t_pred[:5]]}...")
    
    # Correlation between predicted and real targets
    print("\nCalculating prediction correlations...")
    pred_t_list = [ndd.predict_t(v) for v in volumes]
    corr_sum = 0.0
    for i in range(vec_dim):
        actu = [t[i] for t in t_list]
        pred = [t[i] for t in pred_t_list]
        corr, _ = pearsonr(actu, pred)
        print(f"Dimension {i} correlation: {corr:.4f}")
        corr_sum += corr
    print(f"Average correlation: {corr_sum / vec_dim:.4f}")
    
    # Reconstruction
    print("\nGenerating reconstructed volumes...")
    recon_det = ndd.reconstruct(D=10, H=10, W=10, tau=0.0)
    recon_rand = ndd.reconstruct(D=10, H=10, W=10, tau=0.5)
    print(f"Deterministic reconstruction shape: {recon_det.shape}")
    print(f"Stochastic reconstruction shape: {recon_rand.shape}")
    print(f"Deterministic mean: {np.mean(recon_det):.4f}, std: {np.std(recon_det):.4f}")
    print(f"Stochastic mean: {np.mean(recon_rand):.4f}, std: {np.std(recon_rand):.4f}")
    
    # Classification task
    print("\n" + "="*50)
    print("Classification Task")
    print("="*50)
    num_classes = 3
    class_vols = []
    class_labels = []
    for class_id in range(num_classes):
        for _ in range(30):  # 30 volumes per class
            D = random.randint(8, 12)
            H = random.randint(8, 12)
            W = random.randint(8, 12)
            if class_id == 0:
                vol = torch.randn(D, H, W, vec_dim) + 1.0
            elif class_id == 1:
                vol = torch.randn(D, H, W, vec_dim) - 1.0
            else:
                vol = torch.randn(D, H, W, vec_dim)
            class_vols.append(vol)
            class_labels.append(class_id)
    
    ndd_cls = SpatialNumDualDescriptorPM3(vec_dim=vec_dim, rank=rank, mode='nonlinear', user_step=user_step)
    print("\nStarting Classification Training")
    history = ndd_cls.cls_train(class_vols, class_labels, num_classes, max_iters=20, learning_rate=0.05,
                                decay_rate=0.99, batch_size=4, print_every=5)
    
    correct = 0
    for v, lbl in zip(class_vols, class_labels):
        pred, _ = ndd_cls.predict_c(v)
        if pred == lbl:
            correct += 1
    acc = correct / len(class_vols)
    print(f"Classification accuracy: {acc:.4f} ({correct}/{len(class_vols)})")
    
    # Multi-label classification
    print("\n" + "="*50)
    print("Multi-Label Classification Task")
    print("="*50)
    num_labels = 3   # reduced for speed
    ml_vols = []
    ml_labels = []
    for _ in range(50):
        D = random.randint(8, 12)
        H = random.randint(8, 12)
        W = random.randint(8, 12)
        vol = torch.randn(D, H, W, vec_dim)
        ml_vols.append(vol)
        label_vec = [1.0 if random.random() > 0.7 else 0.0 for _ in range(num_labels)]
        ml_labels.append(label_vec)
    
    ndd_lbl = SpatialNumDualDescriptorPM3(vec_dim=vec_dim, rank=rank, mode='nonlinear', user_step=user_step)
    loss_hist, acc_hist = ndd_lbl.lbl_train(ml_vols, ml_labels, num_labels, max_iters=20,
                                             learning_rate=0.05, decay_rate=0.99, batch_size=4, print_every=5)
    print(f"Multi-label final training loss: {loss_hist[-1]:.6f}, accuracy: {acc_hist[-1]:.4f}")
    
    test_vol = torch.randn(10, 10, 10, vec_dim)
    bin_pred, prob_pred = ndd_lbl.predict_l(test_vol, threshold=0.5)
    print(f"\nTest volume prediction: {bin_pred}, probabilities: {prob_pred}")
    
    # Self-training example
    print("\n" + "="*50)
    print("Self-Training Example")
    print("="*50)
    ndd_self = SpatialNumDualDescriptorPM3(vec_dim=vec_dim, rank=rank, mode='nonlinear', user_step=user_step)
    self_seqs = [torch.randn(10, 10, 10, vec_dim) for _ in range(5)]  # equal size for fast path
    self_history = ndd_self.self_train(self_seqs, max_iters=10, learning_rate=0.01, batch_size=2)
    plt.figure(figsize=(8,5))
    plt.plot(self_history)
    plt.title('Self-Training Loss (3D)')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig('self_training_loss_3d.png')
    print("Self-training loss plot saved as 'self_training_loss_3d.png'")
    
    print("\nAll tests completed successfully!")
    print("="*50)
