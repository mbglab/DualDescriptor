# Copyright (C) 2005-2026, Bin-Guang Ma (mbg@mail.hzau.edu.cn); SPDX-License-Identifier: MIT
# The Spatial Numerical Dual Descriptor Vector class (P Matrix form) for 4D array implemented with PyTorch
# This program is for the demonstration of methodology and not fully refined.
# Author: Bin-Guang Ma (assisted by DeepSeek); Date: 2025-8-28 ~ 2026-03-31

import math
import random
import itertools
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy

class SpatialNumDualDescriptorPM4(nn.Module):
    """
    Numerical Vector Dual Descriptor for 4D spatial hypervolumes with GPU acceleration using PyTorch:
      - Processes 4D grids (dim1 × dim2 × dim3 × dim4) of m-dimensional real vectors
      - matrix P ∈ R^{m×m} of basis coefficients
      - square mapping matrix M ∈ R^{m×m} for vector transformation (assumes input_dim = model_dim)
      - indexed periods: period1[i,j] = i*m + j + 2, similarly for period2, period3, period4
      - basis function phi_{i,j}(k1,k2,k3,k4) = cos(2π*k1/period1) * cos(2π*k2/period2) * cos(2π*k3/period3) * cos(2π*k4/period4)
      - supports 'linear' (step=1) or 'nonlinear' (step-by-rank) window extraction
      - batch acceleration for equal-sized hypervolumes when rank_mode == 'drop'
    """
    def __init__(self, vec_dim, rank=1, rank_op='avg', rank_mode='drop', mode='linear', user_step=None, device='cuda'):
        super().__init__()
        self.vec_dim = vec_dim          # Dimension of input vectors and internal representation
        self.rank = rank                # Window size (hypercube side length)
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
        
        # Precompute indexed periods for four dimensions (fixed, not trainable)
        periods = torch.zeros(self.vec_dim, self.vec_dim, dtype=torch.float32)
        for i in range(self.vec_dim):
            for j in range(self.vec_dim):
                periods[i, j] = i * self.vec_dim + j + 2
        self.register_buffer('periods1', periods)   # period1 matrix
        self.register_buffer('periods2', periods)   # period2 matrix
        self.register_buffer('periods3', periods)   # period3 matrix
        self.register_buffer('periods4', periods)   # period4 matrix

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

    def extract_windows(self, hypervol):
        """
        Extract hypercubic windows from a 4D volume and apply rank operation to each window.
        
        Args:
            hypervol: torch.Tensor of shape (D1, D2, D3, D4, vec_dim) or nested list of vectors
            
        Returns:
            tuple: (window_vectors, window_coords)
                window_vectors: torch.Tensor of shape (num_windows, vec_dim)
                window_coords: tuple of four tensors (k1_coords, k2_coords, k3_coords, k4_coords) each of shape (num_windows,)
        """
        # Convert to tensor if needed
        if not isinstance(hypervol, torch.Tensor):
            # Assume hypervol is nested list: D1 list of D2 list of D3 list of D4 list of vectors
            D1 = len(hypervol)
            D2 = len(hypervol[0]) if D1 > 0 else 0
            D3 = len(hypervol[0][0]) if D2 > 0 else 0
            D4 = len(hypervol[0][0][0]) if D3 > 0 else 0
            vol_tensor = torch.zeros(D1, D2, D3, D4, self.vec_dim, dtype=torch.float32, device=self.device)
            for d1 in range(D1):
                for d2 in range(D2):
                    for d3 in range(D3):
                        for d4 in range(D4):
                            vec = hypervol[d1][d2][d3][d4]
                            if not isinstance(vec, torch.Tensor):
                                vec = torch.tensor(vec, dtype=torch.float32)
                            vol_tensor[d1, d2, d3, d4] = vec.to(self.device)
        else:
            vol_tensor = hypervol.to(self.device)
        
        D1, D2, D3, D4 = vol_tensor.shape[0], vol_tensor.shape[1], vol_tensor.shape[2], vol_tensor.shape[3]
        
        # Determine step size
        if self.mode == 'linear':
            step = 1
        else:  # nonlinear
            step = self.step if self.step is not None else self.rank
        
        # Collect all windows
        windows = []
        coords_k1 = []  # dim1 coordinate
        coords_k2 = []  # dim2 coordinate
        coords_k3 = []  # dim3 coordinate
        coords_k4 = []  # dim4 coordinate
        
        # Iterate over all possible window starting positions
        for k1 in range(0, D1 - self.rank + 1, step):
            for k2 in range(0, D2 - self.rank + 1, step):
                for k3 in range(0, D3 - self.rank + 1, step):
                    for k4 in range(0, D4 - self.rank + 1, step):
                        # Extract hypercube of shape (rank, rank, rank, rank, vec_dim)
                        window = vol_tensor[k1:k1+self.rank, k2:k2+self.rank, k3:k3+self.rank, k4:k4+self.rank, :]
                        # Flatten to (rank*rank*rank*rank, vec_dim)
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
                        coords_k4.append(k4)
        
        # Handle incomplete windows if rank_mode == 'pad'
        if self.rank_mode == 'pad':
            # For each dimension, consider starting positions that would create incomplete windows.
            # We iterate over all start positions that produce at least one element, then pad.
            for k1 in range(0, D1, step):
                for k2 in range(0, D2, step):
                    for k3 in range(0, D3, step):
                        for k4 in range(0, D4, step):
                            # Skip already processed full windows (if step equals rank, this condition may skip some)
                            if (k1 + self.rank <= D1 and k2 + self.rank <= D2 and
                                k3 + self.rank <= D3 and k4 + self.rank <= D4):
                                continue
                            # Extract partial window
                            end1 = min(k1 + self.rank, D1)
                            end2 = min(k2 + self.rank, D2)
                            end3 = min(k3 + self.rank, D3)
                            end4 = min(k4 + self.rank, D4)
                            window = vol_tensor[k1:end1, k2:end2, k3:end3, k4:end4, :]  # [d1_rem, d2_rem, d3_rem, d4_rem, vec_dim]
                            if window.numel() == 0:
                                continue
                            # Pad to full rank x rank x rank x rank
                            d1_rem, d2_rem, d3_rem, d4_rem = window.shape[0], window.shape[1], window.shape[2], window.shape[3]
                            padded = torch.zeros(self.rank, self.rank, self.rank, self.rank, self.vec_dim, device=self.device)
                            padded[:d1_rem, :d2_rem, :d3_rem, :d4_rem, :] = window
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
                            coords_k4.append(k4)
        
        if not windows:
            # No valid windows
            return torch.empty(0, self.vec_dim, device=self.device), (torch.empty(0), torch.empty(0), torch.empty(0), torch.empty(0))
        
        window_vectors = torch.stack(windows)  # [num_windows, vec_dim]
        coords_k1 = torch.tensor(coords_k1, dtype=torch.float32, device=self.device)
        coords_k2 = torch.tensor(coords_k2, dtype=torch.float32, device=self.device)
        coords_k3 = torch.tensor(coords_k3, dtype=torch.float32, device=self.device)
        coords_k4 = torch.tensor(coords_k4, dtype=torch.float32, device=self.device)
        return window_vectors, (coords_k1, coords_k2, coords_k3, coords_k4)

    def batch_compute_Nk(self, k1_tensor, k2_tensor, k3_tensor, k4_tensor, vectors):
        """
        Vectorized computation of N(k1,k2,k3,k4) vectors for a batch of positions and vectors
        N(k1,k2,k3,k4) = sum_{i,j} P_{i,j} * (M(v))_i * cos(2π*k1/periods1[i,j]) * cos(2π*k2/periods2[i,j]) * cos(2π*k3/periods3[i,j]) * cos(2π*k4/periods4[i,j])
        
        Args:
            k1_tensor: Tensor of dim1 indices [batch_size]
            k2_tensor: Tensor of dim2 indices [batch_size]
            k3_tensor: Tensor of dim3 indices [batch_size]
            k4_tensor: Tensor of dim4 indices [batch_size]
            vectors: Tensor of window vectors [batch_size, vec_dim]
            
        Returns:
            Tensor of N(k1,k2,k3,k4) vectors [batch_size, vec_dim]
        """
        # Apply square mapping matrix M to each vector
        x = self.M(vectors)  # [batch_size, vec_dim]
        
        # Expand dimensions for broadcasting
        k1_exp = k1_tensor.view(-1, 1, 1)  # [batch_size, 1, 1]
        k2_exp = k2_tensor.view(-1, 1, 1)  # [batch_size, 1, 1]
        k3_exp = k3_tensor.view(-1, 1, 1)  # [batch_size, 1, 1]
        k4_exp = k4_tensor.view(-1, 1, 1)  # [batch_size, 1, 1]
        
        # Calculate basis functions: product of four cosines
        phi = (torch.cos(2 * math.pi * k1_exp / self.periods1) *
               torch.cos(2 * math.pi * k2_exp / self.periods2) *
               torch.cos(2 * math.pi * k3_exp / self.periods3) *
               torch.cos(2 * math.pi * k4_exp / self.periods4))  # [batch_size, vec_dim, vec_dim]
        
        # Optimized computation using einsum
        Nk = torch.einsum('bj,ij,bij->bi', x, self.P, phi)
        return Nk

    def compute_Nk(self, k1, k2, k3, k4, vector):
        """Compute N(k1,k2,k3,k4) for single position and vector (uses batch internally)"""
        if not isinstance(vector, torch.Tensor):
            vector = torch.tensor(vector, dtype=torch.float32, device=self.device)
        elif vector.device != self.device:
            vector = vector.to(self.device)
        
        k1_t = torch.tensor([k1], dtype=torch.float32, device=self.device)
        k2_t = torch.tensor([k2], dtype=torch.float32, device=self.device)
        k3_t = torch.tensor([k3], dtype=torch.float32, device=self.device)
        k4_t = torch.tensor([k4], dtype=torch.float32, device=self.device)
        vec_t = vector.unsqueeze(0)
        result = self.batch_compute_Nk(k1_t, k2_t, k3_t, k4_t, vec_t)
        return result[0]

    # --------------------------------------------------------------------------
    # Batch acceleration methods for equal-sized hypervolumes (rank_mode == 'drop')
    # --------------------------------------------------------------------------
    def _get_starts(self, size):
        """Compute starting indices for windows along one dimension."""
        if self.mode == 'linear':
            step = 1
        else:
            step = self.step if self.step is not None else self.rank
        if self.rank_mode == 'drop':
            starts = list(range(0, size - self.rank + 1, step))
        else:  # 'pad'
            starts = list(range(0, size, step))
        return starts

    def _aggregate_window(self, window_vectors):
        """
        Aggregate vectors within a window according to rank_op.
        window_vectors: shape [batch, n_windows, n_vecs, m]
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

    def batch_represent(self, batch_hypervols):
        """
        Compute sequence representations for a batch of hypervolumes efficiently.
        Assumes all hypervolumes have the same dimensions (D1, D2, D3, D4) and rank_mode == 'drop'.
        
        Args:
            batch_hypervols: Tensor of shape [batch_size, D1, D2, D3, D4, vec_dim]
        
        Returns:
            representations: Tensor of shape [batch_size, vec_dim]
        """
        batch_hypervols = batch_hypervols.to(self.device)
        batch_size, D1, D2, D3, D4, m = batch_hypervols.shape
        
        # Get window start indices
        starts1 = self._get_starts(D1)
        starts2 = self._get_starts(D2)
        starts3 = self._get_starts(D3)
        starts4 = self._get_starts(D4)
        if not starts1 or not starts2 or not starts3 or not starts4:
            return torch.zeros(batch_size, m, device=self.device)
        
        n1 = len(starts1)
        n2 = len(starts2)
        n3 = len(starts3)
        n4 = len(starts4)
        n_windows_total = n1 * n2 * n3 * n4
        
        # Precompute the coordinates for starts3 and starts4
        s3_vals = torch.tensor(starts3, dtype=torch.float32, device=self.device)
        s4_vals = torch.tensor(starts4, dtype=torch.float32, device=self.device)
        s3_grid, s4_grid = torch.meshgrid(s3_vals, s4_vals, indexing='ij')
        k3_coords = s3_grid.flatten()   # [n3*n4]
        k4_coords = s4_grid.flatten()
        
        step = 1 if self.mode == 'linear' else (self.step if self.step is not None else self.rank)
        
        all_windows = []
        # Loop over the first two dimensions (starts1 and starts2)
        for s1 in starts1:
            for s2 in starts2:
                # Extract slice along dim1 and dim2
                vol_slice = batch_hypervols[:, s1:s1+self.rank, s2:s2+self.rank, :, :, :]  # [batch, rank, rank, D3, D4, m]
                # Permute to bring m to front: [batch, m, rank, rank, D3, D4]
                x = vol_slice.permute(0, 5, 1, 2, 3, 4).contiguous()
                # Combine m, rank, rank into channels: [batch, m*rank*rank, D3, D4]
                x = x.view(batch_size, m * self.rank * self.rank, D3, D4)
                # Apply unfold on (D3, D4) dimensions
                windows_2d = torch.nn.functional.unfold(
                    x, kernel_size=(self.rank, self.rank), stride=(step, step)
                )  # [batch, m*rank*rank*rank*rank, n_hw] where n_hw = n3 * n4
                # Reshape to separate m, rank, rank, rank, rank, and window index
                # windows_2d shape: [batch, m*rank^4, n_hw]
                windows_2d = windows_2d.view(batch_size, m, self.rank, self.rank, self.rank, self.rank, n3 * n4)
                # Permute to bring window index to dim1 and m to last: [batch, n_hw, rank, rank, rank, rank, m]
                windows_2d = windows_2d.permute(0, 6, 5, 2, 3, 4, 1).contiguous()
                # Flatten the spatial dimensions of the hypercube
                windows_flat = windows_2d.reshape(batch_size, n3 * n4, self.rank ** 4, m)  # [batch, n_hw, n_vecs, m]
                all_windows.append(windows_flat)
        
        # Concatenate all windows from different (s1,s2) pairs
        all_windows = torch.cat(all_windows, dim=1)  # [batch, n_windows_total, n_vecs, m]
        
        # Aggregate each window according to rank_op (no M yet)
        agg = self._aggregate_window(all_windows)  # [batch, n_windows_total, m]
        
        # Build k1, k2, k3, k4 coordinates for all windows
        s1_vals = torch.tensor(starts1, dtype=torch.float32, device=self.device)
        s2_vals = torch.tensor(starts2, dtype=torch.float32, device=self.device)
        # Create meshgrid of all four dimensions
        k1_grid, k2_grid, k3_grid, k4_grid = torch.meshgrid(s1_vals, s2_vals, s3_vals, s4_vals, indexing='ij')
        k1_coords = k1_grid.flatten()  # [n_windows_total]
        k2_coords = k2_grid.flatten()
        k3_coords = k3_grid.flatten()
        k4_coords = k4_grid.flatten()
        
        # Expand to batch dimension
        k1_expanded = k1_coords.unsqueeze(0).expand(batch_size, -1)  # [batch, n_windows_total]
        k2_expanded = k2_coords.unsqueeze(0).expand(batch_size, -1)
        k3_expanded = k3_coords.unsqueeze(0).expand(batch_size, -1)
        k4_expanded = k4_coords.unsqueeze(0).expand(batch_size, -1)
        
        # Flatten agg and coordinates to compute Nk
        agg_flat = agg.reshape(batch_size * n_windows_total, m)  # [batch*n_windows_total, m]
        k1_flat = k1_expanded.reshape(batch_size * n_windows_total)
        k2_flat = k2_expanded.reshape(batch_size * n_windows_total)
        k3_flat = k3_expanded.reshape(batch_size * n_windows_total)
        k4_flat = k4_expanded.reshape(batch_size * n_windows_total)
        
        # Compute Nk for all windows (this applies M inside)
        Nk_flat = self.batch_compute_Nk(k1_flat, k2_flat, k3_flat, k4_flat, agg_flat)  # [batch*n_windows_total, m]
        Nk = Nk_flat.reshape(batch_size, n_windows_total, m)  # [batch, n_windows_total, m]
        
        # Average over windows to get hypervolume-level representation
        rep = Nk.mean(dim=1)  # [batch, m]
        return rep

    def batch_compute_Nk_and_targets(self, batch_hypervols):
        """
        Compute Nk vectors and target vectors for each window in a batch of hypervolumes.
        Assumes all hypervolumes have the same dimensions (D1, D2, D3, D4) and rank_mode == 'drop'.
        Used in self-training.
        
        Args:
            batch_hypervols: Tensor of shape [batch_size, D1, D2, D3, D4, m]
        
        Returns:
            Nk_all: Tensor [batch_size, n_windows_total, m] - Nk vectors for each window
            targets: Tensor [batch_size, n_windows_total, m] - target vectors = M(aggregated(window))
        """
        batch_hypervols = batch_hypervols.to(self.device)
        batch_size, D1, D2, D3, D4, m = batch_hypervols.shape
        
        # Get window start indices
        starts1 = self._get_starts(D1)
        starts2 = self._get_starts(D2)
        starts3 = self._get_starts(D3)
        starts4 = self._get_starts(D4)
        if not starts1 or not starts2 or not starts3 or not starts4:
            return torch.empty(batch_size, 0, m, device=self.device), torch.empty(batch_size, 0, m, device=self.device)
        
        n1 = len(starts1)
        n2 = len(starts2)
        n3 = len(starts3)
        n4 = len(starts4)
        n_windows_total = n1 * n2 * n3 * n4
        
        # Precompute coordinates for starts3 and starts4
        s3_vals = torch.tensor(starts3, dtype=torch.float32, device=self.device)
        s4_vals = torch.tensor(starts4, dtype=torch.float32, device=self.device)
        s3_grid, s4_grid = torch.meshgrid(s3_vals, s4_vals, indexing='ij')
        k3_coords = s3_grid.flatten()
        k4_coords = s4_grid.flatten()
        
        step = 1 if self.mode == 'linear' else (self.step if self.step is not None else self.rank)
        
        all_windows = []
        for s1 in starts1:
            for s2 in starts2:
                vol_slice = batch_hypervols[:, s1:s1+self.rank, s2:s2+self.rank, :, :, :]  # [batch, rank, rank, D3, D4, m]
                x = vol_slice.permute(0, 5, 1, 2, 3, 4).contiguous()  # [batch, m, rank, rank, D3, D4]
                x = x.view(batch_size, m * self.rank * self.rank, D3, D4)  # [batch, m*rank^2, D3, D4]
                windows_2d = torch.nn.functional.unfold(
                    x, kernel_size=(self.rank, self.rank), stride=(step, step)
                )  # [batch, m*rank^4, n_hw]
                windows_2d = windows_2d.view(batch_size, m, self.rank, self.rank, self.rank, self.rank, n3 * n4)
                windows_2d = windows_2d.permute(0, 6, 5, 2, 3, 4, 1).contiguous()
                windows_flat = windows_2d.reshape(batch_size, n3 * n4, self.rank ** 4, m)
                all_windows.append(windows_flat)
        
        all_windows = torch.cat(all_windows, dim=1)  # [batch, n_windows_total, n_vecs, m]
        
        # Aggregate each window (no M)
        agg = self._aggregate_window(all_windows)  # [batch, n_windows_total, m]
        
        # Compute targets: M(agg)
        agg_flat = agg.reshape(batch_size * n_windows_total, m)
        targets_flat = self.M(agg_flat)  # [batch*n_windows_total, m]
        targets = targets_flat.reshape(batch_size, n_windows_total, m)
        
        # Build coordinates
        s1_vals = torch.tensor(starts1, dtype=torch.float32, device=self.device)
        s2_vals = torch.tensor(starts2, dtype=torch.float32, device=self.device)
        k1_grid, k2_grid, k3_grid, k4_grid = torch.meshgrid(s1_vals, s2_vals, s3_vals, s4_vals, indexing='ij')
        k1_coords = k1_grid.flatten()
        k2_coords = k2_grid.flatten()
        k3_coords = k3_grid.flatten()
        k4_coords = k4_grid.flatten()
        
        k1_expanded = k1_coords.unsqueeze(0).expand(batch_size, -1)
        k2_expanded = k2_coords.unsqueeze(0).expand(batch_size, -1)
        k3_expanded = k3_coords.unsqueeze(0).expand(batch_size, -1)
        k4_expanded = k4_coords.unsqueeze(0).expand(batch_size, -1)
        
        # Compute Nk from agg (this applies M inside)
        agg_flat2 = agg.reshape(batch_size * n_windows_total, m)
        k1_flat = k1_expanded.reshape(batch_size * n_windows_total)
        k2_flat = k2_expanded.reshape(batch_size * n_windows_total)
        k3_flat = k3_expanded.reshape(batch_size * n_windows_total)
        k4_flat = k4_expanded.reshape(batch_size * n_windows_total)
        Nk_flat = self.batch_compute_Nk(k1_flat, k2_flat, k3_flat, k4_flat, agg_flat2)  # [batch*n_windows_total, m]
        Nk_all = Nk_flat.reshape(batch_size, n_windows_total, m)
        
        return Nk_all, targets

    # --------------------------------------------------------------------------
    # Core methods (describe, S, D, d)
    # --------------------------------------------------------------------------
    def describe(self, hypervol):
        """Compute N(k1,k2,k3,k4) vectors for each window in the 4D hypervolume"""
        window_vectors, (k1_coords, k2_coords, k3_coords, k4_coords) = self.extract_windows(hypervol)
        if window_vectors.shape[0] == 0:
            return []
        
        N_batch = self.batch_compute_Nk(k1_coords, k2_coords, k3_coords, k4_coords, window_vectors)
        return N_batch.detach().cpu().numpy()

    def S(self, hypervol):
        """
        Compute cumulative sum of N(k1,k2,k3,k4) vectors over windows in row-major order.
        Returns list of S(l) for l = 1..L where L = number of windows.
        """
        window_vectors, (k1_coords, k2_coords, k3_coords, k4_coords) = self.extract_windows(hypervol)
        if window_vectors.shape[0] == 0:
            return []
        
        N_batch = self.batch_compute_Nk(k1_coords, k2_coords, k3_coords, k4_coords, window_vectors)
        S_cum = torch.cumsum(N_batch, dim=0)
        return [s.detach().cpu().numpy() for s in S_cum]

    def D(self, hypervols, t_list):
        """
        Compute mean squared deviation D across 4D hypervolumes:
        D = average over all windows of (N(k1,k2,k3,k4)-t)^2
        """
        total_loss = 0.0
        total_windows = 0
        
        # Convert target vectors to tensor
        t_tensors = [torch.tensor(t, dtype=torch.float32, device=self.device) for t in t_list]
        
        for hypervol, t in zip(hypervols, t_tensors):
            window_vectors, (k1_coords, k2_coords, k3_coords, k4_coords) = self.extract_windows(hypervol)
            if window_vectors.shape[0] == 0:
                continue
            
            N_batch = self.batch_compute_Nk(k1_coords, k2_coords, k3_coords, k4_coords, window_vectors)
            losses = torch.sum((N_batch - t) ** 2, dim=1)
            total_loss += losses.sum().item()
            total_windows += window_vectors.shape[0]
        
        return total_loss / total_windows if total_windows else 0.0

    def d(self, hypervol, t):
        """Compute pattern deviation value (d) for a single 4D hypervolume."""
        return self.D([hypervol], [t])

    # --------------------------------------------------------------------------
    # Training methods with batch acceleration
    # --------------------------------------------------------------------------
    def reg_train(self, hypervols, t_list, max_iters=1000, tol=1e-8, learning_rate=0.01,
                  continued=False, decay_rate=1.0, print_every=10, batch_size=32,
                  checkpoint_file=None, checkpoint_interval=10):
        """
        Train the model using gradient descent with hypervolume-level batch processing.
        Optimized for GPU memory efficiency by processing hypervolumes individually, but
        uses fast batch processing when all hypervolumes in a batch have equal size
        and rank_mode == 'drop'.
        
        Args:
            hypervols: List of 4D hypervolumes (each is list of lists of lists of lists of vectors or torch.Tensor of shape (D1,D2,D3,D4,vec_dim))
            t_list: List of target vectors corresponding to hypervolumes
            max_iters: Maximum number of training iterations
            tol: Convergence tolerance
            learning_rate: Initial learning rate for optimizer
            continued: Whether to continue training from existing parameters
            decay_rate: Learning rate decay rate
            print_every: Print progress every N iterations
            batch_size: Number of hypervolumes to process in each batch
            checkpoint_file: Path to save training checkpoints
            checkpoint_interval: Save checkpoint every N iterations
            
        Returns:
            list: Training loss history
        """
        if not continued:
            self.reset_parameters()
        
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
            total_vols = 0
            
            indices = list(range(len(hypervols)))
            random.shuffle(indices)
            
            for batch_start in range(0, len(indices), batch_size):
                batch_indices = indices[batch_start:batch_start + batch_size]
                batch_vols = [hypervols[idx] for idx in batch_indices]
                batch_targets = [t_tensors[idx] for idx in batch_indices]
                
                optimizer.zero_grad()
                batch_loss = 0.0
                
                # Check if all volumes in this batch have the same shape
                shapes = []
                for v in batch_vols:
                    if isinstance(v, torch.Tensor):
                        shapes.append(v.shape)
                    else:
                        # Assume nested list: D1 x D2 x D3 x D4 x vec_dim
                        D1 = len(v)
                        D2 = len(v[0]) if D1 > 0 else 0
                        D3 = len(v[0][0]) if D2 > 0 else 0
                        D4 = len(v[0][0][0]) if D3 > 0 else 0
                        shapes.append((D1, D2, D3, D4, self.vec_dim))
                if use_fast_batch and len(set(shapes)) == 1:
                    # All volumes have equal dimensions -> use fast batch processing
                    # Convert volumes to a single tensor, moving to device
                    batch_vols_tensor = torch.stack([v.to(self.device) if isinstance(v, torch.Tensor) else torch.tensor(v, dtype=torch.float32, device=self.device) for v in batch_vols], dim=0)
                    # Compute representations (fast path)
                    reps = self.batch_represent(batch_vols_tensor)  # [batch, m]
                    # Stack targets
                    targets_tensor = torch.stack(batch_targets, dim=0)  # [batch, m]
                    # Compute loss
                    loss = torch.mean(torch.sum((reps - targets_tensor) ** 2, dim=1))
                    loss.backward()
                    optimizer.step()
                    batch_loss = loss.item() * len(batch_vols)
                    total_loss += batch_loss
                    total_vols += len(batch_vols)
                else:
                    # Process each volume individually
                    for hypervol, target in zip(batch_vols, batch_targets):
                        window_vectors, (k1_coords, k2_coords, k3_coords, k4_coords) = self.extract_windows(hypervol)
                        if window_vectors.shape[0] == 0:
                            continue
                        
                        N_batch = self.batch_compute_Nk(k1_coords, k2_coords, k3_coords, k4_coords, window_vectors)
                        # Average over windows to get hypervolume-level prediction
                        vol_pred = torch.mean(N_batch, dim=0)
                        seq_loss = torch.sum((vol_pred - target) ** 2)
                        batch_loss += seq_loss
                        
                        del N_batch, vol_pred, window_vectors
                    
                    if len(batch_vols) > 0:
                        batch_loss = batch_loss / len(batch_vols)
                        batch_loss.backward()
                        optimizer.step()
                        total_loss += batch_loss.item() * len(batch_vols)
                        total_vols += len(batch_vols)
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            avg_loss = total_loss / total_vols if total_vols else 0.0
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
        
        self._compute_training_statistics(hypervols)
        self.trained = True
        return history

    def cls_train(self, hypervols, labels, num_classes, max_iters=1000, tol=1e-8, learning_rate=0.01,
                  continued=False, decay_rate=1.0, print_every=10, batch_size=32,
                  checkpoint_file=None, checkpoint_interval=10):
        """
        Train the model for multi-class classification using cross-entropy loss.
        Optimized for GPU memory efficiency by processing hypervolumes individually, but
        uses fast batch processing when all hypervolumes in a batch have equal size
        and rank_mode == 'drop'.
        """
        if self.classifier is None or self.num_classes != num_classes:
            self.classifier = nn.Linear(self.vec_dim, num_classes).to(self.device)
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
            total_vols = 0
            correct = 0
            
            indices = list(range(len(hypervols)))
            random.shuffle(indices)
            
            for batch_start in range(0, len(indices), batch_size):
                batch_indices = indices[batch_start:batch_start + batch_size]
                batch_vols = [hypervols[idx] for idx in batch_indices]
                batch_labels = label_tensors[batch_indices]
                
                optimizer.zero_grad()
                
                shapes = []
                for v in batch_vols:
                    if isinstance(v, torch.Tensor):
                        shapes.append(v.shape)
                    else:
                        D1 = len(v)
                        D2 = len(v[0]) if D1 > 0 else 0
                        D3 = len(v[0][0]) if D2 > 0 else 0
                        D4 = len(v[0][0][0]) if D3 > 0 else 0
                        shapes.append((D1, D2, D3, D4, self.vec_dim))
                if use_fast_batch and len(set(shapes)) == 1:
                    batch_vols_tensor = torch.stack([v.to(self.device) if isinstance(v, torch.Tensor) else torch.tensor(v, dtype=torch.float32, device=self.device) for v in batch_vols], dim=0)
                    reps = self.batch_represent(batch_vols_tensor)  # [batch, m]
                    logits = self.classifier(reps)
                    loss = criterion(logits, batch_labels)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item() * len(batch_vols)
                    total_vols += len(batch_vols)
                    with torch.no_grad():
                        pred = torch.argmax(logits, dim=1)
                        correct += (pred == batch_labels).sum().item()
                else:
                    batch_logits = []
                    for hypervol in batch_vols:
                        window_vectors, (k1_coords, k2_coords, k3_coords, k4_coords) = self.extract_windows(hypervol)
                        if window_vectors.shape[0] == 0:
                            seq_vector = torch.zeros(self.vec_dim, device=self.device)
                        else:
                            N_batch = self.batch_compute_Nk(k1_coords, k2_coords, k3_coords, k4_coords, window_vectors)
                            seq_vector = torch.mean(N_batch, dim=0)
                            del N_batch, window_vectors
                        logits = self.classifier(seq_vector.unsqueeze(0))
                        batch_logits.append(logits)
                    
                    if batch_logits:
                        all_logits = torch.cat(batch_logits, dim=0)
                        loss = criterion(all_logits, batch_labels)
                        loss.backward()
                        optimizer.step()
                        total_loss += loss.item() * len(batch_vols)
                        total_vols += len(batch_vols)
                        with torch.no_grad():
                            pred = torch.argmax(all_logits, dim=1)
                            correct += (pred == batch_labels).sum().item()
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            avg_loss = total_loss / total_vols if total_vols else 0.0
            accuracy = correct / total_vols if total_vols else 0.0
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

    def lbl_train(self, hypervols, labels, num_labels, max_iters=1000, tol=1e-8, learning_rate=0.01,
                  continued=False, decay_rate=1.0, print_every=10, batch_size=32,
                  checkpoint_file=None, checkpoint_interval=10, pos_weight=None):
        """
        Train the model for multi-label classification using binary cross-entropy loss.
        Optimized for GPU memory efficiency by processing hypervolumes individually, but
        uses fast batch processing when all hypervolumes in a batch have equal size
        and rank_mode == 'drop'.
        """
        if self.labeller is None or self.num_labels != num_labels:
            self.labeller = nn.Linear(self.vec_dim, num_labels).to(self.device)
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
            total_vols = 0
            
            indices = list(range(len(hypervols)))
            random.shuffle(indices)
            
            for batch_start in range(0, len(indices), batch_size):
                batch_indices = indices[batch_start:batch_start + batch_size]
                batch_vols = [hypervols[idx] for idx in batch_indices]
                batch_labels = labels_tensor[batch_indices]
                
                optimizer.zero_grad()
                
                shapes = []
                for v in batch_vols:
                    if isinstance(v, torch.Tensor):
                        shapes.append(v.shape)
                    else:
                        D1 = len(v)
                        D2 = len(v[0]) if D1 > 0 else 0
                        D3 = len(v[0][0]) if D2 > 0 else 0
                        D4 = len(v[0][0][0]) if D3 > 0 else 0
                        shapes.append((D1, D2, D3, D4, self.vec_dim))
                if use_fast_batch and len(set(shapes)) == 1:
                    batch_vols_tensor = torch.stack([v.to(self.device) if isinstance(v, torch.Tensor) else torch.tensor(v, dtype=torch.float32, device=self.device) for v in batch_vols], dim=0)
                    reps = self.batch_represent(batch_vols_tensor)  # [batch, m]
                    logits = self.labeller(reps)
                    loss = criterion(logits, batch_labels)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item() * len(batch_vols)
                    total_vols += len(batch_vols)
                    with torch.no_grad():
                        probs = torch.sigmoid(logits)
                        preds = (probs > 0.5).float()
                        batch_correct = (preds == batch_labels).sum().item()
                        batch_predictions = batch_labels.numel()
                        total_correct += batch_correct
                        total_predictions += batch_predictions
                else:
                    batch_logits = []
                    for hypervol in batch_vols:
                        window_vectors, (k1_coords, k2_coords, k3_coords, k4_coords) = self.extract_windows(hypervol)
                        if window_vectors.shape[0] == 0:
                            continue
                        N_batch = self.batch_compute_Nk(k1_coords, k2_coords, k3_coords, k4_coords, window_vectors)
                        seq_rep = torch.mean(N_batch, dim=0)
                        logits = self.labeller(seq_rep)
                        batch_logits.append(logits)
                        del N_batch, seq_rep, window_vectors
                    
                    if batch_logits:
                        batch_logits = torch.stack(batch_logits, dim=0)
                        loss = criterion(batch_logits, batch_labels)
                        loss.backward()
                        optimizer.step()
                        total_loss += loss.item() * len(batch_vols)
                        total_vols += len(batch_vols)
                        with torch.no_grad():
                            probs = torch.sigmoid(batch_logits)
                            preds = (probs > 0.5).float()
                            batch_correct = (preds == batch_labels).sum().item()
                            batch_predictions = batch_labels.numel()
                            total_correct += batch_correct
                            total_predictions += batch_predictions
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            avg_loss = total_loss / total_vols if total_vols else 0.0
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

    def self_train(self, hypervols, max_iters=100, tol=1e-6, learning_rate=0.01,
                   continued=False, decay_rate=1.0, print_every=10,
                   batch_size=32, checkpoint_file=None, checkpoint_interval=5):
        """
        Self-training method for self-consistency (gap mode) with memory-efficient hypervolume processing.
        Trains the model so that N(k1,k2,k3,k4) vectors match the transformed window vectors at each position.
        Uses fast batch processing when all hypervolumes have equal size and rank_mode == 'drop'.
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
            
            indices = list(range(len(hypervols)))
            random.shuffle(indices)
            
            for batch_start in range(0, len(indices), batch_size):
                batch_indices = indices[batch_start:batch_start + batch_size]
                batch_vols = [hypervols[idx] for idx in batch_indices]
                
                optimizer.zero_grad()
                
                shapes = []
                for v in batch_vols:
                    if isinstance(v, torch.Tensor):
                        shapes.append(v.shape)
                    else:
                        D1 = len(v)
                        D2 = len(v[0]) if D1 > 0 else 0
                        D3 = len(v[0][0]) if D2 > 0 else 0
                        D4 = len(v[0][0][0]) if D3 > 0 else 0
                        shapes.append((D1, D2, D3, D4, self.vec_dim))
                if use_fast_batch and len(set(shapes)) == 1:
                    # All volumes have equal dimensions -> fast path
                    batch_vols_tensor = torch.stack([v.to(self.device) if isinstance(v, torch.Tensor) else torch.tensor(v, dtype=torch.float32, device=self.device) for v in batch_vols], dim=0)
                    Nk_all, targets = self.batch_compute_Nk_and_targets(batch_vols_tensor)
                    # Compute MSE loss over all windows
                    loss = torch.mean(torch.sum((Nk_all - targets) ** 2, dim=-1))
                    loss.backward()
                    optimizer.step()
                    batch_loss = loss.item() * batch_vols_tensor.shape[0] * Nk_all.shape[1]
                    total_loss += batch_loss
                    total_windows += batch_vols_tensor.shape[0] * Nk_all.shape[1]
                else:
                    # Process each volume individually
                    batch_loss = 0.0
                    batch_window_count = 0
                    for hypervol in batch_vols:
                        window_vectors, (k1_coords, k2_coords, k3_coords, k4_coords) = self.extract_windows(hypervol)
                        if window_vectors.shape[0] == 0:
                            continue
                        
                        N_batch = self.batch_compute_Nk(k1_coords, k2_coords, k3_coords, k4_coords, window_vectors)
                        target_vectors = self.M(window_vectors)  # [num_windows, vec_dim]
                        loss = torch.mean(torch.sum((N_batch - target_vectors) ** 2, dim=1))
                        batch_loss += loss
                        batch_window_count += window_vectors.shape[0]
                        del N_batch, target_vectors, window_vectors
                    
                    if batch_window_count > 0:
                        batch_loss = batch_loss / batch_window_count
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
        
        self._compute_training_statistics(hypervols)
        self.trained = True
        return history

    def _compute_training_statistics(self, hypervols, batch_size=50):
        """
        Compute training statistics for reconstruction and generation.
        """
        total_window_count = 0
        total_t = torch.zeros(self.vec_dim, device=self.device)
        
        with torch.no_grad():
            for i in range(0, len(hypervols), batch_size):
                batch_vols = hypervols[i:i+batch_size]
                batch_window_count = 0
                batch_t_sum = torch.zeros(self.vec_dim, device=self.device)
                
                for hypervol in batch_vols:
                    window_vectors, (k1_coords, k2_coords, k3_coords, k4_coords) = self.extract_windows(hypervol)
                    if window_vectors.shape[0] == 0:
                        continue
                    N_batch = self.batch_compute_Nk(k1_coords, k2_coords, k3_coords, k4_coords, window_vectors)
                    batch_window_count += window_vectors.shape[0]
                    batch_t_sum += N_batch.sum(dim=0)
                    del N_batch, window_vectors
                
                total_window_count += batch_window_count
                total_t += batch_t_sum
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        self.mean_window_count = total_window_count / len(hypervols) if hypervols else 0
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

    def predict_t(self, hypervol):
        """
        Predict target vector for a 4D hypervolume.
        Returns the average of all N(k1,k2,k3,k4) vectors over windows.
        """
        window_vectors, (k1_coords, k2_coords, k3_coords, k4_coords) = self.extract_windows(hypervol)
        if window_vectors.shape[0] == 0:
            return np.zeros(self.vec_dim)
        
        N_batch = self.batch_compute_Nk(k1_coords, k2_coords, k3_coords, k4_coords, window_vectors)
        avg = torch.mean(N_batch, dim=0)
        return avg.detach().cpu().numpy()

    def predict_c(self, hypervol):
        """
        Predict class label for a 4D hypervolume using the classification head.
        Returns: (predicted_class, class_probabilities)
        """
        if self.classifier is None:
            raise ValueError("Model must be trained first for classification")
        
        seq_vector = self.predict_t(hypervol)
        seq_vector_tensor = torch.tensor(seq_vector, dtype=torch.float32, device=self.device)
        
        with torch.no_grad():
            logits = self.classifier(seq_vector_tensor.unsqueeze(0))
            probs = torch.softmax(logits, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
        return pred_class, probs[0].cpu().numpy()

    def predict_l(self, hypervol, threshold=0.5):
        """
        Predict multi-label classification for a 4D hypervolume.
        Returns: (binary_predictions, probabilities)
        """
        assert self.labeller is not None, "Model must be trained first for label prediction"
        
        window_vectors, (k1_coords, k2_coords, k3_coords, k4_coords) = self.extract_windows(hypervol)
        if window_vectors.shape[0] == 0:
            return np.zeros(self.num_labels, dtype=np.float32), np.zeros(self.num_labels, dtype=np.float32)
        
        N_batch = self.batch_compute_Nk(k1_coords, k2_coords, k3_coords, k4_coords, window_vectors)
        seq_rep = torch.mean(N_batch, dim=0)
        
        with torch.no_grad():
            logits = self.labeller(seq_rep)
            probs = torch.sigmoid(logits).cpu().numpy()
        binary = (probs > threshold).astype(np.float32)
        return binary, probs

    def reconstruct(self, D1, D2, D3, D4, tau=0.0):
        """
        Reconstruct a representative 4D hypervolume of size D1 x D2 x D3 x D4 by minimizing error with temperature-controlled randomness.
        Assumes non-overlapping windows (step = rank) for reconstruction.
        """
        assert self.trained, "Model must be trained first"
        if tau < 0:
            raise ValueError("Temperature must be non-negative")
        
        # Number of windows in each dimension (assuming step = rank)
        num_windows1 = (D1 + self.rank - 1) // self.rank
        num_windows2 = (D2 + self.rank - 1) // self.rank
        num_windows3 = (D3 + self.rank - 1) // self.rank
        num_windows4 = (D4 + self.rank - 1) // self.rank
        mean_t_tensor = torch.tensor(self.mean_t, dtype=torch.float32, device=self.device)
        
        # Pre-generate candidate vectors (for simplicity, random normal)
        num_candidates = 100
        candidate_vectors = torch.randn(num_candidates, self.vec_dim, device=self.device)
        
        # For each window position, select a vector
        window_vectors = []
        window_coords = []
        for k1 in range(num_windows1):
            for k2 in range(num_windows2):
                for k3 in range(num_windows3):
                    for k4 in range(num_windows4):
                        # Compute Nk for all candidates at this (k1,k2,k3,k4)
                        k1_t = torch.full((num_candidates,), k1, dtype=torch.float32, device=self.device)
                        k2_t = torch.full((num_candidates,), k2, dtype=torch.float32, device=self.device)
                        k3_t = torch.full((num_candidates,), k3, dtype=torch.float32, device=self.device)
                        k4_t = torch.full((num_candidates,), k4, dtype=torch.float32, device=self.device)
                        N_all = self.batch_compute_Nk(k1_t, k2_t, k3_t, k4_t, candidate_vectors)
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
                        window_coords.append((k1, k2, k3, k4))
        
        # Build the full hypervolume by placing each window's representative vector into the corresponding hypercube.
        hypervol = torch.zeros(D1, D2, D3, D4, self.vec_dim, device=self.device)
        for (k1, k2, k3, k4), vec in zip(window_coords, window_vectors):
            d1_start = k1 * self.rank
            d1_end = min(d1_start + self.rank, D1)
            d2_start = k2 * self.rank
            d2_end = min(d2_start + self.rank, D2)
            d3_start = k3 * self.rank
            d3_end = min(d3_start + self.rank, D3)
            d4_start = k4 * self.rank
            d4_end = min(d4_start + self.rank, D4)
            hypervol[d1_start:d1_end, d2_start:d2_end, d3_start:d3_end, d4_start:d4_end, :] = vec
        return hypervol.detach().cpu().numpy()

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
    print("Spatial Numerical Dual Descriptor PM4 - 4D Hypervolume Version")
    print("Processes 4D arrays of m-dimensional real vectors with 4D basis functions")
    print("="*50)
    
    # Set random seeds
    torch.manual_seed(11)
    random.seed(11)
    np.random.seed(11)
    
    # Parameters
    vec_dim = 4         # Dimension of vectors (reduced for speed in 4D)
    rank = 2            # Window size (hypercube side length)
    user_step = 2       # Step size for nonlinear mode (non-overlapping)
    
    # Initialize model
    ndd = SpatialNumDualDescriptorPM4(
        vec_dim=vec_dim,
        rank=rank,
        mode='nonlinear',
        user_step=user_step,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    print(f"\nUsing device: {ndd.device}")
    print(f"Vector dimension: {vec_dim}")
    print(f"Window size: {rank} x {rank} x {rank} x {rank}")
    print(f"P matrix shape: {ndd.P.shape}")
    print(f"M matrix shape: {ndd.M.weight.shape}")
    
    # Generate 30 random 4D hypervolumes with random target vectors (reduced for speed)
    print("\nGenerating training data (4D hypervolumes)...")
    hypervols = []
    t_list = []
    for _ in range(30):
        D1 = random.randint(6, 8)   # dimension1
        D2 = random.randint(6, 8)   # dimension2
        D3 = random.randint(6, 8)   # dimension3
        D4 = random.randint(6, 8)   # dimension4
        hypervol = torch.randn(D1, D2, D3, D4, vec_dim)  # random vectors
        hypervols.append(hypervol)
        t_list.append(np.random.uniform(-1.0, 1.0, vec_dim))
    
    # Regression training
    print("\n" + "="*50)
    print("Starting Regression Training")
    print("="*50)
    ndd.reg_train(hypervols, t_list, max_iters=20, tol=1e-9, learning_rate=1.0, decay_rate=0.95, batch_size=4)
    
    # Predict target of first hypervolume
    first_vol = hypervols[0]
    t_pred = ndd.predict_t(first_vol)
    print(f"\nPredicted t for first hypervolume: {[round(x, 4) for x in t_pred[:5]]}...")
    
    # Correlation between predicted and real targets
    print("\nCalculating prediction correlations...")
    pred_t_list = [ndd.predict_t(v) for v in hypervols]
    corr_sum = 0.0
    for i in range(vec_dim):
        actu = [t[i] for t in t_list]
        pred = [t[i] for t in pred_t_list]
        corr, _ = pearsonr(actu, pred)
        print(f"Dimension {i} correlation: {corr:.4f}")
        corr_sum += corr
    print(f"Average correlation: {corr_sum / vec_dim:.4f}")
    
    # Reconstruction
    print("\nGenerating reconstructed hypervolumes...")
    recon_det = ndd.reconstruct(D1=6, D2=6, D3=6, D4=6, tau=0.0)
    recon_rand = ndd.reconstruct(D1=6, D2=6, D3=6, D4=6, tau=0.5)
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
        for _ in range(20):  # 20 volumes per class
            D1 = random.randint(5, 7)
            D2 = random.randint(5, 7)
            D3 = random.randint(5, 7)
            D4 = random.randint(5, 7)
            if class_id == 0:
                vol = torch.randn(D1, D2, D3, D4, vec_dim) + 1.0
            elif class_id == 1:
                vol = torch.randn(D1, D2, D3, D4, vec_dim) - 1.0
            else:
                vol = torch.randn(D1, D2, D3, D4, vec_dim)
            class_vols.append(vol)
            class_labels.append(class_id)
    
    ndd_cls = SpatialNumDualDescriptorPM4(vec_dim=vec_dim, rank=rank, mode='nonlinear', user_step=user_step)
    print("\nStarting Classification Training")
    history = ndd_cls.cls_train(class_vols, class_labels, num_classes, max_iters=15, learning_rate=0.05,
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
    num_labels = 2   # reduced for speed
    ml_vols = []
    ml_labels = []
    for _ in range(30):
        D1 = random.randint(5, 7)
        D2 = random.randint(5, 7)
        D3 = random.randint(5, 7)
        D4 = random.randint(5, 7)
        vol = torch.randn(D1, D2, D3, D4, vec_dim)
        ml_vols.append(vol)
        label_vec = [1.0 if random.random() > 0.7 else 0.0 for _ in range(num_labels)]
        ml_labels.append(label_vec)
    
    ndd_lbl = SpatialNumDualDescriptorPM4(vec_dim=vec_dim, rank=rank, mode='nonlinear', user_step=user_step)
    loss_hist, acc_hist = ndd_lbl.lbl_train(ml_vols, ml_labels, num_labels, max_iters=15,
                                             learning_rate=0.05, decay_rate=0.99, batch_size=4, print_every=5)
    print(f"Multi-label final training loss: {loss_hist[-1]:.6f}, accuracy: {acc_hist[-1]:.4f}")
    
    test_vol = torch.randn(6, 6, 6, 6, vec_dim)
    bin_pred, prob_pred = ndd_lbl.predict_l(test_vol, threshold=0.5)
    print(f"\nTest hypervolume prediction: {bin_pred}, probabilities: {prob_pred}")
    
    # Self-training example (with equal-sized volumes to trigger fast path)
    print("\n" + "="*50)
    print("Self-Training Example (with equal-sized hypervolumes for fast path)")
    print("="*50)
    ndd_self = SpatialNumDualDescriptorPM4(vec_dim=vec_dim, rank=rank, mode='nonlinear', user_step=user_step)
    # Generate equal-sized hypervolumes for fast batch processing
    fixed_d1, fixed_d2, fixed_d3, fixed_d4 = 5, 5, 5, 5
    self_seqs = [torch.randn(fixed_d1, fixed_d2, fixed_d3, fixed_d4, vec_dim) for _ in range(4)]
    self_history = ndd_self.self_train(self_seqs, max_iters=8, learning_rate=0.01, batch_size=2)
    plt.figure(figsize=(8,5))
    plt.plot(self_history)
    plt.title('Self-Training Loss (4D) - Fast Path')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig('self_training_loss_4d.png')
    print("Self-training loss plot saved as 'self_training_loss_4d.png'")
    
    print("\nAll tests completed successfully!")
    print("="*50)
