# Copyright (C) 2005-2026, Bin-Guang Ma (mbg@mail.hzau.edu.cn); SPDX-License-Identifier: MIT
# The Spatial Numerical Dual Descriptor Vector class (P Matrix form) for 4D array implemented with PyTorch
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

        # Precomputed phi table for all possible (k1,k2,k3,k4) pairs (initially None, built when needed)
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

    def _precompute_phi_table(self, max_k1, max_k2, max_k3, max_k4):
        """
        Precompute phi values for all k1 = 0..max_k1-1, k2 = 0..max_k2-1, k3 = 0..max_k3-1, k4 = 0..max_k4-1
        and store as buffer. This significantly speeds up batch_compute_Nk by avoiding repeated trig calls.
        
        Args:
            max_k1: Number of possible k1 indices (largest index + 1)
            max_k2: Number of possible k2 indices (largest index + 1)
            max_k3: Number of possible k3 indices (largest index + 1)
            max_k4: Number of possible k4 indices (largest index + 1)
        """
        if self.phi_table is not None and \
           self.phi_table.size(0) >= max_k1 and \
           self.phi_table.size(1) >= max_k2 and \
           self.phi_table.size(2) >= max_k3 and \
           self.phi_table.size(3) >= max_k4:
            return  # already have enough precomputed values
        # Allocate and compute: shape [max_k1, max_k2, max_k3, max_k4, vec_dim, vec_dim]
        phi_table = torch.zeros(max_k1, max_k2, max_k3, max_k4, self.vec_dim, self.vec_dim,
                                device=self.device, dtype=torch.float32)
        for k1 in range(max_k1):
            cos1 = torch.cos(2 * math.pi * k1 / self.periods1)  # [vec_dim, vec_dim]
            for k2 in range(max_k2):
                cos12 = cos1 * torch.cos(2 * math.pi * k2 / self.periods2)  # [vec_dim, vec_dim]
                for k3 in range(max_k3):
                    cos123 = cos12 * torch.cos(2 * math.pi * k3 / self.periods3)  # [vec_dim, vec_dim]
                    for k4 in range(max_k4):
                        phi_table[k1, k2, k3, k4] = cos123 * torch.cos(2 * math.pi * k4 / self.periods4)
        self.register_buffer('phi_table', phi_table)

    def _get_window_starts_4d(self, D1, D2, D3, D4):
        """
        Compute start indices of windows in 4D based on mode and rank_mode.
        Returns four lists: k1_starts, k2_starts, k3_starts, k4_starts.
        """
        if self.mode == 'linear':
            step = 1
        else:
            step = self.step if self.step is not None else self.rank
        
        if self.rank_mode == 'drop':
            k1_starts = list(range(0, D1 - self.rank + 1, step))
            k2_starts = list(range(0, D2 - self.rank + 1, step))
            k3_starts = list(range(0, D3 - self.rank + 1, step))
            k4_starts = list(range(0, D4 - self.rank + 1, step))
        else:  # 'pad'
            k1_starts = list(range(0, D1, step))
            k2_starts = list(range(0, D2, step))
            k3_starts = list(range(0, D3, step))
            k4_starts = list(range(0, D4, step))
        return k1_starts, k2_starts, k3_starts, k4_starts

    def _compute_Nk_from_x_4d(self, k1_tensor, k2_tensor, k3_tensor, k4_tensor, x):
        """
        Compute N(k1,k2,k3,k4) vectors directly from mapped vectors x (already transformed by M).
        This avoids double mapping in batch processing.
        
        Args:
            k1_tensor: Tensor of dim1 indices [batch_size] (long)
            k2_tensor: Tensor of dim2 indices [batch_size] (long)
            k3_tensor: Tensor of dim3 indices [batch_size] (long)
            k4_tensor: Tensor of dim4 indices [batch_size] (long)
            x: Tensor of mapped vectors [batch_size, vec_dim]
        
        Returns:
            Nk: Tensor [batch_size, vec_dim]
        """
        # Use precomputed phi table if available
        if self.phi_table is not None:
            # phi: [batch_size, vec_dim, vec_dim]
            phi = self.phi_table[k1_tensor, k2_tensor, k3_tensor, k4_tensor]
        else:
            # Fallback: compute on the fly (slower, but works)
            k1_exp = k1_tensor.view(-1, 1, 1).float()
            k2_exp = k2_tensor.view(-1, 1, 1).float()
            k3_exp = k3_tensor.view(-1, 1, 1).float()
            k4_exp = k4_tensor.view(-1, 1, 1).float()
            phi = (torch.cos(2 * math.pi * k1_exp / self.periods1) *
                   torch.cos(2 * math.pi * k2_exp / self.periods2) *
                   torch.cos(2 * math.pi * k3_exp / self.periods3) *
                   torch.cos(2 * math.pi * k4_exp / self.periods4))
        # Nk = sum_{i,j} P[i,j] * x_j * phi_{i,j}
        Nk = torch.einsum('bj,ij,bij->bi', x, self.P, phi)
        return Nk

    def extract_windows(self, hypervol):
        """
        Extract hypercubic windows from a 4D volume and apply rank operation to each window.
        
        Args:
            hypervol: torch.Tensor of shape (D1, D2, D3, D4, vec_dim) or nested list of vectors
            
        Returns:
            tuple: (window_vectors, window_coords)
                window_vectors: torch.Tensor of shape (num_windows, vec_dim)
                window_coords: tuple of four tensors (k1_coords, k2_coords, k3_coords, k4_coords) each of shape (num_windows,)
                coordinates are returned as torch.long for indexing purposes.
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
        coords_k1 = torch.tensor(coords_k1, dtype=torch.long, device=self.device)
        coords_k2 = torch.tensor(coords_k2, dtype=torch.long, device=self.device)
        coords_k3 = torch.tensor(coords_k3, dtype=torch.long, device=self.device)
        coords_k4 = torch.tensor(coords_k4, dtype=torch.long, device=self.device)
        return window_vectors, (coords_k1, coords_k2, coords_k3, coords_k4)

    def batch_extract_windows_4d(self, vol_batch):
        """
        Batch extract windows from a batch of equal-sized 4D hypervolumes.
        Only works when rank_mode == 'drop'.
        
        Args:
            vol_batch: Tensor of shape [batch_size, D1, D2, D3, D4, vec_dim]
            
        Returns:
            tuple: (window_vectors_batch, coords_batch)
                window_vectors_batch: Tensor of shape [batch_size, n_windows, vec_dim]
                coords_batch: tuple of four tensors (k1_coords, k2_coords, k3_coords, k4_coords)
                              each of shape [batch_size, n_windows]
        """
        batch_size, D1, D2, D3, D4, m = vol_batch.shape
        k1_starts, k2_starts, k3_starts, k4_starts = self._get_window_starts_4d(D1, D2, D3, D4)
        n_windows_d1 = len(k1_starts)
        n_windows_d2 = len(k2_starts)
        n_windows_d3 = len(k3_starts)
        n_windows_d4 = len(k4_starts)
        n_windows = n_windows_d1 * n_windows_d2 * n_windows_d3 * n_windows_d4
        
        step = 1 if self.mode == 'linear' else (self.step if self.step is not None else self.rank)
        
        # Prepare containers
        all_windows = []   # list of [batch, n_windows_34, m] tensors for each (k1,k2) pair
        all_k1 = []
        all_k2 = []
        all_k3 = []
        all_k4 = []
        
        # Loop over first two dimensions (the two that will be handled by Python loop)
        for k1 in k1_starts:
            for k2 in k2_starts:
                # Extract sub-volume for fixed (k1,k2) window
                subvol = vol_batch[:, k1:k1+self.rank, k2:k2+self.rank, :, :, :]  # [batch, rank, rank, D3, D4, m]
                # Merge batch and the first two rank dimensions for 2D unfold
                subvol_flat = subvol.reshape(batch_size * self.rank * self.rank, D3, D4, m)
                subvol_perm = subvol_flat.permute(0, 3, 1, 2)  # [batch*rank^2, m, D3, D4]
                
                # Extract windows over dimensions 3 and 4
                windows_34 = torch.nn.functional.unfold(subvol_perm, kernel_size=(self.rank, self.rank), stride=(step, step))
                n_windows_34 = windows_34.shape[2]
                # Reshape back to batch structure
                windows_34 = windows_34.view(batch_size, self.rank, self.rank, m, self.rank, self.rank, n_windows_34)
                windows_34 = windows_34.permute(0, 6, 1, 2, 4, 5, 3).contiguous()
                windows_34 = windows_34.view(batch_size, n_windows_34, self.rank*self.rank*self.rank*self.rank, m)
                
                # Apply mapping M
                mapped = self.M(windows_34)  # [batch, n_windows_34, rank^4, m]
                # Aggregate over rank dimension
                if self.rank_op == 'sum':
                    agg = mapped.sum(dim=2)
                elif self.rank_op == 'pick':
                    idx = torch.randint(0, self.rank*self.rank*self.rank*self.rank,
                                        (batch_size, n_windows_34, 1), device=self.device)
                    agg = torch.gather(mapped, 2, idx.expand(-1, -1, -1, m)).squeeze(2)
                elif self.rank_op == 'user_func':
                    avg = mapped.mean(dim=2)
                    agg = torch.sigmoid(avg)
                else:  # 'avg'
                    agg = mapped.mean(dim=2)  # [batch, n_windows_34, m]
                all_windows.append(agg)
                
                # Record coordinates for this (k1,k2) pair
                for k3 in k3_starts:
                    for k4 in k4_starts:
                        all_k1.append(k1)
                        all_k2.append(k2)
                        all_k3.append(k3)
                        all_k4.append(k4)
        
        # Concatenate all windows across (k1,k2) pairs
        windows = torch.cat(all_windows, dim=1)  # [batch, n_windows, m]
        
        # Build coordinate tensors for all windows
        k1_tensor = torch.tensor(all_k1, dtype=torch.long, device=self.device)  # [n_windows]
        k2_tensor = torch.tensor(all_k2, dtype=torch.long, device=self.device)
        k3_tensor = torch.tensor(all_k3, dtype=torch.long, device=self.device)
        k4_tensor = torch.tensor(all_k4, dtype=torch.long, device=self.device)
        # Expand to batch dimension
        k1_exp = k1_tensor.unsqueeze(0).expand(batch_size, -1)
        k2_exp = k2_tensor.unsqueeze(0).expand(batch_size, -1)
        k3_exp = k3_tensor.unsqueeze(0).expand(batch_size, -1)
        k4_exp = k4_tensor.unsqueeze(0).expand(batch_size, -1)
        
        return windows, (k1_exp, k2_exp, k3_exp, k4_exp)

    def batch_represent_4d(self, vol_batch):
        """
        Compute hypervolume representations for a batch of 4D volumes efficiently.
        Supports arbitrary window size and step as long as all volumes in the batch have
        identical dimensions, and rank_mode == 'drop'.
        
        Args:
            vol_batch: Tensor of shape [batch_size, D1, D2, D3, D4, vec_dim]
        
        Returns:
            representations: Tensor of shape [batch_size, vec_dim]
        """
        # Use the batch extraction method
        windows, (k1_exp, k2_exp, k3_exp, k4_exp) = self.batch_extract_windows_4d(vol_batch)
        batch_size, n_windows, m = windows.shape
        
        # Flatten to compute all Nk in one go
        windows_flat = windows.reshape(batch_size * n_windows, m)
        k1_flat = k1_exp.reshape(batch_size * n_windows)
        k2_flat = k2_exp.reshape(batch_size * n_windows)
        k3_flat = k3_exp.reshape(batch_size * n_windows)
        k4_flat = k4_exp.reshape(batch_size * n_windows)
        
        # Compute Nk
        Nk_flat = self._compute_Nk_from_x_4d(k1_flat, k2_flat, k3_flat, k4_flat, windows_flat)
        Nk = Nk_flat.view(batch_size, n_windows, m)
        
        # Average over windows
        rep = Nk.mean(dim=1)  # [batch, m]
        return rep

    def batch_compute_Nk_and_targets_4d(self, vol_batch):
        """
        Compute Nk vectors and target vectors for each window in a batch of volumes.
        Supports arbitrary window size and step, but only for rank_mode == 'drop'.
        
        Args:
            vol_batch: Tensor of shape [batch_size, D1, D2, D3, D4, m]
        
        Returns:
            Nk_all: Tensor [batch_size, n_windows, m] - Nk vectors for each window
            targets: Tensor [batch_size, n_windows, m] - target vectors (aggregated mapped vectors)
        """
        batch_size, D1, D2, D3, D4, m = vol_batch.shape
        k1_starts, k2_starts, k3_starts, k4_starts = self._get_window_starts_4d(D1, D2, D3, D4)
        if not k1_starts or not k2_starts or not k3_starts or not k4_starts:
            return (torch.empty(batch_size, 0, m, device=self.device),
                    torch.empty(batch_size, 0, m, device=self.device))
        
        n_windows_d1 = len(k1_starts)
        n_windows_d2 = len(k2_starts)
        n_windows_d3 = len(k3_starts)
        n_windows_d4 = len(k4_starts)
        n_windows = n_windows_d1 * n_windows_d2 * n_windows_d3 * n_windows_d4
        
        step = 1 if self.mode == 'linear' else (self.step if self.step is not None else self.rank)
        
        # Collect all targets across dim1 and dim2
        all_targets = []
        all_k1 = []
        all_k2 = []
        all_k3 = []
        all_k4 = []
        
        for k1 in k1_starts:
            for k2 in k2_starts:
                subvol = vol_batch[:, k1:k1+self.rank, k2:k2+self.rank, :, :, :]  # [batch, rank, rank, D3, D4, m]
                subvol_flat = subvol.reshape(batch_size * self.rank * self.rank, D3, D4, m)
                subvol_perm = subvol_flat.permute(0, 3, 1, 2)
                
                windows_34 = torch.nn.functional.unfold(subvol_perm, kernel_size=(self.rank, self.rank), stride=(step, step))
                n_windows_34 = windows_34.shape[2]
                windows_34 = windows_34.view(batch_size, self.rank, self.rank, m, self.rank, self.rank, n_windows_34)
                windows_34 = windows_34.permute(0, 6, 1, 2, 4, 5, 3).contiguous()
                windows_34 = windows_34.view(batch_size, n_windows_34, self.rank*self.rank*self.rank*self.rank, m)
                
                mapped = self.M(windows_34)  # [batch, n_windows_34, rank^4, m]
                
                # Aggregate to get targets
                if self.rank_op == 'sum':
                    targets = mapped.sum(dim=2)
                elif self.rank_op == 'pick':
                    idx = torch.randint(0, self.rank*self.rank*self.rank*self.rank,
                                        (batch_size, n_windows_34, 1), device=self.device)
                    targets = torch.gather(mapped, 2, idx.expand(-1, -1, -1, m)).squeeze(2)
                elif self.rank_op == 'user_func':
                    avg = mapped.mean(dim=2)
                    targets = torch.sigmoid(avg)
                else:  # 'avg'
                    targets = mapped.mean(dim=2)
                
                all_targets.append(targets)  # [batch, n_windows_34, m]
                for k3 in k3_starts:
                    for k4 in k4_starts:
                        all_k1.append(k1)
                        all_k2.append(k2)
                        all_k3.append(k3)
                        all_k4.append(k4)
        
        targets = torch.cat(all_targets, dim=1)  # [batch, n_windows, m]
        
        # Build k tensors
        k1_tensor = torch.tensor(all_k1, dtype=torch.long, device=self.device)
        k2_tensor = torch.tensor(all_k2, dtype=torch.long, device=self.device)
        k3_tensor = torch.tensor(all_k3, dtype=torch.long, device=self.device)
        k4_tensor = torch.tensor(all_k4, dtype=torch.long, device=self.device)
        k1_expanded = k1_tensor.unsqueeze(0).expand(batch_size, -1)
        k2_expanded = k2_tensor.unsqueeze(0).expand(batch_size, -1)
        k3_expanded = k3_tensor.unsqueeze(0).expand(batch_size, -1)
        k4_expanded = k4_tensor.unsqueeze(0).expand(batch_size, -1)
        
        # Flatten and compute Nk
        targets_flat = targets.reshape(batch_size * n_windows, m)
        k1_flat = k1_expanded.reshape(batch_size * n_windows)
        k2_flat = k2_expanded.reshape(batch_size * n_windows)
        k3_flat = k3_expanded.reshape(batch_size * n_windows)
        k4_flat = k4_expanded.reshape(batch_size * n_windows)
        Nk_flat = self._compute_Nk_from_x_4d(k1_flat, k2_flat, k3_flat, k4_flat, targets_flat)
        Nk_all = Nk_flat.view(batch_size, n_windows, m)
        
        return Nk_all, targets

    def batch_compute_Nk(self, k1_tensor, k2_tensor, k3_tensor, k4_tensor, vectors):
        """
        Vectorized computation of N(k1,k2,k3,k4) vectors for a batch of positions and vectors
        N(k1,k2,k3,k4) = sum_{i,j} P_{i,j} * (M(v))_i * cos(2π*k1/periods1[i,j]) * cos(2π*k2/periods2[i,j]) * cos(2π*k3/periods3[i,j]) * cos(2π*k4/periods4[i,j])
        
        Args:
            k1_tensor: Tensor of dim1 indices [batch_size] (long)
            k2_tensor: Tensor of dim2 indices [batch_size] (long)
            k3_tensor: Tensor of dim3 indices [batch_size] (long)
            k4_tensor: Tensor of dim4 indices [batch_size] (long)
            vectors: Tensor of window vectors [batch_size, vec_dim]
            
        Returns:
            Tensor of N(k1,k2,k3,k4) vectors [batch_size, vec_dim]
        """
        # Apply square mapping matrix M to each vector
        x = self.M(vectors)  # [batch_size, vec_dim]
        return self._compute_Nk_from_x_4d(k1_tensor, k2_tensor, k3_tensor, k4_tensor, x)

    def compute_Nk(self, k1, k2, k3, k4, vector):
        """Compute N(k1,k2,k3,k4) for single position and vector (uses batch internally)"""
        if not isinstance(vector, torch.Tensor):
            vector = torch.tensor(vector, dtype=torch.float32, device=self.device)
        elif vector.device != self.device:
            vector = vector.to(self.device)
        
        k1_t = torch.tensor([k1], dtype=torch.long, device=self.device)
        k2_t = torch.tensor([k2], dtype=torch.long, device=self.device)
        k3_t = torch.tensor([k3], dtype=torch.long, device=self.device)
        k4_t = torch.tensor([k4], dtype=torch.long, device=self.device)
        vec_t = vector.unsqueeze(0)
        result = self.batch_compute_Nk(k1_t, k2_t, k3_t, k4_t, vec_t)
        return result[0]

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

    def reg_train(self, hypervols, t_list, max_iters=1000, tol=1e-8, learning_rate=0.01,
                  continued=False, decay_rate=1.0, print_every=10, batch_size=32,
                  checkpoint_file=None, checkpoint_interval=10):
        """
        Train the model using gradient descent with hypervolume-level batch processing.
        Optimized for GPU memory efficiency by processing hypervolumes individually, but
        uses fast batch processing when all hypervolumes in a batch have the same dimensions
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
        
        # Ensure all hypervolumes are on the correct device and are tensors
        hypervols = [v.to(self.device) if isinstance(v, torch.Tensor) else 
                     torch.tensor(v, dtype=torch.float32, device=self.device) for v in hypervols]
        
        # Determine if all hypervolumes have the same shape (for fast pre-extraction)
        vol_shapes = [v.shape for v in hypervols]
        all_equal = len(set(vol_shapes)) == 1
        use_fast_batch = (self.rank_mode == 'drop')
        
        # Pre-extract windows
        if use_fast_batch and all_equal:
            # All volumes equal length, use batch extraction
            print("Using batch extraction for pre-processing (fast path).")
            batch_vol_tensor = torch.stack(hypervols, dim=0)  # [num, D1, D2, D3, D4, m]
            windows_batch, coords_batch = self.batch_extract_windows_4d(batch_vol_tensor)
            # Convert to list format for compatibility
            extracted_list = []
            for i in range(len(hypervols)):
                extracted_list.append((windows_batch[i], (coords_batch[0][i], coords_batch[1][i], coords_batch[2][i], coords_batch[3][i])))
        else:
            # Extract individually
            extracted_list = []
            for vol in hypervols:
                window_vectors, (k1_coords, k2_coords, k3_coords, k4_coords) = self.extract_windows(vol)
                extracted_list.append((window_vectors, (k1_coords, k2_coords, k3_coords, k4_coords)))
        
        # Record max coordinates for phi table precomputation
        max_k1 = max_k2 = max_k3 = max_k4 = 0
        for window_vectors, (k1_coords, k2_coords, k3_coords, k4_coords) in extracted_list:
            if window_vectors.shape[0] > 0:
                max_k1 = max(max_k1, k1_coords.max().item() + 1)
                max_k2 = max(max_k2, k2_coords.max().item() + 1)
                max_k3 = max(max_k3, k3_coords.max().item() + 1)
                max_k4 = max(max_k4, k4_coords.max().item() + 1)
        max_k1 = max(1, max_k1)
        max_k2 = max(1, max_k2)
        max_k3 = max(1, max_k3)
        max_k4 = max(1, max_k4)
        self._precompute_phi_table(max_k1, max_k2, max_k3, max_k4)
        
        # Convert target vectors to tensor
        t_tensors = [torch.tensor(t, dtype=torch.float32, device=self.device) for t in t_list]
        
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)
        
        history = []
        prev_loss = float('inf')
        best_loss = float('inf')
        best_model_state = None
        
        for it in range(max_iters):
            total_loss = 0.0
            total_vols = 0
            
            indices = list(range(len(hypervols)))
            random.shuffle(indices)
            
            for batch_start in range(0, len(indices), batch_size):
                batch_indices = indices[batch_start:batch_start + batch_size]
                batch_vols = [hypervols[idx] for idx in batch_indices]
                batch_targets = [t_tensors[idx] for idx in batch_indices]
                batch_extracted = [extracted_list[idx] for idx in batch_indices]
                
                optimizer.zero_grad()
                batch_loss = 0.0
                
                # Check if all volumes in this batch have the same dimensions
                vol_shapes_batch = [v.shape for v in batch_vols]
                if use_fast_batch and len(set(vol_shapes_batch)) == 1:
                    # Fast batch processing
                    batch_vol_tensor = torch.stack(batch_vols, dim=0)  # [batch, D1, D2, D3, D4, m]
                    reps = self.batch_represent_4d(batch_vol_tensor)  # [batch, m]
                    targets_tensor = torch.stack(batch_targets, dim=0)  # [batch, m]
                    loss = torch.mean(torch.sum((reps - targets_tensor) ** 2, dim=1))
                    loss.backward()
                    optimizer.step()
                    batch_loss = loss.item() * len(batch_vols)
                    total_loss += batch_loss
                    total_vols += len(batch_vols)
                else:
                    # Process each volume individually using pre-extracted windows
                    for (window_vectors, (k1_coords, k2_coords, k3_coords, k4_coords)), target in zip(batch_extracted, batch_targets):
                        if window_vectors.shape[0] == 0:
                            continue
                        N_batch = self.batch_compute_Nk(k1_coords, k2_coords, k3_coords, k4_coords, window_vectors)
                        vol_pred = torch.mean(N_batch, dim=0)
                        seq_loss = torch.sum((vol_pred - target) ** 2)
                        batch_loss += seq_loss
                        del N_batch, vol_pred
                    
                    if len(batch_extracted) > 0:
                        batch_loss = batch_loss / len(batch_extracted)
                        batch_loss.backward()
                        optimizer.step()
                        total_loss += batch_loss.item() * len(batch_extracted)
                        total_vols += len(batch_extracted)
                
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
        
        self._compute_training_statistics(hypervols, extracted_list)
        self.trained = True
        return history

    def cls_train(self, hypervols, labels, num_classes, max_iters=1000, tol=1e-8, learning_rate=0.01,
                  continued=False, decay_rate=1.0, print_every=10, batch_size=32,
                  checkpoint_file=None, checkpoint_interval=10):
        """
        Train the model for multi-class classification using cross-entropy loss.
        Optimized with fast batch processing for equal-sized hypervolumes.
        """
        if self.classifier is None or self.num_classes != num_classes:
            self.classifier = nn.Linear(self.vec_dim, num_classes).to(self.device)
            self.num_classes = num_classes
        
        if not continued:
            self.reset_parameters()
        
        # Ensure all hypervolumes are on the correct device
        hypervols = [v.to(self.device) if isinstance(v, torch.Tensor) else 
                     torch.tensor(v, dtype=torch.float32, device=self.device) for v in hypervols]
        
        # Determine if all hypervolumes have the same shape (for fast pre-extraction)
        vol_shapes = [v.shape for v in hypervols]
        all_equal = len(set(vol_shapes)) == 1
        use_fast_batch = (self.rank_mode == 'drop')
        
        # Pre-extract windows
        if use_fast_batch and all_equal:
            print("Using batch extraction for pre-processing (fast path).")
            batch_vol_tensor = torch.stack(hypervols, dim=0)
            windows_batch, coords_batch = self.batch_extract_windows_4d(batch_vol_tensor)
            extracted_list = []
            for i in range(len(hypervols)):
                extracted_list.append((windows_batch[i], (coords_batch[0][i], coords_batch[1][i], coords_batch[2][i], coords_batch[3][i])))
        else:
            extracted_list = []
            for vol in hypervols:
                window_vectors, coords = self.extract_windows(vol)
                extracted_list.append((window_vectors, coords))
        
        # Record max coordinates for phi table
        max_k1 = max_k2 = max_k3 = max_k4 = 0
        for window_vectors, (k1_coords, k2_coords, k3_coords, k4_coords) in extracted_list:
            if window_vectors.shape[0] > 0:
                max_k1 = max(max_k1, k1_coords.max().item() + 1)
                max_k2 = max(max_k2, k2_coords.max().item() + 1)
                max_k3 = max(max_k3, k3_coords.max().item() + 1)
                max_k4 = max(max_k4, k4_coords.max().item() + 1)
        max_k1 = max(1, max_k1)
        max_k2 = max(1, max_k2)
        max_k3 = max(1, max_k3)
        max_k4 = max(1, max_k4)
        self._precompute_phi_table(max_k1, max_k2, max_k3, max_k4)
        
        label_tensors = torch.tensor(labels, dtype=torch.long, device=self.device)
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)
        criterion = nn.CrossEntropyLoss()
        
        history = []
        prev_loss = float('inf')
        best_loss = float('inf')
        best_model_state = None
        
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
                batch_extracted = [extracted_list[idx] for idx in batch_indices]
                
                optimizer.zero_grad()
                
                vol_shapes_batch = [v.shape for v in batch_vols]
                if use_fast_batch and len(set(vol_shapes_batch)) == 1:
                    # Fast path
                    batch_vol_tensor = torch.stack(batch_vols, dim=0)
                    reps = self.batch_represent_4d(batch_vol_tensor)
                    logits = self.classifier(reps)
                    loss = criterion(logits, batch_labels)
                    loss.backward()
                    optimizer.step()
                    batch_loss = loss.item() * len(batch_vols)
                    total_loss += batch_loss
                    total_vols += len(batch_vols)
                    with torch.no_grad():
                        pred = torch.argmax(logits, dim=1)
                        correct += (pred == batch_labels).sum().item()
                else:
                    # Slow path: process individually
                    batch_logits = []
                    for (window_vectors, (k1_coords, k2_coords, k3_coords, k4_coords)) in batch_extracted:
                        if window_vectors.shape[0] == 0:
                            seq_vector = torch.zeros(self.vec_dim, device=self.device)
                        else:
                            N_batch = self.batch_compute_Nk(k1_coords, k2_coords, k3_coords, k4_coords, window_vectors)
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
        Optimized with fast batch processing for equal-sized hypervolumes.
        """
        if self.labeller is None or self.num_labels != num_labels:
            self.labeller = nn.Linear(self.vec_dim, num_labels).to(self.device)
            self.num_labels = num_labels
        
        if not continued:
            self.reset_parameters()
        
        # Ensure all hypervolumes are on the correct device
        hypervols = [v.to(self.device) if isinstance(v, torch.Tensor) else 
                     torch.tensor(v, dtype=torch.float32, device=self.device) for v in hypervols]
        
        # Determine if all hypervolumes have the same shape (for fast pre-extraction)
        vol_shapes = [v.shape for v in hypervols]
        all_equal = len(set(vol_shapes)) == 1
        use_fast_batch = (self.rank_mode == 'drop')
        
        # Pre-extract windows
        if use_fast_batch and all_equal:
            print("Using batch extraction for pre-processing (fast path).")
            batch_vol_tensor = torch.stack(hypervols, dim=0)
            windows_batch, coords_batch = self.batch_extract_windows_4d(batch_vol_tensor)
            extracted_list = []
            for i in range(len(hypervols)):
                extracted_list.append((windows_batch[i], (coords_batch[0][i], coords_batch[1][i], coords_batch[2][i], coords_batch[3][i])))
        else:
            extracted_list = []
            for vol in hypervols:
                window_vectors, coords = self.extract_windows(vol)
                extracted_list.append((window_vectors, coords))
        
        # Record max coordinates for phi table
        max_k1 = max_k2 = max_k3 = max_k4 = 0
        for window_vectors, (k1_coords, k2_coords, k3_coords, k4_coords) in extracted_list:
            if window_vectors.shape[0] > 0:
                max_k1 = max(max_k1, k1_coords.max().item() + 1)
                max_k2 = max(max_k2, k2_coords.max().item() + 1)
                max_k3 = max(max_k3, k3_coords.max().item() + 1)
                max_k4 = max(max_k4, k4_coords.max().item() + 1)
        max_k1 = max(1, max_k1)
        max_k2 = max(1, max_k2)
        max_k3 = max(1, max_k3)
        max_k4 = max(1, max_k4)
        self._precompute_phi_table(max_k1, max_k2, max_k3, max_k4)
        
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
                batch_extracted = [extracted_list[idx] for idx in batch_indices]
                
                optimizer.zero_grad()
                
                vol_shapes_batch = [v.shape for v in batch_vols]
                if use_fast_batch and len(set(vol_shapes_batch)) == 1:
                    # Fast path
                    batch_vol_tensor = torch.stack(batch_vols, dim=0)
                    reps = self.batch_represent_4d(batch_vol_tensor)
                    logits = self.labeller(reps)
                    loss = criterion(logits, batch_labels)
                    loss.backward()
                    optimizer.step()
                    batch_loss = loss.item() * len(batch_vols)
                    total_loss += batch_loss
                    total_vols += len(batch_vols)
                    
                    with torch.no_grad():
                        probs = torch.sigmoid(logits)
                        preds = (probs > 0.5).float()
                        batch_correct = (preds == batch_labels).sum().item()
                        batch_predictions = batch_labels.numel()
                        total_correct += batch_correct
                        total_predictions += batch_predictions
                else:
                    # Slow path: process individually
                    batch_logits = []
                    for (window_vectors, (k1_coords, k2_coords, k3_coords, k4_coords)) in batch_extracted:
                        if window_vectors.shape[0] == 0:
                            continue
                        N_batch = self.batch_compute_Nk(k1_coords, k2_coords, k3_coords, k4_coords, window_vectors)
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
        Uses fast batch processing when all hypervolumes have the same dimensions and rank_mode == 'drop'.
        """
        if not continued:
            self.reset_parameters()
        
        # Ensure all hypervolumes are on the correct device
        hypervols = [v.to(self.device) if isinstance(v, torch.Tensor) else 
                     torch.tensor(v, dtype=torch.float32, device=self.device) for v in hypervols]
        
        # Determine if all hypervolumes have the same shape (for fast pre-extraction)
        vol_shapes = [v.shape for v in hypervols]
        all_equal = len(set(vol_shapes)) == 1
        use_fast_batch = (self.rank_mode == 'drop')
        
        # Pre-extract windows
        if use_fast_batch and all_equal:
            print("Using batch extraction for pre-processing (fast path).")
            batch_vol_tensor = torch.stack(hypervols, dim=0)
            windows_batch, coords_batch = self.batch_extract_windows_4d(batch_vol_tensor)
            extracted_list = []
            for i in range(len(hypervols)):
                extracted_list.append((windows_batch[i], (coords_batch[0][i], coords_batch[1][i], coords_batch[2][i], coords_batch[3][i])))
        else:
            extracted_list = []
            for vol in hypervols:
                window_vectors, coords = self.extract_windows(vol)
                extracted_list.append((window_vectors, coords))
        
        # Record max coordinates for phi table
        max_k1 = max_k2 = max_k3 = max_k4 = 0
        for window_vectors, (k1_coords, k2_coords, k3_coords, k4_coords) in extracted_list:
            if window_vectors.shape[0] > 0:
                max_k1 = max(max_k1, k1_coords.max().item() + 1)
                max_k2 = max(max_k2, k2_coords.max().item() + 1)
                max_k3 = max(max_k3, k3_coords.max().item() + 1)
                max_k4 = max(max_k4, k4_coords.max().item() + 1)
        max_k1 = max(1, max_k1)
        max_k2 = max(1, max_k2)
        max_k3 = max(1, max_k3)
        max_k4 = max(1, max_k4)
        self._precompute_phi_table(max_k1, max_k2, max_k3, max_k4)
        
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)
        
        history = []
        prev_loss = float('inf')
        best_loss = float('inf')
        best_model_state = None
        
        # Check if all volumes have the same dimensions (for fast batch processing)
        if use_fast_batch and all_equal:
            # Fast path: all volumes equal size, batch process all windows together
            D1, D2, D3, D4, _ = vol_shapes[0]  # Extract dimensions, ignore last (vec_dim)
            all_vols = torch.stack(hypervols, dim=0)  # [num_vols, D1, D2, D3, D4, m]
            num_vols = len(hypervols)
            seq_batch_size = batch_size  # number of volumes per batch
            
            for it in range(max_iters):
                total_loss = 0.0
                total_windows = 0
                
                indices = list(range(num_vols))
                random.shuffle(indices)
                
                for batch_start in range(0, num_vols, seq_batch_size):
                    batch_indices = indices[batch_start:batch_start + seq_batch_size]
                    batch_vols = all_vols[batch_indices]  # [batch, D1, D2, D3, D4, m]
                    
                    optimizer.zero_grad()
                    
                    # Compute Nk and targets for all windows in this batch of volumes
                    Nk_batch, targets_batch = self.batch_compute_Nk_and_targets_4d(batch_vols)
                    
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
                
                indices = list(range(len(hypervols)))
                random.shuffle(indices)
                
                for batch_start in range(0, len(indices), batch_size):
                    batch_indices = indices[batch_start:batch_start + batch_size]
                    batch_extracted = [extracted_list[idx] for idx in batch_indices]
                    
                    optimizer.zero_grad()
                    batch_loss = 0.0
                    batch_window_count = 0
                    
                    for (window_vectors, (k1_coords, k2_coords, k3_coords, k4_coords)) in batch_extracted:
                        if window_vectors.shape[0] == 0:
                            continue
                        
                        N_batch = self.batch_compute_Nk(k1_coords, k2_coords, k3_coords, k4_coords, window_vectors)
                        target_vectors = self.M(window_vectors)  # [num_windows, vec_dim]
                        
                        # Self-consistency loss: N(k1,k2,k3,k4) should match transformed window vector
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
        
        self._compute_training_statistics(hypervols, extracted_list)
        self.trained = True
        return history

    def _compute_training_statistics(self, hypervols, extracted_list=None, batch_size=50):
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
                for window_vectors, (k1_coords, k2_coords, k3_coords, k4_coords) in extracted_list:
                    if window_vectors.shape[0] == 0:
                        continue
                    N_batch = self.batch_compute_Nk(k1_coords, k2_coords, k3_coords, k4_coords, window_vectors)
                    total_window_count += window_vectors.shape[0]
                    total_t += N_batch.sum(dim=0)
            self.mean_window_count = total_window_count / len(hypervols) if hypervols else 0
            self.mean_t = (total_t / total_window_count).cpu().numpy() if total_window_count > 0 else np.zeros(self.vec_dim)
            return
        
        # Fallback to original slower method
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
        # Use batch_represent_4d for efficiency (single hypervolume)
        if not isinstance(hypervol, torch.Tensor):
            hypervol = torch.tensor(hypervol, dtype=torch.float32, device=self.device)
        elif hypervol.device != self.device:
            hypervol = hypervol.to(self.device)
        # Add batch dimension
        vol_batch = hypervol.unsqueeze(0)  # [1, D1, D2, D3, D4, m]
        rep = self.batch_represent_4d(vol_batch)  # [1, m]
        return rep[0].detach().cpu().numpy()

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
        
        if not isinstance(hypervol, torch.Tensor):
            hypervol = torch.tensor(hypervol, dtype=torch.float32, device=self.device)
        elif hypervol.device != self.device:
            hypervol = hypervol.to(self.device)
        
        vol_batch = hypervol.unsqueeze(0)  # [1, D1, D2, D3, D4, m]
        rep = self.batch_represent_4d(vol_batch)  # [1, m]
        seq_rep = rep[0]
        
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
                        k1_t = torch.full((num_candidates,), k1, dtype=torch.long, device=self.device)
                        k2_t = torch.full((num_candidates,), k2, dtype=torch.long, device=self.device)
                        k3_t = torch.full((num_candidates,), k3, dtype=torch.long, device=self.device)
                        k4_t = torch.full((num_candidates,), k4, dtype=torch.long, device=self.device)
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
    print("Spatial Numerical Dual Descriptor PM4 - 4D Hypervolume Version (Accelerated)")
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
    
    # Self-training example
    print("\n" + "="*50)
    print("Self-Training Example")
    print("="*50)
    ndd_self = SpatialNumDualDescriptorPM4(vec_dim=vec_dim, rank=rank, mode='nonlinear', user_step=user_step)
    self_seqs = [torch.randn(5, 5, 5, 5, vec_dim) for _ in range(4)]  # equal size for fast path
    self_history = ndd_self.self_train(self_seqs, max_iters=8, learning_rate=0.01, batch_size=2)
    plt.figure(figsize=(8,5))
    plt.plot(self_history)
    plt.title('Self-Training Loss (4D)')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig('self_training_loss_4d.png')
    print("Self-training loss plot saved as 'self_training_loss_4d.png'")
    
    print("\nAll tests completed successfully!")
    print("="*50)
