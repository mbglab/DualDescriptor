# Copyright (C) 2005-2026, Bin-Guang Ma (mbg@mail.hzau.edu.cn); SPDX-License-Identifier: MIT
# The Spatial Numerical Dual Descriptor Vector class (Tensor form) for 3D array implemented with PyTorch
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

class SpatialNumDualDescriptorTS3(nn.Module):
    """
    Spatial Numerical Dual Descriptor for 3D arrays with GPU acceleration using PyTorch:
      - Processes 3D grids of m-dimensional real vectors (H x W x D x m) instead of 1D sequences
      - tensor P ∈ R^{m×m×o} of basis coefficients
      - mapping matrix M ∈ R^{m×m} for vector transformation (square matrix)
      - indexed periods for three spatial dimensions:
          period1[i,j,g] = i*(m*o) + j*o + g + 2
          period2[i,j,g] = i*(m*o) + j*o + g + 3
          period3[i,j,g] = i*(m*o) + j*o + g + 4
      - basis function phi_{i,j,g}(k1,k2,k3) = cos(2π*k1/period1) * cos(2π*k2/period2) * cos(2π*k3/period3)
      - supports 'linear' or 'nonlinear' (step-by-step) window extraction
      - batch acceleration for equal-sized grids when rank_mode == 'drop'
    """
    def __init__(self, vec_dim, rank=1, rank_op='avg', rank_mode='drop', num_basis=5, mode='linear',
                 user_step=None, device='cuda'):
        """
        Initialize the Spatial Numerical Dual Descriptor model for 3D arrays.
        
        Args:
            vec_dim: Dimension of input vectors and internal representation
            rank: Window size (int or tuple (rank_h, rank_w, rank_d)); if int, use cubic windows of size rank^3
            rank_op: 'avg', 'sum', 'pick', 'user_func'
            rank_mode: 'pad' or 'drop' (how to handle incomplete windows at borders)
            num_basis: number of basis terms
            mode: 'linear' (step=1) or 'nonlinear' (step=user_step or rank)
            user_step: custom step size for nonlinear mode (int or tuple (step_h, step_w, step_d))
            device: 'cuda' or 'cpu'
        """
        super().__init__()
        self.vec_dim = vec_dim
        # Handle rank as int or tuple
        if isinstance(rank, int):
            self.rank_h = rank
            self.rank_w = rank
            self.rank_d = rank
        else:
            self.rank_h, self.rank_w, self.rank_d = rank
        self.rank_op = rank_op
        self.rank_mode = rank_mode
        self.m = vec_dim
        self.o = num_basis
        assert mode in ('linear', 'nonlinear')
        self.mode = mode
        # Handle step as int or tuple
        if user_step is None:
            self.step_h = self.step_w = self.step_d = None
        elif isinstance(user_step, int):
            self.step_h = self.step_w = self.step_d = user_step
        else:
            self.step_h, self.step_w, self.step_d = user_step
        self.trained = False
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Mapping matrix M
        self.M = nn.Linear(self.vec_dim, self.vec_dim, bias=False)
        
        # Position-weight tensor P[i][j][g]
        self.P = nn.Parameter(torch.empty(self.m, self.m, self.o))
        
        # Precompute indexed periods for three dimensions (fixed, not trainable)
        periods1 = torch.zeros(self.m, self.m, self.o, dtype=torch.float32)
        periods2 = torch.zeros(self.m, self.m, self.o, dtype=torch.float32)
        periods3 = torch.zeros(self.m, self.m, self.o, dtype=torch.float32)
        for i in range(self.m):
            for j in range(self.m):
                for g in range(self.o):
                    base = i*(self.m*self.o) + j*self.o + g
                    periods1[i, j, g] = base + 2
                    periods2[i, j, g] = base + 3
                    periods3[i, j, g] = base + 4
        self.register_buffer('periods1', periods1)
        self.register_buffer('periods2', periods2)
        self.register_buffer('periods3', periods3)

        # Precomputed phi tables (initially None, built when needed)
        self.register_buffer('phi1_table', None)
        self.register_buffer('phi2_table', None)
        self.register_buffer('phi3_table', None)

        # Classifier head (initialized later)
        self.num_classes = None
        self.classifier = None

        # Label head (initialized later)
        self.num_labels = None
        self.labeller = None

        # User function for custom rank operation
        self.user_func = None

        # Cache for window indices for batch acceleration (key: (H,W,D) tuple)
        self._window_index_cache = {}

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

    def _precompute_phi_tables(self, max_k1, max_k2, max_k3):
        """
        Precompute phi tables for all possible k1, k2, k3 values.
        Each table is of shape [max_k, m, m, o] where phi_table[k] = cos(2π*k/periods).
        This significantly speeds up batch_compute_Nk by avoiding repeated trig calls.
        """
        # Check if already have enough precomputed values
        if (self.phi1_table is not None and self.phi2_table is not None and self.phi3_table is not None
            and self.phi1_table.size(0) >= max_k1 and self.phi2_table.size(0) >= max_k2 and self.phi3_table.size(0) >= max_k3):
            return
        
        # Allocate and compute phi1 table
        phi1 = torch.zeros(max_k1, self.m, self.m, self.o, device=self.device, dtype=torch.float32)
        for k in range(max_k1):
            phi1[k] = torch.cos(2 * math.pi * k / self.periods1)
        self.register_buffer('phi1_table', phi1)
        
        # Phi2 table
        phi2 = torch.zeros(max_k2, self.m, self.m, self.o, device=self.device, dtype=torch.float32)
        for k in range(max_k2):
            phi2[k] = torch.cos(2 * math.pi * k / self.periods2)
        self.register_buffer('phi2_table', phi2)
        
        # Phi3 table
        phi3 = torch.zeros(max_k3, self.m, self.m, self.o, device=self.device, dtype=torch.float32)
        for k in range(max_k3):
            phi3[k] = torch.cos(2 * math.pi * k / self.periods3)
        self.register_buffer('phi3_table', phi3)

    def _compute_window_indices_3d(self, H, W, D):
        """
        Precompute linear indices for all windows in a 3D grid of given dimensions.
        Only works for rank_mode == 'drop' (no padding).
        Returns:
            flat_indices: torch.LongTensor of shape [num_windows * window_size] with linear indices
                          into the flattened grid (row-major: H first, then W, then D).
            k1_list: list of starting row indices for each window
            k2_list: list of starting column indices
            k3_list: list of starting depth indices
            num_windows: int
            window_size: int
        """
        if (H, W, D) in self._window_index_cache:
            cached = self._window_index_cache[(H, W, D)]
            return cached['flat_indices'], cached['k1_list'], cached['k2_list'], cached['k3_list'], cached['num_windows'], cached['window_size']
        
        # Determine step sizes
        step_h = 1 if self.mode == 'linear' else (self.step_h if self.step_h is not None else self.rank_h)
        step_w = 1 if self.mode == 'linear' else (self.step_w if self.step_w is not None else self.rank_w)
        step_d = 1 if self.mode == 'linear' else (self.step_d if self.step_d is not None else self.rank_d)
        
        # Generate all window starting positions
        k1_starts = list(range(0, H - self.rank_h + 1, step_h))
        k2_starts = list(range(0, W - self.rank_w + 1, step_w))
        k3_starts = list(range(0, D - self.rank_d + 1, step_d))
        
        if not k1_starts or not k2_starts or not k3_starts:
            cache = {'flat_indices': None, 'k1_list': [], 'k2_list': [], 'k3_list': [], 'num_windows': 0, 'window_size': 0}
            self._window_index_cache[(H, W, D)] = cache
            return None, [], [], [], 0, 0
        
        num_windows = len(k1_starts) * len(k2_starts) * len(k3_starts)
        window_size = self.rank_h * self.rank_w * self.rank_d
        
        # Preallocate index tensor
        indices = torch.zeros(num_windows, window_size, dtype=torch.long, device=self.device)
        k1_list = []
        k2_list = []
        k3_list = []
        
        # Linear index function: idx = i*W*D + j*D + k
        stride_wd = W * D
        stride_d = D
        
        idx = 0
        for i in k1_starts:
            for j in k2_starts:
                for k in k3_starts:
                    k1_list.append(i)
                    k2_list.append(j)
                    k3_list.append(k)
                    # Fill indices for this window
                    offset = 0
                    for di in range(self.rank_h):
                        for dj in range(self.rank_w):
                            for dk in range(self.rank_d):
                                linear = (i+di) * stride_wd + (j+dj) * stride_d + (k+dk)
                                indices[idx, offset] = linear
                                offset += 1
                    idx += 1
        
        flat_indices = indices.view(-1)  # [num_windows * window_size]
        cache = {
            'flat_indices': flat_indices,
            'k1_list': k1_list,
            'k2_list': k2_list,
            'k3_list': k3_list,
            'num_windows': num_windows,
            'window_size': window_size
        }
        self._window_index_cache[(H, W, D)] = cache
        return flat_indices, k1_list, k2_list, k3_list, num_windows, window_size

    def _compute_Nk_from_mapped(self, k1_tensor, k2_tensor, k3_tensor, x):
        """
        Compute N(k1,k2,k3) vectors directly from mapped vectors x (already transformed by M).
        This avoids double mapping in batch processing.
        
        Args:
            k1_tensor: Tensor of row indices [batch_size] (long)
            k2_tensor: Tensor of column indices [batch_size] (long)
            k3_tensor: Tensor of depth indices [batch_size] (long)
            x: Tensor of mapped vectors [batch_size, m]
        
        Returns:
            Nk: Tensor [batch_size, m]
        """
        # Ensure indices are long
        k1_idx = k1_tensor.long()
        k2_idx = k2_tensor.long()
        k3_idx = k3_tensor.long()
        
        # Use precomputed phi tables if available
        if (self.phi1_table is not None and self.phi2_table is not None and self.phi3_table is not None and
            k1_idx.max().item() < self.phi1_table.size(0) and
            k2_idx.max().item() < self.phi2_table.size(0) and
            k3_idx.max().item() < self.phi3_table.size(0)):
            phi1 = self.phi1_table[k1_idx]  # [batch, m, m, o]
            phi2 = self.phi2_table[k2_idx]
            phi3 = self.phi3_table[k3_idx]
            phi = phi1 * phi2 * phi3
        else:
            # Fallback: direct computation
            k1_exp = k1_idx.view(-1, 1, 1, 1)
            k2_exp = k2_idx.view(-1, 1, 1, 1)
            k3_exp = k3_idx.view(-1, 1, 1, 1)
            phi1 = torch.cos(2 * math.pi * k1_exp / self.periods1)
            phi2 = torch.cos(2 * math.pi * k2_exp / self.periods2)
            phi3 = torch.cos(2 * math.pi * k3_exp / self.periods3)
            phi = phi1 * phi2 * phi3
        
        # Nk = sum_{j,g} P[i,j,g] * x_j * phi_{i,j,g}
        Nk = torch.einsum('bj,ijg,bijg->bi', x, self.P, phi)
        return Nk

    def batch_represent_3d(self, grid_batch):
        """
        Compute sequence representations for a batch of 3D grids efficiently.
        Supports arbitrary window size and step as long as all grids in the batch have
        identical dimensions (H, W, D) and rank_mode == 'drop'.
        
        Args:
            grid_batch: Tensor of shape [batch_size, H, W, D, m]
        
        Returns:
            representations: Tensor of shape [batch_size, m]
        """
        grid_batch = grid_batch.to(self.device)
        batch_size, H, W, D, m = grid_batch.shape
        assert self.rank_mode == 'drop', "Batch acceleration only works with rank_mode='drop'"
        
        # Get precomputed window indices
        flat_indices, k1_list, k2_list, k3_list, num_windows, window_size = self._compute_window_indices_3d(H, W, D)
        if flat_indices is None:
            return torch.zeros(batch_size, m, device=self.device)
        
        # Flatten the spatial dimensions
        flat_grid = grid_batch.view(batch_size, H * W * D, m)  # [batch, total_pos, m]
        
        # Gather all window vectors at once
        # flat_indices: [num_windows * window_size]
        indices_expanded = flat_indices.unsqueeze(0).expand(batch_size, -1)  # [batch, num_windows*window_size]
        # Gather along dimension 1 (total_pos)
        gathered = torch.gather(flat_grid, 1, indices_expanded.unsqueeze(-1).expand(-1, -1, m))
        # gathered: [batch, num_windows*window_size, m]
        windows = gathered.view(batch_size, num_windows, window_size, m)  # [batch, num_windows, window_size, m]
        
        # Apply mapping M to each vector
        mapped = self.M(windows)  # [batch, num_windows, window_size, m]
        
        # Aggregate over window_size
        if self.rank_op == 'sum':
            agg = mapped.sum(dim=2)  # [batch, num_windows, m]
        elif self.rank_op == 'pick':
            idx = torch.randint(0, window_size, (batch_size, num_windows, 1), device=self.device)
            agg = torch.gather(mapped, 2, idx.unsqueeze(-1).expand(-1, -1, -1, m)).squeeze(2)
        elif self.rank_op == 'user_func':
            # Default: average + sigmoid
            avg = mapped.mean(dim=2)
            agg = torch.sigmoid(avg)
        else:  # 'avg'
            agg = mapped.mean(dim=2)  # [batch, num_windows, m]
        
        # Prepare k1, k2, k3 tensors for all windows
        k1_tensor = torch.tensor(k1_list, dtype=torch.long, device=self.device)
        k2_tensor = torch.tensor(k2_list, dtype=torch.long, device=self.device)
        k3_tensor = torch.tensor(k3_list, dtype=torch.long, device=self.device)
        
        # Expand to batch dimension
        k1_exp = k1_tensor.unsqueeze(0).expand(batch_size, -1)  # [batch, num_windows]
        k2_exp = k2_tensor.unsqueeze(0).expand(batch_size, -1)
        k3_exp = k3_tensor.unsqueeze(0).expand(batch_size, -1)
        
        # Flatten to compute all Nk in one go
        agg_flat = agg.reshape(batch_size * num_windows, m)  # [batch*num_windows, m]
        k1_flat = k1_exp.reshape(batch_size * num_windows)
        k2_flat = k2_exp.reshape(batch_size * num_windows)
        k3_flat = k3_exp.reshape(batch_size * num_windows)
        
        # Compute Nk for all windows
        Nk_flat = self._compute_Nk_from_mapped(k1_flat, k2_flat, k3_flat, agg_flat)
        Nk = Nk_flat.view(batch_size, num_windows, m)  # [batch, num_windows, m]
        
        # Average over windows to get grid representation
        rep = Nk.mean(dim=1)  # [batch, m]
        return rep

    def batch_compute_Nk_and_targets_3d(self, grid_batch):
        """
        Compute Nk vectors and target vectors for each window in a batch of grids.
        Supports arbitrary window size and step, but only for rank_mode == 'drop'.
        
        Args:
            grid_batch: Tensor of shape [batch_size, H, W, D, m]
        
        Returns:
            Nk_all: Tensor [batch_size, n_windows, m] - Nk vectors for each window
            targets: Tensor [batch_size, n_windows, m] - target vectors (aggregated mapped vectors)
        """
        grid_batch = grid_batch.to(self.device)
        batch_size, H, W, D, m = grid_batch.shape
        assert self.rank_mode == 'drop', "Batch acceleration only works with rank_mode='drop'"
        
        flat_indices, k1_list, k2_list, k3_list, num_windows, window_size = self._compute_window_indices_3d(H, W, D)
        if flat_indices is None:
            return (torch.empty(batch_size, 0, m, device=self.device),
                    torch.empty(batch_size, 0, m, device=self.device))
        
        # Flatten and extract windows
        flat_grid = grid_batch.view(batch_size, H * W * D, m)
        indices_expanded = flat_indices.unsqueeze(0).expand(batch_size, -1)
        gathered = torch.gather(flat_grid, 1, indices_expanded.unsqueeze(-1).expand(-1, -1, m))
        windows = gathered.view(batch_size, num_windows, window_size, m)  # [batch, num_windows, window_size, m]
        
        # Apply mapping M
        mapped = self.M(windows)  # [batch, num_windows, window_size, m]
        
        # Aggregate to get targets
        if self.rank_op == 'sum':
            targets = mapped.sum(dim=2)  # [batch, num_windows, m]
        elif self.rank_op == 'pick':
            idx = torch.randint(0, window_size, (batch_size, num_windows, 1), device=self.device)
            targets = torch.gather(mapped, 2, idx.unsqueeze(-1).expand(-1, -1, -1, m)).squeeze(2)
        elif self.rank_op == 'user_func':
            avg = mapped.mean(dim=2)
            targets = torch.sigmoid(avg)
        else:  # 'avg'
            targets = mapped.mean(dim=2)  # [batch, num_windows, m]
        
        # Prepare k tensors
        k1_tensor = torch.tensor(k1_list, dtype=torch.long, device=self.device)
        k2_tensor = torch.tensor(k2_list, dtype=torch.long, device=self.device)
        k3_tensor = torch.tensor(k3_list, dtype=torch.long, device=self.device)
        
        k1_exp = k1_tensor.unsqueeze(0).expand(batch_size, -1)
        k2_exp = k2_tensor.unsqueeze(0).expand(batch_size, -1)
        k3_exp = k3_tensor.unsqueeze(0).expand(batch_size, -1)
        
        targets_flat = targets.reshape(batch_size * num_windows, m)
        k1_flat = k1_exp.reshape(batch_size * num_windows)
        k2_flat = k2_exp.reshape(batch_size * num_windows)
        k3_flat = k3_exp.reshape(batch_size * num_windows)
        
        Nk_flat = self._compute_Nk_from_mapped(k1_flat, k2_flat, k3_flat, targets_flat)
        Nk_all = Nk_flat.view(batch_size, num_windows, m)
        
        return Nk_all, targets

    def extract_vectors(self, array):
        """
        Extract window vectors from 3D array based on processing mode and rank operation.
        
        - 'linear': Slide window by step 1 in all directions, extracting contiguous windows of size rank_h x rank_w x rank_d
        - 'nonlinear': Slide window by custom step (step_h, step_w, step_d) in all directions
        
        For nonlinear mode, handles incomplete trailing windows using:
        - 'pad': Pads with zero vectors to maintain full window size
        - 'drop': Discards incomplete windows
        
        Args:
            array (torch.Tensor): 3D array of shape [H, W, D, vec_dim] (or convertible)
            
        Returns:
            torch.Tensor: Tensor of shape [num_windows, vec_dim] after applying rank operation to each window
        """
        # Convert to tensor if needed
        if not isinstance(array, torch.Tensor):
            array = torch.tensor(array, dtype=torch.float32, device=self.device)
        if array.device != self.device:
            array = array.to(self.device)
        
        H, W, D, _ = array.shape
        
        def apply_op(win_tensor):
            """Apply rank operation (avg/sum/pick/user_func) to a window of shape [rank_h, rank_w, rank_d, vec_dim]"""
            if self.rank_op == 'sum':
                return torch.sum(win_tensor, dim=(0, 1, 2))
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
                    avg = torch.mean(win_tensor, dim=(0, 1, 2))
                    return torch.sigmoid(avg)
            else:  # 'avg' is default
                return torch.mean(win_tensor, dim=(0, 1, 2))
        
        # Determine step sizes
        if self.mode == 'linear':
            step_h, step_w, step_d = 1, 1, 1
        else:
            step_h = self.step_h if self.step_h is not None else self.rank_h
            step_w = self.step_w if self.step_w is not None else self.rank_w
            step_d = self.step_d if self.step_d is not None else self.rank_d
        
        windows = []
        # Slide over height
        for i in range(0, H, step_h):
            if i + self.rank_h > H:
                if self.rank_mode == 'pad':
                    frag_h = array[i:H, :, :, :]
                    pad_h = self.rank_h - frag_h.shape[0]
                    padding = torch.zeros(pad_h, frag_h.shape[1], frag_h.shape[2], self.vec_dim, device=self.device)
                    frag_h = torch.cat([frag_h, padding], dim=0)
                else:
                    continue
            else:
                frag_h = array[i:i+self.rank_h, :, :, :]
            
            # Slide over width
            for j in range(0, W, step_w):
                if j + self.rank_w > W:
                    if self.rank_mode == 'pad':
                        frag_w = frag_h[:, j:W, :, :]
                        pad_w = self.rank_w - frag_w.shape[1]
                        padding = torch.zeros(frag_w.shape[0], pad_w, frag_w.shape[2], self.vec_dim, device=self.device)
                        frag_w = torch.cat([frag_w, padding], dim=1)
                    else:
                        continue
                else:
                    frag_w = frag_h[:, j:j+self.rank_w, :, :]
                
                # Slide over depth
                for k in range(0, D, step_d):
                    if k + self.rank_d > D:
                        if self.rank_mode == 'pad':
                            frag_d = frag_w[:, :, k:D, :]
                            pad_d = self.rank_d - frag_d.shape[2]
                            padding = torch.zeros(frag_d.shape[0], frag_d.shape[1], pad_d, self.vec_dim, device=self.device)
                            frag_d = torch.cat([frag_d, padding], dim=2)
                        else:
                            continue
                    else:
                        frag_d = frag_w[:, :, k:k+self.rank_d, :]
                    # frag_d shape: [rank_h, rank_w, rank_d, m]
                    windows.append(apply_op(frag_d))
        
        if windows:
            return torch.stack(windows)  # [num_windows, m]
        else:
            return torch.empty(0, self.vec_dim, device=self.device)

    def batch_compute_Nk(self, k1_tensor, k2_tensor, k3_tensor, vectors):
        """
        Vectorized computation of N(k1,k2,k3) vectors for a batch of positions and vectors.
        Optimized using precomputed phi tables if available, otherwise falls back to direct calculation.
        
        Args:
            k1_tensor: Tensor of row indices [batch_size]
            k2_tensor: Tensor of column indices [batch_size]
            k3_tensor: Tensor of depth indices [batch_size]
            vectors: Tensor of vectors [batch_size, vec_dim]
            
        Returns:
            Tensor of N(k1,k2,k3) vectors [batch_size, m]
        """
        # Apply mapping matrix M to each vector
        x = self.M(vectors)  # [batch_size, m]
        return self._compute_Nk_from_mapped(k1_tensor, k2_tensor, k3_tensor, x)

    def describe(self, array):
        """
        Compute N(k1,k2,k3) vectors for each window in the 3D array.
        
        Args:
            array: 3D tensor of shape [H, W, D, m]
            
        Returns:
            list of numpy arrays, each of shape [m]
        """
        extracted_vectors = self.extract_vectors(array)
        if extracted_vectors.shape[0] == 0:
            return []
        
        # Generate positions for each window (row-major order)
        H, W, D, _ = array.shape
        step_h = 1 if self.mode == 'linear' else (self.step_h if self.step_h is not None else self.rank_h)
        step_w = 1 if self.mode == 'linear' else (self.step_w if self.step_w is not None else self.rank_w)
        step_d = 1 if self.mode == 'linear' else (self.step_d if self.step_d is not None else self.rank_d)
        
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
                    positions.append((i, j, k))
        
        # Convert to tensors
        k1_tensor = torch.tensor([p[0] for p in positions], dtype=torch.float32, device=self.device)
        k2_tensor = torch.tensor([p[1] for p in positions], dtype=torch.float32, device=self.device)
        k3_tensor = torch.tensor([p[2] for p in positions], dtype=torch.float32, device=self.device)
        
        N_batch = self.batch_compute_Nk(k1_tensor, k2_tensor, k3_tensor, extracted_vectors)
        return N_batch.detach().cpu().numpy()

    def S(self, array):
        """
        Compute cumulative sums of N(k1,k2,k3) vectors along window order (row-major).
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
        Compute mean squared deviation D across 3D arrays:
        D = average over all windows of (N(k1,k2,k3)-t)^2
        """
        total_loss = 0.0
        total_windows = 0
        
        t_tensors = [torch.tensor(t, dtype=torch.float32, device=self.device) for t in t_list]
        
        for array, t in zip(arrays, t_tensors):
            extracted_vectors = self.extract_vectors(array)
            if extracted_vectors.shape[0] == 0:
                continue
            
            # Generate positions (same as in describe)
            H, W, D, _ = array.shape
            step_h = 1 if self.mode == 'linear' else (self.step_h if self.step_h is not None else self.rank_h)
            step_w = 1 if self.mode == 'linear' else (self.step_w if self.step_w is not None else self.rank_w)
            step_d = 1 if self.mode == 'linear' else (self.step_d if self.step_d is not None else self.rank_d)
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
                        positions.append((i, j, k))
            k1_tensor = torch.tensor([p[0] for p in positions], dtype=torch.float32, device=self.device)
            k2_tensor = torch.tensor([p[1] for p in positions], dtype=torch.float32, device=self.device)
            k3_tensor = torch.tensor([p[2] for p in positions], dtype=torch.float32, device=self.device)
            
            N_batch = self.batch_compute_Nk(k1_tensor, k2_tensor, k3_tensor, extracted_vectors)
            losses = torch.sum((N_batch - t) ** 2, dim=1)
            total_loss += losses.sum().item()
            total_windows += extracted_vectors.shape[0]
        
        return total_loss / total_windows if total_windows else 0.0

    def d(self, array, t):
        """Compute pattern deviation value (d) for a single 3D array."""
        return self.D([array], [t])

    def reg_train(self, arrays, t_list, max_iters=1000, tol=1e-8, learning_rate=0.01,
                  continued=False, decay_rate=1.0, print_every=10, batch_size=32,
                  checkpoint_file=None, checkpoint_interval=10):
        """
        Train the model using gradient descent for regression on 3D arrays.
        Optimized for GPU memory efficiency by processing arrays individually, but
        uses fast batch processing when all arrays in a batch have the same dimensions
        and rank_mode == 'drop'.
        """
        if not continued:
            self.reset_parameters()
        
        # Precompute phi tables based on max H and W and D
        max_H = max(arr.shape[0] for arr in arrays)
        max_W = max(arr.shape[1] for arr in arrays)
        max_D = max(arr.shape[2] for arr in arrays)
        self._precompute_phi_tables(max_H, max_W, max_D)
        
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
                
                # Check if all grids in this batch have the same dimensions
                grid_shapes = [(arr.shape[0], arr.shape[1], arr.shape[2]) for arr in batch_arrays]
                if use_fast_batch and len(set(grid_shapes)) == 1:
                    # All grids have equal dimensions -> use fast batch processing
                    batch_arrays_on_device = [arr.to(self.device) if arr.device != self.device else arr for arr in batch_arrays]
                    batch_grid_tensor = torch.stack(batch_arrays_on_device, dim=0)  # [batch, H, W, D, m]
                    # Compute grid representations
                    reps = self.batch_represent_3d(batch_grid_tensor)  # [batch, m]
                    targets_tensor = torch.stack(batch_targets, dim=0)  # [batch, m]
                    loss = torch.mean(torch.sum((reps - targets_tensor) ** 2, dim=1))
                    loss.backward()
                    optimizer.step()
                    batch_loss = loss.item() * len(batch_arrays)
                    total_loss += batch_loss
                    total_arrays += len(batch_arrays)
                else:
                    # Process each array individually
                    for arr, target in zip(batch_arrays, batch_targets):
                        extracted_vectors = self.extract_vectors(arr)
                        if extracted_vectors.shape[0] == 0:
                            continue
                        
                        # Compute positions
                        H, W, D, _ = arr.shape
                        step_h = 1 if self.mode == 'linear' else (self.step_h if self.step_h is not None else self.rank_h)
                        step_w = 1 if self.mode == 'linear' else (self.step_w if self.step_w is not None else self.rank_w)
                        step_d = 1 if self.mode == 'linear' else (self.step_d if self.step_d is not None else self.rank_d)
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
                                    positions.append((i, j, k))
                        k1_tensor = torch.tensor([p[0] for p in positions], dtype=torch.float32, device=self.device)
                        k2_tensor = torch.tensor([p[1] for p in positions], dtype=torch.float32, device=self.device)
                        k3_tensor = torch.tensor([p[2] for p in positions], dtype=torch.float32, device=self.device)
                        
                        N_batch = self.batch_compute_Nk(k1_tensor, k2_tensor, k3_tensor, extracted_vectors)
                        seq_pred = torch.mean(N_batch, dim=0)
                        seq_loss = torch.sum((seq_pred - target) ** 2)
                        batch_loss += seq_loss
                        
                        del N_batch, seq_pred, extracted_vectors, k1_tensor, k2_tensor, k3_tensor
                    
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
        """Train for multi-class classification on 3D arrays with optional batch acceleration."""
        if self.classifier is None or self.num_classes != num_classes:
            self.classifier = nn.Linear(self.m, num_classes).to(self.device)
            self.num_classes = num_classes
        if not continued:
            self.reset_parameters()
        
        # Precompute phi tables based on maximum array dimensions
        max_H = max(arr.shape[0] for arr in arrays)
        max_W = max(arr.shape[1] for arr in arrays)
        max_D = max(arr.shape[2] for arr in arrays)
        self._precompute_phi_tables(max_H, max_W, max_D)
        
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
                
                grid_shapes = [(arr.shape[0], arr.shape[1], arr.shape[2]) for arr in batch_arrays]
                if use_fast_batch and len(set(grid_shapes)) == 1:
                    # Fast path
                    batch_arrays_on_device = [arr.to(self.device) if arr.device != self.device else arr for arr in batch_arrays]
                    batch_grid_tensor = torch.stack(batch_arrays_on_device, dim=0)
                    reps = self.batch_represent_3d(batch_grid_tensor)  # [batch, m]
                    logits = self.classifier(reps)
                    loss = criterion(logits, batch_labels)
                    loss.backward()
                    optimizer.step()
                    
                    batch_loss = loss.item() * len(batch_arrays)
                    total_loss += batch_loss
                    total_arrays += len(batch_arrays)
                    with torch.no_grad():
                        pred = torch.argmax(logits, dim=1)
                        correct += (pred == batch_labels).sum().item()
                else:
                    # Slow path: process individually
                    batch_logits = []
                    for arr in batch_arrays:
                        extracted_vectors = self.extract_vectors(arr)
                        if extracted_vectors.shape[0] == 0:
                            seq_vector = torch.zeros(self.m, device=self.device)
                        else:
                            # Compute positions
                            H, W, D, _ = arr.shape
                            step_h = 1 if self.mode == 'linear' else (self.step_h if self.step_h is not None else self.rank_h)
                            step_w = 1 if self.mode == 'linear' else (self.step_w if self.step_w is not None else self.rank_w)
                            step_d = 1 if self.mode == 'linear' else (self.step_d if self.step_d is not None else self.rank_d)
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
                                        positions.append((i, j, k))
                            k1_tensor = torch.tensor([p[0] for p in positions], dtype=torch.float32, device=self.device)
                            k2_tensor = torch.tensor([p[1] for p in positions], dtype=torch.float32, device=self.device)
                            k3_tensor = torch.tensor([p[2] for p in positions], dtype=torch.float32, device=self.device)
                            N_batch = self.batch_compute_Nk(k1_tensor, k2_tensor, k3_tensor, extracted_vectors)
                            seq_vector = torch.mean(N_batch, dim=0)
                            del N_batch, extracted_vectors, k1_tensor, k2_tensor, k3_tensor
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
        """Train for multi-label classification on 3D arrays with optional batch acceleration."""
        if self.labeller is None or self.num_labels != num_labels:
            self.labeller = nn.Linear(self.m, num_labels).to(self.device)
            self.num_labels = num_labels
        if not continued:
            self.reset_parameters()
        
        # Precompute phi tables based on maximum array dimensions
        max_H = max(arr.shape[0] for arr in arrays)
        max_W = max(arr.shape[1] for arr in arrays)
        max_D = max(arr.shape[2] for arr in arrays)
        self._precompute_phi_tables(max_H, max_W, max_D)
        
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
                
                grid_shapes = [(arr.shape[0], arr.shape[1], arr.shape[2]) for arr in batch_arrays]
                if use_fast_batch and len(set(grid_shapes)) == 1:
                    # Fast path
                    batch_arrays_on_device = [arr.to(self.device) if arr.device != self.device else arr for arr in batch_arrays]
                    batch_grid_tensor = torch.stack(batch_arrays_on_device, dim=0)
                    reps = self.batch_represent_3d(batch_grid_tensor)  # [batch, m]
                    logits = self.labeller(reps)  # [batch, num_labels]
                    loss = criterion(logits, batch_labels)
                    loss.backward()
                    optimizer.step()
                    
                    batch_loss = loss.item() * len(batch_arrays)
                    total_loss += batch_loss
                    total_arrays += len(batch_arrays)
                    
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
                    for arr in batch_arrays:
                        extracted_vectors = self.extract_vectors(arr)
                        if extracted_vectors.shape[0] == 0:
                            continue
                        H, W, D, _ = arr.shape
                        step_h = 1 if self.mode == 'linear' else (self.step_h if self.step_h is not None else self.rank_h)
                        step_w = 1 if self.mode == 'linear' else (self.step_w if self.step_w is not None else self.rank_w)
                        step_d = 1 if self.mode == 'linear' else (self.step_d if self.step_d is not None else self.rank_d)
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
                                    positions.append((i, j, k))
                        k1_tensor = torch.tensor([p[0] for p in positions], dtype=torch.float32, device=self.device)
                        k2_tensor = torch.tensor([p[1] for p in positions], dtype=torch.float32, device=self.device)
                        k3_tensor = torch.tensor([p[2] for p in positions], dtype=torch.float32, device=self.device)
                        N_batch = self.batch_compute_Nk(k1_tensor, k2_tensor, k3_tensor, extracted_vectors)
                        seq_rep = torch.mean(N_batch, dim=0)
                        logits = self.labeller(seq_rep)
                        batch_logits.append(logits)
                        del N_batch, seq_rep, extracted_vectors, k1_tensor, k2_tensor, k3_tensor
                    
                    if batch_logits:
                        batch_logits = torch.stack(batch_logits, dim=0)
                        loss = criterion(batch_logits, batch_labels)
                        loss.backward()
                        optimizer.step()
                        total_loss += loss.item() * len(batch_arrays)
                        total_arrays += len(batch_arrays)
                        
                        with torch.no_grad():
                            probs = torch.sigmoid(batch_logits)
                            preds = (probs > 0.5).float()
                            batch_correct = (preds == batch_labels).sum().item()
                            batch_predictions = batch_labels.numel()
                            total_correct += batch_correct
                            total_predictions += batch_predictions
                
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
        Self-training for self-consistency on 3D arrays.
        Trains so that N(k1,k2,k3) matches M(window_vector) at each window.
        Uses fast batch processing when all grids have the same dimensions and rank_mode == 'drop'.
        """
        if not continued:
            self.reset_parameters()
        
        # Precompute phi tables based on maximum array dimensions
        max_H = max(arr.shape[0] for arr in arrays)
        max_W = max(arr.shape[1] for arr in arrays)
        max_D = max(arr.shape[2] for arr in arrays)
        self._precompute_phi_tables(max_H, max_W, max_D)
        
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)
        
        history = []
        prev_loss = float('inf')
        best_loss = float('inf')
        best_model_state = None
        
        # Check if all grids have the same dimensions (for fast batch processing)
        grid_shapes = [(arr.shape[0], arr.shape[1], arr.shape[2]) for arr in arrays]
        all_equal = len(set(grid_shapes)) == 1
        use_fast_batch = (self.rank_mode == 'drop' and all_equal)
        
        if use_fast_batch:
            # Fast path: all grids equal size, batch process all windows together
            H, W, D = grid_shapes[0]
            arrays_on_device = [arr.to(self.device) if arr.device != self.device else arr for arr in arrays]
            all_grids = torch.stack(arrays_on_device, dim=0)  # [num_grids, H, W, D, m]
            num_grids = len(arrays)
            seq_batch_size = batch_size  # number of grids per batch
            
            for it in range(max_iters):
                total_loss = 0.0
                total_windows = 0
                
                indices = list(range(num_grids))
                random.shuffle(indices)
                
                for batch_start in range(0, num_grids, seq_batch_size):
                    batch_indices = indices[batch_start:batch_start + seq_batch_size]
                    batch_grids = all_grids[batch_indices]  # [batch, H, W, D, m]
                    
                    optimizer.zero_grad()
                    
                    # Compute Nk and targets for all windows in this batch of grids
                    Nk_batch, targets_batch = self.batch_compute_Nk_and_targets_3d(batch_grids)
                    # both [batch, n_windows, m]
                    
                    # Compute MSE loss over all windows
                    loss = torch.mean(torch.sum((Nk_batch - targets_batch) ** 2, dim=-1))
                    loss.backward()
                    optimizer.step()
                    
                    batch_loss = loss.item() * batch_grids.shape[0] * Nk_batch.shape[1]
                    total_loss += batch_loss
                    total_windows += batch_grids.shape[0] * Nk_batch.shape[1]
                    
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
            # Slow path: arrays have different sizes or rank_mode == 'pad'
            for it in range(max_iters):
                total_loss = 0.0
                total_windows = 0
                
                indices = list(range(len(arrays)))
                random.shuffle(indices)
                
                for batch_start in range(0, len(indices), batch_size):
                    batch_indices = indices[batch_start:batch_start + batch_size]
                    batch_arrays = [arrays[idx] for idx in batch_indices]
                    
                    optimizer.zero_grad()
                    batch_loss = 0.0
                    batch_count = 0
                    
                    for arr in batch_arrays:
                        extracted_vectors = self.extract_vectors(arr)
                        if extracted_vectors.shape[0] == 0:
                            continue
                        H, W, D, _ = arr.shape
                        step_h = 1 if self.mode == 'linear' else (self.step_h if self.step_h is not None else self.rank_h)
                        step_w = 1 if self.mode == 'linear' else (self.step_w if self.step_w is not None else self.rank_w)
                        step_d = 1 if self.mode == 'linear' else (self.step_d if self.step_d is not None else self.rank_d)
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
                                    positions.append((i, j, k))
                        k1_tensor = torch.tensor([p[0] for p in positions], dtype=torch.float32, device=self.device)
                        k2_tensor = torch.tensor([p[1] for p in positions], dtype=torch.float32, device=self.device)
                        k3_tensor = torch.tensor([p[2] for p in positions], dtype=torch.float32, device=self.device)
                        N_batch = self.batch_compute_Nk(k1_tensor, k2_tensor, k3_tensor, extracted_vectors)
                        target_vectors = self.M(extracted_vectors)  # [num_windows, m]
                        
                        loss = torch.mean(torch.sum((N_batch - target_vectors) ** 2, dim=1))
                        batch_loss += loss
                        batch_count += 1
                        del N_batch, target_vectors, extracted_vectors, k1_tensor, k2_tensor, k3_tensor
                    
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
                H, W, D, _ = arr.shape
                step_h = 1 if self.mode == 'linear' else (self.step_h if self.step_h is not None else self.rank_h)
                step_w = 1 if self.mode == 'linear' else (self.step_w if self.step_w is not None else self.rank_w)
                step_d = 1 if self.mode == 'linear' else (self.step_d if self.step_d is not None else self.rank_d)
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
                            positions.append((i, j, k))
                k1_tensor = torch.tensor([p[0] for p in positions], dtype=torch.float32, device=self.device)
                k2_tensor = torch.tensor([p[1] for p in positions], dtype=torch.float32, device=self.device)
                k3_tensor = torch.tensor([p[2] for p in positions], dtype=torch.float32, device=self.device)
                N_batch = self.batch_compute_Nk(k1_tensor, k2_tensor, k3_tensor, extracted_vectors)
                total_t += N_batch.sum(dim=0)
                total_windows += extracted_vectors.shape[0]
                del N_batch, extracted_vectors, k1_tensor, k2_tensor, k3_tensor
        
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
        """Predict target vector for a 3D array (average of all N(k1,k2,k3) vectors)."""
        extracted_vectors = self.extract_vectors(array)
        if extracted_vectors.shape[0] == 0:
            return [0.0] * self.m
        H, W, D, _ = array.shape
        step_h = 1 if self.mode == 'linear' else (self.step_h if self.step_h is not None else self.rank_h)
        step_w = 1 if self.mode == 'linear' else (self.step_w if self.step_w is not None else self.rank_w)
        step_d = 1 if self.mode == 'linear' else (self.step_d if self.step_d is not None else self.rank_d)
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
                    positions.append((i, j, k))
        k1_tensor = torch.tensor([p[0] for p in positions], dtype=torch.float32, device=self.device)
        k2_tensor = torch.tensor([p[1] for p in positions], dtype=torch.float32, device=self.device)
        k3_tensor = torch.tensor([p[2] for p in positions], dtype=torch.float32, device=self.device)
        N_batch = self.batch_compute_Nk(k1_tensor, k2_tensor, k3_tensor, extracted_vectors)
        avg = torch.mean(N_batch, dim=0).detach().cpu().numpy()
        return avg

    def predict_c(self, array):
        """Predict class label for a 3D array using the classification head."""
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
        """Predict multi-label classification for a 3D array."""
        assert self.labeller is not None, "Model must be trained first for label prediction"
        extracted_vectors = self.extract_vectors(array)
        if extracted_vectors.shape[0] == 0:
            return np.zeros(self.num_labels, dtype=np.float32), np.zeros(self.num_labels, dtype=np.float32)
        H, W, D, _ = array.shape
        step_h = 1 if self.mode == 'linear' else (self.step_h if self.step_h is not None else self.rank_h)
        step_w = 1 if self.mode == 'linear' else (self.step_w if self.step_w is not None else self.rank_w)
        step_d = 1 if self.mode == 'linear' else (self.step_d if self.step_d is not None else self.rank_d)
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
                    positions.append((i, j, k))
        k1_tensor = torch.tensor([p[0] for p in positions], dtype=torch.float32, device=self.device)
        k2_tensor = torch.tensor([p[1] for p in positions], dtype=torch.float32, device=self.device)
        k3_tensor = torch.tensor([p[2] for p in positions], dtype=torch.float32, device=self.device)
        N_batch = self.batch_compute_Nk(k1_tensor, k2_tensor, k3_tensor, extracted_vectors)
        seq_rep = torch.mean(N_batch, dim=0)
        with torch.no_grad():
            logits = self.labeller(seq_rep)
            probs = torch.sigmoid(logits).cpu().numpy()
        binary = (probs > threshold).astype(np.float32)
        return binary, probs

    def reconstruct(self, H, W, D, tau=0.0):
        """
        Reconstruct a 3D array of shape (H, W, D, m) by minimizing error with temperature.
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
        
        # Determine number of windows
        nh = (H - self.rank_h) // step_h + 1 if self.rank_mode == 'drop' else (H + step_h - 1) // step_h
        nw = (W - self.rank_w) // step_w + 1 if self.rank_mode == 'drop' else (W + step_w - 1) // step_w
        nd = (D - self.rank_d) // step_d + 1 if self.rank_mode == 'drop' else (D + step_d - 1) // step_d
        
        # Pre-generate candidate vectors
        num_candidates = 100
        candidates = torch.randn(num_candidates, self.vec_dim, device=self.device)
        
        # For each window position, choose a candidate vector
        chosen_vectors = {}  # (i,j,k) -> vector
        for i_idx in range(nh):
            i = i_idx * step_h
            for j_idx in range(nw):
                j = j_idx * step_w
                for k_idx in range(nd):
                    k = k_idx * step_d
                    # Compute N for all candidates at this position
                    k1_tensor = torch.full((num_candidates,), i, dtype=torch.float32, device=self.device)
                    k2_tensor = torch.full((num_candidates,), j, dtype=torch.float32, device=self.device)
                    k3_tensor = torch.full((num_candidates,), k, dtype=torch.float32, device=self.device)
                    N_all = self.batch_compute_Nk(k1_tensor, k2_tensor, k3_tensor, candidates)
                    errors = torch.sum((N_all - mean_t_tensor) ** 2, dim=1)
                    scores = -errors
                    if tau == 0:
                        idx = torch.argmax(scores).item()
                    else:
                        probs = torch.softmax(scores / tau, dim=0).detach().cpu().numpy()
                        idx = random.choices(range(num_candidates), weights=probs, k=1)[0]
                    chosen_vectors[(i, j, k)] = candidates[idx]
        
        # Build the output array by averaging overlapping windows
        result = torch.zeros(H, W, D, self.vec_dim, device=self.device)
        count = torch.zeros(H, W, D, device=self.device)
        for (i, j, k), vec in chosen_vectors.items():
            for di in range(self.rank_h):
                if i+di >= H:
                    continue
                for dj in range(self.rank_w):
                    if j+dj >= W:
                        continue
                    for dk in range(self.rank_d):
                        if k+dk >= D:
                            continue
                        result[i+di, j+dj, k+dk] += vec
                        count[i+di, j+dj, k+dk] += 1
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
    print("Spatial Numerical Dual Descriptor TS3 - 3D Array Version (Accelerated)")
    print("Processes 3D grids of m-dimensional real vectors")
    print("="*50)
    
    torch.manual_seed(11)
    random.seed(11)
    np.random.seed(11)
    
    # Parameters
    vec_dim = 5
    num_basis = 5
    rank = 2                # cubic window 2x2x2
    user_step = 2           # step size for nonlinear mode
    
    ndd = SpatialNumDualDescriptorTS3(
        vec_dim=vec_dim,
        rank=rank,
        num_basis=num_basis,
        mode='nonlinear',
        user_step=user_step,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    print(f"\nUsing device: {ndd.device}")
    print(f"Vector dimension: {vec_dim}")
    print(f"Window size: {rank}x{rank}x{rank}, step: {user_step}")
    
    # Generate 30 random 3D arrays (H x W x D x vec_dim) with random target vectors
    print("\nGenerating training data...")
    arrays, t_list = [], []
    for _ in range(30):
        H = random.randint(15, 20)
        W = random.randint(15, 20)
        D = random.randint(15, 20)
        arr = torch.randn(H, W, D, vec_dim)
        arrays.append(arr)
        t_list.append(np.random.uniform(-1.0, 1.0, vec_dim))
    
    # Regression training
    print("\n" + "="*50)
    print("Starting Regression Training")
    print("="*50)
    ndd.reg_train(arrays, t_list, max_iters=50, tol=1e-9, learning_rate=0.05,
                  decay_rate=0.95, batch_size=4, print_every=5)
    
    # Predict target for first array
    arr0 = arrays[0]
    t_pred = ndd.predict_t(arr0)
    print(f"\nPredicted t for first array: {[round(x, 4) for x in t_pred[:5]]}...")
    
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
    print("\nReconstructing a 3D array...")
    H_rec, W_rec, D_rec = 15, 15, 15
    rec_det = ndd.reconstruct(H_rec, W_rec, D_rec, tau=0.0)
    rec_rand = ndd.reconstruct(H_rec, W_rec, D_rec, tau=0.5)
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
        for _ in range(20):
            H = random.randint(15, 20)
            W = random.randint(15, 20)
            D = random.randint(15, 20)
            if class_id == 0:
                arr = torch.randn(H, W, D, vec_dim) + 1.0
            elif class_id == 1:
                arr = torch.randn(H, W, D, vec_dim) - 1.0
            else:
                arr = torch.randn(H, W, D, vec_dim)
            class_arrays.append(arr)
            class_labels.append(class_id)
    
    ndd_cls = SpatialNumDualDescriptorTS3(
        vec_dim=vec_dim, rank=rank, num_basis=num_basis,
        mode='nonlinear', user_step=user_step, device=ndd.device
    )
    print("\nStarting Classification Training")
    ndd_cls.cls_train(class_arrays, class_labels, num_classes,
                      max_iters=15, tol=1e-8, learning_rate=0.05,
                      decay_rate=0.99, batch_size=4, print_every=5)
    
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
    num_labels = 4
    label_arrays = []
    ml_labels = []
    for _ in range(30):
        H = random.randint(15, 20)
        W = random.randint(15, 20)
        D = random.randint(15, 20)
        arr = torch.randn(H, W, D, vec_dim)
        label_arrays.append(arr)
        label_vec = [random.random() > 0.7 for _ in range(num_labels)]
        ml_labels.append([1.0 if x else 0.0 for x in label_vec])
    
    ndd_lbl = SpatialNumDualDescriptorTS3(
        vec_dim=vec_dim, rank=rank, num_basis=num_basis,
        mode='nonlinear', user_step=user_step, device=ndd.device
    )
    print("\nStarting Multi-Label Training")
    loss_hist, acc_hist = ndd_lbl.lbl_train(
        label_arrays, ml_labels, num_labels,
        max_iters=15, tol=1e-8, learning_rate=0.05,
        decay_rate=0.99, batch_size=4, print_every=5
    )
    print(f"Final multi-label accuracy: {acc_hist[-1]:.4f}")
    
    # Self-training
    print("\n" + "="*50)
    print("Self-Training Example")
    print("="*50)
    ndd_self = SpatialNumDualDescriptorTS3(
        vec_dim=vec_dim, rank=rank, num_basis=num_basis,
        mode='nonlinear', user_step=user_step, device=ndd.device
    )
    # Use equal-sized arrays to trigger fast batch path
    self_arrays = [torch.randn(15, 15, 15, vec_dim) for _ in range(8)]
    print("Training for self-consistency (fast batch path enabled)...")
    self_hist = ndd_self.self_train(self_arrays, max_iters=10, learning_rate=0.01, batch_size=2)
    
    plt.figure(figsize=(10,6))
    plt.plot(self_hist)
    plt.title('Self-Training Loss History (3D)')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig('self_training_loss_3d.png')
    print("\nSelf-training loss plot saved as 'self_training_loss_3d.png'")
    
    print("\nAll tests completed successfully!")
