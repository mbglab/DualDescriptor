# Copyright (C) 2005-2026, Bin-Guang Ma (mbg@mail.hzau.edu.cn); SPDX-License-Identifier: MIT
# The Spatial Numerical Dual Descriptor Vector class (Tensor form) for 2D array implemented with PyTorch
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

class SpatialNumDualDescriptorTS2(nn.Module):
    """
    Spatial Numerical Dual Descriptor for 2D arrays with GPU acceleration using PyTorch:
      - Processes 2D grids of m-dimensional real vectors (H x W x m) instead of 1D sequences
      - tensor P ∈ R^{m×m×o} of basis coefficients
      - mapping matrix M ∈ R^{m×m} for vector transformation (square matrix)
      - indexed periods for two spatial dimensions:
          period1[i,j,g] = i*(m*o) + j*o + g + 2
          period2[i,j,g] = i*(m*o) + j*o + g + 3   (or similar offset)
      - basis function phi_{i,j,g}(k1,k2) = cos(2π*k1/period1[i,j,g]) * cos(2π*k2/period2[i,j,g])
      - supports 'linear' or 'nonlinear' (step-by-step) window extraction
      - batch acceleration for equal-sized grids when rank_mode == 'drop'
    """
    def __init__(self, vec_dim, rank=1, rank_op='avg', rank_mode='drop', num_basis=5, mode='linear',
                 user_step=None, device='cuda'):
        """
        Initialize the Spatial Numerical Dual Descriptor model for 2D arrays.

        Args:
            vec_dim: Dimension of input vectors and internal representation (square matrix)
            rank: Window size (int or tuple (rank_h, rank_w)); if int, use square windows of size rank x rank
            rank_op: 'avg', 'sum', 'pick', 'user_func'
            rank_mode: 'pad' or 'drop' (how to handle incomplete windows at borders)
            num_basis: number of basis terms
            mode: 'linear' (step=1) or 'nonlinear' (step=user_step or rank)
            user_step: custom step size for nonlinear mode (int or tuple (step_h, step_w))
            device: 'cuda' or 'cpu'
        """
        super().__init__()
        self.vec_dim = vec_dim
        # Handle rank as int or tuple
        if isinstance(rank, int):
            self.rank_h = rank
            self.rank_w = rank
        else:
            self.rank_h, self.rank_w = rank
        self.rank_op = rank_op
        self.rank_mode = rank_mode
        self.m = vec_dim
        self.o = num_basis
        assert mode in ('linear', 'nonlinear')
        self.mode = mode
        # Handle step as int or tuple
        if user_step is None:
            self.step_h = self.step_w = None
        elif isinstance(user_step, int):
            self.step_h = self.step_w = user_step
        else:
            self.step_h, self.step_w = user_step
        self.trained = False
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # Mapping matrix M
        self.M = nn.Linear(self.vec_dim, self.vec_dim, bias=False)

        # Position-weight tensor P[i][j][g]
        self.P = nn.Parameter(torch.empty(self.m, self.m, self.o))

        # Precompute indexed periods for both dimensions (fixed, not trainable)
        periods1 = torch.zeros(self.m, self.m, self.o, dtype=torch.float32)
        periods2 = torch.zeros(self.m, self.m, self.o, dtype=torch.float32)
        for i in range(self.m):
            for j in range(self.m):
                for g in range(self.o):
                    base = i*(self.m*self.o) + j*self.o + g
                    periods1[i, j, g] = base + 2
                    periods2[i, j, g] = base + 3   # different offset for second dimension
        self.register_buffer('periods1', periods1)
        self.register_buffer('periods2', periods2)

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

    def extract_vectors(self, array):
        """
        Extract window vectors from 2D array based on processing mode and rank operation.

        - 'linear': Slide window by step 1 in both directions, extracting contiguous windows of size rank_h x rank_w
        - 'nonlinear': Slide window by custom step (step_h, step_w) in both directions

        For nonlinear mode, handles incomplete trailing windows using:
        - 'pad': Pads with zero vectors to maintain full window size
        - 'drop': Discards incomplete windows

        Args:
            array (torch.Tensor): 2D array of shape [H, W, vec_dim] (or convertible)

        Returns:
            torch.Tensor: Tensor of shape [num_windows, vec_dim] after applying rank operation to each window
        """
        # Convert to tensor if needed
        if not isinstance(array, torch.Tensor):
            array = torch.tensor(array, dtype=torch.float32, device=self.device)
        if array.device != self.device:
            array = array.to(self.device)

        H, W, _ = array.shape

        def apply_op(win_tensor):
            """Apply rank operation (avg/sum/pick/user_func) to a window of shape [rank_h, rank_w, vec_dim]"""
            if self.rank_op == 'sum':
                return torch.sum(win_tensor, dim=(0, 1))
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
                    avg = torch.mean(win_tensor, dim=(0, 1))
                    return torch.sigmoid(avg)
            else:  # 'avg' is default
                return torch.mean(win_tensor, dim=(0, 1))

        # Determine step sizes
        if self.mode == 'linear':
            step_h, step_w = 1, 1
        else:
            step_h = self.step_h if self.step_h is not None else self.rank_h
            step_w = self.step_w if self.step_w is not None else self.rank_w

        windows = []
        # Slide over height
        for i in range(0, H, step_h):
            if i + self.rank_h > H:
                if self.rank_mode == 'pad':
                    # Pad the last window with zeros along height
                    frag = array[i:H, :, :]
                    pad_h = self.rank_h - frag.shape[0]
                    # Pad with zeros in height dimension
                    padding = torch.zeros(pad_h, frag.shape[1], self.vec_dim, device=self.device)
                    frag = torch.cat([frag, padding], dim=0)
                    # Now frag is [rank_h, W, m] – but we still need to handle width
                else:
                    continue  # drop incomplete
            else:
                frag = array[i:i+self.rank_h, :, :]  # [rank_h, W, m]

            # Slide over width
            for j in range(0, W, step_w):
                if j + self.rank_w > W:
                    if self.rank_mode == 'pad':
                        # Pad the last window along width
                        sub = frag[:, j:W, :]
                        pad_w = self.rank_w - sub.shape[1]
                        padding = torch.zeros(sub.shape[0], pad_w, self.vec_dim, device=self.device)
                        sub = torch.cat([sub, padding], dim=1)
                    else:
                        continue
                else:
                    sub = frag[:, j:j+self.rank_w, :]
                # sub shape: [rank_h, rank_w, m]
                windows.append(apply_op(sub))

        if windows:
            return torch.stack(windows)  # [num_windows, m]
        else:
            return torch.empty(0, self.vec_dim, device=self.device)

    # --------------------------------------------------------------------------
    # Batch acceleration methods (for rank_mode == 'drop' and equal-sized grids)
    # --------------------------------------------------------------------------
    def _aggregate_window(self, window_vectors):
        """
        Aggregate vectors within a window according to rank_op.
        window_vectors: shape [batch, n_windows, rank_h*rank_w, m]
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
        Compute sequence representations for a batch of grids efficiently.
        Assumes all grids have the same dimensions (H, W) and rank_mode == 'drop'.

        Args:
            batch_grids: Tensor of shape [batch_size, H, W, vec_dim]

        Returns:
            representations: Tensor of shape [batch_size, vec_dim]
        """
        batch_grids = batch_grids.to(self.device)
        batch_size, H, W, m = batch_grids.shape

        # Get window start indices
        row_starts = self._get_row_starts(H)
        col_starts = self._get_col_starts(W)
        if not row_starts or not col_starts:
            return torch.zeros(batch_size, m, device=self.device)

        n_rows = len(row_starts)
        n_cols = len(col_starts)
        n_windows = n_rows * n_cols

        # Permute to [batch, channels, height, width] for unfold
        x = batch_grids.permute(0, 3, 1, 2)  # [batch, m, H, W]

        step_h = 1 if self.mode == 'linear' else (self.step_h if self.step_h is not None else self.rank_h)
        step_w = 1 if self.mode == 'linear' else (self.step_w if self.step_w is not None else self.rank_w)

        # Use unfold to extract all windows (raw vectors, no M applied yet)
        windows = torch.nn.functional.unfold(
            x, kernel_size=(self.rank_h, self.rank_w), stride=(step_h, step_w)
        )  # [batch, m*rank_h*rank_w, n_windows]

        # Reshape to [batch, n_windows, rank_h, rank_w, m]
        windows = windows.view(batch_size, m, self.rank_h, self.rank_w, n_windows).permute(0, 4, 2, 3, 1)
        # [batch, n_windows, rank_h, rank_w, m]

        # Flatten the spatial dimensions of the window
        windows_flat = windows.reshape(batch_size, n_windows, self.rank_h * self.rank_w, m)  # [batch, n_windows, n_vecs, m]

        # Aggregate each window according to rank_op (no M yet)
        agg = self._aggregate_window(windows_flat)  # [batch, n_windows, m]

        # Build k1 and k2 coordinates for each window
        k1_vals = torch.tensor(row_starts, dtype=torch.float32, device=self.device)
        k2_vals = torch.tensor(col_starts, dtype=torch.float32, device=self.device)
        k1_grid, k2_grid = torch.meshgrid(k1_vals, k2_vals, indexing='ij')
        k1_coords = k1_grid.flatten()  # [n_windows]
        k2_coords = k2_grid.flatten()  # [n_windows]

        # Expand to batch dimension
        k1_expanded = k1_coords.unsqueeze(0).expand(batch_size, -1)  # [batch, n_windows]
        k2_expanded = k2_coords.unsqueeze(0).expand(batch_size, -1)

        # Flatten agg and coordinates to compute Nk
        agg_flat = agg.reshape(batch_size * n_windows, m)  # [batch*n_windows, m]
        k1_flat = k1_expanded.reshape(batch_size * n_windows)
        k2_flat = k2_expanded.reshape(batch_size * n_windows)

        # Compute Nk for all windows (this applies M inside)
        Nk_flat = self.batch_compute_Nk(k1_flat, k2_flat, agg_flat)  # [batch*n_windows, m]
        Nk = Nk_flat.reshape(batch_size, n_windows, m)  # [batch, n_windows, m]

        # Average over windows to get grid-level representation
        rep = Nk.mean(dim=1)  # [batch, m]
        return rep

    def batch_compute_Nk_and_targets(self, batch_grids):
        """
        Compute Nk vectors and target vectors for each window in a batch of grids.
        Assumes all grids have the same dimensions (H, W) and rank_mode == 'drop'.
        Used in self-training.

        Args:
            batch_grids: Tensor of shape [batch_size, H, W, m]

        Returns:
            Nk_all: Tensor [batch_size, n_windows, m] - Nk vectors for each window
            targets: Tensor [batch_size, n_windows, m] - target vectors = M(aggregated(window))
        """
        batch_grids = batch_grids.to(self.device)
        batch_size, H, W, m = batch_grids.shape

        # Get window start indices
        row_starts = self._get_row_starts(H)
        col_starts = self._get_col_starts(W)
        if not row_starts or not col_starts:
            return torch.empty(batch_size, 0, m, device=self.device), torch.empty(batch_size, 0, m, device=self.device)

        n_rows = len(row_starts)
        n_cols = len(col_starts)
        n_windows = n_rows * n_cols

        # Permute to [batch, channels, height, width] for unfold
        x = batch_grids.permute(0, 3, 1, 2)  # [batch, m, H, W]

        step_h = 1 if self.mode == 'linear' else (self.step_h if self.step_h is not None else self.rank_h)
        step_w = 1 if self.mode == 'linear' else (self.step_w if self.step_w is not None else self.rank_w)

        # Extract windows (raw vectors, no M)
        windows = torch.nn.functional.unfold(
            x, kernel_size=(self.rank_h, self.rank_w), stride=(step_h, step_w)
        )  # [batch, m*rank_h*rank_w, n_windows]

        windows = windows.view(batch_size, m, self.rank_h, self.rank_w, n_windows).permute(0, 4, 2, 3, 1)
        windows_flat = windows.reshape(batch_size, n_windows, self.rank_h * self.rank_w, m)  # [batch, n_windows, n_vecs, m]

        # Aggregate each window (no M)
        agg = self._aggregate_window(windows_flat)  # [batch, n_windows, m]

        # Compute targets: M(agg)
        agg_flat = agg.reshape(batch_size * n_windows, m)
        targets_flat = self.M(agg_flat)  # [batch*n_windows, m]
        targets = targets_flat.reshape(batch_size, n_windows, m)

        # Build coordinates
        k1_vals = torch.tensor(row_starts, dtype=torch.float32, device=self.device)
        k2_vals = torch.tensor(col_starts, dtype=torch.float32, device=self.device)
        k1_grid, k2_grid = torch.meshgrid(k1_vals, k2_vals, indexing='ij')
        k1_coords = k1_grid.flatten()
        k2_coords = k2_grid.flatten()
        k1_expanded = k1_coords.unsqueeze(0).expand(batch_size, -1)
        k2_expanded = k2_coords.unsqueeze(0).expand(batch_size, -1)

        # Compute Nk from agg (this applies M inside)
        agg_flat2 = agg.reshape(batch_size * n_windows, m)
        k1_flat = k1_expanded.reshape(batch_size * n_windows)
        k2_flat = k2_expanded.reshape(batch_size * n_windows)
        Nk_flat = self.batch_compute_Nk(k1_flat, k2_flat, agg_flat2)  # [batch*n_windows, m]
        Nk_all = Nk_flat.reshape(batch_size, n_windows, m)

        return Nk_all, targets

    # --------------------------------------------------------------------------
    # Core methods (describe, S, D, d)
    # --------------------------------------------------------------------------
    def batch_compute_Nk(self, k1_tensor, k2_tensor, vectors):
        """
        Vectorized computation of N(k1,k2) vectors for a batch of positions and vectors.
        Optimized using einsum for better performance.

        Args:
            k1_tensor: Tensor of row indices [batch_size]
            k2_tensor: Tensor of column indices [batch_size]
            vectors: Tensor of vectors [batch_size, vec_dim]

        Returns:
            Tensor of N(k1,k2) vectors [batch_size, m]
        """
        # Apply mapping matrix M to each vector
        x = self.M(vectors)  # [batch_size, m]

        # Expand dimensions for broadcasting
        k1_exp = k1_tensor.view(-1, 1, 1, 1)  # [batch, 1, 1, 1]
        k2_exp = k2_tensor.view(-1, 1, 1, 1)

        # Calculate basis functions for both dimensions
        phi1 = torch.cos(2 * math.pi * k1_exp / self.periods1)  # [batch, m, m, o]
        phi2 = torch.cos(2 * math.pi * k2_exp / self.periods2)  # [batch, m, m, o]
        phi = phi1 * phi2

        # Optimized computation using einsum
        Nk = torch.einsum('bj,ijg,bijg->bi', x, self.P, phi)
        return Nk

    def describe(self, array):
        """
        Compute N(k1,k2) vectors for each window in the 2D array.

        Args:
            array: 2D tensor of shape [H, W, m]

        Returns:
            list of numpy arrays, each of shape [m]
        """
        extracted_vectors = self.extract_vectors(array)
        if extracted_vectors.shape[0] == 0:
            return []

        # Generate grid coordinates for each window (row-major order)
        H, W, _ = array.shape
        step_h = 1 if self.mode == 'linear' else (self.step_h if self.step_h is not None else self.rank_h)
        step_w = 1 if self.mode == 'linear' else (self.step_w if self.step_w is not None else self.rank_w)

        # Count number of windows along height and width
        nh = (H - self.rank_h) // step_h + 1 if self.rank_mode == 'drop' else (H + step_h - 1) // step_h
        nw = (W - self.rank_w) // step_w + 1 if self.rank_mode == 'drop' else (W + step_w - 1) // step_w

        k1_vals = []
        k2_vals = []
        for i in range(0, H, step_h):
            for j in range(0, W, step_w):
                k1_vals.append(i)
                k2_vals.append(j)
        # If pad mode, we may have more windows; we need to generate them consistently
        # For simplicity, we recompute windows positions by iterating again
        # In describe we only need the extracted_vectors count, so we assume they match
        # We'll generate k1,k2 lists by reusing the same logic as extract_vectors
        # To avoid duplication, we can compute positions during extraction, but for simplicity,
        # we assume windows are generated in the same order as extract_vectors.
        # We'll re-implement the sliding logic to get positions.
        positions = []
        # Loop similarly to extract_vectors
        for i in range(0, H, step_h):
            if i + self.rank_h > H and self.rank_mode == 'drop':
                continue
            for j in range(0, W, step_w):
                if j + self.rank_w > W and self.rank_mode == 'drop':
                    continue
                positions.append((i, j))
        if len(positions) != extracted_vectors.shape[0]:
            # In pad mode, we may have extra windows; we need to align.
            # For simplicity, we'll regenerate using the exact same loop as in extract_vectors
            # but that's complex. We'll just recompute using the same method but without storing.
            # Here we trust that positions correspond exactly.
            pass
        # Convert to tensors
        k1_tensor = torch.tensor([p[0] for p in positions], dtype=torch.float32, device=self.device)
        k2_tensor = torch.tensor([p[1] for p in positions], dtype=torch.float32, device=self.device)

        N_batch = self.batch_compute_Nk(k1_tensor, k2_tensor, extracted_vectors)
        return N_batch.detach().cpu().numpy()

    def S(self, array):
        """
        Compute cumulative sums of N(k1,k2) vectors along row-major order.
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
        Compute mean squared deviation D across 2D arrays:
        D = average over all windows of (N(k1,k2)-t)^2
        """
        total_loss = 0.0
        total_windows = 0

        t_tensors = [torch.tensor(t, dtype=torch.float32, device=self.device) for t in t_list]

        for array, t in zip(arrays, t_tensors):
            extracted_vectors = self.extract_vectors(array)
            if extracted_vectors.shape[0] == 0:
                continue

            # Generate positions (same as in describe)
            H, W, _ = array.shape
            step_h = 1 if self.mode == 'linear' else (self.step_h if self.step_h is not None else self.rank_h)
            step_w = 1 if self.mode == 'linear' else (self.step_w if self.step_w is not None else self.rank_w)
            positions = []
            for i in range(0, H, step_h):
                if i + self.rank_h > H and self.rank_mode == 'drop':
                    continue
                for j in range(0, W, step_w):
                    if j + self.rank_w > W and self.rank_mode == 'drop':
                        continue
                    positions.append((i, j))
            k1_tensor = torch.tensor([p[0] for p in positions], dtype=torch.float32, device=self.device)
            k2_tensor = torch.tensor([p[1] for p in positions], dtype=torch.float32, device=self.device)

            N_batch = self.batch_compute_Nk(k1_tensor, k2_tensor, extracted_vectors)
            losses = torch.sum((N_batch - t) ** 2, dim=1)
            total_loss += losses.sum().item()
            total_windows += extracted_vectors.shape[0]

        return total_loss / total_windows if total_windows else 0.0

    def d(self, array, t):
        """Compute pattern deviation value (d) for a single 2D array."""
        return self.D([array], [t])

    # --------------------------------------------------------------------------
    # Training methods with batch acceleration
    # --------------------------------------------------------------------------
    def reg_train(self, arrays, t_list, max_iters=1000, tol=1e-8, learning_rate=0.01,
                  continued=False, decay_rate=1.0, print_every=10, batch_size=32,
                  checkpoint_file=None, checkpoint_interval=10):
        """
        Train the model using gradient descent for regression on 2D arrays.
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
                shapes = [arr.shape if isinstance(arr, torch.Tensor) else (len(arr), len(arr[0]), self.vec_dim) for arr in batch_arrays]
                if use_fast_batch and len(set(shapes)) == 1:
                    # All grids have equal dimensions -> use fast batch processing
                    # Convert grids to a single tensor, moving to device
                    batch_grids_tensor = torch.stack([arr.to(self.device) if isinstance(arr, torch.Tensor) else torch.tensor(arr, dtype=torch.float32, device=self.device) for arr in batch_arrays], dim=0)
                    # Compute representations (fast path)
                    reps = self.batch_represent(batch_grids_tensor)  # [batch, m]
                    # Stack targets
                    targets_tensor = torch.stack(batch_targets, dim=0)  # [batch, m]
                    # Compute loss
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

                        # Compute positions (same as before)
                        H, W, _ = arr.shape
                        step_h = 1 if self.mode == 'linear' else (self.step_h if self.step_h is not None else self.rank_h)
                        step_w = 1 if self.mode == 'linear' else (self.step_w if self.step_w is not None else self.rank_w)
                        positions = []
                        for i in range(0, H, step_h):
                            if i + self.rank_h > H and self.rank_mode == 'drop':
                                continue
                            for j in range(0, W, step_w):
                                if j + self.rank_w > W and self.rank_mode == 'drop':
                                    continue
                                positions.append((i, j))
                        k1_tensor = torch.tensor([p[0] for p in positions], dtype=torch.float32, device=self.device)
                        k2_tensor = torch.tensor([p[1] for p in positions], dtype=torch.float32, device=self.device)

                        N_batch = self.batch_compute_Nk(k1_tensor, k2_tensor, extracted_vectors)
                        seq_pred = torch.mean(N_batch, dim=0)
                        seq_loss = torch.sum((seq_pred - target) ** 2)
                        batch_loss += seq_loss

                        del N_batch, seq_pred, extracted_vectors, k1_tensor, k2_tensor

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
        """Train for multi-class classification on 2D arrays with batch acceleration."""
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

                shapes = [arr.shape if isinstance(arr, torch.Tensor) else (len(arr), len(arr[0]), self.vec_dim) for arr in batch_arrays]
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
                            H, W, _ = arr.shape
                            step_h = 1 if self.mode == 'linear' else (self.step_h if self.step_h is not None else self.rank_h)
                            step_w = 1 if self.mode == 'linear' else (self.step_w if self.step_w is not None else self.rank_w)
                            positions = []
                            for i in range(0, H, step_h):
                                if i + self.rank_h > H and self.rank_mode == 'drop':
                                    continue
                                for j in range(0, W, step_w):
                                    if j + self.rank_w > W and self.rank_mode == 'drop':
                                        continue
                                    positions.append((i, j))
                            k1_tensor = torch.tensor([p[0] for p in positions], dtype=torch.float32, device=self.device)
                            k2_tensor = torch.tensor([p[1] for p in positions], dtype=torch.float32, device=self.device)
                            N_batch = self.batch_compute_Nk(k1_tensor, k2_tensor, extracted_vectors)
                            seq_vector = torch.mean(N_batch, dim=0)
                            del N_batch, extracted_vectors, k1_tensor, k2_tensor
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
        """Train for multi-label classification on 2D arrays with batch acceleration."""
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

                shapes = [arr.shape if isinstance(arr, torch.Tensor) else (len(arr), len(arr[0]), self.vec_dim) for arr in batch_arrays]
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
                        H, W, _ = arr.shape
                        step_h = 1 if self.mode == 'linear' else (self.step_h if self.step_h is not None else self.rank_h)
                        step_w = 1 if self.mode == 'linear' else (self.step_w if self.step_w is not None else self.rank_w)
                        positions = []
                        for i in range(0, H, step_h):
                            if i + self.rank_h > H and self.rank_mode == 'drop':
                                continue
                            for j in range(0, W, step_w):
                                if j + self.rank_w > W and self.rank_mode == 'drop':
                                    continue
                                positions.append((i, j))
                        k1_tensor = torch.tensor([p[0] for p in positions], dtype=torch.float32, device=self.device)
                        k2_tensor = torch.tensor([p[1] for p in positions], dtype=torch.float32, device=self.device)
                        N_batch = self.batch_compute_Nk(k1_tensor, k2_tensor, extracted_vectors)
                        seq_rep = torch.mean(N_batch, dim=0)
                        logits = self.labeller(seq_rep)
                        batch_predictions_list.append(logits)
                        del N_batch, seq_rep, extracted_vectors, k1_tensor, k2_tensor

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
        Self-training for self-consistency on 2D arrays with batch acceleration.
        Trains so that N(k1,k2) matches M(window_vector) at each window.
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

                shapes = [arr.shape if isinstance(arr, torch.Tensor) else (len(arr), len(arr[0]), self.vec_dim) for arr in batch_arrays]
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
                        H, W, _ = arr.shape
                        step_h = 1 if self.mode == 'linear' else (self.step_h if self.step_h is not None else self.rank_h)
                        step_w = 1 if self.mode == 'linear' else (self.step_w if self.step_w is not None else self.rank_w)
                        positions = []
                        for i in range(0, H, step_h):
                            if i + self.rank_h > H and self.rank_mode == 'drop':
                                continue
                            for j in range(0, W, step_w):
                                if j + self.rank_w > W and self.rank_mode == 'drop':
                                    continue
                                positions.append((i, j))
                        k1_tensor = torch.tensor([p[0] for p in positions], dtype=torch.float32, device=self.device)
                        k2_tensor = torch.tensor([p[1] for p in positions], dtype=torch.float32, device=self.device)
                        N_batch = self.batch_compute_Nk(k1_tensor, k2_tensor, extracted_vectors)
                        target_vectors = self.M(extracted_vectors)  # [num_windows, m]

                        loss = torch.mean(torch.sum((N_batch - target_vectors) ** 2, dim=1))
                        batch_loss += loss
                        batch_count += 1
                        del N_batch, target_vectors, extracted_vectors, k1_tensor, k2_tensor

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
                H, W, _ = arr.shape
                step_h = 1 if self.mode == 'linear' else (self.step_h if self.step_h is not None else self.rank_h)
                step_w = 1 if self.mode == 'linear' else (self.step_w if self.step_w is not None else self.rank_w)
                positions = []
                for i in range(0, H, step_h):
                    if i + self.rank_h > H and self.rank_mode == 'drop':
                        continue
                    for j in range(0, W, step_w):
                        if j + self.rank_w > W and self.rank_mode == 'drop':
                            continue
                        positions.append((i, j))
                k1_tensor = torch.tensor([p[0] for p in positions], dtype=torch.float32, device=self.device)
                k2_tensor = torch.tensor([p[1] for p in positions], dtype=torch.float32, device=self.device)
                N_batch = self.batch_compute_Nk(k1_tensor, k2_tensor, extracted_vectors)
                total_t += N_batch.sum(dim=0)
                total_windows += extracted_vectors.shape[0]
                del N_batch, extracted_vectors, k1_tensor, k2_tensor

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
        """Predict target vector for a 2D array (average of all N(k1,k2) vectors)."""
        extracted_vectors = self.extract_vectors(array)
        if extracted_vectors.shape[0] == 0:
            return [0.0] * self.m
        H, W, _ = array.shape
        step_h = 1 if self.mode == 'linear' else (self.step_h if self.step_h is not None else self.rank_h)
        step_w = 1 if self.mode == 'linear' else (self.step_w if self.step_w is not None else self.rank_w)
        positions = []
        for i in range(0, H, step_h):
            if i + self.rank_h > H and self.rank_mode == 'drop':
                continue
            for j in range(0, W, step_w):
                if j + self.rank_w > W and self.rank_mode == 'drop':
                    continue
                positions.append((i, j))
        k1_tensor = torch.tensor([p[0] for p in positions], dtype=torch.float32, device=self.device)
        k2_tensor = torch.tensor([p[1] for p in positions], dtype=torch.float32, device=self.device)
        N_batch = self.batch_compute_Nk(k1_tensor, k2_tensor, extracted_vectors)
        avg = torch.mean(N_batch, dim=0).detach().cpu().numpy()
        return avg

    def predict_c(self, array):
        """Predict class label for a 2D array using the classification head."""
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
        """Predict multi-label classification for a 2D array."""
        assert self.labeller is not None, "Model must be trained first for label prediction"
        extracted_vectors = self.extract_vectors(array)
        if extracted_vectors.shape[0] == 0:
            return np.zeros(self.num_labels, dtype=np.float32), np.zeros(self.num_labels, dtype=np.float32)
        H, W, _ = array.shape
        step_h = 1 if self.mode == 'linear' else (self.step_h if self.step_h is not None else self.rank_h)
        step_w = 1 if self.mode == 'linear' else (self.step_w if self.step_w is not None else self.rank_w)
        positions = []
        for i in range(0, H, step_h):
            if i + self.rank_h > H and self.rank_mode == 'drop':
                continue
            for j in range(0, W, step_w):
                if j + self.rank_w > W and self.rank_mode == 'drop':
                    continue
                positions.append((i, j))
        k1_tensor = torch.tensor([p[0] for p in positions], dtype=torch.float32, device=self.device)
        k2_tensor = torch.tensor([p[1] for p in positions], dtype=torch.float32, device=self.device)
        N_batch = self.batch_compute_Nk(k1_tensor, k2_tensor, extracted_vectors)
        seq_rep = torch.mean(N_batch, dim=0)
        with torch.no_grad():
            logits = self.labeller(seq_rep)
            probs = torch.sigmoid(logits).cpu().numpy()
        binary = (probs > threshold).astype(np.float32)
        return binary, probs

    def reconstruct(self, H, W, tau=0.0):
        """
        Reconstruct a 2D array of shape (H, W, m) by minimizing error with temperature.
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

        # Determine number of windows
        nh = (H - self.rank_h) // step_h + 1 if self.rank_mode == 'drop' else (H + step_h - 1) // step_h
        nw = (W - self.rank_w) // step_w + 1 if self.rank_mode == 'drop' else (W + step_w - 1) // step_w

        # Pre-generate candidate vectors
        num_candidates = 100
        candidates = torch.randn(num_candidates, self.vec_dim, device=self.device)

        # For each window position, choose a candidate vector
        chosen_vectors = {}  # (i,j) -> vector
        for i_idx in range(nh):
            i = i_idx * step_h
            for j_idx in range(nw):
                j = j_idx * step_w
                # Compute N for all candidates at this position
                k1_tensor = torch.full((num_candidates,), i, dtype=torch.float32, device=self.device)
                k2_tensor = torch.full((num_candidates,), j, dtype=torch.float32, device=self.device)
                N_all = self.batch_compute_Nk(k1_tensor, k2_tensor, candidates)
                errors = torch.sum((N_all - mean_t_tensor) ** 2, dim=1)
                scores = -errors
                if tau == 0:
                    idx = torch.argmax(scores).item()
                else:
                    probs = torch.softmax(scores / tau, dim=0).detach().cpu().numpy()
                    idx = random.choices(range(num_candidates), weights=probs, k=1)[0]
                chosen_vectors[(i, j)] = candidates[idx]

        # Build the output array by averaging overlapping windows
        # We'll create a tensor of zeros and a count tensor
        result = torch.zeros(H, W, self.vec_dim, device=self.device)
        count = torch.zeros(H, W, device=self.device)
        for (i, j), vec in chosen_vectors.items():
            for di in range(self.rank_h):
                if i+di >= H:
                    continue
                for dj in range(self.rank_w):
                    if j+dj >= W:
                        continue
                    result[i+di, j+dj] += vec
                    count[i+di, j+dj] += 1
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
    print("Spatial Numerical Dual Descriptor TS2 - 2D Array Version with Batch Acceleration")
    print("Processes 2D grids of m-dimensional real vectors")
    print("="*50)

    torch.manual_seed(11)
    random.seed(11)
    np.random.seed(11)

    # Parameters
    vec_dim = 10
    num_basis = 10
    rank = 3                # square window 3x3
    user_step = 3           # step size for nonlinear mode

    ndd = SpatialNumDualDescriptorTS2(
        vec_dim=vec_dim,
        rank=rank,
        num_basis=num_basis,
        mode='nonlinear',
        user_step=user_step,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    print(f"\nUsing device: {ndd.device}")
    print(f"Vector dimension: {vec_dim}")
    print(f"Window size: {rank}x{rank}, step: {user_step}")

    # Generate 50 random 2D arrays (H x W x vec_dim) with random target vectors
    print("\nGenerating training data...")
    arrays, t_list = [], []
    for _ in range(50):
        H = random.randint(20, 30)
        W = random.randint(20, 30)
        arr = torch.randn(H, W, vec_dim)
        arrays.append(arr)
        t_list.append(np.random.uniform(-1.0, 1.0, vec_dim))

    # Regression training
    print("\n" + "="*50)
    print("Starting Regression Training")
    print("="*50)
    ndd.reg_train(arrays, t_list, max_iters=50, tol=1e-9, learning_rate=0.1,
                  decay_rate=0.99, batch_size=8, print_every=5)

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
    print("\nReconstructing a 2D array...")
    H_rec, W_rec = 20, 20
    rec_det = ndd.reconstruct(H_rec, W_rec, tau=0.0)
    rec_rand = ndd.reconstruct(H_rec, W_rec, tau=0.5)
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
        for _ in range(30):
            H = random.randint(20, 30)
            W = random.randint(20, 30)
            if class_id == 0:
                arr = torch.randn(H, W, vec_dim) + 1.0
            elif class_id == 1:
                arr = torch.randn(H, W, vec_dim) - 1.0
            else:
                arr = torch.randn(H, W, vec_dim)
            class_arrays.append(arr)
            class_labels.append(class_id)

    ndd_cls = SpatialNumDualDescriptorTS2(
        vec_dim=vec_dim, rank=rank, num_basis=num_basis,
        mode='nonlinear', user_step=user_step, device=ndd.device
    )
    print("\nStarting Classification Training")
    ndd_cls.cls_train(class_arrays, class_labels, num_classes,
                      max_iters=20, tol=1e-8, learning_rate=0.05,
                      decay_rate=0.99, batch_size=8, print_every=5)

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
    for _ in range(50):
        H = random.randint(20, 30)
        W = random.randint(20, 30)
        arr = torch.randn(H, W, vec_dim)
        label_arrays.append(arr)
        label_vec = [random.random() > 0.7 for _ in range(num_labels)]
        ml_labels.append([1.0 if x else 0.0 for x in label_vec])

    ndd_lbl = SpatialNumDualDescriptorTS2(
        vec_dim=vec_dim, rank=rank, num_basis=num_basis,
        mode='nonlinear', user_step=user_step, device=ndd.device
    )
    print("\nStarting Multi-Label Training")
    loss_hist, acc_hist = ndd_lbl.lbl_train(
        label_arrays, ml_labels, num_labels,
        max_iters=50, tol=1e-8, learning_rate=0.1,
        decay_rate=0.99, batch_size=8, print_every=5
    )
    print(f"Final multi-label accuracy: {acc_hist[-1]:.4f}")

    # Self-training
    print("\n" + "="*50)
    print("Self-Training Example")
    print("="*50)
    ndd_self = SpatialNumDualDescriptorTS2(
        vec_dim=vec_dim, rank=rank, num_basis=num_basis,
        mode='nonlinear', user_step=user_step, device=ndd.device
    )
    self_arrays = [torch.randn(25, 25, vec_dim) for _ in range(10)]
    print("Training for self-consistency...")
    self_hist = ndd_self.self_train(self_arrays, max_iters=15, learning_rate=0.01, batch_size=4)

    plt.figure(figsize=(10,6))
    plt.plot(self_hist)
    plt.title('Self-Training Loss History')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig('self_training_loss_2d.png')
    print("\nSelf-training loss plot saved as 'self_training_loss_2d.png'")

    print("\nAll tests completed successfully!")
