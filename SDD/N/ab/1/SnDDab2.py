# Copyright (C) 2005-2026, Bin-Guang Ma (mbg@mail.hzau.edu.cn); SPDX-License-Identifier: MIT
# The Spatial Numeric Dual Descriptor Vector class (AB matrix form) for 2D array implemented with PyTorch
# This program is for the demonstration of methodology and not fully refined.
# Author: Bin-Guang Ma (assisted by DeepSeek); Date: 2025-8-28 ~ 2026-04-02

import math
import itertools
import random
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
import os

class SpatialNumDualDescriptorAB2(nn.Module):
    """
    Spatial Numeric Dual Descriptor for 2D arrays of vectors with GPU acceleration using PyTorch:
      - learnable coefficient matrix Acoeff ∈ R^{m×(L1*L2)}
      - fixed basis matrix Bbasis ∈ R^{(L1*L2)×m}, 
        Bbasis[j1*L2+j2][i] = cos(2π*(j1+1)/(i+2)) * cos(2π*(j2+1)/(i+2))
      - learnable mapping matrix M ∈ R^{m×m} for input vector transformation
      - Supports both linear and nonlinear tokenization of 2D vector arrays
      - Batch processing for GPU acceleration
      - Supports regression, classification, and multi-label classification tasks
    """
    def __init__(self, vec_dim=4, bas_dim1=7, bas_dim2=7, 
                 rank1=2, rank2=2, rank_op='avg', rank_mode='drop', 
                 mode='linear', user_step1=None, user_step2=None, device='cuda'):
        """
        Initialize the Spatial Dual Descriptor for 2D vector arrays.
        
        Args:
            vec_dim (int): Dimension m of input vectors
            bas_dim1 (int): Basis dimension in first direction (L1)
            bas_dim2 (int): Basis dimension in second direction (L2)
            rank1 (int): Window size in first direction
            rank2 (int): Window size in second direction
            rank_op (str): 'avg', 'sum', 'pick', or 'user_func'
            rank_mode (str): 'pad' or 'drop' for handling incomplete windows
            mode (str): 'linear' (sliding window by 1) or 'nonlinear' (stepped window)
            user_step1 (int): Custom step size in first direction for nonlinear mode
            user_step2 (int): Custom step size in second direction for nonlinear mode
            device (str): 'cuda' or 'cpu'
        """
        super().__init__()
        self.m = vec_dim
        self.L1 = bas_dim1
        self.L2 = bas_dim2
        self.L = self.L1 * self.L2   # total basis dimension
        self.rank1 = rank1
        self.rank2 = rank2
        self.rank_op = rank_op
        self.rank_mode = rank_mode
        assert mode in ('linear','nonlinear')
        self.mode = mode
        self.step1 = user_step1 if user_step1 is not None else rank1
        self.step2 = user_step2 if user_step2 is not None else rank2
        self.trained = False
        self.mean_t = None  
        self.mean_window_count = None
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Learnable mapping matrix M (m x m)
        self.M = nn.Parameter(torch.empty(self.m, self.m))
        
        # Coefficient matrix A (m x L)
        self.Acoeff = nn.Parameter(torch.empty(self.m, self.L))
        
        # Fixed basis matrix B (L x m)
        Bbasis = torch.empty(self.L, self.m)
        for idx in range(self.L):
            j1 = idx // self.L2
            j2 = idx % self.L2
            for i in range(self.m):
                Bbasis[idx, i] = (math.cos(2 * math.pi * (j1+1) / (i+2)) *
                                  math.cos(2 * math.pi * (j2+1) / (i+2)))
        self.register_buffer('Bbasis', Bbasis)
        
        # Classification head for multi-class tasks (initialized when needed)
        self.num_classes = None
        self.classifier = None
        
        # Label head for multi-label tasks (initialized when needed)
        self.num_labels = None
        self.labeller = None
        
        # Initialize parameters
        self.reset_parameters()
        self.to(self.device)
    
    def reset_parameters(self):
        """Initialize model parameters"""
        nn.init.eye_(self.M)  # Initialize M as identity matrix
        nn.init.uniform_(self.Acoeff, -0.1, 0.1)
        
        # Initialize classifier if it exists
        if self.classifier is not None:
            nn.init.normal_(self.classifier.weight, 0, 0.01)
            if self.classifier.bias is not None:
                nn.init.constant_(self.classifier.bias, 0)
                
        # Initialize labeller if it exists
        if self.labeller is not None:
            nn.init.xavier_uniform_(self.labeller.weight)
            nn.init.zeros_(self.labeller.bias)
    
    def extract_vectors(self, vec_arr):
        """
        Extract and aggregate vector groups from a 2D vector array based on vectorization mode.
        
        - 'linear': Slide window by 1 step in both directions, extracting contiguous windows of size (rank1, rank2)
        - 'nonlinear': Slide window by step1/step2 (or rank1/rank2 if steps not specified)
        
        Handles incomplete trailing fragments using:
        - 'pad': Pads with zero vectors to maintain window size
        - 'drop': Discards incomplete fragments
        
        The method applies mapping matrix M to each vector in the window and aggregates using rank_op.
        
        Args:
            vec_arr (list or tensor): Input 2D vector array of shape [H, W, m]
            
        Returns:
            list: List of aggregated vectors from extracted windows, each is a tensor of shape [m]
            list: List of corresponding window coordinates (top-left corner) as tuples (k1, k2)
        """
        # Convert to tensor if needed
        if not isinstance(vec_arr, torch.Tensor):
            vec_arr = torch.tensor(vec_arr, dtype=torch.float32, device=self.device)
        elif vec_arr.device != self.device:
            vec_arr = vec_arr.to(self.device)
        
        H, W, _ = vec_arr.shape  # height, width, vector dimension

        # Helper function to apply vector operations to a window
        def apply_op(window):
            """
            Apply mapping and aggregation to a window of vectors.
            window shape: [rank1, rank2, m]
            """
            # Flatten window to a list of vectors
            flat_window = window.reshape(-1, self.m)  # [rank1*rank2, m]
            # Apply mapping matrix M to each vector
            transformed = torch.matmul(flat_window, self.M.T)  # [rank1*rank2, m]
            
            if self.rank_op == 'sum':
                return torch.sum(transformed, dim=0)
            elif self.rank_op == 'pick':
                idx = random.randint(0, transformed.size(0)-1)
                return transformed[idx]
            elif self.rank_op == 'user_func':
                # Default: average + sigmoid
                avg = torch.mean(transformed, dim=0)
                return torch.sigmoid(avg)
            else:  # 'avg' is default
                return torch.mean(transformed, dim=0)
        
        # Determine step sizes
        if self.mode == 'linear':
            step1 = 1
            step2 = 1
        else:  # nonlinear
            step1 = self.step1
            step2 = self.step2
        
        vectors = []
        coords = []
        
        # Slide over the array
        for i in range(0, H, step1):
            for j in range(0, W, step2):
                # Extract window fragment
                frag = vec_arr[i:i+self.rank1, j:j+self.rank2, :]
                frag_h, frag_w, _ = frag.shape
                
                # Pad or drop based on rank_mode
                if self.rank_mode == 'pad':
                    # Pad with zeros if the window is smaller than expected
                    if frag_h < self.rank1 or frag_w < self.rank2:
                        padded = torch.zeros((self.rank1, self.rank2, self.m), device=self.device)
                        padded[:frag_h, :frag_w, :] = frag
                        frag = padded
                elif self.rank_mode == 'drop':
                    # Only keep windows that are exactly the required size
                    if frag_h != self.rank1 or frag_w != self.rank2:
                        continue
                
                vectors.append(apply_op(frag))
                coords.append((i, j))  # store top-left coordinates
        
        return vectors, coords

    def _get_window_coords_and_indices(self, H, W):
        """
        Compute window top-left coordinates and corresponding basis indices for a given array size.
        Used by batch processing methods.
        
        Args:
            H, W: Height and width of the array
            
        Returns:
            coords: List of (i, j) tuples for each window
            idx_tensor: Tensor of shape [n_windows] containing flattened basis indices
        """
        step1 = 1 if self.mode == 'linear' else self.step1
        step2 = 1 if self.mode == 'linear' else self.step2
        
        if self.rank_mode == 'drop':
            # Only windows fully inside
            i_range = range(0, H - self.rank1 + 1, step1)
            j_range = range(0, W - self.rank2 + 1, step2)
        else:  # 'pad'
            # Include all windows that start before H and W (pad with zeros if needed)
            i_range = range(0, H, step1)
            j_range = range(0, W, step2)
        
        coords = []
        j1_list = []
        j2_list = []
        for i in i_range:
            for j in j_range:
                coords.append((i, j))
                j1_list.append(i % self.L1)
                j2_list.append(j % self.L2)
        
        if not coords:
            return [], torch.empty(0, dtype=torch.long, device=self.device)
        
        j1_t = torch.tensor(j1_list, dtype=torch.long, device=self.device)
        j2_t = torch.tensor(j2_list, dtype=torch.long, device=self.device)
        idx_tensor = j1_t * self.L2 + j2_t
        return coords, idx_tensor

    def batch_represent(self, arr_batch):
        """
        Compute array representations for a batch of 2D vector arrays efficiently.
        Supports arbitrary window size and step (linear/nonlinear) as long as all arrays
        in the batch have identical dimensions.
        
        Args:
            arr_batch: Tensor of shape [batch_size, H, W, m]
        
        Returns:
            representations: Tensor of shape [batch_size, m]
        """
        B, H, W, m = arr_batch.shape
        
        # Step sizes
        step1 = 1 if self.mode == 'linear' else self.step1
        step2 = 1 if self.mode == 'linear' else self.step2
        
        # Convert to (B, m, H, W) for unfold
        x = arr_batch.permute(0, 3, 1, 2)  # [B, m, H, W]
        
        # Pad if necessary (for 'pad' mode)
        if self.rank_mode == 'pad':
            # Pad such that all windows (starting at i=0,step1,2*step1,...) are fully inside
            pad_h = (self.rank1 - 1) if step1 == 1 else (self.rank1 - step1)
            pad_w = (self.rank2 - 1) if step2 == 1 else (self.rank2 - step2)
            x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode='constant', value=0)
            H_pad = H + pad_h
            W_pad = W + pad_w
        else:
            H_pad, W_pad = H, W
        
        # Use unfold to extract sliding windows
        # unfold expects (B, C, H, W) and returns (B, C*kernel_h*kernel_w, n_windows)
        windows = torch.nn.functional.unfold(
            x, kernel_size=(self.rank1, self.rank2), stride=(step1, step2)
        )  # shape: [B, m*rank1*rank2, n_windows]
        
        n_windows = windows.shape[2]
        # Reshape to [B, n_windows, rank1*rank2, m]
        windows = windows.view(B, m, self.rank1 * self.rank2, n_windows).permute(0, 3, 2, 1)
        # Now windows: [B, n_windows, win_size, m]
        
        # Apply mapping matrix M to each vector in each window
        # M is (m x m), we want to transform the last dimension: (..., m) -> (..., m)
        # Use batch matrix multiplication: windows @ M.T
        mapped = torch.matmul(windows, self.M.T)  # [B, n_windows, win_size, m]
        
        # Aggregate over win_size dimension according to rank_op
        if self.rank_op == 'sum':
            agg = mapped.sum(dim=2)  # [B, n_windows, m]
        elif self.rank_op == 'pick':
            # Random pick from each window (same pick across all windows? Not exactly)
            # For batch, we pick one random index per window (different per window)
            idx = torch.randint(0, self.rank1 * self.rank2, (B, n_windows, 1), device=self.device)
            agg = torch.gather(mapped, 2, idx.expand(-1, -1, -1, self.m)).squeeze(2)
        elif self.rank_op == 'user_func':
            avg = mapped.mean(dim=2)
            agg = torch.sigmoid(avg)
        else:  # 'avg'
            agg = mapped.mean(dim=2)  # [B, n_windows, m]
        
        # Now we have aggregated vectors for each window
        # Need to compute basis indices for each window
        # We need the top-left coordinates of each window.
        # Compute them based on stride and whether we padded.
        if self.rank_mode == 'drop':
            i_start = 0
            i_end = H - self.rank1 + 1
            j_start = 0
            j_end = W - self.rank2 + 1
        else:  # 'pad'
            i_start = 0
            i_end = H
            j_start = 0
            j_end = W
        
        i_coords = torch.arange(i_start, i_end, step1, device=self.device).view(-1, 1).expand(-1, len(range(j_start, j_end, step2))).reshape(-1)
        j_coords = torch.arange(j_start, j_end, step2, device=self.device).view(1, -1).expand(len(range(i_start, i_end, step1)), -1).reshape(-1)
        # i_coords, j_coords: [n_windows]
        
        # Compute basis indices
        j1 = i_coords % self.L1
        j2 = j_coords % self.L2
        idx_basis = j1 * self.L2 + j2  # [n_windows]
        
        # Get corresponding B rows
        B_rows = self.Bbasis[idx_basis]  # [n_windows, m]
        B_rows = B_rows.unsqueeze(0)  # [1, n_windows, m]
        
        # Get A columns
        A_cols = self.Acoeff[:, idx_basis]  # [m, n_windows]
        A_cols = A_cols.T  # [n_windows, m]
        A_cols = A_cols.unsqueeze(0)  # [1, n_windows, m]
        
        # Compute scalar = B_row • agg
        scalar = torch.sum(B_rows * agg, dim=2)  # [B, n_windows]
        
        # Compute Nk = A_cols * scalar
        Nk = A_cols * scalar.unsqueeze(2)  # [B, n_windows, m]
        
        # Average over windows to get representation
        rep = Nk.mean(dim=1)  # [B, m]
        return rep

    def batch_compute_Nk_and_targets(self, arr_batch):
        """
        Compute Nk vectors and target vectors for each position (window) in a batch of 2D arrays.
        Supports arbitrary window size and step.
        
        Args:
            arr_batch: Tensor of shape [batch_size, H, W, m]
        
        Returns:
            Nk_all: Tensor [batch_size, n_windows, m] - Nk vectors for each window
            targets: Tensor [batch_size, n_windows, m] - target vectors (aggregated mapped vectors)
        """
        B, H, W, m = arr_batch.shape
        
        step1 = 1 if self.mode == 'linear' else self.step1
        step2 = 1 if self.mode == 'linear' else self.step2
        
        x = arr_batch.permute(0, 3, 1, 2)  # [B, m, H, W]
        
        if self.rank_mode == 'pad':
            pad_h = (self.rank1 - 1) if step1 == 1 else (self.rank1 - step1)
            pad_w = (self.rank2 - 1) if step2 == 1 else (self.rank2 - step2)
            x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode='constant', value=0)
            H_pad = H + pad_h
            W_pad = W + pad_w
        else:
            H_pad, W_pad = H, W
        
        windows = torch.nn.functional.unfold(
            x, kernel_size=(self.rank1, self.rank2), stride=(step1, step2)
        )  # [B, m*win_size, n_windows]
        
        n_windows = windows.shape[2]
        windows = windows.view(B, m, self.rank1 * self.rank2, n_windows).permute(0, 3, 2, 1)
        # [B, n_windows, win_size, m]
        
        # Apply mapping
        mapped = torch.matmul(windows, self.M.T)  # [B, n_windows, win_size, m]
        
        # Aggregate to get targets
        if self.rank_op == 'sum':
            targets = mapped.sum(dim=2)
        elif self.rank_op == 'pick':
            idx = torch.randint(0, self.rank1 * self.rank2, (B, n_windows, 1), device=self.device)
            targets = torch.gather(mapped, 2, idx.expand(-1, -1, -1, self.m)).squeeze(2)
        elif self.rank_op == 'user_func':
            avg = mapped.mean(dim=2)
            targets = torch.sigmoid(avg)
        else:
            targets = mapped.mean(dim=2)  # [B, n_windows, m]
        
        # Compute basis indices for windows
        if self.rank_mode == 'drop':
            i_start = 0
            i_end = H - self.rank1 + 1
            j_start = 0
            j_end = W - self.rank2 + 1
        else:
            i_start = 0
            i_end = H
            j_start = 0
            j_end = W
        
        i_coords = torch.arange(i_start, i_end, step1, device=self.device).view(-1, 1).expand(-1, len(range(j_start, j_end, step2))).reshape(-1)
        j_coords = torch.arange(j_start, j_end, step2, device=self.device).view(1, -1).expand(len(range(i_start, i_end, step1)), -1).reshape(-1)
        
        j1 = i_coords % self.L1
        j2 = j_coords % self.L2
        idx_basis = j1 * self.L2 + j2  # [n_windows]
        
        B_rows = self.Bbasis[idx_basis]  # [n_windows, m]
        B_rows = B_rows.unsqueeze(0)  # [1, n_windows, m]
        A_cols = self.Acoeff[:, idx_basis].T  # [n_windows, m]
        A_cols = A_cols.unsqueeze(0)  # [1, n_windows, m]
        
        scalar = torch.sum(B_rows * targets, dim=2)  # [B, n_windows]
        Nk = A_cols * scalar.unsqueeze(2)  # [B, n_windows, m]
        
        return Nk, targets

    def describe(self, vec_arr):
        """Compute N(k1,k2) vectors for each window in the 2D vector array"""
        agg_vecs, coords = self.extract_vectors(vec_arr)
        if not agg_vecs:
            return []
        
        # Stack vectors for batch processing
        agg_tensor = torch.stack(agg_vecs, dim=0)  # [num_windows, m]
        num_windows = len(agg_vecs)
        
        # Compute basis indices (j1, j2) for each window
        j1_list = []
        j2_list = []
        for (k1, k2) in coords:
            j1 = k1 % self.L1
            j2 = k2 % self.L2
            j1_list.append(j1)
            j2_list.append(j2)
        
        # Convert to tensor
        j1_t = torch.tensor(j1_list, dtype=torch.long, device=self.device)
        j2_t = torch.tensor(j2_list, dtype=torch.long, device=self.device)
        # Flatten index: idx = j1 * L2 + j2
        idx = j1_t * self.L2 + j2_t  # [num_windows]
        
        # Get corresponding B rows
        B_rows = self.Bbasis[idx]  # [num_windows, m]
        
        # Compute scalar = B[idx] • agg_vec
        scalar = torch.sum(B_rows * agg_tensor, dim=1)  # [num_windows]
        
        # Get A columns for each window
        A_cols = self.Acoeff[:, idx]  # [m, num_windows] -> need to transpose for multiplication
        # Nk = scalar * A[:, idx]  -> shape [num_windows, m]
        Nk = (A_cols.T) * scalar.unsqueeze(1)  # [num_windows, m]
        
        return Nk.detach().cpu().numpy()
    
    def S(self, vec_arr):
        """Compute cumulative sum of N(k1,k2) vectors (in row-major order)"""
        agg_vecs, coords = self.extract_vectors(vec_arr)
        if not agg_vecs:
            return []
        
        agg_tensor = torch.stack(agg_vecs, dim=0)  # [num_windows, m]
        num_windows = len(agg_vecs)
        
        # Compute basis indices (j1, j2) for each window
        j1_list = []
        j2_list = []
        for (k1, k2) in coords:
            j1 = k1 % self.L1
            j2 = k2 % self.L2
            j1_list.append(j1)
            j2_list.append(j2)
        
        j1_t = torch.tensor(j1_list, dtype=torch.long, device=self.device)
        j2_t = torch.tensor(j2_list, dtype=torch.long, device=self.device)
        idx = j1_t * self.L2 + j2_t
        
        B_rows = self.Bbasis[idx]
        scalar = torch.sum(B_rows * agg_tensor, dim=1)
        A_cols = self.Acoeff[:, idx]
        Nk = (A_cols.T) * scalar.unsqueeze(1)
        
        S_cum = torch.cumsum(Nk, dim=0)
        return S_cum.detach().cpu().numpy()

    def D(self, vec_arrs, t_list):
        """
        Compute mean squared deviation D across vector arrays:
        D = average over all windows of (N(k1,k2)-t)^2
        
        Args:
            vec_arrs: List of 2D vector arrays (each is a tensor of shape [H, W, m])
            t_list: List of target vectors corresponding to each array
            
        Returns:
            float: Average mean squared deviation across all windows and arrays
        """
        total_loss = 0.0
        total_windows = 0
        
        # Convert target vectors to tensor and move to device
        t_tensors = [torch.tensor(t, dtype=torch.float32, device=self.device) for t in t_list]
        
        for vec_arr, t in zip(vec_arrs, t_tensors):
            agg_vecs, coords = self.extract_vectors(vec_arr)
            if not agg_vecs:
                continue
                
            num_windows = len(agg_vecs)
            agg_tensor = torch.stack(agg_vecs, dim=0)  # [num_windows, m]
            
            # Compute basis indices for each window
            j1_list = []
            j2_list = []
            for (k1, k2) in coords:
                j1 = k1 % self.L1
                j2 = k2 % self.L2
                j1_list.append(j1)
                j2_list.append(j2)
            j1_t = torch.tensor(j1_list, dtype=torch.long, device=self.device)
            j2_t = torch.tensor(j2_list, dtype=torch.long, device=self.device)
            idx = j1_t * self.L2 + j2_t
            
            B_rows = self.Bbasis[idx]
            scalar = torch.sum(B_rows * agg_tensor, dim=1)
            A_cols = self.Acoeff[:, idx]
            Nk = (A_cols.T) * scalar.unsqueeze(1)
            
            losses = torch.sum((Nk - t) ** 2, dim=1)
            total_loss += losses.sum().item()
            total_windows += num_windows
            
            # Clean up
            del agg_tensor, B_rows, A_cols, Nk, losses
        
        return total_loss / total_windows if total_windows else 0.0

    def d(self, vec_arr, t):
        """Compute pattern deviation value (d) for a single vector array."""
        return self.D([vec_arr], [t])

    def reg_train(self, vec_arrs, t_list, max_iters=1000, tol=1e-8, learning_rate=0.01, 
                  continued=False, decay_rate=1.0, print_every=10, batch_size=32,
                  checkpoint_file=None, checkpoint_interval=10):
        """
        Train model using gradient descent with array-level batch processing.
        
        Args:
            vec_arrs: List of 2D vector arrays (each shape [H, W, m])
            t_list: List of target vectors
            max_iters: Maximum training iterations
            tol: Convergence tolerance
            learning_rate: Initial learning rate
            continued: Whether to continue from existing parameters
            decay_rate: Learning rate decay rate
            print_every: Print interval
            batch_size: Number of arrays to process in each batch
            checkpoint_file: Path to save/load checkpoint file for resuming training
            checkpoint_interval: Interval (in iterations) for saving checkpoints
            
        Returns:
            list: Training loss history
        """
        
        # Load checkpoint if continuing
        start_iter = 0
        best_loss = float('inf')
        best_model_state = None
        
        if continued and checkpoint_file and os.path.exists(checkpoint_file):
            checkpoint = torch.load(checkpoint_file, map_location=self.device, weights_only=False)
            self.load_state_dict(checkpoint['model_state_dict'])
            optimizer_state = checkpoint['optimizer_state_dict']
            scheduler_state = checkpoint['scheduler_state_dict']
            history = checkpoint['history']
            start_iter = checkpoint['iteration'] + 1
            best_loss = checkpoint.get('best_loss', float('inf'))
            print(f"Resumed training from checkpoint at iteration {start_iter}, best loss: {best_loss:.6e}")
        else:
            if not continued:
                self.reset_parameters()
            history = []
        
        # Convert target vectors to tensor
        t_tensors = [torch.tensor(t, dtype=torch.float32, device=self.device) for t in t_list]
        
        # Ensure all input arrays are on the correct device
        vec_arrs = [arr.to(self.device) if isinstance(arr, torch.Tensor) else 
                    torch.tensor(arr, dtype=torch.float32, device=self.device) 
                    for arr in vec_arrs]
        
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)
        
        if continued and checkpoint_file and os.path.exists(checkpoint_file):
            optimizer.load_state_dict(optimizer_state)
            scheduler.load_state_dict(scheduler_state)
        
        prev_loss = float('inf') if start_iter == 0 else history[-1] if history else float('inf')
        
        # Fast batch processing is possible when all arrays in a batch have identical dimensions
        # and rank_mode == 'drop' (padding would require extra handling, but could be extended)
        use_fast_batch = (self.rank_mode == 'drop')
        
        for it in range(start_iter, max_iters):
            total_loss = 0.0
            total_arrays = 0
            
            # Shuffle arrays for each epoch
            indices = list(range(len(vec_arrs)))
            random.shuffle(indices)
            
            for batch_start in range(0, len(indices), batch_size):
                batch_indices = indices[batch_start:batch_start + batch_size]
                batch_arrs = [vec_arrs[idx] for idx in batch_indices]
                batch_targets = [t_tensors[idx] for idx in batch_indices]
                
                optimizer.zero_grad()
                
                # Check if all arrays in this batch have the same dimensions
                shapes = [arr.shape for arr in batch_arrs]
                all_equal = len(set(shapes)) == 1
                
                if use_fast_batch and all_equal:
                    # Fast batch processing
                    H, W, _ = batch_arrs[0].shape
                    batch_tensor = torch.stack(batch_arrs, dim=0)  # [batch, H, W, m]
                    reps = self.batch_represent(batch_tensor)  # [batch, m]
                    targets_tensor = torch.stack(batch_targets, dim=0)  # [batch, m]
                    loss = torch.mean(torch.sum((reps - targets_tensor) ** 2, dim=1))
                    loss.backward()
                    optimizer.step()
                    batch_loss = loss.item() * len(batch_arrs)
                    total_loss += batch_loss
                    total_arrays += len(batch_arrs)
                else:
                    # Process each array individually
                    batch_loss = 0.0
                    batch_array_count = 0
                    for vec_arr, target in zip(batch_arrs, batch_targets):
                        agg_vecs, coords = self.extract_vectors(vec_arr)
                        if not agg_vecs:
                            continue
                            
                        num_windows = len(agg_vecs)
                        agg_tensor = torch.stack(agg_vecs, dim=0)  # [num_windows, m]
                        
                        # Compute basis indices
                        j1_list = []
                        j2_list = []
                        for (k1, k2) in coords:
                            j1 = k1 % self.L1
                            j2 = k2 % self.L2
                            j1_list.append(j1)
                            j2_list.append(j2)
                        j1_t = torch.tensor(j1_list, dtype=torch.long, device=self.device)
                        j2_t = torch.tensor(j2_list, dtype=torch.long, device=self.device)
                        idx = j1_t * self.L2 + j2_t
                        
                        B_rows = self.Bbasis[idx]
                        scalar = torch.sum(B_rows * agg_tensor, dim=1)
                        A_cols = self.Acoeff[:, idx]
                        Nk = (A_cols.T) * scalar.unsqueeze(1)
                        
                        # Array-level prediction: average of all N(k1,k2)
                        arr_pred = torch.mean(Nk, dim=0)
                        arr_loss = torch.sum((arr_pred - target) ** 2)
                        batch_loss += arr_loss
                        batch_array_count += 1
                        
                        # Clean up
                        del Nk, arr_pred, agg_tensor, B_rows, A_cols, scalar
                        if batch_array_count % 10 == 0 and torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    
                    if batch_array_count > 0:
                        batch_loss = batch_loss / batch_array_count
                        batch_loss.backward()
                        optimizer.step()
                        total_loss += batch_loss.item() * batch_array_count
                        total_arrays += batch_array_count
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            if total_arrays > 0:
                avg_loss = total_loss / total_arrays
            else:
                avg_loss = 0.0
            history.append(avg_loss)
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_model_state = copy.deepcopy(self.state_dict())
            
            if it % print_every == 0 or it == max_iters - 1:
                current_lr = scheduler.get_last_lr()[0]
                print(f"GD Iter {it:3d}: Loss = {avg_loss:.6e}, LR = {current_lr:.6f}")
            
            if checkpoint_file and (it % checkpoint_interval == 0 or it == max_iters - 1):
                self._save_checkpoint(
                    checkpoint_file, it, history, optimizer, scheduler, best_loss
                )
            
            if abs(prev_loss - avg_loss) < tol:
                print(f"Converged after {it+1} iterations.")
                if best_model_state is not None and avg_loss > prev_loss:
                    self.load_state_dict(best_model_state)
                    print(f"Restored best model state with loss = {best_loss:.6e}")
                    history[-1] = best_loss
                break
            prev_loss = avg_loss
            scheduler.step()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        if best_model_state is not None and best_loss < history[-1]:
            self.load_state_dict(best_model_state)
            print(f"Training ended. Restored best model state with loss = {best_loss:.6e}")
            history[-1] = best_loss
        
        self._compute_training_statistics(vec_arrs)
        self.trained = True
        return history

    def cls_train(self, vec_arrs, labels, num_classes, max_iters=1000, tol=1e-8, learning_rate=0.01,
                  continued=False, decay_rate=1.0, print_every=10, batch_size=32,
                  checkpoint_file=None, checkpoint_interval=10):
        """
        Train the model for multi-class classification using cross-entropy loss.
        
        Args:
            vec_arrs: List of 2D vector arrays
            labels: List of integer class labels (0 to num_classes-1)
            num_classes: Number of classes
            max_iters: Maximum number of training iterations
            tol: Convergence tolerance
            learning_rate: Initial learning rate
            continued: Whether to continue training from existing parameters
            decay_rate: Learning rate decay rate
            print_every: Print progress every N iterations
            batch_size: Number of arrays to process in each batch
            checkpoint_file: Path to save training checkpoints
            checkpoint_interval: Save checkpoint every N iterations
            
        Returns:
            list: Training loss history
        """
        
        if self.classifier is None or self.num_classes != num_classes:
            self.classifier = nn.Linear(self.m, num_classes).to(self.device)
            self.num_classes = num_classes
        
        if not continued:
            self.reset_parameters()
        
        label_tensors = torch.tensor(labels, dtype=torch.long, device=self.device)
        vec_arrs = [arr.to(self.device) if isinstance(arr, torch.Tensor) else 
                    torch.tensor(arr, dtype=torch.float32, device=self.device) 
                    for arr in vec_arrs]
        
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
            correct_predictions = 0
            
            indices = list(range(len(vec_arrs)))
            random.shuffle(indices)
            
            for batch_start in range(0, len(indices), batch_size):
                batch_indices = indices[batch_start:batch_start + batch_size]
                batch_arrs = [vec_arrs[idx] for idx in batch_indices]
                batch_labels = label_tensors[batch_indices]
                
                optimizer.zero_grad()
                
                shapes = [arr.shape for arr in batch_arrs]
                all_equal = len(set(shapes)) == 1
                
                if use_fast_batch and all_equal:
                    # Fast batch processing
                    H, W, _ = batch_arrs[0].shape
                    batch_tensor = torch.stack(batch_arrs, dim=0)  # [batch, H, W, m]
                    reps = self.batch_represent(batch_tensor)  # [batch, m]
                    logits = self.classifier(reps)  # [batch, num_classes]
                    loss = criterion(logits, batch_labels)
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item() * len(batch_arrs)
                    total_arrays += len(batch_arrs)
                    with torch.no_grad():
                        predictions = torch.argmax(logits, dim=1)
                        correct_predictions += (predictions == batch_labels).sum().item()
                else:
                    # Process each array individually
                    batch_logits = []
                    for vec_arr in batch_arrs:
                        agg_vecs, coords = self.extract_vectors(vec_arr)
                        if not agg_vecs:
                            seq_vector = torch.zeros(self.m, device=self.device)
                        else:
                            agg_tensor = torch.stack(agg_vecs, dim=0)  # [num_windows, m]
                            num_windows = len(agg_vecs)
                            
                            j1_list = []
                            j2_list = []
                            for (k1, k2) in coords:
                                j1 = k1 % self.L1
                                j2 = k2 % self.L2
                                j1_list.append(j1)
                                j2_list.append(j2)
                            j1_t = torch.tensor(j1_list, dtype=torch.long, device=self.device)
                            j2_t = torch.tensor(j2_list, dtype=torch.long, device=self.device)
                            idx = j1_t * self.L2 + j2_t
                            
                            B_rows = self.Bbasis[idx]
                            scalar = torch.sum(B_rows * agg_tensor, dim=1)
                            A_cols = self.Acoeff[:, idx]
                            Nk = (A_cols.T) * scalar.unsqueeze(1)
                            
                            seq_vector = torch.mean(Nk, dim=0)
                            del Nk, agg_tensor, B_rows, A_cols, scalar
                        
                        logits = self.classifier(seq_vector.unsqueeze(0))
                        batch_logits.append(logits)
                    
                    if batch_logits:
                        all_logits = torch.cat(batch_logits, dim=0)
                        loss = criterion(all_logits, batch_labels)
                        loss.backward()
                        optimizer.step()
                        
                        total_loss += loss.item() * len(batch_arrs)
                        total_arrays += len(batch_arrs)
                        with torch.no_grad():
                            predictions = torch.argmax(all_logits, dim=1)
                            correct_predictions += (predictions == batch_labels).sum().item()
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            if total_arrays > 0:
                avg_loss = total_loss / total_arrays
                accuracy = correct_predictions / total_arrays
            else:
                avg_loss = 0.0
                accuracy = 0.0
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

    def lbl_train(self, vec_arrs, labels, num_labels, max_iters=1000, tol=1e-8, learning_rate=0.01, 
                 continued=False, decay_rate=1.0, print_every=10, batch_size=32,
                 checkpoint_file=None, checkpoint_interval=10, pos_weight=None):
        """
        Train the model for multi-label classification using binary cross-entropy loss.
        
        Args:
            vec_arrs: List of 2D vector arrays
            labels: List of binary label vectors
            num_labels: Number of labels
            max_iters: Maximum number of training iterations
            tol: Convergence tolerance
            learning_rate: Initial learning rate
            continued: Whether to continue training from existing parameters
            decay_rate: Learning rate decay rate
            print_every: Print progress every N iterations
            batch_size: Number of arrays to process in each batch
            checkpoint_file: Path to save training checkpoints
            checkpoint_interval: Save checkpoint every N iterations
            pos_weight: Weight for positive class (torch.Tensor of shape [num_labels])
            
        Returns:
            tuple: (loss_history, acc_history)
        """
        
        if self.labeller is None or self.num_labels != num_labels:
            self.labeller = nn.Linear(self.m, num_labels).to(self.device)
            self.num_labels = num_labels
        
        if not continued:
            self.reset_parameters()
        
        if isinstance(labels, list):
            labels_tensor = torch.tensor(labels, dtype=torch.float32, device=self.device)
        else:
            labels_tensor = torch.as_tensor(labels, dtype=torch.float32, device=self.device)
        
        vec_arrs = [arr.to(self.device) if isinstance(arr, torch.Tensor) else 
                    torch.tensor(arr, dtype=torch.float32, device=self.device) 
                    for arr in vec_arrs]
        
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
            
            indices = list(range(len(vec_arrs)))
            random.shuffle(indices)
            
            for batch_start in range(0, len(indices), batch_size):
                batch_indices = indices[batch_start:batch_start + batch_size]
                batch_arrs = [vec_arrs[idx] for idx in batch_indices]
                batch_labels = labels_tensor[batch_indices]
                
                optimizer.zero_grad()
                
                shapes = [arr.shape for arr in batch_arrs]
                all_equal = len(set(shapes)) == 1
                
                if use_fast_batch and all_equal:
                    # Fast batch processing
                    H, W, _ = batch_arrs[0].shape
                    batch_tensor = torch.stack(batch_arrs, dim=0)  # [batch, H, W, m]
                    reps = self.batch_represent(batch_tensor)  # [batch, m]
                    logits = self.labeller(reps)  # [batch, num_labels]
                    loss = criterion(logits, batch_labels)
                    loss.backward()
                    optimizer.step()
                    
                    batch_loss = loss.item() * len(batch_arrs)
                    total_loss += batch_loss
                    total_arrays += len(batch_arrs)
                    
                    with torch.no_grad():
                        probs = torch.sigmoid(logits)
                        predictions = (probs > 0.5).float()
                        batch_correct = (predictions == batch_labels).sum().item()
                        batch_predictions = batch_labels.numel()
                        total_correct += batch_correct
                        total_predictions += batch_predictions
                else:
                    # Process each array individually
                    batch_predictions_list = []
                    for vec_arr in batch_arrs:
                        agg_vecs, coords = self.extract_vectors(vec_arr)
                        if not agg_vecs:
                            continue
                        
                        agg_tensor = torch.stack(agg_vecs, dim=0)
                        num_windows = len(agg_vecs)
                        
                        j1_list = []
                        j2_list = []
                        for (k1, k2) in coords:
                            j1 = k1 % self.L1
                            j2 = k2 % self.L2
                            j1_list.append(j1)
                            j2_list.append(j2)
                        j1_t = torch.tensor(j1_list, dtype=torch.long, device=self.device)
                        j2_t = torch.tensor(j2_list, dtype=torch.long, device=self.device)
                        idx = j1_t * self.L2 + j2_t
                        
                        B_rows = self.Bbasis[idx]
                        scalar = torch.sum(B_rows * agg_tensor, dim=1)
                        A_cols = self.Acoeff[:, idx]
                        Nk = (A_cols.T) * scalar.unsqueeze(1)
                        
                        seq_rep = torch.mean(Nk, dim=0)
                        logits = self.labeller(seq_rep)
                        batch_predictions_list.append(logits)
                        
                        del Nk, seq_rep, agg_tensor, B_rows, A_cols, scalar
                    
                    if batch_predictions_list:
                        batch_logits = torch.stack(batch_predictions_list, dim=0)
                        loss = criterion(batch_logits, batch_labels)
                        loss.backward()
                        optimizer.step()
                        
                        with torch.no_grad():
                            probs = torch.sigmoid(batch_logits)
                            predictions = (probs > 0.5).float()
                            batch_correct = (predictions == batch_labels).sum().item()
                            batch_predictions = batch_labels.numel()
                        
                        total_loss += loss.item() * len(batch_arrs)
                        total_correct += batch_correct
                        total_predictions += batch_predictions
                        total_arrays += len(batch_arrs)
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            if total_arrays > 0:
                avg_loss = total_loss / total_arrays
                avg_acc = total_correct / total_predictions if total_predictions > 0 else 0.0
            else:
                avg_loss = 0.0
                avg_acc = 0.0
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

    def self_train(self, vec_arrs, max_iters=1000, tol=1e-8, learning_rate=0.01, 
                   continued=False, decay_rate=1.0, print_every=10, batch_size=1024, 
                   checkpoint_file=None, checkpoint_interval=10):
        """
        Train the model using self-supervised learning with memory-optimized batch processing.
        
        Args:
            vec_arrs: List of 2D vector arrays
            max_iters: Maximum training iterations
            tol: Convergence tolerance
            learning_rate: Initial learning rate
            continued: Whether to continue from existing parameters
            decay_rate: Learning rate decay rate
            print_every: Print interval
            batch_size: Batch size for training samples
            checkpoint_file: Path to save/load checkpoint file for resuming training
            checkpoint_interval: Interval (in iterations) for saving checkpoints
            
        Returns:
            list: Training loss history
        """
        
        start_iter = 0
        best_loss = float('inf')
        best_model_state = None
        
        if continued and checkpoint_file and os.path.exists(checkpoint_file):
            checkpoint = torch.load(checkpoint_file, map_location=self.device, weights_only=False)
            self.load_state_dict(checkpoint['model_state_dict'])
            optimizer_state = checkpoint['optimizer_state_dict']
            scheduler_state = checkpoint['scheduler_state_dict']
            history = checkpoint['history']
            start_iter = checkpoint['iteration'] + 1
            best_loss = checkpoint.get('best_loss', float('inf'))
            print(f"Resumed self-training from checkpoint at iteration {start_iter}, best loss: {best_loss:.6f}")
        else:
            if not continued:
                self.reset_parameters()
            history = []
        
        vec_arrs = [arr.to(self.device) if isinstance(arr, torch.Tensor) else 
                    torch.tensor(arr, dtype=torch.float32, device=self.device) 
                    for arr in vec_arrs]
        
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)
        
        if continued and checkpoint_file and os.path.exists(checkpoint_file):
            optimizer.load_state_dict(optimizer_state)
            scheduler.load_state_dict(scheduler_state)
        
        prev_loss = float('inf') if start_iter == 0 else history[-1] if history else float('inf')
        
        # Fast batch processing possible if all arrays have same dimensions and rank_mode == 'drop'
        all_shapes = [arr.shape for arr in vec_arrs]
        all_equal = len(set(all_shapes)) == 1
        use_fast_batch = (self.rank_mode == 'drop' and all_equal)
        
        if use_fast_batch:
            # Fast path: all arrays same size, use batch_compute_Nk_and_targets
            H, W, m = vec_arrs[0].shape
            num_seqs = len(vec_arrs)
            all_seqs = torch.stack(vec_arrs, dim=0)  # [num_seqs, H, W, m]
            seq_batch_size = batch_size  # number of arrays per batch
            
            for it in range(start_iter, max_iters):
                total_loss = 0.0
                total_windows = 0
                
                indices = list(range(num_seqs))
                random.shuffle(indices)
                
                for batch_start in range(0, num_seqs, seq_batch_size):
                    batch_indices = indices[batch_start:batch_start + seq_batch_size]
                    batch_seqs = all_seqs[batch_indices]  # [batch, H, W, m]
                    
                    optimizer.zero_grad()
                    Nk_batch, targets_batch = self.batch_compute_Nk_and_targets(batch_seqs)  # both [batch, n_windows, m]
                    loss = torch.mean(torch.sum((Nk_batch - targets_batch) ** 2, dim=-1))
                    loss.backward()
                    optimizer.step()
                    
                    batch_loss = loss.item() * batch_seqs.shape[0] * Nk_batch.shape[1]
                    total_loss += batch_loss
                    total_windows += batch_seqs.shape[0] * Nk_batch.shape[1]
                    
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                avg_loss = total_loss / total_windows if total_windows > 0 else 0.0
                history.append(avg_loss)
                
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    best_model_state = copy.deepcopy(self.state_dict())
                
                if it % print_every == 0 or it == max_iters - 1:
                    current_lr = scheduler.get_last_lr()[0]
                    print(f"SelfTrain Iter {it:3d}: loss = {avg_loss:.6f}, LR = {current_lr:.6f}, Windows = {total_windows}")
                
                if checkpoint_file and (it % checkpoint_interval == 0 or it == max_iters - 1):
                    self._save_checkpoint(
                        checkpoint_file, it, history, optimizer, scheduler, best_loss
                    )
                
                if abs(prev_loss - avg_loss) < tol:
                    print(f"Converged after {it+1} iterations")
                    if best_model_state is not None and avg_loss > prev_loss:
                        self.load_state_dict(best_model_state)
                        print(f"Restored best model state with loss = {best_loss:.6f}")
                        history[-1] = best_loss
                    break
                prev_loss = avg_loss
                scheduler.step()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        else:
            # Slow path: original per-array, per-window processing
            sample_batch_size = batch_size  # number of windows per gradient step
            for it in range(start_iter, max_iters):
                total_loss = 0.0
                total_samples = 0
                
                indices = list(range(len(vec_arrs)))
                random.shuffle(indices)
                
                for idx in indices:
                    vec_arr = vec_arrs[idx]
                    agg_vecs, coords = self.extract_vectors(vec_arr)
                    if not agg_vecs:
                        continue
                    
                    agg_tensor = torch.stack(agg_vecs, dim=0)  # [num_windows, m]
                    num_windows = len(agg_vecs)
                    
                    # Compute basis indices
                    j1_list = []
                    j2_list = []
                    for (k1, k2) in coords:
                        j1 = k1 % self.L1
                        j2 = k2 % self.L2
                        j1_list.append(j1)
                        j2_list.append(j2)
                    j1_t = torch.tensor(j1_list, dtype=torch.long, device=self.device)
                    j2_t = torch.tensor(j2_list, dtype=torch.long, device=self.device)
                    idx_t = j1_t * self.L2 + j2_t
                    
                    B_rows = self.Bbasis[idx_t]
                    scalar = torch.sum(B_rows * agg_tensor, dim=1)
                    A_cols = self.Acoeff[:, idx_t]
                    Nk = (A_cols.T) * scalar.unsqueeze(1)
                    
                    # Self-supervised loss: Nk should reconstruct the aggregated vector itself
                    loss = torch.mean(torch.sum((Nk - agg_tensor) ** 2, dim=1))
                    loss.backward()
                    optimizer.step()
                    
                    batch_loss = loss.item() * num_windows
                    total_loss += batch_loss
                    total_samples += num_windows
                    
                    # Clean up
                    del agg_tensor, Nk, B_rows, A_cols, scalar
                    if total_samples % 1000 == 0 and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                if total_samples > 0:
                    avg_loss = total_loss / total_samples
                else:
                    avg_loss = 0.0
                history.append(avg_loss)
                
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    best_model_state = copy.deepcopy(self.state_dict())
                
                if it % print_every == 0 or it == max_iters - 1:
                    current_lr = scheduler.get_last_lr()[0]
                    print(f"SelfTrain Iter {it:3d}: loss = {avg_loss:.6f}, LR = {current_lr:.6f}, Samples = {total_samples}")
                
                if checkpoint_file and (it % checkpoint_interval == 0 or it == max_iters - 1):
                    self._save_checkpoint(
                        checkpoint_file, it, history, optimizer, scheduler, best_loss
                    )
                
                if abs(prev_loss - avg_loss) < tol:
                    print(f"Converged after {it+1} iterations")
                    if best_model_state is not None and avg_loss > prev_loss:
                        self.load_state_dict(best_model_state)
                        print(f"Restored best model state with loss = {best_loss:.6f}")
                        history[-1] = best_loss
                    break
                prev_loss = avg_loss
                scheduler.step()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        if best_model_state is not None and best_loss < history[-1]:
            self.load_state_dict(best_model_state)
            print(f"Training ended. Restored best model state with loss = {best_loss:.6f}")
            history[-1] = best_loss
        
        self._compute_training_statistics(vec_arrs)
        self.trained = True
        return history

    def _save_checkpoint(self, checkpoint_file, iteration, history, optimizer, scheduler, best_loss):
        checkpoint = {
            'iteration': iteration,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'history': history,
            'best_loss': best_loss,
            'config': {
                'vec_dim': self.m,
                'bas_dim1': self.L1,
                'bas_dim2': self.L2,
                'rank1': self.rank1,
                'rank2': self.rank2,
                'rank_op': self.rank_op,
                'rank_mode': self.rank_mode,
                'mode': self.mode,
                'step1': self.step1,
                'step2': self.step2
            },
            'training_stats': {
                'mean_t': self.mean_t if hasattr(self, 'mean_t') else None,
                'mean_window_count': self.mean_window_count if hasattr(self, 'mean_window_count') else None
            }
        }
        torch.save(checkpoint, checkpoint_file)
        print(f"Checkpoint saved at iteration {iteration} to {checkpoint_file}")

    def _compute_training_statistics(self, vec_arrs, batch_size=10):
        """
        Compute and store statistics for reconstruction and generation with memory optimization.
        Calculates mean window count and mean target vector across all arrays.
        
        Args:
            vec_arrs: List of 2D vector arrays
            batch_size: Batch size for processing arrays to optimize memory usage
        """
        total_window_count = 0
        total_t = torch.zeros(self.m, device=self.device)
        
        with torch.no_grad():
            for i in range(0, len(vec_arrs), batch_size):
                batch_arrs = vec_arrs[i:i+batch_size]
                batch_window_count = 0
                batch_t_sum = torch.zeros(self.m, device=self.device)
                
                for vec_arr in batch_arrs:
                    agg_vecs, coords = self.extract_vectors(vec_arr)
                    batch_window_count += len(agg_vecs)
                    
                    if agg_vecs:
                        agg_tensor = torch.stack(agg_vecs, dim=0)
                        num_windows = len(agg_vecs)
                        
                        j1_list = []
                        j2_list = []
                        for (k1, k2) in coords:
                            j1 = k1 % self.L1
                            j2 = k2 % self.L2
                            j1_list.append(j1)
                            j2_list.append(j2)
                        j1_t = torch.tensor(j1_list, dtype=torch.long, device=self.device)
                        j2_t = torch.tensor(j2_list, dtype=torch.long, device=self.device)
                        idx = j1_t * self.L2 + j2_t
                        
                        B_rows = self.Bbasis[idx]
                        scalar = torch.sum(B_rows * agg_tensor, dim=1)
                        A_cols = self.Acoeff[:, idx]
                        Nk = (A_cols.T) * scalar.unsqueeze(1)
                        
                        batch_t_sum += Nk.sum(dim=0)
                        del agg_tensor, Nk, B_rows, A_cols, scalar
                
                total_window_count += batch_window_count
                total_t += batch_t_sum
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        self.mean_window_count = total_window_count / len(vec_arrs) if vec_arrs else 0
        self.mean_t = (total_t / total_window_count).cpu().numpy() if total_window_count > 0 else np.zeros(self.m)

    def predict_t(self, vec_arr):
        """Predict target vector as average of N(k1,k2) vectors"""
        agg_vecs, coords = self.extract_vectors(vec_arr)
        if not agg_vecs:
            return np.zeros(self.m)
        
        agg_tensor = torch.stack(agg_vecs, dim=0)  # [num_windows, m]
        num_windows = len(agg_vecs)
        
        j1_list = []
        j2_list = []
        for (k1, k2) in coords:
            j1 = k1 % self.L1
            j2 = k2 % self.L2
            j1_list.append(j1)
            j2_list.append(j2)
        j1_t = torch.tensor(j1_list, dtype=torch.long, device=self.device)
        j2_t = torch.tensor(j2_list, dtype=torch.long, device=self.device)
        idx = j1_t * self.L2 + j2_t
        
        B_rows = self.Bbasis[idx]
        scalar = torch.sum(B_rows * agg_tensor, dim=1)
        A_cols = self.Acoeff[:, idx]
        Nk = (A_cols.T) * scalar.unsqueeze(1)
        
        return torch.mean(Nk, dim=0).detach().cpu().numpy()

    def predict_c(self, vec_arr):
        """
        Predict class label for a 2D vector array using the classification head.
        
        Args:
            vec_arr: Input 2D vector array
            
        Returns:
            tuple: (predicted_class, class_probabilities)
        """
        if self.classifier is None:
            raise ValueError("Model must be trained first for classification")
        
        seq_vector = self.predict_t(vec_arr)
        seq_vector_tensor = torch.tensor(seq_vector, dtype=torch.float32, device=self.device)
        
        with torch.no_grad():
            logits = self.classifier(seq_vector_tensor.unsqueeze(0))
            probabilities = torch.softmax(logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            
        return predicted_class, probabilities[0].cpu().numpy()

    def predict_l(self, vec_arr, threshold=0.5):
        """
        Predict multi-label classification for a 2D vector array.
        
        Args:
            vec_arr: Input 2D vector array
            threshold: Probability threshold for binary classification (default: 0.5)
            
        Returns:
            tuple: (binary_predictions, probability_scores)
        """
        assert self.labeller is not None, "Model must be trained first for label prediction"
        
        agg_vecs, coords = self.extract_vectors(vec_arr)
        if not agg_vecs:
            return np.zeros(self.num_labels, dtype=np.float32), np.zeros(self.num_labels, dtype=np.float32)
        
        agg_tensor = torch.stack(agg_vecs, dim=0)
        num_windows = len(agg_vecs)
        
        j1_list = []
        j2_list = []
        for (k1, k2) in coords:
            j1 = k1 % self.L1
            j2 = k2 % self.L2
            j1_list.append(j1)
            j2_list.append(j2)
        j1_t = torch.tensor(j1_list, dtype=torch.long, device=self.device)
        j2_t = torch.tensor(j2_list, dtype=torch.long, device=self.device)
        idx = j1_t * self.L2 + j2_t
        
        B_rows = self.Bbasis[idx]
        scalar = torch.sum(B_rows * agg_tensor, dim=1)
        A_cols = self.Acoeff[:, idx]
        Nk = (A_cols.T) * scalar.unsqueeze(1)
        
        seq_rep = torch.mean(Nk, dim=0)
        
        with torch.no_grad():
            logits = self.labeller(seq_rep)
            probs = torch.sigmoid(logits).cpu().numpy()
        
        binary_preds = (probs > threshold).astype(np.float32)
        return binary_preds, probs

    def reconstruct(self, H, W, tau=0.0):
        """
        Reconstruct a representative 2D vector array of size H x W with temperature-controlled randomness.
        Uses non-overlapping windows (step = rank1/rank2) to fill the grid.
        
        Args:
            H (int): Desired height of output array
            W (int): Desired width of output array
            tau (float): Temperature for stochastic selection (0 = deterministic)
            
        Returns:
            torch.Tensor: Reconstructed array of shape [H, W, m]
        """
        assert self.trained, "Model must be trained first"
        if tau < 0:
            raise ValueError("Temperature must be non-negative")
        
        # Number of windows needed to cover the grid (using step sizes equal to rank)
        h_win = (H + self.rank1 - 1) // self.rank1
        w_win = (W + self.rank2 - 1) // self.rank2
        
        mean_t_tensor = torch.tensor(self.mean_t, dtype=torch.float32, device=self.device)
        candidate_vecs = torch.randn(100, self.m, device=self.device)  # 100 random candidates
        
        generated_vectors = []  # list of tensors for each window
        
        for k1 in range(h_win):
            for k2 in range(w_win):
                j1 = k1 % self.L1
                j2 = k2 % self.L2
                idx = j1 * self.L2 + j2
                B_row = self.Bbasis[idx].unsqueeze(0)  # [1, m]
                
                # Transform candidates through M
                transformed_candidates = torch.matmul(candidate_vecs, self.M.T)  # [100, m]
                
                # Simulate window aggregation: we need to produce an aggregated vector that
                # would come from a window of size (rank1, rank2) filled with these candidates.
                # Since we are generating each window independently, we can simply take the
                # candidate vector as the aggregated vector after applying the appropriate
                # rank_op. For simplicity, we treat the candidate as the aggregated vector.
                # Compute scalar = B[j] • candidate
                scalar = torch.sum(B_row * transformed_candidates, dim=1)  # [100]
                A_col = self.Acoeff[:, idx]  # [m]
                Nk_all = A_col * scalar.unsqueeze(1)  # [100, m]
                
                # Compute scores (negative MSE to mean_t)
                errors = torch.sum((Nk_all - mean_t_tensor) ** 2, dim=1)
                scores = -errors
                
                if tau == 0:
                    best_idx = torch.argmax(scores).item()
                    best_vec = candidate_vecs[best_idx]
                    generated_vectors.append(best_vec)
                else:
                    probs = torch.softmax(scores / tau, dim=0).detach().cpu().numpy()
                    chosen_idx = np.random.choice(len(candidate_vecs), p=probs)
                    chosen_vec = candidate_vecs[chosen_idx]
                    generated_vectors.append(chosen_vec)
        
        # Build the full array: each window is a block of size (rank1, rank2) filled with the same vector
        full_arr = torch.zeros(H, W, self.m, device=self.device)
        win_idx = 0
        for k1 in range(h_win):
            for k2 in range(w_win):
                vec = generated_vectors[win_idx]
                h_start = k1 * self.rank1
                h_end = min(h_start + self.rank1, H)
                w_start = k2 * self.rank2
                w_end = min(w_start + self.rank2, W)
                full_arr[h_start:h_end, w_start:w_end, :] = vec
                win_idx += 1
        
        return full_arr

    def save(self, filename):
        """Save model state to file"""
        save_dict = {
            'state_dict': self.state_dict(),
            'mean_t': self.mean_t if hasattr(self, 'mean_t') else None,
            'mean_window_count': self.mean_window_count if hasattr(self, 'mean_window_count') else None,
            'trained': self.trained,
            'num_classes': self.num_classes,
            'num_labels': self.num_labels,
            'config': {
                'vec_dim': self.m,
                'bas_dim1': self.L1,
                'bas_dim2': self.L2,
                'rank1': self.rank1,
                'rank2': self.rank2,
                'rank_op': self.rank_op,
                'rank_mode': self.rank_mode,
                'mode': self.mode,
                'step1': self.step1,
                'step2': self.step2
            }
        }
        torch.save(save_dict, filename)
        print(f"Model saved to {filename}")

    def load(self, filename):
        """Load model state from file"""
        try:
            save_dict = torch.load(filename, map_location=self.device, weights_only=False)
        except TypeError:
            save_dict = torch.load(filename, map_location=self.device)
        
        self.load_state_dict(save_dict['state_dict'])
        self.mean_t = save_dict.get('mean_t', None)
        self.mean_window_count = save_dict.get('mean_window_count', None)
        self.trained = save_dict.get('trained', False)
        self.num_classes = save_dict.get('num_classes', None)
        self.num_labels = save_dict.get('num_labels', None)
        
        if 'config' in save_dict:
            config = save_dict['config']
            # Update attributes from config
            self.rank_op = config.get('rank_op', 'avg')
            self.rank_mode = config.get('rank_mode', 'drop')
            self.mode = config.get('mode', 'linear')
            self.step1 = config.get('step1', self.rank1)
            self.step2 = config.get('step2', self.rank2)
            # Note: rank1, rank2, L1, L2 are determined from model structure and not overwritten
        
        # Recreate classifier and labeller if needed
        if self.num_classes is not None:
            self.classifier = nn.Linear(self.m, self.num_classes).to(self.device)
        if self.num_labels is not None:
            self.labeller = nn.Linear(self.m, self.num_labels).to(self.device)
            
        print(f"Model loaded from {filename}")
        return self


# === Example Usage ===
if __name__ == "__main__":

    from statistics import correlation
    
    print("="*60)
    print("Spatial Numeric Dual Descriptor AB2 - 2D Array Version (PyTorch GPU Accelerated)")
    print("="*60)
    
    # Set random seeds for reproducibility
    torch.manual_seed(11)
    random.seed(11)
    
    vec_dim = 3
    bas_dim1 = 5
    bas_dim2 = 5
    rank1 = 2
    rank2 = 2
    arr_num = 50
    
    # Generate 2D vector arrays and random targets
    vec_arrs, t_list = [], []
    for _ in range(arr_num):
        H = random.randint(20, 30)
        W = random.randint(20, 30)
        arr = torch.randn(H, W, vec_dim)  # Random 2D array of vectors
        vec_arrs.append(arr)
        t_list.append([random.uniform(-1,1) for _ in range(vec_dim)])
    
    # Create model
    dd = SpatialNumDualDescriptorAB2(
        vec_dim=vec_dim, 
        bas_dim1=bas_dim1, bas_dim2=bas_dim2,
        rank1=rank1, rank2=rank2,
        mode='linear',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    print(f"\nUsing device: {dd.device}")
    print(f"Vector dimension: {dd.m}")
    print(f"Basis dimensions: {dd.L1} x {dd.L2}")
    print(f"Window size: {dd.rank1} x {dd.rank2}")
    
    # === Gradient Descent Training ===
    print("\n" + "="*50)
    print("Testing Gradient Descent Training (Regression)")
    print("="*50)
    
    print("\nStarting gradient descent training...")
    reg_history = dd.reg_train(
        vec_arrs, 
        t_list,
        learning_rate=0.1,
        max_iters=50,
        tol=1e-66,
        print_every=10,
        decay_rate=0.99,
        batch_size=16
    )
    
    # Predict target for first array
    arr0 = vec_arrs[0]
    t_pred = dd.predict_t(arr0)
    print(f"\nPredicted t for first array: {[round(x.item(), 4) for x in t_pred]}")
    
    # Calculate prediction correlation
    pred_t_list = [dd.predict_t(arr) for arr in vec_arrs]
    
    corr_sum = 0.0
    for i in range(dd.m):
        actu_t = [t_vec[i] for t_vec in t_list]
        pred_t = [t_vec[i] for t_vec in pred_t_list]
        corr = correlation(actu_t, pred_t)
        print(f"Prediction correlation for dimension {i}: {corr:.4f}")
        corr_sum += corr
    corr_avg = corr_sum / dd.m
    print(f"Average correlation: {corr_avg:.4f}")   
    
    # Reconstruct representative arrays
    arr_det = dd.reconstruct(H=20, W=20, tau=0.0)
    arr_rand = dd.reconstruct(H=20, W=20, tau=0.5)
    print(f"\nDeterministic Reconstruction shape: {arr_det.shape}")
    print(f"Stochastic Reconstruction shape (tau=0.5): {arr_rand.shape}")  
    
    # === Classification Task ===
    print("\n" + "="*50)
    print("Classification Task")
    print("="*50)
    
    # Generate classification data
    num_classes = 3
    class_arrs = []
    class_labels = []
    
    for class_id in range(num_classes):
        for _ in range(30):  # 30 arrays per class
            H = random.randint(20, 30)
            W = random.randint(20, 30)
            if class_id == 0:
                # Class 0: pattern with positive mean
                arr = torch.randn(H, W, vec_dim) + 0.5
            elif class_id == 1:
                # Class 1: pattern with negative mean
                arr = torch.randn(H, W, vec_dim) - 0.5
            else:
                # Class 2: normal distribution
                arr = torch.randn(H, W, vec_dim)
            class_arrs.append(arr)
            class_labels.append(class_id)
    
    dd_cls = SpatialNumDualDescriptorAB2(
        vec_dim=vec_dim, bas_dim1=bas_dim1, bas_dim2=bas_dim2,
        rank1=rank1, rank2=rank2, mode='linear',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    print("\nStarting Classification Training")
    history = dd_cls.cls_train(class_arrs, class_labels, num_classes, 
                              max_iters=20, tol=1e-8, learning_rate=0.05,
                              decay_rate=0.99, batch_size=16, print_every=5)
    
    # Prediction results
    correct = 0
    for arr, true_label in zip(class_arrs, class_labels):
        pred_class, probs = dd_cls.predict_c(arr)
        if pred_class == true_label:
            correct += 1
    accuracy = correct / len(class_arrs)
    print(f"\nClassification accuracy: {accuracy:.4f} ({correct}/{len(class_arrs)})")
    
    # === Multi-Label Classification Task ===
    print("\n" + "="*50)
    print("Multi-Label Classification Task")
    print("="*50)
    
    num_labels = 4
    label_arrs = []
    labels = []
    for _ in range(60):
        H = random.randint(20, 30)
        W = random.randint(20, 30)
        arr = torch.randn(H, W, vec_dim)
        label_arrs.append(arr)
        label_vec = [random.random() > 0.7 for _ in range(num_labels)]
        labels.append([1.0 if x else 0.0 for x in label_vec])
    
    dd_lbl = SpatialNumDualDescriptorAB2(
        vec_dim=vec_dim, bas_dim1=bas_dim1, bas_dim2=bas_dim2,
        rank1=rank1, rank2=rank2, mode='linear',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    loss_history, acc_history = dd_lbl.lbl_train(
        label_arrs, labels, num_labels,
        max_iters=30, tol=1e-16, learning_rate=0.05, 
        decay_rate=0.99, print_every=10, batch_size=16
    )
    
    print(f"\nFinal multi-label training loss: {loss_history[-1]:.6f}")
    print(f"Final multi-label training accuracy: {acc_history[-1]:.4f}")
    
    # Example predictions
    test_arr = torch.randn(25, 25, vec_dim)
    binary_pred, probs_pred = dd_lbl.predict_l(test_arr, threshold=0.5)
    print("\nMulti-label prediction example:")
    print(f"Predicted binary labels: {binary_pred}")
    print(f"Predicted probabilities: {[f'{p:.4f}' for p in probs_pred]}")
    
    # === Self-Training Example ===
    print("\n" + "="*50)
    print("Self-Supervised Learning (Self-Training)")
    print("="*50)
    
    self_arrs = []
    for _ in range(20):
        H = random.randint(20, 30)
        W = random.randint(20, 30)
        arr = torch.randn(H, W, vec_dim)
        self_arrs.append(arr)
    
    dd_self = SpatialNumDualDescriptorAB2(
        vec_dim=vec_dim, bas_dim1=bas_dim1, bas_dim2=bas_dim2,
        rank1=rank1, rank2=rank2, mode='linear',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    self_history = dd_self.self_train(
        self_arrs,
        max_iters=50,
        learning_rate=0.001,
        decay_rate=0.995,
        print_every=20,
        batch_size=512
    )
    
    self_arr_recon = dd_self.reconstruct(H=20, W=20, tau=0.0)
    print(f"\nSelf-trained model reconstruction shape: {self_arr_recon.shape}")
    
    # === Model Persistence Test ===
    print("\n" + "="*50)
    print("Model Persistence Test")
    print("="*50)
    
    dd_self.save("self_trained_model_2d.pkl")
    dd_loaded = SpatialNumDualDescriptorAB2(
        vec_dim=vec_dim, bas_dim1=bas_dim1, bas_dim2=bas_dim2,
        rank1=rank1, rank2=rank2, mode='linear',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    ).load("self_trained_model_2d.pkl")
    
    print("Model loaded successfully. Reconstructing with loaded model:")
    recon = dd_loaded.reconstruct(H=15, W=15, tau=0.0)
    print(f"Reconstruction shape: {recon.shape}")
    
    print("\n=== All Tests Completed ===")
