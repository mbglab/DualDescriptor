# Copyright (C) 2005-2026, Bin-Guang Ma (mbg@mail.hzau.edu.cn); SPDX-License-Identifier: MIT
# The Spatial Numeric Dual Descriptor Vector class (AB matrix form) for 4D array implemented with PyTorch
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

class SpatialNumDualDescriptorAB4(nn.Module):
    """
    Spatial Numeric Dual Descriptor for 4D arrays of vectors with GPU acceleration using PyTorch:
      - learnable coefficient matrix Acoeff ∈ R^{m×(L1*L2*L3*L4)}
      - fixed basis matrix Bbasis ∈ R^{(L1*L2*L3*L4)×m},
        Bbasis[idx][i] = cos(2π*(j1+1)/(i+2)) * cos(2π*(j2+1)/(i+2)) *
                         cos(2π*(j3+1)/(i+2)) * cos(2π*(j4+1)/(i+2))
      - learnable mapping matrix M ∈ R^{m×m} for input vector transformation
      - Supports both linear and nonlinear tokenization of 4D vector arrays
      - Batch processing for GPU acceleration
      - Supports regression, classification, and multi-label classification tasks
    """
    def __init__(self, vec_dim=4, bas_dim1=3, bas_dim2=3, bas_dim3=3, bas_dim4=3,
                 rank1=2, rank2=2, rank3=2, rank4=2, rank_op='avg', rank_mode='drop',
                 mode='linear', user_step1=None, user_step2=None, user_step3=None, user_step4=None,
                 device='cuda'):
        """
        Initialize the Spatial Dual Descriptor for 4D vector arrays.

        Args:
            vec_dim (int): Dimension m of input vectors
            bas_dim1 (int): Basis dimension in first direction (L1)
            bas_dim2 (int): Basis dimension in second direction (L2)
            bas_dim3 (int): Basis dimension in third direction (L3)
            bas_dim4 (int): Basis dimension in fourth direction (L4)
            rank1 (int): Window size in first direction
            rank2 (int): Window size in second direction
            rank3 (int): Window size in third direction
            rank4 (int): Window size in fourth direction
            rank_op (str): 'avg', 'sum', 'pick', or 'user_func'
            rank_mode (str): 'pad' or 'drop' for handling incomplete windows
            mode (str): 'linear' (sliding window by 1) or 'nonlinear' (stepped window)
            user_step1 (int): Custom step size in first direction for nonlinear mode
            user_step2 (int): Custom step size in second direction for nonlinear mode
            user_step3 (int): Custom step size in third direction for nonlinear mode
            user_step4 (int): Custom step size in fourth direction for nonlinear mode
            device (str): 'cuda' or 'cpu'
        """
        super().__init__()
        self.m = vec_dim
        self.L1 = bas_dim1
        self.L2 = bas_dim2
        self.L3 = bas_dim3
        self.L4 = bas_dim4
        self.L = self.L1 * self.L2 * self.L3 * self.L4   # total basis dimension
        self.rank1 = rank1
        self.rank2 = rank2
        self.rank3 = rank3
        self.rank4 = rank4
        self.rank_op = rank_op
        self.rank_mode = rank_mode
        assert mode in ('linear','nonlinear')
        self.mode = mode
        self.step1 = user_step1 if user_step1 is not None else rank1
        self.step2 = user_step2 if user_step2 is not None else rank2
        self.step3 = user_step3 if user_step3 is not None else rank3
        self.step4 = user_step4 if user_step4 is not None else rank4
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
            # Decompose flat index into j1,j2,j3,j4
            j1 = idx // (self.L2 * self.L3 * self.L4)
            rem1 = idx % (self.L2 * self.L3 * self.L4)
            j2 = rem1 // (self.L3 * self.L4)
            rem2 = rem1 % (self.L3 * self.L4)
            j3 = rem2 // self.L4
            j4 = rem2 % self.L4
            for i in range(self.m):
                Bbasis[idx, i] = (math.cos(2 * math.pi * (j1+1) / (i+2)) *
                                  math.cos(2 * math.pi * (j2+1) / (i+2)) *
                                  math.cos(2 * math.pi * (j3+1) / (i+2)) *
                                  math.cos(2 * math.pi * (j4+1) / (i+2)))
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

    # ----------------------------------------------------------------------
    #  Batch acceleration methods (inspired by SnDDab3.py)
    # ----------------------------------------------------------------------
    def _get_window_coords_and_indices_4d(self, D1, D2, D3, D4):
        """
        Compute window top-left coordinates and corresponding basis indices for a given 4D array size.
        Used by batch processing methods.

        Args:
            D1, D2, D3, D4: Dimensions of the array

        Returns:
            coords: List of (d1, d2, d3, d4) tuples for each window
            idx_tensor: Tensor of shape [n_windows] containing flattened basis indices
        """
        step1 = 1 if self.mode == 'linear' else self.step1
        step2 = 1 if self.mode == 'linear' else self.step2
        step3 = 1 if self.mode == 'linear' else self.step3
        step4 = 1 if self.mode == 'linear' else self.step4

        if self.rank_mode == 'drop':
            # Only windows fully inside
            range1 = range(0, D1 - self.rank1 + 1, step1)
            range2 = range(0, D2 - self.rank2 + 1, step2)
            range3 = range(0, D3 - self.rank3 + 1, step3)
            range4 = range(0, D4 - self.rank4 + 1, step4)
        else:  # 'pad'
            # Include all windows that start before D1,D2,D3,D4 (pad with zeros if needed)
            range1 = range(0, D1, step1)
            range2 = range(0, D2, step2)
            range3 = range(0, D3, step3)
            range4 = range(0, D4, step4)

        coords = []
        j1_list = []
        j2_list = []
        j3_list = []
        j4_list = []
        for d1 in range1:
            for d2 in range2:
                for d3 in range3:
                    for d4 in range4:
                        coords.append((d1, d2, d3, d4))
                        j1_list.append(d1 % self.L1)
                        j2_list.append(d2 % self.L2)
                        j3_list.append(d3 % self.L3)
                        j4_list.append(d4 % self.L4)

        if not coords:
            return [], torch.empty(0, dtype=torch.long, device=self.device)

        j1_t = torch.tensor(j1_list, dtype=torch.long, device=self.device)
        j2_t = torch.tensor(j2_list, dtype=torch.long, device=self.device)
        j3_t = torch.tensor(j3_list, dtype=torch.long, device=self.device)
        j4_t = torch.tensor(j4_list, dtype=torch.long, device=self.device)
        idx_tensor = (j1_t * (self.L2 * self.L3 * self.L4) +
                      j2_t * (self.L3 * self.L4) +
                      j3_t * self.L4 +
                      j4_t)
        return coords, idx_tensor

    def batch_represent(self, arr_batch):
        """
        Compute array representations for a batch of 4D vector arrays efficiently.
        Supports arbitrary window size and step (linear/nonlinear) as long as all arrays
        in the batch have identical dimensions.

        Args:
            arr_batch: Tensor of shape [batch_size, D1, D2, D3, D4, m]

        Returns:
            representations: Tensor of shape [batch_size, m]
        """
        B, D1, D2, D3, D4, m = arr_batch.shape
        step1 = 1 if self.mode == 'linear' else self.step1
        step2 = 1 if self.mode == 'linear' else self.step2
        step3 = 1 if self.mode == 'linear' else self.step3
        step4 = 1 if self.mode == 'linear' else self.step4

        # Convert to (B, m, D1, D2, D3, D4) for unfolding
        x = arr_batch.permute(0, 5, 1, 2, 3, 4)  # [B, m, D1, D2, D3, D4]

        # Pad if necessary (for 'pad' mode)
        if self.rank_mode == 'pad':
            # For each dimension, pad to make sure we can slide windows with given step
            # We pad to the right such that the last window starts at D - step + 1? Actually we pad to include all windows.
            # We pad to (D + (rank - step))? SnDDab3 pads by (rank-1) when step=1, else (rank-step).
            # Let's use same logic: pad_d1 = (self.rank1 - step1) if step1 != 1 else (self.rank1 - 1)
            pad_d1 = (self.rank1 - 1) if step1 == 1 else (self.rank1 - step1)
            pad_d2 = (self.rank2 - 1) if step2 == 1 else (self.rank2 - step2)
            pad_d3 = (self.rank3 - 1) if step3 == 1 else (self.rank3 - step3)
            pad_d4 = (self.rank4 - 1) if step4 == 1 else (self.rank4 - step4)
            # Padding order: (pad_last, pad_first) for each dimension. We pad only at the end.
            x = torch.nn.functional.pad(x, (0, pad_d4, 0, pad_d3, 0, pad_d2, 0, pad_d1, 0, 0), mode='constant', value=0)
            D1_pad = D1 + pad_d1
            D2_pad = D2 + pad_d2
            D3_pad = D3 + pad_d3
            D4_pad = D4 + pad_d4
        else:
            D1_pad, D2_pad, D3_pad, D4_pad = D1, D2, D3, D4

        # Compute number of windows along each dimension
        n_windows1 = ((D1_pad - self.rank1) // step1 + 1) if D1_pad >= self.rank1 else 0
        n_windows2 = ((D2_pad - self.rank2) // step2 + 1) if D2_pad >= self.rank2 else 0
        n_windows3 = ((D3_pad - self.rank3) // step3 + 1) if D3_pad >= self.rank3 else 0
        n_windows4 = ((D4_pad - self.rank4) // step4 + 1) if D4_pad >= self.rank4 else 0
        total_windows = n_windows1 * n_windows2 * n_windows3 * n_windows4
        if total_windows == 0:
            # No windows, return zero representations
            return torch.zeros(B, self.m, device=self.device)

        # Prepare to collect aggregated vectors for all windows
        agg_all = torch.zeros(B, total_windows, self.m, device=self.device)

        # Iterate over first two dimensions (d1, d2) and use 2D unfold for the last two
        win_size = self.rank1 * self.rank2 * self.rank3 * self.rank4
        idx_offset = 0
        for d1_start in range(0, D1_pad - self.rank1 + 1, step1):
            for d2_start in range(0, D2_pad - self.rank2 + 1, step2):
                # Extract slice: [B, m, rank1, rank2, D3_pad, D4_pad]
                x_slice = x[:, :, d1_start:d1_start+self.rank1, d2_start:d2_start+self.rank2, :, :]
                # Merge first two window dimensions into channel: [B, m*rank1*rank2, D3_pad, D4_pad]
                x_merged = x_slice.contiguous().view(B, self.m * self.rank1 * self.rank2, D3_pad, D4_pad)
                # Apply 2D unfold to get windows over D3 and D4
                windows_2d = torch.nn.functional.unfold(
                    x_merged, kernel_size=(self.rank3, self.rank4), stride=(step3, step4)
                )  # [B, m*rank1*rank2*rank3*rank4, n_windows3 * n_windows4]
                # Reshape to [B, n_spatial, win_size, m]
                windows_2d = windows_2d.view(B, self.m, win_size, -1).permute(0, 3, 2, 1)
                # windows_2d: [B, n_spatial, win_size, m]
                # Apply mapping matrix M
                mapped = torch.matmul(windows_2d, self.M.T)  # [B, n_spatial, win_size, m]
                # Aggregate over win_size dimension
                if self.rank_op == 'sum':
                    agg = mapped.sum(dim=2)
                elif self.rank_op == 'pick':
                    idx = torch.randint(0, win_size, (B, windows_2d.size(1), 1), device=self.device)
                    agg = torch.gather(mapped, 2, idx.expand(-1, -1, -1, self.m)).squeeze(2)
                elif self.rank_op == 'user_func':
                    avg = mapped.mean(dim=2)
                    agg = torch.sigmoid(avg)
                else:  # 'avg'
                    agg = mapped.mean(dim=2)  # [B, n_spatial, m]
                n_spatial = agg.size(1)
                # Store in agg_all
                agg_all[:, idx_offset:idx_offset+n_spatial, :] = agg
                idx_offset += n_spatial

        # Now compute basis indices for each window
        coords, idx_basis = self._get_window_coords_and_indices_4d(D1, D2, D3, D4)
        if not coords:
            return torch.zeros(B, self.m, device=self.device)
        idx_basis = idx_basis.to(self.device)  # [n_windows]

        # Get corresponding B rows and A columns
        B_rows = self.Bbasis[idx_basis]  # [n_windows, m]
        B_rows = B_rows.unsqueeze(0)  # [1, n_windows, m]
        A_cols = self.Acoeff[:, idx_basis].T  # [n_windows, m]
        A_cols = A_cols.unsqueeze(0)  # [1, n_windows, m]

        # Compute scalar = B_row • agg_all
        scalar = torch.sum(B_rows * agg_all, dim=2)  # [B, n_windows]
        # Compute Nk = A_cols * scalar
        Nk = A_cols * scalar.unsqueeze(2)  # [B, n_windows, m]
        # Average over windows to get representation
        rep = Nk.mean(dim=1)  # [B, m]
        return rep

    def batch_compute_Nk_and_targets(self, arr_batch):
        """
        Compute Nk vectors and target vectors for each position (window) in a batch of 4D arrays.
        Supports arbitrary window size and step.

        Args:
            arr_batch: Tensor of shape [batch_size, D1, D2, D3, D4, m]

        Returns:
            Nk_all: Tensor [batch_size, n_windows, m] - Nk vectors for each window
            targets: Tensor [batch_size, n_windows, m] - target vectors (aggregated mapped vectors)
        """
        B, D1, D2, D3, D4, m = arr_batch.shape
        step1 = 1 if self.mode == 'linear' else self.step1
        step2 = 1 if self.mode == 'linear' else self.step2
        step3 = 1 if self.mode == 'linear' else self.step3
        step4 = 1 if self.mode == 'linear' else self.step4

        x = arr_batch.permute(0, 5, 1, 2, 3, 4)  # [B, m, D1, D2, D3, D4]

        if self.rank_mode == 'pad':
            pad_d1 = (self.rank1 - 1) if step1 == 1 else (self.rank1 - step1)
            pad_d2 = (self.rank2 - 1) if step2 == 1 else (self.rank2 - step2)
            pad_d3 = (self.rank3 - 1) if step3 == 1 else (self.rank3 - step3)
            pad_d4 = (self.rank4 - 1) if step4 == 1 else (self.rank4 - step4)
            x = torch.nn.functional.pad(x, (0, pad_d4, 0, pad_d3, 0, pad_d2, 0, pad_d1, 0, 0), mode='constant', value=0)
            D1_pad = D1 + pad_d1
            D2_pad = D2 + pad_d2
            D3_pad = D3 + pad_d3
            D4_pad = D4 + pad_d4
        else:
            D1_pad, D2_pad, D3_pad, D4_pad = D1, D2, D3, D4

        n_windows1 = ((D1_pad - self.rank1) // step1 + 1) if D1_pad >= self.rank1 else 0
        n_windows2 = ((D2_pad - self.rank2) // step2 + 1) if D2_pad >= self.rank2 else 0
        n_windows3 = ((D3_pad - self.rank3) // step3 + 1) if D3_pad >= self.rank3 else 0
        n_windows4 = ((D4_pad - self.rank4) // step4 + 1) if D4_pad >= self.rank4 else 0
        total_windows = n_windows1 * n_windows2 * n_windows3 * n_windows4
        if total_windows == 0:
            return torch.empty(B, 0, self.m, device=self.device), torch.empty(B, 0, self.m, device=self.device)

        targets_all = torch.zeros(B, total_windows, self.m, device=self.device)
        win_size = self.rank1 * self.rank2 * self.rank3 * self.rank4
        idx_offset = 0
        for d1_start in range(0, D1_pad - self.rank1 + 1, step1):
            for d2_start in range(0, D2_pad - self.rank2 + 1, step2):
                x_slice = x[:, :, d1_start:d1_start+self.rank1, d2_start:d2_start+self.rank2, :, :]
                x_merged = x_slice.contiguous().view(B, self.m * self.rank1 * self.rank2, D3_pad, D4_pad)
                windows_2d = torch.nn.functional.unfold(
                    x_merged, kernel_size=(self.rank3, self.rank4), stride=(step3, step4)
                )  # [B, m*rank1*rank2*rank3*rank4, n_spatial]
                windows_2d = windows_2d.view(B, self.m, win_size, -1).permute(0, 3, 2, 1)
                # Apply mapping
                mapped = torch.matmul(windows_2d, self.M.T)  # [B, n_spatial, win_size, m]
                if self.rank_op == 'sum':
                    targets = mapped.sum(dim=2)
                elif self.rank_op == 'pick':
                    idx = torch.randint(0, win_size, (B, windows_2d.size(1), 1), device=self.device)
                    targets = torch.gather(mapped, 2, idx.expand(-1, -1, -1, self.m)).squeeze(2)
                elif self.rank_op == 'user_func':
                    avg = mapped.mean(dim=2)
                    targets = torch.sigmoid(avg)
                else:  # 'avg'
                    targets = mapped.mean(dim=2)  # [B, n_spatial, m]
                n_spatial = targets.size(1)
                targets_all[:, idx_offset:idx_offset+n_spatial, :] = targets
                idx_offset += n_spatial

        # Compute basis indices for all windows
        coords, idx_basis = self._get_window_coords_and_indices_4d(D1, D2, D3, D4)
        if not coords:
            return torch.empty(B, 0, self.m, device=self.device), targets_all
        idx_basis = idx_basis.to(self.device)
        B_rows = self.Bbasis[idx_basis].unsqueeze(0)  # [1, n_windows, m]
        A_cols = self.Acoeff[:, idx_basis].T.unsqueeze(0)  # [1, n_windows, m]
        scalar = torch.sum(B_rows * targets_all, dim=2)  # [B, n_windows]
        Nk_all = A_cols * scalar.unsqueeze(2)  # [B, n_windows, m]
        return Nk_all, targets_all

    # ----------------------------------------------------------------------
    #  Original methods remain unchanged (extract_vectors, describe, S, D, d, etc.)
    # ----------------------------------------------------------------------
    def extract_vectors(self, vec_arr):
        """
        Extract and aggregate vector groups from a 4D vector array based on vectorization mode.

        - 'linear': Slide window by 1 step in all directions, extracting contiguous windows of size (rank1, rank2, rank3, rank4)
        - 'nonlinear': Slide window by step1/step2/step3/step4 (or rank1/rank2/rank3/rank4 if steps not specified)

        Handles incomplete trailing fragments using:
        - 'pad': Pads with zero vectors to maintain window size
        - 'drop': Discards incomplete fragments

        The method applies mapping matrix M to each vector in the window and aggregates using rank_op.

        Args:
            vec_arr (list or tensor): Input 4D vector array of shape [D1, D2, D3, D4, m]

        Returns:
            tuple: (list of aggregated vectors (each shape [m]), list of corresponding window coordinates (k1,k2,k3,k4))
        """
        # Convert to tensor if needed
        if not isinstance(vec_arr, torch.Tensor):
            vec_arr = torch.tensor(vec_arr, dtype=torch.float32, device=self.device)
        elif vec_arr.device != self.device:
            vec_arr = vec_arr.to(self.device)

        D1, D2, D3, D4, _ = vec_arr.shape  # dimensions

        # Helper function to apply vector operations to a window
        def apply_op(window):
            """
            Apply mapping and aggregation to a window of vectors.
            window shape: [rank1, rank2, rank3, rank4, m]
            """
            # Flatten window to a list of vectors
            flat_window = window.reshape(-1, self.m)  # [rank1*rank2*rank3*rank4, m]
            # Apply mapping matrix M to each vector
            transformed = torch.matmul(flat_window, self.M.T)  # [rank1*rank2*rank3*rank4, m]

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
            step3 = 1
            step4 = 1
        else:  # nonlinear
            step1 = self.step1
            step2 = self.step2
            step3 = self.step3
            step4 = self.step4

        vectors = []
        coords = []

        # Slide over the array
        for d1 in range(0, D1, step1):
            for d2 in range(0, D2, step2):
                for d3 in range(0, D3, step3):
                    for d4 in range(0, D4, step4):
                        # Extract window fragment
                        frag = vec_arr[d1:d1+self.rank1,
                                       d2:d2+self.rank2,
                                       d3:d3+self.rank3,
                                       d4:d4+self.rank4, :]
                        frag_d1, frag_d2, frag_d3, frag_d4, _ = frag.shape

                        # Pad or drop based on rank_mode
                        if self.rank_mode == 'pad':
                            # Pad with zeros if the window is smaller than expected
                            if (frag_d1 < self.rank1 or frag_d2 < self.rank2 or
                                frag_d3 < self.rank3 or frag_d4 < self.rank4):
                                padded = torch.zeros((self.rank1, self.rank2, self.rank3, self.rank4, self.m),
                                                     device=self.device)
                                padded[:frag_d1, :frag_d2, :frag_d3, :frag_d4, :] = frag
                                frag = padded
                        elif self.rank_mode == 'drop':
                            # Only keep windows that are exactly the required size
                            if (frag_d1 != self.rank1 or frag_d2 != self.rank2 or
                                frag_d3 != self.rank3 or frag_d4 != self.rank4):
                                continue

                        vectors.append(apply_op(frag))
                        coords.append((d1, d2, d3, d4))

        return vectors, coords

    def describe(self, vec_arr):
        """Compute N(k1,k2,k3,k4) vectors for each window in the 4D vector array"""
        agg_vecs, coords = self.extract_vectors(vec_arr)
        if not agg_vecs:
            return []

        # Stack vectors for batch processing
        agg_tensor = torch.stack(agg_vecs, dim=0)  # [num_windows, m]
        num_windows = len(agg_vecs)

        # Compute basis indices (j1, j2, j3, j4) for each window
        j1_list = []
        j2_list = []
        j3_list = []
        j4_list = []
        for (k1, k2, k3, k4) in coords:
            j1_list.append(k1 % self.L1)
            j2_list.append(k2 % self.L2)
            j3_list.append(k3 % self.L3)
            j4_list.append(k4 % self.L4)

        # Flatten index: idx = j1*(L2*L3*L4) + j2*(L3*L4) + j3*L4 + j4
        idx = (torch.tensor(j1_list, dtype=torch.long, device=self.device) * (self.L2 * self.L3 * self.L4) +
               torch.tensor(j2_list, dtype=torch.long, device=self.device) * (self.L3 * self.L4) +
               torch.tensor(j3_list, dtype=torch.long, device=self.device) * self.L4 +
               torch.tensor(j4_list, dtype=torch.long, device=self.device))

        # Get corresponding B rows
        B_rows = self.Bbasis[idx]  # [num_windows, m]

        # Compute scalar = B[idx] • agg_vec
        scalar = torch.sum(B_rows * agg_tensor, dim=1)  # [num_windows]

        # Get A columns for each window
        A_cols = self.Acoeff[:, idx]  # [m, num_windows]
        # Nk = scalar * A[:, idx]  -> shape [num_windows, m]
        Nk = (A_cols.T) * scalar.unsqueeze(1)  # [num_windows, m]

        return Nk.detach().cpu().numpy()

    def S(self, vec_arr):
        """Compute cumulative sum of N(k1,k2,k3,k4) vectors (in row-major order)"""
        agg_vecs, coords = self.extract_vectors(vec_arr)
        if not agg_vecs:
            return []

        agg_tensor = torch.stack(agg_vecs, dim=0)  # [num_windows, m]
        num_windows = len(agg_vecs)

        j1_list, j2_list, j3_list, j4_list = [], [], [], []
        for (k1, k2, k3, k4) in coords:
            j1_list.append(k1 % self.L1)
            j2_list.append(k2 % self.L2)
            j3_list.append(k3 % self.L3)
            j4_list.append(k4 % self.L4)

        idx = (torch.tensor(j1_list, dtype=torch.long, device=self.device) * (self.L2 * self.L3 * self.L4) +
               torch.tensor(j2_list, dtype=torch.long, device=self.device) * (self.L3 * self.L4) +
               torch.tensor(j3_list, dtype=torch.long, device=self.device) * self.L4 +
               torch.tensor(j4_list, dtype=torch.long, device=self.device))

        B_rows = self.Bbasis[idx]
        scalar = torch.sum(B_rows * agg_tensor, dim=1)
        A_cols = self.Acoeff[:, idx]
        Nk = (A_cols.T) * scalar.unsqueeze(1)

        S_cum = torch.cumsum(Nk, dim=0)
        return S_cum.detach().cpu().numpy()

    def D(self, vec_arrs, t_list):
        """
        Compute mean squared deviation D across vector arrays:
        D = average over all windows of (N(k1,k2,k3,k4)-t)^2

        Args:
            vec_arrs: List of 4D vector arrays (each shape [D1, D2, D3, D4, m])
            t_list: List of target vectors corresponding to each array

        Returns:
            float: Average mean squared deviation across all windows and arrays
        """
        total_loss = 0.0
        total_windows = 0

        # Convert target vectors to tensor
        t_tensors = [torch.tensor(t, dtype=torch.float32, device=self.device) for t in t_list]

        for vec_arr, t in zip(vec_arrs, t_tensors):
            agg_vecs, coords = self.extract_vectors(vec_arr)
            if not agg_vecs:
                continue

            num_windows = len(agg_vecs)
            agg_tensor = torch.stack(agg_vecs, dim=0)  # [num_windows, m]

            # Compute basis indices
            j1_list, j2_list, j3_list, j4_list = [], [], [], []
            for (k1, k2, k3, k4) in coords:
                j1_list.append(k1 % self.L1)
                j2_list.append(k2 % self.L2)
                j3_list.append(k3 % self.L3)
                j4_list.append(k4 % self.L4)

            idx = (torch.tensor(j1_list, dtype=torch.long, device=self.device) * (self.L2 * self.L3 * self.L4) +
                   torch.tensor(j2_list, dtype=torch.long, device=self.device) * (self.L3 * self.L4) +
                   torch.tensor(j3_list, dtype=torch.long, device=self.device) * self.L4 +
                   torch.tensor(j4_list, dtype=torch.long, device=self.device))

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
                  continued=False, decay_rate=1.0, print_every=10, batch_size=16,
                  checkpoint_file=None, checkpoint_interval=10):
        """
        Train model using gradient descent with array-level batch processing.

        Args:
            vec_arrs: List of 4D vector arrays (each shape [D1, D2, D3, D4, m])
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

        # Fast batch processing possible when all arrays have identical dimensions and rank_mode == 'drop'
        all_shapes = [arr.shape for arr in vec_arrs]
        all_equal = len(set(all_shapes)) == 1
        use_fast_batch = (self.rank_mode == 'drop' and all_equal)

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

                if use_fast_batch:
                    # Fast batch processing using batch_represent
                    # Check if all arrays in this batch have the same dimensions
                    shapes = [arr.shape for arr in batch_arrs]
                    if len(set(shapes)) == 1:
                        D1, D2, D3, D4, _ = batch_arrs[0].shape
                        batch_tensor = torch.stack(batch_arrs, dim=0)  # [batch, D1, D2, D3, D4, m]
                        reps = self.batch_represent(batch_tensor)  # [batch, m]
                        targets_tensor = torch.stack(batch_targets, dim=0)  # [batch, m]
                        loss = torch.mean(torch.sum((reps - targets_tensor) ** 2, dim=1))
                        loss.backward()
                        optimizer.step()
                        batch_loss = loss.item() * len(batch_arrs)
                        total_loss += batch_loss
                        total_arrays += len(batch_arrs)
                    else:
                        # Fallback to individual processing
                        batch_loss = 0.0
                        batch_array_count = 0
                        for vec_arr, target in zip(batch_arrs, batch_targets):
                            agg_vecs, coords = self.extract_vectors(vec_arr)
                            if not agg_vecs:
                                continue
                            agg_tensor = torch.stack(agg_vecs, dim=0)
                            num_windows = len(agg_vecs)

                            j1_list, j2_list, j3_list, j4_list = [], [], [], []
                            for (k1, k2, k3, k4) in coords:
                                j1_list.append(k1 % self.L1)
                                j2_list.append(k2 % self.L2)
                                j3_list.append(k3 % self.L3)
                                j4_list.append(k4 % self.L4)

                            idx = (torch.tensor(j1_list, dtype=torch.long, device=self.device) * (self.L2 * self.L3 * self.L4) +
                                   torch.tensor(j2_list, dtype=torch.long, device=self.device) * (self.L3 * self.L4) +
                                   torch.tensor(j3_list, dtype=torch.long, device=self.device) * self.L4 +
                                   torch.tensor(j4_list, dtype=torch.long, device=self.device))

                            B_rows = self.Bbasis[idx]
                            scalar = torch.sum(B_rows * agg_tensor, dim=1)
                            A_cols = self.Acoeff[:, idx]
                            Nk = (A_cols.T) * scalar.unsqueeze(1)

                            arr_pred = torch.mean(Nk, dim=0)
                            arr_loss = torch.sum((arr_pred - target) ** 2)
                            batch_loss += arr_loss
                            batch_array_count += 1

                            del Nk, arr_pred, agg_tensor, B_rows, A_cols, scalar
                            if batch_array_count % 10 == 0 and torch.cuda.is_available():
                                torch.cuda.empty_cache()

                        if batch_array_count > 0:
                            batch_loss = batch_loss / batch_array_count
                            batch_loss.backward()
                            optimizer.step()
                            total_loss += batch_loss.item() * batch_array_count
                            total_arrays += batch_array_count
                else:
                    # Original per-array processing
                    batch_loss = 0.0
                    batch_array_count = 0
                    for vec_arr, target in zip(batch_arrs, batch_targets):
                        agg_vecs, coords = self.extract_vectors(vec_arr)
                        if not agg_vecs:
                            continue
                        agg_tensor = torch.stack(agg_vecs, dim=0)
                        num_windows = len(agg_vecs)

                        j1_list, j2_list, j3_list, j4_list = [], [], [], []
                        for (k1, k2, k3, k4) in coords:
                            j1_list.append(k1 % self.L1)
                            j2_list.append(k2 % self.L2)
                            j3_list.append(k3 % self.L3)
                            j4_list.append(k4 % self.L4)

                        idx = (torch.tensor(j1_list, dtype=torch.long, device=self.device) * (self.L2 * self.L3 * self.L4) +
                               torch.tensor(j2_list, dtype=torch.long, device=self.device) * (self.L3 * self.L4) +
                               torch.tensor(j3_list, dtype=torch.long, device=self.device) * self.L4 +
                               torch.tensor(j4_list, dtype=torch.long, device=self.device))

                        B_rows = self.Bbasis[idx]
                        scalar = torch.sum(B_rows * agg_tensor, dim=1)
                        A_cols = self.Acoeff[:, idx]
                        Nk = (A_cols.T) * scalar.unsqueeze(1)

                        arr_pred = torch.mean(Nk, dim=0)
                        arr_loss = torch.sum((arr_pred - target) ** 2)
                        batch_loss += arr_loss
                        batch_array_count += 1

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
                  continued=False, decay_rate=1.0, print_every=10, batch_size=16,
                  checkpoint_file=None, checkpoint_interval=10):
        """
        Train the model for multi-class classification using cross-entropy loss.

        Args:
            vec_arrs: List of 4D vector arrays
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

        # Fast batch processing possible when all arrays have identical dimensions and rank_mode == 'drop'
        all_shapes = [arr.shape for arr in vec_arrs]
        all_equal = len(set(all_shapes)) == 1
        use_fast_batch = (self.rank_mode == 'drop' and all_equal)

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

                if use_fast_batch:
                    # Fast batch processing using batch_represent
                    shapes = [arr.shape for arr in batch_arrs]
                    if len(set(shapes)) == 1:
                        D1, D2, D3, D4, _ = batch_arrs[0].shape
                        batch_tensor = torch.stack(batch_arrs, dim=0)  # [batch, D1, D2, D3, D4, m]
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
                        # Fallback to individual processing
                        batch_logits = []
                        for vec_arr in batch_arrs:
                            agg_vecs, coords = self.extract_vectors(vec_arr)
                            if not agg_vecs:
                                seq_vector = torch.zeros(self.m, device=self.device)
                            else:
                                agg_tensor = torch.stack(agg_vecs, dim=0)
                                j1_list, j2_list, j3_list, j4_list = [], [], [], []
                                for (k1, k2, k3, k4) in coords:
                                    j1_list.append(k1 % self.L1)
                                    j2_list.append(k2 % self.L2)
                                    j3_list.append(k3 % self.L3)
                                    j4_list.append(k4 % self.L4)

                                idx = (torch.tensor(j1_list, dtype=torch.long, device=self.device) * (self.L2 * self.L3 * self.L4) +
                                       torch.tensor(j2_list, dtype=torch.long, device=self.device) * (self.L3 * self.L4) +
                                       torch.tensor(j3_list, dtype=torch.long, device=self.device) * self.L4 +
                                       torch.tensor(j4_list, dtype=torch.long, device=self.device))

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
                else:
                    # Original per-array processing
                    batch_logits = []
                    for vec_arr in batch_arrs:
                        agg_vecs, coords = self.extract_vectors(vec_arr)
                        if not agg_vecs:
                            seq_vector = torch.zeros(self.m, device=self.device)
                        else:
                            agg_tensor = torch.stack(agg_vecs, dim=0)
                            j1_list, j2_list, j3_list, j4_list = [], [], [], []
                            for (k1, k2, k3, k4) in coords:
                                j1_list.append(k1 % self.L1)
                                j2_list.append(k2 % self.L2)
                                j3_list.append(k3 % self.L3)
                                j4_list.append(k4 % self.L4)

                            idx = (torch.tensor(j1_list, dtype=torch.long, device=self.device) * (self.L2 * self.L3 * self.L4) +
                                   torch.tensor(j2_list, dtype=torch.long, device=self.device) * (self.L3 * self.L4) +
                                   torch.tensor(j3_list, dtype=torch.long, device=self.device) * self.L4 +
                                   torch.tensor(j4_list, dtype=torch.long, device=self.device))

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
                 continued=False, decay_rate=1.0, print_every=10, batch_size=16,
                 checkpoint_file=None, checkpoint_interval=10, pos_weight=None):
        """
        Train the model for multi-label classification using binary cross-entropy loss.

        Args:
            vec_arrs: List of 4D vector arrays
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

        # Fast batch processing possible when all arrays have identical dimensions and rank_mode == 'drop'
        all_shapes = [arr.shape for arr in vec_arrs]
        all_equal = len(set(all_shapes)) == 1
        use_fast_batch = (self.rank_mode == 'drop' and all_equal)

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

                if use_fast_batch:
                    shapes = [arr.shape for arr in batch_arrs]
                    if len(set(shapes)) == 1:
                        D1, D2, D3, D4, _ = batch_arrs[0].shape
                        batch_tensor = torch.stack(batch_arrs, dim=0)  # [batch, D1, D2, D3, D4, m]
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
                        # Fallback to individual processing
                        batch_predictions_list = []
                        for vec_arr in batch_arrs:
                            agg_vecs, coords = self.extract_vectors(vec_arr)
                            if not agg_vecs:
                                continue
                            agg_tensor = torch.stack(agg_vecs, dim=0)
                            j1_list, j2_list, j3_list, j4_list = [], [], [], []
                            for (k1, k2, k3, k4) in coords:
                                j1_list.append(k1 % self.L1)
                                j2_list.append(k2 % self.L2)
                                j3_list.append(k3 % self.L3)
                                j4_list.append(k4 % self.L4)

                            idx = (torch.tensor(j1_list, dtype=torch.long, device=self.device) * (self.L2 * self.L3 * self.L4) +
                                   torch.tensor(j2_list, dtype=torch.long, device=self.device) * (self.L3 * self.L4) +
                                   torch.tensor(j3_list, dtype=torch.long, device=self.device) * self.L4 +
                                   torch.tensor(j4_list, dtype=torch.long, device=self.device))

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
                else:
                    # Original per-array processing
                    batch_predictions_list = []
                    for vec_arr in batch_arrs:
                        agg_vecs, coords = self.extract_vectors(vec_arr)
                        if not agg_vecs:
                            continue
                        agg_tensor = torch.stack(agg_vecs, dim=0)
                        j1_list, j2_list, j3_list, j4_list = [], [], [], []
                        for (k1, k2, k3, k4) in coords:
                            j1_list.append(k1 % self.L1)
                            j2_list.append(k2 % self.L2)
                            j3_list.append(k3 % self.L3)
                            j4_list.append(k4 % self.L4)

                        idx = (torch.tensor(j1_list, dtype=torch.long, device=self.device) * (self.L2 * self.L3 * self.L4) +
                               torch.tensor(j2_list, dtype=torch.long, device=self.device) * (self.L3 * self.L4) +
                               torch.tensor(j3_list, dtype=torch.long, device=self.device) * self.L4 +
                               torch.tensor(j4_list, dtype=torch.long, device=self.device))

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
                   continued=False, decay_rate=1.0, print_every=10, batch_size=256,
                   checkpoint_file=None, checkpoint_interval=10):
        """
        Train the model using self-supervised learning with memory-optimized batch processing.

        Args:
            vec_arrs: List of 4D vector arrays
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
            D1, D2, D3, D4, m = vec_arrs[0].shape
            num_seqs = len(vec_arrs)
            all_seqs = torch.stack(vec_arrs, dim=0)  # [num_seqs, D1, D2, D3, D4, m]
            seq_batch_size = batch_size  # number of arrays per batch

            for it in range(start_iter, max_iters):
                total_loss = 0.0
                total_windows = 0

                indices = list(range(num_seqs))
                random.shuffle(indices)

                for batch_start in range(0, num_seqs, seq_batch_size):
                    batch_indices = indices[batch_start:batch_start + seq_batch_size]
                    batch_seqs = all_seqs[batch_indices]  # [batch, D1, D2, D3, D4, m]

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
                    j1_list, j2_list, j3_list, j4_list = [], [], [], []
                    for (k1, k2, k3, k4) in coords:
                        j1_list.append(k1 % self.L1)
                        j2_list.append(k2 % self.L2)
                        j3_list.append(k3 % self.L3)
                        j4_list.append(k4 % self.L4)

                    idx_t = (torch.tensor(j1_list, dtype=torch.long, device=self.device) * (self.L2 * self.L3 * self.L4) +
                             torch.tensor(j2_list, dtype=torch.long, device=self.device) * (self.L3 * self.L4) +
                             torch.tensor(j3_list, dtype=torch.long, device=self.device) * self.L4 +
                             torch.tensor(j4_list, dtype=torch.long, device=self.device))

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
                'bas_dim3': self.L3,
                'bas_dim4': self.L4,
                'rank1': self.rank1,
                'rank2': self.rank2,
                'rank3': self.rank3,
                'rank4': self.rank4,
                'rank_op': self.rank_op,
                'rank_mode': self.rank_mode,
                'mode': self.mode,
                'step1': self.step1,
                'step2': self.step2,
                'step3': self.step3,
                'step4': self.step4
            },
            'training_stats': {
                'mean_t': self.mean_t if hasattr(self, 'mean_t') else None,
                'mean_window_count': self.mean_window_count if hasattr(self, 'mean_window_count') else None
            }
        }
        torch.save(checkpoint, checkpoint_file)
        print(f"Checkpoint saved at iteration {iteration} to {checkpoint_file}")

    def _compute_training_statistics(self, vec_arrs, batch_size=5):
        """
        Compute and store statistics for reconstruction and generation with memory optimization.
        Calculates mean window count and mean target vector across all arrays.

        Args:
            vec_arrs: List of 4D vector arrays
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

                        j1_list, j2_list, j3_list, j4_list = [], [], [], []
                        for (k1, k2, k3, k4) in coords:
                            j1_list.append(k1 % self.L1)
                            j2_list.append(k2 % self.L2)
                            j3_list.append(k3 % self.L3)
                            j4_list.append(k4 % self.L4)

                        idx = (torch.tensor(j1_list, dtype=torch.long, device=self.device) * (self.L2 * self.L3 * self.L4) +
                               torch.tensor(j2_list, dtype=torch.long, device=self.device) * (self.L3 * self.L4) +
                               torch.tensor(j3_list, dtype=torch.long, device=self.device) * self.L4 +
                               torch.tensor(j4_list, dtype=torch.long, device=self.device))

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
        """Predict target vector as average of N(k1,k2,k3,k4) vectors"""
        agg_vecs, coords = self.extract_vectors(vec_arr)
        if not agg_vecs:
            return np.zeros(self.m)

        agg_tensor = torch.stack(agg_vecs, dim=0)  # [num_windows, m]
        num_windows = len(agg_vecs)

        # Compute basis indices
        j1_list, j2_list, j3_list, j4_list = [], [], [], []
        for (k1, k2, k3, k4) in coords:
            j1_list.append(k1 % self.L1)
            j2_list.append(k2 % self.L2)
            j3_list.append(k3 % self.L3)
            j4_list.append(k4 % self.L4)

        idx = (torch.tensor(j1_list, dtype=torch.long, device=self.device) * (self.L2 * self.L3 * self.L4) +
               torch.tensor(j2_list, dtype=torch.long, device=self.device) * (self.L3 * self.L4) +
               torch.tensor(j3_list, dtype=torch.long, device=self.device) * self.L4 +
               torch.tensor(j4_list, dtype=torch.long, device=self.device))

        B_rows = self.Bbasis[idx]
        scalar = torch.sum(B_rows * agg_tensor, dim=1)
        A_cols = self.Acoeff[:, idx]
        Nk = (A_cols.T) * scalar.unsqueeze(1)

        return torch.mean(Nk, dim=0).detach().cpu().numpy()

    def predict_c(self, vec_arr):
        """
        Predict class label for a 4D vector array using the classification head.

        Args:
            vec_arr: Input 4D vector array

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
        Predict multi-label classification for a 4D vector array.

        Args:
            vec_arr: Input 4D vector array
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

        # Compute basis indices
        j1_list, j2_list, j3_list, j4_list = [], [], [], []
        for (k1, k2, k3, k4) in coords:
            j1_list.append(k1 % self.L1)
            j2_list.append(k2 % self.L2)
            j3_list.append(k3 % self.L3)
            j4_list.append(k4 % self.L4)

        idx = (torch.tensor(j1_list, dtype=torch.long, device=self.device) * (self.L2 * self.L3 * self.L4) +
               torch.tensor(j2_list, dtype=torch.long, device=self.device) * (self.L3 * self.L4) +
               torch.tensor(j3_list, dtype=torch.long, device=self.device) * self.L4 +
               torch.tensor(j4_list, dtype=torch.long, device=self.device))

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

    def reconstruct(self, D1, D2, D3, D4, tau=0.0):
        """
        Reconstruct a representative 4D vector array of size D1 x D2 x D3 x D4 with temperature-controlled randomness.
        Uses non-overlapping windows (step = rank1/rank2/rank3/rank4) to fill the grid.

        Args:
            D1 (int): Desired size in first dimension
            D2 (int): Desired size in second dimension
            D3 (int): Desired size in third dimension
            D4 (int): Desired size in fourth dimension
            tau (float): Temperature for stochastic selection (0 = deterministic)

        Returns:
            torch.Tensor: Reconstructed array of shape [D1, D2, D3, D4, m]
        """
        assert self.trained, "Model must be trained first"
        if tau < 0:
            raise ValueError("Temperature must be non-negative")

        # Number of windows needed to cover the grid (using step = rank)
        win1 = (D1 + self.rank1 - 1) // self.rank1
        win2 = (D2 + self.rank2 - 1) // self.rank2
        win3 = (D3 + self.rank3 - 1) // self.rank3
        win4 = (D4 + self.rank4 - 1) // self.rank4

        mean_t_tensor = torch.tensor(self.mean_t, dtype=torch.float32, device=self.device)
        candidate_vecs = torch.randn(100, self.m, device=self.device)  # 100 random candidates

        generated_vectors = []  # list of tensors for each window

        for w1 in range(win1):
            for w2 in range(win2):
                for w3 in range(win3):
                    for w4 in range(win4):
                        j1 = w1 % self.L1
                        j2 = w2 % self.L2
                        j3 = w3 % self.L3
                        j4 = w4 % self.L4
                        idx = j1 * (self.L2 * self.L3 * self.L4) + j2 * (self.L3 * self.L4) + j3 * self.L4 + j4
                        B_row = self.Bbasis[idx].unsqueeze(0)  # [1, m]

                        # Transform candidates through M
                        transformed_candidates = torch.matmul(candidate_vecs, self.M.T)  # [100, m]

                        # Compute scalar = B[j] • candidate (candidate is the aggregated vector)
                        scalar = torch.sum(B_row * transformed_candidates, dim=1)  # [100]
                        A_col = self.Acoeff[:, idx]  # [m]
                        Nk_all = A_col * scalar.unsqueeze(1)  # [100, m]

                        # Scores: negative MSE to mean_t
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

        # Build the full array: each window is a block filled with the same vector
        full_arr = torch.zeros(D1, D2, D3, D4, self.m, device=self.device)
        win_idx = 0
        for w1 in range(win1):
            for w2 in range(win2):
                for w3 in range(win3):
                    for w4 in range(win4):
                        vec = generated_vectors[win_idx]
                        d1_start = w1 * self.rank1
                        d1_end = min(d1_start + self.rank1, D1)
                        d2_start = w2 * self.rank2
                        d2_end = min(d2_start + self.rank2, D2)
                        d3_start = w3 * self.rank3
                        d3_end = min(d3_start + self.rank3, D3)
                        d4_start = w4 * self.rank4
                        d4_end = min(d4_start + self.rank4, D4)
                        full_arr[d1_start:d1_end, d2_start:d2_end, d3_start:d3_end, d4_start:d4_end, :] = vec
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
                'bas_dim3': self.L3,
                'bas_dim4': self.L4,
                'rank1': self.rank1,
                'rank2': self.rank2,
                'rank3': self.rank3,
                'rank4': self.rank4,
                'rank_op': self.rank_op,
                'rank_mode': self.rank_mode,
                'mode': self.mode,
                'step1': self.step1,
                'step2': self.step2,
                'step3': self.step3,
                'step4': self.step4
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
            self.rank_op = config.get('rank_op', 'avg')
            self.rank_mode = config.get('rank_mode', 'drop')
            self.mode = config.get('mode', 'linear')
            self.step1 = config.get('step1', self.rank1)
            self.step2 = config.get('step2', self.rank2)
            self.step3 = config.get('step3', self.rank3)
            self.step4 = config.get('step4', self.rank4)

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
    print("Spatial Numeric Dual Descriptor AB4 - 4D Array Version (PyTorch GPU Accelerated)")
    print("="*60)

    # Set random seeds for reproducibility
    torch.manual_seed(11)
    random.seed(11)

    vec_dim = 3
    bas_dim1 = 3
    bas_dim2 = 3
    bas_dim3 = 3
    bas_dim4 = 3
    rank1 = 2
    rank2 = 2
    rank3 = 2
    rank4 = 2
    arr_num = 20

    # Generate 4D vector arrays and random targets
    vec_arrs, t_list = [], []
    for _ in range(arr_num):
        D1 = random.randint(6, 8)
        D2 = random.randint(6, 8)
        D3 = random.randint(6, 8)
        D4 = random.randint(6, 8)
        arr = torch.randn(D1, D2, D3, D4, vec_dim)  # Random 4D array of vectors
        vec_arrs.append(arr)
        t_list.append([random.uniform(-1,1) for _ in range(vec_dim)])

    # Create model
    dd = SpatialNumDualDescriptorAB4(
        vec_dim=vec_dim,
        bas_dim1=bas_dim1, bas_dim2=bas_dim2, bas_dim3=bas_dim3, bas_dim4=bas_dim4,
        rank1=rank1, rank2=rank2, rank3=rank3, rank4=rank4,
        mode='linear',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    print(f"\nUsing device: {dd.device}")
    print(f"Vector dimension: {dd.m}")
    print(f"Basis dimensions: {dd.L1} x {dd.L2} x {dd.L3} x {dd.L4}")
    print(f"Window size: {dd.rank1} x {dd.rank2} x {dd.rank3} x {dd.rank4}")

    # === Gradient Descent Training ===
    print("\n" + "="*50)
    print("Testing Gradient Descent Training (Regression)")
    print("="*50)

    print("\nStarting gradient descent training...")
    reg_history = dd.reg_train(
        vec_arrs,
        t_list,
        learning_rate=0.1,
        max_iters=30,
        tol=1e-66,
        print_every=10,
        decay_rate=0.99,
        batch_size=8
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
    arr_det = dd.reconstruct(D1=8, D2=8, D3=8, D4=8, tau=0.0)
    arr_rand = dd.reconstruct(D1=8, D2=8, D3=8, D4=8, tau=0.5)
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
        for _ in range(15):  # 15 arrays per class
            D1 = random.randint(6, 8)
            D2 = random.randint(6, 8)
            D3 = random.randint(6, 8)
            D4 = random.randint(6, 8)
            if class_id == 0:
                # Class 0: pattern with positive mean
                arr = torch.randn(D1, D2, D3, D4, vec_dim) + 0.5
            elif class_id == 1:
                # Class 1: pattern with negative mean
                arr = torch.randn(D1, D2, D3, D4, vec_dim) - 0.5
            else:
                # Class 2: normal distribution
                arr = torch.randn(D1, D2, D3, D4, vec_dim)
            class_arrs.append(arr)
            class_labels.append(class_id)

    dd_cls = SpatialNumDualDescriptorAB4(
        vec_dim=vec_dim, bas_dim1=bas_dim1, bas_dim2=bas_dim2,
        bas_dim3=bas_dim3, bas_dim4=bas_dim4,
        rank1=rank1, rank2=rank2, rank3=rank3, rank4=rank4,
        mode='linear',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    print("\nStarting Classification Training")
    history = dd_cls.cls_train(class_arrs, class_labels, num_classes,
                              max_iters=15, tol=1e-8, learning_rate=0.05,
                              decay_rate=0.99, batch_size=8, print_every=5)

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
    for _ in range(25):
        D1 = random.randint(6, 8)
        D2 = random.randint(6, 8)
        D3 = random.randint(6, 8)
        D4 = random.randint(6, 8)
        arr = torch.randn(D1, D2, D3, D4, vec_dim)
        label_arrs.append(arr)
        label_vec = [random.random() > 0.7 for _ in range(num_labels)]
        labels.append([1.0 if x else 0.0 for x in label_vec])

    dd_lbl = SpatialNumDualDescriptorAB4(
        vec_dim=vec_dim, bas_dim1=bas_dim1, bas_dim2=bas_dim2,
        bas_dim3=bas_dim3, bas_dim4=bas_dim4,
        rank1=rank1, rank2=rank2, rank3=rank3, rank4=rank4,
        mode='linear',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    loss_history, acc_history = dd_lbl.lbl_train(
        label_arrs, labels, num_labels,
        max_iters=20, tol=1e-16, learning_rate=0.05,
        decay_rate=0.99, print_every=10, batch_size=8
    )

    print(f"\nFinal multi-label training loss: {loss_history[-1]:.6f}")
    print(f"Final multi-label training accuracy: {acc_history[-1]:.4f}")

    # Example predictions
    test_arr = torch.randn(6, 6, 6, 6, vec_dim)
    binary_pred, probs_pred = dd_lbl.predict_l(test_arr, threshold=0.5)
    print("\nMulti-label prediction example:")
    print(f"Predicted binary labels: {binary_pred}")
    print(f"Predicted probabilities: {[f'{p:.4f}' for p in probs_pred]}")

    # === Self-Training Example ===
    print("\n" + "="*50)
    print("Self-Supervised Learning (Self-Training)")
    print("="*50)

    self_arrs = []
    for _ in range(10):
        D1 = random.randint(6, 8)
        D2 = random.randint(6, 8)
        D3 = random.randint(6, 8)
        D4 = random.randint(6, 8)
        arr = torch.randn(D1, D2, D3, D4, vec_dim)
        self_arrs.append(arr)

    dd_self = SpatialNumDualDescriptorAB4(
        vec_dim=vec_dim, bas_dim1=bas_dim1, bas_dim2=bas_dim2,
        bas_dim3=bas_dim3, bas_dim4=bas_dim4,
        rank1=rank1, rank2=rank2, rank3=rank3, rank4=rank4,
        mode='linear',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    self_history = dd_self.self_train(
        self_arrs,
        max_iters=30,
        learning_rate=0.001,
        decay_rate=0.995,
        print_every=15,
        batch_size=128
    )

    self_arr_recon = dd_self.reconstruct(D1=6, D2=6, D3=6, D4=6, tau=0.0)
    print(f"\nSelf-trained model reconstruction shape: {self_arr_recon.shape}")

    # === Model Persistence Test ===
    print("\n" + "="*50)
    print("Model Persistence Test")
    print("="*50)

    dd_self.save("self_trained_model_4d.pkl")
    dd_loaded = SpatialNumDualDescriptorAB4(
        vec_dim=vec_dim, bas_dim1=bas_dim1, bas_dim2=bas_dim2,
        bas_dim3=bas_dim3, bas_dim4=bas_dim4,
        rank1=rank1, rank2=rank2, rank3=rank3, rank4=rank4,
        mode='linear',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    ).load("self_trained_model_4d.pkl")

    print("Model loaded successfully. Reconstructing with loaded model:")
    recon = dd_loaded.reconstruct(D1=5, D2=5, D3=5, D4=5, tau=0.0)
    print(f"Reconstruction shape: {recon.shape}")

    print("\n=== All Tests Completed ===")
