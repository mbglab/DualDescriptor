# Copyright (C) 2005-2026, Bin-Guang Ma (mbg@mail.hzau.edu.cn); SPDX-License-Identifier: MIT
# The Spatial Numeric Dual Descriptor Vector class (AB matrix form) for 3D array implemented with PyTorch
# This program is for the demonstration of methodology and not fully refined.
# Author: Bin-Guang Ma (assisted by DeepSeek); Date: 2025-8-28 ~ 2026-04-02

import math
import itertools
import random
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import copy
import os

class SpatialNumDualDescriptorAB3(nn.Module):
    """
    Spatial Numeric Dual Descriptor for 3D arrays of vectors with GPU acceleration using PyTorch:
      - learnable coefficient matrix Acoeff ∈ R^{m×(L1*L2*L3)}
      - fixed basis matrix Bbasis ∈ R^{(L1*L2*L3)×m},
        Bbasis[idx][i] = cos(2π*(j1+1)/(i+2)) * cos(2π*(j2+1)/(i+2)) * cos(2π*(j3+1)/(i+2))
      - learnable mapping matrix M ∈ R^{m×m} for input vector transformation
      - Supports both linear and nonlinear tokenization of 3D vector arrays
      - Batch processing for GPU acceleration
      - Supports regression, classification, and multi-label classification tasks
    """
    def __init__(self, vec_dim=4, bas_dim1=5, bas_dim2=5, bas_dim3=5,
                 rank1=2, rank2=2, rank3=2, rank_op='avg', rank_mode='drop',
                 mode='linear', user_step1=None, user_step2=None, user_step3=None, device='cuda'):
        """
        Initialize the Spatial Dual Descriptor for 3D vector arrays.

        Args:
            vec_dim (int): Dimension m of input vectors
            bas_dim1 (int): Basis dimension in first direction (L1)
            bas_dim2 (int): Basis dimension in second direction (L2)
            bas_dim3 (int): Basis dimension in third direction (L3)
            rank1 (int): Window size in first direction
            rank2 (int): Window size in second direction
            rank3 (int): Window size in third direction
            rank_op (str): 'avg', 'sum', 'pick', or 'user_func'
            rank_mode (str): 'pad' or 'drop' for handling incomplete windows
            mode (str): 'linear' (sliding window by 1) or 'nonlinear' (stepped window)
            user_step1 (int): Custom step size in first direction for nonlinear mode
            user_step2 (int): Custom step size in second direction for nonlinear mode
            user_step3 (int): Custom step size in third direction for nonlinear mode
            device (str): 'cuda' or 'cpu'
        """
        super().__init__()
        self.m = vec_dim
        self.L1 = bas_dim1
        self.L2 = bas_dim2
        self.L3 = bas_dim3
        self.L = self.L1 * self.L2 * self.L3   # total basis dimension
        self.rank1 = rank1
        self.rank2 = rank2
        self.rank3 = rank3
        self.rank_op = rank_op
        self.rank_mode = rank_mode
        assert mode in ('linear','nonlinear')
        self.mode = mode
        self.step1 = user_step1 if user_step1 is not None else rank1
        self.step2 = user_step2 if user_step2 is not None else rank2
        self.step3 = user_step3 if user_step3 is not None else rank3
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
            j1 = idx // (self.L2 * self.L3)
            rem = idx % (self.L2 * self.L3)
            j2 = rem // self.L3
            j3 = rem % self.L3
            for i in range(self.m):
                Bbasis[idx, i] = (math.cos(2 * math.pi * (j1+1) / (i+2)) *
                                  math.cos(2 * math.pi * (j2+1) / (i+2)) *
                                  math.cos(2 * math.pi * (j3+1) / (i+2)))
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
        Extract and aggregate vector groups from a 3D vector array based on vectorization mode.

        - 'linear': Slide window by 1 step in all directions, extracting contiguous windows of size (rank1, rank2, rank3)
        - 'nonlinear': Slide window by step1/step2/step3 (or rank1/rank2/rank3 if steps not specified)

        Handles incomplete trailing fragments using:
        - 'pad': Pads with zero vectors to maintain window size
        - 'drop': Discards incomplete fragments

        The method applies mapping matrix M to each vector in the window and aggregates using rank_op.

        Args:
            vec_arr (list or tensor): Input 3D vector array of shape [D, H, W, m]

        Returns:
            tuple: (list of aggregated vectors (each shape [m]), list of corresponding window coordinates (k1,k2,k3))
        """
        # Convert to tensor if needed
        if not isinstance(vec_arr, torch.Tensor):
            vec_arr = torch.tensor(vec_arr, dtype=torch.float32, device=self.device)
        elif vec_arr.device != self.device:
            vec_arr = vec_arr.to(self.device)

        D, H, W, _ = vec_arr.shape
        if D < self.rank1 or H < self.rank2 or W < self.rank3:
            return [], []

        # Use vectorized approach for linear mode with avg/sum
        if self.mode == 'linear' and self.rank_op in ('avg', 'sum'):
            # Prepare input for 3D convolution: (1, m, D, H, W)
            x = vec_arr.permute(3, 0, 1, 2).unsqueeze(0)  # (1, m, D, H, W)
            # Create convolution kernel: (m, 1, rank1, rank2, rank3) filled with ones
            weight = torch.ones(self.m, 1, self.rank1, self.rank2, self.rank3,
                                dtype=torch.float32, device=self.device)
            # Perform grouped convolution to compute sum for each channel independently
            out = F.conv3d(x, weight, groups=self.m, stride=1, padding=0)
            # out shape: (1, m, out_D, out_H, out_W)
            out = out.squeeze(0)  # (m, out_D, out_H, out_W)
            out_D, out_H, out_W = out.shape[1], out.shape[2], out.shape[3]
            if self.rank_op == 'avg':
                window_size = self.rank1 * self.rank2 * self.rank3
                out = out / window_size
            # Reshape to (num_windows, m)
            agg = out.permute(1, 2, 3, 0).reshape(-1, self.m)  # (num_windows, m)
            # Apply mapping matrix M
            transformed = torch.matmul(agg, self.M.T)  # (num_windows, m)

            # Generate coordinates for each window
            coords = []
            for d in range(out_D):
                for h in range(out_H):
                    for w in range(out_W):
                        coords.append((d, h, w))
            vectors = [transformed[i] for i in range(transformed.size(0))]
            return vectors, coords

        # Fallback: loop-based extraction for nonlinear mode or other rank_op
        def apply_op(window):
            """Apply mapping and aggregation to a window of vectors."""
            flat_window = window.reshape(-1, self.m)
            transformed = torch.matmul(flat_window, self.M.T)
            if self.rank_op == 'sum':
                return torch.sum(transformed, dim=0)
            elif self.rank_op == 'pick':
                idx = random.randint(0, transformed.size(0)-1)
                return transformed[idx]
            elif self.rank_op == 'user_func':
                avg = torch.mean(transformed, dim=0)
                return torch.sigmoid(avg)
            else:  # 'avg'
                return torch.mean(transformed, dim=0)

        vectors = []
        coords = []
        step1 = 1 if self.mode == 'linear' else self.step1
        step2 = 1 if self.mode == 'linear' else self.step2
        step3 = 1 if self.mode == 'linear' else self.step3

        for d in range(0, D, step1):
            for h in range(0, H, step2):
                for w in range(0, W, step3):
                    frag = vec_arr[d:d+self.rank1, h:h+self.rank2, w:w+self.rank3, :]
                    frag_d, frag_h, frag_w, _ = frag.shape
                    if self.rank_mode == 'pad':
                        if frag_d < self.rank1 or frag_h < self.rank2 or frag_w < self.rank3:
                            padded = torch.zeros((self.rank1, self.rank2, self.rank3, self.m), device=self.device)
                            padded[:frag_d, :frag_h, :frag_w, :] = frag
                            frag = padded
                    elif self.rank_mode == 'drop':
                        if frag_d != self.rank1 or frag_h != self.rank2 or frag_w != self.rank3:
                            continue
                    vectors.append(apply_op(frag))
                    coords.append((d, h, w))
        return vectors, coords

    def _get_window_data(self, vec_arr):
        """
        Internal helper: returns window aggregated vectors and their coordinates (k1,k2,k3).
        Optimized version using vectorized operations when possible.

        Args:
            vec_arr: tensor of shape (D, H, W, m)

        Returns:
            agg_tensor: torch.Tensor of shape (num_windows, m)
            k1_tensor: torch.Tensor of shape (num_windows,)
            k2_tensor: torch.Tensor of shape (num_windows,)
            k3_tensor: torch.Tensor of shape (num_windows,)
        """
        agg_list, coords = self.extract_vectors(vec_arr)
        if not agg_list:
            return torch.empty(0, self.m, device=self.device), torch.empty(0, device=self.device), torch.empty(0, device=self.device), torch.empty(0, device=self.device)
        agg_tensor = torch.stack(agg_list)  # (num_windows, m)
        k1_tensor = torch.tensor([c[0] for c in coords], dtype=torch.long, device=self.device)
        k2_tensor = torch.tensor([c[1] for c in coords], dtype=torch.long, device=self.device)
        k3_tensor = torch.tensor([c[2] for c in coords], dtype=torch.long, device=self.device)
        return agg_tensor, k1_tensor, k2_tensor, k3_tensor

    def _batch_window_data_3d(self, arr_batch):
        """
        Efficiently compute aggregated window vectors and their positions for a batch of 3D arrays.
        Assumes all arrays have the same shape (D, H, W, m) and conditions for vectorized extraction are met:
            - mode == 'linear'
            - rank_op in ('avg', 'sum')
            - D >= rank1 and H >= rank2 and W >= rank3
        Otherwise returns None to indicate fallback.

        Args:
            arr_batch: torch.Tensor of shape [B, D, H, W, m]

        Returns:
            agg_flat: torch.Tensor of shape [total_windows, m]
            k1_flat: torch.Tensor of shape [total_windows]
            k2_flat: torch.Tensor of shape [total_windows]
            k3_flat: torch.Tensor of shape [total_windows]
            window_counts: torch.Tensor of shape [B] (number of windows per array)
        """
        B, D, H, W, m = arr_batch.shape
        # Check conditions
        if self.mode != 'linear' or self.rank_op not in ('avg', 'sum') or D < self.rank1 or H < self.rank2 or W < self.rank3:
            return None

        # Prepare for 3D convolution: [B, m, D, H, W]
        x = arr_batch.permute(0, 4, 1, 2, 3)  # [B, m, D, H, W]
        # Create weight: (m, 1, rank1, rank2, rank3) filled with ones
        weight = torch.ones(self.m, 1, self.rank1, self.rank2, self.rank3,
                            dtype=torch.float32, device=self.device)
        # Convolution (groups=m ensures each channel is processed independently)
        out = F.conv3d(x, weight, groups=self.m, stride=1, padding=0)  # [B, m, out_D, out_H, out_W]
        out_D, out_H, out_W = out.shape[2], out.shape[3], out.shape[4]
        if self.rank_op == 'avg':
            window_size = self.rank1 * self.rank2 * self.rank3
            out = out / window_size
        # Reshape to [B, num_windows, m]
        agg = out.permute(0, 2, 3, 4, 1).reshape(B, -1, self.m)  # [B, num_windows, m]
        # Apply M
        agg = torch.matmul(agg, self.M.T)  # [B, num_windows, m]

        # Flatten across batch
        agg_flat = agg.reshape(-1, self.m)  # [B*num_windows, m]

        # Generate coordinates for a single array (top-left corner)
        d_coords = torch.arange(out_D, device=self.device).repeat_interleave(out_H * out_W)
        h_coords = torch.arange(out_H, device=self.device).repeat(out_D).repeat_interleave(out_W)
        w_coords = torch.arange(out_W, device=self.device).repeat(out_D * out_H)
        # Repeat for batch
        k1_flat = d_coords.repeat(B)
        k2_flat = h_coords.repeat(B)
        k3_flat = w_coords.repeat(B)
        window_counts = torch.full((B,), out_D * out_H * out_W, dtype=torch.long, device=self.device)

        return agg_flat, k1_flat, k2_flat, k3_flat, window_counts

    def describe(self, vec_arr):
        """Compute N(k1,k2,k3) vectors for each window in the 3D vector array"""
        agg_tensor, k1_tensor, k2_tensor, k3_tensor = self._get_window_data(vec_arr)
        if agg_tensor.shape[0] == 0:
            return []

        j1 = k1_tensor % self.L1
        j2 = k2_tensor % self.L2
        j3 = k3_tensor % self.L3
        idx = j1 * (self.L2 * self.L3) + j2 * self.L3 + j3  # (num_windows,)
        B_rows = self.Bbasis[idx]  # (num_windows, m)
        scalar = torch.sum(B_rows * agg_tensor, dim=1)  # (num_windows,)
        A_cols = self.Acoeff[:, idx].t()  # (num_windows, m)
        Nk = A_cols * scalar.unsqueeze(1)  # (num_windows, m)
        return Nk.detach().cpu().numpy()

    def S(self, vec_arr):
        """Compute cumulative sum of N(k1,k2,k3) vectors (in row-major order)"""
        agg_tensor, k1_tensor, k2_tensor, k3_tensor = self._get_window_data(vec_arr)
        if agg_tensor.shape[0] == 0:
            return []

        j1 = k1_tensor % self.L1
        j2 = k2_tensor % self.L2
        j3 = k3_tensor % self.L3
        idx = j1 * (self.L2 * self.L3) + j2 * self.L3 + j3
        B_rows = self.Bbasis[idx]
        scalar = torch.sum(B_rows * agg_tensor, dim=1)
        A_cols = self.Acoeff[:, idx].t()
        Nk = A_cols * scalar.unsqueeze(1)
        S_cum = torch.cumsum(Nk, dim=0)
        return S_cum.detach().cpu().numpy()

    def D(self, vec_arrs, t_list):
        """
        Compute mean squared deviation D across vector arrays:
        D = average over all windows of (N(k1,k2,k3)-t)^2

        Args:
            vec_arrs: List of 3D vector arrays (each shape [D, H, W, m])
            t_list: List of target vectors corresponding to each array

        Returns:
            float: Average mean squared deviation across all windows and arrays
        """
        total_loss = 0.0
        total_windows = 0
        t_tensors = [torch.tensor(t, dtype=torch.float32, device=self.device) for t in t_list]

        for vec_arr, t in zip(vec_arrs, t_tensors):
            agg_tensor, k1, k2, k3 = self._get_window_data(vec_arr)
            if agg_tensor.shape[0] == 0:
                continue
            j1 = k1 % self.L1
            j2 = k2 % self.L2
            j3 = k3 % self.L3
            idx = j1 * (self.L2 * self.L3) + j2 * self.L3 + j3
            B_rows = self.Bbasis[idx]
            scalar = torch.sum(B_rows * agg_tensor, dim=1)
            A_cols = self.Acoeff[:, idx].t()
            Nk = A_cols * scalar.unsqueeze(1)
            losses = torch.sum((Nk - t) ** 2, dim=1)
            total_loss += losses.sum().item()
            total_windows += agg_tensor.shape[0]
        return total_loss / total_windows if total_windows else 0.0

    def d(self, vec_arr, t):
        """Compute pattern deviation value (d) for a single vector array."""
        return self.D([vec_arr], [t])

    def reg_train(self, vec_arrs, t_list, max_iters=1000, tol=1e-8, learning_rate=0.01,
                  continued=False, decay_rate=1.0, print_every=10, batch_size=32,
                  checkpoint_file=None, checkpoint_interval=10):
        """
        Train model using gradient descent with array-level batch processing.
        Optimized version: uses vectorized batch window extraction when all arrays in a batch have the same shape.
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

        for it in range(start_iter, max_iters):
            total_loss = 0.0
            total_arrays = 0

            indices = list(range(len(vec_arrs)))
            random.shuffle(indices)

            for batch_start in range(0, len(indices), batch_size):
                batch_indices = indices[batch_start:batch_start + batch_size]
                batch_arrs = [vec_arrs[idx] for idx in batch_indices]
                batch_targets = [t_tensors[idx] for idx in batch_indices]

                optimizer.zero_grad()

                # Check if all arrays in the batch have the same shape
                shapes = [(arr.shape[0], arr.shape[1], arr.shape[2]) for arr in batch_arrs]
                if len(set(shapes)) == 1 and self.mode == 'linear' and self.rank_op in ('avg', 'sum'):
                    D, H, W = shapes[0]
                    if D >= self.rank1 and H >= self.rank2 and W >= self.rank3:
                        # Stack into a single tensor [B, D, H, W, m]
                        batch_tensor = torch.stack(batch_arrs, dim=0)
                        result = self._batch_window_data_3d(batch_tensor)
                        if result is not None:
                            agg_flat, d_flat, h_flat, w_flat, window_counts = result
                            # Compute Nk for all windows
                            j1 = d_flat % self.L1
                            j2 = h_flat % self.L2
                            j3 = w_flat % self.L3
                            idx = j1 * (self.L2 * self.L3) + j2 * self.L3 + j3
                            B_rows = self.Bbasis[idx]               # [total_w, m]
                            scalar = torch.sum(B_rows * agg_flat, dim=1)  # [total_w]
                            A_cols = self.Acoeff[:, idx].t()        # [total_w, m]
                            Nk_all = A_cols * scalar.unsqueeze(1)   # [total_w, m]
                            # Split back to arrays
                            array_reps = []
                            start_idx = 0
                            for nw in window_counts:
                                if nw > 0:
                                    arr_rep = Nk_all[start_idx:start_idx+nw].mean(dim=0)
                                    start_idx += nw
                                else:
                                    arr_rep = torch.zeros(self.m, device=self.device)
                                array_reps.append(arr_rep)
                            reps_tensor = torch.stack(array_reps)
                            targets_tensor = torch.stack(batch_targets)
                            loss = torch.mean(torch.sum((reps_tensor - targets_tensor) ** 2, dim=1))
                            loss.backward()
                            optimizer.step()
                            total_loss += loss.item() * len(array_reps)
                            total_arrays += len(array_reps)
                            # Cleanup
                            del batch_tensor, agg_flat, d_flat, h_flat, w_flat, window_counts, Nk_all, array_reps, reps_tensor
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            continue  # skip the fallback loop

                # Fallback: per-array processing for unequal shapes or conditions not met
                all_agg = []
                all_k1 = []
                all_k2 = []
                all_k3 = []
                array_window_counts = []
                for arr in batch_arrs:
                    agg, k1, k2, k3 = self._get_window_data(arr)
                    if agg.shape[0] == 0:
                        array_window_counts.append(0)
                        continue
                    all_agg.append(agg)
                    all_k1.append(k1)
                    all_k2.append(k2)
                    all_k3.append(k3)
                    array_window_counts.append(agg.shape[0])

                if not all_agg:
                    continue

                all_agg = torch.cat(all_agg, dim=0)  # (total_w, m)
                all_k1 = torch.cat(all_k1, dim=0)    # (total_w,)
                all_k2 = torch.cat(all_k2, dim=0)    # (total_w,)
                all_k3 = torch.cat(all_k3, dim=0)    # (total_w,)

                j1 = all_k1 % self.L1
                j2 = all_k2 % self.L2
                j3 = all_k3 % self.L3
                idx = j1 * (self.L2 * self.L3) + j2 * self.L3 + j3
                B_rows = self.Bbasis[idx]          # (total_w, m)
                scalar = torch.sum(B_rows * all_agg, dim=1)  # (total_w,)
                A_cols = self.Acoeff[:, idx].t()   # (total_w, m)
                Nk_all = A_cols * scalar.unsqueeze(1)  # (total_w, m)

                array_reps = []
                start_idx = 0
                for nw in array_window_counts:
                    if nw > 0:
                        arr_rep = Nk_all[start_idx:start_idx+nw].mean(dim=0)
                        start_idx += nw
                    else:
                        arr_rep = torch.zeros(self.m, device=self.device)
                    array_reps.append(arr_rep)

                if array_reps:
                    reps_tensor = torch.stack(array_reps)
                    targets_tensor = torch.stack(batch_targets)
                    loss = torch.mean(torch.sum((reps_tensor - targets_tensor) ** 2, dim=1))
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item() * len(array_reps)
                    total_arrays += len(array_reps)

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            avg_loss = total_loss / total_arrays if total_arrays else 0.0
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
        Optimized version: uses vectorized batch window extraction when all arrays in a batch have the same shape.
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

                # Check if all arrays in the batch have the same shape
                shapes = [(arr.shape[0], arr.shape[1], arr.shape[2]) for arr in batch_arrs]
                if len(set(shapes)) == 1 and self.mode == 'linear' and self.rank_op in ('avg', 'sum'):
                    D, H, W = shapes[0]
                    if D >= self.rank1 and H >= self.rank2 and W >= self.rank3:
                        batch_tensor = torch.stack(batch_arrs, dim=0)
                        result = self._batch_window_data_3d(batch_tensor)
                        if result is not None:
                            agg_flat, d_flat, h_flat, w_flat, window_counts = result
                            j1 = d_flat % self.L1
                            j2 = h_flat % self.L2
                            j3 = w_flat % self.L3
                            idx = j1 * (self.L2 * self.L3) + j2 * self.L3 + j3
                            B_rows = self.Bbasis[idx]
                            scalar = torch.sum(B_rows * agg_flat, dim=1)
                            A_cols = self.Acoeff[:, idx].t()
                            Nk_all = A_cols * scalar.unsqueeze(1)
                            array_reps = []
                            start_idx = 0
                            for nw in window_counts:
                                if nw > 0:
                                    arr_rep = Nk_all[start_idx:start_idx+nw].mean(dim=0)
                                    start_idx += nw
                                else:
                                    arr_rep = torch.zeros(self.m, device=self.device)
                                array_reps.append(arr_rep)
                            reps_tensor = torch.stack(array_reps)
                            logits = self.classifier(reps_tensor)
                            loss = criterion(logits, batch_labels)
                            loss.backward()
                            optimizer.step()
                            total_loss += loss.item() * len(array_reps)
                            total_arrays += len(array_reps)
                            with torch.no_grad():
                                predictions = torch.argmax(logits, dim=1)
                                correct_predictions += (predictions == batch_labels).sum().item()
                            # Cleanup
                            del batch_tensor, agg_flat, d_flat, h_flat, w_flat, window_counts, Nk_all, array_reps, reps_tensor
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            continue  # skip fallback

                # Fallback: per-array processing
                all_agg = []
                all_k1 = []
                all_k2 = []
                all_k3 = []
                array_window_counts = []
                for arr in batch_arrs:
                    agg, k1, k2, k3 = self._get_window_data(arr)
                    if agg.shape[0] == 0:
                        array_window_counts.append(0)
                        continue
                    all_agg.append(agg)
                    all_k1.append(k1)
                    all_k2.append(k2)
                    all_k3.append(k3)
                    array_window_counts.append(agg.shape[0])

                if not all_agg:
                    continue

                all_agg = torch.cat(all_agg, dim=0)
                all_k1 = torch.cat(all_k1, dim=0)
                all_k2 = torch.cat(all_k2, dim=0)
                all_k3 = torch.cat(all_k3, dim=0)

                j1 = all_k1 % self.L1
                j2 = all_k2 % self.L2
                j3 = all_k3 % self.L3
                idx = j1 * (self.L2 * self.L3) + j2 * self.L3 + j3
                B_rows = self.Bbasis[idx]
                scalar = torch.sum(B_rows * all_agg, dim=1)
                A_cols = self.Acoeff[:, idx].t()
                Nk_all = A_cols * scalar.unsqueeze(1)

                array_reps = []
                start_idx = 0
                for nw in array_window_counts:
                    if nw > 0:
                        arr_rep = Nk_all[start_idx:start_idx+nw].mean(dim=0)
                        start_idx += nw
                    else:
                        arr_rep = torch.zeros(self.m, device=self.device)
                    array_reps.append(arr_rep)

                if array_reps:
                    reps_tensor = torch.stack(array_reps)
                    logits = self.classifier(reps_tensor)
                    loss = criterion(logits, batch_labels)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item() * len(array_reps)
                    total_arrays += len(array_reps)
                    with torch.no_grad():
                        predictions = torch.argmax(logits, dim=1)
                        correct_predictions += (predictions == batch_labels).sum().item()

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            avg_loss = total_loss / total_arrays if total_arrays else 0.0
            accuracy = correct_predictions / total_arrays if total_arrays else 0.0
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
        Optimized version: uses vectorized batch window extraction when all arrays in a batch have the same shape.
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

                # Check if all arrays in the batch have the same shape
                shapes = [(arr.shape[0], arr.shape[1], arr.shape[2]) for arr in batch_arrs]
                if len(set(shapes)) == 1 and self.mode == 'linear' and self.rank_op in ('avg', 'sum'):
                    D, H, W = shapes[0]
                    if D >= self.rank1 and H >= self.rank2 and W >= self.rank3:
                        batch_tensor = torch.stack(batch_arrs, dim=0)
                        result = self._batch_window_data_3d(batch_tensor)
                        if result is not None:
                            agg_flat, d_flat, h_flat, w_flat, window_counts = result
                            j1 = d_flat % self.L1
                            j2 = h_flat % self.L2
                            j3 = w_flat % self.L3
                            idx = j1 * (self.L2 * self.L3) + j2 * self.L3 + j3
                            B_rows = self.Bbasis[idx]
                            scalar = torch.sum(B_rows * agg_flat, dim=1)
                            A_cols = self.Acoeff[:, idx].t()
                            Nk_all = A_cols * scalar.unsqueeze(1)
                            array_reps = []
                            start_idx = 0
                            for nw in window_counts:
                                if nw > 0:
                                    arr_rep = Nk_all[start_idx:start_idx+nw].mean(dim=0)
                                    start_idx += nw
                                else:
                                    arr_rep = torch.zeros(self.m, device=self.device)
                                array_reps.append(arr_rep)
                            reps_tensor = torch.stack(array_reps)
                            logits = self.labeller(reps_tensor)
                            loss = criterion(logits, batch_labels)
                            loss.backward()
                            optimizer.step()
                            total_loss += loss.item() * len(array_reps)
                            total_arrays += len(array_reps)
                            with torch.no_grad():
                                probs = torch.sigmoid(logits)
                                predictions = (probs > 0.5).float()
                                total_correct += (predictions == batch_labels).sum().item()
                                total_predictions += batch_labels.numel()
                            # Cleanup
                            del batch_tensor, agg_flat, d_flat, h_flat, w_flat, window_counts, Nk_all, array_reps, reps_tensor
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            continue  # skip fallback

                # Fallback: per-array processing
                all_agg = []
                all_k1 = []
                all_k2 = []
                all_k3 = []
                array_window_counts = []
                for arr in batch_arrs:
                    agg, k1, k2, k3 = self._get_window_data(arr)
                    if agg.shape[0] == 0:
                        array_window_counts.append(0)
                        continue
                    all_agg.append(agg)
                    all_k1.append(k1)
                    all_k2.append(k2)
                    all_k3.append(k3)
                    array_window_counts.append(agg.shape[0])

                if not all_agg:
                    continue

                all_agg = torch.cat(all_agg, dim=0)
                all_k1 = torch.cat(all_k1, dim=0)
                all_k2 = torch.cat(all_k2, dim=0)
                all_k3 = torch.cat(all_k3, dim=0)

                j1 = all_k1 % self.L1
                j2 = all_k2 % self.L2
                j3 = all_k3 % self.L3
                idx = j1 * (self.L2 * self.L3) + j2 * self.L3 + j3
                B_rows = self.Bbasis[idx]
                scalar = torch.sum(B_rows * all_agg, dim=1)
                A_cols = self.Acoeff[:, idx].t()
                Nk_all = A_cols * scalar.unsqueeze(1)

                array_reps = []
                start_idx = 0
                for nw in array_window_counts:
                    if nw > 0:
                        arr_rep = Nk_all[start_idx:start_idx+nw].mean(dim=0)
                        start_idx += nw
                    else:
                        arr_rep = torch.zeros(self.m, device=self.device)
                    array_reps.append(arr_rep)

                if array_reps:
                    reps_tensor = torch.stack(array_reps)
                    logits = self.labeller(reps_tensor)
                    loss = criterion(logits, batch_labels)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item() * len(array_reps)
                    total_arrays += len(array_reps)
                    with torch.no_grad():
                        probs = torch.sigmoid(logits)
                        predictions = (probs > 0.5).float()
                        total_correct += (predictions == batch_labels).sum().item()
                        total_predictions += batch_labels.numel()

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            avg_loss = total_loss / total_arrays if total_arrays else 0.0
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

    def self_train(self, vec_arrs, max_iters=1000, tol=1e-8, learning_rate=0.01,
                   continued=False, decay_rate=1.0, print_every=10, batch_size=32,
                   checkpoint_file=None, checkpoint_interval=10):
        """
        Train the model using self-supervised learning with memory-optimized batch processing.
        Optimized version: uses vectorized batch window extraction when all arrays in a batch have the same shape.
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

        for it in range(start_iter, max_iters):
            total_loss = 0.0
            total_samples = 0

            indices = list(range(len(vec_arrs)))
            random.shuffle(indices)

            for batch_start in range(0, len(indices), batch_size):
                batch_indices = indices[batch_start:batch_start + batch_size]
                batch_arrs = [vec_arrs[idx] for idx in batch_indices]

                optimizer.zero_grad()

                # Check if all arrays in the batch have the same shape
                shapes = [(arr.shape[0], arr.shape[1], arr.shape[2]) for arr in batch_arrs]
                if len(set(shapes)) == 1 and self.mode == 'linear' and self.rank_op in ('avg', 'sum'):
                    D, H, W = shapes[0]
                    if D >= self.rank1 and H >= self.rank2 and W >= self.rank3:
                        batch_tensor = torch.stack(batch_arrs, dim=0)
                        result = self._batch_window_data_3d(batch_tensor)
                        if result is not None:
                            agg_flat, d_flat, h_flat, w_flat, window_counts = result
                            j1 = d_flat % self.L1
                            j2 = h_flat % self.L2
                            j3 = w_flat % self.L3
                            idx = j1 * (self.L2 * self.L3) + j2 * self.L3 + j3
                            B_rows = self.Bbasis[idx]
                            scalar = torch.sum(B_rows * agg_flat, dim=1)
                            A_cols = self.Acoeff[:, idx].t()
                            Nk_all = A_cols * scalar.unsqueeze(1)
                            # Self-supervised loss: each window should reconstruct itself
                            loss = torch.mean(torch.sum((Nk_all - agg_flat) ** 2, dim=1))
                            loss.backward()
                            optimizer.step()
                            total_loss += loss.item() * agg_flat.size(0)
                            total_samples += agg_flat.size(0)
                            # Cleanup
                            del batch_tensor, agg_flat, d_flat, h_flat, w_flat, window_counts, Nk_all
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            continue  # skip fallback

                # Fallback: per-array processing
                for arr in batch_arrs:
                    agg, k1, k2, k3 = self._get_window_data(arr)
                    if agg.shape[0] == 0:
                        continue
                    j1 = k1 % self.L1
                    j2 = k2 % self.L2
                    j3 = k3 % self.L3
                    idx = j1 * (self.L2 * self.L3) + j2 * self.L3 + j3
                    B_rows = self.Bbasis[idx]
                    scalar = torch.sum(B_rows * agg, dim=1)
                    A_cols = self.Acoeff[:, idx].t()
                    Nk = A_cols * scalar.unsqueeze(1)
                    loss = torch.mean(torch.sum((Nk - agg) ** 2, dim=1))
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item() * agg.size(0)
                    total_samples += agg.size(0)

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            avg_loss = total_loss / total_samples if total_samples else 0.0
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
                'rank1': self.rank1,
                'rank2': self.rank2,
                'rank3': self.rank3,
                'rank_op': self.rank_op,
                'rank_mode': self.rank_mode,
                'mode': self.mode,
                'step1': self.step1,
                'step2': self.step2,
                'step3': self.step3
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
        Optimized version using batch window collection.
        """
        total_window_count = 0
        total_t = torch.zeros(self.m, device=self.device)

        with torch.no_grad():
            for i in range(0, len(vec_arrs), batch_size):
                batch_arrs = vec_arrs[i:i+batch_size]
                batch_window_count = 0
                batch_t_sum = torch.zeros(self.m, device=self.device)

                for arr in batch_arrs:
                    agg, k1, k2, k3 = self._get_window_data(arr)
                    if agg.shape[0] == 0:
                        continue
                    batch_window_count += agg.shape[0]
                    j1 = k1 % self.L1
                    j2 = k2 % self.L2
                    j3 = k3 % self.L3
                    idx = j1 * (self.L2 * self.L3) + j2 * self.L3 + j3
                    B_rows = self.Bbasis[idx]
                    scalar = torch.sum(B_rows * agg, dim=1)
                    A_cols = self.Acoeff[:, idx].t()
                    Nk = A_cols * scalar.unsqueeze(1)
                    batch_t_sum += Nk.sum(dim=0)

                total_window_count += batch_window_count
                total_t += batch_t_sum

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        self.mean_window_count = total_window_count / len(vec_arrs) if vec_arrs else 0
        self.mean_t = (total_t / total_window_count).cpu().numpy() if total_window_count > 0 else np.zeros(self.m)

    def predict_t(self, vec_arr):
        """Predict target vector as average of N(k1,k2,k3) vectors"""
        agg, k1, k2, k3 = self._get_window_data(vec_arr)
        if agg.shape[0] == 0:
            return np.zeros(self.m)
        j1 = k1 % self.L1
        j2 = k2 % self.L2
        j3 = k3 % self.L3
        idx = j1 * (self.L2 * self.L3) + j2 * self.L3 + j3
        B_rows = self.Bbasis[idx]
        scalar = torch.sum(B_rows * agg, dim=1)
        A_cols = self.Acoeff[:, idx].t()
        Nk = A_cols * scalar.unsqueeze(1)
        return torch.mean(Nk, dim=0).detach().cpu().numpy()

    def predict_c(self, vec_arr):
        """
        Predict class label for a 3D vector array using the classification head.

        Args:
            vec_arr: Input 3D vector array

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
        Predict multi-label classification for a 3D vector array.

        Args:
            vec_arr: Input 3D vector array
            threshold: Probability threshold for binary classification (default: 0.5)

        Returns:
            tuple: (binary_predictions, probability_scores)
        """
        assert self.labeller is not None, "Model must be trained first for label prediction"

        agg, k1, k2, k3 = self._get_window_data(vec_arr)
        if agg.shape[0] == 0:
            return np.zeros(self.num_labels, dtype=np.float32), np.zeros(self.num_labels, dtype=np.float32)

        j1 = k1 % self.L1
        j2 = k2 % self.L2
        j3 = k3 % self.L3
        idx = j1 * (self.L2 * self.L3) + j2 * self.L3 + j3
        B_rows = self.Bbasis[idx]
        scalar = torch.sum(B_rows * agg, dim=1)
        A_cols = self.Acoeff[:, idx].t()
        Nk = A_cols * scalar.unsqueeze(1)
        seq_rep = torch.mean(Nk, dim=0)

        with torch.no_grad():
            logits = self.labeller(seq_rep)
            probs = torch.sigmoid(logits).cpu().numpy()

        binary_preds = (probs > threshold).astype(np.float32)
        return binary_preds, probs

    def reconstruct(self, D, H, W, tau=0.0):
        """
        Reconstruct a representative 3D vector array of size D x H x W with temperature-controlled randomness.
        Uses non-overlapping windows (step = rank1/rank2/rank3) to fill the grid.

        Args:
            D (int): Desired depth of output array
            H (int): Desired height of output array
            W (int): Desired width of output array
            tau (float): Temperature for stochastic selection (0 = deterministic)

        Returns:
            torch.Tensor: Reconstructed array of shape [D, H, W, m]
        """
        assert self.trained, "Model must be trained first"
        if tau < 0:
            raise ValueError("Temperature must be non-negative")

        # Number of windows needed to cover the grid (using step = rank)
        d_win = (D + self.rank1 - 1) // self.rank1
        h_win = (H + self.rank2 - 1) // self.rank2
        w_win = (W + self.rank3 - 1) // self.rank3

        mean_t_tensor = torch.tensor(self.mean_t, dtype=torch.float32, device=self.device)
        candidate_vecs = torch.randn(100, self.m, device=self.device)  # 100 random candidates

        generated_vectors = []  # list of tensors for each window

        for d in range(d_win):
            for h in range(h_win):
                for w in range(w_win):
                    j1 = d % self.L1
                    j2 = h % self.L2
                    j3 = w % self.L3
                    idx = j1 * (self.L2 * self.L3) + j2 * self.L3 + j3
                    B_row = self.Bbasis[idx].unsqueeze(0)  # [1, m]

                    # Transform candidates through M
                    transformed_candidates = torch.matmul(candidate_vecs, self.M.T)  # [100, m]

                    # Simulate window aggregation (the candidate vector is the aggregated vector for the window)
                    # For simplicity, we treat the candidate as the aggregated vector after M and aggregation.
                    # So we directly use candidate as the aggregated vector for scalar computation.
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
        full_arr = torch.zeros(D, H, W, self.m, device=self.device)
        win_idx = 0
        for d in range(d_win):
            for h in range(h_win):
                for w in range(w_win):
                    vec = generated_vectors[win_idx]
                    d_start = d * self.rank1
                    d_end = min(d_start + self.rank1, D)
                    h_start = h * self.rank2
                    h_end = min(h_start + self.rank2, H)
                    w_start = w * self.rank3
                    w_end = min(w_start + self.rank3, W)
                    full_arr[d_start:d_end, h_start:h_end, w_start:w_end, :] = vec
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
                'rank1': self.rank1,
                'rank2': self.rank2,
                'rank3': self.rank3,
                'rank_op': self.rank_op,
                'rank_mode': self.rank_mode,
                'mode': self.mode,
                'step1': self.step1,
                'step2': self.step2,
                'step3': self.step3
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
    print("Spatial Numeric Dual Descriptor AB3 - 3D Array Version (PyTorch GPU Accelerated)")
    print("Optimized with vectorized window extraction and batch processing")
    print("Equal-length batch acceleration added")
    print("="*60)

    # Set random seeds for reproducibility
    torch.manual_seed(11)
    random.seed(11)

    vec_dim = 3
    bas_dim1 = 3
    bas_dim2 = 3
    bas_dim3 = 3
    rank1 = 2
    rank2 = 2
    rank3 = 2
    arr_num = 30

    # Generate 3D vector arrays and random targets
    vec_arrs, t_list = [], []
    for _ in range(arr_num):
        D = random.randint(10, 15)
        H = random.randint(10, 15)
        W = random.randint(10, 15)
        arr = torch.randn(D, H, W, vec_dim)  # Random 3D array of vectors
        vec_arrs.append(arr)
        t_list.append([random.uniform(-1,1) for _ in range(vec_dim)])

    # Create model
    dd = SpatialNumDualDescriptorAB3(
        vec_dim=vec_dim,
        bas_dim1=bas_dim1, bas_dim2=bas_dim2, bas_dim3=bas_dim3,
        rank1=rank1, rank2=rank2, rank3=rank3,
        mode='linear',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    print(f"\nUsing device: {dd.device}")
    print(f"Vector dimension: {dd.m}")
    print(f"Basis dimensions: {dd.L1} x {dd.L2} x {dd.L3}")
    print(f"Window size: {dd.rank1} x {dd.rank2} x {dd.rank3}")

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
    arr_det = dd.reconstruct(D=10, H=10, W=10, tau=0.0)
    arr_rand = dd.reconstruct(D=10, H=10, W=10, tau=0.5)
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
        for _ in range(20):  # 20 arrays per class
            D = random.randint(8, 12)
            H = random.randint(8, 12)
            W = random.randint(8, 12)
            if class_id == 0:
                # Class 0: pattern with positive mean
                arr = torch.randn(D, H, W, vec_dim) + 0.5
            elif class_id == 1:
                # Class 1: pattern with negative mean
                arr = torch.randn(D, H, W, vec_dim) - 0.5
            else:
                # Class 2: normal distribution
                arr = torch.randn(D, H, W, vec_dim)
            class_arrs.append(arr)
            class_labels.append(class_id)

    dd_cls = SpatialNumDualDescriptorAB3(
        vec_dim=vec_dim, bas_dim1=bas_dim1, bas_dim2=bas_dim2, bas_dim3=bas_dim3,
        rank1=rank1, rank2=rank2, rank3=rank3, mode='linear',
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
    for _ in range(40):
        D = random.randint(8, 12)
        H = random.randint(8, 12)
        W = random.randint(8, 12)
        arr = torch.randn(D, H, W, vec_dim)
        label_arrs.append(arr)
        label_vec = [random.random() > 0.7 for _ in range(num_labels)]
        labels.append([1.0 if x else 0.0 for x in label_vec])

    dd_lbl = SpatialNumDualDescriptorAB3(
        vec_dim=vec_dim, bas_dim1=bas_dim1, bas_dim2=bas_dim2, bas_dim3=bas_dim3,
        rank1=rank1, rank2=rank2, rank3=rank3, mode='linear',
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
    test_arr = torch.randn(10, 10, 10, vec_dim)
    binary_pred, probs_pred = dd_lbl.predict_l(test_arr, threshold=0.5)
    print("\nMulti-label prediction example:")
    print(f"Predicted binary labels: {binary_pred}")
    print(f"Predicted probabilities: {[f'{p:.4f}' for p in probs_pred]}")

    # === Self-Training Example ===
    print("\n" + "="*50)
    print("Self-Supervised Learning (Self-Training)")
    print("="*50)

    self_arrs = []
    for _ in range(15):
        D = random.randint(8, 12)
        H = random.randint(8, 12)
        W = random.randint(8, 12)
        arr = torch.randn(D, H, W, vec_dim)
        self_arrs.append(arr)

    dd_self = SpatialNumDualDescriptorAB3(
        vec_dim=vec_dim, bas_dim1=bas_dim1, bas_dim2=bas_dim2, bas_dim3=bas_dim3,
        rank1=rank1, rank2=rank2, rank3=rank3, mode='linear',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    self_history = dd_self.self_train(
        self_arrs,
        max_iters=30,
        learning_rate=0.001,
        decay_rate=0.995,
        print_every=15,
        batch_size=256
    )

    self_arr_recon = dd_self.reconstruct(D=8, H=8, W=8, tau=0.0)
    print(f"\nSelf-trained model reconstruction shape: {self_arr_recon.shape}")

    # === Model Persistence Test ===
    print("\n" + "="*50)
    print("Model Persistence Test")
    print("="*50)

    dd_self.save("self_trained_model_3d.pkl")
    dd_loaded = SpatialNumDualDescriptorAB3(
        vec_dim=vec_dim, bas_dim1=bas_dim1, bas_dim2=bas_dim2, bas_dim3=bas_dim3,
        rank1=rank1, rank2=rank2, rank3=rank3, mode='linear',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    ).load("self_trained_model_3d.pkl")

    print("Model loaded successfully. Reconstructing with loaded model:")
    recon = dd_loaded.reconstruct(D=6, H=6, W=6, tau=0.0)
    print(f"Reconstruction shape: {recon.shape}")

    print("\n=== All Tests Completed ===")
