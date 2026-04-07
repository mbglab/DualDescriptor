# Copyright (C) 2005-2026, Bin-Guang Ma (mbg@mail.hzau.edu.cn); SPDX-License-Identifier: MIT
# The Spatial Numeric Dual Descriptor Vector class (Random AB matrix form) for 3D array implemented with PyTorch
# This program is for the demonstration of methodology and not fully refined.
# Author: Bin-Guang Ma (assisted by DeepSeek); Date: 2025-8-28 ~ 2026-04-03

import math
import random
import itertools
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import copy
import os

class SpatialNumDualDescriptorRN3(nn.Module):
    """
    Dual Descriptor for 3D vector volumes with GPU acceleration using PyTorch:
      - Learnable coefficient matrix Acoeff ∈ R^{m×(L1·L2·L3)}
      - Learnable basis matrix Bbasis ∈ R^{(L1·L2·L3)×m}
      - Learnable mapping matrix M ∈ R^{m×m}
      - Input: 3D volumes of m-dimensional vectors (D × H × W × m)
      - Optimized with batch processing for GPU acceleration
      - Supports both linear and nonlinear 3D windowing
      - Supports regression, classification, and multi-label classification tasks
      - Batch acceleration for same-sized volumes (inspired by SnDDab3.py)
    """
    def __init__(self, vec_dim=4, bas_dim1=5, bas_dim2=5, bas_dim3=5, rank=1, rank_op='avg', rank_mode='drop',
                 mode='linear', user_step=None, device='cuda'):
        """
        Initialize the Spatial Dual Descriptor for 3D vector volumes.
        
        Args:
            vec_dim (int): Dimension m of input vectors
            bas_dim1 (int): Basis dimension for first index (depth)
            bas_dim2 (int): Basis dimension for second index (height)
            bas_dim3 (int): Basis dimension for third index (width)
            rank (int): Window size (cubic window, side length) for vector aggregation
            rank_op (str): 'avg', 'sum', 'pick', or 'user_func'
            rank_mode (str): 'pad' or 'drop' for handling incomplete windows
            mode (str): 'linear' (sliding window, step=1) or 'nonlinear' (stepped window)
            user_step (int): Custom step size for nonlinear mode (same for all dimensions)
            device (str): 'cuda' or 'cpu'
        """
        super().__init__()
        self.m = vec_dim                     # Vector dimension
        self.L1 = bas_dim1                   # Basis dimension for depth
        self.L2 = bas_dim2                   # Basis dimension for height
        self.L3 = bas_dim3                   # Basis dimension for width
        self.total_basis = self.L1 * self.L2 * self.L3
        self.rank = rank                     # Window size (cube)
        self.rank_op = rank_op
        self.rank_mode = rank_mode
        assert mode in ('linear', 'nonlinear')
        self.mode = mode
        self.step = user_step or rank        # Step size (default = rank)
        self.trained = False
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # No tokens needed for numerical vectors
        self.tokens = []
        self.token_to_idx = {}
        self.idx_to_token = {}
        
        # Mapping matrix M: m×m
        self.M = nn.Parameter(torch.empty(self.m, self.m))
        
        # Coefficient matrix Acoeff: m × total_basis
        self.Acoeff = nn.Parameter(torch.empty(self.m, self.total_basis))
        
        # Basis matrix Bbasis: total_basis × m
        self.Bbasis = nn.Parameter(torch.empty(self.total_basis, self.m))
        
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
        nn.init.uniform_(self.M, -0.5, 0.5)
        nn.init.uniform_(self.Acoeff, -0.1, 0.1)
        nn.init.uniform_(self.Bbasis, -0.1, 0.1)
        
        if self.classifier is not None:
            nn.init.normal_(self.classifier.weight, 0, 0.01)
            if self.classifier.bias is not None:
                nn.init.constant_(self.classifier.bias, 0)
                
        if self.labeller is not None:
            nn.init.xavier_uniform_(self.labeller.weight)
            nn.init.zeros_(self.labeller.bias)
    
    def map_vector(self, vector):
        """Apply mapping matrix M to input vector."""
        if vector.dim() == 1:
            return torch.mv(self.M, vector)
        else:
            return torch.mm(vector, self.M.t())
    
    def _get_composite_index(self, k1, k2, k3):
        """
        Convert 3D position indices to a composite index.
        Composite index = k1 * L2 * L3 + k2 * L3 + k3
        Modulo total_basis yields (k1 % L1, k2 % L2, k3 % L3).
        """
        return k1 * self.L2 * self.L3 + k2 * self.L3 + k3
    
    def _compute_basis_index(self, composite_k):
        """Compute basis index j = composite_k % total_basis. Supports tensor input."""
        if isinstance(composite_k, torch.Tensor):
            return (composite_k % self.total_basis).long()
        else:
            return (composite_k % self.total_basis)
    
    def extract_vectors(self, volume):
        """
        Extract vector groups from a 3D vector volume using sliding windows and return
        aggregated vectors along with their composite position indices.
        
        The volume is expected to have shape (D, H, W, m) where D, H, W are dimensions.
        Windows are cubic of size rank × rank × rank. They are extracted in row-major order
        (depth-first, then height, then width).
        
        - 'linear': slide window by 1 step in all dimensions
        - 'nonlinear': slide window by step (given or rank) in all dimensions
        
        For incomplete windows at the boundaries, handles them using:
        - 'pad': pads with zero vectors
        - 'drop': discards incomplete windows
        
        Returns:
            tuple: (vectors_tensor, composite_indices_tensor)
                vectors_tensor: [num_windows, m] aggregated vectors
                composite_indices_tensor: [num_windows] composite indices
        """
        # Convert to tensor if needed, ensure shape (D, H, W, m)
        if not isinstance(volume, torch.Tensor):
            volume = torch.tensor(np.array(volume), dtype=torch.float32, device=self.device)
        if volume.dim() == 3:
            volume = volume.unsqueeze(-1)  # (D, H, W, 1)
        if volume.dim() == 4 and volume.shape[-1] != self.m:
            # Possibly volume is (D, H, W, something) but not m? assume it's fine
            pass
        
        D, H, W = volume.shape[0], volume.shape[1], volume.shape[2]
        step = self.step
        
        windows = []
        composite_indices = []
        
        def apply_op(vectors):
            # vectors: (rank*rank*rank, m)
            if self.rank_op == 'sum':
                return torch.sum(vectors, dim=0)
            elif self.rank_op == 'pick':
                idx = random.randint(0, vectors.size(0)-1)
                return vectors[idx]
            elif self.rank_op == 'user_func':
                avg = torch.mean(vectors, dim=0)
                return torch.sigmoid(avg)
            else:  # 'avg' is default
                return torch.mean(vectors, dim=0)
        
        if self.mode == 'linear':
            # Step 1 in all dimensions
            for i in range(D - self.rank + 1):
                for j in range(H - self.rank + 1):
                    for k in range(W - self.rank + 1):
                        window_vectors = volume[i:i+self.rank, j:j+self.rank, k:k+self.rank].reshape(-1, self.m)
                        agg_vec = apply_op(window_vectors)
                        windows.append(agg_vec)
                        composite_indices.append(self._get_composite_index(i, j, k))
        else:
            # Nonlinear mode: step stride
            for i in range(0, D, step):
                for j in range(0, H, step):
                    for k in range(0, W, step):
                        i_end = i + self.rank
                        j_end = j + self.rank
                        k_end = k + self.rank
                        if i_end > D or j_end > H or k_end > W:
                            if self.rank_mode == 'pad':
                                # Collect existing vectors and pad missing ones
                                window_list = []
                                for di in range(i, min(i_end, D)):
                                    for dj in range(j, min(j_end, H)):
                                        for dk in range(k, min(k_end, W)):
                                            window_list.append(volume[di, dj, dk])
                                pad_count = self.rank*self.rank*self.rank - len(window_list)
                                if pad_count > 0:
                                    zero_pad = torch.zeros(pad_count, self.m, device=self.device)
                                    window_vectors = torch.cat([torch.stack(window_list), zero_pad])
                                else:
                                    window_vectors = torch.stack(window_list)
                            else:  # drop
                                continue
                        else:
                            window_vectors = volume[i:i+self.rank, j:j+self.rank, k:k+self.rank].reshape(-1, self.m)
                        
                        agg_vec = apply_op(window_vectors)
                        windows.append(agg_vec)
                        composite_indices.append(self._get_composite_index(i, j, k))
        
        if windows:
            return torch.stack(windows), torch.tensor(composite_indices, dtype=torch.long, device=self.device)
        else:
            return torch.empty(0, self.m, device=self.device), torch.empty(0, dtype=torch.long, device=self.device)
    
    def batch_compute_Nk(self, composite_k_tensor, vectors):
        """
        Vectorized computation of N(k1,k2,k3) vectors for a batch of composite positions and vectors.
        
        Args:
            composite_k_tensor: Tensor of composite indices [batch_size]
            vectors: Tensor of input vectors [batch_size, m]
            
        Returns:
            Tensor of N(k1,k2,k3) vectors [batch_size, m]
        """
        # Apply mapping matrix M to input vectors
        x = self.map_vector(vectors)  # [batch_size, m]
        
        # Calculate basis index j = composite_k % total_basis
        j_indices = self._compute_basis_index(composite_k_tensor)
        
        # Get Bbasis vectors: [batch_size, m]
        B_j = self.Bbasis[j_indices]
        
        # Compute scalar projection: B_j • x [batch_size]
        scalar = torch.sum(B_j * x, dim=1, keepdim=True)
        
        # Get Acoeff vectors: [batch_size, m]
        A_j = self.Acoeff.permute(1, 0)[j_indices]
        
        # Compute Nk = scalar * A_j [batch_size, m]
        Nk = scalar * A_j
            
        return Nk
    
    # ----------------------------------------------------------------------
    #  Batch acceleration methods (inspired by SnDDab3.py)
    # ----------------------------------------------------------------------
    def _get_window_coords_and_indices_3d(self, D, H, W):
        """
        Compute window top-left coordinates and corresponding composite indices for a given array size.
        Used by batch processing methods.
        
        Args:
            D, H, W: Depth, height, width of the array
        
        Returns:
            coords: List of (d, h, w) tuples for each window
            comp_indices: Tensor of shape [n_windows] containing composite indices
        """
        step = 1 if self.mode == 'linear' else self.step
        
        if self.rank_mode == 'drop':
            # Only windows fully inside
            d_range = range(0, D - self.rank + 1, step)
            h_range = range(0, H - self.rank + 1, step)
            w_range = range(0, W - self.rank + 1, step)
        else:  # 'pad'
            # Include all windows that start before D,H,W (pad with zeros if needed)
            d_range = range(0, D, step)
            h_range = range(0, H, step)
            w_range = range(0, W, step)
        
        coords = []
        comp_list = []
        for d in d_range:
            for h in h_range:
                for w in w_range:
                    coords.append((d, h, w))
                    comp_list.append(self._get_composite_index(d, h, w))
        
        if not coords:
            return [], torch.empty(0, dtype=torch.long, device=self.device)
        
        comp_tensor = torch.tensor(comp_list, dtype=torch.long, device=self.device)
        return coords, comp_tensor
    
    def batch_represent(self, vol_batch):
        """
        Compute array representations for a batch of 3D vector volumes efficiently.
        Supports arbitrary window size and step (linear/nonlinear) as long as all volumes
        in the batch have identical dimensions.
        
        Args:
            vol_batch: Tensor of shape [batch_size, D, H, W, m]
        
        Returns:
            representations: Tensor of shape [batch_size, m]
        """
        B, D, H, W, m = vol_batch.shape
        step = 1 if self.mode == 'linear' else self.step
        
        # Convert to (B, m, D, H, W) for 3D unfolding
        x = vol_batch.permute(0, 4, 1, 2, 3)  # [B, m, D, H, W]
        
        # Pad if necessary (for 'pad' mode)
        if self.rank_mode == 'pad':
            pad_d = (self.rank - 1) if step == 1 else (self.rank - step)
            pad_h = pad_d
            pad_w = pad_d
            # Padding order: (pad_w_last, pad_w_first, pad_h_last, pad_h_first, pad_d_last, pad_d_first)
            x = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_d), mode='constant', value=0)
            D_pad = D + pad_d
            H_pad = H + pad_h
            W_pad = W + pad_w
        else:
            D_pad, H_pad, W_pad = D, H, W
        
        # Compute number of windows along each dimension
        n_windows_d = ((D_pad - self.rank) // step + 1) if D_pad >= self.rank else 0
        n_windows_h = ((H_pad - self.rank) // step + 1) if H_pad >= self.rank else 0
        n_windows_w = ((W_pad - self.rank) // step + 1) if W_pad >= self.rank else 0
        total_windows = n_windows_d * n_windows_h * n_windows_w
        if total_windows == 0:
            # No windows, return zero representations
            return torch.zeros(B, m, device=self.device)
        
        # Prepare to collect aggregated vectors for all windows
        agg_all = torch.zeros(B, total_windows, m, device=self.device)
        win_size = self.rank * self.rank * self.rank
        
        idx_offset = 0
        for d_start in range(0, D_pad - self.rank + 1, step):
            # Extract depth slice of size (rank) for each batch
            x_d = x[:, :, d_start:d_start+self.rank, :, :]  # [B, m, rank, H_pad, W_pad]
            # Merge depth into channel dimension: [B, m*rank, H_pad, W_pad]
            x_d = x_d.contiguous().view(B, m * self.rank, H_pad, W_pad)
            # Apply 2D unfold to get all spatial windows for this depth offset
            windows_2d = F.unfold(
                x_d, kernel_size=(self.rank, self.rank), stride=(step, step)
            )  # [B, m*rank*rank*rank, n_spatial]
            n_spatial = windows_2d.size(2)
            # Reshape to [B, n_spatial, win_size, m]
            windows_2d = windows_2d.view(B, m, win_size, n_spatial).permute(0, 3, 2, 1)
            # Apply mapping matrix M
            mapped = torch.matmul(windows_2d, self.M.T)  # [B, n_spatial, win_size, m]
            # Aggregate over win_size dimension
            if self.rank_op == 'sum':
                agg = mapped.sum(dim=2)
            elif self.rank_op == 'pick':
                idx = torch.randint(0, win_size, (B, n_spatial, 1), device=self.device)
                agg = torch.gather(mapped, 2, idx.expand(-1, -1, -1, m)).squeeze(2)
            elif self.rank_op == 'user_func':
                avg = mapped.mean(dim=2)
                agg = torch.sigmoid(avg)
            else:  # 'avg'
                agg = mapped.mean(dim=2)  # [B, n_spatial, m]
            # Store in agg_all
            agg_all[:, idx_offset:idx_offset+n_spatial, :] = agg
            idx_offset += n_spatial
        
        # Get composite indices for each window
        coords, comp_indices = self._get_window_coords_and_indices_3d(D, H, W)
        if not coords:
            return torch.zeros(B, m, device=self.device)
        comp_indices = comp_indices.to(self.device)  # [n_windows]
        
        # Get corresponding B rows and A columns
        j_indices = self._compute_basis_index(comp_indices)  # [n_windows]
        B_rows = self.Bbasis[j_indices]  # [n_windows, m]
        B_rows = B_rows.unsqueeze(0)  # [1, n_windows, m]
        A_cols = self.Acoeff[:, j_indices].T  # [n_windows, m]
        A_cols = A_cols.unsqueeze(0)  # [1, n_windows, m]
        
        # Compute scalar = B_row • agg
        scalar = torch.sum(B_rows * agg_all, dim=2)  # [B, n_windows]
        # Compute Nk = A_cols * scalar
        Nk = A_cols * scalar.unsqueeze(2)  # [B, n_windows, m]
        # Average over windows to get representation
        rep = Nk.mean(dim=1)  # [B, m]
        return rep
    
    def batch_compute_Nk_and_targets(self, vol_batch):
        """
        Compute Nk vectors and target vectors for each position (window) in a batch of 3D volumes.
        Supports arbitrary window size and step.
        
        Args:
            vol_batch: Tensor of shape [batch_size, D, H, W, m]
        
        Returns:
            Nk_all: Tensor [batch_size, n_windows, m] - Nk vectors for each window
            targets: Tensor [batch_size, n_windows, m] - target vectors (aggregated mapped vectors)
        """
        B, D, H, W, m = vol_batch.shape
        step = 1 if self.mode == 'linear' else self.step
        
        x = vol_batch.permute(0, 4, 1, 2, 3)  # [B, m, D, H, W]
        
        if self.rank_mode == 'pad':
            pad_d = (self.rank - 1) if step == 1 else (self.rank - step)
            pad_h = pad_d
            pad_w = pad_d
            x = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_d), mode='constant', value=0)
            D_pad = D + pad_d
            H_pad = H + pad_h
            W_pad = W + pad_w
        else:
            D_pad, H_pad, W_pad = D, H, W
        
        n_windows_d = ((D_pad - self.rank) // step + 1) if D_pad >= self.rank else 0
        n_windows_h = ((H_pad - self.rank) // step + 1) if H_pad >= self.rank else 0
        n_windows_w = ((W_pad - self.rank) // step + 1) if W_pad >= self.rank else 0
        total_windows = n_windows_d * n_windows_h * n_windows_w
        if total_windows == 0:
            return torch.empty(B, 0, m, device=self.device), torch.empty(B, 0, m, device=self.device)
        
        targets_all = torch.zeros(B, total_windows, m, device=self.device)
        win_size = self.rank * self.rank * self.rank
        idx_offset = 0
        
        for d_start in range(0, D_pad - self.rank + 1, step):
            x_d = x[:, :, d_start:d_start+self.rank, :, :]  # [B, m, rank, H_pad, W_pad]
            x_d = x_d.contiguous().view(B, m * self.rank, H_pad, W_pad)
            windows_2d = F.unfold(
                x_d, kernel_size=(self.rank, self.rank), stride=(step, step)
            )  # [B, m*rank*rank*rank, n_spatial]
            n_spatial = windows_2d.size(2)
            windows_2d = windows_2d.view(B, m, win_size, n_spatial).permute(0, 3, 2, 1)
            # Apply mapping
            mapped = torch.matmul(windows_2d, self.M.T)  # [B, n_spatial, win_size, m]
            if self.rank_op == 'sum':
                targets = mapped.sum(dim=2)
            elif self.rank_op == 'pick':
                idx = torch.randint(0, win_size, (B, n_spatial, 1), device=self.device)
                targets = torch.gather(mapped, 2, idx.expand(-1, -1, -1, m)).squeeze(2)
            elif self.rank_op == 'user_func':
                avg = mapped.mean(dim=2)
                targets = torch.sigmoid(avg)
            else:  # 'avg'
                targets = mapped.mean(dim=2)  # [B, n_spatial, m]
            targets_all[:, idx_offset:idx_offset+n_spatial, :] = targets
            idx_offset += n_spatial
        
        # Compute composite indices for all windows
        coords, comp_indices = self._get_window_coords_and_indices_3d(D, H, W)
        if not coords:
            return torch.empty(B, 0, m, device=self.device), targets_all
        comp_indices = comp_indices.to(self.device)
        j_indices = self._compute_basis_index(comp_indices)  # [n_windows]
        B_rows = self.Bbasis[j_indices].unsqueeze(0)  # [1, n_windows, m]
        A_cols = self.Acoeff[:, j_indices].T.unsqueeze(0)  # [1, n_windows, m]
        scalar = torch.sum(B_rows * targets_all, dim=2)  # [B, n_windows]
        Nk_all = A_cols * scalar.unsqueeze(2)  # [B, n_windows, m]
        return Nk_all, targets_all
    
    # ----------------------------------------------------------------------
    #  Original methods (describe, S, D, d, etc.)
    # ----------------------------------------------------------------------
    def describe(self, volume):
        """Compute N(k1,k2,k3) vectors for each window in the volume."""
        vectors, comp_indices = self.extract_vectors(volume)
        if vectors.shape[0] == 0:
            return np.array([])
        N_batch = self.batch_compute_Nk(comp_indices, vectors)
        return N_batch.detach().cpu().numpy()
    
    def S(self, volume):
        """
        Compute list of cumulative sums S(l) = sum_{i=1..l} N(k_i) for windows in row-major order.
        """
        vectors, comp_indices = self.extract_vectors(volume)
        if vectors.shape[0] == 0:
            return []
        N_batch = self.batch_compute_Nk(comp_indices, vectors)
        S_cum = torch.cumsum(N_batch, dim=0)
        return [s.detach().cpu().numpy() for s in S_cum]
    
    def D(self, volumes, t_list):
        """
        Compute mean squared deviation D across volumes:
        D = average over all windows of (N(k1,k2,k3)-t_seq)^2
        """
        total_loss = 0.0
        total_positions = 0
        
        t_tensors = [torch.tensor(t, dtype=torch.float32, device=self.device) for t in t_list]
        
        for vol, t in zip(volumes, t_tensors):
            vectors, comp_indices = self.extract_vectors(vol)
            if vectors.shape[0] == 0:
                continue
            N_batch = self.batch_compute_Nk(comp_indices, vectors)
            losses = torch.sum((N_batch - t) ** 2, dim=1)
            total_loss += losses.sum().item()
            total_positions += vectors.shape[0]
        return total_loss / total_positions if total_positions else 0.0
    
    def d(self, volume, t):
        """Compute pattern deviation value (d) for a single volume."""
        return self.D([volume], [t])
    
    def reg_train(self, volumes, t_list, max_iters=1000, tol=1e-8, learning_rate=0.01,
                  continued=False, decay_rate=1.0, print_every=10, batch_size=32,
                  checkpoint_file=None, checkpoint_interval=10):
        """
        Train model using gradient descent with volume-level batch processing.
        
        Args:
            volumes: List of 3D vector volumes (each as numpy array or torch tensor, shape D×H×W×m)
            t_list: List of target vectors
            ... (other parameters same as original)
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
        
        t_tensors = [torch.tensor(t, dtype=torch.float32, device=self.device) for t in t_list]
        # Ensure all volumes are tensors on device
        volumes = [vol.to(self.device) if isinstance(vol, torch.Tensor) else
                   torch.tensor(vol, dtype=torch.float32, device=self.device)
                   for vol in volumes]
        
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)
        
        if continued and checkpoint_file and os.path.exists(checkpoint_file):
            optimizer.load_state_dict(optimizer_state)
            scheduler.load_state_dict(scheduler_state)
        
        prev_loss = float('inf') if start_iter == 0 else history[-1] if history else float('inf')
        
        # Fast batch processing possible when all volumes have identical dimensions and rank_mode == 'drop'
        all_shapes = [vol.shape for vol in volumes]
        all_equal = len(set(all_shapes)) == 1
        use_fast_batch = (self.rank_mode == 'drop' and all_equal)
        
        for it in range(start_iter, max_iters):
            total_loss = 0.0
            total_volumes = 0
            
            indices = list(range(len(volumes)))
            random.shuffle(indices)
            
            for batch_start in range(0, len(indices), batch_size):
                batch_indices = indices[batch_start:batch_start + batch_size]
                batch_volumes = [volumes[idx] for idx in batch_indices]
                batch_targets = [t_tensors[idx] for idx in batch_indices]
                
                optimizer.zero_grad()
                
                if use_fast_batch:
                    # Check if all volumes in this batch have same shape
                    shapes = [vol.shape for vol in batch_volumes]
                    if len(set(shapes)) == 1:
                        batch_tensor = torch.stack(batch_volumes, dim=0)  # [batch, D, H, W, m]
                        reps = self.batch_represent(batch_tensor)  # [batch, m]
                        targets_tensor = torch.stack(batch_targets, dim=0)  # [batch, m]
                        loss = torch.mean(torch.sum((reps - targets_tensor) ** 2, dim=1))
                        loss.backward()
                        optimizer.step()
                        batch_loss = loss.item() * len(batch_volumes)
                        total_loss += batch_loss
                        total_volumes += len(batch_volumes)
                    else:
                        # Fallback to individual processing
                        batch_loss = 0.0
                        batch_count = 0
                        for vol, target in zip(batch_volumes, batch_targets):
                            vectors, comp_indices = self.extract_vectors(vol)
                            if vectors.shape[0] == 0:
                                continue
                            Nk_batch = self.batch_compute_Nk(comp_indices, vectors)
                            vol_pred = torch.mean(Nk_batch, dim=0)
                            vol_loss = torch.sum((vol_pred - target) ** 2)
                            batch_loss += vol_loss
                            batch_count += 1
                            del Nk_batch, vol_pred, comp_indices
                        if batch_count > 0:
                            batch_loss = batch_loss / batch_count
                            batch_loss.backward()
                            optimizer.step()
                            total_loss += batch_loss.item() * batch_count
                            total_volumes += batch_count
                else:
                    # Original per-volume processing
                    batch_loss = 0.0
                    batch_count = 0
                    for vol, target in zip(batch_volumes, batch_targets):
                        vectors, comp_indices = self.extract_vectors(vol)
                        if vectors.shape[0] == 0:
                            continue
                        Nk_batch = self.batch_compute_Nk(comp_indices, vectors)
                        vol_pred = torch.mean(Nk_batch, dim=0)
                        vol_loss = torch.sum((vol_pred - target) ** 2)
                        batch_loss += vol_loss
                        batch_count += 1
                        del Nk_batch, vol_pred, comp_indices
                    if batch_count > 0:
                        batch_loss = batch_loss / batch_count
                        batch_loss.backward()
                        optimizer.step()
                        total_loss += batch_loss.item() * batch_count
                        total_volumes += batch_count
                
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
        
        self._compute_training_statistics(volumes)
        self.trained = True
        return history
    
    def cls_train(self, volumes, labels, num_classes, max_iters=1000, tol=1e-8, learning_rate=0.01,
                  continued=False, decay_rate=1.0, print_every=10, batch_size=32,
                  checkpoint_file=None, checkpoint_interval=10):
        """Multi-class classification training."""
        if self.classifier is None or self.num_classes != num_classes:
            self.classifier = nn.Linear(self.m, num_classes).to(self.device)
            self.num_classes = num_classes
        
        if not continued:
            self.reset_parameters()
        
        label_tensors = torch.tensor(labels, dtype=torch.long, device=self.device)
        volumes = [vol.to(self.device) if isinstance(vol, torch.Tensor) else
                   torch.tensor(vol, dtype=torch.float32, device=self.device)
                   for vol in volumes]
        
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)
        criterion = nn.CrossEntropyLoss()
        
        history = []
        prev_loss = float('inf')
        best_loss = float('inf')
        best_model_state = None
        
        # Fast batch processing possible when all volumes have identical dimensions and rank_mode == 'drop'
        all_shapes = [vol.shape for vol in volumes]
        all_equal = len(set(all_shapes)) == 1
        use_fast_batch = (self.rank_mode == 'drop' and all_equal)
        
        for it in range(max_iters):
            total_loss = 0.0
            total_volumes = 0
            correct = 0
            
            indices = list(range(len(volumes)))
            random.shuffle(indices)
            
            for batch_start in range(0, len(indices), batch_size):
                batch_indices = indices[batch_start:batch_start + batch_size]
                batch_volumes = [volumes[idx] for idx in batch_indices]
                batch_labels = label_tensors[batch_indices]
                
                optimizer.zero_grad()
                
                if use_fast_batch:
                    shapes = [vol.shape for vol in batch_volumes]
                    if len(set(shapes)) == 1:
                        batch_tensor = torch.stack(batch_volumes, dim=0)  # [batch, D, H, W, m]
                        reps = self.batch_represent(batch_tensor)  # [batch, m]
                        logits = self.classifier(reps)  # [batch, num_classes]
                        loss = criterion(logits, batch_labels)
                        loss.backward()
                        optimizer.step()
                        
                        total_loss += loss.item() * len(batch_volumes)
                        total_volumes += len(batch_volumes)
                        with torch.no_grad():
                            preds = torch.argmax(logits, dim=1)
                            correct += (preds == batch_labels).sum().item()
                    else:
                        # Fallback to individual processing
                        batch_logits = []
                        for vol in batch_volumes:
                            vectors, comp_indices = self.extract_vectors(vol)
                            if vectors.shape[0] == 0:
                                vol_vec = torch.zeros(self.m, device=self.device)
                            else:
                                Nk_batch = self.batch_compute_Nk(comp_indices, vectors)
                                vol_vec = torch.mean(Nk_batch, dim=0)
                                del Nk_batch, comp_indices
                            logits = self.classifier(vol_vec.unsqueeze(0))
                            batch_logits.append(logits)
                        if batch_logits:
                            all_logits = torch.cat(batch_logits, dim=0)
                            loss = criterion(all_logits, batch_labels)
                            loss.backward()
                            optimizer.step()
                            total_loss += loss.item() * len(batch_volumes)
                            total_volumes += len(batch_volumes)
                            with torch.no_grad():
                                preds = torch.argmax(all_logits, dim=1)
                                correct += (preds == batch_labels).sum().item()
                else:
                    # Original per-volume processing
                    batch_logits = []
                    for vol in batch_volumes:
                        vectors, comp_indices = self.extract_vectors(vol)
                        if vectors.shape[0] == 0:
                            vol_vec = torch.zeros(self.m, device=self.device)
                        else:
                            Nk_batch = self.batch_compute_Nk(comp_indices, vectors)
                            vol_vec = torch.mean(Nk_batch, dim=0)
                            del Nk_batch, comp_indices
                        logits = self.classifier(vol_vec.unsqueeze(0))
                        batch_logits.append(logits)
                    if batch_logits:
                        all_logits = torch.cat(batch_logits, dim=0)
                        loss = criterion(all_logits, batch_labels)
                        loss.backward()
                        optimizer.step()
                        total_loss += loss.item() * len(batch_volumes)
                        total_volumes += len(batch_volumes)
                        with torch.no_grad():
                            preds = torch.argmax(all_logits, dim=1)
                            correct += (preds == batch_labels).sum().item()
                
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
        """Multi-label classification training."""
        if self.labeller is None or self.num_labels != num_labels:
            self.labeller = nn.Linear(self.m, num_labels).to(self.device)
            self.num_labels = num_labels
        
        if not continued:
            self.reset_parameters()
        
        if isinstance(labels, list):
            labels_tensor = torch.tensor(labels, dtype=torch.float32, device=self.device)
        else:
            labels_tensor = torch.as_tensor(labels, dtype=torch.float32, device=self.device)
        
        volumes = [vol.to(self.device) if isinstance(vol, torch.Tensor) else
                   torch.tensor(vol, dtype=torch.float32, device=self.device)
                   for vol in volumes]
        
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
        
        # Fast batch processing possible when all volumes have identical dimensions and rank_mode == 'drop'
        all_shapes = [vol.shape for vol in volumes]
        all_equal = len(set(all_shapes)) == 1
        use_fast_batch = (self.rank_mode == 'drop' and all_equal)
        
        for it in range(max_iters):
            total_loss = 0.0
            total_correct = 0
            total_preds = 0
            total_volumes = 0
            
            indices = list(range(len(volumes)))
            random.shuffle(indices)
            
            for batch_start in range(0, len(indices), batch_size):
                batch_indices = indices[batch_start:batch_start + batch_size]
                batch_volumes = [volumes[idx] for idx in batch_indices]
                batch_labels = labels_tensor[batch_indices]
                
                optimizer.zero_grad()
                
                if use_fast_batch:
                    shapes = [vol.shape for vol in batch_volumes]
                    if len(set(shapes)) == 1:
                        batch_tensor = torch.stack(batch_volumes, dim=0)  # [batch, D, H, W, m]
                        reps = self.batch_represent(batch_tensor)  # [batch, m]
                        logits = self.labeller(reps)  # [batch, num_labels]
                        loss = criterion(logits, batch_labels)
                        loss.backward()
                        optimizer.step()
                        
                        total_loss += loss.item() * len(batch_volumes)
                        total_volumes += len(batch_volumes)
                        with torch.no_grad():
                            probs = torch.sigmoid(logits)
                            preds = (probs > 0.5).float()
                            total_correct += (preds == batch_labels).sum().item()
                            total_preds += batch_labels.numel()
                    else:
                        # Fallback to individual processing
                        batch_logits = []
                        for vol in batch_volumes:
                            vectors, comp_indices = self.extract_vectors(vol)
                            if vectors.shape[0] == 0:
                                continue
                            Nk_batch = self.batch_compute_Nk(comp_indices, vectors)
                            vol_repr = torch.mean(Nk_batch, dim=0)
                            logits = self.labeller(vol_repr)
                            batch_logits.append(logits)
                            del Nk_batch, vol_repr, comp_indices
                        if batch_logits:
                            all_logits = torch.stack(batch_logits, dim=0)
                            loss = criterion(all_logits, batch_labels)
                            loss.backward()
                            optimizer.step()
                            total_loss += loss.item() * len(batch_logits)
                            total_volumes += len(batch_logits)
                            with torch.no_grad():
                                probs = torch.sigmoid(all_logits)
                                preds = (probs > 0.5).float()
                                total_correct += (preds == batch_labels).sum().item()
                                total_preds += batch_labels.numel()
                else:
                    # Original per-volume processing
                    batch_logits = []
                    for vol in batch_volumes:
                        vectors, comp_indices = self.extract_vectors(vol)
                        if vectors.shape[0] == 0:
                            continue
                        Nk_batch = self.batch_compute_Nk(comp_indices, vectors)
                        vol_repr = torch.mean(Nk_batch, dim=0)
                        logits = self.labeller(vol_repr)
                        batch_logits.append(logits)
                        del Nk_batch, vol_repr, comp_indices
                    if batch_logits:
                        all_logits = torch.stack(batch_logits, dim=0)
                        loss = criterion(all_logits, batch_labels)
                        loss.backward()
                        optimizer.step()
                        total_loss += loss.item() * len(batch_logits)
                        total_volumes += len(batch_logits)
                        with torch.no_grad():
                            probs = torch.sigmoid(all_logits)
                            preds = (probs > 0.5).float()
                            total_correct += (preds == batch_labels).sum().item()
                            total_preds += batch_labels.numel()
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            avg_loss = total_loss / total_volumes if total_volumes else 0.0
            avg_acc = total_correct / total_preds if total_preds > 0 else 0.0
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
            
            if abs(prev_loss - avg_loss) < tol:
                print(f"Converged after {it+1} iterations.")
                if best_model_state is not None:
                    self.load_state_dict(best_model_state)
                break
            prev_loss = avg_loss
            scheduler.step()
        
        self.trained = True
        return loss_history, acc_history
    
    def self_train(self, volumes, max_iters=1000, tol=1e-8, learning_rate=0.01,
                   continued=False, decay_rate=1.0, print_every=10, batch_size=1024,
                   checkpoint_file=None, checkpoint_interval=10):
        """Self-supervised training using gap-filling objective, with memory-optimized batch processing."""
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
        
        volumes = [vol.to(self.device) if isinstance(vol, torch.Tensor) else
                   torch.tensor(vol, dtype=torch.float32, device=self.device)
                   for vol in volumes]
        
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)
        
        if continued and checkpoint_file and os.path.exists(checkpoint_file):
            optimizer.load_state_dict(optimizer_state)
            scheduler.load_state_dict(scheduler_state)
        
        prev_loss = float('inf') if start_iter == 0 else history[-1] if history else float('inf')
        
        # Fast batch processing possible when all volumes have identical dimensions and rank_mode == 'drop'
        all_shapes = [vol.shape for vol in volumes]
        all_equal = len(set(all_shapes)) == 1
        use_fast_batch = (self.rank_mode == 'drop' and all_equal)
        
        if use_fast_batch:
            # Fast path: all volumes same size, use batch_compute_Nk_and_targets
            D, H, W, m = volumes[0].shape
            num_vols = len(volumes)
            all_vols = torch.stack(volumes, dim=0)  # [num_vols, D, H, W, m]
            vol_batch_size = batch_size  # number of volumes per gradient step
            
            for it in range(start_iter, max_iters):
                total_loss = 0.0
                total_windows = 0
                
                indices = list(range(num_vols))
                random.shuffle(indices)
                
                for batch_start in range(0, num_vols, vol_batch_size):
                    batch_indices = indices[batch_start:batch_start + vol_batch_size]
                    batch_vols = all_vols[batch_indices]  # [batch, D, H, W, m]
                    
                    optimizer.zero_grad()
                    Nk_batch, targets_batch = self.batch_compute_Nk_and_targets(batch_vols)  # both [batch, n_windows, m]
                    loss = torch.mean(torch.sum((Nk_batch - targets_batch) ** 2, dim=-1))
                    loss.backward()
                    optimizer.step()
                    
                    batch_loss = loss.item() * batch_vols.shape[0] * Nk_batch.shape[1]
                    total_loss += batch_loss
                    total_windows += batch_vols.shape[0] * Nk_batch.shape[1]
                    
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                avg_loss = total_loss / total_windows if total_windows > 0 else 0.0
                history.append(avg_loss)
                
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    best_model_state = copy.deepcopy(self.state_dict())
                
                if it % print_every == 0 or it == max_iters - 1:
                    current_lr = scheduler.get_last_lr()[0]
                    print(f"Self-Train Iter {it:3d}: loss = {avg_loss:.6f}, LR = {current_lr:.6f}, Windows = {total_windows}")
                
                if checkpoint_file and (it % checkpoint_interval == 0 or it == max_iters - 1):
                    self._save_checkpoint(checkpoint_file, it, history, optimizer, scheduler, best_loss)
                
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
            # Slow path: original per-volume, per-window processing
            sample_batch_size = batch_size  # number of windows per gradient step
            for it in range(start_iter, max_iters):
                total_loss = 0.0
                total_samples = 0
                
                indices = list(range(len(volumes)))
                random.shuffle(indices)
                
                for vol_idx in indices:
                    vol = volumes[vol_idx]
                    vectors, comp_indices = self.extract_vectors(vol)
                    if vectors.shape[0] == 0:
                        continue
                    
                    samples = [(comp_indices[i].item(), vectors[i]) for i in range(vectors.shape[0])]
                    if not samples:
                        continue
                    
                    for batch_start in range(0, len(samples), sample_batch_size):
                        batch = samples[batch_start:batch_start + sample_batch_size]
                        optimizer.zero_grad()
                        
                        k_list = [s[0] for s in batch]
                        v_list = [s[1] for s in batch]
                        k_tensor = torch.tensor(k_list, dtype=torch.long, device=self.device)
                        v_tensor = torch.stack(v_list)
                        
                        Nk_batch = self.batch_compute_Nk(k_tensor, v_tensor)
                        targets = self.map_vector(v_tensor)
                        loss = torch.mean(torch.sum((Nk_batch - targets) ** 2, dim=1))
                        loss.backward()
                        optimizer.step()
                        
                        batch_loss = loss.item() * len(batch)
                        total_loss += batch_loss
                        total_samples += len(batch)
                        
                        del k_tensor, v_tensor, Nk_batch, targets, loss
                        if total_samples % 1000 == 0 and torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    
                    del samples
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                avg_loss = total_loss / total_samples if total_samples else 0.0
                history.append(avg_loss)
                
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    best_model_state = copy.deepcopy(self.state_dict())
                
                if it % print_every == 0 or it == max_iters - 1:
                    current_lr = scheduler.get_last_lr()[0]
                    print(f"Self-Train Iter {it:3d}: loss = {avg_loss:.6f}, LR = {current_lr:.6f}, Samples = {total_samples}")
                
                if checkpoint_file and (it % checkpoint_interval == 0 or it == max_iters - 1):
                    self._save_checkpoint(checkpoint_file, it, history, optimizer, scheduler, best_loss)
                
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
        
        self._compute_training_statistics(volumes)
        self.trained = True
        return history
    
    def _save_checkpoint(self, checkpoint_file, iteration, history, optimizer, scheduler, best_loss):
        """Save training checkpoint."""
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
                'rank': self.rank,
                'rank_op': self.rank_op,
                'rank_mode': self.rank_mode,
                'mode': self.mode,
                'user_step': self.step
            },
            'training_stats': {
                'mean_t': self.mean_t if hasattr(self, 'mean_t') else None,
                'mean_window_count': self.mean_window_count if hasattr(self, 'mean_window_count') else None
            }
        }
        torch.save(checkpoint, checkpoint_file)
        print(f"Checkpoint saved at iteration {iteration} to {checkpoint_file}")
    
    def _compute_training_statistics(self, volumes, batch_size=50):
        """Compute and store statistics for reconstruction and generation."""
        total_window_count = 0
        total_t = torch.zeros(self.m, device=self.device)
        
        with torch.no_grad():
            for i in range(0, len(volumes), batch_size):
                batch_vols = volumes[i:i+batch_size]
                batch_window_count = 0
                batch_t_sum = torch.zeros(self.m, device=self.device)
                
                for vol in batch_vols:
                    vectors, comp_indices = self.extract_vectors(vol)
                    batch_window_count += vectors.shape[0]
                    if vectors.shape[0] > 0:
                        Nk_batch = self.batch_compute_Nk(comp_indices, vectors)
                        batch_t_sum += Nk_batch.sum(dim=0)
                        del comp_indices, Nk_batch
                
                total_window_count += batch_window_count
                total_t += batch_t_sum
                del batch_t_sum
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        self.mean_window_count = total_window_count / len(volumes) if volumes else 0
        self.mean_t = (total_t / total_window_count).cpu().numpy() if total_window_count > 0 else np.zeros(self.m)
    
    def predict_t(self, volume):
        """Predict target vector as average of N(k1,k2,k3) vectors over all windows."""
        vectors, comp_indices = self.extract_vectors(volume)
        if vectors.shape[0] == 0:
            return np.zeros(self.m)
        Nk = self.batch_compute_Nk(comp_indices, vectors)
        return torch.mean(Nk, dim=0).detach().cpu().numpy()
    
    def predict_c(self, volume):
        """Predict class label for a volume."""
        if self.classifier is None:
            raise ValueError("Model must be trained first for classification")
        vol_vector = self.predict_t(volume)
        vol_vector_tensor = torch.tensor(vol_vector, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits = self.classifier(vol_vector_tensor.unsqueeze(0))
            probabilities = torch.softmax(logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
        return predicted_class, probabilities[0].cpu().numpy()
    
    def predict_l(self, volume, threshold=0.5):
        """Predict multi-label classification for a volume."""
        assert self.labeller is not None, "Model must be trained first for label prediction"
        vectors, comp_indices = self.extract_vectors(volume)
        if vectors.shape[0] == 0:
            return np.zeros(self.num_labels, dtype=np.float32), np.zeros(self.num_labels, dtype=np.float32)
        Nk_batch = self.batch_compute_Nk(comp_indices, vectors)
        vol_repr = torch.mean(Nk_batch, dim=0)
        with torch.no_grad():
            logits = self.labeller(vol_repr)
            probs = torch.sigmoid(logits).cpu().numpy()
        binary_preds = (probs > threshold).astype(np.float32)
        return binary_preds, probs
    
    def reconstruct(self, depth, height, width, tau=0.0, num_candidates=1000):
        """
        Reconstruct representative 3D vector volume of size depth×height×width.
        
        Args:
            depth (int): Number of slices in depth dimension
            height (int): Number of rows
            width (int): Number of columns
            tau (float): Temperature parameter (0: deterministic, >0: stochastic)
            num_candidates (int): Number of candidate vectors to consider
            
        Returns:
            numpy.ndarray: Reconstructed volume of shape [depth, height, width, m]
        """
        assert self.trained, "Model must be trained first"
        if tau < 0:
            raise ValueError("Temperature must be non-negative")
            
        mean_t_tensor = torch.tensor(self.mean_t, dtype=torch.float32, device=self.device)
        generated_volume = []
        
        # Generate random candidate vectors
        candidate_vectors = torch.randn(num_candidates, self.m, device=self.device)
        
        for i in range(depth):
            plane = []
            for j in range(height):
                row = []
                for k in range(width):
                    comp_k = self._get_composite_index(i, j, k)
                    k_tensor = torch.tensor([comp_k] * num_candidates, dtype=torch.long, device=self.device)
                    Nk_all = self.batch_compute_Nk(k_tensor, candidate_vectors)
                    
                    errors = torch.sum((Nk_all - mean_t_tensor) ** 2, dim=1)
                    scores = -errors
                    
                    if tau == 0:
                        max_idx = torch.argmax(scores).item()
                        best_vec = candidate_vectors[max_idx].cpu().numpy()
                        row.append(best_vec)
                    else:
                        probs = torch.softmax(scores / tau, dim=0).detach().cpu().numpy()
                        chosen_idx = random.choices(range(num_candidates), weights=probs, k=1)[0]
                        chosen_vec = candidate_vectors[chosen_idx].cpu().numpy()
                        row.append(chosen_vec)
                plane.append(row)
            generated_volume.append(plane)
        return np.stack(generated_volume)
    
    def save(self, filename):
        """Save model state to file."""
        save_dict = {
            'state_dict': self.state_dict(),
            'mean_t': self.mean_t if hasattr(self, 'mean_t') else None,
            'mean_window_count': self.mean_window_count if hasattr(self, 'mean_window_count') else None,
            'trained': self.trained,
            'num_classes': self.num_classes,
            'num_labels': self.num_labels,
            'rank_op': self.rank_op,
            'L1': self.L1,
            'L2': self.L2,
            'L3': self.L3
        }
        torch.save(save_dict, filename)
        print(f"Model saved to {filename}")
    
    def load(self, filename):
        """Load model state from file."""
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
        self.rank_op = save_dict.get('rank_op', 'avg')
        
        if self.num_classes is not None:
            self.classifier = nn.Linear(self.m, self.num_classes).to(self.device)
        if self.num_labels is not None:
            self.labeller = nn.Linear(self.m, self.num_labels).to(self.device)
            
        print(f"Model loaded from {filename}")
        return self


# ================================================
# Example Usage with 3D Vector Volumes
# ================================================
if __name__ == "__main__":
    from statistics import correlation
    
    print("="*60)
    print("SpatialNumDualDescriptorRN3 - PyTorch GPU Accelerated Version (3D)")
    print("Optimized for 3D Vector Volumes with Batch Acceleration")
    print("="*60)
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    
    # Parameters
    vec_dim = 3
    bas_dim1 = 3
    bas_dim2 = 3
    bas_dim3 = 3
    rank = 2                  # window size 2x2x2
    volume_num = 300
    volume_shape_range = ((5, 10), (5, 10), (5, 10))  # (min_D, max_D), (min_H, max_H), (min_W, max_W)
    
    # ------------------------------------------------------------------
    # Example 1: Regression Task
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("Example 1: Regression Task")
    print("="*60)
    
    # Generate random 3D volumes and target vectors
    volumes, t_list = [], []
    for _ in range(volume_num):
        D = np.random.randint(volume_shape_range[0][0], volume_shape_range[0][1])
        H = np.random.randint(volume_shape_range[1][0], volume_shape_range[1][1])
        W = np.random.randint(volume_shape_range[2][0], volume_shape_range[2][1])
        volume = np.random.randn(D, H, W, vec_dim).astype(np.float32)
        volumes.append(volume)
        t_list.append(np.random.uniform(-1, 1, vec_dim).astype(np.float32))
    
    model = SpatialNumDualDescriptorRN3(
        vec_dim=vec_dim,
        bas_dim1=bas_dim1,
        bas_dim2=bas_dim2,
        bas_dim3=bas_dim3,
        rank=rank,
        rank_mode='drop',
        mode='linear',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    print(f"\nUsing device: {model.device}")
    print(f"Vector dimension: {model.m}")
    print(f"Basis dimensions: {model.L1} × {model.L2} × {model.L3}")
    print(f"Window size: {model.rank}×{model.rank}×{model.rank}")
    
    # Train regression
    print("\nStarting gradient descent training...")
    history = model.reg_train(
        volumes, t_list,
        learning_rate=0.05,
        max_iters=30,
        tol=1e-6,
        decay_rate=0.99,
        print_every=10,
        batch_size=8
    )
    
    # Predict on first volume
    test_vol = volumes[0]
    t_pred = model.predict_t(test_vol)
    print(f"\nPredicted t for first volume: {t_pred}")
    print(f"Actual t: {t_list[0]}")
    print(f"Mean squared error: {np.mean((t_pred - t_list[0]) ** 2):.6f}")
    
    # Correlation across all volumes
    pred_t_list = [model.predict_t(v) for v in volumes]
    corr_sum = 0.0
    for i in range(model.m):
        actu_t = [t[i] for t in t_list]
        pred_t = [p[i] for p in pred_t_list]
        corr = correlation(actu_t, pred_t)
        print(f"Prediction correlation for dimension {i}: {corr:.4f}")
        corr_sum += corr
    print(f"Average correlation: {corr_sum / model.m:.4f}")
    
    # Reconstruction
    print("\nReconstructing representative 3D volume (4×4×4)...")
    recon_vol = model.reconstruct(depth=4, height=4, width=4, tau=0.0)
    print(f"Reconstructed volume shape: {recon_vol.shape}")
    print("First slice (depth=0) of first channel:")
    print(recon_vol[0, :, :, 0])
    
    # ------------------------------------------------------------------
    # Example 2: Classification Task
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("Example 2: Classification Task")
    print("="*60)
    
    num_classes = 3
    class_volumes = []
    class_labels = []
    
    for class_id in range(num_classes):
        for _ in range(20):
            D = np.random.randint(6, 10)
            H = np.random.randint(6, 10)
            W = np.random.randint(6, 10)
            if class_id == 0:
                vol = np.random.normal(loc=1.0, scale=0.2, size=(D, H, W, vec_dim)).astype(np.float32)
            elif class_id == 1:
                vol = np.random.normal(loc=-1.0, scale=0.5, size=(D, H, W, vec_dim)).astype(np.float32)
            else:
                vol = np.random.normal(loc=0.0, scale=1.0, size=(D, H, W, vec_dim)).astype(np.float32)
            class_volumes.append(vol)
            class_labels.append(class_id)
    
    model_cls = SpatialNumDualDescriptorRN3(
        vec_dim=vec_dim, bas_dim1=bas_dim1, bas_dim2=bas_dim2, bas_dim3=bas_dim3,
        rank=rank, mode='linear', device=model.device
    )
    
    print("\nStarting classification training...")
    cls_history = model_cls.cls_train(
        class_volumes, class_labels, num_classes,
        max_iters=20, tol=1e-6, learning_rate=0.05,
        decay_rate=0.99, batch_size=8, print_every=5
    )
    
    # Evaluate
    correct = 0
    for v, true_lbl in zip(class_volumes, class_labels):
        pred, _ = model_cls.predict_c(v)
        if pred == true_lbl:
            correct += 1
    accuracy = correct / len(class_volumes)
    print(f"\nClassification accuracy: {accuracy:.4f} ({correct}/{len(class_volumes)})")
    
    # ------------------------------------------------------------------
    # Example 3: Multi-Label Classification
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("Example 3: Multi-Label Classification")
    print("="*60)
    
    num_labels = 3
    ml_volumes = []
    ml_labels = []
    for _ in range(30):
        D = np.random.randint(5, 10)
        H = np.random.randint(5, 10)
        W = np.random.randint(5, 10)
        vol = np.random.randn(D, H, W, vec_dim).astype(np.float32)
        ml_volumes.append(vol)
        label_vec = [1.0 if np.random.random() > 0.7 else 0.0 for _ in range(num_labels)]
        ml_labels.append(label_vec)
    
    model_lbl = SpatialNumDualDescriptorRN3(
        vec_dim=vec_dim, bas_dim1=bas_dim1, bas_dim2=bas_dim2, bas_dim3=bas_dim3,
        rank=rank, mode='linear', device=model.device
    )
    
    print("\nStarting multi-label training...")
    loss_hist, acc_hist = model_lbl.lbl_train(
        ml_volumes, ml_labels, num_labels,
        max_iters=15, tol=1e-6, learning_rate=0.05,
        decay_rate=0.99, batch_size=8, print_every=5
    )
    print(f"Final loss: {loss_hist[-1]:.6f}, Final accuracy: {acc_hist[-1]:.4f}")
    
    # ------------------------------------------------------------------
    # Example 4: Self-Training
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("Example 4: Self-Training")
    print("="*60)
    
    self_volumes = []
    for _ in range(15):
        D = np.random.randint(6, 10)
        H = np.random.randint(6, 10)
        W = np.random.randint(6, 10)
        vol = np.random.randn(D, H, W, vec_dim).astype(np.float32)
        self_volumes.append(vol)
    
    model_self = SpatialNumDualDescriptorRN3(
        vec_dim=vec_dim, bas_dim1=bas_dim1, bas_dim2=bas_dim2, bas_dim3=bas_dim3,
        rank=rank, mode='linear', device=model.device
    )
    
    print("\nStarting self-training...")
    self_history = model_self.self_train(
        self_volumes, max_iters=20, learning_rate=0.01,
        decay_rate=0.995, batch_size=512, print_every=5
    )
    
    # Reconstruction after self-training
    recon_self = model_self.reconstruct(depth=3, height=3, width=3, tau=0.0)
    print(f"\nSelf-trained reconstruction (3×3×3) shape: {recon_self.shape}")
    
    # ------------------------------------------------------------------
    # Example 5: Model Persistence
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("Example 5: Model Persistence")
    print("="*60)
    
    model.save("spatial_dual_descriptor_3d.pt")
    model_loaded = SpatialNumDualDescriptorRN3(
        vec_dim=vec_dim, bas_dim1=bas_dim1, bas_dim2=bas_dim2, bas_dim3=bas_dim3,
        rank=rank, mode='linear', device=model.device
    )
    model_loaded.load("spatial_dual_descriptor_3d.pt")
    
    test_pred_orig = model.predict_t(test_vol)
    test_pred_load = model_loaded.predict_t(test_vol)
    print(f"Original prediction: {test_pred_orig}")
    print(f"Loaded prediction: {test_pred_load}")
    print(f"Match: {np.allclose(test_pred_orig, test_pred_load, rtol=1e-6)}")
    
    print("\n" + "="*60)
    print("All examples completed successfully!")
    print("="*60)
