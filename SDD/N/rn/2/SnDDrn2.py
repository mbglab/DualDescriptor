# Copyright (C) 2005-2026, Bin-Guang Ma (mbg@mail.hzau.edu.cn); SPDX-License-Identifier: MIT
# The Spatial Numeric Dual Descriptor Vector class (Random AB matrix form) for 2D array implemented with PyTorch
# This program is for the demonstration of methodology and not fully refined.
# Author: Bin-Guang Ma (assisted by DeepSeek); Date: 2025-8-28 ~ 2026-04-03

import math
import random
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import copy
import os

class SpatialNumDualDescriptorRN2(nn.Module):
    """
    Dual Descriptor for 2D vector grids with GPU acceleration using PyTorch:
      - Learnable coefficient matrix Acoeff ∈ R^{m×(L1·L2)}
      - Learnable basis matrix Bbasis ∈ R^{(L1·L2)×m}
      - Learnable mapping matrix M ∈ R^{m×m}
      - Input: 2D grids of m-dimensional vectors (H × W × m)
      - Optimized with batch processing for GPU acceleration
      - Supports both linear and nonlinear 2D windowing
      - Supports regression, classification, and multi-label classification tasks
      - Added equal-length batch acceleration for linear mode with avg/sum aggregation
    """
    def __init__(self, vec_dim=4, bas_dim1=5, bas_dim2=5, rank=1, rank_op='avg', rank_mode='drop',
                 mode='linear', user_step=None, device='cuda'):
        """
        Initialize the Spatial Dual Descriptor for 2D vector grids.
        
        Args:
            vec_dim (int): Dimension m of input vectors
            bas_dim1 (int): Basis dimension for first index (rows)
            bas_dim2 (int): Basis dimension for second index (cols)
            rank (int): Window size (square window, side length) for vector aggregation
            rank_op (str): 'avg', 'sum', 'pick', or 'user_func'
            rank_mode (str): 'pad' or 'drop' for handling incomplete windows
            mode (str): 'linear' (sliding window, step=1) or 'nonlinear' (stepped window)
            user_step (int): Custom step size for nonlinear mode (same for both dimensions)
            device (str): 'cuda' or 'cpu'
        """
        super().__init__()
        self.m = vec_dim                     # Vector dimension
        self.L1 = bas_dim1                   # Basis dimension for rows
        self.L2 = bas_dim2                   # Basis dimension for cols
        self.total_basis = self.L1 * self.L2 # Total number of basis functions
        self.rank = rank                     # Window size (square)
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
    
    def _get_composite_index(self, k1, k2):
        """
        Convert 2D position indices to a composite index that respects 2D periodicity.
        Composite index = k1 * L2 + k2, then modulo total_basis yields (k1 % L1, k2 % L2).
        """
        return k1 * self.L2 + k2
    
    def _compute_basis_index(self, composite_k):
        """
        Compute basis index j = composite_k % total_basis.
        """
        return (composite_k % self.total_basis).long()
    
    def extract_vectors(self, grid):
        """
        Extract vector groups from a 2D vector grid using sliding windows and return
        aggregated vectors along with their composite position indices.
        
        The grid is expected to have shape (H, W, m) or (H, W, ...) with last dim = m.
        Windows are square of size rank × rank. They are extracted in row-major order.
        
        - 'linear': slide window by 1 step in both directions
        - 'nonlinear': slide window by step (given or rank) in both directions
        
        For incomplete windows at the boundaries, handles them using:
        - 'pad': pads with zero vectors
        - 'drop': discards incomplete windows
        
        Returns:
            tuple: (vectors_tensor, composite_indices_tensor)
                vectors_tensor: [num_windows, m] aggregated vectors (NOT mapped by M)
                composite_indices_tensor: [num_windows] composite indices
        """
        # Convert to tensor if needed, ensure shape (H, W, m)
        if not isinstance(grid, torch.Tensor):
            grid = torch.tensor(np.array(grid), dtype=torch.float32, device=self.device)
        # If grid is 2D (H, W) or 3D (H, W, something), ensure last dim is m
        if grid.dim() == 2:
            grid = grid.unsqueeze(-1)  # (H, W, 1)
        if grid.dim() == 3 and grid.shape[-1] != self.m:
            # Possibly grid is (H, W, ...) but not m? assume it's fine
            pass
        
        H, W = grid.shape[0], grid.shape[1]
        step = self.step
        
        # ========== Vectorized path for linear mode with avg/sum ==========
        if self.mode == 'linear' and self.rank_op in ('avg', 'sum'):
            # Check if grid is large enough
            if H < self.rank or W < self.rank:
                return torch.empty(0, self.m, device=self.device), torch.empty(0, dtype=torch.float32, device=self.device)
            
            # Reshape grid to (1, m, H, W) for unfold
            grid_perm = grid.permute(2, 0, 1).unsqueeze(0)  # (1, m, H, W)
            # Unfold: extract all rank×rank windows, stride=1
            # Output shape: (1, m*rank*rank, num_windows)
            windows = F.unfold(grid_perm, kernel_size=(self.rank, self.rank), stride=1)
            num_windows = windows.shape[2]
            # Reshape to (num_windows, rank, rank, m)
            windows = windows.view(1, self.m, self.rank, self.rank, num_windows).squeeze(0)  # (m, rank, rank, num_windows)
            windows = windows.permute(3, 1, 2, 0)  # (num_windows, rank, rank, m)
            # Aggregate over window positions
            if self.rank_op == 'avg':
                agg = windows.mean(dim=(1, 2))  # (num_windows, m)
            else:  # sum
                agg = windows.sum(dim=(1, 2))   # (num_windows, m)
            
            # Generate composite indices for each window (top-left corner)
            # For linear mode, windows are in row-major order, top-left (i,j)
            # i = row index, j = col index; i from 0 to H-rank, j from 0 to W-rank
            # The order from unfold is row-major: first all windows for i=0, then i=1,...
            i_indices = torch.arange(H - self.rank + 1, device=self.device).repeat_interleave(W - self.rank + 1)
            j_indices = torch.arange(W - self.rank + 1, device=self.device).repeat(H - self.rank + 1)
            comp_indices = self._get_composite_index(i_indices, j_indices)
            
            # Return aggregated vectors (not mapped by M) and composite indices
            return agg, comp_indices
        
        # ========== Fallback loop-based extraction for other modes ==========
        # Helper for aggregation
        def apply_op(vectors):
            # vectors: (rank*rank, m)
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
        
        windows = []
        composite_indices = []
        
        if self.mode == 'linear':
            # Slide step=1
            for i in range(H - self.rank + 1):
                for j in range(W - self.rank + 1):
                    window_vectors = grid[i:i+self.rank, j:j+self.rank].reshape(-1, self.m)
                    agg_vec = apply_op(window_vectors)
                    windows.append(agg_vec)
                    composite_indices.append(self._get_composite_index(i, j))
        else:
            # Nonlinear mode with step
            for i in range(0, H, step):
                for j in range(0, W, step):
                    i_end = i + self.rank
                    j_end = j + self.rank
                    if i_end > H or j_end > W:
                        if self.rank_mode == 'pad':
                            # Pad missing parts with zeros
                            window_vectors_list = []
                            for ii in range(i, min(i_end, H)):
                                for jj in range(j, min(j_end, W)):
                                    window_vectors_list.append(grid[ii, jj])
                            pad_count = self.rank*self.rank - len(window_vectors_list)
                            if pad_count > 0:
                                zero_pad = torch.zeros(pad_count, self.m, device=self.device)
                                window_vectors = torch.cat([torch.stack(window_vectors_list), zero_pad])
                            else:
                                window_vectors = torch.stack(window_vectors_list)
                        else:
                            continue
                    else:
                        window_vectors = grid[i:i+self.rank, j:j+self.rank].reshape(-1, self.m)
                    agg_vec = apply_op(window_vectors)
                    windows.append(agg_vec)
                    composite_indices.append(self._get_composite_index(i, j))
        
        if windows:
            return torch.stack(windows), torch.tensor(composite_indices, dtype=torch.float32, device=self.device)
        else:
            return torch.empty(0, self.m, device=self.device), torch.empty(0, dtype=torch.float32, device=self.device)
    
    def _batch_extract_vectors_raw(self, grid_batch):
        """
        Vectorized extraction of raw (non-mapped) aggregated window vectors for a batch of equal-sized grids.
        Only works for linear mode and rank_op in ('avg', 'sum').
        
        Args:
            grid_batch: torch.Tensor of shape [batch_size, H, W, m]
            
        Returns:
            agg_raw_flat: torch.Tensor of shape [total_windows, m] (raw aggregated, no M applied)
            comp_indices_flat: torch.Tensor of shape [total_windows] (composite indices)
        """
        B, H, W, m = grid_batch.shape
        if H < self.rank or W < self.rank:
            return torch.empty(0, m, device=self.device), torch.empty(0, device=self.device)
        
        # Reshape to [B, m, H, W] for unfold
        grid_perm = grid_batch.permute(0, 3, 1, 2)  # [B, m, H, W]
        # Unfold: extract all rank×rank windows, stride=1
        windows = F.unfold(grid_perm, kernel_size=(self.rank, self.rank), stride=1)  # [B, m*rank*rank, N_w]
        N_w = windows.shape[2]  # number of windows per grid
        # Reshape to [B, m, rank, rank, N_w]
        windows = windows.view(B, m, self.rank, self.rank, N_w)
        # Permute to [B, N_w, rank, rank, m]
        windows = windows.permute(0, 4, 2, 3, 1)
        # Aggregate over window positions
        if self.rank_op == 'avg':
            agg_raw = windows.mean(dim=(2, 3))   # [B, N_w, m]
        else:  # sum
            agg_raw = windows.sum(dim=(2, 3))    # [B, N_w, m]
        
        # Flatten across batch and windows
        agg_raw_flat = agg_raw.reshape(-1, m)    # [B*N_w, m]
        
        # Generate composite indices for each window (same for all grids in batch)
        # i indices: 0..(H-rank), j indices: 0..(W-rank), row-major order
        i_indices = torch.arange(H - self.rank + 1, device=self.device).repeat_interleave(W - self.rank + 1)
        j_indices = torch.arange(W - self.rank + 1, device=self.device).repeat(H - self.rank + 1)
        comp_indices_base = self._get_composite_index(i_indices, j_indices)  # [N_w]
        # Repeat for each grid in batch
        comp_indices_flat = comp_indices_base.unsqueeze(0).repeat(B, 1).reshape(-1)  # [B*N_w]
        return agg_raw_flat, comp_indices_flat
    
    def batch_represent(self, grid_batch):
        """
        Compute sequence representations for a batch of equal-sized 2D vector grids.
        This method computes the full N(k) representation (including A and B matrices)
        and then averages over windows to obtain a fixed-size vector per grid.
        
        Args:
            grid_batch: torch.Tensor of shape [batch_size, H, W, m]
        
        Returns:
            reps: torch.Tensor of shape [batch_size, m]
        """
        B, H, W, m = grid_batch.shape
        agg_raw_flat, comp_indices_flat = self._batch_extract_vectors_raw(grid_batch)
        if agg_raw_flat.shape[0] == 0:
            return torch.zeros(B, m, device=self.device)
        
        # Compute Nk for all windows (batch_compute_Nk internally applies M to agg_raw_flat)
        Nk_all = self.batch_compute_Nk(comp_indices_flat, agg_raw_flat)  # [total_w, m]
        
        # Split back to grids and average
        N_w = agg_raw_flat.shape[0] // B
        Nk_grid = Nk_all.view(B, N_w, m)   # [B, N_w, m]
        reps = Nk_grid.mean(dim=1)          # [B, m]
        return reps
    
    def batch_compute_Nk(self, composite_k_tensor, vectors):
        """
        Vectorized computation of N(k1,k2) vectors for a batch of composite positions and vectors.
        
        Args:
            composite_k_tensor: Tensor of composite indices [batch_size]
            vectors: Tensor of input vectors (aggregated windows) [batch_size, m]
            
        Returns:
            Tensor of N(k1,k2) vectors [batch_size, m]
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
        # Acoeff is [m, total_basis] -> permute to [total_basis, m] then index
        A_j = self.Acoeff.permute(1, 0)[j_indices]
        
        # Compute Nk = scalar * A_j [batch_size, m]
        Nk = scalar * A_j
            
        return Nk

    def describe(self, grid):
        """Compute N(k1,k2) vectors for each window in the grid."""
        vectors, comp_indices = self.extract_vectors(grid)
        if vectors.shape[0] == 0:
            return np.array([])
        N_batch = self.batch_compute_Nk(comp_indices, vectors)
        return N_batch.detach().cpu().numpy()

    def S(self, grid):
        """
        Compute list of cumulative sums S(l) = sum_{i=1..l} N(k_i) for windows in row-major order.
        """
        vectors, comp_indices = self.extract_vectors(grid)
        if vectors.shape[0] == 0:
            return []
        N_batch = self.batch_compute_Nk(comp_indices, vectors)
        S_cum = torch.cumsum(N_batch, dim=0)
        return [s.detach().cpu().numpy() for s in S_cum]

    def D(self, grids, t_list):
        """
        Compute mean squared deviation D across grids:
        D = average over all windows of (N(k1,k2)-t_seq)^2
        """
        total_loss = 0.0
        total_positions = 0
        
        t_tensors = [torch.tensor(t, dtype=torch.float32, device=self.device) for t in t_list]
        
        for grid, t in zip(grids, t_tensors):
            vectors, comp_indices = self.extract_vectors(grid)
            if vectors.shape[0] == 0:
                continue
            N_batch = self.batch_compute_Nk(comp_indices, vectors)
            losses = torch.sum((N_batch - t) ** 2, dim=1)
            total_loss += losses.sum().item()
            total_positions += vectors.shape[0]
        return total_loss / total_positions if total_positions else 0.0

    def d(self, grid, t):
        """Compute pattern deviation value (d) for a single grid."""
        return self.D([grid], [t])

    def reg_train(self, grids, t_list, max_iters=1000, tol=1e-8, learning_rate=0.01,
                  continued=False, decay_rate=1.0, print_every=10, batch_size=32,
                  checkpoint_file=None, checkpoint_interval=10):
        """
        Train model using gradient descent with grid-level batch processing.
        Optimized version: if all grids in a batch have equal size and conditions allow,
        uses vectorized batch_represent; otherwise falls back to per-grid window collection.
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
        # Ensure grids are on correct device
        grids = [grid.to(self.device) if isinstance(grid, torch.Tensor) else
                 torch.tensor(grid, dtype=torch.float32, device=self.device)
                 for grid in grids]
        
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)
        
        if continued and checkpoint_file and os.path.exists(checkpoint_file):
            optimizer.load_state_dict(optimizer_state)
            scheduler.load_state_dict(scheduler_state)
        
        prev_loss = float('inf') if start_iter == 0 else history[-1] if history else float('inf')
        
        for it in range(start_iter, max_iters):
            total_loss = 0.0
            total_grids = 0
            
            indices = list(range(len(grids)))
            random.shuffle(indices)
            
            for batch_start in range(0, len(indices), batch_size):
                batch_indices = indices[batch_start:batch_start + batch_size]
                batch_grids = [grids[idx] for idx in batch_indices]
                batch_targets = [t_tensors[idx] for idx in batch_indices]
                
                optimizer.zero_grad()
                
                # Check if all grids in the batch have the same dimensions
                grid_shapes = [grid.shape for grid in batch_grids]  # each is (H, W, m)
                if len(set(grid_shapes)) == 1 and self.mode == 'linear' and self.rank_op in ('avg', 'sum'):
                    # Equal size and conditions allow: use batch_represent
                    grid_tensor = torch.stack(batch_grids, dim=0)  # [B, H, W, m]
                    reps = self.batch_represent(grid_tensor)      # [B, m]
                    targets_tensor = torch.stack(batch_targets, dim=0)
                    loss = torch.mean(torch.sum((reps - targets_tensor) ** 2, dim=1))
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item() * len(batch_indices)
                    total_grids += len(batch_indices)
                else:
                    # Unequal sizes or unsupported mode: fallback to per-grid window collection
                    all_vectors = []
                    all_comp = []
                    grid_window_counts = []
                    
                    for grid in batch_grids:
                        vectors, comp_indices = self.extract_vectors(grid)
                        if vectors.shape[0] == 0:
                            grid_window_counts.append(0)
                            continue
                        all_vectors.append(vectors)
                        all_comp.append(comp_indices)
                        grid_window_counts.append(vectors.shape[0])
                    
                    if not all_vectors:
                        continue
                    
                    all_vectors = torch.cat(all_vectors, dim=0)
                    all_comp = torch.cat(all_comp, dim=0)
                    
                    Nk_all = self.batch_compute_Nk(all_comp, all_vectors)
                    
                    grid_reps = []
                    start_idx = 0
                    for nw in grid_window_counts:
                        if nw > 0:
                            grid_rep = Nk_all[start_idx:start_idx+nw].mean(dim=0)
                            start_idx += nw
                        else:
                            grid_rep = torch.zeros(self.m, device=self.device)
                        grid_reps.append(grid_rep)
                    
                    if grid_reps:
                        grid_reps_tensor = torch.stack(grid_reps)
                        targets_tensor = torch.stack(batch_targets)
                        loss = torch.mean(torch.sum((grid_reps_tensor - targets_tensor) ** 2, dim=1))
                        loss.backward()
                        optimizer.step()
                        total_loss += loss.item() * len(grid_reps)
                        total_grids += len(grid_reps)
                
                # Clean up
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            avg_loss = total_loss / total_grids if total_grids else 0.0
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
        
        self._compute_training_statistics(grids)
        self.trained = True
        return history

    def cls_train(self, grids, labels, num_classes, max_iters=1000, tol=1e-8, learning_rate=0.01,
                  continued=False, decay_rate=1.0, print_every=10, batch_size=32,
                  checkpoint_file=None, checkpoint_interval=10):
        """Multi-class classification training with batch window processing."""
        if self.classifier is None or self.num_classes != num_classes:
            self.classifier = nn.Linear(self.m, num_classes).to(self.device)
            self.num_classes = num_classes
        
        if not continued:
            self.reset_parameters()
        
        label_tensors = torch.tensor(labels, dtype=torch.long, device=self.device)
        grids = [grid.to(self.device) if isinstance(grid, torch.Tensor) else
                 torch.tensor(grid, dtype=torch.float32, device=self.device)
                 for grid in grids]
        
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)
        criterion = nn.CrossEntropyLoss()
        
        history = []
        prev_loss = float('inf')
        best_loss = float('inf')
        best_model_state = None
        
        for it in range(max_iters):
            total_loss = 0.0
            total_grids = 0
            correct = 0
            
            indices = list(range(len(grids)))
            random.shuffle(indices)
            
            for batch_start in range(0, len(indices), batch_size):
                batch_indices = indices[batch_start:batch_start + batch_size]
                batch_grids = [grids[idx] for idx in batch_indices]
                batch_labels = label_tensors[batch_indices]
                
                optimizer.zero_grad()
                
                # Check if all grids in the batch have the same dimensions
                grid_shapes = [grid.shape for grid in batch_grids]
                if len(set(grid_shapes)) == 1 and self.mode == 'linear' and self.rank_op in ('avg', 'sum'):
                    # Equal size and conditions allow: use batch_represent
                    grid_tensor = torch.stack(batch_grids, dim=0)
                    reps = self.batch_represent(grid_tensor)
                    logits = self.classifier(reps)
                    loss = criterion(logits, batch_labels)
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item() * len(batch_indices)
                    total_grids += len(batch_indices)
                    with torch.no_grad():
                        preds = torch.argmax(logits, dim=1)
                        correct += (preds == batch_labels).sum().item()
                else:
                    # Unequal sizes or unsupported mode: fallback to per-grid window collection
                    all_vectors = []
                    all_comp = []
                    grid_window_counts = []
                    
                    for grid in batch_grids:
                        vectors, comp_indices = self.extract_vectors(grid)
                        if vectors.shape[0] == 0:
                            grid_window_counts.append(0)
                            continue
                        all_vectors.append(vectors)
                        all_comp.append(comp_indices)
                        grid_window_counts.append(vectors.shape[0])
                    
                    if not all_vectors:
                        continue
                    
                    all_vectors = torch.cat(all_vectors, dim=0)
                    all_comp = torch.cat(all_comp, dim=0)
                    Nk_all = self.batch_compute_Nk(all_comp, all_vectors)
                    
                    grid_reps = []
                    start_idx = 0
                    for nw in grid_window_counts:
                        if nw > 0:
                            grid_rep = Nk_all[start_idx:start_idx+nw].mean(dim=0)
                            start_idx += nw
                        else:
                            grid_rep = torch.zeros(self.m, device=self.device)
                        grid_reps.append(grid_rep)
                    
                    if grid_reps:
                        grid_reps_tensor = torch.stack(grid_reps)
                        logits = self.classifier(grid_reps_tensor)
                        loss = criterion(logits, batch_labels)
                        loss.backward()
                        optimizer.step()
                        
                        total_loss += loss.item() * len(grid_reps)
                        total_grids += len(grid_reps)
                        with torch.no_grad():
                            preds = torch.argmax(logits, dim=1)
                            correct += (preds == batch_labels).sum().item()
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            avg_loss = total_loss / total_grids if total_grids else 0.0
            accuracy = correct / total_grids if total_grids else 0.0
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

    def lbl_train(self, grids, labels, num_labels, max_iters=1000, tol=1e-8, learning_rate=0.01,
                  continued=False, decay_rate=1.0, print_every=10, batch_size=32,
                  checkpoint_file=None, checkpoint_interval=10, pos_weight=None):
        """Multi-label classification training with batch window processing."""
        if self.labeller is None or self.num_labels != num_labels:
            self.labeller = nn.Linear(self.m, num_labels).to(self.device)
            self.num_labels = num_labels
        
        if not continued:
            self.reset_parameters()
        
        if isinstance(labels, list):
            labels_tensor = torch.tensor(labels, dtype=torch.float32, device=self.device)
        else:
            labels_tensor = torch.as_tensor(labels, dtype=torch.float32, device=self.device)
        
        grids = [grid.to(self.device) if isinstance(grid, torch.Tensor) else
                 torch.tensor(grid, dtype=torch.float32, device=self.device)
                 for grid in grids]
        
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
            total_preds = 0
            total_grids = 0
            
            indices = list(range(len(grids)))
            random.shuffle(indices)
            
            for batch_start in range(0, len(indices), batch_size):
                batch_indices = indices[batch_start:batch_start + batch_size]
                batch_grids = [grids[idx] for idx in batch_indices]
                batch_labels = labels_tensor[batch_indices]
                
                optimizer.zero_grad()
                
                # Check if all grids in the batch have the same dimensions
                grid_shapes = [grid.shape for grid in batch_grids]
                if len(set(grid_shapes)) == 1 and self.mode == 'linear' and self.rank_op in ('avg', 'sum'):
                    # Equal size and conditions allow: use batch_represent
                    grid_tensor = torch.stack(batch_grids, dim=0)
                    reps = self.batch_represent(grid_tensor)
                    logits = self.labeller(reps)
                    loss = criterion(logits, batch_labels)
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item() * len(batch_indices)
                    total_grids += len(batch_indices)
                    with torch.no_grad():
                        probs = torch.sigmoid(logits)
                        preds = (probs > 0.5).float()
                        total_correct += (preds == batch_labels).sum().item()
                        total_preds += batch_labels.numel()
                else:
                    # Unequal sizes or unsupported mode: fallback to per-grid window collection
                    all_vectors = []
                    all_comp = []
                    grid_window_counts = []
                    
                    for grid in batch_grids:
                        vectors, comp_indices = self.extract_vectors(grid)
                        if vectors.shape[0] == 0:
                            grid_window_counts.append(0)
                            continue
                        all_vectors.append(vectors)
                        all_comp.append(comp_indices)
                        grid_window_counts.append(vectors.shape[0])
                    
                    if not all_vectors:
                        continue
                    
                    all_vectors = torch.cat(all_vectors, dim=0)
                    all_comp = torch.cat(all_comp, dim=0)
                    Nk_all = self.batch_compute_Nk(all_comp, all_vectors)
                    
                    grid_reps = []
                    start_idx = 0
                    for nw in grid_window_counts:
                        if nw > 0:
                            grid_rep = Nk_all[start_idx:start_idx+nw].mean(dim=0)
                            start_idx += nw
                        else:
                            grid_rep = torch.zeros(self.m, device=self.device)
                        grid_reps.append(grid_rep)
                    
                    if grid_reps:
                        grid_reps_tensor = torch.stack(grid_reps)
                        logits = self.labeller(grid_reps_tensor)
                        loss = criterion(logits, batch_labels)
                        loss.backward()
                        optimizer.step()
                        
                        total_loss += loss.item() * len(grid_reps)
                        total_grids += len(grid_reps)
                        with torch.no_grad():
                            probs = torch.sigmoid(logits)
                            preds = (probs > 0.5).float()
                            total_correct += (preds == batch_labels).sum().item()
                            total_preds += batch_labels.numel()
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            avg_loss = total_loss / total_grids if total_grids else 0.0
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

    def self_train(self, grids, max_iters=1000, tol=1e-8, learning_rate=0.01,
                   continued=False, decay_rate=1.0, print_every=10, batch_size=1024,
                   checkpoint_file=None, checkpoint_interval=10):
        """
        Self-supervised training using gap-filling objective.
        Optimized version: if all grids in a batch have equal size and conditions allow,
        uses vectorized batch extraction; otherwise falls back to per-grid processing.
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
        
        grids = [grid.to(self.device) if isinstance(grid, torch.Tensor) else
                 torch.tensor(grid, dtype=torch.float32, device=self.device)
                 for grid in grids]
        
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)
        
        if continued and checkpoint_file and os.path.exists(checkpoint_file):
            optimizer.load_state_dict(optimizer_state)
            scheduler.load_state_dict(scheduler_state)
        
        prev_loss = float('inf') if start_iter == 0 else history[-1] if history else float('inf')
        
        for it in range(start_iter, max_iters):
            total_loss = 0.0
            total_samples = 0
            
            indices = list(range(len(grids)))
            random.shuffle(indices)
            
            for batch_start in range(0, len(indices), batch_size):
                batch_indices = indices[batch_start:batch_start + batch_size]
                batch_grids = [grids[idx] for idx in batch_indices]
                
                optimizer.zero_grad()
                
                # Check if all grids in the batch have the same dimensions
                grid_shapes = [grid.shape for grid in batch_grids]
                if len(set(grid_shapes)) == 1 and self.mode == 'linear' and self.rank_op in ('avg', 'sum'):
                    # Equal size and conditions allow: use vectorized batch extraction
                    grid_tensor = torch.stack(batch_grids, dim=0)  # [B, H, W, m]
                    agg_raw_flat, comp_indices_flat = self._batch_extract_vectors_raw(grid_tensor)
                    if agg_raw_flat.shape[0] == 0:
                        continue
                    # Compute Nk for all windows
                    Nk_all = self.batch_compute_Nk(comp_indices_flat, agg_raw_flat)
                    # Self-supervised target: mapped aggregated vectors (M * raw)
                    targets = self.map_vector(agg_raw_flat)
                    loss = torch.mean(torch.sum((Nk_all - targets) ** 2, dim=1))
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item() * agg_raw_flat.size(0)
                    total_samples += agg_raw_flat.size(0)
                else:
                    # Unequal sizes or unsupported mode: fallback to per-grid processing
                    for grid in batch_grids:
                        vectors, comp_indices = self.extract_vectors(grid)
                        if vectors.shape[0] == 0:
                            continue
                        Nk_batch = self.batch_compute_Nk(comp_indices, vectors)
                        # Self-supervised target: mapped aggregated vectors
                        targets = self.map_vector(vectors)
                        loss = torch.mean(torch.sum((Nk_batch - targets) ** 2, dim=1))
                        loss.backward()
                        optimizer.step()
                        total_loss += loss.item() * vectors.size(0)
                        total_samples += vectors.size(0)
                
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
        
        self._compute_training_statistics(grids)
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

    def _compute_training_statistics(self, grids, batch_size=50):
        """Compute and store statistics for reconstruction and generation."""
        total_window_count = 0
        total_t = torch.zeros(self.m, device=self.device)
        
        with torch.no_grad():
            for i in range(0, len(grids), batch_size):
                batch_grids = grids[i:i+batch_size]
                batch_window_count = 0
                batch_t_sum = torch.zeros(self.m, device=self.device)
                
                for grid in batch_grids:
                    vectors, comp_indices = self.extract_vectors(grid)
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
        
        self.mean_window_count = total_window_count / len(grids) if grids else 0
        self.mean_t = (total_t / total_window_count).cpu().numpy() if total_window_count > 0 else np.zeros(self.m)

    def predict_t(self, grid):
        """Predict target vector as average of N(k1,k2) vectors over all windows."""
        vectors, comp_indices = self.extract_vectors(grid)
        if vectors.shape[0] == 0:
            return np.zeros(self.m)
        Nk = self.batch_compute_Nk(comp_indices, vectors)
        return torch.mean(Nk, dim=0).detach().cpu().numpy()

    def predict_c(self, grid):
        """Predict class label for a grid."""
        if self.classifier is None:
            raise ValueError("Model must be trained first for classification")
        seq_vector = self.predict_t(grid)
        seq_vector_tensor = torch.tensor(seq_vector, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits = self.classifier(seq_vector_tensor.unsqueeze(0))
            probabilities = torch.softmax(logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
        return predicted_class, probabilities[0].cpu().numpy()

    def predict_l(self, grid, threshold=0.5):
        """Predict multi-label classification for a grid."""
        assert self.labeller is not None, "Model must be trained first for label prediction"
        vectors, comp_indices = self.extract_vectors(grid)
        if vectors.shape[0] == 0:
            return np.zeros(self.num_labels, dtype=np.float32), np.zeros(self.num_labels, dtype=np.float32)
        Nk_batch = self.batch_compute_Nk(comp_indices, vectors)
        grid_repr = torch.mean(Nk_batch, dim=0)
        with torch.no_grad():
            logits = self.labeller(grid_repr)
            probs = torch.sigmoid(logits).cpu().numpy()
        binary_preds = (probs > threshold).astype(np.float32)
        return binary_preds, probs

    def reconstruct(self, height, width, tau=0.0, num_candidates=1000):
        """
        Reconstruct representative 2D vector grid of size height×width.
        
        Args:
            height (int): Number of rows in output grid
            width (int): Number of columns in output grid
            tau (float): Temperature parameter (0: deterministic, >0: stochastic)
            num_candidates (int): Number of candidate vectors to consider
            
        Returns:
            numpy.ndarray: Reconstructed grid of shape [height, width, m]
        """
        assert self.trained, "Model must be trained first"
        if tau < 0:
            raise ValueError("Temperature must be non-negative")
            
        mean_t_tensor = torch.tensor(self.mean_t, dtype=torch.float32, device=self.device)
        generated_grid = []
        
        # Generate random candidate vectors
        candidate_vectors = torch.randn(num_candidates, self.m, device=self.device)
        
        for i in range(height):
            row = []
            for j in range(width):
                # Compute composite index for (i, j)
                comp_k = self._get_composite_index(i, j)
                k_tensor = torch.tensor([comp_k] * num_candidates, dtype=torch.float32, device=self.device)
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
            generated_grid.append(row)
        return np.stack(generated_grid)

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
            'L2': self.L2
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
# Example Usage with 2D Vector Grids
# ================================================
if __name__ == "__main__":
    from statistics import correlation
    
    print("="*60)
    print("SpatialNumDualDescriptorRN2 - PyTorch GPU Accelerated Version (2D)")
    print("Optimized for 2D Vector Grids with Equal-Length Batch Acceleration")
    print("="*60)
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    
    # Parameters
    vec_dim = 3
    bas_dim1 = 5
    bas_dim2 = 5
    rank = 2                  # window size 2x2
    grid_num = 50
    grid_shape_range = ((10, 20), (10, 20))  # (min_H, max_H), (min_W, max_W)
    
    # ------------------------------------------------------------------
    # Example 1: Regression Task
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("Example 1: Regression Task")
    print("="*60)
    
    # Generate random 2D grids and target vectors
    grids, t_list = [], []
    for _ in range(grid_num):
        H = np.random.randint(grid_shape_range[0][0], grid_shape_range[0][1])
        W = np.random.randint(grid_shape_range[1][0], grid_shape_range[1][1])
        # Random grid: (H, W, vec_dim)
        grid = np.random.randn(H, W, vec_dim).astype(np.float32)
        grids.append(grid)
        # Random target vector
        t_list.append(np.random.uniform(-1, 1, vec_dim).astype(np.float32))
    
    model = SpatialNumDualDescriptorRN2(
        vec_dim=vec_dim,
        bas_dim1=bas_dim1,
        bas_dim2=bas_dim2,
        rank=rank,
        rank_mode='drop',
        mode='linear',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    print(f"\nUsing device: {model.device}")
    print(f"Vector dimension: {model.m}")
    print(f"Basis dimensions: {model.L1} × {model.L2}")
    print(f"Window size: {model.rank}×{model.rank}")
    
    # Train regression
    print("\nStarting gradient descent training...")
    history = model.reg_train(
        grids, t_list,
        learning_rate=0.05,
        max_iters=30,
        tol=1e-6,
        decay_rate=0.99,
        print_every=10,
        batch_size=8
    )
    
    # Predict on first grid
    test_grid = grids[0]
    t_pred = model.predict_t(test_grid)
    print(f"\nPredicted t for first grid: {t_pred}")
    print(f"Actual t: {t_list[0]}")
    print(f"Mean squared error: {np.mean((t_pred - t_list[0]) ** 2):.6f}")
    
    # Correlation across all grids
    pred_t_list = [model.predict_t(g) for g in grids]
    corr_sum = 0.0
    for i in range(model.m):
        actu_t = [t[i] for t in t_list]
        pred_t = [p[i] for p in pred_t_list]
        corr = correlation(actu_t, pred_t)
        print(f"Prediction correlation for dimension {i}: {corr:.4f}")
        corr_sum += corr
    print(f"Average correlation: {corr_sum / model.m:.4f}")
    
    # Reconstruction
    print("\nReconstructing representative 2D grid (8×8)...")
    recon_grid = model.reconstruct(height=8, width=8, tau=0.0)
    print(f"Reconstructed grid shape: {recon_grid.shape}")
    print("First 2 rows of first channel:")
    print(recon_grid[:2, :, 0])
    
    # ------------------------------------------------------------------
    # Example 2: Classification Task
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("Example 2: Classification Task")
    print("="*60)
    
    num_classes = 3
    class_grids = []
    class_labels = []
    
    for class_id in range(num_classes):
        for _ in range(30):
            H = np.random.randint(12, 18)
            W = np.random.randint(12, 18)
            if class_id == 0:
                grid = np.random.normal(loc=1.0, scale=0.2, size=(H, W, vec_dim)).astype(np.float32)
            elif class_id == 1:
                grid = np.random.normal(loc=-1.0, scale=0.5, size=(H, W, vec_dim)).astype(np.float32)
            else:
                grid = np.random.normal(loc=0.0, scale=1.0, size=(H, W, vec_dim)).astype(np.float32)
            class_grids.append(grid)
            class_labels.append(class_id)
    
    model_cls = SpatialNumDualDescriptorRN2(
        vec_dim=vec_dim, bas_dim1=bas_dim1, bas_dim2=bas_dim2,
        rank=rank, mode='linear', device=model.device
    )
    
    print("\nStarting classification training...")
    cls_history = model_cls.cls_train(
        class_grids, class_labels, num_classes,
        max_iters=20, tol=1e-6, learning_rate=0.05,
        decay_rate=0.99, batch_size=8, print_every=5
    )
    
    # Evaluate
    correct = 0
    for g, true_lbl in zip(class_grids, class_labels):
        pred, _ = model_cls.predict_c(g)
        if pred == true_lbl:
            correct += 1
    accuracy = correct / len(class_grids)
    print(f"\nClassification accuracy: {accuracy:.4f} ({correct}/{len(class_grids)})")
    
    # ------------------------------------------------------------------
    # Example 3: Multi-Label Classification
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("Example 3: Multi-Label Classification")
    print("="*60)
    
    num_labels = 4
    ml_grids = []
    ml_labels = []
    for _ in range(50):
        H = np.random.randint(10, 20)
        W = np.random.randint(10, 20)
        grid = np.random.randn(H, W, vec_dim).astype(np.float32)
        ml_grids.append(grid)
        label_vec = [1.0 if np.random.random() > 0.7 else 0.0 for _ in range(num_labels)]
        ml_labels.append(label_vec)
    
    model_lbl = SpatialNumDualDescriptorRN2(
        vec_dim=vec_dim, bas_dim1=bas_dim1, bas_dim2=bas_dim2,
        rank=rank, mode='linear', device=model.device
    )
    
    print("\nStarting multi-label training...")
    loss_hist, acc_hist = model_lbl.lbl_train(
        ml_grids, ml_labels, num_labels,
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
    
    self_grids = []
    for _ in range(20):
        H = np.random.randint(10, 20)
        W = np.random.randint(10, 20)
        grid = np.random.randn(H, W, vec_dim).astype(np.float32)
        self_grids.append(grid)
    
    model_self = SpatialNumDualDescriptorRN2(
        vec_dim=vec_dim, bas_dim1=bas_dim1, bas_dim2=bas_dim2,
        rank=rank, mode='linear', device=model.device
    )
    
    print("\nStarting self-training...")
    self_history = model_self.self_train(
        self_grids, max_iters=30, learning_rate=0.01,
        decay_rate=0.995, batch_size=512, print_every=5
    )
    
    # Reconstruction after self-training
    recon_self = model_self.reconstruct(height=5, width=5, tau=0.0)
    print(f"\nSelf-trained reconstruction (5×5) shape: {recon_self.shape}")
    
    # ------------------------------------------------------------------
    # Example 5: Model Persistence
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("Example 5: Model Persistence")
    print("="*60)
    
    model.save("spatial_dual_descriptor_2d.pt")
    model_loaded = SpatialNumDualDescriptorRN2(
        vec_dim=vec_dim, bas_dim1=bas_dim1, bas_dim2=bas_dim2,
        rank=rank, mode='linear', device=model.device
    )
    model_loaded.load("spatial_dual_descriptor_2d.pt")
    
    test_pred_orig = model.predict_t(test_grid)
    test_pred_load = model_loaded.predict_t(test_grid)
    print(f"Original prediction: {test_pred_orig}")
    print(f"Loaded prediction: {test_pred_load}")
    print(f"Match: {np.allclose(test_pred_orig, test_pred_load, rtol=1e-6)}")
    
    print("\n" + "="*60)
    print("All examples completed successfully!")
    print("="*60)
