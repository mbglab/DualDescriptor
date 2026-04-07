# Copyright (C) 2005-2026, Bin-Guang Ma (mbg@mail.hzau.edu.cn); SPDX-License-Identifier: MIT
# The Spatial Numeric Dual Descriptor Vector class (Random AB matrix form) for 2D array implemented with PyTorch
# This program is for the demonstration of methodology and not fully refined.
# Author: Bin-Guang Ma (assisted by DeepSeek); Date: 2025-8-28 ~ 2026-04-02

import math
import random
import itertools
import torch
import torch.nn as nn
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
      - Optimized with batch processing for GPU acceleration (unfold-based)
      - Supports both linear and nonlinear 2D windowing
      - Supports regression, classification, and multi-label classification tasks
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
        
        # Basis matrix Bbasis: total_basis × m (learnable)
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
                vectors_tensor: [num_windows, m] aggregated vectors
                composite_indices_tensor: [num_windows] composite indices
        """
        # Convert to tensor if needed, ensure shape (H, W, m)
        if not isinstance(grid, torch.Tensor):
            grid = torch.tensor(np.array(grid), dtype=torch.float32, device=self.device)
        if grid.dim() == 2:
            grid = grid.unsqueeze(-1)  # (H, W, 1)
        if grid.dim() == 3 and grid.shape[-1] != self.m:
            # Possibly grid is (H, W, ...) but not m? assume it's fine
            pass
        
        H, W = grid.shape[0], grid.shape[1]
        step = self.step
        
        windows = []
        composite_indices = []
        
        # Define helper for window aggregation
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
        
        # Iterate over window top-left corners
        if self.mode == 'linear':
            # Step 1
            for i in range(H - self.rank + 1):
                for j in range(W - self.rank + 1):
                    window_vectors = grid[i:i+self.rank, j:j+self.rank].reshape(-1, self.m)
                    agg_vec = apply_op(window_vectors)
                    windows.append(agg_vec)
                    composite_indices.append(self._get_composite_index(i, j))
        else:
            # Nonlinear mode: step stride
            for i in range(0, H, step):
                for j in range(0, W, step):
                    i_end = i + self.rank
                    j_end = j + self.rank
                    if i_end > H or j_end > W:
                        if self.rank_mode == 'pad':
                            window_vectors = []
                            for ii in range(i, min(i_end, H)):
                                for jj in range(j, min(j_end, W)):
                                    window_vectors.append(grid[ii, jj])
                            pad_count = self.rank*self.rank - len(window_vectors)
                            if pad_count > 0:
                                zero_pad = torch.zeros(pad_count, self.m, device=self.device)
                                window_vectors = torch.cat([torch.stack(window_vectors), zero_pad])
                            else:
                                window_vectors = torch.stack(window_vectors)
                        else:  # drop
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
    
    def batch_compute_Nk(self, composite_k_tensor, vectors):
        """
        Vectorized computation of N(k1,k2) vectors for a batch of composite positions and vectors.
        
        Args:
            composite_k_tensor: Tensor of composite indices [batch_size]
            vectors: Tensor of input vectors [batch_size, m]
            
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

    # ----------------------------------------------------------------------
    # Batch acceleration methods (unfold-based) for same-size grids
    # ----------------------------------------------------------------------
    def _get_window_coords_and_indices(self, H, W):
        """
        Compute window top-left coordinates and corresponding composite indices
        for a given grid size. Used by batch methods.
        
        Args:
            H, W: Height and width of the grid
            
        Returns:
            coords: List of (i, j) tuples for each window
            idx_tensor: Tensor of shape [n_windows] containing composite indices
        """
        step = 1 if self.mode == 'linear' else self.step
        
        if self.rank_mode == 'drop':
            i_range = range(0, H - self.rank + 1, step)
            j_range = range(0, W - self.rank + 1, step)
        else:  # 'pad'
            i_range = range(0, H, step)
            j_range = range(0, W, step)
        
        coords = []
        comp_indices = []
        for i in i_range:
            for j in j_range:
                coords.append((i, j))
                comp_indices.append(self._get_composite_index(i, j))
        
        if not coords:
            return [], torch.empty(0, dtype=torch.long, device=self.device)
        
        idx_tensor = torch.tensor(comp_indices, dtype=torch.long, device=self.device)
        return coords, idx_tensor

    def batch_represent(self, batch_grids):
        """
        Compute array representations for a batch of 2D vector grids efficiently.
        Supports arbitrary window size and step (linear/nonlinear) as long as all grids
        in the batch have identical dimensions and rank_mode == 'drop'.
        
        Args:
            batch_grids: Tensor of shape [batch_size, H, W, m]
        
        Returns:
            representations: Tensor of shape [batch_size, m]
        """
        B, H, W, m = batch_grids.shape
        assert self.rank_mode == 'drop', "batch_represent only supports rank_mode='drop'"
        
        step = 1 if self.mode == 'linear' else self.step
        
        # Convert to (B, m, H, W) for unfold
        x = batch_grids.permute(0, 3, 1, 2)  # [B, m, H, W]
        
        # Use unfold to extract sliding windows
        windows = torch.nn.functional.unfold(
            x, kernel_size=(self.rank, self.rank), stride=(step, step)
        )  # shape: [B, m*rank*rank, n_windows]
        
        n_windows = windows.shape[2]
        # Reshape to [B, n_windows, rank*rank, m]
        windows = windows.view(B, m, self.rank * self.rank, n_windows).permute(0, 3, 2, 1)
        # windows: [B, n_windows, win_size, m]
        
        # Apply mapping matrix M to each vector in each window
        mapped = torch.matmul(windows, self.M.T)  # [B, n_windows, win_size, m]
        
        # Aggregate over win_size dimension according to rank_op
        if self.rank_op == 'sum':
            agg = mapped.sum(dim=2)  # [B, n_windows, m]
        elif self.rank_op == 'pick':
            idx = torch.randint(0, self.rank * self.rank, (B, n_windows, 1), device=self.device)
            agg = torch.gather(mapped, 2, idx.expand(-1, -1, -1, self.m)).squeeze(2)
        elif self.rank_op == 'user_func':
            avg = mapped.mean(dim=2)
            agg = torch.sigmoid(avg)
        else:  # 'avg'
            agg = mapped.mean(dim=2)  # [B, n_windows, m]
        
        # Compute composite indices for windows
        # Generate top-left coordinates
        if self.rank_mode == 'drop':
            i_start = 0
            i_end = H - self.rank + 1
            j_start = 0
            j_end = W - self.rank + 1
        else:
            # For completeness (though not used due to assert)
            i_start = 0
            i_end = H
            j_start = 0
            j_end = W
        
        i_coords = torch.arange(i_start, i_end, step, device=self.device).view(-1, 1).expand(-1, len(range(j_start, j_end, step))).reshape(-1)
        j_coords = torch.arange(j_start, j_end, step, device=self.device).view(1, -1).expand(len(range(i_start, i_end, step)), -1).reshape(-1)
        # i_coords, j_coords: [n_windows]
        
        # Compute composite indices
        comp_k = i_coords * self.L2 + j_coords  # [n_windows]
        # Basis index = comp_k % total_basis
        idx_basis = comp_k % self.total_basis  # [n_windows]
        
        # Get corresponding B rows
        B_rows = self.Bbasis[idx_basis]  # [n_windows, m]
        B_rows = B_rows.unsqueeze(0)  # [1, n_windows, m]
        
        # Get A columns
        A_cols = self.Acoeff[:, idx_basis].T  # [n_windows, m]
        A_cols = A_cols.unsqueeze(0)  # [1, n_windows, m]
        
        # Compute scalar = B_row • agg
        scalar = torch.sum(B_rows * agg, dim=2)  # [B, n_windows]
        
        # Compute Nk = A_cols * scalar
        Nk = A_cols * scalar.unsqueeze(2)  # [B, n_windows, m]
        
        # Average over windows to get representation
        rep = Nk.mean(dim=1)  # [B, m]
        return rep

    def batch_compute_Nk_and_targets(self, batch_grids):
        """
        Compute Nk vectors and target vectors for each window in a batch of 2D grids.
        Supports arbitrary window size and step (linear/nonlinear) when rank_mode='drop'.
        
        Args:
            batch_grids: Tensor of shape [batch_size, H, W, m]
        
        Returns:
            Nk_all: Tensor [batch_size, n_windows, m] - Nk vectors for each window
            targets: Tensor [batch_size, n_windows, m] - target vectors (aggregated mapped vectors)
        """
        B, H, W, m = batch_grids.shape
        assert self.rank_mode == 'drop', "batch_compute_Nk_and_targets only supports rank_mode='drop'"
        
        step = 1 if self.mode == 'linear' else self.step
        
        x = batch_grids.permute(0, 3, 1, 2)  # [B, m, H, W]
        
        windows = torch.nn.functional.unfold(
            x, kernel_size=(self.rank, self.rank), stride=(step, step)
        )  # [B, m*rank*rank, n_windows]
        
        n_windows = windows.shape[2]
        windows = windows.view(B, m, self.rank * self.rank, n_windows).permute(0, 3, 2, 1)
        # [B, n_windows, win_size, m]
        
        # Apply mapping
        mapped = torch.matmul(windows, self.M.T)  # [B, n_windows, win_size, m]
        
        # Aggregate to get targets
        if self.rank_op == 'sum':
            targets = mapped.sum(dim=2)
        elif self.rank_op == 'pick':
            idx = torch.randint(0, self.rank * self.rank, (B, n_windows, 1), device=self.device)
            targets = torch.gather(mapped, 2, idx.expand(-1, -1, -1, self.m)).squeeze(2)
        elif self.rank_op == 'user_func':
            avg = mapped.mean(dim=2)
            targets = torch.sigmoid(avg)
        else:
            targets = mapped.mean(dim=2)  # [B, n_windows, m]
        
        # Compute composite indices for windows
        i_start = 0
        i_end = H - self.rank + 1
        j_start = 0
        j_end = W - self.rank + 1
        
        i_coords = torch.arange(i_start, i_end, step, device=self.device).view(-1, 1).expand(-1, len(range(j_start, j_end, step))).reshape(-1)
        j_coords = torch.arange(j_start, j_end, step, device=self.device).view(1, -1).expand(len(range(i_start, i_end, step)), -1).reshape(-1)
        
        comp_k = i_coords * self.L2 + j_coords  # [n_windows]
        idx_basis = comp_k % self.total_basis
        
        B_rows = self.Bbasis[idx_basis]  # [n_windows, m]
        B_rows = B_rows.unsqueeze(0)  # [1, n_windows, m]
        A_cols = self.Acoeff[:, idx_basis].T  # [n_windows, m]
        A_cols = A_cols.unsqueeze(0)  # [1, n_windows, m]
        
        scalar = torch.sum(B_rows * targets, dim=2)  # [B, n_windows]
        Nk = A_cols * scalar.unsqueeze(2)  # [B, n_windows, m]
        
        return Nk, targets

    # ----------------------------------------------------------------------
    # Core description methods
    # ----------------------------------------------------------------------
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

    # ----------------------------------------------------------------------
    # Regression training (with batch acceleration)
    # ----------------------------------------------------------------------
    def reg_train(self, grids, t_list, max_iters=1000, tol=1e-8, learning_rate=0.01,
                  continued=False, decay_rate=1.0, print_every=10, batch_size=32,
                  checkpoint_file=None, checkpoint_interval=10):
        """
        Train model using gradient descent with grid-level batch processing.
        Automatically uses fast batch path if all grids in a batch have identical dimensions
        and rank_mode == 'drop'.
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
        
        # Convert targets and grids to tensors on device
        t_tensors = [torch.tensor(t, dtype=torch.float32, device=self.device) for t in t_list]
        grid_tensors = []
        shapes = []
        for g in grids:
            if not isinstance(g, torch.Tensor):
                g = torch.tensor(g, dtype=torch.float32, device=self.device)
            else:
                g = g.to(self.device)
            grid_tensors.append(g)
            shapes.append(g.shape)
        
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)
        
        if continued and checkpoint_file and os.path.exists(checkpoint_file):
            optimizer.load_state_dict(optimizer_state)
            scheduler.load_state_dict(scheduler_state)
        
        prev_loss = float('inf') if start_iter == 0 else history[-1] if history else float('inf')
        use_fast_batch = (self.rank_mode == 'drop')
        
        for it in range(start_iter, max_iters):
            total_loss = 0.0
            total_grids = 0
            
            indices = list(range(len(grid_tensors)))
            random.shuffle(indices)
            
            for batch_start in range(0, len(indices), batch_size):
                batch_indices = indices[batch_start:batch_start + batch_size]
                batch_grids = [grid_tensors[idx] for idx in batch_indices]
                batch_targets = [t_tensors[idx] for idx in batch_indices]
                
                optimizer.zero_grad()
                
                # Check if all grids in batch have same shape
                batch_shapes = [g.shape for g in batch_grids]
                all_equal = len(set(batch_shapes)) == 1
                
                if use_fast_batch and all_equal:
                    # Fast batch processing
                    H, W, _ = batch_grids[0].shape
                    batch_tensor = torch.stack(batch_grids, dim=0)  # [batch, H, W, m]
                    reps = self.batch_represent(batch_tensor)  # [batch, m]
                    targets_tensor = torch.stack(batch_targets, dim=0)  # [batch, m]
                    loss = torch.mean(torch.sum((reps - targets_tensor) ** 2, dim=1))
                    loss.backward()
                    optimizer.step()
                    batch_loss = loss.item() * len(batch_grids)
                    total_loss += batch_loss
                    total_grids += len(batch_grids)
                else:
                    # Process each grid individually
                    batch_loss = 0.0
                    batch_count = 0
                    for grid, target in zip(batch_grids, batch_targets):
                        vectors, comp_indices = self.extract_vectors(grid)
                        if vectors.shape[0] == 0:
                            continue
                        Nk_batch = self.batch_compute_Nk(comp_indices, vectors)
                        grid_pred = torch.mean(Nk_batch, dim=0)
                        grid_loss = torch.sum((grid_pred - target) ** 2)
                        batch_loss += grid_loss
                        batch_count += 1
                        del Nk_batch, grid_pred, comp_indices
                        if batch_count % 10 == 0 and torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    
                    if batch_count > 0:
                        batch_loss = batch_loss / batch_count
                        batch_loss.backward()
                        optimizer.step()
                        total_loss += batch_loss.item() * batch_count
                        total_grids += batch_count
                
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
        
        self._compute_training_statistics(grid_tensors)
        self.trained = True
        return history

    # ----------------------------------------------------------------------
    # Classification training (with batch acceleration)
    # ----------------------------------------------------------------------
    def cls_train(self, grids, labels, num_classes, max_iters=1000, tol=1e-8, learning_rate=0.01,
                  continued=False, decay_rate=1.0, print_every=10, batch_size=32,
                  checkpoint_file=None, checkpoint_interval=10):
        """Multi-class classification training with optional batch acceleration."""
        if self.classifier is None or self.num_classes != num_classes:
            self.classifier = nn.Linear(self.m, num_classes).to(self.device)
            self.num_classes = num_classes
        
        if not continued:
            self.reset_parameters()
        
        # Convert grids to tensors
        grid_tensors = []
        for g in grids:
            if not isinstance(g, torch.Tensor):
                g = torch.tensor(g, dtype=torch.float32, device=self.device)
            else:
                g = g.to(self.device)
            grid_tensors.append(g)
        
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
            total_grids = 0
            correct = 0
            
            indices = list(range(len(grid_tensors)))
            random.shuffle(indices)
            
            for batch_start in range(0, len(indices), batch_size):
                batch_indices = indices[batch_start:batch_start + batch_size]
                batch_grids = [grid_tensors[idx] for idx in batch_indices]
                batch_labels = label_tensors[batch_indices]
                
                optimizer.zero_grad()
                
                batch_shapes = [g.shape for g in batch_grids]
                all_equal = len(set(batch_shapes)) == 1
                
                if use_fast_batch and all_equal:
                    # Fast batch processing
                    H, W, _ = batch_grids[0].shape
                    batch_tensor = torch.stack(batch_grids, dim=0)
                    reps = self.batch_represent(batch_tensor)  # [batch, m]
                    logits = self.classifier(reps)
                    loss = criterion(logits, batch_labels)
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item() * len(batch_grids)
                    total_grids += len(batch_grids)
                    with torch.no_grad():
                        preds = torch.argmax(logits, dim=1)
                        correct += (preds == batch_labels).sum().item()
                else:
                    # Individual processing
                    batch_logits = []
                    for grid in batch_grids:
                        vectors, comp_indices = self.extract_vectors(grid)
                        if vectors.shape[0] == 0:
                            grid_vec = torch.zeros(self.m, device=self.device)
                        else:
                            Nk_batch = self.batch_compute_Nk(comp_indices, vectors)
                            grid_vec = torch.mean(Nk_batch, dim=0)
                            del Nk_batch, comp_indices
                        logits = self.classifier(grid_vec.unsqueeze(0))
                        batch_logits.append(logits)
                    
                    if batch_logits:
                        all_logits = torch.cat(batch_logits, dim=0)
                        loss = criterion(all_logits, batch_labels)
                        loss.backward()
                        optimizer.step()
                        
                        total_loss += loss.item() * len(batch_grids)
                        total_grids += len(batch_grids)
                        with torch.no_grad():
                            preds = torch.argmax(all_logits, dim=1)
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

    # ----------------------------------------------------------------------
    # Multi-label classification training (with batch acceleration)
    # ----------------------------------------------------------------------
    def lbl_train(self, grids, labels, num_labels, max_iters=1000, tol=1e-8, learning_rate=0.01,
                  continued=False, decay_rate=1.0, print_every=10, batch_size=32,
                  checkpoint_file=None, checkpoint_interval=10, pos_weight=None):
        """Multi-label classification training with optional batch acceleration."""
        if self.labeller is None or self.num_labels != num_labels:
            self.labeller = nn.Linear(self.m, num_labels).to(self.device)
            self.num_labels = num_labels
        
        if not continued:
            self.reset_parameters()
        
        # Convert grids to tensors
        grid_tensors = []
        for g in grids:
            if not isinstance(g, torch.Tensor):
                g = torch.tensor(g, dtype=torch.float32, device=self.device)
            else:
                g = g.to(self.device)
            grid_tensors.append(g)
        
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
            total_preds = 0
            total_grids = 0
            
            indices = list(range(len(grid_tensors)))
            random.shuffle(indices)
            
            for batch_start in range(0, len(indices), batch_size):
                batch_indices = indices[batch_start:batch_start + batch_size]
                batch_grids = [grid_tensors[idx] for idx in batch_indices]
                batch_labels = labels_tensor[batch_indices]
                
                optimizer.zero_grad()
                
                batch_shapes = [g.shape for g in batch_grids]
                all_equal = len(set(batch_shapes)) == 1
                
                if use_fast_batch and all_equal:
                    H, W, _ = batch_grids[0].shape
                    batch_tensor = torch.stack(batch_grids, dim=0)
                    reps = self.batch_represent(batch_tensor)
                    logits = self.labeller(reps)
                    loss = criterion(logits, batch_labels)
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item() * len(batch_grids)
                    total_grids += len(batch_grids)
                    with torch.no_grad():
                        probs = torch.sigmoid(logits)
                        preds = (probs > 0.5).float()
                        total_correct += (preds == batch_labels).sum().item()
                        total_preds += batch_labels.numel()
                else:
                    batch_logits = []
                    for grid in batch_grids:
                        vectors, comp_indices = self.extract_vectors(grid)
                        if vectors.shape[0] == 0:
                            continue
                        Nk_batch = self.batch_compute_Nk(comp_indices, vectors)
                        grid_repr = torch.mean(Nk_batch, dim=0)
                        logits = self.labeller(grid_repr)
                        batch_logits.append(logits)
                        del Nk_batch, grid_repr, comp_indices
                    
                    if batch_logits:
                        all_logits = torch.stack(batch_logits, dim=0)
                        loss = criterion(all_logits, batch_labels)
                        loss.backward()
                        optimizer.step()
                        
                        total_loss += loss.item() * len(batch_logits)
                        total_grids += len(batch_logits)
                        with torch.no_grad():
                            probs = torch.sigmoid(all_logits)
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

    # ----------------------------------------------------------------------
    # Self-supervised training (with batch acceleration)
    # ----------------------------------------------------------------------
    def self_train(self, grids, max_iters=1000, tol=1e-8, learning_rate=0.01,
                   continued=False, decay_rate=1.0, print_every=10, batch_size=1024,
                   checkpoint_file=None, checkpoint_interval=10):
        """Self-supervised training using gap-filling objective with optional batch acceleration."""
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
        
        # Convert grids to tensors
        grid_tensors = []
        for g in grids:
            if not isinstance(g, torch.Tensor):
                g = torch.tensor(g, dtype=torch.float32, device=self.device)
            else:
                g = g.to(self.device)
            grid_tensors.append(g)
        
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)
        
        if continued and checkpoint_file and os.path.exists(checkpoint_file):
            optimizer.load_state_dict(optimizer_state)
            scheduler.load_state_dict(scheduler_state)
        
        prev_loss = float('inf') if start_iter == 0 else history[-1] if history else float('inf')
        
        # Check if all grids have same shape
        shapes = [g.shape for g in grid_tensors]
        all_same_shape = len(set(shapes)) == 1
        use_fast_batch = (self.rank_mode == 'drop' and all_same_shape)
        
        if use_fast_batch:
            # Fast path: all grids same size, use batch_compute_Nk_and_targets
            H, W, m = grid_tensors[0].shape
            num_seqs = len(grid_tensors)
            all_seqs = torch.stack(grid_tensors, dim=0)  # [num_seqs, H, W, m]
            seq_batch_size = batch_size  # number of grids per batch
            
            for it in range(start_iter, max_iters):
                total_loss = 0.0
                total_windows = 0
                
                indices = list(range(num_seqs))
                random.shuffle(indices)
                
                for batch_start in range(0, num_seqs, seq_batch_size):
                    batch_indices = indices[batch_start:batch_start + seq_batch_size]
                    batch_seqs = all_seqs[batch_indices]  # [batch, H, W, m]
                    
                    optimizer.zero_grad()
                    Nk_batch, targets_batch = self.batch_compute_Nk_and_targets(batch_seqs)
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
            # Slow path: original per-grid, per-window processing
            for it in range(start_iter, max_iters):
                total_loss = 0.0
                total_samples = 0
                
                indices = list(range(len(grid_tensors)))
                random.shuffle(indices)
                
                for grid_idx in indices:
                    grid = grid_tensors[grid_idx]
                    vectors, comp_indices = self.extract_vectors(grid)
                    if vectors.shape[0] == 0:
                        continue
                    
                    # Generate samples: each window gives (composite_index, vector)
                    samples = [(comp_indices[i].item(), vectors[i]) for i in range(vectors.shape[0])]
                    if not samples:
                        continue
                    
                    # Process samples in batches
                    for b_start in range(0, len(samples), batch_size):
                        batch = samples[b_start:b_start + batch_size]
                        optimizer.zero_grad()
                        
                        k_list = [s[0] for s in batch]
                        v_list = [s[1] for s in batch]
                        k_tensor = torch.tensor(k_list, dtype=torch.float32, device=self.device)
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
        
        self._compute_training_statistics(grid_tensors)
        self.trained = True
        return history

    # ----------------------------------------------------------------------
    # Helper methods
    # ----------------------------------------------------------------------
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
    print("Optimized for 2D Vector Grids with Batch Acceleration")
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
