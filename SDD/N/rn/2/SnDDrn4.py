# Copyright (C) 2005-2026, Bin-Guang Ma (mbg@mail.hzau.edu.cn); SPDX-License-Identifier: MIT
# The Spatial Numeric Dual Descriptor Vector class (Random AB matrix form) for 4D array implemented with PyTorch
# This program is for the demonstration of methodology and not fully refined.
# Author: Bin-Guang Ma (assisted by DeepSeek); Date: 2025-8-28 ~ 2026-03-28

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

class SpatialNumDualDescriptorRN4(nn.Module):
    """
    Dual Descriptor for 4D vector spatiotemporal volumes with GPU acceleration using PyTorch:
      - Learnable coefficient matrix Acoeff ∈ R^{m×(L1·L2·L3·L4)}
      - Learnable basis matrix Bbasis ∈ R^{(L1·L2·L3·L4)×m}
      - Learnable mapping matrix M ∈ R^{m×m}
      - Input: 4D volumes of m-dimensional vectors (T × D × H × W × m)
      - Optimized with batch processing for GPU acceleration
      - Supports both linear and nonlinear 4D windowing
      - Supports regression, classification, and multi-label classification tasks
    """
    def __init__(self, vec_dim=4, bas_dim1=5, bas_dim2=5, bas_dim3=5, bas_dim4=5, rank=1, rank_op='avg', rank_mode='drop',
                 mode='linear', user_step=None, device='cuda'):
        """
        Initialize the Spatial Dual Descriptor for 4D vector spatiotemporal volumes.
        
        Args:
            vec_dim (int): Dimension m of input vectors
            bas_dim1 (int): Basis dimension for first index (time)
            bas_dim2 (int): Basis dimension for second index (depth)
            bas_dim3 (int): Basis dimension for third index (height)
            bas_dim4 (int): Basis dimension for fourth index (width)
            rank (int): Window size (hypercubic window, side length) for vector aggregation
            rank_op (str): 'avg', 'sum', 'pick', or 'user_func'
            rank_mode (str): 'pad' or 'drop' for handling incomplete windows
            mode (str): 'linear' (sliding window, step=1) or 'nonlinear' (stepped window)
            user_step (int): Custom step size for nonlinear mode (same for all dimensions)
            device (str): 'cuda' or 'cpu'
        """
        super().__init__()
        self.m = vec_dim                     # Vector dimension
        self.L1 = bas_dim1                   # Basis dimension for time
        self.L2 = bas_dim2                   # Basis dimension for depth
        self.L3 = bas_dim3                   # Basis dimension for height
        self.L4 = bas_dim4                   # Basis dimension for width
        self.total_basis = self.L1 * self.L2 * self.L3 * self.L4
        self.rank = rank                     # Window size (hypercube)
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
    
    def _get_composite_index(self, k1, k2, k3, k4):
        """
        Convert 4D position indices to a composite index.
        Composite index = k1 * L2*L3*L4 + k2 * L3*L4 + k3 * L4 + k4
        Modulo total_basis yields (k1 % L1, k2 % L2, k3 % L3, k4 % L4).
        """
        return k1 * (self.L2 * self.L3 * self.L4) + k2 * (self.L3 * self.L4) + k3 * self.L4 + k4
    
    def _compute_basis_index(self, composite_k):
        """Compute basis index j = composite_k % total_basis."""
        return (composite_k % self.total_basis).long()
    
    def extract_vectors(self, volume):
        """
        Extract vector groups from a 4D vector volume using sliding windows and return
        aggregated vectors along with their composite position indices.
        
        The volume is expected to have shape (T, D, H, W, m) where T, D, H, W are dimensions.
        Windows are hypercubic of size rank × rank × rank × rank. They are extracted in
        row-major order (time, depth, height, width).
        
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
        # Convert to tensor if needed, ensure shape (T, D, H, W, m)
        if not isinstance(volume, torch.Tensor):
            volume = torch.tensor(np.array(volume), dtype=torch.float32, device=self.device)
        # Handle different input shapes: if 4D (T, D, H, W) then add last dimension of 1
        if volume.dim() == 4:
            volume = volume.unsqueeze(-1)  # (T, D, H, W, 1)
        if volume.dim() == 5 and volume.shape[-1] != self.m:
            # Assume the last dimension is the vector dimension (might be different)
            # In case of mismatch, raise warning but proceed
            print(f"Warning: volume last dimension {volume.shape[-1]} != vec_dim {self.m}")
        
        T, D, H, W = volume.shape[0], volume.shape[1], volume.shape[2], volume.shape[3]
        step = self.step
        
        # ========== Vectorized path for linear mode with avg/sum ==========
        if self.mode == 'linear' and self.rank_op in ('avg', 'sum') and T >= self.rank and D >= self.rank and H >= self.rank and W >= self.rank:
            # Compute number of windows along each dimension
            T_w = T - self.rank + 1
            D_w = D - self.rank + 1
            H_w = H - self.rank + 1
            W_w = W - self.rank + 1
            n_windows = T_w * D_w * H_w * W_w
            
            # Generate all starting indices for windows (grid)
            t_start = torch.arange(T_w, device=self.device)
            d_start = torch.arange(D_w, device=self.device)
            h_start = torch.arange(H_w, device=self.device)
            w_start = torch.arange(W_w, device=self.device)
            t_grid, d_grid, h_grid, w_grid = torch.meshgrid(t_start, d_start, h_start, w_start, indexing='ij')
            t_start_all = t_grid.reshape(-1)  # (n_windows,)
            d_start_all = d_grid.reshape(-1)
            h_start_all = h_grid.reshape(-1)
            w_start_all = w_grid.reshape(-1)
            
            # Generate offsets within a window
            offsets = torch.arange(self.rank, device=self.device)
            t_off, d_off, h_off, w_off = torch.meshgrid(offsets, offsets, offsets, offsets, indexing='ij')
            t_off_all = t_off.reshape(-1)  # (rank^4,)
            d_off_all = d_off.reshape(-1)
            h_off_all = h_off.reshape(-1)
            w_off_all = w_off.reshape(-1)
            
            # Broadcast: all window starts + offsets
            t_all = t_start_all[:, None] + t_off_all[None, :]  # (n_windows, rank^4)
            d_all = d_start_all[:, None] + d_off_all[None, :]
            h_all = h_start_all[:, None] + h_off_all[None, :]
            w_all = w_start_all[:, None] + w_off_all[None, :]
            
            # Compute linear indices in the flattened volume (T*D*H*W)
            stride_T = D * H * W
            stride_D = H * W
            stride_H = W
            linear_idx = t_all * stride_T + d_all * stride_D + h_all * stride_H + w_all  # (n_windows, rank^4)
            
            # Flatten volume to (T*D*H*W, m)
            vol_flat = volume.reshape(-1, self.m)  # (total_positions, m)
            
            # Gather all window vectors in one go
            all_vectors = vol_flat[linear_idx]  # (n_windows, rank^4, m)
            
            # Aggregate along window positions
            if self.rank_op == 'avg':
                agg = all_vectors.mean(dim=1)  # (n_windows, m)
            else:  # sum
                agg = all_vectors.sum(dim=1)   # (n_windows, m)
            
            # Generate composite indices for each window
            comp_indices = self._get_composite_index(t_start_all, d_start_all, h_start_all, w_start_all)
            
            return agg, comp_indices
        
        # ========== Fallback loop-based extraction for other modes ==========
        def apply_op(vectors):
            # vectors: (rank^4, m)
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
            # Step 1 in all dimensions
            for t in range(T - self.rank + 1):
                for d in range(D - self.rank + 1):
                    for h in range(H - self.rank + 1):
                        for w in range(W - self.rank + 1):
                            window_vectors = volume[t:t+self.rank, d:d+self.rank, h:h+self.rank, w:w+self.rank].reshape(-1, self.m)
                            agg_vec = apply_op(window_vectors)
                            windows.append(agg_vec)
                            composite_indices.append(self._get_composite_index(t, d, h, w))
        else:
            # Nonlinear mode: step stride
            for t in range(0, T, step):
                for d in range(0, D, step):
                    for h in range(0, H, step):
                        for w in range(0, W, step):
                            t_end = t + self.rank
                            d_end = d + self.rank
                            h_end = h + self.rank
                            w_end = w + self.rank
                            if t_end > T or d_end > D or h_end > H or w_end > W:
                                if self.rank_mode == 'pad':
                                    # Collect existing vectors and pad missing ones
                                    window_list = []
                                    for ti in range(t, min(t_end, T)):
                                        for di in range(d, min(d_end, D)):
                                            for hi in range(h, min(h_end, H)):
                                                for wi in range(w, min(w_end, W)):
                                                    window_list.append(volume[ti, di, hi, wi])
                                    pad_count = self.rank**4 - len(window_list)
                                    if pad_count > 0:
                                        zero_pad = torch.zeros(pad_count, self.m, device=self.device)
                                        window_vectors = torch.cat([torch.stack(window_list), zero_pad])
                                    else:
                                        window_vectors = torch.stack(window_list)
                                else:  # drop
                                    continue
                            else:
                                window_vectors = volume[t:t+self.rank, d:d+self.rank, h:h+self.rank, w:w+self.rank].reshape(-1, self.m)
                            
                            agg_vec = apply_op(window_vectors)
                            windows.append(agg_vec)
                            composite_indices.append(self._get_composite_index(t, d, h, w))
        
        if windows:
            return torch.stack(windows), torch.tensor(composite_indices, dtype=torch.float32, device=self.device)
        else:
            return torch.empty(0, self.m, device=self.device), torch.empty(0, dtype=torch.float32, device=self.device)
    
    def batch_compute_Nk(self, composite_k_tensor, vectors):
        """
        Vectorized computation of N(k1,k2,k3,k4) vectors for a batch of composite positions and vectors.
        
        Args:
            composite_k_tensor: Tensor of composite indices [batch_size]
            vectors: Tensor of input vectors [batch_size, m]
            
        Returns:
            Tensor of N(k1,k2,k3,k4) vectors [batch_size, m]
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
    
    def describe(self, volume):
        """Compute N(k1,k2,k3,k4) vectors for each window in the volume."""
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
        D = average over all windows of (N(k1,k2,k3,k4)-t_seq)^2
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
        Optimized: collects all windows from a batch, computes Nk in bulk.
        
        Args:
            volumes: List of 4D vector volumes (each as numpy array or torch tensor, shape T×D×H×W×m)
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
        # Ensure volumes are on correct device
        volumes = [vol.to(self.device) if isinstance(vol, torch.Tensor) else
                   torch.tensor(vol, dtype=torch.float32, device=self.device)
                   for vol in volumes]
        
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)
        
        if continued and checkpoint_file and os.path.exists(checkpoint_file):
            optimizer.load_state_dict(optimizer_state)
            scheduler.load_state_dict(scheduler_state)
        
        prev_loss = float('inf') if start_iter == 0 else history[-1] if history else float('inf')
        
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
                
                # Collect all windows from all volumes in the batch
                all_vectors = []
                all_comp = []
                volume_window_counts = []
                
                for vol in batch_volumes:
                    vectors, comp_indices = self.extract_vectors(vol)
                    if vectors.shape[0] == 0:
                        volume_window_counts.append(0)
                        continue
                    all_vectors.append(vectors)
                    all_comp.append(comp_indices)
                    volume_window_counts.append(vectors.shape[0])
                
                if not all_vectors:
                    continue
                
                # Concatenate all windows across volumes
                all_vectors = torch.cat(all_vectors, dim=0)   # (total_w, m)
                all_comp = torch.cat(all_comp, dim=0)         # (total_w,)
                
                # Compute Nk for all windows at once
                Nk_all = self.batch_compute_Nk(all_comp, all_vectors)  # (total_w, m)
                
                # Split Nk_all back to volumes and compute volume-level representation (mean)
                volume_reps = []
                start_idx = 0
                for nw in volume_window_counts:
                    if nw > 0:
                        vol_rep = Nk_all[start_idx:start_idx+nw].mean(dim=0)
                        start_idx += nw
                    else:
                        vol_rep = torch.zeros(self.m, device=self.device)
                    volume_reps.append(vol_rep)
                
                # Compute loss for this batch
                if volume_reps:
                    volume_reps_tensor = torch.stack(volume_reps)
                    targets_tensor = torch.stack(batch_targets)
                    loss = torch.mean(torch.sum((volume_reps_tensor - targets_tensor) ** 2, dim=1))
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item() * len(volume_reps)
                    total_volumes += len(volume_reps)
                
                # Clean up
                del all_vectors, all_comp, Nk_all, volume_reps, loss
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
        """Multi-class classification training with batch window processing."""
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
                
                # Collect windows from all volumes in batch
                all_vectors = []
                all_comp = []
                volume_window_counts = []
                
                for vol in batch_volumes:
                    vectors, comp_indices = self.extract_vectors(vol)
                    if vectors.shape[0] == 0:
                        volume_window_counts.append(0)
                        continue
                    all_vectors.append(vectors)
                    all_comp.append(comp_indices)
                    volume_window_counts.append(vectors.shape[0])
                
                if not all_vectors:
                    continue
                
                all_vectors = torch.cat(all_vectors, dim=0)
                all_comp = torch.cat(all_comp, dim=0)
                Nk_all = self.batch_compute_Nk(all_comp, all_vectors)
                
                volume_reps = []
                start_idx = 0
                for nw in volume_window_counts:
                    if nw > 0:
                        vol_rep = Nk_all[start_idx:start_idx+nw].mean(dim=0)
                        start_idx += nw
                    else:
                        vol_rep = torch.zeros(self.m, device=self.device)
                    volume_reps.append(vol_rep)
                
                if volume_reps:
                    volume_reps_tensor = torch.stack(volume_reps)
                    logits = self.classifier(volume_reps_tensor)
                    loss = criterion(logits, batch_labels)
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item() * len(volume_reps)
                    total_volumes += len(volume_reps)
                    with torch.no_grad():
                        preds = torch.argmax(logits, dim=1)
                        correct += (preds == batch_labels).sum().item()
                
                del all_vectors, all_comp, Nk_all, volume_reps, logits, loss
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
                
                # Collect windows
                all_vectors = []
                all_comp = []
                volume_window_counts = []
                
                for vol in batch_volumes:
                    vectors, comp_indices = self.extract_vectors(vol)
                    if vectors.shape[0] == 0:
                        volume_window_counts.append(0)
                        continue
                    all_vectors.append(vectors)
                    all_comp.append(comp_indices)
                    volume_window_counts.append(vectors.shape[0])
                
                if not all_vectors:
                    continue
                
                all_vectors = torch.cat(all_vectors, dim=0)
                all_comp = torch.cat(all_comp, dim=0)
                Nk_all = self.batch_compute_Nk(all_comp, all_vectors)
                
                volume_reps = []
                start_idx = 0
                for nw in volume_window_counts:
                    if nw > 0:
                        vol_rep = Nk_all[start_idx:start_idx+nw].mean(dim=0)
                        start_idx += nw
                    else:
                        vol_rep = torch.zeros(self.m, device=self.device)
                    volume_reps.append(vol_rep)
                
                if volume_reps:
                    volume_reps_tensor = torch.stack(volume_reps)
                    logits = self.labeller(volume_reps_tensor)
                    loss = criterion(logits, batch_labels)
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item() * len(volume_reps)
                    total_volumes += len(volume_reps)
                    
                    with torch.no_grad():
                        probs = torch.sigmoid(logits)
                        preds = (probs > 0.5).float()
                        total_correct += (preds == batch_labels).sum().item()
                        total_preds += batch_labels.numel()
                
                del all_vectors, all_comp, Nk_all, volume_reps, logits, loss
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
        """
        Self-supervised training using gap-filling objective.
        Optimized: collects all windows from a batch, computes Nk in bulk.
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
        
        volumes = [vol.to(self.device) if isinstance(vol, torch.Tensor) else
                   torch.tensor(vol, dtype=torch.float32, device=self.device)
                   for vol in volumes]
        
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)
        
        if continued and checkpoint_file and os.path.exists(checkpoint_file):
            optimizer.load_state_dict(optimizer_state)
            scheduler.load_state_dict(scheduler_state)
        
        prev_loss = float('inf') if start_iter == 0 else history[-1] if history else float('inf')
        
        for it in range(start_iter, max_iters):
            total_loss = 0.0
            total_samples = 0
            
            indices = list(range(len(volumes)))
            random.shuffle(indices)
            
            for batch_start in range(0, len(indices), batch_size):
                batch_indices = indices[batch_start:batch_start + batch_size]
                batch_volumes = [volumes[idx] for idx in batch_indices]
                
                optimizer.zero_grad()
                
                # Collect all windows from all volumes in the batch
                all_vectors = []
                all_comp = []
                
                for vol in batch_volumes:
                    vectors, comp_indices = self.extract_vectors(vol)
                    if vectors.shape[0] == 0:
                        continue
                    all_vectors.append(vectors)
                    all_comp.append(comp_indices)
                
                if not all_vectors:
                    continue
                
                all_vectors = torch.cat(all_vectors, dim=0)
                all_comp = torch.cat(all_comp, dim=0)
                
                # Compute Nk for all windows
                Nk_all = self.batch_compute_Nk(all_comp, all_vectors)
                # Self-supervised target: mapped aggregated vectors (M * aggregated vectors)
                targets = self.map_vector(all_vectors)
                loss = torch.mean(torch.sum((Nk_all - targets) ** 2, dim=1))
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item() * all_vectors.size(0)
                total_samples += all_vectors.size(0)
                
                del all_vectors, all_comp, Nk_all, targets, loss
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
                'bas_dim4': self.L4,
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
        """Predict target vector as average of N(k1,k2,k3,k4) vectors over all windows."""
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
    
    def reconstruct(self, time, depth, height, width, tau=0.0, num_candidates=1000):
        """
        Reconstruct representative 4D vector volume of size time×depth×height×width.
        
        Args:
            time (int): Number of time steps
            depth (int): Number of depth slices
            height (int): Number of rows
            width (int): Number of columns
            tau (float): Temperature parameter (0: deterministic, >0: stochastic)
            num_candidates (int): Number of candidate vectors to consider
            
        Returns:
            numpy.ndarray: Reconstructed volume of shape [time, depth, height, width, m]
        """
        assert self.trained, "Model must be trained first"
        if tau < 0:
            raise ValueError("Temperature must be non-negative")
            
        mean_t_tensor = torch.tensor(self.mean_t, dtype=torch.float32, device=self.device)
        generated_volume = []
        
        # Generate random candidate vectors
        candidate_vectors = torch.randn(num_candidates, self.m, device=self.device)
        
        for t in range(time):
            time_slice = []
            for d in range(depth):
                depth_slice = []
                for h in range(height):
                    row = []
                    for w in range(width):
                        comp_k = self._get_composite_index(t, d, h, w)
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
                    depth_slice.append(row)
                time_slice.append(depth_slice)
            generated_volume.append(time_slice)
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
            'L3': self.L3,
            'L4': self.L4
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
# Example Usage with 4D Spatiotemporal Vector Volumes
# ================================================
if __name__ == "__main__":
    from statistics import correlation
    
    print("="*60)
    print("SpatialNumDualDescriptorRN4 - PyTorch GPU Accelerated Version (4D)")
    print("Optimized for 4D Spatiotemporal Vector Volumes")
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
    bas_dim4 = 3
    rank = 2                  # window size 2x2x2x2
    volume_num = 20
    # (min_T, max_T), (min_D, max_D), (min_H, max_H), (min_W, max_W)
    volume_shape_range = ((3, 6), (3, 6), (3, 6), (3, 6))
    
    # ------------------------------------------------------------------
    # Example 1: Regression Task
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("Example 1: Regression Task")
    print("="*60)
    
    # Generate random 4D volumes and target vectors
    volumes, t_list = [], []
    for _ in range(volume_num):
        T = np.random.randint(volume_shape_range[0][0], volume_shape_range[0][1])
        D = np.random.randint(volume_shape_range[1][0], volume_shape_range[1][1])
        H = np.random.randint(volume_shape_range[2][0], volume_shape_range[2][1])
        W = np.random.randint(volume_shape_range[3][0], volume_shape_range[3][1])
        volume = np.random.randn(T, D, H, W, vec_dim).astype(np.float32)
        volumes.append(volume)
        t_list.append(np.random.uniform(-1, 1, vec_dim).astype(np.float32))
    
    model = SpatialNumDualDescriptorRN4(
        vec_dim=vec_dim,
        bas_dim1=bas_dim1,
        bas_dim2=bas_dim2,
        bas_dim3=bas_dim3,
        bas_dim4=bas_dim4,
        rank=rank,
        rank_mode='drop',
        mode='linear',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    print(f"\nUsing device: {model.device}")
    print(f"Vector dimension: {model.m}")
    print(f"Basis dimensions: {model.L1} × {model.L2} × {model.L3} × {model.L4}")
    print(f"Window size: {model.rank}×{model.rank}×{model.rank}×{model.rank}")
    
    # Train regression
    print("\nStarting gradient descent training...")
    history = model.reg_train(
        volumes, t_list,
        learning_rate=0.05,
        max_iters=25,
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
    print("\nReconstructing representative 4D volume (3×3×3×3)...")
    recon_vol = model.reconstruct(time=3, depth=3, height=3, width=3, tau=0.0)
    print(f"Reconstructed volume shape: {recon_vol.shape}")
    print("First time slice (t=0) of first channel:")
    print(recon_vol[0, :, :, :, 0])
    
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
        for _ in range(15):
            T = np.random.randint(4, 6)
            D = np.random.randint(4, 6)
            H = np.random.randint(4, 6)
            W = np.random.randint(4, 6)
            if class_id == 0:
                vol = np.random.normal(loc=1.0, scale=0.2, size=(T, D, H, W, vec_dim)).astype(np.float32)
            elif class_id == 1:
                vol = np.random.normal(loc=-1.0, scale=0.5, size=(T, D, H, W, vec_dim)).astype(np.float32)
            else:
                vol = np.random.normal(loc=0.0, scale=1.0, size=(T, D, H, W, vec_dim)).astype(np.float32)
            class_volumes.append(vol)
            class_labels.append(class_id)
    
    model_cls = SpatialNumDualDescriptorRN4(
        vec_dim=vec_dim, bas_dim1=bas_dim1, bas_dim2=bas_dim2, bas_dim3=bas_dim3, bas_dim4=bas_dim4,
        rank=rank, mode='linear', device=model.device
    )
    
    print("\nStarting classification training...")
    cls_history = model_cls.cls_train(
        class_volumes, class_labels, num_classes,
        max_iters=15, tol=1e-6, learning_rate=0.05,
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
    for _ in range(20):
        T = np.random.randint(3, 6)
        D = np.random.randint(3, 6)
        H = np.random.randint(3, 6)
        W = np.random.randint(3, 6)
        vol = np.random.randn(T, D, H, W, vec_dim).astype(np.float32)
        ml_volumes.append(vol)
        label_vec = [1.0 if np.random.random() > 0.7 else 0.0 for _ in range(num_labels)]
        ml_labels.append(label_vec)
    
    model_lbl = SpatialNumDualDescriptorRN4(
        vec_dim=vec_dim, bas_dim1=bas_dim1, bas_dim2=bas_dim2, bas_dim3=bas_dim3, bas_dim4=bas_dim4,
        rank=rank, mode='linear', device=model.device
    )
    
    print("\nStarting multi-label training...")
    loss_hist, acc_hist = model_lbl.lbl_train(
        ml_volumes, ml_labels, num_labels,
        max_iters=12, tol=1e-6, learning_rate=0.05,
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
    for _ in range(12):
        T = np.random.randint(3, 5)
        D = np.random.randint(3, 5)
        H = np.random.randint(3, 5)
        W = np.random.randint(3, 5)
        vol = np.random.randn(T, D, H, W, vec_dim).astype(np.float32)
        self_volumes.append(vol)
    
    model_self = SpatialNumDualDescriptorRN4(
        vec_dim=vec_dim, bas_dim1=bas_dim1, bas_dim2=bas_dim2, bas_dim3=bas_dim3, bas_dim4=bas_dim4,
        rank=rank, mode='linear', device=model.device
    )
    
    print("\nStarting self-training...")
    self_history = model_self.self_train(
        self_volumes, max_iters=15, learning_rate=0.01,
        decay_rate=0.995, batch_size=512, print_every=5
    )
    
    # Reconstruction after self-training
    recon_self = model_self.reconstruct(time=2, depth=2, height=2, width=2, tau=0.0)
    print(f"\nSelf-trained reconstruction (2×2×2×2) shape: {recon_self.shape}")
    
    # ------------------------------------------------------------------
    # Example 5: Model Persistence
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("Example 5: Model Persistence")
    print("="*60)
    
    model.save("spatial_dual_descriptor_4d.pt")
    model_loaded = SpatialNumDualDescriptorRN4(
        vec_dim=vec_dim, bas_dim1=bas_dim1, bas_dim2=bas_dim2, bas_dim3=bas_dim3, bas_dim4=bas_dim4,
        rank=rank, mode='linear', device=model.device
    )
    model_loaded.load("spatial_dual_descriptor_4d.pt")
    
    test_pred_orig = model.predict_t(test_vol)
    test_pred_load = model_loaded.predict_t(test_vol)
    print(f"Original prediction: {test_pred_orig}")
    print(f"Loaded prediction: {test_pred_load}")
    print(f"Match: {np.allclose(test_pred_orig, test_pred_load, rtol=1e-6)}")
    
    print("\n" + "="*60)
    print("All examples completed successfully!")
    print("="*60)
