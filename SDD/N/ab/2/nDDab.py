# Copyright (C) 2005-2026, Bin-Guang Ma (mbg@mail.hzau.edu.cn); SPDX-License-Identifier: MIT
# The Numeric Dual Descriptor Vector class (AB matrix form) implemented with PyTorch
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

class NumDualDescriptorAB(nn.Module):
    """
    Numeric Dual Descriptor with GPU acceleration using PyTorch:
      - learnable coefficient matrix Acoeff ∈ R^{m×L}
      - fixed basis matrix Bbasis ∈ R^{L×m}, Bbasis[k][i] = cos(2π*(k+1)/(i+2))
      - learnable mapping matrix M ∈ R^{m×m} for input vector transformation
      - Supports both linear and nonlinear tokenization of vector sequences
      - Batch processing for GPU acceleration with equal-length sequence optimization
      - Supports regression, classification, and multi-label classification tasks
    """
    def __init__(self, vec_dim=4, bas_dim=50, rank=1, rank_op='avg', rank_mode='drop', 
                 mode='linear', user_step=None, device='cuda'):
        """
        Initialize the Dual Descriptor for vector sequences.
        
        Args:
            vec_dim (int): Dimension m of input vectors
            bas_dim (int): Basis dimension L
            rank (int): Window size for vector aggregation
            rank_op (str): 'avg', 'sum', 'pick', or 'user_func'
            rank_mode (str): 'pad' or 'drop' for handling incomplete windows
            mode (str): 'linear' (sliding window) or 'nonlinear' (stepped window)
            user_step (int): Custom step size for nonlinear mode
            device (str): 'cuda' or 'cpu'
        """
        super().__init__()
        self.m = vec_dim
        self.L = bas_dim
        self.rank = rank
        self.rank_op = rank_op
        self.rank_mode = rank_mode
        assert mode in ('linear','nonlinear')
        self.mode = mode
        self.step = user_step
        self.trained = False
        self.mean_t = None  
        self.mean_L = None
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Learnable mapping matrix M (m x m)
        self.M = nn.Parameter(torch.empty(self.m, self.m))
        
        # Coefficient matrix A (m x L)
        self.Acoeff = nn.Parameter(torch.empty(self.m, self.L))
        
        # Fixed basis matrix B (L x m)
        Bbasis = torch.empty(self.L, self.m)
        for k in range(self.L):
            for i in range(self.m):
                Bbasis[k, i] = math.cos(2 * math.pi * (k+1) / (i+2))
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
    
    def extract_vectors(self, vec_seq):
        """
        Extract and aggregate vector groups from a vector sequence based on vectorization mode.
        Optimized for linear mode with avg/sum aggregation using vectorized operations.
        
        - 'linear': Slide window by 1 step, extracting contiguous vectors of length = rank
        - 'nonlinear': Slide window by custom step (or rank length if step not specified)
        
        For nonlinear mode, handles incomplete trailing fragments using:
        - 'pad': Pads with zero vectors to maintain group length
        - 'drop': Discards incomplete fragments
        
        The method applies mapping matrix M to each window and aggregates using rank_op.
        
        Args:
            vec_seq (list or tensor): Input vector sequence to vectorize
            
        Returns:
            list: List of aggregated vectors from extracted vector groups
        """
        # Convert to tensor if needed
        if not isinstance(vec_seq, torch.Tensor):
            vec_seq = torch.tensor(vec_seq, dtype=torch.float32, device=self.device)
        elif vec_seq.device != self.device:
            vec_seq = vec_seq.to(self.device)
        
        # Linear mode with avg/sum: use vectorized unfold
        if self.mode == 'linear' and self.rank_op in ('avg', 'sum'):
            L = vec_seq.shape[0]
            if L < self.rank:
                return []
            
            # Prepare for unfold: shape (1, m, 1, L)
            seq_reshaped = vec_seq.unsqueeze(0).permute(0, 2, 1).unsqueeze(2)  # (1, m, 1, L)
            # Unfold along the length dimension (height=1, width=L)
            windows = F.unfold(seq_reshaped, kernel_size=(1, self.rank), stride=1)  # (1, m*rank, num_windows)
            num_windows = windows.shape[2]
            # Reshape to (num_windows, rank, m)
            windows = windows.view(1, self.m, self.rank, num_windows).squeeze(0)  # (m, rank, num_windows)
            windows = windows.permute(2, 1, 0)  # (num_windows, rank, m)
            
            # Aggregate along rank dimension
            if self.rank_op == 'avg':
                agg = windows.mean(dim=1)  # (num_windows, m)
            else:  # sum
                agg = windows.sum(dim=1)   # (num_windows, m)
            
            # Apply mapping matrix M
            transformed = torch.matmul(agg, self.M.T)  # (num_windows, m)
            return [transformed[i] for i in range(transformed.size(0))]
        
        # For nonlinear mode or other rank_op, fall back to original loop-based implementation
        # Helper function to apply vector operations
        def apply_op(vectors):
            """Apply mapping and aggregation to a window of vectors"""
            # Apply mapping matrix M to each vector in the window
            transformed = torch.matmul(vectors, self.M.T)  # [rank, m]
            
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
        
        # Linear mode (non-avg/sum) or nonlinear mode
        if self.mode == 'linear':
            L = vec_seq.shape[0]
            return [apply_op(vec_seq[i:i+self.rank]) for i in range(L - self.rank + 1)]
        
        # Nonlinear mode: stepping with custom step size
        vectors = []
        step = self.step or self.rank  # Use custom step if defined, else use rank length
        
        for i in range(0, vec_seq.shape[0], step):
            frag = vec_seq[i:i+self.rank]
            frag_len = frag.shape[0]
            
            # Pad or drop based on rank_mode setting
            if self.rank_mode == 'pad':
                # Pad fragment with zero vectors if shorter than rank
                if frag_len < self.rank:
                    padding = torch.zeros(self.rank - frag_len, self.m, device=self.device)
                    frag = torch.cat([frag, padding], dim=0)
                vectors.append(apply_op(frag)) 
            elif self.rank_mode == 'drop':
                # Only add fragments that match full rank length
                if frag_len == self.rank:
                    vectors.append(apply_op(frag)) 
        return vectors

    def _get_window_data(self, vec_seq):
        """
        Internal helper: returns window aggregated vectors and their positions (k values).
        Optimized version using vectorized operations for linear+avg/sum, otherwise loops.
        
        Args:
            vec_seq: tensor of shape (L, m)
            
        Returns:
            agg_vecs: torch.Tensor of shape (num_windows, m)
            k_positions: torch.Tensor of shape (num_windows,)
        """
        if not isinstance(vec_seq, torch.Tensor):
            vec_seq = torch.tensor(vec_seq, dtype=torch.float32, device=self.device)
        elif vec_seq.device != self.device:
            vec_seq = vec_seq.to(self.device)
        
        # Use the same logic as extract_vectors but return tensors
        if self.mode == 'linear' and self.rank_op in ('avg', 'sum'):
            L = vec_seq.shape[0]
            if L < self.rank:
                return torch.empty(0, self.m, device=self.device), torch.empty(0, device=self.device)
            
            # Prepare for unfold: shape (1, m, 1, L)
            seq_reshaped = vec_seq.unsqueeze(0).permute(0, 2, 1).unsqueeze(2)  # (1, m, 1, L)
            windows = F.unfold(seq_reshaped, kernel_size=(1, self.rank), stride=1)  # (1, m*rank, num_windows)
            num_windows = windows.shape[2]
            windows = windows.view(1, self.m, self.rank, num_windows).squeeze(0)  # (m, rank, num_windows)
            windows = windows.permute(2, 1, 0)  # (num_windows, rank, m)
            
            if self.rank_op == 'avg':
                agg = windows.mean(dim=1)
            else:
                agg = windows.sum(dim=1)
            
            # Apply M
            transformed = torch.matmul(agg, self.M.T)
            k_positions = torch.arange(num_windows, dtype=torch.float32, device=self.device)
            return transformed, k_positions
        
        # Fallback to loop for other modes
        agg_list = self.extract_vectors(vec_seq)  # returns list
        if not agg_list:
            return torch.empty(0, self.m, device=self.device), torch.empty(0, device=self.device)
        agg_tensor = torch.stack(agg_list)
        k_positions = torch.arange(len(agg_list), dtype=torch.float32, device=self.device)
        return agg_tensor, k_positions

    def batch_window_data(self, seq_batch):
        """
        Efficiently compute aggregated window vectors and their positions for a batch of sequences.
        If all sequences have the same length and conditions allow, uses vectorized window extraction.
        Otherwise, falls back to per-sequence extraction and concatenation.
        
        Args:
            seq_batch: torch.Tensor of shape [batch_size, seq_len, m]
        
        Returns:
            agg_flat: torch.Tensor of shape [total_windows, m]
            k_positions: torch.Tensor of shape [total_windows]
        """
        B, L, m = seq_batch.shape
        # Use vectorized extraction if conditions allow (linear mode, avg/sum rank_op, and length >= rank)
        if self.mode == 'linear' and self.rank_op in ('avg', 'sum') and L >= self.rank:
            # seq_batch: [B, L, m] -> [B, m, L] -> unsqueeze(2) -> [B, m, 1, L]
            seq_reshaped = seq_batch.permute(0, 2, 1).unsqueeze(2)  # [B, m, 1, L]
            windows = F.unfold(seq_reshaped, kernel_size=(1, self.rank), stride=1)  # [B, m*rank, N_w]
            N_w = windows.shape[2]  # number of windows per sequence
            windows = windows.view(B, self.m, self.rank, N_w)  # [B, m, rank, N_w]
            windows = windows.permute(0, 3, 2, 1)  # [B, N_w, rank, m]
            
            if self.rank_op == 'avg':
                agg = windows.mean(dim=2)   # [B, N_w, m]
            else:  # sum
                agg = windows.sum(dim=2)    # [B, N_w, m]
            
            # Apply mapping matrix M
            agg = torch.matmul(agg, self.M.T)  # [B, N_w, m]
            agg_flat = agg.reshape(-1, self.m)  # [B*N_w, m]
            # k positions: for each sequence, k = 0..N_w-1
            k_positions = torch.arange(N_w, device=self.device).unsqueeze(0).repeat(B, 1).reshape(-1).float()
            return agg_flat, k_positions
        else:
            # Fallback to per-sequence processing
            all_agg = []
            all_k = []
            for seq in seq_batch:
                agg, k = self._get_window_data(seq)
                if agg.shape[0] > 0:
                    all_agg.append(agg)
                    all_k.append(k)
            if all_agg:
                agg_tensor = torch.cat(all_agg, dim=0)
                k_tensor = torch.cat(all_k, dim=0)
                return agg_tensor, k_tensor
            else:
                return torch.empty(0, self.m, device=self.device), torch.empty(0, device=self.device)

    def batch_represent(self, seq_batch):
        """
        Compute sequence representations for a batch of equal-length vector sequences.
        This method computes the full N(k) representation (including A and B matrices).
        
        Args:
            seq_batch: torch.Tensor of shape [batch_size, seq_len, m]
        
        Returns:
            reps: torch.Tensor of shape [batch_size, m]
        """
        B, L, m = seq_batch.shape
        agg_flat, k_positions = self.batch_window_data(seq_batch)
        if agg_flat.shape[0] == 0:
            return torch.zeros(B, self.m, device=self.device)
        
        # Compute Nk for all windows
        j_indices = (k_positions % self.L).long()
        B_rows = self.Bbasis[j_indices]          # [total_w, m]
        scalar = torch.sum(B_rows * agg_flat, dim=1)  # [total_w]
        A_cols = self.Acoeff[:, j_indices].t()   # [total_w, m]
        Nk_all = A_cols * scalar.unsqueeze(1)     # [total_w, m]
        
        # Split back to sequences and average
        N_w = agg_flat.shape[0] // B
        Nk_seq = Nk_all.view(B, N_w, self.m)     # [B, N_w, m]
        reps = Nk_seq.mean(dim=1)                # [B, m]
        return reps

    def describe(self, vec_seq):
        """Compute N(k) vectors for each window in the vector sequence"""
        agg_vecs, k_positions = self._get_window_data(vec_seq)
        if agg_vecs.shape[0] == 0:
            return []
        
        j_indices = (k_positions % self.L).long()
        B_rows = self.Bbasis[j_indices]  # [num_windows, m]
        scalar = torch.sum(B_rows * agg_vecs, dim=1)  # [num_windows]
        A_cols = self.Acoeff[:, j_indices].t()  # [num_windows, m]
        Nk = A_cols * scalar.unsqueeze(1)  # [num_windows, m]
        return Nk.detach().cpu().numpy()
    
    def S(self, vec_seq):
        """Compute cumulative sum of N(k) vectors"""
        agg_vecs, k_positions = self._get_window_data(vec_seq)
        if agg_vecs.shape[0] == 0:
            return []
        
        j_indices = (k_positions % self.L).long()
        B_rows = self.Bbasis[j_indices]
        scalar = torch.sum(B_rows * agg_vecs, dim=1)
        A_cols = self.Acoeff[:, j_indices].t()
        Nk = A_cols * scalar.unsqueeze(1)
        S_cum = torch.cumsum(Nk, dim=0)
        return S_cum.detach().cpu().numpy()

    def D(self, vec_seqs, t_list):
        """
        Compute mean squared deviation D across vector sequences:
        D = average over all positions of (N(k)-t_seq)^2
        
        Args:
            vec_seqs: List of vector sequences (each is a tensor of shape [seq_len, m])
            t_list: List of target vectors corresponding to each sequence
            
        Returns:
            float: Average mean squared deviation across all positions and sequences
        """
        total_loss = 0.0
        total_windows = 0
        t_tensors = [torch.tensor(t, dtype=torch.float32, device=self.device) for t in t_list]
        
        for vec_seq, t in zip(vec_seqs, t_tensors):
            agg_vecs, k_positions = self._get_window_data(vec_seq)
            if agg_vecs.shape[0] == 0:
                continue
            num_windows = agg_vecs.shape[0]
            j_indices = (k_positions % self.L).long()
            B_rows = self.Bbasis[j_indices]
            scalar = torch.sum(B_rows * agg_vecs, dim=1)
            A_cols = self.Acoeff[:, j_indices].t()
            Nk_batch = A_cols * scalar.unsqueeze(1)
            losses = torch.sum((Nk_batch - t) ** 2, dim=1)
            total_loss += losses.sum().item()
            total_windows += num_windows
        return total_loss / total_windows if total_windows else 0.0

    def d(self, vec_seq, t):
        """Compute pattern deviation value (d) for a single vector sequence."""
        return self.D([vec_seq], [t])

    def reg_train(self, vec_seqs, t_list, max_iters=1000, tol=1e-8, learning_rate=0.01, 
                  continued=False, decay_rate=1.0, print_every=10, batch_size=32,
                  checkpoint_file=None, checkpoint_interval=10):
        """
        Train model using gradient descent with batch processing.
        Optimized: uses batch_represent (full N(k) computation) when sequences in a batch have equal length.
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
        
        # Convert targets to tensors and ensure sequences on correct device
        t_tensors = [torch.tensor(t, dtype=torch.float32, device=self.device) for t in t_list]
        vec_seqs = [seq.to(self.device) if isinstance(seq, torch.Tensor) else 
                    torch.tensor(seq, dtype=torch.float32, device=self.device) 
                    for seq in vec_seqs]
        
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)
        if continued and checkpoint_file and os.path.exists(checkpoint_file):
            optimizer.load_state_dict(optimizer_state)
            scheduler.load_state_dict(scheduler_state)
        
        prev_loss = float('inf') if start_iter == 0 else history[-1] if history else float('inf')
        
        for it in range(start_iter, max_iters):
            total_loss = 0.0
            total_sequences = 0
            
            indices = list(range(len(vec_seqs)))
            random.shuffle(indices)
            
            for batch_start in range(0, len(indices), batch_size):
                batch_indices = indices[batch_start:batch_start + batch_size]
                batch_seqs = [vec_seqs[idx] for idx in batch_indices]
                batch_targets = [t_tensors[idx] for idx in batch_indices]
                
                optimizer.zero_grad()
                
                # Check if all sequences in the batch have the same length
                seq_lengths = [seq.shape[0] for seq in batch_seqs]
                if len(set(seq_lengths)) == 1:
                    # Equal length: use efficient batch processing with full N(k) computation
                    seq_tensor = torch.stack(batch_seqs, dim=0)  # [B, L, m]
                    reps = self.batch_represent(seq_tensor)      # [B, m] (full N(k) representation)
                    targets_tensor = torch.stack(batch_targets, dim=0)
                    loss = torch.mean(torch.sum((reps - targets_tensor) ** 2, dim=1))
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item() * len(batch_indices)
                    total_sequences += len(batch_indices)
                else:
                    # Unequal lengths: fallback to per-sequence processing
                    # Collect all window data from sequences in this batch
                    all_agg = []
                    all_k = []
                    seq_window_counts = []
                    for seq in batch_seqs:
                        agg_vecs, k_positions = self._get_window_data(seq)
                        if agg_vecs.shape[0] == 0:
                            seq_window_counts.append(0)
                            continue
                        all_agg.append(agg_vecs)
                        all_k.append(k_positions)
                        seq_window_counts.append(agg_vecs.shape[0])
                    
                    if not all_agg:
                        continue
                    
                    all_agg = torch.cat(all_agg, dim=0)
                    all_k = torch.cat(all_k, dim=0)
                    
                    j_indices = (all_k % self.L).long()
                    B_rows = self.Bbasis[j_indices]
                    scalar = torch.sum(B_rows * all_agg, dim=1)
                    A_cols = self.Acoeff[:, j_indices].t()
                    Nk_all = A_cols * scalar.unsqueeze(1)
                    
                    seq_reps = []
                    start_idx = 0
                    for nw in seq_window_counts:
                        if nw > 0:
                            seq_rep = Nk_all[start_idx:start_idx+nw].mean(dim=0)
                            start_idx += nw
                        else:
                            seq_rep = torch.zeros(self.m, device=self.device)
                        seq_reps.append(seq_rep)
                    
                    if seq_reps:
                        seq_reps_tensor = torch.stack(seq_reps)
                        targets_tensor = torch.stack(batch_targets)
                        loss = torch.mean(torch.sum((seq_reps_tensor - targets_tensor) ** 2, dim=1))
                        loss.backward()
                        optimizer.step()
                        total_loss += loss.item() * len(seq_reps)
                        total_sequences += len(seq_reps)
                
                # Clear GPU cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            avg_loss = total_loss / total_sequences if total_sequences else 0.0
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
        
        if best_model_state is not None and best_loss < history[-1]:
            self.load_state_dict(best_model_state)
            print(f"Training ended. Restored best model state with loss = {best_loss:.6e}")
            history[-1] = best_loss
        
        self._compute_training_statistics(vec_seqs)
        self.trained = True
        return history

    def cls_train(self, vec_seqs, labels, num_classes, max_iters=1000, tol=1e-8, learning_rate=0.01,
                  continued=False, decay_rate=1.0, print_every=10, batch_size=32,
                  checkpoint_file=None, checkpoint_interval=10):
        """Optimized classification training using batch_represent (full N(k)) for equal-length sequences."""
        if self.classifier is None or self.num_classes != num_classes:
            self.classifier = nn.Linear(self.m, num_classes).to(self.device)
            self.num_classes = num_classes
        
        if not continued:
            self.reset_parameters()
        
        label_tensors = torch.tensor(labels, dtype=torch.long, device=self.device)
        vec_seqs = [seq.to(self.device) if isinstance(seq, torch.Tensor) else 
                    torch.tensor(seq, dtype=torch.float32, device=self.device) 
                    for seq in vec_seqs]
        
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)
        criterion = nn.CrossEntropyLoss()
        
        history = []
        prev_loss = float('inf')
        best_loss = float('inf')
        best_model_state = None
        
        for it in range(max_iters):
            total_loss = 0.0
            total_sequences = 0
            correct_predictions = 0
            
            indices = list(range(len(vec_seqs)))
            random.shuffle(indices)
            
            for batch_start in range(0, len(indices), batch_size):
                batch_indices = indices[batch_start:batch_start + batch_size]
                batch_seqs = [vec_seqs[idx] for idx in batch_indices]
                batch_labels = label_tensors[batch_indices]
                
                optimizer.zero_grad()
                
                # Check if all sequences in the batch have the same length
                seq_lengths = [seq.shape[0] for seq in batch_seqs]
                if len(set(seq_lengths)) == 1:
                    # Equal length: use batch_represent (full N(k) representation)
                    seq_tensor = torch.stack(batch_seqs, dim=0)  # [B, L, m]
                    reps = self.batch_represent(seq_tensor)      # [B, m]
                    logits = self.classifier(reps)
                    loss = criterion(logits, batch_labels)
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item() * len(batch_indices)
                    total_sequences += len(batch_indices)
                    with torch.no_grad():
                        predictions = torch.argmax(logits, dim=1)
                        correct_predictions += (predictions == batch_labels).sum().item()
                else:
                    # Unequal lengths: fallback to per-sequence processing
                    all_agg = []
                    all_k = []
                    seq_window_counts = []
                    for seq in batch_seqs:
                        agg_vecs, k_positions = self._get_window_data(seq)
                        if agg_vecs.shape[0] == 0:
                            seq_window_counts.append(0)
                            continue
                        all_agg.append(agg_vecs)
                        all_k.append(k_positions)
                        seq_window_counts.append(agg_vecs.shape[0])
                    
                    if not all_agg:
                        continue
                    
                    all_agg = torch.cat(all_agg, dim=0)
                    all_k = torch.cat(all_k, dim=0)
                    
                    j_indices = (all_k % self.L).long()
                    B_rows = self.Bbasis[j_indices]
                    scalar = torch.sum(B_rows * all_agg, dim=1)
                    A_cols = self.Acoeff[:, j_indices].t()
                    Nk_all = A_cols * scalar.unsqueeze(1)
                    
                    seq_reps = []
                    start_idx = 0
                    for nw in seq_window_counts:
                        if nw > 0:
                            seq_rep = Nk_all[start_idx:start_idx+nw].mean(dim=0)
                            start_idx += nw
                        else:
                            seq_rep = torch.zeros(self.m, device=self.device)
                        seq_reps.append(seq_rep)
                    
                    if seq_reps:
                        seq_reps_tensor = torch.stack(seq_reps)
                        logits = self.classifier(seq_reps_tensor)
                        loss = criterion(logits, batch_labels)
                        loss.backward()
                        optimizer.step()
                        
                        total_loss += loss.item() * len(seq_reps)
                        total_sequences += len(seq_reps)
                        with torch.no_grad():
                            predictions = torch.argmax(logits, dim=1)
                            correct_predictions += (predictions == batch_labels).sum().item()
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            avg_loss = total_loss / total_sequences if total_sequences else 0.0
            accuracy = correct_predictions / total_sequences if total_sequences else 0.0
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

    def lbl_train(self, vec_seqs, labels, num_labels, max_iters=1000, tol=1e-8, learning_rate=0.01, 
                 continued=False, decay_rate=1.0, print_every=10, batch_size=32,
                 checkpoint_file=None, checkpoint_interval=10, pos_weight=None):
        """Optimized multi-label classification training with batch_represent (full N(k)) for equal-length sequences."""
        if self.labeller is None or self.num_labels != num_labels:
            self.labeller = nn.Linear(self.m, num_labels).to(self.device)
            self.num_labels = num_labels
        
        if not continued:
            self.reset_parameters()
        
        if isinstance(labels, list):
            labels_tensor = torch.tensor(labels, dtype=torch.float32, device=self.device)
        else:
            labels_tensor = torch.as_tensor(labels, dtype=torch.float32, device=self.device)
        
        vec_seqs = [seq.to(self.device) if isinstance(seq, torch.Tensor) else 
                    torch.tensor(seq, dtype=torch.float32, device=self.device) 
                    for seq in vec_seqs]
        
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
            total_sequences = 0
            
            indices = list(range(len(vec_seqs)))
            random.shuffle(indices)
            
            for batch_start in range(0, len(indices), batch_size):
                batch_indices = indices[batch_start:batch_start + batch_size]
                batch_seqs = [vec_seqs[idx] for idx in batch_indices]
                batch_labels = labels_tensor[batch_indices]
                
                optimizer.zero_grad()
                
                # Check if all sequences in the batch have the same length
                seq_lengths = [seq.shape[0] for seq in batch_seqs]
                if len(set(seq_lengths)) == 1:
                    # Equal length: use batch_represent (full N(k) representation)
                    seq_tensor = torch.stack(batch_seqs, dim=0)  # [B, L, m]
                    reps = self.batch_represent(seq_tensor)      # [B, m]
                    logits = self.labeller(reps)
                    loss = criterion(logits, batch_labels)
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item() * len(batch_indices)
                    total_sequences += len(batch_indices)
                    with torch.no_grad():
                        probs = torch.sigmoid(logits)
                        predictions = (probs > 0.5).float()
                        total_correct += (predictions == batch_labels).sum().item()
                        total_predictions += batch_labels.numel()
                else:
                    # Unequal lengths: fallback to per-sequence processing
                    all_agg = []
                    all_k = []
                    seq_window_counts = []
                    for seq in batch_seqs:
                        agg_vecs, k_positions = self._get_window_data(seq)
                        if agg_vecs.shape[0] == 0:
                            seq_window_counts.append(0)
                            continue
                        all_agg.append(agg_vecs)
                        all_k.append(k_positions)
                        seq_window_counts.append(agg_vecs.shape[0])
                    
                    if not all_agg:
                        continue
                    
                    all_agg = torch.cat(all_agg, dim=0)
                    all_k = torch.cat(all_k, dim=0)
                    
                    j_indices = (all_k % self.L).long()
                    B_rows = self.Bbasis[j_indices]
                    scalar = torch.sum(B_rows * all_agg, dim=1)
                    A_cols = self.Acoeff[:, j_indices].t()
                    Nk_all = A_cols * scalar.unsqueeze(1)
                    
                    seq_reps = []
                    start_idx = 0
                    for nw in seq_window_counts:
                        if nw > 0:
                            seq_rep = Nk_all[start_idx:start_idx+nw].mean(dim=0)
                            start_idx += nw
                        else:
                            seq_rep = torch.zeros(self.m, device=self.device)
                        seq_reps.append(seq_rep)
                    
                    if seq_reps:
                        seq_reps_tensor = torch.stack(seq_reps)
                        logits = self.labeller(seq_reps_tensor)
                        loss = criterion(logits, batch_labels)
                        loss.backward()
                        optimizer.step()
                        
                        total_loss += loss.item() * len(seq_reps)
                        total_sequences += len(seq_reps)
                        with torch.no_grad():
                            probs = torch.sigmoid(logits)
                            predictions = (probs > 0.5).float()
                            total_correct += (predictions == batch_labels).sum().item()
                            total_predictions += batch_labels.numel()
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            avg_loss = total_loss / total_sequences if total_sequences else 0.0
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

    def self_train(self, vec_seqs, max_iters=1000, tol=1e-8, learning_rate=0.01, 
                   continued=False, decay_rate=1.0, print_every=10, batch_size=32, 
                   checkpoint_file=None, checkpoint_interval=10):
        """
        Train the model using self-supervised learning with batch acceleration for equal-length sequences.
        For each batch, if all sequences have the same length, uses vectorized batch_window_data.
        """
        # Load checkpoint if continuing and checkpoint file exists
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
        
        # Ensure all input sequences are on the correct device
        vec_seqs = [seq.to(self.device) if isinstance(seq, torch.Tensor) else 
                    torch.tensor(seq, dtype=torch.float32, device=self.device) 
                    for seq in vec_seqs]
        
        # Set up optimizer and scheduler
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)
        
        # Load optimizer and scheduler states if resuming
        if continued and checkpoint_file and os.path.exists(checkpoint_file):
            optimizer.load_state_dict(optimizer_state)
            scheduler.load_state_dict(scheduler_state)
        
        prev_loss = float('inf') if start_iter == 0 else history[-1] if history else float('inf')
        
        for it in range(start_iter, max_iters):
            total_loss = 0.0
            total_samples = 0
            
            # Shuffle sequences for each epoch
            indices = list(range(len(vec_seqs)))
            random.shuffle(indices)
            
            # Process sequences in batches
            for batch_start in range(0, len(indices), batch_size):
                batch_indices = indices[batch_start:batch_start + batch_size]
                batch_seqs = [vec_seqs[idx] for idx in batch_indices]
                
                optimizer.zero_grad()
                
                # Check if all sequences in the batch have the same length
                seq_lengths = [seq.shape[0] for seq in batch_seqs]
                if len(set(seq_lengths)) == 1:
                    # Equal length: use batch_window_data for all windows at once
                    seq_tensor = torch.stack(batch_seqs, dim=0)  # [B, L, m]
                    agg_flat, k_positions = self.batch_window_data(seq_tensor)  # [total_windows, m], [total_windows]
                    if agg_flat.shape[0] == 0:
                        continue
                    
                    # Compute Nk for all windows
                    j_indices = (k_positions % self.L).long()
                    B_rows = self.Bbasis[j_indices]
                    scalar = torch.sum(B_rows * agg_flat, dim=1)
                    A_cols = self.Acoeff[:, j_indices].t()
                    Nk_all = A_cols * scalar.unsqueeze(1)
                    
                    # Self-supervised loss: each window should reconstruct itself
                    loss = torch.mean(torch.sum((Nk_all - agg_flat) ** 2, dim=1))
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item() * agg_flat.size(0)
                    total_samples += agg_flat.size(0)
                else:
                    # Unequal lengths: fallback to per-sequence processing
                    for vec_seq in batch_seqs:
                        agg_vecs, k_positions = self._get_window_data(vec_seq)
                        if agg_vecs.shape[0] == 0:
                            continue
                        j_indices = (k_positions % self.L).long()
                        B_rows = self.Bbasis[j_indices]
                        scalar = torch.sum(B_rows * agg_vecs, dim=1)
                        A_cols = self.Acoeff[:, j_indices].t()
                        Nk_all = A_cols * scalar.unsqueeze(1)
                        loss = torch.mean(torch.sum((Nk_all - agg_vecs) ** 2, dim=1))
                        loss.backward()
                        optimizer.step()
                        total_loss += loss.item() * agg_vecs.size(0)
                        total_samples += agg_vecs.size(0)
                
                # Clear GPU cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Calculate average loss for this iteration
            avg_loss = total_loss / total_samples if total_samples else 0.0
            history.append(avg_loss)
            
            # Update best model state if current loss is lower
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_model_state = copy.deepcopy(self.state_dict())
            
            # Print training progress
            if it % print_every == 0 or it == max_iters - 1:
                current_lr = scheduler.get_last_lr()[0]
                print(f"SelfTrain Iter {it:3d}: loss = {avg_loss:.6f}, LR = {current_lr:.6f}, "
                      f"Samples = {total_samples}")
            
            # Save checkpoint at specified intervals
            if checkpoint_file and (it % checkpoint_interval == 0 or it == max_iters - 1):
                self._save_checkpoint(
                    checkpoint_file, it, history, optimizer, scheduler, best_loss
                )
            
            # Check convergence
            if abs(prev_loss - avg_loss) < tol:
                print(f"Converged after {it+1} iterations")
                # Restore the best model state before breaking
                if best_model_state is not None and avg_loss > prev_loss:
                    self.load_state_dict(best_model_state)
                    print(f"Restored best model state with loss = {best_loss:.6f}")
                    history[-1] = best_loss
                break
            prev_loss = avg_loss
            
            # Update learning rate
            scheduler.step()
            
            # Final GPU memory cleanup for this iteration
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # If training ended without convergence, restore best model state
        if best_model_state is not None and best_loss < history[-1]:
            self.load_state_dict(best_model_state)
            print(f"Training ended. Restored best model state with loss = {best_loss:.6f}")
            history[-1] = best_loss
        
        # Compute and store statistics for reconstruction/generation
        self._compute_training_statistics(vec_seqs)
        self.trained = True
        
        return history

    def _save_checkpoint(self, checkpoint_file, iteration, history, optimizer, scheduler, best_loss):
        """Save training checkpoint with complete training state"""
        checkpoint = {
            'iteration': iteration,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'history': history,
            'best_loss': best_loss,
            'config': {
                'vec_dim': self.m,
                'bas_dim': self.L,
                'rank': self.rank,
                'rank_op': self.rank_op,
                'rank_mode': self.rank_mode,
                'mode': self.mode,
                'user_step': self.step
            },
            'training_stats': {
                'mean_t': self.mean_t if hasattr(self, 'mean_t') else None,
                'mean_token_count': self.mean_token_count if hasattr(self, 'mean_token_count') else None
            }
        }
        torch.save(checkpoint, checkpoint_file)
        print(f"Checkpoint saved at iteration {iteration} to {checkpoint_file}")

    def _compute_training_statistics(self, vec_seqs, batch_size=50):
        """Compute and store statistics for reconstruction and generation."""
        total_window_count = 0
        total_t = torch.zeros(self.m, device=self.device)
        
        with torch.no_grad():
            for i in range(0, len(vec_seqs), batch_size):
                batch_seqs = vec_seqs[i:i+batch_size]
                batch_window_count = 0
                batch_t_sum = torch.zeros(self.m, device=self.device)
                
                for vec_seq in batch_seqs:
                    agg_vecs, k_positions = self._get_window_data(vec_seq)
                    batch_window_count += agg_vecs.shape[0]
                    if agg_vecs.shape[0] == 0:
                        continue
                    j_indices = (k_positions % self.L).long()
                    B_rows = self.Bbasis[j_indices]
                    scalar = torch.sum(B_rows * agg_vecs, dim=1)
                    A_cols = self.Acoeff[:, j_indices].t()
                    Nk_batch = A_cols * scalar.unsqueeze(1)
                    batch_t_sum += Nk_batch.sum(dim=0)
                
                total_window_count += batch_window_count
                total_t += batch_t_sum
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        self.mean_window_count = total_window_count / len(vec_seqs) if vec_seqs else 0
        self.mean_t = (total_t / total_window_count).cpu().numpy() if total_window_count > 0 else np.zeros(self.m)

    def predict_t(self, vec_seq):
        """Predict target vector as average of N(k) vectors"""
        agg_vecs, k_positions = self._get_window_data(vec_seq)
        if agg_vecs.shape[0] == 0:
            return np.zeros(self.m)
        j_indices = (k_positions % self.L).long()
        B_rows = self.Bbasis[j_indices]
        scalar = torch.sum(B_rows * agg_vecs, dim=1)
        A_cols = self.Acoeff[:, j_indices].t()
        Nk_batch = A_cols * scalar.unsqueeze(1)
        return torch.mean(Nk_batch, dim=0).detach().cpu().numpy()

    def predict_c(self, vec_seq):
        """Predict class label for a vector sequence."""
        if self.classifier is None:
            raise ValueError("Model must be trained first for classification")
        seq_vector = self.predict_t(vec_seq)
        seq_vector_tensor = torch.tensor(seq_vector, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits = self.classifier(seq_vector_tensor.unsqueeze(0))
            probabilities = torch.softmax(logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
        return predicted_class, probabilities[0].cpu().numpy()

    def predict_l(self, vec_seq, threshold=0.5):
        """Predict multi-label classification for a vector sequence."""
        assert self.labeller is not None, "Model must be trained first for label prediction"
        agg_vecs, k_positions = self._get_window_data(vec_seq)
        if agg_vecs.shape[0] == 0:
            return np.zeros(self.num_labels, dtype=np.float32), np.zeros(self.num_labels, dtype=np.float32)
        j_indices = (k_positions % self.L).long()
        B_rows = self.Bbasis[j_indices]
        scalar = torch.sum(B_rows * agg_vecs, dim=1)
        A_cols = self.Acoeff[:, j_indices].t()
        Nk_batch = A_cols * scalar.unsqueeze(1)
        seq_representation = torch.mean(Nk_batch, dim=0)
        with torch.no_grad():
            logits = self.labeller(seq_representation)
            probs = torch.sigmoid(logits).cpu().numpy()
        binary_preds = (probs > threshold).astype(np.float32)
        return binary_preds, probs

    def reconstruct(self, L, tau=0.0):
        """Reconstruct representative vector sequence of length L with temperature-controlled randomness"""
        assert self.trained, "Model must be trained first"
        if tau < 0:
            raise ValueError("Temperature must be non-negative")
            
        num_windows = (L + self.rank - 1) // self.rank
        mean_t_tensor = torch.tensor(self.mean_t, dtype=torch.float32, device=self.device)
        
        candidate_vecs = torch.randn(100, self.m, device=self.device)
        generated_vectors = []
        
        for k in range(num_windows):
            j = k % self.L
            B_row = self.Bbasis[j].unsqueeze(0)
            transformed_candidates = torch.matmul(candidate_vecs, self.M.T)
            
            if self.rank_op == 'sum':
                expanded_candidates = transformed_candidates.unsqueeze(1).repeat(1, self.rank, 1)
                aggregated_candidates = torch.sum(expanded_candidates, dim=1)
            elif self.rank_op == 'pick':
                idx = random.randint(0, self.rank-1)
                expanded_candidates = transformed_candidates.unsqueeze(1).repeat(1, self.rank, 1)
                aggregated_candidates = expanded_candidates[:, idx, :]
            elif self.rank_op == 'user_func':
                expanded_candidates = transformed_candidates.unsqueeze(1).repeat(1, self.rank, 1)
                avg = torch.mean(expanded_candidates, dim=1)
                aggregated_candidates = torch.sigmoid(avg)
            else:  # 'avg'
                expanded_candidates = transformed_candidates.unsqueeze(1).repeat(1, self.rank, 1)
                aggregated_candidates = torch.mean(expanded_candidates, dim=1)
            
            scalar = torch.sum(B_row * aggregated_candidates, dim=1)
            A_col = self.Acoeff[:, j]
            Nk_all = A_col * scalar.unsqueeze(1)
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
        
        full_seq = torch.stack(generated_vectors, dim=0)
        return full_seq[:L]

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
                'bas_dim': self.L,
                'rank': self.rank,
                'rank_op': self.rank_op,
                'rank_mode': self.rank_mode,
                'mode': self.mode,
                'user_step': self.step
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
            self.step = config.get('user_step', None)
        
        if self.num_classes is not None:
            self.classifier = nn.Linear(self.m, self.num_classes).to(self.device)
        if self.num_labels is not None:
            self.labeller = nn.Linear(self.m, self.num_labels).to(self.device)
            
        print(f"Model loaded from {filename}")
        return self


# === Example Usage ===
if __name__ == "__main__":

    from statistics import correlation
    
    print("="*50)
    print("Numeric Dual Descriptor AB - PyTorch GPU Accelerated Version")
    print("Optimized for vector sequence processing with equal-length batch acceleration (fixed)")
    print("="*50)
    
    # Set random seeds for reproducibility
    torch.manual_seed(11)
    random.seed(11)
    
    vec_dim = 3
    bas_dim = 300
    seq_num = 100
    
    # Generate vector sequences and random targets
    vec_seqs, t_list = [], []
    for _ in range(seq_num):
        L = random.randint(200, 300)
        seq = torch.randn(L, vec_dim)  # Random vector sequence
        vec_seqs.append(seq)
        t_list.append([random.uniform(-1,1) for _ in range(vec_dim)])
    
    # Create model
    dd = NumDualDescriptorAB(
        vec_dim=vec_dim, 
        bas_dim=bas_dim, 
        rank=1,         
        mode='linear',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Display device information
    print(f"\nUsing device: {dd.device}")
    print(f"Vector dimension: {dd.m}")
    print(f"Basis dimension: {dd.L}")    
    
    # === Gradient Descent Training ===
    print("\n" + "="*50)
    print("Testing Gradient Descent Training")
    print("="*50)
    
    # Train using gradient descent
    print("\nStarting gradient descent training...")
    reg_history = dd.reg_train(
        vec_seqs, 
        t_list,
        learning_rate=0.1,
        max_iters=50,
        tol=1e-66,
        print_every=5,
        decay_rate=0.99,
        batch_size=1024
    )
    
    # Predict target for first sequence
    aseq = vec_seqs[0]
    t_pred = dd.predict_t(aseq)
    print(f"\nPredicted t for first sequence: {[round(x.item(), 4) for x in t_pred]}")
    
    # Calculate prediction correlation
    pred_t_list = [dd.predict_t(seq) for seq in vec_seqs]
    
    corr_sum = 0.0
    for i in range(dd.m):
        actu_t = [t_vec[i] for t_vec in t_list]
        pred_t = [t_vec[i] for t_vec in pred_t_list]
        corr = correlation(actu_t, pred_t)
        print(f"Prediction correlation for dimension {i}: {corr:.4f}")
        corr_sum += corr
    corr_avg = corr_sum / dd.m
    print(f"Average correlation: {corr_avg:.4f}")   
    
    # Reconstruct representative sequences
    seq_det = dd.reconstruct(L=100, tau=0.0)
    seq_rand = dd.reconstruct(L=100, tau=0.5)
    print(f"\nDeterministic Reconstruction shape: {seq_det.shape}")
    print(f"Stochastic Reconstruction shape (tau=0.5): {seq_rand.shape}")  
    
    # === Classification Task ===
    print("\n" + "="*50)
    print("Classification Task")
    print("="*50)
    
    # Generate classification data
    num_classes = 3
    class_seqs = []
    class_labels = []
    
    # Create sequences with different patterns for each class
    for class_id in range(num_classes):
        for _ in range(50):  # 50 sequences per class
            L = random.randint(150, 250)
            if class_id == 0:
                # Class 0: Pattern with positive mean
                seq = torch.randn(L, vec_dim) + 0.5
            elif class_id == 1:
                # Class 1: Pattern with negative mean
                seq = torch.randn(L, vec_dim) - 0.5
            else:
                # Class 2: Normal distribution
                seq = torch.randn(L, vec_dim)
            
            class_seqs.append(seq)
            class_labels.append(class_id)
    
    # Initialize new model for classification
    dd_cls = NumDualDescriptorAB(
        vec_dim=vec_dim, 
        bas_dim=bas_dim, 
        rank=1,        
        mode='linear',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Train for classification
    print("\n" + "="*50)
    print("Starting Classification Training")
    print("="*50)
    history = dd_cls.cls_train(class_seqs, class_labels, num_classes, 
                              max_iters=10, tol=1e-8, learning_rate=0.05,
                              decay_rate=0.99, batch_size=32, print_every=1)
    
    # Show prediction results on the training dataset
    print("\n" + "="*50)
    print("Prediction results")
    print("="*50)
    
    correct = 0
    all_predictions = []
    
    for seq, true_label in zip(class_seqs, class_labels):
        pred_class, probs = dd_cls.predict_c(seq)
        all_predictions.append(pred_class)
        
        if pred_class == true_label:
            correct += 1
    
    accuracy = correct / len(class_seqs)
    print(f"Accuracy: {accuracy:.4f} ({correct}/{len(class_seqs)})")
    
    # Show some example predictions
    print("\nExample predictions:")
    for i in range(min(5, len(class_seqs))):
        pred_class, probs = dd_cls.predict_c(class_seqs[i])
        print(f"Seq {i+1}: True={class_labels[i]}, Pred={pred_class}, Probs={[f'{p:.3f}' for p in probs]}")
    
    # === Multi-Label Classification Task ===
    print("\n\n" + "="*50)
    print("Multi-Label Classification Model")
    print("="*50)
    
    # Generate 100 sequences with random multi-labels for classification
    num_labels = 4  # Example: 4 different patterns
    label_seqs = []
    labels = []
    for _ in range(100):
        L = random.randint(200, 300)
        seq = torch.randn(L, vec_dim)
        label_seqs.append(seq)
        # Create random binary labels (multi-label classification)
        # Each sequence can have 0-4 active labels
        label_vec = [random.random() > 0.7 for _ in range(num_labels)]
        labels.append([1.0 if x else 0.0 for x in label_vec])
    
    dd_lbl = NumDualDescriptorAB(
        vec_dim=vec_dim, 
        bas_dim=bas_dim, 
        rank=1,        
        mode='linear',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Training multi-label classification model
    print("\n" + "="*50)
    print("Starting Gradient Descent Training for Multi-Label Classification")
    print("="*50)
       
    # Train the model
    loss_history, acc_history = dd_lbl.lbl_train(
        label_seqs, labels, num_labels,
        max_iters=50, 
        tol=1e-16, 
        learning_rate=0.05, 
        decay_rate=0.99, 
        print_every=10, 
        batch_size=32
    )
    
    print(f"\nFinal training loss: {loss_history[-1]:.6f}")
    print(f"Final training accuracy: {acc_history[-1]:.4f}")
    
    # Show prediction results on training set
    print("\n" + "="*50)
    print("Prediction Results")
    print("="*50)
    
    all_correct = 0
    total = 0
    
    for seq, true_labels in zip(label_seqs, labels):
        pred_binary, pred_probs = dd_lbl.predict_l(seq, threshold=0.5)
        
        # Convert true labels to numpy array
        true_labels_np = np.array(true_labels)
        
        # Calculate accuracy for this sequence
        correct = np.all(pred_binary == true_labels_np)
        all_correct += correct
        total += 1
        
        # Print detailed results for first few sequences
        if total <= 3:
            print(f"\nSequence {total}:")
            print(f"True labels: {true_labels_np}")
            print(f"Predicted binary: {pred_binary}")
            print(f"Predicted probabilities: {[f'{p:.4f}' for p in pred_probs]}")
            print(f"Correct: {correct}")
    
    accuracy = all_correct / total if total > 0 else 0.0
    print(f"\nOverall prediction accuracy: {accuracy:.4f} ({all_correct}/{total} sequences)")
    
    # Example of label prediction for a new sequence
    print("\n" + "="*50)
    print("Label Prediction Example")
    print("="*50)
    
    # Create a test sequence
    test_seq = torch.randn(250, vec_dim)
    print(f"Test sequence shape: {test_seq.shape}")
    
    # Predict labels
    binary_pred, probs_pred = dd_lbl.predict_l(test_seq, threshold=0.5)
    print(f"\nPredicted binary labels: {binary_pred}")
    print(f"Predicted probabilities: {[f'{p:.4f}' for p in probs_pred]}")
    
    # Interpret the predictions
    label_names = ["Pattern_A", "Pattern_B", "Pattern_C", "Pattern_D"]
    print("\nLabel interpretation:")
    for i, (binary, prob) in enumerate(zip(binary_pred, probs_pred)):
        status = "ACTIVE" if binary > 0.5 else "INACTIVE"
        print(f"  {label_names[i]}: {status} (confidence: {prob:.4f})")

    # === Self-Training Example ===
    # Set random seeds
    torch.manual_seed(2)
    random.seed(2)
    
    # Define parameters
    vec_dim = 3
    bas_dim = 100
    
    # Generate training sequences
    print("\n=== Generating Training Sequences ===")
    vec_seqs = []
    for i in range(30):
        L = random.randint(100, 200)
        seq = torch.randn(L, vec_dim)
        vec_seqs.append(seq)
    
    # Create model for self-training
    print("\n=== Creating Numeric Dual Descriptor Model for Self-Training ===")
    dd_self = NumDualDescriptorAB(
        vec_dim=vec_dim,
        bas_dim=bas_dim,
        rank=1,        
        mode='linear',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Train in self-consistency mode 
    print("\n=== Starting self-consistency training ===")
    self_history = dd_self.self_train(
        vec_seqs,
        max_iters=100,
        learning_rate=0.001,
        decay_rate=0.995,
        print_every=20,
        batch_size=32  # Reduced from 1024 to avoid memory issues
    ) 
    
    # Reconstruct sequence from self-trained model
    print("\n=== Reconstructing Sequence from Self-Trained Model ===")
    self_seq = dd_self.reconstruct(L=40, tau=0.0)
    print(f"Self-trained model Reconstruction shape: {self_seq.shape}")   
    
    # Save and load model
    print("\n=== Model Persistence Test ===")
    dd_self.save("self_trained_model.pkl")
    
    # Load model
    dd_loaded = NumDualDescriptorAB(
        vec_dim=vec_dim,
        bas_dim=bas_dim,
        rank=1,        
        mode='linear',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    ).load("self_trained_model.pkl")
    
    print("Model loaded successfully. Reconstructing with loaded model:")
    print(dd_loaded.reconstruct(L=20, tau=0.0).shape)   

    print("\n=== All Tests Completed ===")
