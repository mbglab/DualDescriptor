# Copyright (C) 2005-2026, Bin-Guang Ma (mbg@mail.hzau.edu.cn); SPDX-License-Identifier: MIT
# The Numeric Dual Descriptor Vector class (Random AB matrix form) implemented with PyTorch
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

class NumDualDescriptorRN(nn.Module):
    """
    Dual Descriptor for numerical vector sequences with GPU acceleration using PyTorch:
      - Learnable coefficient matrix Acoeff ∈ R^{m×L}
      - Learnable basis matrix Bbasis ∈ R^{L×m}
      - Learnable mapping matrix M ∈ R^{m×m} (replaces token embeddings)
      - Input: sequences of m-dimensional vectors
      - Optimized with batch processing for GPU acceleration
      - Supports both linear and nonlinear vector windowing
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
        self.m = vec_dim  # Vector dimension
        self.L = bas_dim    # Basis dimension
        self.rank = rank    # Window size for aggregation
        self.rank_op = rank_op
        self.rank_mode = rank_mode
        assert mode in ('linear','nonlinear')
        self.mode = mode
        self.step = user_step
        self.trained = False
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # No tokens needed for numerical vectors
        self.tokens = []
        self.token_to_idx = {}
        self.idx_to_token = {}
        
        # Mapping matrix M: m×m (replaces token embeddings)
        self.M = nn.Parameter(torch.empty(self.m, self.m))
        
        # Coefficient matrix Acoeff: m×L
        self.Acoeff = nn.Parameter(torch.empty(self.m, self.L))
        
        # Basis matrix Bbasis: L×m
        self.Bbasis = nn.Parameter(torch.empty(self.L, self.m))
        
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
        nn.init.uniform_(self.M, -0.5, 0.5)          # Initialize mapping matrix
        nn.init.uniform_(self.Acoeff, -0.1, 0.1)    # Initialize coefficient matrix
        nn.init.uniform_(self.Bbasis, -0.1, 0.1)    # Initialize basis matrix
        
        # Initialize classifier if it exists
        if self.classifier is not None:
            nn.init.normal_(self.classifier.weight, 0, 0.01)
            if self.classifier.bias is not None:
                nn.init.constant_(self.classifier.bias, 0)
                
        # Initialize labeller if it exists
        if self.labeller is not None:
            nn.init.xavier_uniform_(self.labeller.weight)
            nn.init.zeros_(self.labeller.bias)
    
    def map_vector(self, vector):
        """
        Apply mapping matrix M to input vector.
        
        Args:
            vector (torch.Tensor): Input vector of shape [batch_size, m] or [m]
            
        Returns:
            torch.Tensor: Mapped vector of same shape as input
        """
        if vector.dim() == 1:
            # Single vector: [m] -> [m]
            return torch.mv(self.M, vector)
        else:
            # Batch of vectors: [batch_size, m] -> [batch_size, m]
            return torch.mm(vector, self.M.t())
    
    def extract_vectors(self, vec_seq):
        """
        Extract vector groups from a vector sequence based on vectorization mode
        and return the aggregated vectors.
        
        - 'linear': Slide window by 1 step, extracting contiguous vectors of length = rank
        - 'nonlinear': Slide window by custom step (or rank length if step not specified)
        
        For nonlinear mode, handles incomplete trailing fragments using:
        - 'pad': Pads with zero vectors to maintain group length
        - 'drop': Discards incomplete fragments
        
        Args:
            vec_seq (list or tensor): Input sequence of m-dimensional vectors
            
        Returns:
            torch.Tensor: Tensor of aggregated vectors from extracted vector groups
        """
        # Convert to tensor if needed
        if not isinstance(vec_seq, torch.Tensor):
            vec_seq = torch.tensor(np.array(vec_seq), dtype=torch.float32, device=self.device)        
        
        L = vec_seq.shape[0]  # Sequence length

        # Helper function to apply vector operations
        def apply_op(vectors):
            """Apply rank operation to a list of vectors"""
            if self.rank_op == 'sum':
                return torch.sum(vectors, dim=0)
            elif self.rank_op == 'pick':
                idx = random.randint(0, vectors.size(0)-1)
                return vectors[idx]
            elif self.rank_op == 'user_func':
                # Default: average + sigmoid
                avg = torch.mean(vectors, dim=0)
                return torch.sigmoid(avg)
            else:  # 'avg' is default
                return torch.mean(vectors, dim=0)
        
        # Linear mode: sliding window with step=1
        if self.mode == 'linear':
            windows = []
            for i in range(L - self.rank + 1):
                window = vec_seq[i:i+self.rank]
                window_vector = apply_op(window)
                windows.append(window_vector)
            
            if windows:
                return torch.stack(windows)
            else:
                return torch.empty(0, self.m, device=self.device)
        
        # Nonlinear mode: stepping with custom step size
        vectors = []
        step = self.step or self.rank  # Use custom step if defined, else use rank length
        
        for i in range(0, L, step):
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
        
        if vectors:
            return torch.stack(vectors)
        else:
            return torch.empty(0, self.m, device=self.device)

    def _get_window_starts(self, seq_len):
        """Compute start indices of windows based on mode and rank_mode."""
        if self.mode == 'linear':
            step = 1
            if self.rank_mode == 'drop':
                starts = list(range(0, seq_len - self.rank + 1, step))
            else:  # 'pad'
                starts = list(range(0, seq_len, step))
        else:  # nonlinear
            step = self.step or self.rank
            if self.rank_mode == 'drop':
                starts = list(range(0, seq_len - self.rank + 1, step))
            else:
                starts = list(range(0, seq_len, step))
        return starts

    def batch_represent(self, seq_batch):
        """
        Compute sequence representations for a batch of vector sequences efficiently.
        Supports arbitrary window size and step (linear/nonlinear) as long as all sequences
        in the batch have identical length.
        
        Args:
            seq_batch: Tensor of shape [batch_size, seq_len, m]
        
        Returns:
            representations: Tensor of shape [batch_size, m]
        """
        batch_size, seq_len, m = seq_batch.shape
        
        # Compute window start indices
        starts = self._get_window_starts(seq_len)
        if not starts:
            return torch.zeros(batch_size, m, device=self.device)
        
        n_windows = len(starts)
        
        # Use unfold to extract sliding windows efficiently
        # We need to reshape to [batch_size, m, seq_len] and then unfold
        x = seq_batch.permute(0, 2, 1)  # [batch, m, seq_len]
        
        # Pad if necessary (for 'pad' mode)
        if self.rank_mode == 'pad':
            # Compute how many extra positions needed at the end to ensure all windows are fully inside
            step = 1 if self.mode == 'linear' else (self.step or self.rank)
            last_start = starts[-1]
            needed = last_start + self.rank - seq_len
            if needed > 0:
                x = torch.nn.functional.pad(x, (0, needed), mode='constant', value=0)
                seq_len_pad = seq_len + needed
            else:
                seq_len_pad = seq_len
        else:
            seq_len_pad = seq_len
        
        # Unfold: output shape [batch, m * rank, n_windows]
        windows = torch.nn.functional.unfold(
            x, kernel_size=(1, self.rank), stride=(1, step if self.mode == 'nonlinear' else 1)
        )  # shape: [batch, m*rank, n_windows]
        
        # Reshape to [batch, n_windows, rank, m]
        windows = windows.view(batch_size, m, self.rank, n_windows).permute(0, 3, 2, 1)
        # [batch, n_windows, rank, m]
        
        # Apply mapping M to each vector in each window
        mapped = torch.matmul(windows, self.M.T)  # [batch, n_windows, rank, m]
        
        # Aggregate over rank dimension
        if self.rank_op == 'sum':
            agg = mapped.sum(dim=2)  # [batch, n_windows, m]
        elif self.rank_op == 'pick':
            idx = torch.randint(0, self.rank, (batch_size, n_windows, 1), device=self.device)
            agg = torch.gather(mapped, 2, idx.expand(-1, -1, -1, self.m)).squeeze(2)
        elif self.rank_op == 'user_func':
            avg = mapped.mean(dim=2)
            agg = torch.sigmoid(avg)
        else:  # 'avg'
            agg = mapped.mean(dim=2)  # [batch, n_windows, m]
        
        # Compute basis indices for each window position
        # Basis index depends on window's start index (k)
        k_tensor = torch.tensor(starts, dtype=torch.long, device=self.device)  # [n_windows]
        j_indices = k_tensor % self.L  # [n_windows]
        
        # Get B rows and A columns for each window
        B_rows = self.Bbasis[j_indices]  # [n_windows, m]
        B_rows = B_rows.unsqueeze(0)  # [1, n_windows, m]
        A_cols = self.Acoeff[:, j_indices].T  # [n_windows, m]
        A_cols = A_cols.unsqueeze(0)  # [1, n_windows, m]
        
        # Compute scalar = B_j · agg
        scalar = torch.sum(B_rows * agg, dim=2)  # [batch, n_windows]
        
        # Compute Nk = A_j * scalar
        Nk = A_cols * scalar.unsqueeze(2)  # [batch, n_windows, m]
        
        # Average over windows to get representation
        rep = Nk.mean(dim=1)  # [batch, m]
        return rep

    def batch_compute_Nk_and_targets(self, seq_batch):
        """
        Compute Nk vectors and target vectors for each window in a batch of sequences.
        Supports arbitrary window size and step.
        
        Args:
            seq_batch: Tensor of shape [batch_size, seq_len, m]
        
        Returns:
            Nk_all: Tensor [batch_size, n_windows, m] - Nk vectors for each window
            targets: Tensor [batch_size, n_windows, m] - target vectors (aggregated mapped vectors)
        """
        batch_size, seq_len, m = seq_batch.shape
        
        starts = self._get_window_starts(seq_len)
        if not starts:
            return torch.empty(batch_size, 0, m, device=self.device), torch.empty(batch_size, 0, m, device=self.device)
        
        n_windows = len(starts)
        step = 1 if self.mode == 'linear' else (self.step or self.rank)
        
        x = seq_batch.permute(0, 2, 1)  # [batch, m, seq_len]
        
        if self.rank_mode == 'pad':
            last_start = starts[-1]
            needed = last_start + self.rank - seq_len
            if needed > 0:
                x = torch.nn.functional.pad(x, (0, needed), mode='constant', value=0)
                seq_len_pad = seq_len + needed
            else:
                seq_len_pad = seq_len
        else:
            seq_len_pad = seq_len
        
        windows = torch.nn.functional.unfold(
            x, kernel_size=(1, self.rank), stride=(1, step)
        )  # [batch, m*rank, n_windows]
        
        windows = windows.view(batch_size, m, self.rank, n_windows).permute(0, 3, 2, 1)
        # [batch, n_windows, rank, m]
        
        # Apply mapping
        mapped = torch.matmul(windows, self.M.T)  # [batch, n_windows, rank, m]
        
        # Aggregate to get targets
        if self.rank_op == 'sum':
            targets = mapped.sum(dim=2)  # [batch, n_windows, m]
        elif self.rank_op == 'pick':
            idx = torch.randint(0, self.rank, (batch_size, n_windows, 1), device=self.device)
            targets = torch.gather(mapped, 2, idx.expand(-1, -1, -1, self.m)).squeeze(2)
        elif self.rank_op == 'user_func':
            avg = mapped.mean(dim=2)
            targets = torch.sigmoid(avg)
        else:  # 'avg'
            targets = mapped.mean(dim=2)  # [batch, n_windows, m]
        
        # Compute basis indices
        k_tensor = torch.tensor(starts, dtype=torch.long, device=self.device)
        j_indices = k_tensor % self.L
        
        B_rows = self.Bbasis[j_indices]  # [n_windows, m]
        B_rows = B_rows.unsqueeze(0)  # [1, n_windows, m]
        A_cols = self.Acoeff[:, j_indices].T  # [n_windows, m]
        A_cols = A_cols.unsqueeze(0)  # [1, n_windows, m]
        
        scalar = torch.sum(B_rows * targets, dim=2)  # [batch, n_windows]
        Nk = A_cols * scalar.unsqueeze(2)  # [batch, n_windows, m]
        
        return Nk, targets

    def batch_compute_Nk(self, k_tensor, vectors):
        """
        Vectorized computation of N(k) vectors for a batch of positions and vectors.
        
        Args:
            k_tensor: Tensor of position indices [batch_size]
            vectors: Tensor of input vectors [batch_size, m]
            
        Returns:
            Tensor of N(k) vectors [batch_size, m]
        """
        # Apply mapping matrix M to input vectors
        x = self.map_vector(vectors)  # [batch_size, m]
        
        # Calculate basis indices j = k % L [batch_size]
        j_indices = (k_tensor % self.L).long()
        
        # Get Bbasis vectors: [batch_size, m]
        B_j = self.Bbasis[j_indices]
        
        # Compute scalar projection: B_j • x [batch_size]
        scalar = torch.sum(B_j * x, dim=1, keepdim=True)
        
        # Get Acoeff vectors: [batch_size, m]
        # Note: Acoeff is [m, L] -> permute to [L, m] then index with j_indices
        A_j = self.Acoeff.permute(1, 0)[j_indices]
        
        # Compute Nk = scalar * A_j [batch_size, m]
        Nk = scalar * A_j
            
        return Nk

    def describe(self, seq):
        """Compute N(k) vectors for each window in sequence"""
        vectors = self.extract_vectors(seq)
        if vectors.shape[0] == 0:
            return np.array([])
        
        k_positions = torch.arange(vectors.shape[0], dtype=torch.float32, device=self.device)
        
        # Batch compute all Nk vectors
        N_batch = self.batch_compute_Nk(k_positions, vectors)
        return N_batch.detach().cpu().numpy()

    def S(self, seq):
        """
        Compute list of S(l)=sum(N(k)) (k=1,...,l; l=1,...,L) for a given sequence.
        """
        vectors = self.extract_vectors(seq)
        if vectors.shape[0] == 0:
            return []
            
        k_positions = torch.arange(vectors.shape[0], dtype=torch.float32, device=self.device)
        
        # Batch compute all Nk vectors
        N_batch = self.batch_compute_Nk(k_positions, vectors)
        
        # Compute cumulative sum (S vectors)
        S_cum = torch.cumsum(N_batch, dim=0)
        return [s.detach().cpu().numpy() for s in S_cum]

    def D(self, seqs, t_list):
        """
        Compute mean squared deviation D across sequences:
        D = average over all positions of (N(k)-t_seq)^2
        """
        total_loss = 0.0
        total_positions = 0
        
        # Convert target vectors to tensor
        t_tensors = [torch.tensor(t, dtype=torch.float32, device=self.device) for t in t_list]
        
        for seq, t in zip(seqs, t_tensors):
            vectors = self.extract_vectors(seq)
            if vectors.shape[0] == 0:
                continue
                
            k_positions = torch.arange(vectors.shape[0], dtype=torch.float32, device=self.device)
            
            # Batch compute all Nk vectors
            N_batch = self.batch_compute_Nk(k_positions, vectors)
            
            # Compute loss for each position
            losses = torch.sum((N_batch - t) ** 2, dim=1)
            total_loss += losses.sum().item()
            total_positions += vectors.shape[0]
                
        return total_loss / total_positions if total_positions else 0.0

    def d(self, seq, t):
        """Compute pattern deviation value (d) for a single sequence."""
        d_value = self.D([seq], [t])
        return d_value     

    def reg_train(self, seqs, t_list, max_iters=1000, tol=1e-8, learning_rate=0.01, 
               continued=False, decay_rate=1.0, print_every=10, batch_size=32,
               checkpoint_file=None, checkpoint_interval=10):
        """
        Train model using gradient descent with sequence-level batch processing.
        
        Args:
            seqs: List of vector sequences (each as numpy array or torch tensor)
            t_list: List of target vectors
            max_iters: Maximum training iterations
            tol: Convergence tolerance
            learning_rate: Initial learning rate
            continued: Whether to continue from existing parameters
            decay_rate: Learning rate decay rate
            print_every: Print interval
            batch_size: Number of sequences to process in each batch
            checkpoint_file: Path to save/load checkpoint file for resuming training
            checkpoint_interval: Interval (in iterations) for saving checkpoints
            
        Returns:
            list: Training loss history
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
            print(f"Resumed training from checkpoint at iteration {start_iter}, best loss: {best_loss:.6e}")
        else:
            if not continued:
                self.reset_parameters()
            history = []
        
        # Convert target vectors to tensor
        t_tensors = [torch.tensor(t, dtype=torch.float32, device=self.device) for t in t_list]
        
        # Ensure all input sequences are on the correct device
        seqs = [seq.to(self.device) if isinstance(seq, torch.Tensor) else 
                torch.tensor(seq, dtype=torch.float32, device=self.device) 
                for seq in seqs]
        
        # Set up optimizer and scheduler
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)
        
        # Load optimizer and scheduler states if resuming
        if continued and checkpoint_file and os.path.exists(checkpoint_file):
            optimizer.load_state_dict(optimizer_state)
            scheduler.load_state_dict(scheduler_state)
        
        prev_loss = float('inf') if start_iter == 0 else history[-1] if history else float('inf')
        
        # Fast batch processing possible when rank_mode == 'drop'
        use_fast_batch = (self.rank_mode == 'drop')
        
        for it in range(start_iter, max_iters):
            total_loss = 0.0
            total_sequences = 0
            
            # Shuffle sequences for each epoch to ensure diverse batches
            indices = list(range(len(seqs)))
            random.shuffle(indices)
            
            # Process sequences in batches
            for batch_start in range(0, len(indices), batch_size):
                batch_indices = indices[batch_start:batch_start + batch_size]
                batch_seqs = [seqs[idx] for idx in batch_indices]
                batch_targets = [t_tensors[idx] for idx in batch_indices]
                
                optimizer.zero_grad()
                batch_loss = 0.0
                batch_sequence_count = 0
                
                # Check if all sequences in this batch have the same length
                seq_lengths = [seq.shape[0] for seq in batch_seqs]
                if use_fast_batch and len(set(seq_lengths)) == 1:
                    # All sequences have equal length -> use fast batch processing
                    batch_seq_tensor = torch.stack(batch_seqs, dim=0)  # [batch, seq_len, m]
                    # Compute sequence representations
                    reps = self.batch_represent(batch_seq_tensor)  # [batch, m]
                    # Convert targets to tensor
                    targets_tensor = torch.stack(batch_targets, dim=0)  # [batch, m]
                    # Compute loss for all sequences in batch
                    loss = torch.mean(torch.sum((reps - targets_tensor) ** 2, dim=1))
                    loss.backward()
                    optimizer.step()
                    batch_loss = loss.item() * len(batch_seqs)
                    batch_sequence_count = len(batch_seqs)
                    total_loss += batch_loss
                    total_sequences += batch_sequence_count
                else:
                    # Process each sequence in the current batch individually
                    for seq, target in zip(batch_seqs, batch_targets):
                        vectors = self.extract_vectors(seq)
                        if vectors.shape[0] == 0:
                            continue  # Skip empty sequences
                            
                        k_positions = torch.arange(vectors.shape[0], dtype=torch.float32, device=self.device)
                        # Batch compute all Nk vectors for current sequence
                        Nk_batch = self.batch_compute_Nk(k_positions, vectors)
                        # Compute sequence-level prediction: average of all N(k) vectors
                        seq_pred = torch.mean(Nk_batch, dim=0)
                        # Calculate loss for this sequence (MSE between prediction and target)
                        seq_loss = torch.sum((seq_pred - target) ** 2)
                        batch_loss += seq_loss
                        batch_sequence_count += 1
                        # Clean up intermediate tensors
                        del Nk_batch, seq_pred, k_positions
                        # Periodically clear GPU cache
                        if batch_sequence_count % 10 == 0 and torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    
                    # Backpropagate batch loss if we have valid sequences
                    if batch_sequence_count > 0:
                        batch_loss = batch_loss / batch_sequence_count
                        batch_loss.backward()
                        optimizer.step()
                        total_loss += batch_loss.item() * batch_sequence_count
                        total_sequences += batch_sequence_count
                
                # Clear GPU cache after processing each batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Calculate average loss for this iteration
            if total_sequences > 0:
                avg_loss = total_loss / total_sequences
            else:
                avg_loss = 0.0
                
            history.append(avg_loss)
            
            # Update best model state if current loss is lower
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_model_state = copy.deepcopy(self.state_dict())
            
            # Print training progress
            if it % print_every == 0 or it == max_iters - 1:
                current_lr = scheduler.get_last_lr()[0]
                print(f"GD Iter {it:3d}: Loss = {avg_loss:.6e}, LR = {current_lr:.6f}")
            
            # Save checkpoint at specified intervals
            if checkpoint_file and (it % checkpoint_interval == 0 or it == max_iters - 1):
                self._save_checkpoint(
                    checkpoint_file, it, history, optimizer, scheduler, best_loss
                )
            
            # Check convergence
            if abs(prev_loss - avg_loss) < tol:
                print(f"Converged after {it+1} iterations.")
                # Restore the best model state before breaking
                if best_model_state is not None and avg_loss > prev_loss:
                    self.load_state_dict(best_model_state)
                    print(f"Restored best model state with loss = {best_loss:.6e}")
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
            print(f"Training ended. Restored best model state with loss = {best_loss:.6e}")
            history[-1] = best_loss
        
        # Compute and store statistics for reconstruction/generation
        self._compute_training_statistics(seqs)
        self.trained = True
        
        return history

    def cls_train(self, seqs, labels, num_classes, max_iters=1000, tol=1e-8, learning_rate=0.01,
                  continued=False, decay_rate=1.0, print_every=10, batch_size=32,
                  checkpoint_file=None, checkpoint_interval=10):
        """
        Train the model for multi-class classification using cross-entropy loss.
        
        Args:
            seqs: List of vector sequences for training
            labels: List of integer class labels (0 to num_classes-1)
            num_classes: Number of classes in the classification problem
            max_iters: Maximum number of training iterations
            tol: Convergence tolerance
            learning_rate: Initial learning rate for optimizer
            continued: Whether to continue training from existing parameters
            decay_rate: Learning rate decay rate
            print_every: Print progress every N iterations
            batch_size: Number of sequences to process in each batch
            checkpoint_file: Path to save training checkpoints
            checkpoint_interval: Save checkpoint every N iterations
            
        Returns:
            list: Training loss history
        """
        
        # Initialize classification head if not already done
        if self.classifier is None or self.num_classes != num_classes:
            self.classifier = nn.Linear(self.m, num_classes).to(self.device)
            self.num_classes = num_classes
        
        if not continued:
            self.reset_parameters()
        
        # Convert labels to tensor
        label_tensors = torch.tensor(labels, dtype=torch.long, device=self.device)
        
        # Ensure all input sequences are on the correct device
        seqs = [seq.to(self.device) if isinstance(seq, torch.Tensor) else 
                torch.tensor(seq, dtype=torch.float32, device=self.device) 
                for seq in seqs]
        
        # Setup optimizer and scheduler
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)
        
        # Cross-entropy loss
        criterion = nn.CrossEntropyLoss()
        
        # Fast batch processing possible when rank_mode == 'drop'
        use_fast_batch = (self.rank_mode == 'drop')
        
        # Training state variables
        history = []
        prev_loss = float('inf')
        best_loss = float('inf')
        best_model_state = None
        
        for it in range(max_iters):
            total_loss = 0.0
            total_sequences = 0
            correct_predictions = 0
            
            # Shuffle sequences for each epoch
            indices = list(range(len(seqs)))
            random.shuffle(indices)
            
            # Process sequences in batches
            for batch_start in range(0, len(indices), batch_size):
                batch_indices = indices[batch_start:batch_start + batch_size]
                batch_seqs = [seqs[idx] for idx in batch_indices]
                batch_labels = label_tensors[batch_indices]
                
                optimizer.zero_grad()
                
                # Check if all sequences in this batch have the same length
                seq_lengths = [seq.shape[0] for seq in batch_seqs]
                if use_fast_batch and len(set(seq_lengths)) == 1:
                    # All equal: use efficient batch processing
                    batch_seq_tensor = torch.stack(batch_seqs, dim=0)  # [batch, seq_len, m]
                    reps = self.batch_represent(batch_seq_tensor)  # [batch, m]
                else:
                    # Unequal lengths: fallback to per-sequence processing
                    reps_list = []
                    for seq in batch_seqs:
                        vectors = self.extract_vectors(seq)
                        if vectors.shape[0] == 0:
                            seq_vector = torch.zeros(self.m, device=self.device)
                        else:
                            k_positions = torch.arange(vectors.shape[0], dtype=torch.float32, device=self.device)
                            Nk_batch = self.batch_compute_Nk(k_positions, vectors)
                            seq_vector = torch.mean(Nk_batch, dim=0)
                        reps_list.append(seq_vector)
                    reps = torch.stack(reps_list, dim=0)
                
                # Get logits through classification head
                logits = self.classifier(reps)  # [batch, num_classes]
                loss = criterion(logits, batch_labels)
                loss.backward()
                optimizer.step()
                
                # Accumulate statistics
                batch_loss = loss.item() * len(batch_seqs)
                total_loss += batch_loss
                total_sequences += len(batch_seqs)
                
                # Calculate accuracy
                with torch.no_grad():
                    predictions = torch.argmax(logits, dim=1)
                    correct_predictions += (predictions == batch_labels).sum().item()
                
                # Clear GPU cache periodically
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Calculate average loss and accuracy for this iteration
            if total_sequences > 0:
                avg_loss = total_loss / total_sequences
                accuracy = correct_predictions / total_sequences
            else:
                avg_loss = 0.0
                accuracy = 0.0
                
            history.append(avg_loss)
            
            # Update best model state
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_model_state = copy.deepcopy(self.state_dict())
            
            # Print training progress
            if it % print_every == 0 or it == max_iters - 1:
                current_lr = scheduler.get_last_lr()[0]
                print(f"CLS-Train Iter {it:3d}: Loss = {avg_loss:.6e}, Acc = {accuracy:.4f}, LR = {current_lr:.6f}")
            
            # Save checkpoint if specified
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
            
            # Check convergence
            if abs(prev_loss - avg_loss) < tol:
                print(f"Converged after {it+1} iterations.")
                # Restore best model state
                if best_model_state is not None:
                    self.load_state_dict(best_model_state)
                break
                
            prev_loss = avg_loss
            
            # Learning rate scheduling
            scheduler.step()
        
        self.trained = True
        
        return history

    def lbl_train(self, seqs, labels, num_labels, max_iters=1000, tol=1e-8, learning_rate=0.01, 
                 continued=False, decay_rate=1.0, print_every=10, batch_size=32,
                 checkpoint_file=None, checkpoint_interval=10, pos_weight=None):
        """
        Train the model for multi-label classification using binary cross-entropy loss.
        
        Args:
            seqs: List of vector sequences for training
            labels: List of binary label vectors (list of lists) or 2D numpy array/torch tensor
            num_labels: Number of labels for multi-label prediction task
            max_iters: Maximum number of training iterations
            tol: Convergence tolerance
            learning_rate: Initial learning rate for optimizer
            continued: Whether to continue training from existing parameters
            decay_rate: Learning rate decay rate
            print_every: Print progress every N iterations
            batch_size: Number of sequences to process in each batch
            checkpoint_file: Path to save training checkpoints
            checkpoint_interval: Save checkpoint every N iterations
            pos_weight: Weight for positive class (torch.Tensor of shape [num_labels])
            
        Returns:
            list: Training loss history
            list: Training accuracy history
        """
        
        # Initialize label head if not already done
        if self.labeller is None or self.num_labels != num_labels:
            self.labeller = nn.Linear(self.m, num_labels).to(self.device)
            self.num_labels = num_labels
        
        if not continued:
            self.reset_parameters()
        
        # Convert labels to tensor
        if isinstance(labels, list):
            labels_tensor = torch.tensor(labels, dtype=torch.float32, device=self.device)
        else:
            labels_tensor = torch.as_tensor(labels, dtype=torch.float32, device=self.device)
        
        # Ensure all input sequences are on the correct device
        seqs = [seq.to(self.device) if isinstance(seq, torch.Tensor) else 
                torch.tensor(seq, dtype=torch.float32, device=self.device) 
                for seq in seqs]
        
        # Setup loss function with optional positive class weighting
        if pos_weight is not None:
            pos_weight_tensor = torch.tensor(pos_weight, dtype=torch.float32, device=self.device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        else:
            criterion = nn.BCEWithLogitsLoss()
        
        # Setup optimizer and scheduler
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)
        
        # Fast batch processing possible when rank_mode == 'drop'
        use_fast_batch = (self.rank_mode == 'drop')
        
        # Training state variables
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
            
            # Shuffle sequences for each epoch
            indices = list(range(len(seqs)))
            random.shuffle(indices)
            
            # Process sequences in batches
            for batch_start in range(0, len(indices), batch_size):
                batch_indices = indices[batch_start:batch_start + batch_size]
                batch_seqs = [seqs[idx] for idx in batch_indices]
                batch_labels = labels_tensor[batch_indices]
                
                optimizer.zero_grad()
                
                # Check if all sequences in this batch have the same length
                seq_lengths = [seq.shape[0] for seq in batch_seqs]
                if use_fast_batch and len(set(seq_lengths)) == 1:
                    # All equal: use efficient batch processing
                    batch_seq_tensor = torch.stack(batch_seqs, dim=0)  # [batch, seq_len, m]
                    reps = self.batch_represent(batch_seq_tensor)  # [batch, m]
                    logits = self.labeller(reps)  # [batch, num_labels]
                    loss = criterion(logits, batch_labels)
                    loss.backward()
                    optimizer.step()
                    
                    batch_loss = loss.item() * len(batch_seqs)
                    total_loss += batch_loss
                    total_sequences += len(batch_seqs)
                    
                    # Calculate accuracy
                    with torch.no_grad():
                        probs = torch.sigmoid(logits)
                        predictions = (probs > 0.5).float()
                        batch_correct = (predictions == batch_labels).sum().item()
                        batch_predictions = batch_labels.numel()
                        total_correct += batch_correct
                        total_predictions += batch_predictions
                else:
                    # Unequal lengths: process each sequence individually
                    batch_loss = 0.0
                    batch_correct = 0
                    batch_predictions = 0
                    batch_predictions_list = []
                    
                    for seq in batch_seqs:
                        vectors = self.extract_vectors(seq)
                        if vectors.shape[0] == 0:
                            continue
                        k_positions = torch.arange(vectors.shape[0], dtype=torch.float32, device=self.device)
                        Nk_batch = self.batch_compute_Nk(k_positions, vectors)
                        seq_representation = torch.mean(Nk_batch, dim=0)
                        logits = self.labeller(seq_representation)
                        batch_predictions_list.append(logits)
                        # Clean up
                        del Nk_batch, seq_representation, k_positions
                    
                    if batch_predictions_list:
                        batch_logits = torch.stack(batch_predictions_list, dim=0)
                        loss = criterion(batch_logits, batch_labels)
                        loss.backward()
                        optimizer.step()
                        
                        batch_loss = loss.item() * len(batch_seqs)
                        total_loss += batch_loss
                        total_sequences += len(batch_seqs)
                        
                        with torch.no_grad():
                            probs = torch.sigmoid(batch_logits)
                            predictions = (probs > 0.5).float()
                            batch_correct = (predictions == batch_labels).sum().item()
                            batch_predictions = batch_labels.numel()
                            total_correct += batch_correct
                            total_predictions += batch_predictions
                
                # Clear GPU cache periodically
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Calculate average loss and accuracy for this iteration
            if total_sequences > 0:
                avg_loss = total_loss / total_sequences
                avg_acc = total_correct / total_predictions if total_predictions > 0 else 0.0
            else:
                avg_loss = 0.0
                avg_acc = 0.0
                
            loss_history.append(avg_loss)
            acc_history.append(avg_acc)
            
            # Update best model state
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_model_state = copy.deepcopy(self.state_dict())
            
            # Print training progress
            if it % print_every == 0 or it == max_iters - 1:
                current_lr = scheduler.get_last_lr()[0]
                print(f"MLC-Train Iter {it:3d}: Loss = {avg_loss:.6e}, Acc = {avg_acc:.4f}, LR = {current_lr:.6f}")
            
            # Save checkpoint if specified
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
            
            # Check convergence
            if abs(prev_loss - avg_loss) < tol:
                print(f"Converged after {it+1} iterations.")
                # Restore best model state
                if best_model_state is not None:
                    self.load_state_dict(best_model_state)
                break
                
            prev_loss = avg_loss
            
            # Learning rate scheduling
            scheduler.step()
        
        self.trained = True
        
        return loss_history, acc_history

    def self_train(self, seqs, max_iters=1000, tol=1e-8, 
               learning_rate=0.01, continued=False, decay_rate=1.0, 
               print_every=10, batch_size=1024, checkpoint_file=None, 
               checkpoint_interval=10):
        """
        Train the model using self-supervised learning with memory-optimized batch processing.
        Self-training uses the gap-filling objective: predict vector mappings from position information.
        
        Args:
            seqs: List of vector sequences
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
        seqs = [seq.to(self.device) if isinstance(seq, torch.Tensor) else 
                torch.tensor(seq, dtype=torch.float32, device=self.device) 
                for seq in seqs]
        
        # Set up optimizer and scheduler
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)
        
        # Load optimizer and scheduler states if resuming
        if continued and checkpoint_file and os.path.exists(checkpoint_file):
            optimizer.load_state_dict(optimizer_state)
            scheduler.load_state_dict(scheduler_state)
        
        prev_loss = float('inf') if start_iter == 0 else history[-1] if history else float('inf')
        
        # Check if all sequences have the same length (for fast batch processing)
        seq_lengths = [seq.shape[0] for seq in seqs]
        all_equal = len(set(seq_lengths)) == 1
        use_fast_batch = (self.rank_mode == 'drop' and all_equal)
        
        if use_fast_batch:
            # Fast path: all sequences equal length, we can batch process all windows together
            seq_len = seq_lengths[0]
            # Convert list to single tensor for efficiency
            all_seqs = torch.stack(seqs, dim=0)  # [num_seqs, seq_len, m]
            num_seqs = len(seqs)
            
            # Determine sequence batch size for the fast path
            seq_batch_size = batch_size  # batch_size now means number of sequences per batch
            
            for it in range(start_iter, max_iters):
                total_loss = 0.0
                total_windows = 0
                
                # Shuffle sequence indices
                indices = list(range(num_seqs))
                random.shuffle(indices)
                
                # Process sequences in batches
                for batch_start in range(0, num_seqs, seq_batch_size):
                    batch_indices = indices[batch_start:batch_start + seq_batch_size]
                    batch_seqs = all_seqs[batch_indices]  # [batch, seq_len, m]
                    
                    optimizer.zero_grad()
                    
                    # Compute Nk and targets for all windows in this batch of sequences
                    Nk_batch, targets_batch = self.batch_compute_Nk_and_targets(batch_seqs)  # both [batch, n_windows, m]
                    
                    # Compute MSE loss over all windows
                    loss = torch.mean(torch.sum((Nk_batch - targets_batch) ** 2, dim=-1))
                    loss.backward()
                    optimizer.step()
                    
                    batch_loss = loss.item() * batch_seqs.shape[0] * Nk_batch.shape[1]
                    total_loss += batch_loss
                    total_windows += batch_seqs.shape[0] * Nk_batch.shape[1]
                    
                    # Clear GPU cache
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                avg_loss = total_loss / total_windows if total_windows > 0 else 0.0
                history.append(avg_loss)
                
                # Update best model state if current loss is lower
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    best_model_state = copy.deepcopy(self.state_dict())
                
                # Print training progress
                if it % print_every == 0 or it == max_iters - 1:
                    current_lr = scheduler.get_last_lr()[0]
                    print(f"Self-Train Iter {it:3d}: loss = {avg_loss:.6f}, LR = {current_lr:.6f}, "
                          f"Windows = {total_windows}")
                
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
        else:
            # Slow path: sequences have different lengths or rank_mode == 'pad'
            # Use original per-sequence, per-window sampling with sample batch size
            sample_batch_size = batch_size
            
            for it in range(start_iter, max_iters):
                total_loss = 0.0
                total_samples = 0
                
                # Shuffle sequences for each epoch to ensure diverse samples
                indices = list(range(len(seqs)))
                random.shuffle(indices)
                
                # Process sequences in shuffled order
                for seq_idx in indices:
                    seq = seqs[seq_idx]
                    vectors = self.extract_vectors(seq)
                    if vectors.shape[0] == 0:
                        continue
                        
                    seq_samples = []
                    
                    # Generate samples for current sequence in gap-filling mode
                    # Each vector window is a sample (position k, vector)
                    for k, vector in enumerate(vectors):
                        seq_samples.append((k, vector))
                    
                    if not seq_samples:
                        continue
                    
                    # Process samples from current sequence in batches
                    for batch_start in range(0, len(seq_samples), sample_batch_size):
                        batch_samples = seq_samples[batch_start:batch_start + sample_batch_size]
                        
                        optimizer.zero_grad()
                        
                        # Prepare batch data directly as tensors
                        k_list = []
                        vector_list = []
                        
                        for sample in batch_samples:
                            k, vector = sample
                            k_list.append(k)
                            vector_list.append(vector)
                        
                        # Create tensors directly on GPU
                        k_tensor = torch.tensor(k_list, dtype=torch.float32, device=self.device)
                        vectors_tensor = torch.stack(vector_list)
                        
                        # Batch compute Nk for current vectors
                        Nk_batch = self.batch_compute_Nk(k_tensor, vectors_tensor)
                        
                        # Get target mappings (same as current vectors mapped through M)
                        targets = self.map_vector(vectors_tensor)
                        
                        # Compute loss: mean squared error between Nk and mapped vectors
                        loss = torch.mean(torch.sum((Nk_batch - targets) ** 2, dim=1))
                        loss.backward()
                        optimizer.step()
                        
                        batch_loss = loss.item() * len(batch_samples)
                        total_loss += batch_loss
                        total_samples += len(batch_samples)
                        
                        # Clean up to free memory
                        del k_tensor, vectors_tensor, Nk_batch, targets, loss
                        
                        # Periodically clear GPU cache
                        if total_samples % 1000 == 0 and torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    
                    # Clear sequence-specific tensors
                    del seq_samples
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                # Calculate average loss for this iteration
                if total_samples > 0:
                    avg_loss = total_loss / total_samples
                else:
                    avg_loss = 0.0
                    
                history.append(avg_loss)
                
                # Update best model state if current loss is lower
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    best_model_state = copy.deepcopy(self.state_dict())
                
                # Print training progress
                if it % print_every == 0 or it == max_iters - 1:
                    current_lr = scheduler.get_last_lr()[0]
                    print(f"Self-Train Iter {it:3d}: loss = {avg_loss:.6f}, LR = {current_lr:.6f}, "
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
        self._compute_training_statistics(seqs)
        self.trained = True
        
        return history

    def _save_checkpoint(self, checkpoint_file, iteration, history, optimizer, scheduler, best_loss):
        """
        Save training checkpoint with complete training state
        
        Args:
            checkpoint_file: Path to save checkpoint file
            iteration: Current training iteration
            history: Training loss history
            optimizer: Optimizer instance
            scheduler: Learning rate scheduler instance
            best_loss: Best loss achieved so far
        """
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

    def _compute_training_statistics(self, seqs, batch_size=50):
        """Compute and store statistics for reconstruction and generation with memory optimization"""
        total_token_count = 0
        total_t = torch.zeros(self.m, device=self.device)
        
        with torch.no_grad():
            for i in range(0, len(seqs), batch_size):
                batch_seqs = seqs[i:i+batch_size]
                batch_token_count = 0
                batch_t_sum = torch.zeros(self.m, device=self.device)
                
                for seq in batch_seqs:
                    try:
                        vectors = self.extract_vectors(seq)
                        batch_token_count += vectors.shape[0]
                        
                        if vectors.shape[0] > 0:
                            k_positions = torch.arange(vectors.shape[0], dtype=torch.float32, device=self.device)
                            Nk_batch = self.batch_compute_Nk(k_positions, vectors)
                            batch_t_sum += Nk_batch.sum(dim=0)
                            
                            # Clean up
                            del k_positions, Nk_batch
                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            print(f"Warning: Memory error computing statistics for sequence. Skipping.")
                            continue
                        else:
                            raise e
                
                total_token_count += batch_token_count
                total_t += batch_t_sum
                
                # Clean batch
                del batch_t_sum
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        self.mean_token_count = total_token_count / len(seqs) if seqs else 0
        self.mean_t = (total_t / total_token_count).cpu().numpy() if total_token_count > 0 else np.zeros(self.m)
    
    def predict_t(self, seq):
        """Predict target vector as average of N(k) vectors"""
        vectors = self.extract_vectors(seq)
        if vectors.shape[0] == 0:
            return np.zeros(self.m)
        
        # Compute all Nk vectors
        Nk = self.describe(seq)
        return np.mean(Nk, axis=0)
    
    def predict_c(self, seq):
        """
        Predict class label for a sequence using the classification head.
        
        Args:
            seq: Input vector sequence
            
        Returns:
            tuple: (predicted_class, class_probabilities)
        """
        if self.classifier is None:
            raise ValueError("Model must be trained first for classification")
        
        # Get sequence vector representation
        seq_vector = self.predict_t(seq)
        seq_vector_tensor = torch.tensor(seq_vector, dtype=torch.float32, device=self.device)
        
        # Get logits through classification head
        with torch.no_grad():
            logits = self.classifier(seq_vector_tensor.unsqueeze(0))
            probabilities = torch.softmax(logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            
        return predicted_class, probabilities[0].cpu().numpy()

    def predict_l(self, seq, threshold=0.5):
        """
        Predict multi-label classification for a sequence.
        
        Args:
            seq: Input vector sequence
            threshold: Probability threshold for binary classification (default: 0.5)
            
        Returns:
            numpy.ndarray: Binary label predictions (0 or 1 for each label)
            numpy.ndarray: Probability scores for each label
        """
        assert self.labeller is not None, "Model must be trained first for label prediction"
        
        vectors = self.extract_vectors(seq)
        if vectors.shape[0] == 0:
            # Return zeros if no vectors
            return np.zeros(self.num_labels, dtype=np.float32), np.zeros(self.num_labels, dtype=np.float32)
        
        k_positions = torch.arange(vectors.shape[0], dtype=torch.float32, device=self.device)
        
        # Compute N(k) vectors for all windows in the sequence
        Nk_batch = self.batch_compute_Nk(k_positions, vectors)
        
        # Compute sequence representation: average of all N(k) vectors
        seq_representation = torch.mean(Nk_batch, dim=0)
        
        # Pass through classification head to get logits
        with torch.no_grad():
            logits = self.labeller(seq_representation)
            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(logits).cpu().numpy()
        
        # Apply threshold to get binary predictions
        binary_preds = (probs > threshold).astype(np.float32)
        
        return binary_preds, probs

    def reconstruct(self, L, tau=0.0, num_candidates=1000):
        """
        Reconstruct representative vector sequence of length L by minimizing error with temperature-controlled randomness.
        
        Args:
            L (int): Length of sequence to generate (number of vector windows)
            tau (float): Temperature parameter (0: deterministic, >0: stochastic)
            num_candidates (int): Number of candidate vectors to consider
            
        Returns:
            numpy.ndarray: Reconstructed vector sequence of shape [L, m]
        """
        assert self.trained, "Model must be trained first"
        if tau < 0:
            raise ValueError("Temperature must be non-negative")
            
        mean_t_tensor = torch.tensor(self.mean_t, dtype=torch.float32, device=self.device)
        generated_vectors = []
        
        # Generate random candidate vectors (could be improved with more sophisticated sampling)
        candidate_vectors = torch.randn(num_candidates, self.m, device=self.device)
        
        for k in range(L):
            # Compute Nk for all candidate vectors at position k
            k_tensor = torch.tensor([k] * num_candidates, dtype=torch.float32, device=self.device)
            Nk_all = self.batch_compute_Nk(k_tensor, candidate_vectors)
            
            # Compute scores
            errors = torch.sum((Nk_all - mean_t_tensor) ** 2, dim=1)
            scores = -errors  # Convert to score (higher = better)
            
            if tau == 0:  # Deterministic selection
                max_idx = torch.argmax(scores).item()
                best_vec = candidate_vectors[max_idx].cpu().numpy()
                generated_vectors.append(best_vec)
            else:  # Stochastic selection
                probs = torch.softmax(scores / tau, dim=0).detach().cpu().numpy()
                chosen_idx = random.choices(range(num_candidates), weights=probs, k=1)[0]
                chosen_vec = candidate_vectors[chosen_idx].cpu().numpy()
                generated_vectors.append(chosen_vec)
                
        return np.stack(generated_vectors)

    def save(self, filename):
        """Save model state to file"""
        save_dict = {
            'state_dict': self.state_dict(),
            'mean_t': self.mean_t if hasattr(self, 'mean_t') else None,
            'mean_L': self.mean_L if hasattr(self, 'mean_L') else None,
            'trained': self.trained,
            'num_classes': self.num_classes,
            'num_labels': self.num_labels,
            'rank_op': self.rank_op
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
        self.mean_L = save_dict.get('mean_L', None)
        self.trained = save_dict.get('trained', False)
        self.num_classes = save_dict.get('num_classes', None)
        self.num_labels = save_dict.get('num_labels', None)
        self.rank_op = save_dict.get('rank_op', 'avg')
        
        # Recreate classifier and labeller if needed
        if self.num_classes is not None:
            self.classifier = nn.Linear(self.m, self.num_classes).to(self.device)
        if self.num_labels is not None:
            self.labeller = nn.Linear(self.m, self.num_labels).to(self.device)
            
        print(f"Model loaded from {filename}")
        return self


# ================================================
# Example Usage with Numerical Vector Sequences
# ================================================
if __name__ == "__main__":
    # Import correlation function for evaluation
    from statistics import correlation
    
    print("="*60)
    print("NumDualDescriptorRN - PyTorch GPU Accelerated Version")
    print("Optimized for Numerical Vector Sequences")
    print("="*60)
    
    # Set random seeds for reproducibility
    torch.manual_seed(2)
    random.seed(2)
    np.random.seed(2)
    
    # Define parameters
    vec_dim = 3          # Dimension of input vectors (m)
    bas_dim = 100        # Basis dimension (L)
    rank = 1             # Window size for vector averaging
    seq_num = 100         # Number of sequences
    seq_len_range = (200, 300)  # Sequence length range
    
    # ================================================
    # Example 1: Regression Task
    # ================================================
    print("\n" + "="*60)
    print("Example 1: Regression Task")
    print("="*60)
    
    # Generate random vector sequences and target vectors
    seqs, t_list = [], []
    for _ in range(seq_num):
        L = np.random.randint(seq_len_range[0], seq_len_range[1])
        # Generate random vector sequence
        seq = np.random.randn(L, vec_dim).astype(np.float32)
        seqs.append(seq)
        # Generate random target vector
        t_list.append(np.random.uniform(-1, 1, vec_dim).astype(np.float32))
    
    # Create model instance
    model = NumDualDescriptorRN(
        vec_dim=vec_dim,
        bas_dim=bas_dim,
        rank=rank,
        rank_mode='drop',
        mode='linear',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    print(f"\nUsing device: {model.device}")
    print(f"Vector dimension: {model.m}")
    print(f"Basis dimension: {model.L}")
    print(f"Window size (rank): {model.rank}")
    
    # Train the model using gradient descent
    print("\nStarting gradient descent training...")
    history = model.reg_train(
        seqs, 
        t_list,
        learning_rate=0.05,
        max_iters=50,
        tol=1e-6,
        decay_rate=0.99,
        print_every=10,
        batch_size=16
    )
    
    # Predict target vector for first sequence
    test_seq = seqs[0]
    t_pred = model.predict_t(test_seq)
    print(f"\nPredicted t for first sequence: {t_pred}")
    print(f"Actual t: {t_list[0]}")
    print(f"Mean squared error: {np.mean((t_pred - t_list[0]) ** 2):.6f}")
    
    # Calculate correlation between predicted and actual targets
    pred_t_list = [model.predict_t(seq) for seq in seqs]
    corr_sum = 0.0
    for i in range(model.m):
        actu_t = [t_vec[i] for t_vec in t_list]
        pred_t = [t_vec[i] for t_vec in pred_t_list]
        corr = correlation(actu_t, pred_t)
        print(f"Prediction correlation for dimension {i}: {corr:.4f}")
        corr_sum += corr
    corr_avg = corr_sum / model.m
    print(f"Average correlation: {corr_avg:.4f}")
    
    # Reconstruct representative vector sequence
    print("\nReconstructing representative sequence...")
    recon_seq = model.reconstruct(L=10, tau=0.0)
    print(f"Deterministic reconstruction (first 3 windows):")
    print(recon_seq[:3])
    
    # ================================================
    # Example 2: Classification Task
    # ================================================
    print("\n" + "="*60)
    print("Example 2: Classification Task")
    print("="*60)
    
    # Generate classification data
    num_classes = 3
    class_seqs = []
    class_labels = []
    
    # Create sequences with different patterns for each class
    for class_id in range(num_classes):
        for _ in range(50):  # 50 sequences per class
            L = np.random.randint(100, 200)
            if class_id == 0:
                # Class 0: Positive mean with low variance
                seq = np.random.normal(loc=1.0, scale=0.2, size=(L, vec_dim)).astype(np.float32)
            elif class_id == 1:
                # Class 1: Negative mean with medium variance
                seq = np.random.normal(loc=-1.0, scale=0.5, size=(L, vec_dim)).astype(np.float32)
            else:
                # Class 2: Zero mean with high variance
                seq = np.random.normal(loc=0.0, scale=1.0, size=(L, vec_dim)).astype(np.float32)
            
            class_seqs.append(seq)
            class_labels.append(class_id)
    
    # Initialize new model for classification
    model_cls = NumDualDescriptorRN(
        vec_dim=vec_dim,
        bas_dim=bas_dim,
        rank=rank,
        rank_mode='drop',
        mode='linear',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Train for classification
    print("\nStarting classification training...")
    cls_history = model_cls.cls_train(
        class_seqs, class_labels, num_classes, 
        max_iters=20, tol=1e-6, learning_rate=0.05,
        decay_rate=0.99, batch_size=16, print_every=5
    )
    
    # Evaluate on training set
    print("\nEvaluating classification accuracy...")
    correct = 0
    all_predictions = []
    
    for seq, true_label in zip(class_seqs, class_labels):
        pred_class, probs = model_cls.predict_c(seq)
        all_predictions.append(pred_class)
        
        if pred_class == true_label:
            correct += 1
    
    accuracy = correct / len(class_seqs)
    print(f"Accuracy: {accuracy:.4f} ({correct}/{len(class_seqs)})")
    
    # Show some example predictions
    print("\nExample predictions:")
    for i in range(min(3, len(class_seqs))):
        pred_class, probs = model_cls.predict_c(class_seqs[i])
        print(f"Seq {i+1}: True={class_labels[i]}, Pred={pred_class}, Probs={[f'{p:.3f}' for p in probs]}")
    
    # ================================================
    # Example 3: Multi-Label Classification Task
    # ================================================
    print("\n" + "="*60)
    print("Example 3: Multi-Label Classification Task")
    print("="*60)
    
    # Generate multi-label data
    num_labels = 4
    label_seqs = []
    labels = []
    
    for _ in range(100):
        L = np.random.randint(100, 200)
        # Create random vector sequence
        seq = np.random.randn(L, vec_dim).astype(np.float32)
        label_seqs.append(seq)
        # Create random binary labels (each sequence can have 0-4 active labels)
        label_vec = [(np.random.random() > 0.7) for _ in range(num_labels)]
        labels.append([1.0 if x else 0.0 for x in label_vec])
    
    # Initialize model for multi-label classification
    model_lbl = NumDualDescriptorRN(
        vec_dim=vec_dim,
        bas_dim=bas_dim,
        rank=rank,
        rank_mode='drop',
        mode='linear',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Train the model
    print("\nStarting multi-label classification training...")
    loss_history, acc_history = model_lbl.lbl_train(
        label_seqs, labels, num_labels,
        max_iters=20, 
        tol=1e-6, 
        learning_rate=0.05, 
        decay_rate=0.99, 
        print_every=5, 
        batch_size=16
    )
    
    print(f"\nFinal training loss: {loss_history[-1]:.6f}")
    print(f"Final training accuracy: {acc_history[-1]:.4f}")
    
    # Show prediction results
    print("\nPrediction results (first 3 sequences):")
    for i in range(min(3, len(label_seqs))):
        pred_binary, pred_probs = model_lbl.predict_l(label_seqs[i], threshold=0.5)
        print(f"\nSequence {i+1}:")
        print(f"True labels: {labels[i]}")
        print(f"Predicted binary: {pred_binary}")
        print(f"Predicted probabilities: {[f'{p:.3f}' for p in pred_probs]}")
        correct = np.all(pred_binary == np.array(labels[i]))
        print(f"Correct: {correct}")
    
    # ================================================
    # Example 4: Self-Training
    # ================================================
    print("\n" + "="*60)
    print("Example 4: Self-Training")
    print("="*60)
    
    # Generate sequences for self-training
    self_seqs = []
    for _ in range(30):
        L = np.random.randint(80, 120)
        seq = np.random.randn(L, vec_dim).astype(np.float32)
        self_seqs.append(seq)
    
    # Initialize model for self-training
    model_self = NumDualDescriptorRN(
        vec_dim=vec_dim,
        bas_dim=bas_dim,
        rank=rank,
        rank_mode='drop',
        mode='linear',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Run self-supervised training
    print("\nStarting self-training...")
    self_history = model_self.self_train(
        self_seqs,
        max_iters=50,
        learning_rate=0.01,
        decay_rate=0.995,
        print_every=5,
        batch_size=512
    )
    
    # Reconstruct sequences
    print("\nReconstructing sequences...")
    recon_det = model_self.reconstruct(L=5, tau=0.0)
    recon_sto = model_self.reconstruct(L=5, tau=0.5)
    
    print(f"Deterministic reconstruction (tau=0.0):")
    print(recon_det)
    print(f"\nStochastic reconstruction (tau=0.5):")
    print(recon_sto)   
    
    # ================================================
    # Example 5: Model Persistence
    # ================================================
    print("\n" + "="*60)
    print("Example 5: Model Persistence")
    print("="*60)
    
    # Save the regression model
    model.save("num_dual_descriptor_model.pt")
    
    # Load the model
    model_loaded = NumDualDescriptorRN(
        vec_dim=vec_dim,
        bas_dim=bas_dim,
        rank=rank,
        rank_mode='drop',
        mode='linear',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    model_loaded.load("num_dual_descriptor_model.pt")
    
    # Test that loaded model gives same predictions
    test_pred_original = model.predict_t(test_seq)
    test_pred_loaded = model_loaded.predict_t(test_seq)
    
    print(f"Original model prediction: {test_pred_original}")
    print(f"Loaded model prediction: {test_pred_loaded}")
    print(f"Predictions match: {np.allclose(test_pred_original, test_pred_loaded, rtol=1e-6)}")
    
    print("\n" + "="*60)
    print("All examples completed successfully!")
    print("="*60)
