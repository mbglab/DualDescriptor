# Copyright (C) 2005-2025, Bin-Guang Ma (mbg@mail.hzau.edu.cn); SPDX-License-Identifier: MIT
# Spatial Dual Descriptor Vector class (Random AB matrix form) for 2D character arrays implemented with PyTorch
# This program is for the demonstration of methodology and not fully refined.
# Author: Bin-Guang Ma (assisted by DeepSeek); Date: 2025-7-29 ~ 2026-3-25

import math
import random
import itertools
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
import os

class SpatialDualDescriptorRN2(nn.Module):
    """
    Spatial Dual Descriptor for 2D character arrays with GPU acceleration using PyTorch:
      - Learnable coefficient matrix Acoeff ∈ R^{m×L} (L = bas_row * bas_col)
      - Learnable basis matrices: B_row ∈ R^{bas_row×m}, B_col ∈ R^{bas_col×m}
      - Token embeddings M: token → R^m
      - Optimized with batch processing for GPU acceleration
      - Supports both linear and nonlinear tokenization
      - Supports regression, classification, and multi-label classification tasks
    """
    def __init__(self, charset, vec_dim=4, bas_row=10, bas_col=10, rank=1, rank_mode='drop', 
                 mode='linear', user_step=None, device='cuda'):
        super().__init__()
        self.charset = list(charset)
        self.m = vec_dim
        self.bas_row = bas_row
        self.bas_col = bas_col
        self.L = bas_row * bas_col          # total number of basis indices
        self.rank = rank
        self.rank_mode = rank_mode
        assert mode in ('linear','nonlinear')
        self.mode = mode
        self.step = user_step
        self.trained = False
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Generate all possible tokens from square windows of size rank×rank
        # Each window is flattened row-major into a string.
        toks = []
        if self.rank_mode=='pad':
            # For padding mode, we generate tokens of length rank^2 by padding with '_'
            # but note that the window may be smaller than rank×rank if at the border.
            # However, for token vocabulary we only need all possible full-size combinations.
            # In pad mode we still consider all possible full windows (no padding in vocabulary).
            # The padding will be applied during extraction.
            for r in range(1, self.rank+1):
                for prefix in itertools.product(self.charset, repeat=r):
                    tok = ''.join(prefix).ljust(self.rank, '_')
                    toks.append(tok)
        else:
            # 'drop' mode: only full-size windows are considered
            toks = [''.join(p) for p in itertools.product(self.charset, repeat=self.rank*self.rank)]
        # Ensure tokens are sorted and unique
        self.tokens = sorted(set(toks))
        self.token_to_idx = {token: idx for idx, token in enumerate(self.tokens)}
        self.idx_to_token = {idx: token for idx, token in enumerate(self.tokens)}
        
        # Token embeddings
        self.embedding = nn.Embedding(len(self.tokens), self.m)
        
        # Coefficient matrix Acoeff: m×L (L = bas_row * bas_col)
        self.Acoeff = nn.Parameter(torch.empty(self.m, self.L))
        
        # Basis matrices for rows and columns
        self.B_row = nn.Parameter(torch.empty(self.bas_row, self.m))
        self.B_col = nn.Parameter(torch.empty(self.bas_col, self.m))
        
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
        nn.init.uniform_(self.embedding.weight, -0.5, 0.5)
        nn.init.uniform_(self.Acoeff, -0.1, 0.1)
        nn.init.uniform_(self.B_row, -0.1, 0.1)
        nn.init.uniform_(self.B_col, -0.1, 0.1)
        
        # Initialize classifier if it exists
        if self.classifier is not None:
            nn.init.normal_(self.classifier.weight, 0, 0.01)
            if self.classifier.bias is not None:
                nn.init.constant_(self.classifier.bias, 0)
                
        # Initialize labeller if it exists
        if self.labeller is not None:
            nn.init.xavier_uniform_(self.labeller.weight)
            nn.init.zeros_(self.labeller.bias)
    
    def token_to_indices(self, token_list):
        """Convert list of tokens to tensor of indices"""
        return torch.tensor([self.token_to_idx[tok] for tok in token_list], device=self.device)
    
    def extract_tokens_and_positions(self, arr_2d):
        """
        Extract square windows of size rank×rank from a 2D character array.
        Returns:
            tokens: list of window strings (flattened row-major)
            k1_list: list of row indices of the top-left corner of each window
            k2_list: list of column indices of the top-left corner of each window
        """
        H = len(arr_2d)
        W = len(arr_2d[0]) if H>0 else 0
        tokens = []
        k1_list = []
        k2_list = []
        
        # Determine step size
        if self.mode == 'linear':
            step = 1
        else:
            step = self.step or self.rank
        
        # Slide over the array
        for i in range(0, H, step):
            # Stop if window would exceed height
            if i + self.rank > H:
                if self.rank_mode == 'pad':
                    # Pad rows with '_' to make height rank
                    rows_needed = self.rank - (H - i)
                    rows = []
                    for r in range(i, H):
                        rows.append(arr_2d[r][:])  # copy row
                    # Pad missing rows with '_' characters
                    for _ in range(rows_needed):
                        rows.append(['_'] * W)
                    # Now process columns for this padded block
                    for j in range(0, W, step):
                        if j + self.rank > W:
                            if self.rank_mode == 'pad':
                                # Pad columns with '_'
                                cols_needed = self.rank - (W - j)
                                # Extract submatrix of size (self.rank, self.rank) from rows and columns
                                window_rows = rows[:self.rank]  # exactly rank rows
                                # For each row, pad columns if needed
                                window_chars = []
                                for r in range(self.rank):
                                    row = window_rows[r]
                                    if j < len(row):
                                        seg = row[j:j+self.rank]
                                        if len(seg) < self.rank:
                                            seg += '_' * (self.rank - len(seg))
                                    else:
                                        seg = '_' * self.rank
                                    window_chars.extend(seg)
                                token = ''.join(window_chars)
                                tokens.append(token)
                                k1_list.append(i)
                                k2_list.append(j)
                            else:
                                # drop incomplete windows
                                continue
                        else:
                            # Full column window within padded rows
                            window_rows = rows[:self.rank]
                            window_chars = []
                            for r in range(self.rank):
                                seg = window_rows[r][j:j+self.rank]
                                window_chars.extend(seg)
                            token = ''.join(window_chars)
                            tokens.append(token)
                            k1_list.append(i)
                            k2_list.append(j)
                else:
                    # drop incomplete rows
                    continue
            else:
                # Full height window
                for j in range(0, W, step):
                    if j + self.rank > W:
                        if self.rank_mode == 'pad':
                            # Pad columns with '_'
                            cols_needed = self.rank - (W - j)
                            window_chars = []
                            for r in range(i, i+self.rank):
                                row = arr_2d[r]
                                if j < len(row):
                                    seg = row[j:j+self.rank]
                                    if len(seg) < self.rank:
                                        seg += '_' * (self.rank - len(seg))
                                else:
                                    seg = '_' * self.rank
                                window_chars.extend(seg)
                            token = ''.join(window_chars)
                            tokens.append(token)
                            k1_list.append(i)
                            k2_list.append(j)
                        else:
                            continue
                    else:
                        # Full window
                        window_chars = []
                        for r in range(i, i+self.rank):
                            row = arr_2d[r]
                            seg = row[j:j+self.rank]
                            window_chars.extend(seg)
                        token = ''.join(window_chars)
                        tokens.append(token)
                        k1_list.append(i)
                        k2_list.append(j)
        return tokens, k1_list, k2_list

    def batch_compute_Nk(self, k1_tensor, k2_tensor, token_indices):
        """
        Vectorized computation of N(k1,k2) vectors for a batch of positions and tokens.
        Args:
            k1_tensor: Tensor of row indices [batch_size]
            k2_tensor: Tensor of column indices [batch_size]
            token_indices: Tensor of token indices [batch_size]
        Returns:
            Tensor of N(k) vectors [batch_size, m]
        """
        # Get embeddings for all tokens [batch_size, m]
        x = self.embedding(token_indices)
        
        # Compute basis indices
        j1 = (k1_tensor % self.bas_row).long()
        j2 = (k2_tensor % self.bas_col).long()
        
        # Get row and column basis vectors
        B_row_j = self.B_row[j1]       # [batch_size, m]
        B_col_j = self.B_col[j2]       # [batch_size, m]
        
        # Compute scalar product: (B_row·x) * (B_col·x)
        # dot product per sample: sum over m
        s1 = torch.sum(B_row_j * x, dim=1)   # [batch_size]
        s2 = torch.sum(B_col_j * x, dim=1)   # [batch_size]
        scalar = s1 * s2                     # [batch_size]
        
        # Compute composite index j = j1 * bas_col + j2
        j = j1 * self.bas_col + j2           # [batch_size]
        
        # Get Acoeff vectors: A is [m, L], we want columns indexed by j -> [batch_size, m]
        A_j = self.Acoeff[:, j].permute(1, 0)  # [batch_size, m] after permute
        
        # Compute Nk = scalar * A_j [batch_size, m]
        Nk = scalar.unsqueeze(1) * A_j        # [batch_size, m]
            
        return Nk

    def describe(self, arr_2d):
        """Compute N(k) vectors for each window in the 2D array"""
        tokens, k1_list, k2_list = self.extract_tokens_and_positions(arr_2d)
        if not tokens:
            return []
        
        token_indices = self.token_to_indices(tokens)
        k1_tensor = torch.tensor(k1_list, dtype=torch.float32, device=self.device)
        k2_tensor = torch.tensor(k2_list, dtype=torch.float32, device=self.device)
        
        # Batch compute all Nk vectors
        N_batch = self.batch_compute_Nk(k1_tensor, k2_tensor, token_indices)
        return N_batch.detach().cpu().numpy()

    def S(self, arr_2d):
        """
        Compute list of S(l) = sum_{windows up to position order} N(k) for a 2D array.
        Note: The order is row-major (by k1, then k2) based on extraction order.
        """
        tokens, k1_list, k2_list = self.extract_tokens_and_positions(arr_2d)
        if not tokens:
            return []
            
        token_indices = self.token_to_indices(tokens)
        k1_tensor = torch.tensor(k1_list, dtype=torch.float32, device=self.device)
        k2_tensor = torch.tensor(k2_list, dtype=torch.float32, device=self.device)
        
        # Batch compute all Nk vectors
        N_batch = self.batch_compute_Nk(k1_tensor, k2_tensor, token_indices)
        
        # Compute cumulative sum (S vectors)
        S_cum = torch.cumsum(N_batch, dim=0)
        return [s.detach().cpu().numpy() for s in S_cum]

    def D(self, arrs_2d, t_list):
        """
        Compute mean squared deviation D across arrays:
        D = average over all positions of (N(k1,k2)-t_seq)^2
        """
        total_loss = 0.0
        total_positions = 0
        
        # Convert target vectors to tensor
        t_tensors = [torch.tensor(t, dtype=torch.float32, device=self.device) for t in t_list]
        
        for arr_2d, t in zip(arrs_2d, t_tensors):
            tokens, k1_list, k2_list = self.extract_tokens_and_positions(arr_2d)
            if not tokens:
                continue
                
            token_indices = self.token_to_indices(tokens)
            k1_tensor = torch.tensor(k1_list, dtype=torch.float32, device=self.device)
            k2_tensor = torch.tensor(k2_list, dtype=torch.float32, device=self.device)
            
            # Batch compute all Nk vectors
            N_batch = self.batch_compute_Nk(k1_tensor, k2_tensor, token_indices)
            
            # Compute loss for each position
            losses = torch.sum((N_batch - t) ** 2, dim=1)
            total_loss += losses.sum().item()
            total_positions += len(tokens)
                
        return total_loss / total_positions if total_positions else 0.0

    def d(self, arr_2d, t):
        """
        Compute pattern deviation value (d) for a single 2D array.
        """
        d_value = self.D([arr_2d], [t])
        return d_value     

    def reg_train(self, arrs_2d, t_list, max_iters=1000, tol=1e-8, learning_rate=0.01, 
               continued=False, decay_rate=1.0, print_every=10, batch_size=32,
               checkpoint_file=None, checkpoint_interval=10):
        """
        Train model using gradient descent with sequence-level batch processing.
        Memory-optimized version that processes 2D arrays in batches.
        
        Args:
            arrs_2d: List of 2D character arrays (list of list of list of chars)
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
            total_arrays = 0
            
            # Shuffle arrays for each epoch
            indices = list(range(len(arrs_2d)))
            random.shuffle(indices)
            
            # Process arrays in batches
            for batch_start in range(0, len(indices), batch_size):
                batch_indices = indices[batch_start:batch_start + batch_size]
                batch_arrs = [arrs_2d[idx] for idx in batch_indices]
                batch_targets = [t_tensors[idx] for idx in batch_indices]
                
                optimizer.zero_grad()
                batch_loss = 0.0
                batch_array_count = 0
                
                # Process each array in the current batch
                for arr_2d, target in zip(batch_arrs, batch_targets):
                    # Extract tokens and positions for current array
                    tokens, k1_list, k2_list = self.extract_tokens_and_positions(arr_2d)
                    if not tokens:
                        continue  # Skip empty arrays
                        
                    # Convert tokens to indices
                    token_indices = self.token_to_indices(tokens)
                    k1_tensor = torch.tensor(k1_list, dtype=torch.float32, device=self.device)
                    k2_tensor = torch.tensor(k2_list, dtype=torch.float32, device=self.device)
                    
                    # Batch compute all Nk vectors for current array
                    Nk_batch = self.batch_compute_Nk(k1_tensor, k2_tensor, token_indices)
                    
                    # Compute array-level prediction: average of all N(k) vectors
                    array_pred = torch.mean(Nk_batch, dim=0)
                    
                    # Calculate loss for this array (MSE between prediction and target)
                    array_loss = torch.sum((array_pred - target) ** 2)
                    batch_loss += array_loss
                    batch_array_count += 1
                    
                    # Clean up intermediate tensors
                    del Nk_batch, array_pred, token_indices, k1_tensor, k2_tensor
                    
                    # Periodically clear GPU cache
                    if batch_array_count % 10 == 0 and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                # Backpropagate batch loss if we have valid arrays
                if batch_array_count > 0:
                    batch_loss = batch_loss / batch_array_count
                    batch_loss.backward()
                    optimizer.step()
                    
                    total_loss += batch_loss.item() * batch_array_count
                    total_arrays += batch_array_count
                
                # Clear GPU cache after processing each batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Calculate average loss for this iteration
            if total_arrays > 0:
                avg_loss = total_loss / total_arrays
            else:
                avg_loss = 0.0
                
            history.append(avg_loss)
            
            # Update best model state
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_model_state = copy.deepcopy(self.state_dict())
            
            # Print training progress
            if it % print_every == 0 or it == max_iters - 1:
                current_lr = scheduler.get_last_lr()[0]
                print(f"GD Iter {it:3d}: Loss = {avg_loss:.6e}, LR = {current_lr:.6f}")
            
            # Save checkpoint
            if checkpoint_file and (it % checkpoint_interval == 0 or it == max_iters - 1):
                self._save_checkpoint(
                    checkpoint_file, it, history, optimizer, scheduler, best_loss
                )
            
            # Check convergence
            if abs(prev_loss - avg_loss) < tol:
                print(f"Converged after {it+1} iterations.")
                if best_model_state is not None and avg_loss > prev_loss:
                    self.load_state_dict(best_model_state)
                    print(f"Restored best model state with loss = {best_loss:.6e}")
                    history[-1] = best_loss
                break
            prev_loss = avg_loss
            
            # Update learning rate
            scheduler.step()
            
            # Final GPU memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Restore best model state if not converged
        if best_model_state is not None and best_loss < history[-1]:
            self.load_state_dict(best_model_state)
            print(f"Training ended. Restored best model state with loss = {best_loss:.6e}")
            history[-1] = best_loss
        
        # Compute and store statistics for reconstruction/generation
        self._compute_training_statistics(arrs_2d)
        self.trained = True
        
        return history

    def cls_train(self, arrs_2d, labels, num_classes, max_iters=1000, tol=1e-8, learning_rate=0.01,
                  continued=False, decay_rate=1.0, print_every=10, batch_size=32,
                  checkpoint_file=None, checkpoint_interval=10):
        """
        Train the model for multi-class classification using cross-entropy loss.
        
        Args:
            arrs_2d: List of 2D character arrays
            labels: List of integer class labels (0 to num_classes-1)
            num_classes: Number of classes
            max_iters: Maximum training iterations
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
        
        # Initialize classification head if not already done
        if self.classifier is None or self.num_classes != num_classes:
            self.classifier = nn.Linear(self.m, num_classes).to(self.device)
            self.num_classes = num_classes
        
        if not continued:
            self.reset_parameters()
        
        # Convert labels to tensor
        label_tensors = torch.tensor(labels, dtype=torch.long, device=self.device)
        
        # Setup optimizer and scheduler
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)
        
        # Cross-entropy loss
        criterion = nn.CrossEntropyLoss()
        
        # Training state variables
        history = []
        prev_loss = float('inf')
        best_loss = float('inf')
        best_model_state = None
        
        for it in range(max_iters):
            total_loss = 0.0
            total_arrays = 0
            correct_predictions = 0
            
            # Shuffle arrays for each epoch
            indices = list(range(len(arrs_2d)))
            random.shuffle(indices)
            
            # Process arrays in batches
            for batch_start in range(0, len(indices), batch_size):
                batch_indices = indices[batch_start:batch_start + batch_size]
                batch_arrs = [arrs_2d[idx] for idx in batch_indices]
                batch_labels = label_tensors[batch_indices]
                
                optimizer.zero_grad()
                batch_loss = 0.0
                batch_logits = []
                
                # Process each array in the batch
                for arr_2d in batch_arrs:
                    # Extract tokens and positions
                    tokens, k1_list, k2_list = self.extract_tokens_and_positions(arr_2d)
                    if not tokens:
                        # For empty arrays, use zero vector
                        array_vector = torch.zeros(self.m, device=self.device)
                    else:
                        token_indices = self.token_to_indices(tokens)
                        k1_tensor = torch.tensor(k1_list, dtype=torch.float32, device=self.device)
                        k2_tensor = torch.tensor(k2_list, dtype=torch.float32, device=self.device)
                        
                        # Compute Nk vectors for all windows
                        Nk_batch = self.batch_compute_Nk(k1_tensor, k2_tensor, token_indices)
                        
                        # Compute array-level vector: average of all N(k) vectors
                        array_vector = torch.mean(Nk_batch, dim=0)
                        
                        # Clean up
                        del Nk_batch, token_indices, k1_tensor, k2_tensor
                    
                    # Get logits through classification head
                    logits = self.classifier(array_vector.unsqueeze(0))
                    batch_logits.append(logits)
                
                # Stack all logits and compute loss
                if batch_logits:
                    all_logits = torch.cat(batch_logits, dim=0)
                    loss = criterion(all_logits, batch_labels)
                    loss.backward()
                    optimizer.step()
                    
                    batch_loss = loss.item()
                    total_loss += batch_loss * len(batch_arrs)
                    total_arrays += len(batch_arrs)
                    
                    # Calculate accuracy
                    with torch.no_grad():
                        predictions = torch.argmax(all_logits, dim=1)
                        correct_predictions += (predictions == batch_labels).sum().item()
                
                # Clear GPU cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Calculate average loss and accuracy
            if total_arrays > 0:
                avg_loss = total_loss / total_arrays
                accuracy = correct_predictions / total_arrays
            else:
                avg_loss = 0.0
                accuracy = 0.0
                
            history.append(avg_loss)
            
            # Update best model state
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_model_state = copy.deepcopy(self.state_dict())
            
            # Print progress
            if it % print_every == 0 or it == max_iters - 1:
                current_lr = scheduler.get_last_lr()[0]
                print(f"CLS-Train Iter {it:3d}: Loss = {avg_loss:.6e}, Acc = {accuracy:.4f}, LR = {current_lr:.6f}")
            
            # Save checkpoint
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
                if best_model_state is not None:
                    self.load_state_dict(best_model_state)
                break
                
            prev_loss = avg_loss
            
            # Learning rate scheduling
            scheduler.step()
        
        self.trained = True
        
        return history

    def lbl_train(self, arrs_2d, labels, num_labels, max_iters=1000, tol=1e-8, learning_rate=0.01, 
                 continued=False, decay_rate=1.0, print_every=10, batch_size=32,
                 checkpoint_file=None, checkpoint_interval=10, pos_weight=None):
        """
        Train the model for multi-label classification using binary cross-entropy loss.
        
        Args:
            arrs_2d: List of 2D character arrays
            labels: List of binary label vectors (list of lists) or 2D numpy array/torch tensor
            num_labels: Number of labels
            max_iters: Maximum training iterations
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
        
        # Setup loss function with optional positive class weighting
        if pos_weight is not None:
            pos_weight_tensor = torch.tensor(pos_weight, dtype=torch.float32, device=self.device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        else:
            criterion = nn.BCEWithLogitsLoss()
        
        # Setup optimizer and scheduler
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)
        
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
            total_arrays = 0
            
            # Shuffle arrays for each epoch
            indices = list(range(len(arrs_2d)))
            random.shuffle(indices)
            
            # Process arrays in batches
            for batch_start in range(0, len(indices), batch_size):
                batch_indices = indices[batch_start:batch_start + batch_size]
                batch_arrs = [arrs_2d[idx] for idx in batch_indices]
                batch_labels = labels_tensor[batch_indices]
                
                optimizer.zero_grad()
                batch_loss = 0.0
                batch_correct = 0
                batch_predictions = 0
                
                # Process each array in the batch
                batch_predictions_list = []
                for arr_2d in batch_arrs:
                    # Extract tokens and positions
                    tokens, k1_list, k2_list = self.extract_tokens_and_positions(arr_2d)
                    if not tokens:
                        continue
                        
                    token_indices = self.token_to_indices(tokens)
                    k1_tensor = torch.tensor(k1_list, dtype=torch.float32, device=self.device)
                    k2_tensor = torch.tensor(k2_list, dtype=torch.float32, device=self.device)
                    
                    # Compute Nk vectors
                    Nk_batch = self.batch_compute_Nk(k1_tensor, k2_tensor, token_indices)
                    
                    # Compute array representation: average of all N(k) vectors
                    array_representation = torch.mean(Nk_batch, dim=0)
                    
                    # Pass through label head to get logits
                    logits = self.labeller(array_representation)
                    batch_predictions_list.append(logits)
                    
                    # Clean up
                    del Nk_batch, array_representation, token_indices, k1_tensor, k2_tensor
                
                # Stack predictions for the batch
                if batch_predictions_list:
                    batch_logits = torch.stack(batch_predictions_list, dim=0)
                    
                    # Calculate loss
                    batch_loss = criterion(batch_logits, batch_labels)
                    
                    # Calculate accuracy
                    with torch.no_grad():
                        probs = torch.sigmoid(batch_logits)
                        predictions = (probs > 0.5).float()
                        batch_correct = (predictions == batch_labels).sum().item()
                        batch_predictions = batch_labels.numel()
                    
                    # Backpropagate
                    batch_loss.backward()
                    optimizer.step()
                    
                    total_loss += batch_loss.item() * len(batch_arrs)
                    total_correct += batch_correct
                    total_predictions += batch_predictions
                    total_arrays += len(batch_arrs)
                
                # Clear GPU cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Calculate average loss and accuracy
            if total_arrays > 0:
                avg_loss = total_loss / total_arrays
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
            
            # Print progress
            if it % print_every == 0 or it == max_iters - 1:
                current_lr = scheduler.get_last_lr()[0]
                print(f"MLC-Train Iter {it:3d}: Loss = {avg_loss:.6e}, Acc = {avg_acc:.4f}, LR = {current_lr:.6f}")
            
            # Save checkpoint
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
                if best_model_state is not None:
                    self.load_state_dict(best_model_state)
                break
                
            prev_loss = avg_loss
            
            # Learning rate scheduling
            scheduler.step()
        
        self.trained = True
        
        return loss_history, acc_history

    def self_train(self, arrs_2d, max_iters=1000, tol=1e-8, 
               learning_rate=0.01, continued=False, decay_rate=1.0, 
               print_every=10, batch_size=1024, checkpoint_file=None, 
               checkpoint_interval=10):
        """
        Train the model using self-supervised learning with memory-optimized batch processing.
        Self-training uses the gap-filling objective: predict token embeddings from position information.
        
        Args:
            arrs_2d: List of 2D character arrays
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
            
            # Shuffle arrays for each epoch
            indices = list(range(len(arrs_2d)))
            random.shuffle(indices)
            
            # Process arrays in shuffled order
            for arr_idx in indices:
                arr_2d = arrs_2d[arr_idx]
                tokens, k1_list, k2_list = self.extract_tokens_and_positions(arr_2d)
                if not tokens:
                    continue
                    
                token_indices = self.token_to_indices(tokens)
                array_samples = []
                
                # Generate samples for current array: each window is a sample (k1, k2, token_idx)
                for k1, k2, token_idx in zip(k1_list, k2_list, token_indices):
                    array_samples.append((k1, k2, token_idx.item()))
                
                if not array_samples:
                    continue
                
                # Process samples from current array in batches
                for batch_start in range(0, len(array_samples), batch_size):
                    batch_samples = array_samples[batch_start:batch_start + batch_size]
                    
                    optimizer.zero_grad()
                    
                    # Prepare batch data
                    k1_list_batch = []
                    k2_list_batch = []
                    token_idx_list = []
                    
                    for sample in batch_samples:
                        k1, k2, token_idx = sample
                        k1_list_batch.append(k1)
                        k2_list_batch.append(k2)
                        token_idx_list.append(token_idx)
                    
                    # Create tensors
                    k1_tensor = torch.tensor(k1_list_batch, dtype=torch.float32, device=self.device)
                    k2_tensor = torch.tensor(k2_list_batch, dtype=torch.float32, device=self.device)
                    token_indices_tensor = torch.tensor(token_idx_list, device=self.device)
                    
                    # Batch compute Nk for current tokens
                    Nk_batch = self.batch_compute_Nk(k1_tensor, k2_tensor, token_indices_tensor)
                    
                    # Get target embeddings (same as current tokens for gap-filling)
                    targets = self.embedding(token_indices_tensor)
                    
                    # Compute loss: mean squared error between Nk and token embeddings
                    loss = torch.mean(torch.sum((Nk_batch - targets) ** 2, dim=1))
                    loss.backward()
                    optimizer.step()
                    
                    batch_loss = loss.item() * len(batch_samples)
                    total_loss += batch_loss
                    total_samples += len(batch_samples)
                    
                    # Clean up
                    del k1_tensor, k2_tensor, token_indices_tensor, Nk_batch, targets, loss
                    
                    # Periodically clear GPU cache
                    if total_samples % 1000 == 0 and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                # Clear array-specific tensors
                del token_indices, array_samples
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Calculate average loss for this iteration
            if total_samples > 0:
                avg_loss = total_loss / total_samples
            else:
                avg_loss = 0.0
                
            history.append(avg_loss)
            
            # Update best model state
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_model_state = copy.deepcopy(self.state_dict())
            
            # Print progress
            if it % print_every == 0 or it == max_iters - 1:
                current_lr = scheduler.get_last_lr()[0]
                print(f"Self-Train Iter {it:3d}: loss = {avg_loss:.6f}, LR = {current_lr:.6f}, "
                      f"Samples = {total_samples}")
            
            # Save checkpoint
            if checkpoint_file and (it % checkpoint_interval == 0 or it == max_iters - 1):
                self._save_checkpoint(
                    checkpoint_file, it, history, optimizer, scheduler, best_loss
                )
            
            # Check convergence
            if abs(prev_loss - avg_loss) < tol:
                print(f"Converged after {it+1} iterations")
                if best_model_state is not None and avg_loss > prev_loss:
                    self.load_state_dict(best_model_state)
                    print(f"Restored best model state with loss = {best_loss:.6f}")
                    history[-1] = best_loss
                break
            prev_loss = avg_loss
            
            # Update learning rate
            scheduler.step()
            
            # Final GPU memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Restore best model state if needed
        if best_model_state is not None and best_loss < history[-1]:
            self.load_state_dict(best_model_state)
            print(f"Training ended. Restored best model state with loss = {best_loss:.6f}")
            history[-1] = best_loss
        
        # Compute and store statistics for reconstruction/generation
        self._compute_training_statistics(arrs_2d)
        self.trained = True
        
        return history

    def _save_checkpoint(self, checkpoint_file, iteration, history, optimizer, scheduler, best_loss):
        """
        Save training checkpoint with complete training state
        """
        checkpoint = {
            'iteration': iteration,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'history': history,
            'best_loss': best_loss,
            'config': {
                'charset': self.charset,
                'vec_dim': self.m,
                'bas_row': self.bas_row,
                'bas_col': self.bas_col,
                'rank': self.rank,
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

    def _compute_training_statistics(self, arrs_2d, batch_size=10):
        """Compute and store statistics for reconstruction and generation with memory optimization"""
        total_window_count = 0
        total_t = torch.zeros(self.m, device=self.device)
        
        with torch.no_grad():
            for i in range(0, len(arrs_2d), batch_size):
                batch_arrs = arrs_2d[i:i+batch_size]
                batch_window_count = 0
                batch_t_sum = torch.zeros(self.m, device=self.device)
                
                for arr_2d in batch_arrs:
                    try:
                        tokens, k1_list, k2_list = self.extract_tokens_and_positions(arr_2d)
                        batch_window_count += len(tokens)
                        
                        if tokens:
                            token_indices = self.token_to_indices(tokens)
                            k1_tensor = torch.tensor(k1_list, dtype=torch.float32, device=self.device)
                            k2_tensor = torch.tensor(k2_list, dtype=torch.float32, device=self.device)
                            Nk_batch = self.batch_compute_Nk(k1_tensor, k2_tensor, token_indices)
                            batch_t_sum += Nk_batch.sum(dim=0)
                            
                            # Clean up
                            del token_indices, k1_tensor, k2_tensor, Nk_batch
                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            print(f"Warning: Memory error computing statistics for array. Skipping.")
                            continue
                        else:
                            raise e
                
                total_window_count += batch_window_count
                total_t += batch_t_sum
                
                # Clean batch
                del batch_t_sum
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        self.mean_window_count = total_window_count / len(arrs_2d) if arrs_2d else 0
        self.mean_t = (total_t / total_window_count).cpu().numpy() if total_window_count > 0 else np.zeros(self.m)
    
    def predict_t(self, arr_2d):
        """Predict target vector as average of N(k) vectors over all windows"""
        tokens, k1_list, k2_list = self.extract_tokens_and_positions(arr_2d)
        if not tokens:
            return np.zeros(self.m)
        
        # Compute all Nk vectors
        Nk = self.describe(arr_2d)
        return np.mean(Nk, axis=0)
    
    def predict_c(self, arr_2d):
        """
        Predict class label for a 2D array using the classification head.
        
        Args:
            arr_2d: Input 2D character array
            
        Returns:
            tuple: (predicted_class, class_probabilities)
        """
        if self.classifier is None:
            raise ValueError("Model must be trained first for classification")
        
        # Get array vector representation
        array_vector = self.predict_t(arr_2d)
        array_vector_tensor = torch.tensor(array_vector, dtype=torch.float32, device=self.device)
        
        # Get logits through classification head
        with torch.no_grad():
            logits = self.classifier(array_vector_tensor.unsqueeze(0))
            probabilities = torch.softmax(logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            
        return predicted_class, probabilities[0].cpu().numpy()

    def predict_l(self, arr_2d, threshold=0.5):
        """
        Predict multi-label classification for a 2D array.
        
        Args:
            arr_2d: Input 2D character array
            threshold: Probability threshold for binary classification (default: 0.5)
            
        Returns:
            numpy.ndarray: Binary label predictions (0 or 1 for each label)
            numpy.ndarray: Probability scores for each label
        """
        assert self.labeller is not None, "Model must be trained first for label prediction"
        
        tokens, k1_list, k2_list = self.extract_tokens_and_positions(arr_2d)
        if not tokens:
            # Return zeros if no tokens
            return np.zeros(self.num_labels, dtype=np.float32), np.zeros(self.num_labels, dtype=np.float32)
        
        token_indices = self.token_to_indices(tokens)
        k1_tensor = torch.tensor(k1_list, dtype=torch.float32, device=self.device)
        k2_tensor = torch.tensor(k2_list, dtype=torch.float32, device=self.device)
        
        # Compute N(k) vectors for all windows
        Nk_batch = self.batch_compute_Nk(k1_tensor, k2_tensor, token_indices)
        
        # Compute array representation: average of all N(k) vectors
        array_representation = torch.mean(Nk_batch, dim=0)
        
        # Pass through label head to get logits
        with torch.no_grad():
            logits = self.labeller(array_representation)
            probs = torch.sigmoid(logits).cpu().numpy()
        
        # Apply threshold to get binary predictions
        binary_preds = (probs > threshold).astype(np.float32)
        
        return binary_preds, probs

    def reconstruct(self, H, W, tau=0.0):
        """
        Reconstruct a 2D array of size H×W by generating window tokens in row-major order.
        Assumes step size = rank (non-overlapping) for simplicity.
        """
        assert self.trained, "Model must be trained first"
        if tau < 0:
            raise ValueError("Temperature must be non-negative")
        
        # Determine number of windows in each dimension
        # Since we reconstruct with step = rank, the array must be divisible by rank,
        # but we can pad if needed. Here we simply require H and W to be multiples of rank.
        # For simplicity, we adjust H and W to be multiples of rank.
        rank = self.rank
        H_adj = ((H + rank - 1) // rank) * rank
        W_adj = ((W + rank - 1) // rank) * rank
        
        num_windows_h = H_adj // rank
        num_windows_w = W_adj // rank
        
        mean_t_tensor = torch.tensor(self.mean_t, dtype=torch.float32, device=self.device)
        
        # Precompute all token embeddings
        all_token_indices = torch.arange(len(self.tokens), device=self.device)
        all_embeddings = self.embedding(all_token_indices)  # not directly used
        
        # Initialize a grid for the reconstructed characters
        # We'll fill it window by window, possibly with overlap if step<rank, but we use step=rank.
        # For simplicity, we create a 2D array of '_' and fill.
        result = [['_' for _ in range(W_adj)] for __ in range(H_adj)]
        
        # For each window position (i, j) in row-major order
        for i in range(0, num_windows_h):
            for j in range(0, num_windows_w):
                k1 = i * rank  # top-left row index
                k2 = j * rank  # top-left column index
                # Compute Nk for all tokens at this position
                k1_tensor = torch.tensor([k1] * len(self.tokens), dtype=torch.float32, device=self.device)
                k2_tensor = torch.tensor([k2] * len(self.tokens), dtype=torch.float32, device=self.device)
                Nk_all = self.batch_compute_Nk(k1_tensor, k2_tensor, all_token_indices)
                
                # Compute errors with respect to mean_t
                errors = torch.sum((Nk_all - mean_t_tensor) ** 2, dim=1)
                scores = -errors  # higher score = better
                
                if tau == 0:  # Deterministic selection
                    max_idx = torch.argmax(scores).item()
                    best_tok = self.idx_to_token[max_idx]
                else:  # Stochastic selection
                    probs = torch.softmax(scores / tau, dim=0).detach().cpu().numpy()
                    chosen_idx = random.choices(range(len(self.tokens)), weights=probs, k=1)[0]
                    best_tok = self.idx_to_token[chosen_idx]
                
                # Fill the window in the result array
                # The token is a string of length rank*rank, row-major
                for r in range(rank):
                    for c in range(rank):
                        result[k1 + r][k2 + c] = best_tok[r * rank + c]
        
        # Trim to original H, W
        final_result = [row[:W] for row in result[:H]]
        return final_result

    def save(self, filename):
        """Save model state to file"""
        save_dict = {
            'state_dict': self.state_dict(),
            'mean_t': self.mean_t if hasattr(self, 'mean_t') else None,
            'mean_window_count': self.mean_window_count if hasattr(self, 'mean_window_count') else None,
            'trained': self.trained,
            'num_classes': self.num_classes,
            'num_labels': self.num_labels
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
    
    print("="*50)
    print("Spatial Dual Descriptor RN2 - PyTorch GPU Accelerated Version (2D)")
    print("Optimized with batch processing")
    print("="*50)
    
    # Set random seeds to ensure reproducibility
    torch.manual_seed(11)
    random.seed(11)
    
    charset = ['A','C','G','T']
    vec_dim = 3
    bas_row = 10
    bas_col = 10
    array_num = 50

    # Generate 2D arrays of random size (height 20-30, width 20-30) and random targets
    arrays_2d = []
    t_list = []
    for _ in range(array_num):
        H = random.randint(20, 30)
        W = random.randint(20, 30)
        arr = [[random.choice(charset) for _ in range(W)] for __ in range(H)]
        arrays_2d.append(arr)
        t_list.append([random.uniform(-1,1) for _ in range(vec_dim)])

    # === Gradient Descent Training Example ===
    print("\n" + "="*50)
    print("Testing Gradient Descent Training on 2D Arrays")
    print("="*50)

    # Create new model instance with GPU acceleration
    dd = SpatialDualDescriptorRN2(
        charset, 
        rank=2, 
        vec_dim=vec_dim, 
        bas_row=bas_row,
        bas_col=bas_col,
        mode='linear', 
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Display device information
    print(f"\nUsing device: {dd.device}")
    print(f"Number of tokens: {len(dd.tokens)}")

    # Train using gradient descent
    print("\nStarting gradient descent training...")
    grad_history = dd.reg_train(
        arrays_2d, 
        t_list,
        learning_rate=0.05,
        max_iters=50,
        tol=1e-8,
        decay_rate=0.99,
        print_every=5,
        batch_size=8
    )

    # Predict target vector for first array
    arr = arrays_2d[0]
    t_pred = dd.predict_t(arr)
    print(f"\nPredicted t for first array: {[round(x.item(), 4) for x in t_pred]}")    
    
    # Calculate flattened correlation between predicted and actual targets
    pred_t_list = [dd.predict_t(arr) for arr in arrays_2d]
    
    # Predictions and actuals for correlation calculation
    corr_sum = 0.0
    for i in range(dd.m):
        actu_t = [t_vec[i] for t_vec in t_list]
        pred_t = [t_vec[i] for t_vec in pred_t_list]
        corr = correlation(actu_t, pred_t)
        print(f"Prediction correlation for dimension {i}: {corr:.4f}")
        corr_sum += corr
    corr_avg = corr_sum / dd.m
    print(f"Average correlation: {corr_avg:.4f}")

    # Reconstruct representative 2D array
    H_rec = 20
    W_rec = 20
    rec_det = dd.reconstruct(H_rec, W_rec, tau=0.0)
    rec_sto = dd.reconstruct(H_rec, W_rec, tau=0.5)
    print("\nDeterministic Reconstruction (first 5 rows):")
    for row in rec_det[:5]:
        print(''.join(row))
    print("Stochastic Reconstruction (tau=0.5, first 5 rows):")
    for row in rec_sto[:5]:
        print(''.join(row))
    
    # === Classification Task ===
    print("\n" + "="*50)
    print("Classification Task on 2D Arrays")
    print("="*50)
    
    # Generate classification data
    num_classes = 3
    class_arrays = []
    class_labels = []
    
    # Create arrays with different patterns for each class
    for class_id in range(num_classes):
        for _ in range(30):  # 30 arrays per class
            H = random.randint(15, 25)
            W = random.randint(15, 25)
            if class_id == 0:
                # Class 0: High A content
                arr = [[random.choices(['A','C','G','T'], weights=[0.6,0.1,0.1,0.2])[0] for _ in range(W)] for __ in range(H)]
            elif class_id == 1:
                # Class 1: High GC content
                arr = [[random.choices(['A','C','G','T'], weights=[0.1,0.4,0.4,0.1])[0] for _ in range(W)] for __ in range(H)]
            else:
                # Class 2: Balanced
                arr = [[random.choice(charset) for _ in range(W)] for __ in range(H)]
            class_arrays.append(arr)
            class_labels.append(class_id)
    
    # Initialize new model for classification
    dd_cls = SpatialDualDescriptorRN2(
        charset, 
        vec_dim=vec_dim, 
        bas_row=bas_row,
        bas_col=bas_col,
        rank=3, 
        mode='linear',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Train for classification
    print("\n" + "="*50)
    print("Starting Classification Training")
    print("="*50)
    history = dd_cls.cls_train(class_arrays, class_labels, num_classes, 
                              max_iters=10, tol=1e-8, learning_rate=0.05,
                              decay_rate=0.99, batch_size=8, print_every=5)
    
    # Show prediction results on the training dataset
    print("\n" + "="*50)
    print("Prediction results")
    print("="*50)
    
    correct = 0
    for arr, true_label in zip(class_arrays, class_labels):
        pred_class, probs = dd_cls.predict_c(arr)
        if pred_class == true_label:
            correct += 1
    
    accuracy = correct / len(class_arrays)
    print(f"Accuracy: {accuracy:.4f} ({correct}/{len(class_arrays)})")
    
    # Show some example predictions
    print("\nExample predictions:")
    for i in range(min(5, len(class_arrays))):
        pred_class, probs = dd_cls.predict_c(class_arrays[i])
        print(f"Array {i+1}: True={class_labels[i]}, Pred={pred_class}, Probs={[f'{p:.3f}' for p in probs]}")

    # === Multi-Label Classification Task ===
    print("\n\n" + "="*50)
    print("Multi-Label Classification Model on 2D Arrays")
    print("="*50)
    
    # Generate arrays with random multi-labels
    num_labels = 4
    label_arrays = []
    labels = []
    for _ in range(30):
        H = random.randint(20, 30)
        W = random.randint(20, 30)
        arr = [[random.choice(charset) for _ in range(W)] for __ in range(H)]
        label_arrays.append(arr)
        # Random binary labels
        label_vec = [random.random() > 0.7 for _ in range(num_labels)]
        labels.append([1.0 if x else 0.0 for x in label_vec])
    
    dd_lbl = SpatialDualDescriptorRN2(
        charset, 
        vec_dim=vec_dim, 
        bas_row=bas_row,
        bas_col=bas_col,
        rank=3, 
        mode='linear',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Training multi-label classification model
    print("\n" + "="*50)
    print("Starting Multi-Label Classification Training")
    print("="*50)
       
    loss_history, acc_history = dd_lbl.lbl_train(
        label_arrays, labels, num_labels,
        max_iters=50, 
        tol=1e-8, 
        learning_rate=0.05, 
        decay_rate=0.99, 
        print_every=10, 
        batch_size=8
    )
    
    print(f"\nFinal training loss: {loss_history[-1]:.6f}")
    print(f"Final training accuracy: {acc_history[-1]:.4f}")
    
    # Show prediction results on training set
    print("\n" + "="*50)
    print("Prediction Results")
    print("="*50)
    
    all_correct = 0
    total = 0
    
    for arr, true_labels in zip(label_arrays, labels):
        pred_binary, pred_probs = dd_lbl.predict_l(arr, threshold=0.5)
        true_labels_np = np.array(true_labels)
        correct = np.all(pred_binary == true_labels_np)
        all_correct += correct
        total += 1
        
        if total <= 3:
            print(f"\nArray {total}:")
            print(f"True labels: {true_labels_np}")
            print(f"Predicted binary: {pred_binary}")
            print(f"Predicted probabilities: {[f'{p:.4f}' for p in pred_probs]}")
            print(f"Correct: {correct}")
    
    accuracy = all_correct / total if total > 0 else 0.0
    print(f"\nOverall prediction accuracy: {accuracy:.4f} ({all_correct}/{total} arrays)")
    
    # Example of label prediction for a new array
    print("\n" + "="*50)
    print("Label Prediction Example")
    print("="*50)
    
    test_arr = [[random.choice(charset) for _ in range(25)] for __ in range(25)]
    binary_pred, probs_pred = dd_lbl.predict_l(test_arr, threshold=0.5)
    print(f"Test array shape: {len(test_arr)}x{len(test_arr[0])}")
    print(f"Predicted binary labels: {binary_pred}")
    print(f"Predicted probabilities: {[f'{p:.4f}' for p in probs_pred]}")
    
    # === Self-Training Example ===
    random.seed(1)
    print("\n=== Self-Training on 2D Arrays ===")
    
    # Generate training arrays
    train_arrays = []
    for i in range(20):
        H = random.randint(15, 25)
        W = random.randint(15, 25)
        arr = [[random.choice(charset) for _ in range(W)] for __ in range(H)]
        train_arrays.append(arr)
    
    dd_self = SpatialDualDescriptorRN2(
        charset, 
        rank=3,
        vec_dim=vec_dim,
        bas_row=bas_row,
        bas_col=bas_col,
        mode='linear',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )    
    
    print("\n=== Starting Self-Training ===")
    self_history = dd_self.self_train(
        train_arrays,
        max_iters=30,
        learning_rate=0.01,
        decay_rate=0.995,
        print_every=10,
        batch_size=512
    )
    
    # Reconstruct a 2D array
    print("\n=== Reconstructing 2D Array ===")
    rec_det = dd_self.reconstruct(H=20, W=20, tau=0.0)
    print("Deterministic Reconstruction (first 5 rows):")
    for row in rec_det[:5]:
        print(''.join(row))
    
    # Save and load model
    print("\n=== Model Persistence Test ===")
    dd_self.save("spatial_self_trained_model.pt")
    dd_loaded = SpatialDualDescriptorRN2(
        charset, 
        rank=3,
        vec_dim=vec_dim,
        bas_row=bas_row,
        bas_col=bas_col,
        mode='linear',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    dd_loaded.load("spatial_self_trained_model.pt")
    print("Model loaded successfully. Reconstructing with loaded model:")
    rec_loaded = dd_loaded.reconstruct(H=10, W=10, tau=0.0)
    for row in rec_loaded:
        print(''.join(row))
    
    print("\n=== All Demos Completed Successfully ===")
