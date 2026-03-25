# Copyright (C) 2005-2025, Bin-Guang Ma (mbg@mail.hzau.edu.cn); SPDX-License-Identifier: MIT
# Spatial Dual Descriptor Vector class (Random AB matrix form) for 4D character arrays implemented with PyTorch
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

class SpatialDualDescriptorRN4(nn.Module):
    """
    Spatial Dual Descriptor for 4D character arrays with GPU acceleration using PyTorch:
      - Learnable coefficient matrix Acoeff ∈ R^{m×L} (L = bas1 * bas2 * bas3 * bas4)
      - Learnable basis matrices: B1, B2, B3, B4 (each of shape (bas_i, m))
      - Token embeddings M: token → R^m
      - Optimized with batch processing for GPU acceleration
      - Supports both linear and nonlinear tokenization
      - Supports regression, classification, and multi-label classification tasks
    """
    def __init__(self, charset, vec_dim=4, bas1=4, bas2=4, bas3=4, bas4=4, rank=1, rank_mode='drop', 
                 mode='linear', user_step=None, device='cuda'):
        super().__init__()
        self.charset = list(charset)
        self.m = vec_dim
        self.bas1 = bas1
        self.bas2 = bas2
        self.bas3 = bas3
        self.bas4 = bas4
        self.L = bas1 * bas2 * bas3 * bas4          # total number of basis indices
        self.rank = rank
        self.rank_mode = rank_mode
        assert mode in ('linear','nonlinear')
        self.mode = mode
        self.step = user_step
        self.trained = False
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Generate all possible tokens from hypercubic windows of size rank^4.
        # Each window is flattened into a string: order: dim1 fastest, then dim2, dim3, dim4 slowest.
        toks = []
        if self.rank_mode == 'pad':
            # For padding mode, we generate tokens of length rank^4 by padding with '_'
            for r in range(1, self.rank+1):
                for prefix in itertools.product(self.charset, repeat=r):
                    tok = ''.join(prefix).ljust(self.rank, '_')
                    toks.append(tok)
        else:
            # 'drop' mode: only full-size windows are considered
            toks = [''.join(p) for p in itertools.product(self.charset, repeat=self.rank**4)]
        # Ensure tokens are sorted and unique
        self.tokens = sorted(set(toks))
        self.token_to_idx = {token: idx for idx, token in enumerate(self.tokens)}
        self.idx_to_token = {idx: token for idx, token in enumerate(self.tokens)}
        
        # Token embeddings
        self.embedding = nn.Embedding(len(self.tokens), self.m)
        
        # Coefficient matrix Acoeff: m × L (L = bas1 * bas2 * bas3 * bas4)
        self.Acoeff = nn.Parameter(torch.empty(self.m, self.L))
        
        # Basis matrices for each dimension
        self.B1 = nn.Parameter(torch.empty(self.bas1, self.m))
        self.B2 = nn.Parameter(torch.empty(self.bas2, self.m))
        self.B3 = nn.Parameter(torch.empty(self.bas3, self.m))
        self.B4 = nn.Parameter(torch.empty(self.bas4, self.m))
        
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
        nn.init.uniform_(self.B1, -0.1, 0.1)
        nn.init.uniform_(self.B2, -0.1, 0.1)
        nn.init.uniform_(self.B3, -0.1, 0.1)
        nn.init.uniform_(self.B4, -0.1, 0.1)
        
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
    
    def extract_tokens_and_positions(self, arr_4d):
        """
        Extract hypercubic windows of size rank^4 from a 4D character array.
        The array is assumed to be a nested list: arr_4d[d1][d2][d3][d4] (dim1, dim2, dim3, dim4).
        Returns:
            tokens: list of window strings (flattened in order: dim1 fastest, then dim2, dim3, dim4 slowest)
            k1_list: list of indices in dimension 1
            k2_list: list of indices in dimension 2
            k3_list: list of indices in dimension 3
            k4_list: list of indices in dimension 4
        """
        D1 = len(arr_4d)
        D2 = len(arr_4d[0]) if D1 > 0 else 0
        D3 = len(arr_4d[0][0]) if D2 > 0 else 0
        D4 = len(arr_4d[0][0][0]) if D3 > 0 else 0
        tokens = []
        k1_list = []
        k2_list = []
        k3_list = []
        k4_list = []
        
        # Determine step size
        if self.mode == 'linear':
            step = 1
        else:
            step = self.step or self.rank
        
        # Helper to get character at (i1,i2,i3,i4) with padding '_' if out of bounds
        def get_char(i1, i2, i3, i4):
            if 0 <= i1 < D1 and 0 <= i2 < D2 and 0 <= i3 < D3 and 0 <= i4 < D4:
                return arr_4d[i1][i2][i3][i4]
            else:
                return '_'
        
        # Slide over the 4D array
        for i1 in range(0, D1, step):
            for i2 in range(0, D2, step):
                for i3 in range(0, D3, step):
                    for i4 in range(0, D4, step):
                        # Check if we have a full hypercube (rank×rank×rank×rank) within bounds without padding
                        full_in_bounds = (i1 + self.rank <= D1) and (i2 + self.rank <= D2) and \
                                         (i3 + self.rank <= D3) and (i4 + self.rank <= D4)
                        if full_in_bounds:
                            # Extract full hypercube
                            window_chars = []
                            for d1 in range(self.rank):
                                for d2 in range(self.rank):
                                    for d3 in range(self.rank):
                                        for d4 in range(self.rank):
                                            window_chars.append(arr_4d[i1+d1][i2+d2][i3+d3][i4+d4])
                            token = ''.join(window_chars)
                            tokens.append(token)
                            k1_list.append(i1)
                            k2_list.append(i2)
                            k3_list.append(i3)
                            k4_list.append(i4)
                        else:
                            if self.rank_mode == 'pad':
                                # Pad with '_' to fill the hypercube
                                window_chars = []
                                for d1 in range(self.rank):
                                    for d2 in range(self.rank):
                                        for d3 in range(self.rank):
                                            for d4 in range(self.rank):
                                                window_chars.append(get_char(i1+d1, i2+d2, i3+d3, i4+d4))
                                token = ''.join(window_chars)
                                tokens.append(token)
                                k1_list.append(i1)
                                k2_list.append(i2)
                                k3_list.append(i3)
                                k4_list.append(i4)
                            # else drop: skip this window
        return tokens, k1_list, k2_list, k3_list, k4_list

    def batch_compute_Nk(self, k1_tensor, k2_tensor, k3_tensor, k4_tensor, token_indices):
        """
        Vectorized computation of N(k1,k2,k3,k4) vectors for a batch of positions and tokens.
        Args:
            k1_tensor: Tensor of indices in dimension 1 [batch_size]
            k2_tensor: Tensor of indices in dimension 2 [batch_size]
            k3_tensor: Tensor of indices in dimension 3 [batch_size]
            k4_tensor: Tensor of indices in dimension 4 [batch_size]
            token_indices: Tensor of token indices [batch_size]
        Returns:
            Tensor of N(k) vectors [batch_size, m]
        """
        # Get embeddings for all tokens [batch_size, m]
        x = self.embedding(token_indices)
        
        # Compute basis indices
        j1 = (k1_tensor % self.bas1).long()
        j2 = (k2_tensor % self.bas2).long()
        j3 = (k3_tensor % self.bas3).long()
        j4 = (k4_tensor % self.bas4).long()
        
        # Get basis vectors
        B1_j = self.B1[j1]       # [batch_size, m]
        B2_j = self.B2[j2]       # [batch_size, m]
        B3_j = self.B3[j3]       # [batch_size, m]
        B4_j = self.B4[j4]       # [batch_size, m]
        
        # Compute scalar product: (B1·x) * (B2·x) * (B3·x) * (B4·x)
        s1 = torch.sum(B1_j * x, dim=1)   # [batch_size]
        s2 = torch.sum(B2_j * x, dim=1)   # [batch_size]
        s3 = torch.sum(B3_j * x, dim=1)   # [batch_size]
        s4 = torch.sum(B4_j * x, dim=1)   # [batch_size]
        scalar = s1 * s2 * s3 * s4        # [batch_size]
        
        # Compute composite index j = j1*bas2*bas3*bas4 + j2*bas3*bas4 + j3*bas4 + j4
        j = j1 * self.bas2 * self.bas3 * self.bas4 + \
            j2 * self.bas3 * self.bas4 + \
            j3 * self.bas4 + \
            j4   # [batch_size]
        
        # Get Acoeff vectors: A is [m, L], we want columns indexed by j -> [batch_size, m]
        A_j = self.Acoeff[:, j].permute(1, 0)  # [batch_size, m] after permute
        
        # Compute Nk = scalar * A_j [batch_size, m]
        Nk = scalar.unsqueeze(1) * A_j        # [batch_size, m]
            
        return Nk

    def describe(self, arr_4d):
        """Compute N(k) vectors for each window in the 4D array"""
        tokens, k1_list, k2_list, k3_list, k4_list = self.extract_tokens_and_positions(arr_4d)
        if not tokens:
            return []
        
        token_indices = self.token_to_indices(tokens)
        k1_tensor = torch.tensor(k1_list, dtype=torch.float32, device=self.device)
        k2_tensor = torch.tensor(k2_list, dtype=torch.float32, device=self.device)
        k3_tensor = torch.tensor(k3_list, dtype=torch.float32, device=self.device)
        k4_tensor = torch.tensor(k4_list, dtype=torch.float32, device=self.device)
        
        # Batch compute all Nk vectors
        N_batch = self.batch_compute_Nk(k1_tensor, k2_tensor, k3_tensor, k4_tensor, token_indices)
        return N_batch.detach().cpu().numpy()

    def S(self, arr_4d):
        """
        Compute list of S(l) = sum_{windows up to position order} N(k) for a 4D array.
        Note: The order is row-major (by k1, then k2, then k3, then k4) based on extraction order.
        """
        tokens, k1_list, k2_list, k3_list, k4_list = self.extract_tokens_and_positions(arr_4d)
        if not tokens:
            return []
            
        token_indices = self.token_to_indices(tokens)
        k1_tensor = torch.tensor(k1_list, dtype=torch.float32, device=self.device)
        k2_tensor = torch.tensor(k2_list, dtype=torch.float32, device=self.device)
        k3_tensor = torch.tensor(k3_list, dtype=torch.float32, device=self.device)
        k4_tensor = torch.tensor(k4_list, dtype=torch.float32, device=self.device)
        
        # Batch compute all Nk vectors
        N_batch = self.batch_compute_Nk(k1_tensor, k2_tensor, k3_tensor, k4_tensor, token_indices)
        
        # Compute cumulative sum (S vectors)
        S_cum = torch.cumsum(N_batch, dim=0)
        return [s.detach().cpu().numpy() for s in S_cum]

    def D(self, arrs_4d, t_list):
        """
        Compute mean squared deviation D across arrays:
        D = average over all positions of (N(k1,k2,k3,k4)-t_seq)^2
        """
        total_loss = 0.0
        total_positions = 0
        
        # Convert target vectors to tensor
        t_tensors = [torch.tensor(t, dtype=torch.float32, device=self.device) for t in t_list]
        
        for arr_4d, t in zip(arrs_4d, t_tensors):
            tokens, k1_list, k2_list, k3_list, k4_list = self.extract_tokens_and_positions(arr_4d)
            if not tokens:
                continue
                
            token_indices = self.token_to_indices(tokens)
            k1_tensor = torch.tensor(k1_list, dtype=torch.float32, device=self.device)
            k2_tensor = torch.tensor(k2_list, dtype=torch.float32, device=self.device)
            k3_tensor = torch.tensor(k3_list, dtype=torch.float32, device=self.device)
            k4_tensor = torch.tensor(k4_list, dtype=torch.float32, device=self.device)
            
            # Batch compute all Nk vectors
            N_batch = self.batch_compute_Nk(k1_tensor, k2_tensor, k3_tensor, k4_tensor, token_indices)
            
            # Compute loss for each position
            losses = torch.sum((N_batch - t) ** 2, dim=1)
            total_loss += losses.sum().item()
            total_positions += len(tokens)
                
        return total_loss / total_positions if total_positions else 0.0

    def d(self, arr_4d, t):
        """
        Compute pattern deviation value (d) for a single 4D array.
        """
        d_value = self.D([arr_4d], [t])
        return d_value     

    def reg_train(self, arrs_4d, t_list, max_iters=1000, tol=1e-8, learning_rate=0.01, 
               continued=False, decay_rate=1.0, print_every=10, batch_size=32,
               checkpoint_file=None, checkpoint_interval=10):
        """
        Train model using gradient descent with batch processing for 4D arrays.
        
        Args:
            arrs_4d: List of 4D character arrays (list of list of list of list of list of chars)
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
            indices = list(range(len(arrs_4d)))
            random.shuffle(indices)
            
            # Process arrays in batches
            for batch_start in range(0, len(indices), batch_size):
                batch_indices = indices[batch_start:batch_start + batch_size]
                batch_arrs = [arrs_4d[idx] for idx in batch_indices]
                batch_targets = [t_tensors[idx] for idx in batch_indices]
                
                optimizer.zero_grad()
                batch_loss = 0.0
                batch_array_count = 0
                
                # Process each array in the current batch
                for arr_4d, target in zip(batch_arrs, batch_targets):
                    # Extract tokens and positions for current array
                    tokens, k1_list, k2_list, k3_list, k4_list = self.extract_tokens_and_positions(arr_4d)
                    if not tokens:
                        continue  # Skip empty arrays
                        
                    # Convert tokens to indices
                    token_indices = self.token_to_indices(tokens)
                    k1_tensor = torch.tensor(k1_list, dtype=torch.float32, device=self.device)
                    k2_tensor = torch.tensor(k2_list, dtype=torch.float32, device=self.device)
                    k3_tensor = torch.tensor(k3_list, dtype=torch.float32, device=self.device)
                    k4_tensor = torch.tensor(k4_list, dtype=torch.float32, device=self.device)
                    
                    # Batch compute all Nk vectors for current array
                    Nk_batch = self.batch_compute_Nk(k1_tensor, k2_tensor, k3_tensor, k4_tensor, token_indices)
                    
                    # Compute array-level prediction: average of all N(k) vectors
                    array_pred = torch.mean(Nk_batch, dim=0)
                    
                    # Calculate loss for this array (MSE between prediction and target)
                    array_loss = torch.sum((array_pred - target) ** 2)
                    batch_loss += array_loss
                    batch_array_count += 1
                    
                    # Clean up intermediate tensors
                    del Nk_batch, array_pred, token_indices, k1_tensor, k2_tensor, k3_tensor, k4_tensor
                    
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
        self._compute_training_statistics(arrs_4d)
        self.trained = True
        
        return history

    def cls_train(self, arrs_4d, labels, num_classes, max_iters=1000, tol=1e-8, learning_rate=0.01,
                  continued=False, decay_rate=1.0, print_every=10, batch_size=32,
                  checkpoint_file=None, checkpoint_interval=10):
        """
        Train the model for multi-class classification using cross-entropy loss.
        
        Args:
            arrs_4d: List of 4D character arrays
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
            indices = list(range(len(arrs_4d)))
            random.shuffle(indices)
            
            # Process arrays in batches
            for batch_start in range(0, len(indices), batch_size):
                batch_indices = indices[batch_start:batch_start + batch_size]
                batch_arrs = [arrs_4d[idx] for idx in batch_indices]
                batch_labels = label_tensors[batch_indices]
                
                optimizer.zero_grad()
                batch_loss = 0.0
                batch_logits = []
                
                # Process each array in the batch
                for arr_4d in batch_arrs:
                    # Extract tokens and positions
                    tokens, k1_list, k2_list, k3_list, k4_list = self.extract_tokens_and_positions(arr_4d)
                    if not tokens:
                        # For empty arrays, use zero vector
                        array_vector = torch.zeros(self.m, device=self.device)
                    else:
                        token_indices = self.token_to_indices(tokens)
                        k1_tensor = torch.tensor(k1_list, dtype=torch.float32, device=self.device)
                        k2_tensor = torch.tensor(k2_list, dtype=torch.float32, device=self.device)
                        k3_tensor = torch.tensor(k3_list, dtype=torch.float32, device=self.device)
                        k4_tensor = torch.tensor(k4_list, dtype=torch.float32, device=self.device)
                        
                        # Compute Nk vectors for all windows
                        Nk_batch = self.batch_compute_Nk(k1_tensor, k2_tensor, k3_tensor, k4_tensor, token_indices)
                        
                        # Compute array-level vector: average of all N(k) vectors
                        array_vector = torch.mean(Nk_batch, dim=0)
                        
                        # Clean up
                        del Nk_batch, token_indices, k1_tensor, k2_tensor, k3_tensor, k4_tensor
                    
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

    def lbl_train(self, arrs_4d, labels, num_labels, max_iters=1000, tol=1e-8, learning_rate=0.01, 
                 continued=False, decay_rate=1.0, print_every=10, batch_size=32,
                 checkpoint_file=None, checkpoint_interval=10, pos_weight=None):
        """
        Train the model for multi-label classification using binary cross-entropy loss.
        
        Args:
            arrs_4d: List of 4D character arrays
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
            indices = list(range(len(arrs_4d)))
            random.shuffle(indices)
            
            # Process arrays in batches
            for batch_start in range(0, len(indices), batch_size):
                batch_indices = indices[batch_start:batch_start + batch_size]
                batch_arrs = [arrs_4d[idx] for idx in batch_indices]
                batch_labels = labels_tensor[batch_indices]
                
                optimizer.zero_grad()
                batch_loss = 0.0
                batch_correct = 0
                batch_predictions = 0
                
                # Process each array in the batch
                batch_predictions_list = []
                for arr_4d in batch_arrs:
                    # Extract tokens and positions
                    tokens, k1_list, k2_list, k3_list, k4_list = self.extract_tokens_and_positions(arr_4d)
                    if not tokens:
                        continue
                        
                    token_indices = self.token_to_indices(tokens)
                    k1_tensor = torch.tensor(k1_list, dtype=torch.float32, device=self.device)
                    k2_tensor = torch.tensor(k2_list, dtype=torch.float32, device=self.device)
                    k3_tensor = torch.tensor(k3_list, dtype=torch.float32, device=self.device)
                    k4_tensor = torch.tensor(k4_list, dtype=torch.float32, device=self.device)
                    
                    # Compute Nk vectors
                    Nk_batch = self.batch_compute_Nk(k1_tensor, k2_tensor, k3_tensor, k4_tensor, token_indices)
                    
                    # Compute array representation: average of all N(k) vectors
                    array_representation = torch.mean(Nk_batch, dim=0)
                    
                    # Pass through label head to get logits
                    logits = self.labeller(array_representation)
                    batch_predictions_list.append(logits)
                    
                    # Clean up
                    del Nk_batch, array_representation, token_indices, k1_tensor, k2_tensor, k3_tensor, k4_tensor
                
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

    def self_train(self, arrs_4d, max_iters=1000, tol=1e-8, 
               learning_rate=0.01, continued=False, decay_rate=1.0, 
               print_every=10, batch_size=1024, checkpoint_file=None, 
               checkpoint_interval=10):
        """
        Train the model using self-supervised learning with memory-optimized batch processing.
        Self-training uses the gap-filling objective: predict token embeddings from position information.
        
        Args:
            arrs_4d: List of 4D character arrays
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
            indices = list(range(len(arrs_4d)))
            random.shuffle(indices)
            
            # Process arrays in shuffled order
            for arr_idx in indices:
                arr_4d = arrs_4d[arr_idx]
                tokens, k1_list, k2_list, k3_list, k4_list = self.extract_tokens_and_positions(arr_4d)
                if not tokens:
                    continue
                    
                token_indices = self.token_to_indices(tokens)
                array_samples = []
                
                # Generate samples for current array: each window is a sample (k1, k2, k3, k4, token_idx)
                for k1, k2, k3, k4, token_idx in zip(k1_list, k2_list, k3_list, k4_list, token_indices):
                    array_samples.append((k1, k2, k3, k4, token_idx.item()))
                
                if not array_samples:
                    continue
                
                # Process samples from current array in batches
                for batch_start in range(0, len(array_samples), batch_size):
                    batch_samples = array_samples[batch_start:batch_start + batch_size]
                    
                    optimizer.zero_grad()
                    
                    # Prepare batch data
                    k1_list_batch = []
                    k2_list_batch = []
                    k3_list_batch = []
                    k4_list_batch = []
                    token_idx_list = []
                    
                    for sample in batch_samples:
                        k1, k2, k3, k4, token_idx = sample
                        k1_list_batch.append(k1)
                        k2_list_batch.append(k2)
                        k3_list_batch.append(k3)
                        k4_list_batch.append(k4)
                        token_idx_list.append(token_idx)
                    
                    # Create tensors
                    k1_tensor = torch.tensor(k1_list_batch, dtype=torch.float32, device=self.device)
                    k2_tensor = torch.tensor(k2_list_batch, dtype=torch.float32, device=self.device)
                    k3_tensor = torch.tensor(k3_list_batch, dtype=torch.float32, device=self.device)
                    k4_tensor = torch.tensor(k4_list_batch, dtype=torch.float32, device=self.device)
                    token_indices_tensor = torch.tensor(token_idx_list, device=self.device)
                    
                    # Batch compute Nk for current tokens
                    Nk_batch = self.batch_compute_Nk(k1_tensor, k2_tensor, k3_tensor, k4_tensor, token_indices_tensor)
                    
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
                    del k1_tensor, k2_tensor, k3_tensor, k4_tensor, token_indices_tensor, Nk_batch, targets, loss
                    
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
        self._compute_training_statistics(arrs_4d)
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
                'bas1': self.bas1,
                'bas2': self.bas2,
                'bas3': self.bas3,
                'bas4': self.bas4,
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

    def _compute_training_statistics(self, arrs_4d, batch_size=5):
        """Compute and store statistics for reconstruction and generation with memory optimization"""
        total_window_count = 0
        total_t = torch.zeros(self.m, device=self.device)
        
        with torch.no_grad():
            for i in range(0, len(arrs_4d), batch_size):
                batch_arrs = arrs_4d[i:i+batch_size]
                batch_window_count = 0
                batch_t_sum = torch.zeros(self.m, device=self.device)
                
                for arr_4d in batch_arrs:
                    try:
                        tokens, k1_list, k2_list, k3_list, k4_list = self.extract_tokens_and_positions(arr_4d)
                        batch_window_count += len(tokens)
                        
                        if tokens:
                            token_indices = self.token_to_indices(tokens)
                            k1_tensor = torch.tensor(k1_list, dtype=torch.float32, device=self.device)
                            k2_tensor = torch.tensor(k2_list, dtype=torch.float32, device=self.device)
                            k3_tensor = torch.tensor(k3_list, dtype=torch.float32, device=self.device)
                            k4_tensor = torch.tensor(k4_list, dtype=torch.float32, device=self.device)
                            Nk_batch = self.batch_compute_Nk(k1_tensor, k2_tensor, k3_tensor, k4_tensor, token_indices)
                            batch_t_sum += Nk_batch.sum(dim=0)
                            
                            # Clean up
                            del token_indices, k1_tensor, k2_tensor, k3_tensor, k4_tensor, Nk_batch
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
        
        self.mean_window_count = total_window_count / len(arrs_4d) if arrs_4d else 0
        self.mean_t = (total_t / total_window_count).cpu().numpy() if total_window_count > 0 else np.zeros(self.m)
    
    def predict_t(self, arr_4d):
        """Predict target vector as average of N(k) vectors over all windows"""
        tokens, k1_list, k2_list, k3_list, k4_list = self.extract_tokens_and_positions(arr_4d)
        if not tokens:
            return np.zeros(self.m)
        
        # Compute all Nk vectors
        Nk = self.describe(arr_4d)
        return np.mean(Nk, axis=0)
    
    def predict_c(self, arr_4d):
        """
        Predict class label for a 4D array using the classification head.
        
        Args:
            arr_4d: Input 4D character array
            
        Returns:
            tuple: (predicted_class, class_probabilities)
        """
        if self.classifier is None:
            raise ValueError("Model must be trained first for classification")
        
        # Get array vector representation
        array_vector = self.predict_t(arr_4d)
        array_vector_tensor = torch.tensor(array_vector, dtype=torch.float32, device=self.device)
        
        # Get logits through classification head
        with torch.no_grad():
            logits = self.classifier(array_vector_tensor.unsqueeze(0))
            probabilities = torch.softmax(logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            
        return predicted_class, probabilities[0].cpu().numpy()

    def predict_l(self, arr_4d, threshold=0.5):
        """
        Predict multi-label classification for a 4D array.
        
        Args:
            arr_4d: Input 4D character array
            threshold: Probability threshold for binary classification (default: 0.5)
            
        Returns:
            numpy.ndarray: Binary label predictions (0 or 1 for each label)
            numpy.ndarray: Probability scores for each label
        """
        assert self.labeller is not None, "Model must be trained first for label prediction"
        
        tokens, k1_list, k2_list, k3_list, k4_list = self.extract_tokens_and_positions(arr_4d)
        if not tokens:
            # Return zeros if no tokens
            return np.zeros(self.num_labels, dtype=np.float32), np.zeros(self.num_labels, dtype=np.float32)
        
        token_indices = self.token_to_indices(tokens)
        k1_tensor = torch.tensor(k1_list, dtype=torch.float32, device=self.device)
        k2_tensor = torch.tensor(k2_list, dtype=torch.float32, device=self.device)
        k3_tensor = torch.tensor(k3_list, dtype=torch.float32, device=self.device)
        k4_tensor = torch.tensor(k4_list, dtype=torch.float32, device=self.device)
        
        # Compute N(k) vectors for all windows
        Nk_batch = self.batch_compute_Nk(k1_tensor, k2_tensor, k3_tensor, k4_tensor, token_indices)
        
        # Compute array representation: average of all N(k) vectors
        array_representation = torch.mean(Nk_batch, dim=0)
        
        # Pass through label head to get logits
        with torch.no_grad():
            logits = self.labeller(array_representation)
            probs = torch.sigmoid(logits).cpu().numpy()
        
        # Apply threshold to get binary predictions
        binary_preds = (probs > threshold).astype(np.float32)
        
        return binary_preds, probs

    def reconstruct(self, D1, D2, D3, D4, tau=0.0):
        """
        Reconstruct a 4D array of size D1×D2×D3×D4 by generating hypercubic windows in row-major order.
        Assumes step size = rank (non-overlapping) for simplicity.
        """
        assert self.trained, "Model must be trained first"
        if tau < 0:
            raise ValueError("Temperature must be non-negative")
        
        # Determine number of windows in each dimension
        rank = self.rank
        D1_adj = ((D1 + rank - 1) // rank) * rank
        D2_adj = ((D2 + rank - 1) // rank) * rank
        D3_adj = ((D3 + rank - 1) // rank) * rank
        D4_adj = ((D4 + rank - 1) // rank) * rank
        
        num_windows_d1 = D1_adj // rank
        num_windows_d2 = D2_adj // rank
        num_windows_d3 = D3_adj // rank
        num_windows_d4 = D4_adj // rank
        
        mean_t_tensor = torch.tensor(self.mean_t, dtype=torch.float32, device=self.device)
        
        # Precompute all token indices and embeddings
        all_token_indices = torch.arange(len(self.tokens), device=self.device)
        
        # Initialize a 4D grid for the reconstructed characters with '_'
        # We'll fill it window by window with step = rank.
        # Use nested list comprehensions with caution for memory; for demonstration small sizes are used.
        result = [[[[ '_' for _ in range(D4_adj)] for __ in range(D3_adj)] for ___ in range(D2_adj)] for ____ in range(D1_adj)]
        
        # For each window position (i1,i2,i3,i4) in row-major order
        for i1 in range(0, num_windows_d1):
            for i2 in range(0, num_windows_d2):
                for i3 in range(0, num_windows_d3):
                    for i4 in range(0, num_windows_d4):
                        k1 = i1 * rank
                        k2 = i2 * rank
                        k3 = i3 * rank
                        k4 = i4 * rank
                        # Compute Nk for all tokens at this position
                        k1_tensor = torch.tensor([k1] * len(self.tokens), dtype=torch.float32, device=self.device)
                        k2_tensor = torch.tensor([k2] * len(self.tokens), dtype=torch.float32, device=self.device)
                        k3_tensor = torch.tensor([k3] * len(self.tokens), dtype=torch.float32, device=self.device)
                        k4_tensor = torch.tensor([k4] * len(self.tokens), dtype=torch.float32, device=self.device)
                        Nk_all = self.batch_compute_Nk(k1_tensor, k2_tensor, k3_tensor, k4_tensor, all_token_indices)
                        
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
                        # The token is a string of length rank^4, order: dim1 fastest, then dim2, dim3, dim4 slowest.
                        for d1 in range(rank):
                            for d2 in range(rank):
                                for d3 in range(rank):
                                    for d4 in range(rank):
                                        result[k1+d1][k2+d2][k3+d3][k4+d4] = best_tok[d1*rank*rank*rank + d2*rank*rank + d3*rank + d4]
        
        # Trim to original sizes
        final_result = [result[i1][:D2] for i1 in range(D1)]
        final_result = [[[row[:D3] for row in slice_] for slice_ in layer] for layer in final_result]
        final_result = [[[[cell[:D4] for cell in row] for row in slice_] for slice_ in layer] for layer in final_result]
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
    print("Spatial Dual Descriptor RN4 - PyTorch GPU Accelerated Version (4D)")
    print("Optimized with batch processing")
    print("="*50)
    
    # Set random seeds to ensure reproducibility
    torch.manual_seed(11)
    random.seed(11)
    
    charset = ['A','C','G','T']
    vec_dim = 3
    bas1 = 4
    bas2 = 4
    bas3 = 4
    bas4 = 4
    array_num = 10   # small number for 4D due to memory

    # Generate 4D arrays of random size (each dimension 5-8) and random targets
    arrays_4d = []
    t_list = []
    for _ in range(array_num):
        D1 = random.randint(5, 8)
        D2 = random.randint(5, 8)
        D3 = random.randint(5, 8)
        D4 = random.randint(5, 8)
        # Create a 4D list: arr[d1][d2][d3][d4]
        arr = [[[[random.choice(charset) for _ in range(D4)] for __ in range(D3)] for ___ in range(D2)] for ____ in range(D1)]
        arrays_4d.append(arr)
        t_list.append([random.uniform(-1,1) for _ in range(vec_dim)])

    # === Gradient Descent Training Example ===
    print("\n" + "="*50)
    print("Testing Gradient Descent Training on 4D Arrays")
    print("="*50)

    # Create new model instance with GPU acceleration
    dd = SpatialDualDescriptorRN4(
        charset, 
        rank=1,            # smaller rank for 4D to reduce token count
        vec_dim=vec_dim, 
        bas1=bas1,
        bas2=bas2,
        bas3=bas3,
        bas4=bas4,
        mode='linear', 
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Display device information
    print(f"\nUsing device: {dd.device}")
    print(f"Number of tokens: {len(dd.tokens)}")

    # Train using gradient descent
    print("\nStarting gradient descent training...")
    grad_history = dd.reg_train(
        arrays_4d, 
        t_list,
        learning_rate=0.05,
        max_iters=50,      # fewer iterations for demonstration
        tol=1e-8,
        decay_rate=0.99,
        print_every=10,
        batch_size=2
    )

    # Predict target vector for first array
    arr = arrays_4d[0]
    t_pred = dd.predict_t(arr)
    print(f"\nPredicted t for first array: {[round(x.item(), 4) for x in t_pred]}")    
    
    # Calculate flattened correlation between predicted and actual targets
    pred_t_list = [dd.predict_t(arr) for arr in arrays_4d]
    
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

    # Reconstruct representative 4D array (small size for demonstration)
    D1_rec = 4
    D2_rec = 4
    D3_rec = 4
    D4_rec = 4
    rec_det = dd.reconstruct(D1_rec, D2_rec, D3_rec, D4_rec, tau=0.0)
    rec_sto = dd.reconstruct(D1_rec, D2_rec, D3_rec, D4_rec, tau=0.5)
    print("\nDeterministic Reconstruction (first slice along dim1):")
    # Print first 2x2 block of first two dimensions? Actually we can show a small subcube
    for d2 in range(min(2, D2_rec)):
        for d3 in range(min(2, D3_rec)):
            row = ''.join(rec_det[0][d2][d3][:min(4, D4_rec)])
            print(f"d2={d2}, d3={d3}: {row}")
    print("Stochastic Reconstruction (tau=0.5, first slice):")
    for d2 in range(min(2, D2_rec)):
        for d3 in range(min(2, D3_rec)):
            row = ''.join(rec_sto[0][d2][d3][:min(4, D4_rec)])
            print(f"d2={d2}, d3={d3}: {row}")
    
    # === Classification Task ===
    print("\n" + "="*50)
    print("Classification Task on 4D Arrays")
    print("="*50)
    
    # Generate classification data
    num_classes = 2  # binary classification for simplicity
    class_arrays = []
    class_labels = []
    
    # Create arrays with different patterns for each class
    for class_id in range(num_classes):
        for _ in range(15):  # 15 arrays per class
            D1 = random.randint(5, 7)
            D2 = random.randint(5, 7)
            D3 = random.randint(5, 7)
            D4 = random.randint(5, 7)
            if class_id == 0:
                # Class 0: High A content
                arr = [[[[random.choices(['A','C','G','T'], weights=[0.6,0.1,0.1,0.2])[0] for _ in range(D4)] for __ in range(D3)] for ___ in range(D2)] for ____ in range(D1)]
            else:
                # Class 1: Balanced
                arr = [[[[random.choice(charset) for _ in range(D4)] for __ in range(D3)] for ___ in range(D2)] for ____ in range(D1)]
            class_arrays.append(arr)
            class_labels.append(class_id)
    
    # Initialize new model for classification
    dd_cls = SpatialDualDescriptorRN4(
        charset, 
        vec_dim=vec_dim, 
        bas1=bas1,
        bas2=bas2,
        bas3=bas3,
        bas4=bas4,
        rank=1, 
        mode='linear',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Train for classification
    print("\n" + "="*50)
    print("Starting Classification Training")
    print("="*50)
    history = dd_cls.cls_train(class_arrays, class_labels, num_classes, 
                              max_iters=40, tol=1e-8, learning_rate=0.05,
                              decay_rate=0.99, batch_size=2, print_every=5)
    
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
    print("Multi-Label Classification Model on 4D Arrays")
    print("="*50)
    
    # Generate arrays with random multi-labels
    num_labels = 2
    label_arrays = []
    labels = []
    for _ in range(20):
        D1 = random.randint(5, 7)
        D2 = random.randint(5, 7)
        D3 = random.randint(5, 7)
        D4 = random.randint(5, 7)
        arr = [[[[random.choice(charset) for _ in range(D4)] for __ in range(D3)] for ___ in range(D2)] for ____ in range(D1)]
        label_arrays.append(arr)
        # Random binary labels
        label_vec = [random.random() > 0.7 for _ in range(num_labels)]
        labels.append([1.0 if x else 0.0 for x in label_vec])
    
    dd_lbl = SpatialDualDescriptorRN4(
        charset, 
        vec_dim=vec_dim, 
        bas1=bas1,
        bas2=bas2,
        bas3=bas3,
        bas4=bas4,
        rank=1, 
        mode='linear',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Training multi-label classification model
    print("\n" + "="*50)
    print("Starting Multi-Label Classification Training")
    print("="*50)
       
    loss_history, acc_history = dd_lbl.lbl_train(
        label_arrays, labels, num_labels,
        max_iters=40, 
        tol=1e-8, 
        learning_rate=0.05, 
        decay_rate=0.99, 
        print_every=10, 
        batch_size=2
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
    
    test_arr = [[[[random.choice(charset) for _ in range(6)] for __ in range(6)] for ___ in range(6)] for ____ in range(6)]
    binary_pred, probs_pred = dd_lbl.predict_l(test_arr, threshold=0.5)
    print(f"Test array shape: {len(test_arr)}x{len(test_arr[0])}x{len(test_arr[0][0])}x{len(test_arr[0][0][0])}")
    print(f"Predicted binary labels: {binary_pred}")
    print(f"Predicted probabilities: {[f'{p:.4f}' for p in probs_pred]}")
    
    # === Self-Training Example ===
    random.seed(1)
    print("\n=== Self-Training on 4D Arrays ===")
    
    # Generate training arrays
    train_arrays = []
    for i in range(10):
        D1 = random.randint(5, 7)
        D2 = random.randint(5, 7)
        D3 = random.randint(5, 7)
        D4 = random.randint(5, 7)
        arr = [[[[random.choice(charset) for _ in range(D4)] for __ in range(D3)] for ___ in range(D2)] for ____ in range(D1)]
        train_arrays.append(arr)
    
    dd_self = SpatialDualDescriptorRN4(
        charset, 
        rank=1,
        vec_dim=vec_dim,
        bas1=bas1,
        bas2=bas2,
        bas3=bas3,
        bas4=bas4,
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
    
    # Reconstruct a 4D array
    print("\n=== Reconstructing 4D Array ===")
    rec_det = dd_self.reconstruct(D1=4, D2=4, D3=4, D4=4, tau=0.0)
    print("Deterministic Reconstruction (first slice along dim1):")
    for d2 in range(min(2, 4)):
        for d3 in range(min(2, 4)):
            row = ''.join(rec_det[0][d2][d3][:min(4, 4)])
            print(f"d2={d2}, d3={d3}: {row}")
    
    # Save and load model
    print("\n=== Model Persistence Test ===")
    dd_self.save("spatial_self_trained_model_4d.pt")
    dd_loaded = SpatialDualDescriptorRN4(
        charset, 
        rank=1,
        vec_dim=vec_dim,
        bas1=bas1,
        bas2=bas2,
        bas3=bas3,
        bas4=bas4,
        mode='linear',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    dd_loaded.load("spatial_self_trained_model_4d.pt")
    print("Model loaded successfully. Reconstructing with loaded model:")
    rec_loaded = dd_loaded.reconstruct(D1=3, D2=3, D3=3, D4=3, tau=0.0)
    for d2 in range(min(2, 3)):
        for d3 in range(min(2, 3)):
            row = ''.join(rec_loaded[0][d2][d3][:min(3, 3)])
            print(f"d2={d2}, d3={d3}: {row}")
    
    print("\n=== All Demos Completed Successfully ===")
