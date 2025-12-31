# Copyright (C) 2005-2025, Bin-Guang Ma (mbg@mail.hzau.edu.cn); SPDX-License-Identifier: MIT
# The Numeric Dual Descriptor Vector class (AB matrix form) implemented with PyTorch
# This program is for the demonstration of methodology and not fully refined.
# Author: Bin-Guang Ma (assisted by DeepSeek); Date: 2025-8-28 ~ 2025-12-30

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

class NumDualDescriptorAB(nn.Module):
    """
    Numeric Dual Descriptor with GPU acceleration using PyTorch:
      - learnable coefficient matrix Acoeff ∈ R^{m×L}
      - fixed basis matrix Bbasis ∈ R^{L×m}, Bbasis[k][i] = cos(2π*(k+1)/(i+2))
      - learnable mapping matrix M ∈ R^{m×m} for input vector transformation
      - Supports both linear and nonlinear tokenization of vector sequences
      - Batch processing for GPU acceleration
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
        
        L = vec_seq.shape[0]  # Sequence length

        # Helper function to apply vector operations
        def apply_op(vectors):
            """Apply mapping and aggregation to a window of vectors"""
            # Apply mapping matrix M to each vector in the window
            transformed = torch.matmul(vectors, self.M.T)  # [rank, m] × [m, m]^T = [rank, m]
            
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
        
        # Linear mode: sliding window with step=1
        if self.mode == 'linear':
            return [apply_op(vec_seq[i:i+self.rank]) for i in range(L - self.rank + 1)]            
        
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
        return vectors

    def describe(self, vec_seq):
        """Compute N(k) vectors for each window in the vector sequence"""
        # Use extract_vectors to get transformed and aggregated vectors
        transformed_vecs = self.extract_vectors(vec_seq)
        if not transformed_vecs:
            return []
        
        # Stack vectors for batch processing
        transformed_vecs_tensor = torch.stack(transformed_vecs, dim=0)  # [num_windows, m]
        num_windows = transformed_vecs_tensor.shape[0]
        
        # Compute basis indices (k mod L)
        k_positions = torch.arange(num_windows, dtype=torch.float32, device=self.device)
        j_indices = (k_positions % self.L).long()
        
        # Get corresponding B basis rows
        B_rows = self.Bbasis[j_indices]  # [num_windows, m]
        
        # Compute scalar = B[j] • transformed_vec for each window
        scalar = torch.sum(B_rows * transformed_vecs_tensor, dim=1)  # [num_windows]
        
        # Get A columns for each position
        A_cols = self.Acoeff[:, j_indices].t()  # [num_windows, m]
        
        # Compute Nk = scalar * A_cols
        Nk = A_cols * scalar.unsqueeze(1)  # [num_windows, m]
        
        return Nk.detach().cpu().numpy()
    
    def S(self, vec_seq):
        """Compute cumulative sum of N(k) vectors"""
        # Use extract_vectors to get transformed and aggregated vectors
        transformed_vecs = self.extract_vectors(vec_seq)
        if not transformed_vecs:
            return []
        
        # Stack vectors for batch processing
        transformed_vecs_tensor = torch.stack(transformed_vecs, dim=0)  # [num_windows, m]
        num_windows = transformed_vecs_tensor.shape[0]
        
        # Compute basis indices (k mod L)
        k_positions = torch.arange(num_windows, dtype=torch.float32, device=self.device)
        j_indices = (k_positions % self.L).long()
        
        # Get corresponding B basis rows
        B_rows = self.Bbasis[j_indices]  # [num_windows, m]
        
        # Compute scalar = B[j] • transformed_vec for each window
        scalar = torch.sum(B_rows * transformed_vecs_tensor, dim=1)  # [num_windows]
        
        # Get A columns for each position
        A_cols = self.Acoeff[:, j_indices].t()  # [num_windows, m]
        
        # Compute Nk = scalar * A_cols
        Nk = A_cols * scalar.unsqueeze(1)  # [num_windows, m]
        
        # Compute cumulative sum
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
        
        # Convert target vectors to tensor and move to device
        t_tensors = [torch.tensor(t, dtype=torch.float32, device=self.device) for t in t_list]
        
        for vec_seq, t in zip(vec_seqs, t_tensors):
            # Use extract_vectors to get transformed and aggregated vectors
            transformed_vecs = self.extract_vectors(vec_seq)
            if not transformed_vecs:
                continue
                
            num_windows = len(transformed_vecs)
            transformed_vecs_tensor = torch.stack(transformed_vecs, dim=0)  # [num_windows, m]
            
            # Compute basis indices (k mod L)
            k_positions = torch.arange(num_windows, dtype=torch.float32, device=self.device)
            j_indices = (k_positions % self.L).long()
            
            # Get corresponding B basis rows
            B_rows = self.Bbasis[j_indices]  # [num_windows, m]
            
            # Compute scalar = B[j] • transformed_vec for each window
            scalar = torch.sum(B_rows * transformed_vecs_tensor, dim=1)  # [num_windows]
            
            # Get A columns for each position
            A_cols = self.Acoeff[:, j_indices].t()  # [num_windows, m]
            
            # Compute Nk vectors for all windows in the sequence
            Nk_batch = A_cols * scalar.unsqueeze(1)  # [num_windows, m]
            
            # Compute loss for each window
            losses = torch.sum((Nk_batch - t) ** 2, dim=1)
            total_loss += losses.sum().item()
            total_windows += num_windows
            
            # Clean up intermediate tensors
            del transformed_vecs_tensor, B_rows, A_cols, Nk_batch, losses
        
        return total_loss / total_windows if total_windows else 0.0

    def d(self, vec_seq, t):
        """
        Compute pattern deviation value (d) for a single vector sequence.
        """
        d_value = self.D([vec_seq], [t])
        return d_value

    def reg_train(self, vec_seqs, t_list, max_iters=1000, tol=1e-8, learning_rate=0.01, 
                  continued=False, decay_rate=1.0, print_every=10, batch_size=32,
                  checkpoint_file=None, checkpoint_interval=10):
        """
        Train model using gradient descent with sequence-level batch processing.
        Memory-optimized version that processes sequences in batches to avoid 
        large precomputation and storage of all token positions.
        
        Args:
            vec_seqs: List of vector sequences (each is a tensor of shape [seq_len, m])
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
        
        # Convert target vectors to tensor and move to device
        t_tensors = [torch.tensor(t, dtype=torch.float32, device=self.device) for t in t_list]
        
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
            total_sequences = 0
            
            # Shuffle sequences for each epoch to ensure diverse batches
            indices = list(range(len(vec_seqs)))
            random.shuffle(indices)
            
            # Process sequences in batches
            for batch_start in range(0, len(indices), batch_size):
                batch_indices = indices[batch_start:batch_start + batch_size]
                batch_seqs = [vec_seqs[idx] for idx in batch_indices]
                batch_targets = [t_tensors[idx] for idx in batch_indices]
                
                optimizer.zero_grad()
                batch_loss = 0.0
                batch_sequence_count = 0
                
                # Process each sequence in the current batch
                for vec_seq, target in zip(batch_seqs, batch_targets):
                    # Use extract_vectors to get transformed and aggregated vectors
                    transformed_vecs = self.extract_vectors(vec_seq)
                    if not transformed_vecs:
                        continue  # Skip empty sequences
                        
                    num_windows = len(transformed_vecs)
                    transformed_vecs_tensor = torch.stack(transformed_vecs, dim=0)  # [num_windows, m]
                    
                    # Compute basis indices (k mod L)
                    k_positions = torch.arange(num_windows, dtype=torch.float32, device=self.device)
                    j_indices = (k_positions % self.L).long()
                    
                    # Get corresponding B basis rows
                    B_rows = self.Bbasis[j_indices]  # [num_windows, m]
                    
                    # Compute scalar = B[j] • transformed_vec for each window
                    scalar = torch.sum(B_rows * transformed_vecs_tensor, dim=1)  # [num_windows]
                    
                    # Get A columns for each position
                    A_cols = self.Acoeff[:, j_indices].t()  # [num_windows, m]
                    
                    # Compute Nk vectors for all windows in the sequence
                    Nk_batch = A_cols * scalar.unsqueeze(1)  # [num_windows, m]
                    
                    # Compute sequence-level prediction: average of all N(k) vectors
                    seq_pred = torch.mean(Nk_batch, dim=0)
                    
                    # Calculate loss for this sequence (MSE between prediction and target)
                    seq_loss = torch.sum((seq_pred - target) ** 2)
                    batch_loss += seq_loss
                    batch_sequence_count += 1
                    
                    # Clean up intermediate tensors to free GPU memory
                    del Nk_batch, seq_pred, transformed_vecs_tensor, B_rows, A_cols, scalar
                    
                    # Periodically clear GPU cache to prevent memory fragmentation
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
        self._compute_training_statistics(vec_seqs)
        self.trained = True
        
        return history

    def cls_train(self, vec_seqs, labels, num_classes, max_iters=1000, tol=1e-8, learning_rate=0.01,
                  continued=False, decay_rate=1.0, print_every=10, batch_size=32,
                  checkpoint_file=None, checkpoint_interval=10):
        """
        Train the model for multi-class classification using cross-entropy loss.
        
        Args:
            vec_seqs: List of vector sequences for training
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
        
        # Convert labels to tensor and move to device
        label_tensors = torch.tensor(labels, dtype=torch.long, device=self.device)
        
        # Ensure all input sequences are on the correct device
        vec_seqs = [seq.to(self.device) if isinstance(seq, torch.Tensor) else 
                   torch.tensor(seq, dtype=torch.float32, device=self.device) 
                   for seq in vec_seqs]
        
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
            total_sequences = 0
            correct_predictions = 0
            
            # Shuffle sequences for each epoch
            indices = list(range(len(vec_seqs)))
            random.shuffle(indices)
            
            # Process sequences in batches
            for batch_start in range(0, len(indices), batch_size):
                batch_indices = indices[batch_start:batch_start + batch_size]
                batch_seqs = [vec_seqs[idx] for idx in batch_indices]
                batch_labels = label_tensors[batch_indices]
                
                optimizer.zero_grad()
                batch_loss = 0.0
                batch_logits = []
                
                # Process each sequence in the batch
                for vec_seq in batch_seqs:
                    # Use extract_vectors to get transformed and aggregated vectors
                    transformed_vecs = self.extract_vectors(vec_seq)
                    if not transformed_vecs:
                        # For empty sequences, use zero vector
                        seq_vector = torch.zeros(self.m, device=self.device)
                    else:
                        # Stack vectors
                        transformed_vecs_tensor = torch.stack(transformed_vecs, dim=0)  # [num_windows, m]
                        num_windows = transformed_vecs_tensor.shape[0]
                        
                        # Compute basis indices (k mod L)
                        k_positions = torch.arange(num_windows, dtype=torch.float32, device=self.device)
                        j_indices = (k_positions % self.L).long()
                        
                        # Get corresponding B basis rows
                        B_rows = self.Bbasis[j_indices]  # [num_windows, m]
                        
                        # Compute scalar = B[j] • transformed_vec for each window
                        scalar = torch.sum(B_rows * transformed_vecs_tensor, dim=1)  # [num_windows]
                        
                        # Get A columns for each position
                        A_cols = self.Acoeff[:, j_indices].t()  # [num_windows, m]
                        
                        # Compute Nk vectors for all windows in the sequence
                        Nk_batch = A_cols * scalar.unsqueeze(1)  # [num_windows, m]
                        
                        # Compute sequence-level vector: average of all N(k) vectors
                        seq_vector = torch.mean(Nk_batch, dim=0)
                        
                        # Clean up intermediate tensors to free memory
                        del Nk_batch, transformed_vecs_tensor, B_rows, A_cols, scalar
                    
                    # Get logits through classification head
                    logits = self.classifier(seq_vector.unsqueeze(0))
                    batch_logits.append(logits)
                
                # Stack all logits and compute loss
                if batch_logits:
                    all_logits = torch.cat(batch_logits, dim=0)
                    loss = criterion(all_logits, batch_labels)
                    loss.backward()
                    optimizer.step()
                    
                    # Calculate batch statistics
                    batch_loss = loss.item()
                    total_loss += batch_loss * len(batch_seqs)
                    total_sequences += len(batch_seqs)
                    
                    # Calculate accuracy
                    with torch.no_grad():
                        predictions = torch.argmax(all_logits, dim=1)
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

    def lbl_train(self, vec_seqs, labels, num_labels, max_iters=1000, tol=1e-8, learning_rate=0.01, 
                 continued=False, decay_rate=1.0, print_every=10, batch_size=32,
                 checkpoint_file=None, checkpoint_interval=10, pos_weight=None):
        """
        Train the model for multi-label classification using binary cross-entropy loss.
        
        Args:
            vec_seqs: List of vector sequences for training
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
        vec_seqs = [seq.to(self.device) if isinstance(seq, torch.Tensor) else 
                   torch.tensor(seq, dtype=torch.float32, device=self.device) 
                   for seq in vec_seqs]
        
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
            total_sequences = 0
            
            # Shuffle sequences for each epoch
            indices = list(range(len(vec_seqs)))
            random.shuffle(indices)
            
            # Process sequences in batches
            for batch_start in range(0, len(indices), batch_size):
                batch_indices = indices[batch_start:batch_start + batch_size]
                batch_seqs = [vec_seqs[idx] for idx in batch_indices]
                batch_labels = labels_tensor[batch_indices]
                
                optimizer.zero_grad()
                batch_loss = 0.0
                batch_correct = 0
                batch_predictions = 0
                
                # Process each sequence in the batch
                batch_predictions_list = []
                for vec_seq in batch_seqs:
                    # Use extract_vectors to get transformed and aggregated vectors
                    transformed_vecs = self.extract_vectors(vec_seq)
                    if not transformed_vecs:
                        # If no windows, skip this sequence
                        continue
                        
                    # Stack vectors
                    transformed_vecs_tensor = torch.stack(transformed_vecs, dim=0)  # [num_windows, m]
                    num_windows = transformed_vecs_tensor.shape[0]
                    
                    # Compute basis indices (k mod L)
                    k_positions = torch.arange(num_windows, dtype=torch.float32, device=self.device)
                    j_indices = (k_positions % self.L).long()
                    
                    # Get corresponding B basis rows
                    B_rows = self.Bbasis[j_indices]  # [num_windows, m]
                    
                    # Compute scalar = B[j] • transformed_vec for each window
                    scalar = torch.sum(B_rows * transformed_vecs_tensor, dim=1)  # [num_windows]
                    
                    # Get A columns for each position
                    A_cols = self.Acoeff[:, j_indices].t()  # [num_windows, m]
                    
                    # Compute Nk vectors for all windows in the sequence
                    Nk_batch = A_cols * scalar.unsqueeze(1)  # [num_windows, m]
                    
                    # Compute sequence representation: average of all N(k) vectors
                    seq_representation = torch.mean(Nk_batch, dim=0)
                    
                    # Pass through classification head to get logits
                    logits = self.labeller(seq_representation)
                    batch_predictions_list.append(logits)
                    
                    # Clean up intermediate tensors to free memory
                    del Nk_batch, seq_representation, transformed_vecs_tensor, B_rows, A_cols, scalar
                
                # Stack predictions for the batch
                if batch_predictions_list:
                    batch_logits = torch.stack(batch_predictions_list, dim=0)
                    
                    # Calculate loss for the batch
                    batch_loss = criterion(batch_logits, batch_labels)
                    
                    # Calculate accuracy
                    with torch.no_grad():
                        # Apply sigmoid to get probabilities
                        probs = torch.sigmoid(batch_logits)
                        # Threshold at 0.5 for binary predictions
                        predictions = (probs > 0.5).float()
                        # Calculate number of correct predictions
                        batch_correct = (predictions == batch_labels).sum().item()
                        batch_predictions = batch_labels.numel()
                    
                    # Backpropagate
                    batch_loss.backward()
                    optimizer.step()
                    
                    total_loss += batch_loss.item() * len(batch_seqs)
                    total_correct += batch_correct
                    total_predictions += batch_predictions
                    total_sequences += len(batch_seqs)
                
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

    def self_train(self, vec_seqs, max_iters=1000, tol=1e-8, learning_rate=0.01, 
                   continued=False, decay_rate=1.0, print_every=10, batch_size=1024, 
                   checkpoint_file=None, checkpoint_interval=10):
        """
        Train the model using self-supervised learning with memory-optimized batch processing.
        
        Args:
            vec_seqs: List of vector sequences
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
            
            # Shuffle sequences for each epoch to ensure diverse samples
            indices = list(range(len(vec_seqs)))
            random.shuffle(indices)
            
            # Process sequences in shuffled order
            for seq_idx in indices:
                vec_seq = vec_seqs[seq_idx]
                # Use extract_vectors to get transformed and aggregated vectors
                transformed_vecs = self.extract_vectors(vec_seq)
                if not transformed_vecs:
                    continue
                    
                seq_samples = []
                
                # Generate samples for current sequence
                # Each aggregated vector is a sample (position k, aggregated vector)
                for k, agg_vec in enumerate(transformed_vecs):
                    seq_samples.append((k, agg_vec))
                
                if not seq_samples:
                    continue
                
                # Process samples from current sequence in batches
                for batch_start in range(0, len(seq_samples), batch_size):
                    batch_samples = seq_samples[batch_start:batch_start + batch_size]
                    
                    optimizer.zero_grad()
                    
                    # Prepare batch data
                    k_list = []
                    agg_vec_list = []
                    
                    for sample in batch_samples:
                        k, agg_vec = sample
                        k_list.append(k)
                        agg_vec_list.append(agg_vec)
                    
                    # Stack aggregated vectors
                    agg_vec_tensors = torch.stack(agg_vec_list, dim=0)  # [batch_size, m]
                    
                    # Create tensors for position indices
                    k_tensor = torch.tensor(k_list, dtype=torch.float32, device=self.device)
                    
                    # Compute Nk for current aggregated vectors using batch processing
                    j_indices = (k_tensor % self.L).long()
                    B_rows = self.Bbasis[j_indices]
                    scalar = torch.sum(B_rows * agg_vec_tensors, dim=1)
                    A_cols = self.Acoeff[:, j_indices].t()
                    Nk_batch = A_cols * scalar.unsqueeze(1)
                    
                    # Get target vectors (the aggregated vectors themselves)
                    targets = agg_vec_tensors
                    
                    # Compute loss
                    loss = torch.mean(torch.sum((Nk_batch - targets) ** 2, dim=1))
                    loss.backward()
                    optimizer.step()
                    
                    batch_loss = loss.item() * len(batch_samples)
                    total_loss += batch_loss
                    total_samples += len(batch_samples)
                    
                    # Clean up to free memory
                    del k_tensor, agg_vec_tensors, Nk_batch, targets, loss
                    
                    # Periodically clear GPU cache to prevent memory fragmentation
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

    def _compute_training_statistics(self, vec_seqs, batch_size=50):
        """
        Compute and store statistics for reconstruction and generation with memory optimization.
        Calculates mean window count and mean target vector across all sequences.
        
        Args:
            vec_seqs: List of vector sequences
            batch_size: Batch size for processing sequences to optimize memory usage
        """
        total_window_count = 0
        total_t = torch.zeros(self.m, device=self.device)
        
        with torch.no_grad():
            for i in range(0, len(vec_seqs), batch_size):
                batch_seqs = vec_seqs[i:i+batch_size]
                batch_window_count = 0
                batch_t_sum = torch.zeros(self.m, device=self.device)
                
                for vec_seq in batch_seqs:
                    # Use extract_vectors to get transformed and aggregated vectors
                    transformed_vecs = self.extract_vectors(vec_seq)
                    batch_window_count += len(transformed_vecs)
                    
                    if transformed_vecs:
                        # Stack vectors
                        transformed_vecs_tensor = torch.stack(transformed_vecs, dim=0)  # [num_windows, m]
                        num_windows = transformed_vecs_tensor.shape[0]
                        
                        # Compute basis indices (k mod L)
                        k_positions = torch.arange(num_windows, dtype=torch.float32, device=self.device)
                        j_indices = (k_positions % self.L).long()
                        
                        # Get corresponding B basis rows
                        B_rows = self.Bbasis[j_indices]  # [num_windows, m]
                        
                        # Compute scalar = B[j] • transformed_vec for each window
                        scalar = torch.sum(B_rows * transformed_vecs_tensor, dim=1)  # [num_windows]
                        
                        # Get A columns for each position
                        A_cols = self.Acoeff[:, j_indices].t()  # [num_windows, m]
                        
                        # Compute Nk vectors for all windows in the sequence
                        Nk_batch = A_cols * scalar.unsqueeze(1)  # [num_windows, m]
                        
                        batch_t_sum += Nk_batch.sum(dim=0)
                        
                        # Clean up intermediate tensors
                        del transformed_vecs_tensor, Nk_batch, B_rows, A_cols, scalar
                
                total_window_count += batch_window_count
                total_t += batch_t_sum
                
                # Clean batch tensors
                del batch_t_sum
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Store statistics for reconstruction and generation
        self.mean_window_count = total_window_count / len(vec_seqs) if vec_seqs else 0
        self.mean_t = (total_t / total_window_count).cpu().numpy() if total_window_count > 0 else np.zeros(self.m)

    def predict_t(self, vec_seq):
        """Predict target vector as average of N(k) vectors"""
        # Use extract_vectors to get transformed and aggregated vectors
        transformed_vecs = self.extract_vectors(vec_seq)
        if not transformed_vecs:
            return np.zeros(self.m)
        
        # Stack vectors
        transformed_vecs_tensor = torch.stack(transformed_vecs, dim=0)  # [num_windows, m]
        num_windows = transformed_vecs_tensor.shape[0]
        
        # Compute basis indices (k mod L)
        k_positions = torch.arange(num_windows, dtype=torch.float32, device=self.device)
        j_indices = (k_positions % self.L).long()
        
        # Get corresponding B basis rows
        B_rows = self.Bbasis[j_indices]  # [num_windows, m]
        
        # Compute scalar = B[j] • transformed_vec for each window
        scalar = torch.sum(B_rows * transformed_vecs_tensor, dim=1)  # [num_windows]
        
        # Get A columns for each position
        A_cols = self.Acoeff[:, j_indices].t()  # [num_windows, m]
        
        # Compute Nk vectors for all windows in the sequence
        Nk_batch = A_cols * scalar.unsqueeze(1)  # [num_windows, m]
        
        # Return average of all N(k) vectors
        return torch.mean(Nk_batch, dim=0).detach().cpu().numpy()

    def predict_c(self, vec_seq):
        """
        Predict class label for a vector sequence using the classification head.
        
        Args:
            vec_seq: Input vector sequence
            
        Returns:
            tuple: (predicted_class, class_probabilities)
        """
        if self.classifier is None:
            raise ValueError("Model must be trained first for classification")
        
        # Get sequence vector representation
        seq_vector = self.predict_t(vec_seq)
        seq_vector_tensor = torch.tensor(seq_vector, dtype=torch.float32, device=self.device)
        
        # Get logits through classification head
        with torch.no_grad():
            logits = self.classifier(seq_vector_tensor.unsqueeze(0))
            probabilities = torch.softmax(logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            
        return predicted_class, probabilities[0].cpu().numpy()

    def predict_l(self, vec_seq, threshold=0.5):
        """
        Predict multi-label classification for a vector sequence.
        
        Args:
            vec_seq: Input vector sequence
            threshold: Probability threshold for binary classification (default: 0.5)
            
        Returns:
            numpy.ndarray: Binary label predictions (0 or 1 for each label)
            numpy.ndarray: Probability scores for each label
        """
        assert self.labeller is not None, "Model must be trained first for label prediction"
        
        # Use extract_vectors to get transformed and aggregated vectors
        transformed_vecs = self.extract_vectors(vec_seq)
        if not transformed_vecs:
            # Return zeros if no windows
            return np.zeros(self.num_labels, dtype=np.float32), np.zeros(self.num_labels, dtype=np.float32)
        
        # Stack vectors
        transformed_vecs_tensor = torch.stack(transformed_vecs, dim=0)  # [num_windows, m]
        num_windows = transformed_vecs_tensor.shape[0]
        
        # Compute basis indices (k mod L)
        k_positions = torch.arange(num_windows, dtype=torch.float32, device=self.device)
        j_indices = (k_positions % self.L).long()
        
        # Get corresponding B basis rows
        B_rows = self.Bbasis[j_indices]  # [num_windows, m]
        
        # Compute scalar = B[j] • transformed_vec for each window
        scalar = torch.sum(B_rows * transformed_vecs_tensor, dim=1)  # [num_windows]
        
        # Get A columns for each position
        A_cols = self.Acoeff[:, j_indices].t()  # [num_windows, m]
        
        # Compute Nk vectors for all windows in the sequence
        Nk_batch = A_cols * scalar.unsqueeze(1)  # [num_windows, m]
        
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

    def reconstruct(self, L, tau=0.0):
        """Reconstruct representative vector sequence of length L with temperature-controlled randomness"""
        assert self.trained, "Model must be trained first"
        if tau < 0:
            raise ValueError("Temperature must be non-negative")
            
        num_windows = (L + self.rank - 1) // self.rank
        mean_t_tensor = torch.tensor(self.mean_t, dtype=torch.float32, device=self.device)
        
        # Generate random candidate vectors for reconstruction
        candidate_vecs = torch.randn(100, self.m, device=self.device)  # 100 random candidate vectors
        
        generated_vectors = []
        
        for k in range(num_windows):
            # Compute j index for basis
            j = k % self.L
            B_row = self.Bbasis[j].unsqueeze(0)  # [1, m]
            
            # Transform candidate vectors through M
            transformed_candidates = torch.matmul(candidate_vecs, self.M.T)  # [100, m]
            
            # Apply rank operation to transformed candidates
            if self.rank_op == 'sum':
                # For sum operation, we need to simulate rank vectors
                # We'll create rank copies of each candidate and sum them
                expanded_candidates = transformed_candidates.unsqueeze(1).repeat(1, self.rank, 1)
                aggregated_candidates = torch.sum(expanded_candidates, dim=1)
            elif self.rank_op == 'pick':
                # For pick operation, randomly select one from each candidate's rank copies
                idx = random.randint(0, self.rank-1)
                expanded_candidates = transformed_candidates.unsqueeze(1).repeat(1, self.rank, 1)
                aggregated_candidates = expanded_candidates[:, idx, :]
            elif self.rank_op == 'user_func':
                # For user_func, apply average + sigmoid
                expanded_candidates = transformed_candidates.unsqueeze(1).repeat(1, self.rank, 1)
                avg = torch.mean(expanded_candidates, dim=1)
                aggregated_candidates = torch.sigmoid(avg)
            else:  # 'avg' is default
                expanded_candidates = transformed_candidates.unsqueeze(1).repeat(1, self.rank, 1)
                aggregated_candidates = torch.mean(expanded_candidates, dim=1)
            
            # Compute scalar = B[j] • aggregated_candidate for all candidates
            scalar = torch.sum(B_row * aggregated_candidates, dim=1)  # [100]
            
            # Compute Nk = scalar * A[:,j]
            A_col = self.Acoeff[:, j]  # [m]
            Nk_all = A_col * scalar.unsqueeze(1)  # [100, m]
            
            # Compute scores (negative MSE)
            errors = torch.sum((Nk_all - mean_t_tensor) ** 2, dim=1)
            scores = -errors
            
            # Select vector
            if tau == 0:  # Deterministic
                best_idx = torch.argmax(scores).item()
                best_vec = candidate_vecs[best_idx]
                generated_vectors.append(best_vec)
            else:  # Stochastic
                probs = torch.softmax(scores / tau, dim=0).detach().cpu().numpy()
                chosen_idx = np.random.choice(len(candidate_vecs), p=probs)
                chosen_vec = candidate_vecs[chosen_idx]
                generated_vectors.append(chosen_vec)
        
        # Stack and trim to exact length
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
        
        # Load config if available
        if 'config' in save_dict:
            config = save_dict['config']
            # Update attributes from config
            self.rank_op = config.get('rank_op', 'avg')
            self.rank_mode = config.get('rank_mode', 'drop')
            self.mode = config.get('mode', 'linear')
            self.step = config.get('user_step', None)
        
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
    print("Numeric Dual Descriptor AB - PyTorch GPU Accelerated Version")
    print("Optimized for vector sequence processing")
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
        batch_size=1024
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
