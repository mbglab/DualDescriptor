# Copyright (C) 2005-2025, Bin-Guang Ma (mbg@mail.hzau.edu.cn); SPDX-License-Identifier: MIT
# The Numerical Dual Descriptor Vector class (P Matrix form) implemented with PyTorch
# This program is for the demonstration of methodology and not fully refined.
# Author: Bin-Guang Ma (assisted by DeepSeek); Date: 2025-8-28 ~ 2025-12-30

import math
import random
import itertools
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy

class NumDualDescriptorPM(nn.Module):
    """
    Numerical Vector Dual Descriptor with GPU acceleration using PyTorch:
      - Processes sequences of m-dimensional real vectors instead of character sequences
      - matrix P ∈ R^{m×m} of basis coefficients (simplified 2D version)
      - square mapping matrix M ∈ R^{m×m} for vector transformation (assumes input_dim = model_dim)
      - indexed periods: period[i,j] = i*m + j + 2
      - basis function phi_{i,j}(k) = cos(2π * k / period[i,j])
      - supports 'linear' or 'nonlinear' (step-by-rank) vector extraction
    """
    def __init__(self, vec_dim, rank=1, rank_op='avg', rank_mode='drop', mode='linear', user_step=None, device='cuda'):
        super().__init__()
        self.vec_dim = vec_dim          # Dimension of input vectors and internal representation
        self.rank = rank              # r-per/k-mer length
        self.rank_op = rank_op        # 'avg', 'sum', 'pick', 'user_func'
        self.rank_mode = rank_mode    # 'pad' or 'drop'
        assert mode in ('linear', 'nonlinear')
        self.mode = mode
        self.step = user_step
        self.trained = False
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Mapping matrix M for vector transformation
        self.M = nn.Linear(self.vec_dim, self.vec_dim, bias=False)
        
        # Position-weight matrix P[i][j] (simplified 2D version)
        self.P = nn.Parameter(torch.empty(self.vec_dim, self.vec_dim))
        
        # Precompute indexed periods[i][j] (fixed, not trainable)
        periods = torch.zeros(self.vec_dim, self.vec_dim, dtype=torch.float32)
        for i in range(self.vec_dim):
            for j in range(self.vec_dim):
                periods[i, j] = i * self.vec_dim + j + 2
        self.register_buffer('periods', periods)

        # Class head (initialized later when num_classes is known)
        self.num_classes = None # Number of classes in the multi-class prediction task
        self.classifier = None        

        # Label head (initialized later when num_labels is known)
        self.num_labels = None  # Number of labels for multi-label prediction task
        self.labeller = None

        # User function for custom rank operation
        self.user_func = None

        # Initialize parameters
        self.reset_parameters()
        self.to(self.device)
        
    def reset_parameters(self):
        """Initialize model parameters"""
        nn.init.uniform_(self.M.weight, -0.5, 0.5)
        nn.init.uniform_(self.P, -0.1, 0.1)
        if self.classifier is not None:
            nn.init.normal_(self.classifier.weight, 0, 0.01)
            if self.classifier.bias is not None:
                nn.init.constant_(self.classifier.bias, 0)
        if self.num_labels is not None:
            nn.init.xavier_uniform_(self.labeller.weight)
            nn.init.zeros_(self.labeller.bias)
    
    def set_user_func(self, func):
        """Set custom user function for rank operation"""
        if callable(func):
            self.user_func = func
        else:
            raise ValueError("User function must be callable")

    def extract_vectors(self, seq_vectors):
        """
        Extract window vectors from sequence based on processing mode and rank operation.
        
        - 'linear': Slide window by 1 step, extracting contiguous vectors of length = rank
        - 'nonlinear': Slide window by custom step (or rank length if step not specified)
        
        For nonlinear mode, handles incomplete trailing fragments using:
        - 'pad': Pads with zero vectors to maintain group length
        - 'drop': Discards incomplete fragments
        
        Args:
            seq_vectors (list or tensor): Input vector sequence
            
        Returns:
            list: List of vectors after applying rank operation to each extracted vector group
        """
        L = len(seq_vectors)
        # Convert to tensor if needed
        if not isinstance(seq_vectors, torch.Tensor):
            seq_vectors = torch.tensor(seq_vectors, dtype=torch.float32, device=self.device)
        
        # Ensure device consistency
        if seq_vectors.device != self.device:
            seq_vectors = seq_vectors.to(self.device)
        
        def apply_op(vec_tensor):
            """Apply rank operation (avg/sum/pick/user_func) to a list of vectors"""
            
            if self.rank_op == 'sum':
                return torch.sum(vec_tensor, dim=0)
                
            elif self.rank_op == 'pick':
                idx = random.randint(0, len(vec_tensor)-1)
                return vec_tensor[idx]
                
            elif self.rank_op == 'user_func':
                # Use custom function if provided, else default behavior
                if self.user_func is not None and callable(self.user_func):
                    return self.user_func(vec_tensor)
                else:
                    # Default: average + sigmoid
                    avg = torch.mean(vec_tensor, dim=0)
                    return torch.sigmoid(avg)
                    
            else:  # 'avg' is default
                return torch.mean(vec_tensor, dim=0)
        
        # Linear mode: sliding window with step=1
        if self.mode == 'linear':
            vector_groups = [seq_vectors[i:i+self.rank] for i in range(L - self.rank + 1)]
        
        # Nonlinear mode: stepping with custom step size
        else:
            vector_groups = []
            step = self.step or self.rank  # Use custom step if defined, else use rank length
            
            for i in range(0, L, step):
                frag = seq_vectors[i:i+self.rank]
                frag_len = len(frag)
                
                # Pad or drop based on rank_mode setting
                if self.rank_mode == 'pad' and frag_len < self.rank:
                    # Pad fragment with zero vectors if shorter than rank
                    padding = torch.zeros(self.rank - frag_len, self.vec_dim, device=self.device)
                    frag = torch.cat([frag, padding], dim=0)
                    vector_groups.append(frag)
                elif frag_len == self.rank:
                    # Only add fragments that match full rank length
                    vector_groups.append(frag)
        
        # Apply rank operation for each group
        vectors = [apply_op(group) for group in vector_groups]
        
        # Convert list of tensors to single tensor if not empty
        if vectors:
            return torch.stack(vectors)
        else:
            # Return empty tensor with correct dimensions
            return torch.empty(0, self.vec_dim, device=self.device)

    def batch_compute_Nk(self, k_tensor, vectors):
        """
        Vectorized computation of N(k) vectors for a batch of positions and vectors
        Optimized using einsum for better performance
        
        Args:
            k_tensor: Tensor of position indices [batch_size]
            vectors: Tensor of vectors [batch_size, vec_dim]
            
        Returns:
            Tensor of N(k) vectors [batch_size, vec_dim]
        """
        # Apply square mapping matrix M to each vector
        # vectors: [batch_size, vec_dim]
        # After M transformation: [batch_size, vec_dim]
        x = self.M(vectors)  # [batch_size, vec_dim]
        
        # Expand dimensions for broadcasting [batch_size, 1, 1]
        k_expanded = k_tensor.view(-1, 1, 1)
        
        # Calculate basis functions: cos(2π*k/periods) [batch_size, vec_dim, vec_dim]
        phi = torch.cos(2 * math.pi * k_expanded / self.periods)
        
        # Optimized computation using einsum
        Nk = torch.einsum('bj,ij,bij->bi', x, self.P, phi)
            
        return Nk

    def compute_Nk(self, k, vector):
        """Compute N(k) for single position and vector (uses batch internally)"""
        # Convert to tensors and ensure correct device
        if not isinstance(vector, torch.Tensor):
            vector = torch.tensor(vector, dtype=torch.float32, device=self.device)
        elif vector.device != self.device:
            vector = vector.to(self.device)
            
        k_tensor = torch.tensor([k], dtype=torch.float32, device=self.device)
        vector_tensor = vector.unsqueeze(0)  # Add batch dimension
        
        # Use batch computation
        result = self.batch_compute_Nk(k_tensor, vector_tensor)
        return result[0]  # Return first element

    def describe(self, vectors):
        """Compute N(k) vectors for each window in vector sequence"""
        if len(vectors) == 0:
            return []
        
        # Extract and apply rank operation to vectors
        extracted_vectors = self.extract_vectors(vectors)
        if extracted_vectors.shape[0] == 0:
            return []
        
        k_positions = torch.arange(extracted_vectors.shape[0], dtype=torch.float32, device=self.device)
        
        # Batch compute all Nk vectors
        N_batch = self.batch_compute_Nk(k_positions, extracted_vectors)
        return N_batch.detach().cpu().numpy()

    def S(self, vectors):
        """
        Compute list of S(l)=sum(N(k)) (k=1,...,l; l=1,...,L) for a given vector sequence.        
        """
        if len(vectors) == 0:
            return []
        
        # Extract and apply rank operation to vectors
        extracted_vectors = self.extract_vectors(vectors)
        if extracted_vectors.shape[0] == 0:
            return []
        
        k_positions = torch.arange(extracted_vectors.shape[0], dtype=torch.float32, device=self.device)
        
        # Batch compute all Nk vectors
        N_batch = self.batch_compute_Nk(k_positions, extracted_vectors)
        
        # Compute cumulative sum (S vectors)
        S_cum = torch.cumsum(N_batch, dim=0)
        return [s.detach().cpu().numpy() for s in S_cum]

    def D(self, vector_seqs, t_list):
        """
        Compute mean squared deviation D across sequences:
        D = average over all positions of (N(k)-t_seq)^2
        """
        total_loss = 0.0
        total_positions = 0
        
        # Convert target vectors to tensor
        t_tensors = [torch.tensor(t, dtype=torch.float32, device=self.device) for t in t_list]
        
        for vectors, t in zip(vector_seqs, t_tensors):
            # Extract and apply rank operation to vectors
            extracted_vectors = self.extract_vectors(vectors)
            if extracted_vectors.shape[0] == 0:
                continue
                
            k_positions = torch.arange(extracted_vectors.shape[0], dtype=torch.float32, device=self.device)
            
            # Batch compute all Nk vectors
            N_batch = self.batch_compute_Nk(k_positions, extracted_vectors)
            
            # Compute loss for each position
            losses = torch.sum((N_batch - t) ** 2, dim=1)
            total_loss += losses.sum().item()
            total_positions += extracted_vectors.shape[0]
                
        return total_loss / total_positions if total_positions else 0.0

    def d(self, vectors, t):
        """
        Compute pattern deviation value (d) for a single vector sequence. 
        """
        d_value = self.D([vectors], [t])
        return d_value

    def reg_train(self, vector_seqs, t_list, max_iters=1000, tol=1e-8, learning_rate=0.01, 
               continued=False, decay_rate=1.0, print_every=10, batch_size=32,
               checkpoint_file=None, checkpoint_interval=10):
        """
        Train the model using gradient descent with sequence-level batch processing.
        Optimized for GPU memory efficiency by processing sequences individually.
        
        Args:
            vector_seqs: List of vector sequences for training
            t_list: List of target vectors corresponding to sequences
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
        
        if not continued:
            self.reset_parameters()
        
        # Convert target vectors to tensor
        t_tensors = [torch.tensor(t, dtype=torch.float32, device=self.device) for t in t_list]
        
        # Setup optimizer and scheduler
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)
        
        # Training state variables
        history = []
        prev_loss = float('inf')
        best_loss = float('inf')
        best_model_state = None
        
        for it in range(max_iters):
            total_loss = 0.0
            total_sequences = 0
            
            # Shuffle sequences for each epoch
            indices = list(range(len(vector_seqs)))
            random.shuffle(indices)
            
            # Process sequences in batches
            for batch_start in range(0, len(indices), batch_size):
                batch_indices = indices[batch_start:batch_start + batch_size]
                batch_seqs = [vector_seqs[idx] for idx in batch_indices]
                batch_targets = [t_tensors[idx] for idx in batch_indices]
                
                optimizer.zero_grad()
                batch_loss = 0.0
                
                # Process each sequence in the batch
                for vectors, target in zip(batch_seqs, batch_targets):
                    # Extract and apply rank operation to vectors
                    extracted_vectors = self.extract_vectors(vectors)
                    if extracted_vectors.shape[0] == 0:
                        continue
                        
                    k_positions = torch.arange(extracted_vectors.shape[0], dtype=torch.float32, device=self.device)
                    
                    # Compute N(k) vectors for all extracted vectors in the sequence
                    Nk_batch = self.batch_compute_Nk(k_positions, extracted_vectors)
                    
                    # Compute sequence-level target: average of all N(k) vectors
                    seq_pred = torch.mean(Nk_batch, dim=0)
                    
                    # Calculate loss for this sequence
                    seq_loss = torch.sum((seq_pred - target) ** 2)
                    batch_loss += seq_loss
                    
                    # Clean up intermediate tensors to free memory
                    del Nk_batch, seq_pred, extracted_vectors, k_positions
                
                # Average loss over sequences in batch and backpropagate
                if len(batch_seqs) > 0:
                    batch_loss = batch_loss / len(batch_seqs)
                    batch_loss.backward()
                    optimizer.step()
                    
                    total_loss += batch_loss.item() * len(batch_seqs)
                    total_sequences += len(batch_seqs)
                
                # Clear GPU cache periodically
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Calculate average loss for this iteration
            if total_sequences > 0:
                avg_loss = total_loss / total_sequences
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
            
            # Save checkpoint if specified
            if checkpoint_file and (it % checkpoint_interval == 0 or it == max_iters - 1):
                self._save_checkpoint(checkpoint_file, it, history, optimizer, scheduler, best_loss)
            
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
        
        # Compute and store training statistics for reconstruction/generation
        self._compute_training_statistics(vector_seqs)
        self.trained = True
        
        return history

    def cls_train(self, vector_seqs, labels, num_classes, max_iters=1000, tol=1e-8, learning_rate=0.01,
                  continued=False, decay_rate=1.0, print_every=10, batch_size=32,
                  checkpoint_file=None, checkpoint_interval=10):
        """
        Train the model for multi-class classification using cross-entropy loss.
        Optimized for GPU memory efficiency by processing sequences individually.
        
        Args:
            vector_seqs: List of vector sequences for training
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
            self.classifier = nn.Linear(self.vec_dim, num_classes).to(self.device)
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
            total_sequences = 0
            correct_predictions = 0
            
            # Shuffle sequences for each epoch
            indices = list(range(len(vector_seqs)))
            random.shuffle(indices)
            
            # Process sequences in batches
            for batch_start in range(0, len(indices), batch_size):
                batch_indices = indices[batch_start:batch_start + batch_size]
                batch_seqs = [vector_seqs[idx] for idx in batch_indices]
                batch_labels = label_tensors[batch_indices]
                
                optimizer.zero_grad()
                batch_loss = 0.0
                batch_logits = []
                
                # Process each sequence in the batch
                for vectors in batch_seqs:
                    # Extract and apply rank operation to vectors
                    extracted_vectors = self.extract_vectors(vectors)
                    if extracted_vectors.shape[0] == 0:
                        # For empty sequences, use zero vector
                        seq_vector = torch.zeros(self.vec_dim, device=self.device)
                    else:
                        k_positions = torch.arange(extracted_vectors.shape[0], dtype=torch.float32, device=self.device)
                        
                        # Compute N(k) vectors for all extracted vectors in the sequence
                        Nk_batch = self.batch_compute_Nk(k_positions, extracted_vectors)
                        
                        # Compute sequence-level vector: average of all N(k) vectors
                        seq_vector = torch.mean(Nk_batch, dim=0)
                        
                        # Clean up intermediate tensors to free memory
                        del Nk_batch, extracted_vectors, k_positions
                    
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

    def lbl_train(self, vector_seqs, labels, num_labels, max_iters=1000, tol=1e-8, learning_rate=0.01, 
                 continued=False, decay_rate=1.0, print_every=10, batch_size=32,
                 checkpoint_file=None, checkpoint_interval=10, pos_weight=None):
        """
        Train the model for multi-label classification using binary cross-entropy loss.
        
        Args:
            vector_seqs: List of vector sequences for training
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
            self.labeller = nn.Linear(self.vec_dim, num_labels).to(self.device)
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
            total_sequences = 0
            
            # Shuffle sequences for each epoch
            indices = list(range(len(vector_seqs)))
            random.shuffle(indices)
            
            # Process sequences in batches
            for batch_start in range(0, len(indices), batch_size):
                batch_indices = indices[batch_start:batch_start + batch_size]
                batch_seqs = [vector_seqs[idx] for idx in batch_indices]
                batch_labels = labels_tensor[batch_indices]
                
                optimizer.zero_grad()
                batch_loss = 0.0
                batch_correct = 0
                batch_predictions = 0
                
                # Process each sequence in the batch
                batch_predictions_list = []
                for vectors in batch_seqs:
                    # Extract and apply rank operation to vectors
                    extracted_vectors = self.extract_vectors(vectors)
                    if extracted_vectors.shape[0] == 0:
                        # If no vectors extracted, skip this sequence
                        continue
                        
                    k_positions = torch.arange(extracted_vectors.shape[0], dtype=torch.float32, device=self.device)
                    
                    # Compute N(k) vectors for all extracted vectors in the sequence
                    Nk_batch = self.batch_compute_Nk(k_positions, extracted_vectors)
                    
                    # Compute sequence representation: average of all N(k) vectors
                    seq_representation = torch.mean(Nk_batch, dim=0)
                    
                    # Pass through classification head to get logits
                    logits = self.labeller(seq_representation)
                    batch_predictions_list.append(logits)
                    
                    # Clean up intermediate tensors to free memory
                    del Nk_batch, seq_representation, extracted_vectors, k_positions
                
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

    def self_train(self, vector_seqs, max_iters=100, tol=1e-6, learning_rate=0.01, 
                   continued=False, decay_rate=1.0, print_every=10,
                   batch_size=32, checkpoint_file=None, checkpoint_interval=5):
        """
        Self-training method for self-consistency (gap mode) with memory-efficient sequence processing.
        Trains the model so that N(k) vectors match the transformed vector windows at each position.
        
        Args:
            vector_seqs: List of vector sequences for training
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
        
        if not continued:
            self.reset_parameters()
        
        # Setup optimizer and scheduler
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)
        
        # Training state variables
        history = []
        prev_loss = float('inf')
        best_loss = float('inf')
        best_model_state = None
        
        for it in range(max_iters):
            total_loss = 0.0
            total_samples = 0
            
            # Shuffle sequences for each epoch
            indices = list(range(len(vector_seqs)))
            random.shuffle(indices)
            
            # Process sequences in batches
            for batch_start in range(0, len(indices), batch_size):
                batch_indices = indices[batch_start:batch_start + batch_size]
                batch_seqs = [vector_seqs[idx] for idx in batch_indices]
                
                optimizer.zero_grad()
                batch_loss = 0.0
                batch_sample_count = 0
                
                # Process each sequence in the batch
                for vectors in batch_seqs:
                    # Extract and apply rank operation to vectors
                    extracted_vectors = self.extract_vectors(vectors)
                    if extracted_vectors.shape[0] == 0:
                        continue
                        
                    k_positions = torch.arange(extracted_vectors.shape[0], dtype=torch.float32, device=self.device)
                    
                    # Compute N(k) vectors for all extracted vectors
                    Nk_batch = self.batch_compute_Nk(k_positions, extracted_vectors)
                    
                    # Transform extracted vectors using M
                    target_vectors = self.M(extracted_vectors)  # [num_vectors, vec_dim]
                    
                    # Self-consistency loss: N(k) should match transformed vector at position k
                    seq_loss = 0.0
                    valid_positions = 0
                    
                    for k in range(extracted_vectors.shape[0]):
                        target = target_vectors[k]
                        pred = Nk_batch[k]
                        seq_loss += torch.sum((pred - target) ** 2)
                        valid_positions += 1
                    
                    if valid_positions > 0:
                        seq_loss = seq_loss / valid_positions
                        batch_loss += seq_loss
                        batch_sample_count += 1
                    
                    # Clean up intermediate tensors
                    del Nk_batch, target_vectors, extracted_vectors, k_positions
                
                # Backpropagate batch loss
                if batch_sample_count > 0:
                    batch_loss = batch_loss / batch_sample_count
                    batch_loss.backward()
                    optimizer.step()
                    
                    total_loss += batch_loss.item() * batch_sample_count
                    total_samples += batch_sample_count
                
                # Clear GPU cache
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
            
            # Print training progress
            if it % print_every == 0 or it == max_iters - 1:
                current_lr = scheduler.get_last_lr()[0]
                print(f"Self-Train Iter {it:3d}: Loss = {avg_loss:.6f}, LR = {current_lr:.6f}")
            
            # Save checkpoint if specified
            if checkpoint_file and (it % checkpoint_interval == 0 or it == max_iters - 1):
                self._save_checkpoint(checkpoint_file, it, history, optimizer, scheduler, best_loss)
            
            # Check convergence
            if abs(prev_loss - avg_loss) < tol:
                print(f"Converged after {it+1} iterations")
                # Restore best model state
                if best_model_state is not None:
                    self.load_state_dict(best_model_state)
                break
                
            prev_loss = avg_loss
            
            # Learning rate scheduling
            scheduler.step()
        
        # Compute and store training statistics
        self._compute_training_statistics(vector_seqs)
        self.trained = True
        
        return history

    def _compute_training_statistics(self, vector_seqs, batch_size=50):
        """
        Compute training statistics for reconstruction and generation.
        Processes sequences in batches to manage memory usage.
        
        Args:
            vector_seqs: List of training vector sequences
            batch_size: Number of sequences to process in each batch
        """
        total_vector_count = 0
        total_t = torch.zeros(self.vec_dim, device=self.device)
        
        with torch.no_grad():
            for i in range(0, len(vector_seqs), batch_size):
                batch_seqs = vector_seqs[i:i + batch_size]
                batch_vector_count = 0
                batch_vec_sum = torch.zeros(self.vec_dim, device=self.device)
                
                for vectors in batch_seqs:
                    # Extract and apply rank operation to vectors
                    extracted_vectors = self.extract_vectors(vectors)
                    if extracted_vectors.shape[0] == 0:
                        continue
                        
                    batch_vector_count += extracted_vectors.shape[0]
                    k_positions = torch.arange(extracted_vectors.shape[0], dtype=torch.float32, device=self.device)
                    
                    Nk_batch = self.batch_compute_Nk(k_positions, extracted_vectors)
                    batch_vec_sum += Nk_batch.sum(dim=0)
                    
                    # Clean up
                    del Nk_batch, extracted_vectors, k_positions
                
                total_vector_count += batch_vector_count
                total_t += batch_vec_sum
                
                # Clear GPU cache between batches
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        self.mean_vector_count = total_vector_count / len(vector_seqs) if vector_seqs else 0
        self.mean_t = (total_t / total_vector_count).cpu().numpy() if total_vector_count > 0 else np.zeros(self.vec_dim)

    def _save_checkpoint(self, checkpoint_file, iteration, history, optimizer, scheduler, best_loss):
        """
        Save training checkpoint with complete training state.
        
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
            'training_stats': {
                'mean_t': self.mean_t,
                'mean_vector_count': self.mean_vector_count
            }
        }
        torch.save(checkpoint, checkpoint_file)
        print(f"Checkpoint saved at iteration {iteration}")

    def predict_t(self, vectors):
        """
        Predict target vector for a vector sequence
        Returns the average of all N(k) vectors in the sequence
        """
        if len(vectors) == 0:
            return [0.0] * self.vec_dim
        
        # Extract and apply rank operation to vectors
        extracted_vectors = self.extract_vectors(vectors)
        if extracted_vectors.shape[0] == 0:
            return [0.0] * self.vec_dim
        
        k_positions = torch.arange(extracted_vectors.shape[0], dtype=torch.float32, device=self.device)
        
        # Batch compute all Nk vectors
        Nk_batch = self.batch_compute_Nk(k_positions, extracted_vectors)
        Nk_sum = torch.sum(Nk_batch, dim=0)
        
        return (Nk_sum / extracted_vectors.shape[0]).detach().cpu().numpy()

    def predict_c(self, vectors):
        """
        Predict class label for a vector sequence using the classification head.
        
        Args:
            vectors: Input vector sequence
            
        Returns:
            tuple: (predicted_class, class_probabilities)
        """
        if self.classifier is None:
            raise ValueError("Model must be trained first for classification")
        
        # Get sequence vector representation
        seq_vector = self.predict_t(vectors)
        seq_vector_tensor = torch.tensor(seq_vector, dtype=torch.float32, device=self.device)
        
        # Get logits through classification head
        with torch.no_grad():
            logits = self.classifier(seq_vector_tensor.unsqueeze(0))
            probabilities = torch.softmax(logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            
        return predicted_class, probabilities[0].cpu().numpy()

    def predict_l(self, vectors, threshold=0.5):
        """
        Predict multi-label classification for a vector sequence.
        
        Args:
            vectors: Input vector sequence
            threshold: Probability threshold for binary classification (default: 0.5)
            
        Returns:
            numpy.ndarray: Binary label predictions (0 or 1 for each label)
            numpy.ndarray: Probability scores for each label
        """
        assert self.labeller is not None, "Model must be trained first for label prediction"
        
        if len(vectors) == 0:
            # Return zeros if no vectors
            return np.zeros(self.num_labels, dtype=np.float32), np.zeros(self.num_labels, dtype=np.float32)
        
        # Extract and apply rank operation to vectors
        extracted_vectors = self.extract_vectors(vectors)
        if extracted_vectors.shape[0] == 0:
            return np.zeros(self.num_labels, dtype=np.float32), np.zeros(self.num_labels, dtype=np.float32)
        
        k_positions = torch.arange(extracted_vectors.shape[0], dtype=torch.float32, device=self.device)
        
        # Compute N(k) vectors for all extracted vectors in the sequence
        Nk_batch = self.batch_compute_Nk(k_positions, extracted_vectors)
        
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
        """Reconstruct representative vector sequence of length L by minimizing error with temperature-controlled randomness"""
        assert self.trained, "Model must be trained first"
        if tau < 0:
            raise ValueError("Temperature must be non-negative")
            
        # For reconstruction, we need to generate rank-length vector windows
        # Since we're dealing with continuous vectors, we need a different approach
        # We'll generate random vectors and select the best ones
        
        num_windows = (L + self.rank - 1) // self.rank
        mean_t_tensor = torch.tensor(self.mean_t, dtype=torch.float32, device=self.device)
        
        # We need to generate candidate vectors
        # For simplicity, we'll generate random vectors from a normal distribution
        # In practice, you might want to use a more sophisticated generation method
        
        generated_vectors = []
        
        # Pre-generate some candidate vectors
        num_candidates = 100
        candidate_vectors = torch.randn(num_candidates, self.vec_dim, device=self.device)
        
        for k in range(num_windows):
            # Compute Nk for all candidate vectors at position k
            k_tensor = torch.tensor([k] * num_candidates, dtype=torch.float32, device=self.device)
            Nk_all = self.batch_compute_Nk(k_tensor, candidate_vectors)
            
            # Compute scores
            errors = torch.sum((Nk_all - mean_t_tensor) ** 2, dim=1)
            scores = -errors  # Convert to score (higher = better)
            
            if tau == 0:  # Deterministic selection
                max_idx = torch.argmax(scores).item()
                best_vector = candidate_vectors[max_idx]
            else:  # Stochastic selection
                probs = torch.softmax(scores / tau, dim=0).detach().cpu().numpy()
                chosen_idx = random.choices(range(num_candidates), weights=probs, k=1)[0]
                best_vector = candidate_vectors[chosen_idx]
            
            generated_vectors.append(best_vector)
        
        # Concatenate all vectors to form the full sequence
        full_sequence = torch.stack(generated_vectors)[:L]
        return full_sequence.detach().cpu().numpy()

    def save(self, filename):
        """Save model state to file"""
        torch.save(self.state_dict(), filename)
        print(f"Model saved to {filename}")

    def load(self, filename):
        """Load model state from file"""        
        # Use weights only=True to avoid security warnings
        try:
            # PyTorch 1.13+ supports the "weights only" parameter
            state_dict = torch.load(filename, map_location=self.device, weights_only=True)
        except TypeError:
            # Rollback solution for old versions of PyTorch
            state_dict = torch.load(filename, map_location=self.device)
        
        self.load_state_dict(state_dict)
        print(f"Model loaded from {filename}")
        return self


# === Example Usage ===
if __name__=="__main__":
    import matplotlib.pyplot as plt
    from scipy.stats import pearsonr
    
    print("="*50)
    print("Numerical Dual Descriptor PM - PyTorch GPU Accelerated Version")
    print("Processes sequences of m-dimensional real vectors with 2D matrix P")   
    print("="*50)
    
    # Set random seeds to ensure reproducibility
    torch.manual_seed(11)
    random.seed(11)
    np.random.seed(11)
    
    # Parameters
    vec_dim = 10     # Dimension of input vectors and internal representation
    rank = 1          # Window size for vector sequences
    user_step = 1     # Step size for nonlinear mode
    
    # Initialize the model (using vec_dim instead of input_dim and model_dim)
    ndd = NumDualDescriptorPM(
        vec_dim=vec_dim,
        rank=rank, 
        mode='nonlinear', 
        user_step=user_step,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Display device information
    print(f"\nUsing device: {ndd.device}")
    print(f"Vector dimension: {vec_dim}")
    print(f"Rank (window size): {rank}")
    print(f"P matrix shape: {ndd.P.shape}")  # Should be [vec_dim, vec_dim]
    print(f"M matrix shape: {ndd.M.weight.shape}")  # Should be [vec_dim, vec_dim]
    
    # Generate 100 vector sequences with random target vectors
    print("\nGenerating training data...")
    seqs, t_list = [], []
    for _ in range(100):
        L = random.randint(200, 300)
        # Generate random vector sequence
        seq = torch.randn(L, vec_dim)
        seqs.append(seq)
        # Create a random vector target
        t_list.append(np.random.uniform(-1.0, 1.0, vec_dim))

    # Training model
    print("\n" + "="*50)
    print("Starting Gradient Descent Training")
    print("="*50)
    ndd.reg_train(seqs, t_list, max_iters=100, tol=1e-9, learning_rate=1.0, decay_rate=0.95, batch_size=32)  
   
    # Predict the target vector of the first sequence
    aseq = seqs[0]
    t_pred = ndd.predict_t(aseq)
    print(f"\nPredicted t for first sequence: {[round(x.item(), 4) for x in t_pred[:5]]}...")    
    
    # Calculate the correlation between the predicted and the real target
    print("\nCalculating prediction correlations...")
    pred_t_list = [ndd.predict_t(seq) for seq in seqs]
    
    # The predicted values and actual values used for correlation calculation
    corr_sum = 0.0
    for i in range(ndd.vec_dim):
        actu_t = [t_vec[i] for t_vec in t_list]
        pred_t = [t_vec[i] for t_vec in pred_t_list]
        corr, _ = pearsonr(actu_t, pred_t)
        print(f"Dimension {i} prediction correlation: {corr:.4f}")
        corr_sum += corr
    corr_avg = corr_sum / ndd.vec_dim
    print(f"Average correlation: {corr_avg:.4f}")         
   
    # Reconstruct representative sequences
    print("\nGenerating reconstructed sequences...")
    # Note: Reconstruction for continuous vectors generates new vector sequences
    seq_det = ndd.reconstruct(L=100, tau=0.0)
    seq_rand = ndd.reconstruct(L=100, tau=0.5)
    print(f"Deterministic reconstruction shape: {seq_det.shape}")
    print(f"Stochastic reconstruction shape: {seq_rand.shape}")
    print(f"Deterministic mean: {np.mean(seq_det):.4f}, std: {np.std(seq_det):.4f}")
    print(f"Stochastic mean: {np.mean(seq_rand):.4f}, std: {np.std(seq_rand):.4f}")

    # Classification task 
    print("\n" + "="*50)
    print("Classification Task")
    print("="*50)
    
    # Generate classification data
    num_classes = 3
    class_seqs = []
    class_labels = []
    
    # Create vector sequences with different patterns for each class
    for class_id in range(num_classes):
        for _ in range(50):  # 50 sequences per class
            L = random.randint(150, 250)
            if class_id == 0:
                # Class 0: Vectors with positive mean
                seq = torch.randn(L, vec_dim) + 1.0
            elif class_id == 1:
                # Class 1: Vectors with negative mean
                seq = torch.randn(L, vec_dim) - 1.0
            else:
                # Class 2: Standard normal vectors
                seq = torch.randn(L, vec_dim)
            
            class_seqs.append(seq)
            class_labels.append(class_id)    

    # Initialize new model for classification
    ndd_cls = NumDualDescriptorPM(
        vec_dim=vec_dim,
        rank=rank, 
        mode='nonlinear', 
        user_step=user_step,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Train for classification
    print("\n" + "="*50)
    print("Starting Classification Training")
    print("="*50)
    history = ndd_cls.cls_train(class_seqs, class_labels, num_classes, 
                               max_iters=50, tol=1e-8, learning_rate=0.05,
                               decay_rate=0.99, batch_size=16, print_every=5)
    
    # Show prediction results on the training dataset
    print("\n" + "="*50)
    print("Prediction results")
    print("="*50)
    
    correct = 0
    all_predictions = []
    
    for seq, true_label in zip(class_seqs, class_labels):
        pred_class, probs = ndd_cls.predict_c(seq)
        all_predictions.append(pred_class)
        
        if pred_class == true_label:
            correct += 1
    
    accuracy = correct / len(class_seqs)
    print(f"Accuracy: {accuracy:.4f} ({correct}/{len(class_seqs)})")
    
    # Show some example predictions
    print("\nExample predictions:")
    for i in range(min(5, len(class_seqs))):
        pred_class, probs = ndd_cls.predict_c(class_seqs[i])
        print(f"Seq {i+1}: True={class_labels[i]}, Pred={pred_class}, Probs={[f'{p:.3f}' for p in probs[:3]]}...")

    # Multi-label classification
    print("\n\n" + "="*50)
    print("Multi-Label Classification Model")
    print("="*50)

    # Generate 100 vector sequences with random multi-labels for classification
    num_labels = 4  # Example: 4 different functions    
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

    
    ndd_lbl = NumDualDescriptorPM(
        vec_dim=vec_dim,
        rank=rank, 
        mode='nonlinear', 
        user_step=user_step,        
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Training multi-label classification model
    print("\n" + "="*50)
    print("Starting Gradient Descent Training for Multi-Label Classification")
    print("="*50)
       
    # Train the model
    loss_history, acc_history = ndd_lbl.lbl_train(
        label_seqs, labels, num_labels,
        max_iters=50, 
        tol=1e-16, 
        learning_rate=0.05, 
        decay_rate=0.99, 
        print_every=10, 
        batch_size=16
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
        pred_binary, pred_probs = ndd_lbl.predict_l(seq, threshold=0.5)
        
        # Convert true labels to numpy array
        true_labels_np = np.array(true_labels)
        
        # Calculate accuracy for this sequence (exact match)
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
    binary_pred, probs_pred = ndd_lbl.predict_l(test_seq, threshold=0.5)
    print(f"\nPredicted binary labels: {binary_pred}")
    print(f"Predicted probabilities: {[f'{p:.4f}' for p in probs_pred]}")
    
    # Interpret the predictions
    label_names = ["Function_A", "Function_B", "Function_C", "Function_D"]
    print("\nLabel interpretation:")
    for i, (binary, prob) in enumerate(zip(binary_pred, probs_pred)):
        status = "ACTIVE" if binary > 0.5 else "INACTIVE"
        print(f"  {label_names[i]}: {status} (confidence: {prob:.4f})")

    # Self-training examples
    print("\n" + "="*50)
    print("Self-Training Example")
    print("="*50)
    
    # Create a new model
    ndd_self = NumDualDescriptorPM(
        vec_dim=vec_dim,
        rank=rank, 
        mode='nonlinear', 
        user_step=user_step,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Generate sample sequences
    self_seqs = []
    for _ in range(10):
        L = random.randint(200, 300)
        self_seqs.append(torch.randn(L, vec_dim))
    
    # Conduct self-consistency training
    print("\nTraining for self-consistency:")
    self_history = ndd_self.self_train(
        self_seqs, 
        max_iters=30, 
        tol=1e-8, 
        learning_rate=0.01,         
        batch_size=4
    )
    
    # Visualize training loss
    plt.figure(figsize=(10, 6))
    plt.plot(self_history)
    plt.title('Self-Training Loss History')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig('self_training_loss.png')
    print("\nSelf-training loss plot saved as 'self_training_loss.png'")
    
    # Test reconstruction
    print("\nTesting reconstruction...")
    rec_vectors = ndd_self.reconstruct(L=100, tau=0.2)
    print(f"Reconstructed vector sequence shape: {rec_vectors.shape}")
    print(f"Mean: {np.mean(rec_vectors):.4f}, Std: {np.std(rec_vectors):.4f}")

    print("\nAll tests completed successfully!")
    print("="*50)
