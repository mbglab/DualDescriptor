# Copyright (C) 2005-2025, Bin-Guang Ma (mbg@mail.hzau.edu.cn); SPDX-License-Identifier: MIT
# The Hierarchical Numeric Dual Descriptor (Random AB matrix form) with Linker matrices in PyTorch
# With layer normalization and residual connections and generation capability
# This program is for the demonstration of methodology and not fully refined.
# Author: Bin-Guang Ma (assisted by DeepSeek); Date: 2025-7-22 ~ 2026-1-13

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import os

class Layer(nn.Module):
    """
    A single layer of the Hierarchical Numeric Dual Descriptor with Linker
    
    Args:
        in_dim (int): Input dimension
        out_dim (int): Output dimension
        basis_dim (int): Basis dimension
        in_seq_len (int): Input sequence length
        out_seq_len (int): Output sequence length
        linker_trainable (bool): Whether linker matrix is trainable
        residual_mode (str or None): Residual connection type. Options are:
            - 'separate': use separate projection and Linker for residual
            - 'shared': share M and Linker for residual
            - None: no residual connection
        device (str): Device to place the layer on ('cuda' or 'cpu')
    """
    def __init__(self, in_dim, out_dim, basis_dim, in_seq_len, out_seq_len, 
                 linker_trainable, residual_mode=None, device='cuda'):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.basis_dim = basis_dim
        self.in_seq_len = in_seq_len
        self.out_seq_len = out_seq_len
        self.residual_mode = residual_mode
        self.linker_trainable = linker_trainable
        self.device = device
        
        # Initialize transformation matrix M
        self.M = nn.Parameter(torch.empty(out_dim, in_dim, device=device))
        nn.init.uniform_(self.M, -0.1, 0.1)
        
        # Initialize coefficient matrix Acoeff
        self.Acoeff = nn.Parameter(torch.empty(out_dim, basis_dim, device=device))
        nn.init.uniform_(self.Acoeff, -0.1, 0.1)
        
        # Initialize basis matrix Bbasis
        self.Bbasis = nn.Parameter(torch.empty(basis_dim, out_dim, device=device))
        nn.init.uniform_(self.Bbasis, -0.1, 0.1)
        
        # Initialize Linker matrix
        self.Linker = nn.Parameter(torch.empty(in_seq_len, out_seq_len, device=device))
        nn.init.uniform_(self.Linker, -0.1, 0.1)
        
        # Freeze Linker if not trainable
        if not linker_trainable:
            self.Linker.requires_grad = False
            
        # Layer normalization
        self.layer_norm = nn.LayerNorm(out_dim, device=device)
        
        # Precompute basis indices for sequence positions
        self.register_buffer('basis_indices', 
                            torch.tensor([i % basis_dim for i in range(in_seq_len)], device=device))
        
        # Residual connection setup
        if self.residual_mode == 'separate':
            # Separate projection and Linker for residual path
            # Feature dimension projection
            self.residual_proj = nn.Linear(in_dim, out_dim, bias=False, device=device)
            nn.init.uniform_(self.residual_proj.weight, -0.1, 0.1)
            
            # Sequence length transformation matrix for residual
            self.residual_linker = nn.Parameter(torch.empty(in_seq_len, out_seq_len, device=device))
            nn.init.uniform_(self.residual_linker, -0.1, 0.1)
            
            # Freeze residual Linker if main Linker is not trainable
            if not linker_trainable:
                self.residual_linker.requires_grad = False
            
    def forward(self, x):
        """
        Forward pass for the layer
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, in_seq_len, in_dim)
            
        Returns:
            Tensor: Output tensor of shape (batch_size, out_seq_len, out_dim)
        """
        # Save for residual connection
        residual = x
        
        # Main path processing
        # Apply linear transformation: (batch_size, in_seq_len, in_dim) -> (batch_size, in_seq_len, out_dim)
        x = torch.matmul(x, self.M.T)
        
        # Apply layer normalization
        x = self.layer_norm(x)

        # Position-wise processing
        # Select basis vectors for each position: (in_seq_len, out_dim)
        basis_vectors = self.Bbasis[self.basis_indices]
        
        # Compute scalars: (batch_size, in_seq_len)
        scalars = torch.einsum('bsd,sd->bs', x, basis_vectors)
        
        # Select coefficients for each position: (in_seq_len, out_dim)
        coeffs = self.Acoeff[:, self.basis_indices].permute(1, 0)
        
        # Compute position outputs: (batch_size, in_seq_len, out_dim)
        u = coeffs * scalars.unsqueeze(-1)
        
        # Apply sequence length transformation using Linker matrix
        # (batch_size, in_seq_len, out_dim) x (in_seq_len, out_seq_len) -> (batch_size, out_dim, out_seq_len)
        v = torch.matmul(u.permute(0, 2, 1), self.Linker)
        
        # Transpose to (batch_size, out_seq_len, out_dim)
        main_output = v.permute(0, 2, 1)
        
        # Residual connection processing
        if self.residual_mode == 'separate':
            # Separate projection and Linker for residual
            # Feature projection
            residual_feat = self.residual_proj(residual)
            
            # Sequence transformation for residual
            residual_seq = torch.matmul(residual_feat.permute(0, 2, 1), self.residual_linker)
            residual_out = residual_seq.permute(0, 2, 1)
            
            # Add residual
            out = main_output + residual_out
            
        elif self.residual_mode == 'shared':
            # Shared M and Linker for residual
            # Feature projection using shared M
            residual_feat = torch.matmul(residual, self.M.T)
            
            # Sequence transformation using shared Linker
            residual_seq = torch.matmul(residual_feat.permute(0, 2, 1), self.Linker)
            residual_out = residual_seq.permute(0, 2, 1)
            
            # Add residual
            out = main_output + residual_out
            
        else:
            # Identity residual only if dimensions match
            if self.in_dim == self.out_dim and self.in_seq_len == self.out_seq_len:
                # Sequence transformation for residual
                residual_seq = torch.matmul(residual.permute(0, 2, 1), self.Linker)
                residual_out = residual_seq.permute(0, 2, 1)
                out = main_output + residual_out
            else:
                out = main_output
       
        return out

class HierDDLrn(nn.Module):
    """
    Hierarchical Numeric Dual Descriptor with Linker for vector sequences
    
    Args:
        input_dim (int): Dimension of input vectors
        model_dims (list): Output dimensions for each layer
        basis_dims (list): Basis dimensions for each layer
        linker_dims (list): Output sequence lengths for each layer
        input_seq_len (int): Fixed input sequence length
        linker_trainable (bool|list): Controls if Linker matrices are trainable
        residual_mode_list (str|list|None): Residual connection mode for each layer. Options:
            - 'separate': use separate projection and Linker for residual
            - 'shared': share M and Linker for residual
            - None: no residual connection
        device (str): Device to place the model on ('cuda' or 'cpu')
    """
    def __init__(self, input_dim=4, model_dims=[4], basis_dims=[50], 
                 linker_dims=[50], input_seq_len=100, linker_trainable=False,
                 residual_mode_list=None, device='cuda'):
        super().__init__()
        self.input_dim = input_dim
        self.model_dims = model_dims
        self.basis_dims = basis_dims
        self.linker_dims = linker_dims
        self.input_seq_len = input_seq_len
        self.num_layers = len(model_dims)
        self.trained = False
        self.device = device
        
        # Validate dimensions
        if len(basis_dims) != self.num_layers:
            raise ValueError("basis_dims length must match model_dims length")
        if len(linker_dims) != self.num_layers:
            raise ValueError("linker_dims length must match model_dims length")
        
        # Process linker_trainable parameter
        if isinstance(linker_trainable, bool):
            self.linker_trainable = [linker_trainable] * self.num_layers
        elif isinstance(linker_trainable, list):
            if len(linker_trainable) != self.num_layers:
                raise ValueError("linker_trainable list length must match number of layers")
            self.linker_trainable = linker_trainable
        else:
            raise TypeError("linker_trainable must be bool or list of bools")
        
        # Process residual_mode_list
        if residual_mode_list is None:
            self.residual_mode_list = [None] * self.num_layers
        elif isinstance(residual_mode_list, str):
            self.residual_mode_list = [residual_mode_list] * self.num_layers
        elif isinstance(residual_mode_list, list):
            if len(residual_mode_list) != self.num_layers:
                raise ValueError("residual_mode_list length must match number of layers")
            self.residual_mode_list = residual_mode_list
        else:
            raise TypeError("residual_mode_list must be str, list of str, or None")
        
        # Create layers
        self.layers = nn.ModuleList()
        for i, out_dim in enumerate(model_dims):
            # Determine input dimensions for this layer
            in_dim = input_dim if i == 0 else model_dims[i-1]
            in_seq = input_seq_len if i == 0 else linker_dims[i-1]
            out_seq = linker_dims[i]
            residual_mode = self.residual_mode_list[i]
            
            layer = Layer(
                in_dim=in_dim,
                out_dim=out_dim,
                basis_dim=basis_dims[i],
                in_seq_len=in_seq,
                out_seq_len=out_seq,
                linker_trainable=self.linker_trainable[i],
                residual_mode=residual_mode,
                device=device
            )
            self.layers.append(layer)
    
    def forward(self, x):
        """
        Forward pass through all layers
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, input_seq_len, input_dim)
            
        Returns:
            Tensor: Output tensor of shape (batch_size, linker_dims[-1], model_dims[-1])
        """
        for layer in self.layers:
            x = layer(x)
        return x
    
    def describe(self, vec_seq):
        """
        Compute descriptor vectors for a sequence
        
        Args:
            vec_seq (list or Tensor): Sequence of vectors (seq_len, input_dim)
            
        Returns:
            list: List of output vectors (linker_dims[-1], model_dims[-1])
        """
        # Convert to tensor and add batch dimension
        if not isinstance(vec_seq, torch.Tensor):
            vec_seq = torch.tensor(vec_seq, dtype=torch.float32, device=self.device)
        # Move to same device as model
        x = vec_seq.unsqueeze(0)  # (1, seq_len, input_dim)
        
        # Forward pass
        with torch.no_grad():
            output = self.forward(x)
            
        # Remove batch dimension and convert to list of vectors
        return output.squeeze(0).tolist()      
    
    def deviation(self, seqs, t_list):
        """
        Compute mean squared error loss
        
        Args:
            seqs (list): List of sequences, each is (input_seq_len, input_dim)
            t_list (list): List of target vectors (model_dims[-1])
            
        Returns:
            Tensor: Mean squared error loss
        """
        # Convert to tensors and move to device
        seqs_tensor = torch.tensor(np.array(seqs), dtype=torch.float32, device=self.device)
        targets = torch.tensor(np.array(t_list), dtype=torch.float32, device=self.device)
        
        # Forward pass
        outputs = self.forward(seqs_tensor)  # (batch_size, out_seq, out_dim)
        
        # Compute average vector for each sequence
        avg_vectors = outputs.mean(dim=1)  # (batch_size, out_dim)
        
        # Compute MSE loss
        loss = nn.functional.mse_loss(avg_vectors, targets)
        return loss   
    
    def reg_train(self, seqs, t_list, max_iters=1000, tol=1e-8, 
                   learning_rate=0.01, decay_rate=1.0, print_every=10,
                   continued=False, checkpoint_file=None, checkpoint_interval=10,
                   batch_size=256):
        """
        Train the model using Adam optimizer with batch processing and checkpoint support
        
        Args:
            seqs (list): Training sequences
            t_list (list): Target vectors
            max_iters (int): Maximum training iterations
            tol (float): Tolerance for convergence
            learning_rate (float): Initial learning rate
            decay_rate (float): Learning rate decay rate
            print_every (int): Print progress every n iterations
            continued (bool): Whether to continue from existing parameters
            checkpoint_file (str): File to save/load checkpoints
            checkpoint_interval (int): Save checkpoint every n iterations
            batch_size (int): Batch size for training
            
        Returns:
            list: Training loss history
        """
        # Load checkpoint if continuing
        start_iter = 0
        best_loss = float('inf')
        best_model_state = None
        history = []
        
        if continued and checkpoint_file and os.path.exists(checkpoint_file):
            checkpoint = torch.load(checkpoint_file, map_location=self.device, weights_only=False)
            self.load_state_dict(checkpoint['model_state_dict'])
            optimizer_state = checkpoint['optimizer_state_dict']
            scheduler_state = checkpoint['scheduler_state_dict']
            history = checkpoint['history']
            start_iter = checkpoint['iteration'] + 1
            best_loss = checkpoint.get('best_loss', float('inf'))
            print(f"Resumed training from checkpoint at iteration {start_iter}, best loss: {best_loss:.6e}")
        
        # Convert data to tensors
        seqs_tensor = torch.tensor(np.array(seqs), dtype=torch.float32, device=self.device)
        targets = torch.tensor(np.array(t_list), dtype=torch.float32, device=self.device)
        num_samples = len(seqs)
        
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)
        
        # Load optimizer state if continuing
        if continued and checkpoint_file and os.path.exists(checkpoint_file):
            optimizer.load_state_dict(optimizer_state)
            scheduler.load_state_dict(scheduler_state)
        
        for it in range(start_iter, max_iters):
            total_loss = 0.0
            
            # Shuffle indices
            perm = torch.randperm(num_samples)
            
            for i in range(0, num_samples, batch_size):
                # Get batch indices
                batch_idx = perm[i:i+batch_size]
                batch_seqs = seqs_tensor[batch_idx]
                batch_targets = targets[batch_idx]
                
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.forward(batch_seqs)
                avg_vectors = outputs.mean(dim=1)
                
                # Compute loss
                loss = nn.functional.mse_loss(avg_vectors, batch_targets)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                
                # Update parameters
                optimizer.step()
                
                total_loss += loss.item() * len(batch_idx)
            
            # Average loss
            avg_loss = total_loss / num_samples
            history.append(avg_loss)
            
            # Update best loss
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_model_state = copy.deepcopy(self.state_dict())
            
            # Print progress
            if it % print_every == 0 or it == max_iters - 1:
                lr = optimizer.param_groups[0]['lr']
                print(f"GD Iter {it:3d}: Loss = {avg_loss:.6e}, LR = {lr:.6e}")
            
            # Save checkpoint
            if checkpoint_file and (it % checkpoint_interval == 0 or it == max_iters - 1):
                self._save_checkpoint(checkpoint_file, it, history, optimizer, scheduler, best_loss)
            
            # Check convergence
            if len(history) > 1 and abs(history[-2] - avg_loss) < tol:
                print(f"Converged after {it+1} iterations")
                break
            
            # Decay learning rate
            scheduler.step()
        
        # Restore best model if available
        if best_model_state is not None and best_loss < history[-1]:
            self.load_state_dict(best_model_state)
            print(f"Restored best model state with loss = {best_loss:.6e}")
            history[-1] = best_loss
        
        self.trained = True
        return history
    
    def _save_checkpoint(self, checkpoint_file, iteration, history, optimizer, scheduler, best_loss):
        """
        Save training checkpoint
        
        Args:
            checkpoint_file (str): File to save checkpoint
            iteration (int): Current iteration
            history (list): Training loss history
            optimizer: Optimizer object
            scheduler: Learning rate scheduler
            best_loss (float): Best loss so far
        """
        checkpoint = {
            'iteration': iteration,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'history': history,
            'best_loss': best_loss,
            'config': {
                'input_dim': self.input_dim,
                'model_dims': self.model_dims,
                'basis_dims': self.basis_dims,
                'input_seq_len': self.input_seq_len,
                'linker_dims': self.linker_dims,
                'linker_trainable': self.linker_trainable,
                'residual_mode_list': self.residual_mode_list,
                'device': str(self.device)
            },
            'training_stats': {
                'mean_t': self.mean_t if hasattr(self, 'mean_t') else None
            }
        }
        torch.save(checkpoint, checkpoint_file)

    def predict_t(self, vec_seq):
        """
        Predict target vector as average of all final layer outputs
        
        Args:
            vec_seq (list or Tensor): Sequence of vectors (seq_len, input_dim)
            
        Returns:
            list: Average output vector (model_dims[-1])
        """
        outputs = self.describe(vec_seq)
        if not outputs:
            return [0.0] * self.model_dims[-1]
        
        # Compute average vector
        return np.mean(outputs, axis=0).tolist()

    def self_train(self, seqs, max_iters=100, tol=1e-6, learning_rate=0.01, 
                   continued=False, self_mode='gap', decay_rate=1.0, print_every=10,
                   batch_size=256, checkpoint_file=None, checkpoint_interval=5):
        """
        Self-training for the ENTIRE hierarchical model with two modes:
          - 'gap': Predicts current input vector (self-consistency)
          - 'reg': Predicts next input vector (auto-regressive) with causal masking
        
        Now trains ALL layers of the hierarchical model, not just the first layer.
        Stores statistical information for reconstruction and generation.
        
        Args:
            seqs: List of input sequences
            max_iters: Maximum training iterations
            tol: Convergence tolerance
            learning_rate: Initial learning rate
            continued: Whether to continue from existing parameters
            self_mode: Training mode ('gap' or 'reg')
            decay_rate: Learning rate decay rate
            print_every: Print interval
            batch_size: Batch size for training
            checkpoint_file: File to save/load checkpoints
            checkpoint_interval: Save checkpoint every n iterations
        
        Returns:
            Training loss history
        """
        if self_mode not in ('gap', 'reg'):
            raise ValueError("self_mode must be either 'gap' or 'reg'")
        
        # Validate input sequences
        for seq in seqs:
            if len(seq) != self.input_seq_len:
                raise ValueError(f"All sequences must have length {self.input_seq_len}")
        
        # Load checkpoint if continuing
        start_iter = 0
        best_loss = float('inf')
        best_model_state = None
        history = []
        
        if continued and checkpoint_file and os.path.exists(checkpoint_file):
            checkpoint = torch.load(checkpoint_file, map_location=self.device, weights_only=False)
            self.load_state_dict(checkpoint['model_state_dict'])
            optimizer_state = checkpoint['optimizer_state_dict']
            scheduler_state = checkpoint['scheduler_state_dict']
            history = checkpoint['history']
            start_iter = checkpoint['iteration'] + 1
            best_loss = checkpoint.get('best_loss', float('inf'))
            print(f"Resumed self-training from checkpoint at iteration {start_iter}, best loss: {best_loss:.6f}")
        
        # Convert sequences to tensors
        seqs_tensor = torch.tensor(np.array(seqs), dtype=torch.float32, device=self.device)
        num_samples = len(seqs)
        
        # Initialize all layers if not continuing training
        if not continued:
            for layer in self.layers:
                # Reinitialize parameters with small random values
                nn.init.uniform_(layer.M, -0.5, 0.5)
                nn.init.uniform_(layer.Acoeff, -0.1, 0.1)
                nn.init.uniform_(layer.Bbasis, -0.1, 0.1)
                nn.init.uniform_(layer.Linker, -0.1, 0.1)
                
                # Reinitialize residual components if using 'separate' mode
                if layer.residual_mode == 'separate':
                    nn.init.uniform_(layer.residual_proj.weight, -0.1, 0.1)
                    if hasattr(layer, 'residual_linker'):
                        nn.init.uniform_(layer.residual_linker, -0.1, 0.1)
            
            # Calculate global mean vector of input sequences
            total = torch.zeros(self.input_dim, device=self.device)
            total_vectors = 0
            for seq in seqs_tensor:
                total += torch.sum(seq, dim=0)
                total_vectors += seq.size(0)
            self.mean_t = (total / total_vectors).cpu().numpy()
        
        # Setup optimizer
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)
        
        # Load optimizer state if continuing
        if continued and checkpoint_file and os.path.exists(checkpoint_file):
            optimizer.load_state_dict(optimizer_state)
            scheduler.load_state_dict(scheduler_state)
        
        for it in range(start_iter, max_iters):
            total_loss = 0.0
            total_batches = 0  # Track number of processed batches
            
            # Shuffle indices
            perm = torch.randperm(num_samples)
            
            for i in range(0, num_samples, batch_size):
                # Get batch indices
                batch_idx = perm[i:i+batch_size]
                batch_seqs = seqs_tensor[batch_idx]
                
                optimizer.zero_grad()
                
                # Apply causal masking to Linker matrices in reg mode
                original_linkers = []
                original_residual_linkers = []
                
                if self_mode == 'reg':
                    for layer in self.layers:
                        # Create causal mask for main Linker
                        causal_mask = torch.tril(torch.ones_like(layer.Linker))
                        original_linker = layer.Linker.clone()
                        layer.Linker.data = layer.Linker.data * causal_mask
                        original_linkers.append(original_linker)
                        
                        # Create causal mask for residual Linker if exists
                        if hasattr(layer, 'residual_linker') and layer.residual_linker is not None:
                            causal_mask_residual = torch.tril(torch.ones_like(layer.residual_linker))
                            original_residual_linker = layer.residual_linker.clone()
                            layer.residual_linker.data = layer.residual_linker.data * causal_mask_residual
                            original_residual_linkers.append(original_residual_linker)
                        else:
                            original_residual_linkers.append(None)
                
                # Forward pass through all layers
                x = batch_seqs
                for layer in self.layers:
                    x = layer(x)
                
                # Restore original Linker matrices after forward pass in reg mode
                if self_mode == 'reg':
                    for i, layer in enumerate(self.layers):
                        layer.Linker.data = original_linkers[i]
                        if original_residual_linkers[i] is not None:
                            layer.residual_linker.data = original_residual_linkers[i]
                
                # Calculate loss based on self_mode
                loss = 0.0
                
                # Vectorized loss calculation
                if self_mode == 'gap':
                    # For gap mode, we need to compare hierarchical output with original input
                    # Check if output dimensions match input dimensions
                    if x.shape[1:] == batch_seqs.shape[1:]:
                        # Direct comparison when dimensions match
                        loss = nn.functional.mse_loss(x, batch_seqs)
                    else:
                        # When dimensions don't match, we need to handle differently
                        # For demonstration, we'll compute loss on overlapping dimensions
                        min_seq_len = min(x.shape[1], batch_seqs.shape[1])
                        min_dim = min(x.shape[2], batch_seqs.shape[2])
                        
                        if min_seq_len > 0 and min_dim > 0:
                            # Compare only overlapping parts
                            pred_slice = x[:, :min_seq_len, :min_dim]
                            target_slice = batch_seqs[:, :min_seq_len, :min_dim]
                            loss = nn.functional.mse_loss(pred_slice, target_slice)
                        else:
                            # Skip this batch if no overlap
                            continue
                else:  # 'reg' mode
                    # Predict next vector: Hier[k] vs Input[k+1]
                    # We need at least 2 positions for meaningful comparison
                    if batch_seqs.shape[1] > 1 and x.shape[1] >= 1:
                        # Targets: Input from position 1 to end
                        targets = batch_seqs[:, 1:, :]
                        
                        # Predictions: Hierarchical output from position 0 to end-1
                        # We need to ensure we have enough output positions
                        max_pred_len = min(x.shape[1], targets.shape[1])
                        
                        if max_pred_len > 0:
                            preds = x[:, :max_pred_len, :]
                            targets_slice = targets[:, :max_pred_len, :]
                            loss = nn.functional.mse_loss(preds, targets_slice)
                        else:
                            continue
                    else:
                        # Skip this batch if not enough positions
                        continue
                
                # Only proceed if we have a valid loss
                if loss.item() > 0:
                    # Backward pass and optimization
                    loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                    
                    # Update parameters
                    optimizer.step()
                    
                    total_loss += loss.item()
                    total_batches += 1
            
            # Calculate average loss only if we processed any batches
            if total_batches > 0:
                avg_loss = total_loss / total_batches
            else:
                # If no batches were processed, use a high loss to continue training
                avg_loss = 1.0  # Arbitrary high value to continue training
                print(f"Iteration {it}: No valid batches processed, using default loss")
            
            history.append(avg_loss)
            
            # Update best loss
            if avg_loss < best_loss and total_batches > 0:
                best_loss = avg_loss
                best_model_state = copy.deepcopy(self.state_dict())
            
            # Print progress
            if it % print_every == 0 or it == max_iters - 1:
                current_lr = scheduler.get_last_lr()[0]
                mode_display = "Gap" if self_mode == 'gap' else "Reg"
                print(f"SelfTrain({mode_display}) Iter {it:3d}: loss = {avg_loss:.6f}, LR = {current_lr:.6f}, Batches = {total_batches}")
            
            # Save checkpoint
            if checkpoint_file and (it % checkpoint_interval == 0 or it == max_iters - 1):
                self._save_checkpoint(checkpoint_file, it, history, optimizer, scheduler, best_loss)
            
            # Check convergence - only if we processed batches and loss is not default
            if len(history) > 1 and total_batches > 0 and avg_loss != 1.0:
                loss_change = abs(history[-2] - avg_loss)
                if loss_change < tol:
                    print(f"Converged after {it+1} iterations with loss change {loss_change:.6e}")
                    break
            
            # Update learning rate scheduler - only if optimizer.step() was called
            if total_batches > 0:
                scheduler.step()
        
        # Restore best model if available
        if best_model_state is not None and best_loss < history[-1] and total_batches > 0:
            self.load_state_dict(best_model_state)
            print(f"Restored best model state with loss = {best_loss:.6f}")
            history[-1] = best_loss
        
        self.trained = True
        return history
    
    def reconstruct(self, L, tau=0.0, discrete_mode=False, vocab_size=None):
        """
        Generate sequence of vectors with temperature-controlled randomness.
        Supports both continuous and discrete generation modes.
        
        Args:
            L (int): Number of vectors to generate
            tau (float): Temperature parameter
            discrete_mode (bool): If True, use discrete sampling
            vocab_size (int): Required for discrete mode
        
        Returns:
            Generated sequence as numpy array
        """
        assert self.trained and hasattr(self, 'mean_t'), "Model must be self-trained first"
        if tau < 0:
            raise ValueError("Temperature must be non-negative")
        if discrete_mode and vocab_size is None:
            raise ValueError("vocab_size must be specified for discrete mode")

        # Set model to evaluation mode
        self.eval()
        
        generated = []
        
        # Create initial sequence with proper shape and values
        current_seq = torch.normal(
            mean=0.0,  # Use float mean instead of tensor
            std=0.1,
            size=(self.input_seq_len, self.input_dim),
            device=self.device
        )
        # Add the learned mean vector to the initial sequence
        mean_tensor = torch.tensor(self.mean_t, device=self.device, dtype=torch.float32)
        current_seq = current_seq + mean_tensor.unsqueeze(0)  # Broadcast mean to all positions
        
        with torch.no_grad():
            for _ in range(L):
                # Forward pass through all layers
                x = current_seq.unsqueeze(0)  # Add batch dimension
                for layer in self.layers:
                    x = layer(x)
                
                # Get the last output vector
                output_vector = x.squeeze(0)[-1]  # Last position of last sequence
                
                if discrete_mode:
                    # Discrete generation mode
                    discrete_vector = []
                    for value in output_vector:
                        # Create logits and sample
                        logits = torch.full((vocab_size,), value.item(), device=self.device)
                        if tau > 0:
                            logits += torch.normal(0, tau, size=(vocab_size,), device=self.device)
                        
                        if tau == 0:
                            sampled_index = torch.argmax(logits)
                        else:
                            probs = torch.softmax(logits / tau, dim=0)
                            sampled_index = torch.multinomial(probs, 1)
                        
                        discrete_vector.append(sampled_index.item())
                    
                    output_vector = torch.tensor(discrete_vector, device=self.device).float()
                else:
                    # Continuous generation mode
                    if tau > 0:
                        noise = torch.normal(0, tau * torch.abs(output_vector) + 0.01)
                        output_vector = output_vector + noise
                
                generated.append(output_vector.cpu().numpy())
                
                # Update current sequence (shift window)
                current_seq = torch.cat([current_seq[1:], output_vector.unsqueeze(0)])
        
        return np.array(generated)
    
    def count_parameters(self):
        """
        Calculate and print the number of learnable parameters in table format
        """
        total_params = 0
        trainable_params = 0
        
        print("="*70)
        print(f"{'Component':<15} | {'Layer/Param':<25} | {'Count':<15} | {'Shape':<20}")
        print("-"*70)
        
        for l_idx, layer in enumerate(self.layers):
            in_dim = self.input_dim if l_idx == 0 else self.model_dims[l_idx-1]
            out_dim = self.model_dims[l_idx]
            basis_dim = self.basis_dims[l_idx]
            in_seq = self.input_seq_len if l_idx == 0 else self.linker_dims[l_idx-1]
            out_seq = self.linker_dims[l_idx]
            
            print(f"Layer {l_idx:<13} | {'Config':<25} | {f'in={in_dim}, out={out_dim}':<15} | {f'L={basis_dim}, seq={in_seq}→{out_seq}':<20}")
            print("-"*70)
            
            layer_params = 0
            layer_trainable = 0
            
            # Count parameters for each parameter in the layer
            for name, param in layer.named_parameters():
                if param.requires_grad:
                    num = param.numel()
                    layer_trainable += num
                    print(f"{'':<15} | {name:<25} | {num:<15} | {str(tuple(param.shape)):<20}")
                layer_params += param.numel()
            
            # Count buffers (non-trainable parameters)
            for name, buffer in layer.named_buffers():
                if buffer is not None:
                    num = buffer.numel()
                    print(f"{'':<15} | {name+'(buffer)':<25} | {num:<15} | {str(tuple(buffer.shape)):<20}")
            
            total_params += layer_params
            trainable_params += layer_trainable
            
            print(f"{'':<15} | {'TOTAL':<25} | {layer_params:<15} |")
            print(f"{'':<15} | {'Trainable':<25} | {layer_trainable:<15} |")
            print("-"*70)
        
        # Print summary
        print(f"{'SUMMARY':<15} | {'Total Parameters':<25} | {total_params:<15} |")
        print(f"{'':<15} | {'Trainable Parameters':<25} | {trainable_params:<15} |")
        print("="*70)
        
        return total_params, trainable_params
    
    def save(self, filename):
        """Save model state to file"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': {
                'input_dim': self.input_dim,
                'model_dims': self.model_dims,
                'basis_dims': self.basis_dims,
                'input_seq_len': self.input_seq_len,
                'linker_dims': self.linker_dims,
                'linker_trainable': self.linker_trainable,
                'residual_mode_list': self.residual_mode_list,
                'device': self.device
            },
            'mean_t': self.mean_t if hasattr(self, 'mean_t') else None            
        }, filename)
        print(f"Model saved to {filename}")
    
    @classmethod
    def load(cls, filename, device=None):
        """Load model from file"""
        # Device configuration
        device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        
        # Load checkpoint
        checkpoint = torch.load(filename, map_location=device, weights_only=False)
        config = checkpoint['config']
        
        # Create model instance
        model = cls(
            input_dim=config['input_dim'],
            model_dims=config['model_dims'],
            basis_dims=config['basis_dims'],
            input_seq_len=config['input_seq_len'],
            linker_dims=config['linker_dims'],
            linker_trainable=config['linker_trainable'],
            residual_mode_list=config.get('residual_mode_list', None),
            device=config.get('device', 'cuda')
        ).to(device)
        
        # Load state
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load additional attributes
        if 'mean_t' in checkpoint:
            model.mean_t = checkpoint['mean_t']
        
        model.trained = True
        print(f"Model loaded from {filename}")
        return model
    

# === Example Usage ===
if __name__ == "__main__":

    from statistics import correlation
    
    # Set random seeds for reproducibility
    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)
    
    # Configuration
    input_dim = 10          # Input vector dimension
    input_seq_len = 200     # Fixed input sequence length
    model_dims = [8, 6, 3]  # Output dimensions for each layer
    basis_dims = [100, 40, 20] # Basis dimensions for each layer
    linker_dims = [50, 20, 10] # Output sequence lengths for each layer
    seq_count = 100         # Number of training sequences
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Generate training data with fixed sequence length
    print("\nGenerating training data...")
    print(f"Input dim: {input_dim}, Seq len: {input_seq_len}")
    print(f"Layer dims: {model_dims}, Basis dims: {basis_dims}, Linker dims: {linker_dims}")
    
    seqs = []   # List of sequences
    t_list = [] # List of target vectors

    for i in range(seq_count):
        # Generate vector sequence with fixed length
        sequence = np.random.uniform(-1, 1, (input_seq_len, input_dim))
        seqs.append(sequence)
        
        # Generate random target vector
        target = np.random.uniform(-1, 1, model_dims[-1])
        t_list.append(target)
    
    # Test Case 1: Mixed residual strategies
    print("\n=== Test Case 1: Mixed Residual Strategies ===")
    model_mixed = HierDDLrn(
        input_dim=input_dim,
        model_dims=model_dims,
        basis_dims=basis_dims,
        linker_dims=linker_dims,
        input_seq_len=input_seq_len,
        linker_trainable=[True, False, True],
        residual_mode_list=['separate', 'shared', None],
        device=device
    )
    
    # Show parameter counts
    print("\nModel parameter counts (table format):")
    model_mixed.count_parameters()
    
    # Train model with batch processing and checkpoint
    print("\nTraining model with batch processing...")
    history = model_mixed.reg_train(
        seqs, 
        t_list,
        learning_rate=0.01,
        max_iters=100,
        decay_rate=0.99,
        print_every=10,
        batch_size=256,
        checkpoint_file='hierddlrn_gd_checkpoint.pth',
        checkpoint_interval=20
    )
    
    # Evaluate predictions
    pred_t_list = [model_mixed.predict_t(seq) for seq in seqs]
    print("\nPrediction correlations per dimension:")
    corrs = []
    for d in range(model_dims[-1]):
        actuals = [t[d] for t in t_list]
        preds = [p[d] for p in pred_t_list]
        corr = correlation(actuals, preds)
        corrs.append(corr)
        print(f"  Dim {d}: correlation = {corr:.4f}")
    
    print(f"Average correlation: {np.mean(corrs):.4f}")
    
    # Test Case 2: Self-training and reconstruction
    print("\n=== Test Case 2: Self-training and Reconstruction ===")
    
    # Create a fresh model for self-training
    model_self = HierDDLrn(
        input_dim=input_dim,
        model_dims=[10, 20, 10],
        basis_dims=basis_dims,
        linker_dims=linker_dims,
        input_seq_len=input_seq_len,
        linker_trainable=True,
        residual_mode_list='separate',
        device=device
    )
    
    # Self-train in gap mode with checkpoint
    print("\nSelf-training in gap mode...")
    self_history = model_self.self_train(
        seqs,
        max_iters=50,
        learning_rate=0.01,
        self_mode='gap',
        print_every=5,
        batch_size=32,
        checkpoint_file='hierddlrn_self_checkpoint.pth',
        checkpoint_interval=10
    )
    
    # Reconstruct new sequences
    print("\nReconstructing sequences...")
    reconstructed_seqs = model_self.reconstruct(
        L=20,
        tau=0.1,
        discrete_mode=False
    )
    
    print(f"Reconstructed {len(reconstructed_seqs)} sequences of length {len(reconstructed_seqs[0])}")
    print(f"First reconstructed vector: {reconstructed_seqs[0]}")
    
    # Test continued training
    print("\n=== Test Case 3: Continued Training ===")
    
    # Continue training from checkpoint
    print("\nContinuing self-training from checkpoint...")
    continued_history = model_self.self_train(
        seqs,
        max_iters=30,
        learning_rate=0.005,
        self_mode='gap',
        print_every=5,
        batch_size=32,
        continued=True,
        checkpoint_file='hierddlrn_self_checkpoint.pth',
        checkpoint_interval=10
    )
    
    print(f"Initial training had {len(self_history)} iterations")
    print(f"Continued training added {len(continued_history)} iterations")
    
    # Test Case 4: Save and load model
    print("\n=== Test Case 4: Save and Load Model ===")
    
    # Save model
    model_self.save("test_model.pth")
    
    # Load model
    loaded_model = HierDDLrn.load("test_model.pth", device=device)
    
    # Verify loaded model works
    test_pred = loaded_model.predict_t(seqs[0])
    print(f"Loaded model prediction: {test_pred[:3]}...")  # Show first 3 values
    
    # Compare parameter counts
    print("\nOriginal model parameters:")
    total_orig, trainable_orig = model_self.count_parameters()
    
    print("\nLoaded model parameters:")
    total_loaded, trainable_loaded = loaded_model.count_parameters()
    
    if total_orig == total_loaded and trainable_orig == trainable_loaded:
        print("✓ Parameter counts match between original and loaded models")
    else:
        print("✗ Parameter counts do not match")
    
    print("\n=== Hierarchical Vector Sequence Processing with Linker Matrices Demo Completed ===")
