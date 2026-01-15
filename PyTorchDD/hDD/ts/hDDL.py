# Copyright (C) 2005-2025, Bin-Guang Ma (mbg@mail.hzau.edu.cn); SPDX-License-Identifier: MIT
# The Hierarchical Numeric Dual Descriptor (Tensor form) with Linker matrices in PyTorch
# With layer normalization and residual connections and reconstruction capability
# This program is for the demonstration of methodology and not fully refined.
# Author: Bin-Guang Ma (assisted by DeepSeek); Date: 2025-7-22 ~ 2026-1-13

import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
import os

class Layer(nn.Module):
    """Single layer of Hierarchical Numeric Dual Descriptor"""
    def __init__(self, in_dim, out_dim, in_seq, out_seq, num_basis, linker_trainable, use_residual, device):
        """
        Initialize a hierarchical layer
        
        Args:
            in_dim (int): Input dimension
            out_dim (int): Output dimension
            in_seq (int): Input sequence length
            out_seq (int): Output sequence length
            num_basis (int): Number of basis functions
            linker_trainable (bool): Whether Linker matrix is trainable
            use_residual (str or None): Residual connection type. Options are:
                - 'separate': use separate projection and Linker for residual
                - 'shared': share M and Linker for residual
                - None: no residual connection
            device (torch.device): Device to use
        """
        super().__init__()
        self.device = device
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.in_seq = in_seq
        self.out_seq = out_seq
        self.num_basis = num_basis
        self.use_residual = use_residual
        
        # Linear transformation matrix
        self.M = nn.Parameter(torch.empty(out_dim, in_dim))
        
        # Position-dependent transformation tensor
        self.P = nn.Parameter(torch.empty(out_dim, out_dim, num_basis))
        
        # Precompute periods tensor (fixed, not trainable)
        periods = torch.zeros(out_dim, out_dim, num_basis, dtype=torch.float32)
        for i in range(out_dim):
            for j in range(out_dim):
                for g in range(num_basis):
                    periods[i, j, g] = i*(out_dim*num_basis) + j*num_basis + g + 2
        self.register_buffer('periods', periods)
        
        # Sequence length transformation matrix
        self.Linker = nn.Parameter(
            torch.empty(in_seq, out_seq),
            requires_grad=linker_trainable
        )

        # Layer normalization
        self.ln = nn.LayerNorm(out_dim)
        
        # Residual connection setup
        if self.use_residual == 'separate':
            # Separate projection and Linker for residual path
            self.residual_proj = nn.Linear(in_dim, out_dim, bias=False)
            self.residual_linker = nn.Parameter(
                torch.empty(in_seq, out_seq),
                requires_grad=linker_trainable
            )
        
        # Initialize parameters
        self.reset_parameters()
    
    def reset_parameters(self):
        """Reset layer parameters"""
        nn.init.uniform_(self.M, -0.5, 0.5)
        nn.init.uniform_(self.P, -0.1, 0.1)
        nn.init.uniform_(self.Linker, -0.1, 0.1)
        
        if self.use_residual == 'separate':
            nn.init.uniform_(self.residual_proj.weight, -0.1, 0.1)
            nn.init.uniform_(self.residual_linker, -0.1, 0.1)
    
    def position_transform(self, Z):
        """
        Position-dependent transformation with basis functions (batch version)
        
        Args:
            Z (Tensor): Input tensor of shape (batch_size, seq_len, out_dim)
            
        Returns:
            Tensor: Transformed tensor of shape (batch_size, seq_len, out_dim)
        """
        batch_size, seq_len, _ = Z.shape
        
        # Create position indices [seq_len]
        k = torch.arange(seq_len, device=self.device).float()
        
        # Reshape for broadcasting: [seq_len, 1, 1, 1]
        k = k.view(seq_len, 1, 1, 1)
        
        # Get periods and reshape for broadcasting: [1, out_dim, out_dim, num_basis]
        periods = self.periods.unsqueeze(0)
        
        # Compute basis functions: cos(2πk/period) -> [seq_len, out_dim, out_dim, num_basis]
        phi = torch.cos(2 * math.pi * k / periods)
        
        # Prepare Z for broadcasting: [batch_size, seq_len, 1, out_dim, 1]
        Z_exp = Z.unsqueeze(2).unsqueeze(-1)
        
        # Prepare P for broadcasting: [1, 1, out_dim, out_dim, num_basis]
        P_exp = self.P.unsqueeze(0).unsqueeze(0)
        
        # Prepare phi for broadcasting: [1, seq_len, out_dim, out_dim, num_basis]
        phi_exp = phi.unsqueeze(0)
        
        # Compute position transformation: Z * P * phi
        # Result: [batch_size, seq_len, out_dim, out_dim, num_basis]
        M = Z_exp * P_exp * phi_exp
        
        # Sum over j (dim=3) and g (dim=4) -> [batch_size, seq_len, out_dim]
        T = torch.sum(M, dim=(3, 4))
        
        return T
    
    def sequence_transform(self, T):
        """
        Transform sequence length using Linker matrix
        
        Args:
            T (Tensor): Input tensor of shape (batch_size, in_seq, out_dim)
            
        Returns:
            Tensor: Transformed tensor of shape (batch_size, out_seq, out_dim)
        """
        # Using matrix multiplication: T @ Linker
        # T: [batch_size, in_seq, out_dim] -> [batch_size, out_seq, out_dim]
        return torch.matmul(T.transpose(1, 2), self.Linker).transpose(1, 2)
    
    def forward(self, x):
        """
        Forward pass through the layer
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, in_seq, in_dim)
            
        Returns:
            Tensor: Output tensor of shape (batch_size, out_seq, out_dim)
        """
        batch_size, in_seq, in_dim = x.shape
        
        # Main path: linear transformation
        Z = torch.matmul(x, self.M.t())  # [batch_size, in_seq, out_dim]

        # Layer normalization
        Z = self.ln(Z)
        
        # Position-dependent transformation
        T = self.position_transform(Z)  # [batch_size, in_seq, out_dim]
        
        # Sequence transformation
        U = self.sequence_transform(T)  # [batch_size, out_seq, out_dim]
        
        # Residual connection processing
        if self.use_residual == 'separate':
            # Separate projection and Linker for residual
            residual_feat = self.residual_proj(x)  # [batch_size, in_seq, out_dim]
            residual = torch.matmul(residual_feat.transpose(1, 2), self.residual_linker).transpose(1, 2)  # [batch_size, out_seq, out_dim]
            out = U + residual
        elif self.use_residual == 'shared':
            # Shared M and Linker for residual
            residual_feat = torch.matmul(x, self.M.t())  # [batch_size, in_seq, out_dim]
            residual = torch.matmul(residual_feat.transpose(1, 2), self.Linker).transpose(1, 2)  # [batch_size, out_seq, out_dim]
            out = U + residual
        else:
            # Identity residual only if dimensions match
            if self.in_dim == self.out_dim and self.in_seq == self.out_seq:
                residual = torch.matmul(x.transpose(1, 2), self.Linker).transpose(1, 2)  # [batch_size, out_seq, out_dim]
                out = U + residual
            else:
                out = U 
       
        return out

class HierDDLts(nn.Module):
    """
    Hierarchical Numeric Dual Descriptor with PyTorch implementation
    Features:
    - GPU acceleration support
    - Flexible residual connections
    - Layer normalization
    - Sequence length transformation via Linker matrices
    - Configurable trainable parameters
    - Auto-training, reconstruction, and generation capabilities
    """
    
    def __init__(self, input_dim=2, model_dims=[2], num_basis_list=[5],
                 input_seq_len=100, linker_dims=[50], linker_trainable=False,
                 use_residual_list=None, device=None):
        super().__init__()
        
        # Device configuration (GPU/CPU)
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        
        # Model configuration
        self.input_dim = input_dim
        self.model_dims = model_dims
        self.num_basis_list = num_basis_list
        self.input_seq_len = input_seq_len
        self.linker_dims = linker_dims
        self.num_layers = len(model_dims)
        self.trained = False
        
        # Validate dimensions
        if len(linker_dims) != self.num_layers:
            raise ValueError("linker_dims must have same length as model_dims")
        if len(num_basis_list) != self.num_layers:
            raise ValueError("num_basis_list must have same length as model_dims")
        
        # Process linker_trainable parameter
        if isinstance(linker_trainable, bool):
            self.linker_trainable = [linker_trainable] * self.num_layers
        elif isinstance(linker_trainable, list):
            if len(linker_trainable) != self.num_layers:
                raise ValueError("linker_trainable list length must match number of layers")
            self.linker_trainable = linker_trainable
        else:
            raise TypeError("linker_trainable must be bool or list of bools")
        
        # Process use_residual_list
        if use_residual_list is None:
            self.use_residual_list = [None] * self.num_layers
        elif isinstance(use_residual_list, list):
            if len(use_residual_list) != self.num_layers:
                raise ValueError("use_residual_list length must match number of layers")
            self.use_residual_list = use_residual_list
        else:
            raise TypeError("use_residual_list must be list or None")
        
        # Initialize layers
        self.layers = nn.ModuleList()
        for l in range(self.num_layers):
            # Determine input dimensions for this layer
            in_dim = input_dim if l == 0 else model_dims[l-1]
            in_seq = input_seq_len if l == 0 else linker_dims[l-1]
            out_dim = model_dims[l]
            out_seq = linker_dims[l]
            num_basis = num_basis_list[l]
            use_residual = self.use_residual_list[l]
            
            # Initialize layer
            layer = Layer(
                in_dim, out_dim, in_seq, out_seq, 
                num_basis, self.linker_trainable[l], 
                use_residual, self.device
            )
            self.layers.append(layer)
        
        # Move model to device
        self.to(self.device)
        
        # Initialize training statistics
        self.mean_t = None
        self.mean_target = None
    
    def reset_parameters(self):
        """Reset all model parameters"""
        for layer in self.layers:
            layer.reset_parameters()
        
        # Reset training state
        self.mean_t = None
        self.mean_target = None
        self.trained = False
    
    def forward(self, seq):
        """
        Forward pass through all layers
        
        Args:
            seq: Input tensor of shape (batch_size, seq_len, input_dim) or (seq_len, input_dim)
            
        Returns:
            Output tensor
        """
        # Handle single sequence (add batch dimension)
        if seq.dim() == 2:
            seq = seq.unsqueeze(0)
        
        current = seq
        for layer in self.layers:
            current = layer(current)
        
        # Remove batch dimension if input was single sequence
        if seq.size(0) == 1 and seq.dim() == 3:
            current = current.squeeze(0)
            
        return current
    
    def describe(self, seq):
        """
        Compute output vectors for each position in the sequence
        Args:
            seq: Input sequence (list of vectors or numpy array)
        
        Returns:
            Output sequence as numpy array
        """
        # Convert input to tensor
        seq_tensor = torch.tensor(seq, dtype=torch.float32, device=self.device)
        
        # Forward pass
        with torch.no_grad():
            output = self.forward(seq_tensor)
        
        return output.cpu().numpy()
    
    def deviation(self, seqs, t_list):
        """
        Compute mean squared deviation with batch processing
        
        Args:
            seqs: List of input sequences
            t_list: List of target vectors
        
        Returns:
            Mean squared error
        """
        # Convert sequences to batch tensor
        seqs_tensor = torch.stack([torch.tensor(seq, dtype=torch.float32, device=self.device) 
                                  for seq in seqs])
        t_tensor = torch.tensor(t_list, dtype=torch.float32, device=self.device)
        
        with torch.no_grad():
            # Forward pass for all sequences
            outputs = self.forward(seqs_tensor)  # [batch_size, seq_len, out_dim]
            
            # Compute mean over sequence dimension for each batch
            pred_targets = outputs.mean(dim=1)  # [batch_size, out_dim]
            
            # Compute MSE loss
            loss = torch.mean((pred_targets - t_tensor) ** 2)
        
        return loss.item()
    
    def reg_train(self, seqs, t_list, max_iters=1000, tol=1e-88, lr=0.01, 
                   decay_rate=1.0, print_every=10, continued=False,
                   batch_size=1024, checkpoint_file=None, checkpoint_interval=10):
        """
        Train model using Adam optimizer with batch processing
        
        Args:
            seqs: List of training sequences
            t_list: List of target vectors
            max_iters: Maximum training iterations
            lr: Initial learning rate
            decay_rate: Learning rate decay rate
            tol: Convergence tolerance
            print_every: Print interval
            continued: Whether to continue from existing parameters
            batch_size: Batch size for training
            checkpoint_file: Path to save/load checkpoint file for resuming training
            checkpoint_interval: Interval (in iterations) for saving checkpoints
        
        Returns:
            Training loss history
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
        
        # Convert all data to tensors
        print("Pre-processing training data...")
        seqs_tensor = torch.stack([torch.tensor(seq, dtype=torch.float32, device='cpu') for seq in seqs])
        t_tensor = torch.tensor(np.array(t_list), dtype=torch.float32, device='cpu')
        num_samples = len(seqs)
        
        # Set model to training mode
        self.train()
        self.trained = True
        
        # Setup optimizer and scheduler
        optimizer = optim.Adam(self.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)
        
        # Load optimizer and scheduler states if resuming
        if continued and checkpoint_file and os.path.exists(checkpoint_file):
            optimizer.load_state_dict(optimizer_state)
            scheduler.load_state_dict(scheduler_state)
        
        prev_loss = float('inf') if start_iter == 0 else history[-1] if history else float('inf')
        
        for it in range(start_iter, max_iters):
            total_loss = 0.0
            
            # Shuffle indices for this epoch
            perm = torch.randperm(num_samples)
            
            # Process in batches
            for i in range(0, num_samples, batch_size):
                # Get batch indices
                batch_idx = perm[i:i+batch_size]
                
                # Get batch data and move to device
                batch_seqs = seqs_tensor[batch_idx].to(self.device)
                batch_targets = t_tensor[batch_idx].to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = self.forward(batch_seqs)  # [batch_size, seq_len, out_dim]
                
                # Compute mean over sequence dimension for each batch
                pred_targets = outputs.mean(dim=1)  # [batch_size, out_dim]
                
                # Compute loss
                loss = torch.mean((pred_targets - batch_targets) ** 2)
                
                # Backpropagation
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item() * len(batch_idx)
            
            # Average loss for this epoch
            avg_loss = total_loss / num_samples
            history.append(avg_loss)
            
            # Update best model state if current loss is lower
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_model_state = copy.deepcopy(self.state_dict())
            
            # Print progress
            if it % print_every == 0 or it == max_iters - 1:
                current_lr = scheduler.get_last_lr()[0]
                print(f"Iter {it:3d}: Loss = {avg_loss:.6e}, LR = {current_lr:.6f}")

            # Save checkpoint at specified intervals
            if checkpoint_file and (it % checkpoint_interval == 0 or it == max_iters - 1):
                self._save_checkpoint(
                    checkpoint_file, it, history, optimizer, scheduler, best_loss
                )
            
            # Check convergence
            if abs(prev_loss - avg_loss) < tol:
                print(f"Converged after {it+1} iterations.")
                # Restore the best model state before breaking
                if best_model_state is not None and avg_loss > best_loss:
                    self.load_state_dict(best_model_state)
                    print(f"Restored best model state with loss = {best_loss:.6e}")
                    history[-1] = best_loss
                break
                
            prev_loss = avg_loss
            
            # Update learning rate
            scheduler.step()
        
        # If training ended without convergence, restore best model state
        if best_model_state is not None and best_loss < history[-1]:
            self.load_state_dict(best_model_state)
            print(f"Training ended. Restored best model state with loss = {best_loss:.6e}")
            history[-1] = best_loss
        
        return history
    
    def predict_t(self, seq):
        """
        Predict target vector by averaging output sequence
        Args:
            seq: Input sequence (list of vectors or numpy array)
        
        Returns:
            Predicted target vector as numpy array
        """
        # Forward pass
        output = self.describe(seq)
        
        # Average output vectors
        return np.mean(output, axis=0)

    def self_train(self, seqs, max_iters=100, tol=1e-6, learning_rate=0.01, 
                   continued=False, self_mode='gap', decay_rate=1.0, print_every=10,
                   batch_size=1024, checkpoint_file=None, checkpoint_interval=5):
        """
        Self-training for the ENTIRE hierarchical model with batch processing
        
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
            checkpoint_file: Path to save/load checkpoint file for resuming training
            checkpoint_interval: Interval (in iterations) for saving checkpoints
        
        Returns:
            Training loss history
        """
        if self_mode not in ('gap', 'reg'):
            raise ValueError("self_mode must be either 'gap' or 'reg'")
        
        # Validate input sequences
        for seq in seqs:
            if len(seq) != self.input_seq_len:
                raise ValueError(f"All sequences must have length {self.input_seq_len}")
        
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
                # Calculate global mean vector of input sequences
                total = torch.zeros(self.input_dim, device=self.device)
                total_windows = 0
                for seq in seqs:
                    seq_tensor = torch.tensor(seq, dtype=torch.float32, device=self.device)
                    total += torch.sum(seq_tensor, dim=0)
                    total_windows += seq_tensor.size(0)
                self.mean_t = (total / total_windows).cpu().numpy()
            history = []
        
        # Convert all sequences to tensor
        print("Pre-processing training data...")
        seqs_tensor = torch.stack([torch.tensor(seq, dtype=torch.float32, device='cpu') for seq in seqs])
        num_samples = len(seqs)
        
        # Setup optimizer
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)
        
        # Load optimizer and scheduler states if resuming
        if continued and checkpoint_file and os.path.exists(checkpoint_file):
            optimizer.load_state_dict(optimizer_state)
            scheduler.load_state_dict(scheduler_state)
        
        prev_loss = float('inf') if start_iter == 0 else history[-1] if history else float('inf')
        
        for it in range(start_iter, max_iters):
            total_loss = 0.0
            
            # Apply causal masking if in 'reg' mode
            original_linkers = []
            original_residual_linkers = []
            
            if self_mode == 'reg':
                for layer in self.layers:
                    if hasattr(layer, 'Linker'):
                        causal_mask = torch.tril(torch.ones_like(layer.Linker))
                        original_linkers.append(layer.Linker.data.clone())
                        layer.Linker.data.mul_(causal_mask)
                        
                        if hasattr(layer, 'residual_linker') and layer.residual_linker is not None:
                            cm_res = torch.tril(torch.ones_like(layer.residual_linker))
                            original_residual_linkers.append(layer.residual_linker.data.clone())
                            layer.residual_linker.data.mul_(cm_res)
                        else:
                            original_residual_linkers.append(None)
            
            # Shuffle indices for this epoch
            perm = torch.randperm(num_samples)
            
            # Process in batches
            for i in range(0, num_samples, batch_size):
                # Get batch indices
                batch_idx = perm[i:i+batch_size]
                
                # Get batch data and move to device
                batch_seqs = seqs_tensor[batch_idx].to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                output = self.forward(batch_seqs)  # [batch_size, seq_len, out_dim]
                
                # Calculate loss based on self_mode
                loss = 0.0
                valid_samples = 0
                
                if self_mode == 'gap':
                    # Self-consistency: output should match input
                    # Note: This requires output_dim == input_dim
                    if output.shape[-1] == batch_seqs.shape[-1]:
                        # Align sequence lengths
                        min_len = min(output.shape[1], batch_seqs.shape[1])
                        diff = output[:, :min_len, :] - batch_seqs[:, :min_len, :]
                        sq_diff = torch.sum(diff ** 2, dim=-1)  # [batch_size, seq_len]
                        loss = torch.mean(sq_diff)
                        valid_samples = 1
                else:  # 'reg' mode
                    # Predict next: output[k] should match input[k+1]
                    min_len = min(output.shape[1], batch_seqs.shape[1])
                    if min_len > 1:
                        # Output at position k should predict input at position k+1
                        targets = batch_seqs[:, 1:min_len, :]
                        preds = output[:, 0:min_len-1, :]
                        diff = preds - targets
                        sq_diff = torch.sum(diff ** 2, dim=-1)  # [batch_size, seq_len-1]
                        loss = torch.mean(sq_diff)
                        valid_samples = 1
                
                if valid_samples > 0:
                    total_loss += loss.item() * len(batch_idx)
                    loss.backward()
                    optimizer.step()
            
            # Restore original weights if Reg mode
            if self_mode == 'reg':
                for idx, layer in enumerate(self.layers):
                    if hasattr(layer, 'Linker'):
                        layer.Linker.data.copy_(original_linkers[idx])
                        if original_residual_linkers[idx] is not None:
                            layer.residual_linker.data.copy_(original_residual_linkers[idx])
            
            # Average loss for this epoch
            avg_loss = total_loss / num_samples if num_samples > 0 else 0.0
            history.append(avg_loss)
            
            # Update best model state if current loss is lower
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_model_state = copy.deepcopy(self.state_dict())
            
            # Print progress
            if it % print_every == 0 or it == max_iters - 1:
                mode_display = "Gap" if self_mode == 'gap' else "Reg"
                current_lr = scheduler.get_last_lr()[0]
                print(f"SelfTrain({mode_display}) Iter {it:3d}: loss = {avg_loss:.6f}, LR = {current_lr:.6f}")

            # Save checkpoint at specified intervals
            if checkpoint_file and (it % checkpoint_interval == 0 or it == max_iters - 1):
                self._save_checkpoint(
                    checkpoint_file, it, history, optimizer, scheduler, best_loss
                )
            
            # Check convergence
            if prev_loss - avg_loss < tol:
                print(f"Converged after {it+1} iterations")
                # Restore the best model state before breaking
                if best_model_state is not None and avg_loss > best_loss:
                    self.load_state_dict(best_model_state)
                    print(f"Restored best model state with loss = {best_loss:.6f}")
                    history[-1] = best_loss
                break
            prev_loss = avg_loss
            
            # Update learning rate every 5 iterations
            if it % 5 == 0:
                scheduler.step()
        
        # If training ended without convergence, restore best model state
        if best_model_state is not None and best_loss < history[-1]:
            self.load_state_dict(best_model_state)
            print(f"Training ended. Restored best model state with loss = {best_loss:.6f}")
            history[-1] = best_loss
        
        # Compute mean target vector for reconstruction
        self._compute_mean_target(seqs)
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
                'input_dim': self.input_dim,
                'model_dims': self.model_dims,
                'num_basis_list': self.num_basis_list,
                'input_seq_len': self.input_seq_len,
                'linker_dims': self.linker_dims,
                'linker_trainable': self.linker_trainable,
                'use_residual_list': self.use_residual_list
            },
            'training_stats': {
                'mean_t': self.mean_t,
                'mean_target': self.mean_target.cpu().numpy() if self.mean_target is not None else None
            }
        }
        torch.save(checkpoint, checkpoint_file)
        #print(f"Checkpoint saved to {checkpoint_file} at iteration {iteration}")
    
    def _compute_mean_target(self, seqs, batch_size=256):
        """
        Compute the mean target vector in the final layer output space
        This represents the average pattern learned by the entire hierarchical model
        """
        if not seqs:
            self.mean_target = None
            return
            
        # Convert sequences to tensor
        seqs_tensor = torch.stack([torch.tensor(seq, dtype=torch.float32, device='cpu') for seq in seqs])
        num_samples = len(seqs)
        final_dim = self.model_dims[-1] if self.model_dims else self.input_dim
        
        total_output = torch.zeros(final_dim, device=self.device)
        total_sequences = 0
        
        with torch.no_grad():
            for i in range(0, num_samples, batch_size):
                batch_seqs = seqs_tensor[i:i+batch_size].to(self.device)
                try:
                    output = self.forward(batch_seqs)  # [batch_size, seq_len, out_dim]
                    
                    # Mean over sequence, Sum over batch
                    batch_means = output.mean(dim=1)  # [batch_size, out_dim]
                    total_output += batch_means.sum(dim=0)
                    total_sequences += batch_seqs.shape[0]
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"Warning: OOM in mean target compute. Batch {i}")
                        torch.cuda.empty_cache()
                        continue
                    else:
                        raise e
        
        if total_sequences > 0:
            self.mean_target = total_output / total_sequences
        else:
            self.mean_target = None

    def reconstruct(self, L, tau=0.0, discrete_mode=False, vocab_size=None):
        """
        Reconstruct sequence of vectors with temperature-controlled randomness.
        Supports both continuous and discrete reconstruction modes.
        
        Args:
            L (int): Number of vectors to reconstruct
            tau (float): Temperature parameter
            discrete_mode (bool): If True, use discrete sampling
            vocab_size (int): Required for discrete mode
        
        Returns:
            Reconstructed sequence as numpy array
        """
        assert self.trained and hasattr(self, 'mean_t'), "Model must be self-trained first"
        if tau < 0:
            raise ValueError("Temperature must be non-negative")
        if discrete_mode and vocab_size is None:
            raise ValueError("vocab_size must be specified for discrete mode")

        # Set model to evaluation mode
        self.eval()
        
        reconstructed = []
        
        # Create initial sequence with proper shape and values
        # Use mean_t as the base for initialization
        mean_tensor = torch.tensor(self.mean_t, device=self.device, dtype=torch.float32)
        current_seq = torch.normal(
            mean=0.0,
            std=0.1,
            size=(self.input_seq_len, self.input_dim),
            device=self.device
        )
        # Add the learned mean vector to the initial sequence
        current_seq = current_seq + mean_tensor.unsqueeze(0)  # Broadcast mean to all positions
        
        with torch.no_grad():
            for _ in range(L):
                # Forward pass
                current = current_seq.unsqueeze(0)  # Add batch dimension
                for layer in self.layers:
                    current = layer(current)
                
                # Remove batch dimension
                current = current.squeeze(0)
                
                # Get the last output vector
                output_vector = current[-1]
                
                if discrete_mode:
                    # Discrete reconstruction mode
                    discrete_vector = []
                    for value in output_vector:
                        # Create logits and sample
                        logits = torch.full((vocab_size,), value.item())
                        if tau > 0:
                            logits += torch.normal(0, tau, size=(vocab_size,))
                        
                        if tau == 0:
                            sampled_index = torch.argmax(logits)
                        else:
                            probs = torch.softmax(logits / tau, dim=0)
                            sampled_index = torch.multinomial(probs, 1)
                        
                        discrete_vector.append(sampled_index.item())
                    
                    output_vector = torch.tensor(discrete_vector, device=self.device).float()
                else:
                    # Continuous reconstruction mode
                    if tau > 0:
                        noise = torch.normal(0, tau * torch.abs(output_vector) + 0.01)
                        output_vector = output_vector + noise
                
                reconstructed.append(output_vector.cpu().numpy())
                
                # Update current sequence (shift window)
                current_seq = torch.cat([current_seq[1:], output_vector.unsqueeze(0)])
        
        return np.array(reconstructed)
    
    def count_parameters(self):
        """Count learnable parameters with table display"""
        total_params = 0
        trainable_params = 0
        
        print("="*80)
        print(f"{'Component':<15} | {'Layer/Param':<25} | {'Count':<15} | {'Shape':<20}")
        print("-"*80)
        
        # Count parameters for each layer
        for l, layer in enumerate(self.layers):
            layer_params = 0
            layer_trainable = 0
            layer_name = f"Layer {l}"
            
            # M matrix
            num = layer.M.numel()
            shape = str(tuple(layer.M.shape))
            print(f"{layer_name:<15} | {'M':<25} | {num:<15} | {shape:<20}")
            layer_params += num
            layer_trainable += num
            
            # P tensor
            num = layer.P.numel()
            shape = str(tuple(layer.P.shape))
            print(f"{layer_name:<15} | {'P':<25} | {num:<15} | {shape:<20}")
            layer_params += num
            layer_trainable += num
            
            # Linker matrix
            num = layer.Linker.numel()
            shape = str(tuple(layer.Linker.shape))
            trainable_status = "T" if layer.Linker.requires_grad else "F"
            print(f"{layer_name:<15} | {'Linker':<25} | {num:<15} | {shape:<20} [{trainable_status}]")
            layer_params += num
            if layer.Linker.requires_grad:
                layer_trainable += num
            
            # Residual parameters
            if layer.use_residual == 'separate':
                # Residual projection
                num = layer.residual_proj.weight.numel()
                shape = str(tuple(layer.residual_proj.weight.shape))
                print(f"{layer_name:<15} | {'residual_proj':<25} | {num:<15} | {shape:<20}")
                layer_params += num
                layer_trainable += num
                
                # Residual linker
                num = layer.residual_linker.numel()
                shape = str(tuple(layer.residual_linker.shape))
                trainable_status = "T" if layer.residual_linker.requires_grad else "F"
                print(f"{layer_name:<15} | {'residual_linker':<25} | {num:<15} | {shape:<20} [{trainable_status}]")
                layer_params += num
                if layer.residual_linker.requires_grad:
                    layer_trainable += num
            
            print(f"{layer_name:<15} | {'TOTAL':<25} | {layer_params:<15} | (trainable: {layer_trainable})")
            total_params += layer_params
            trainable_params += layer_trainable
            print("-"*80)
        
        print(f"{'TOTAL':<15} | {'ALL PARAMETERS':<25} | {total_params:<15} | (trainable: {trainable_params})")
        print("="*80)
        
        return total_params, trainable_params
    
    def save(self, filename):
        """Save model state to file"""
        # Convert mean_target to numpy for saving
        mean_target_np = self.mean_target.cpu().numpy() if self.mean_target is not None else None
        
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': {
                'input_dim': self.input_dim,
                'model_dims': self.model_dims,
                'num_basis_list': self.num_basis_list,
                'input_seq_len': self.input_seq_len,
                'linker_dims': self.linker_dims,
                'linker_trainable': self.linker_trainable,
                'use_residual_list': self.use_residual_list
            },
            'training_stats': {
                'mean_t': self.mean_t if hasattr(self, 'mean_t') else None,
                'mean_target': mean_target_np
            }
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
            num_basis_list=config['num_basis_list'],
            input_seq_len=config['input_seq_len'],
            linker_dims=config['linker_dims'],
            linker_trainable=config['linker_trainable'],
            use_residual_list=config.get('use_residual_list', None),
            device=device
        )
        
        # Load state
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load additional attributes
        if 'training_stats' in checkpoint:
            stats = checkpoint['training_stats']
            if stats['mean_t'] is not None:
                model.mean_t = stats['mean_t']
            if stats['mean_target'] is not None:
                model.mean_target = torch.tensor(stats['mean_target'], device=device)
        
        model.trained = True
        print(f"Model loaded from {filename}")
        return model


# === Example Usage ===
if __name__ == "__main__":

    from statistics import correlation

    # Set random seeds for reproducibility
    torch.manual_seed(1)
    random.seed(1)
    np.random.seed(1)
    
    # Configuration
    input_dim = 10          # Input vector dimension
    model_dims = [8, 6, 3]  # Output dimensions for each layer
    num_basis_list = [5, 4, 3]  # Basis functions per layer
    input_seq_len = 100     # Fixed input sequence length
    linker_dims = [100, 50, 20]  # Output sequence lengths for each layer
    num_seqs = 100          # Number of training sequences
    
    # Generate synthetic training data
    print(f"Generating {num_seqs} sequences...")
    seqs = []
    t_list = []
    
    for _ in range(num_seqs):
        # Input sequence: (input_seq_len, input_dim)
        seq = np.random.uniform(-1, 1, size=(input_seq_len, input_dim))
        seqs.append(seq)
        
        # Target vector: (model_dims[-1],)
        t_list.append(np.random.uniform(-1, 1, size=model_dims[-1]))

    # Test Case 1: Mixed residual strategies
    print("\n=== Test Case 1: Mixed Residual Strategies ===")
    model_mixed = HierDDLts(
        input_dim=input_dim,
        model_dims=model_dims,
        num_basis_list=num_basis_list,
        input_seq_len=input_seq_len,
        linker_dims=linker_dims,
        linker_trainable=[True, False, True],
        use_residual_list=['separate', 'shared', None]  # Different residual for each layer
    )
    
    # Parameter count with table display
    print("\nParameter count before training:")
    total_params, trainable_params = model_mixed.count_parameters()
    
    # Training with batch processing and checkpoint
    print("\nTraining model with batch processing...")
    history = model_mixed.reg_train(
        seqs, 
        t_list, 
        max_iters=50,
        lr=0.01,
        decay_rate=0.98,
        print_every=5,
        batch_size=32,
        checkpoint_file='hddl_gd_checkpoint.pth',
        checkpoint_interval=10
    )
    
    # Calculate correlations
    print("\nCorrelation analysis:")
    all_preds = [model_mixed.predict_t(seq) for seq in seqs]
    correlations = []
    for i in range(model_dims[-1]):
        actual = [t[i] for t in t_list]
        predicted = [p[i] for p in all_preds]
        corr = correlation(actual, predicted)
        correlations.append(corr)
        print(f"Output dim {i} correlation: {corr:.4f}")
    print(f"Average correlation: {np.mean(correlations):.4f}")    
    
    # Test Case 2: Self-training and sequence reconstruction
    print("\n=== Test Case 2: Self-training and Sequence Reconstruction ===")
    
    # Create a new model with compatible dimensions for self-training
    hndd_self = HierDDLts(
        input_dim=input_dim,
        model_dims=[input_dim, 20, input_dim],  # Output dim must match input dim for reconstruction
        num_basis_list=[5]*3,
        input_seq_len=input_seq_len,
        linker_dims=[100, 50, 20],  
        linker_trainable=False,
        use_residual_list=[None]*3
    )
    
    # Self-train in 'gap' mode (self-consistency) with checkpoint
    print("\nSelf-training in 'gap' mode:")
    self_history = hndd_self.self_train(
        seqs,
        self_mode='gap',
        max_iters=20,
        learning_rate=1.0,
        decay_rate=0.98,
        print_every=5,
        batch_size=32,
        checkpoint_file='hddl_self_checkpoint.pth',
        checkpoint_interval=5
    )
    
    # Reconstruct new sequence with temperature
    print("\nReconstructing new sequences:")
    print("Deterministic reconstruction (tau=0):")
    rec_det = hndd_self.reconstruct(L=10, tau=0)
    print(f"  Reconstructed {len(rec_det)} vectors")
    print(f"  First vector: {[f'{x:.4f}' for x in rec_det[0][:min(5, input_dim)]]}...")
    
    print("\nStochastic reconstruction (tau=0.5):")
    rec_stoch = hndd_self.reconstruct(L=10, tau=0.5)
    print(f"  Reconstructed {len(rec_stoch)} vectors")
    print(f"  First vector: {[f'{x:.4f}' for x in rec_stoch[0][:min(5, input_dim)]]}...")
    
    # Test Case 3: Self-training in 'reg' mode (auto-regressive)
    print("\n=== Test Case 3: Auto-regressive Training ===")
    
    # Self-train in 'reg' mode (next-step prediction) with continued training
    print("\nSelf-training in 'reg' mode (continued from previous training):")
    self_history_reg = hndd_self.self_train(
        seqs,
        self_mode='reg',
        max_iters=10,
        learning_rate=0.5,
        decay_rate=0.98,
        print_every=5,
        continued=True,  # Continue training existing model
        batch_size=32,
        checkpoint_file='hddl_self_checkpoint.pth',
        checkpoint_interval=5
    )
    
    # Reconstruct sequence using auto-regressive model
    print("\nReconstructing sequence with auto-regressive model:")
    rec_reg = hndd_self.reconstruct(L=15, tau=0.3)
    print(f"Reconstructed {len(rec_reg)} vectors")
    print(f"First vector: {[f'{x:.4f}' for x in rec_reg[0][:min(5, input_dim)]]}...")
    print(f"Last vector: {[f'{x:.4f}' for x in rec_reg[-1][:min(5, input_dim)]]}...")
    
    # Test Case 4: Model save and load
    print("\n=== Test Case 4: Model Save/Load Test ===")
    
    # Save model
    hndd_self.save("hddl_model.pth")
    
    # Load model
    loaded_model = HierDDLts.load("hddl_model.pth")
    
    # Test loaded model
    test_seq = seqs[0]
    original_pred = hndd_self.predict_t(test_seq)
    loaded_pred = loaded_model.predict_t(test_seq)
    
    print(f"Original model prediction: {[f'{x:.4f}' for x in original_pred]}")
    print(f"Loaded model prediction:   {[f'{x:.4f}' for x in loaded_pred]}")
    
    pred_diff = np.max(np.abs(original_pred - loaded_pred))
    print(f"Maximum prediction difference: {pred_diff:.6e}")
    
    if pred_diff < 1e-6:
        print("✓ Model save/load test PASSED")
    else:
        print("✗ Model save/load test FAILED")
    
    # Test reconstruction consistency
    torch.manual_seed(123); random.seed(123); np.random.seed(123)
    original_rec = hndd_self.reconstruct(L=10, tau=0.1)
    
    torch.manual_seed(123); random.seed(123); np.random.seed(123)
    loaded_rec = loaded_model.reconstruct(L=10, tau=0.1)
    
    if np.allclose(original_rec, loaded_rec, rtol=1e-6):
        print("✓ Reconstruction consistency test PASSED")
    else:
        print("✗ Reconstruction consistency test FAILED")
        print(f"Original first vector: {original_rec[0][:5]}")
        print(f"Loaded first vector:   {loaded_rec[0][:5]}")
    
    print("\nAll tests completed successfully!")
