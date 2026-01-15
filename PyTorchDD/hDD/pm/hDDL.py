# Copyright (C) 2005-2025, Bin-Guang Ma (mbg@mail.hzau.edu.cn); SPDX-License-Identifier: MIT
# The Hierarchical Numeric Dual Descriptor (P Matrix form) with Linker matrices in PyTorch
# With layer normalization and residual connections and generation capability
# This program is for the demonstration of methodology and not fully refined.
# Author: Bin-Guang Ma (assisted by DeepSeek); Date: 2025-8-26 ~ 2026-1-13

import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import os

class Layer(nn.Module):
    """Single layer of Hierarchical Numeric Dual Descriptor"""
    def __init__(self, in_dim, out_dim, in_seq, out_seq, linker_trainable, use_residual, device):
        """
        Initialize a hierarchical layer
        
        Args:
            in_dim (int): Input dimension
            out_dim (int): Output dimension
            in_seq (int): Input sequence length
            out_seq (int): Output sequence length
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
        self.use_residual = use_residual
        
        # Linear transformation matrix
        self.M = nn.Parameter(torch.randn(out_dim, in_dim) * 0.5)
        
        # Position-dependent transformation matrix (simplified to 2D)
        self.P = nn.Parameter(torch.randn(out_dim, out_dim) * 0.1)
        
        # Precompute periods matrix (fixed, not trainable)
        periods = torch.zeros(out_dim, out_dim, dtype=torch.float32)
        for i in range(out_dim):
            for j in range(out_dim):
                # Simplified period calculation without basis dimension
                periods[i, j] = i * out_dim + j + 2
        self.register_buffer('periods', periods)
        
        # Sequence length transformation matrix
        self.Linker = nn.Parameter(
            torch.randn(in_seq, out_seq) * 0.1,
            requires_grad=linker_trainable
        )

        # Layer normalization
        self.ln = nn.LayerNorm(out_dim)
        
        # Residual connection setup
        if self.use_residual == 'separate':
            # Separate projection and Linker for residual path
            self.residual_proj = nn.Linear(in_dim, out_dim, bias=False)
            self.residual_linker = nn.Parameter(
                torch.randn(in_seq, out_seq) * 0.1,
                requires_grad=linker_trainable
            )
        
        # Initialize parameters
        self.reset_parameters()
    
    def reset_parameters(self):
        """Reset layer parameters to initial values"""
        nn.init.uniform_(self.M, -0.5, 0.5)
        nn.init.uniform_(self.P, -0.1, 0.1)
        nn.init.uniform_(self.Linker, -0.1, 0.1)
        if self.use_residual == 'separate':
            nn.init.uniform_(self.residual_proj.weight, -0.1, 0.1)
            if hasattr(self, 'residual_linker'):
                nn.init.uniform_(self.residual_linker, -0.1, 0.1)
    
    def position_transform(self, Z):
        """
        Position-dependent transformation with simplified matrix form
        Supports batch processing: Z shape [batch_size, seq_len, out_dim]
        """
        batch_size, seq_len, _ = Z.shape
        
        # Create position indices [1, seq_len, 1, 1]
        k = torch.arange(seq_len, device=self.device).float()
        k = k.view(1, seq_len, 1, 1)
        
        # Get periods and reshape for broadcasting: [1, 1, out_dim, out_dim]
        periods = self.periods.unsqueeze(0).unsqueeze(0)
        
        # Compute basis function: cos(2πk/period) -> [1, seq_len, out_dim, out_dim]
        phi = torch.cos(2 * math.pi * k / periods)
        
        # Prepare Z for broadcasting: [batch_size, seq_len, 1, out_dim]
        Z_exp = Z.unsqueeze(2)
        
        # Prepare P for broadcasting: [1, 1, out_dim, out_dim]
        P_exp = self.P.unsqueeze(0).unsqueeze(0)
        
        # Compute position transformation: Z * P * phi
        # Dimensions: 
        #   Z_exp: [batch_size, seq_len, 1, out_dim]
        #   P_exp: [1, 1, out_dim, out_dim]
        #   phi:   [1, seq_len, out_dim, out_dim]
        # Result: [batch_size, seq_len, out_dim, out_dim]
        M = Z_exp * P_exp * phi
        
        # Sum over j (dim=3) -> [batch_size, seq_len, out_dim]
        T = torch.sum(M, dim=3)
        
        return T
    
    def sequence_transform(self, T):
        """
        Transform sequence length using Linker matrix
        Supports batch processing: T shape [batch_size, in_seq, out_dim]
        """
        batch_size, in_seq, out_dim = T.shape
        
        # Transpose to [batch_size, out_dim, in_seq]
        T_transposed = T.transpose(1, 2)
        
        # Apply Linker transformation: [batch_size, out_dim, in_seq] @ [in_seq, out_seq] = [batch_size, out_dim, out_seq]
        U = torch.matmul(T_transposed, self.Linker)
        
        # Transpose back to [batch_size, out_seq, out_dim]
        return U.transpose(1, 2)
    
    def forward(self, x):
        """
        Forward pass through the layer
        Supports batch processing: x shape [batch_size, seq_len, in_dim]
        """
        # Main path: linear transformation
        Z = torch.matmul(x, self.M.t())  # [batch_size, seq_len, out_dim]

        # Layer normalization
        Z = self.ln(Z)
        
        # Position-dependent transformation
        T = self.position_transform(Z)  # [batch_size, seq_len, out_dim]
        
        # Sequence transformation
        U = self.sequence_transform(T)  # [batch_size, out_seq, out_dim]
        
        # Residual connection processing
        if self.use_residual == 'separate':
            # Separate projection and Linker for residual
            residual_feat = self.residual_proj(x)  # [batch_size, seq_len, out_dim]
            residual = self.sequence_transform(residual_feat)  # [batch_size, out_seq, out_dim]
            out = U + residual
        elif self.use_residual == 'shared':
            # Shared M and Linker for residual
            residual_feat = torch.matmul(x, self.M.t())  # [batch_size, seq_len, out_dim]
            residual = self.sequence_transform(residual_feat)  # [batch_size, out_seq, out_dim]
            out = U + residual
        else:
            # Identity residual only if dimensions match
            if self.in_dim == self.out_dim and self.in_seq == self.out_seq:
                residual = self.sequence_transform(x)  # [batch_size, out_seq, out_dim]
                out = U + residual
            else:
                out = U
       
        return out

class HierDDLpm(nn.Module):
    """
    Hierarchical Numeric Dual Descriptor with PyTorch implementation
    Features:
    - GPU acceleration support
    - Flexible residual connections
    - Layer normalization
    - Sequence length transformation via Linker matrices
    - Configurable trainable parameters
    """
    
    def __init__(self, input_dim=2, model_dims=[2],
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
        self.input_seq_len = input_seq_len
        self.linker_dims = linker_dims
        self.num_layers = len(model_dims)
        self.trained = False
        
        # Validate dimensions
        if len(linker_dims) != self.num_layers:
            raise ValueError("linker_dims must have same length as model_dims")
        
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
            use_residual = self.use_residual_list[l]
            
            # Initialize layer
            layer = Layer(
                in_dim, out_dim, in_seq, out_seq, 
                self.linker_trainable[l], 
                use_residual, self.device
            )
            self.layers.append(layer)
        
        # Move model to device
        self.to(self.device)
        
        # Training statistics
        self.mean_t = None
        self.mean_target = None
    
    def forward(self, x):
        """
        Forward pass through all layers
        Supports batch processing: x shape [batch_size, seq_len, input_dim]
        """
        # If input is 2D (single sequence), add batch dimension
        if x.dim() == 2:
            x = x.unsqueeze(0)  # [1, seq_len, input_dim]
        
        current = x
        for layer in self.layers:
            current = layer(current)
        
        # If input was single sequence, remove batch dimension
        if x.dim() == 3 and x.size(0) == 1:
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
        Compute mean squared deviation
        Args:
            seqs: List of input sequences
            t_list: List of target vectors
        
        Returns:
            Mean squared error
        """
        total_loss = 0.0
        count = 0
        
        for seq, t in zip(seqs, t_list):
            # Convert to tensors
            seq_tensor = torch.tensor(seq, dtype=torch.float32, device=self.device)
            t_tensor = torch.tensor(t, dtype=torch.float32, device=self.device)
            
            # Forward pass
            output = self.forward(seq_tensor)
            
            # Compute loss
            target = t_tensor.repeat(output.size(0), 1)
            loss = torch.mean((output - target) ** 2)
            
            total_loss += loss.item() * output.size(0)
            count += output.size(0)
        
        return total_loss / count if count else 0.0
    
    def reg_train(self, seqs, t_list, max_iters=1000, tol=1e-88, lr=0.01, 
                   decay_rate=1.0, print_every=10, continued=False,
                   checkpoint_file=None, checkpoint_interval=10, batch_size=32):
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
            checkpoint_file: Path to save/load checkpoint file for resuming training
            checkpoint_interval: Interval (in iterations) for saving checkpoints
            batch_size: Batch size for training
        
        Returns:
            Training loss history
        """
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
            if 'mean_t' in checkpoint:
                self.mean_t = checkpoint['mean_t']
            if 'mean_target' in checkpoint:
                self.mean_target = torch.tensor(checkpoint['mean_target'], device=self.device)
            print(f"Resumed training from checkpoint at iteration {start_iter}, best loss: {best_loss:.6e}")
        else:
            if not continued:
                self.reset_parameters()
            history = []
        
        # Set model to training mode
        self.train()
        self.trained = True
        
        # Convert all sequences to tensors
        seqs_tensor = torch.stack([torch.tensor(seq, dtype=torch.float32, device=self.device) for seq in seqs])
        t_tensors = torch.tensor(np.array(t_list), dtype=torch.float32, device=self.device)
        num_samples = len(seqs)
        
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
            
            # Shuffle indices
            perm = torch.randperm(num_samples)
            
            # Process in batches
            for i in range(0, num_samples, batch_size):
                batch_idx = perm[i:i+batch_size]
                
                # Get batch data
                batch_seqs = seqs_tensor[batch_idx]  # [batch_size, seq_len, input_dim]
                batch_targets = t_tensors[batch_idx]  # [batch_size, output_dim]
                
                # Forward pass for the whole batch
                output = self.forward(batch_seqs)  # [batch_size, output_seq, output_dim]
                
                # Prepare target - repeat for each position in output sequence
                batch_size_actual, output_seq, output_dim = output.shape
                target = batch_targets.unsqueeze(1).expand(batch_size_actual, output_seq, output_dim)
                
                # Compute loss (mean squared error) for the whole batch
                loss = torch.mean((output - target) ** 2)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item() * batch_size_actual
            
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
            
            # Update learning rate
            scheduler.step()
            
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
        
        # If training ended without convergence, restore best model state
        if best_model_state is not None and best_loss < history[-1]:
            self.load_state_dict(best_model_state)
            print(f"Training ended. Restored best model state with loss = {best_loss:.6e}")
            history[-1] = best_loss
        
        # Store training statistics
        self._compute_training_statistics(seqs)
        self.trained = True
        
        return history
    
    def self_train(self, seqs, max_iters=100, tol=1e-16, learning_rate=0.01, 
                   continued=False, self_mode='gap', decay_rate=1.0, print_every=10,
                   batch_size=32, checkpoint_file=None, checkpoint_interval=5):
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
            if 'mean_t' in checkpoint:
                self.mean_t = checkpoint['mean_t']
            if 'mean_target' in checkpoint:
                self.mean_target = torch.tensor(checkpoint['mean_target'], device=self.device)
            print(f"Resumed self-training from checkpoint at iteration {start_iter}, best loss: {best_loss:.6f}")
        else:
            if not continued:
                self.reset_parameters()
            history = []
        
        # Convert sequences to tensors
        seqs_tensor = torch.stack([torch.tensor(seq, dtype=torch.float32, device=self.device) for seq in seqs])
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
            
            # Shuffle indices
            perm = torch.randperm(num_samples)
            
            # Apply causal masks if in 'reg' mode
            original_linkers = []
            original_residual_linkers = []
            
            if self_mode == 'reg':
                for layer in self.layers:
                    if hasattr(layer, 'Linker'):
                        causal_mask = torch.tril(torch.ones_like(layer.Linker))
                        original_linkers.append(layer.Linker.data.clone())
                        layer.Linker.data.mul_(causal_mask)
                        
                        if layer.use_residual == 'separate' and hasattr(layer, 'residual_linker'):
                            cm_res = torch.tril(torch.ones_like(layer.residual_linker))
                            original_residual_linkers.append(layer.residual_linker.data.clone())
                            layer.residual_linker.data.mul_(cm_res)
                        else:
                            original_residual_linkers.append(None)
            
            # Process in batches
            for i in range(0, num_samples, batch_size):
                batch_idx = perm[i:i+batch_size]
                batch_seqs = seqs_tensor[batch_idx]  # [batch_size, seq_len, input_dim]
                
                # Forward pass for the whole batch
                current = batch_seqs
                for layer in self.layers:
                    # Apply linear transformation
                    linear_out = torch.matmul(current, layer.M.t())  # [batch_size, seq_len, out_dim]
                    
                    # Apply layer normalization
                    normalized_out = layer.ln(linear_out)
                    
                    # Position-dependent transformation
                    T = layer.position_transform(normalized_out)  # [batch_size, seq_len, out_dim]
                    
                    # Sequence transformation (Linker already has causal mask if in 'reg' mode)
                    U = layer.sequence_transform(T)  # [batch_size, out_seq, out_dim]
                    
                    # Handle residual connections
                    if layer.use_residual == 'separate':
                        residual_feat = layer.residual_proj(current)  # [batch_size, seq_len, out_dim]
                        residual = layer.sequence_transform(residual_feat)  # [batch_size, out_seq, out_dim]
                        current = U + residual
                    elif layer.use_residual == 'shared':
                        residual_feat = torch.matmul(current, layer.M.t())  # [batch_size, seq_len, out_dim]
                        residual = layer.sequence_transform(residual_feat)  # [batch_size, out_seq, out_dim]
                        current = U + residual
                    else:
                        if current.shape == U.shape:
                            residual = layer.sequence_transform(current)  # [batch_size, out_seq, out_dim]
                            current = U + residual
                        else:
                            current = U
                
                # Calculate loss based on self_mode for the whole batch
                batch_size_actual, output_seq, output_dim = current.shape
                seq_len = batch_seqs.shape[1]
                
                if self_mode == 'gap':
                    # Self-consistency: output should match input
                    # We need to match dimensions - average output sequence to match input
                    target = batch_seqs
                    # For gap mode, we compare averaged output to input
                    # This is a simplified approach - in practice you might need different matching
                    pred = current.mean(dim=1, keepdim=True).expand(-1, seq_len, -1)
                    loss = torch.mean((pred - target) ** 2)
                else:  # 'reg' mode
                    # Predict next vector in sequence
                    # For simplicity, we'll use a simplified approach here
                    # In practice, you might need a more sophisticated autoregressive setup
                    target = batch_seqs[:, 1:, :]  # Skip first position
                    pred = current[:, :-1, :]  # Skip last position (if dimensions match)
                    min_len = min(target.shape[1], pred.shape[1])
                    loss = torch.mean((pred[:, :min_len, :] - target[:, :min_len, :]) ** 2)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item() * batch_size_actual
            
            # Restore original weights if in 'reg' mode
            if self_mode == 'reg':
                for idx, layer in enumerate(self.layers):
                    if hasattr(layer, 'Linker'):
                        layer.Linker.data.copy_(original_linkers[idx])
                        if original_residual_linkers[idx] is not None:
                            layer.residual_linker.data.copy_(original_residual_linkers[idx])
            
            # Average loss
            avg_loss = total_loss / num_samples
            history.append(avg_loss)
            
            # Update best model state if current loss is lower
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_model_state = copy.deepcopy(self.state_dict())
            
            # Print progress
            if it % print_every == 0 or it == max_iters - 1:
                current_lr = scheduler.get_last_lr()[0]
                mode_display = "Gap" if self_mode == 'gap' else "Reg"
                print(f"SelfTrain({mode_display}) Iter {it:3d}: loss = {avg_loss:.6f}, LR = {current_lr:.6f}")
            
            # Save checkpoint at specified intervals
            if checkpoint_file and (it % checkpoint_interval == 0 or it == max_iters - 1):
                self._save_checkpoint(
                    checkpoint_file, it, history, optimizer, scheduler, best_loss
                )
            
            # Update learning rate
            if it % 5 == 0:  # Decay every 5 iterations
                scheduler.step()
            
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
        
        # If training ended without convergence, restore best model state
        if best_model_state is not None and best_loss < history[-1]:
            self.load_state_dict(best_model_state)
            print(f"Training ended. Restored best model state with loss = {best_loss:.6f}")
            history[-1] = best_loss
        
        self.trained = True
        # Store training statistics
        self._compute_training_statistics(seqs)
        
        return history

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
                # Forward pass (single sequence, no batch dimension)
                current = current_seq
                for layer in self.layers:
                    # For single sequence, add batch dimension temporarily
                    if current.dim() == 2:
                        current_batch = current.unsqueeze(0)  # [1, seq_len, dim]
                    else:
                        current_batch = current
                    
                    # Apply linear transformation
                    linear_out = torch.matmul(current_batch, layer.M.t())
                    
                    # Layer normalization
                    normalized_out = layer.ln(linear_out)
                    
                    # Position-dependent transformation
                    T = layer.position_transform(normalized_out)
                    
                    # Sequence transformation
                    U = layer.sequence_transform(T)
                    
                    # Handle residual connections
                    if layer.use_residual == 'separate':
                        residual_feat = layer.residual_proj(current_batch)
                        residual = layer.sequence_transform(residual_feat)
                        current_batch = U + residual
                    elif layer.use_residual == 'shared':
                        residual_feat = torch.matmul(current_batch, layer.M.t())
                        residual = layer.sequence_transform(residual_feat)
                        current_batch = U + residual
                    else:
                        if current_batch.shape == U.shape:
                            residual = layer.sequence_transform(current_batch)
                            current_batch = U + residual
                        else:
                            current_batch = U
                    
                    # Remove batch dimension if we added it
                    if current.dim() == 2:
                        current = current_batch.squeeze(0)
                    else:
                        current = current_batch
                
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

    def reset_parameters(self):
        """Reset all model parameters to initial values"""
        for layer in self.layers:
            layer.reset_parameters()
        
        # Reset training state
        self.mean_t = None
        self.mean_target = None
        self.trained = False

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
        # Convert mean_target to numpy for saving
        mean_target_np = self.mean_target.cpu().numpy() if self.mean_target is not None else None
        
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
                'input_seq_len': self.input_seq_len,
                'linker_dims': self.linker_dims,
                'linker_trainable': self.linker_trainable,
                'use_residual_list': self.use_residual_list
            },
            'training_stats': {
                'mean_t': self.mean_t,
                'mean_target': mean_target_np
            }
        }
        torch.save(checkpoint, checkpoint_file)
        #print(f"Checkpoint saved at iteration {iteration} to {checkpoint_file}")

    def _compute_training_statistics(self, seqs, batch_size=256):
        """Compute and store statistics for reconstruction"""
        if not seqs:
            self.mean_t = None
            return
            
        total_vectors = 0
        total_sum = torch.zeros(self.input_dim, device=self.device)
        
        # Convert sequences to tensors
        seqs_tensor = torch.stack([torch.tensor(seq, dtype=torch.float32, device=self.device) for seq in seqs])
        
        with torch.no_grad():
            total_sum = torch.sum(seqs_tensor, dim=[0, 1])  # Sum over batch and sequence dimensions
            total_vectors = seqs_tensor.shape[0] * seqs_tensor.shape[1]
        
        self.mean_t = (total_sum / total_vectors).cpu().numpy()
    
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

    def count_parameters(self):
        """Count and print learnable parameters in the model by layer and type in table format"""
        total_params = 0
        
        print("="*70)
        print(f"{'Layer':<10} | {'Parameter':<15} | {'Count':<15} | {'Shape':<20} | {'Trainable':<10}")
        print("-"*70)
        
        for l, layer in enumerate(self.layers):
            layer_params = 0
            
            # M matrix
            num = layer.M.numel()
            shape = str(tuple(layer.M.shape))
            print(f"{f'Layer {l}':<10} | {'M':<15} | {num:<15} | {shape:<20} | {'Yes':<10}")
            layer_params += num
            total_params += num
            
            # P matrix
            num = layer.P.numel()
            shape = str(tuple(layer.P.shape))
            print(f"{f'Layer {l}':<10} | {'P':<15} | {num:<15} | {shape:<20} | {'Yes':<10}")
            layer_params += num
            total_params += num
            
            # Linker matrix
            num = layer.Linker.numel()
            shape = str(tuple(layer.Linker.shape))
            trainable = "Yes" if layer.Linker.requires_grad else "No"
            print(f"{f'Layer {l}':<10} | {'Linker':<15} | {num:<15} | {shape:<20} | {trainable:<10}")
            layer_params += num
            total_params += num
            
            # Residual components
            if layer.use_residual == 'separate':
                # Residual projection
                num = layer.residual_proj.weight.numel()
                shape = str(tuple(layer.residual_proj.weight.shape))
                print(f"{f'Layer {l}':<10} | {'residual_proj':<15} | {num:<15} | {shape:<20} | {'Yes':<10}")
                layer_params += num
                total_params += num
                
                # Residual linker
                if hasattr(layer, 'residual_linker'):
                    num = layer.residual_linker.numel()
                    shape = str(tuple(layer.residual_linker.shape))
                    trainable = "Yes" if layer.residual_linker.requires_grad else "No"
                    print(f"{f'Layer {l}':<10} | {'residual_linker':<15} | {num:<15} | {shape:<20} | {trainable:<10}")
                    layer_params += num
                    total_params += num
            
            # Layer summary
            print(f"{f'Layer {l}':<10} | {'TOTAL':<15} | {layer_params:<15} | {'':<20} | {'':<10}")
            print("-"*70)
        
        print(f"{'ALL':<10} | {'TOTAL':<15} | {total_params:<15} | {'':<20} | {'':<10}")
        print("="*70)
        
        return total_params
    
    def save(self, filename):
        """Save model state to file"""
        # Convert mean_target to numpy for saving
        mean_target_np = self.mean_target.cpu().numpy() if self.mean_target is not None else None
        
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': {
                'input_dim': self.input_dim,
                'model_dims': self.model_dims,
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
    model_mixed = HierDDLpm(
        input_dim=input_dim,
        model_dims=model_dims,
        input_seq_len=input_seq_len,
        linker_dims=linker_dims,
        linker_trainable=[True, False, True],
        use_residual_list=['separate', 'shared', None]  # Different residual for each layer
    )
    
    # Parameter count
    print("\nParameter count before training:")
    total_params = model_mixed.count_parameters()
    
    # Training with checkpoint support and batch processing
    print("\nTraining model with checkpoint support and batch processing...")
    history = model_mixed.reg_train(
        seqs, 
        t_list, 
        max_iters=50,
        lr=0.01,
        decay_rate=0.98,
        print_every=5,
        batch_size=32,
        checkpoint_file='hddlpm_gd_checkpoint.pth',
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
    
    # Test Case 2: Save and load model
    print("\n=== Test Case 2: Save and Load Model ===")
    model_mixed.save("hddlpm_model_residual.pth")
    loaded_model = HierDDLpm.load("hddlpm_model_residual.pth")
    
    # Check parameter equality
    param_match = True
    for (n1, p1), (n2, p2) in zip(model_mixed.named_parameters(), 
                                  loaded_model.named_parameters()):
        if not torch.allclose(p1, p2, atol=1e-6):
            print(f"Parameter mismatch: {n1}")
            param_match = False
    if param_match:
        print("All parameters match successfully!")
    
    # Test Case 3: Predictions consistency
    test_seq = seqs[0]
    orig_pred = model_mixed.predict_t(test_seq)
    loaded_pred = loaded_model.predict_t(test_seq)
    if np.allclose(orig_pred, loaded_pred, atol=1e-5):
        print("Predictions consistent after loading")
    else:
        print("Predictions differ after loading")
        print(f"Original: {orig_pred}")
        print(f"Loaded: {loaded_pred}")
    
    # Test Case 4: Self-training functionality (previously auto_train)
    print("\n=== Test Case 4: Self-training ===")
    # Create a new model for self-training
    model_self = HierDDLpm(
        input_dim=input_dim,
        model_dims=[10, 20, 10],
        input_seq_len=input_seq_len,
        linker_dims=linker_dims,
        linker_trainable=[True, True, True],
        use_residual_list=['separate', 'shared', None]
    )
    
    # Self-train in gap mode (self-consistency) with checkpoint support
    print("\nSelf-training in gap mode with checkpoint support...")
    self_history_gap = model_self.self_train(
        seqs[:10],  # Use subset for faster training
        max_iters=20,
        learning_rate=0.01,
        self_mode='gap',
        print_every=5,
        batch_size=16,
        checkpoint_file='hddlpm_self_checkpoint.pth',
        checkpoint_interval=5
    )
    
    # Reconstruct sequences using the self-trained model (previously generate)
    print("\nReconstructing sequences with self-trained model...")
    reconstructed_seq = model_self.reconstruct(
        L=50,
        tau=0.1,
        discrete_mode=False
    )
    print(f"Reconstructed sequence shape: {reconstructed_seq.shape}")
    print(f"First few vectors:\n{reconstructed_seq[:3]}")
    
    # Test Case 5: Self-training in regression mode
    print("\n=== Test Case 5: Self-training in regression mode ===")
    model_self_reg = HierDDLpm(
        input_dim=input_dim,
        model_dims=[10, 20, 10],
        input_seq_len=input_seq_len,
        linker_dims=linker_dims,
        linker_trainable=[True, True, True],
        use_residual_list=['separate', 'shared', None]
    )
    
    print("\nSelf-training in regression mode...")
    self_history_reg = model_self_reg.self_train(
        seqs[:10],  # Use subset for faster training
        max_iters=20,
        learning_rate=0.01,
        self_mode='reg',
        print_every=5,
        batch_size=16
    )
    
    # Reconstruct sequences using regression-trained model
    print("\nReconstructing sequences with regression-trained model...")
    reconstructed_seq_reg = model_self_reg.reconstruct(
        L=30,
        tau=0.05,
        discrete_mode=False
    )
    print(f"Reconstructed sequence shape: {reconstructed_seq_reg.shape}")
    print(f"First few vectors:\n{reconstructed_seq_reg[:3]}")
    
    # Test Case 6: Save and load self-trained model
    print("\n=== Test Case 6: Save/Load Self-trained Model ===")
    model_self.save("hddlpm_self_model.pth")
    loaded_self_model = HierDDLpm.load("hddlpm_self_model.pth")
    
    # Test reconstruction consistency
    torch.manual_seed(123); random.seed(123); np.random.seed(123)
    orig_reconstruct = model_self.reconstruct(L=10, tau=0.0)
    
    torch.manual_seed(123); random.seed(123); np.random.seed(123)
    loaded_reconstruct = loaded_self_model.reconstruct(L=10, tau=0.0)
    
    if np.allclose(orig_reconstruct, loaded_reconstruct, atol=1e-5):
        print("Reconstruction consistent after loading self-trained model")
    else:
        print("Reconstruction differs after loading")
        print(f"Original mean: {np.mean(orig_reconstruct):.4f}, std: {np.std(orig_reconstruct):.4f}")
        print(f"Loaded mean: {np.mean(loaded_reconstruct):.4f}, std: {np.std(loaded_reconstruct):.4f}")
    
    # Test Case 7: Continued training from checkpoint
    print("\n=== Test Case 7: Continued Training from Checkpoint ===")
    model_continued = HierDDLpm(
        input_dim=input_dim,
        model_dims=[8, 6, 3],
        input_seq_len=input_seq_len,
        linker_dims=linker_dims,
        linker_trainable=[True, False, True],
        use_residual_list=['separate', 'shared', None]
    )
    
    print("\nInitial training for 10 iterations...")
    history1 = model_continued.reg_train(
        seqs[:20], 
        t_list[:20],
        max_iters=10,
        lr=0.01,
        print_every=2,
        batch_size=8,
        checkpoint_file='hddlpm_continued.pth',
        checkpoint_interval=5
    )
    
    print("\nContinuing training from checkpoint for 10 more iterations...")
    history2 = model_continued.reg_train(
        seqs[:20], 
        t_list[:20],
        max_iters=20,
        lr=0.01,
        print_every=2,
        continued=True,
        batch_size=8,
        checkpoint_file='hddlpm_continued.pth',
        checkpoint_interval=5
    )
    
    print(f"Initial training loss: {history1[-1]:.6f}")
    print(f"Continued training loss: {history2[-1]:.6f}")
    
    print("\nAll tests completed successfully!")
