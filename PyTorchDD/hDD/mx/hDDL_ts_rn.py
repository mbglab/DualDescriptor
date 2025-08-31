# Copyright (C) 2005-2025, Bin-Guang Ma (mbg@mail.hzau.edu.cn); SPDX-License-Identifier: MIT
# The Hierarchical Numeric Dual Descriptor (Mixed TS and RN Layers) with Linker matrices in PyTorch
# With layer normalization and residual connections and generation capability
# This program combines both random AB matrix form and tensor form layers
# Author: Bin-Guang Ma (assisted by DeepSeek); Date: 2025-8-30

import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class LayerRN(nn.Module):
    """
    A single layer of the Hierarchical Numeric Dual Descriptor with Linker (Random AB matrix form)
    
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

class LayerTS(nn.Module):
    """Single layer of Hierarchical Numeric Dual Descriptor (Tensor form)"""
    def __init__(self, in_dim, out_dim, in_seq, out_seq, num_basis, linker_trainable, use_residual, device):
        """
        Initialize a hierarchical layer (Tensor form)
        
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
        self.M = nn.Parameter(torch.randn(out_dim, in_dim, device=device) * 0.5)
        
        # Position-dependent transformation tensor
        self.P = nn.Parameter(torch.randn(out_dim, out_dim, num_basis, device=device) * 0.1)
        
        # Precompute periods tensor (fixed, not trainable)
        periods = torch.zeros(out_dim, out_dim, num_basis, dtype=torch.float32, device=device)
        for i in range(out_dim):
            for j in range(out_dim):
                for g in range(num_basis):
                    periods[i, j, g] = i*(out_dim*num_basis) + j*num_basis + g + 2
        self.register_buffer('periods', periods)
        
        # Sequence length transformation matrix
        self.Linker = nn.Parameter(
            torch.randn(in_seq, out_seq, device=device) * 0.1,
            requires_grad=linker_trainable
        )

        # Layer normalization
        self.ln = nn.LayerNorm(out_dim, device=device)
        
        # Residual connection setup
        if self.use_residual == 'separate':
            # Separate projection and Linker for residual path
            self.residual_proj = nn.Linear(in_dim, out_dim, bias=False, device=device)
            self.residual_linker = nn.Parameter(
                torch.randn(in_seq, out_seq, device=device) * 0.1,
                requires_grad=linker_trainable
            )
    
    def position_transform(self, Z):
        """Position-dependent transformation with basis functions (vectorized)"""
        seq_len, _ = Z.shape
        
        # Create position indices [seq_len]
        k = torch.arange(seq_len, device=self.device).float()
        
        # Reshape for broadcasting: [seq_len, 1, 1, 1]
        k = k.view(seq_len, 1, 1, 1)
        
        # Get periods and reshape for broadcasting: [1, out_dim, out_dim, num_basis]
        periods = self.periods.unsqueeze(0)
        
        # Compute basis functions: cos(2πk/period) -> [seq_len, out_dim, out_dim, num_basis]
        phi = torch.cos(2 * math.pi * k / periods)
        
        # Prepare Z for broadcasting: [seq_len, 1, out_dim, 1]
        Z_exp = Z.unsqueeze(1).unsqueeze(-1)
        
        # Prepare P for broadcasting: [1, out_dim, out_dim, num_basis]
        P_exp = self.P.unsqueeze(0)
        
        # Compute position transformation: Z * P * phi
        # Dimensions: 
        #   Z_exp: [seq_len, 1, out_dim, 1]
        #   P_exp: [1, out_dim, out_dim, num_basis]
        #   phi:   [seq_len, out_dim, out_dim, num_basis]
        # Result: [seq_len, out_dim, out_dim, num_basis]
        M = Z_exp * P_exp * phi
        
        # Sum over j (dim=2) and g (dim=3) -> [seq_len, out_dim]
        T = torch.sum(M, dim=(2, 3))
        
        return T
    
    def sequence_transform(self, T):
        """Transform sequence length using Linker matrix"""
        # Using matrix multiplication: Linker^T @ T
        return torch.mm(self.Linker.t(), T)
    
    def forward(self, x):
        """Forward pass through the layer"""
        # Main path: linear transformation
        Z = torch.mm(x, self.M.t())

        # Layer normalization
        Z = self.ln(Z)
        
        # Position-dependent transformation
        T = self.position_transform(Z)
        
        # Sequence transformation
        U = self.sequence_transform(T)
        
        # Residual connection processing
        if self.use_residual == 'separate':
            # Separate projection and Linker for residual
            residual_feat = self.residual_proj(x)
            residual = torch.mm(self.residual_linker.t(), residual_feat)
            out = U + residual
        elif self.use_residual == 'shared':
            # Shared M and Linker for residual
            residual_feat = torch.mm(x, self.M.t())
            residual = torch.mm(self.Linker.t(), residual_feat)
            out = U + residual
        else:
            # Identity residual only if dimensions match
            if self.in_dim == self.out_dim and self.in_seq == self.out_seq:
                residual = torch.mm(self.Linker.t(), x)
                out = U + residual
            else:
                out = U 
       
        return out 

class HierDDLmx(nn.Module):
    """
    Hierarchical Numeric Dual Descriptor with Mixed Layer Types
    
    Args:
        input_dim (int): Dimension of input vectors
        model_dims (list): Output dimensions for each layer
        layer_types (list): Layer types for each layer ('rn' or 'ts')
        rn_params (dict): Parameters for RN layers
        ts_params (dict): Parameters for TS layers
        input_seq_len (int): Fixed input sequence length
        linker_dims (list): Output sequence lengths for each layer
        device (str): Device to place the model on ('cuda' or 'cpu')
    """
    def __init__(self, input_dim=4, model_dims=[4], layer_types=['rn'],
                 rn_params=None, ts_params=None, input_seq_len=100, 
                 linker_dims=[50], device='cuda'):
        super().__init__()
        self.input_dim = input_dim
        self.model_dims = model_dims
        self.layer_types = layer_types
        self.input_seq_len = input_seq_len
        self.linker_dims = linker_dims
        self.num_layers = len(model_dims)
        self.trained = False
        self.device = device
        
        # Validate dimensions
        if len(layer_types) != self.num_layers:
            raise ValueError("layer_types length must match model_dims length")
        if len(linker_dims) != self.num_layers:
            raise ValueError("linker_dims length must match model_dims length")
        
        # Set default parameters for RN and TS layers
        if rn_params is None:
            rn_params = {
                'basis_dims': [50] * self.num_layers,
                'linker_trainable': [False] * self.num_layers,
                'residual_mode_list': [None] * self.num_layers
            }
        
        if ts_params is None:
            ts_params = {
                'num_basis_list': [5] * self.num_layers,
                'linker_trainable': [False] * self.num_layers,
                'use_residual_list': [None] * self.num_layers
            }
        
        # Validate parameter lengths
        if len(rn_params['basis_dims']) != self.num_layers:
            raise ValueError("rn_params['basis_dims'] length must match number of layers")
        if len(rn_params['linker_trainable']) != self.num_layers:
            raise ValueError("rn_params['linker_trainable'] length must match number of layers")
        if len(rn_params['residual_mode_list']) != self.num_layers:
            raise ValueError("rn_params['residual_mode_list'] length must match number of layers")
        
        if len(ts_params['num_basis_list']) != self.num_layers:
            raise ValueError("ts_params['num_basis_list'] length must match number of layers")
        if len(ts_params['linker_trainable']) != self.num_layers:
            raise ValueError("ts_params['linker_trainable'] length must match number of layers")
        if len(ts_params['use_residual_list']) != self.num_layers:
            raise ValueError("ts_params['use_residual_list'] length must match number of layers")
        
        # Create layers
        self.layers = nn.ModuleList()
        for i, (out_dim, layer_type) in enumerate(zip(model_dims, layer_types)):
            # Determine input dimensions for this layer
            in_dim = input_dim if i == 0 else model_dims[i-1]
            in_seq = input_seq_len if i == 0 else linker_dims[i-1]
            out_seq = linker_dims[i]
            
            if layer_type == 'rn':
                # Create RN layer
                layer = LayerRN(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    basis_dim=rn_params['basis_dims'][i],
                    in_seq_len=in_seq,
                    out_seq_len=out_seq,
                    linker_trainable=rn_params['linker_trainable'][i],
                    residual_mode=rn_params['residual_mode_list'][i],
                    device=device
                )
            elif layer_type == 'ts':
                # Create TS layer
                layer = LayerTS(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    in_seq=in_seq,
                    out_seq=out_seq,
                    num_basis=ts_params['num_basis_list'][i],
                    linker_trainable=ts_params['linker_trainable'][i],
                    use_residual=ts_params['use_residual_list'][i],
                    device=device
                )
            else:
                raise ValueError(f"Unknown layer type: {layer_type}. Must be 'rn' or 'ts'")
            
            self.layers.append(layer)
    
    def forward(self, x):
        """
        Forward pass through all layers
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, input_seq_len, input_dim)
            
        Returns:
            Tensor: Output tensor of shape (batch_size, linker_dims[-1], model_dims[-1])
        """
        # Handle TS layers which don't support batch processing
        for i, layer in enumerate(self.layers):
            if isinstance(layer, LayerTS):
                # TS layers don't support batch processing, so we handle each sample individually
                if x.dim() == 3:  # Batch processing
                    batch_size, seq_len, dim = x.shape
                    outputs = []
                    for j in range(batch_size):
                        sample = x[j]  # (seq_len, dim)
                        output = layer(sample)  # (out_seq, out_dim)
                        outputs.append(output.unsqueeze(0))
                    x = torch.cat(outputs, dim=0)  # (batch_size, out_seq, out_dim)
                else:
                    x = layer(x)  # Single sample
            else:
                # RN layers support batch processing
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
        
        if vec_seq.dim() == 2:
            vec_seq = vec_seq.unsqueeze(0)  # Add batch dimension
        
        # Forward pass
        with torch.no_grad():
            output = self.forward(vec_seq)
            
        # Remove batch dimension and convert to list of vectors
        return output.squeeze(0).cpu().numpy().tolist() if output.dim() == 3 else output.cpu().numpy()
    
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
    
    def grad_train(self, seqs, t_list, max_iters=1000, tol=1e-88, 
                   learning_rate=0.01, decay_rate=1.0, print_every=10):
        """
        Train the model using Adam optimizer
        
        Args:
            seqs (list): Training sequences
            t_list (list): Target vectors
            max_iters (int): Maximum training iterations
            tol (float): Tolerance for convergence
            learning_rate (float): Initial learning rate
            decay_rate (float): Learning rate decay rate
            print_every (int): Print progress every n iterations
            
        Returns:
            list: Training loss history
        """
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)
        history = []
        best_loss = float('inf')
        
        for it in range(max_iters):
            optimizer.zero_grad()
            
            # Compute loss
            loss = self.deviation(seqs, t_list)
            
            # Backpropagate
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            
            # Update parameters
            optimizer.step()                  
            
            # Record loss
            current_loss = loss.item()
            history.append(current_loss)
            
            # Print progress
            if it % print_every == 0 or it == max_iters - 1:
                lr = optimizer.param_groups[0]['lr']
                print(f"Iter {it:3d}: Loss = {current_loss:.6f}, LR = {lr:.6f}")

            # Decay learning rate
            scheduler.step()
            
            # Check convergence
            if current_loss < best_loss:
                best_loss = current_loss
            elif best_loss - current_loss < tol:
                print(f"Converged after {it+1} iterations")
                break
                
        self.trained = True
        return history

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

    def auto_train(self, seqs, max_iters=100, tol=1e-16, learning_rate=0.01, 
              continued=False, auto_mode='gap', decay_rate=1.0, print_every=10):
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
            auto_mode: Training mode ('gap' or 'reg')
            decay_rate: Learning rate decay rate
            print_every: Print interval
        
        Returns:
            Training loss history
        """
        if auto_mode not in ('gap', 'reg'):
            raise ValueError("auto_mode must be either 'gap' or 'reg'")
        
        # Validate input sequences
        for seq in seqs:
            if len(seq) != self.input_seq_len:
                raise ValueError(f"All sequences must have length {self.input_seq_len}")
        
        # Convert sequences to tensors
        seqs_tensor = [torch.tensor(seq, dtype=torch.float32, device=self.device) for seq in seqs]
        
        # Initialize all layers if not continuing training
        if not continued:
            for layer in self.layers:
                if isinstance(layer, LayerRN):
                    # Reinitialize RN layer parameters
                    nn.init.uniform_(layer.M, -0.5, 0.5)
                    nn.init.uniform_(layer.Acoeff, -0.1, 0.1)
                    nn.init.uniform_(layer.Bbasis, -0.1, 0.1)
                    nn.init.uniform_(layer.Linker, -0.1, 0.1)
                    
                    # Reinitialize residual components if using 'separate' mode
                    if layer.residual_mode == 'separate':
                        nn.init.uniform_(layer.residual_proj.weight, -0.1, 0.1)
                        if hasattr(layer, 'residual_linker'):
                            nn.init.uniform_(layer.residual_linker, -0.1, 0.1)
                else:
                    # Reinitialize TS layer parameters
                    nn.init.uniform_(layer.M, -0.5, 0.5)
                    nn.init.uniform_(layer.P, -0.1, 0.1)
                    
                    if layer.Linker.requires_grad:
                        nn.init.uniform_(layer.Linker, -0.1, 0.1)
                    
                    # Reinitialize residual components if using 'separate' mode
                    if layer.use_residual == 'separate':
                        nn.init.uniform_(layer.residual_proj.weight, -0.1, 0.1)
                        if hasattr(layer, 'residual_linker') and layer.residual_linker.requires_grad:
                            nn.init.uniform_(layer.residual_linker, -0.1, 0.1)
            
            # Calculate global mean vector of input sequences
            total = torch.zeros(self.input_dim, device=self.device)
            total_vectors = 0
            for seq in seqs_tensor:
                total += torch.sum(seq, dim=0)
                total_vectors += seq.size(0)
            self.mean_t = (total / total_vectors).cpu().numpy()            
        
        # Calculate total training samples
        total_samples = 0
        for seq in seqs_tensor:
            if auto_mode == 'gap':
                total_samples += seq.size(0)  # All positions are samples
            else:  # 'reg' mode
                total_samples += max(0, seq.size(0) - 1)  # All except last position
        
        if total_samples == 0:
            raise ValueError("No training samples found")
        
        # Setup optimizer
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)
        
        history = []
        prev_loss = float('inf')
        
        for it in range(max_iters):
            total_loss = 0.0
            
            for seq in seqs_tensor:
                optimizer.zero_grad()
                
                # Forward pass with causal masking for reg mode
                x = seq.unsqueeze(0)  # Add batch dimension
                
                # Apply causal masking to Linker matrices in reg mode
                if auto_mode == 'reg':
                    # Store original Linker matrices and apply causal masks
                    original_linkers = []
                    original_residual_linkers = []
                    
                    for layer in self.layers:
                        if isinstance(layer, LayerRN):
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
                        else:
                            # TS layer
                            causal_mask = torch.tril(torch.ones_like(layer.Linker))
                            original_linker = layer.Linker.clone()
                            layer.Linker.data = layer.Linker.data * causal_mask
                            original_linkers.append(original_linker)
                            
                            if hasattr(layer, 'residual_linker') and layer.residual_linker is not None:
                                causal_mask_residual = torch.tril(torch.ones_like(layer.residual_linker))
                                original_residual_linker = layer.residual_linker.clone()
                                layer.residual_linker.data = layer.residual_linker.data * causal_mask_residual
                                original_residual_linkers.append(original_residual_linker)
                            else:
                                original_residual_linkers.append(None)
                
                # Forward pass through all layers
                for layer in self.layers:
                    if isinstance(layer, LayerTS):
                        # TS layers need special handling for batch processing
                        if x.dim() == 3:  # Batch processing
                            batch_size, seq_len, dim = x.shape
                            outputs = []
                            for j in range(batch_size):
                                sample = x[j]  # (seq_len, dim)
                                output = layer(sample)  # (out_seq, out_dim)
                                outputs.append(output.unsqueeze(0))
                            x = torch.cat(outputs, dim=0)  # (batch_size, out_seq, out_dim)
                        else:
                            x = layer(x)  # Single sample
                    else:
                        # RN layers support batch processing
                        x = layer(x)
                
                # Remove batch dimension for TS layers
                if any(isinstance(layer, LayerTS) for layer in self.layers) and x.dim() == 3:
                    current = x.squeeze(0)
                else:
                    current = x
                
                # Restore original Linker matrices after forward pass in reg mode
                if auto_mode == 'reg':
                    for i, layer in enumerate(self.layers):
                        layer.Linker.data = original_linkers[i]
                        if original_residual_linkers[i] is not None:
                            if isinstance(layer, LayerRN):
                                layer.residual_linker.data = original_residual_linkers[i]
                            else:
                                layer.residual_linker.data = original_residual_linkers[i]
                
                # Calculate loss based on auto_mode
                loss = 0.0
                valid_positions = 0
                
                for k in range(current.size(0)):
                    # Skip last position in 'reg' mode
                    if auto_mode == 'reg' and k == seq.size(0) - 1:
                        continue
                    
                    # Determine target based on mode
                    if auto_mode == 'gap':
                        target = seq[k]
                    else:  # 'reg' mode
                        target = seq[k + 1]
                    
                    # Calculate MSE loss
                    loss += torch.sum((current[k] - target) ** 2)
                    valid_positions += 1
                
                if valid_positions > 0:
                    loss = loss / valid_positions
                    total_loss += loss.item()
                    loss.backward()
                    optimizer.step()
            
            # Average loss
            avg_loss = total_loss / len(seqs_tensor)
            history.append(avg_loss)            
            
            # Print progress
            if it % print_every == 0 or it == max_iters - 1:
                current_lr = scheduler.get_last_lr()[0]
                mode_display = "Gap" if auto_mode == 'gap' else "Reg"
                print(f"AutoTrain({mode_display}) Iter {it:3d}: loss = {avg_loss:.6f}, LR = {current_lr:.6f}")

            # Update learning rate
            if it % 5 == 0:  # Decay every 5 iterations
                scheduler.step()
            
            # Check convergence
            if prev_loss - avg_loss < tol:
                print(f"Converged after {it+1} iterations")
                break
            prev_loss = avg_loss
        
        self.trained = True
        return history
    
    def generate(self, L, tau=0.0, discrete_mode=False, vocab_size=None):
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
        assert self.trained and hasattr(self, 'mean_t'), "Model must be auto-trained first"
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
                    if isinstance(layer, LayerTS):
                        # TS layers need special handling for batch processing
                        if x.dim() == 3:  # Batch processing
                            batch_size, seq_len, dim = x.shape
                            outputs = []
                            for j in range(batch_size):
                                sample = x[j]  # (seq_len, dim)
                                output = layer(sample)  # (out_seq, out_dim)
                                outputs.append(output.unsqueeze(0))
                            x = torch.cat(outputs, dim=0)  # (batch_size, out_seq, out_dim)
                        else:
                            x = layer(x)  # Single sample
                    else:
                        # RN layers support batch processing
                        x = layer(x)
                
                # Get the last output vector
                if x.dim() == 3:  # Batch processing
                    output_vector = x.squeeze(0)[-1]  # Last position of last sequence
                else:
                    output_vector = x[-1]  # Last position
                
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
                        noise = torch.normal(0, tau * torch.abs(output_vector) + 0.01) #, device=self.device
                        output_vector = output_vector + noise
                
                generated.append(output_vector.cpu().numpy())
                
                # Update current sequence (shift window)
                current_seq = torch.cat([current_seq[1:], output_vector.unsqueeze(0)])
        
        return np.array(generated)
    
    def count_parameters(self):
        """
        Calculate and print the number of learnable parameters
        """
        total_params = 0
        trainable_params = 0
        print("Model Parameter Counts:")
        
        for l_idx, layer in enumerate(self.layers):
            layer_params = sum(p.numel() for p in layer.parameters())
            layer_trainable = sum(p.numel() for p in layer.parameters() if p.requires_grad)
            
            total_params += layer_params
            trainable_params += layer_trainable
            
            in_dim = self.input_dim if l_idx == 0 else self.model_dims[l_idx-1]
            in_seq = self.input_seq_len if l_idx == 0 else self.linker_dims[l_idx-1]
            out_dim = self.model_dims[l_idx]
            out_seq = self.linker_dims[l_idx]
            
            if isinstance(layer, LayerRN):
                layer_type = "RN"
                L_i = layer.basis_dim
                residual_mode = layer.residual_mode
                print(f"  Layer {l_idx} ({layer_type}): in_dim={in_dim}, out_dim={out_dim}, L={L_i}, in_seq={in_seq}, out_seq={out_seq}")
                print(f"    Parameters: {layer_params} (Trainable: {layer_trainable})")
                print(f"    Residual mode: {residual_mode}")
            else:
                layer_type = "TS"
                num_basis = layer.num_basis
                use_residual = layer.use_residual
                print(f"  Layer {l_idx} ({layer_type}): in_dim={in_dim}, out_dim={out_dim}, num_basis={num_basis}, in_seq={in_seq}, out_seq={out_seq}")
                print(f"    Parameters: {layer_params} (Trainable: {layer_trainable})")
                print(f"    Residual type: {use_residual}")
        
        print(f"Total parameters: {total_params}")
        print(f"Trainable parameters: {trainable_params}")        
        return total_params, trainable_params

    def save(self, filename):
        """Save model state to file"""
        # Get layer types
        layer_types = []
        for layer in self.layers:
            if isinstance(layer, LayerRN):
                layer_types.append('rn')
            else:
                layer_types.append('ts')
        
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': {
                'input_dim': self.input_dim,
                'model_dims': self.model_dims,
                'layer_types': layer_types,
                'input_seq_len': self.input_seq_len,
                'linker_dims': self.linker_dims,
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
            layer_types=config['layer_types'],
            input_seq_len=config['input_seq_len'],
            linker_dims=config['linker_dims'],
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
    input_seq_len = 100     # Fixed input sequence length
    model_dims = [8, 6, 3]  # Output dimensions for each layer
    layer_types = ['rn', 'ts', 'rn']  # Layer types for each layer
    linker_dims = [50, 20, 10]  # Output sequence lengths for each layer
    seq_count = 100         # Number of training sequences
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # RN layer parameters
    rn_params = {
        'basis_dims': [100, 40, 20],  # Basis dimensions for each RN layer
        'linker_trainable': [True, False, True],  # Linker trainable for each RN layer
        'residual_mode_list': ['separate', 'shared', None]  # Residual mode for each RN layer
    }
    
    # TS layer parameters
    ts_params = {
        'num_basis_list': [5, 4, 3],  # Basis functions for each TS layer
        'linker_trainable': [False, True, False],  # Linker trainable for each TS layer
        'use_residual_list': [None, 'separate', 'shared']  # Residual mode for each TS layer
    }
    
    # Generate training data with fixed sequence length
    print("\nGenerating training data...")
    print(f"Input dim: {input_dim}, Seq len: {input_seq_len}")
    print(f"Layer dims: {model_dims}, Layer types: {layer_types}, Linker dims: {linker_dims}")
    
    seqs = []   # List of sequences
    t_list = [] # List of target vectors

    for i in range(seq_count):
        # Generate vector sequence with fixed length
        sequence = np.random.uniform(-1, 1, (input_seq_len, input_dim))
        seqs.append(sequence)
        
        # Generate random target vector
        target = np.random.uniform(-1, 1, model_dims[-1])
        t_list.append(target)
    
    # Create mixed model
    print("\n=== Creating Mixed Model ===")
    model_mixed = HierDDLmx(
        input_dim=input_dim,
        model_dims=model_dims,
        layer_types=layer_types,
        rn_params=rn_params,
        ts_params=ts_params,
        input_seq_len=input_seq_len,
        linker_dims=linker_dims,
        device=device
    )
    
    # Show parameter counts
    print("\nModel parameter counts:")
    model_mixed.count_parameters()
    
    # Train model
    print("\nTraining model...")
    history = model_mixed.grad_train(
        seqs, 
        t_list,
        learning_rate=0.01,
        max_iters=100,
        decay_rate=0.99,
        print_every=10
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
    
    # Test Case 2: Auto-training and generation
    print("\n=== Test Case 2: Auto-training and Generation ===")
    
    # Create a fresh model for auto-training
    model_auto = HierDDLmx(
        input_dim=input_dim,
        model_dims=[10, 20, 10],
        layer_types=['rn', 'ts', 'rn'],
        input_seq_len=input_seq_len,
        linker_dims=linker_dims,
        device=device
    )
    
    # Auto-train in gap mode
    print("\nAuto-training in gap mode...")
    auto_history = model_auto.auto_train(
        seqs,
        max_iters=50,
        learning_rate=0.01,
        auto_mode='gap',
        print_every=1
    )
    
    # Generate new sequences
    print("\nGenerating sequences...")
    generated_seqs = model_auto.generate(
        L=20,
        tau=0.1,
        discrete_mode=False
    )
    
    print(f"Generated {len(generated_seqs)} sequences of length {len(generated_seqs[0])}")
    print(f"First generated vector: {generated_seqs[0]}")
    
    # Test Case 3: Save and load model
    print("\n=== Test Case 3: Save and Load Model ===")
    
    # Save model
    model_auto.save("test_model_mixed.pth")
    
    # Load model
    loaded_model = HierDDLmx.load("test_model_mixed.pth", device=device)
    
    # Verify loaded model works
    test_pred = loaded_model.predict_t(seqs[0])
    print(f"Loaded model prediction: {test_pred[:3]}...")  # Show first 3 values
    
    print("\n=== Hierarchical Vector Sequence Processing with Mixed Layer Types Demo Completed ===")
