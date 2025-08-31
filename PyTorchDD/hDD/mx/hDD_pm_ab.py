# Copyright (C) 2005-2025, Bin-Guang Ma (mbg@mail.hzau.edu.cn); SPDX-License-Identifier: MIT
# The Hierarchical Numeric Dual Descriptor (Mixed AB and PM Layers) implemented with PyTorch
# With layer normalization and residual connections and generation capability
# This program is for the demonstration of methodology and not fully refined.
# Author: Bin-Guang Ma (assisted by DeepSeek); Date: 2025-8-30

import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class LayerAB(nn.Module):
    """
    Single layer of the Hierarchical Numeric Dual Descriptor (AB matrix form) with Residual Connections.
    
    Args:
        in_dim (int): Input dimension
        out_dim (int): Output dimension
        basis_dim (int): Basis dimension for this layer
        use_residual (str or None): Residual connection type. Options are:
            - 'separate': use a separate linear projection for the residual connection.
            - 'shared': share the linear transformation matrix M for the residual.
            - None or other: no residual connection, unless in_dim==out_dim, then use identity residual.
        device (str): Device to run computations on ('cuda' or 'cpu')
    """
    def __init__(self, in_dim, out_dim, basis_dim, use_residual=None, device='cuda'):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.basis_dim = basis_dim
        self.use_residual = use_residual
        self.device = device
        
        # Linear transformation
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        
        # Layer normalization
        self.norm = nn.LayerNorm(out_dim)
        
        # Coefficient matrix: out_dim x basis_dim (learnable)
        self.Acoeff = nn.Parameter(torch.Tensor(out_dim, basis_dim))
        
        # Fixed basis matrix B (basis_dim x out_dim) calculated with:
        #   Bbasis[k, i] = cos(2π*(k+1)/(i+2))
        Bbasis = torch.empty(basis_dim, out_dim)
        for k in range(basis_dim):
            for i in range(out_dim):
                Bbasis[k, i] = math.cos(2 * math.pi * (k+1) / (i+2))
        self.register_buffer('Bbasis', Bbasis)  # Fixed, non-trainable buffer
        
        # Initialize parameters
        nn.init.uniform_(self.linear.weight, -0.1, 0.1)
        nn.init.uniform_(self.Acoeff, -0.1, 0.1)
        
        # Handle residual connection based on mode
        if use_residual == 'separate':
            # Separate linear projection for residual
            self.residual_proj = nn.Linear(in_dim, out_dim, bias=False)
            nn.init.uniform_(self.residual_proj.weight, -0.1, 0.1)
        else:
            self.residual_proj = None
            
        self.to(device)
    
    def forward(self, x, positions=None):
        """
        Forward pass through the layer with residual connections
        
        Args:
            x (torch.Tensor): Input sequence of shape (batch_size, seq_len, in_dim)
            positions (torch.Tensor): Position indices (not used in AB layer, for compatibility)
        
        Returns:
            torch.Tensor: Output sequence of shape (batch_size, seq_len, out_dim)
        """
        batch_size, seq_len, in_dim = x.shape
        
        # Compute residual based on mode
        residual = 0
        if self.use_residual == 'separate':
            residual = self.residual_proj(x)
        elif self.use_residual == 'shared':
            residual = self.linear(x)  # Use shared linear transformation
        else:  # None or other
            if self.in_dim == self.out_dim:
                residual = x  # Identity connection
        
        # Main path: linear transformation
        transformed = self.linear(x)
        
        # Apply layer normalization
        normalized = self.norm(transformed)
        
        # Compute basis indices: j = k % basis_dim for each position
        j_indices = torch.arange(seq_len, device=self.device) % self.basis_dim
        
        # Select basis vectors: (seq_len, out_dim)
        B_vectors = self.Bbasis[j_indices]
        
        # Compute scalars: dot product of normalized and B_vectors
        # normalized: (batch_size, seq_len, out_dim)
        # B_vectors: (seq_len, out_dim) -> expand to (batch_size, seq_len, out_dim)
        B_vectors_expanded = B_vectors.unsqueeze(0).expand(batch_size, -1, -1)
        scalars = torch.sum(normalized * B_vectors_expanded, dim=2)  # (batch_size, seq_len)
        
        # Select coefficient vectors: (seq_len, out_dim)
        A_vectors = self.Acoeff[:, j_indices].t()  # (seq_len, out_dim)
        A_vectors_expanded = A_vectors.unsqueeze(0).expand(batch_size, -1, -1)  # (batch_size, seq_len, out_dim)
        
        # Compute new features: scalars * A_vectors
        new_features = scalars.unsqueeze(2) * A_vectors_expanded  # (batch_size, seq_len, out_dim)
        
        # Add residual connection
        output = new_features + residual
        
        return output

class LayerPM(nn.Module):
    """Single layer of Hierarchical Dual Descriptor (with 2D P matrix)"""
    def __init__(self, in_dim, out_dim, use_residual, device='cpu'):
        """
        Initialize a hierarchical layer with simplified P matrix
        
        Args:
            in_dim (int): Input dimension
            out_dim (int): Output dimension
            use_residual (str or None): Residual connection type. Options are:
                - 'separate': use a separate linear projection for the residual connection.
                - 'shared': share the linear transformation matrix M for the residual.
                - None or other: no residual connection, unless in_dim==out_dim, then use identity residual.
            device (str): Device to run computations on ('cuda' or 'cpu')
        """
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.use_residual = use_residual
        self.device = device
        
        # Linear transformation matrix (shared for main path and residual)
        self.M = nn.Parameter(torch.empty(out_dim, in_dim))
        
        # Simplified P matrix (out_dim, out_dim)
        self.P = nn.Parameter(torch.empty(out_dim, out_dim))
        
        # Precompute periods matrix (fixed, not trainable)
        periods = torch.zeros(out_dim, out_dim)
        for i in range(out_dim):
            for j in range(out_dim):
                # Simplified period calculation
                periods[i, j] = i * out_dim + j + 2
        self.register_buffer('periods', periods)

        # Residual projection processing
        if self.use_residual == 'separate':
            self.residual_proj = nn.Linear(in_dim, out_dim, bias=False) if in_dim != out_dim else nn.Identity()
        else:
            self.residual_proj = None
        
        # Layer normalization
        self.norm = nn.LayerNorm(out_dim)
        
        # Initialize parameters        
        nn.init.uniform_(self.M, -0.5, 0.5)
        nn.init.uniform_(self.P, -0.1, 0.1)
        if isinstance(self.residual_proj, nn.Linear):            
            nn.init.uniform_(self.residual_proj.weight, -0.1, 0.1)
            
        self.to(device)
    
    def forward(self, x, positions):
        """
        Forward pass for the layer with simplified P matrix
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, in_dim)
            positions (Tensor): Position indices of shape (seq_len,)
        
        Returns:
            Tensor: Output of shape (batch_size, seq_len, out_dim)
        """
        batch_size, seq_len, _ = x.shape
        
        # Linear transformation: x' = x @ M^T
        x_trans = torch.matmul(x, self.M.t())  # (batch_size, seq_len, out_dim)

        # Layer normalization
        x_trans = self.norm(x_trans)
        
        # Prepare position-dependent transformation
        k = positions.view(-1, 1, 1).float()  # (seq_len, 1, 1)
        
        # Compute basis function: phi = cos(2π * k / period)
        # periods shape: (1, out_dim, out_dim)
        periods = self.periods.unsqueeze(0)  # add dimension for broadcasting
        phi = torch.cos(2 * math.pi * k / periods)  # (seq_len, out_dim, out_dim)
        
        # Compute position-dependent transformation matrix
        M_k = self.P.unsqueeze(0) * phi  # (seq_len, out_dim, out_dim)
        
        # Apply transformation using Einstein summation
        # 'bsj,sij->bsi': for each position s, multiply input vector with transformation matrix
        Nk = torch.einsum('bsj,sij->bsi', x_trans, M_k)
        
        # Residual connection processing 
        if self.use_residual == 'separate':
            residual = self.residual_proj(x)
            out = Nk + residual            
        elif self.use_residual == 'shared':        
            residual = torch.matmul(x, self.M.t())
            out = Nk + residual
        else:
            if self.in_dim == self.out_dim and self.use_residual != None:
                residual = x
                out = Nk + residual
            else:
                out = Nk
                
        return out

class HierDDmx(nn.Module):
    """
    Hierarchical Numeric Dual Descriptor with Mixed AB and PM Layers.
    Supports variable dimensions for each layer and configurable residual modes.
    
    Args:
        input_dim (int): Dimension of input vectors
        model_dims (list): List of output dimensions for each layer
        basis_dims (list): List of basis dimensions for AB layers (ignored for PM layers)
        use_residual_list (list or None): List of residual modes for each layer
        layer_types (list): List of layer types ('ab' or 'pm') for each layer
        device (str): Device to run computations on ('cuda' or 'cpu')
    """
    def __init__(self, input_dim=4, model_dims=[4], basis_dims=[50], 
                 use_residual_list=None, layer_types=None, device='cuda'):
        super().__init__()
        self.input_dim = input_dim
        self.model_dims = model_dims
        self.basis_dims = basis_dims
        self.num_layers = len(model_dims)
        self.device = device
        self.trained = False
        
        # Validate dimensions
        if len(basis_dims) != self.num_layers:
            raise ValueError("basis_dims length must match model_dims length")
        
        # Handle residual modes
        if use_residual_list is None:
            use_residual_list = [None] * self.num_layers
        elif len(use_residual_list) != self.num_layers:
            raise ValueError("use_residual_list length must match model_dims length")
        
        # Handle layer types
        if layer_types is None:
            # Default: alternate between AB and PM layers
            layer_types = ['ab' if i % 2 == 0 else 'pm' for i in range(self.num_layers)]
        elif len(layer_types) != self.num_layers:
            raise ValueError("layer_types length must match model_dims length")
        
        self.layer_types = layer_types
        
        # Create layers
        self.layers = nn.ModuleList()
        in_dim = input_dim
        for i in range(self.num_layers):
            out_dim = model_dims[i]
            basis_dim = basis_dims[i]
            use_residual = use_residual_list[i]
            layer_type = layer_types[i]
            
            if layer_type == 'ab':
                layer = LayerAB(in_dim, out_dim, basis_dim, use_residual, device)
            elif layer_type == 'pm':
                layer = LayerPM(in_dim, out_dim, use_residual, device)
            else:
                raise ValueError(f"Unknown layer type: {layer_type}")
                
            self.layers.append(layer)
            in_dim = out_dim  # Next layer input is current layer output
        
        self.to(device)
    
    def forward(self, x):
        """
        Forward pass through the entire hierarchical model
        
        Args:
            x (torch.Tensor): Input sequence of shape (batch_size, seq_len, input_dim)
        
        Returns:
            torch.Tensor: Output sequence of shape (batch_size, seq_len, model_dims[-1])
        """
        batch_size, seq_len, _ = x.shape
        positions = torch.arange(seq_len, device=self.device)
        
        current = x
        for i, layer in enumerate(self.layers):
            if self.layer_types[i] == 'ab':
                current = layer(current)  # AB layers don't need positions
            else:  # PM layers
                current = layer(current, positions)
        return current
    
    def describe(self, vec_seq):
        """
        Compute descriptor vectors for each position in the sequence
        
        Args:
            vec_seq (list): List of input vectors (each vector is list of floats)
        
        Returns:
            list: List of output vectors from the final layer
        """
        # Convert to tensor and move to device
        input_tensor = torch.tensor(vec_seq, dtype=torch.float32).to(self.device)
        input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension
        
        # Forward pass
        output_tensor = self.forward(input_tensor)
        
        # Remove batch dimension and convert back to list of vectors
        return output_tensor.squeeze(0).cpu().detach().numpy().tolist()
    
    def deviation(self, seqs, t_list):
        """
        Compute mean squared error between descriptors and targets
        
        Args:
            seqs (list): List of input sequences
            t_list (list): List of target vectors
        
        Returns:
            float: Mean squared error
        """
        total_loss = 0.0
        total_positions = 0
        
        for seq, t in zip(seqs, t_list):
            # Convert to tensors
            input_tensor = torch.tensor(seq, dtype=torch.float32).to(self.device)
            input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension
            target_tensor = torch.tensor(t, dtype=torch.float32).to(self.device)
            
            # Forward pass
            outputs = self.forward(input_tensor)
            outputs = outputs.squeeze(0)  # Remove batch dimension
            
            # Compute MSE for each position
            for pos in range(len(seq)):
                loss = torch.mean((outputs[pos] - target_tensor) ** 2)
                total_loss += loss.item()
                total_positions += 1
        
        return total_loss / total_positions if total_positions > 0 else 0.0
    
    def grad_train(self, seqs, t_list, max_iters=1000, tol=1e-18,
                    learning_rate=0.01, decay_rate=1.0, print_every=10):
        """
        Train the model using Adam optimizer with learning rate decay and early stopping
        
        Args:
            seqs (list): List of training sequences
            t_list (list): List of target vectors
            max_iters (int): Maximum training iterations
            tol (float): Convergence tolerance
            learning_rate (float): Initial learning rate
            decay_rate (float): Learning rate decay factor
            print_every (int): Print progress every N iterations
        
        Returns:
            list: Training loss history
        """
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)
        history = []
        best_loss = float('inf')
        best_state = self.state_dict()
        
        # Convert targets to tensors
        target_tensors = [torch.tensor(t, dtype=torch.float32).to(self.device) for t in t_list]
        
        for it in range(max_iters):
            total_loss = 0.0
            total_positions = 0
            
            # Training pass
            self.train()
            for seq, target in zip(seqs, target_tensors):
                # Convert sequence to tensor
                input_tensor = torch.tensor(seq, dtype=torch.float32).to(self.device)
                input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension
                
                # Forward pass
                outputs = self.forward(input_tensor)
                outputs = outputs.squeeze(0)  # Remove batch dimension
                
                # Compute loss (mean across positions and dimensions)
                loss = torch.mean((outputs - target) ** 2)
                total_loss += loss.item() * len(seq)
                total_positions += len(seq)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()               
            
            # Calculate average loss
            avg_loss = total_loss / total_positions
            history.append(avg_loss)
            
            # Print progress
            if it % print_every == 0 or it == max_iters - 1:
                lr = optimizer.param_groups[0]['lr']
                print(f"Iter {it:4d}: Loss = {avg_loss:.8f}, LR = {lr:.6f}")

            # Update learning rate
            scheduler.step()
            
            # Check for improvement
            if avg_loss < best_loss - tol:
                best_loss = avg_loss
                best_state = self.state_dict()
            else:
                # No improvement - revert to best state and stop
                self.load_state_dict(best_state)
                print(f"Converged after {it} iterations")
                break
        
        self.trained = True
        return history
    
    def predict_t(self, vec_seq):
        """
        Predict target vector as average of all final layer outputs
        
        Args:
            vec_seq (list): Input sequence of vectors
        
        Returns:
            list: Predicted target vector
        """
        outputs = self.describe(vec_seq)
        if not outputs:
            return [0.0] * self.model_dims[-1]
        
        # Compute average across positions
        return np.array(outputs).mean(axis=0).tolist()     
    
    def auto_train(self, seqs, max_iters=100, tol=1e-16, learning_rate=0.01, 
                  continued=False, auto_mode='gap', decay_rate=1.0, print_every=10):
        """
        Self-training for the hierarchical model with two modes:
          - 'gap': Predicts current input vector (self-consistency)
          - 'reg': Predicts next input vector (auto-regressive) with causal masking          
        
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
        
        # Convert sequences to tensors
        seqs_tensor = [torch.tensor(seq, dtype=torch.float32, device=self.device) for seq in seqs]
        
        # Initialize all layers if not continuing training
        if not continued:
            for layer in self.layers:
                # Reinitialize parameters with small random values
                if hasattr(layer, 'linear'):  # AB layer
                    nn.init.uniform_(layer.linear.weight, -0.1, 0.1)
                    nn.init.uniform_(layer.Acoeff, -0.1, 0.1)
                    
                    # Reinitialize residual components if using 'separate' mode
                    if layer.use_residual == 'separate' and hasattr(layer, 'residual_proj'):
                        if isinstance(layer.residual_proj, nn.Linear):
                            nn.init.uniform_(layer.residual_proj.weight, -0.1, 0.1)
                else:  # PM layer
                    nn.init.uniform_(layer.M, -0.5, 0.5)
                    nn.init.uniform_(layer.P, -0.1, 0.1)
                    
                    # Reinitialize residual components if using 'separate' mode
                    if layer.use_residual == 'separate' and hasattr(layer, 'residual_proj'):
                        if isinstance(layer.residual_proj, nn.Linear):
                            nn.init.uniform_(layer.residual_proj.weight, -0.1, 0.1)
            
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
                
                # Add batch dimension
                seq_batch = seq.unsqueeze(0)
                
                # Forward pass
                outputs = self.forward(seq_batch)
                outputs = outputs.squeeze(0)  # Remove batch dimension
                
                # Calculate loss based on auto_mode
                loss = 0.0
                valid_positions = 0
                
                for k in range(outputs.size(0)):
                    # Skip last position in 'reg' mode
                    if auto_mode == 'reg' and k == seq.size(0) - 1:
                        continue
                    
                    # Determine target based on mode
                    if auto_mode == 'gap':
                        target = seq[k]
                    else:  # 'reg' mode
                        target = seq[k + 1]
                    
                    # Calculate MSE loss
                    loss += torch.sum((outputs[k] - target) ** 2)
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
        seq_len = 10  # Fixed sequence length for generation
        current_seq = torch.normal(
            mean=0.0,
            std=0.1,
            size=(seq_len, self.input_dim),
            device=self.device
        )
        # Add the learned mean vector to the initial sequence
        mean_tensor = torch.tensor(self.mean_t, device=self.device, dtype=torch.float32)
        current_seq = current_seq + mean_tensor.unsqueeze(0)  # Broadcast mean to all positions
        
        # Generate position indices
        positions = torch.arange(seq_len, device=self.device)
        
        with torch.no_grad():
            for _ in range(L):
                # Add batch dimension
                current_seq_batch = current_seq.unsqueeze(0)
                
                # Forward pass
                outputs = self.forward(current_seq_batch)
                outputs = outputs.squeeze(0)  # Remove batch dimension
                
                # Get the last output vector
                output_vector = outputs[-1]
                
                if discrete_mode:
                    # Discrete generation mode
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
        Calculate and print the number of learnable parameters
        """
        total_params = 0
        print("Model Parameter Counts:")
        
        in_dim = self.input_dim
        for l_idx, (layer, layer_type) in enumerate(zip(self.layers, self.layer_types)):
            out_dim = self.model_dims[l_idx]
            basis_dim = self.basis_dims[l_idx] if layer_type == 'ab' else 0
            
            # Parameter counts
            if layer_type == 'ab':
                linear_params = sum(p.numel() for p in layer.linear.parameters())
                A_params = layer.Acoeff.numel()
                norm_params = sum(p.numel() for p in layer.norm.parameters())
                residual_params = sum(p.numel() for p in layer.residual_proj.parameters()) if layer.residual_proj else 0
                
                layer_params = linear_params + A_params + norm_params + residual_params
                total_params += layer_params
                
                print(f"  Layer {l_idx} (AB, in_dim: {in_dim}, out_dim: {out_dim}, basis_dim: {basis_dim}, residual: {layer.use_residual}):")
                print(f"    Linear: {linear_params} params")
                print(f"    Acoeff: {A_params} params")
                print(f"    Bbasis: Fixed (non-trainable)")
                print(f"    LayerNorm: {norm_params} params")
                if layer.use_residual == 'separate':
                    print(f"    Residual projection: {residual_params} params")
            else:  # PM layer
                M_params = layer.M.numel()
                P_params = layer.P.numel()
                norm_params = sum(p.numel() for p in layer.norm.parameters())
                residual_params = sum(p.numel() for p in layer.residual_proj.parameters()) if layer.residual_proj else 0
                
                layer_params = M_params + P_params + norm_params + residual_params
                total_params += layer_params
                
                print(f"  Layer {l_idx} (PM, in_dim: {in_dim}, out_dim: {out_dim}, residual: {layer.use_residual}):")
                print(f"    M matrix: {M_params} params")
                print(f"    P matrix: {P_params} params")
                print(f"    Periods: Fixed (non-trainable)")
                print(f"    LayerNorm: {norm_params} params")
                if layer.use_residual == 'separate':
                    print(f"    Residual projection: {residual_params} params")
            
            print(f"    Layer total: {layer_params}")
            
            in_dim = out_dim  # Update for next layer
        
        print(f"Total parameters: {total_params}")
        return total_params
    
    def save(self, filename):
        """Save model state to file"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': {
                'input_dim': self.input_dim,
                'model_dims': self.model_dims,
                'basis_dims': self.basis_dims,
                'use_residual_list': [layer.use_residual for layer in self.layers],
                'layer_types': self.layer_types
            },
            'mean_t': self.mean_t if hasattr(self, 'mean_t') else None,
            'trained': self.trained
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
            use_residual_list=config.get('use_residual_list', None),
            layer_types=config.get('layer_types', None),
            device=device
        )
        
        # Load state
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load additional attributes
        if 'mean_t' in checkpoint:
            model.mean_t = checkpoint['mean_t']
        if 'trained' in checkpoint:
            model.trained = checkpoint['trained']
            
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
    input_dim = 10            # Input vector dimension
    model_dims = [8, 6, 4]    # Output dimensions for each layer
    basis_dims = [150, 100, 50] # Basis dimensions for AB layers
    use_residual_list = ['separate', 'separate', 'separate']  # Residual modes for each layer
    layer_types = ['ab', 'pm', 'ab']  # Layer types for each layer
    seq_count = 100            # Number of training sequences
    min_len = 100              # Minimum sequence length
    max_len = 200              # Maximum sequence length
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Generate training data
    print("\nGenerating training data...")
    print(f"Input dimension: {input_dim}, Model dims: {model_dims}, Basis dims: {basis_dims}")
    print(f"Residual modes: {use_residual_list}")
    print(f"Layer types: {layer_types}")
    seqs = []     # List of input sequences
    t_list = []   # List of target vectors (dimension = model_dims[-1])
    
    for i in range(seq_count):
        # Random sequence length
        length = random.randint(min_len, max_len)
        # Generate vector sequence
        sequence = np.random.uniform(-1, 1, (length, input_dim)).tolist()
        seqs.append(sequence)
        
        # Generate random target vector (dimension = last model_dim)
        target = np.random.uniform(-1, 1, model_dims[-1]).tolist()
        t_list.append(target)
    
    # Create model
    print("\nCreating Hierarchical HierDDmx model...")
    model = HierDDmx(
        input_dim=input_dim,
        model_dims=model_dims,
        basis_dims=basis_dims,
        use_residual_list=use_residual_list,
        layer_types=layer_types,
        device=device
    )
    
    # Show model structure
    print("\nModel structure:")
    model.count_parameters()
    
    # Train model
    print("\nTraining model...")
    history = model.grad_train(
        seqs,
        t_list,
        learning_rate=0.01,
        max_iters=50,
        tol=1e-88,
        decay_rate=0.98,
        print_every=10
    )
    
    # Evaluate predictions
    print("\nEvaluating predictions...")
    pred_t_list = [model.predict_t(seq) for seq in seqs]
    
    print("Prediction correlations per dimension:")
    corrs = []
    for d in range(model_dims[-1]):
        actuals = [t[d] for t in t_list]
        preds = [p[d] for p in pred_t_list]
        corr = correlation(actuals, preds)
        corrs.append(corr)
        print(f"  Dim {d}: correlation = {corr:.4f}")
    
    print(f"Average correlation: {np.mean(corrs):.4f}")     
    
    # Save and load demonstration
    print("\nSave and load demonstration:")
    model.save("hierarchical_model_mx.pth")
    
    # Load the model
    loaded_model = HierDDmx.load("hierarchical_model_mx.pth", device=device)
    
    # Verify loaded model works
    test_seq = seqs[0][:10]  # First 10 vectors of first sequence
    original_output = model.predict_t(test_seq)
    loaded_output = loaded_model.predict_t(test_seq)
    
    print(f"Original model prediction: {original_output}")
    print(f"Loaded model prediction: {loaded_output}")
    print(f"Predictions match: {np.allclose(original_output, loaded_output, atol=1e-6)}")

    # Example of using the new auto_train and generate methods
    print("\n" + "="*50)
    print("Example of using auto_train and generate methods")
    print("="*50)

    # Create a new model for auto-training
    auto_model = HierDDmx(
        input_dim=input_dim,
        model_dims=[10, 20, 10],
        basis_dims=[100, 80, 60],
        use_residual_list=use_residual_list,
        layer_types=['ab', 'pm', 'ab'],
        device=device
    )
    
    # Auto-train the model in 'gap' mode
    print("\nAuto-training model in 'gap' mode...")
    auto_history = auto_model.auto_train(
        seqs[:10],
        max_iters=20,
        learning_rate=0.01,
        auto_mode='gap',
        print_every=5
    )
    
    # Generate new sequences
    print("\nGenerating new sequences...")
    generated_seq = auto_model.generate(L=5, tau=0.1)
    print(f"Generated sequence shape: {generated_seq.shape}")
    print("First 5 generated vectors:")
    for i, vec in enumerate(generated_seq):
        print(f"Vector {i+1}: {[f'{x:.4f}' for x in vec]}")
    
    # Save and load model
    print("\nSaving and loading model...")
    auto_model.save("auto_model_mx.pth")
    loaded_model = HierDDmx.load("auto_model_mx.pth", device=device)
    
    # Generate with loaded model
    loaded_generated = loaded_model.generate(L=3, tau=0.05)
    print("Generated with loaded model:")
    for i, vec in enumerate(loaded_generated):
        print(f"Vector {i+1}: {[f'{x:.4f}' for x in vec]}")    
    
    print("Training and testing completed!")
        
    print("\n=== Hierarchical Vector Sequence Processing with Mixed AB and PM Layers Completed ===")
