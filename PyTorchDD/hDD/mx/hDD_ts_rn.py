# Copyright (C) 2005-2025, Bin-Guang Ma (mbg@mail.hzau.edu.cn); SPDX-License-Identifier: MIT
# Hierarchical Numeric Dual Descriptor (Mixed TS and RN Layers) implemented with PyTorch
# With layer normalization and residual connections and generation capability
# This program is for the demonstration of methodology and not fully refined.
# Author: Bin-Guang Ma (assisted by DeepSeek); Date: 2025-8-30

import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class LayerRN(nn.Module):
    """
    Single layer from HierDDrn (Random AB matrix form).
    
    Args:
        in_dim (int): Input dimension
        out_dim (int): Output dimension
        basis_dim (int): Basis dimension for this layer
        use_residual (str or None): Residual connection type. Options: 'separate', 'shared', or None.
        device (str): Device to run computations on ('cuda' or 'cpu')
    """
    def __init__(self, in_dim, out_dim, basis_dim, use_residual='separate', device='cuda'):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.basis_dim = basis_dim
        self.use_residual = use_residual
        self.device = device
        
        # Linear transformation from input to output dimension
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        
        # Layer normalization for output dimension
        self.norm = nn.LayerNorm(out_dim)
        
        # Coefficient matrix: out_dim x basis_dim
        self.Acoeff = nn.Parameter(torch.Tensor(out_dim, basis_dim))
        
        # Basis matrix: basis_dim x out_dim
        self.Bbasis = nn.Parameter(torch.Tensor(basis_dim, out_dim))
        
        # Initialize parameters
        nn.init.uniform_(self.linear.weight, -0.1, 0.1)
        nn.init.uniform_(self.Acoeff, -0.1, 0.1)
        nn.init.uniform_(self.Bbasis, -0.1, 0.1)
        
        # Residual projection processing
        if self.use_residual == 'separate' and in_dim != out_dim:
            self.residual_proj = nn.Linear(in_dim, out_dim, bias=False)
            nn.init.uniform_(self.residual_proj.weight, -0.1, 0.1)
        else:
            self.residual_proj = None
            
        self.to(device)
    
    def forward(self, x):
        """
        Forward pass for RN layer.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, in_dim) or (seq_len, in_dim)
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, out_dim) or (seq_len, out_dim)
        """
        # Handle batch input: if x has 3 dimensions, we flatten batch and seq_len for processing
        has_batch = True
        if x.dim() == 2:
            # If input is (seq_len, in_dim), add batch dimension
            x = x.unsqueeze(0)
            has_batch = False
        batch_size, seq_len, in_dim = x.shape
        
        # Save input for residual connection
        residual = x
        
        # Apply linear transformation
        transformed = self.linear(x)  # (batch_size, seq_len, out_dim)
        
        # Apply layer normalization
        normalized = self.norm(transformed)  # (batch_size, seq_len, out_dim)
        
        # Compute basis indices: j = k % basis_dim for each position
        j_indices = torch.arange(seq_len, device=self.device) % self.basis_dim
        
        # Select basis vectors: (seq_len, out_dim) -> expanded to (batch_size, seq_len, out_dim)
        B_vectors = self.Bbasis[j_indices]  # (seq_len, out_dim)
        B_vectors = B_vectors.unsqueeze(0).expand(batch_size, -1, -1)  # (batch_size, seq_len, out_dim)
        
        # Compute scalars: dot product of normalized and B_vectors along last dimension
        scalars = torch.einsum('bsd,bsd->bs', normalized, B_vectors)  # (batch_size, seq_len)
        
        # Select coefficient vectors: (seq_len, out_dim) -> expanded to (batch_size, seq_len, out_dim)
        A_vectors = self.Acoeff[:, j_indices].t()  # (seq_len, out_dim)
        A_vectors = A_vectors.unsqueeze(0).expand(batch_size, -1, -1)  # (batch_size, seq_len, out_dim)
        
        # Compute new features: scalars * A_vectors
        new_features = scalars.unsqueeze(2) * A_vectors  # (batch_size, seq_len, out_dim)
        
        # Residual connection processing 
        if self.use_residual == 'separate':
            if self.residual_proj is not None:
                residual = self.residual_proj(residual)
            out = new_features + residual
        elif self.use_residual == 'shared':        
            residual = self.linear(residual)
            out = new_features + residual
        else:
            if self.in_dim == self.out_dim and self.use_residual is not None:
                out = new_features + residual
            else:
                out = new_features
                
        # Remove batch dimension if input was without batch
        if not has_batch:
            out = out.squeeze(0)
        return out

class LayerTS(nn.Module):
    """
    Single layer from HierDDts (Tensor form).
    
    Args:
        in_dim (int): Input dimension
        out_dim (int): Output dimension
        num_basis (int): Number of basis functions
        use_residual (str or None): Residual connection type. Options: 'separate', 'shared', or None.
        device (str): Device to run computations on ('cuda' or 'cpu')
    """
    def __init__(self, in_dim, out_dim, num_basis, use_residual, device='cpu'):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_basis = num_basis
        self.use_residual = use_residual
        self.device = device
        
        # Linear transformation matrix (shared for main path and residual)
        self.M = nn.Parameter(torch.empty(out_dim, in_dim))
        
        # Tensor of basis coefficients
        self.P = nn.Parameter(torch.empty(out_dim, out_dim, num_basis))
        
        # Precompute periods tensor (fixed, not trainable)
        periods = torch.zeros(out_dim, out_dim, num_basis, device=device)
        for i in range(out_dim):
            for j in range(out_dim):
                for g in range(num_basis):
                    periods[i, j, g] = i*(out_dim*num_basis) + j*num_basis + g + 2
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
        if self.residual_proj is not None and isinstance(self.residual_proj, nn.Linear):            
            nn.init.uniform_(self.residual_proj.weight, -0.1, 0.1)
        
        self.to(device)
    
    def forward(self, x, positions):
        """
        Forward pass for TS layer.
        
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
        # Expand tensors for broadcasting: positions (seq_len, 1, 1, 1)
        k = positions.view(-1, 1, 1, 1).float()
        
        # Compute basis functions: phi = cos(2π * k / period)
        # periods shape: (1, out_dim, out_dim, num_basis)
        periods = self.periods.unsqueeze(0)
        phi = torch.cos(2 * math.pi * k / periods)  # (seq_len, out_dim, out_dim, num_basis)
        
        # Apply position-dependent transformation
        # x_trans: (batch_size, seq_len, out_dim) -> (batch_size, seq_len, 1, out_dim, 1)
        x_exp = x_trans.unsqueeze(2).unsqueeze(4)
        
        # P: (out_dim, out_dim, num_basis) -> (1, 1, out_dim, out_dim, num_basis)
        P_exp = self.P.unsqueeze(0).unsqueeze(0)
        
        # Compute: P * x' * phi
        # Result shape: (batch_size, seq_len, out_dim, out_dim, num_basis)
        product = P_exp * x_exp * phi
        
        # Sum over j (input dim) and g (basis index)
        Nk = torch.sum(product, dim=(3, 4))  # (batch_size, seq_len, out_dim)
       
        # Residual connection processing 
        if self.use_residual == 'separate':
            if self.residual_proj is not None:
                residual = self.residual_proj(x)
            else:
                residual = x
            out = Nk + residual            
        elif self.use_residual == 'shared':        
            residual = torch.matmul(x, self.M.t())
            out = Nk + residual
        else:
            if self.in_dim == self.out_dim and self.use_residual is not None:
                residual = x
                out = Nk + residual
            else:
                out = Nk
                
        return out

class HierDDmx(nn.Module):
    """
    Hierarchical Numeric Dual Descriptor with mixed layer types (RN and TS).
    
    Args:
        input_dim (int): Dimension of input vectors
        model_dims (list): List of output dimensions for each layer
        layer_types (list): List of layer types for each layer, 'rn' or 'ts'
        basis_list (list): List of basis dimensions for each layer. For 'rn' layers, this is basis_dim; for 'ts' layers, this is num_basis.
        use_residual_list (list): Residual connection type for each layer
        device (str): Device to run computations on ('cuda' or 'cpu')
    """
    def __init__(self, input_dim=4, model_dims=[4], layer_types=['rn'], basis_list=[50], 
                 use_residual_list=None, device='cuda'):
        super().__init__()
        self.input_dim = input_dim
        self.model_dims = model_dims
        self.layer_types = layer_types
        self.basis_list = basis_list
        self.num_layers = len(model_dims)
        self.device = device
        self.trained = False
        
        # Validate dimensions
        if len(layer_types) != self.num_layers or len(basis_list) != self.num_layers:
            raise ValueError("layer_types and basis_list length must match model_dims length")
        
        # Set default residual connection types
        if use_residual_list is None:
            self.use_residual_list = ['separate'] * self.num_layers
        else:
            self.use_residual_list = use_residual_list
            
        # Create layers
        self.layers = nn.ModuleList()
        in_dim = input_dim
        for i in range(self.num_layers):
            out_dim = model_dims[i]
            layer_type = layer_types[i]
            basis_val = basis_list[i]
            use_residual = self.use_residual_list[i]
            
            if layer_type == 'rn':
                layer = LayerRN(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    basis_dim=basis_val,
                    use_residual=use_residual,
                    device=device
                )
            elif layer_type == 'ts':
                layer = LayerTS(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    num_basis=basis_val,
                    use_residual=use_residual,
                    device=device
                )
            else:
                raise ValueError("layer_type must be 'rn' or 'ts'")
                
            self.layers.append(layer)
            in_dim = out_dim  # Next layer's input is this layer's output
        
        self.to(device)
    
    def forward(self, x):
        """
        Forward pass through the entire hierarchical model.
        
        Args:
            x (torch.Tensor): Input sequence of shape (batch_size, seq_len, input_dim) or (seq_len, input_dim)
        
        Returns:
            torch.Tensor: Output sequence of shape (batch_size, seq_len, model_dims[-1]) or (seq_len, model_dims[-1])
        """
        # Check if input has batch dimension
        has_batch = True
        if x.dim() == 2:
            x = x.unsqueeze(0)
            has_batch = False
            
        batch_size, seq_len, _ = x.shape
        
        # Generate position indices for TS layers
        positions = torch.arange(seq_len, device=self.device)
        
        current = x
        for layer in self.layers:
            if isinstance(layer, LayerTS):
                current = layer(current, positions)
            else:  # LayerRN
                # For RN layers, we process without positions
                current = layer(current)
        
        if not has_batch:
            current = current.squeeze(0)
        return current
    
    def describe(self, vec_seq):
        """
        Compute descriptor vectors for each position in the sequence.
        
        Args:
            vec_seq (list): List of input vectors (each vector is list of floats)
        
        Returns:
            list: List of output vectors from the final layer
        """
        # Convert to tensor and move to device
        input_tensor = torch.tensor(vec_seq, dtype=torch.float32).to(self.device)
        
        # Forward pass
        output_tensor = self.forward(input_tensor)
        
        # Convert back to list of vectors
        return output_tensor.cpu().detach().numpy().tolist()
    
    def deviation(self, seqs, t_list):
        """
        Compute mean squared error between descriptors and targets using batch processing.
        
        Args:
            seqs (list): List of input sequences
            t_list (list): List of target vectors (must match final layer dimension)
        
        Returns:
            float: Mean squared error
        """
        # Convert all sequences and targets to tensors in one go
        seqs_tensor = [torch.tensor(seq, dtype=torch.float32, device=self.device) for seq in seqs]
        targets_tensor = torch.tensor(t_list, dtype=torch.float32, device=self.device)
        
        total_loss = 0.0
        total_positions = 0
        
        # Process all sequences in a batch-like manner
        for i, seq_tensor in enumerate(seqs_tensor):
            # Forward pass
            outputs = self.forward(seq_tensor)
            
            # Compute average output for this sequence
            avg_output = outputs.mean(dim=0)
            
            # Compute MSE between average output and target
            loss = torch.mean((avg_output - targets_tensor[i]) ** 2)
            total_loss += loss.item() * len(seq_tensor)
            total_positions += len(seq_tensor)
        
        return total_loss / total_positions if total_positions > 0 else 0.0

    def grad_train(self, seqs, t_list, max_iters=1000, tol=1e-8,
                    learning_rate=0.01, decay_rate=1.0, print_every=10):
        """
        Train the model using Adam optimizer with learning rate decay and early stopping.
        
        Args:
            seqs (list): List of training sequences
            t_list (list): List of target vectors (must match final layer dimension)
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
        
        # Convert all data to tensors in one go
        seqs_tensor = [torch.tensor(seq, dtype=torch.float32, device=self.device) for seq in seqs]
        targets_tensor = torch.tensor(t_list, dtype=torch.float32, device=self.device)
        
        for it in range(max_iters):
            total_loss = 0.0
            total_positions = 0
            
            # Training pass - process all sequences
            self.train()
            for i, seq_tensor in enumerate(seqs_tensor):
                # Forward pass
                outputs = self.forward(seq_tensor)
                
                # Compute average output for this sequence
                avg_output = outputs.mean(dim=0)
                
                # Compute loss (MSE between average output and target)
                loss = torch.mean((avg_output - targets_tensor[i]) ** 2)
                total_loss += loss.item() * len(seq_tensor)
                total_positions += len(seq_tensor)
                
                # Backward pass and optimization
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
        Predict target vector as average of all final layer outputs.
        
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
        Self-training for the hierarchical model with two modes: 'gap' or 'reg'.
        
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
                if isinstance(layer, LayerRN):
                    nn.init.uniform_(layer.linear.weight, -0.5, 0.5)
                    nn.init.uniform_(layer.Acoeff, -0.1, 0.1)
                    nn.init.uniform_(layer.Bbasis, -0.1, 0.1)
                    if layer.residual_proj is not None:
                        nn.init.uniform_(layer.residual_proj.weight, -0.1, 0.1)
                else:  # LayerTS
                    nn.init.uniform_(layer.M, -0.5, 0.5)
                    nn.init.uniform_(layer.P, -0.1, 0.1)
                    if layer.residual_proj is not None and isinstance(layer.residual_proj, nn.Linear):
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
                
                # For 'reg' mode with causal masking: process each position sequentially
                if auto_mode == 'reg':
                    # Process each position with causal masking
                    loss = 0.0
                    valid_positions = 0
                    
                    for k in range(seq.size(0) - 1):  # Skip last position
                        # Use only the prefix up to position k for prediction
                        prefix = seq[:k+1]  # Input sequence up to current position
                        
                        # Forward pass with causal masking
                        outputs = self.forward(prefix)
                        
                        # Get the prediction for the next position (last output)
                        prediction = outputs[-1]
                        target = seq[k + 1]
                        
                        # Calculate MSE loss
                        position_loss = torch.sum((prediction - target) ** 2)
                        loss += position_loss
                        valid_positions += 1
                    
                    if valid_positions > 0:
                        loss = loss / valid_positions
                        total_loss += loss.item()
                        loss.backward()
                        optimizer.step()
                
                else:  # 'gap' mode
                    # Forward pass
                    outputs = self.forward(seq)
                    
                    # Calculate loss for gap mode
                    loss = 0.0
                    valid_positions = 0
                    
                    for k in range(outputs.size(0)):
                        target = seq[k]
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
        
        with torch.no_grad():
            for _ in range(L):
                # Forward pass
                outputs = self.forward(current_seq)
                
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
        Calculate and print the number of learnable parameters.
        """
        total_params = 0
        print("Model Parameter Counts:")
        
        # Track input dimensions for each layer
        in_dims = [self.input_dim] + self.model_dims[:-1]
        
        for l_idx, (in_dim, out_dim, basis_val, layer_type, layer) in enumerate(zip(
            in_dims, self.model_dims, self.basis_list, self.layer_types, self.layers)):
            
            layer_params = 0
            print(f"  Layer {l_idx} (type: {layer_type}, in_dim: {in_dim}, out_dim: {out_dim}, basis: {basis_val}):")
            
            if layer_type == 'rn':
                # Count parameters for RN layer
                linear_params = sum(p.numel() for p in layer.linear.parameters())
                A_params = layer.Acoeff.numel()
                B_params = layer.Bbasis.numel()
                norm_params = sum(p.numel() for p in layer.norm.parameters())
                residual_params = 0
                if layer.residual_proj is not None:
                    residual_params = sum(p.numel() for p in layer.residual_proj.parameters())
                
                layer_params = linear_params + A_params + B_params + norm_params + residual_params
                total_params += layer_params
                
                print(f"    Linear: {linear_params} params")
                print(f"    Acoeff: {A_params} params")
                print(f"    Bbasis: {B_params} params")
                print(f"    LayerNorm: {norm_params} params")
                print(f"    Residual Proj: {residual_params} params")
                
            else:  # 'ts'
                # Count parameters for TS layer
                M_params = layer.M.numel()
                P_params = layer.P.numel()
                norm_params = sum(p.numel() for p in layer.norm.parameters())
                residual_params = 0
                if layer.residual_proj is not None:
                    residual_params = sum(p.numel() for p in layer.residual_proj.parameters())
                
                layer_params = M_params + P_params + norm_params + residual_params
                total_params += layer_params
                
                print(f"    M: {M_params} params")
                print(f"    P: {P_params} params")
                print(f"    LayerNorm: {norm_params} params")
                print(f"    Residual Proj: {residual_params} params")
            
            print(f"    Layer total: {layer_params}")
        
        print(f"Total parameters: {total_params}")
        return total_params
    
    def save(self, filename):
        """Save model state to file"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': {
                'input_dim': self.input_dim,
                'model_dims': self.model_dims,
                'layer_types': self.layer_types,
                'basis_list': self.basis_list,
                'use_residual_list': self.use_residual_list
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
            basis_list=config['basis_list'],
            use_residual_list=config.get('use_residual_list', None),
            device=device
        )
        
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
    input_dim = 10       # Input vector dimension
    model_dims = [16, 8, 4] # Output dimensions for each layer
    layer_types = ['rn', 'ts', 'rn']  # Layer types for each layer
    basis_list = [150, 5, 100]  # For RN layers: basis_dim, for TS layers: num_basis
    use_residual_list = ['separate', 'shared', 'separate']  # Residual types for each layer
    seq_count = 100      # Number of training sequences
    min_len = 100        # Minimum sequence length
    max_len = 200        # Maximum sequence length
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Generate training data
    print("\nGenerating training data...")
    print(f"Input dimension: {input_dim}, Model dims: {model_dims}, Layer types: {layer_types}")
    print(f"Basis list: {basis_list}, Residual types: {use_residual_list}")
    seqs = []     # List of input sequences
    t_list = []   # List of target vectors (dimension = model_dims[-1])
    
    for _ in range(seq_count):
        L = random.randint(min_len, max_len)
        # Generate n-dimensional input sequence
        seq = [[random.uniform(-1, 1) for _ in range(input_dim)] for __ in range(L)]
        seqs.append(seq)
        # Generate m-dimensional target vector
        t_list.append([random.uniform(-1, 1) for _ in range(model_dims[-1])])
    
    # Create model with mixed layer types
    print("\nCreating Hierarchical HierDDmx model...")
    model = HierDDmx(
        input_dim=input_dim,
        model_dims=model_dims,
        layer_types=layer_types,
        basis_list=basis_list,
        use_residual_list=use_residual_list,
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
        learning_rate=0.005,
        max_iters=100,
        tol=1e-18,
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
    
    # Save and load model
    print("\n=== Testing Save/Load Functionality ===")
    model.save("test_model_mx.pth")
    
    # Load the model
    loaded_model = HierDDmx.load("test_model_mx.pth", device=device)
    
    # Verify loaded model works
    test_output = loaded_model.predict_t(seqs[0])
    print(f"Loaded model prediction: {test_output[:3]}...")  # Show first 3 elements

    # Example of using the new auto_train and generate methods
    print("\n" + "="*50)
    print("Example of using auto_train and generate methods")
    print("="*50)

    # Create a new model for auto-training
    auto_model = HierDDmx(
        input_dim=input_dim,
        model_dims=[10, 20, 10],
        layer_types=['ts', 'rn', 'ts'],
        basis_list=[5, 100, 3],
        use_residual_list=use_residual_list,
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
    
    print("\n=== Hierarchical Vector Sequence Processing Completed ===")
