# Copyright (C) 2005-2025, Bin-Guang Ma (mbg@mail.hzau.edu.cn); SPDX-License-Identifier: MIT
# The Hierarchical Numeric Dual Descriptor (Tensor form) with Linker matrices in PyTorch
# With layer normalization and residual connections and generation capability
# This program is for the demonstration of methodology and not fully refined.
# Author: Bin-Guang Ma (assisted by DeepSeek); Date: 2025-7-22

import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

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
        self.M = nn.Parameter(torch.randn(out_dim, in_dim) * 0.5)
        
        # Position-dependent transformation tensor
        self.P = nn.Parameter(torch.randn(out_dim, out_dim, num_basis) * 0.1)
        
        # Precompute periods tensor (fixed, not trainable)
        periods = torch.zeros(out_dim, out_dim, num_basis, dtype=torch.float32)
        for i in range(out_dim):
            for j in range(out_dim):
                for g in range(num_basis):
                    periods[i, j, g] = i*(out_dim*num_basis) + j*num_basis + g + 2
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
    
    def forward(self, seq):
        """Forward pass through all layers"""
        current = seq
        for layer in self.layers:
            current = layer(current)
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
    
    def grad_train(self, seqs, t_list, max_iters=1000, tol=1e-88, lr=0.01, 
                   decay_rate=1.0, print_every=10):
        """
        Train model using Adam optimizer
        Args:
            seqs: List of training sequences
            t_list: List of target vectors
            max_iters: Maximum training iterations
            lr: Initial learning rate
            decay_rate: Learning rate decay rate
            tol: Convergence tolerance
            print_every: Print interval
        
        Returns:
            Training loss history
        """
        # Set model to training mode
        self.train()
        self.trained = True
        
        # Setup optimizer and scheduler
        optimizer = optim.Adam(self.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)
        
        history = []
        prev_loss = float('inf')
        
        for it in range(max_iters):
            total_loss = 0.0
            n_batches = 0
            
            for seq, t in zip(seqs, t_list):
                # Convert to tensors
                seq_tensor = torch.tensor(seq, dtype=torch.float32, device=self.device)
                t_tensor = torch.tensor(t, dtype=torch.float32, device=self.device)
                
                # Forward pass
                output = self.forward(seq_tensor)
                
                # Prepare target - repeat for each position in output sequence
                target = t_tensor.expand_as(output)
                
                # Compute loss (mean squared error)
                loss = torch.mean((output - target) ** 2)
                
                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                n_batches += 1
            
            # Average loss for this epoch
            avg_loss = total_loss / n_batches
            history.append(avg_loss)             
            
            # Print progress
            if it % print_every == 0 or it == max_iters - 1:
                current_lr = scheduler.get_last_lr()[0]
                print(f"Iter {it:3d}: Loss = {avg_loss:.6e}, LR = {current_lr:.6f}")

            # Update learning rate
            scheduler.step()
            
            # Check convergence
            if abs(prev_loss - avg_loss) < tol:
                print(f"Converged after {it+1} iterations.")
                break
                
            prev_loss = avg_loss
        
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

    def auto_train(self, seqs, max_iters=100, tol=1e-6, learning_rate=0.01, 
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
            for l, layer in enumerate(self.layers):
                # Reinitialize parameters with small random values
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
            total_windows = 0
            for seq in seqs_tensor:
                total += torch.sum(seq, dim=0)
                total_windows += seq.size(0)
            self.mean_t = (total / total_windows).cpu().numpy()            
        
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
                
                # Forward pass
                current = seq
                intermediates = []
                
                for l, layer in enumerate(self.layers):
                    # Apply linear transformation
                    linear_out = torch.mm(current, layer.M.t())
                    
                    # Apply layer normalization with causal masking for reg mode
                    if auto_mode == 'reg':
                        normalized_out = []
                        for k in range(linear_out.size(0)):
                            # Use only historical positions for normalization
                            historical = linear_out[:k+1]
                            normalized = layer.ln(historical)
                            normalized_out.append(normalized[-1:])  # Take only the last position
                        normalized_out = torch.cat(normalized_out, dim=0)
                    else:
                        normalized_out = layer.ln(linear_out)
                    
                    # Position-dependent transformation
                    T = layer.position_transform(normalized_out)
                    
                    # Apply causal masking for Linker matrix in reg mode
                    if auto_mode == 'reg' and layer.Linker.requires_grad:
                        # Create causal mask
                        causal_mask = torch.tril(torch.ones_like(layer.Linker))
                        used_linker = layer.Linker * causal_mask
                    else:
                        used_linker = layer.Linker
                    
                    # Sequence transformation
                    U = torch.mm(used_linker.t(), T)
                    
                    # Handle residual connections
                    if layer.use_residual == 'separate':
                        residual_feat = layer.residual_proj(current)
                        if auto_mode == 'reg' and hasattr(layer, 'residual_linker'):
                            causal_mask_residual = torch.tril(torch.ones_like(layer.residual_linker))
                            residual = torch.mm(layer.residual_linker.t() * causal_mask_residual, residual_feat)
                        else:
                            residual = torch.mm(layer.residual_linker.t(), residual_feat)
                        current = U + residual
                    elif layer.use_residual == 'shared':
                        residual_feat = torch.mm(current, layer.M.t())
                        residual = torch.mm(used_linker.t(), residual_feat)
                        current = U + residual
                    else:
                        if current.shape == U.shape:
                            residual = torch.mm(used_linker.t(), current)
                            current = U + residual
                        else:
                            current = U
                    
                    intermediates.append({
                        'linear_out': linear_out,
                        'normalized_out': normalized_out,
                        'T': T,
                        'U': U,
                        'used_linker': used_linker
                    })
                
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
        # Fix: Create initial sequence with proper shape and values
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
                # Forward pass
                current = current_seq
                for layer in self.layers:
                    linear_out = torch.mm(current, layer.M.t())
                    normalized_out = layer.ln(linear_out)
                    T = layer.position_transform(normalized_out)
                    U = torch.mm(layer.Linker.t(), T)
                    
                    if layer.use_residual == 'separate':
                        residual_feat = layer.residual_proj(current)
                        residual = torch.mm(layer.residual_linker.t(), residual_feat)
                        current = U + residual
                    elif layer.use_residual == 'shared':
                        residual_feat = torch.mm(current, layer.M.t())
                        residual = torch.mm(layer.Linker.t(), residual_feat)
                        current = U + residual
                    else:
                        if current.shape == U.shape:
                            residual = torch.mm(layer.Linker.t(), current)
                            current = U + residual
                        else:
                            current = U
                
                # Get the last output vector
                output_vector = current[-1]
                
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
        """Count learnable parameters"""
        total_params = 0
        trainable_params = 0
        print("Parameter Count:")
        
        for l, layer in enumerate(self.layers):
            M_params = layer.M.numel()
            P_params = layer.P.numel()
            Linker_params = layer.Linker.numel()
            
            layer_params = M_params + P_params + Linker_params            
            
            # Count trainable parameters
            layer_trainable = M_params + P_params
            if layer.Linker.requires_grad:
                layer_trainable += Linker_params
                
            # Add residual parameters if using separate residual
            if layer.use_residual == 'separate':
                residual_proj_params = layer.residual_proj.weight.numel()
                residual_linker_params = layer.residual_linker.numel()
                layer_params += residual_proj_params + residual_linker_params                
                layer_trainable += residual_proj_params
                if layer.residual_linker.requires_grad:
                    layer_trainable += residual_linker_params

            total_params += layer_params            
            trainable_params += layer_trainable
            
            print(f"  Layer {l}:")
            print(f"    M matrix: {list(layer.M.shape)} = {M_params} parameters")
            print(f"    P tensor: {list(layer.P.shape)} = {P_params} parameters")
            print(f"    Linker matrix: {list(layer.Linker.shape)} = {Linker_params} parameters")
            if layer.use_residual == 'separate':
                print(f"    Residual proj: {list(layer.residual_proj.weight.shape)} = {residual_proj_params} parameters")
                print(f"    Residual linker: {list(layer.residual_linker.shape)} = {residual_linker_params} parameters")
            print(f"    Residual type: {layer.use_residual}")
            print(f"    Layer total: {layer_params} (trainable: {layer_trainable})")
        
        print(f"Total parameters: {total_params}")
        print(f"Trainable parameters: {trainable_params}")
        return total_params, trainable_params
    
    def save(self, filename):
        """Save model state to file"""
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
        checkpoint = torch.load(filename, map_location=device, weights_only=True)
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
    
    # Parameter count
    print("\nParameter count before training:")
    total_params, trainable_params = model_mixed.count_parameters()
    
    # Training
    print("\nTraining model...")
    history = model_mixed.grad_train(
        seqs, 
        t_list, 
        max_iters=50,
        lr=0.01,
        decay_rate=0.98,
        print_every=5
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
    
    # Test Case 2: Auto-training and sequence generation
    print("\n=== Test Case 2: Auto-training and Sequence Generation ===")
    
    # Create a new model with compatible dimensions for auto-training
    hndd_auto = HierDDLts(
        input_dim=input_dim,
        model_dims=[input_dim, 20, input_dim],  # Output dim must match input dim for reconstruction
        num_basis_list=[5]*3,
        input_seq_len=input_seq_len,
        linker_dims=[100, 50, 20],  
        linker_trainable=False,
        use_residual_list=[None]*3
    )
    
    # Auto-train in 'gap' mode (self-consistency)
    print("\nAuto-training in 'gap' mode:")
    auto_history = hndd_auto.auto_train(
        seqs,
        auto_mode='gap',
        max_iters=20,
        learning_rate=1.0,
        decay_rate=0.98,
        print_every=5
    )
    
    # Generate new sequence with temperature
    print("\nGenerating new sequences:")
    print("Deterministic generation (tau=0):")
    gen_det = hndd_auto.generate(L=10, tau=0)
    print(f"  Generated {len(gen_det)} vectors")
    print(f"  First vector: {[f'{x:.4f}' for x in gen_det[0][:min(5, input_dim)]]}...")
    
    print("\nStochastic generation (tau=0.5):")
    gen_stoch = hndd_auto.generate(L=10, tau=0.5)
    print(f"  Generated {len(gen_stoch)} vectors")
    print(f"  First vector: {[f'{x:.4f}' for x in gen_stoch[0][:min(5, input_dim)]]}...")
    
    # Test Case 3: Auto-training in 'reg' mode (auto-regressive)
    print("\n=== Test Case 3: Auto-regressive Training ===")
    
    # Auto-train in 'reg' mode (next-step prediction)
    print("\nAuto-training in 'reg' mode:")
    auto_history_reg = hndd_auto.auto_train(
        seqs,
        auto_mode='reg',
        max_iters=10,
        learning_rate=0.5,
        decay_rate=0.98,
        print_every=5,
        continued=True  # Continue training existing model
    )
    
    # Generate sequence using auto-regressive model
    print("\nGenerating sequence with auto-regressive model:")
    gen_reg = hndd_auto.generate(L=15, tau=0.3)
    print(f"Generated {len(gen_reg)} vectors")
    print(f"First vector: {[f'{x:.4f}' for x in gen_reg[0][:min(5, input_dim)]]}...")
    print(f"Last vector: {[f'{x:.4f}' for x in gen_reg[-1][:min(5, input_dim)]]}...")
    
    print("\nAll tests completed successfully!")
