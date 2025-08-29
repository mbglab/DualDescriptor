# Copyright (C) 2005-2025, Bin-Guang Ma (mbg@mail.hzau.edu.cn); SPDX-License-Identifier: MIT
# The Hierarchical Numeric Dual Descriptor (AB matrix form) with Linker matrices in PyTorch
# With layer normalization and residual connections and generation capability
# This program is for the demonstration of methodology and not fully refined.
# Author: Bin-Guang Ma (assisted by DeepSeek); Date: 2025-8-26

import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

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
        use_residual (str): Residual connection type ('separate', 'shared', or None)
        device (str): Device to run the layer on ('cuda' or 'cpu')
    """
    def __init__(self, in_dim, out_dim, basis_dim, in_seq_len, out_seq_len, 
                 linker_trainable, use_residual=None, device='cuda'):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.basis_dim = basis_dim
        self.in_seq_len = in_seq_len
        self.out_seq_len = out_seq_len
        self.use_residual = use_residual
        self.linker_trainable = linker_trainable
        self.device = device
        
        # Initialize transformation matrix M
        self.M = nn.Parameter(torch.empty(out_dim, in_dim, device=device))
        nn.init.uniform_(self.M, -0.1, 0.1)
        
        # Initialize coefficient matrix Acoeff
        self.Acoeff = nn.Parameter(torch.empty(out_dim, basis_dim, device=device))
        nn.init.uniform_(self.Acoeff, -0.1, 0.1)
        
        # Fixed basis matrix Bbasis (non-trainable)
        Bbasis = torch.empty(basis_dim, out_dim, device=device)
        for k in range(basis_dim):
            for i in range(out_dim):
                Bbasis[k, i] = math.cos(2 * math.pi * (k + 1) / (i + 1))
        self.register_buffer('Bbasis', Bbasis)
        
        # Initialize Linker matrix
        self.Linker = nn.Parameter(torch.empty(in_seq_len, out_seq_len, device=device))
        nn.init.uniform_(self.Linker, -0.1, 0.1)
        
        # Freeze Linker if not trainable
        if not linker_trainable:
            self.Linker.requires_grad = False
            
        # Layer normalization
        self.layer_norm = nn.LayerNorm(out_dim, device=device)
        
        # Handle different residual connection types
        if self.use_residual == 'separate':
            # Separate projection and linker for residual path
            self.residual_proj = nn.Linear(in_dim, out_dim, bias=False, device=device)
            nn.init.uniform_(self.residual_proj.weight, -0.1, 0.1)
            
            self.residual_linker = nn.Parameter(torch.empty(in_seq_len, out_seq_len, device=device))
            nn.init.uniform_(self.residual_linker, -0.1, 0.1)
            if not linker_trainable:
                self.residual_linker.requires_grad = False
        else:
            self.residual_proj = None
            self.residual_linker = None
            
        # Precompute basis indices for sequence positions
        self.register_buffer('basis_indices', 
                            torch.tensor([i % basis_dim for i in range(in_seq_len)], device=device))
        
    def forward(self, x):
        """
        Forward pass for the layer
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, in_seq_len, in_dim)
            
        Returns:
            Tensor: Output tensor of shape (batch_size, out_seq_len, out_dim)
        """
        # Main path processing
        # Apply linear transformation
        z = torch.matmul(x, self.M.T)
        
        # Apply layer normalization
        z = self.layer_norm(z)

        # Position-wise processing
        # Select basis vectors for each position: (in_seq_len, out_dim)
        basis_vectors = self.Bbasis[self.basis_indices]
        
        # Compute scalars: (batch_size, in_seq_len)
        scalars = torch.einsum('bsd,sd->bs', z, basis_vectors)
        
        # Select coefficients for each position: (in_seq_len, out_dim)
        coeffs = self.Acoeff[:, self.basis_indices].permute(1, 0)
        
        # Compute position outputs: (batch_size, in_seq_len, out_dim)
        u = coeffs * scalars.unsqueeze(-1)
        
        # Apply sequence length transformation using Linker matrix
        # (batch_size, in_seq_len, out_dim) x (in_seq_len, out_seq_len) -> (batch_size, out_dim, out_seq_len)
        v = torch.matmul(u.permute(0, 2, 1), self.Linker)
        
        # Transpose to (batch_size, out_seq_len, out_dim)
        main_output = v.permute(0, 2, 1)
        
        # Process residual connections
        if self.use_residual == 'separate':
            # Separate projection and linker for residual
            residual_feat = self.residual_proj(x)
            residual = torch.matmul(
                residual_feat.permute(0, 2, 1), 
                self.residual_linker
            ).permute(0, 2, 1)
            return main_output + residual
            
        elif self.use_residual == 'shared':
            # Shared M and Linker for residual
            residual_feat = torch.matmul(x, self.M.T)
            residual = torch.matmul(
                residual_feat.permute(0, 2, 1), 
                self.Linker
            ).permute(0, 2, 1)
            return main_output + residual
            
        elif self.use_residual is None:
            # Identity residual only if dimensions match
            if self.in_dim == self.out_dim and self.in_seq_len == self.out_seq_len:
                residual = torch.matmul(
                    x.permute(0, 2, 1), 
                    self.Linker
                ).permute(0, 2, 1)
                return main_output + residual
            else:
                return main_output
                
        return main_output

class HierDDLab(nn.Module):
    """
    Hierarchical Numeric Dual Descriptor with Linker for vector sequences
    
    Args:
        input_dim (int): Dimension of input vectors
        model_dims (list): Output dimensions for each layer
        basis_dims (list): Basis dimensions for each layer
        linker_dims (list): Output sequence lengths for each layer
        input_seq_len (int): Fixed input sequence length
        linker_trainable (bool|list): Controls if Linker matrices are trainable
        use_residual (str|list): Residual connection type ('separate', 'shared', or None)
        device (str): Device to run the model on ('cuda' or 'cpu')
    """
    def __init__(self, input_dim=4, model_dims=[4], basis_dims=[50], 
                 linker_dims=[50], input_seq_len=100, linker_trainable=False,
                 use_residual=None, device='cuda'):
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
        
        # Process use_residual parameter
        if use_residual is None or isinstance(use_residual, str):
            self.use_residual = [use_residual] * self.num_layers
        elif isinstance(use_residual, list):
            if len(use_residual) != self.num_layers:
                raise ValueError("use_residual list length must match number of layers")
            self.use_residual = use_residual
        else:
            raise TypeError("use_residual must be str, None, or list of str/None")
        
        # Create layers
        self.layers = nn.ModuleList()
        for i, out_dim in enumerate(model_dims):
            # Determine input dimensions for this layer
            in_dim = input_dim if i == 0 else model_dims[i-1]
            in_seq = input_seq_len if i == 0 else linker_dims[i-1]
            out_seq = linker_dims[i]
            
            layer = Layer(
                in_dim=in_dim,
                out_dim=out_dim,
                basis_dim=basis_dims[i],
                in_seq_len=in_seq,
                out_seq_len=out_seq,
                linker_trainable=self.linker_trainable[i],
                use_residual=self.use_residual[i],
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
        else:
            vec_seq = vec_seq.to(self.device)
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
    
    def train_model(self, seqs, t_list, max_iters=1000, tol=1e-88, 
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
                # Reinitialize parameters with small random values
                nn.init.uniform_(layer.M, -0.5, 0.5)
                nn.init.uniform_(layer.Acoeff, -0.1, 0.1)
                
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
                
                # Forward pass through all layers
                current = seq.unsqueeze(0)  # Add batch dimension
                for layer in self.layers:
                    current = layer(current)
                
                # Remove batch dimension
                current = current.squeeze(0)
                
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
                current = current_seq.unsqueeze(0)  # Add batch dimension
                for layer in self.layers:
                    current = layer(current)
                
                # Remove batch dimension
                current = current.squeeze(0)
                
                # Get the last output vector
                output_vector = current[-1]
                
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
            L_i = self.basis_dims[l_idx]
            out_seq = self.linker_dims[l_idx]
            
            print(f"  Layer {l_idx} (in_dim: {in_dim}, out_dim: {out_dim}, L: {L_i}, in_seq: {in_seq}, out_seq: {out_seq}):")
            print(f"    Parameters: {layer_params} (Trainable: {layer_trainable})")
            print(f"    Residual type: {self.use_residual[l_idx]}")
            
            # Count residual parameters if using separate residual
            if self.use_residual[l_idx] == 'separate':
                residual_params = layer.residual_proj.weight.numel()
                residual_params += layer.residual_linker.numel()
                print(f"    Residual params: {residual_params} (Trainable: {residual_params if layer.residual_linker.requires_grad else 0})")
        
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
                'basis_dims': self.basis_dims,
                'linker_dims': self.linker_dims,
                'input_seq_len': self.input_seq_len,
                'linker_trainable': self.linker_trainable,
                'use_residual': self.use_residual,
                'device': self.device
            },
            'mean_t': self.mean_t if hasattr(self, 'mean_t') else None            
        }, filename)
        print(f"Model saved to {filename}")
    
    @classmethod
    def load(cls, filename, device=None):
        """Load model from file"""
        # Device configuration
        if device is None:
            # Try to get device from saved config, default to cuda if available
            checkpoint = torch.load(filename, map_location='cpu', weights_only=False)
            device = checkpoint['config'].get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
            device = torch.device(device)
        
        # Load checkpoint
        checkpoint = torch.load(filename, map_location=device, weights_only=False)
        config = checkpoint['config']
        
        # Create model instance
        model = cls(
            input_dim=config['input_dim'],
            model_dims=config['model_dims'],
            basis_dims=config['basis_dims'],
            linker_dims=config['linker_dims'],
            input_seq_len=config['input_seq_len'],
            linker_trainable=config['linker_trainable'],
            use_residual=config['use_residual'],
            device=device
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
    model_dims = [8, 6, 3] # Output dimensions for each layer
    basis_dims = [150, 100, 50] # Basis dimensions for each layer
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

    L = input_seq_len
    for i in range(seq_count):
        # Generate vector sequence with fixed length
        sequence = np.random.uniform(-1, 1, (input_seq_len, input_dim))
        seqs.append(sequence)
        
        # Generate random target vector
        target = np.random.uniform(-1, 1, model_dims[-1])
        t_list.append(target)
    
    # Test Case 1: Mixed residual strategies
    print("\n=== Test Case 1: Mixed Residual Strategies ===")
    model_mixed = HierDDLab(
        input_dim=input_dim,
        model_dims=model_dims,
        basis_dims=basis_dims,
        linker_dims=linker_dims,
        input_seq_len=input_seq_len,
        linker_trainable=[True, False, True],
        use_residual=['separate', 'shared', None],  # Different residual for each layer
        device=device
    )
    
    # Show parameter counts
    print("\nModel parameter counts:")
    total_params, trainable_params = model_mixed.count_parameters()
    
    # Train model
    print("\nTraining model...")
    history = model_mixed.train_model(
        seqs, 
        t_list,
        learning_rate=0.01,
        tol=1e-88,
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
    
    # Test Case 2: Save and load model
    print("\n=== Test Case 3: Save and Load Model ===")
    
    # Save model
    model_mixed.save("hier_ddl_model.pth")
    
    # Load model
    loaded_model = HierDDLab.load("hier_ddl_model.pth", device=device)
    
    # Verify loaded model works
    test_pred = loaded_model.predict_t(seqs[0])
    print(f"Original prediction: {pred_t_list[0][:3]}...")
    print(f"Loaded model prediction: {test_pred[:3]}...")

    # Test Case 2: Example of using the new auto_train and generate methods
    print("\n" + "="*50)
    print("Example of using auto_train and generate methods")
    print("="*50)

    # Create a new model for auto-training
    auto_model = HierDDLab(
        input_dim=input_dim,
        model_dims=[10, 20, 10],
        basis_dims=basis_dims,
        linker_dims=linker_dims,
        input_seq_len=input_seq_len,
        linker_trainable=[True, False, True],
        use_residual=['separate', 'shared', None],  # Different residual for each layer
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
    auto_model.save("auto_model.pth")
    loaded_model = HierDDLab.load("auto_model.pth", device=device)
    
    # Generate with loaded model
    loaded_generated = loaded_model.generate(L=3, tau=0.05)
    print("Generated with loaded model:")
    for i, vec in enumerate(loaded_generated):
        print(f"Vector {i+1}: {[f'{x:.4f}' for x in vec]}")    
    
    print("Training and testing completed!")
