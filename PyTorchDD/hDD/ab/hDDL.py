# Copyright (C) 2005-2025, Bin-Guang Ma (mbg@mail.hzau.edu.cn). All rights reserved.
# The Hierarchical Numeric Dual Descriptor class with Linker Matrices for Sequence Length Transformation
# Modified to support hierarchical structure with multiple layers and sequence length transformation
# PyTorch GPU Accelerated Version
# Author: Bin-Guang Ma (assisted by DeepSeek); Date: 2025-7-7

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import pickle

class HierDDLabLayer(nn.Module):
    """
    A single layer of the Hierarchical Numeric Dual Descriptor with Linker
    
    Args:
        in_dim (int): Input dimension
        out_dim (int): Output dimension
        basis_dim (int): Basis dimension
        in_seq_len (int): Input sequence length
        out_seq_len (int): Output sequence length
        linker_trainable (bool): Whether linker matrix is trainable
        use_residual (bool): Whether to use residual connections
    """
    def __init__(self, in_dim, out_dim, basis_dim, in_seq_len, out_seq_len, 
                 linker_trainable, use_residual=True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.basis_dim = basis_dim
        self.in_seq_len = in_seq_len
        self.out_seq_len = out_seq_len
        self.use_residual = use_residual
        self.linker_trainable = linker_trainable
        
        # Initialize transformation matrix M
        self.M = nn.Parameter(torch.empty(out_dim, in_dim))
        nn.init.uniform_(self.M, -0.1, 0.1)
        
        # Initialize coefficient matrix Acoeff
        self.Acoeff = nn.Parameter(torch.empty(out_dim, basis_dim))
        nn.init.uniform_(self.Acoeff, -0.1, 0.1)
        
        # Initialize basis matrix Bbasis
        self.Bbasis = nn.Parameter(torch.empty(basis_dim, out_dim))
        nn.init.uniform_(self.Bbasis, -0.1, 0.1)
        
        # Initialize Linker matrix
        self.Linker = nn.Parameter(torch.empty(in_seq_len, out_seq_len))
        nn.init.uniform_(self.Linker, -0.1, 0.1)
        
        # Freeze Linker if not trainable
        if not linker_trainable:
            self.Linker.requires_grad = False
            
        # Layer normalization
        self.layer_norm = nn.LayerNorm(out_dim)
        
        # Residual projection if dimensions change
        if use_residual and in_dim != out_dim:
            self.residual_proj = nn.Linear(in_dim, out_dim, bias=False)
            nn.init.uniform_(self.residual_proj.weight, -0.1, 0.1)
        else:
            self.residual_proj = None
            
        # Precompute basis indices for sequence positions
        self.register_buffer('basis_indices', 
                            torch.tensor([i % basis_dim for i in range(in_seq_len)]))
        
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
        
        # Apply linear transformation: (batch_size, in_seq_len, in_dim) -> (batch_size, in_seq_len, out_dim)
        x = torch.matmul(x, self.M.T)
        
        # Apply layer normalization
        x = self.layer_norm(x)

        # Apply residual connection
        if self.use_residual:
            if self.residual_proj is not None:
                residual = self.residual_proj(residual)
            x = x + residual
                   
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
        return v.permute(0, 2, 1)

class HierDDLabTorch(nn.Module):
    """
    Hierarchical Numeric Dual Descriptor with Linker for vector sequences
    
    Args:
        input_dim (int): Dimension of input vectors
        model_dims (list): Output dimensions for each layer
        basis_dims (list): Basis dimensions for each layer
        linker_dims (list): Output sequence lengths for each layer
        input_seq_len (int): Fixed input sequence length
        linker_trainable (bool|list): Controls if Linker matrices are trainable
        use_residual (bool): Whether to use residual connections
    """
    def __init__(self, input_dim=4, model_dims=[4], basis_dims=[50], 
                 linker_dims=[50], input_seq_len=100, linker_trainable=False,
                 use_residual=True):
        super().__init__()
        self.input_dim = input_dim
        self.model_dims = model_dims
        self.basis_dims = basis_dims
        self.linker_dims = linker_dims
        self.input_seq_len = input_seq_len
        self.num_layers = len(model_dims)
        self.use_residual = use_residual
        self.trained = False
        
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
        
        # Create layers
        self.layers = nn.ModuleList()
        for i, out_dim in enumerate(model_dims):
            # Determine input dimensions for this layer
            in_dim = input_dim if i == 0 else model_dims[i-1]
            in_seq = input_seq_len if i == 0 else linker_dims[i-1]
            out_seq = linker_dims[i]
            
            layer = HierDDLabLayer(
                in_dim=in_dim,
                out_dim=out_dim,
                basis_dim=basis_dims[i],
                in_seq_len=in_seq,
                out_seq_len=out_seq,
                linker_trainable=self.linker_trainable[i],
                use_residual=use_residual
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
            vec_seq = torch.tensor(vec_seq, dtype=torch.float32)
        # Move to same device as model
        device = next(self.parameters()).device
        vec_seq = vec_seq.to(device)
        x = vec_seq.unsqueeze(0)  # (1, seq_len, input_dim)
        
        # Forward pass
        with torch.no_grad():
            output = self.forward(x)
            
        # Remove batch dimension and convert to list of vectors
        return output.squeeze(0).tolist()
    
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
    
    def compute_loss(self, seqs, t_list):
        """
        Compute mean squared error loss
        
        Args:
            seqs (list): List of sequences, each is (input_seq_len, input_dim)
            t_list (list): List of target vectors (model_dims[-1])
            
        Returns:
            Tensor: Mean squared error loss
        """
        # Get device of model
        device = next(self.parameters()).device
        
        # Convert to tensors and move to device
        seqs_tensor = torch.tensor(np.array(seqs), dtype=torch.float32).to(device)
        targets = torch.tensor(np.array(t_list), dtype=torch.float32).to(device)
        
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
            loss = self.compute_loss(seqs, t_list)
            
            # Backpropagate
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            
            # Update parameters
            optimizer.step()
            
            # Decay learning rate
            scheduler.step()
            
            # Record loss
            current_loss = loss.item()
            history.append(current_loss)
            
            # Print progress
            if it % print_every == 0 or it == max_iters - 1:
                lr = optimizer.param_groups[0]['lr']
                print(f"Iter {it:3d}: Loss = {current_loss:.6f}, LR = {lr:.6f}")
            
            # Check convergence
            if current_loss < best_loss:
                best_loss = current_loss
            elif best_loss - current_loss < tol:
                print(f"Converged after {it+1} iterations")
                break
                
        self.trained = True
        return history
    
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
        
        print(f"Total parameters: {total_params}")
        print(f"Trainable parameters: {trainable_params}")        
        return total_params, trainable_params
    
    def save(self, filename):
        """Save model to file"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': {
                'input_dim': self.input_dim,
                'model_dims': self.model_dims,
                'basis_dims': self.basis_dims,
                'linker_dims': self.linker_dims,
                'input_seq_len': self.input_seq_len,
                'linker_trainable': self.linker_trainable,
                'use_residual': self.use_residual
            }
        }, filename)
        print(f"Model saved to {filename}")
    
    @classmethod
    def load(cls, filename, device='cpu'):
        """Load model from file"""
        checkpoint = torch.load(filename, map_location=device)
        config = checkpoint['config']
        model = cls(
            input_dim=config['input_dim'],
            model_dims=config['model_dims'],
            basis_dims=config['basis_dims'],
            linker_dims=config['linker_dims'],
            input_seq_len=config['input_seq_len'],
            linker_trainable=config['linker_trainable'],
            use_residual=config['use_residual']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.trained = True
        model = model.to(device)
        print(f"Model loaded from {filename}")
        return model

# === Example Usage ===
if __name__ == "__main__":

    from scipy.stats import pearsonr
    
    # Set random seeds for reproducibility
    #torch.manual_seed(1)
    #np.random.seed(1)
    #random.seed(1)
    
    # Configuration
    input_dim = 10          # Input vector dimension
    input_seq_len = 100    # Fixed input sequence length
    model_dims = [8, 6, 3]    # Output dimensions for each layer
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

    L = input_seq_len
    for i in range(seq_count):
        # Generate vector sequence with fixed length
        sequence = np.random.uniform(-1, 1, (input_seq_len, input_dim))
        seqs.append(sequence)
        
        # Generate random target vector
        #target = np.random.uniform(-1, 1, model_dims[-1])
        #target = np.sum(sequence, axis=0) / L**0.2
        target = [sum(vec[d] for vec in sequence) / L for d in range(model_dims[-1])]
        t_list.append(target)
    
    # Test Case 1: All Linker Matrices Trainable
    print("\n=== Test Case 1: All Linker Matrices Trainable ===")
    model1 = HierDDLabTorch(
        input_dim=input_dim,
        model_dims=model_dims,
        basis_dims=basis_dims,
        linker_dims=linker_dims,
        input_seq_len=input_seq_len,
        linker_trainable=True,  # All Linker matrices trainable
        use_residual=True
    ).to(device)
    
    # Show parameter counts
    print("\nModel 1 parameter counts:")
    model1.count_parameters()
    
    # Train model
    print("\nTraining Model 1...")
    history1 = model1.train_model(
        seqs, 
        t_list,
        learning_rate=0.01,
        max_iters=200,
        decay_rate=0.99,
        print_every=10
    )
    
    # Evaluate predictions
    pred_t_list = [model1.predict_t(seq) for seq in seqs]
    print("\nPrediction correlations per dimension:")
    corrs = []
    for d in range(model_dims[-1]):
        actuals = [t[d] for t in t_list]
        preds = [p[d] for p in pred_t_list]
        corr, _ = pearsonr(actuals, preds)
        corrs.append(corr)
        print(f"  Dim {d}: correlation = {corr:.4f}")
    
    print(f"Average correlation: {np.mean(corrs):.4f}")
    
    # Test Case 2: Mixed Linker Trainability
    print("\n\n=== Test Case 2: Mixed Linker Trainability ===")
    model2 = HierDDLabTorch(
        input_dim=input_dim,
        model_dims=model_dims,
        basis_dims=basis_dims,
        linker_dims=linker_dims,
        input_seq_len=input_seq_len,
        linker_trainable=[True, False, True],  # First layer trainable, second not
        use_residual=True
    ).to(device)
    
    print("\nTraining Model 2...")
    history2 = model2.train_model(
        seqs, 
        t_list,
        learning_rate=0.01,
        max_iters=200,
        decay_rate=0.99,
        print_every=10
    )
    
    # Evaluate predictions
    pred_t_list2 = [model2.predict_t(seq) for seq in seqs]
    print("\nPrediction correlations per dimension (Mixed):")
    corrs2 = []
    for d in range(model_dims[-1]):
        actuals = [t[d] for t in t_list]
        preds = [p[d] for p in pred_t_list2]
        corr, _ = pearsonr(actuals, preds)
        corrs2.append(corr)
        print(f"  Dim {d}: correlation = {corr:.4f}")
    
    print(f"Average correlation: {np.mean(corrs2):.4f}")
    
    # Test Case 3: No Linker Matrices Trainable
    print("\n\n=== Test Case 3: No Linker Matrices Trainable ===")
    model3 = HierDDLabTorch(
        input_dim=input_dim,
        model_dims=model_dims,
        basis_dims=basis_dims,
        linker_dims=linker_dims,
        input_seq_len=input_seq_len,
        linker_trainable=False,  # No Linker matrices trainable
        use_residual=True
    ).to(device)
    
    print("\nTraining Model 3...")
    history3 = model3.train_model(
        seqs, 
        t_list,
        learning_rate=0.01,
        max_iters=200,
        decay_rate=0.99,
        print_every=10
    )
    
    # Evaluate predictions
    pred_t_list3 = [model3.predict_t(seq) for seq in seqs]
    print("\nPrediction correlations per dimension (Fixed Linkers):")
    corrs3 = []
    for d in range(model_dims[-1]):
        actuals = [t[d] for t in t_list]
        preds = [p[d] for p in pred_t_list3]
        corr, _ = pearsonr(actuals, preds)
        corrs3.append(corr)
        print(f"  Dim {d}: correlation = {corr:.4f}")
    
    print(f"Average correlation: {np.mean(corrs3):.4f}")
    
    # Save and load model
    print("\nTesting model persistence...")
    model1.save("hierarchical_vector_model_linker_torch.pth")
    loaded_model = HierDDLabTorch.load("hierarchical_vector_model_linker_torch.pth", device=device)
    
    print("Loaded model prediction for first sequence:")
    pred = loaded_model.predict_t(seqs[0])
    print(f"  Predicted target: {[round(p, 4) for p in pred]}")
    
    print("\n=== Hierarchical Vector Sequence Processing with Linker Matrices Demo Completed ===")
