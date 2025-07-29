# Copyright (C) 2005-2025, Bin-Guang Ma (mbg@mail.hzau.edu.cn). All rights reserved.
# The Hierarchical Numeric Dual Descriptor class (Random AB matrix form) for Vector Sequences
# Modified to support hierarchical structure with multiple layers - PyTorch GPU Accelerated Version
# Author: Bin-Guang Ma (assisted by DeepSeek); Date: 2025-7-7

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import pickle
import math

class HierarchicalLayer(nn.Module):
    """
    Single layer of the Hierarchical Numeric Dual Descriptor with Residual Connections.
    
    Args:
        model_dim (int): Dimension of input/output vectors
        basis_dim (int): Basis dimension for this layer
        device (str): Device to run computations on ('cuda' or 'cpu')
    """
    def __init__(self, model_dim, basis_dim, device='cuda'):
        super().__init__()
        self.model_dim = model_dim
        self.basis_dim = basis_dim
        self.device = device
        
        # Linear transformation
        self.linear = nn.Linear(model_dim, model_dim, bias=False)
        
        # Layer normalization
        self.norm = nn.LayerNorm(model_dim)
        
        # Coefficient matrix: model_dim x basis_dim
        self.Acoeff = nn.Parameter(torch.Tensor(model_dim, basis_dim))
        
        # Basis matrix: basis_dim x model_dim
        self.Bbasis = nn.Parameter(torch.Tensor(basis_dim, model_dim))
        
        # Initialize parameters
        nn.init.uniform_(self.linear.weight, -0.1, 0.1)
        nn.init.uniform_(self.Acoeff, -0.1, 0.1)
        nn.init.uniform_(self.Bbasis, -0.1, 0.1)
        
        self.to(device)
    
    def forward(self, x):
        """
        Forward pass through the layer with residual connections
        
        Args:
            x (torch.Tensor): Input sequence of shape (seq_len, model_dim)
        
        Returns:
            torch.Tensor: Output sequence of shape (seq_len, model_dim)
        """
        # Apply linear transformation
        transformed = self.linear(x)
        
        # Apply layer normalization
        normalized = self.norm(transformed)
        
        # Get sequence length
        seq_len = x.size(0)
        
        # Compute basis indices: j = k % basis_dim for each position
        j_indices = torch.arange(seq_len) % self.basis_dim
        j_indices = j_indices.to(self.device)
        
        # Select basis vectors: (seq_len, model_dim)
        B_vectors = self.Bbasis[j_indices]
        
        # Compute scalars: dot product of normalized and B_vectors
        scalars = torch.einsum('sd,sd->s', normalized, B_vectors)
        
        # Select coefficient vectors: (seq_len, model_dim)
        A_vectors = self.Acoeff[:, j_indices].t()
        
        # Compute new features: scalars * A_vectors
        new_features = scalars.unsqueeze(1) * A_vectors
        
        # Residual connection: add input to new features
        return new_features + x

class HierDDabResidual(nn.Module):
    """
    Hierarchical Numeric Dual Descriptor with Residual Connections and Layer Normalization.
    All layers have the same input/output dimension (model_dim), while basis dimensions can vary.
    
    Args:
        model_dim (int): Dimension of input vectors and all layer outputs
        basis_dims (list): List of basis dimensions for each layer
        device (str): Device to run computations on ('cuda' or 'cpu')
    """
    def __init__(self, model_dim=4, basis_dims=[50], device='cuda'):
        super().__init__()
        self.model_dim = model_dim
        self.basis_dims = basis_dims
        self.num_layers = len(basis_dims)
        self.device = device
        self.trained = False
        
        # Create layers
        self.layers = nn.ModuleList()
        for basis_dim in basis_dims:
            layer = HierarchicalLayer(model_dim, basis_dim, device)
            self.layers.append(layer)
        
        self.to(device)
    
    def forward(self, x):
        """
        Forward pass through the entire hierarchical model
        
        Args:
            x (torch.Tensor): Input sequence of shape (seq_len, model_dim)
        
        Returns:
            torch.Tensor: Output sequence of shape (seq_len, model_dim)
        """
        current = x
        for layer in self.layers:
            current = layer(current)
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
        
        # Forward pass
        output_tensor = self.forward(input_tensor)
        
        # Convert back to list of vectors
        return output_tensor.cpu().detach().numpy().tolist()
    
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
            target_tensor = torch.tensor(t, dtype=torch.float32).to(self.device)
            
            # Forward pass
            outputs = self.forward(input_tensor)
            
            # Compute MSE for each position
            for pos in range(len(seq)):
                loss = torch.mean((outputs[pos] - target_tensor) ** 2)
                total_loss += loss.item()
                total_positions += 1
        
        return total_loss / total_positions if total_positions > 0 else 0.0
    
    def train_model(self, seqs, t_list, max_iters=1000, tol=1e-8,
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
                
                # Forward pass
                outputs = self.forward(input_tensor)
                
                # Compute loss (mean across positions and dimensions)
                loss = torch.mean((outputs - target) ** 2)
                total_loss += loss.item() * len(seq)
                total_positions += len(seq)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            # Update learning rate
            scheduler.step()
            
            # Calculate average loss
            avg_loss = total_loss / total_positions
            history.append(avg_loss)
            
            # Print progress
            if it % print_every == 0 or it == max_iters - 1:
                lr = optimizer.param_groups[0]['lr']
                print(f"Iter {it:4d}: Loss = {avg_loss:.8f}, LR = {lr:.6f}")
            
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
            return [0.0] * self.model_dim
        
        # Compute average across positions
        return np.array(outputs).mean(axis=0).tolist()
    
    def count_parameters(self):
        """
        Calculate and print the number of learnable parameters
        """
        total_params = 0
        print("Model Parameter Counts:")
        
        for l_idx, layer in enumerate(self.layers):
            # Parameter counts
            linear_params = sum(p.numel() for p in layer.linear.parameters())
            A_params = layer.Acoeff.numel()
            B_params = layer.Bbasis.numel()
            norm_params = sum(p.numel() for p in layer.norm.parameters())
            
            layer_params = linear_params + A_params + B_params + norm_params
            total_params += layer_params
            
            print(f"  Layer {l_idx} (model_dim: {self.model_dim}, basis_dim: {self.basis_dims[l_idx]}):")
            print(f"    Linear: {linear_params} params")
            print(f"    Acoeff: {A_params} params")
            print(f"    Bbasis: {B_params} params")
            print(f"    LayerNorm: {norm_params} params")
            print(f"    Layer total: {layer_params}")
        
        print(f"Total parameters: {total_params}")
        return total_params
    
    def save(self, filename):
        """Save model to file"""
        torch.save({
            'state_dict': self.state_dict(),
            'model_dim': self.model_dim,
            'basis_dims': self.basis_dims,
            'trained': self.trained
        }, filename)
        print(f"Model saved to {filename}")
    
    @classmethod
    def load(cls, filename, device='cuda'):
        """Load model from file"""
        checkpoint = torch.load(filename, map_location=device)
        model = cls(
            model_dim=checkpoint['model_dim'],
            basis_dims=checkpoint['basis_dims'],
            device=device
        )
        model.load_state_dict(checkpoint['state_dict'])
        model.trained = checkpoint['trained']
        print(f"Model loaded from {filename}")
        return model

# === Example Usage ===
if __name__ == "__main__":

    from statistics import mean, correlation
    
    # Set random seeds for reproducibility
    torch.manual_seed(2)
    np.random.seed(2)
    random.seed(2)
    
    # Configuration
    model_dim = 10       # Input/output dimension for all layers
    basis_dims = [150, 150]  # Basis dimensions for each layer
    seq_count = 100       # Number of training sequences
    min_len = 100        # Minimum sequence length
    max_len = 200        # Maximum sequence length
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Generate training data
    print("\nGenerating training data...")
    print(f"Model dimension: {model_dim}, Basis dims: {basis_dims}")
    seqs = []     # List of input sequences
    t_list = []   # List of target vectors
    
    for i in range(seq_count):
        # Random sequence length
        length = random.randint(min_len, max_len)
        # Generate vector sequence
        sequence = np.random.uniform(-1, 1, (length, model_dim)).tolist()
        seqs.append(sequence)
        
        # Generate random target vector
        target = np.random.uniform(-1, 1, model_dim).tolist()
        t_list.append(target)        
    
    # Create model
    print("\nCreating Hierarchical HierDDabResidual model...")
    model = HierDDabResidual(
        model_dim=model_dim,
        basis_dims=basis_dims,
        device=device
    )
    
    # Show model structure
    print("\nModel structure:")
    model.count_parameters()
    
    # Train model
    print("\nTraining model...")
    history = model.train_model(
        seqs,
        t_list,
        learning_rate=0.01,
        max_iters=300,
        tol=1e-88,
        decay_rate=0.98,
        print_every=10
    )
    
    # Evaluate predictions
    print("\nEvaluating predictions...")
    pred_t_list = [model.predict_t(seq) for seq in seqs]
    
    print("Prediction correlations per dimension:")
    corrs = []
    for d in range(model_dim):
        actuals = [t[d] for t in t_list]
        preds = [p[d] for p in pred_t_list]
        corr = correlation(actuals, preds)
        corrs.append(corr)
        print(f"  Dim {d}: correlation = {corr:.4f}")
    
    print(f"Average correlation: {mean(corrs):.4f}")
    
    # Save and load model
    print("\nTesting model persistence...")
    model.save("hierarchical_model_residual.pth")
    loaded_model = HierDDabResidual.load("hierarchical_model_residual.pth", device=device)
    
    # Verify loaded model
    print("Loaded model prediction for first sequence:")
    pred = loaded_model.predict_t(seqs[0])
    print(f"  Predicted target: {[round(p, 4) for p in pred]}")
    
    print("\n=== Hierarchical Vector Sequence Processing with Residual Connections Completed ===")
