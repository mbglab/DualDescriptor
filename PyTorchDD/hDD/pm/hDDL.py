# Copyright (C) Bin-Guang Ma (mbg@mail.hzau.edu.cn). All rights reserved.
# The Numeric Dual Descriptor class (Tensor form) with hierarchical structure
# Modified to support sequence length transformation between layers using Linker matrices
# PyTorch implementation of Hierarchical Numeric Dual Descriptor with GPU support
# Author: Bin-Guang Ma; Date: 2025-7-15

import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle

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
        self.M = nn.Parameter(torch.randn(out_dim, in_dim) * 0.1)
        
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

class HierDDLpmTorch(nn.Module):
    """
    Hierarchical Numeric Dual Descriptor with PyTorch implementation
    Features:
    - GPU acceleration support
    - Flexible residual connections
    - Layer normalization
    - Sequence length transformation via Linker matrices
    - Configurable trainable parameters
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
    
    def train_model(self, seqs, t_list, max_iters=1000, lr=0.01, 
                   decay_rate=1.0, tol=1e-88, print_every=10):
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
            
            # Update learning rate
            scheduler.step()
            
            # Print progress
            if it % print_every == 0 or it == max_iters - 1:
                current_lr = scheduler.get_last_lr()[0]
                print(f"Iter {it:3d}: Loss = {avg_loss:.6e}, LR = {current_lr:.6f}")
            
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
        checkpoint = torch.load(filename, map_location=device)
        config = checkpoint['config']
        
        # Create model instance
        model = cls(
            input_dim=config['input_dim'],
            model_dims=config['model_dims'],
            num_basis_list=config['num_basis_list'],
            input_seq_len=config['input_seq_len'],
            linker_dims=config['linker_dims'],
            linker_trainable=config['linker_trainable'],
            use_residual_list=config.get('use_residual_list', None),  # Backward compatibility
            device=device
        )
        
        # Load state
        model.load_state_dict(checkpoint['model_state_dict'])
        model.trained = True
        print(f"Model loaded from {filename}")
        return model


# === Example Usage ===
if __name__ == "__main__":

    from scipy.stats import pearsonr

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
    num_seqs = 300          # Number of training sequences
    
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
    model_mixed = HierDDLpmTorch(
        input_dim=input_dim,
        model_dims=model_dims,
        num_basis_list=num_basis_list,
        input_seq_len=input_seq_len,
        linker_dims=linker_dims,
        linker_trainable=[True, False, True],
        use_residual_list=['separate', 'shared', None]  # Different residual for each layer
        #use_residual_list=[None] * 3
    )
    
    # Parameter count
    print("\nParameter count before training:")
    total_params, trainable_params = model_mixed.count_parameters()
    
    # Training
    print("\nTraining model...")
    history = model_mixed.train_model(
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
        corr, p_value = pearsonr(actual, predicted)
        correlations.append(corr)
        print(f"Output dim {i} correlation: {corr:.4f} (p={p_value:.4e})")
    print(f"Average correlation: {np.mean(correlations):.4f}")
    
    # Test Case 2: Save and load model
    print("\nTesting save/load functionality...")
    model_mixed.save("hddlpm_model_residual.pth")
    loaded_model = HierDDLpmTorch.load("hddlpm_model_residual.pth")
    
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
