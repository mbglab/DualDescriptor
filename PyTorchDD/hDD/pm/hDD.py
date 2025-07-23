# Copyright (C) Bin-Guang Ma (mbg@mail.hzau.edu.cn). All rights reserved.
# The Numeric Dual Descriptor class (Tensor form) with hierarchical structure
# PyTorch implementation of Hierarchical Numeric Dual Descriptor with GPU support
# Author: Bin-Guang Ma; Date: 2025-7-15

import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class HierLayer(nn.Module):
    """Single layer of Hierarchical Dual Descriptor"""
    def __init__(self, in_dim, out_dim, num_basis, use_residual):
        """
        Initialize a hierarchical layer
        
        Args:
            in_dim (int): Input dimension
            out_dim (int): Output dimension
            num_basis (int): Number of basis functions
            use_residual (str or None): Residual connection type. Options are:
                - 'separate': use a separate linear projection for the residual connection.
                - 'shared': share the linear transformation matrix M for the residual.
                - None or other: no residual connection, unless in_dim==out_dim, then use identity residual.
        """
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_basis = num_basis
        self.use_residual = use_residual
        
        # Linear transformation matrix (shared for main path and residual)
        self.M = nn.Parameter(torch.empty(out_dim, in_dim))
        
        # Tensor of basis coefficients
        self.P = nn.Parameter(torch.empty(out_dim, out_dim, num_basis))
        
        # Precompute periods tensor (fixed, not trainable)
        periods = torch.zeros(out_dim, out_dim, num_basis)
        for i in range(out_dim):
            for j in range(out_dim):
                for g in range(num_basis):
                    periods[i, j, g] = i*(out_dim*num_basis) + j*num_basis + g + 2
        self.register_buffer('periods', periods)

        # Residual projection processing
        if self.use_residual == 'separate':
            self.residual_proj = nn.Linear(in_dim, out_dim, bias=False) if in_dim != out_dim else nn.Identity()
        
        # Layer normalization
        self.norm = nn.LayerNorm(out_dim)
        
        # Initialize parameters        
        nn.init.uniform_(self.M, -0.5, 0.5)
        nn.init.uniform_(self.P, -0.1, 0.1)
        if isinstance(self.residual_proj, nn.Linear):            
            nn.init.uniform_(self.residual_proj.weight, -0.1, 0.1)      
    
    def forward(self, x, positions):
        """
        Forward pass for the layer
        
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

class HierDDpm(nn.Module):
    """Hierarchical Numeric Dual Descriptor with PyTorch and GPU support"""
    def __init__(self, input_dim=2, model_dims=[2], num_basis_list=[5], use_residual_list=None, device='cpu'):
        """
        Initialize hierarchical HierDDpm
        
        Args:
            input_dim (int): Input vector dimension
            model_dims (list): Output dimensions for each layer
            num_basis_list (list): Number of basis functions for each layer
            device (str): Device to use ('cpu' or 'cuda')
        """
        super().__init__()
        self.input_dim = input_dim
        self.model_dims = model_dims
        self.num_basis_list = num_basis_list
        if use_residual_list == None:
            self.use_residual_list = ['separate'] * len(self.model_dims)
        else:
            self.use_residual_list = use_residual_list            
        self.num_layers = len(model_dims)
        self.device = device
        
        # Create hierarchical layers
        layers = []
        in_dim = input_dim
        for out_dim, num_basis, use_residual in zip(self.model_dims, self.num_basis_list, self.use_residual_list):
            layers.append(HierLayer(in_dim, out_dim, num_basis, use_residual))
            in_dim = out_dim  # Next layer input is current layer output
        
        self.layers = nn.ModuleList(layers)
        self.to(device)
    
    def forward(self, seq):
        """
        Forward pass through all layers
        
        Args:
            seq (Tensor): Input sequence of shape (batch_size, seq_len, input_dim)
        
        Returns:
            Tensor: Output sequence of shape (batch_size, seq_len, model_dims[-1])
        """
        # Generate position indices
        batch_size, seq_len, _ = seq.shape
        positions = torch.arange(seq_len, device=self.device)
        
        x = seq
        for layer in self.layers:
            x = layer(x, positions)
        return x
    
    def predict_t(self, seq):
        """
        Predict target vector for a sequence
        Returns the average of all output vectors in the sequence
        
        Args:
            seq (Tensor): Input sequence of shape (batch_size, seq_len, input_dim)
        
        Returns:
            Tensor: Predicted target vector of shape (batch_size, model_dims[-1])
        """
        outputs = self.forward(seq)
        return outputs.mean(dim=1)  # Average over sequence length
    
    def deviation(self, seqs, t_list):
        """
        Compute mean squared deviation D across sequences
        
        Args:
            seqs (list): List of input sequences
            t_list (list): List of target vectors
        
        Returns:
            float: Mean squared deviation
        """
        total_loss = 0.0
        count = 0
        
        for seq, t in zip(seqs, t_list):
            # Convert to tensors
            seq_tensor = torch.tensor(seq, dtype=torch.float32, device=self.device)
            t_tensor = torch.tensor(t, dtype=torch.float32, device=self.device)
            
            # Predict and compute loss
            pred = self.predict_t(seq_tensor.unsqueeze(0)).squeeze(0)
            loss = torch.sum((pred - t_tensor) ** 2)
            
            total_loss += loss.item()
            count += 1
        
        return total_loss / count if count else 0.0
    
    def train_model(self, seqs, t_list, max_iters=1000, lr=0.01, 
                    decay_rate=0.999, print_every=10):
        """
        Train model using Adam optimizer
        
        Args:
            seqs (list): List of training sequences
            t_list (list): List of target vectors
            max_iters (int): Maximum training iterations
            lr (float): Learning rate
            decay_rate (float): Learning rate decay rate
            print_every (int): Print progress every N iterations
        
        Returns:
            list: Training loss history
        """
        optimizer = optim.Adam(self.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)
        history = []
        
        for it in range(max_iters):
            total_loss = 0.0
            count = 0
            
            for seq, t in zip(seqs, t_list):
                # Convert to tensors
                seq_tensor = torch.tensor(seq, dtype=torch.float32, device=self.device)
                t_tensor = torch.tensor(t, dtype=torch.float32, device=self.device)
                
                # Forward pass
                pred = self.predict_t(seq_tensor.unsqueeze(0)).squeeze(0)
                
                # Compute loss
                loss = torch.sum((pred - t_tensor) ** 2)
                total_loss += loss.item()
                count += 1
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            # Update learning rate
            scheduler.step()
            
            # Record and print progress
            avg_loss = total_loss / count
            history.append(avg_loss)
            
            if it % print_every == 0 or it == max_iters - 1:
                current_lr = scheduler.get_last_lr()[0]
                print(f"Iter {it:4d}: Loss = {avg_loss:.6e}, LR = {current_lr:.6f}")
        
        return history
    
    def count_parameters(self):
        """Count and print learnable parameters in the model by layer and parameter type"""
        total_params = 0
        layer_params = []
        
        # Print header
        print("="*60)
        print(f"{'Layer':<10} | {'Param Type':<20} | {'Count':<15} | {'Shape':<20}")
        print("-"*60)
        
        # Iterate through each layer
        for i, layer in enumerate(self.layers):
            layer_total = 0
            layer_name = f"Layer {i}"
            
            # Count parameters in this layer
            for name, param in layer.named_parameters():
                if param.requires_grad:
                    num = param.numel()
                    shape = str(tuple(param.shape))
                    
                    # Print parameter details
                    print(f"{layer_name:<10} | {name:<20} | {num:<15} | {shape:<20}")
                    
                    layer_total += num
                    total_params += num
            
            # Store layer summary
            layer_params.append((layer_name, layer_total))
        
        # Print summary by layer
        print("-"*60)
        for name, count in layer_params:
            print(f"{name} total parameters: {count}")
        
        # Print grand total
        print("="*60)
        print(f"Total trainable parameters: {total_params}")
        print("="*60)
        
        return total_params

# === Example Usage ===
if __name__ == "__main__":

    from scipy.stats import pearsonr
    
    # Set random seeds for reproducibility
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    
    # Configuration
    input_dim = 10          # Input vector dimension
    model_dims = [8, 6, 3]  # Output dimensions for each layer
    num_basis_list = [5, 4, 3]  # Basis functions per layer
    use_residual_list = ['separate'] * 3 
    num_seqs = 100          # Number of training sequences
    min_len, max_len = 100, 200  # Sequence length range
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print(f"Model config: input_dim={input_dim}, model_dims={model_dims}, num_basis={num_basis_list}, use_residual={use_residual_list}")
    
    # Generate synthetic training data(random)
    seqs = []     # List of sequences (each sequence: list of n-dim vectors)
    t_list = []   # List of target vectors (m-dim)
    
    for _ in range(num_seqs):
        L = random.randint(min_len, max_len)
        # Generate n-dimensional input sequence
        seq = [[random.uniform(-1, 1) for _ in range(input_dim)] for __ in range(L)]
        seqs.append(seq)
        # Generate m-dimensional target vector
        t_list.append([random.uniform(-1, 1) for _ in range(model_dims[-1])])

##    # Generate synthetic training data (content-dependent)
##    print("\nGenerating synthetic training data...")
##    seqs = []     # List of sequences (each sequence: list of n-dim vectors)
##    t_list = []   # List of target vectors (m-dim)
##    
##    for _ in range(num_seqs):
##        L = random.randint(min_len, max_len)
##        # Generate n-dimensional input sequence with time dependence
##        base = [random.uniform(-1, 1) for _ in range(input_dim)]
##        seq = []
##        for i in range(L):
##            # Add time-dependent noise
##            vec = [base[d] + 0.1 * math.sin(i * 0.1 + d * 0.5) for d in range(input_dim)]
##            seq.append(vec)
##        seqs.append(seq)
##        
##        # Create the target vector related to the sequence content
##        #t_vec = [sum(math.cos(vec[d]**2) for vec in seq) / math.sin(L) for d in range(model_dims[-1])]
##        t_vec = [sum(vec[d]**2 for vec in seq) / L**0.5 for d in range(model_dims[-1])]
##        t_list.append(t_vec)
    
    # Create model with shared M matrix
    print("\nCreating Hierarchical HierDDpm model...")
    model = HierDDpm(
        input_dim=input_dim,
        model_dims=model_dims,
        num_basis_list=num_basis_list,
        use_residual_list=use_residual_list,
        device=device
    )
    model.count_parameters()
    
    # Train model
    print("\nTraining model...")
    history = model.train_model(
        seqs, 
        t_list,
        max_iters=200,
        lr=0.1,
        decay_rate=0.97,
        print_every=10
    )
    
    # Predict targets
    print("\nPredictions (first 3 sequences):")
    predictions = []
    for i, seq in enumerate(seqs[:3]):
        seq_tensor = torch.tensor(seq, dtype=torch.float32, device=device)
        t_pred = model.predict_t(seq_tensor.unsqueeze(0)).squeeze(0).detach().cpu().numpy()
        
        print(f"Seq {i+1}:")
        print(f"  Target:    {[f'{x:.4f}' for x in t_list[i]]}")
        print(f"  Predicted: {[f'{x:.4f}' for x in t_pred]}")
        predictions.append(t_pred)
    
    # Calculate prediction correlations
    print("\nCalculating prediction correlations...")
    all_preds = []
    for seq in seqs:
        seq_tensor = torch.tensor(seq, dtype=torch.float32, device=device)
        t_pred = model.predict_t(seq_tensor.unsqueeze(0)).squeeze(0).detach().cpu().numpy()
        all_preds.append(t_pred)
    
    output_dim = model_dims[-1]
    correlations = []
    for i in range(output_dim):
        actual = [t[i] for t in t_list]
        predicted = [p[i] for p in all_preds]
        corr, p_value = pearsonr(actual, predicted)
        correlations.append(corr)
        print(f"Output dim {i} correlation: {corr:.4f} (p={p_value:.4e})")
    
    avg_corr = np.mean(correlations)
    print(f"\nAverage correlation: {avg_corr:.4f}")
    print("Training completed!")
