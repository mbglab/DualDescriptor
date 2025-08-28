# Copyright (C) Bin-Guang Ma (mbg@mail.hzau.edu.cn). All rights reserved.
# Hybrid Descriptor-Transformer Network (DDNet) implementation
# PyTorch implementation with GPU support
# Author: Bin-Guang Ma; Date: 2025-8-16

import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class DescriptorLayer(nn.Module):
    """Descriptor layer implementation with improved period calculation"""
    def __init__(self, in_dim, out_dim, num_basis, use_residual):
        """
        Initialize a descriptor layer
        
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
        
        # Improved period calculation - exponential growth instead of linear
        periods = torch.zeros(out_dim, out_dim, num_basis)
        for i in range(out_dim):
            for j in range(out_dim):
                for g in range(num_basis):
                    # Use more reasonable period range with exponential growth
                    base_period = 2 + (i + j) * 0.5
                    periods[i, j, g] = base_period * (1.5 ** g)
        self.register_buffer('periods', periods)

        # Residual projection processing
        if self.use_residual == 'separate':
            self.residual_proj = nn.Linear(in_dim, out_dim, bias=False) if in_dim != out_dim else nn.Identity()
        
        # Layer normalization
        self.norm = nn.LayerNorm(out_dim)
        
        # Initialize parameters        
        nn.init.xavier_uniform_(self.M)  # Better initialization
        nn.init.xavier_uniform_(self.P)  # Better initialization
        if isinstance(self.residual_proj, nn.Linear):            
            nn.init.xavier_uniform_(self.residual_proj.weight)  # Better initialization
    
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

class DDNet(nn.Module):
    """Hybrid Descriptor-Transformer Network (DDNet)"""
    def __init__(self, input_dim=2, model_dim=64, num_blocks=3, num_basis=5, 
                 num_heads=4, dim_feedforward=256, dropout=0.1, 
                 use_residual='separate', target_dim=3, device='cpu'):
        """
        Initialize DDNet model
        
        Args:
            input_dim (int): Input vector dimension
            model_dim (int): Hidden dimension for all layers
            num_blocks (int): Number of hybrid blocks
            num_basis (int): Number of basis functions for descriptor layers
            num_heads (int): Number of attention heads for transformer layers
            dim_feedforward (int): Feedforward dimension in transformer
            dropout (float): Dropout rate
            use_residual (str): Residual connection type for descriptor layers
            target_dim (int): Output target dimension
            device (str): Device to use ('cpu' or 'cuda')
        """
        super().__init__()
        self.input_dim = input_dim
        self.model_dim = model_dim
        self.num_blocks = num_blocks
        self.target_dim = target_dim
        self.device = device
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, model_dim)
        
        # Create hybrid blocks
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            # Descriptor layer with improved period calculation
            desc_layer = DescriptorLayer(
                in_dim=model_dim,
                out_dim=model_dim,
                num_basis=num_basis,
                use_residual=use_residual
            )
            
            # Transformer layer with layer normalization and GELU activation
            trans_layer = TransformerEncoderLayer(
                d_model=model_dim,
                nhead=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation='gelu',  # Use GELU for better performance
                batch_first=True  # Use (batch, seq, features) format
            )
            
            self.blocks.append(nn.ModuleList([desc_layer, trans_layer]))
        
        # Output layer for target prediction
        self.output_layer = nn.Linear(model_dim, target_dim)
        
        # Positional encoding initialization
        self.pos_encoder = PositionalEncoding(model_dim, dropout)
        
        # Initialize weights properly
        self._init_weights()
        
        self.to(device)
    
    def _init_weights(self):
        """Initialize model weights with proper initialization"""
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    def forward(self, x):
        """
        Forward pass through all hybrid blocks
        
        Args:
            x (Tensor): Input sequence of shape (batch_size, seq_len, input_dim)
        
        Returns:
            Tensor: Output sequence of shape (batch_size, seq_len, model_dim)
        """
        # Generate position indices
        batch_size, seq_len, _ = x.shape
        positions = torch.arange(seq_len, device=self.device)
        
        # Input projection
        x = self.input_proj(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Process through hybrid blocks
        for desc_layer, trans_layer in self.blocks:
            # Descriptor layer processing
            x_desc = desc_layer(x, positions)
            
            # Transformer layer processing (no padding mask needed for full sequences)
            x = trans_layer(x_desc)
        
        return x
    
    def predict_t(self, seq):
        """
        Predict target vector for a sequence
        Returns the average of all output vectors in the sequence
        
        Args:
            seq (Tensor): Input sequence of shape (batch_size, seq_len, input_dim)
        
        Returns:
            Tensor: Predicted target vector of shape (batch_size, target_dim)
        """
        # Get final sequence representation
        outputs = self.forward(seq)
        
        # Average pooling over sequence length
        pooled = outputs.mean(dim=1)
        
        # Project to target dimension
        return self.output_layer(pooled)
    
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
            loss = torch.mean((pred - t_tensor) ** 2)  # Use MSE instead of sum
            
            total_loss += loss.item()
            count += 1
        
        return total_loss / count if count else 0.0
    
    def train_model(self, seqs, t_list, max_iters=1000, lr=0.001, 
                    batch_size=8, decay_rate=0.998, print_every=10):
        """
        Train model using Adam optimizer with batch processing
        
        Args:
            seqs (list): List of training sequences
            t_list (list): List of target vectors
            max_iters (int): Maximum training iterations
            lr (float): Learning rate
            batch_size (int): Batch size for training
            decay_rate (float): Learning rate decay rate
            print_every (int): Print progress every N iterations
        
        Returns:
            list: Training loss history
        """
        optimizer = optim.Adam(self.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)
        history = []
        
        # Create dataset
        dataset = list(zip(seqs, t_list))
        
        for it in range(max_iters):
            total_loss = 0.0
            count = 0
            
            # Shuffle dataset for each epoch
            random.shuffle(dataset)
            
            # Process in batches
            for i in range(0, len(dataset), batch_size):
                batch = dataset[i:i+batch_size]
                batch_loss = 0.0
                
                for seq, t in batch:
                    # Convert to tensors
                    seq_tensor = torch.tensor(seq, dtype=torch.float32, device=self.device)
                    t_tensor = torch.tensor(t, dtype=torch.float32, device=self.device)
                    
                    # Forward pass
                    pred = self.predict_t(seq_tensor.unsqueeze(0)).squeeze(0)
                    
                    # Compute loss (MSE)
                    loss = torch.mean((pred - t_tensor) ** 2)
                    batch_loss += loss
                    count += 1
                
                # Backward pass for the batch
                optimizer.zero_grad()
                (batch_loss / len(batch)).backward()
                optimizer.step()
                
                total_loss += batch_loss.item()
            
            # Update learning rate
            scheduler.step()
            
            # Record and print progress
            avg_loss = total_loss / count if count else 0.0
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
        print(f"{'Layer':<20} | {'Param Type':<25} | {'Count':<15} | {'Shape':<20}")
        print("-"*60)
        
        # Input projection layer
        layer_name = "Input Projection"
        for name, param in self.input_proj.named_parameters():
            if param.requires_grad:
                num = param.numel()
                shape = str(tuple(param.shape))
                print(f"{layer_name:<20} | {name:<25} | {num:<15} | {shape:<20}")
                total_params += num
                layer_params.append((layer_name, num))
        
        # Positional encoding
        layer_name = "Positional Encoding"
        for name, param in self.pos_encoder.named_parameters():
            if param.requires_grad:
                num = param.numel()
                shape = str(tuple(param.shape))
                print(f"{layer_name:<20} | {name:<25} | {num:<15} | {shape:<20}")
                total_params += num
                layer_params.append((layer_name, num))
        
        # Hybrid blocks
        for block_idx, (desc, trans) in enumerate(self.blocks):
            # Descriptor layer
            layer_name = f"Block{block_idx}-Descriptor"
            for name, param in desc.named_parameters():
                if param.requires_grad:
                    num = param.numel()
                    shape = str(tuple(param.shape))
                    print(f"{layer_name:<20} | {name:<25} | {num:<15} | {shape:<20}")
                    total_params += num
                    layer_params.append((layer_name, num))
            
            # Transformer layer
            layer_name = f"Block{block_idx}-Transformer"
            for name, param in trans.named_parameters():
                if param.requires_grad:
                    num = param.numel()
                    shape = str(tuple(param.shape))
                    print(f"{layer_name:<20} | {name:<25} | {num:<15} | {shape:<20}")
                    total_params += num
                    layer_params.append((layer_name, num))
        
        # Output layer
        layer_name = "Output Layer"
        for name, param in self.output_layer.named_parameters():
            if param.requires_grad:
                num = param.numel()
                shape = str(tuple(param.shape))
                print(f"{layer_name:<20} | {name:<25} | {num:<15} | {shape:<20}")
                total_params += num
                layer_params.append((layer_name, num))
        
        # Print summary
        print("="*60)
        print(f"Total trainable parameters: {total_params}")
        print("="*60)
        
        return total_params

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models"""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Add positional encoding to input tensor
        
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:x.size(1)].transpose(0, 1)
        return self.dropout(x)

def generate_meaningful_data(num_seqs, min_len, max_len, input_dim, target_dim):
    """
    Generate synthetic training data with meaningful patterns
    
    Args:
        num_seqs (int): Number of sequences to generate
        min_len (int): Minimum sequence length
        max_len (int): Maximum sequence length
        input_dim (int): Input dimension
        target_dim (int): Target dimension
    
    Returns:
        tuple: (sequences, targets)
    """
    seqs = []
    t_list = []
    
    for _ in range(num_seqs):
        L = random.randint(min_len, max_len)
        
        # Generate meaningful base pattern
        base_pattern = np.random.randn(input_dim)
        frequency = random.uniform(0.05, 0.2)
        phase_shift = random.uniform(0, 2 * math.pi)
        amplitude = random.uniform(0.5, 2.0)
        
        seq = []
        for i in range(L):
            # Create time-dependent pattern with clear structure
            time_factor = amplitude * math.sin(frequency * i + phase_shift)
            # Add some non-linearity
            quadratic_term = 0.01 * (i - L/2) ** 2 / (L/2) ** 2
            noise = 0.02 * np.random.randn(input_dim)
            
            vec = base_pattern * (time_factor + quadratic_term) + noise
            seq.append(vec.tolist())
        
        seqs.append(seq)
        
        # Create target that is clearly related to sequence content
        seq_array = np.array(seq)
        
        # Use meaningful statistics as targets
        mean_vals = np.mean(seq_array, axis=0)
        std_vals = np.std(seq_array, axis=0)
        max_vals = np.max(seq_array, axis=0)
        
        # Combine statistics to create target vector
        t_vec = []
        for d in range(target_dim):
            if d < input_dim:
                # Weighted combination of statistics
                weight = 0.4 * mean_vals[d] + 0.3 * std_vals[d] + 0.3 * max_vals[d]
                t_vec.append(weight)
            else:
                # For extra dimensions, use global statistics
                t_vec.append(np.mean(mean_vals) if d % 2 == 0 else np.mean(std_vals))
        
        t_list.append(t_vec)
    
    return seqs, t_list

# === Example Usage ===
if __name__ == "__main__":
    from scipy.stats import pearsonr
    
    # Set random seeds for reproducibility
    torch.manual_seed(2)
    random.seed(2)
    np.random.seed(2)
    
    # Configuration
    input_dim = 10          # Input vector dimension
    model_dim = 64          # Hidden dimension for all layers
    num_blocks = 3          # Number of hybrid blocks
    num_basis = 5           # Basis functions per descriptor layer
    num_heads = 4           # Number of attention heads
    dim_feedforward = 256   # Feedforward dimension in transformer
    dropout = 0.1           # Dropout rate
    target_dim = 3          # Output target dimension
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    num_seqs = 100          # Number of training sequences
    min_len, max_len = 100, 200  # Sequence length range
    
    print(f"Using device: {device}")
    print(f"Model config: input_dim={input_dim}, model_dim={model_dim}, num_blocks={num_blocks}")
    print(f"num_basis={num_basis}, num_heads={num_heads}, dim_feedforward={dim_feedforward}")
    print(f"dropout={dropout}, target_dim={target_dim}")
    
    # Generate meaningful synthetic training data
    #print("\nGenerating meaningful synthetic training data...")
    #seqs, t_list = generate_meaningful_data(num_seqs, min_len, max_len, input_dim, target_dim)

    print("\nGenerating synthetic training data...")
    seqs = []     # List of sequences (each sequence: list of n-dim vectors)
    t_list = []   # List of target vectors (m-dim)

##    for _ in range(num_seqs):
##        L = random.randint(min_len, max_len)
##        # Generate n-dimensional input sequence
##        seq = [[random.uniform(-1, 1) for _ in range(input_dim)] for __ in range(L)]
##        seqs.append(seq)
##        # Generate m-dimensional target vector
##        t_list.append([random.uniform(-1, 1) for _ in range(target_dim)])

    for _ in range(num_seqs):
        L = random.randint(min_len, max_len)
        # Generate n-dimensional input sequence with time dependence
        base = [random.uniform(-1, 1) for _ in range(input_dim)]
        seq = []
        for i in range(L):
            # Add time-dependent noise
            vec = [base[d] + 0.1 * math.sin(i * 0.1 + d * 0.5) for d in range(input_dim)]
            seq.append(vec)
        seqs.append(seq)
        
        # Create the target vector related to the sequence content
        t_vec = [sum(vec[d]**2 for vec in seq) / L**0.5 for d in range(target_dim)]
        t_list.append(t_vec)
        
    # Create hybrid DDNet model with improved architecture
    print("\nCreating improved DDNet model...")
    model = DDNet(
        input_dim=input_dim,
        model_dim=model_dim,
        num_blocks=num_blocks,
        num_basis=num_basis,
        num_heads=num_heads,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        target_dim=target_dim,
        device=device
    )
    model.count_parameters()
    
    # Train model with improved settings
    print("\nTraining model with improved settings...")
    history = model.train_model(
        seqs, 
        t_list,
        max_iters=50,
        lr=0.001,           # Lower learning rate
        batch_size=8,       # Batch processing
        decay_rate=0.998,   # Slower decay
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
    
    correlations = []
    for i in range(target_dim):
        actual = [t[i] for t in t_list]
        predicted = [p[i] for p in all_preds]
        corr, p_value = pearsonr(actual, predicted)
        correlations.append(corr)
        print(f"Output dim {i} correlation: {corr:.4f} (p={p_value:.4e})")
    
    avg_corr = np.mean(correlations)
    print(f"\nAverage correlation: {avg_corr:.4f}")
    print("Training completed!")
