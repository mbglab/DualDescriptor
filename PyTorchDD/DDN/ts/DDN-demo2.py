# Copyright (C) 2005-2025, Bin-Guang Ma (mbg@mail.hzau.edu.cn); SPDX-License-Identifier: MIT
# A Dual Descriptor Network class demo: hDD (Tensor form) mixed with Transformer encoder
# This program is for the demonstration of methodology and not fully refined.
# Author: Bin-Guang Ma (assisted by ChatGPT & DeepSeek); Date: 2025-8-25

import torch
import torch.nn as nn
import torch.optim as optim
import math
import random
import numpy as np
from scipy.stats import pearsonr

class DescLayer(nn.Module):
    """
    Descriptor Layer with periodic basis functions
    Implements the transformation:
        ln_input = LayerNorm(input)
        x = M * ln_input
        Nk = sum_{i,j,g} P[i,j,g] * x_j * cos(2πk/period)
        res = R * ln_input
        out = res + Nk
    
    Optimized with vectorized operations for batch and sequence processing.
    """
    def __init__(self, in_dim, out_dim, num_basis, ln_eps=1e-5):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_basis = num_basis
        
        # Linear transformations
        self.M = nn.Linear(in_dim, out_dim, bias=False)
        self.R = nn.Linear(in_dim, out_dim, bias=False)
        
        # Basis coefficients
        self.P = nn.Parameter(torch.empty(out_dim, out_dim, num_basis))
        
        # LayerNorm parameters
        self.ln_in = nn.LayerNorm(in_dim, eps=ln_eps)
        
        # Fixed periods tensor (not learnable)
        periods = torch.zeros(out_dim, out_dim, num_basis)
        for i in range(out_dim):
            for j in range(out_dim):
                for g in range(num_basis):
                    periods[i, j, g] = i*(out_dim*num_basis) + j*num_basis + g + 2
        self.register_buffer('periods', periods)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        # Initialize weights
        nn.init.uniform_(self.M.weight, -0.5, 0.5)
        nn.init.uniform_(self.R.weight, -0.01, 0.01)
        nn.init.uniform_(self.P, -0.1, 0.1)
        
        # Initialize residual connection as identity if dimensions match
        if self.in_dim == self.out_dim:
            self.R.weight.data = torch.eye(self.out_dim) + torch.randn_like(self.R.weight) * 0.01
    
    def forward(self, x, k):
        """
        Vectorized forward pass for batch and sequence processing.
        Args:
            x: Input tensor of shape (batch_size, seq_len, in_dim)
            k: Position indices tensor of shape (batch_size, seq_len)
        Returns:
            Output tensor of shape (batch_size, seq_len, out_dim)
        """
        # Layer normalization
        ln_x = self.ln_in(x)  # (batch_size, seq_len, in_dim)
        
        # Linear transformations
        x_proj = self.M(ln_x)  # (batch_size, seq_len, out_dim)
        res = self.R(ln_x)     # (batch_size, seq_len, out_dim)
        
        # Prepare tensors for broadcasting
        # P: (1, 1, out_dim, out_dim, num_basis)
        P_exp = self.P.unsqueeze(0).unsqueeze(0)
        # periods: (1, 1, out_dim, out_dim, num_basis)
        periods_exp = self.periods.unsqueeze(0).unsqueeze(0)
        # k: (batch_size, seq_len, 1, 1, 1)
        k_exp = k.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        # x_proj: (batch_size, seq_len, 1, out_dim, 1)
        x_proj_exp = x_proj.unsqueeze(2).unsqueeze(4)
        
        # Compute cosine terms using broadcasting
        # cos_terms: (batch_size, seq_len, out_dim, out_dim, num_basis)
        cos_terms = torch.cos(2 * math.pi * k_exp / periods_exp)
        
        # Combine all terms
        terms = P_exp * x_proj_exp * cos_terms
        
        # Sum over j (dim 3) and g (dim 4) dimensions
        Nk = terms.sum(dim=(-1, -2))  # (batch_size, seq_len, out_dim)
        
        # Output: residual + Nk
        return res + Nk


class TransformerBlock(nn.Module):
    """
    Transformer Encoder Block with Pre-Layer Normalization
    Implements:
        ln1 = LayerNorm(x)
        attn_out = MultiHeadAttention(ln1)
        x = x + attn_out
        ln2 = LayerNorm(x)
        ffn_out = FFN(ln2)
        x = x + ffn_out
    """
    def __init__(self, dim, ff_hidden_factor=2, ln_eps=1e-5):
        super().__init__()
        self.dim = dim
        self.ff_hidden = max(1, int(ff_hidden_factor * dim))
        
        # Layer Normalizations
        self.ln1 = nn.LayerNorm(dim, eps=ln_eps)
        self.ln2 = nn.LayerNorm(dim, eps=ln_eps)
        
        # Self-attention layers
        self.Wq = nn.Linear(dim, dim, bias=False)
        self.Wk = nn.Linear(dim, dim, bias=False)
        self.Wv = nn.Linear(dim, dim, bias=False)
        self.Wo = nn.Linear(dim, dim, bias=True)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(dim, self.ff_hidden),
            nn.ReLU(),
            nn.Linear(self.ff_hidden, dim)
        )
        
        self.reset_parameters()
    
    def reset_parameters(self):
        # Initialize attention weights
        nn.init.uniform_(self.Wq.weight, -0.2, 0.2)
        nn.init.uniform_(self.Wk.weight, -0.2, 0.2)
        nn.init.uniform_(self.Wv.weight, -0.2, 0.2)
        nn.init.uniform_(self.Wo.weight, -0.2, 0.2)
        nn.init.zeros_(self.Wo.bias)
        
        # Initialize FFN weights
        for layer in self.ffn:
            if isinstance(layer, nn.Linear):
                nn.init.uniform_(layer.weight, -0.2, 0.2)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, dim)
        Returns:
            Output tensor of shape (batch_size, seq_len, dim)
        """
        batch_size, seq_len, _ = x.shape
        
        # Pre-LN for attention
        ln1 = self.ln1(x)
        
        # Compute Q, K, V
        Q = self.Wq(ln1)  # (batch_size, seq_len, dim)
        K = self.Wk(ln1)  # (batch_size, seq_len, dim)
        V = self.Wv(ln1)  # (batch_size, seq_len, dim)
        
        # Scaled dot-product attention
        attn_scores = torch.matmul(Q, K.transpose(1, 2)) / math.sqrt(self.dim)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_out = torch.matmul(attn_weights, V)
        
        # Output projection
        attn_out = self.Wo(attn_out)
        
        # Residual connection
        x = x + attn_out
        
        # Pre-LN for FFN
        ln2 = self.ln2(x)
        ffn_out = self.ffn(ln2)
        
        # Residual connection
        return x + ffn_out


class DDNet(nn.Module):
    """
    Dual Descriptor Network with hierarchical structure
    Alternates Descriptor layers and Transformer encoder layers
    
    Optimized with vectorized operations for batch and sequence processing.
    """
    def __init__(self, input_dim=10, model_dims=[8, 6, 3], num_basis_list=[5, 4, 3],
                 attn_ff_hidden_factor=2, ln_eps=1e-5, device='cuda'):
        super().__init__()
        assert len(model_dims) == len(num_basis_list), "model_dims and num_basis_list must align"
        self.input_dim = input_dim
        self.model_dims = model_dims
        self.num_basis_list = num_basis_list
        self.num_desc_layers = len(model_dims)
        self.ln_eps = ln_eps
        self.device = device
        self.trained = False
        
        # Create descriptor layers
        self.desc_layers = nn.ModuleList()
        for l, out_dim in enumerate(model_dims):
            in_dim = input_dim if l == 0 else model_dims[l-1]
            self.desc_layers.append(DescLayer(in_dim, out_dim, num_basis_list[l], ln_eps))
        
        # Create transformer layers
        self.trans_layers = nn.ModuleList()
        for l in range(self.num_desc_layers - 1):
            dim = model_dims[l]
            self.trans_layers.append(TransformerBlock(dim, attn_ff_hidden_factor, ln_eps))
        
        # Readout layer (decoder)
        out_dim = model_dims[-1]
        self.read_W = nn.Linear(out_dim, input_dim, bias=True)
        
        # Initialize readout
        nn.init.uniform_(self.read_W.weight, -0.1, 0.1)
        nn.init.zeros_(self.read_W.bias)
        
        self.to(device)
    
    def forward(self, seq, positions=None):
        """
        Vectorized forward pass for batch and sequence processing.
        Args:
            seq: Input sequence tensor of shape (batch_size, seq_len, input_dim)
            positions: Optional position indices tensor of shape (batch_size, seq_len)
        Returns:
            Output sequence tensor of shape (batch_size, seq_len, model_dims[-1])
        """
        batch_size, seq_len, _ = seq.shape
        
        # Generate position indices if not provided
        if positions is None:
            # Create position indices: [0, 1, 2, ..., seq_len-1] for each batch
            positions = torch.arange(seq_len, device=self.device).expand(batch_size, seq_len)
        
        # Process through descriptor and transformer layers
        current = seq
        for idx in range(self.num_desc_layers):
            # Apply descriptor layer to entire sequence
            current = self.desc_layers[idx](current, positions)
            
            # Apply transformer layer if available
            if idx < len(self.trans_layers):
                current = self.trans_layers[idx](current)
        
        return current
    
    def grad_train(self, seqs, t_list, max_iters=200, lr=0.5, decay_rate=0.995,
                   print_every=10, tol=1e-9):
        """
        Supervised training with target vectors
        Args:
            seqs: List of input sequences, each of shape (seq_len, input_dim)
            t_list: List of target vectors, each of shape (model_dims[-1])
        """
        self.train()
        optimizer = optim.Adam(self.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)
        
        # Convert data to tensors
        seq_tensors = [torch.tensor(seq, dtype=torch.float32, device=self.device) for seq in seqs]
        t_tensors = [torch.tensor(t, dtype=torch.float32, device=self.device) for t in t_list]
        
        history = []
        best_loss = float('inf')
        
        for it in range(max_iters):
            total_loss = 0.0
            total_count = 0
            
            for seq, target in zip(seq_tensors, t_tensors):
                # Forward pass
                outputs = self(seq.unsqueeze(0))  # Add batch dimension
                
                # Average over sequence positions
                pred = outputs.mean(dim=1).squeeze(0)  # (model_dims[-1])
                
                # Compute loss
                loss = torch.mean((pred - target) ** 2)
                total_loss += loss.item()
                total_count += 1
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            # Update learning rate
            scheduler.step()
            
            # Calculate average loss
            avg_loss = total_loss / total_count if total_count > 0 else 0.0
            history.append(avg_loss)
            
            # Print progress
            if it % print_every == 0 or it == max_iters - 1:
                print(f"Iter {it:3d}: Loss = {avg_loss:.6e}, LR = {scheduler.get_last_lr()[0]:.6f}")
            
            # Check for convergence
            if avg_loss < best_loss:
                best_loss = avg_loss
            elif avg_loss >= best_loss - tol:
                print(f"Converged after {it+1} iterations.")
                break
        
        self.trained = True
        return history

    def predict_t(self, seq):
        """Predict target vector for a sequence"""
        self.eval()
        with torch.no_grad():
            seq_tensor = torch.tensor(seq, dtype=torch.float32, device=self.device).unsqueeze(0)
            outputs = self(seq_tensor)
            t_pred = outputs.mean(dim=1).squeeze(0).cpu().numpy()
        return t_pred
    
    def auto_train(self, seqs, mode='gap', max_iters=200, lr=0.5, decay_rate=0.995,
                   print_every=10, tol=1e-9):
        """
        Self-supervised training for sequence modeling
        Args:
            seqs: List of input sequences, each of shape (seq_len, input_dim)
            mode: 'gap' (predict current vector) or 'reg' (predict next vector)
        """
        self.train()
        optimizer = optim.Adam(self.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)
        
        # Convert data to tensors
        seq_tensors = [torch.tensor(seq, dtype=torch.float32, device=self.device) for seq in seqs]
        
        history = []
        best_loss = float('inf')
        
        for it in range(max_iters):
            total_loss = 0.0
            total_count = 0
            
            for seq in seq_tensors:
                # Handle both 2D (seq_len, input_dim) and 3D (batch, seq_len, input_dim) tensors
                if seq.dim() == 2:
                    seq_len, input_dim = seq.shape
                    batch_size = 1
                else:  # dim == 3
                    batch_size, seq_len, input_dim = seq.shape
                
                # Create targets based on mode
                if mode == 'gap':
                    targets = seq  # Predict current vector
                    # Remove mask as it's unused in loss calculation
                else:  # 'reg'
                    targets = seq[1:]  # Predict next vector
                    # Remove last position from input sequence
                    seq = seq[:-1]  # Shape becomes (seq_len-1, input_dim) for 2D
                    seq_len -= 1
                
                # Add batch dimension if needed (for 2D tensors)
                if seq.dim() == 2:
                    seq = seq.unsqueeze(0)  # (1, seq_len, input_dim)
                
                # Forward pass
                outputs = self(seq)  # (batch_size, seq_len, model_dim)
                
                # Decode outputs to input space
                decoded = self.read_W(outputs)  # (batch_size, seq_len, input_dim)
                
                # Remove batch dimension if needed for loss calculation
                if decoded.dim() == 3 and decoded.size(0) == 1:
                    decoded = decoded.squeeze(0)  # (seq_len, input_dim)
                
                # Compute loss
                loss = torch.mean((decoded - targets) ** 2)
                total_loss += loss.item()
                total_count += 1
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            # Update learning rate
            scheduler.step()
            
            # Calculate average loss
            avg_loss = total_loss / total_count if total_count > 0 else 0.0
            history.append(avg_loss)
            
            # Print progress
            if it % print_every == 0 or it == max_iters - 1:
                print(f"Iter {it:3d}: Loss = {avg_loss:.6e}, LR = {scheduler.get_last_lr()[0]:.6f}, mode={mode}")
            
            # Check for convergence
            if avg_loss < best_loss:
                best_loss = avg_loss
            elif avg_loss >= best_loss - tol:
                print(f"Converged after {it+1} iterations.")
                break
        
        self.trained = True
        return history     
    
    def generate(self, L, tau=0.0, start_vec=None):
        """
        Generate a sequence of vectors
        Args:
            L: Length of sequence to generate
            tau: Temperature/noise scale (0 = deterministic)
            start_vec: Optional starting vector (input_dim)
        Returns:
            Generated sequence (L, input_dim)
        """
        self.eval()
        generated = []
        
        # Initialize context
        if start_vec is None:
            context = torch.zeros(1, 1, self.input_dim, device=self.device)
        else:
            context = torch.tensor(start_vec, dtype=torch.float32, device=self.device).view(1, 1, -1)
        
        # Generate sequence
        with torch.no_grad():
            for step in range(L):
                # Forward pass through the network
                outputs = self(context)
                
                # Get last position output
                last_out = outputs[:, -1, :]
                
                # Decode to input space
                decoded = self.read_W(last_out)
                
                # Add noise if specified
                if tau > 0:
                    noise = torch.randn_like(decoded) * tau
                    decoded += noise
                
                # Append to generated sequence and update context
                generated.append(decoded.squeeze(0).cpu().numpy())
                context = torch.cat([context, decoded.unsqueeze(1)], dim=1)
        
        return np.array(generated)
    
    def count_parameters(self):
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def save(self, filename):
        """Save model to file"""
        torch.save(self.state_dict(), filename)
        print(f"Model saved to {filename}")
    
    def load(self, filename):
        """Load model from file"""
        self.load_state_dict(torch.load(filename, map_location=self.device, weights_only=True))
        self.trained = True
        print(f"Model loaded from {filename}")


# ---------------------------
# --- Example usage & test ---
# ---------------------------

def pearson_corr(x, y):
    """Calculate Pearson correlation between two arrays"""
    return pearsonr(x, y)[0]

if __name__ == "__main__":
    # Configuration
    input_dim = 10
    model_dims = [8, 6, 3]
    num_basis_list = [8, 4, 3]
    num_seqs = 100
    min_len, max_len = 100, 200
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {device}")
    print(f"Vectorized DDNet implementation with batch and sequence optimization")
    
    # Create synthetic data
    seqs = []
    t_list = []
    for _ in range(num_seqs):
        L = random.randint(min_len, max_len)
        seq = np.random.uniform(-1, 1, (L, input_dim))
        seqs.append(seq)
        t_list.append(np.random.uniform(-1, 1, model_dims[-1]))
    
    # Initialize model
    dd = DDNet(input_dim, model_dims, num_basis_list, device=device)
    print(f"Model created with {dd.count_parameters()} parameters")
    
    # Supervised training
    print("\nSupervised training...")
    hist_sup = dd.grad_train(seqs, t_list, max_iters=50, lr=0.01, print_every=5)
    print("Supervised training history:", hist_sup[-5:])
    
    # Evaluate predictions
    print("\nEvaluating predictions...")
    preds = [dd.predict_t(seq) for seq in seqs]
    
    # Calculate correlations
    corrs = []
    for dim in range(model_dims[-1]):
        actual = [t[dim] for t in t_list]
        predicted = [p[dim] for p in preds]
        c = pearson_corr(actual, predicted)
        corrs.append(c)
        print(f"Dim {dim}: Correlation = {c:.4f}")
    
    print(f"Average correlation: {np.mean(corrs):.4f}")
    
    # Self-supervised training (gap filling)
    print("\nSelf-supervised training (gap filling)...")
    hist_auto_gap = dd.auto_train(seqs[:3], mode='gap', max_iters=20, lr=0.01, print_every=2)
    print("Gap filling history:", hist_auto_gap[-5:])
    
    # Self-supervised training (next vector prediction)
    print("\nSelf-supervised training (next vector prediction)...")
    hist_auto_reg = dd.auto_train(seqs[:3], mode='reg', max_iters=20, lr=0.01, print_every=2)
    print("Next vector prediction history:", hist_auto_reg[-5:])
    
    # Save and load model
    dd.save("ddnet_model.pth")
    dd_loaded = DDNet(input_dim, model_dims, num_basis_list, device=device)
    dd_loaded.load("ddnet_model.pth")
    
    # Generate sequence
    print("\nGenerating sequence...")
    start_vec = np.random.uniform(-0.5, 0.5, input_dim)
    generated = dd.generate(L=5, tau=0.1, start_vec=start_vec)
    print("Generated sequence shape:", generated.shape)
    print("First 2 vectors:")
    print(generated[:2])
    
    # Test loaded model
    print("\nTesting loaded model...")
    pred_original = dd.predict_t(seqs[0])
    pred_loaded = dd_loaded.predict_t(seqs[0])
    diff = np.mean(np.abs(pred_original - pred_loaded))
    print(f"Prediction difference: {diff:.6f}")
    assert diff < 1e-6, "Loaded model produces different results!"
    
    print("\nAll tests completed successfully!")
