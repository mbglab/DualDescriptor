# Copyright (C) 2005-2025, Bin-Guang Ma (mbg@mail.hzau.edu.cn); SPDX-License-Identifier: MIT
# Hierarchical Dual Descriptor (Random AB matrix form) with Character Sequence Input (HierDDLrnC)
# Combines character-level processing and hierarchical vector processing with Linker matrices
# This program is for the demonstration of methodology and not fully refined.
# Author: Bin-Guang Ma (assisted by DeepSeek, Gemini); Date: 2025-10-12 ~ 2026-01-17

import math
import random
import itertools
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy, os

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
        residual_mode (str or None): Residual connection type. Options are:
            - 'separate': use separate projection and Linker for residual
            - 'shared': share M and Linker for residual
            - None: no residual connection
        device (str): Device to place the layer on ('cuda' or 'cpu')
    """
    def __init__(self, in_dim, out_dim, basis_dim, in_seq_len, out_seq_len, 
                 linker_trainable, residual_mode=None, device='cuda'):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.basis_dim = basis_dim
        self.in_seq_len = in_seq_len
        self.out_seq_len = out_seq_len
        self.residual_mode = residual_mode
        self.linker_trainable = linker_trainable
        self.device = device
        
        # Initialize transformation matrix M
        self.M = nn.Parameter(torch.empty(out_dim, in_dim, device=device))
        nn.init.uniform_(self.M, -0.1, 0.1)
        
        # Initialize coefficient matrix Acoeff
        self.Acoeff = nn.Parameter(torch.empty(out_dim, basis_dim, device=device))
        nn.init.uniform_(self.Acoeff, -0.1, 0.1)
        
        # Initialize basis matrix Bbasis
        self.Bbasis = nn.Parameter(torch.empty(basis_dim, out_dim, device=device))
        nn.init.uniform_(self.Bbasis, -0.1, 0.1)
        
        # Initialize Linker matrix
        self.Linker = nn.Parameter(torch.empty(in_seq_len, out_seq_len, device=device))
        nn.init.uniform_(self.Linker, -0.1, 0.1)
        
        # Freeze Linker if not trainable
        if not linker_trainable:
            self.Linker.requires_grad = False
            
        # Layer normalization
        self.layer_norm = nn.LayerNorm(out_dim, device=device)
        
        # Precompute basis indices for sequence positions
        # Optimization: Buffer is useful for broadcasting in forward
        self.register_buffer('basis_indices', 
                            torch.tensor([i % basis_dim for i in range(in_seq_len)], device=device))
        
        # Residual connection setup
        if self.residual_mode == 'separate':
            # Separate projection and Linker for residual path
            # Feature dimension projection
            self.residual_proj = nn.Linear(in_dim, out_dim, bias=False, device=device)
            nn.init.uniform_(self.residual_proj.weight, -0.1, 0.1)
            
            # Sequence length transformation matrix for residual
            self.residual_linker = nn.Parameter(torch.empty(in_seq_len, out_seq_len, device=device))
            nn.init.uniform_(self.residual_linker, -0.1, 0.1)
            
            # Freeze residual Linker if main Linker is not trainable
            if not linker_trainable:
                self.residual_linker.requires_grad = False
            
    def forward(self, x):
        """
        Forward pass for the layer
        Optimization: Fully vectorized for (batch_size, in_seq_len, in_dim)
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, in_seq_len, in_dim)
            
        Returns:
            Tensor: Output tensor of shape (batch_size, out_seq_len, out_dim)
        """
        # Save for residual connection
        residual = x
        
        # Main path processing
        # Apply linear transformation: (batch_size, in_seq_len, in_dim) -> (batch_size, in_seq_len, out_dim)
        x = torch.matmul(x, self.M.T)
        
        # Apply layer normalization
        x = self.layer_norm(x)

        # Position-wise processing
        # Select basis vectors for each position: (in_seq_len, out_dim)
        basis_vectors = self.Bbasis[self.basis_indices]
        
        # Compute scalars: (batch_size, in_seq_len)
        # Optimization: einsum handles batch dimension naturally
        scalars = torch.einsum('bsd,sd->bs', x, basis_vectors)
        
        # Select coefficients for each position: (in_seq_len, out_dim)
        coeffs = self.Acoeff[:, self.basis_indices].permute(1, 0)
        
        # Compute position outputs: (batch_size, in_seq_len, out_dim)
        # Broadcasting scalars (batch, seq, 1) * coeffs (1, seq, dim) -> (batch, seq, dim)
        u = coeffs.unsqueeze(0) * scalars.unsqueeze(-1)
        
        # Apply sequence length transformation using Linker matrix
        # (batch_size, in_seq_len, out_dim) -> permute -> (batch_size, out_dim, in_seq_len)
        # matmul with Linker (in_seq, out_seq) -> (batch_size, out_dim, out_seq)
        v = torch.matmul(u.permute(0, 2, 1), self.Linker)
        
        # Transpose to (batch_size, out_seq_len, out_dim)
        main_output = v.permute(0, 2, 1)
        
        # Residual connection processing
        if self.residual_mode == 'separate':
            # Separate projection and Linker for residual
            # Feature projection
            residual_feat = self.residual_proj(residual)
            
            # Sequence transformation for residual
            residual_seq = torch.matmul(residual_feat.permute(0, 2, 1), self.residual_linker)
            residual_out = residual_seq.permute(0, 2, 1)
            
            # Add residual
            out = main_output + residual_out
            
        elif self.residual_mode == 'shared':
            # Shared M and Linker for residual
            # Feature projection using shared M
            residual_feat = torch.matmul(residual, self.M.T)
            
            # Sequence transformation using shared Linker
            residual_seq = torch.matmul(residual_feat.permute(0, 2, 1), self.Linker)
            residual_out = residual_seq.permute(0, 2, 1)
            
            # Add residual
            out = main_output + residual_out
            
        else:
            # Identity residual only if dimensions match
            if self.in_dim == self.out_dim and self.in_seq_len == self.out_seq_len:
                # Sequence transformation for residual
                residual_seq = torch.matmul(residual.permute(0, 2, 1), self.Linker)
                residual_out = residual_seq.permute(0, 2, 1)
                out = main_output + residual_out
            else:
                out = main_output
       
        return out

class HierDDLrnC(nn.Module):
    """
    Hierarchical Dual Descriptor with Linker and Character Sequence Input
    Combines character-level processing and hierarchical Linker-based layers
    """
    def __init__(self, charset, rank=1, rank_mode='drop', vec_dim=2, 
                 mode='linear', user_step=None, model_dims=[2], basis_dims=[50],
                 linker_dims=[50], input_seq_len=100, linker_trainable=False,
                 use_residual_list=None, device='cuda'):
        super().__init__()
        self.charset = list(charset)
        self.rank = rank    # r-per/k-mer length
        self.rank_mode = rank_mode # 'pad' or 'drop'
        self.vec_dim = vec_dim    # embedding dimension
        self.mode = mode
        self.step = user_step
        self.model_dims = model_dims
        self.basis_dims = basis_dims
        self.linker_dims = linker_dims
        self.input_seq_len = input_seq_len  # Fixed input sequence length
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.trained = False        
        
        # Character-level processing components
        # 1) Generate all possible tokens (k-mers + right-padded with '_')
        toks = []
        if self.rank_mode=='pad':
            for r in range(1, self.rank+1):
                for prefix in itertools.product(self.charset, repeat=r):
                    tok = ''.join(prefix).ljust(self.rank, '_')
                    toks.append(tok)
        else:
            toks = [''.join(p) for p in itertools.product(self.charset, repeat=self.rank)]
        self.tokens = sorted(set(toks))
        self.token_to_idx = {token: idx for idx, token in enumerate(self.tokens)}
        self.idx_to_token = {idx: token for idx, token in enumerate(self.tokens)}
        
        # Embedding layer for tokens
        self.char_embedding = nn.Embedding(len(self.tokens), self.vec_dim)
        
        # Character layer random AB matrix components
        self.char_basis_dim = 50  # Fixed basis dimension for character layer
        self.char_linear = nn.Linear(self.vec_dim, self.vec_dim, bias=False)
        self.char_Acoeff = nn.Parameter(torch.Tensor(self.vec_dim, self.char_basis_dim))
        self.char_Bbasis = nn.Parameter(torch.Tensor(self.char_basis_dim, self.vec_dim))
        
        # Initialize character layer parameters
        nn.init.uniform_(self.char_embedding.weight, -0.5, 0.5)
        nn.init.uniform_(self.char_linear.weight, -0.1, 0.1)
        nn.init.uniform_(self.char_Acoeff, -0.1, 0.1)
        nn.init.uniform_(self.char_Bbasis, -0.1, 0.1)
        
        # Training statistics for character layer
        self.mean_token_count = None
        self.mean_t = None
        
        # Calculate token sequence length from character sequence length
        self.token_seq_len = self._calculate_token_seq_len(input_seq_len)
        
        # Hierarchical layers with Linker (starting from Layer 1)
        self.hierarchical_layers = nn.ModuleList()
        in_dim = vec_dim  # Input to first hierarchical layer is output from char layer
        in_seq_len = self.token_seq_len  # Input sequence length for first hierarchical layer
        
        if use_residual_list is None:
            use_residual_list = [None] * len(model_dims)
            
        if len(basis_dims) != len(model_dims):
            raise ValueError("basis_dims length must match model_dims length")
        if len(linker_dims) != len(model_dims):
            raise ValueError("linker_dims length must match model_dims length")
            
        # Process linker_trainable parameter
        if isinstance(linker_trainable, bool):
            self.linker_trainable = [linker_trainable] * len(model_dims)
        elif isinstance(linker_trainable, list):
            if len(linker_trainable) != len(model_dims):
                raise ValueError("linker_trainable list length must match number of layers")
            self.linker_trainable = linker_trainable
        else:
            raise TypeError("linker_trainable must be bool or list of bools")
            
        for i, (out_dim, basis_dim, out_seq_len) in enumerate(zip(model_dims, basis_dims, linker_dims)):
            layer = Layer(
                in_dim=in_dim,
                out_dim=out_dim,
                basis_dim=basis_dim,
                in_seq_len=in_seq_len,
                out_seq_len=out_seq_len,
                linker_trainable=self.linker_trainable[i],
                residual_mode=use_residual_list[i],
                device=self.device
            )
            self.hierarchical_layers.append(layer)
            in_dim = out_dim  # Next layer input is current layer output
            in_seq_len = out_seq_len  # Next layer input sequence length is current layer output sequence length
        
        # Classification head for multi-class tasks (initialized when needed)
        self.num_classes = None
        self.classifier = None
        
        # Label head for multi-label tasks (initialized when needed)
        self.num_labels = None
        self.labeller = None
        
        # Mean target vector for the final hierarchical layer output
        self.mean_target = None
        
        self.to(self.device)
    
    def _calculate_token_seq_len(self, char_seq_len):
        """Calculate token sequence length from character sequence length based on tokenization mode"""
        if self.mode == 'linear':
            # Linear mode: sliding window with step=1
            return max(0, char_seq_len - self.rank + 1)
        else:
            # Nonlinear mode: stepping with custom step size
            step = self.step or self.rank
            if self.rank_mode == 'pad':
                # Always pad to get full tokens
                return (char_seq_len + step - 1) // step
            else:
                # Drop incomplete tokens
                return max(0, (char_seq_len - self.rank + step) // step)
    
    def extract_tokens(self, seq):
        """
        Extract k-mer tokens from a character sequence based on tokenization mode.
        Optimization: This is used for generation/reconstruction or single inference.
        For training, we use batch_token_indices.
        """
        L = len(seq)
        if self.mode == 'linear':
            return [seq[i:i+self.rank] for i in range(L - self.rank + 1)]
        
        toks = []
        step = self.step or self.rank
        
        for i in range(0, L, step):
            frag = seq[i:i+self.rank]
            frag_len = len(frag)
            
            if self.rank_mode == 'pad':
                toks.append(frag if frag_len == self.rank else frag.ljust(self.rank, '_'))
            elif self.rank_mode == 'drop':
                if frag_len == self.rank:
                    toks.append(frag)
        return toks

    def batch_token_indices(self, seqs, device=None):
        """
        Optimization: Convert a batch of strings to indices tensor in one go.
        Performs tokenization on CPU using list comprehensions (fast for strings)
        then moves to target device as a single tensor.
        
        Args:
            seqs (list): List of strings.
            device (torch.device): Target device (defaults to self.device)
        
        Returns:
            Tensor: LongTensor of shape [batch_size, token_seq_len]
        """
        target_device = device if device is not None else self.device
        
        # Prepare params
        step = 1 if self.mode == 'linear' else (self.step or self.rank)
        
        all_indices = []
        
        # Optimization: Use fast python string slicing
        # We assume all sequences are input_seq_len (checked in caller)
        # Pre-calculating ranges avoids re-computing range object every time
        L = self.input_seq_len
        
        if self.mode == 'linear':
            ranges = range(L - self.rank + 1)
            for seq in seqs:
                # Direct lookup is faster than function call
                toks = [seq[i:i+self.rank] for i in ranges]
                all_indices.append([self.token_to_idx[t] for t in toks])
                
        else: # nonlinear
            ranges = range(0, L, step)
            for seq in seqs:
                seq_indices = []
                for i in ranges:
                    frag = seq[i:i+self.rank]
                    if len(frag) == self.rank:
                        seq_indices.append(self.token_to_idx[frag])
                    else:
                        if self.rank_mode == 'pad':
                            frag_pad = frag.ljust(self.rank, '_')
                            seq_indices.append(self.token_to_idx[frag_pad])
                        # if drop, we do nothing (but fixed length implies we usually shouldn't drop variably)
                all_indices.append(seq_indices)

        if not all_indices:
             return torch.zeros((len(seqs), 0), dtype=torch.long, device=target_device)

        return torch.tensor(all_indices, dtype=torch.long, device=target_device)

    def char_forward(self, token_indices):
        """
        Optimization: Vectorized computation of N(k) vectors for a BATCH of sequences.
        Replaces char_batch_compute_Nk (which was 1D) with 2D support.
        
        Args:
            token_indices: Tensor of token indices [batch_size, seq_len]
        Returns:
            Tensor of N(k) vectors [batch_size, seq_len, vec_dim]
        """
        batch_size, seq_len = token_indices.shape
        
        # Get embeddings: [batch_size, seq_len, vec_dim]
        x = self.char_embedding(token_indices)
        
        # Linear transformation (broadcasts over batch and seq)
        transformed = self.char_linear(x)
        
        # Apply layer normalization
        normalized = torch.nn.functional.layer_norm(transformed, (self.vec_dim,))
        
        # Compute basis indices: j = k % basis_dim for each position
        # Optimization: Pre-compute/broadcast indices. Same for all batches.
        j_indices = torch.arange(seq_len, device=self.device) % self.char_basis_dim
        
        # Select basis vectors: (seq_len, vec_dim)
        B_vectors = self.char_Bbasis[j_indices]
        
        # Compute scalars: dot product of normalized and B_vectors
        # Optimization: einsum over batch(b), seq(s), dim(d)
        # normalized [b, s, d], B_vectors [s, d] -> scalars [b, s]
        scalars = torch.einsum('bsd,sd->bs', normalized, B_vectors)
        
        # Select coefficient vectors: (seq_len, vec_dim)
        # Acoeff [dim, basis_dim] -> select cols based on j_indices -> [dim, seq_len] -> transpose -> [seq_len, dim]
        A_vectors = self.char_Acoeff[:, j_indices].t()
        
        # Compute new features: scalars * A_vectors
        # scalars [b, s] -> [b, s, 1]
        # A_vectors [s, d] -> broadcast to [b, s, d]
        Nk = scalars.unsqueeze(2) * A_vectors.unsqueeze(0)
            
        return Nk

    # Kept for backward compatibility/single sequence generation calls
    def char_batch_compute_Nk(self, token_indices):
        """Wrapper for old 1D calls -> delegates to vectorized implementation"""
        if token_indices.dim() == 1:
            return self.char_forward(token_indices.unsqueeze(0)).squeeze(0)
        return self.char_forward(token_indices)

    def char_describe(self, seq):
        """Compute N(k) vectors for each k-mer in sequence (character-level processing)"""
        toks = self.extract_tokens(seq)
        if not toks:
            return []
        token_indices = torch.tensor([self.token_to_idx[tok] for tok in toks], device=self.device)
        return self.char_batch_compute_Nk(token_indices).detach().cpu().numpy()

    def char_reconstruct(self):
        """Reconstruct representative sequence by minimizing error (character-level)"""
        assert self.trained, "Model must be trained first"
        n_tokens = round(self.mean_token_count)
        mean_t_tensor = torch.tensor(self.mean_t, dtype=torch.float32, device=self.device)
        seq_tokens = []
        
        all_token_indices = torch.arange(len(self.tokens), device=self.device)
        
        # Optimization: We can compute ALL candidates for ALL positions in batches if needed,
        # but reconstruction is usually one-off. Keep logic simple but correct.
        for k in range(n_tokens):
            # Simulate position k for all tokens
            # Create a fake "batch" where each row is a candidate token at position k
            # But char_batch_compute_Nk expects position to be implicit in sequence.
            # We must manually construct the computation for position k.
            
            # Manual computation for efficiency
            emb = self.char_embedding(all_token_indices) # [NumTokens, Dim]
            trans = self.char_linear(emb)
            norm = torch.nn.functional.layer_norm(trans, (self.vec_dim,))
            
            # Basis at position k
            j = k % self.char_basis_dim
            B_vec = self.char_Bbasis[j] # [Dim]
            scalar = torch.matmul(norm, B_vec) # [NumTokens]
            
            A_vec = self.char_Acoeff[:, j] # [Dim]
            Nk_candidates = scalar.unsqueeze(1) * A_vec.unsqueeze(0) # [NumTokens, Dim]
            
            errors = torch.sum((Nk_candidates - mean_t_tensor) ** 2, dim=1)
            min_idx = torch.argmin(errors).item()
            seq_tokens.append(self.idx_to_token[min_idx])
            
        reconstructed_seq = ''.join(seq_tokens)
        if '_' in reconstructed_seq:
            reconstructed_seq = reconstructed_seq.replace('_', '')
        return reconstructed_seq

    def char_generate(self, L, tau=0.0):
        """Generate sequence of length L with temperature-controlled randomness (character-level)"""
        assert self.trained, "Model must be trained first"
        if tau < 0: raise ValueError("Temperature must be non-negative")
            
        num_blocks = (L + self.rank - 1) // self.rank
        mean_t_tensor = torch.tensor(self.mean_t, dtype=torch.float32, device=self.device)
        generated_tokens = []
        all_token_indices = torch.arange(len(self.tokens), device=self.device)
        
        for k in range(num_blocks):
            # Fast manual compute for position k
            emb = self.char_embedding(all_token_indices)
            trans = self.char_linear(emb)
            norm = torch.nn.functional.layer_norm(trans, (self.vec_dim,))
            
            j = k % self.char_basis_dim
            B_vec = self.char_Bbasis[j]
            scalar = torch.matmul(norm, B_vec)
            A_vec = self.char_Acoeff[:, j]
            Nk_candidates = scalar.unsqueeze(1) * A_vec.unsqueeze(0)
            
            errors = torch.sum((Nk_candidates - mean_t_tensor) ** 2, dim=1)
            scores = -errors
            
            if tau == 0:
                max_idx = torch.argmax(scores).item()
                generated_tokens.append(self.idx_to_token[max_idx])
            else:
                probs = torch.softmax(scores / tau, dim=0).detach().cpu().numpy()
                chosen_idx = np.random.choice(len(self.tokens), p=probs)
                generated_tokens.append(self.idx_to_token[chosen_idx])
                
        full_seq = ''.join(generated_tokens)
        if '_' in full_seq: full_seq = full_seq.replace('_', '')
        return full_seq[:L]
    
    def char_sequence_to_tensor(self, seq):
        """Convert single character sequence to tensor"""
        if len(seq) != self.input_seq_len:
            raise ValueError(f"Input sequence length must be {self.input_seq_len}, got {len(seq)}")
        
        # Optimization: reuse batch logic for consistency
        return self.batch_sequence_to_tensor([seq])
    
    def batch_sequence_to_tensor(self, seqs):
        """Helper to go from strings -> token indices -> char layer output"""
        indices = self.batch_token_indices(seqs) # [Batch, SeqLen]
        return self.char_forward(indices) # [Batch, SeqLen, Dim]
    
    def forward(self, input_data):
        """
        Forward pass through entire hierarchical model.
        Optimization: Accepts either raw strings OR pre-computed token indices.
        Process inputs as a batch directly.
        
        Args:
            input_data: List of strings OR Tensor of token indices [Batch, SeqLen]
        """
        # Handle input types
        if isinstance(input_data, list):
            # Batch of strings
            token_indices = self.batch_token_indices(input_data)
        elif isinstance(input_data, torch.Tensor):
            # Indices tensor (already on device or needs move)
            token_indices = input_data.to(self.device)
            if token_indices.dim() == 1: # Single sequence case
                token_indices = token_indices.unsqueeze(0)
        elif isinstance(input_data, str):
            # Single string
            token_indices = self.batch_token_indices([input_data])
        else:
            raise TypeError("Input must be list of strings, single string, or Tensor")
            
        # 1. Character Layer Forward (Vectorized)
        x = self.char_forward(token_indices) # [Batch, SeqLen, Dim]
        
        # 2. Hierarchical Layers Forward
        for layer in self.hierarchical_layers:
            x = layer(x)
            
        return x
    
    def predict_t(self, seq):
        """Predict target vector for a character sequence"""
        with torch.no_grad():
            output = self.forward(seq)  # [1, final_linker_dim, final_model_dim]
            target = output.mean(dim=1)  # Average over sequence length
            return target.squeeze(0).cpu().numpy()
    
    def reg_train(self, seqs, t_list, max_iters=1000, tol=1e-8, learning_rate=0.01, 
                   continued=False, decay_rate=1.0, print_every=10, batch_size=1024,
                   checkpoint_file=None, checkpoint_interval=10):
        """
        Optimization: Fully vectorized batch training.
        Pre-computes indices to avoid CPU overhead during training loop.
        """
        # Validate input lengths
        if any(len(s) != self.input_seq_len for s in seqs):
            raise ValueError(f"All sequences must have length {self.input_seq_len}")
        
        # Load checkpoint logic (kept same as original)
        start_iter = 0
        best_loss = float('inf')
        best_model_state = None
        
        if continued and checkpoint_file and os.path.exists(checkpoint_file):
            checkpoint = torch.load(checkpoint_file, map_location=self.device, weights_only=False)
            self.load_state_dict(checkpoint['model_state_dict'])
            optimizer_state = checkpoint['optimizer_state_dict']
            scheduler_state = checkpoint['scheduler_state_dict']
            history = checkpoint['history']
            start_iter = checkpoint['iteration'] + 1
            best_loss = checkpoint.get('best_loss', float('inf'))
            print(f"Resumed training from checkpoint at iteration {start_iter}, best loss: {best_loss:.6e}")
        else:
            if not continued: self.reset_parameters() 
            history = []
        
        # Optimization: Pre-process data
        # Convert strings to indices on CPU first to save GPU memory if dataset is huge,
        # or move to GPU if it fits. Here we keep on CPU and transfer batches.
        print("Pre-processing training data...")
        all_indices = self.batch_token_indices(seqs, device='cpu') 
        all_targets = torch.tensor(t_list, dtype=torch.float32, device='cpu')
        num_samples = len(seqs)
        
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)
        
        if continued and checkpoint_file and os.path.exists(checkpoint_file):
            optimizer.load_state_dict(optimizer_state)
            scheduler.load_state_dict(scheduler_state)
        
        prev_loss = float('inf') if start_iter == 0 else history[-1] if history else float('inf')
        
        for it in range(start_iter, max_iters):
            total_loss = 0.0
            
            # Shuffle indices
            perm = torch.randperm(num_samples)
            
            for i in range(0, num_samples, batch_size):
                # Optimization: Efficient batch slicing and transfer
                batch_idx = perm[i:i+batch_size]
                batch_indices = all_indices[batch_idx].to(self.device)
                batch_t = all_targets[batch_idx].to(self.device) # [B, TargetDim]
                
                optimizer.zero_grad()
                
                # Optimization: Single forward pass for the whole batch
                # forward() now accepts tensor input directly
                output = self.forward(batch_indices) # [B, OutSeqLen, OutDim]
                
                # Vectorized Loss Calculation
                # Mean over sequence dimension
                pred_target = output.mean(dim=1) # [B, OutDim]
                
                # Sum of squared errors (sum over dim, sum over batch)
                loss = torch.sum((pred_target - batch_t) ** 2)
                
                # Normalize loss for reporting (average per sample) 
                # Original code summed loss, then divided by batch size later.
                # Here we follow original logic: sum squared error.
                batch_loss_val = loss.item()
                
                # Normalize for backward to be scale-invariant of batch size?
                # Original code: `batch_loss = batch_loss / len(batch_seqs)`
                loss = loss / len(batch_idx)
                
                loss.backward()
                optimizer.step()
                
                total_loss += batch_loss_val # Sum of all SE
                
            avg_loss = total_loss / num_samples
            history.append(avg_loss)
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_model_state = copy.deepcopy(self.state_dict())
            
            if it % print_every == 0 or it == max_iters - 1:
                print(f"GD Iter {it:3d}: Loss = {avg_loss:.6e}, LR = {scheduler.get_last_lr()[0]:.6f}")
            
            if checkpoint_file and (it % checkpoint_interval == 0 or it == max_iters - 1):
                self._save_checkpoint(checkpoint_file, it, history, optimizer, scheduler, best_loss)
            
            if abs(prev_loss - avg_loss) < tol:
                print(f"Converged after {it+1} iterations.")
                if best_model_state is not None and avg_loss > prev_loss:
                    self.load_state_dict(best_model_state)                    
                    print(f"Restored best model state with loss = {best_loss:.6e}")
                    history[-1] = best_loss
                break
            prev_loss = avg_loss
            scheduler.step()

        if best_model_state is not None and best_loss < history[-1]:
            self.load_state_dict(best_model_state)  
            print(f"Training ended. Restored best model state with loss = {best_loss:.6e}")
            history[-1] = best_loss
        
        self._compute_training_statistics(seqs)
        self._compute_hierarchical_mean_target(seqs)
        self.trained = True
        return history

    def cls_train(self, seqs, labels, num_classes, max_iters=1000, tol=1e-8, learning_rate=0.01,
                  continued=False, decay_rate=1.0, print_every=10, batch_size=32,
                  checkpoint_file=None, checkpoint_interval=10):
        """
        Train the model for multi-class classification using cross-entropy loss.
        Optimized for HierDDLrnC architecture with batch processing.
        
        Args:
            seqs: List of character sequences for training
            labels: List of integer class labels (0 to num_classes-1)
            num_classes: Number of classes in the classification problem
            max_iters: Maximum number of training iterations
            tol: Convergence tolerance
            learning_rate: Initial learning rate for optimizer
            continued: Whether to continue training from existing parameters
            decay_rate: Learning rate decay rate
            print_every: Print progress every N iterations
            batch_size: Number of sequences to process in each batch
            checkpoint_file: Path to save training checkpoints
            checkpoint_interval: Save checkpoint every N iterations
            
        Returns:
            list: Training loss history
        """
        
        # Validate input lengths
        if any(len(s) != self.input_seq_len for s in seqs):
            raise ValueError(f"All sequences must have length {self.input_seq_len}")
        
        # Initialize classification head if not already done
        if self.classifier is None or self.num_classes != num_classes:
            self.classifier = nn.Linear(self.model_dims[-1], num_classes).to(self.device)
            self.num_classes = num_classes
        
        if not continued:
            self.reset_parameters()
        
        # Load checkpoint if continuing
        start_iter = 0
        best_loss = float('inf')
        best_model_state = None
        
        if continued and checkpoint_file and os.path.exists(checkpoint_file):
            checkpoint = torch.load(checkpoint_file, map_location=self.device, weights_only=False)
            self.load_state_dict(checkpoint['model_state_dict'])
            optimizer_state = checkpoint['optimizer_state_dict']
            scheduler_state = checkpoint['scheduler_state_dict']
            history = checkpoint['history']
            start_iter = checkpoint['iteration'] + 1
            best_loss = checkpoint.get('best_loss', float('inf'))
            print(f"Resumed training from checkpoint at iteration {start_iter}, best loss: {best_loss:.6e}")
            
            # Load classifier state if available
            if 'classifier_state_dict' in checkpoint:
                self.classifier.load_state_dict(checkpoint['classifier_state_dict'])
        else:
            history = []
        
        # Convert labels to tensor
        label_tensors = torch.tensor(labels, dtype=torch.long, device=self.device)
        
        # Pre-process sequences to indices
        all_indices = self.batch_token_indices(seqs, device='cpu')
        num_samples = len(seqs)
        
        # Setup optimizer and scheduler
        # Include classifier parameters in optimization
        optimizer = optim.Adam(list(self.parameters()), # + list(self.classifier.parameters())
                              lr=learning_rate)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)
        
        # Load optimizer and scheduler states if resuming
        if continued and checkpoint_file and os.path.exists(checkpoint_file):
            optimizer.load_state_dict(optimizer_state)
            scheduler.load_state_dict(scheduler_state)
        
        # Cross-entropy loss
        criterion = nn.CrossEntropyLoss()
        
        # Training state variables
        prev_loss = float('inf') if start_iter == 0 else history[-1] if history else float('inf')
        
        for it in range(start_iter, max_iters):
            total_loss = 0.0
            total_sequences = 0
            correct_predictions = 0
            
            # Shuffle indices for each epoch
            perm = torch.randperm(num_samples)
            
            # Process sequences in batches
            for i in range(0, num_samples, batch_size):
                batch_idx = perm[i:i+batch_size]
                batch_indices = all_indices[batch_idx].to(self.device)
                batch_labels = label_tensors[batch_idx]
                
                optimizer.zero_grad()
                
                # Forward pass through hierarchical model
                output = self.forward(batch_indices)  # [B, SeqLen, FinalDim]
                
                # Compute sequence representation: average over sequence length
                seq_vectors = output.mean(dim=1)  # [B, FinalDim]
                
                # Get class logits through classification head
                logits = self.classifier(seq_vectors)  # [B, NumClasses]
                
                # Calculate loss
                loss = criterion(logits, batch_labels)
                loss.backward()
                optimizer.step()
                
                # Calculate batch statistics
                batch_loss = loss.item()
                total_loss += batch_loss * len(batch_idx)
                total_sequences += len(batch_idx)
                
                # Calculate accuracy for this batch
                with torch.no_grad():
                    predictions = torch.argmax(logits, dim=1)
                    correct_predictions += (predictions == batch_labels).sum().item()
                
                # Clear GPU cache periodically
                if torch.cuda.is_available() and (i // batch_size) % 10 == 0:
                    torch.cuda.empty_cache()
            
            # Calculate average loss and accuracy for this iteration
            if total_sequences > 0:
                avg_loss = total_loss / total_sequences
                accuracy = correct_predictions / total_sequences
            else:
                avg_loss = 0.0
                accuracy = 0.0
                
            history.append(avg_loss)
            
            # Update best model state
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_model_state = copy.deepcopy(self.state_dict())
                best_classifier_state = copy.deepcopy(self.classifier.state_dict())
            
            # Print training progress
            if it % print_every == 0 or it == max_iters - 1:
                current_lr = scheduler.get_last_lr()[0]
                print(f"CLS-Train Iter {it:3d}: Loss = {avg_loss:.6e}, Acc = {accuracy:.4f}, LR = {current_lr:.6f}")
            
            # Save checkpoint if specified
            if checkpoint_file and (it % checkpoint_interval == 0 or it == max_iters - 1):
                checkpoint = {
                    'iteration': it,
                    'model_state_dict': self.state_dict(),
                    'classifier_state_dict': self.classifier.state_dict() if self.classifier else None,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'history': history,
                    'best_loss': best_loss,
                    'num_classes': self.num_classes
                }
                torch.save(checkpoint, checkpoint_file)
                #print(f"Checkpoint saved at iteration {it}")
            
            # Check convergence
            if abs(prev_loss - avg_loss) < tol:
                print(f"Converged after {it+1} iterations.")
                # Restore best model state
                if best_model_state is not None:
                    self.load_state_dict(best_model_state)
                    self.classifier.load_state_dict(best_classifier_state)
                break
                
            prev_loss = avg_loss
            
            # Learning rate scheduling
            scheduler.step()
        
        self.trained = True
        
        return history

    def lbl_train(self, seqs, labels, num_labels, max_iters=1000, tol=1e-8, learning_rate=0.01, 
                 continued=False, decay_rate=1.0, print_every=10, batch_size=32,
                 checkpoint_file=None, checkpoint_interval=10, pos_weight=None):
        """
        Train the model for multi-label classification using binary cross-entropy loss.
        Optimized for HierDDLrnC architecture with batch processing.
        
        Args:
            seqs: List of character sequences for training
            labels: List of binary label vectors (list of lists) or 2D numpy array/torch tensor
            num_labels: Number of labels for multi-label prediction task
            max_iters: Maximum number of training iterations
            tol: Convergence tolerance
            learning_rate: Initial learning rate for optimizer
            continued: Whether to continue training from existing parameters
            decay_rate: Learning rate decay rate
            print_every: Print progress every N iterations
            batch_size: Number of sequences to process in each batch
            checkpoint_file: Path to save training checkpoints
            checkpoint_interval: Save checkpoint every N iterations
            pos_weight: Weight for positive class (torch.Tensor of shape [num_labels])
            
        Returns:
            list: Training loss history
            list: Training accuracy history
        """
        
        # Validate input lengths
        if any(len(s) != self.input_seq_len for s in seqs):
            raise ValueError(f"All sequences must have length {self.input_seq_len}")
        
        # Initialize label head if not already done
        if self.labeller is None or self.num_labels != num_labels:
            self.labeller = nn.Linear(self.model_dims[-1], num_labels).to(self.device)
            self.num_labels = num_labels
        
        if not continued:
            self.reset_parameters()
        
        # Load checkpoint if continuing
        start_iter = 0
        best_loss = float('inf')
        best_model_state = None
        
        if continued and checkpoint_file and os.path.exists(checkpoint_file):
            checkpoint = torch.load(checkpoint_file, map_location=self.device, weights_only=False)
            self.load_state_dict(checkpoint['model_state_dict'])
            optimizer_state = checkpoint['optimizer_state_dict']
            scheduler_state = checkpoint['scheduler_state_dict']
            loss_history = checkpoint.get('loss_history', [])
            acc_history = checkpoint.get('acc_history', [])
            start_iter = checkpoint['iteration'] + 1
            best_loss = checkpoint.get('best_loss', float('inf'))
            print(f"Resumed training from checkpoint at iteration {start_iter}, best loss: {best_loss:.6e}")
            
            # Load labeller state if available
            if 'labeller_state_dict' in checkpoint:
                self.labeller.load_state_dict(checkpoint['labeller_state_dict'])
        else:
            loss_history = []
            acc_history = []
        
        # Convert labels to tensor
        if isinstance(labels, list):
            labels_tensor = torch.tensor(labels, dtype=torch.float32, device=self.device)
        else:
            labels_tensor = torch.as_tensor(labels, dtype=torch.float32, device=self.device)
        
        # Pre-process sequences to indices
        all_indices = self.batch_token_indices(seqs, device='cpu')
        num_samples = len(seqs)
        
        # Setup loss function with optional positive class weighting
        if pos_weight is not None:
            pos_weight_tensor = torch.tensor(pos_weight, dtype=torch.float32, device=self.device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        else:
            criterion = nn.BCEWithLogitsLoss()
        
        # Setup optimizer and scheduler
        # Include labeller parameters in optimization
        optimizer = optim.Adam(list(self.parameters()), # + list(self.labeller.parameters())
                              lr=learning_rate)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)
        
        # Load optimizer and scheduler states if resuming
        if continued and checkpoint_file and os.path.exists(checkpoint_file):
            optimizer.load_state_dict(optimizer_state)
            scheduler.load_state_dict(scheduler_state)
        
        # Training state variables
        prev_loss = float('inf') if start_iter == 0 else loss_history[-1] if loss_history else float('inf')
        
        for it in range(start_iter, max_iters):
            total_loss = 0.0
            total_correct = 0
            total_predictions = 0
            total_sequences = 0
            
            # Shuffle indices for each epoch
            perm = torch.randperm(num_samples)
            
            # Process sequences in batches
            for i in range(0, num_samples, batch_size):
                batch_idx = perm[i:i+batch_size]
                batch_indices = all_indices[batch_idx].to(self.device)
                batch_labels = labels_tensor[batch_idx]
                
                optimizer.zero_grad()
                
                # Forward pass through hierarchical model
                output = self.forward(batch_indices)  # [B, SeqLen, FinalDim]
                
                # Compute sequence representation: average over sequence length
                seq_vectors = output.mean(dim=1)  # [B, FinalDim]
                
                # Get label logits through classification head
                logits = self.labeller(seq_vectors)  # [B, NumLabels]
                
                # Calculate loss
                loss = criterion(logits, batch_labels)
                loss.backward()
                optimizer.step()
                
                # Calculate batch statistics
                batch_loss = loss.item()
                total_loss += batch_loss * len(batch_idx)
                total_sequences += len(batch_idx)
                
                # Calculate accuracy for this batch
                with torch.no_grad():
                    # Apply sigmoid to get probabilities
                    probs = torch.sigmoid(logits)
                    # Threshold at 0.5 for binary predictions
                    predictions = (probs > 0.5).float()
                    # Calculate number of correct predictions
                    batch_correct = (predictions == batch_labels).sum().item()
                    batch_predictions = batch_labels.numel()
                    
                total_correct += batch_correct
                total_predictions += batch_predictions
                
                # Clear GPU cache periodically
                if torch.cuda.is_available() and (i // batch_size) % 10 == 0:
                    torch.cuda.empty_cache()
            
            # Calculate average loss and accuracy for this iteration
            if total_sequences > 0:
                avg_loss = total_loss / total_sequences
                avg_acc = total_correct / total_predictions if total_predictions > 0 else 0.0
            else:
                avg_loss = 0.0
                avg_acc = 0.0
                
            loss_history.append(avg_loss)
            acc_history.append(avg_acc)
            
            # Update best model state
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_model_state = copy.deepcopy(self.state_dict())
                best_labeller_state = copy.deepcopy(self.labeller.state_dict())
            
            # Print training progress
            if it % print_every == 0 or it == max_iters - 1:
                current_lr = scheduler.get_last_lr()[0]
                print(f"MLC-Train Iter {it:3d}: Loss = {avg_loss:.6e}, Acc = {avg_acc:.4f}, LR = {current_lr:.6f}")
            
            # Save checkpoint if specified
            if checkpoint_file and (it % checkpoint_interval == 0 or it == max_iters - 1):
                checkpoint = {
                    'iteration': it,
                    'model_state_dict': self.state_dict(),
                    'labeller_state_dict': self.labeller.state_dict() if self.labeller else None,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss_history': loss_history,
                    'acc_history': acc_history,
                    'best_loss': best_loss,
                    'num_labels': self.num_labels
                }
                torch.save(checkpoint, checkpoint_file)
                #print(f"Checkpoint saved at iteration {it}")
            
            # Check convergence
            if abs(prev_loss - avg_loss) < tol:
                print(f"Converged after {it+1} iterations.")
                # Restore best model state
                if best_model_state is not None:
                    self.load_state_dict(best_model_state)
                    self.labeller.load_state_dict(best_labeller_state)
                break
                
            prev_loss = avg_loss
            
            # Learning rate scheduling
            scheduler.step()
        
        self.trained = True
        
        return loss_history, acc_history

    def predict_c(self, seq):
        """
        Predict class label for a sequence using the classification head.
        
        Args:
            seq (str): Input character sequence
            
        Returns:
            tuple: (predicted_class, class_probabilities)
        """
        if self.classifier is None:
            raise ValueError("Model must be trained first for classification")
        
        # Ensure sequence has correct length
        if len(seq) != self.input_seq_len:
            raise ValueError(f"Input sequence length must be {self.input_seq_len}, got {len(seq)}")
        
        # Get sequence vector representation through forward pass
        with torch.no_grad():
            output = self.forward(seq)  # [1, SeqLen, FinalDim]
            seq_vector = output.mean(dim=1)  # [1, FinalDim]
            
            # Get logits through classification head
            logits = self.classifier(seq_vector)
            probabilities = torch.softmax(logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            
        return predicted_class, probabilities[0].cpu().numpy()

    def predict_l(self, seq, threshold=0.5):
        """
        Predict multi-label classification for a sequence.
        
        Args:
            seq: Input character sequence
            threshold: Probability threshold for binary classification (default: 0.5)
            
        Returns:
            numpy.ndarray: Binary label predictions (0 or 1 for each label)
            numpy.ndarray: Probability scores for each label
        """
        if self.labeller is None:
            raise ValueError("Model must be trained first for label prediction")
        
        # Ensure sequence has correct length
        if len(seq) != self.input_seq_len:
            raise ValueError(f"Input sequence length must be {self.input_seq_len}, got {len(seq)}")
        
        # Get sequence vector representation through forward pass
        with torch.no_grad():
            output = self.forward(seq)  # [1, SeqLen, FinalDim]
            seq_vector = output.mean(dim=1)  # [1, FinalDim]
            
            # Get logits through label head
            logits = self.labeller(seq_vector)
            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(logits).cpu().numpy()
        
        # Apply threshold to get binary predictions
        binary_preds = (probs > threshold).astype(np.float32)
        
        return binary_preds[0], probs[0]

    def self_train(self, seqs, max_iters=100, tol=1e-6, learning_rate=0.01, 
                   continued=False, self_mode='gap', decay_rate=1.0, print_every=10,
                   batch_size=256, checkpoint_file=None, checkpoint_interval=5):
        """
        Optimization: Fully vectorized self-training.
        Handles causal masking efficiently by applying it once per epoch/batch group.
        """
        if any(len(s) != self.input_seq_len for s in seqs):
            raise ValueError(f"All sequences must have length {self.input_seq_len}")
        if self_mode not in ('gap', 'reg'):
            raise ValueError("self_mode must be either 'gap' or 'reg'")
            
        start_iter = 0
        best_loss = float('inf')
        best_model_state = None
        
        if continued and checkpoint_file and os.path.exists(checkpoint_file):
            checkpoint = torch.load(checkpoint_file, map_location=self.device, weights_only=False)
            self.load_state_dict(checkpoint['model_state_dict'])
            optimizer_state = checkpoint['optimizer_state_dict']
            scheduler_state = checkpoint['scheduler_state_dict']
            history = checkpoint['history']
            start_iter = checkpoint['iteration'] + 1
            best_loss = checkpoint.get('best_loss', float('inf'))
        else:
            if not continued: self.reset_parameters()   
            history = []
        
        # Pre-process data
        all_indices = self.batch_token_indices(seqs, device='cpu')        
        num_samples = len(seqs)
        
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)
        
        if continued and checkpoint_file and os.path.exists(checkpoint_file):
            optimizer.load_state_dict(optimizer_state)
            scheduler.load_state_dict(scheduler_state)
            
        prev_loss = float('inf') if start_iter == 0 else history[-1] if history else float('inf')
        
        # Optimization: Pre-calculate Causal Masks if in 'reg' mode
        # Since structure is fixed, we can prepare masks once.
        # But we must apply them to the weights *before* forward.
        
        for it in range(start_iter, max_iters):
            total_loss = 0.0
            
            # Shuffle
            perm = torch.randperm(num_samples)
            
            # Apply Mask globally for the epoch (or per batch, but globally is faster) if reg mode
            original_linkers = []
            original_residual_linkers = []
            
            if self_mode == 'reg':
                for layer in self.hierarchical_layers:
                    if hasattr(layer, 'Linker'):
                        causal_mask = torch.tril(torch.ones_like(layer.Linker))
                        original_linkers.append(layer.Linker.data.clone())
                        layer.Linker.data.mul_(causal_mask) # In-place mul
                        
                        if hasattr(layer, 'residual_linker') and layer.residual_linker is not None:
                            cm_res = torch.tril(torch.ones_like(layer.residual_linker))
                            original_residual_linkers.append(layer.residual_linker.data.clone())
                            layer.residual_linker.data.mul_(cm_res)
                        else:
                            original_residual_linkers.append(None)

            # Iterate batches
            for i in range(0, num_samples, batch_size):
                batch_idx = perm[i:i+batch_size]
                batch_indices = all_indices[batch_idx].to(self.device)
                
                optimizer.zero_grad()
                
                # 1. Compute Char Output (Target for Gap/Reg)
                # Optimization: No need to detach char_output usually unless we freeze char layer?
                # Original code optimizers all params.
                char_output = self.char_forward(batch_indices) # [B, S_char, D]
                
                # 2. Compute Hierarchical Output
                # We can't use self.forward because we need to inject char_output or process from it
                # And we already applied masks if needed.
                hier_out = char_output
                for layer in self.hierarchical_layers:
                    hier_out = layer(hier_out) # [B, S_hier, D_hier]
                
                # 3. Vectorized Loss Calculation
                loss = 0.0
                
                # Determine valid comparison length
                # S_char might be different from S_hier if layers change sequence length
                # Original code loops: for k in range(hier_output.shape[1])
                min_len = min(char_output.shape[1], hier_out.shape[1])
                
                if self_mode == 'gap':
                    # ||Hier[k] - Char[k]||^2
                    # Slicing up to min_len
                    diff = hier_out[:, :min_len, :] - char_output[:, :min_len, :]
                    # Mean over dim and valid length, Sum over batch (to match logic)
                    # Original: sum((pred-target)**2) per seq per pos, then avg over valid pos.
                    sq_diff = torch.sum(diff ** 2, dim=-1) # [B, S_valid]
                    loss = torch.sum(torch.mean(sq_diff, dim=1)) # Sum of means
                    
                else: # 'reg' mode
                    # Predict next: Hier[k] vs Char[k+1]
                    if min_len > 1:
                        # Targets: Char 1 to min_len
                        targets = char_output[:, 1:min_len, :]
                        # Preds: Hier 0 to min_len-1
                        preds = hier_out[:, 0:min_len-1, :]
                        
                        diff = preds - targets
                        sq_diff = torch.sum(diff ** 2, dim=-1)
                        loss = torch.sum(torch.mean(sq_diff, dim=1))
                    else:
                        loss = torch.tensor(0.0, device=self.device)

                batch_loss_scalar = loss.item()
                
                # Normalize by batch size for gradient step
                if batch_loss_scalar > 0:
                    (loss / len(batch_idx)).backward()
                    optimizer.step()
                
                total_loss += batch_loss_scalar

            # Restore original weights if Reg mode
            if self_mode == 'reg':
                for idx, layer in enumerate(self.hierarchical_layers):
                    if hasattr(layer, 'Linker'):
                        layer.Linker.data.copy_(original_linkers[idx])
                        if original_residual_linkers[idx] is not None:
                            layer.residual_linker.data.copy_(original_residual_linkers[idx])

            avg_loss = total_loss / num_samples
            history.append(avg_loss)
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_model_state = copy.deepcopy(self.state_dict())
            
            if it % print_every == 0 or it == max_iters - 1:
                print(f"AutoTrain({self_mode}) Iter {it:3d}: Loss = {avg_loss:.6f}, LR = {scheduler.get_last_lr()[0]:.6f}")
            
            if checkpoint_file and (it % checkpoint_interval == 0 or it == max_iters - 1):
                self._save_checkpoint(checkpoint_file, it, history, optimizer, scheduler, best_loss)
            
            if prev_loss - avg_loss < tol:
                print(f"Converged after {it+1} iterations")
                if best_model_state is not None and avg_loss > prev_loss:
                    self.load_state_dict(best_model_state)
                    history[-1] = best_loss                    
                break
            prev_loss = avg_loss
            
            # Decay every 5 iters (as per original logic logic)
            if it % 5 == 0: 
                scheduler.step()

        if best_model_state is not None and best_loss < history[-1]:
            self.load_state_dict(best_model_state)
            print(f"Training ended. Restored best model state with loss = {best_loss:.6f}")            
            history[-1] = best_loss
        
        self._compute_training_statistics(seqs)
        self._compute_hierarchical_mean_target(seqs)
        self.trained = True
        return history

    def _save_checkpoint(self, checkpoint_file, iteration, history, optimizer, scheduler, best_loss):
        """Save training checkpoint"""
        mean_target_np = self.mean_target.cpu().numpy() if self.mean_target is not None else None
        checkpoint = {
            'iteration': iteration,
            'model_state_dict': self.state_dict(),
            'classifier_state_dict': self.classifier.state_dict() if self.classifier else None,
            'labeller_state_dict': self.labeller.state_dict() if self.labeller else None,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'history': history,
            'best_loss': best_loss,
            'config': {
                'charset': self.charset,
                'rank': self.rank,
                'vec_dim': self.vec_dim,
                'model_dims': self.model_dims,
                'basis_dims': self.basis_dims,
                'linker_dims': self.linker_dims,
                'input_seq_len': self.input_seq_len,
                'linker_trainable': self.linker_trainable,
                'use_residual_list': [layer.residual_mode for layer in self.hierarchical_layers],
                'char_layer_config': {'rank_mode': self.rank_mode, 'mode': self.mode, 'user_step': self.step}
            },
            'training_stats': {'mean_t': self.mean_t, 'mean_token_count': self.mean_token_count, 'mean_target': mean_target_np}
        }
        torch.save(checkpoint, checkpoint_file)

    def _compute_training_statistics(self, seqs, batch_size=256):
        """
        Optimization: Vectorized stats computation.
        """
        total_token_count = 0
        total_vec_sum = torch.zeros(self.vec_dim, device=self.device)
        
        all_indices = self.batch_token_indices(seqs, device='cpu')
        
        with torch.no_grad():
            for i in range(0, len(seqs), batch_size):
                batch_indices = all_indices[i:i+batch_size].to(self.device)
                
                # Get char layer output [B, S, D]
                nk_batch = self.char_forward(batch_indices)
                
                # Sum over Batch and Sequence dim
                total_vec_sum += nk_batch.sum(dim=[0, 1])
                
                # Count tokens (assuming fixed length with padding, padding should ideally be masked if we want true token count
                # but original code just did len(toks) which included padding tokens in 'pad' mode or dropped in 'drop' mode)
                # self.token_seq_len is fixed per seq.
                total_token_count += batch_indices.shape[0] * batch_indices.shape[1]
        
        self.mean_token_count = total_token_count / len(seqs) if seqs else 0
        self.mean_t = (total_vec_sum / total_token_count).cpu().numpy() if total_token_count > 0 else np.zeros(self.vec_dim)

    def _compute_hierarchical_mean_target(self, seqs, batch_size=256):
        """
        Optimization: Vectorized mean target computation.
        """
        if not seqs:
            self.mean_target = None
            return
            
        final_dim = self.model_dims[-1] if self.model_dims else self.vec_dim
        total_output = torch.zeros(final_dim, device=self.device)
        total_sequences = 0
        
        all_indices = self.batch_token_indices(seqs, device='cpu')
        
        with torch.no_grad():
            for i in range(0, len(seqs), batch_size):
                batch_indices = all_indices[i:i+batch_size].to(self.device)
                try:
                    output = self.forward(batch_indices) # [B, S_out, D_out]
                    
                    # Mean over sequence, Sum over batch
                    # Mean over seq: [B, D_out]
                    batch_means = output.mean(dim=1)
                    total_output += batch_means.sum(dim=0)
                    total_sequences += batch_indices.shape[0]
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"Warning: OOM in mean target compute. Batch {i}")
                        torch.cuda.empty_cache()
                        continue
                    else:
                        raise e
        
        if total_sequences > 0:
            self.mean_target = total_output / total_sequences
        else:
            self.mean_target = None   

    def reconstruct(self, L, tau=0.0):
        """Reconstruct representative character sequence of length L using the entire hierarchical model"""
        assert self.trained, "Model must be trained first"
        assert self.mean_target is not None, "Mean target vector must be computed first"
        
        if L <= 0: return ""
        if L > self.input_seq_len: L = self.input_seq_len
        
        generated_sequence = ""
        
        while len(generated_sequence) < L:
            # Batch candidates
            candidates = []
            valid_tokens = []
            
            for token in self.tokens:
                cand = generated_sequence + token
                if len(cand) > L: continue
                
                # Pad for model input
                if len(cand) < self.input_seq_len:
                    padded = cand.ljust(self.input_seq_len, 'A')
                else:
                    padded = cand[:self.input_seq_len]
                
                candidates.append(padded)
                valid_tokens.append(token)
            
            if not candidates:
                generated_sequence += self.char_generate(L - len(generated_sequence), tau)
                break
                
            with torch.no_grad():
                outputs = self.forward(candidates)
                means = outputs.mean(dim=1)
                
                # Calculate scores (negative error)
                diff = means - self.mean_target.unsqueeze(0)
                errors = torch.sum(diff ** 2, dim=1)
                scores = -errors
                
                scores_np = scores.cpu().numpy()
                
                if tau == 0:
                    best_idx = np.argmax(scores_np)
                    chosen_token = valid_tokens[best_idx]
                else:
                    scaled_scores = scores_np / tau
                    exp_scores = np.exp(scaled_scores - np.max(scaled_scores))
                    probs = exp_scores / np.sum(exp_scores)
                    chosen_idx = np.random.choice(len(valid_tokens), p=probs)
                    chosen_token = valid_tokens[chosen_idx]
            
            generated_sequence += chosen_token
        
        final_sequence = generated_sequence[:L]
        if '_' in final_sequence: final_sequence = final_sequence.replace('_', '')
        return final_sequence
    
    def reset_parameters(self):
        """Reset all model parameters"""
        nn.init.uniform_(self.char_embedding.weight, -0.5, 0.5)
        nn.init.uniform_(self.char_linear.weight, -0.1, 0.1)
        nn.init.uniform_(self.char_Acoeff, -0.1, 0.1)
        nn.init.uniform_(self.char_Bbasis, -0.1, 0.1)
        
        for layer in self.hierarchical_layers:
            nn.init.uniform_(layer.M, -0.1, 0.1)
            nn.init.uniform_(layer.Acoeff, -0.1, 0.1)
            nn.init.uniform_(layer.Bbasis, -0.1, 0.1)
            nn.init.uniform_(layer.Linker, -0.1, 0.1)
            if hasattr(layer, 'residual_proj') and isinstance(layer.residual_proj, nn.Linear):
                nn.init.uniform_(layer.residual_proj.weight, -0.1, 0.1)
            if hasattr(layer, 'residual_linker') and layer.residual_linker is not None:
                nn.init.uniform_(layer.residual_linker, -0.1, 0.1)
        
        # Reset classifier and labeller if they exist
        if self.classifier is not None:
            nn.init.normal_(self.classifier.weight, 0, 0.01)
            if self.classifier.bias is not None:
                nn.init.constant_(self.classifier.bias, 0)
        
        if self.labeller is not None:
            nn.init.xavier_uniform_(self.labeller.weight)
            nn.init.zeros_(self.labeller.bias)
        
        self.mean_target = None
        self.trained = False
        self.mean_token_count = None
        self.mean_t = None
    
    def count_parameters(self):
        """Count and print learnable parameters"""
        total_params = 0
        print("="*70)
        print(f"{'Component':<15} | {'Layer/Param':<25} | {'Count':<15} | {'Shape':<20}")
        print("-"*70)
        
        char_params = 0
        for name, param in [('char_embedding', self.char_embedding.weight), 
                           ('char_linear', self.char_linear.weight),
                           ('char_Acoeff', self.char_Acoeff),
                           ('char_Bbasis', self.char_Bbasis)]:
            num = param.numel()
            print(f"{'Char Layer':<15} | {name:<25} | {num:<15} | {str(tuple(param.shape)):<20}")
            char_params += num
            total_params += num
            
        print(f"{'Char Layer':<15} | {'TOTAL':<25} | {char_params:<15} |")
        print("-"*70)
        
        for i, layer in enumerate(self.hierarchical_layers):
            layer_params = 0
            layer_name = f"Hier Layer {i}"
            for name, param in layer.named_parameters():
                if param.requires_grad:
                    num = param.numel()
                    print(f"{layer_name:<15} | {name:<25} | {num:<15} | {str(tuple(param.shape)):<20}")
                    layer_params += num
                    total_params += num
            print(f"{layer_name:<15} | {'TOTAL':<25} | {layer_params:<15} |")
            print("-"*70)
        
        # Count classifier and labeller parameters if they exist
        if self.classifier is not None:
            classifier_params = sum(p.numel() for p in self.classifier.parameters())
            print(f"{'Classifier':<15} | {'TOTAL':<25} | {classifier_params:<15} |")
            total_params += classifier_params
            print("-"*70)
        
        if self.labeller is not None:
            labeller_params = sum(p.numel() for p in self.labeller.parameters())
            print(f"{'Labeller':<15} | {'TOTAL':<25} | {labeller_params:<15} |")
            total_params += labeller_params
            print("-"*70)
        
        print(f"{'TOTAL':<15} | {'ALL PARAMETERS':<25} | {total_params:<15} |")
        print("="*70)
        return total_params
    
    def save(self, filename):
        """Save model state to file"""
        # Create a state dictionary that excludes classifier and labeller parameters
        # since they will be saved separately
        model_state_dict = {}
        for name, param in self.state_dict().items():
            # Skip classifier and labeller parameters
            if not (name.startswith('classifier.') or name.startswith('labeller.')):
                model_state_dict[name] = param
        
        mean_target_np = self.mean_target.cpu().numpy() if self.mean_target is not None else None
        save_dict = {
            'model_state_dict': model_state_dict,
            'classifier_state_dict': self.classifier.state_dict() if self.classifier else None,
            'labeller_state_dict': self.labeller.state_dict() if self.labeller else None,
            'config': {
                'charset': self.charset,
                'rank': self.rank,
                'vec_dim': self.vec_dim,
                'model_dims': self.model_dims,
                'basis_dims': self.basis_dims,
                'linker_dims': self.linker_dims,
                'input_seq_len': self.input_seq_len,
                'linker_trainable': self.linker_trainable,
                'use_residual_list': [layer.residual_mode for layer in self.hierarchical_layers],
                'char_layer_config': {'rank_mode': self.rank_mode, 'mode': self.mode, 'user_step': self.step}
            },
            'training_stats': {
                'mean_t': self.mean_t, 
                'mean_token_count': self.mean_token_count, 
                'mean_target': mean_target_np
            },
            'num_classes': self.num_classes,
            'num_labels': self.num_labels,
            'trained': self.trained
        }
        torch.save(save_dict, filename)
        print(f"Model saved to {filename}")
    
    @classmethod
    def load(cls, filename, device=None):
        """Load model from file"""
        device = device or torch.device("cuda" if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(filename, map_location=device, weights_only=False)
        config = checkpoint['config']
        char_layer_config = config['char_layer_config']
        
        # Create model instance
        model = cls(
            charset=config['charset'],
            rank=config['rank'],
            vec_dim=config['vec_dim'],
            model_dims=config['model_dims'],
            basis_dims=config['basis_dims'],
            linker_dims=config['linker_dims'],
            input_seq_len=config['input_seq_len'],
            rank_mode=char_layer_config['rank_mode'],
            mode=char_layer_config['mode'],
            user_step=char_layer_config['user_step'],            
            linker_trainable=config.get('linker_trainable', False),
            use_residual_list=config.get('use_residual_list', [None] * len(config['model_dims'])),
            device=device
        )
        
        # Load main model parameters (excluding classifier/labeller)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        
        # Load classifier if it exists in checkpoint
        if 'num_classes' in checkpoint and checkpoint['num_classes'] is not None:
            model.num_classes = checkpoint['num_classes']
            model.classifier = nn.Linear(model.model_dims[-1], model.num_classes).to(device)
            if 'classifier_state_dict' in checkpoint and checkpoint['classifier_state_dict'] is not None:
                model.classifier.load_state_dict(checkpoint['classifier_state_dict'])
        
        # Load labeller if it exists in checkpoint
        if 'num_labels' in checkpoint and checkpoint['num_labels'] is not None:
            model.num_labels = checkpoint['num_labels']
            model.labeller = nn.Linear(model.model_dims[-1], model.num_labels).to(device)
            if 'labeller_state_dict' in checkpoint and checkpoint['labeller_state_dict'] is not None:
                model.labeller.load_state_dict(checkpoint['labeller_state_dict'])
        
        # Load training statistics
        if 'training_stats' in checkpoint:
            stats = checkpoint['training_stats']
            if stats['mean_t'] is not None: 
                model.mean_t = stats['mean_t']
            if stats['mean_token_count'] is not None: 
                model.mean_token_count = stats['mean_token_count']
            if stats['mean_target'] is not None: 
                model.mean_target = torch.tensor(stats['mean_target'], device=device)
            model.trained = checkpoint.get('trained', True)
        
        print(f"Model loaded from {filename}")
        return model

# === Example Usage ===
if __name__ == "__main__":
    from statistics import correlation
    
    print("="*60)
    print("HierDDLrnC - Hierarchical Dual Descriptor with Linker and Character Input")
    print("Linker-based Model with Character Sequence Processing Demonstration")
    print("="*60)
    
    # Set random seeds for reproducibility
    torch.manual_seed(2)
    random.seed(2)
    np.random.seed(2)
    
    # Configuration
    charset = ['A', 'C', 'G', 'T']
    rank = 1
    vec_dim = 8
    input_seq_len = 300  # Fixed input sequence length
    model_dims = [16, 12, 8]  # Hierarchical layer dimensions
    basis_dims = [100, 80, 60]  # Basis dimensions for hierarchical layers
    linker_dims = [80, 60, 40]  # Output sequence lengths for hierarchical layers
    num_seqs = 100
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {device}")
    print(f"Character set: {charset}")
    print(f"Rank (k-mer length): {rank}")
    print(f"Character layer output dim: {vec_dim}")
    print(f"Input sequence length: {input_seq_len}")
    print(f"Hierarchical layer dims: {model_dims}")
    print(f"Hierarchical basis dims: {basis_dims}")
    print(f"Hierarchical linker dims: {linker_dims}")
    print(f"Using Linker-based architecture with character input")
    
    # Generate synthetic training data with fixed length
    print("\nGenerating synthetic training data...")
    seqs, t_list = [], []
    for _ in range(num_seqs):
        seq = ''.join(random.choices(charset, k=input_seq_len))
        seqs.append(seq)
        # Create target vector (final layer dimension)
        t_list.append([random.uniform(-1.0, 1.0) for _ in range(model_dims[-1])])
    
    # Create model
    print("\nCreating HierDDLrnC model...")
    model = HierDDLrnC(
        charset=charset,
        rank=rank,
        vec_dim=vec_dim,
        model_dims=model_dims,
        basis_dims=basis_dims,
        linker_dims=linker_dims,
        input_seq_len=input_seq_len,
        mode='nonlinear',
        user_step=1,
        linker_trainable=[True, False, True],
        use_residual_list=['separate', 'shared', None],
        device=device
    )
    
    # Count parameters
    print("\nModel Parameter Count:")
    total_params = model.count_parameters()
    
    # Gradient descent training
    print("\n" + "="*50)
    print("Gradient Descent Training")
    print("="*50)
    
    reg_history = model.reg_train(
        seqs, t_list,
        max_iters=100,
        learning_rate=0.1,
        decay_rate=0.98,
        print_every=5,
        batch_size=32,
        checkpoint_file='hierddlrnc_gd_checkpoint.pth',
        checkpoint_interval=10
    )
    
    # Predictions and correlation analysis
    print("\n" + "="*50)
    print("Prediction and Correlation Analysis")
    print("="*50)
    
    pred_t_list = [model.predict_t(seq) for seq in seqs]
    
    # Calculate correlations for each dimension
    output_dim = model_dims[-1]
    correlations = []
    for i in range(output_dim):
        actual = [t_vec[i] for t_vec in t_list]
        predicted = [t_vec[i] for t_vec in pred_t_list]
        corr = correlation(actual, predicted)
        correlations.append(corr)
        print(f"Dimension {i} correlation: {corr:.4f}")
    
    avg_correlation = sum(correlations) / len(correlations)
    print(f"Average correlation: {avg_correlation:.4f}")
    
    # Sequence reconstruction
    print("\n" + "="*50)
    print("Sequence Reconstruction")
    print("="*50)    
    
    # Deterministic reconstruction
    det_seq = model.reconstruct(L=100, tau=0.0)
    print(f"Deterministic reconstruction (tau=0.0): {det_seq}")
    
    # Stochastic reconstruction
    sto_seq = model.reconstruct(L=100, tau=0.5)
    print(f"Stochastic reconstruction (tau=0.5): {sto_seq}")
    
    # Classification Task Example
    print("\n" + "="*50)
    print("Classification Task Example")
    print("="*50)
    
    # Generate classification data
    num_classes = 3
    class_seqs = []
    class_labels = []
    
    # Create sequences with different patterns for each class
    for class_id in range(num_classes):
        for _ in range(50):  # 50 sequences per class
            if class_id == 0:
                # Class 0: High A content
                seq = ''.join(random.choices(['A', 'C', 'G', 'T'], weights=[0.6, 0.1, 0.1, 0.2], k=input_seq_len))
            elif class_id == 1:
                # Class 1: High GC content
                seq = ''.join(random.choices(['A', 'C', 'G', 'T'], weights=[0.1, 0.4, 0.4, 0.1], k=input_seq_len))
            else:
                # Class 2: Balanced
                seq = ''.join(random.choices(['A', 'C', 'G', 'T'], k=input_seq_len))
            
            class_seqs.append(seq)
            class_labels.append(class_id)
    
    # Create new model for classification
    cls_model = HierDDLrnC(
        charset=charset,
        rank=rank,
        vec_dim=vec_dim,
        model_dims=model_dims,
        basis_dims=basis_dims,
        linker_dims=linker_dims,
        input_seq_len=input_seq_len,
        mode='nonlinear',
        user_step=1,
        linker_trainable=True,
        use_residual_list=['separate', 'shared', None],
        device=device
    )
    
    # Train for classification
    print("\nTraining classification model...")
    cls_history = cls_model.cls_train(
        class_seqs, class_labels, num_classes,
        max_iters=50,
        learning_rate=0.05,
        decay_rate=0.99,
        print_every=5,
        batch_size=16,
        checkpoint_file='hierddlrnc_cls_checkpoint.pth',
        checkpoint_interval=10
    )
    
    # Test classification predictions
    print("\nClassification Prediction Results:")
    correct = 0
    for i, (seq, true_label) in enumerate(zip(class_seqs[:10], class_labels[:10])):
        pred_class, probs = cls_model.predict_c(seq)
        status = "✓" if pred_class == true_label else "✗"
        print(f"Seq {i+1}: True={true_label}, Pred={pred_class}, Probs={[f'{p:.3f}' for p in probs]} {status}")
        if pred_class == true_label:
            correct += 1
    
    print(f"\nAccuracy on first 10 sequences: {correct}/10 = {correct/10:.2f}")
    
    # Multi-Label Classification Task Example
    print("\n" + "="*50)
    print("Multi-Label Classification Task Example")
    print("="*50)
    
    # Generate multi-label data
    num_labels = 4
    label_seqs = []
    labels = []
    
    for _ in range(100):
        seq = ''.join(random.choices(charset, k=input_seq_len))
        label_seqs.append(seq)
        # Create random binary labels (multi-label classification)
        # Each sequence can have 0-4 active labels
        label_vec = [random.random() > 0.7 for _ in range(num_labels)]
        labels.append([1.0 if x else 0.0 for x in label_vec])
    
    # Create new model for multi-label classification
    lbl_model = HierDDLrnC(
        charset=charset,
        rank=rank,
        vec_dim=vec_dim,
        model_dims=model_dims,
        basis_dims=basis_dims,
        linker_dims=linker_dims,
        input_seq_len=input_seq_len,
        mode='nonlinear',
        user_step=1,
        linker_trainable=True,
        use_residual_list=['separate', 'shared', None],
        device=device
    )
    
    # Train for multi-label classification
    print("\nTraining multi-label classification model...")
    loss_history, acc_history = lbl_model.lbl_train(
        label_seqs, labels, num_labels,
        max_iters=50,
        learning_rate=0.05,
        decay_rate=0.99,
        print_every=5,
        batch_size=16,
        checkpoint_file='hierddlrnc_lbl_checkpoint.pth',
        checkpoint_interval=10
    )
    
    print(f"\nFinal training loss: {loss_history[-1]:.6f}")
    print(f"Final training accuracy: {acc_history[-1]:.4f}")
    
    # Test multi-label predictions
    print("\nMulti-Label Prediction Results (first 5 sequences):")
    label_names = ["Function_A", "Function_B", "Function_C", "Function_D"]
    
    for i in range(min(5, len(label_seqs))):
        binary_pred, probs = lbl_model.predict_l(label_seqs[i], threshold=0.5)
        true_labels = labels[i]
        
        print(f"\nSequence {i+1}:")
        print(f"True labels:    {true_labels}")
        print(f"Binary preds:   {binary_pred}")
        print(f"Probabilities:  {[f'{p:.4f}' for p in probs]}")
        
        # Check which predictions are correct
        correct = all(abs(binary_pred[j] - true_labels[j]) < 0.5 for j in range(num_labels))
        print(f"All correct: {'✓' if correct else '✗'}")
        
        # Show interpretation
        print("Label interpretation:")
        for j in range(num_labels):
            status = "ACTIVE" if binary_pred[j] > 0.5 else "INACTIVE"
            truth = "ACTIVE" if true_labels[j] > 0.5 else "INACTIVE"
            match = "✓" if (binary_pred[j] > 0.5) == (true_labels[j] > 0.5) else "✗"
            print(f"  {label_names[j]}: Predicted {status}, True {truth} {match}")
    
    # Self-training example
    print("\n" + "="*50)
    print("Auto-Training Example")
    print("="*50)
    
    # Create a new model for self-training
    self_model = HierDDLrnC(
        charset=charset,
        rank=rank,
        vec_dim=vec_dim,
        model_dims=model_dims,
        basis_dims=basis_dims,
        linker_dims=linker_dims,
        input_seq_len=input_seq_len,
        mode='nonlinear',
        user_step=1,
        linker_trainable=True,
        use_residual_list='separate',
        device=device
    )
    
    # Use sequences for self-training
    auto_seqs = seqs[:50]  # Use first 50 sequences
    
    # Self-train in gap mode
    print("\nSelf-training in 'gap' mode...")
    auto_history_gap = self_model.self_train(
        auto_seqs,
        max_iters=30,
        learning_rate=0.01,
        self_mode='gap',
        print_every=5,
        batch_size=256,
        checkpoint_file='hierddlrnc_auto_checkpoint.pth',
        checkpoint_interval=5
    )
    
    # Reconstruct from self-trained model
    auto_rec_seq = self_model.reconstruct(L=80, tau=0.2)
    print(f"Reconstructed from self-trained model: {auto_rec_seq}")
    
    # Model save and load test
    print("\n" + "="*50)
    print("Model Save/Load Test")
    print("="*50)
    
    # Save classification model
    cls_model.save("hierddlrnc_cls_model.pth")
    
    # Load classification model
    loaded_cls_model = HierDDLrnC.load("hierddlrnc_cls_model.pth", device=device)
    
    # Test loaded classification model
    test_seq = class_seqs[0]
    original_pred, original_probs = cls_model.predict_c(test_seq)
    loaded_pred, loaded_probs = loaded_cls_model.predict_c(test_seq)
    
    print(f"Original model prediction: Class {original_pred}, Probs {[f'{p:.4f}' for p in original_probs]}")
    print(f"Loaded model prediction:   Class {loaded_pred}, Probs {[f'{p:.4f}' for p in loaded_probs]}")
    
    pred_diff = np.max(np.abs(original_probs - loaded_probs))
    print(f"Maximum probability difference: {pred_diff:.6e}")
    
    if pred_diff < 1e-6 and original_pred == loaded_pred:
        print("✓ Classification model save/load test PASSED")
    else:
        print("✗ Classification model save/load test FAILED")    
    
    print("\nAll tests completed successfully!")
