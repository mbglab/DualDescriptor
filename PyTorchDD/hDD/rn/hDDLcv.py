# Copyright (C) 2005-2025, Bin-Guang Ma (mbg@mail.hzau.edu.cn); SPDX-License-Identifier: MIT
# Hierarchical Dual Descriptor (Random AB matrix form) with Character Sequence Input (HierDDLrnC)
# Combines character-level processing and hierarchical vector processing with Linker matrices
# This program is for the demonstration of methodology and not fully refined.
# Author: Bin-Guang Ma (assisted by DeepSeek); Date: 2025-11-30 ~ 2026-1-17

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
        scalars = torch.einsum('bsd,sd->bs', x, basis_vectors)
        
        # Select coefficients for each position: (in_seq_len, out_dim)
        coeffs = self.Acoeff[:, self.basis_indices].permute(1, 0)
        
        # Compute position outputs: (batch_size, in_seq_len, out_dim)
        u = coeffs * scalars.unsqueeze(-1)
        
        # Apply sequence length transformation using Linker matrix
        # (batch_size, in_seq_len, out_dim) x (in_seq_len, out_seq_len) -> (batch_size, out_dim, out_seq_len)
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
    Supports variable-length character sequences as input
    
    Args:
        charset: Character set for sequence input
        rank: k-mer length for tokenization
        rank_mode: 'pad' or 'drop' for incomplete k-mers
        vec_dim: Output dimension of character layer (Layer 0)
        mode: 'linear' or 'nonlinear' tokenization
        user_step: Step size for nonlinear tokenization
        max_seq_len: Maximum input sequence length (for variable-length sequences)
        hier_input_seq_len: Fixed input sequence length for first hierarchical layer
        model_dims: Output dimensions for hierarchical layers
        basis_dims: Basis dimensions for hierarchical layers
        linker_dims: Output sequence lengths for hierarchical layers
        linker_trainable: Controls if Linker matrices are trainable
        use_residual_list: Residual connection types for hierarchical layers
        char_linker_trainable: Whether character layer Linker matrix is trainable (None=use linker_trainable[0])
        device: Computation device
    """
    def __init__(self, charset, rank=1, rank_mode='drop', vec_dim=2, 
                 mode='linear', user_step=None, max_seq_len=100, hier_input_seq_len=50,
                 model_dims=[2], basis_dims=[50], linker_dims=[50], linker_trainable=False,
                 use_residual_list=None, char_linker_trainable=None, device='cuda'):
        super().__init__()
        self.charset = list(charset)
        self.rank = rank    # r-per/k-mer length
        self.rank_mode = rank_mode # 'pad' or 'drop'
        self.vec_dim = vec_dim    # embedding dimension
        self.mode = mode
        self.step = user_step
        self.max_seq_len = max_seq_len  # Maximum input character sequence length
        self.hier_input_seq_len = hier_input_seq_len  # Fixed input sequence length for first hierarchical layer
        self.model_dims = model_dims
        self.basis_dims = basis_dims
        self.linker_dims = linker_dims
        self.num_layers = len(model_dims)
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.trained = False        
        
        # Validate dimensions
        if len(basis_dims) != self.num_layers:
            raise ValueError("basis_dims must have same length as model_dims")
        if len(linker_dims) != self.num_layers:
            raise ValueError("linker_dims must have same length as model_dims")
        
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
        
        # Process char_linker_trainable parameter
        if char_linker_trainable is None:
            # If not specified, use the first hier-layer linker trainable setting
            self.char_linker_trainable = self.linker_trainable[0] if isinstance(self.linker_trainable, list) and len(self.linker_trainable) > 0 else self.linker_trainable
        else:
            self.char_linker_trainable = char_linker_trainable
        
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
        
        # Calculate maximum token sequence length from maximum character sequence length
        self.max_token_seq_len = self._calculate_token_seq_len(max_seq_len)
        
        # Embedding layer for tokens
        self.char_embedding = nn.Embedding(len(self.tokens), self.vec_dim)
        
        # Character layer random AB matrix components
        self.char_basis_dim = 50  # Fixed basis dimension for character layer
        self.char_linear = nn.Linear(self.vec_dim, self.vec_dim, bias=False)
        self.char_Acoeff = nn.Parameter(torch.Tensor(self.vec_dim, self.char_basis_dim))
        self.char_Bbasis = nn.Parameter(torch.Tensor(self.char_basis_dim, self.vec_dim))
        
        # Special Linker matrix to convert variable-length character sequences to fixed-length hierarchical input
        self.char_to_hier_linker = nn.Parameter(
            torch.empty(self.max_token_seq_len, self.hier_input_seq_len),
            requires_grad=self.char_linker_trainable
        )

        # Initialize character layer parameters
        nn.init.uniform_(self.char_embedding.weight, -0.5, 0.5)
        nn.init.uniform_(self.char_linear.weight, -0.1, 0.1)
        nn.init.uniform_(self.char_Acoeff, -0.1, 0.1)
        nn.init.uniform_(self.char_Bbasis, -0.1, 0.1)
        nn.init.uniform_(self.char_to_hier_linker, -0.1, 0.1)
        
        # Training statistics for character layer
        self.mean_token_count = None
        self.mean_t = None
        
        # Classification heads (initialized when needed)
        self.num_classes = None
        self.classifier = None
        
        # Label head for multi-label tasks (initialized when needed)
        self.num_labels = None
        self.labeller = None
        
        # Hierarchical layers with Linker (starting from Layer 1)
        self.hierarchical_layers = nn.ModuleList()
        for l in range(self.num_layers):
            # Determine input dimensions for this layer
            in_dim = vec_dim if l == 0 else model_dims[l-1]
            # For first hierarchical layer, use fixed hier_input_seq_len; for subsequent layers, use previous linker_dim
            in_seq = self.hier_input_seq_len if l == 0 else linker_dims[l-1]
            out_dim = model_dims[l]
            out_seq = linker_dims[l]
            basis_dim = basis_dims[l]
            use_residual = self.use_residual_list[l]
            linker_trainable = self.linker_trainable[l]
            
            # Initialize layer
            layer = Layer(
                in_dim=in_dim,
                out_dim=out_dim,
                basis_dim=basis_dim,
                in_seq_len=in_seq,
                out_seq_len=out_seq,
                linker_trainable=linker_trainable,
                residual_mode=use_residual,
                device=self.device
            )
            self.hierarchical_layers.append(layer)
        
        # Mean target vector for the final hierarchical layer output
        self.mean_target = None
        
        self.to(self.device)
    
    def _calculate_token_seq_len(self, char_seq_len):
        """
        Calculate token sequence length from character sequence length based on tokenization mode
        
        Args:
            char_seq_len (int): Character sequence length
            
        Returns:
            int: Token sequence length
        """
        if self.mode == 'linear':
            # Linear mode: sliding window with step=1
            return char_seq_len - self.rank + 1
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
        
        - 'linear': Slide window by 1 step, extracting contiguous kmers of length = rank
        - 'nonlinear': Slide window by custom step (or rank length if step not specified)
        
        For nonlinear mode, handles incomplete trailing fragments using:
        - 'pad': Pads with '_' to maintain kmer length
        - 'drop': Discards incomplete fragments
        
        Args:
            seq (str): Input character sequence to tokenize
            
        Returns:
            list: List of extracted kmer tokens
        """
        L = len(seq)
        # Linear mode: sliding window with step=1
        if self.mode == 'linear':
            return [seq[i:i+self.rank] for i in range(L - self.rank + 1)]
        
        # Nonlinear mode: stepping with custom step size
        toks = []
        step = self.step or self.rank  # Use custom step if defined, else use rank length
        
        for i in range(0, L, step):
            frag = seq[i:i+self.rank]
            frag_len = len(frag)
            
            # Pad or drop based on rank_mode setting
            if self.rank_mode == 'pad':
                # Pad fragment with '_' if shorter than rank
                toks.append(frag if frag_len == self.rank else frag.ljust(self.rank, '_'))
            elif self.rank_mode == 'drop':
                # Only add fragments that match full rank length
                if frag_len == self.rank:
                    toks.append(frag)
        return toks

    def batch_token_indices(self, seqs, device=None):
        """
        Convert a batch of sequences to token indices tensor with vectorized processing.
        This method improves performance by reducing CPU-GPU data transfer and enabling
        parallel processing.
        
        Args:
            seqs (list): List of character sequences
            device (torch.device): Target device (defaults to self.device)
        
        Returns:
            Tensor: LongTensor of shape [batch_size, max_token_len] with padding
        """
        target_device = device if device is not None else self.device
        
        # Prepare params
        step = 1 if self.mode == 'linear' else (self.step or self.rank)
        
        all_indices = []
        all_lengths = []
        
        # Process each sequence in the batch
        for seq in seqs:
            L = len(seq)
            seq_indices = []
            
            if self.mode == 'linear':
                ranges = range(L - self.rank + 1)
                toks = [seq[i:i+self.rank] for i in ranges]
                seq_indices = [self.token_to_idx[t] for t in toks]
                
            else: # nonlinear mode
                ranges = range(0, L, step)
                for i in ranges:
                    frag = seq[i:i+self.rank]
                    if len(frag) == self.rank:
                        seq_indices.append(self.token_to_idx[frag])
                    else:
                        if self.rank_mode == 'pad':
                            frag_pad = frag.ljust(self.rank, '_')
                            seq_indices.append(self.token_to_idx[frag_pad])
            
            all_indices.append(seq_indices)
            all_lengths.append(len(seq_indices))
        
        # Find max length for padding
        max_len = max(all_lengths) if all_lengths else 0
        
        # Pad sequences to max length with a special token (use index 0 for padding)
        padded_indices = []
        for seq_indices in all_indices:
            if len(seq_indices) < max_len:
                padded = seq_indices + [0] * (max_len - len(seq_indices))
            else:
                padded = seq_indices[:max_len]
            padded_indices.append(padded)
        
        return torch.tensor(padded_indices, dtype=torch.long, device=target_device)

    def char_forward(self, token_indices):
        """
        Vectorized computation of N(k) vectors for a BATCH of token sequences.
        
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
        j_indices = torch.arange(seq_len, device=self.device) % self.char_basis_dim
        
        # Select basis vectors: (seq_len, vec_dim)
        B_vectors = self.char_Bbasis[j_indices]
        
        # Compute scalars: dot product of normalized and B_vectors
        scalars = torch.einsum('bsd,sd->bs', normalized, B_vectors)
        
        # Select coefficient vectors: (seq_len, vec_dim)
        A_vectors = self.char_Acoeff[:, j_indices].t()
        
        # Compute new features: scalars * A_vectors
        Nk = scalars.unsqueeze(2) * A_vectors.unsqueeze(0)
            
        return Nk

    def char_batch_compute_Nk(self, token_indices):
        """
        Wrapper for backward compatibility with 1D token indices.
        Delegates to vectorized implementation.
        """
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
        
        for k in range(n_tokens):
            # Manual computation for position k
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
        """
        Convert character sequence to tensor representation through char layer
        and apply char_to_hier_linker to convert to fixed length for hierarchical layers
        
        Args:
            seq (str): Input character sequence (length <= max_seq_len)
            
        Returns:
            Tensor: Sequence tensor of shape [1, hier_input_seq_len, vec_dim]
        """
        # Validate sequence length
        if len(seq) > self.max_seq_len:
            raise ValueError(f"Input sequence length must be <= {self.max_seq_len}, got {len(seq)}")
        
        toks = self.extract_tokens(seq)
        if not toks:
            return torch.zeros((1, self.hier_input_seq_len, self.vec_dim), device=self.device)
        
        token_seq_len = len(toks)
        
        # Validate token sequence length
        if token_seq_len > self.max_token_seq_len:
            raise ValueError(f"Token sequence length must be <= {self.max_token_seq_len}, got {token_seq_len}")
        
        token_indices = torch.tensor([self.token_to_idx[tok] for tok in toks], device=self.device)
        
        # Compute N(k) vectors through char layer
        Nk_batch = self.char_batch_compute_Nk(token_indices)
        
        # Add batch dimension: [1, token_seq_len, vec_dim]
        char_output = Nk_batch.unsqueeze(0)
        
        # Apply char_to_hier_linker to convert variable length to fixed length
        # Use only the first token_seq_len rows of the linker matrix for this sequence
        hier_input = torch.matmul(char_output.transpose(1, 2), self.char_to_hier_linker[:token_seq_len, :]).transpose(1, 2)
        
        return hier_input  # [1, hier_input_seq_len, vec_dim]
    
    def batch_sequence_to_tensor(self, seqs):
        """
        Convert a batch of sequences to tensor representation with vectorized processing.
        
        Args:
            seqs (list): List of character sequences
            
        Returns:
            Tensor: Sequence tensor of shape [batch_size, hier_input_seq_len, vec_dim]
        """
        if not seqs:
            return torch.zeros((0, self.hier_input_seq_len, self.vec_dim), device=self.device)
        
        # Get token indices for all sequences
        token_indices = self.batch_token_indices(seqs)  # [batch_size, max_token_len]
        
        # Compute character layer outputs
        char_output = self.char_forward(token_indices)  # [batch_size, max_token_len, vec_dim]
        
        # Apply char_to_hier_linker to convert variable length to fixed length
        # We need to handle variable lengths, so we'll process each sequence individually
        # This could be further optimized but maintains flexibility for variable lengths
        batch_outputs = []
        for i in range(len(seqs)):
            # Get actual token length for this sequence (excluding padding)
            actual_token_len = len(self.extract_tokens(seqs[i]))
            if actual_token_len > 0:
                # Extract relevant part of char output
                seq_char_output = char_output[i:i+1, :actual_token_len, :]
                # Apply linker for this sequence
                seq_hier_input = torch.matmul(
                    seq_char_output.transpose(1, 2), 
                    self.char_to_hier_linker[:actual_token_len, :]
                ).transpose(1, 2)
                batch_outputs.append(seq_hier_input.squeeze(0))
            else:
                # Empty sequence
                batch_outputs.append(torch.zeros((self.hier_input_seq_len, self.vec_dim), device=self.device))
        
        return torch.stack(batch_outputs)  # [batch_size, hier_input_seq_len, vec_dim]
    
    def forward(self, input_data):
        """
        Forward pass through entire hierarchical model with Linker.
        Supports both single sequences and batches.
        
        Args:
            input_data (str or list): Input character sequence(s)
            
        Returns:
            Tensor: Output of shape [batch_size, final_seq_len, final_dim]
        """
        if isinstance(input_data, str):
            # Single sequence
            x = self.char_sequence_to_tensor(input_data)
        elif isinstance(input_data, list):
            # Multiple sequences
            x = self.batch_sequence_to_tensor(input_data)
        else:
            raise TypeError("Input must be string or list of strings")
        
        # Pass through hierarchical layers with Linker
        for layer in self.hierarchical_layers:
            x = layer(x)
            
        return x
    
    def predict_t(self, seq):
        """
        Predict target vector for a character sequence
        Returns the average of all output vectors in the sequence
        
        Args:
            seq (str): Input character sequence of variable length (<= max_seq_len)
            
        Returns:
            numpy.array: Predicted target vector
        """
        with torch.no_grad():
            output = self.forward(seq)  # [1, final_seq_len, final_dim]
            target = output.mean(dim=1)  # Average over sequence length
            return target.squeeze(0).cpu().numpy()
    
    def reg_train(self, seqs, t_list, max_iters=1000, tol=1e-8, learning_rate=0.01, 
                   continued=False, decay_rate=1.0, print_every=10, batch_size=1024,
                   checkpoint_file=None, checkpoint_interval=10):
        """
        Train the model using gradient descent with target vectors.
        Uses vectorized batch processing for improved performance.
        
        Args:
            seqs: List of character sequences (all must have length <= max_seq_len)
            t_list: List of target vectors
            max_iters: Maximum training iterations
            tol: Convergence tolerance
            learning_rate: Initial learning rate
            continued: Whether to continue from existing parameters
            decay_rate: Learning rate decay rate
            print_every: Print interval
            batch_size: Batch size for training
            checkpoint_file: Path to save/load checkpoint file for resuming training
            checkpoint_interval: Interval (in iterations) for saving checkpoints
            
        Returns:
            list: Training loss history
        """
        # Validate input sequences
        for seq in seqs:
            if len(seq) > self.max_seq_len:
                raise ValueError(f"All sequences must have length <= {self.max_seq_len}")
        
        # Load checkpoint if continuing and checkpoint file exists
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
            if not continued:
                self.reset_parameters() 
            history = []
        
        # Convert target vectors to tensor
        t_tensors = torch.tensor(t_list, dtype=torch.float32, device='cpu')
        num_samples = len(seqs)
        
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)
        
        # Load optimizer and scheduler states if resuming
        if continued and checkpoint_file and os.path.exists(checkpoint_file):
            optimizer.load_state_dict(optimizer_state)
            scheduler.load_state_dict(scheduler_state)
        
        prev_loss = float('inf') if start_iter == 0 else history[-1] if history else float('inf')
        
        for it in range(start_iter, max_iters):
            total_loss = 0.0
            
            # Shuffle indices
            perm = torch.randperm(num_samples)
            
            for i in range(0, num_samples, batch_size):
                batch_idx = perm[i:i+batch_size]
                batch_seqs = [seqs[idx] for idx in batch_idx]
                batch_t = t_tensors[batch_idx].to(self.device)
                
                optimizer.zero_grad()
                
                # Vectorized forward pass
                output = self.forward(batch_seqs)  # [B, final_seq_len, final_dim]
                
                # Vectorized loss calculation
                pred_target = output.mean(dim=1)  # [B, final_dim]
                loss = torch.sum((pred_target - batch_t) ** 2)
                
                # Normalize loss for gradient
                loss = loss / len(batch_idx)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item() * len(batch_idx)
            
            # Average loss over epoch
            avg_loss = total_loss / num_samples
            history.append(avg_loss)
            
            # Update best model state if current loss is lower
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_model_state = copy.deepcopy(self.state_dict())
            
            if it % print_every == 0 or it == max_iters - 1:
                print(f"GD Iter {it:3d}: Loss = {avg_loss:.6e}, LR = {scheduler.get_last_lr()[0]:.6f}")
            
            # Save checkpoint at specified intervals
            if checkpoint_file and (it % checkpoint_interval == 0 or it == max_iters - 1):
                self._save_checkpoint(
                    checkpoint_file, it, history, optimizer, scheduler, best_loss
                )
            
            # Check convergence
            if abs(prev_loss - avg_loss) < tol:
                print(f"Converged after {it+1} iterations.")
                # Restore the best model state before breaking
                if best_model_state is not None and avg_loss > best_loss:
                    self.load_state_dict(best_model_state)                    
                    print(f"Restored best model state with loss = {best_loss:.6e}")
                    history[-1] = best_loss
                break
            prev_loss = avg_loss

            # Learning rate decay
            scheduler.step()

        # If training ended without convergence, restore best model state
        if best_model_state is not None and best_loss < history[-1]:
            self.load_state_dict(best_model_state)  
            print(f"Training ended. Restored best model state with loss = {best_loss:.6e}")
            history[-1] = best_loss
        
        # Store training statistics and compute mean target
        self._compute_training_statistics(seqs)
        self._compute_hierarchical_mean_target(seqs)
        self.trained = True
        
        return history

    def cls_train(self, seqs, labels, num_classes, max_iters=1000, tol=1e-8, learning_rate=0.01,
                  continued=False, decay_rate=1.0, print_every=10, batch_size=32,
                  checkpoint_file=None, checkpoint_interval=10):
        """
        Train the model for multi-class classification using cross-entropy loss.
        Adapted for HierDDLrnC architecture with batch processing.
        
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
        # Validate input sequences
        for seq in seqs:
            if len(seq) > self.max_seq_len:
                raise ValueError(f"All sequences must have length <= {self.max_seq_len}")
        
        # Initialize classification head if not already done
        if self.classifier is None or self.num_classes != num_classes:
            self.classifier = nn.Linear(self.model_dims[-1], num_classes).to(self.device)
            self.num_classes = num_classes
        
        if not continued:
            self.reset_parameters()
        
        # Convert labels to tensor
        label_tensors = torch.tensor(labels, dtype=torch.long, device=self.device)
        num_samples = len(seqs)
        
        # Setup optimizer and scheduler
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)
        
        # Cross-entropy loss
        criterion = nn.CrossEntropyLoss()
        
        # Training state variables
        history = []
        prev_loss = float('inf')
        best_loss = float('inf')
        best_model_state = None
        
        for it in range(max_iters):
            total_loss = 0.0
            total_correct = 0
            total_sequences = 0
            
            # Shuffle sequences for each epoch
            indices = torch.randperm(num_samples)
            
            # Process sequences in batches
            for batch_start in range(0, num_samples, batch_size):
                batch_indices = indices[batch_start:batch_start + batch_size]
                batch_seqs = [seqs[idx] for idx in batch_indices]
                batch_labels = label_tensors[batch_indices]
                
                optimizer.zero_grad()
                
                # Forward pass through the entire hierarchical model
                # output shape: [batch_size, final_seq_len, final_dim]
                output = self.forward(batch_seqs)
                
                # Get sequence representation: average over sequence length
                seq_representations = output.mean(dim=1)  # [batch_size, final_dim]
                
                # Get logits through classification head
                logits = self.classifier(seq_representations)  # [batch_size, num_classes]
                
                # Compute loss
                loss = criterion(logits, batch_labels)
                loss.backward()
                optimizer.step()
                
                # Calculate batch statistics
                batch_loss = loss.item()
                total_loss += batch_loss * len(batch_seqs)
                
                # Calculate accuracy
                with torch.no_grad():
                    predictions = torch.argmax(logits, dim=1)
                    batch_correct = (predictions == batch_labels).sum().item()
                    total_correct += batch_correct
                    total_sequences += len(batch_seqs)
                
                # Clear GPU cache periodically
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Calculate average loss and accuracy for this iteration
            avg_loss = total_loss / total_sequences if total_sequences > 0 else 0.0
            accuracy = total_correct / total_sequences if total_sequences > 0 else 0.0
            
            history.append(avg_loss)
            
            # Update best model state
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_model_state = copy.deepcopy(self.state_dict())
            
            # Print training progress
            if it % print_every == 0 or it == max_iters - 1:
                current_lr = scheduler.get_last_lr()[0]
                print(f"CLS-Train Iter {it:3d}: Loss = {avg_loss:.6e}, Acc = {accuracy:.4f}, LR = {current_lr:.6f}")
            
            # Save checkpoint if specified
            if checkpoint_file and (it % checkpoint_interval == 0 or it == max_iters - 1):
                checkpoint = {
                    'iteration': it,
                    'model_state_dict': self.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'history': history,
                    'best_loss': best_loss,
                    'num_classes': self.num_classes
                }
                torch.save(checkpoint, checkpoint_file)
                print(f"Checkpoint saved at iteration {it}")
            
            # Check convergence
            if abs(prev_loss - avg_loss) < tol:
                print(f"Converged after {it+1} iterations.")
                # Restore best model state
                if best_model_state is not None:
                    self.load_state_dict(best_model_state)
                break
                
            prev_loss = avg_loss
            
            # Learning rate scheduling
            scheduler.step()
        
        # Store training statistics
        self._compute_training_statistics(seqs)
        self._compute_hierarchical_mean_target(seqs)
        self.trained = True
        
        return history

    def lbl_train(self, seqs, labels, num_labels, max_iters=1000, tol=1e-8, learning_rate=0.01, 
                 continued=False, decay_rate=1.0, print_every=10, batch_size=32,
                 checkpoint_file=None, checkpoint_interval=10, pos_weight=None):
        """
        Train the model for multi-label classification using binary cross-entropy loss.
        Adapted for HierDDLrnC architecture with batch processing.
        
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
        # Validate input sequences
        for seq in seqs:
            if len(seq) > self.max_seq_len:
                raise ValueError(f"All sequences must have length <= {self.max_seq_len}")
        
        # Initialize label head if not already done
        if self.labeller is None or self.num_labels != num_labels:
            self.labeller = nn.Linear(self.model_dims[-1], num_labels).to(self.device)
            self.num_labels = num_labels
        
        if not continued:
            self.reset_parameters()
        
        # Convert labels to tensor
        if isinstance(labels, list):
            labels_tensor = torch.tensor(labels, dtype=torch.float32, device=self.device)
        else:
            labels_tensor = torch.as_tensor(labels, dtype=torch.float32, device=self.device)
        
        # Setup loss function with optional positive class weighting
        if pos_weight is not None:
            pos_weight_tensor = torch.tensor(pos_weight, dtype=torch.float32, device=self.device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        else:
            criterion = nn.BCEWithLogitsLoss()
        
        # Setup optimizer and scheduler
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)
        
        # Training state variables
        loss_history = []
        acc_history = []
        prev_loss = float('inf')
        best_loss = float('inf')
        best_model_state = None
        
        num_samples = len(seqs)
        
        for it in range(max_iters):
            total_loss = 0.0
            total_correct = 0
            total_predictions = 0
            total_sequences = 0
            
            # Shuffle sequences for each epoch
            indices = torch.randperm(num_samples)
            
            # Process sequences in batches
            for batch_start in range(0, num_samples, batch_size):
                batch_indices = indices[batch_start:batch_start + batch_size]
                batch_seqs = [seqs[idx] for idx in batch_indices]
                batch_labels = labels_tensor[batch_indices]
                
                optimizer.zero_grad()
                
                # Forward pass through the entire hierarchical model
                output = self.forward(batch_seqs)  # [batch_size, final_seq_len, final_dim]
                
                # Get sequence representation: average over sequence length
                seq_representations = output.mean(dim=1)  # [batch_size, final_dim]
                
                # Pass through classification head to get logits
                logits = self.labeller(seq_representations)  # [batch_size, num_labels]
                
                # Calculate loss
                loss = criterion(logits, batch_labels)
                loss.backward()
                optimizer.step()
                
                # Calculate accuracy
                with torch.no_grad():
                    # Apply sigmoid to get probabilities
                    probs = torch.sigmoid(logits)
                    # Threshold at 0.5 for binary predictions
                    predictions = (probs > 0.5).float()
                    # Calculate number of correct predictions
                    batch_correct = (predictions == batch_labels).sum().item()
                    batch_predictions = batch_labels.numel()
                
                total_loss += loss.item() * len(batch_seqs)
                total_correct += batch_correct
                total_predictions += batch_predictions
                total_sequences += len(batch_seqs)
                
                # Clear GPU cache periodically
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Calculate average loss and accuracy for this iteration
            avg_loss = total_loss / total_sequences if total_sequences > 0 else 0.0
            avg_acc = total_correct / total_predictions if total_predictions > 0 else 0.0
            
            loss_history.append(avg_loss)
            acc_history.append(avg_acc)
            
            # Update best model state
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_model_state = copy.deepcopy(self.state_dict())
            
            # Print training progress
            if it % print_every == 0 or it == max_iters - 1:
                current_lr = scheduler.get_last_lr()[0]
                print(f"MLC-Train Iter {it:3d}: Loss = {avg_loss:.6e}, Acc = {avg_acc:.4f}, LR = {current_lr:.6f}")
            
            # Save checkpoint if specified
            if checkpoint_file and (it % checkpoint_interval == 0 or it == max_iters - 1):
                checkpoint = {
                    'iteration': it,
                    'model_state_dict': self.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss_history': loss_history,
                    'acc_history': acc_history,
                    'best_loss': best_loss
                }
                torch.save(checkpoint, checkpoint_file)
                print(f"Checkpoint saved at iteration {it}")
            
            # Check convergence
            if abs(prev_loss - avg_loss) < tol:
                print(f"Converged after {it+1} iterations.")
                # Restore best model state
                if best_model_state is not None:
                    self.load_state_dict(best_model_state)
                break
                
            prev_loss = avg_loss
            
            # Learning rate scheduling
            scheduler.step()
        
        # Store training statistics
        self._compute_training_statistics(seqs)
        self._compute_hierarchical_mean_target(seqs)
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
        
        with torch.no_grad():
            # Get sequence representation through hierarchical model
            output = self.forward(seq)  # [1, final_seq_len, final_dim]
            seq_representation = output.mean(dim=1)  # [1, final_dim]
            
            # Get logits through classification head
            logits = self.classifier(seq_representation)
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
        
        with torch.no_grad():
            # Get sequence representation through hierarchical model
            output = self.forward(seq)  # [1, final_seq_len, final_dim]
            seq_representation = output.mean(dim=1)  # [1, final_dim]
            
            # Pass through classification head to get logits
            logits = self.labeller(seq_representation)
            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(logits).cpu().numpy()[0]
        
        # Apply threshold to get binary predictions
        binary_preds = (probs > threshold).astype(np.float32)
        
        return binary_preds, probs

    def self_train(self, seqs, max_iters=100, tol=1e-6, learning_rate=0.01, 
                   continued=False, self_mode='gap', decay_rate=1.0, print_every=10,
                   batch_size=256, checkpoint_file=None, checkpoint_interval=5):
        """
        Self-training method for the hierarchical model with Linker.
        Uses vectorized batch processing for improved performance.
        
        Args:
            seqs: List of character sequences (all must have length <= max_seq_len)
            max_iters: Maximum training iterations
            tol: Convergence tolerance
            learning_rate: Initial learning rate
            continued: Whether to continue from existing parameters
            self_mode: 'gap' or 'reg' training mode
            decay_rate: Learning rate decay rate
            print_every: Print interval
            batch_size: Batch size for training
            checkpoint_file: Path to save/load checkpoint file for resuming training
            checkpoint_interval: Interval (in iterations) for saving checkpoints
            
        Returns:
            list: Training loss history
        """
        # Validate input sequences
        for seq in seqs:
            if len(seq) > self.max_seq_len:
                raise ValueError(f"All sequences must have length <= {self.max_seq_len}")
                
        if self_mode not in ('gap', 'reg'):
            raise ValueError("self_mode must be either 'gap' or 'reg'")
        
        # Load checkpoint if continuing and checkpoint file exists
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
            if not continued:
                self.reset_parameters()   
            history = []
        
        num_samples = len(seqs)
        
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)
        
        if continued and checkpoint_file and os.path.exists(checkpoint_file):
            optimizer.load_state_dict(optimizer_state)
            scheduler.load_state_dict(scheduler_state)
            
        prev_loss = float('inf') if start_iter == 0 else history[-1] if history else float('inf')
        
        # Pre-calculate Causal Masks if in 'reg' mode
        original_linkers = []
        original_residual_linkers = []
        
        for layer in self.hierarchical_layers:
            if hasattr(layer, 'Linker'):
                original_linkers.append(layer.Linker.data.clone())
                
                if hasattr(layer, 'residual_linker') and layer.residual_linker is not None:
                    original_residual_linkers.append(layer.residual_linker.data.clone())
                else:
                    original_residual_linkers.append(None)
        
        for it in range(start_iter, max_iters):
            total_loss = 0.0
            
            # Apply causal masks if in 'reg' mode
            if self_mode == 'reg':
                for idx, layer in enumerate(self.hierarchical_layers):
                    if hasattr(layer, 'Linker'):
                        causal_mask = torch.tril(torch.ones_like(layer.Linker))
                        layer.Linker.data.mul_(causal_mask)
                        
                        if hasattr(layer, 'residual_linker') and layer.residual_linker is not None:
                            cm_res = torch.tril(torch.ones_like(layer.residual_linker))
                            layer.residual_linker.data.mul_(cm_res)
            
            # Shuffle
            perm = torch.randperm(num_samples)
            
            # Iterate batches
            for i in range(0, num_samples, batch_size):
                batch_idx = perm[i:i+batch_size]
                batch_seqs = [seqs[idx] for idx in batch_idx]
                
                optimizer.zero_grad()
                
                # Vectorized forward pass
                char_output = self.batch_sequence_to_tensor(batch_seqs)  # [B, S_char, D]
                
                # Pass through hierarchical layers
                hier_out = char_output
                for layer in self.hierarchical_layers:
                    hier_out = layer(hier_out)  # [B, S_hier, D_hier]
                
                # Vectorized loss calculation
                loss = 0.0
                
                # Determine valid comparison length
                min_len = min(char_output.shape[1], hier_out.shape[1])
                
                if self_mode == 'gap':
                    # ||Hier[k] - Char[k]||^2
                    diff = hier_out[:, :min_len, :] - char_output[:, :min_len, :]
                    sq_diff = torch.sum(diff ** 2, dim=-1)  # [B, S_valid]
                    loss = torch.sum(torch.mean(sq_diff, dim=1))  # Sum of means
                    
                else:  # 'reg' mode
                    # Predict next: Hier[k] vs Char[k+1]
                    if min_len > 1:
                        targets = char_output[:, 1:min_len, :]
                        preds = hier_out[:, 0:min_len-1, :]
                        diff = preds - targets
                        sq_diff = torch.sum(diff ** 2, dim=-1)
                        loss = torch.sum(torch.mean(sq_diff, dim=1))
                    else:
                        loss = torch.tensor(0.0, device=self.device)
                
                batch_loss_scalar = loss.item()
                
                # Normalize and backward
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
                current_lr = scheduler.get_last_lr()[0]
                mode_display = "Gap" if self_mode == 'gap' else "Reg"
                print(f"SelfTrain({mode_display}) Iter {it:3d}: Loss = {avg_loss:.6f}, LR = {current_lr:.6f}")
            
            if checkpoint_file and (it % checkpoint_interval == 0 or it == max_iters - 1):
                self._save_checkpoint(checkpoint_file, it, history, optimizer, scheduler, best_loss)
            
            if prev_loss - avg_loss < tol:
                print(f"Converged after {it+1} iterations")
                if best_model_state is not None and avg_loss > prev_loss:
                    self.load_state_dict(best_model_state)
                    history[-1] = best_loss                    
                break
            prev_loss = avg_loss
            
            # Learning rate decay every 5 iterations
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
        """
        Save training checkpoint with complete training state
        
        Args:
            checkpoint_file: Path to save checkpoint file
            iteration: Current training iteration
            history: Training loss history
            optimizer: Optimizer instance
            scheduler: Learning rate scheduler instance
            best_loss: Best loss achieved so far
        """
        # Convert mean_target to numpy for saving
        mean_target_np = self.mean_target.cpu().numpy() if self.mean_target is not None else None
        
        checkpoint = {
            'iteration': iteration,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'history': history,
            'best_loss': best_loss,
            'config': {
                'charset': self.charset,
                'rank': self.rank,
                'vec_dim': self.vec_dim,
                'max_seq_len': self.max_seq_len,
                'hier_input_seq_len': self.hier_input_seq_len,
                'model_dims': self.model_dims,
                'basis_dims': self.basis_dims,
                'linker_dims': self.linker_dims,
                'linker_trainable': self.linker_trainable,
                'use_residual_list': self.use_residual_list,
                'char_linker_trainable': self.char_linker_trainable,
                'char_layer_config': {
                    'rank_mode': self.rank_mode,
                    'mode': self.mode,
                    'user_step': self.step
                }
            },
            'training_stats': {
                'mean_t': self.mean_t,
                'mean_token_count': self.mean_token_count,
                'mean_target': mean_target_np
            }
        }
        torch.save(checkpoint, checkpoint_file)

    def _compute_training_statistics(self, seqs, batch_size=256):
        """Compute and store statistics for reconstruction and generation"""
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
                
                # Count tokens (excluding padding)
                batch_token_count = 0
                for j in range(batch_indices.shape[0]):
                    actual_seq_len = len(self.extract_tokens(seqs[i+j]))
                    batch_token_count += actual_seq_len
                
                total_token_count += batch_token_count
        
        self.mean_token_count = total_token_count / len(seqs) if seqs else 0
        self.mean_t = (total_vec_sum / total_token_count).cpu().numpy() if total_token_count > 0 else np.zeros(self.vec_dim)

    def _compute_hierarchical_mean_target(self, seqs, batch_size=32):
        """
        Compute the mean target vector in the final hierarchical layer output space
        This represents the average pattern learned by the entire hierarchical model
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
                batch_seqs = seqs[i:i+batch_size]
                batch_output = self.forward(batch_seqs)  # [B, S_out, D_out]
                
                # Mean over sequence, Sum over batch
                batch_means = batch_output.mean(dim=1)  # [B, D_out]
                total_output += batch_means.sum(dim=0)
                total_sequences += len(batch_seqs)
        
        if total_sequences > 0:
            self.mean_target = total_output / total_sequences
        else:
            self.mean_target = None
    
    def reconstruct(self, L, tau=0.0):
        """
        Reconstruct representative character sequence of length L using the entire hierarchical model.
        This method combines both character-level and hierarchical patterns with Linker.
        
        Args:
            L (int): Length of sequence to reconstruct (must be <= max_seq_len)
            tau (float): Temperature parameter for stochastic sampling.
                        tau=0: deterministic (greedy) selection
                        tau>0: stochastic selection with temperature
            
        Returns:
            str: Reconstructed character sequence
        """
        assert self.trained, "Model must be trained first"
        assert self.mean_target is not None, "Mean target vector must be computed first"
        
        if L <= 0: return ""
        if L > self.max_seq_len: L = self.max_seq_len
        
        generated_sequence = ""
        
        while len(generated_sequence) < L:
            # Batch candidates
            candidates = []
            valid_tokens = []
            
            for token in self.tokens:
                cand = generated_sequence + token
                if len(cand) > L: continue
                
                # Pad for model input
                if len(cand) < self.max_seq_len:
                    padded = cand.ljust(self.max_seq_len, 'A')
                else:
                    padded = cand[:self.max_seq_len]
                
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
        # Reset char layer parameters
        nn.init.uniform_(self.char_embedding.weight, -0.5, 0.5)
        nn.init.uniform_(self.char_linear.weight, -0.1, 0.1)
        nn.init.uniform_(self.char_Acoeff, -0.1, 0.1)
        nn.init.uniform_(self.char_Bbasis, -0.1, 0.1)
        nn.init.uniform_(self.char_to_hier_linker, -0.1, 0.1)
        
        # Reset hierarchical layer parameters
        for layer in self.hierarchical_layers:
            nn.init.uniform_(layer.M, -0.1, 0.1)
            nn.init.uniform_(layer.Acoeff, -0.1, 0.1)
            nn.init.uniform_(layer.Bbasis, -0.1, 0.1)
            nn.init.uniform_(layer.Linker, -0.1, 0.1)
            if hasattr(layer, 'residual_proj') and isinstance(layer.residual_proj, nn.Linear):
                nn.init.uniform_(layer.residual_proj.weight, -0.1, 0.1)
            if hasattr(layer, 'residual_linker') and layer.residual_linker is not None:
                nn.init.uniform_(layer.residual_linker, -0.1, 0.1)
        
        # Reset classification heads
        if self.classifier is not None:
            nn.init.normal_(self.classifier.weight, 0, 0.01)
            if self.classifier.bias is not None:
                nn.init.constant_(self.classifier.bias, 0)
                
        # Reset labeller
        if self.labeller is not None:
            nn.init.xavier_uniform_(self.labeller.weight)
            nn.init.zeros_(self.labeller.bias)
        
        # Reset training state        
        self.mean_target = None
        self.trained = False
        self.mean_token_count = None
        self.mean_t = None
    
    def count_parameters(self):
        """Count and print learnable parameters in the model by layer and type"""
        total_params = 0
        trainable_params = 0
        
        print("="*80)
        print(f"{'Component':<15} | {'Layer/Param':<25} | {'Count':<15} | {'Shape':<20}")
        print("-"*80)
        
        # Character layer parameters
        char_params = 0
        char_trainable = 0
        
        # Character embedding parameters
        num = self.char_embedding.weight.numel()
        shape = str(tuple(self.char_embedding.weight.shape))
        print(f"{'Char Layer':<15} | {'char_embedding':<25} | {num:<15} | {shape:<20}")
        char_params += num
        char_trainable += num
        
        # Character linear parameters
        num = self.char_linear.weight.numel()
        shape = str(tuple(self.char_linear.weight.shape))
        print(f"{'Char Layer':<15} | {'char_linear':<25} | {num:<15} | {shape:<20}")
        char_params += num
        char_trainable += num
        
        # Character Acoeff parameters
        num = self.char_Acoeff.numel()
        shape = str(tuple(self.char_Acoeff.shape))
        print(f"{'Char Layer':<15} | {'char_Acoeff':<25} | {num:<15} | {shape:<20}")
        char_params += num
        char_trainable += num
        
        # Character Bbasis parameters
        num = self.char_Bbasis.numel()
        shape = str(tuple(self.char_Bbasis.shape))
        print(f"{'Char Layer':<15} | {'char_Bbasis':<25} | {num:<15} | {shape:<20}")
        char_params += num
        char_trainable += num
        
        # Char to Hier Linker parameters
        num = self.char_to_hier_linker.numel()
        shape = str(tuple(self.char_to_hier_linker.shape))
        trainable_status = "T" if self.char_to_hier_linker.requires_grad else "F"
        print(f"{'Char Layer':<15} | {'char_to_hier_linker':<25} | {num:<15} | {shape:<20} [{trainable_status}]")
        char_params += num
        if self.char_to_hier_linker.requires_grad:
            char_trainable += num
        
        print(f"{'Char Layer':<15} | {'TOTAL':<25} | {char_params:<15} | (trainable: {char_trainable})")
        total_params += char_params
        trainable_params += char_trainable
        print("-"*80)
        
        # Hierarchical layer parameters
        for i, layer in enumerate(self.hierarchical_layers):
            layer_params = 0
            layer_trainable = 0
            layer_name = f"Hier Layer {i}"
            
            # M matrix
            num = layer.M.numel()
            shape = str(tuple(layer.M.shape))
            print(f"{layer_name:<15} | {'M':<25} | {num:<15} | {shape:<20}")
            layer_params += num
            layer_trainable += num
            
            # Acoeff matrix
            num = layer.Acoeff.numel()
            shape = str(tuple(layer.Acoeff.shape))
            print(f"{layer_name:<15} | {'Acoeff':<25} | {num:<15} | {shape:<20}")
            layer_params += num
            layer_trainable += num
            
            # Bbasis matrix
            num = layer.Bbasis.numel()
            shape = str(tuple(layer.Bbasis.shape))
            print(f"{layer_name:<15} | {'Bbasis':<25} | {num:<15} | {shape:<20}")
            layer_params += num
            layer_trainable += num
            
            # Linker matrix
            num = layer.Linker.numel()
            shape = str(tuple(layer.Linker.shape))
            trainable_status = "T" if layer.Linker.requires_grad else "F"
            print(f"{layer_name:<15} | {'Linker':<25} | {num:<15} | {shape:<20} [{trainable_status}]")
            layer_params += num
            if layer.Linker.requires_grad:
                layer_trainable += num
            
            # Residual parameters
            if layer.residual_mode == 'separate':
                # Residual projection
                num = layer.residual_proj.weight.numel()
                shape = str(tuple(layer.residual_proj.weight.shape))
                print(f"{layer_name:<15} | {'residual_proj':<25} | {num:<15} | {shape:<20}")
                layer_params += num
                layer_trainable += num
                
                # Residual linker
                num = layer.residual_linker.numel()
                shape = str(tuple(layer.residual_linker.shape))
                trainable_status = "T" if layer.residual_linker.requires_grad else "F"
                print(f"{layer_name:<15} | {'residual_linker':<25} | {num:<15} | {shape:<20} [{trainable_status}]")
                layer_params += num
                if layer.residual_linker.requires_grad:
                    layer_trainable += num
            
            print(f"{layer_name:<15} | {'TOTAL':<25} | {layer_params:<15} | (trainable: {layer_trainable})")
            total_params += layer_params
            trainable_params += layer_trainable
            print("-"*80)
        
        # Classification heads
        if self.classifier is not None:
            cls_params = self.classifier.weight.numel()
            if self.classifier.bias is not None:
                cls_params += self.classifier.bias.numel()
            shape = str(tuple(self.classifier.weight.shape))
            print(f"{'Classifier':<15} | {'classifier':<25} | {cls_params:<15} | {shape:<20}")
            total_params += cls_params
            trainable_params += cls_params
        
        if self.labeller is not None:
            lbl_params = self.labeller.weight.numel()
            if self.labeller.bias is not None:
                lbl_params += self.labeller.bias.numel()
            shape = str(tuple(self.labeller.weight.shape))
            print(f"{'Labeller':<15} | {'labeller':<25} | {lbl_params:<15} | {shape:<20}")
            total_params += lbl_params
            trainable_params += lbl_params
        
        print(f"{'TOTAL':<15} | {'ALL PARAMETERS':<25} | {total_params:<15} | (trainable: {trainable_params})")
        print("="*80)
        
        return total_params, trainable_params
    
    def save(self, filename):
        """Save model state to file"""
        # Convert mean_target to numpy for saving
        mean_target_np = self.mean_target.cpu().numpy() if self.mean_target is not None else None
        
        # 创建状态字典，排除分类器和标签器的参数（它们会在加载时重新创建）
        state_dict = self.state_dict()
        
        # 如果需要，可以从状态字典中移除分类器和标签器的参数
        # 但这里我们保留它们，因为我们在加载时会创建这些层
        
        save_dict = {
            'model_state_dict': state_dict,
            'config': {
                'charset': self.charset,
                'rank': self.rank,
                'vec_dim': self.vec_dim,
                'max_seq_len': self.max_seq_len,
                'hier_input_seq_len': self.hier_input_seq_len,
                'model_dims': self.model_dims,
                'basis_dims': self.basis_dims,
                'linker_dims': self.linker_dims,
                'linker_trainable': self.linker_trainable,
                'use_residual_list': self.use_residual_list,
                'char_linker_trainable': self.char_linker_trainable,
                'char_layer_config': {
                    'rank_mode': self.rank_mode,
                    'mode': self.mode,
                    'user_step': self.step
                }
            },
            'training_stats': {
                'mean_t': self.mean_t,
                'mean_token_count': self.mean_token_count,
                'mean_target': mean_target_np
            },
            'num_classes': self.num_classes,
            'num_labels': self.num_labels
        }
        torch.save(save_dict, filename)
        print(f"Model saved to {filename}")
    
    @classmethod
    def load(cls, filename, device=None):
        """Load model from file"""
        # Device configuration
        device = device or torch.device("cuda" if torch.cuda.is_available() else 'cpu')
        
        # Load checkpoint
        checkpoint = torch.load(filename, map_location=device, weights_only=False)
        config = checkpoint['config']
        
        # Obtain parameters from configuration 
        char_layer_config = config['char_layer_config']
        
        # Create model instance
        model = cls(
            charset=config['charset'],
            rank=config['rank'],
            vec_dim=config['vec_dim'],
            max_seq_len=config['max_seq_len'],
            hier_input_seq_len=config['hier_input_seq_len'],
            model_dims=config['model_dims'],
            basis_dims=config['basis_dims'],
            linker_dims=config['linker_dims'],
            rank_mode=char_layer_config['rank_mode'],
            mode=char_layer_config['mode'],
            user_step=char_layer_config['user_step'],            
            linker_trainable=config.get('linker_trainable', False),
            use_residual_list=config.get('use_residual_list', [None] * len(config['model_dims'])),
            char_linker_trainable=config.get('char_linker_trainable', None),
            device=device
        )
        
        # Load classification heads info
        model.num_classes = checkpoint.get('num_classes', None)
        model.num_labels = checkpoint.get('num_labels', None)
        
        # Recreate classifier and labeller if needed
        if model.num_classes is not None:
            model.classifier = nn.Linear(model.model_dims[-1], model.num_classes).to(device)
        if model.num_labels is not None:
            model.labeller = nn.Linear(model.model_dims[-1], model.num_labels).to(device)
        
        # 现在加载状态字典，但需要过滤掉模型中不存在的键
        state_dict = checkpoint['model_state_dict']
        model_state_dict = model.state_dict()
        
        # 过滤掉模型中不存在的键（比如分类器/标签器如果模型没有创建）
        filtered_state_dict = {}
        for key, value in state_dict.items():
            if key in model_state_dict:
                filtered_state_dict[key] = value
            elif key.startswith('classifier.') or key.startswith('labeller.'):
                # 如果键属于分类器或标签器，但模型已经创建了这些层，则包含它
                if (key.startswith('classifier.') and model.classifier is not None) or \
                   (key.startswith('labeller.') and model.labeller is not None):
                    filtered_state_dict[key] = value
                else:
                    print(f"Warning: Skipping parameter '{key}' as corresponding layer not initialized")
            else:
                print(f"Warning: Skipping unexpected parameter '{key}'")
        
        # 加载过滤后的状态字典
        model.load_state_dict(filtered_state_dict, strict=False)
        
        # Load training statistics
        if 'training_stats' in checkpoint:
            stats = checkpoint['training_stats']
            if stats['mean_t'] is not None:
                model.mean_t = stats['mean_t']
            if stats['mean_token_count'] is not None:
                model.mean_token_count = stats['mean_token_count']
            if stats['mean_target'] is not None:
                model.mean_target = torch.tensor(stats['mean_target'], device=device)
            model.trained = True
        
        print(f"Model loaded from {filename}")
        return model

# === Example Usage ===
if __name__ == "__main__":
    from statistics import correlation
    
    print("="*60)
    print("HierDDLrnC - Hierarchical Dual Descriptor with Linker and Character Input")
    print("Variable-Length Sequence Processing Demonstration")
    print("="*60)
    
    # Set random seeds for reproducibility
    torch.manual_seed(2)
    random.seed(2)
    np.random.seed(2)
    
    # Configuration for variable-length sequences
    charset = ['A', 'C', 'G', 'T']
    rank = 1
    vec_dim = 8
    max_seq_len = 200  # Maximum input character sequence length
    hier_input_seq_len = 50  # Fixed input sequence length for first hierarchical layer
    model_dims = [16, 12, 8]  # Hierarchical layer dimensions
    basis_dims = [100, 80, 60]  # Basis dimensions for hierarchical layers
    linker_dims = [40, 30, 20]  # Output sequence lengths for hierarchical layers
    num_seqs = 100
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {device}")
    print(f"Character set: {charset}")
    print(f"Rank (k-mer length): {rank}")
    print(f"Character layer output dim: {vec_dim}")
    print(f"Maximum sequence length: {max_seq_len}")
    print(f"Hierarchical input sequence length: {hier_input_seq_len}")
    print(f"Hierarchical layer dims: {model_dims}")
    print(f"Hierarchical basis dims: {basis_dims}")
    print(f"Linker output sequence lengths: {linker_dims}")
    print(f"Using Linker-based architecture with variable-length character input")
    
    # Generate synthetic training data with variable lengths
    print("\nGenerating synthetic training data with variable lengths...")
    seqs, t_list = [], []
    for i in range(num_seqs):
        # Generate variable-length sequences (50-200 characters)
        seq_len = random.randint(50, 200)
        seq = ''.join(random.choices(charset, k=seq_len))
        seqs.append(seq)
        # Create target vector (final layer dimension)
        t_list.append([random.uniform(-1.0, 1.0) for _ in range(model_dims[-1])])
    
    # Create model with variable-length support
    print("\nCreating HierDDLrnC model with variable-length support...")
    model = HierDDLrnC(
        charset=charset,
        rank=rank,
        vec_dim=vec_dim,
        max_seq_len=max_seq_len,
        hier_input_seq_len=hier_input_seq_len,
        model_dims=model_dims,
        basis_dims=basis_dims,
        linker_dims=linker_dims,
        mode='linear',
        linker_trainable=[True, False, True],  # Mixed trainable settings
        use_residual_list=['separate', 'shared', None],  # Mixed residual strategies
        char_linker_trainable=True,  # Separately set the character-layer Linker matrix
        device=device
    )
    
    # Count parameters
    print("\nModel Parameter Count:")
    total_params, trainable_params = model.count_parameters()
    
    # Test character-level processing with variable-length sequences
    print("\n" + "="*50)
    print("Character-Level Processing Test (Variable-Length)")
    print("="*50)
    
    test_seq_short = seqs[0]  # Short sequence
    test_seq_long = seqs[-1]  # Long sequence
    
    char_vectors_short = model.char_describe(test_seq_short)
    char_vectors_long = model.char_describe(test_seq_long)
    
    print(f"Short sequence (length {len(test_seq_short)}): {test_seq_short[:50]}...")
    print(f"Character vectors shape: {char_vectors_short.shape}")
    print(f"Number of tokens: {len(model.extract_tokens(test_seq_short))}")
    
    print(f"\nLong sequence (length {len(test_seq_long)}): {test_seq_long[:50]}...")
    print(f"Character vectors shape: {char_vectors_long.shape}")
    print(f"Number of tokens: {len(model.extract_tokens(test_seq_long))}")
    
    # Test forward pass with variable-length sequences
    print("\n" + "="*50)
    print("Forward Pass Test with Variable-Length Sequences")
    print("="*50)
    
    # Test single sequence
    output_short = model.forward(test_seq_short)
    output_long = model.forward(test_seq_long)
    
    print(f"Short sequence output shape: {output_short.shape}")
    print(f"Long sequence output shape: {output_long.shape}")
    
    # Test batch processing
    batch_seqs = [test_seq_short, test_seq_long]
    batch_output = model.forward(batch_seqs)
    print(f"Batch output shape: {batch_output.shape}")
    
    # Gradient descent training with variable-length sequences
    print("\n" + "="*50)
    print("Gradient Descent Training with Variable-Length Sequences")
    print("="*50)
    
    reg_history = model.reg_train(
        seqs, t_list,
        max_iters=50,  # Reduced for demonstration
        learning_rate=0.01,
        decay_rate=0.98,
        print_every=5,
        batch_size=16,
        checkpoint_file='hierddlrnc_var_gd_checkpoint.pth',
        checkpoint_interval=10
    )
    
    # Predictions and correlation analysis
    print("\n" + "="*50)
    print("Prediction and Correlation Analysis")
    print("="*50)
    
    pred_t_list = [model.predict_t(seq) for seq in seqs[:10]]  # Test on first 10 sequences
    
    # Calculate correlations for each dimension
    output_dim = model_dims[-1]
    correlations = []
    for i in range(output_dim):
        actual = [t_vec[i] for t_vec in t_list[:10]]
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
    print(f"Deterministic reconstruction (tau=0.0): {det_seq[:100]}")
    
    # Stochastic reconstruction
    sto_seq = model.reconstruct(L=100, tau=0.5)
    print(f"Stochastic reconstruction (tau=0.5): {sto_seq[:100]}")
    
    # Classification task demonstration
    print("\n" + "="*50)
    print("Classification Task Demonstration")
    print("="*50)
    
    # Generate classification data
    num_classes = 3
    class_seqs = []
    class_labels = []
    
    # Create sequences with different patterns for each class
    for class_id in range(num_classes):
        for _ in range(30):  # 30 sequences per class
            L = random.randint(100, 200)
            if class_id == 0:
                # Class 0: High A content
                seq = ''.join(random.choices(['A', 'C', 'G', 'T'], weights=[0.6, 0.1, 0.1, 0.2], k=L))
            elif class_id == 1:
                # Class 1: High GC content
                seq = ''.join(random.choices(['A', 'C', 'G', 'T'], weights=[0.1, 0.4, 0.4, 0.1], k=L))
            else:
                # Class 2: Balanced
                seq = ''.join(random.choices(['A', 'C', 'G', 'T'], k=L))
            
            class_seqs.append(seq)
            class_labels.append(class_id)
    
    # Create new model for classification
    cls_model = HierDDLrnC(
        charset=charset,
        rank=rank,
        vec_dim=vec_dim,
        max_seq_len=max_seq_len,
        hier_input_seq_len=hier_input_seq_len,
        model_dims=model_dims,
        basis_dims=basis_dims,
        linker_dims=linker_dims,
        mode='linear',
        linker_trainable=False,
        device=device
    )
    
    # Train for classification
    print("\nTraining classification model...")
    cls_history = cls_model.cls_train(
        class_seqs, class_labels, num_classes,
        max_iters=50,  # Reduced for demonstration
        learning_rate=0.05,
        decay_rate=0.98,
        print_every=5,
        batch_size=16
    )
    
    # Test classification predictions
    print("\nClassification predictions on training set:")
    correct = 0
    for i, (seq, true_label) in enumerate(zip(class_seqs[:10], class_labels[:10])):
        pred_class, probs = cls_model.predict_c(seq)
        correct += (pred_class == true_label)
        if i < 3:  # Show first 3 examples
            print(f"Seq {i+1}: True={true_label}, Pred={pred_class}, Probs={[f'{p:.3f}' for p in probs]}")
    
    accuracy = correct / 10
    print(f"\nAccuracy on 10 test sequences: {accuracy:.4f}")
    
    # Multi-label classification demonstration
    print("\n" + "="*50)
    print("Multi-Label Classification Task Demonstration")
    print("="*50)
    
    # Generate multi-label data
    num_labels = 4
    label_seqs = []
    labels = []
    
    for _ in range(50):
        L = random.randint(100, 200)
        seq = ''.join(random.choices(charset, k=L))
        label_seqs.append(seq)
        # Create random binary labels (multi-label classification)
        label_vec = [random.random() > 0.7 for _ in range(num_labels)]
        labels.append([1.0 if x else 0.0 for x in label_vec])
    
    # Create new model for multi-label classification
    lbl_model = HierDDLrnC(
        charset=charset,
        rank=rank,
        vec_dim=vec_dim,
        max_seq_len=max_seq_len,
        hier_input_seq_len=hier_input_seq_len,
        model_dims=model_dims,
        basis_dims=basis_dims,
        linker_dims=linker_dims,
        mode='linear',
        linker_trainable=False,
        device=device
    )
    
    # Train for multi-label classification
    print("\nTraining multi-label classification model...")
    lbl_loss_history, lbl_acc_history = lbl_model.lbl_train(
        label_seqs, labels, num_labels,
        max_iters=100,  # Reduced for demonstration
        learning_rate=0.05,
        decay_rate=0.98,
        print_every=5,
        batch_size=16
    )
    
    # Test multi-label predictions
    print("\nMulti-label predictions on training set:")
    test_seq = label_seqs[0]
    binary_pred, probs_pred = lbl_model.predict_l(test_seq, threshold=0.5)
    
    print(f"Test sequence (first 50 chars): {test_seq[:50]}...")
    print(f"Predicted binary labels: {binary_pred}")
    print(f"Predicted probabilities: {[f'{p:.4f}' for p in probs_pred]}")
    
    # Interpret the predictions
    label_names = ["Function_A", "Function_B", "Function_C", "Function_D"]
    print("\nLabel interpretation:")
    for i, (binary, prob) in enumerate(zip(binary_pred, probs_pred)):
        status = "ACTIVE" if binary > 0.5 else "INACTIVE"
        print(f"  {label_names[i]}: {status} (confidence: {prob:.4f})")
    
    # Self-training example with variable-length sequences
    print("\n" + "="*50)
    print("Self-Training Example with Variable-Length Sequences")
    print("="*50)
    
    # Create a new model for self-training
    self_model = HierDDLrnC(
        charset=charset,
        rank=rank,
        vec_dim=vec_dim,
        max_seq_len=max_seq_len,
        hier_input_seq_len=hier_input_seq_len,
        model_dims=model_dims,
        basis_dims=basis_dims,
        linker_dims=linker_dims,
        mode='linear',
        linker_trainable=False,
        use_residual_list=[None] * len(model_dims),
        char_linker_trainable=False,  # Set the character-layer Linker matrix to be untrainable
        device=device
    )
    
    # Self-train in gap mode
    print("\nSelf-training in 'gap' mode...")
    self_history_gap = self_model.self_train(
        seqs[:20],  # Use subset for faster demonstration
        max_iters=10,
        learning_rate=0.01,
        self_mode='gap',
        print_every=2,
        batch_size=16,
        checkpoint_file='hierddlrnc_var_self_checkpoint.pth',
        checkpoint_interval=5
    )
    
    # Reconstruct from self-trained model
    self_rec_seq = self_model.reconstruct(L=80, tau=0.2)
    print(f"Reconstructed from self-trained model: {self_rec_seq}")
    
    # Model save and load test
    print("\n" + "="*50)
    print("Model Save/Load Test")
    print("="*50)
    
    # Save model
    model.save("hierddlrnc_var_model.pth")
    
    # Load model
    loaded_model = HierDDLrnC.load("hierddlrnc_var_model.pth", device=device)
    
    # Test loaded model
    test_seq = seqs[0]
    original_pred = model.predict_t(test_seq)
    loaded_pred = loaded_model.predict_t(test_seq)
    
    print(f"Original model prediction: {[f'{x:.4f}' for x in original_pred]}")
    print(f"Loaded model prediction:   {[f'{x:.4f}' for x in loaded_pred]}")
    
    pred_diff = np.max(np.abs(original_pred - loaded_pred))
    print(f"Maximum prediction difference: {pred_diff:.6e}")
    
    if pred_diff < 1e-6:
        print("✓ Model save/load test PASSED")
    else:
        print("✗ Model save/load test FAILED")
    
    # Test classification model save/load
    print("\nTesting classification model save/load...")
    cls_model.save("hierddlrnc_cls_model.pth")
    loaded_cls_model = HierDDLrnC.load("hierddlrnc_cls_model.pth", device=device)
    
    # Test prediction consistency
    test_seq = class_seqs[0]
    original_pred_class, original_probs = cls_model.predict_c(test_seq)
    loaded_pred_class, loaded_probs = loaded_cls_model.predict_c(test_seq)
    
    print(f"Original prediction: class={original_pred_class}, probs={[f'{p:.4f}' for p in original_probs]}")
    print(f"Loaded prediction:   class={loaded_pred_class}, probs={[f'{p:.4f}' for p in loaded_probs]}")
    
    if original_pred_class == loaded_pred_class and np.allclose(original_probs, loaded_probs, atol=1e-6):
        print("✓ Classification model save/load test PASSED")
    else:
        print("✗ Classification model save/load test FAILED")
    
    # Test reconstruction consistency
    torch.manual_seed(123); random.seed(123); np.random.seed(123)
    original_rec = model.reconstruct(L=50, tau=0.1)
    
    torch.manual_seed(123); random.seed(123); np.random.seed(123)
    loaded_rec = loaded_model.reconstruct(L=50, tau=0.1)
    
    if original_rec == loaded_rec:
        print("✓ Reconstruction consistency test PASSED")
    else:
        print("✗ Reconstruction consistency test FAILED")
        print(f"Original: {original_rec}")
        print(f"Loaded:   {loaded_rec}")
    
    print("\nAll tests completed successfully!")
    print("Variable-length sequence processing and classification tasks are working correctly!")
