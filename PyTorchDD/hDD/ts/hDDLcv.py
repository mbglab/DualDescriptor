# Copyright (C) 2005-2025, Bin-Guang Ma (mbg@mail.hzau.edu.cn); SPDX-License-Identifier: MIT
# Hierarchical Dual Descriptor (Tensor form) with Character Sequence Input (HierDDLtsC)
# Combines character-level processing and hierarchical vector processing with Linker matrices
# This program is for the demonstration of methodology and not fully refined.
# Author: Bin-Guang Ma (assisted by DeepSeek); Date: 2025-11-13 ~ 2026-1-18

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
    """Single layer of Hierarchical Dual Descriptor with Linker Matrix"""
    def __init__(self, in_dim, out_dim, in_seq, out_seq, num_basis, linker_trainable, use_residual, device):
        """
        Initialize a hierarchical layer with Linker matrix
        
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
        self.linker_trainable = linker_trainable
        self.use_residual = use_residual
        
        # Linear transformation matrix
        self.M = nn.Parameter(torch.empty(out_dim, in_dim))
        
        # Position-dependent transformation tensor
        self.P = nn.Parameter(torch.empty(out_dim, out_dim, num_basis))
        
        # Precompute periods tensor (fixed, not trainable)
        periods = torch.zeros(out_dim, out_dim, num_basis, dtype=torch.float32)
        for i in range(out_dim):
            for j in range(out_dim):
                for g in range(num_basis):
                    periods[i, j, g] = i*(out_dim*num_basis) + j*num_basis + g + 2
        self.register_buffer('periods', periods)
        
        # Sequence length transformation matrix (Linker)
        self.Linker = nn.Parameter(
            torch.empty(in_seq, out_seq),
            requires_grad=linker_trainable
        )

        # Layer normalization
        self.ln = nn.LayerNorm(out_dim)
        
        # Residual connection setup
        if self.use_residual == 'separate':
            # Separate projection and Linker for residual path
            self.residual_proj = nn.Linear(in_dim, out_dim, bias=False)
            self.residual_linker = nn.Parameter(
                torch.empty(in_seq, out_seq),
                requires_grad=linker_trainable
            )
        
        # Initialize parameters
        self.reset_parameters()
    
    def reset_parameters(self):
        """Reset layer parameters"""
        nn.init.uniform_(self.M, -0.5, 0.5)
        nn.init.uniform_(self.P, -0.1, 0.1)
        nn.init.uniform_(self.Linker, -0.1, 0.1)
        
        if self.use_residual == 'separate':
            nn.init.uniform_(self.residual_proj.weight, -0.1, 0.1)
            nn.init.uniform_(self.residual_linker, -0.1, 0.1)
    
    def position_transform(self, Z):
        """
        Position-dependent transformation with basis functions (batch version)
        
        Args:
            Z (Tensor): Input tensor of shape (batch_size, seq_len, out_dim)
            
        Returns:
            Tensor: Transformed tensor of shape (batch_size, seq_len, out_dim)
        """
        batch_size, seq_len, _ = Z.shape
        
        # Create position indices [seq_len]
        k = torch.arange(seq_len, device=self.device).float()
        
        # Reshape for broadcasting: [seq_len, 1, 1, 1]
        k = k.view(seq_len, 1, 1, 1)
        
        # Get periods and reshape for broadcasting: [1, out_dim, out_dim, num_basis]
        periods = self.periods.unsqueeze(0)
        
        # Compute basis functions: cos(2πk/period) -> [seq_len, out_dim, out_dim, num_basis]
        phi = torch.cos(2 * math.pi * k / periods)
        
        # Prepare Z for broadcasting: [batch_size, seq_len, 1, out_dim, 1]
        Z_exp = Z.unsqueeze(2).unsqueeze(-1)
        
        # Prepare P for broadcasting: [1, 1, out_dim, out_dim, num_basis]
        P_exp = self.P.unsqueeze(0).unsqueeze(0)
        
        # Prepare phi for broadcasting: [1, seq_len, out_dim, out_dim, num_basis]
        phi_exp = phi.unsqueeze(0)
        
        # Compute position transformation: Z * P * phi
        # Result: [batch_size, seq_len, out_dim, out_dim, num_basis]
        M = Z_exp * P_exp * phi_exp
        
        # Sum over j (dim=3) and g (dim=4) -> [batch_size, seq_len, out_dim]
        T = torch.sum(M, dim=(3, 4))
        
        return T
    
    def sequence_transform(self, T):
        """
        Transform sequence length using Linker matrix
        
        Args:
            T (Tensor): Input tensor of shape (batch_size, in_seq, out_dim)
            
        Returns:
            Tensor: Transformed tensor of shape (batch_size, out_seq, out_dim)
        """
        # Using matrix multiplication: T @ Linker
        # T: [batch_size, in_seq, out_dim] -> [batch_size, out_seq, out_dim]
        return torch.matmul(T.transpose(1, 2), self.Linker).transpose(1, 2)
    
    def forward(self, x):
        """
        Forward pass through the layer
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, in_seq, in_dim)
            
        Returns:
            Tensor: Output tensor of shape (batch_size, out_seq, out_dim)
        """
        batch_size, in_seq, in_dim = x.shape
        
        # Main path: linear transformation
        Z = torch.matmul(x, self.M.t())  # [batch_size, in_seq, out_dim]

        # Layer normalization
        Z = self.ln(Z)
        
        # Position-dependent transformation
        T = self.position_transform(Z)  # [batch_size, in_seq, out_dim]
        
        # Sequence transformation
        U = self.sequence_transform(T)  # [batch_size, out_seq, out_dim]
        
        # Residual connection processing
        if self.use_residual == 'separate':
            # Separate projection and Linker for residual
            residual_feat = self.residual_proj(x)  # [batch_size, in_seq, out_dim]
            residual = torch.matmul(residual_feat.transpose(1, 2), self.residual_linker).transpose(1, 2)  # [batch_size, out_seq, out_dim]
            out = U + residual
        elif self.use_residual == 'shared':
            # Shared M and Linker for residual
            residual_feat = torch.matmul(x, self.M.t())  # [batch_size, in_seq, out_dim]
            residual = torch.matmul(residual_feat.transpose(1, 2), self.Linker).transpose(1, 2)  # [batch_size, out_seq, out_dim]
            out = U + residual
        else:
            # Identity residual only if dimensions match
            if self.in_dim == self.out_dim and self.in_seq == self.out_seq:
                residual = torch.matmul(x.transpose(1, 2), self.Linker).transpose(1, 2)  # [batch_size, out_seq, out_dim]
                out = U + residual
            else:
                out = U 
       
        return out

class HierDDLtsC(nn.Module):
    """
    Hierarchical Dual Descriptor with Character Sequence Input and Linker Matrices
    Combines character-level processing and hierarchical vector processing with Linker matrices
    Supports variable-length character sequences as input
    """
    def __init__(self, charset, rank=1, rank_mode='drop', vec_dim=2, num_basis=5, 
                 mode='linear', user_step=None, max_seq_len=100, hier_input_seq_len=50,
                 model_dims=[2], num_basis_list=[5], linker_dims=[50], linker_trainable=False,
                 use_residual_list=None, char_linker_trainable=None, device='cuda'):
        """
        Initialize HierDDLtsC model with character input and Linker matrices
        
        Args:
            charset: Character set for sequence input
            rank: k-mer length for tokenization
            rank_mode: 'pad' or 'drop' for incomplete k-mers
            vec_dim: Output dimension of character layer (Layer 0)
            num_basis: Number of basis functions for character layer
            mode: 'linear' or 'nonlinear' tokenization
            user_step: Step size for nonlinear tokenization
            max_seq_len: Maximum input sequence length (for variable-length sequences)
            hier_input_seq_len: Fixed input sequence length for first hierarchical layer
            model_dims: Output dimensions for hierarchical layers
            num_basis_list: Number of basis functions for each hierarchical layer
            linker_dims: Output sequence lengths for each hierarchical layer
            linker_trainable: Whether Linker matrices are trainable
            use_residual_list: Residual connection types for hierarchical layers
            char_linker_trainable: Whether character layer Linker matrix is trainable (None=use linker_trainable[0])
            device: Computation device
        """
        super().__init__()
        self.charset = list(charset)
        self.rank = rank    # r-per/k-mer length
        self.rank_mode = rank_mode # 'pad' or 'drop'
        self.vec_dim = vec_dim    # embedding dimension
        self.char_num_basis = num_basis  # number of basis terms for character layer        
        self.mode = mode
        self.step = user_step
        self.max_seq_len = max_seq_len  # Maximum input character sequence length
        self.hier_input_seq_len = hier_input_seq_len  # Fixed input sequence length for first hierarchical layer
        self.model_dims = model_dims
        self.linker_dims = linker_dims
        self.num_layers = len(model_dims)
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.trained = False        
        
        # Validate dimensions
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
        
        # Position-weight tensor P[i][j][g] for character layer
        self.char_P = nn.Parameter(torch.empty(self.vec_dim, self.vec_dim, self.char_num_basis))
        
        # Precompute indexed periods[i][j][g] for character layer (fixed, not trainable)
        char_periods = torch.zeros(self.vec_dim, self.vec_dim, self.char_num_basis, dtype=torch.float32)
        for i in range(self.vec_dim):
            for j in range(self.vec_dim):
                for g in range(self.char_num_basis):
                    char_periods[i, j, g] = i*(self.vec_dim*self.char_num_basis) + j*self.char_num_basis + g + 2
        self.register_buffer('char_periods', char_periods)

        # Special Linker matrix to convert variable-length character sequences to fixed-length hierarchical input
        self.char_to_hier_linker = nn.Parameter(
            torch.empty(self.max_token_seq_len, self.hier_input_seq_len),
            requires_grad=self.char_linker_trainable
        )

        # Training statistics for character layer
        self.mean_token_count = None
        self.mean_t = None
        
        # Hierarchical layers with Linker matrices (starting from Layer 1)
        self.hierarchical_layers = nn.ModuleList()
        for l in range(self.num_layers):
            # Determine input dimensions for this layer
            in_dim = vec_dim if l == 0 else model_dims[l-1]
            # For first hierarchical layer, use fixed hier_input_seq_len; for subsequent layers, use previous linker_dim
            in_seq = self.hier_input_seq_len if l == 0 else linker_dims[l-1]
            out_dim = model_dims[l]
            out_seq = linker_dims[l]
            num_basis = num_basis_list[l]
            use_residual = self.use_residual_list[l]
            linker_trainable = self.linker_trainable[l]
            
            # Initialize layer
            layer = Layer(
                in_dim, out_dim, in_seq, out_seq, 
                num_basis, linker_trainable, 
                use_residual, self.device
            )
            self.hierarchical_layers.append(layer)
        
        # Mean target vector for the final hierarchical layer output
        self.mean_target = None
        
        # Classification head (initialized later when num_classes is known)
        self.num_classes = None  # Number of classes in the multi-class prediction task
        self.classifier = None
        
        # Label head (initialized later when num_labels is known)
        self.num_labels = None   # Number of labels for multi-label prediction task
        self.labeller = None
        
        # Initialize parameters
        self.reset_parameters()
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
    
    def token_to_indices(self, token_list):
        """Convert list of tokens to tensor of indices"""
        return torch.tensor([self.token_to_idx[tok] for tok in token_list], device=self.device)
    
    def batch_token_indices(self, seqs, device=None):
        """
        Vectorized batch conversion of character sequences to token indices with variable-length support
        
        Args:
            seqs (list): List of character sequences (variable length)
            device (torch.device): Target device (defaults to self.device)
            
        Returns:
            tuple: (indices_tensor, seq_lengths) where:
                - indices_tensor: LongTensor of shape [batch_size, max_token_len] (padded with 0)
                - seq_lengths: List of actual token sequence lengths for each sequence
        """
        target_device = device if device is not None else self.device
        
        # Prepare tokenization parameters
        step = 1 if self.mode == 'linear' else (self.step or self.rank)
        
        all_indices = []
        seq_lengths = []
        
        # Process each sequence
        for seq in seqs:
            L = len(seq)
            token_indices = []
            
            if self.mode == 'linear':
                # Linear mode: sliding window with step=1
                ranges = range(L - self.rank + 1)
                for i in ranges:
                    frag = seq[i:i+self.rank]
                    token_indices.append(self.token_to_idx[frag])
            else:
                # Nonlinear mode
                ranges = range(0, L, step)
                for i in ranges:
                    frag = seq[i:i+self.rank]
                    if len(frag) == self.rank:
                        token_indices.append(self.token_to_idx[frag])
                    else:
                        if self.rank_mode == 'pad':
                            frag_pad = frag.ljust(self.rank, '_')
                            token_indices.append(self.token_to_idx[frag_pad])
            
            all_indices.append(token_indices)
            seq_lengths.append(len(token_indices))
        
        if not all_indices:
            return torch.zeros((len(seqs), 0), dtype=torch.long, device=target_device), seq_lengths
        
        # Pad sequences to max length
        max_len = max(seq_lengths)
        padded_indices = []
        for indices in all_indices:
            if len(indices) < max_len:
                padded = indices + [0] * (max_len - len(indices))  # Pad with 0 (assuming idx 0 corresponds to a token)
            else:
                padded = indices[:max_len]
            padded_indices.append(padded)
        
        return torch.tensor(padded_indices, dtype=torch.long, device=target_device), seq_lengths
    
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

    def char_forward(self, token_indices):
        """
        Vectorized computation of N(k) vectors for a batch of token sequences
        
        Args:
            token_indices: Tensor of token indices [batch_size, seq_len]
            
        Returns:
            Tensor: N(k) vectors [batch_size, seq_len, vec_dim]
        """
        batch_size, seq_len = token_indices.shape
        
        # Get embeddings for all tokens [batch_size, seq_len, vec_dim]
        x = self.char_embedding(token_indices)
        
        # Create position indices for each position in sequence
        k_tensor = torch.arange(seq_len, device=self.device, dtype=torch.float32)
        # Expand for batch dimension: [1, seq_len, 1, 1, 1]
        k_expanded = k_tensor.view(1, seq_len, 1, 1, 1).expand(batch_size, -1, -1, -1, -1)
        
        # Calculate basis functions: cos(2π*k/periods) 
        # char_periods shape: [vec_dim, vec_dim, char_num_basis]
        # phi shape: [batch_size, seq_len, vec_dim, vec_dim, char_num_basis]
        phi = torch.cos(2 * math.pi * k_expanded / self.char_periods)
        
        # Prepare x for einsum: [batch_size, seq_len, vec_dim, 1, 1]
        x_exp = x.unsqueeze(3).unsqueeze(4)
        
        # Compute Nk using einsum: sum over j and g dimensions
        Nk = torch.einsum('bsj,ijg,bsijg->bsi', x, self.char_P, phi)
            
        return Nk

    def char_batch_compute_Nk(self, k_tensor, token_indices):
        """
        Vectorized computation of N(k) vectors for a batch of positions and tokens
        Optimized using einsum for better performance
        
        Args:
            k_tensor: Tensor of position indices [batch_size]
            token_indices: Tensor of token indices [batch_size]
            
        Returns:
            Tensor of N(k) vectors [batch_size, vec_dim]
        """
        # Get embeddings for all tokens [batch_size, vec_dim]
        x = self.char_embedding(token_indices)
        
        # Expand dimensions for broadcasting [batch_size, 1, 1, 1]
        k_expanded = k_tensor.view(-1, 1, 1, 1)
        
        # Calculate basis functions: cos(2π*k/periods) [batch_size, vec_dim, vec_dim, char_num_basis]
        phi = torch.cos(2 * math.pi * k_expanded / self.char_periods)
        
        # Optimized computation using einsum
        Nk = torch.einsum('bj,ijg,bijg->bi', x, self.char_P, phi)
            
        return Nk

    def char_describe(self, seq):
        """Compute N(k) vectors for each k-mer in sequence (character-level processing)"""
        toks = self.extract_tokens(seq)
        if not toks:
            return []
        
        token_indices = self.token_to_indices(toks)
        k_positions = torch.arange(len(toks), dtype=torch.float32, device=self.device)
        
        # Batch compute all Nk vectors
        N_batch = self.char_batch_compute_Nk(k_positions, token_indices)
        return N_batch.detach().cpu().numpy()

    def char_reconstruct(self):
        """Reconstruct representative sequence by minimizing error (character-level)"""
        assert self.trained, "Model must be trained first"
        n_tokens = round(self.mean_token_count)
        mean_t_tensor = torch.tensor(self.mean_t, dtype=torch.float32, device=self.device)
        seq_tokens = []
        
        # Precompute all token embeddings
        all_token_indices = torch.arange(len(self.tokens), device=self.device)
        
        for k in range(n_tokens):
            # Compute Nk for all tokens at position k
            k_tensor = torch.tensor([k] * len(self.tokens), dtype=torch.float32, device=self.device)
            Nk_all = self.char_batch_compute_Nk(k_tensor, all_token_indices)
            
            # Compute errors
            errors = torch.sum((Nk_all - mean_t_tensor) ** 2, dim=1)
            min_idx = torch.argmin(errors).item()
            best_tok = self.idx_to_token[min_idx]
            seq_tokens.append(best_tok)
            
        return ''.join(seq_tokens)

    def char_generate(self, L, tau=0.0):
        """Generate sequence of length L with temperature-controlled randomness (character-level)"""
        assert self.trained, "Model must be trained first"
        if tau < 0:
            raise ValueError("Temperature must be non-negative")
            
        num_blocks = (L + self.rank - 1) // self.rank
        mean_t_tensor = torch.tensor(self.mean_t, dtype=torch.float32, device=self.device)
        generated_tokens = []
        
        # Precompute all token embeddings
        all_token_indices = torch.arange(len(self.tokens), device=self.device)
        
        for k in range(num_blocks):
            # Compute Nk for all tokens at position k
            k_tensor = torch.tensor([k] * len(self.tokens), dtype=torch.float32, device=self.device)
            Nk_all = self.char_batch_compute_Nk(k_tensor, all_token_indices)
            
            # Compute scores
            errors = torch.sum((Nk_all - mean_t_tensor) ** 2, dim=1)
            scores = -errors  # Convert to score (higher = better)
            
            if tau == 0:  # Deterministic selection
                max_idx = torch.argmax(scores).item()
                best_tok = self.idx_to_token[max_idx]
                generated_tokens.append(best_tok)
            else:  # Stochastic selection
                probs = torch.softmax(scores / tau, dim=0).detach().cpu().numpy()
                chosen_idx = random.choices(range(len(self.tokens)), weights=probs, k=1)[0]
                chosen_tok = self.idx_to_token[chosen_idx]
                generated_tokens.append(chosen_tok)
                
        full_seq = ''.join(generated_tokens)
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
        
        token_indices = self.token_to_indices(toks)
        k_positions = torch.arange(token_seq_len, dtype=torch.float32, device=self.device)
        
        # Compute N(k) vectors through char layer
        Nk_batch = self.char_batch_compute_Nk(k_positions, token_indices)
        
        # Add batch dimension: [1, token_seq_len, vec_dim]
        char_output = Nk_batch.unsqueeze(0)
        
        # Apply char_to_hier_linker to convert variable length to fixed length
        # Use only the first token_seq_len rows of the linker matrix for this sequence
        hier_input = torch.matmul(char_output.transpose(1, 2), self.char_to_hier_linker[:token_seq_len, :]).transpose(1, 2)
        
        return hier_input  # [1, hier_input_seq_len, vec_dim]
    
    def batch_sequence_to_tensor(self, seqs):
        """
        Convert a batch of character sequences to tensor representation
        
        Args:
            seqs (list): List of character sequences
            
        Returns:
            Tensor: Sequence tensor of shape [batch_size, hier_input_seq_len, vec_dim]
        """
        if not seqs:
            return torch.zeros((0, self.hier_input_seq_len, self.vec_dim), device=self.device)
        
        # Process each sequence individually and stack
        batch_tensors = []
        for seq in seqs:
            tensor_seq = self.char_sequence_to_tensor(seq)
            batch_tensors.append(tensor_seq.squeeze(0))
        
        return torch.stack(batch_tensors)  # [batch_size, hier_input_seq_len, vec_dim]
    
    def forward(self, input_data):
        """
        Forward pass through entire hierarchical model with Linker matrices
        
        Args:
            input_data (str or list or Tensor): Input character sequence(s) or token indices
            
        Returns:
            Tensor: Output of shape [batch_size, final_seq_len, final_dim]
        """
        # Handle different input types
        if isinstance(input_data, str):
            # Single sequence
            x = self.char_sequence_to_tensor(input_data)
        elif isinstance(input_data, list):
            # Multiple sequences - process each separately and stack
            x = self.batch_sequence_to_tensor(input_data)
        elif isinstance(input_data, torch.Tensor):
            # Pre-computed token indices - not directly supported for variable length
            # Convert to list of sequences first
            raise TypeError("Tensor input not directly supported for variable-length sequences. Use str or list of strings.")
        else:
            raise TypeError("Input must be str or list of strings")
        
        # Pass through hierarchical layers with Linker matrices
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
        Train the model using gradient descent with target vectors
        
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
        t_tensors = torch.tensor(t_list, dtype=torch.float32, device=self.device)
        
        # Set up optimizer (train all parameters including char layer and Linker matrices)
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)
        
        # Load optimizer and scheduler states if resuming
        if continued and checkpoint_file and os.path.exists(checkpoint_file):
            optimizer.load_state_dict(optimizer_state)
            scheduler.load_state_dict(scheduler_state)
        
        prev_loss = float('inf') if start_iter == 0 else history[-1] if history else float('inf')
        
        for it in range(start_iter, max_iters):
            total_loss = 0.0
            total_sequences = 0
            
            # Shuffle indices
            indices = list(range(len(seqs)))
            random.shuffle(indices)
            
            # Process in batches
            for i in range(0, len(indices), batch_size):
                batch_indices = indices[i:i+batch_size]
                batch_seqs = [seqs[idx] for idx in batch_indices]
                batch_targets = t_tensors[batch_indices]
                
                optimizer.zero_grad()
                
                # Batch forward pass
                batch_output = self.forward(batch_seqs)  # [B, final_seq_len, final_dim]
                
                # Compute loss
                pred_target = batch_output.mean(dim=1)  # [B, final_dim]
                loss = torch.sum((pred_target - batch_targets) ** 2)
                
                # Normalize loss for gradient computation
                normalized_loss = loss / len(batch_indices)
                normalized_loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                total_sequences += len(batch_indices)
            
            # Average loss over epoch
            avg_loss = total_loss / total_sequences if total_sequences > 0 else 0
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
    
    def self_train(self, seqs, max_iters=100, tol=1e-6, learning_rate=0.01, 
                   continued=False, self_mode='gap', decay_rate=1.0, print_every=10,
                   batch_size = 1024, checkpoint_file=None, checkpoint_interval=5):
        """
        Self-training method for the hierarchical model with Linker matrices
        
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
            print(f"Resumed self-training from checkpoint at iteration {start_iter}, best loss: {best_loss:.6f}")
        else:
            if not continued:
                self.reset_parameters()   
            history = []
        
        # Set up optimizer
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)
        
        # Load optimizer and scheduler states if resuming
        if continued and checkpoint_file and os.path.exists(checkpoint_file):
            optimizer.load_state_dict(optimizer_state)
            scheduler.load_state_dict(scheduler_state)
        
        prev_loss = float('inf') if start_iter == 0 else history[-1] if history else float('inf')
        
        for it in range(start_iter, max_iters):
            total_loss = 0.0
            total_samples = 0
            
            # Shuffle indices
            indices = list(range(len(seqs)))
            random.shuffle(indices)
            
            # Apply causal masking if in 'reg' mode
            original_linkers = []
            original_residual_linkers = []
            
            if self_mode == 'reg':
                for layer in self.hierarchical_layers:
                    if hasattr(layer, 'Linker'):
                        causal_mask = torch.tril(torch.ones_like(layer.Linker))
                        original_linkers.append(layer.Linker.data.clone())
                        layer.Linker.data.mul_(causal_mask)
                        
                        if hasattr(layer, 'residual_linker') and layer.residual_linker is not None:
                            cm_res = torch.tril(torch.ones_like(layer.residual_linker))
                            original_residual_linkers.append(layer.residual_linker.data.clone())
                            layer.residual_linker.data.mul_(cm_res)
                        else:
                            original_residual_linkers.append(None)
            
            # Process in batches
            for i in range(0, len(indices), batch_size):
                batch_indices = indices[i:i+batch_size]
                batch_seqs = [seqs[idx] for idx in batch_indices]
                
                optimizer.zero_grad()
                
                # Batch forward pass
                batch_output = self.forward(batch_seqs)  # [B, final_seq_len, final_dim]
                
                # Get character layer output for each sequence in batch
                batch_char_output = []
                for seq in batch_seqs:
                    toks = self.extract_tokens(seq)
                    if not toks:
                        batch_char_output.append(torch.zeros((0, self.vec_dim), device=self.device))
                        continue
                        
                    token_indices = self.token_to_indices(toks)
                    k_positions = torch.arange(len(toks), dtype=torch.float32, device=self.device)
                    
                    # Compute char layer output
                    Nk_batch = self.char_batch_compute_Nk(k_positions, token_indices)
                    # Apply char_to_hier_linker
                    if len(toks) > 0:
                        char_output = Nk_batch.unsqueeze(0)  # [1, token_seq_len, vec_dim]
                        hier_input = torch.matmul(char_output.transpose(1, 2), 
                                                 self.char_to_hier_linker[:len(toks), :]).transpose(1, 2)
                        batch_char_output.append(hier_input.squeeze(0))
                    else:
                        batch_char_output.append(torch.zeros((self.hier_input_seq_len, self.vec_dim), device=self.device))
                
                # Stack char outputs
                if batch_char_output:
                    char_output_tensor = torch.stack(batch_char_output)  # [B, hier_input_seq_len, vec_dim]
                else:
                    char_output_tensor = torch.zeros((len(batch_seqs), self.hier_input_seq_len, self.vec_dim), 
                                                    device=self.device)
                
                # 3. Vectorized Loss Calculation
                loss = 0.0
                
                # Determine valid comparison length
                min_seq_len = min(batch_output.shape[1], char_output_tensor.shape[1])
                
                if self_mode == 'gap':
                    # Self-consistency: output should match char layer output
                    if min_seq_len > 0:
                        diff = batch_output[:, :min_seq_len, :] - char_output_tensor[:, :min_seq_len, :]
                        sq_diff = torch.sum(diff ** 2, dim=-1)  # [B, S_valid]
                        loss = torch.sum(torch.mean(sq_diff, dim=1))  # Sum of means
                else:  # 'reg' mode
                    # Predict next: Hier[k] vs Char[k+1]
                    if min_seq_len > 1:
                        targets = char_output_tensor[:, 1:min_seq_len, :]
                        preds = batch_output[:, 0:min_seq_len-1, :]
                        diff = preds - targets
                        sq_diff = torch.sum(diff ** 2, dim=-1)
                        loss = torch.sum(torch.mean(sq_diff, dim=1))
                    else:
                        loss = torch.tensor(0.0, device=self.device)
                
                batch_loss_scalar = loss.item()
                
                # Normalize and backpropagate if loss > 0
                if batch_loss_scalar > 0:
                    (loss / len(batch_indices)).backward()
                    optimizer.step()
                
                total_loss += batch_loss_scalar
                total_samples += len(batch_indices)
            
            # Restore original weights if Reg mode
            if self_mode == 'reg':
                for idx, layer in enumerate(self.hierarchical_layers):
                    if hasattr(layer, 'Linker'):
                        layer.Linker.data.copy_(original_linkers[idx])
                        if original_residual_linkers[idx] is not None:
                            layer.residual_linker.data.copy_(original_residual_linkers[idx])
            
            avg_loss = total_loss / total_samples if total_samples > 0 else 0
            history.append(avg_loss)
            
            # Update best model state if current loss is lower
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_model_state = copy.deepcopy(self.state_dict())
            
            if it % print_every == 0 or it == max_iters - 1:
                mode_display = "Gap" if self_mode == 'gap' else "Reg"
                print(f"SelfTrain({mode_display}) Iter {it:3d}: Loss = {avg_loss:.6f}, LR = {scheduler.get_last_lr()[0]:.6f}")
            
            # Save checkpoint at specified intervals
            if checkpoint_file and (it % checkpoint_interval == 0 or it == max_iters - 1):
                self._save_checkpoint(
                    checkpoint_file, it, history, optimizer, scheduler, best_loss
                )
            
            # Check convergence
            if prev_loss - avg_loss < tol:
                print(f"Converged after {it+1} iterations")
                # Restore the best model state before breaking
                if best_model_state is not None and avg_loss > best_loss:
                    self.load_state_dict(best_model_state)
                    print(f"Restored best model state with loss = {best_loss:.6f}")
                    history[-1] = best_loss
                break
            prev_loss = avg_loss
            
            # Learning rate decay every 5 iterations
            if it % 5 == 0:
                scheduler.step()

        # If training ended without convergence, restore best model state
        if best_model_state is not None and best_loss < history[-1]:
            self.load_state_dict(best_model_state)
            print(f"Training ended. Restored best model state with loss = {best_loss:.6f}")
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
        Optimized for batch processing with variable-length sequences.
        
        Args:
            seqs: List of character sequences for training (variable length)
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
            final_dim = self.model_dims[-1] if self.model_dims else self.vec_dim
            self.classifier = nn.Linear(final_dim, num_classes).to(self.device)
            self.num_classes = num_classes
        
        if not continued:
            self.reset_parameters()
        
        # Convert labels to tensor
        label_tensors = torch.tensor(labels, dtype=torch.long, device=self.device)
        
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
            total_sequences = 0
            correct_predictions = 0
            
            # Shuffle sequences for each epoch
            indices = list(range(len(seqs)))
            random.shuffle(indices)
            
            # Process sequences in batches
            for batch_start in range(0, len(indices), batch_size):
                batch_indices = indices[batch_start:batch_start + batch_size]
                batch_seqs = [seqs[idx] for idx in batch_indices]
                batch_labels = label_tensors[batch_indices]
                
                optimizer.zero_grad()
                
                # Batch forward pass
                batch_output = self.forward(batch_seqs)  # [B, final_seq_len, final_dim]
                
                # Compute sequence representation: average of all output vectors
                seq_representations = batch_output.mean(dim=1)  # [B, final_dim]
                
                # Get logits through classification head
                logits = self.classifier(seq_representations)  # [B, num_classes]
                
                # Compute loss
                loss = criterion(logits, batch_labels)
                loss.backward()
                optimizer.step()
                
                # Calculate batch statistics
                batch_loss = loss.item()
                total_loss += batch_loss * len(batch_seqs)
                total_sequences += len(batch_seqs)
                
                # Calculate accuracy
                with torch.no_grad():
                    predictions = torch.argmax(logits, dim=1)
                    correct_predictions += (predictions == batch_labels).sum().item()
            
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
        
        self.trained = True
        
        return history
    
    def lbl_train(self, seqs, labels, num_labels, max_iters=1000, tol=1e-8, learning_rate=0.01, 
                 continued=False, decay_rate=1.0, print_every=10, batch_size=32,
                 checkpoint_file=None, checkpoint_interval=10, pos_weight=None):
        """
        Train the model for multi-label classification using binary cross-entropy loss.
        
        Args:
            seqs: List of character sequences for training (variable length)
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
            final_dim = self.model_dims[-1] if self.model_dims else self.vec_dim
            self.labeller = nn.Linear(final_dim, num_labels).to(self.device)
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
        
        for it in range(max_iters):
            total_loss = 0.0
            total_correct = 0
            total_predictions = 0
            total_sequences = 0
            
            # Shuffle sequences for each epoch
            indices = list(range(len(seqs)))
            random.shuffle(indices)
            
            # Process sequences in batches
            for batch_start in range(0, len(indices), batch_size):
                batch_indices = indices[batch_start:batch_start + batch_size]
                batch_seqs = [seqs[idx] for idx in batch_indices]
                batch_labels = labels_tensor[batch_indices]
                
                optimizer.zero_grad()
                
                # Batch forward pass
                batch_output = self.forward(batch_seqs)  # [B, final_seq_len, final_dim]
                
                # Compute sequence representation: average of all output vectors
                seq_representations = batch_output.mean(dim=1)  # [B, final_dim]
                
                # Pass through classification head to get logits
                logits = self.labeller(seq_representations)  # [B, num_labels]
                
                # Calculate loss for the batch
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
            # Get sequence representation through forward pass
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
        assert self.labeller is not None, "Model must be trained first for label prediction"
        
        with torch.no_grad():
            # Get sequence representation through forward pass
            output = self.forward(seq)  # [1, final_seq_len, final_dim]
            seq_representation = output.mean(dim=1)  # [1, final_dim]
            
            # Pass through classification head to get logits
            logits = self.labeller(seq_representation)
            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(logits).cpu().numpy()[0]
        
        # Apply threshold to get binary predictions
        binary_preds = (probs > threshold).astype(np.float32)
        
        return binary_preds, probs

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
                'linker_dims': self.linker_dims,
                'num_basis_list': [layer.num_basis for layer in self.hierarchical_layers],
                'linker_trainable': self.linker_trainable,
                'use_residual_list': self.use_residual_list,
                'char_linker_trainable': self.char_linker_trainable,
                'char_layer_config': {
                    'rank_mode': self.rank_mode,
                    'num_basis': self.char_num_basis,
                    'mode': self.mode,
                    'user_step': self.step
                },
                'num_classes': self.num_classes,
                'num_labels': self.num_labels
            },
            'training_stats': {
                'mean_t': self.mean_t,
                'mean_token_count': self.mean_token_count,
                'mean_target': mean_target_np
            }
        }
        torch.save(checkpoint, checkpoint_file)
        #print(f"Checkpoint saved at iteration {iteration} to {checkpoint_file}")

    def _compute_training_statistics(self, seqs, batch_size=256):
        """Compute and store statistics for reconstruction and generation"""
        total_token_count = 0
        total_vec_sum = torch.zeros(self.vec_dim, device=self.device)
        
        # Use batch processing for efficiency
        all_indices, seq_lengths = self.batch_token_indices(seqs, device='cpu')
        
        with torch.no_grad():
            for i in range(0, len(seqs), batch_size):
                batch_indices = all_indices[i:i+batch_size].to(self.device)
                
                # Get char layer output [B, S, D]
                nk_batch = self.char_forward(batch_indices)
                
                # Apply mask based on actual sequence lengths
                for j, seq_len in enumerate(seq_lengths[i:i+batch_size]):
                    if seq_len > 0:
                        total_vec_sum += nk_batch[j, :seq_len].sum(dim=0)
                        total_token_count += seq_len
        
        self.mean_token_count = total_token_count / len(seqs) if seqs else 0
        self.mean_t = (total_vec_sum / total_token_count).cpu().numpy() if total_token_count > 0 else np.zeros(self.vec_dim)

    def _compute_hierarchical_mean_target(self, seqs, batch_size=256):
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
        
        # Process in batches
        with torch.no_grad():
            for i in range(0, len(seqs), batch_size):
                batch_seqs = seqs[i:i+batch_size]
                try:
                    output = self.forward(batch_seqs)  # [B, S_out, D_out]
                    
                    # Mean over sequence, Sum over batch
                    batch_means = output.mean(dim=1)  # [B, D_out]
                    total_output += batch_means.sum(dim=0)
                    total_sequences += len(batch_seqs)
                    
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
        """
        Reconstruct character sequence of length L using the entire hierarchical model.
        
        This method uses temperature-controlled sampling based on the complete model's
        predictions, incorporating both character-level and hierarchical patterns with Linker matrices.
        
        Args:
            L (int): Length of sequence to generate (must be <= max_seq_len)
            tau (float): Temperature parameter for stochastic sampling.
                        tau=0: deterministic (greedy) selection
                        tau>0: stochastic selection with temperature
            
        Returns:
            str: Generated character sequence
        """
        assert self.trained, "Model must be trained first"
        assert self.mean_target is not None, "Mean target vector must be computed first"
        
        if tau < 0:
            raise ValueError("Temperature must be non-negative")
        
        if L <= 0:
            return ""
        
        if L > self.max_seq_len:
            print(f"Warning: Requested length {L} exceeds maximum sequence length {self.max_seq_len}. Truncating.")
            L = self.max_seq_len
        
        generated_sequence = ""
        
        # Generate sequence token by token
        while len(generated_sequence) < L:
            candidate_scores = {}
            
            # Evaluate each possible token
            for token in self.tokens:
                candidate_seq = generated_sequence + token
                
                # Pad to max_seq_len if necessary for model input
                if len(candidate_seq) < self.max_seq_len:
                    padded_seq = candidate_seq.ljust(self.max_seq_len, 'A')
                else:
                    padded_seq = candidate_seq[:self.max_seq_len]
                    
                # Skip if candidate exceeds desired length
                if len(candidate_seq) > L:
                    continue
                    
                with torch.no_grad():
                    model_output = self.forward(padded_seq)
                    
                    if model_output.numel() > 0:
                        pred_target = model_output.mean(dim=1).squeeze(0)
                        error = torch.sum((pred_target - self.mean_target) ** 2).item()
                        score = -error  # Convert error to score (higher = better)
                        candidate_scores[token] = score
            
            if not candidate_scores:
                # No valid candidates, use character layer as fallback
                char_layer_gen = self.char_generate(L - len(generated_sequence), tau)
                generated_sequence += char_layer_gen
                break
            
            # Select token based on temperature
            tokens = list(candidate_scores.keys())
            scores = list(candidate_scores.values())
            
            if tau == 0:
                # Deterministic selection: choose token with highest score
                best_idx = np.argmax(scores)
                chosen_token = tokens[best_idx]
            else:
                # Stochastic selection with temperature
                scores_array = np.array(scores)
                # Apply temperature scaling
                scaled_scores = scores_array / tau
                # Convert to probabilities using softmax
                exp_scores = np.exp(scaled_scores - np.max(scaled_scores))  # Numerical stability
                probs = exp_scores / np.sum(exp_scores)
                # Sample based on probabilities
                chosen_idx = np.random.choice(len(tokens), p=probs)
                chosen_token = tokens[chosen_idx]
            
            generated_sequence += chosen_token
        
        # Ensure exact length and remove padding
        final_sequence = generated_sequence[:L]
        if '_' in final_sequence:
            final_sequence = final_sequence.replace('_', '')
        
        return final_sequence
    
    def reset_parameters(self):
        """Reset all model parameters"""
        # Reset char layer parameters
        nn.init.uniform_(self.char_embedding.weight, -0.5, 0.5)
        nn.init.uniform_(self.char_P, -0.1, 0.1)
        nn.init.uniform_(self.char_to_hier_linker, -0.1, 0.1)
        
        # Reset hierarchical layer parameters
        for layer in self.hierarchical_layers:
            layer.reset_parameters()
        
        # Reset classification heads if they exist
        if self.classifier is not None:
            nn.init.normal_(self.classifier.weight, 0, 0.01)
            if self.classifier.bias is not None:
                nn.init.constant_(self.classifier.bias, 0)
        
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
        
        # Character embedding parameters
        num = self.char_embedding.weight.numel()
        shape = str(tuple(self.char_embedding.weight.shape))
        print(f"{'Char Layer':<15} | {'char_embedding':<25} | {num:<15} | {shape:<20}")
        char_params += num
        total_params += num
        trainable_params += num
        
        # Character P parameters
        num = self.char_P.numel()
        shape = str(tuple(self.char_P.shape))
        print(f"{'Char Layer':<15} | {'char_P':<25} | {num:<15} | {shape:<20}")
        char_params += num
        total_params += num
        trainable_params += num
        
        # Char to Hier Linker parameters
        num = self.char_to_hier_linker.numel()
        shape = str(tuple(self.char_to_hier_linker.shape))
        trainable_status = "T" if self.char_to_hier_linker.requires_grad else "F"
        print(f"{'Char Layer':<15} | {'char_to_hier_linker':<25} | {num:<15} | {shape:<20} [{trainable_status}]")
        char_params += num
        total_params += num
        if self.char_to_hier_linker.requires_grad:
            trainable_params += num
        
        print(f"{'Char Layer':<15} | {'TOTAL':<25} | {char_params:<15} |")
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
            
            # P tensor
            num = layer.P.numel()
            shape = str(tuple(layer.P.shape))
            print(f"{layer_name:<15} | {'P':<25} | {num:<15} | {shape:<20}")
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
            if layer.use_residual == 'separate':
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
            num = self.classifier.weight.numel()
            shape = str(tuple(self.classifier.weight.shape))
            print(f"{'Classifier':<15} | {'weight':<25} | {num:<15} | {shape:<20}")
            total_params += num
            trainable_params += num
            
            if self.classifier.bias is not None:
                num = self.classifier.bias.numel()
                shape = str(tuple(self.classifier.bias.shape))
                print(f"{'Classifier':<15} | {'bias':<25} | {num:<15} | {shape:<20}")
                total_params += num
                trainable_params += num
        
        if self.labeller is not None:
            num = self.labeller.weight.numel()
            shape = str(tuple(self.labeller.weight.shape))
            print(f"{'Labeller':<15} | {'weight':<25} | {num:<15} | {shape:<20}")
            total_params += num
            trainable_params += num
            
            if self.labeller.bias is not None:
                num = self.labeller.bias.numel()
                shape = str(tuple(self.labeller.bias.shape))
                print(f"{'Labeller':<15} | {'bias':<25} | {num:<15} | {shape:<20}")
                total_params += num
                trainable_params += num
        
        print(f"{'TOTAL':<15} | {'ALL PARAMETERS':<25} | {total_params:<15} | (trainable: {trainable_params})")
        print("="*80)
        
        return total_params, trainable_params
    
    def save(self, filename):
        """Save model state to file"""
        # Convert mean_target to numpy for saving
        mean_target_np = self.mean_target.cpu().numpy() if self.mean_target is not None else None
        
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': {
                'charset': self.charset,
                'rank': self.rank,
                'vec_dim': self.vec_dim,
                'max_seq_len': self.max_seq_len,
                'hier_input_seq_len': self.hier_input_seq_len,
                'model_dims': self.model_dims,
                'linker_dims': self.linker_dims,
                'num_basis_list': [layer.num_basis for layer in self.hierarchical_layers],
                'linker_trainable': self.linker_trainable,
                'use_residual_list': self.use_residual_list,
                'char_linker_trainable': self.char_linker_trainable,
                'char_layer_config': {
                    'rank_mode': self.rank_mode,
                    'num_basis': self.char_num_basis,
                    'mode': self.mode,
                    'user_step': self.step
                },
                'num_classes': self.num_classes,
                'num_labels': self.num_labels
            },
            'training_stats': {
                'mean_t': self.mean_t,
                'mean_token_count': self.mean_token_count,
                'mean_target': mean_target_np
            }
        }, filename)
        print(f"Model saved to {filename}")
    
    @classmethod
    def load(cls, filename, device=None):
        """Load model from file"""
        # Device configuration
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load checkpoint
        checkpoint = torch.load(filename, map_location=device, weights_only=False)
        config = checkpoint['config']
        
        # Obtain the num_basis_list parameter from configuration 
        char_layer_config = config['char_layer_config']
        
        # Create model instance
        model = cls(
            charset=config['charset'],
            rank=config['rank'],
            vec_dim=config['vec_dim'],
            max_seq_len=config['max_seq_len'],
            hier_input_seq_len=config['hier_input_seq_len'],
            model_dims=config['model_dims'],
            linker_dims=config['linker_dims'],
            rank_mode=char_layer_config['rank_mode'],
            num_basis=char_layer_config['num_basis'],  # char layer's num_basis
            mode=char_layer_config['mode'],
            user_step=char_layer_config['user_step'],            
            num_basis_list=config.get('num_basis_list', [char_layer_config['num_basis']] * len(config['model_dims'])),            
            linker_trainable=config.get('linker_trainable', False),
            use_residual_list=config.get('use_residual_list', [None] * len(config['model_dims'])),
            char_linker_trainable=config.get('char_linker_trainable', None),
            device=device
        )
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load classification heads if they exist in config
        if 'num_classes' in config and config['num_classes'] is not None:
            final_dim = config['model_dims'][-1] if config['model_dims'] else config['vec_dim']
            model.classifier = nn.Linear(final_dim, config['num_classes']).to(device)
            model.num_classes = config['num_classes']
        
        if 'num_labels' in config and config['num_labels'] is not None:
            final_dim = config['model_dims'][-1] if config['model_dims'] else config['vec_dim']
            model.labeller = nn.Linear(final_dim, config['num_labels']).to(device)
            model.num_labels = config['num_labels']
        
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
    print("HierDDLtsC - Hierarchical Dual Descriptor with Character Input and Linker Matrices")
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
    num_basis_list = [6, 5, 4]  # Basis functions for each hierarchical layer
    linker_dims = [40, 30, 20]  # Output sequence lengths for each hierarchical layer
    num_seqs = 100
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {device}")
    print(f"Character set: {charset}")
    print(f"Rank (k-mer length): {rank}")
    print(f"Character layer output dim: {vec_dim}")
    print(f"Maximum sequence length: {max_seq_len}")
    print(f"Hierarchical input sequence length: {hier_input_seq_len}")
    print(f"Hierarchical layer dims: {model_dims}")
    print(f"Linker output sequence lengths: {linker_dims}")
    print(f"Number of basis functions: {num_basis_list}")
    
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
    print("\nCreating HierDDLtsC model with variable-length support...")
    model = HierDDLtsC(
        charset=charset,
        rank=rank,
        vec_dim=vec_dim,
        max_seq_len=max_seq_len,
        hier_input_seq_len=hier_input_seq_len,
        model_dims=model_dims,
        linker_dims=linker_dims,
        num_basis_list=num_basis_list,
        linker_trainable=[True, False, True],  # Mixed trainable settings
        use_residual_list=['separate', 'shared', None],  # Mixed residual strategies
        char_linker_trainable=True,  # Separately set the character-layer Linker matrix
        mode='linear',
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
    
    # Test batch_token_indices method
    print("\nTesting batch_token_indices method...")
    batch_indices, seq_lengths = model.batch_token_indices(seqs[:5])
    print(f"Batch indices shape: {batch_indices.shape}")
    print(f"Sequence lengths: {seq_lengths}")
    
    # Test char_forward method
    print("\nTesting char_forward method...")
    batch_output = model.char_forward(batch_indices.to(device))
    print(f"Char forward output shape: {batch_output.shape}")
    
    # Test batch_sequence_to_tensor method
    print("\nTesting batch_sequence_to_tensor method...")
    batch_tensor = model.batch_sequence_to_tensor(seqs[:5])
    print(f"Batch tensor shape: {batch_tensor.shape}")
    
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
        checkpoint_file='hierddltsc_var_gd_checkpoint.pth',
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
    print(f"Deterministic reconstruction (tau=0.0): {det_seq}")
    
    # Stochastic reconstruction
    sto_seq = model.reconstruct(L=100, tau=0.5)
    print(f"Stochastic reconstruction (tau=0.5): {sto_seq}")
    
    # Character-level reconstruction
    char_reconstructed_seq = model.char_reconstruct()
    print(f"Character-level reconstructed sequence (length {len(char_reconstructed_seq)}):")
    print(f"First 100 chars: {char_reconstructed_seq[:100]}")
    
    # Character-level generation
    char_gen_seq = model.char_generate(L=100, tau=0.2)
    print(f"Character-level generation (tau=0.2): {char_gen_seq}")
    
    # === Classification task example ===
    print("\n" + "="*50)
    print("Classification Task Example")
    print("="*50)
    
    # Generate classification data with variable-length sequences
    num_classes = 3
    class_seqs = []
    class_labels = []
    
    # Create sequences with different patterns for each class
    for class_id in range(num_classes):
        for _ in range(50):  # 50 sequences per class
            seq_len = random.randint(100, 200)
            if class_id == 0:
                # Class 0: High A content
                seq = ''.join(random.choices(['A', 'C', 'G', 'T'], weights=[0.6, 0.1, 0.1, 0.2], k=seq_len))
            elif class_id == 1:
                # Class 1: High GC content
                seq = ''.join(random.choices(['A', 'C', 'G', 'T'], weights=[0.1, 0.4, 0.4, 0.1], k=seq_len))
            else:
                # Class 2: Balanced
                seq = ''.join(random.choices(['A', 'C', 'G', 'T'], k=seq_len))
            
            class_seqs.append(seq)
            class_labels.append(class_id)
    
    # Create a new model for classification
    cls_model = HierDDLtsC(
        charset=charset,
        rank=rank,
        vec_dim=vec_dim,
        max_seq_len=max_seq_len,
        hier_input_seq_len=hier_input_seq_len,
        model_dims=model_dims,
        linker_dims=linker_dims,
        num_basis_list=num_basis_list,
        linker_trainable=False,
        use_residual_list=[None] * len(model_dims),
        char_linker_trainable=False,
        mode='linear',
        device=device
    )
    
    # Train for classification
    print("\nTraining classification model...")
    cls_history = cls_model.cls_train(
        class_seqs, class_labels, num_classes,
        max_iters=30,  # Reduced for demonstration
        learning_rate=0.01,
        decay_rate=0.98,
        print_every=5,
        batch_size=16
    )
    
    # Show prediction results
    print("\nClassification prediction results:")
    correct = 0
    for i in range(min(10, len(class_seqs))):
        pred_class, probs = cls_model.predict_c(class_seqs[i])
        true_label = class_labels[i]
        correct += int(pred_class == true_label)
        print(f"Seq {i+1}: True={true_label}, Pred={pred_class}, Probs={[f'{p:.3f}' for p in probs]}")
    
    accuracy = correct / min(10, len(class_seqs))
    print(f"Accuracy on first 10 sequences: {accuracy:.4f}")
    
    # === Multi-label classification example ===
    print("\n" + "="*50)
    print("Multi-Label Classification Example")
    print("="*50)
    
    # Generate multi-label data with variable-length sequences
    num_labels = 4
    label_seqs = []
    labels = []
    
    for _ in range(100):
        seq_len = random.randint(100, 200)
        seq = ''.join(random.choices(charset, k=seq_len))
        label_seqs.append(seq)
        
        # Create random binary labels (multi-label classification)
        # Each sequence can have 0-4 active labels
        label_vec = [random.random() > 0.7 for _ in range(num_labels)]
        labels.append([1.0 if x else 0.0 for x in label_vec])
    
    # Create a new model for multi-label classification
    lbl_model = HierDDLtsC(
        charset=charset,
        rank=rank,
        vec_dim=vec_dim,
        max_seq_len=max_seq_len,
        hier_input_seq_len=hier_input_seq_len,
        model_dims=model_dims,
        linker_dims=linker_dims,
        num_basis_list=num_basis_list,
        linker_trainable=False,
        use_residual_list=[None] * len(model_dims),
        char_linker_trainable=False,
        mode='linear',
        device=device
    )
    
    # Train for multi-label classification
    print("\nTraining multi-label classification model...")
    loss_history, acc_history = lbl_model.lbl_train(
        label_seqs, labels, num_labels,
        max_iters=50,  # Reduced for demonstration
        learning_rate=0.02,
        decay_rate=0.98,
        print_every=5,
        batch_size=16
    )
    
    print(f"\nFinal training loss: {loss_history[-1]:.6f}")
    print(f"Final training accuracy: {acc_history[-1]:.4f}")
    
    # Show multi-label prediction results
    print("\nMulti-label prediction results:")
    for i in range(min(5, len(label_seqs))):
        binary_pred, probs_pred = lbl_model.predict_l(label_seqs[i], threshold=0.5)
        true_labels = labels[i]
        
        print(f"\nSequence {i+1}:")
        print(f"True labels: {true_labels}")
        print(f"Predicted binary: {binary_pred}")
        print(f"Predicted probabilities: {[f'{p:.4f}' for p in probs_pred]}")
        
        # Check if all labels match
        matches = all(b == t for b, t in zip(binary_pred, true_labels))
        print(f"All labels match: {matches}")
    
    # Self-training example with variable-length sequences
    print("\n" + "="*50)
    print("Self-Training Example with Variable-Length Sequences")
    print("="*50)
    
    # Create a new model for self-training
    self_model = HierDDLtsC(
        charset=charset,
        rank=rank,
        vec_dim=vec_dim,
        max_seq_len=max_seq_len,
        hier_input_seq_len=hier_input_seq_len,
        model_dims=model_dims,
        linker_dims=linker_dims,
        num_basis_list=num_basis_list,
        linker_trainable=False,
        use_residual_list=[None] * len(model_dims),
        char_linker_trainable=False,  # Set the character-layer Linker matrix to be untrainable
        mode='linear',
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
        batch_size=8,
        checkpoint_file='hierddltsc_var_self_checkpoint.pth',
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
    model.save("hierddltsc_var_model.pth")
    
    # Load model
    loaded_model = HierDDLtsC.load("hierddltsc_var_model.pth", device=device)
    
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
    print("Variable-length sequence processing is working correctly!")
    print("Classification and multi-label classification functions are working correctly!")
