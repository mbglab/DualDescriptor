# Copyright (C) 2005-2025, Bin-Guang Ma (mbg@mail.hzau.edu.cn); SPDX-License-Identifier: MIT
# Hierarchical Dual Descriptor (AB matrix form) with Character Sequence Input (HierDDLabC)
# Combines character-level processing and hierarchical vector processing with Linker matrices
# Modified to support variable-length sequences with adaptive Linker matrices
# This program is for the demonstration of methodology and not fully refined.
# Author: Bin-Guang Ma (assisted by DeepSeek); Date: 2025-11-30 ~ 2026-1-15

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
    Single layer of Hierarchical Dual Descriptor with Linker matrix form
    Uses the Linker matrix architecture from HierDDLab with character sequence input
    Modified to support variable-length sequences with adaptive Linker matrices
    """
    def __init__(self, in_dim, out_dim, basis_dim, max_in_seq, max_out_seq, 
                 linker_trainable, use_residual, device='cuda', reduction_factor=2):
        """
        Initialize a hierarchical layer with Linker matrix form for variable-length sequences
        
        Args:
            in_dim (int): Input dimension
            out_dim (int): Output dimension
            basis_dim (int): Basis dimension for this layer
            max_in_seq (int): Maximum input sequence length in the dataset
            max_out_seq (int): Maximum output sequence length
            linker_trainable (bool): Whether linker matrix is trainable
            use_residual (str or None): Residual connection type
            device (str): Computation device
            reduction_factor (int): Factor by which to reduce sequence length (default: 2)
        """
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.basis_dim = basis_dim
        self.max_in_seq = max_in_seq
        self.max_out_seq = max_out_seq
        self.use_residual = use_residual
        self.linker_trainable = linker_trainable
        self.device = device
        self.reduction_factor = reduction_factor
        
        # Linear transformation matrix M: out_dim x in_dim
        self.M = nn.Parameter(torch.Tensor(out_dim, in_dim))
        
        # Coefficient matrix: out_dim x basis_dim (learnable)
        self.Acoeff = nn.Parameter(torch.Tensor(out_dim, basis_dim))
        
        # Fixed basis matrix B (basis_dim x out_dim)
        Bbasis = torch.empty(basis_dim, out_dim)
        for k in range(basis_dim):
            for i in range(out_dim):
                Bbasis[k, i] = math.cos(2 * math.pi * (k+1) / (i+2))
        self.register_buffer('Bbasis', Bbasis)  # Fixed, non-trainable buffer
        
        # Adaptive Linker matrix for variable-length sequences: max_in_seq x max_out_seq
        self.Linker = nn.Parameter(torch.Tensor(max_in_seq, max_out_seq))
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(out_dim)
        
        # Initialize parameters
        nn.init.uniform_(self.M, -0.1, 0.1)
        nn.init.uniform_(self.Acoeff, -0.1, 0.1)
        nn.init.uniform_(self.Linker, -0.1, 0.1)
        
        # Freeze Linker if not trainable
        if not linker_trainable:
            self.Linker.requires_grad = False
            
        # Handle residual connection based on mode
        if use_residual == 'separate':
            # Separate linear projection for residual
            self.residual_proj = nn.Linear(in_dim, out_dim, bias=False)
            nn.init.uniform_(self.residual_proj.weight, -0.1, 0.1)
            
            # Separate linker for residual
            self.residual_linker = nn.Parameter(torch.Tensor(max_in_seq, max_out_seq))
            nn.init.uniform_(self.residual_linker, -0.1, 0.1)
            if not linker_trainable:
                self.residual_linker.requires_grad = False
        else:
            self.residual_proj = None
            self.residual_linker = None
        
        self.to(device)
    
    def adaptive_sequence_transform(self, u, actual_in_seq_len):
        """
        Transform sequence length using adaptive Linker matrix for variable-length sequences
        
        Args:
            u (Tensor): Input tensor of shape (batch_size, in_seq, out_dim)
            actual_in_seq_len (int): Actual sequence length for this batch
            
        Returns:
            Tensor: Transformed tensor of shape (batch_size, out_seq, out_dim)
        """
        batch_size, in_seq, out_dim = u.shape
        
        # Calculate output sequence length based on reduction factor
        actual_out_seq_len = max(1, actual_in_seq_len // self.reduction_factor)
        
        # Extract the relevant portion of the Linker matrix
        # Use first 'actual_in_seq_len' rows and first 'actual_out_seq_len' columns
        adaptive_linker = self.Linker[:actual_in_seq_len, :actual_out_seq_len]
        
        # Using matrix multiplication: u @ adaptive_linker
        # u: [batch_size, in_seq, out_dim] -> [batch_size, out_seq, out_dim]
        return torch.matmul(u.transpose(1, 2), adaptive_linker).transpose(1, 2)
    
    def forward(self, x, actual_seq_len=None):
        """
        Forward pass through the layer with adaptive Linker matrix form for variable-length sequences
        
        Args:
            x (torch.Tensor): Input sequence of shape (batch_size, in_seq_len, in_dim)
            actual_seq_len (int): Actual sequence length (if None, use x.shape[1])
        
        Returns:
            torch.Tensor: Output sequence of shape (batch_size, out_seq_len, out_dim)
            int: Actual output sequence length
        """
        batch_size, seq_len, _ = x.shape
        
        # Use provided actual sequence length or default to input sequence length
        if actual_seq_len is None:
            actual_seq_len = seq_len
        
        # Precompute basis indices for actual sequence positions
        basis_indices = torch.tensor([i % self.basis_dim for i in range(actual_seq_len)], device=self.device)
        
        # Main path processing
        # Apply linear transformation: (batch_size, in_seq_len, in_dim) -> (batch_size, in_seq_len, out_dim)
        z = torch.matmul(x, self.M.T)
        
        # Apply layer normalization (only on the actual sequence part)
        z_actual = z[:, :actual_seq_len, :]
        z_actual = self.layer_norm(z_actual)
        
        # Position-wise processing
        # Select basis vectors for each position: (actual_seq_len, out_dim)
        basis_vectors = self.Bbasis[basis_indices]
        
        # Compute scalars: dot product of z and basis_vectors
        scalars = torch.einsum('bsd,sd->bs', z_actual, basis_vectors)  # (batch_size, actual_seq_len)
        
        # Select coefficient vectors: (actual_seq_len, out_dim)
        coeffs = self.Acoeff[:, basis_indices].permute(1, 0)  # (actual_seq_len, out_dim)
        
        # Compute position outputs: scalars * coeffs
        u = coeffs * scalars.unsqueeze(-1)  # (batch_size, actual_seq_len, out_dim)
        
        # Apply adaptive sequence length transformation using Linker matrix
        main_output = self.adaptive_sequence_transform(u, actual_seq_len)
        
        # Calculate actual output sequence length
        actual_out_seq_len = main_output.shape[1]
        
        # Process residual connections
        residual = 0
        if self.use_residual == 'separate':
            # Separate projection and linker for residual
            residual_feat = self.residual_proj(x)  # (batch_size, in_seq_len, out_dim)
            residual_linker = self.residual_linker[:actual_seq_len, :actual_out_seq_len]
            residual = torch.matmul(
                residual_feat.permute(0, 2, 1), 
                residual_linker
            ).permute(0, 2, 1)  # (batch_size, out_seq_len, out_dim)
            
        elif self.use_residual == 'shared':
            # Shared M and Linker for residual
            residual_feat = torch.matmul(x, self.M.T)  # (batch_size, in_seq_len, out_dim)
            residual_linker = self.Linker[:actual_seq_len, :actual_out_seq_len]
            residual = torch.matmul(
                residual_feat.permute(0, 2, 1), 
                residual_linker
            ).permute(0, 2, 1)  # (batch_size, out_seq_len, out_dim)
            
        elif self.use_residual is None:
            # Identity residual only if dimensions match
            if self.in_dim == self.out_dim and actual_seq_len == actual_out_seq_len:
                residual_linker = self.Linker[:actual_seq_len, :actual_out_seq_len]
                residual = torch.matmul(
                    x.permute(0, 2, 1), 
                    residual_linker
                ).permute(0, 2, 1)  # (batch_size, out_seq_len, out_dim)
        
        # Add residual connection
        output = main_output + residual
        
        return output, actual_out_seq_len

class HierDDLabC(nn.Module):
    """
    Hierarchical Dual Descriptor with Linker Matrix and Character Sequence Input
    Combines character-level processing and hierarchical linker matrix layers
    Modified to support variable-length sequences with adaptive Linker matrices
    """
    def __init__(self, charset, rank=1, rank_mode='drop', vec_dim=2, 
                 mode='linear', user_step=None, max_input_seq_len=100,
                 model_dims=[2], basis_dims=[50], linker_dims=[50],
                 linker_trainable=False, use_residual_list=None, 
                 reduction_factors=None, device='cuda'):
        """
        Initialize HierDDLabC model with embedded character-level processing and adaptive Linker matrix layers
        
        Args:
            charset: Character set for sequence input
            rank: k-mer length for tokenization
            rank_mode: 'pad' or 'drop' for incomplete k-mers
            vec_dim: Output dimension of character layer (Layer 0)
            mode: 'linear' or 'nonlinear' tokenization
            user_step: Step size for nonlinear tokenization
            max_input_seq_len: Maximum input character sequence length in the dataset
            model_dims: Output dimensions for hierarchical layers
            basis_dims: Basis dimensions for hierarchical layers
            linker_dims: Maximum output sequence lengths for hierarchical layers
            linker_trainable: Whether linker matrices are trainable (bool or list)
            use_residual_list: Residual connection types for hierarchical layers
            reduction_factors: Sequence length reduction factors for each layer (default: 2 for all)
            device: Computation device
        """
        super().__init__()
        self.charset = list(charset)
        self.rank = rank    # r-per/k-mer length
        self.rank_mode = rank_mode # 'pad' or 'drop'
        self.vec_dim = vec_dim    # embedding dimension
        self.mode = mode
        self.step = user_step
        self.max_input_seq_len = max_input_seq_len  # Maximum input character sequence length
        self.model_dims = model_dims
        self.basis_dims = basis_dims
        self.linker_dims = linker_dims
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.trained = False
        self.num_layers = len(model_dims)
        
        # Process reduction_factors
        if reduction_factors is None:
            self.reduction_factors = [2] * self.num_layers  # Default: reduce by factor of 2
        elif isinstance(reduction_factors, list):
            if len(reduction_factors) != self.num_layers:
                raise ValueError("reduction_factors list length must match number of layers")
            self.reduction_factors = reduction_factors
        else:
            raise TypeError("reduction_factors must be list or None")        
        
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
        
        # Character layer AB matrix components
        self.char_basis_dim = 50  # Fixed basis dimension for character layer
        self.char_linear = nn.Linear(self.vec_dim, self.vec_dim, bias=False)
        self.char_Acoeff = nn.Parameter(torch.Tensor(self.vec_dim, self.char_basis_dim))
        
        # Fixed basis matrix for character layer
        char_Bbasis = torch.empty(self.char_basis_dim, self.vec_dim)
        for k in range(self.char_basis_dim):
            for i in range(self.vec_dim):
                char_Bbasis[k, i] = math.cos(2 * math.pi * (k+1) / (i+2))
        self.register_buffer('char_Bbasis', char_Bbasis)
        
        # Training statistics for character layer
        self.mean_token_count = None
        self.mean_t = None
        
        # Classification head for multi-class tasks (initialized when needed)
        self.num_classes = None
        self.classifier = None
        
        # Label head for multi-label tasks (initialized when needed)
        self.num_labels = None
        self.labeller = None
        
        # Calculate maximum token sequence length from maximum character sequence length
        self.max_token_seq_len = self._calculate_token_seq_len(max_input_seq_len)
        
        # Process linker_trainable parameter
        if isinstance(linker_trainable, bool):
            self.linker_trainable = [linker_trainable] * len(model_dims)
        elif isinstance(linker_trainable, list):
            if len(linker_trainable) != len(model_dims):
                raise ValueError("linker_trainable list length must match model_dims length")
            self.linker_trainable = linker_trainable
        else:
            raise TypeError("linker_trainable must be bool or list of bools")
            
        # Process use_residual parameter
        if use_residual_list is None:
            self.use_residual_list = [None] * len(model_dims)
        elif isinstance(use_residual_list, list):
            if len(use_residual_list) != len(model_dims):
                raise ValueError("use_residual_list length must match model_dims length")
            self.use_residual_list = use_residual_list
        else:
            raise TypeError("use_residual_list must be list of str/None")
        
        # Validate dimensions
        if len(basis_dims) != len(model_dims):
            raise ValueError("basis_dims length must match model_dims length")
        if len(linker_dims) != len(model_dims):
            raise ValueError("linker_dims length must match model_dims length")
        
        # Hierarchical layers with adaptive Linker matrices for variable-length sequences (starting from Layer 1)
        self.hierarchical_layers = nn.ModuleList()
        in_dim = vec_dim  # Input to first hierarchical layer is output from char layer
        max_in_seq = self.max_token_seq_len  # Maximum input sequence length for first hierarchical layer
        
        for i, (out_dim, basis_dim, max_out_seq) in enumerate(zip(model_dims, basis_dims, linker_dims)):
            layer = Layer(
                in_dim=in_dim,
                out_dim=out_dim,
                basis_dim=basis_dim,
                max_in_seq=max_in_seq,
                max_out_seq=max_out_seq,
                linker_trainable=self.linker_trainable[i],
                use_residual=self.use_residual_list[i],
                device=self.device,
                reduction_factor=self.reduction_factors[i]
            )
            self.hierarchical_layers.append(layer)
            in_dim = out_dim  # Next layer input is current layer output
            max_in_seq = max_out_seq  # Next layer maximum input sequence length is current layer maximum output sequence length
        
        # Mean target vector for the final hierarchical layer output
        self.mean_target = None
        
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

    def char_batch_compute_Nk(self, token_indices):
        """
        Vectorized computation of N(k) vectors for a batch of tokens using AB matrix form
        Args:
            token_indices: Tensor of token indices [batch_size]
        Returns:
            Tensor of N(k) vectors [batch_size, vec_dim]
        """
        batch_size = token_indices.shape[0]
        
        # Get embeddings for all tokens [batch_size, vec_dim]
        x = self.char_embedding(token_indices)
        
        # Linear transformation
        transformed = self.char_linear(x)
        
        # Apply layer normalization
        normalized = torch.nn.functional.layer_norm(transformed, (self.vec_dim,))
        
        # Compute basis indices: j = k % basis_dim for each position
        seq_len = batch_size  # Each token is at a different position
        j_indices = torch.tensor([i % self.char_basis_dim for i in range(seq_len)], device=self.device)
        
        # Select basis vectors: (seq_len, vec_dim)
        B_vectors = self.char_Bbasis[j_indices]
        
        # Compute scalars: dot product of normalized and B_vectors
        scalars = torch.einsum('bd,bd->b', normalized, B_vectors)
        
        # Select coefficient vectors: (seq_len, vec_dim)
        A_vectors = self.char_Acoeff[:, j_indices].t()
        
        # Compute new features: scalars * A_vectors
        Nk = scalars.unsqueeze(1) * A_vectors
            
        return Nk

    def char_describe(self, seq):
        """Compute N(k) vectors for each k-mer in sequence (character-level processing)"""
        toks = self.extract_tokens(seq)
        if not toks:
            return []
        
        token_indices = self.token_to_indices(toks)
        
        # Batch compute all Nk vectors
        N_batch = self.char_batch_compute_Nk(token_indices)
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
            # We need to simulate the position for each token
            # For AB matrix, position affects basis selection via j = k % basis_dim
            # So we create a batch with the same position for all tokens
            position_specific_indices = all_token_indices  # All tokens at this position
            
            # Compute Nk for all tokens at position k
            Nk_all = self.char_batch_compute_Nk(position_specific_indices)
            
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
            position_specific_indices = all_token_indices
            Nk_all = self.char_batch_compute_Nk(position_specific_indices)
            
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
        
        Args:
            seq (str): Input character sequence (can be variable length, up to max_input_seq_len)
            
        Returns:
            Tensor: Sequence tensor of shape [1, token_seq_len, vec_dim]
            int: Actual token sequence length
        """
        # Validate sequence length doesn't exceed maximum
        if len(seq) > self.max_input_seq_len:
            raise ValueError(f"Input sequence length {len(seq)} exceeds maximum allowed {self.max_input_seq_len}")
        
        toks = self.extract_tokens(seq)
        if not toks:
            return torch.zeros((1, 0, self.vec_dim), device=self.device), 0
        
        actual_token_seq_len = len(toks)
        
        token_indices = self.token_to_indices(toks)
        
        # Compute N(k) vectors through char layer
        Nk_batch = self.char_batch_compute_Nk(token_indices)
        
        # Add batch dimension
        return Nk_batch.unsqueeze(0), actual_token_seq_len  # [1, token_seq_len, vec_dim], actual_length
    
    def forward(self, seq):
        """
        Forward pass through entire hierarchical model with adaptive Linker matrices for variable-length sequences
        
        Args:
            seq (str or list): Input character sequence(s) of variable length (up to max_input_seq_len)
            
        Returns:
            Tensor: Output of shape [batch_size, final_seq_len, final_dim]
        """
        if isinstance(seq, str):
            # Single sequence
            x, actual_seq_len = self.char_sequence_to_tensor(seq)
            batch_actual_seq_lens = [actual_seq_len]
        else:
            # Multiple sequences - process each separately and stack
            batch_tensors = []
            batch_actual_seq_lens = []
            for s in seq:
                tensor_seq, actual_len = self.char_sequence_to_tensor(s)
                batch_tensors.append(tensor_seq.squeeze(0))
                batch_actual_seq_lens.append(actual_len)
            x = torch.stack(batch_tensors)  # [batch_size, token_seq_len, vec_dim]
        
        # Pass through hierarchical layers with adaptive Linker matrices
        current_actual_seq_lens = batch_actual_seq_lens
        for layer in self.hierarchical_layers:
            # Process each sequence in the batch with its actual length
            if isinstance(seq, str):
                # Single sequence
                x, current_actual_seq_lens = layer(x, current_actual_seq_lens[0])
                current_actual_seq_lens = [current_actual_seq_lens]
            else:
                # Batch processing - we need to handle each sequence separately due to variable lengths
                batch_outputs = []
                new_actual_seq_lens = []
                for i in range(x.shape[0]):
                    seq_tensor = x[i:i+1]  # Get single sequence [1, seq_len, dim]
                    seq_output, seq_out_len = layer(seq_tensor, current_actual_seq_lens[i])
                    batch_outputs.append(seq_output.squeeze(0))
                    new_actual_seq_lens.append(seq_out_len)
                x = torch.stack(batch_outputs)
                current_actual_seq_lens = new_actual_seq_lens
            
        return x
    
    def predict_t(self, seq):
        """
        Predict target vector for a character sequence
        Returns the average of all output vectors in the sequence
        
        Args:
            seq (str): Input character sequence of variable length
            
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
        Train the model using gradient descent with target vectors for variable-length sequences
        
        Args:
            seqs: List of character sequences (variable length, up to max_input_seq_len)
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
        # Validate input sequences don't exceed maximum length
        for seq in seqs:
            if len(seq) > self.max_input_seq_len:
                raise ValueError(f"Sequence length {len(seq)} exceeds maximum allowed {self.max_input_seq_len}")
        
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
        t_tensors = [torch.tensor(t, dtype=torch.float32, device=self.device) for t in t_list]
        
        # Set up optimizer (train all parameters including char layer)
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
            
            # Process sequences in random order
            indices = list(range(len(seqs)))
            random.shuffle(indices)
            
            for i in range(0, len(indices), batch_size):
                batch_indices = indices[i:i+batch_size]
                batch_seqs = [seqs[idx] for idx in batch_indices]
                batch_targets = [t_tensors[idx] for idx in batch_indices]
                
                optimizer.zero_grad()
                batch_loss = 0.0
                
                for seq, target in zip(batch_seqs, batch_targets):
                    # Forward pass
                    output = self.forward(seq)  # [1, final_seq_len, final_dim]
                    pred_target = output.mean(dim=1).squeeze(0)  # [final_dim]
                    
                    # Compute loss
                    loss = torch.sum((pred_target - target) ** 2)
                    batch_loss += loss
                
                # Average loss over batch
                batch_loss = batch_loss / len(batch_seqs)
                batch_loss.backward()
                optimizer.step()
                
                total_loss += batch_loss.item() * len(batch_seqs)
                total_sequences += len(batch_seqs)
            
            # Average loss over epoch
            avg_loss = total_loss / total_sequences
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
                if best_model_state is not None and avg_loss > prev_loss:
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
        # Validate input sequences don't exceed maximum length
        for seq in seqs:
            if len(seq) > self.max_input_seq_len:
                raise ValueError(f"Sequence length {len(seq)} exceeds maximum allowed {self.max_input_seq_len}")
        
        # Initialize classification head if not already done
        if self.classifier is None or self.num_classes != num_classes:
            self.classifier = nn.Linear(self.model_dims[-1] if self.model_dims else self.vec_dim, num_classes).to(self.device)
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
                batch_loss = 0.0
                batch_logits = []
                
                # Process each sequence in the batch
                for seq in batch_seqs:
                    # Get sequence representation through the entire model
                    with torch.no_grad():
                        output = self.forward(seq)  # [1, final_seq_len, final_dim]
                        seq_vector = output.mean(dim=1).squeeze(0)  # [final_dim]
                    
                    # Get logits through classification head
                    logits = self.classifier(seq_vector.unsqueeze(0))
                    batch_logits.append(logits)
                
                # Stack all logits and compute loss
                if batch_logits:
                    all_logits = torch.cat(batch_logits, dim=0)
                    loss = criterion(all_logits, batch_labels)
                    loss.backward()
                    optimizer.step()
                    
                    # Calculate batch statistics
                    batch_loss = loss.item()
                    total_loss += batch_loss * len(batch_seqs)
                    total_sequences += len(batch_seqs)
                    
                    # Calculate accuracy
                    with torch.no_grad():
                        predictions = torch.argmax(all_logits, dim=1)
                        correct_predictions += (predictions == batch_labels).sum().item()
                
                # Clear GPU cache periodically
                if torch.cuda.is_available():
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
        # Validate input sequences don't exceed maximum length
        for seq in seqs:
            if len(seq) > self.max_input_seq_len:
                raise ValueError(f"Sequence length {len(seq)} exceeds maximum allowed {self.max_input_seq_len}")
        
        # Initialize label head if not already done
        if self.labeller is None or self.num_labels != num_labels:
            self.labeller = nn.Linear(self.model_dims[-1] if self.model_dims else self.vec_dim, num_labels).to(self.device)
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
                batch_loss = 0.0
                batch_correct = 0
                batch_predictions = 0
                
                # Process each sequence in the batch
                batch_predictions_list = []
                for seq in batch_seqs:
                    # Get sequence representation through the entire model
                    with torch.no_grad():
                        output = self.forward(seq)  # [1, final_seq_len, final_dim]
                        seq_representation = output.mean(dim=1).squeeze(0)  # [final_dim]
                    
                    # Pass through classification head to get logits
                    logits = self.labeller(seq_representation)
                    batch_predictions_list.append(logits)
                
                # Stack predictions for the batch
                if batch_predictions_list:
                    batch_logits = torch.stack(batch_predictions_list, dim=0)
                    
                    # Calculate loss for the batch
                    batch_loss = criterion(batch_logits, batch_labels)
                    
                    # Calculate accuracy
                    with torch.no_grad():
                        # Apply sigmoid to get probabilities
                        probs = torch.sigmoid(batch_logits)
                        # Threshold at 0.5 for binary predictions
                        predictions = (probs > 0.5).float()
                        # Calculate number of correct predictions
                        batch_correct = (predictions == batch_labels).sum().item()
                        batch_predictions = batch_labels.numel()
                    
                    # Backpropagate
                    batch_loss.backward()
                    optimizer.step()
                    
                    total_loss += batch_loss.item() * len(batch_seqs)
                    total_correct += batch_correct
                    total_predictions += batch_predictions
                    total_sequences += len(batch_seqs)
                
                # Clear GPU cache periodically
                if torch.cuda.is_available():
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
        
        # Get sequence vector representation
        with torch.no_grad():
            output = self.forward(seq)  # [1, final_seq_len, final_dim]
            seq_vector = output.mean(dim=1).squeeze(0)  # [final_dim]
            
            # Get logits through classification head
            logits = self.classifier(seq_vector.unsqueeze(0))
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
        
        # Get sequence representation
        with torch.no_grad():
            output = self.forward(seq)  # [1, final_seq_len, final_dim]
            seq_representation = output.mean(dim=1).squeeze(0)  # [final_dim]
            
            # Pass through classification head to get logits
            logits = self.labeller(seq_representation)
            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(logits).cpu().numpy()
        
        # Apply threshold to get binary predictions
        binary_preds = (probs > threshold).astype(np.float32)
        
        return binary_preds, probs
    
    def self_train(self, seqs, max_iters=100, tol=1e-6, learning_rate=0.01, 
                   continued=False, self_mode='gap', decay_rate=1.0, print_every=10,
                   checkpoint_file=None, checkpoint_interval=5):
        """
        Self-training method for the hierarchical model with adaptive Linker matrices for variable-length sequences
        
        Args:
            seqs: List of character sequences (variable length, up to max_input_seq_len)
            max_iters: Maximum training iterations
            tol: Convergence tolerance
            learning_rate: Initial learning rate
            continued: Whether to continue from existing parameters
            self_mode: 'gap' or 'reg' training mode
            decay_rate: Learning rate decay rate
            print_every: Print interval
            checkpoint_file: Path to save/load checkpoint file for resuming training
            checkpoint_interval: Interval (in iterations) for saving checkpoints
            
        Returns:
            list: Training loss history
        """
        # Validate input sequences don't exceed maximum length
        for seq in seqs:
            if len(seq) > self.max_input_seq_len:
                raise ValueError(f"Sequence length {len(seq)} exceeds maximum allowed {self.max_input_seq_len}")
        
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
            
            # Process each sequence
            for seq in seqs:
                # Convert to tensor through char layer
                char_output, actual_token_seq_len = self.char_sequence_to_tensor(seq)  # [1, token_seq_len, vec_dim]
                char_seq_len = char_output.shape[1]
                
                if char_seq_len <= 1 and self_mode == 'reg':
                    continue  # Skip sequences that are too short for reg mode
                
                # Forward pass through hierarchical layers with adaptive Linker matrices
                hierarchical_output = char_output
                current_seq_len = actual_token_seq_len
                for layer in self.hierarchical_layers:
                    hierarchical_output, current_seq_len = layer(hierarchical_output, current_seq_len)
                
                hier_seq_len = hierarchical_output.shape[1]
                
                # Compute loss based on self_mode
                seq_loss = 0.0
                valid_positions = 0
                
                # Use the minimum sequence length to avoid index errors
                min_seq_len = min(char_seq_len, hier_seq_len)
                
                for k in range(min_seq_len):
                    if self_mode == 'gap':
                        # Self-consistency: output should match char layer output
                        target = char_output[0, k]
                        pred = hierarchical_output[0, k]
                        seq_loss += torch.sum((pred - target) ** 2)
                        valid_positions += 1
                    else:  # 'reg' mode
                        if k < char_seq_len - 1 and k < hier_seq_len:
                            # Predict next position's char layer output
                            target = char_output[0, k + 1]
                            pred = hierarchical_output[0, k]
                            seq_loss += torch.sum((pred - target) ** 2)
                            valid_positions += 1
                
                if valid_positions > 0:
                    seq_loss = seq_loss / valid_positions
                    total_loss += seq_loss.item()
                    total_samples += 1
                    
                    # Backward pass
                    optimizer.zero_grad()
                    seq_loss.backward()
                    optimizer.step()
            
            if total_samples == 0:
                avg_loss = 0.0
            else:
                avg_loss = total_loss / total_samples
                
            history.append(avg_loss)
            
            # Update best model state if current loss is lower
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_model_state = copy.deepcopy(self.state_dict())
            
            if it % print_every == 0 or it == max_iters - 1:
                current_lr = scheduler.get_last_lr()[0]
                mode_display = "Gap" if self_mode == 'gap' else "Reg"
                print(f"SelfTrain({mode_display}) Iter {it:3d}: Loss = {avg_loss:.6f}, LR = {current_lr:.6f}")
            
            # Save checkpoint at specified intervals
            if checkpoint_file and (it % checkpoint_interval == 0 or it == max_iters - 1):
                self._save_checkpoint(
                    checkpoint_file, it, history, optimizer, scheduler, best_loss
                )
            
            # Check convergence
            if prev_loss - avg_loss < tol:
                print(f"Converged after {it+1} iterations")
                # Restore the best model state before breaking
                if best_model_state is not None and avg_loss > prev_loss:
                    self.load_state_dict(best_model_state)
                    print(f"Restored best model state with loss = {best_loss:.6f}")
                    history[-1] = best_loss                    
                break
            prev_loss = avg_loss

            # Learning rate decay
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
                'max_input_seq_len': self.max_input_seq_len,
                'model_dims': self.model_dims,
                'basis_dims': self.basis_dims,
                'linker_dims': self.linker_dims,
                'linker_trainable': self.linker_trainable,
                'use_residual_list': self.use_residual_list,
                'reduction_factors': self.reduction_factors,
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
        #print(f"Checkpoint saved at iteration {iteration} to {checkpoint_file}")

    def _compute_training_statistics(self, seqs, batch_size=50):
        """Compute and store statistics for reconstruction and generation"""
        total_token_count = 0
        total_vec_sum = torch.zeros(self.vec_dim, device=self.device)
        
        with torch.no_grad():
            for i in range(0, len(seqs), batch_size):
                batch_seqs = seqs[i:i+batch_size]
                batch_token_count = 0
                batch_vec_sum = torch.zeros(self.vec_dim, device=self.device)
                
                for seq in batch_seqs:
                    toks = self.extract_tokens(seq)
                    if not toks:
                        continue
                        
                    batch_token_count += len(toks)
                    token_indices = self.token_to_indices(toks)
                    
                    Nk_batch = self.char_batch_compute_Nk(token_indices)
                    batch_vec_sum += Nk_batch.sum(dim=0)
                
                total_token_count += batch_token_count
                total_vec_sum += batch_vec_sum
                
                # Clean batch
                del batch_vec_sum
        
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
        
        with torch.no_grad():
            # Batch processing sequences
            for i in range(0, len(seqs), batch_size):
                batch_seqs = seqs[i:i+batch_size]
                batch_outputs = []
                
                for seq in batch_seqs:
                    try:
                        output = self.forward(seq)
                        if output.numel() > 0:
                            seq_output = output.mean(dim=1).squeeze(0)
                            batch_outputs.append(seq_output)
                    except RuntimeError as e:
                        # Handle the situation of insufficient memory
                        if "out of memory" in str(e):
                            print(f"Warning: The sequence is too long, resulting in insufficient memory. Skip the sequence: {seq[:50]}...")
                            continue
                        else:
                            raise e
                
                if batch_outputs:
                    # Immediately accumulate and release the batch memory
                    batch_tensor = torch.stack(batch_outputs)
                    total_output += batch_tensor.sum(dim=0)
                    total_sequences += len(batch_outputs)
                    
                    # Clean batches
                    del batch_tensor, batch_outputs
                
                # Optional: Regularly clean the GPU cache
                if torch.cuda.is_available() and (i // batch_size) % 10 == 0:
                    torch.cuda.empty_cache()
        
        if total_sequences > 0:
            self.mean_target = total_output / total_sequences
        else:
            self.mean_target = None
    
    def reconstruct(self, L, tau=0.0):
        """
        Reconstruct character sequence of length L using the entire hierarchical model with adaptive Linker.
        
        This method uses temperature-controlled sampling based on the complete model's
        predictions, incorporating both character-level and hierarchical patterns with adaptive Linker matrices.
        
        Args:
            L (int): Length of sequence to generate (must be <= max_input_seq_len)
            tau (float): Temperature parameter for stochastic sampling.
                        tau=0: deterministic (greedy) selection
                        tau>0: stochastic selection with temperature
            
        Returns:
            str: Reconstructed character sequence
        """
        assert self.trained, "Model must be trained first"
        assert self.mean_target is not None, "Mean target vector must be computed first"
        
        if tau < 0:
            raise ValueError("Temperature must be non-negative")
        
        if L <= 0:
            return ""
        
        if L > self.max_input_seq_len:
            print(f"Warning: Requested length {L} exceeds maximum input sequence length {self.max_input_seq_len}. Truncating.")
            L = self.max_input_seq_len
        
        generated_sequence = ""
        
        # Generate sequence token by token
        while len(generated_sequence) < L:
            candidate_scores = {}
            
            # Evaluate each possible token
            for token in self.tokens:
                candidate_seq = generated_sequence + token
                
                # Don't pad - use variable length naturally
                if len(candidate_seq) > self.max_input_seq_len:
                    candidate_seq = candidate_seq[:self.max_input_seq_len]
                    
                # Skip if candidate exceeds desired length
                if len(candidate_seq) > L:
                    continue
                    
                with torch.no_grad():
                    model_output = self.forward(candidate_seq)
                    
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
        nn.init.uniform_(self.char_linear.weight, -0.1, 0.1)
        nn.init.uniform_(self.char_Acoeff, -0.1, 0.1)
        
        # Reset hierarchical layer parameters
        for layer in self.hierarchical_layers:
            nn.init.uniform_(layer.M, -0.1, 0.1)
            nn.init.uniform_(layer.Acoeff, -0.1, 0.1)
            nn.init.uniform_(layer.Linker, -0.1, 0.1)
            if hasattr(layer, 'residual_proj') and isinstance(layer.residual_proj, nn.Linear):
                nn.init.uniform_(layer.residual_proj.weight, -0.1, 0.1)
            if hasattr(layer, 'residual_linker') and layer.residual_linker is not None:
                nn.init.uniform_(layer.residual_linker, -0.1, 0.1)
        
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
        
        print("="*70)
        print(f"{'Component':<15} | {'Layer/Param':<25} | {'Count':<15} | {'Shape':<20}")
        print("-"*70)
        
        # Character layer parameters
        char_params = 0
        
        # Character embedding parameters
        num = self.char_embedding.weight.numel()
        shape = str(tuple(self.char_embedding.weight.shape))
        print(f"{'Char Layer':<15} | {'char_embedding':<25} | {num:<15} | {shape:<20}")
        char_params += num
        total_params += num
        
        # Character linear parameters
        num = self.char_linear.weight.numel()
        shape = str(tuple(self.char_linear.weight.shape))
        print(f"{'Char Layer':<15} | {'char_linear':<25} | {num:<15} | {shape:<20}")
        char_params += num
        total_params += num
        
        # Character Acoeff parameters
        num = self.char_Acoeff.numel()
        shape = str(tuple(self.char_Acoeff.shape))
        print(f"{'Char Layer':<15} | {'char_Acoeff':<25} | {num:<15} | {shape:<20}")
        char_params += num
        total_params += num
        
        print(f"{'Char Layer':<15} | {'TOTAL':<25} | {char_params:<15} |")
        print("-"*70)
        
        # Hierarchical layer parameters
        for i, layer in enumerate(self.hierarchical_layers):
            layer_params = 0
            layer_name = f"Hier Layer {i}"
            
            for name, param in layer.named_parameters():
                if param.requires_grad:
                    num = param.numel()
                    shape = str(tuple(param.shape))
                    print(f"{layer_name:<15} | {name:<25} | {num:<15} | {shape:<20}")
                    layer_params += num
                    total_params += num
            
            print(f"{layer_name:<15} | {'TOTAL':<25} | {layer_params:<15} |")
            print("-"*70)
        
        # Classification heads
        if self.classifier is not None:
            cls_params = 0
            for name, param in self.classifier.named_parameters():
                if param.requires_grad:
                    num = param.numel()
                    shape = str(tuple(param.shape))
                    print(f"{'Classifier':<15} | {name:<25} | {num:<15} | {shape:<20}")
                    cls_params += num
                    total_params += num
            print(f"{'Classifier':<15} | {'TOTAL':<25} | {cls_params:<15} |")
            print("-"*70)
        
        if self.labeller is not None:
            lbl_params = 0
            for name, param in self.labeller.named_parameters():
                if param.requires_grad:
                    num = param.numel()
                    shape = str(tuple(param.shape))
                    print(f"{'Labeller':<15} | {name:<25} | {num:<15} | {shape:<20}")
                    lbl_params += num
                    total_params += num
            print(f"{'Labeller':<15} | {'TOTAL':<25} | {lbl_params:<15} |")
            print("-"*70)
        
        print(f"{'TOTAL':<15} | {'ALL PARAMETERS':<25} | {total_params:<15} |")
        print("="*70)
        
        return total_params
    
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
                'max_input_seq_len': self.max_input_seq_len,
                'model_dims': self.model_dims,
                'basis_dims': self.basis_dims,
                'linker_dims': self.linker_dims,
                'linker_trainable': self.linker_trainable,
                'use_residual_list': self.use_residual_list,
                'reduction_factors': self.reduction_factors,
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
            'classification_info': {
                'num_classes': self.num_classes,
                'num_labels': self.num_labels
            }
        }, filename)
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
            max_input_seq_len=config['max_input_seq_len'],
            model_dims=config['model_dims'],
            basis_dims=config['basis_dims'],
            linker_dims=config['linker_dims'],
            rank_mode=char_layer_config['rank_mode'],
            mode=char_layer_config['mode'],
            user_step=char_layer_config['user_step'],            
            linker_trainable=config.get('linker_trainable', False),
            use_residual_list=config.get('use_residual_list', [None] * len(config['model_dims'])),
            reduction_factors=config.get('reduction_factors', [2] * len(config['model_dims'])),
            device=device
        )
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        
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
        
        # Load classification info
        if 'classification_info' in checkpoint:
            cls_info = checkpoint['classification_info']
            if cls_info['num_classes'] is not None:
                model.num_classes = cls_info['num_classes']
                model.classifier = nn.Linear(
                    model.model_dims[-1] if model.model_dims else model.vec_dim, 
                    model.num_classes
                ).to(device)
            if cls_info['num_labels'] is not None:
                model.num_labels = cls_info['num_labels']
                model.labeller = nn.Linear(
                    model.model_dims[-1] if model.model_dims else model.vec_dim,
                    model.num_labels
                ).to(device)
        
        print(f"Model loaded from {filename}")
        return model

# === Example Usage with Variable-Length Sequences ===
if __name__ == "__main__":
    from statistics import correlation
    
    print("="*60)
    print("HierDDLabC - Hierarchical Dual Descriptor with Linker Matrix and Character Input")
    print("Adaptive Linker Model with Variable-Length Sequence Support Demonstration")
    print("="*60)
    
    # Set random seeds for reproducibility
    torch.manual_seed(2)
    random.seed(2)
    np.random.seed(2)
    
    # Configuration for variable-length sequences
    charset = ['A', 'C', 'G', 'T']
    rank = 1
    vec_dim = 8
    max_input_seq_len = 200  # Maximum input character sequence length
    model_dims = [16, 12, 8]  # Hierarchical layer dimensions
    basis_dims = [100, 80, 60]  # Basis dimensions for hierarchical layers
    linker_dims = [100, 50, 25]  # Maximum output sequence lengths for hierarchical layers
    reduction_factors = [2, 2, 2]  # Sequence length reduction factors for each layer
    num_seqs = 100
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {device}")
    print(f"Character set: {charset}")
    print(f"Rank (k-mer length): {rank}")
    print(f"Character layer output dim: {vec_dim}")
    print(f"Maximum input sequence length: {max_input_seq_len}")
    print(f"Hierarchical layer dims: {model_dims}")
    print(f"Hierarchical basis dims: {basis_dims}")
    print(f"Linker maximum output sequence lengths: {linker_dims}")
    print(f"Reduction factors: {reduction_factors}")
    print(f"Using Adaptive Linker-based architecture with character input and variable-length support")
    
    # Generate synthetic training data with variable lengths
    print("\nGenerating synthetic training data with variable lengths...")
    seqs, t_list = [], []
    for i in range(num_seqs):
        # Generate variable-length sequences (50 to 200 characters)
        seq_len = random.randint(50, max_input_seq_len)
        seq = ''.join(random.choices(charset, k=seq_len))
        seqs.append(seq)
        # Create target vector (final layer dimension)
        t_list.append([random.uniform(-1.0, 1.0) for _ in range(model_dims[-1])])
    
    # Display sequence length statistics
    seq_lengths = [len(seq) for seq in seqs]
    print(f"Sequence length statistics: min={min(seq_lengths)}, max={max(seq_lengths)}, avg={np.mean(seq_lengths):.1f}")
    
    # Create model with adaptive Linker matrices for variable-length sequences
    print("\nCreating HierDDLabC model with adaptive Linker matrices...")
    model = HierDDLabC(
        charset=charset,
        rank=rank,
        vec_dim=vec_dim,
        max_input_seq_len=max_input_seq_len,
        model_dims=model_dims,
        basis_dims=basis_dims,
        linker_dims=linker_dims,
        reduction_factors=reduction_factors,
        linker_trainable=[True, False, True],  # Mixed trainability
        use_residual_list=['separate', 'shared', None],  # Mixed residual strategies
        mode='linear',
        device=device
    )
    
    # Count parameters
    print("\nModel Parameter Count:")
    total_params = model.count_parameters()
    
    # Test character-level processing with variable-length sequences
    print("\n" + "="*50)
    print("Character-Level Processing Test with Variable-Length Sequences")
    print("="*50)
    
    # Test with sequences of different lengths
    test_seqs = [seqs[0], seqs[10], seqs[20]]  # Select sequences of different lengths
    for i, test_seq in enumerate(test_seqs):
        char_vectors = model.char_describe(test_seq)
        print(f"Sequence {i+1} (length {len(test_seq)}): {test_seq[:30]}...")
        print(f"  Character vectors shape: {char_vectors.shape}")
        print(f"  Number of tokens: {len(model.extract_tokens(test_seq))}")
    
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
        batch_size=8,   # Smaller batch size for variable lengths
        checkpoint_file='hierddlabc_var_gd_checkpoint.pth',
        checkpoint_interval=10
    )
    
    # Predictions and correlation analysis
    print("\n" + "="*50)
    print("Prediction and Correlation Analysis with Variable-Length Sequences")
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
    print("Sequence Reconstruction with Variable-Length Support")
    print("="*50)
    
    # Deterministic reconstruction
    det_seq = model.reconstruct(L=100, tau=0.0)
    print(f"Deterministic reconstruction (tau=0.0, length {len(det_seq)}): {det_seq[:50]}...")
    
    # Stochastic reconstruction
    stoch_seq = model.reconstruct(L=100, tau=0.5)
    print(f"Stochastic reconstruction (tau=0.5, length {len(stoch_seq)}): {stoch_seq[:50]}...")
    
    # Character-level reconstruction
    char_reconstructed_seq = model.char_reconstruct()
    print(f"Character-level reconstructed sequence (length {len(char_reconstructed_seq)}):")
    print(f"First 100 chars: {char_reconstructed_seq[:100]}")
    
    # Character-level generation
    char_gen_seq = model.char_generate(L=80, tau=0.2)
    print(f"Character-level generation (tau=0.2): {char_gen_seq}")
    
    # === Classification Task Example ===
    print("\n" + "="*50)
    print("Classification Task with Variable-Length Sequences")
    print("="*50)
    
    # Generate classification data
    num_classes = 3
    class_seqs = []
    class_labels = []
    
    # Create sequences with different patterns for each class
    for class_id in range(num_classes):
        for _ in range(30):  # 30 sequences per class
            seq_len = random.randint(50, 150)
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
    cls_model = HierDDLabC(
        charset=charset,
        rank=rank,
        vec_dim=vec_dim,
        max_input_seq_len=max_input_seq_len,
        model_dims=model_dims,
        basis_dims=basis_dims,
        linker_dims=linker_dims,
        reduction_factors=reduction_factors,
        linker_trainable=False,
        use_residual_list=[None] * len(model_dims),
        mode='linear',
        device=device
    )
    
    # Train for classification
    print("\nStarting classification training...")
    cls_history = cls_model.cls_train(
        class_seqs, 
        class_labels, 
        num_classes,
        max_iters=50,
        learning_rate=0.05,
        decay_rate=0.98,
        print_every=5,
        batch_size=16
    )
    
    # Test classification predictions
    print("\nClassification prediction results:")
    correct = 0
    for i, (seq, true_label) in enumerate(zip(class_seqs[:10], class_labels[:10])):
        pred_class, probs = cls_model.predict_c(seq)
        is_correct = pred_class == true_label
        correct += is_correct
        print(f"Seq {i+1}: True={true_label}, Pred={pred_class}, Correct={is_correct}, "
              f"Probs={[f'{p:.3f}' for p in probs]}")
    
    accuracy = correct / 10
    print(f"Accuracy on 10 test sequences: {accuracy:.4f}")
    
    # === Multi-Label Classification Task Example ===
    print("\n" + "="*50)
    print("Multi-Label Classification with Variable-Length Sequences")
    print("="*50)
    
    # Generate multi-label data
    num_labels = 4
    label_seqs = []
    labels = []
    
    for _ in range(80):
        seq_len = random.randint(60, 180)
        seq = ''.join(random.choices(charset, k=seq_len))
        label_seqs.append(seq)
        
        # Create random binary labels (each sequence can have 0-4 active labels)
        label_vec = [random.random() > 0.7 for _ in range(num_labels)]
        labels.append([1.0 if x else 0.0 for x in label_vec])
    
    # Create a new model for multi-label classification
    lbl_model = HierDDLabC(
        charset=charset,
        rank=rank,
        vec_dim=vec_dim,
        max_input_seq_len=max_input_seq_len,
        model_dims=model_dims,
        basis_dims=basis_dims,
        linker_dims=linker_dims,
        reduction_factors=reduction_factors,
        linker_trainable=False,
        use_residual_list=[None] * len(model_dims),
        mode='linear',
        device=device
    )
    
    # Train for multi-label classification
    print("\nStarting multi-label classification training...")
    loss_history, acc_history = lbl_model.lbl_train(
        label_seqs,
        labels,
        num_labels,
        max_iters=30,
        learning_rate=0.05,
        decay_rate=0.98,
        print_every=5,
        batch_size=16
    )
    
    # Test multi-label predictions
    print("\nMulti-label prediction results:")
    for i, (seq, true_labels) in enumerate(zip(label_seqs[:3], labels[:3])):
        binary_pred, probs = lbl_model.predict_l(seq, threshold=0.5)
        true_labels_np = np.array(true_labels)
        
        print(f"Sequence {i+1} (length {len(seq)}):")
        print(f"  True labels: {true_labels_np}")
        print(f"  Predicted binary: {binary_pred}")
        print(f"  Predicted probabilities: {[f'{p:.4f}' for p in probs]}")
        print(f"  Match: {np.array_equal(binary_pred, true_labels_np)}")
    
    # Self-training example with variable-length sequences
    print("\n" + "="*50)
    print("Self-Training Example with Variable-Length Sequences")
    print("="*50)
    
    # Create a new model for self-training
    self_model = HierDDLabC(
        charset=charset,
        rank=rank,
        vec_dim=vec_dim,
        max_input_seq_len=max_input_seq_len,
        model_dims=model_dims,
        basis_dims=basis_dims,
        linker_dims=linker_dims,
        reduction_factors=reduction_factors,
        linker_trainable=False,
        use_residual_list=[None] * len(model_dims),
        mode='linear',
        device=device
    )
    
    # Self-train in gap mode with variable-length sequences
    print("\nSelf-training in 'gap' mode with variable-length sequences...")
    self_history_gap = self_model.self_train(
        seqs[:20],  # Use subset for faster demonstration
        max_iters=10,
        learning_rate=0.01,
        self_mode='gap',
        print_every=2,
        checkpoint_file='hierddlabc_var_self_checkpoint.pth',
        checkpoint_interval=5
    )
    
    # Reconstruct from self-trained model
    self_rec_seq = self_model.reconstruct(L=80, tau=0.2)
    print(f"Reconstructed from self-trained model: {self_rec_seq}")
    
    # Model save and load test
    print("\n" + "="*50)
    print("Model Save/Load Test with Variable-Length Support")
    print("="*50)
    
    # Save model
    model.save("hierddlabc_var_model.pth")
    
    # Load model
    loaded_model = HierDDLabC.load("hierddlabc_var_model.pth", device=device)
    
    # Test loaded model with variable-length sequences
    test_seqs = [seqs[0], seqs[5], seqs[10]]  # Test with different lengths
    for i, test_seq in enumerate(test_seqs):
        original_pred = model.predict_t(test_seq)
        loaded_pred = loaded_model.predict_t(test_seq)
        
        print(f"Sequence {i+1} (length {len(test_seq)}):")
        print(f"  Original model prediction: {[f'{x:.4f}' for x in original_pred[:3]]}...")
        print(f"  Loaded model prediction:   {[f'{x:.4f}' for x in loaded_pred[:3]]}...")
        
        pred_diff = np.max(np.abs(original_pred - loaded_pred))
        print(f"  Maximum prediction difference: {pred_diff:.6e}")
        
        if pred_diff < 1e-6:
            print("  ✓ Prediction consistency test PASSED")
        else:
            print("  ✗ Prediction consistency test FAILED")
    
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
    
    print("\nAll tests with variable-length sequences completed successfully!")
