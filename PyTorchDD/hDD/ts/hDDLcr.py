# Copyright (C) 2005-2025, Bin-Guang Ma (mbg@mail.hzau.edu.cn); SPDX-License-Identifier: MIT
# Hierarchical Dual Descriptor (Tensor form) with Character Sequence Input (HierDDLtsC)
# Combines character-level processing and hierarchical vector processing with Linker matrices
# Modified to support variable-length sequences with adaptive Linker matrices
# This program is for the demonstration of methodology and not fully refined.
# Author: Bin-Guang Ma (assisted by DeepSeek); Date: 2025-11-12

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
    """Single layer of Hierarchical Dual Descriptor with Adaptive Linker Matrix for Variable-Length Sequences"""
    def __init__(self, in_dim, out_dim, max_in_seq, max_out_seq, num_basis, linker_trainable, use_residual, device, reduction_factor=2):
        """
        Initialize a hierarchical layer with adaptive Linker matrix for variable-length sequences
        
        Args:
            in_dim (int): Input dimension
            out_dim (int): Output dimension
            max_in_seq (int): Maximum input sequence length in the dataset
            max_out_seq (int): Maximum output sequence length (max_in_seq // reduction_factor)
            num_basis (int): Number of basis functions
            linker_trainable (bool): Whether Linker matrix is trainable
            use_residual (str or None): Residual connection type. Options are:
                - 'separate': use separate projection and Linker for residual
                - 'shared': share M and Linker for residual
                - None: no residual connection
            device (torch.device): Device to use
            reduction_factor (int): Factor by which to reduce sequence length (default: 2)
        """
        super().__init__()
        self.device = device
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.max_in_seq = max_in_seq
        self.max_out_seq = max_out_seq
        self.num_basis = num_basis
        self.linker_trainable = linker_trainable
        self.use_residual = use_residual
        self.reduction_factor = reduction_factor
        
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
        
        # Adaptive Linker matrix for variable-length sequences
        # Size is based on maximum sequence lengths in the dataset
        self.Linker = nn.Parameter(
            torch.empty(max_in_seq, max_out_seq),
            requires_grad=linker_trainable
        )

        # Layer normalization
        self.ln = nn.LayerNorm(out_dim)
        
        # Residual connection setup
        if self.use_residual == 'separate':
            # Separate projection and Linker for residual path
            self.residual_proj = nn.Linear(in_dim, out_dim, bias=False)
            self.residual_linker = nn.Parameter(
                torch.empty(max_in_seq, max_out_seq),
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
    
    def adaptive_sequence_transform(self, T, actual_in_seq_len):
        """
        Transform sequence length using adaptive Linker matrix for variable-length sequences
        
        Args:
            T (Tensor): Input tensor of shape (batch_size, in_seq, out_dim)
            actual_in_seq_len (int): Actual sequence length for this batch
            
        Returns:
            Tensor: Transformed tensor of shape (batch_size, out_seq, out_dim)
        """
        batch_size, in_seq, out_dim = T.shape
        
        # Calculate output sequence length based on reduction factor
        actual_out_seq_len = max(1, actual_in_seq_len // self.reduction_factor)
        
        # Extract the relevant portion of the Linker matrix
        # Use first 'actual_in_seq_len' rows and first 'actual_out_seq_len' columns
        adaptive_linker = self.Linker[:actual_in_seq_len, :actual_out_seq_len]
        
        # Using matrix multiplication: T @ adaptive_linker
        # T: [batch_size, in_seq, out_dim] -> [batch_size, out_seq, out_dim]
        return torch.matmul(T.transpose(1, 2), adaptive_linker).transpose(1, 2)
    
    def forward(self, x, actual_seq_len=None):
        """
        Forward pass through the layer with variable-length sequence support
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, in_seq, in_dim)
            actual_seq_len (int): Actual sequence length (if None, use x.shape[1])
            
        Returns:
            Tensor: Output tensor of shape (batch_size, out_seq, out_dim)
            int: Actual output sequence length
        """
        batch_size, in_seq, in_dim = x.shape
        
        # Use provided actual sequence length or default to input sequence length
        if actual_seq_len is None:
            actual_seq_len = in_seq
        
        # Main path: linear transformation
        Z = torch.matmul(x, self.M.t())  # [batch_size, in_seq, out_dim]

        # Layer normalization
        Z = self.ln(Z)
        
        # Position-dependent transformation
        T = self.position_transform(Z)  # [batch_size, in_seq, out_dim]
        
        # Adaptive sequence transformation for variable-length sequences
        U = self.adaptive_sequence_transform(T, actual_seq_len)  # [batch_size, out_seq, out_dim]
        
        # Calculate actual output sequence length
        actual_out_seq_len = U.shape[1]
        
        # Residual connection processing
        if self.use_residual == 'separate':
            # Separate projection and Linker for residual
            residual_feat = self.residual_proj(x)  # [batch_size, in_seq, out_dim]
            residual_linker = self.residual_linker[:actual_seq_len, :actual_out_seq_len]
            residual = torch.matmul(residual_feat.transpose(1, 2), residual_linker).transpose(1, 2)  # [batch_size, out_seq, out_dim]
            out = U + residual
        elif self.use_residual == 'shared':
            # Shared M and Linker for residual
            residual_feat = torch.matmul(x, self.M.t())  # [batch_size, in_seq, out_dim]
            residual_linker = self.Linker[:actual_seq_len, :actual_out_seq_len]
            residual = torch.matmul(residual_feat.transpose(1, 2), residual_linker).transpose(1, 2)  # [batch_size, out_seq, out_dim]
            out = U + residual
        else:
            # Identity residual only if dimensions match
            if self.in_dim == self.out_dim and actual_seq_len == actual_out_seq_len:
                residual_linker = self.Linker[:actual_seq_len, :actual_out_seq_len]
                residual = torch.matmul(x.transpose(1, 2), residual_linker).transpose(1, 2)  # [batch_size, out_seq, out_dim]
                out = U + residual
            else:
                out = U 
       
        return out, actual_out_seq_len

class HierDDLtsC(nn.Module):
    """
    Hierarchical Dual Descriptor with Character Sequence Input and Adaptive Linker Matrices
    Modified to support variable-length sequences with adaptive Linker matrices
    Combines character-level processing and hierarchical vector processing with Linker matrices
    Supports variable-length character sequences as input
    """
    def __init__(self, charset, rank=1, rank_mode='drop', vec_dim=2, num_basis=5, 
                 mode='linear', user_step=None, max_input_seq_len=100, model_dims=[2], 
                 num_basis_list=[5], linker_dims=[50], linker_trainable=False,
                 use_residual_list=None, reduction_factors=None, device='cuda'):
        """
        Initialize HierDDLtsC model with adaptive Linker matrices for variable-length sequences
        
        Args:
            charset: Character set for sequence input
            rank: k-mer length for tokenization
            rank_mode: 'pad' or 'drop' for incomplete k-mers
            vec_dim: Output dimension of character layer (Layer 0)
            num_basis: Number of basis functions for character layer
            mode: 'linear' or 'nonlinear' tokenization
            user_step: Step size for nonlinear tokenization
            max_input_seq_len: Maximum input sequence length in the dataset
            model_dims: Output dimensions for hierarchical layers
            num_basis_list: Number of basis functions for each hierarchical layer
            linker_dims: Maximum output sequence lengths for each hierarchical layer
            linker_trainable: Whether Linker matrices are trainable
            use_residual_list: Residual connection types for hierarchical layers
            reduction_factors: Sequence length reduction factors for each layer (default: 2 for all)
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
        self.max_input_seq_len = max_input_seq_len  # Maximum input character sequence length
        self.model_dims = model_dims
        self.linker_dims = linker_dims
        self.num_layers = len(model_dims)
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.trained = False
        
        # Process reduction_factors
        if reduction_factors is None:
            self.reduction_factors = [2] * self.num_layers  # Default: reduce by factor of 2
        elif isinstance(reduction_factors, list):
            if len(reduction_factors) != self.num_layers:
                raise ValueError("reduction_factors list length must match number of layers")
            self.reduction_factors = reduction_factors
        else:
            raise TypeError("reduction_factors must be list or None")
        
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
        self.max_token_seq_len = self._calculate_token_seq_len(max_input_seq_len)
        
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

        # Training statistics for character layer
        self.mean_token_count = None
        self.mean_t = None
        
        # Hierarchical layers with adaptive Linker matrices for variable-length sequences (starting from Layer 1)
        self.hierarchical_layers = nn.ModuleList()
        for l in range(self.num_layers):
            # Determine input dimensions for this layer
            in_dim = vec_dim if l == 0 else model_dims[l-1]
            max_in_seq = self.max_token_seq_len if l == 0 else linker_dims[l-1]
            out_dim = model_dims[l]
            max_out_seq = linker_dims[l]
            num_basis = num_basis_list[l]
            use_residual = self.use_residual_list[l]
            linker_trainable = self.linker_trainable[l]
            reduction_factor = self.reduction_factors[l]
            
            # Initialize layer with adaptive Linker matrix
            layer = Layer(
                in_dim, out_dim, max_in_seq, max_out_seq, 
                num_basis, linker_trainable, 
                use_residual, self.device, reduction_factor
            )
            self.hierarchical_layers.append(layer)
        
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
        k_positions = torch.arange(len(toks), dtype=torch.float32, device=self.device)
        
        # Compute N(k) vectors through char layer
        Nk_batch = self.char_batch_compute_Nk(k_positions, token_indices)
        
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
    
    def grad_train(self, seqs, t_list, max_iters=1000, tol=1e-8, learning_rate=0.01, 
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
                    # Forward pass with variable-length sequence
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
                if best_model_state is not None and avg_loss > best_loss:
                    self.load_state_dict(best_model_state)                    
                    print(f"Restored best model state with loss = {best_loss:.6e}")
                    history[-1] = best_loss
                break
            prev_loss = avg_loss

            # Learning rate decay
            #if it % 5 == 0:  # Decay every 5 iterations
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
    
    def auto_train(self, seqs, max_iters=100, tol=1e-6, learning_rate=0.01, 
                   continued=False, auto_mode='gap', decay_rate=1.0, print_every=10,
                   checkpoint_file=None, checkpoint_interval=5):
        """
        Self-training method for the hierarchical model with adaptive Linker matrices for variable-length sequences
        
        Args:
            seqs: List of character sequences (variable length, up to max_input_seq_len)
            max_iters: Maximum training iterations
            tol: Convergence tolerance
            learning_rate: Initial learning rate
            continued: Whether to continue from existing parameters
            auto_mode: 'gap' or 'reg' training mode
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
                
        if auto_mode not in ('gap', 'reg'):
            raise ValueError("auto_mode must be either 'gap' or 'reg'")
        
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
            print(f"Resumed auto-training from checkpoint at iteration {start_iter}, best loss: {best_loss:.6f}")
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
                # Convert to tensor through char layer with variable-length support
                char_output, actual_token_seq_len = self.char_sequence_to_tensor(seq)  # [1, token_seq_len, vec_dim]
                char_seq_len = char_output.shape[1]
                
                if char_seq_len <= 1 and auto_mode == 'reg':
                    continue  # Skip sequences that are too short for reg mode
                
                # Forward pass through hierarchical layers with adaptive Linker matrices
                hierarchical_output = char_output
                current_seq_len = actual_token_seq_len
                for layer in self.hierarchical_layers:
                    hierarchical_output, current_seq_len = layer(hierarchical_output, current_seq_len)
                
                hier_seq_len = hierarchical_output.shape[1]
                
                # Compute loss based on auto_mode
                seq_loss = 0.0
                valid_positions = 0
                
                # Use the minimum sequence length to avoid index errors
                min_seq_len = min(char_seq_len, hier_seq_len)
                
                for k in range(min_seq_len):
                    if auto_mode == 'gap':
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
                mode_display = "Gap" if auto_mode == 'gap' else "Reg"
                print(f"AutoTrain({mode_display}) Iter {it:3d}: Loss = {avg_loss:.6f}, LR = {current_lr:.6f}")
            
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

            # Learning rate decay
            if it % 5 == 0:  # Decay every 5 iterations
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
                'linker_dims': self.linker_dims,
                'num_basis_list': [layer.num_basis for layer in self.hierarchical_layers],
                'linker_trainable': self.linker_trainable,
                'use_residual_list': self.use_residual_list,
                'reduction_factors': self.reduction_factors,
                'char_layer_config': {
                    'rank_mode': self.rank_mode,
                    'num_basis': self.char_num_basis,
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
                    k_positions = torch.arange(len(toks), dtype=torch.float32, device=self.device)
                    
                    Nk_batch = self.char_batch_compute_Nk(k_positions, token_indices)
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
    
    def reconstruct(self):
        """
        Reconstruct representative character sequence using the entire hierarchical model.
        
        This method leverages the complete model architecture including both the character layer
        and all hierarchical layers with adaptive Linker matrices to reconstruct a sequence that best 
        represents the learned patterns from the training data.
        
        Returns:
            str: Reconstructed character sequence
        """
        assert self.trained, "Model must be trained first"
        assert self.mean_target is not None, "Mean target vector must be computed first"
        
        # Determine sequence length based on training statistics
        n_tokens = round(self.mean_token_count)
        if n_tokens <= 0:
            n_tokens = 10  # Default minimum length
        
        # Get the target vector from the final hierarchical layer output space
        final_dim = self.model_dims[-1] if self.model_dims else self.vec_dim
        
        # Precompute all token embeddings
        all_token_indices = torch.arange(len(self.tokens), device=self.device)
        seq_tokens = []
        
        # Track current sequence state for hierarchical processing
        current_sequence = ""
        
        for k in range(n_tokens):
            best_tok = None
            min_error = float('inf')
            
            # Evaluate each possible token at this position
            for token_idx, token in enumerate(self.tokens):
                # Build candidate sequence
                candidate_seq = current_sequence + token
                
                # Don't pad - let the model handle variable length naturally
                if len(candidate_seq) > self.max_input_seq_len:
                    candidate_seq = candidate_seq[:self.max_input_seq_len]
                
                # Process through entire hierarchical model
                with torch.no_grad():
                    model_output = self.forward(candidate_seq)
                    
                    # Get the prediction (average over sequence)
                    if model_output.numel() > 0:
                        pred_target = model_output.mean(dim=1).squeeze(0)
                        
                        # Compute error compared to mean target
                        error = torch.sum((pred_target - self.mean_target) ** 2).item()
                        
                        if error < min_error:
                            min_error = error
                            best_tok = token
            
            # Add best token to sequence
            if best_tok is not None:
                seq_tokens.append(best_tok)
                current_sequence = ''.join(seq_tokens)
            else:
                # Fallback: use character layer reconstruction
                char_layer_recon = self.char_reconstruct()
                return char_layer_recon
        
        reconstructed_seq = ''.join(seq_tokens)
        
        # Remove padding characters if present
        if '_' in reconstructed_seq:
            reconstructed_seq = reconstructed_seq.replace('_', '')
        
        return reconstructed_seq[:self.max_input_seq_len]  # Ensure doesn't exceed maximum length

    def generate(self, L, tau=0.0):
        """
        Generate character sequence of length L using the entire hierarchical model.
        
        This method uses temperature-controlled sampling based on the complete model's
        predictions, incorporating both character-level and hierarchical patterns with adaptive Linker matrices.
        
        Args:
            L (int): Length of sequence to generate (must be <= max_input_seq_len)
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
        nn.init.uniform_(self.char_P, -0.1, 0.1)
        
        # Reset hierarchical layer parameters
        for layer in self.hierarchical_layers:
            layer.reset_parameters()
        
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
            
            # Linker matrix (adaptive for variable-length sequences)
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
                'max_input_seq_len': self.max_input_seq_len,
                'model_dims': self.model_dims,
                'linker_dims': self.linker_dims,
                'num_basis_list': [layer.num_basis for layer in self.hierarchical_layers],
                'linker_trainable': self.linker_trainable,
                'use_residual_list': self.use_residual_list,
                'reduction_factors': self.reduction_factors,
                'char_layer_config': {
                    'rank_mode': self.rank_mode,
                    'num_basis': self.char_num_basis,
                    'mode': self.mode,
                    'user_step': self.step
                }
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
            max_input_seq_len=config['max_input_seq_len'],
            model_dims=config['model_dims'],
            linker_dims=config['linker_dims'],
            rank_mode=char_layer_config['rank_mode'],
            num_basis=char_layer_config['num_basis'],  # char layer's num_basis
            mode=char_layer_config['mode'],
            user_step=char_layer_config['user_step'],            
            num_basis_list=config.get('num_basis_list', [char_layer_config['num_basis']] * len(config['model_dims'])),            
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
        
        print(f"Model loaded from {filename}")
        return model


# === Example Usage with Variable-Length Sequences ===
if __name__ == "__main__":
    from statistics import correlation
    
    print("="*60)
    print("HierDDLtsC - Hierarchical Dual Descriptor with Character Input and Adaptive Linker Matrices")
    print("Variable-Length Sequence Support Demonstration")
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
    num_basis_list = [6, 5, 4]  # Basis functions for each hierarchical layer
    linker_dims = [100, 50, 25]  # Maximum output sequence lengths for each hierarchical layer
    reduction_factors = [2, 2, 2]  # Sequence length reduction factors for each layer
    num_seqs = 100
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {device}")
    print(f"Character set: {charset}")
    print(f"Rank (k-mer length): {rank}")
    print(f"Character layer output dim: {vec_dim}")
    print(f"Maximum input sequence length: {max_input_seq_len}")
    print(f"Hierarchical layer dims: {model_dims}")
    print(f"Linker maximum output sequence lengths: {linker_dims}")
    print(f"Reduction factors: {reduction_factors}")
    print(f"Number of basis functions: {num_basis_list}")
    
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
    print("\nCreating HierDDLtsC model with adaptive Linker matrices...")
    model = HierDDLtsC(
        charset=charset,
        rank=rank,
        vec_dim=vec_dim,
        max_input_seq_len=max_input_seq_len,
        model_dims=model_dims,
        linker_dims=linker_dims,
        num_basis_list=num_basis_list,
        reduction_factors=reduction_factors,
        linker_trainable=[True, False, True],  # Mixed trainable settings
        use_residual_list=['separate', 'shared', None],  # Mixed residual strategies
        mode='linear',
        device=device
    )
    
    # Count parameters
    print("\nModel Parameter Count:")
    total_params, trainable_params = model.count_parameters()
    
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
    
    gd_history = model.grad_train(
        seqs, t_list,
        max_iters=50,  # Reduced for demonstration
        learning_rate=0.01,
        decay_rate=0.98,
        print_every=5,
        batch_size=8,   # Smaller batch size for variable lengths
        checkpoint_file='hierddltsc_var_gd_checkpoint.pth',
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
    
    reconstructed_seq = model.reconstruct()
    print(f"Reconstructed sequence (length {len(reconstructed_seq)}):")
    print(f"First 100 chars: {reconstructed_seq[:100]}")
    
    # Character-level reconstruction
    char_reconstructed_seq = model.char_reconstruct()
    print(f"Character-level reconstructed sequence (length {len(char_reconstructed_seq)}):")
    print(f"First 100 chars: {char_reconstructed_seq[:100]}")
    
    # Sequence generation with variable-length support
    print("\n" + "="*50)
    print("Sequence Generation with Variable-Length Support")
    print("="*50)
    
    # Generate sequences of different lengths
    for length in [50, 100, 150]:
        # Deterministic generation
        det_seq = model.generate(L=length, tau=0.0)
        print(f"Deterministic generation (length {length}, tau=0.0): {det_seq[:50]}...")
        
        # Stochastic generation
        stoch_seq = model.generate(L=length, tau=0.5)
        print(f"Stochastic generation (length {length}, tau=0.5): {stoch_seq[:50]}...")
    
    # Character-level generation
    char_gen_seq = model.char_generate(L=80, tau=0.2)
    print(f"Character-level generation (tau=0.2): {char_gen_seq}")
    
    # Auto-training example with variable-length sequences
    print("\n" + "="*50)
    print("Auto-Training Example with Variable-Length Sequences")
    print("="*50)
    
    # Create a new model for auto-training
    auto_model = HierDDLtsC(
        charset=charset,
        rank=rank,
        vec_dim=vec_dim,
        max_input_seq_len=max_input_seq_len,
        model_dims=model_dims,
        linker_dims=linker_dims,
        num_basis_list=num_basis_list,
        reduction_factors=reduction_factors,
        linker_trainable=False,
        use_residual_list=[None] * len(model_dims),
        mode='linear',
        device=device
    )
    
    # Auto-train in gap mode with variable-length sequences
    print("\nAuto-training in 'gap' mode with variable-length sequences...")
    auto_history_gap = auto_model.auto_train(
        seqs[:20],  # Use subset for faster demonstration
        max_iters=10,
        learning_rate=0.01,
        auto_mode='gap',
        print_every=2,
        checkpoint_file='hierddltsc_var_auto_checkpoint.pth',
        checkpoint_interval=5
    )
    
    # Generate from auto-trained model
    auto_gen_seq = auto_model.generate(L=80, tau=0.2)
    print(f"Generated from auto-trained model: {auto_gen_seq}")
    
    # Model save and load test
    print("\n" + "="*50)
    print("Model Save/Load Test with Variable-Length Support")
    print("="*50)
    
    # Save model
    model.save("hierddltsc_var_model.pth")
    
    # Load model
    loaded_model = HierDDLtsC.load("hierddltsc_var_model.pth", device=device)
    
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
    
    # Test generation consistency
    torch.manual_seed(123); random.seed(123); np.random.seed(123)
    original_gen = model.generate(L=50, tau=0.1)
    
    torch.manual_seed(123); random.seed(123); np.random.seed(123)
    loaded_gen = loaded_model.generate(L=50, tau=0.1)
    
    if original_gen == loaded_gen:
        print("✓ Generation consistency test PASSED")
    else:
        print("✗ Generation consistency test FAILED")
        print(f"Original: {original_gen}")
        print(f"Loaded:   {loaded_gen}")
    
    print("\nAll tests with variable-length sequences completed successfully!")
