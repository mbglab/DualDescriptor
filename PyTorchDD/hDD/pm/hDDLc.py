# Copyright (C) 2005-2025, Bin-Guang Ma (mbg@mail.hzau.edu.cn); SPDX-License-Identifier: MIT
# Hierarchical Dual Descriptor (P Matrix form) with Character Sequence Input (HierDDLpmC)
# Combines character-level processing and hierarchical vector processing with Linker matrices
# This program is for the demonstration of methodology and not fully refined.
# Author: Bin-Guang Ma (assisted by DeepSeek); Date: 2025-10-11 ~ 2026-1-16

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
    """Single layer of Hierarchical Dual Descriptor with Linker matrices"""
    def __init__(self, in_dim, out_dim, in_seq, out_seq, linker_trainable, use_residual, device):
        """
        Initialize a hierarchical layer with Linker matrix
        
        Args:
            in_dim (int): Input dimension
            out_dim (int): Output dimension
            in_seq (int): Input sequence length
            out_seq (int): Output sequence length
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
        self.use_residual = use_residual
        
        # Linear transformation matrix
        self.M = nn.Parameter(torch.empty(out_dim, in_dim))
        
        # Position-dependent transformation matrix (simplified to 2D)
        self.P = nn.Parameter(torch.empty(out_dim, out_dim))
        
        # Precompute periods matrix (fixed, not trainable)
        periods = torch.zeros(out_dim, out_dim, dtype=torch.float32)
        for i in range(out_dim):
            for j in range(out_dim):
                periods[i, j] = i * out_dim + j + 2
        self.register_buffer('periods', periods)
        
        # Sequence length transformation matrix
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
        Position-dependent transformation with simplified matrix form
        Assumes 3D input: [batch_size, seq_len, out_dim]
        """
        batch_size, seq_len, out_dim = Z.shape
        
        # Create position indices [1, seq_len, 1, 1]
        k = torch.arange(seq_len, device=self.device).float()
        k = k.view(1, seq_len, 1, 1)
        
        # Get periods and reshape for broadcasting: [1, 1, out_dim, out_dim]
        periods = self.periods.unsqueeze(0).unsqueeze(0)
        
        # Compute basis function: cos(2πk/period) -> [1, seq_len, out_dim, out_dim]
        phi = torch.cos(2 * math.pi * k / periods)
        
        # Prepare Z for broadcasting: [batch_size, seq_len, 1, out_dim]
        Z_exp = Z.unsqueeze(2)
        
        # Prepare P for broadcasting: [1, 1, out_dim, out_dim]
        P_exp = self.P.unsqueeze(0).unsqueeze(0)
        
        # Compute position transformation: Z * P * phi
        # Dimensions: 
        #   Z_exp: [batch_size, seq_len, 1, out_dim]
        #   P_exp: [1, 1, out_dim, out_dim]
        #   phi:   [1, seq_len, out_dim, out_dim]
        # Result: [batch_size, seq_len, out_dim, out_dim]
        M = Z_exp * P_exp * phi
        
        # Sum over j (dim=3) -> [batch_size, seq_len, out_dim]
        T = torch.sum(M, dim=3)
        return T
    
    def sequence_transform(self, T):
        """
        Transform sequence length using Linker matrix
        Assumes 3D input: [batch_size, in_seq, out_dim]
        """
        batch_size, in_seq, out_dim = T.shape
        
        # Transpose to [batch_size, out_dim, in_seq]
        T_transposed = T.transpose(1, 2)
        
        # Apply Linker transformation: [batch_size, out_dim, in_seq] @ [in_seq, out_seq] = [batch_size, out_dim, out_seq]
        U = torch.matmul(T_transposed, self.Linker)
        
        # Transpose back to [batch_size, out_seq, out_dim]
        return U.transpose(1, 2)
    
    def forward(self, x):
        """
        Forward pass through the layer
        Assumes 3D input: [batch_size, seq_len, in_dim]
        """
        batch_size, seq_len, in_dim = x.shape
        
        # Main path: linear transformation
        Z = torch.matmul(x, self.M.t())  # [batch_size, seq_len, out_dim]

        # Layer normalization
        Z = self.ln(Z)
        
        # Position-dependent transformation
        T = self.position_transform(Z)  # [batch_size, seq_len, out_dim]
        
        # Sequence transformation
        U = self.sequence_transform(T)  # [batch_size, out_seq, out_dim]
        
        # Residual connection processing
        residual = 0
        if self.use_residual == 'separate':
            # Separate projection and Linker for residual
            residual_feat = self.residual_proj(x)  # [batch_size, seq_len, out_dim]
            residual = self.sequence_transform(residual_feat)  # [batch_size, out_seq, out_dim]
        elif self.use_residual == 'shared':
            # Shared M and Linker for residual
            residual_feat = torch.matmul(x, self.M.t())  # [batch_size, seq_len, out_dim]
            residual = self.sequence_transform(residual_feat)  # [batch_size, out_seq, out_dim]
        else:
            # Identity residual only if dimensions match
            if self.in_dim == self.out_dim and self.in_seq == self.out_seq:
                residual = self.sequence_transform(x)  # [batch_size, out_seq, out_dim]
        
        # Add residual connection
        out = U + residual
        return out

class HierDDLpmC(nn.Module):
    """
    Hierarchical Dual Descriptor with Linker Matrices and Character Sequence Input
    Combines character-level processing and hierarchical layers for vector processing with Linker matrices for sequence length transformation
    Assumes all input character sequences have the same length
    """
    def __init__(self, charset, rank=1, rank_mode='drop', vec_dim=2, 
                 mode='linear', user_step=None, input_seq_len=100, 
                 model_dims=[2], linker_dims=[50], linker_trainable=False,
                 use_residual_list=None, device='cuda'):
        """
        Initialize HierDDLpmC model with embedded character-level processing and Linker matrices
        
        Args:
            charset: Character set for sequence input
            rank: k-mer length for tokenization
            rank_mode: 'pad' or 'drop' for incomplete k-mers
            vec_dim: Output dimension of character layer (Layer 0)
            mode: 'linear' or 'nonlinear' tokenization
            user_step: Step size for nonlinear tokenization
            input_seq_len: Fixed input sequence length after character processing
            model_dims: Output dimensions for hierarchical layers
            linker_dims: Output sequence lengths for hierarchical layers
            linker_trainable: Whether Linker matrices are trainable
            use_residual_list: Residual connection types for hierarchical layers
            device: Computation device
        """
        super().__init__()
        self.charset = list(charset)
        self.rank = rank    # r-per/k-mer length
        self.rank_mode = rank_mode # 'pad' or 'drop'
        self.vec_dim = vec_dim    # embedding dimension
        self.mode = mode
        self.step = user_step
        self.input_seq_len = input_seq_len
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
        
        # Calculate token sequence length from character sequence length
        self.token_seq_len = self._calculate_token_seq_len(input_seq_len)
        
        # Embedding layer for tokens
        self.char_embedding = nn.Embedding(len(self.tokens), self.vec_dim)
        
        # 2D position matrix P[i][j] for character layer
        self.char_P = nn.Parameter(torch.empty(self.vec_dim, self.vec_dim))
        
        # Precompute indexed periods[i][j] for character layer (fixed, not trainable)
        char_periods = torch.zeros(self.vec_dim, self.vec_dim, dtype=torch.float32)
        for i in range(self.vec_dim):
            for j in range(self.vec_dim):
                char_periods[i, j] = i * self.vec_dim + j + 2
        self.register_buffer('char_periods', char_periods)

        # Training statistics for character layer
        self.mean_token_count = None
        self.mean_t = None
        
        # Classification and multi-label heads (initialized later)
        self.num_classes = None
        self.classifier = None
        self.num_labels = None
        self.labeller = None
        
        # Hierarchical layers (starting from Layer 1)
        self.hierarchical_layers = nn.ModuleList()
        for l in range(self.num_layers):
            # Determine input dimensions for this layer
            in_dim = vec_dim if l == 0 else model_dims[l-1]
            in_seq = self.token_seq_len if l == 0 else linker_dims[l-1]  # 使用token_seq_len而不是input_seq_len
            out_dim = model_dims[l]
            out_seq = linker_dims[l]
            use_residual = self.use_residual_list[l]
            
            # Initialize layer
            layer = Layer(
                in_dim, out_dim, in_seq, out_seq, 
                self.linker_trainable[l], 
                use_residual, self.device
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
    
    def batch_token_indices(self, seqs, device=None):
        """
        Vectorized conversion of a batch of strings to token indices tensor
        
        Args:
            seqs (list): List of input character strings
            device (torch.device): Target device (defaults to self.device)
            
        Returns:
            torch.Tensor: LongTensor of shape [batch_size, token_seq_len]
        """
        target_device = device if device is not None else self.device
        
        # Prepare tokenization parameters
        step = 1 if self.mode == 'linear' else (self.step or self.rank)
        L = self.input_seq_len
        
        all_indices = []
        
        # Vectorized tokenization for all sequences
        if self.mode == 'linear':
            ranges = range(L - self.rank + 1)
            for seq in seqs:
                # Direct string slicing for each position
                toks = [seq[i:i+self.rank] for i in ranges]
                all_indices.append([self.token_to_idx[t] for t in toks])
                
        else:  # nonlinear mode
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
                all_indices.append(seq_indices)

        if not all_indices:
            return torch.zeros((len(seqs), 0), dtype=torch.long, device=target_device)
            
        return torch.tensor(all_indices, dtype=torch.long, device=target_device)
    
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
        Vectorized computation of N(k) vectors for a batch of tokens using P matrix form
        
        Args:
            token_indices: Tensor of token indices [batch_size, seq_len] or [seq_len]
            
        Returns:
            Tensor of N(k) vectors [batch_size, seq_len, vec_dim] or [seq_len, vec_dim]
        """
        # Handle single sequence case
        if token_indices.dim() == 1:
            token_indices = token_indices.unsqueeze(0)
            unsqueeze_output = True
        else:
            unsqueeze_output = False
            
        batch_size, seq_len = token_indices.shape
        
        # Get embeddings for all tokens [batch_size, seq_len, vec_dim]
        x = self.char_embedding(token_indices)
        
        # Expand dimensions for broadcasting [batch_size, seq_len, 1, 1]
        k_positions = torch.arange(seq_len, device=self.device).float()
        k_expanded = k_positions.view(1, seq_len, 1, 1).expand(batch_size, seq_len, 1, 1)
        
        # Calculate basis functions: cos(2π*k/periods) [batch_size, seq_len, vec_dim, vec_dim]
        phi = torch.cos(2 * math.pi * k_expanded / self.char_periods)
        
        # Prepare x for broadcasting: [batch_size, seq_len, 1, vec_dim]
        x_expanded = x.unsqueeze(2)
        
        # Prepare P for broadcasting: [1, 1, vec_dim, vec_dim]
        P_expanded = self.char_P.unsqueeze(0).unsqueeze(0)
        
        # Optimized computation using einsum: x * P * phi
        # Result: [batch_size, seq_len, vec_dim, vec_dim]
        M = x_expanded * P_expanded * phi
        
        # Sum over the last dimension (j) -> [batch_size, seq_len, vec_dim]
        Nk = torch.sum(M, dim=3)
            
        # Remove batch dimension if input was single sequence
        if unsqueeze_output:
            Nk = Nk.squeeze(0)
            
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
            # Create position tensor for all tokens at position k
            position_specific_indices = all_token_indices  # All tokens at this position
            
            # Create position tensor
            k_tensor = torch.tensor([k] * len(self.tokens), dtype=torch.float32, device=self.device)
            
            # Compute Nk for all tokens at position k
            # For batch computation, we need to reshape
            position_indices = torch.tensor([k] * len(self.tokens), device=self.device)
            # Use single token batch computation
            Nk_all = self.char_batch_compute_Nk(position_specific_indices.unsqueeze(0))
            Nk_all = Nk_all.squeeze(0)  # Remove batch dimension
            
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
            # Use batch computation
            Nk_all = self.char_batch_compute_Nk(position_specific_indices.unsqueeze(0))
            Nk_all = Nk_all.squeeze(0)  # Remove batch dimension
            
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
            seq (str): Input character sequence (must have length = input_seq_len)
            
        Returns:
            Tensor: Sequence tensor of shape [1, token_seq_len, vec_dim]
        """
        # Validate sequence length
        if len(seq) != self.input_seq_len:
            raise ValueError(f"Input sequence length must be {self.input_seq_len}, got {len(seq)}")
        
        # Use batch processing for consistency
        return self.batch_sequence_to_tensor([seq])
    
    def batch_sequence_to_tensor(self, seqs):
        """
        Convert batch of character sequences to tensor representation
        
        Args:
            seqs (list): List of input character sequences
            
        Returns:
            Tensor: Sequence tensor of shape [batch_size, token_seq_len, vec_dim]
        """
        # Convert strings to token indices
        indices = self.batch_token_indices(seqs)
        
        # Compute N(k) vectors through char layer
        return self.char_batch_compute_Nk(indices)
    
    def forward(self, input_data):
        """
        Forward pass through entire hierarchical model
        
        Args:
            input_data: Input character sequence(s) - can be:
                - str: Single character sequence
                - list: List of character sequences
                - torch.Tensor: Pre-computed token indices [batch_size, seq_len]
            
        Returns:
            Tensor: Output of shape [batch_size, final_linker_dim, final_model_dim]
        """
        # Handle different input types
        if isinstance(input_data, str):
            # Single sequence
            x = self.char_sequence_to_tensor(input_data)
        elif isinstance(input_data, list):
            # Multiple sequences
            x = self.batch_sequence_to_tensor(input_data)
        elif isinstance(input_data, torch.Tensor):
            # Pre-computed token indices
            if input_data.dim() == 1:
                input_data = input_data.unsqueeze(0)
            x = self.char_batch_compute_Nk(input_data.to(self.device))
        else:
            raise TypeError("Input must be str, list of str, or Tensor")
        
        # Pass through hierarchical layers with Linker matrices
        # x is already 3D: [batch_size, token_seq_len, vec_dim]
        for layer in self.hierarchical_layers:
            x = layer(x)  # Layer now handles 3D input
            
        return x
    
    def predict_t(self, seq):
        """
        Predict target vector for a character sequence
        Returns the average of all output vectors in the sequence
        
        Args:
            seq (str): Input character sequence of fixed length input_seq_len
            
        Returns:
            numpy.array: Predicted target vector
        """
        with torch.no_grad():
            output = self.forward(seq)  # [1, final_linker_dim, final_model_dim]
            target = output.mean(dim=1)  # Average over sequence length
            return target.squeeze(0).cpu().numpy()
    
    def cls_train(self, seqs, labels, num_classes, max_iters=1000, tol=1e-8, learning_rate=0.01,
                  continued=False, decay_rate=1.0, print_every=10, batch_size=32,
                  checkpoint_file=None, checkpoint_interval=10):
        """
        Train the model for multi-class classification using cross-entropy loss.
        
        Args:
            seqs: List of character sequences (all must have length = input_seq_len)
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
            if len(seq) != self.input_seq_len:
                raise ValueError(f"All sequences must have length {self.input_seq_len}")
        
        # Initialize classification head if not already done or if num_classes changed
        if self.classifier is None or self.num_classes != num_classes:
            final_dim = self.model_dims[-1] if self.model_dims else self.vec_dim
            self.classifier = nn.Linear(final_dim, num_classes).to(self.device)
            self.num_classes = num_classes
        
        if not continued:
            self.reset_parameters()
        
        # Convert labels to tensor
        label_tensors = torch.tensor(labels, dtype=torch.long, device=self.device)
        
        # Pre-process token indices for all sequences
        all_indices = self.batch_token_indices(seqs, device='cpu')
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
            total_sequences = 0
            correct_predictions = 0
            
            # Shuffle sequences for each epoch
            perm = torch.randperm(num_samples)
            
            # Process sequences in batches
            for i in range(0, num_samples, batch_size):
                batch_idx = perm[i:i+batch_size]
                batch_indices = all_indices[batch_idx].to(self.device)
                batch_labels = label_tensors[batch_idx]
                
                optimizer.zero_grad()
                
                # Forward pass for the whole batch
                output = self.forward(batch_indices)  # [batch_size, final_linker_dim, final_model_dim]
                
                # Get sequence representations by averaging over sequence dimension
                seq_representations = output.mean(dim=1)  # [batch_size, final_model_dim]
                
                # Get logits through classification head
                logits = self.classifier(seq_representations)  # [batch_size, num_classes]
                
                # Compute loss
                loss = criterion(logits, batch_labels)
                loss.backward()
                optimizer.step()
                
                # Calculate batch statistics
                batch_loss = loss.item()
                total_loss += batch_loss * len(batch_idx)
                total_sequences += len(batch_idx)
                
                # Calculate accuracy
                with torch.no_grad():
                    predictions = torch.argmax(logits, dim=1)
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
            seqs: List of character sequences (all must have length = input_seq_len)
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
            if len(seq) != self.input_seq_len:
                raise ValueError(f"All sequences must have length {self.input_seq_len}")
        
        # Initialize label head if not already done or if num_labels changed
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
        
        # Pre-process token indices for all sequences
        all_indices = self.batch_token_indices(seqs, device='cpu')
        num_samples = len(seqs)
        
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
            perm = torch.randperm(num_samples)
            
            # Process sequences in batches
            for i in range(0, num_samples, batch_size):
                batch_idx = perm[i:i+batch_size]
                batch_indices = all_indices[batch_idx].to(self.device)
                batch_labels = labels_tensor[batch_idx]
                
                optimizer.zero_grad()
                
                # Forward pass for the whole batch
                output = self.forward(batch_indices)  # [batch_size, final_linker_dim, final_model_dim]
                
                # Get sequence representations by averaging over sequence dimension
                seq_representations = output.mean(dim=1)  # [batch_size, final_model_dim]
                
                # Pass through classification head to get logits
                logits = self.labeller(seq_representations)  # [batch_size, num_labels]
                
                # Calculate loss for the batch
                batch_loss = criterion(logits, batch_labels)
                
                # Calculate accuracy
                with torch.no_grad():
                    # Apply sigmoid to get probabilities
                    probs = torch.sigmoid(logits)
                    # Threshold at 0.5 for binary predictions
                    predictions = (probs > 0.5).float()
                    # Calculate number of correct predictions
                    batch_correct = (predictions == batch_labels).sum().item()
                    batch_predictions = batch_labels.numel()
                
                # Backpropagate
                batch_loss.backward()
                optimizer.step()
                
                total_loss += batch_loss.item() * len(batch_idx)
                total_correct += batch_correct
                total_predictions += batch_predictions
                total_sequences += len(batch_idx)
                
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
        
        with torch.no_grad():
            # Get sequence vector representation through hierarchical model
            output = self.forward(seq)  # [1, final_linker_dim, final_model_dim]
            seq_vector = output.mean(dim=1)  # [1, final_model_dim]
            
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
        
        with torch.no_grad():
            # Get sequence vector representation through hierarchical model
            output = self.forward(seq)  # [1, final_linker_dim, final_model_dim]
            seq_vector = output.mean(dim=1)  # [1, final_model_dim]
            
            # Pass through classification head to get logits
            logits = self.labeller(seq_vector)
            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(logits).cpu().numpy()
        
        # Apply threshold to get binary predictions
        binary_preds = (probs > threshold).astype(np.float32)
        
        return binary_preds[0], probs[0]
    
    def reg_train(self, seqs, t_list, max_iters=1000, tol=1e-8, learning_rate=0.01, 
                   continued=False, decay_rate=1.0, print_every=10, batch_size=1024,
                   checkpoint_file=None, checkpoint_interval=10):
        """
        Train the model using gradient descent with target vectors
        
        Args:
            seqs: List of character sequences (all must have length = input_seq_len)
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
            if len(seq) != self.input_seq_len:
                raise ValueError(f"All sequences must have length {self.input_seq_len}")
        
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
        
        # Pre-process token indices for all sequences
        all_indices = self.batch_token_indices(seqs, device='cpu')
        num_samples = len(seqs)
        
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
            
            # Shuffle indices
            perm = torch.randperm(num_samples)
            
            for i in range(0, num_samples, batch_size):
                batch_idx = perm[i:i+batch_size]
                batch_indices = all_indices[batch_idx].to(self.device)
                batch_targets = t_tensors[batch_idx]
                
                optimizer.zero_grad()
                
                # Forward pass for the whole batch
                output = self.forward(batch_indices)
                pred_targets = output.mean(dim=1)  # [batch_size, final_model_dim]
                
                # Compute loss
                loss = torch.sum((pred_targets - batch_targets) ** 2)
                batch_loss_val = loss.item()
                
                # Normalize by batch size
                loss = loss / len(batch_idx)
                loss.backward()
                optimizer.step()
                
                total_loss += batch_loss_val
            
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
    
    def self_train(self, seqs, max_iters=100, tol=1e-6, learning_rate=0.01, 
                   continued=False, self_mode='gap', decay_rate=1.0, print_every=10,
                   batch_size = 1024, checkpoint_file=None, checkpoint_interval=5):
        """
        Self-training method for the hierarchical model with Linker matrices
        
        Args:
            seqs: List of character sequences (all must have length = input_seq_len)
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
        # Validate input sequences
        for seq in seqs:
            if len(seq) != self.input_seq_len:
                raise ValueError(f"All sequences must have length {self.input_seq_len}")
        
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
        
        # Pre-process token indices for all sequences
        all_indices = self.batch_token_indices(seqs, device='cpu')
        num_samples = len(seqs)        
        
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
            
            # Shuffle indices
            perm = torch.randperm(num_samples)
            
            # Apply causal masks if in 'reg' mode
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
            
            # Process batches
            for i in range(0, num_samples, batch_size):
                batch_idx = perm[i:i+batch_size]
                batch_indices = all_indices[batch_idx].to(self.device)
                
                optimizer.zero_grad()
                
                # Compute character layer output
                char_output = self.char_batch_compute_Nk(batch_indices)
                
                # Forward through hierarchical layers
                hier_out = char_output
                for layer in self.hierarchical_layers:
                    hier_out = layer(hier_out)
                
                # Compute loss based on self_mode
                loss = 0.0
                min_len = min(char_output.shape[1], hier_out.shape[1])
                
                if self_mode == 'gap':
                    # Self-consistency: hierarchical output should match character layer output
                    if min_len > 0:
                        diff = hier_out[:, :min_len, :] - char_output[:, :min_len, :]
                        sq_diff = torch.sum(diff ** 2, dim=-1)
                        loss = torch.sum(torch.mean(sq_diff, dim=1))
                else:  # 'reg' mode
                    # Predict next position's character layer output
                    if min_len > 1:
                        targets = char_output[:, 1:min_len, :]
                        preds = hier_out[:, :min_len-1, :]
                        diff = preds - targets
                        sq_diff = torch.sum(diff ** 2, dim=-1)
                        loss = torch.sum(torch.mean(sq_diff, dim=1))
                
                batch_loss_scalar = loss.item()
                
                # Backward pass if loss > 0
                if batch_loss_scalar > 0:
                    (loss / len(batch_idx)).backward()
                    optimizer.step()
                
                total_loss += batch_loss_scalar
            
            # Restore original weights if in 'reg' mode
            if self_mode == 'reg':
                for idx, layer in enumerate(self.hierarchical_layers):
                    if hasattr(layer, 'Linker'):
                        layer.Linker.data.copy_(original_linkers[idx])
                        if original_residual_linkers[idx] is not None:
                            layer.residual_linker.data.copy_(original_residual_linkers[idx])
            
            avg_loss = total_loss / num_samples
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
                'input_seq_len': self.input_seq_len,
                'model_dims': self.model_dims,
                'linker_dims': self.linker_dims,
                'linker_trainable': self.linker_trainable,
                'use_residual_list': self.use_residual_list,
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

    def _compute_training_statistics(self, seqs, batch_size=256):
        """Compute and store statistics for reconstruction and generation"""
        total_token_count = 0
        total_vec_sum = torch.zeros(self.vec_dim, device=self.device)
        
        # Pre-process all sequences
        all_indices = self.batch_token_indices(seqs, device='cpu')
        
        with torch.no_grad():
            for i in range(0, len(seqs), batch_size):
                batch_indices = all_indices[i:i+batch_size].to(self.device)
                
                # Get character layer output for the batch
                nk_batch = self.char_batch_compute_Nk(batch_indices)
                
                # Sum over batch and sequence dimensions
                total_vec_sum += nk_batch.sum(dim=[0, 1])
                
                # Count tokens
                total_token_count += batch_indices.shape[0] * batch_indices.shape[1]
        
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
        
        # Pre-process all sequences
        all_indices = self.batch_token_indices(seqs, device='cpu')
        
        with torch.no_grad():
            for i in range(0, len(seqs), batch_size):
                batch_indices = all_indices[i:i+batch_size].to(self.device)
                try:
                    output = self.forward(batch_indices)
                    
                    # Mean over sequence dimension, sum over batch
                    batch_means = output.mean(dim=1)
                    total_output += batch_means.sum(dim=0)
                    total_sequences += batch_indices.shape[0]
                    
                except RuntimeError as e:
                    # Handle out of memory errors
                    if "out of memory" in str(e):
                        print(f"Warning: The sequence batch is too large, resulting in insufficient memory. Skipping batch...")
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
        Reconstruct character sequence of length L using the entire hierarchical model with Linker matrices.
        
        This method uses temperature-controlled sampling based on the complete model's
        predictions, incorporating both character-level and hierarchical patterns with Linker matrices.
        
        Args:
            L (int): Length of sequence to generate (must be <= input_seq_len)
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
        
        if L > self.input_seq_len:
            print(f"Warning: Requested length {L} exceeds fixed input sequence length {self.input_seq_len}. Truncating.")
            L = self.input_seq_len
        
        generated_sequence = ""
        
        # Generate sequence token by token
        while len(generated_sequence) < L:
            candidate_scores = {}
            
            # Evaluate each possible token
            for token in self.tokens:
                candidate_seq = generated_sequence + token
                
                # Pad to fixed length if necessary for model input
                if len(candidate_seq) < self.input_seq_len:
                    padded_seq = candidate_seq.ljust(self.input_seq_len, 'A')
                else:
                    padded_seq = candidate_seq[:self.input_seq_len]
                    
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
        
        # Character P parameters (2D matrix)
        num = self.char_P.numel()
        shape = str(tuple(self.char_P.shape))
        print(f"{'Char Layer':<15} | {'char_P':<25} | {num:<15} | {shape:<20}")
        char_params += num
        total_params += num
        
        print(f"{'Char Layer':<15} | {'TOTAL':<25} | {char_params:<15} |")
        print("-"*70)
        
        # Hierarchical layer parameters
        for i, layer in enumerate(self.hierarchical_layers):
            layer_params = 0
            layer_name = f"Hier Layer {i}"
            
            # M matrix
            num = layer.M.numel()
            shape = str(tuple(layer.M.shape))
            print(f"{layer_name:<15} | {'M':<25} | {num:<15} | {shape:<20}")
            layer_params += num
            total_params += num
            
            # P matrix
            num = layer.P.numel()
            shape = str(tuple(layer.P.shape))
            print(f"{layer_name:<15} | {'P':<25} | {num:<15} | {shape:<20}")
            layer_params += num
            total_params += num
            
            # Linker matrix
            num = layer.Linker.numel()
            shape = str(tuple(layer.Linker.shape))
            trainable = "T" if layer.Linker.requires_grad else "F"
            print(f"{layer_name:<15} | {'Linker':<25} | {num:<15} | {shape:<20} [{trainable}]")
            layer_params += num
            total_params += num
            
            # Residual components
            if layer.use_residual == 'separate':
                # Residual projection
                num = layer.residual_proj.weight.numel()
                shape = str(tuple(layer.residual_proj.weight.shape))
                print(f"{layer_name:<15} | {'residual_proj':<25} | {num:<15} | {shape:<20}")
                layer_params += num
                total_params += num
                
                # Residual linker
                num = layer.residual_linker.numel()
                shape = str(tuple(layer.residual_linker.shape))
                trainable = "T" if layer.residual_linker.requires_grad else "F"
                print(f"{layer_name:<15} | {'residual_linker':<25} | {num:<15} | {shape:<20} [{trainable}]")
                layer_params += num
                total_params += num
            
            print(f"{layer_name:<15} | {'TOTAL':<25} | {layer_params:<15} |")
            print("-"*70)
        
        # Classification heads
        if self.classifier is not None:
            num = self.classifier.weight.numel()
            shape = str(tuple(self.classifier.weight.shape))
            print(f"{'Classifier':<15} | {'weight':<25} | {num:<15} | {shape:<20}")
            total_params += num
            
            if self.classifier.bias is not None:
                num = self.classifier.bias.numel()
                shape = str(tuple(self.classifier.bias.shape))
                print(f"{'Classifier':<15} | {'bias':<25} | {num:<15} | {shape:<20}")
                total_params += num
        
        if self.labeller is not None:
            num = self.labeller.weight.numel()
            shape = str(tuple(self.labeller.weight.shape))
            print(f"{'Labeller':<15} | {'weight':<25} | {num:<15} | {shape:<20}")
            total_params += num
            
            if self.labeller.bias is not None:
                num = self.labeller.bias.numel()
                shape = str(tuple(self.labeller.bias.shape))
                print(f"{'Labeller':<15} | {'bias':<25} | {num:<15} | {shape:<20}")
                total_params += num
        
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
                'input_seq_len': self.input_seq_len,
                'model_dims': self.model_dims,
                'linker_dims': self.linker_dims,
                'linker_trainable': self.linker_trainable,
                'use_residual_list': self.use_residual_list,
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
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
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
            input_seq_len=config['input_seq_len'],
            model_dims=config['model_dims'],
            linker_dims=config['linker_dims'],
            linker_trainable=config['linker_trainable'],
            rank_mode=char_layer_config['rank_mode'],
            mode=char_layer_config['mode'],
            user_step=char_layer_config['user_step'],            
            use_residual_list=config.get('use_residual_list', [None] * len(config['model_dims'])),
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
        
        # Load classification info and initialize heads if needed
        if 'classification_info' in checkpoint:
            info = checkpoint['classification_info']
            if info['num_classes'] is not None:
                final_dim = model.model_dims[-1] if model.model_dims else model.vec_dim
                model.classifier = nn.Linear(final_dim, info['num_classes']).to(device)
                model.num_classes = info['num_classes']
            
            if info['num_labels'] is not None:
                final_dim = model.model_dims[-1] if model.model_dims else model.vec_dim
                model.labeller = nn.Linear(final_dim, info['num_labels']).to(device)
                model.num_labels = info['num_labels']
        
        print(f"Model loaded from {filename}")
        return model

# === Example Usage ===
if __name__ == "__main__":
    from statistics import correlation
    
    print("="*60)
    print("HierDDLpmC - Hierarchical Dual Descriptor with Linker Matrices and Character Input")
    print("Linker Matrix Model with Fixed Sequence Length")
    print("="*60)
    
    # Set random seeds for reproducibility
    torch.manual_seed(2)
    random.seed(2)
    np.random.seed(2)
    
    # Configuration
    charset = ['A', 'C', 'G', 'T']
    rank = 1
    vec_dim = 8
    input_seq_len = 200  # Fixed input sequence length
    model_dims = [16, 12, 8]  # Hierarchical layer dimensions
    linker_dims = [40, 30, 20]  # Output sequence lengths for hierarchical layers
    num_seqs = 100
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {device}")
    print(f"Character set: {charset}")
    print(f"Rank (k-mer length): {rank}")
    print(f"Character layer output dim: {vec_dim}")
    print(f"Input sequence length: {input_seq_len}")
    print(f"Hierarchical layer dims: {model_dims}")
    print(f"Linker output dims: {linker_dims}")
    print(f"Using Linker matrices for sequence length transformation")
    
    # Generate synthetic training data with fixed length
    print("\nGenerating synthetic training data...")
    seqs, t_list = [], []
    for _ in range(num_seqs):
        seq = ''.join(random.choices(charset, k=input_seq_len))
        seqs.append(seq)
        # Create target vector (final layer dimension)
        t_list.append([random.uniform(-1.0, 1.0) for _ in range(model_dims[-1])])
    
    # Create model
    print("\nCreating HierDDLpmC model...")
    model = HierDDLpmC(
        charset=charset,
        rank=rank,
        vec_dim=vec_dim,
        input_seq_len=input_seq_len,
        model_dims=model_dims,
        linker_dims=linker_dims,
        linker_trainable=[True, False, True],  # Mixed trainable settings
        mode='nonlinear',
        user_step=1,
        use_residual_list=['separate', 'shared', None],  # Mixed residual strategies
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
        learning_rate=0.01,
        decay_rate=0.99,
        print_every=5,
        batch_size=32,
        checkpoint_file='hddlpmc_gd_checkpoint.pth',
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
    stoch_seq = model.reconstruct(L=100, tau=0.5)
    print(f"Stochastic reconstruction (tau=0.5): {stoch_seq}")
    
    # Self-training example
    print("\n" + "="*50)
    print("Self-Training Example")
    print("="*50)
    
    # Create a new model for self-training
    self_model = HierDDLpmC(
        charset=charset,
        rank=rank,
        vec_dim=vec_dim,
        input_seq_len=input_seq_len,
        model_dims=model_dims,
        linker_dims=linker_dims,
        linker_trainable=[True, True, True],
        mode='nonlinear',
        user_step=1,
        device=device
    )
    
    # Use sequences for self-training
    self_seqs = seqs[:50]  # Use subset for self-training
    
    # Self-train in gap mode
    print("\nSelf-training in 'gap' mode...")
    self_history_gap = self_model.self_train(
        self_seqs,
        max_iters=30,
        learning_rate=0.01,
        self_mode='gap',
        print_every=5,
        batch_size = 256,
        checkpoint_file='hddlpmc_self_checkpoint.pth',
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
    model.save("hierddlpmc_model.pth")
    
    # Load model
    loaded_model = HierDDLpmC.load("hierddlpmc_model.pth", device=device)
    
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
    
    # Character-level functionality test
    print("\n" + "="*50)
    print("Character-Level Functionality Test")
    print("="*50)
    
    test_seq = "ACGTACGTACGT"
    char_vectors = model.char_describe(test_seq)
    print(f"Character vectors for '{test_seq}': shape {char_vectors.shape}")
    
    char_recon = model.char_reconstruct()
    print(f"Character-level reconstruction: {char_recon[:50]}...")
    
    char_gen = model.char_generate(L=30, tau=0.1)
    print(f"Character-level generation: {char_gen}")
    
    # Classification task example
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
            L = input_seq_len
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
    
    # Initialize new model for classification
    cls_model = HierDDLpmC(
        charset=charset,
        rank=rank,
        vec_dim=vec_dim,
        input_seq_len=input_seq_len,
        model_dims=model_dims,
        linker_dims=linker_dims,
        linker_trainable=[True, False, True],
        mode='nonlinear',
        user_step=1,
        device=device
    )
    
    # Train for classification
    print("\nTraining for classification...")
    cls_history = cls_model.cls_train(
        class_seqs, class_labels, num_classes,
        max_iters=50,
        learning_rate=0.01,
        decay_rate=0.99,
        print_every=5,
        batch_size=32
    )
    
    # Test classification predictions
    print("\n" + "="*50)
    print("Classification Prediction Results")
    print("="*50)
    
    correct = 0
    for i, (seq, true_label) in enumerate(zip(class_seqs[:10], class_labels[:10])):
        pred_class, probs = cls_model.predict_c(seq)
        is_correct = pred_class == true_label
        if is_correct:
            correct += 1
        print(f"Seq {i+1}: True={true_label}, Pred={pred_class}, Probs={[f'{p:.3f}' for p in probs]}, Correct={is_correct}")
    
    accuracy = correct / 10
    print(f"\nAccuracy on first 10 sequences: {accuracy:.4f} ({correct}/10)")
    
    # Multi-label classification example
    print("\n" + "="*50)
    print("Multi-Label Classification Example")
    print("="*50)
    
    # Generate multi-label data
    num_labels = 4
    label_seqs = []
    labels = []
    
    for _ in range(100):
        L = input_seq_len
        seq = ''.join(random.choices(charset, k=L))
        label_seqs.append(seq)
        # Create random binary labels (multi-label classification)
        # Each sequence can have 0-4 active labels
        label_vec = [random.random() > 0.7 for _ in range(num_labels)]
        labels.append([1.0 if x else 0.0 for x in label_vec])
    
    # Initialize new model for multi-label classification
    lbl_model = HierDDLpmC(
        charset=charset,
        rank=rank,
        vec_dim=vec_dim,
        input_seq_len=input_seq_len,
        model_dims=model_dims,
        linker_dims=linker_dims,
        linker_trainable=[True, False, True],
        mode='nonlinear',
        user_step=1,
        device=device
    )
    
    # Train for multi-label classification
    print("\nTraining for multi-label classification...")
    loss_history, acc_history = lbl_model.lbl_train(
        label_seqs, labels, num_labels,
        max_iters=50,
        learning_rate=0.01,
        decay_rate=0.99,
        print_every=5,
        batch_size=32
    )
    
    print(f"\nFinal training loss: {loss_history[-1]:.6f}")
    print(f"Final training accuracy: {acc_history[-1]:.4f}")
    
    # Test multi-label predictions
    print("\n" + "="*50)
    print("Multi-Label Prediction Results")
    print("="*50)
    
    label_names = ["Function_A", "Function_B", "Function_C", "Function_D"]
    
    for i, (seq, true_labels) in enumerate(zip(label_seqs[:5], labels[:5])):
        binary_pred, probs = lbl_model.predict_l(seq, threshold=0.5)
        
        print(f"\nSequence {i+1}:")
        print(f"True labels: {true_labels}")
        print(f"Predicted binary: {binary_pred}")
        print(f"Predicted probabilities: {[f'{p:.4f}' for p in probs]}")
        
        # Interpret predictions
        print("Interpretation:")
        for j, (binary, prob) in enumerate(zip(binary_pred, probs)):
            status = "ACTIVE" if binary > 0.5 else "INACTIVE"
            print(f"  {label_names[j]}: {status} (confidence: {prob:.4f})")
    
    # Create a test sequence
    test_seq = "".join(random.choices(charset, k=input_seq_len))
    print(f"\nTest sequence (first 50 chars): {test_seq[:50]}...")
    
    # Predict labels for test sequence
    binary_pred, probs_pred = lbl_model.predict_l(test_seq, threshold=0.5)
    print(f"Predicted binary labels: {binary_pred}")
    print(f"Predicted probabilities: {[f'{p:.4f}' for p in probs_pred]}")
    
    print("\nAll tests completed successfully!")
