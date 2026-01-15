# Copyright (C) 2005-2025, Bin-Guang Ma (mbg@mail.hzau.edu.cn); SPDX-License-Identifier: MIT
# Hierarchical Dual Descriptor (AB matrix form) and Character Sequence Input (HierDDabC)
# Combines character-level processing and hierarchical vector processing
# This program is for the demonstration of methodology and not fully refined.
# Author: Bin-Guang Ma (assisted by DeepSeek); Date: 2025-10-11 ~ 2026-1-13

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
    Single layer of Hierarchical Dual Descriptor with AB matrix form
    Uses the AB matrix architecture from HierDDab with character sequence input
    """
    def __init__(self, in_dim, out_dim, basis_dim, use_residual, device='cuda'):
        """
        Initialize a hierarchical layer with AB matrix form
        
        Args:
            in_dim (int): Input dimension
            out_dim (int): Output dimension
            basis_dim (int): Basis dimension for this layer
            use_residual (str or None): Residual connection type
            device (str): Computation device
        """
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.basis_dim = basis_dim
        self.use_residual = use_residual
        self.device = device
        
        # Linear transformation
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        
        # Layer normalization
        self.norm = nn.LayerNorm(out_dim)
        
        # Coefficient matrix: out_dim x basis_dim (learnable)
        self.Acoeff = nn.Parameter(torch.Tensor(out_dim, basis_dim))
        
        # Fixed basis matrix B (basis_dim x out_dim)
        Bbasis = torch.empty(basis_dim, out_dim)
        for k in range(basis_dim):
            for i in range(out_dim):
                Bbasis[k, i] = math.cos(2 * math.pi * (k+1) / (i+2))
        self.register_buffer('Bbasis', Bbasis)  # Fixed, non-trainable buffer
        
        # Initialize parameters
        nn.init.uniform_(self.linear.weight, -0.1, 0.1)
        nn.init.uniform_(self.Acoeff, -0.1, 0.1)
        
        # Handle residual connection based on mode
        if use_residual == 'separate':
            # Separate linear projection for residual
            self.residual_proj = nn.Linear(in_dim, out_dim, bias=False)
            nn.init.uniform_(self.residual_proj.weight, -0.1, 0.1)
        else:
            self.residual_proj = None
            
        self.to(device)
    
    def forward(self, x):
        """
        Forward pass through the layer with AB matrix form
        
        Args:
            x (torch.Tensor): Input sequence of shape (batch_size, seq_len, in_dim)
        
        Returns:
            torch.Tensor: Output sequence of shape (batch_size, seq_len, out_dim)
        """
        batch_size, seq_len, _ = x.shape
        
        # Compute residual based on mode
        residual = 0
        if self.use_residual == 'separate':
            residual = self.residual_proj(x)
        elif self.use_residual == 'shared':
            residual = self.linear(x)  # Use shared linear transformation
        else:  # None or other
            if self.in_dim == self.out_dim:
                residual = x  # Identity connection
        
        # Main path: linear transformation
        transformed = self.linear(x)
        
        # Apply layer normalization
        normalized = self.norm(transformed)
        
        # Compute basis indices: j = k % basis_dim for each position
        j_indices = torch.arange(seq_len) % self.basis_dim
        j_indices = j_indices.to(self.device)
        
        # Select basis vectors: (seq_len, out_dim)
        B_vectors = self.Bbasis[j_indices]  # (seq_len, out_dim)
        
        # Expand for batch processing
        B_vectors_exp = B_vectors.unsqueeze(0).expand(batch_size, -1, -1)  # (batch_size, seq_len, out_dim)
        
        # Compute scalars: dot product of normalized and B_vectors
        scalars = torch.einsum('bsd,bsd->bs', normalized, B_vectors_exp)  # (batch_size, seq_len)
        
        # Select coefficient vectors: (seq_len, out_dim)
        A_vectors = self.Acoeff[:, j_indices].t()  # (seq_len, out_dim)
        A_vectors_exp = A_vectors.unsqueeze(0).expand(batch_size, -1, -1)  # (batch_size, seq_len, out_dim)
        
        # Compute new features: scalars * A_vectors
        new_features = scalars.unsqueeze(2) * A_vectors_exp  # (batch_size, seq_len, out_dim)
        
        # Add residual connection
        output = new_features + residual
        
        return output

class HierDDabC(nn.Module):
    """
    Hierarchical Dual Descriptor with AB Matrix and Character Sequence Input
    Combines character-level processing and hierarchical AB matrix layers
    """
    def __init__(self, charset, rank=1, rank_mode='drop', vec_dim=2, 
                 mode='linear', user_step=None, model_dims=[2], basis_dims=[50],
                 use_residual_list=None, device='cuda'):
        """
        Initialize HierDDabC model with embedded character-level processing and AB matrix layers
        
        Args:
            charset: Character set for sequence input
            rank: k-mer length for tokenization
            rank_mode: 'pad' or 'drop' for incomplete k-mers
            vec_dim: Output dimension of character layer (Layer 0)
            mode: 'linear' or 'nonlinear' tokenization
            user_step: Step size for nonlinear tokenization
            model_dims: Output dimensions for hierarchical layers
            basis_dims: Basis dimensions for hierarchical layers
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
        self.model_dims = model_dims
        self.basis_dims = basis_dims
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
        
        # Hierarchical layers (starting from Layer 1)
        self.hierarchical_layers = nn.ModuleList()
        in_dim = vec_dim  # Input to first hierarchical layer is output from char layer
        
        if use_residual_list is None:
            use_residual_list = ['separate'] * len(model_dims)
            
        if len(basis_dims) != len(model_dims):
            raise ValueError("basis_dims length must match model_dims length")
            
        for out_dim, basis_dim, use_residual in zip(model_dims, basis_dims, use_residual_list):
            layer = Layer(in_dim, out_dim, basis_dim, use_residual, self.device)
            self.hierarchical_layers.append(layer)
            in_dim = out_dim  # Next layer input is current layer output
        
        # Mean target vector for the final hierarchical layer output
        self.mean_target = None
        
        # Initialize parameters
        self.reset_parameters()
        self.to(self.device)
    
    def token_to_indices(self, token_list):
        """Convert list of tokens to tensor of indices"""
        return torch.tensor([self.token_to_idx[tok] for tok in token_list], device=self.device)
    
    def batch_token_indices(self, seqs):
        """
        Optimization: Convert a batch of strings to indices tensor in one go.
        Performs tokenization on CPU using list comprehensions (fast for strings)
        then moves to target device as a single tensor.
        
        Args:
            seqs (list): List of strings.
        
        Returns:
            Tensor: LongTensor of shape [batch_size, token_seq_len]
        """
        all_indices = []
        
        for seq in seqs:
            toks = self.extract_tokens(seq)
            all_indices.append([self.token_to_idx[t] for t in toks])
        
        if not all_indices:
            return torch.zeros((len(seqs), 0), dtype=torch.long, device=self.device)
        
        # Find max length for padding
        max_len = max(len(indices) for indices in all_indices)
        
        # Pad sequences to same length
        padded_indices = []
        for indices in all_indices:
            if len(indices) < max_len:
                # Pad with 0 (index of first token)
                padded = indices + [0] * (max_len - len(indices))
            else:
                padded = indices
            padded_indices.append(padded)
        
        return torch.tensor(padded_indices, dtype=torch.long, device=self.device)
    
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
        Optimization: Vectorized computation of N(k) vectors for a BATCH of sequences.
        
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
        
        for k in range(n_tokens):
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
        if tau < 0:
            raise ValueError("Temperature must be non-negative")
            
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
        if '_' in full_seq:
            full_seq = full_seq.replace('_', '')
        return full_seq[:L]
    
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
        """
        Predict target vector for a character sequence
        Returns the average of all output vectors in the sequence
        
        Args:
            seq (str): Input character sequence
            
        Returns:
            numpy.array: Predicted target vector
        """
        with torch.no_grad():
            output = self.forward(seq)  # [1, seq_len, final_dim]
            target = output.mean(dim=1)  # Average over sequence length
            return target.squeeze(0).cpu().numpy()
    
    def reg_train(self, seqs, t_list, max_iters=1000, tol=1e-8, learning_rate=0.01, 
                   continued=False, decay_rate=1.0, print_every=10, batch_size=1024,
                   checkpoint_file=None, checkpoint_interval=10):
        """
        Train the model using gradient descent with target vectors
        
        Args:
            seqs: List of character sequences
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
        
        # Pre-process data: convert strings to indices on CPU
        all_indices = self.batch_token_indices(seqs)  # On CPU
        
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
            num_samples = len(seqs)
            
            # Shuffle indices
            perm = torch.randperm(num_samples)
            
            for i in range(0, num_samples, batch_size):
                batch_idx = perm[i:i+batch_size]
                batch_indices = all_indices[batch_idx].to(self.device)
                batch_t = t_tensors[batch_idx].to(self.device) # [B, TargetDim]
                
                optimizer.zero_grad()
                
                # Single forward pass for the whole batch
                output = self.forward(batch_indices) # [B, SeqLen, OutDim]
                
                # Vectorized Loss Calculation
                # Mean over sequence dimension
                pred_target = output.mean(dim=1) # [B, OutDim]
                
                # Sum of squared errors (sum over dim, sum over batch)
                loss = torch.sum((pred_target - batch_t) ** 2)
                
                # Normalize loss for backward pass
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
                if best_model_state is not None and avg_loss > prev_loss:
                    self.load_state_dict(best_model_state)                    
                    print(f"Restored best model state with loss = {best_loss:.6e}")
                    history[-1] = best_loss
                break
            prev_loss = avg_loss

            # Decay learning rate every iteration
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
                   batch_size=256, checkpoint_file=None, checkpoint_interval=5):
        """
        Self-training method for the hierarchical model with batch processing
        
        Args:
            seqs: List of character sequences
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
        
        # Pre-process data: convert strings to indices on CPU
        all_indices = self.batch_token_indices(seqs)
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
            
            # Process each batch
            for i in range(0, num_samples, batch_size):
                batch_idx = perm[i:i+batch_size]
                batch_indices = all_indices[batch_idx].to(self.device)
                
                optimizer.zero_grad()
                
                # 1. Compute Char Output (Target for Gap/Reg)
                char_output = self.char_forward(batch_indices) # [B, S_char, D]
                
                # 2. Compute Hierarchical Output
                hier_out = char_output
                for layer in self.hierarchical_layers:
                    hier_out = layer(hier_out) # [B, S_hier, D_hier]
                
                # 3. Vectorized Loss Calculation
                loss = 0.0
                
                # Determine valid comparison length
                # S_char might be different from S_hier if layers change sequence length
                min_len = min(char_output.shape[1], hier_out.shape[1])
                
                if self_mode == 'gap':
                    # ||Hier[k] - Char[k]||^2
                    # Slicing up to min_len
                    diff = hier_out[:, :min_len, :] - char_output[:, :min_len, :]
                    # Mean over dim and valid length, Sum over batch
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

            # Decay every 5 iterations (as per original logic)
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
                'model_dims': self.model_dims,
                'basis_dims': self.basis_dims,
                'use_residual_list': [layer.use_residual for layer in self.hierarchical_layers],
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
        all_indices = self.batch_token_indices(seqs)
        
        with torch.no_grad():
            total_vec_sum = torch.zeros(self.vec_dim, device=self.device)
            total_token_count = 0
            
            for i in range(0, len(seqs), batch_size):
                batch_indices = all_indices[i:i+batch_size].to(self.device)
                
                # Get char layer output [B, S, D]
                nk_batch = self.char_forward(batch_indices)
                
                # Sum over Batch and Sequence dim
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
        
        all_indices = self.batch_token_indices(seqs)
        
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
        """
        Reconstruct representative character sequence using the entire hierarchical model.
        
        This method generates a sequence of length L that best represents the learned
        patterns from the training data, using temperature-controlled sampling.
        
        Args:
            L (int): Length of sequence to reconstruct
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
        
        generated_sequence = ""
        
        # Generate sequence token by token
        while len(generated_sequence) < L:
            candidate_scores = {}
            valid_tokens = []
            
            # Evaluate each possible token
            for token in self.tokens:
                candidate_seq = generated_sequence + token
                
                # Skip if candidate exceeds desired length
                if len(candidate_seq) > L:
                    continue
                    
                valid_tokens.append(token)
            
            if not valid_tokens:
                # No valid candidates, use character layer as fallback
                char_layer_gen = self.char_generate(L - len(generated_sequence), tau)
                generated_sequence += char_layer_gen
                break
            
            # Create batch of candidate sequences
            candidate_seqs = [generated_sequence + token for token in valid_tokens]
            
            with torch.no_grad():
                model_output = self.forward(candidate_seqs)
                
                # Get predictions (average over sequence)
                pred_targets = model_output.mean(dim=1)  # [num_candidates, final_dim]
                
                # Compute errors compared to mean target
                diff = pred_targets - self.mean_target.unsqueeze(0)
                errors = torch.sum(diff ** 2, dim=1).cpu().numpy()
                scores = -errors  # Convert error to score (higher = better)
                
                # Select token based on temperature
                if tau == 0:
                    # Deterministic selection: choose token with highest score
                    best_idx = np.argmax(scores)
                    chosen_token = valid_tokens[best_idx]
                else:
                    # Stochastic selection with temperature
                    scaled_scores = scores / tau
                    exp_scores = np.exp(scaled_scores - np.max(scaled_scores))  # Numerical stability
                    probs = exp_scores / np.sum(exp_scores)
                    # Sample based on probabilities
                    chosen_idx = np.random.choice(len(valid_tokens), p=probs)
                    chosen_token = valid_tokens[chosen_idx]
            
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
            nn.init.uniform_(layer.linear.weight, -0.1, 0.1)
            nn.init.uniform_(layer.Acoeff, -0.1, 0.1)
            if hasattr(layer, 'residual_proj') and isinstance(layer.residual_proj, nn.Linear):
                nn.init.uniform_(layer.residual_proj.weight, -0.1, 0.1)
        
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
                'model_dims': self.model_dims,
                'basis_dims': self.basis_dims,
                'use_residual_list': [layer.use_residual for layer in self.hierarchical_layers],
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
            model_dims=config['model_dims'],
            basis_dims=config['basis_dims'],
            rank_mode=char_layer_config['rank_mode'],
            mode=char_layer_config['mode'],
            user_step=char_layer_config['user_step'],            
            use_residual_list=config.get('use_residual_list', ['separate'] * len(config['model_dims'])),
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

# === Example Usage ===
if __name__ == "__main__":
    from statistics import correlation
    
    print("="*60)
    print("HierDDabC - Hierarchical Dual Descriptor with AB Matrix and Character Input")
    print("AB Matrix Model with Character Sequence Processing Demonstration")
    print("="*60)
    
    # Set random seeds for reproducibility
    torch.manual_seed(2)
    random.seed(2)
    np.random.seed(2)
    
    # Configuration
    charset = ['A', 'C', 'G', 'T']
    rank = 2
    vec_dim = 8
    model_dims = [16, 12, 8]  # Hierarchical layer dimensions
    basis_dims = [100, 80, 60]  # Basis dimensions for hierarchical layers
    num_seqs = 100
    min_len, max_len = 200, 300
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {device}")
    print(f"Character set: {charset}")
    print(f"Rank (k-mer length): {rank}")
    print(f"Character layer output dim: {vec_dim}")
    print(f"Hierarchical layer dims: {model_dims}")
    print(f"Hierarchical basis dims: {basis_dims}")
    print(f"Using AB matrix architecture with character input")
    
    # Generate synthetic training data
    print("\nGenerating synthetic training data...")
    seqs, t_list = [], []
    for _ in range(num_seqs):
        L = random.randint(min_len, max_len)
        seq = ''.join(random.choices(charset, k=L))
        seqs.append(seq)
        # Create target vector (final layer dimension)
        t_list.append([random.uniform(-1.0, 1.0) for _ in range(model_dims[-1])])
    
    # Create model
    print("\nCreating HierDDabC model...")
    model = HierDDabC(
        charset=charset,
        rank=rank,
        vec_dim=vec_dim,
        model_dims=model_dims,
        basis_dims=basis_dims,
        mode='nonlinear',
        user_step=1,
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
        batch_size=16,
        checkpoint_file='hierddabc_gd_checkpoint.pth',
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
    self_model = HierDDabC(
        charset=charset,
        rank=rank,
        vec_dim=vec_dim,
        model_dims=model_dims,
        basis_dims=basis_dims,
        mode='nonlinear',
        user_step=1,
        device=device
    )
    
    # Use shorter sequences for self-training
    self_seqs = []
    for _ in range(100):
        L = random.randint(100, 200)
        self_seqs.append(''.join(random.choices(charset, k=L)))
    
    # Self-train in gap mode
    print("\nSelf-training in 'gap' mode...")
    self_history_gap = self_model.self_train(
        self_seqs,
        max_iters=30,
        learning_rate=0.01,
        self_mode='gap',
        print_every=5,
        batch_size=256,
        checkpoint_file='hierddabc_self_checkpoint.pth',
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
    model.save("hierddabc_model.pth")
    
    # Load model
    loaded_model = HierDDabC.load("hierddabc_model.pth", device=device)
    
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
