# Copyright (C) 2005-2025, Bin-Guang Ma (mbg@mail.hzau.edu.cn); SPDX-License-Identifier: MIT
# Hierarchical Dual Descriptor (P Matrix form) and Character Sequence Input (HierDDpmC)
# Combines character-level processing and hierarchical vector processing
# This program is for the demonstration of methodology and not fully refined.
# Author: Bin-Guang Ma (assisted by DeepSeek); Date: 2025-10-11

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
    """Single layer of Hierarchical Dual Descriptor with 2D position matrix"""
    def __init__(self, in_dim, out_dim, use_residual):
        """
        Initialize a hierarchical layer with 2D position matrix
        
        Args:
            in_dim (int): Input dimension
            out_dim (int): Output dimension
            use_residual (str or None): Residual connection type. Options are:
                - 'separate': use a separate linear projection for the residual connection.
                - 'shared': share the linear transformation matrix M for the residual.
                - None or other: no residual connection, unless in_dim==out_dim, then use identity residual.
        """
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.use_residual = use_residual
        
        # Linear transformation matrix (shared for main path and residual)
        self.M = nn.Parameter(torch.empty(out_dim, in_dim))
        
        # 2D position matrix (simplified from 3D tensor)
        self.P = nn.Parameter(torch.empty(out_dim, out_dim))
        
        # Precompute periods matrix (fixed, not trainable)
        periods = torch.zeros(out_dim, out_dim)
        for i in range(out_dim):
            for j in range(out_dim):
                periods[i, j] = i * out_dim + j + 2
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
        Forward pass for the layer with 2D position matrix
        
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
        # Expand tensors for broadcasting: positions (seq_len, 1, 1)
        k = positions.view(-1, 1, 1).float()
        
        # Compute basis functions: phi = cos(2π * k / period)
        # periods shape: (1, out_dim, out_dim)
        periods = self.periods.unsqueeze(0)
        phi = torch.cos(2 * math.pi * k / periods)  # (seq_len, out_dim, out_dim)
        
        # Apply position-dependent transformation
        # x_trans: (batch_size, seq_len, out_dim) -> (batch_size, seq_len, 1, out_dim)
        x_exp = x_trans.unsqueeze(2)
        
        # P: (out_dim, out_dim) -> (1, 1, out_dim, out_dim)
        P_exp = self.P.unsqueeze(0).unsqueeze(0)
        
        # Compute: P * x' * phi
        # Result shape: (batch_size, seq_len, out_dim, out_dim)
        product = P_exp * x_exp * phi
        
        # Sum over j (input dim)
        Nk = torch.sum(product, dim=3)  # (batch_size, seq_len, out_dim)
       
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

class HierDDpmC(nn.Module):
    """
    Hierarchical Dual Descriptor with Position Matrix and Character Sequence Input
    Combines character-level processing and hierarchical layers for vector processing with simplified 2D position matrices
    """
    def __init__(self, charset, rank=1, rank_mode='drop', vec_dim=2, 
                 mode='linear', user_step=None, model_dims=[2], 
                 use_residual_list=None, device='cuda'):
        """
        Initialize HierDDpmC model with embedded character-level processing and 2D position matrices
        
        Args:
            charset: Character set for sequence input
            rank: k-mer length for tokenization
            rank_mode: 'pad' or 'drop' for incomplete k-mers
            vec_dim: Output dimension of character layer (Layer 0)
            mode: 'linear' or 'nonlinear' tokenization
            user_step: Step size for nonlinear tokenization
            model_dims: Output dimensions for hierarchical layers
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
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.trained = False        
        
        # Character-level processing components (formerly DualDescriptorTS)
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
        
        # 2D position matrix P[i][j] for character layer (simplified from 3D tensor)
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
        
        # Hierarchical layers (starting from Layer 1)
        self.hierarchical_layers = nn.ModuleList()
        in_dim = vec_dim  # Input to first hierarchical layer is output from char layer
        
        if use_residual_list is None:
            use_residual_list = ['separate'] * len(model_dims)
            
        for out_dim, use_residual in zip(model_dims, use_residual_list):
            layer = Layer(in_dim, out_dim, use_residual)
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
        Optimized using einsum for better performance with 2D position matrix
        Args:
            k_tensor: Tensor of position indices [batch_size]
            token_indices: Tensor of token indices [batch_size]
        Returns:
            Tensor of N(k) vectors [batch_size, vec_dim]
        """
        # Get embeddings for all tokens [batch_size, vec_dim]
        x = self.char_embedding(token_indices)
        
        # Expand dimensions for broadcasting [batch_size, 1, 1]
        k_expanded = k_tensor.view(-1, 1, 1)
        
        # Calculate basis functions: cos(2π*k/periods) [batch_size, vec_dim, vec_dim]
        phi = torch.cos(2 * math.pi * k_expanded / self.char_periods)
        
        # Optimized computation using einsum with 2D position matrix
        Nk = torch.einsum('bj,ij,bij->bi', x, self.char_P, phi)
            
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
            seq (str): Input character sequence
            
        Returns:
            Tensor: Sequence tensor of shape [1, seq_len, vec_dim]
        """
        toks = self.extract_tokens(seq)
        if not toks:
            return torch.zeros((1, 0, self.vec_dim), device=self.device)
        
        token_indices = self.token_to_indices(toks)
        k_positions = torch.arange(len(toks), dtype=torch.float32, device=self.device)
        
        # Compute N(k) vectors through char layer
        Nk_batch = self.char_batch_compute_Nk(k_positions, token_indices)
        
        # Add batch dimension
        return Nk_batch.unsqueeze(0)  # [1, seq_len, vec_dim]
    
    def forward(self, seq):
        """
        Forward pass through entire hierarchical model
        
        Args:
            seq (str or list): Input character sequence(s)
            
        Returns:
            Tensor: Output of shape [batch_size, seq_len, final_dim]
        """
        if isinstance(seq, str):
            # Single sequence
            x = self.char_sequence_to_tensor(seq)
        else:
            # Multiple sequences - process each separately and stack
            batch_tensors = []
            for s in seq:
                tensor_seq = self.char_sequence_to_tensor(s)
                batch_tensors.append(tensor_seq.squeeze(0))
            x = torch.stack(batch_tensors)  # [batch_size, seq_len, vec_dim]
        
        # Generate position indices
        seq_len = x.shape[1]
        positions = torch.arange(seq_len, device=self.device)
        
        # Pass through hierarchical layers
        for layer in self.hierarchical_layers:
            x = layer(x, positions)
            
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
    
    def grad_train(self, seqs, t_list, max_iters=1000, tol=1e-8, learning_rate=0.01, 
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
                    output = self.forward(seq)  # [1, seq_len, final_dim]
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
        Self-training method for the hierarchical model
        
        Args:
            seqs: List of character sequences
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
        if auto_mode not in ('gap', 'reg'):
            raise ValueError("auto_mode must be either 'gap' or 'reg'")
        
        # Load checkpoint if continuing and checkpoint file exists
        start_iter = 0
        best_loss = float('inf')
        
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
                # Convert to tensor through char layer
                char_output = self.char_sequence_to_tensor(seq)  # [1, seq_len, vec_dim]
                seq_len = char_output.shape[1]
                
                if seq_len <= 1 and auto_mode == 'reg':
                    continue  # Skip sequences that are too short for reg mode
                
                # Forward pass through hierarchical layers
                positions = torch.arange(seq_len, device=self.device)
                hierarchical_output = char_output
                for layer in self.hierarchical_layers:
                    hierarchical_output = layer(hierarchical_output, positions)
                
                # Compute loss based on auto_mode
                seq_loss = 0.0
                valid_positions = 0
                
                for k in range(seq_len):
                    if auto_mode == 'gap':
                        # Self-consistency: output should match char layer output
                        target = char_output[0, k]
                        pred = hierarchical_output[0, k]
                        seq_loss += torch.sum((pred - target) ** 2)
                        valid_positions += 1
                    else:  # 'reg' mode
                        if k < seq_len - 1:
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
                if best_model_state is not None and avg_loss > prev_loss:
                    self.load_state_dict(best_model_state)
                    print(f"Restored best model state with loss = {best_loss:.6f}")
                    history[-1] = best_loss                    
                break
            prev_loss = avg_loss

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
                'model_dims': self.model_dims,                
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
        and all hierarchical layers to reconstruct a sequence that best represents the learned
        patterns from the training data.
        
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
        
        return reconstructed_seq

    def generate(self, L, tau=0.0):
        """
        Generate character sequence of length L using the entire hierarchical model.
        
        This method uses temperature-controlled sampling based on the complete model's
        predictions, incorporating both character-level and hierarchical patterns.
        
        Args:
            L (int): Length of sequence to generate
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
        
        generated_sequence = ""
        
        # Generate sequence token by token
        while len(generated_sequence) < L:
            candidate_scores = {}
            
            # Evaluate each possible token
            for token in self.tokens:
                candidate_seq = generated_sequence + token
                
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
            nn.init.uniform_(layer.M, -0.5, 0.5)
            nn.init.uniform_(layer.P, -0.1, 0.1)
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
            model_dims=config['model_dims'],
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
    print("HierDDpmC - Hierarchical Dual Descriptor with Position Matrix and Character Input")
    print("Simplified 2D Position Matrix Model Demonstration")
    print("="*60)
    
    # Set random seeds for reproducibility
    torch.manual_seed(2)
    random.seed(2)
    np.random.seed(2)
    
    # Configuration
    charset = ['A', 'C', 'G', 'T']
    rank = 4
    vec_dim = 8
    model_dims = [16, 12, 8]  # Hierarchical layer dimensions
    num_seqs = 100
    min_len, max_len = 200, 300
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {device}")
    print(f"Character set: {charset}")
    print(f"Rank (k-mer length): {rank}")
    print(f"Character layer output dim: {vec_dim}")
    print(f"Hierarchical layer dims: {model_dims}")
    print(f"Using simplified 2D position matrices (no basis dimension)")
    
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
    print("\nCreating HierDDpmC model...")
    model = HierDDpmC(
        charset=charset,
        rank=rank,
        vec_dim=vec_dim,
        model_dims=model_dims,
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
    
    gd_history = model.grad_train(
        seqs, t_list,
        max_iters=100,
        learning_rate=0.1,
        decay_rate=0.98,
        print_every=5,
        batch_size=32,
        checkpoint_file='gd_checkpoint.pth',
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
    
    reconstructed_seq = model.reconstruct()
    print(f"Reconstructed sequence (length {len(reconstructed_seq)}):")
    print(f"First 100 chars: {reconstructed_seq[:100]}")
    
    # Sequence generation
    print("\n" + "="*50)
    print("Sequence Generation")
    print("="*50)
    
    # Deterministic generation
    det_seq = model.generate(L=100, tau=0.0)
    print(f"Deterministic generation (tau=0.0): {det_seq}")
    
    # Stochastic generation
    stoch_seq = model.generate(L=100, tau=0.5)
    print(f"Stochastic generation (tau=0.5): {stoch_seq}")
    
    # Auto-training example
    print("\n" + "="*50)
    print("Auto-Training Example")
    print("="*50)
    
    # Create a new model for auto-training
    auto_model = HierDDpmC(
        charset=charset,
        rank=rank,
        vec_dim=vec_dim,
        model_dims=model_dims,
        mode='nonlinear',
        user_step=1,
        device=device
    )
    
    # Use shorter sequences for auto-training
    auto_seqs = []
    for _ in range(100):
        L = random.randint(100, 200)
        auto_seqs.append(''.join(random.choices(charset, k=L)))
    
    # Auto-train in gap mode
    print("\nAuto-training in 'gap' mode...")
    auto_history_gap = auto_model.auto_train(
        auto_seqs,
        max_iters=30,
        learning_rate=0.01,
        auto_mode='gap',
        print_every=5,
        checkpoint_file='auto_checkpoint.pth',
        checkpoint_interval=5
    )
    
    # Generate from auto-trained model
    auto_gen_seq = auto_model.generate(L=80, tau=0.2)
    print(f"Generated from auto-trained model: {auto_gen_seq}")
    
    # Model save and load test
    print("\n" + "="*50)
    print("Model Save/Load Test")
    print("="*50)
    
    # Save model
    model.save("hierddpmc_model.pth")
    
    # Load model
    loaded_model = HierDDpmC.load("hierddpmc_model.pth", device=device)
    
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
    
    print("\nAll tests completed successfully!")