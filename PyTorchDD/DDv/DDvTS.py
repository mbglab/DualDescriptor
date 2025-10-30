# Copyright (C) 2005-2025, Bin-Guang Ma (mbg@mail.hzau.edu.cn); SPDX-License-Identifier: MIT
# The Dual Descriptor Vector class (Tensor form) implemented with PyTorch
# This program is for the demonstration of methodology and not fully refined.
# Author: Bin-Guang Ma (assisted by DeepSeek); Date: 2025-7-29

import math
import random
import itertools
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy, os

class DualDescriptorTS(nn.Module):
    """
    Vector Dual Descriptor with GPU acceleration using PyTorch:
      - tensor P ∈ R^{m×m×o} of basis coefficients
      - embedding: k-mer token embeddings in R^m
      - indexed periods: period[i,j,g] = i*(m*o) + j*o + g + 2
      - basis function phi_{i,j,g}(k) = cos(2π * k / period[i,j,g])
      - supports 'linear' or 'nonlinear' (step-by-rank) k-mer extraction
    """
    def __init__(self, charset, rank=1, rank_mode='drop', vec_dim=2, num_basis=5, mode='linear', user_step=None, device='cuda'):
        super().__init__()
        self.charset = list(charset)
        self.rank = rank    # r-per/k-mer length
        self.rank_mode = rank_mode # 'pad' or 'drop'
        self.m = vec_dim    # embedding dimension
        self.o = num_basis  # number of basis terms         
        assert mode in ('linear','nonlinear')
        self.mode = mode
        self.step = user_step
        self.trained = False
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
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
        self.embedding = nn.Embedding(len(self.tokens), self.m)
        
        # Position-weight tensor P[i][j][g]
        self.P = nn.Parameter(torch.empty(self.m, self.m, self.o))
        
        # Precompute indexed periods[i][j][g] (fixed, not trainable)
        periods = torch.zeros(self.m, self.m, self.o, dtype=torch.float32)
        for i in range(self.m):
            for j in range(self.m):
                for g in range(self.o):
                    periods[i, j, g] = i*(self.m*self.o) + j*self.o + g + 2
        self.register_buffer('periods', periods)

        # Initialize parameters
        self.reset_parameters()
        self.to(self.device)
        
    def reset_parameters(self):
        """Initialize model parameters"""
        nn.init.uniform_(self.embedding.weight, -0.5, 0.5)
        nn.init.uniform_(self.P, -0.1, 0.1)
        
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

    def batch_compute_Nk(self, k_tensor, token_indices):
        """
        Vectorized computation of N(k) vectors for a batch of positions and tokens
        Optimized using einsum for better performance
        Args:
            k_tensor: Tensor of position indices [batch_size]
            token_indices: Tensor of token indices [batch_size]
        Returns:
            Tensor of N(k) vectors [batch_size, m]
        """
        # Get embeddings for all tokens [batch_size, m]
        x = self.embedding(token_indices)
        
        # Expand dimensions for broadcasting [batch_size, 1, 1, 1]
        k_expanded = k_tensor.view(-1, 1, 1, 1)
        
        # Calculate basis functions: cos(2π*k/periods) [batch_size, m, m, o]
        phi = torch.cos(2 * math.pi * k_expanded / self.periods)
        
        # Optimized computation using einsum
        # Original: term = x.unsqueeze(1).unsqueeze(3) * self.P * phi
        # Original: return term.sum(dim=(2, 3))
        Nk = torch.einsum('bj,ijg,bijg->bi', x, self.P, phi)
            
        return Nk

    def compute_Nk(self, k, token_idx):
        """Compute N(k) for single position and token (uses batch internally)"""
        # Convert to tensors
        k_tensor = torch.tensor([k], dtype=torch.float32, device=self.device)
        idx_tensor = torch.tensor([token_idx], device=self.device)
        
        # Use batch computation
        result = self.batch_compute_Nk(k_tensor, idx_tensor)
        return result[0]  # Return first element

    def describe(self, seq):
        """Compute N(k) vectors for each k-mer in sequence"""
        toks = self.extract_tokens(seq)
        if not toks:
            return []
        
        token_indices = self.token_to_indices(toks)
        k_positions = torch.arange(len(toks), dtype=torch.float32, device=self.device)
        
        # Batch compute all Nk vectors
        N_batch = self.batch_compute_Nk(k_positions, token_indices)
        return N_batch.detach().cpu().numpy()

    def S(self, seq):
        """
        Compute list of S(l)=sum(N(k)) (k=1,...,l; l=1,...,L) for a given sequence.        
        """
        toks = self.extract_tokens(seq)
        if not toks:
            return []
            
        token_indices = self.token_to_indices(toks)
        k_positions = torch.arange(len(toks), dtype=torch.float32, device=self.device)
        
        # Batch compute all Nk vectors
        N_batch = self.batch_compute_Nk(k_positions, token_indices)
        
        # Compute cumulative sum (S vectors)
        S_cum = torch.cumsum(N_batch, dim=0)
        return [s.detach().cpu().numpy() for s in S_cum]

    def D(self, seqs, t_list):
        """
        Compute mean squared deviation D across sequences:
        D = average over all positions of (N(k)-t_seq)^2
        """
        total_loss = 0.0
        total_positions = 0
        
        # Convert target vectors to tensor
        t_tensors = [torch.tensor(t, dtype=torch.float32, device=self.device) for t in t_list]
        
        for seq, t in zip(seqs, t_tensors):
            toks = self.extract_tokens(seq)
            if not toks:
                continue
                
            token_indices = self.token_to_indices(toks)
            k_positions = torch.arange(len(toks), dtype=torch.float32, device=self.device)
            
            # Batch compute all Nk vectors
            N_batch = self.batch_compute_Nk(k_positions, token_indices)
            
            # Compute loss for each position
            losses = torch.sum((N_batch - t) ** 2, dim=1)
            total_loss += losses.sum().item()
            total_positions += len(toks)
                
        return total_loss / total_positions if total_positions else 0.0

    def grad_train(self, seqs, t_list, max_iters=1000, tol=1e-8, learning_rate=0.01, 
               continued=False, decay_rate=1.0, print_every=10, batch_size=32,
               checkpoint_file=None, checkpoint_interval=10):
        """
        Train the model using gradient descent with sequence-level batch processing.
        Optimized for GPU memory efficiency by processing sequences individually.
        
        Args:
            seqs: List of character sequences for training
            t_list: List of target vectors corresponding to sequences
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
        
        if not continued:
            self.reset_parameters()
        
        # Convert target vectors to tensor
        t_tensors = [torch.tensor(t, dtype=torch.float32, device=self.device) for t in t_list]
        
        # Setup optimizer and scheduler
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)
        
        # Training state variables
        history = []
        prev_loss = float('inf')
        best_loss = float('inf')
        best_model_state = None
        
        for it in range(max_iters):
            total_loss = 0.0
            total_sequences = 0
            
            # Shuffle sequences for each epoch
            indices = list(range(len(seqs)))
            random.shuffle(indices)
            
            # Process sequences in batches
            for batch_start in range(0, len(indices), batch_size):
                batch_indices = indices[batch_start:batch_start + batch_size]
                batch_seqs = [seqs[idx] for idx in batch_indices]
                batch_targets = [t_tensors[idx] for idx in batch_indices]
                
                optimizer.zero_grad()
                batch_loss = 0.0
                
                # Process each sequence in the batch
                for seq, target in zip(batch_seqs, batch_targets):
                    # Extract tokens and convert to indices
                    tokens = self.extract_tokens(seq)
                    if not tokens:
                        continue
                        
                    token_indices = self.token_to_indices(tokens)
                    k_positions = torch.arange(len(tokens), dtype=torch.float32, device=self.device)
                    
                    # Compute N(k) vectors for all positions in the sequence
                    Nk_batch = self.batch_compute_Nk(k_positions, token_indices)
                    
                    # Compute sequence-level target: average of all N(k) vectors
                    seq_pred = torch.mean(Nk_batch, dim=0)
                    
                    # Calculate loss for this sequence
                    seq_loss = torch.sum((seq_pred - target) ** 2)
                    batch_loss += seq_loss
                    
                    # Clean up intermediate tensors to free memory
                    del Nk_batch, seq_pred, token_indices, k_positions
                
                # Average loss over sequences in batch and backpropagate
                if len(batch_seqs) > 0:
                    batch_loss = batch_loss / len(batch_seqs)
                    batch_loss.backward()
                    optimizer.step()
                    
                    total_loss += batch_loss.item() * len(batch_seqs)
                    total_sequences += len(batch_seqs)
                
                # Clear GPU cache periodically
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Calculate average loss for this iteration
            if total_sequences > 0:
                avg_loss = total_loss / total_sequences
            else:
                avg_loss = 0.0
                
            history.append(avg_loss)
            
            # Update best model state
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_model_state = copy.deepcopy(self.state_dict())
            
            # Print training progress
            if it % print_every == 0 or it == max_iters - 1:
                current_lr = scheduler.get_last_lr()[0]
                print(f"GD Iter {it:3d}: Loss = {avg_loss:.6e}, LR = {current_lr:.6f}")
            
            # Save checkpoint if specified
            if checkpoint_file and (it % checkpoint_interval == 0 or it == max_iters - 1):
                self._save_checkpoint(checkpoint_file, it, history, optimizer, scheduler, best_loss)
            
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
        
        # Compute and store training statistics for reconstruction/generation
        self._compute_training_statistics(seqs)
        self.trained = True
        
        return history

    def auto_train(self, seqs, max_iters=100, tol=1e-6, learning_rate=0.01, 
                   continued=False, auto_mode='gap', decay_rate=1.0, print_every=10,
                   batch_size=32, checkpoint_file=None, checkpoint_interval=5):
        """
        Self-training method with memory-efficient sequence processing.
        Supports both gap (self-consistency) and reg (next-token prediction) modes.
        
        Args:
            seqs: List of character sequences for training
            max_iters: Maximum number of training iterations
            tol: Convergence tolerance
            learning_rate: Initial learning rate for optimizer
            continued: Whether to continue training from existing parameters
            auto_mode: Training mode - 'gap' for self-consistency, 'reg' for next-token prediction
            decay_rate: Learning rate decay rate
            print_every: Print progress every N iterations
            batch_size: Number of sequences to process in each batch
            checkpoint_file: Path to save training checkpoints
            checkpoint_interval: Save checkpoint every N iterations
            
        Returns:
            list: Training loss history
        """
        
        if auto_mode not in ('gap', 'reg'):
            raise ValueError("auto_mode must be either 'gap' or 'reg'")
        
        if not continued:
            self.reset_parameters()
        
        # Setup optimizer and scheduler
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)
        
        # Training state variables
        history = []
        prev_loss = float('inf')
        best_loss = float('inf')
        best_model_state = None
        
        for it in range(max_iters):
            total_loss = 0.0
            total_samples = 0
            
            # Shuffle sequences for each epoch
            indices = list(range(len(seqs)))
            random.shuffle(indices)
            
            # Process sequences in batches
            for batch_start in range(0, len(indices), batch_size):
                batch_indices = indices[batch_start:batch_start + batch_size]
                batch_seqs = [seqs[idx] for idx in batch_indices]
                
                optimizer.zero_grad()
                batch_loss = 0.0
                batch_sample_count = 0
                
                # Process each sequence in the batch
                for seq in batch_seqs:
                    # Extract tokens and convert to indices
                    tokens = self.extract_tokens(seq)
                    if len(tokens) <= 1 and auto_mode == 'reg':
                        continue  # Skip sequences too short for regression mode
                        
                    token_indices = self.token_to_indices(tokens)
                    k_positions = torch.arange(len(tokens), dtype=torch.float32, device=self.device)
                    
                    # Compute N(k) vectors for all positions
                    Nk_batch = self.batch_compute_Nk(k_positions, token_indices)
                    
                    # Get token embeddings for target computation
                    token_embeddings = self.embedding(token_indices)
                    
                    # Compute loss based on auto_mode
                    seq_loss = 0.0
                    valid_positions = 0
                    
                    for k in range(len(tokens)):
                        if auto_mode == 'gap':
                            # Self-consistency: N(k) should match token embedding at position k
                            target = token_embeddings[k]
                            pred = Nk_batch[k]
                            seq_loss += torch.sum((pred - target) ** 2)
                            valid_positions += 1
                        else:  # 'reg' mode
                            if k < len(tokens) - 1:
                                # Predict next token's embedding
                                target = token_embeddings[k + 1]
                                pred = Nk_batch[k]
                                seq_loss += torch.sum((pred - target) ** 2)
                                valid_positions += 1
                    
                    if valid_positions > 0:
                        seq_loss = seq_loss / valid_positions
                        batch_loss += seq_loss
                        batch_sample_count += 1
                    
                    # Clean up intermediate tensors
                    del Nk_batch, token_embeddings, token_indices, k_positions
                
                # Backpropagate batch loss
                if batch_sample_count > 0:
                    batch_loss = batch_loss / batch_sample_count
                    batch_loss.backward()
                    optimizer.step()
                    
                    total_loss += batch_loss.item() * batch_sample_count
                    total_samples += batch_sample_count
                
                # Clear GPU cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Calculate average loss for this iteration
            if total_samples > 0:
                avg_loss = total_loss / total_samples
            else:
                avg_loss = 0.0
                
            history.append(avg_loss)
            
            # Update best model state
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_model_state = copy.deepcopy(self.state_dict())
            
            # Print training progress
            if it % print_every == 0 or it == max_iters - 1:
                current_lr = scheduler.get_last_lr()[0]
                mode_display = "Gap" if auto_mode == 'gap' else "Reg"
                print(f"AutoTrain({mode_display}) Iter {it:3d}: Loss = {avg_loss:.6f}, LR = {current_lr:.6f}")
            
            # Save checkpoint if specified
            if checkpoint_file and (it % checkpoint_interval == 0 or it == max_iters - 1):
                self._save_checkpoint(checkpoint_file, it, history, optimizer, scheduler, best_loss)
            
            # Check convergence
            if abs(prev_loss - avg_loss) < tol:
                print(f"Converged after {it+1} iterations")
                # Restore best model state
                if best_model_state is not None:
                    self.load_state_dict(best_model_state)
                break
                
            prev_loss = avg_loss
            
            # Learning rate scheduling
            scheduler.step()
        
        # Compute and store training statistics
        self._compute_training_statistics(seqs)
        self.trained = True
        
        return history

    def _compute_training_statistics(self, seqs, batch_size=50):
        """
        Compute training statistics for reconstruction and generation.
        Processes sequences in batches to manage memory usage.
        
        Args:
            seqs: List of training sequences
            batch_size: Number of sequences to process in each batch
        """
        total_token_count = 0
        total_t = torch.zeros(self.m, device=self.device)
        
        with torch.no_grad():
            for i in range(0, len(seqs), batch_size):
                batch_seqs = seqs[i:i + batch_size]
                batch_token_count = 0
                batch_vec_sum = torch.zeros(self.m, device=self.device)
                
                for seq in batch_seqs:
                    tokens = self.extract_tokens(seq)
                    if not tokens:
                        continue
                        
                    batch_token_count += len(tokens)
                    token_indices = self.token_to_indices(tokens)
                    k_positions = torch.arange(len(tokens), dtype=torch.float32, device=self.device)
                    
                    Nk_batch = self.batch_compute_Nk(k_positions, token_indices)
                    batch_vec_sum += Nk_batch.sum(dim=0)
                    
                    # Clean up
                    del Nk_batch, token_indices, k_positions
                
                total_token_count += batch_token_count
                total_t += batch_vec_sum
                
                # Clear GPU cache between batches
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        self.mean_token_count = total_token_count / len(seqs) if seqs else 0
        self.mean_t = (total_t / total_token_count).cpu().numpy() if total_token_count > 0 else np.zeros(self.m)

    def _save_checkpoint(self, checkpoint_file, iteration, history, optimizer, scheduler, best_loss):
        """
        Save training checkpoint with complete training state.
        
        Args:
            checkpoint_file: Path to save checkpoint file
            iteration: Current training iteration
            history: Training loss history
            optimizer: Optimizer instance
            scheduler: Learning rate scheduler instance
            best_loss: Best loss achieved so far
        """
        checkpoint = {
            'iteration': iteration,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'history': history,
            'best_loss': best_loss,
            'training_stats': {
                'mean_t': self.mean_t,
                'mean_token_count': self.mean_token_count
            }
        }
        torch.save(checkpoint, checkpoint_file)
        print(f"Checkpoint saved at iteration {iteration}")

    def predict_t(self, seq):
        """
        Predict target vector for a sequence
        Returns the average of all N(k) vectors in the sequence
        """
        toks = self.extract_tokens(seq)
        if not toks:
            return [0.0] * self.m
            
        token_indices = self.token_to_indices(toks)
        k_positions = torch.arange(len(toks), dtype=torch.float32, device=self.device)
        
        # Batch compute all Nk vectors
        Nk_batch = self.batch_compute_Nk(k_positions, token_indices)
        Nk_sum = torch.sum(Nk_batch, dim=0)
        
        return (Nk_sum / len(toks)).detach().cpu().numpy()
    
    def reconstruct(self):
        """Reconstruct representative sequence by minimizing error"""
        assert self.trained, "Model must be trained first"
        n_tokens = round(self.mean_token_count)
        mean_t_tensor = torch.tensor(self.mean_t, dtype=torch.float32, device=self.device)
        seq_tokens = []
        
        # Precompute all token embeddings
        all_token_indices = torch.arange(len(self.tokens), device=self.device)
        all_embeddings = self.embedding(all_token_indices)
        
        for k in range(n_tokens):
            # Compute Nk for all tokens at position k
            k_tensor = torch.tensor([k] * len(self.tokens), dtype=torch.float32, device=self.device)
            Nk_all = self.batch_compute_Nk(k_tensor, all_token_indices)
            
            # Compute errors
            errors = torch.sum((Nk_all - mean_t_tensor) ** 2, dim=1)
            min_idx = torch.argmin(errors).item()
            best_tok = self.idx_to_token[min_idx]
            seq_tokens.append(best_tok)
            
        return ''.join(seq_tokens)

    def generate(self, L, tau=0.0):
        """Generate sequence of length L with temperature-controlled randomness"""
        assert self.trained, "Model must be trained first"
        if tau < 0:
            raise ValueError("Temperature must be non-negative")
            
        num_blocks = (L + self.rank - 1) // self.rank
        mean_t_tensor = torch.tensor(self.mean_t, dtype=torch.float32, device=self.device)
        generated_tokens = []
        
        # Precompute all token embeddings
        all_token_indices = torch.arange(len(self.tokens), device=self.device)
        all_embeddings = self.embedding(all_token_indices)
        
        for k in range(num_blocks):
            # Compute Nk for all tokens at position k
            k_tensor = torch.tensor([k] * len(self.tokens), dtype=torch.float32, device=self.device)
            Nk_all = self.batch_compute_Nk(k_tensor, all_token_indices)
            
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

    def save(self, filename):
        """Save model state to file"""
        torch.save(self.state_dict(), filename)
        print(f"Model saved to {filename}")

    def load(self, filename):
        """Load model state from file"""        
        # Use weights only=True to avoid security warnings
        try:
            # PyTorch 1.13+ supports the "weights only" parameter
            state_dict = torch.load(filename, map_location=self.device, weights_only=True)
        except TypeError:
            # Rollback solution for old versions of PyTorch
            state_dict = torch.load(filename, map_location=self.device)
        
        self.load_state_dict(state_dict)
        print(f"Model loaded from {filename}")
        return self

    def part_train(self, vec_seqs, max_iters=100, tol=1e-6, learning_rate=0.01, 
                   continued=False, auto_mode='reg', decay_rate=1.0, print_every=10):
        """
        Train the I tensor on vector sequences using gradient descent with PyTorch optimization.
        Supports two modes:
          - 'gap': Predicts current vector (self-consistency)
          - 'reg': Predicts next vector (auto-regressive)
        
        Parameters:
            vec_seqs (list): List of vector sequences (each sequence is list of m-dim vectors)
            learning_rate (float): Step size for gradient updates
            max_iters (int): Maximum training iterations
            tol (float): Convergence tolerance
            auto_mode (str): Training mode - 'gap' or 'reg'
            continued (bool): Continue training existing I tensor
            decay_rate (float): Learning rate decay factor (1.0 = no decay)
            
        Returns:
            list: Training history (loss values per iteration)
        """
        if auto_mode not in ('gap', 'reg'):
            raise ValueError("auto_mode must be either 'gap' or 'reg'")

        # Initialize I tensor if needed
        if not continued or not hasattr(self, 'I'):            
            self.I = nn.Parameter(torch.empty(self.m, self.m, self.o, device=self.device))
            nn.init.uniform_(self.I, -0.1, 0.1)
        
        # Calculate total training samples
        total_samples = 0
        for seq in vec_seqs:
            if auto_mode == 'gap':
                total_samples += len(seq)  # All vectors are samples
            else:  # 'reg' mode
                total_samples += max(0, len(seq) - 1)  # Vectors except last
                
        if total_samples == 0:
            raise ValueError("No training samples found")
            
        # Setup optimizer for I tensor only
        optimizer = optim.Adam([self.I], lr=learning_rate)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)
        
        history = []  # Store loss per iteration
        prev_loss = float('inf')
        best_loss = float('inf')
        best_I_state = None
        
        for it in range(max_iters):
            optimizer.zero_grad()
            total_loss = 0.0
            
            # Process all vector sequences
            for seq in vec_seqs:
                if not seq:
                    continue
                    
                # Convert sequence to tensor
                seq_tensor = torch.tensor(np.array(seq), dtype=torch.float32, device=self.device)
                
                # Process vectors based on mode
                for k in range(len(seq)):
                    # Skip last vector in 'reg' mode (no next vector)
                    if auto_mode == 'reg' and k == len(seq) - 1:
                        continue
                        
                    current_vec = seq_tensor[k]
                    
                    # Compute N(k) for current vector at position k using I tensor
                    k_tensor = torch.tensor([k], dtype=torch.float32, device=self.device)
                    k_expanded = k_tensor.view(-1, 1, 1, 1)
                    
                    # Calculate basis functions: cos(2π*k/periods) [1, m, m, o]
                    phi = torch.cos(2 * math.pi * k_expanded / self.periods)
                    
                    # Compute Nk using I tensor: Nk[i] = Σ_j Σ_g I[i][j][g] * current_vec[j] * phi[i,j,g]
                    Nk = torch.einsum('j,ijg,ijg->i', current_vec, self.I, phi[0])
                    
                    # Determine target based on mode
                    if auto_mode == 'gap':
                        target = current_vec  # Self-consistency
                    else:  # 'reg' mode
                        target = seq_tensor[k + 1]  # Next vector prediction
                    
                    # Compute loss
                    loss = torch.sum((Nk - target) ** 2)
                    total_loss += loss
            
            # Average loss and backpropagate
            avg_loss = total_loss / total_samples
            avg_loss.backward()
            optimizer.step()
            
            history.append(avg_loss.item())
            
            # Update best I state
            if avg_loss.item() < best_loss:
                best_loss = avg_loss.item()
                best_I_state = self.I.detach().clone()
            
            # Print training progress
            if it % print_every == 0 or it == max_iters - 1:
                current_lr = scheduler.get_last_lr()[0]
                mode_display = "Gap" if auto_mode == 'gap' else "Reg"
                print(f"PartTrain({mode_display}) Iter {it:3d}: loss = {avg_loss.item():.6f}, LR = {current_lr:.6f}")
            
            # Check convergence
            if abs(prev_loss - avg_loss.item()) < tol:
                print(f"Converged after {it+1} iterations")
                # Restore best I state
                if best_I_state is not None:
                    self.I.data = best_I_state
                break
            prev_loss = avg_loss.item()
            
            # Learning rate scheduling
            scheduler.step()
        
        # Compute and store mean vector for generation
        total_vectors = 0
        total_vec_sum = torch.zeros(self.m, device=self.device)
        for seq in vec_seqs:
            seq_tensor = torch.tensor(np.array(seq), dtype=torch.float32, device=self.device)
            total_vectors += len(seq)
            total_vec_sum += seq_tensor.sum(dim=0)
        
        self.mean_vector = (total_vec_sum / total_vectors).cpu().numpy()
        
        return history

    def part_generate(self, L, tau=0.0, mode='reg'):
        """
        Generate a sequence of vectors using the trained I tensor with PyTorch optimization
        
        Parameters:
            L (int): Length of sequence to generate
            tau (float): Temperature for randomness (0 = deterministic)
            mode (str): Generation mode - 'gap' or 'reg' (must match training)
                
        Returns:
            list: Generated sequence of m-dimensional vectors
        """
        if not hasattr(self, 'I'):
            raise RuntimeError("I tensor not initialized - train first")
            
        if tau < 0:
            raise ValueError("Temperature must be non-negative")
            
        mean_vector_tensor = torch.tensor(self.mean_vector, dtype=torch.float32, device=self.device)
        sequence = []
        
        if mode == 'gap':
            # Gap mode: Generate independent reconstructions at each position
            for k in range(L):
                # Start with mean vector
                current_vec = mean_vector_tensor
                
                # Compute reconstruction at position k
                k_tensor = torch.tensor([k], dtype=torch.float32, device=self.device)
                k_expanded = k_tensor.view(-1, 1, 1, 1)
                
                # Calculate basis functions: cos(2π*k/periods) [1, m, m, o]
                phi = torch.cos(2 * math.pi * k_expanded / self.periods)
                
                # Compute reconstructed vector using I tensor
                reconstructed_vec = torch.einsum('j,ijg,ijg->i', current_vec, self.I, phi[0])
                
                # Add temperature-controlled noise
                if tau > 0:
                    noise = torch.normal(0, tau, size=(self.m,), device=self.device)
                    reconstructed_vec = reconstructed_vec + noise
                    
                sequence.append(reconstructed_vec.detach().cpu().numpy())
            return sequence
            
        else:  # 'reg' mode
            # Reg mode: Auto-regressive generation
            current_vec = mean_vector_tensor  # Start with mean vector
            
            for k in range(L):
                # Compute prediction for next vector
                k_tensor = torch.tensor([k], dtype=torch.float32, device=self.device)
                k_expanded = k_tensor.view(-1, 1, 1, 1)
                
                # Calculate basis functions: cos(2π*k/periods) [1, m, m, o]
                phi = torch.cos(2 * math.pi * k_expanded / self.periods)
                
                # Compute next vector prediction using I tensor
                next_vec_pred = torch.einsum('j,ijg,ijg->i', current_vec, self.I, phi[0])
                
                # Add temperature-controlled noise
                if tau > 0:
                    noise = torch.normal(0, tau, size=(self.m,), device=self.device)
                    next_vec = next_vec_pred + noise
                else:
                    next_vec = next_vec_pred
                    
                sequence.append(next_vec.detach().cpu().numpy())
                current_vec = next_vec  # Use prediction as next input
                
            return sequence

    def double_train(self, seqs, auto_mode='reg', part_mode='reg', 
                    auto_params=None, part_params=None):
        """
        Two-stage training method with PyTorch optimization: 
          1. First train on character sequences using auto_train (unsupervised)
          2. Then convert sequences to vector sequences using S(l) and train I tensor
        
        Parameters:
            seqs (list): Input character sequences
            auto_mode (str): Training mode for auto_train - 'gap' or 'reg'
            part_mode (str): Training mode for part_train - 'gap' or 'reg'
            auto_params (dict): Parameters for auto_train (max_iters, tol, learning_rate)
            part_params (dict): Parameters for part_train (max_iters, tol, learning_rate)
            
        Returns:
            tuple: (auto_history, part_history) training histories
        """
        # Set default parameters if not provided
        auto_params = auto_params or {'max_iters': 100, 'tol': 1e-6, 'learning_rate': 0.01}
        part_params = part_params or {'max_iters': 100, 'tol': 1e-6, 'learning_rate': 0.01}
        
        # Stage 1: Train character model with auto_train
        print("="*50)
        print("Stage 1: Auto-training on character sequences")
        print("="*50)
        auto_history = self.auto_train(
            seqs, 
            auto_mode=auto_mode,
            max_iters=auto_params['max_iters'],
            tol=auto_params['tol'],
            learning_rate=auto_params['learning_rate']
        )
        
        # Convert sequences to vector sequences using S(l)
        print("\n" + "="*50)
        print("Converting sequences to vector representations")
        print("="*50)
        vec_seqs = []
        for i, seq in enumerate(seqs):
            # Get cumulative S(l) vectors for the sequence
            s_vectors = self.S(seq)
            vec_seqs.append(s_vectors)
            if i < 3:  # Show sample conversion for first 3 sequences
                print(f"Sequence {i+1} (len={len(seq)}) -> {len(s_vectors)} vectors")
                print(f"  First vector: {[round(x, 4) for x in s_vectors[0]]}")
                print(f"  Last vector: {[round(x, 4) for x in s_vectors[-1]]}")
        
        # Stage 2: Train I tensor on vector sequences
        print("\n" + "="*50)
        print("Stage 2: Training I tensor on vector sequences")
        print("="*50)
        part_history = self.part_train(
            vec_seqs,
            max_iters=part_params['max_iters'],
            tol=part_params['tol'],
            learning_rate=part_params['learning_rate'],
            auto_mode=part_mode
        )
        
        return auto_history, part_history

    def double_generate(self, L, tau=0.0, mode='reg'):
        """
        Generate character sequences using a two-stage approach with PyTorch optimization:
          1. Character-level model (auto-trained) for token probabilities
          2. Vector-sequence model (part-trained) for structural coherence
        
        Parameters:
            L (int): Length of sequence to generate
            tau (float): Temperature for stochastic sampling (0=deterministic)
            mode (str): Generation mode for vector refinement - 'gap' or 'reg' (default: 'reg')
        
        Returns:
            str: Generated character sequence
        """
        # Stage 1: Generate initial sequence with character model
        init_seq = self.generate(L, tau=tau)
        
        # Stage 2: Compute S(l) vectors for initial sequence
        s_vectors = self.S(init_seq)
        
        # Stage 3: Refine vectors using I-tensor with specified mode
        refined_vectors = self.part_generate(len(s_vectors), mode=mode, tau=tau)
        
        # Stage 4: Reconstruct character sequence using both models
        generated_tokens = []
        current_s = torch.zeros(self.m, device=self.device)  # Initialize cumulative vector
        
        for k in range(L):
            # Get target vector for current position
            if k < len(refined_vectors):
                target_vec = torch.tensor(refined_vectors[k], dtype=torch.float32, device=self.device)
            else:
                # If beyond refined vectors, use character model prediction
                target_vec = torch.tensor(self.mean_t, dtype=torch.float32, device=self.device)
            
            # Calculate required N(k) vector: ΔS = S(k) - S(k-1)
            required_nk = target_vec - current_s
            
            # Find best matching token
            best_token = None
            min_error = float('inf')
            token_scores = []
            
            # Precompute all token embeddings
            all_token_indices = torch.arange(len(self.tokens), device=self.device)
            
            # Compute Nk for all tokens at position k
            k_tensor = torch.tensor([k] * len(self.tokens), dtype=torch.float32, device=self.device)
            Nk_all = self.batch_compute_Nk(k_tensor, all_token_indices)
            
            # Calculate errors for all tokens
            errors = torch.sum((Nk_all - required_nk) ** 2, dim=1)
            
            # Get scores and find best token
            scores = -errors  # Convert to score (higher = better)
            
            if tau == 0:  # Deterministic selection
                max_idx = torch.argmax(scores).item()
                chosen_token = self.idx_to_token[max_idx]
            else:  # Stochastic selection
                probs = torch.softmax(scores / tau, dim=0).detach().cpu().numpy()
                chosen_idx = random.choices(range(len(self.tokens)), weights=probs, k=1)[0]
                chosen_token = self.idx_to_token[chosen_idx]
            
            # Update sequence and cumulative vector
            generated_tokens.append(chosen_token)
            
            # Compute actual N(k) for chosen token
            token_idx = self.token_to_idx[chosen_token]
            actual_nk = self.compute_Nk(k, token_idx)
            current_s = current_s + actual_nk
        
        return ''.join(generated_tokens)

# === Example Usage ===
if __name__=="__main__":

    from statistics import correlation
    
    print("="*50)
    print("Dual Descriptor TS - PyTorch GPU Accelerated Version")
    print("Optimized with batch processing")
    print("="*50)
    
    # Set random seeds to ensure reproducibility
    torch.manual_seed(11)
    random.seed(11)
    
    charset = ['A','C','G','T']
    vec_dim = 10
    num_basis = 10
    rank = 6
    user_step = 3
    
    # Generate 1000 sequences with random target vectors
    seqs, t_list = [], []
    for _ in range(100):
        L = random.randint(200, 300)
        seq = ''.join(random.choices(charset, k=L))
        seqs.append(seq)
        # Create a random vector target
        t_list.append([random.uniform(-1.0, 1.0) for _ in range(vec_dim)])

    # Initialize the model
    dd = DualDescriptorTS(
        charset, 
        rank=rank, 
        vec_dim=vec_dim, 
        num_basis=num_basis, 
        mode='nonlinear', 
        user_step=user_step,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Display device information
    print(f"\nUsing device: {dd.device}")
    print(f"Number of tokens: {len(dd.tokens)}")

    # Training model
    print("\n" + "="*50)
    print("Starting Gradient Descent Training")
    print("="*50)
    dd.grad_train(seqs, t_list, max_iters=200, tol=1e-199, learning_rate=0.1, decay_rate = 0.99, batch_size=2048)  
   
    # Predict the target vector of the first sequence
    aseq = seqs[0]
    t_pred = dd.predict_t(aseq)
    print(f"\nPredicted t for first sequence: {[round(x, 4) for x in t_pred]}")    
    
    # Calculate the correlation between the predicted and the real target
    pred_t_list = [dd.predict_t(seq) for seq in seqs]
    
    # The predicted values and actual values used for correlation calculation
    corr_sum = 0.0
    for i in range(dd.m):
        actu_t = [t_vec[i] for t_vec in t_list]
        pred_t = [t_vec[i] for t_vec in pred_t_list]
        corr = correlation(actu_t, pred_t)
        print(f"Dimension {i} prediction correlation: {corr:.4f}")
        corr_sum += corr
    corr_avg = corr_sum / dd.m
    print(f"Average correlation: {corr_avg:.4f}")       
  
    # Reconstruct the representative sequence
    repr_seq = dd.reconstruct()
    print(f"\nRepresentative sequence (len={len(repr_seq)}): {repr_seq[:50]}...")
    
    # Generate new sequences
    seq_det = dd.generate(L=100, tau=0.0)
    seq_rand = dd.generate(L=100, tau=0.5)
    print("\nDeterministic generation:", seq_det[:50] + "...")
    print("Stochastic generation (tau=0.5):", seq_rand[:50] + "...")    
   
    # === Combined self-training examples ===
    print("\n" + "="*50)
    print("Combined Auto-Training Example")
    print("="*50)
    
    # Create a new model
    dd_gap = DualDescriptorTS(
        charset, 
        rank=rank, 
        vec_dim=vec_dim, 
        num_basis=num_basis, 
        mode='nonlinear', 
        user_step=user_step,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Generate sample sequences
    auto_seqs = []
    for _ in range(10):
        L = random.randint(200, 300)
        auto_seqs.append(''.join(random.choices(charset, k=L)))
    
    # Conduct self-consistenty training (gap mode)
    print("\nTraining in 'gap' mode (self-consistency):")
    gap_history = dd_gap.auto_train(
        auto_seqs, 
        max_iters=50, 
        tol=1e-8, 
        learning_rate=0.01, 
        auto_mode='gap',
        batch_size=1024
    )
    
    # Generate sequences
    print("\nGenerated sequences from 'gap' model:")
    for i in range(2):
        gen_seq = dd_gap.generate(100, tau=0.2)
        print(f"Sequence {i+1}: {gen_seq[:50]}...") 
    
    # === Part Train/Generate Example ===
    print("\n" + "="*50)
    print("Part Train/Generate Example")
    print("="*50)
    
    # Create new model
    dd_part = DualDescriptorTS(charset="", rank=3, vec_dim=2, num_basis=5)
    
    # Generate sample vector sequences (2D vectors)
    vec_seqs = []
    for _ in range(5):  # 5 sequences
        seq_len = random.randint(100, 150)
        seq = []
        for _ in range(seq_len):
            # Generate random 2D vector
            vec = [random.uniform(-1, 1), random.uniform(-1, 1)]
            seq.append(vec)
        vec_seqs.append(seq)
    
    # Train in self-consistency (gap) mode
    print("\nTraining in 'gap' mode (self-consistency):")
    gap_history = dd_part.part_train(vec_seqs, max_iters=100, 
                                     learning_rate=0.1, auto_mode='gap')
    
    # Generate new vector sequence
    print("\nGenerated vector sequence (gap mode):")
    gen_seq = dd_part.part_generate(10, mode='gap', tau=0.0)
    for i, vec in enumerate(gen_seq):
        print(f"Vec {i+1}: [{vec[0]:.10f}, {vec[1]:.10f}]")
    
    # Train in auto-regressive (reg) mode
    print("\nTraining in 'reg' mode (next-vector prediction):")
    reg_history = dd_part.part_train(vec_seqs, max_iters=100, 
                                     learning_rate=0.1, auto_mode='reg')
    
    # Generate new vector sequence with randomness
    print("\nGenerated vector sequence with temperature (reg mode):")
    gen_seq = dd_part.part_generate(10, mode='reg', tau=0.1)
    for i, vec in enumerate(gen_seq):
        print(f"Vec {i+1}: [{vec[0]:.10f}, {vec[1]:.10f}]")
    
    # === Double Generation Example ===
    print("\n" + "="*50)
    print("Double Generation Example")
    print("="*50)

    # Create and train model using double_train
    dd_double = DualDescriptorTS(
        charset=['A','C','G','T'], 
        rank=3, 
        vec_dim=2, 
        num_basis=5,
        mode='nonlinear',
        user_step=2
    )

    # Generate sample DNA sequences
    dna_seqs = []
    for _ in range(10):  # 5 sequences
        seq_len = random.randint(100, 200)
        dna_seqs.append(''.join(random.choices(['A','C','G','T'], k=seq_len)))

    # Configure training parameters
    auto_config = {
        'max_iters': 50,
        'tol': 1e-6,
        'learning_rate': 0.1
    }
    
    part_config = {
        'max_iters': 50,
        'tol': 1e-6,
        'learning_rate': 0.01
    }

    # Train with double_train (as in previous example)
    auto_hist, part_hist = dd_double.double_train(
        dna_seqs,  # Sample DNA sequences
        auto_mode='reg',
        part_mode='reg',
        auto_params=auto_config,
        part_params=part_config
    )

    # Generate sequences using different methods for comparison
    print("\n1. Character-only generation:")
    char_seq = dd_double.generate(100, tau=0.3)
    print(char_seq)

    print("\n2. Vector-only generation:")
    vec_seq = dd_double.part_generate(10, mode='reg', tau=0.1)
    for i, vec in enumerate(vec_seq):
        print(f"Position {i}: [{vec[0]:.4f}, {vec[1]:.4f}]")

    print("\n3. Double-generation (combined models):")
    double_seq = dd_double.double_generate(100, tau=0.2)
    print(double_seq)

    print("\nAll tests completed successfully!")
