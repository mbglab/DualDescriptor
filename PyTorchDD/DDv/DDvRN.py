# Copyright (C) 2005-2025, Bin-Guang Ma (mbg@mail.hzau.edu.cn); SPDX-License-Identifier: MIT
# The Dual Descriptor Vector class (Random AB matrix form) implemented with PyTorch
# This program is for the demonstration of methodology and not fully refined.
# Author: Bin-Guang Ma (assisted by DeepSeek); Date: 2025-7-29

import math
import random
import itertools
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
import os

class DualDescriptorRN(nn.Module):
    """
    Dual Descriptor with GPU acceleration using PyTorch:
      - Learnable coefficient matrix Acoeff ∈ R^{m×L}
      - Learnable basis matrix Bbasis ∈ R^{L×m}
      - Token embeddings M: token → R^m
      - Optimized with batch processing for GPU acceleration
    """
    def __init__(self, charset, vec_dim=4, bas_dim=50, rank=1, rank_mode='drop', 
                 mode='linear', user_step=None, device='cuda'):
        super().__init__()
        self.charset = list(charset)
        self.m = vec_dim
        self.L = bas_dim
        self.rank = rank
        self.rank_mode = rank_mode
        assert mode in ('linear','nonlinear')
        self.mode = mode
        self.step = user_step
        self.trained = False
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Generate all possible tokens (k-mers + right-padded with '_')
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
        
        # Token embeddings
        self.embedding = nn.Embedding(len(self.tokens), self.m)
        
        # Coefficient matrix Acoeff: m×L
        self.Acoeff = nn.Parameter(torch.empty(self.m, self.L))
        
        # Basis matrix Bbasis: L×m
        self.Bbasis = nn.Parameter(torch.empty(self.L, self.m))
        
        # Initialize parameters
        self.reset_parameters()
        self.to(self.device)
        
    def reset_parameters(self):
        """Initialize model parameters"""
        nn.init.uniform_(self.embedding.weight, -0.5, 0.5)
        nn.init.uniform_(self.Acoeff, -0.1, 0.1)
        nn.init.uniform_(self.Bbasis, -0.1, 0.1)
    
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
        Args:
            k_tensor: Tensor of position indices [batch_size]
            token_indices: Tensor of token indices [batch_size]
        Returns:
            Tensor of N(k) vectors [batch_size, m]
        """
        # Get embeddings for all tokens [batch_size, m]
        x = self.embedding(token_indices)
        
        # Calculate basis indices j = k % L [batch_size]
        j_indices = (k_tensor % self.L).long()
        
        # Get Bbasis vectors: [batch_size, m]
        B_j = self.Bbasis[j_indices]
        
        # Compute scalar projection: B_j • x [batch_size]
        scalar = torch.sum(B_j * x, dim=1, keepdim=True)
        
        # Get Acoeff vectors: [batch_size, m]
        # Note: Acoeff is [m, L] -> permute to [L, m] then index with j_indices
        A_j = self.Acoeff.permute(1, 0)[j_indices]
        
        # Compute Nk = scalar * A_j [batch_size, m]
        Nk = scalar * A_j
            
        return Nk

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
        Train model using gradient descent with sequence-level batch processing.
        Memory-optimized version that processes sequences in batches to avoid 
        large precomputation and storage of all token positions.
        
        Args:
            seqs: List of character sequences
            t_list: List of target vectors
            max_iters: Maximum training iterations
            tol: Convergence tolerance
            learning_rate: Initial learning rate
            continued: Whether to continue from existing parameters
            decay_rate: Learning rate decay rate
            print_every: Print interval
            batch_size: Number of sequences to process in each batch
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
        t_tensors = [torch.tensor(t, dtype=torch.float32, device=self.device) for t in t_list]
        
        # Set up optimizer and scheduler
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
            
            # Shuffle sequences for each epoch to ensure diverse batches
            indices = list(range(len(seqs)))
            random.shuffle(indices)
            
            # Process sequences in batches
            for batch_start in range(0, len(indices), batch_size):
                batch_indices = indices[batch_start:batch_start + batch_size]
                batch_seqs = [seqs[idx] for idx in batch_indices]
                batch_targets = [t_tensors[idx] for idx in batch_indices]
                
                optimizer.zero_grad()
                batch_loss = 0.0
                batch_sequence_count = 0
                
                # Process each sequence in the current batch
                for seq, target in zip(batch_seqs, batch_targets):
                    # Extract tokens for current sequence
                    tokens = self.extract_tokens(seq)
                    if not tokens:
                        continue  # Skip empty sequences
                        
                    # Convert tokens to indices
                    token_indices = self.token_to_indices(tokens)
                    k_positions = torch.arange(len(tokens), dtype=torch.float32, device=self.device)
                    
                    # Batch compute all Nk vectors for current sequence
                    Nk_batch = self.batch_compute_Nk(k_positions, token_indices)
                    
                    # Compute sequence-level prediction: average of all N(k) vectors
                    seq_pred = torch.mean(Nk_batch, dim=0)
                    
                    # Calculate loss for this sequence (MSE between prediction and target)
                    seq_loss = torch.sum((seq_pred - target) ** 2)
                    batch_loss += seq_loss
                    batch_sequence_count += 1
                    
                    # Clean up intermediate tensors to free GPU memory
                    del Nk_batch, seq_pred, token_indices, k_positions
                    
                    # Periodically clear GPU cache to prevent memory fragmentation
                    if batch_sequence_count % 10 == 0 and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                # Backpropagate batch loss if we have valid sequences
                if batch_sequence_count > 0:
                    batch_loss = batch_loss / batch_sequence_count
                    batch_loss.backward()
                    optimizer.step()
                    
                    total_loss += batch_loss.item() * batch_sequence_count
                    total_sequences += batch_sequence_count
                
                # Clear GPU cache after processing each batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Calculate average loss for this iteration
            if total_sequences > 0:
                avg_loss = total_loss / total_sequences
            else:
                avg_loss = 0.0
                
            history.append(avg_loss)
            
            # Update best model state if current loss is lower
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_model_state = copy.deepcopy(self.state_dict())
            
            # Print training progress
            if it % print_every == 0 or it == max_iters - 1:
                current_lr = scheduler.get_last_lr()[0]
                print(f"GD Iter {it:3d}: Loss = {avg_loss:.6e}, LR = {current_lr:.6f}")
            
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
            
            # Update learning rate
            scheduler.step()
            
            # Final GPU memory cleanup for this iteration
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # If training ended without convergence, restore best model state
        if best_model_state is not None and best_loss < history[-1]:
            self.load_state_dict(best_model_state)
            print(f"Training ended. Restored best model state with loss = {best_loss:.6e}")
            history[-1] = best_loss
        
        # Compute and store statistics for reconstruction/generation
        self._compute_training_statistics(seqs)
        self.trained = True
        
        return history

    def auto_train(self, seqs, auto_mode='gap', max_iters=1000, tol=1e-8, 
               learning_rate=0.01, continued=False, decay_rate=1.0, 
               print_every=10, batch_size=1024, checkpoint_file=None, 
               checkpoint_interval=10):
        """
        Train the model using self-supervised learning with memory-optimized batch processing.
        Memory-optimized version that processes sequences and samples in batches to avoid 
        large precomputation and storage of all training samples.
        
        Args:
            seqs: List of character sequences
            auto_mode: 'gap' or 'reg' training mode
            max_iters: Maximum training iterations
            tol: Convergence tolerance
            learning_rate: Initial learning rate
            continued: Whether to continue from existing parameters
            decay_rate: Learning rate decay rate
            print_every: Print interval
            batch_size: Batch size for training samples
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
        
        # Set up optimizer and scheduler
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
            
            # Shuffle sequences for each epoch to ensure diverse samples
            indices = list(range(len(seqs)))
            random.shuffle(indices)
            
            # Process sequences in shuffled order
            for seq_idx in indices:
                seq = seqs[seq_idx]
                tokens = self.extract_tokens(seq)
                if not tokens:
                    continue
                    
                token_indices = self.token_to_indices(tokens)
                seq_samples = []
                
                # Generate samples for current sequence
                if auto_mode == 'gap':
                    # Each token is a sample (position k, token_idx)
                    for k, token_idx in enumerate(token_indices):
                        seq_samples.append((k, token_idx.item()))
                else:  # 'reg' mode
                    # Each token except last is a sample (position k, current token, next token)
                    for k in range(len(tokens) - 1):
                        seq_samples.append((k, token_indices[k].item(), token_indices[k+1].item()))
                
                if not seq_samples:
                    continue
                
                # Process samples from current sequence in batches
                for batch_start in range(0, len(seq_samples), batch_size):
                    batch_samples = seq_samples[batch_start:batch_start + batch_size]
                    
                    optimizer.zero_grad()
                    
                    # Prepare batch data directly as tensors
                    k_list = []
                    current_indices_list = []
                    target_indices_list = [] if auto_mode == 'reg' else None
                    
                    for sample in batch_samples:
                        if auto_mode == 'reg':
                            k, current_idx, next_idx = sample
                            k_list.append(k)
                            current_indices_list.append(current_idx)
                            target_indices_list.append(next_idx)
                        else:  # 'gap' mode
                            k, token_idx = sample
                            k_list.append(k)
                            current_indices_list.append(token_idx)
                    
                    # Create tensors directly on GPU
                    k_tensor = torch.tensor(k_list, dtype=torch.float32, device=self.device)
                    current_indices_tensor = torch.tensor(current_indices_list, device=self.device)
                    
                    # Batch compute Nk for current tokens
                    Nk_batch = self.batch_compute_Nk(k_tensor, current_indices_tensor)
                    
                    # Get target embeddings
                    if auto_mode == 'gap':
                        targets = self.embedding(current_indices_tensor)
                    else:  # 'reg' mode
                        target_indices_tensor = torch.tensor(target_indices_list, device=self.device)
                        targets = self.embedding(target_indices_tensor)
                    
                    # Compute loss
                    loss = torch.mean(torch.sum((Nk_batch - targets) ** 2, dim=1))
                    loss.backward()
                    optimizer.step()
                    
                    batch_loss = loss.item() * len(batch_samples)
                    total_loss += batch_loss
                    total_samples += len(batch_samples)
                    
                    # Clean up to free memory
                    del k_tensor, current_indices_tensor, Nk_batch, targets, loss
                    if auto_mode == 'reg':
                        del target_indices_tensor
                    
                    # Periodically clear GPU cache to prevent memory fragmentation
                    if total_samples % 1000 == 0 and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                # Clear sequence-specific tensors
                del token_indices, seq_samples
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Calculate average loss for this iteration
            if total_samples > 0:
                avg_loss = total_loss / total_samples
            else:
                avg_loss = 0.0
                
            history.append(avg_loss)
            
            # Update best model state if current loss is lower
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_model_state = copy.deepcopy(self.state_dict())
            
            # Print training progress
            if it % print_every == 0 or it == max_iters - 1:
                mode_display = "Gap" if auto_mode == 'gap' else "Reg"
                current_lr = scheduler.get_last_lr()[0]
                print(f"AutoTrain({mode_display}) Iter {it:3d}: loss = {avg_loss:.6f}, LR = {current_lr:.6f}, "
                      f"Samples = {total_samples}")
            
            # Save checkpoint at specified intervals
            if checkpoint_file and (it % checkpoint_interval == 0 or it == max_iters - 1):
                self._save_checkpoint(
                    checkpoint_file, it, history, optimizer, scheduler, best_loss
                )
            
            # Check convergence
            if abs(prev_loss - avg_loss) < tol:
                print(f"Converged after {it+1} iterations")
                # Restore the best model state before breaking
                if best_model_state is not None and avg_loss > prev_loss:
                    self.load_state_dict(best_model_state)
                    print(f"Restored best model state with loss = {best_loss:.6f}")
                    history[-1] = best_loss
                break
            prev_loss = avg_loss
            
            # Update learning rate
            scheduler.step()
            
            # Final GPU memory cleanup for this iteration
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # If training ended without convergence, restore best model state
        if best_model_state is not None and best_loss < history[-1]:
            self.load_state_dict(best_model_state)
            print(f"Training ended. Restored best model state with loss = {best_loss:.6f}")
            history[-1] = best_loss
        
        # Compute and store statistics for reconstruction/generation
        self._compute_training_statistics(seqs)
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
        checkpoint = {
            'iteration': iteration,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'history': history,
            'best_loss': best_loss,
            'config': {
                'charset': self.charset,
                'vec_dim': self.m,
                'bas_dim': self.L,
                'rank': self.rank,
                'rank_mode': self.rank_mode,
                'mode': self.mode,
                'user_step': self.step
            },
            'training_stats': {
                'mean_t': self.mean_t if hasattr(self, 'mean_t') else None,
                'mean_token_count': self.mean_token_count if hasattr(self, 'mean_token_count') else None
            }
        }
        torch.save(checkpoint, checkpoint_file)
        print(f"Checkpoint saved at iteration {iteration} to {checkpoint_file}")

    def _compute_training_statistics(self, seqs, batch_size=50):
        """Compute and store statistics for reconstruction and generation with memory optimization"""
        total_token_count = 0
        total_t = torch.zeros(self.m, device=self.device)
        
        with torch.no_grad():
            for i in range(0, len(seqs), batch_size):
                batch_seqs = seqs[i:i+batch_size]
                batch_token_count = 0
                batch_t_sum = torch.zeros(self.m, device=self.device)
                
                for seq in batch_seqs:
                    try:
                        tokens = self.extract_tokens(seq)
                        batch_token_count += len(tokens)
                        
                        if tokens:
                            token_indices = self.token_to_indices(tokens)
                            k_positions = torch.arange(len(tokens), dtype=torch.float32, device=self.device)
                            Nk_batch = self.batch_compute_Nk(k_positions, token_indices)
                            batch_t_sum += Nk_batch.sum(dim=0)
                            
                            # Clean up
                            del token_indices, k_positions, Nk_batch
                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            print(f"Warning: Memory error computing statistics for sequence. Skipping.")
                            continue
                        else:
                            raise e
                
                total_token_count += batch_token_count
                total_t += batch_t_sum
                
                # Clean batch
                del batch_t_sum
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        self.mean_token_count = total_token_count / len(seqs) if seqs else 0
        self.mean_t = (total_t / total_token_count).cpu().numpy() if total_token_count > 0 else np.zeros(self.m)
    
    def predict_t(self, seq):
        """Predict target vector as average of N(k) vectors"""
        toks = self.extract_tokens(seq)
        if not toks:
            return np.zeros(self.m)
        
        # Compute all Nk vectors
        Nk = self.describe(seq)
        return np.mean(Nk, axis=0)
    
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
        save_dict = {
            'state_dict': self.state_dict(),
            'mean_t': self.mean_t if hasattr(self, 'mean_t') else None,
            'mean_L': self.mean_L if hasattr(self, 'mean_L') else None,
            'trained': self.trained
        }
        torch.save(save_dict, filename)
        print(f"Model saved to {filename}")

    def load(self, filename):
        """Load model state from file"""
        try:
            save_dict = torch.load(filename, map_location=self.device, weights_only=False)
        except TypeError:
            save_dict = torch.load(filename, map_location=self.device)
        
        self.load_state_dict(save_dict['state_dict'])
        self.mean_t = save_dict.get('mean_t', None)
        self.mean_L = save_dict.get('mean_L', None)
        self.trained = save_dict.get('trained', False)
        print(f"Model loaded from {filename}")
        return self

    def part_train(self, vec_seqs, max_iters=100, tol=1e-6, learning_rate=0.01,
                   continued=False, auto_mode='reg', decay_rate=1.0, print_every=10):
        """
        Train the interaction model using gradient descent on vector sequences.
        Uses dual matrices AcoeffI (m×L) and BbasisI (L×m) to represent interactions.
        
        Supports two modes:
          'gap': Predicts current vector (self-consistency)
          'reg': Predicts next vector (auto-regressive)
        
        Args:
            vec_seqs (list): List of vector sequences (each sequence is a list of m-dimensional vectors)
            max_iters (int): Maximum training iterations
            tol (float): Convergence tolerance
            learning_rate (float): Initial learning rate
            continued (bool): Continue training from existing parameters
            auto_mode (str): Training mode - 'gap' or 'reg'
            decay_rate (float): Learning rate decay factor
            print_every (int): Progress printing frequency
            
        Returns:
            list: Training loss history
        """
        # Validate training mode
        if auto_mode not in ('gap', 'reg'):
            raise ValueError("auto_mode must be 'gap' or 'reg'")
        
        m, L = self.m, self.L
        
        # Initialize interaction parameters if not continuing
        if not continued or not hasattr(self, 'AcoeffI'):
            self.AcoeffI = nn.Parameter(torch.empty(m, L).to(self.device))
            self.BbasisI = nn.Parameter(torch.empty(L, m).to(self.device))
            nn.init.uniform_(self.AcoeffI, -0.1, 0.1)
            nn.init.uniform_(self.BbasisI, -0.1, 0.1)
        
        # Precompute total training samples
        total_samples = 0
        for seq in vec_seqs:
            if auto_mode == 'gap':
                total_samples += len(seq)  # All vectors
            else:  # reg mode
                total_samples += max(0, len(seq) - 1)  # Vectors except last
        
        if total_samples == 0:
            raise ValueError("No training samples found")
        
        # Set up optimizer and scheduler
        optimizer = optim.Adam([self.AcoeffI, self.BbasisI], lr=learning_rate)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)
        
        history = []  # Loss history
        prev_loss = float('inf')
        
        for it in range(max_iters):
            total_loss = 0.0
            
            # Process all sequences
            for seq in vec_seqs:
                if not seq:
                    continue
                    
                for k in range(len(seq)):
                    # Skip last position in reg mode
                    if auto_mode == 'reg' and k == len(seq) - 1:
                        continue
                    
                    # Get current vector
                    current_vec = torch.tensor(seq[k], dtype=torch.float32, device=self.device)
                    
                    # Determine target vector
                    if auto_mode == 'gap':
                        target = current_vec  # Self-consistency
                    else:  # reg mode
                        target = torch.tensor(seq[k + 1], dtype=torch.float32, device=self.device)
                    
                    # Compute intermediate projection (BbasisI • current_vec)
                    proj = torch.matmul(self.BbasisI, current_vec)
                    
                    # Compute Nk = AcoeffI • proj
                    Nk = torch.matmul(self.AcoeffI, proj)
                    
                    # Compute loss
                    loss = torch.sum((Nk - target) ** 2)
                    total_loss += loss.item()
                    
                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            
            # Calculate average loss
            avg_loss = total_loss / total_samples
            history.append(avg_loss)
            
            # Print progress
            if it % print_every == 0 or it == max_iters - 1:
                mode_str = "Gap" if auto_mode == 'gap' else "Reg"
                current_lr = scheduler.get_last_lr()[0]
                print(f"PartTrain({mode_str}) Iter {it:3d}: loss={avg_loss:.6e}, LR={current_lr:.4f}")
            
            # Check convergence
            if abs(prev_loss - avg_loss) < tol:
                print(f"Converged after {it+1} iterations")
                break
            prev_loss = avg_loss
            
            # Update learning rate
            scheduler.step()
        
        # Compute mean vector for generation
        total_vecs = 0
        sum_vec = torch.zeros(m, device=self.device)
        for seq in vec_seqs:
            for vec in seq:
                total_vecs += 1
                sum_vec += torch.tensor(vec, dtype=torch.float32, device=self.device)
        
        self.mean_vector = (sum_vec / total_vecs).cpu().numpy()
        return history

    def part_generate(self, L, tau=0.0, mode='reg'):
        """
        Generate a sequence of vectors using the trained interaction model.
        
        Args:
            L (int): Length of sequence to generate
            tau (float): Temperature for randomness (0=deterministic)
            mode (str): Generation mode - 'gap' or 'reg'
                
        Returns:
            list: Generated sequence of m-dimensional vectors
        """
        if not hasattr(self, 'AcoeffI') or not hasattr(self, 'BbasisI'):
            raise RuntimeError("Train interaction model first using part_train()")
        
        m = self.m
        sequence = []
        
        if mode == 'gap':
            # Gap filling mode: independent generations
            for k in range(L):
                # Start with mean vector
                current_vec = torch.tensor(self.mean_vector, dtype=torch.float32, device=self.device)
                
                # Compute projection: BbasisI • current_vec
                proj = torch.matmul(self.BbasisI, current_vec)
                
                # Compute Nk = AcoeffI • proj
                Nk = torch.matmul(self.AcoeffI, proj)
                
                # Apply temperature noise
                if tau > 0:
                    noise = torch.normal(0, tau, size=Nk.shape, device=self.device)
                    Nk = Nk + noise
                
                sequence.append(Nk.detach().cpu().numpy())
        
        else:  # reg mode
            # Auto-regressive mode: sequential generation
            current_vec = torch.tensor(self.mean_vector, dtype=torch.float32, device=self.device)
            
            for k in range(L):
                # Compute projection: BbasisI • current_vec
                proj = torch.matmul(self.BbasisI, current_vec)
                
                # Compute next vector = AcoeffI • proj
                next_vec = torch.matmul(self.AcoeffI, proj)
                
                # Apply temperature noise
                if tau > 0:
                    noise = torch.normal(0, tau, size=next_vec.shape, device=self.device)
                    next_vec = next_vec + noise
                
                sequence.append(next_vec.detach().cpu().numpy())
                current_vec = next_vec  # Update for next iteration
        
        return sequence

    def double_train(self, seqs, auto_mode='reg', part_mode='reg', 
                    auto_params=None, part_params=None):
        """
        Two-stage training method:
          1. First train on character sequences using auto_train (unsupervised)
          2. Then convert sequences to vector sequences using S(l) and train I matrix
        
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
        
        # Stage 2: Convert sequences to vector sequences using S(l)
        print("\n" + "="*50)
        print("Stage 2: Converting sequences to vector representations")
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
        
        # Train I matrix on vector sequences
        print("\n" + "="*50)
        print("Stage 3: Training I matrix on vector sequences")
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
        Generate character sequences using a two-stage approach that combines:
          1. Character-level model (auto-trained) for token probabilities
          2. Vector-sequence model (part-trained) for structural coherence
        
        Steps:
          a. Generate initial sequence with character model
          b. Compute cumulative vectors S(l) for initial sequence
          c. Use I-matrix to refine vector sequence
          d. Select tokens that best match the refined vectors
        
        Parameters:
            L (int): Length of sequence to generate
            tau (float): Temperature for stochastic sampling (0=deterministic)
        
        Returns:
            str: Generated character sequence
        """
        # Helper function to compute Nk for a token at position k
        def compute_Nk(k, token):
            j = k % self.L
            scalar = sum(self.Bbasis[j][i] * self.embedding.weight[self.token_to_idx[token]][i] for i in range(self.m))
            return [self.Acoeff[i][j] * scalar for i in range(self.m)]
        
        # Stage 1: Generate initial sequence with character model
        init_seq = self.generate(L, tau=tau)
        
        # Stage 2: Compute S(l) vectors for initial sequence
        s_vectors = self.S(init_seq)
        
        # Stage 3: Refine vectors using I-matrix with specified mode
        refined_vectors = self.part_generate(len(s_vectors), mode=mode, tau=tau)
        
        # Stage 4: Reconstruct character sequence using both models
        generated_tokens = []
        current_s = [0.0] * self.m  # Initialize cumulative vector
        
        for k in range(L):
            # Get target vector for current position
            if k < len(refined_vectors):
                target_vec = refined_vectors[k]
            else:
                # If beyond refined vectors, use character model prediction
                target_vec = self.mean_t
            
            # Calculate required N(k) vector: ΔS = S(k) - S(k-1)
            required_nk = [target_vec[i] - current_s[i] for i in range(self.m)]
            
            # Find best matching token
            best_token = None
            min_error = float('inf')
            token_scores = []
            
            for token in self.tokens:
                # Predict N(k) for this token at position k
                predicted_nk = compute_Nk(k, token)
                
                # Calculate matching error
                error = 0.0
                for d in range(self.m):
                    diff = predicted_nk[d] - required_nk[d]
                    error += diff * diff
                
                token_scores.append((token, error))
                
                # Track best token
                if error < min_error:
                    min_error = error
                    best_token = token
            
            # Select token (deterministic or stochastic)
            if tau == 0:
                chosen_token = best_token
            else:
                # Convert errors to probabilities
                tokens, errors = zip(*token_scores)
                weights = [math.exp(-err/tau) for err in errors]
                total_weight = sum(weights)
                if total_weight > 0:
                    probs = [w/total_weight for w in weights]
                    chosen_token = random.choices(tokens, weights=probs, k=1)[0]
                else:
                    chosen_token = random.choice(tokens)
            
            # Update sequence and cumulative vector
            generated_tokens.append(chosen_token)
            actual_nk = compute_Nk(k, chosen_token)
            current_s = [current_s[i] + actual_nk[i] for i in range(self.m)]
        
        return ''.join(generated_tokens)


# === Example Usage ===
if __name__ == "__main__":

    from statistics import correlation
    
    print("="*50)
    print("Dual Descriptor RN - PyTorch GPU Accelerated Version")
    print("Optimized with batch processing")
    print("="*50)
    
    # Set random seeds to ensure reproducibility
    torch.manual_seed(11)
    random.seed(11)
    
    charset = ['A','C','G','T']
    vec_dim = 3
    bas_dim = 300
    seq_num = 100

    # Generate sequences of length 100–200 and random targets
    seqs, t_list = [], []
    for _ in range(seq_num):
        L = random.randint(200, 300)
        seq = ''.join(random.choices(charset, k=L))
        seqs.append(seq)
        t_list.append([random.uniform(-1,1) for _ in range(vec_dim)])

    # === Gradient Descent Training Example ===
    print("\n" + "="*50)
    print("Testing Gradient Descent Training")
    print("="*50)

    # Create new model instance with GPU acceleration
    dd = DualDescriptorRN(
        charset, 
        rank=3, 
        vec_dim=vec_dim, 
        bas_dim=bas_dim, 
        mode='linear', 
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Display device information
    print(f"\nUsing device: {dd.device}")
    print(f"Number of tokens: {len(dd.tokens)}")

    # Train using gradient descent
    print("\nStarting gradient descent training...")
    grad_history = dd.grad_train(
        seqs, 
        t_list,
        learning_rate=0.05,
        max_iters=100,
        tol=1e-86,
        decay_rate=0.99,
        print_every=5,
        batch_size=2048
    )

    # Predict target vector for first sequence
    aseq = seqs[0]
    t_pred = dd.predict_t(aseq)
    print(f"\nPredicted t for first sequence: {[round(x.item(), 4) for x in t_pred]}")    
    
    # Calculate flattened correlation between predicted and actual targets
    pred_t_list = [dd.predict_t(seq) for seq in seqs]
    
    # Predictions and actuals for correlation calculation
    corr_sum = 0.0
    for i in range(dd.m):
        actu_t = [t_vec[i] for t_vec in t_list]
        pred_t = [t_vec[i] for t_vec in pred_t_list]
        corr = correlation(actu_t, pred_t)
        print(f"Prediction correlation for dimension {i}: {corr:.4f}")
        corr_sum += corr
    corr_avg = corr_sum / dd.m
    print(f"Average correlation: {corr_avg:.4f}")

    # Reconstruct representative sequence
    repr_seq = dd.reconstruct()
    print(f"\nReconstructed representative sequence: {repr_seq[:50]}...")
    
    # Generate sequences
    print("\nGenerated sequences:")
    seq_det = dd.generate(L=100, tau=0.0)
    print(f"Deterministic (tau=0): {seq_det[:50]}...")

    # === Auto-Training Example ===
    # Set random seed for reproducible results
    random.seed(1)
    
    # Define character set and model parameters
    charset = ['A','C','G','T']
    vec_dim = 3   # Vector dimension
    bas_dim = 100 # Base matrix dimension

    # Generate training sequences
    print("\n=== Generating Training Sequences ===")
    seqs = []
    for i in range(30):  # Generate 30 sequences
        L = random.randint(100, 200)
        seq = ''.join(random.choices(charset, k=L))
        seqs.append(seq)
        print(f"Generated sequence {i+1}: length={L}, first 10 chars: {seq[:10]}...")
    
    print("=== Creating Dual Descriptor Model ===")
    dd_auto_gap = DualDescriptorRN(
        charset, 
        rank=1,
        vec_dim=vec_dim,
        bas_dim=bas_dim,
        mode='linear',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )    
    
    # Run self-supervised training (Gap Filling mode)
    print("\n=== Starting Gap Filling Training ===")
    gap_history = dd_auto_gap.auto_train(
        seqs,
        auto_mode='gap',
        max_iters=300,
        learning_rate=0.01,
        decay_rate=0.995,
        print_every=20,
        batch_size=2048
    )

    print("=== Creating Dual Descriptor Model ===")
    dd_auto_reg = DualDescriptorRN(
        charset, 
        rank=1,
        vec_dim=vec_dim,
        bas_dim=bas_dim,
        mode='linear',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Run self-supervised training (Auto-Regressive mode)
    print("\n=== Starting Auto-Regressive Training ===")
    reg_history = dd_auto_reg.auto_train(
        seqs,
        auto_mode='reg',
        max_iters=300,
        learning_rate=0.001,
        decay_rate=0.99,
        print_every=20,
        batch_size=2048
    )
    
    # Generate new sequences
    print("\n=== Generating New Sequences ===")
    
    # Temperature=0 (deterministic generation)
    seq_det = dd_auto_gap.generate(L=40, tau=0.0)
    print(f"Deterministic Generation (tau=0.0):\n{seq_det}")
    
    # Temperature=0.5 (moderate randomness)
    seq_sto = dd_auto_reg.generate(L=40, tau=0.5)
    print(f"\nStochastic Generation (tau=0.5):\n{seq_sto}")
    
    # Save and load model
    print("\n=== Model Persistence Test ===")
    dd_auto_reg.save("auto_trained_model_rn.pt")
    
    # Load model
    dd_loaded = DualDescriptorRN(
        charset, 
        rank=1,
        vec_dim=vec_dim,
        bas_dim=bas_dim,
        mode='linear',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    dd_loaded.load("auto_trained_model_rn.pt")
    print("Model loaded successfully. Generating with loaded model:")
    print(dd_loaded.generate(L=20, tau=0.0))
    
    print("\n=== Auto-Training Demo Completed ===")

    # === Part Training and Generation Example ===
    print("\n" + "="*50)
    print("Part Training and Generation Example")
    print("="*50)

    # Create new model
    dd_part = DualDescriptorRN(charset=charset, rank=1, vec_dim=2, mode='linear')
    
    # Generate sample vector sequences (2D vectors)
    vec_seqs = []
    for _ in range(10):  # 10 sequences
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
    dd_double = DualDescriptorRN(
        charset=['A','C','G','T'], 
        rank=1, 
        vec_dim=2,
        mode='nonlinear',
        user_step=1
    )

    # Generate sample DNA sequences
    dna_seqs = []
    for _ in range(10):  # 10 sequences
        seq_len = random.randint(100, 200)
        dna_seqs.append(''.join(random.choices(['A','C','G','T'], k=seq_len)))
    
    # Configure training parameters
    auto_config = {
        'max_iters': 50,
        'tol': 1e-6,
        'learning_rate': 0.001
    }
    
    part_config = {
        'max_iters': 50,
        'tol': 1e-20,
        'learning_rate': 0.001
    }

    # Train with double_train
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

    print("\n=== All Demos Completed Successfully ===")
