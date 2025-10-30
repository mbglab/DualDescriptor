# Copyright (C) 2005-2025, Bin-Guang Ma (mbg@mail.hzau.edu.cn); SPDX-License-Identifier: MIT
# The Dual Descriptor Vector class (AB matrix form) implemented with PyTorch
# This program is for the demonstration of methodology and not fully refined.
# Author: Bin-Guang Ma (assisted by DeepSeek); Date: 2025-7-29

import math
import itertools
import random
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
import os

class DualDescriptorAB(nn.Module):
    """
    Dual Descriptor with GPU acceleration using PyTorch:
      - learnable coefficient matrix Acoeff ∈ R^{m×L}
      - fixed basis matrix Bbasis ∈ R^{L×m}, Bbasis[k][i] = cos(2π*(k+1)/(i+2))
      - learnable token embeddings via nn.Embedding
      - Supports both linear and nonlinear tokenization
      - Batch processing for GPU acceleration
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
        self.mean_t = None  
        self.mean_L = None
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Generate tokens
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
        self.vocab_size = len(self.tokens)
        
        # Token embeddings
        self.embedding = nn.Embedding(self.vocab_size, self.m)
        
        # Coefficient matrix A (m x L)
        self.Acoeff = nn.Parameter(torch.empty(self.m, self.L))
        
        # Fixed basis matrix B (L x m)
        Bbasis = torch.empty(self.L, self.m)
        for k in range(self.L):
            for i in range(self.m):
                Bbasis[k, i] = math.cos(2 * math.pi * (k+1) / (i+2))
        self.register_buffer('Bbasis', Bbasis)
        
        # Initialize parameters
        self.reset_parameters()
        self.to(self.device)
    
    def reset_parameters(self):
        """Initialize model parameters"""
        nn.init.uniform_(self.embedding.weight, -0.5, 0.5)
        nn.init.uniform_(self.Acoeff, -0.1, 0.1)
        
    def token_to_indices(self, token_list):
        """Convert list of tokens to tensor of indices"""
        return torch.tensor([self.token_to_idx[tok] for tok in token_list], 
                           device=self.device, dtype=torch.long)
    
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

    def describe(self, seq):
        """Compute N(k) vectors for each token in sequence"""
        toks = self.extract_tokens(seq)
        if not toks:
            return []
        
        token_indices = self.token_to_indices(toks)
        k_positions = torch.arange(len(toks), dtype=torch.float32, device=self.device)
        
        # Get token embeddings
        x = self.embedding(token_indices)  # [seq_len, m]
        
        # Compute basis indices (k mod L)
        j_indices = (k_positions % self.L).long()
        
        # Get corresponding B basis rows
        B_rows = self.Bbasis[j_indices]  # [seq_len, m]
        
        # Compute scalar = B[j] • x for each position
        scalar = torch.sum(B_rows * x, dim=1)  # [seq_len]
        
        # Get A columns for each position
        A_cols = self.Acoeff[:, j_indices].t()  # [seq_len, m]
        
        # Compute Nk = scalar * A_cols
        Nk = A_cols * scalar.unsqueeze(1)  # [seq_len, m]
        
        return Nk.detach().cpu().numpy()
    
    def S(self, seq):
        """Compute cumulative sum of N(k) vectors"""
        toks = self.extract_tokens(seq)
        if not toks:
            return []
        
        token_indices = self.token_to_indices(toks)
        k_positions = torch.arange(len(toks), dtype=torch.float32, device=self.device)
        
        # Compute Nk vectors
        x = self.embedding(token_indices)
        j_indices = (k_positions % self.L).long()
        B_rows = self.Bbasis[j_indices]
        scalar = torch.sum(B_rows * x, dim=1)
        A_cols = self.Acoeff[:, j_indices].t()
        Nk = A_cols * scalar.unsqueeze(1)
        
        # Compute cumulative sum
        S_cum = torch.cumsum(Nk, dim=0)
        return S_cum.detach().cpu().numpy()

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
                    
                    # Compute Nk vectors for all positions in current sequence
                    x = self.embedding(token_indices)
                    j_indices = (k_positions % self.L).long()
                    B_rows = self.Bbasis[j_indices]
                    scalar = torch.sum(B_rows * x, dim=1)
                    A_cols = self.Acoeff[:, j_indices].t()
                    Nk_batch = A_cols * scalar.unsqueeze(1)
                    
                    # Compute sequence-level prediction: average of all N(k) vectors
                    seq_pred = torch.mean(Nk_batch, dim=0)
                    
                    # Calculate loss for this sequence (MSE between prediction and target)
                    seq_loss = torch.sum((seq_pred - target) ** 2)
                    batch_loss += seq_loss
                    batch_sequence_count += 1
                    
                    # Clean up intermediate tensors to free GPU memory
                    del Nk_batch, seq_pred, token_indices, k_positions, x, B_rows, A_cols, scalar
                    
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
                      #f"Sequences = {total_sequences}")
            
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
                    
                    # Compute Nk for current tokens using batch processing
                    x = self.embedding(current_indices_tensor)
                    j_indices = (k_tensor % self.L).long()
                    B_rows = self.Bbasis[j_indices]
                    scalar = torch.sum(B_rows * x, dim=1)
                    A_cols = self.Acoeff[:, j_indices].t()
                    Nk_batch = A_cols * scalar.unsqueeze(1)
                    
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
                    del k_tensor, current_indices_tensor, Nk_batch, targets, loss, x, B_rows, A_cols, scalar
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
        """
        Compute and store statistics for reconstruction and generation with memory optimization.
        Calculates mean token count and mean target vector across all sequences.
        
        Args:
            seqs: List of character sequences
            batch_size: Batch size for processing sequences to optimize memory usage
        """
        total_token_count = 0
        total_t = torch.zeros(self.m, device=self.device)
        
        with torch.no_grad():
            for i in range(0, len(seqs), batch_size):
                batch_seqs = seqs[i:i+batch_size]
                batch_token_count = 0
                batch_t_sum = torch.zeros(self.m, device=self.device)
                
                for seq in batch_seqs:
                    tokens = self.extract_tokens(seq)
                    batch_token_count += len(tokens)
                    
                    if tokens:
                        token_indices = self.token_to_indices(tokens)
                        k_positions = torch.arange(len(tokens), dtype=torch.float32, device=self.device)
                        
                        # Compute Nk vectors using batch processing
                        x = self.embedding(token_indices)
                        j_indices = (k_positions % self.L).long()
                        B_rows = self.Bbasis[j_indices]
                        scalar = torch.sum(B_rows * x, dim=1)
                        A_cols = self.Acoeff[:, j_indices].t()
                        Nk_batch = A_cols * scalar.unsqueeze(1)
                        
                        batch_t_sum += Nk_batch.sum(dim=0)
                        
                        # Clean up intermediate tensors
                        del token_indices, k_positions, Nk_batch, x, B_rows, A_cols, scalar
                
                total_token_count += batch_token_count
                total_t += batch_t_sum
                
                # Clean batch tensors
                del batch_t_sum
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Store statistics for reconstruction and generation
        self.mean_L = total_token_count / len(seqs) if seqs else 0
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
        """Reconstruct representative sequence"""
        assert self.trained, "Model must be trained first"
        n_tokens = round(self.mean_L)
        mean_t_tensor = torch.tensor(self.mean_t, dtype=torch.float32, device=self.device)
        seq_tokens = []
        
        # Precompute all token embeddings
        all_token_indices = torch.arange(self.vocab_size, device=self.device)
        all_embeddings = self.embedding(all_token_indices)
        
        for k in range(n_tokens):
            # Compute j index for basis
            j = k % self.L
            B_row = self.Bbasis[j].unsqueeze(0)  # [1, m]
            
            # Compute scalar = B[j] • x for all tokens
            scalar = torch.sum(B_row * all_embeddings, dim=1)  # [vocab_size]
            
            # Compute Nk = scalar * A[:,j]
            A_col = self.Acoeff[:, j]  # [m]
            Nk_all = A_col * scalar.unsqueeze(1)  # [vocab_size, m]
            
            # Compute error to mean target
            errors = torch.sum((Nk_all - mean_t_tensor) ** 2, dim=1)
            min_idx = torch.argmin(errors).item()
            seq_tokens.append(self.idx_to_token[min_idx])
        
        return ''.join(seq_tokens)

    def generate(self, L, tau=0.0):
        """Generate sequence with temperature-controlled randomness"""
        assert self.trained, "Model must be trained first"
        if tau < 0:
            raise ValueError("Temperature must be non-negative")
            
        num_blocks = (L + self.rank - 1) // self.rank
        mean_t_tensor = torch.tensor(self.mean_t, dtype=torch.float32, device=self.device)
        generated_tokens = []
        
        # Precompute all token embeddings
        all_token_indices = torch.arange(self.vocab_size, device=self.device)
        all_embeddings = self.embedding(all_token_indices)
        
        for k in range(num_blocks):
            # Compute j index for basis
            j = k % self.L
            B_row = self.Bbasis[j].unsqueeze(0)  # [1, m]
            
            # Compute scalar = B[j] • x for all tokens
            scalar = torch.sum(B_row * all_embeddings, dim=1)  # [vocab_size]
            
            # Compute Nk = scalar * A[:,j]
            A_col = self.Acoeff[:, j]  # [m]
            Nk_all = A_col * scalar.unsqueeze(1)  # [vocab_size, m]
            
            # Compute scores (negative MSE)
            errors = torch.sum((Nk_all - mean_t_tensor) ** 2, dim=1)
            scores = -errors
            
            # Select token
            if tau == 0:  # Deterministic
                best_idx = torch.argmax(scores).item()
                best_tok = self.idx_to_token[best_idx]
                generated_tokens.append(best_tok)
            else:  # Stochastic
                probs = torch.softmax(scores / tau, dim=0).detach().cpu().numpy()
                chosen_idx = np.random.choice(self.vocab_size, p=probs)
                chosen_tok = self.idx_to_token[chosen_idx]
                generated_tokens.append(chosen_tok)
        
        # Trim to exact length
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
        Train the interaction matrix I on vector sequences using gradient descent.
        Supports two training modes:
          - 'gap': Predicts current vector (self-consistency)
          - 'reg': Predicts next vector (auto-regressive)
        
        Parameters:
            vec_seqs (list): List of vector sequences (each sequence is a list of m-dimensional vectors)
            learning_rate (float): Step size for gradient updates
            max_iters (int): Maximum training iterations
            tol (float): Convergence tolerance
            auto_mode (str): Training mode - 'gap' or 'reg'
            continued (bool): Continue training existing I matrix
            decay_rate (float): Learning rate decay factor (1.0 = no decay)
            
        Returns:
            list: Training history (loss values per iteration)
        """
        if auto_mode not in ('gap', 'reg'):
            raise ValueError("auto_mode must be either 'gap' or 'reg'")

        # Initialize I matrix if needed
        if not continued or not hasattr(self, 'I'):
            self.I = nn.Parameter(torch.empty(self.m, self.m))
            nn.init.uniform_(self.I, -0.1, 0.1)
            self.to(self.device)
        
        # Calculate total training samples
        total_samples = 0
        for seq in vec_seqs:
            if auto_mode == 'gap':
                total_samples += len(seq)  # All vectors are samples
            else:  # 'reg' mode
                total_samples += max(0, len(seq) - 1)  # Vectors except last
                
        if total_samples == 0:
            raise ValueError("No training samples found")
            
        # Set up optimizer and scheduler
        optimizer = optim.Adam([self.I], lr=learning_rate)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)
        
        history = []  # Store loss per iteration
        prev_loss = float('inf')
        
        for it in range(max_iters):
            # Initialize gradient matrix for I
            total_loss = 0.0
            
            # Process all vector sequences
            for seq in vec_seqs:
                    
                # Convert sequence to tensor
                seq_tensor = torch.tensor(seq, dtype=torch.float32, device=self.device)
                
                # Process vectors based on mode
                for k in range(len(seq)):
                    # Skip last vector in 'reg' mode (no next vector)
                    if auto_mode == 'reg' and k == len(seq) - 1:
                        continue
                        
                    current_vec = seq_tensor[k]
                    j = k % self.L  # Basis index
                    basis_vec = self.Bbasis[j]  # Get basis vector for position k
                    
                    # Compute N(k) for current vector at position k
                    Nk = torch.zeros(self.m, device=self.device)
                    for i in range(self.m):
                        for j_dim in range(self.m):
                            # Basis modulation: I[i][j_dim] * current_vec[j_dim] * basis_vec[j_dim]
                            Nk[i] += self.I[i, j_dim] * current_vec[j_dim] * basis_vec[j_dim]
                    
                    # Determine target based on mode
                    if auto_mode == 'gap':
                        target = current_vec  # Self-consistency
                    else:  # 'reg' mode
                        target = seq_tensor[k + 1]  # Next vector prediction
                    
                    # Compute loss and gradients
                    loss = torch.sum((Nk - target) ** 2)
                    total_loss += loss.item()
                    
                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            
            # Update learning rate
            scheduler.step()
            
            # Record and print progress
            avg_loss = total_loss / total_samples
            history.append(avg_loss)
            if it % print_every == 0 or it == max_iters - 1:
                mode_display = "Gap" if auto_mode == 'gap' else "Reg"
                current_lr = scheduler.get_last_lr()[0]
                print(f"PartTrain({mode_display}) Iter {it:3d}: loss = {avg_loss:.6e}, LR = {current_lr:.6f}")
            
            # Check convergence
            if prev_loss - avg_loss < tol:
                print(f"Converged after {it+1} iterations")
                break
            prev_loss = avg_loss
        
        # Compute and store mean vector for generation
        total_vectors = 0
        total_vec_sum = torch.zeros(self.m, device=self.device)
        for seq in vec_seqs:
            seq_tensor = torch.tensor(seq, dtype=torch.float32, device=self.device)
            total_vectors += len(seq)
            total_vec_sum += seq_tensor.sum(dim=0)
        
        self.mean_vector = (total_vec_sum / total_vectors).cpu().numpy()
        
        return history

    def part_generate(self, L, tau=0.0, mode='reg'):
        """
        Generate a sequence of vectors using the trained I matrix
        
        Parameters:
            L (int): Length of sequence to generate
            tau (float): Temperature for randomness (0 = deterministic)
            mode (str): Generation mode - 'gap' or 'reg' (must match training)
                
        Returns:
            list: Generated sequence of m-dimensional vectors
        """
        if not hasattr(self, 'I'):
            raise RuntimeError("I matrix not initialized - train first")
            
        if tau < 0:
            raise ValueError("Temperature must be non-negative")
            
        sequence = []
        
        if mode == 'gap':
            # Gap mode: Generate independent reconstructions
            for k in range(L):
                j = k % self.L  # Basis index
                basis_vec = self.Bbasis[j]  # Basis vector for position
                
                # Start with mean vector
                current_vec = torch.tensor(self.mean_vector, dtype=torch.float32, device=self.device)
                
                # Compute reconstruction at position k
                reconstructed_vec = torch.zeros(self.m, device=self.device)
                for i in range(self.m):
                    for j_dim in range(self.m):
                        reconstructed_vec[i] += self.I[i, j_dim] * current_vec[j_dim] * basis_vec[j_dim]
                
                # Add temperature-controlled noise
                if tau > 0:
                    noise = torch.normal(0, tau, size=(self.m,), device=self.device)
                    reconstructed_vec += noise
                    
                sequence.append(reconstructed_vec.detach().cpu().numpy())
                
        else:  # 'reg' mode
            # Reg mode: Auto-regressive generation
            current_vec = torch.tensor(self.mean_vector, dtype=torch.float32, device=self.device)
            
            for k in range(L):
                j = k % self.L  # Basis index
                basis_vec = self.Bbasis[j]  # Basis vector for position
                
                # Compute prediction for next vector
                next_vec_pred = torch.zeros(self.m, device=self.device)
                for i in range(self.m):
                    for j_dim in range(self.m):
                        next_vec_pred[i] += self.I[i, j_dim] * current_vec[j_dim] * basis_vec[j_dim]
                
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
        
        # Stage 2: Train I matrix on vector sequences
        print("\n" + "="*50)
        print("Stage 2: Training I matrix on vector sequences")
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
            scalar = torch.sum(self.Bbasis[j] * self.embedding(torch.tensor([self.token_to_idx[token]], device=self.device)))
            return (self.Acoeff[:, j] * scalar).detach().cpu().numpy()
        
        # Stage 1: Generate initial sequence with character model
        init_seq = self.generate(L, tau=tau)
        
        # Stage 2: Compute S(l) vectors for initial sequence
        s_vectors = self.S(init_seq)
        
        # Stage 3: Refine vectors using I-matrix with specified mode
        refined_vectors = self.part_generate(len(s_vectors), mode=mode, tau=tau)
        
        # Stage 4: Reconstruct character sequence using both models
        generated_tokens = []
        current_s = np.zeros(self.m)  # Initialize cumulative vector
        
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
    print("Dual Descriptor AB - PyTorch GPU Accelerated Version")
    print("Optimized with batch processing")
    print("="*50)
    
    # Set random seeds for reproducibility
    torch.manual_seed(11)
    random.seed(11)
    
    charset = ['A','C','G','T']
    vec_dim = 3
    bas_dim = 300
    seq_num = 100
    
    # Generate sequences and random targets
    seqs, t_list = [], []
    for _ in range(seq_num):
        L = random.randint(200, 300)
        seq = ''.join(random.choices(charset, k=L))
        seqs.append(seq)
        t_list.append([random.uniform(-1,1) for _ in range(vec_dim)])
    
    # Create model
    dd = DualDescriptorAB(
        charset, 
        vec_dim=vec_dim, 
        bas_dim=bas_dim, 
        rank=3, 
        mode='linear',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Display device information
    print(f"\nUsing device: {dd.device}")
    print(f"Number of tokens: {len(dd.tokens)}")
    
    # === Gradient Descent Training ===
    print("\n" + "="*50)
    print("Testing Gradient Descent Training")
    print("="*50)
    
    # Train using gradient descent
    print("\nStarting gradient descent training...")
    grad_history = dd.grad_train(
        seqs, 
        t_list,
        learning_rate=0.1,
        max_iters=100,
        tol=1e-66,
        print_every=5,
        decay_rate=0.99,
        batch_size=1024
    )
    
    # Predict target for first sequence
    aseq = seqs[0]
    t_pred = dd.predict_t(aseq)
    print(f"\nPredicted t for first sequence: {[round(x.item(), 4) for x in t_pred]}")
    
    # Calculate prediction correlation
    pred_t_list = [dd.predict_t(seq) for seq in seqs]
    
    corr_sum = 0.0
    for i in range(dd.m):
        actu_t = [t_vec[i] for t_vec in t_list]
        pred_t = [t_vec[i] for t_vec in pred_t_list]
        corr = correlation(actu_t, pred_t)
        print(f"Prediction correlation: {corr:.4f}")
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
    # Set random seeds
    torch.manual_seed(2)
    random.seed(2)
    
    # Define parameters
    charset = ['A','C','G','T']
    vec_dim = 3
    bas_dim = 100
    
    # Generate training sequences
    print("\n=== Generating Training Sequences ===")
    seqs = []
    for i in range(30):
        L = random.randint(100, 200)
        seq = ''.join(random.choices(charset, k=L))
        seqs.append(seq)
        print(f"Generated sequence {i+1}: length={L}, first 10 chars: {seq[:10]}...")
    
    # Create model for gap filling
    print("\n=== Creating Dual Descriptor Model for Gap Filling ===")
    dd_gap = DualDescriptorAB(
        charset,
        vec_dim=vec_dim,
        bas_dim=bas_dim,
        rank=1,
        mode='linear',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Train in gap filling mode
    print("\n=== Starting Gap Filling Training ===")
    gap_history = dd_gap.auto_train(
        seqs,
        auto_mode='gap',
        max_iters=300,
        learning_rate=0.001,
        decay_rate=0.995,
        print_every=20,
        batch_size=1024
    )
    
    # Create model for auto-regressive training
    print("\n=== Creating Dual Descriptor Model for Auto-Regressive Training ===")
    dd_reg = DualDescriptorAB(
        charset,
        vec_dim=vec_dim,
        bas_dim=bas_dim,
        rank=1,
        mode='linear',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Train in auto-regressive mode
    print("\n=== Starting Auto-Regressive Training ===")
    reg_history = dd_reg.auto_train(
        seqs,
        auto_mode='reg',
        max_iters=300,
        learning_rate=0.01,
        decay_rate=0.99,
        print_every=20,
        batch_size=1024
    )
    
    # Generate new sequences
    print("\n=== Generating New Sequences ===")
    
    # From gap model
    gap_seq = dd_gap.generate(L=40, tau=0.0)
    print(f"Gap model generation: {gap_seq}")
    
    # From reg model
    reg_seq = dd_reg.generate(L=40, tau=0.0)
    print(f"Reg model generation: {reg_seq}")
    
    # Save and load model
    print("\n=== Model Persistence Test ===")
    dd_reg.save("auto_trained_model.pkl")
    
    # Load model
    dd_loaded = DualDescriptorAB(
        charset,
        vec_dim=vec_dim,
        bas_dim=bas_dim,
        rank=1,
        mode='linear',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    ).load("auto_trained_model.pkl")
    
    print("Model loaded successfully. Generating with loaded model:")
    print(dd_loaded.generate(L=20, tau=0.0))
    
    # === Part Train/Generate Example ===
    print("\n" + "="*50)
    print("Part Train/Generate Example")
    print("="*50)
    
    # Create new model
    dd_part = DualDescriptorAB(charset=charset, rank=1, vec_dim=2, device='cuda' if torch.cuda.is_available() else 'cpu')
    
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

    # === Double Train/Generate Example ===
    print("\n" + "="*50)
    print("Double Train/Generate Example")
    print("="*50)
    
    # Create and train model using double_train
    dd_double = DualDescriptorAB(
        charset=['A','C','G','T'], 
        rank=1, 
        vec_dim=2,
        mode='nonlinear',
        user_step=1,
        device='cuda' if torch.cuda.is_available() else 'cpu'
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
        'learning_rate': 0.1
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
    
    print("\n=== All Tests Completed ===")
