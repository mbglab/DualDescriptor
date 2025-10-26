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
               continued=False, decay_rate=1.0, print_every=10, batch_size=1024,
               checkpoint_file=None, checkpoint_interval=10):
        """
        Train model using gradient descent with batch processing and memory optimization.
        GPU-accelerated with PyTorch, featuring checkpoint saving and memory management.
        
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
        
        # Precompute token indices for all sequences (memory optimized approach)
        all_samples = []
        total_token_count = 0
        
        for seq_idx, seq in enumerate(seqs):
            toks = self.extract_tokens(seq)
            if not toks:
                continue
                
            token_indices = self.token_to_indices(toks)
            total_token_count += len(toks)
            
            # Store samples with position information
            for pos in range(len(toks)):
                all_samples.append((pos, seq_idx, token_indices[pos].item()))
        
        total_samples = len(all_samples)
        if total_samples == 0:
            print("Warning: No training samples found")
            return []
        
        # Convert target vectors to tensor
        t_tensors = [torch.tensor(t, dtype=torch.float32, device=self.device) for t in t_list]
        
        # Set up optimizer
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)
        
        # Load optimizer and scheduler states if resuming
        if continued and checkpoint_file and os.path.exists(checkpoint_file):
            optimizer.load_state_dict(optimizer_state)
            scheduler.load_state_dict(scheduler_state)
        
        prev_loss = float('inf') if start_iter == 0 else history[-1] if history else float('inf')
        
        for it in range(start_iter, max_iters):
            epoch_loss = 0.0
            processed_samples = 0
            
            # Shuffle samples
            random.shuffle(all_samples)
            
            # Process in batches with fixed batch size
            for batch_start in range(0, total_samples, batch_size):
                batch_end = min(batch_start + batch_size, total_samples)
                batch_samples = all_samples[batch_start:batch_end]
                
                optimizer.zero_grad()
                
                # Prepare batch data directly as tensors
                k_list = []
                token_idx_list = []
                target_list = []
                
                for pos, seq_idx, token_idx in batch_samples:
                    k_list.append(pos)
                    token_idx_list.append(token_idx)
                    target_list.append(t_tensors[seq_idx])
                
                # Create tensors directly on GPU
                k_tensor = torch.tensor(k_list, dtype=torch.float32, device=self.device)
                token_indices_tensor = torch.tensor(token_idx_list, device=self.device)
                targets = torch.stack(target_list)
                
                # Compute Nk vectors using batch processing
                x = self.embedding(token_indices_tensor)
                j_indices = (k_tensor % self.L).long()
                B_rows = self.Bbasis[j_indices]
                scalar = torch.sum(B_rows * x, dim=1)
                A_cols = self.Acoeff[:, j_indices].t()
                Nk_batch = A_cols * scalar.unsqueeze(1)
                
                # Compute loss
                loss = torch.mean(torch.sum((Nk_batch - targets) ** 2, dim=1))
                loss.backward()
                optimizer.step()
                
                batch_loss = loss.item() * len(batch_samples)
                epoch_loss += batch_loss
                processed_samples += len(batch_samples)
                
                # Clean up to free memory
                del k_tensor, token_indices_tensor, targets, Nk_batch, loss, x, B_rows, A_cols, scalar
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Average loss over epoch
            avg_loss = epoch_loss / processed_samples if processed_samples > 0 else 0.0
            history.append(avg_loss)
            
            # Update best model state if current loss is lower
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_model_state = copy.deepcopy(self.state_dict())
            
            if it % print_every == 0 or it == max_iters - 1:
                print(f"GD Iter {it:3d}: D = {avg_loss:.6e}, LR = {scheduler.get_last_lr()[0]:.6f}")
            
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
            
            scheduler.step()
            
            # Clean GPU memory at the end of each iteration
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
        Train the model using self-supervised learning with memory optimization
        and checkpoint mechanism. Supports both 'gap' and 'reg' training modes.
        
        Args:
            seqs: List of character sequences
            auto_mode: 'gap' or 'reg' training mode
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
        
        # Precompute all training samples (memory optimized approach)
        all_samples = []
        
        for seq_idx, seq in enumerate(seqs):
            tokens = self.extract_tokens(seq)
            if not tokens:
                continue
                
            token_indices = self.token_to_indices(tokens)
            
            if auto_mode == 'gap':
                # Each token is a sample (position k, token_idx)
                for k, token_idx in enumerate(token_indices):
                    all_samples.append((k, token_idx.item()))
            else:  # 'reg' mode
                # Each token except last is a sample (position k, current token, next token)
                for k in range(len(tokens) - 1):
                    all_samples.append((k, token_indices[k].item(), token_indices[k+1].item()))
        
        if not all_samples:
            raise ValueError("No training samples found")
            
        # Set up optimizer
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)
        
        # Load optimizer and scheduler states if resuming
        if continued and checkpoint_file and os.path.exists(checkpoint_file):
            optimizer.load_state_dict(optimizer_state)
            scheduler.load_state_dict(scheduler_state)
        
        prev_loss = float('inf') if start_iter == 0 else history[-1] if history else float('inf')
        total_samples = len(all_samples)
        
        for it in range(start_iter, max_iters):
            epoch_loss = 0.0
            processed_samples = 0
            
            # Shuffle samples
            random.shuffle(all_samples)
            
            # Process in batches with fixed batch size
            for batch_start in range(0, total_samples, batch_size):
                batch_end = min(batch_start + batch_size, total_samples)
                batch_samples = all_samples[batch_start:batch_end]
                
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
                epoch_loss += batch_loss
                processed_samples += len(batch_samples)
                
                # Clean up to free memory
                del k_tensor, current_indices_tensor, Nk_batch, targets, loss, x, B_rows, A_cols, scalar
                if auto_mode == 'reg':
                    del target_indices_tensor
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Average loss over epoch
            avg_loss = epoch_loss / processed_samples if processed_samples > 0 else 0.0
            history.append(avg_loss)
            
            # Update best model state if current loss is lower
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_model_state = copy.deepcopy(self.state_dict())
            
            if it % print_every == 0 or it == max_iters - 1:
                mode_display = "Gap" if auto_mode == 'gap' else "Reg"
                current_lr = scheduler.get_last_lr()[0]
                print(f"AutoTrain({mode_display}) Iter {it:3d}: loss = {avg_loss:.6f}, LR = {current_lr:.6f}")
            
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
            
            scheduler.step()
            
            # Clean GPU memory at the end of each iteration
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
    
    print("\n=== All Tests Completed ===")
