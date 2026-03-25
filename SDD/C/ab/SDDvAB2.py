# Copyright (C) 2005-2025, Bin-Guang Ma (mbg@mail.hzau.edu.cn); SPDX-License-Identifier: MIT
# The Spatial Dual Descriptor Vector class (AB matrix form) for 2D character arrays implemented with PyTorch
# This program is for the demonstration of methodology and not fully refined.
# Author: Bin-Guang Ma (assisted by DeepSeek); Date: 2025-7-29 ~ 2026-3-25

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
from collections import defaultdict

class SpatialDualDescriptorAB2(nn.Module):
    """
    Spatial Dual Descriptor for 2D character arrays with GPU acceleration using PyTorch:
      - learnable coefficient matrix Acoeff ∈ R^{m×(Lh×Lw)}
      - fixed basis matrix Bbasis ∈ R^{(Lh×Lw)×m}, 
        Bbasis[ki, kj, i] = cos(2π*(ki+1)/(i+2)) * cos(2π*(kj+1)/(i+2))
      - learnable token embeddings via nn.Embedding (tokens are flattened 2D windows)
      - Supports both linear and nonlinear tokenization in 2D
      - Batch processing for GPU acceleration
      - Supports regression, classification, and multi-label classification tasks
    """
    def __init__(self, charset, vec_dim=4, bas_dim_h=50, bas_dim_w=50, rank=1, rank_mode='drop', 
                 mode='linear', user_step=None, device='cuda'):
        super().__init__()
        self.charset = list(charset)
        self.m = vec_dim
        self.Lh = bas_dim_h
        self.Lw = bas_dim_w
        self.L_total = self.Lh * self.Lw
        self.rank = rank
        self.rank_mode = rank_mode
        assert mode in ('linear','nonlinear')
        self.mode = mode
        self.step = user_step
        self.trained = False
        self.mean_t = None  
        self.mean_L = None
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Generate tokens: all possible rank×rank patterns flattened row-wise
        # Number of tokens = len(charset)^(rank*rank) (may be large, keep rank small)
        self.window_area = rank * rank
        if self.rank_mode == 'pad':
            # For pad mode we need to include padding character '_' which is not in charset
            # So we temporarily extend charset for token generation
            extended_charset = charset + ['_']
            # tokens = all combinations of length window_area, but with padding allowed
            # This is huge; we will generate on-the-fly or limit. For simplicity we assume pad mode not used or rank small.
            # Here we generate tokens from extended charset but this may be large.
            # We'll use a simpler approach: tokens are all combinations of characters from charset + '_'
            # But this may be too many; we'll use the same method as original but with '_' allowed.
            # Actually original pad mode uses '_' for padding, so tokens include '_'. So we need to include '_' in charset for token generation.
            pass
        # For drop mode, tokens are exactly rank×rank patterns from charset
        # We generate all combinations of charset repeated window_area times
        # This is huge for rank>2, but we assume rank is small (e.g., 2) for practical use.
        # In production, one might use a more efficient representation.
        toks = []
        for pattern in itertools.product(self.charset, repeat=self.window_area):
            toks.append(''.join(pattern))
        # For pad mode, we also need patterns containing '_' at the end for incomplete windows.
        # But in pad mode we will pad incomplete windows with '_', so the token vocabulary includes those.
        # We'll generate all patterns from charset + '_' for pad mode.
        if self.rank_mode == 'pad':
            extended_charset = self.charset + ['_']
            pad_toks = []
            for pattern in itertools.product(extended_charset, repeat=self.window_area):
                pad_toks.append(''.join(pattern))
            toks = pad_toks
        # Remove duplicates (shouldn't be any)
        self.tokens = sorted(set(toks))
        self.token_to_idx = {token: idx for idx, token in enumerate(self.tokens)}
        self.idx_to_token = {idx: token for idx, token in enumerate(self.tokens)}
        self.vocab_size = len(self.tokens)
        
        # Token embeddings
        self.embedding = nn.Embedding(self.vocab_size, self.m)
        
        # Coefficient matrix A (m × L_total)
        self.Acoeff = nn.Parameter(torch.empty(self.m, self.L_total))
        
        # Fixed basis matrix B (L_total × m)
        Bbasis = torch.empty(self.L_total, self.m)
        for idx in range(self.L_total):
            ki = idx // self.Lw
            kj = idx % self.Lw
            for i in range(self.m):
                Bbasis[idx, i] = math.cos(2 * math.pi * (ki+1) / (i+2)) * \
                                 math.cos(2 * math.pi * (kj+1) / (i+2))
        self.register_buffer('Bbasis', Bbasis)
        
        # Classification head for multi-class tasks (initialized when needed)
        self.num_classes = None
        self.classifier = None
        
        # Label head for multi-label tasks (initialized when needed)
        self.num_labels = None
        self.labeller = None
        
        # Initialize parameters
        self.reset_parameters()
        self.to(self.device)
    
    def reset_parameters(self):
        """Initialize model parameters"""
        nn.init.uniform_(self.embedding.weight, -0.5, 0.5)
        nn.init.uniform_(self.Acoeff, -0.1, 0.1)
        
        if self.classifier is not None:
            nn.init.normal_(self.classifier.weight, 0, 0.01)
            if self.classifier.bias is not None:
                nn.init.constant_(self.classifier.bias, 0)
                
        if self.labeller is not None:
            nn.init.xavier_uniform_(self.labeller.weight)
            nn.init.zeros_(self.labeller.bias)
    
    def token_to_indices(self, token_list):
        """Convert list of tokens to tensor of indices"""
        return torch.tensor([self.token_to_idx[tok] for tok in token_list], 
                           device=self.device, dtype=torch.long)
    
    def extract_tokens(self, grid):
        """
        Extract 2D window tokens from a 2D character grid.
        
        Args:
            grid: list of list of chars, shape (H, W)
            
        Returns:
            tokens: list of token strings (flattened window)
            positions: list of (row, col) tuples for top-left corner of each window
        """
        H = len(grid)
        if H == 0:
            return [], []
        W = len(grid[0])
        
        # Linear mode: sliding window with step=1 in both dimensions
        if self.mode == 'linear':
            step = 1
        else:
            step = self.step or self.rank
        
        tokens = []
        positions = []
        for i in range(0, H - self.rank + 1, step):
            for j in range(0, W - self.rank + 1, step):
                # Extract window
                window = [grid[i+di][j+dj] for di in range(self.rank) for dj in range(self.rank)]
                token = ''.join(window)
                tokens.append(token)
                positions.append((i, j))
        
        # Handle incomplete windows at edges if rank_mode == 'pad'
        if self.rank_mode == 'pad':
            # For rows
            for i in range(0, H, step):
                if i + self.rank > H:
                    # Incomplete row-wise
                    for j in range(0, W - self.rank + 1, step):
                        # Extract partial row window
                        window = []
                        for di in range(self.rank):
                            for dj in range(self.rank):
                                if i+di < H and j+dj < W:
                                    window.append(grid[i+di][j+dj])
                                else:
                                    window.append('_')
                        token = ''.join(window)
                        tokens.append(token)
                        positions.append((i, j))
                # For columns at right edge
                for j in range(0, W, step):
                    if j + self.rank > W and i + self.rank <= H:
                        # Incomplete column-wise
                        window = []
                        for di in range(self.rank):
                            for dj in range(self.rank):
                                if i+di < H and j+dj < W:
                                    window.append(grid[i+di][j+dj])
                                else:
                                    window.append('_')
                        token = ''.join(window)
                        tokens.append(token)
                        positions.append((i, j))
                # For bottom-right corner when both incomplete
                if i + self.rank > H and j + self.rank > W:
                    window = []
                    for di in range(self.rank):
                        for dj in range(self.rank):
                            if i+di < H and j+dj < W:
                                window.append(grid[i+di][j+dj])
                            else:
                                window.append('_')
                    token = ''.join(window)
                    tokens.append(token)
                    positions.append((i, j))
        return tokens, positions

    def describe(self, grid):
        """Compute N(k) vectors for each window in grid"""
        tokens, positions = self.extract_tokens(grid)
        if not tokens:
            return []
        
        token_indices = self.token_to_indices(tokens)
        # Create tensor of positions (row, col)
        pos_tensor = torch.tensor(positions, device=self.device, dtype=torch.long)  # shape (num_windows, 2)
        
        # Compute j indices: j = (row % Lh) * Lw + (col % Lw)
        j_indices = (pos_tensor[:, 0] % self.Lh) * self.Lw + (pos_tensor[:, 1] % self.Lw)  # shape (num_windows,)
        
        # Get token embeddings
        x = self.embedding(token_indices)  # [num_windows, m]
        
        # Get B rows for each j
        B_rows = self.Bbasis[j_indices]  # [num_windows, m]
        
        # Compute scalar = B[j] • x for each window
        scalar = torch.sum(B_rows * x, dim=1)  # [num_windows]
        
        # Get A columns for each j
        A_cols = self.Acoeff[:, j_indices].t()  # [num_windows, m]
        
        # Compute Nk = scalar * A_cols
        Nk = A_cols * scalar.unsqueeze(1)  # [num_windows, m]
        
        return Nk.detach().cpu().numpy()
    
    def S(self, grid):
        """Compute cumulative sum of N(k) vectors (ordered by window traversal order)"""
        tokens, positions = self.extract_tokens(grid)
        if not tokens:
            return []
        
        token_indices = self.token_to_indices(tokens)
        pos_tensor = torch.tensor(positions, device=self.device, dtype=torch.long)
        j_indices = (pos_tensor[:, 0] % self.Lh) * self.Lw + (pos_tensor[:, 1] % self.Lw)
        
        x = self.embedding(token_indices)
        B_rows = self.Bbasis[j_indices]
        scalar = torch.sum(B_rows * x, dim=1)
        A_cols = self.Acoeff[:, j_indices].t()
        Nk = A_cols * scalar.unsqueeze(1)
        
        S_cum = torch.cumsum(Nk, dim=0)
        return S_cum.detach().cpu().numpy()

    def D(self, grids, t_list):
        """
        Compute mean squared deviation D across grids:
        D = average over all windows of (N(k)-t_grid)^2
        
        Args:
            grids: List of 2D character grids (list of list of list of chars)
            t_list: List of target vectors corresponding to each grid
            
        Returns:
            float: Average mean squared deviation across all windows and grids
        """
        total_loss = 0.0
        total_windows = 0
        
        t_tensors = [torch.tensor(t, dtype=torch.float32, device=self.device) for t in t_list]
        
        for grid, t in zip(grids, t_tensors):
            tokens, positions = self.extract_tokens(grid)
            if not tokens:
                continue
                
            token_indices = self.token_to_indices(tokens)
            pos_tensor = torch.tensor(positions, device=self.device, dtype=torch.long)
            j_indices = (pos_tensor[:, 0] % self.Lh) * self.Lw + (pos_tensor[:, 1] % self.Lw)
            
            x = self.embedding(token_indices)
            B_rows = self.Bbasis[j_indices]
            scalar = torch.sum(B_rows * x, dim=1)
            A_cols = self.Acoeff[:, j_indices].t()
            Nk_batch = A_cols * scalar.unsqueeze(1)
            
            losses = torch.sum((Nk_batch - t) ** 2, dim=1)
            total_loss += losses.sum().item()
            total_windows += len(tokens)
            
            # Clean up
            del x, B_rows, A_cols, Nk_batch, losses
        
        return total_loss / total_windows if total_windows else 0.0

    def d(self, grid, t):
        """
        Compute pattern deviation value (d) for a single grid.
        """
        return self.D([grid], [t])

    def _compute_training_statistics(self, grids, batch_size=10):
        """
        Compute and store statistics for reconstruction and generation.
        Calculates mean token count and mean target vector across all grids.
        
        Args:
            grids: List of 2D character grids
            batch_size: Batch size for processing grids to optimize memory usage
        """
        total_window_count = 0
        total_t = torch.zeros(self.m, device=self.device)
        
        with torch.no_grad():
            for i in range(0, len(grids), batch_size):
                batch_grids = grids[i:i+batch_size]
                batch_window_count = 0
                batch_t_sum = torch.zeros(self.m, device=self.device)
                
                for grid in batch_grids:
                    tokens, positions = self.extract_tokens(grid)
                    batch_window_count += len(tokens)
                    
                    if tokens:
                        token_indices = self.token_to_indices(tokens)
                        pos_tensor = torch.tensor(positions, device=self.device, dtype=torch.long)
                        j_indices = (pos_tensor[:, 0] % self.Lh) * self.Lw + (pos_tensor[:, 1] % self.Lw)
                        
                        x = self.embedding(token_indices)
                        B_rows = self.Bbasis[j_indices]
                        scalar = torch.sum(B_rows * x, dim=1)
                        A_cols = self.Acoeff[:, j_indices].t()
                        Nk_batch = A_cols * scalar.unsqueeze(1)
                        
                        batch_t_sum += Nk_batch.sum(dim=0)
                        
                        del token_indices, pos_tensor, Nk_batch, x, B_rows, A_cols, scalar
                
                total_window_count += batch_window_count
                total_t += batch_t_sum
                
                del batch_t_sum
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        self.mean_window_count = total_window_count / len(grids) if grids else 0
        self.mean_t = (total_t / total_window_count).cpu().numpy() if total_window_count > 0 else np.zeros(self.m)
    
    def predict_t(self, grid):
        """Predict target vector as average of N(k) vectors over all windows"""
        tokens, positions = self.extract_tokens(grid)
        if not tokens:
            return np.zeros(self.m)
        
        Nk = self.describe(grid)
        return np.mean(Nk, axis=0)

    def predict_c(self, grid):
        """
        Predict class label for a grid using the classification head.
        
        Args:
            grid: 2D character grid
            
        Returns:
            tuple: (predicted_class, class_probabilities)
        """
        if self.classifier is None:
            raise ValueError("Model must be trained first for classification")
        
        seq_vector = self.predict_t(grid)
        seq_vector_tensor = torch.tensor(seq_vector, dtype=torch.float32, device=self.device)
        
        with torch.no_grad():
            logits = self.classifier(seq_vector_tensor.unsqueeze(0))
            probabilities = torch.softmax(logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            
        return predicted_class, probabilities[0].cpu().numpy()

    def predict_l(self, grid, threshold=0.5):
        """
        Predict multi-label classification for a grid.
        
        Args:
            grid: 2D character grid
            threshold: Probability threshold for binary classification (default: 0.5)
            
        Returns:
            numpy.ndarray: Binary label predictions (0 or 1 for each label)
            numpy.ndarray: Probability scores for each label
        """
        assert self.labeller is not None, "Model must be trained first for label prediction"
        
        tokens, positions = self.extract_tokens(grid)
        if not tokens:
            return np.zeros(self.num_labels, dtype=np.float32), np.zeros(self.num_labels, dtype=np.float32)
        
        token_indices = self.token_to_indices(tokens)
        pos_tensor = torch.tensor(positions, device=self.device, dtype=torch.long)
        j_indices = (pos_tensor[:, 0] % self.Lh) * self.Lw + (pos_tensor[:, 1] % self.Lw)
        
        x = self.embedding(token_indices)
        B_rows = self.Bbasis[j_indices]
        scalar = torch.sum(B_rows * x, dim=1)
        A_cols = self.Acoeff[:, j_indices].t()
        Nk_batch = A_cols * scalar.unsqueeze(1)
        
        # Sequence representation: average of N(k) vectors
        seq_representation = torch.mean(Nk_batch, dim=0)
        
        with torch.no_grad():
            logits = self.labeller(seq_representation)
            probs = torch.sigmoid(logits).cpu().numpy()
        
        binary_preds = (probs > threshold).astype(np.float32)
        return binary_preds, probs

    def reg_train(self, grids, t_list, max_iters=1000, tol=1e-8, learning_rate=0.01, 
               continued=False, decay_rate=1.0, print_every=10, batch_size=32,
               checkpoint_file=None, checkpoint_interval=10):
        """
        Train model for regression using gradient descent with batch processing.
        
        Args:
            grids: List of 2D character grids
            t_list: List of target vectors
            ... (other parameters same as original)
        """
        # Load checkpoint if continuing
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
        
        t_tensors = [torch.tensor(t, dtype=torch.float32, device=self.device) for t in t_list]
        
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)
        
        if continued and checkpoint_file and os.path.exists(checkpoint_file):
            optimizer.load_state_dict(optimizer_state)
            scheduler.load_state_dict(scheduler_state)
        
        prev_loss = float('inf') if start_iter == 0 else history[-1] if history else float('inf')
        
        for it in range(start_iter, max_iters):
            total_loss = 0.0
            total_grids = 0
            
            indices = list(range(len(grids)))
            random.shuffle(indices)
            
            for batch_start in range(0, len(indices), batch_size):
                batch_indices = indices[batch_start:batch_start + batch_size]
                batch_grids = [grids[idx] for idx in batch_indices]
                batch_targets = [t_tensors[idx] for idx in batch_indices]
                
                optimizer.zero_grad()
                batch_loss = 0.0
                batch_grid_count = 0
                
                for grid, target in zip(batch_grids, batch_targets):
                    tokens, positions = self.extract_tokens(grid)
                    if not tokens:
                        continue
                        
                    token_indices = self.token_to_indices(tokens)
                    pos_tensor = torch.tensor(positions, device=self.device, dtype=torch.long)
                    j_indices = (pos_tensor[:, 0] % self.Lh) * self.Lw + (pos_tensor[:, 1] % self.Lw)
                    
                    x = self.embedding(token_indices)
                    B_rows = self.Bbasis[j_indices]
                    scalar = torch.sum(B_rows * x, dim=1)
                    A_cols = self.Acoeff[:, j_indices].t()
                    Nk_batch = A_cols * scalar.unsqueeze(1)
                    
                    seq_pred = torch.mean(Nk_batch, dim=0)
                    seq_loss = torch.sum((seq_pred - target) ** 2)
                    batch_loss += seq_loss
                    batch_grid_count += 1
                    
                    del Nk_batch, seq_pred, token_indices, pos_tensor, x, B_rows, A_cols, scalar
                
                if batch_grid_count > 0:
                    batch_loss = batch_loss / batch_grid_count
                    batch_loss.backward()
                    optimizer.step()
                    
                    total_loss += batch_loss.item() * batch_grid_count
                    total_grids += batch_grid_count
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            avg_loss = total_loss / total_grids if total_grids else 0.0
            history.append(avg_loss)
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_model_state = copy.deepcopy(self.state_dict())
            
            if it % print_every == 0 or it == max_iters - 1:
                current_lr = scheduler.get_last_lr()[0]
                print(f"GD Iter {it:3d}: Loss = {avg_loss:.6e}, LR = {current_lr:.6f}")
            
            if checkpoint_file and (it % checkpoint_interval == 0 or it == max_iters - 1):
                self._save_checkpoint(checkpoint_file, it, history, optimizer, scheduler, best_loss)
            
            if abs(prev_loss - avg_loss) < tol:
                print(f"Converged after {it+1} iterations.")
                if best_model_state is not None and avg_loss > prev_loss:
                    self.load_state_dict(best_model_state)
                    print(f"Restored best model state with loss = {best_loss:.6e}")
                    history[-1] = best_loss
                break
            prev_loss = avg_loss
            scheduler.step()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        if best_model_state is not None and best_loss < history[-1]:
            self.load_state_dict(best_model_state)
            print(f"Training ended. Restored best model state with loss = {best_loss:.6e}")
            history[-1] = best_loss
        
        self._compute_training_statistics(grids)
        self.trained = True
        return history

    def cls_train(self, grids, labels, num_classes, max_iters=1000, tol=1e-8, learning_rate=0.01,
                  continued=False, decay_rate=1.0, print_every=10, batch_size=32,
                  checkpoint_file=None, checkpoint_interval=10):
        """
        Train the model for multi-class classification using cross-entropy loss.
        
        Args:
            grids: List of 2D character grids
            labels: List of integer class labels (0 to num_classes-1)
            num_classes: Number of classes
            ... (other parameters same as original)
        """
        if self.classifier is None or self.num_classes != num_classes:
            self.classifier = nn.Linear(self.m, num_classes).to(self.device)
            self.num_classes = num_classes
        
        if not continued:
            self.reset_parameters()
        
        label_tensors = torch.tensor(labels, dtype=torch.long, device=self.device)
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)
        criterion = nn.CrossEntropyLoss()
        
        history = []
        prev_loss = float('inf')
        best_loss = float('inf')
        best_model_state = None
        
        for it in range(max_iters):
            total_loss = 0.0
            total_grids = 0
            correct_predictions = 0
            
            indices = list(range(len(grids)))
            random.shuffle(indices)
            
            for batch_start in range(0, len(indices), batch_size):
                batch_indices = indices[batch_start:batch_start + batch_size]
                batch_grids = [grids[idx] for idx in batch_indices]
                batch_labels = label_tensors[batch_indices]
                
                optimizer.zero_grad()
                batch_logits = []
                
                for grid in batch_grids:
                    tokens, positions = self.extract_tokens(grid)
                    if not tokens:
                        seq_vector = torch.zeros(self.m, device=self.device)
                    else:
                        token_indices = self.token_to_indices(tokens)
                        pos_tensor = torch.tensor(positions, device=self.device, dtype=torch.long)
                        j_indices = (pos_tensor[:, 0] % self.Lh) * self.Lw + (pos_tensor[:, 1] % self.Lw)
                        
                        x = self.embedding(token_indices)
                        B_rows = self.Bbasis[j_indices]
                        scalar = torch.sum(B_rows * x, dim=1)
                        A_cols = self.Acoeff[:, j_indices].t()
                        Nk_batch = A_cols * scalar.unsqueeze(1)
                        
                        seq_vector = torch.mean(Nk_batch, dim=0)
                        del Nk_batch, token_indices, pos_tensor, x, B_rows, A_cols, scalar
                    
                    logits = self.classifier(seq_vector.unsqueeze(0))
                    batch_logits.append(logits)
                
                if batch_logits:
                    all_logits = torch.cat(batch_logits, dim=0)
                    loss = criterion(all_logits, batch_labels)
                    loss.backward()
                    optimizer.step()
                    
                    batch_loss = loss.item()
                    total_loss += batch_loss * len(batch_grids)
                    total_grids += len(batch_grids)
                    
                    with torch.no_grad():
                        predictions = torch.argmax(all_logits, dim=1)
                        correct_predictions += (predictions == batch_labels).sum().item()
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            avg_loss = total_loss / total_grids if total_grids else 0.0
            accuracy = correct_predictions / total_grids if total_grids else 0.0
            history.append(avg_loss)
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_model_state = copy.deepcopy(self.state_dict())
            
            if it % print_every == 0 or it == max_iters - 1:
                current_lr = scheduler.get_last_lr()[0]
                print(f"CLS-Train Iter {it:3d}: Loss = {avg_loss:.6e}, Acc = {accuracy:.4f}, LR = {current_lr:.6f}")
            
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
            
            if abs(prev_loss - avg_loss) < tol:
                print(f"Converged after {it+1} iterations.")
                if best_model_state is not None:
                    self.load_state_dict(best_model_state)
                break
            prev_loss = avg_loss
            scheduler.step()
        
        self.trained = True
        return history

    def lbl_train(self, grids, labels, num_labels, max_iters=1000, tol=1e-8, learning_rate=0.01, 
                 continued=False, decay_rate=1.0, print_every=10, batch_size=32,
                 checkpoint_file=None, checkpoint_interval=10, pos_weight=None):
        """
        Train the model for multi-label classification using binary cross-entropy loss.
        
        Args:
            grids: List of 2D character grids
            labels: List of binary label vectors
            num_labels: Number of labels
            ... (other parameters same as original)
        """
        if self.labeller is None or self.num_labels != num_labels:
            self.labeller = nn.Linear(self.m, num_labels).to(self.device)
            self.num_labels = num_labels
        
        if not continued:
            self.reset_parameters()
        
        if isinstance(labels, list):
            labels_tensor = torch.tensor(labels, dtype=torch.float32, device=self.device)
        else:
            labels_tensor = torch.as_tensor(labels, dtype=torch.float32, device=self.device)
        
        if pos_weight is not None:
            pos_weight_tensor = torch.tensor(pos_weight, dtype=torch.float32, device=self.device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        else:
            criterion = nn.BCEWithLogitsLoss()
        
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)
        
        loss_history = []
        acc_history = []
        prev_loss = float('inf')
        best_loss = float('inf')
        best_model_state = None
        
        for it in range(max_iters):
            total_loss = 0.0
            total_correct = 0
            total_predictions = 0
            total_grids = 0
            
            indices = list(range(len(grids)))
            random.shuffle(indices)
            
            for batch_start in range(0, len(indices), batch_size):
                batch_indices = indices[batch_start:batch_start + batch_size]
                batch_grids = [grids[idx] for idx in batch_indices]
                batch_labels = labels_tensor[batch_indices]
                
                optimizer.zero_grad()
                batch_logits_list = []
                
                for grid in batch_grids:
                    tokens, positions = self.extract_tokens(grid)
                    if not tokens:
                        continue
                    
                    token_indices = self.token_to_indices(tokens)
                    pos_tensor = torch.tensor(positions, device=self.device, dtype=torch.long)
                    j_indices = (pos_tensor[:, 0] % self.Lh) * self.Lw + (pos_tensor[:, 1] % self.Lw)
                    
                    x = self.embedding(token_indices)
                    B_rows = self.Bbasis[j_indices]
                    scalar = torch.sum(B_rows * x, dim=1)
                    A_cols = self.Acoeff[:, j_indices].t()
                    Nk_batch = A_cols * scalar.unsqueeze(1)
                    
                    seq_rep = torch.mean(Nk_batch, dim=0)
                    logits = self.labeller(seq_rep)
                    batch_logits_list.append(logits)
                    
                    del Nk_batch, seq_rep, token_indices, pos_tensor, x, B_rows, A_cols, scalar
                
                if batch_logits_list:
                    batch_logits = torch.stack(batch_logits_list, dim=0)
                    loss = criterion(batch_logits, batch_labels)
                    loss.backward()
                    optimizer.step()
                    
                    batch_loss = loss.item()
                    total_loss += batch_loss * len(batch_grids)
                    total_grids += len(batch_grids)
                    
                    with torch.no_grad():
                        probs = torch.sigmoid(batch_logits)
                        predictions = (probs > 0.5).float()
                        batch_correct = (predictions == batch_labels).sum().item()
                        batch_predictions = batch_labels.numel()
                        total_correct += batch_correct
                        total_predictions += batch_predictions
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            avg_loss = total_loss / total_grids if total_grids else 0.0
            avg_acc = total_correct / total_predictions if total_predictions else 0.0
            loss_history.append(avg_loss)
            acc_history.append(avg_acc)
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_model_state = copy.deepcopy(self.state_dict())
            
            if it % print_every == 0 or it == max_iters - 1:
                current_lr = scheduler.get_last_lr()[0]
                print(f"MLC-Train Iter {it:3d}: Loss = {avg_loss:.6e}, Acc = {avg_acc:.4f}, LR = {current_lr:.6f}")
            
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
            
            if abs(prev_loss - avg_loss) < tol:
                print(f"Converged after {it+1} iterations.")
                if best_model_state is not None:
                    self.load_state_dict(best_model_state)
                break
            prev_loss = avg_loss
            scheduler.step()
        
        self.trained = True
        return loss_history, acc_history

    def self_train(self, grids, max_iters=1000, tol=1e-8, learning_rate=0.01, 
               continued=False, decay_rate=1.0, print_every=10, batch_size=1024, 
               checkpoint_file=None, checkpoint_interval=10):
        """
        Self-supervised training: predict each window from its own embedding (gap mode).
        
        Args:
            grids: List of 2D character grids
            ... (other parameters same as original)
        """
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
        
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)
        
        if continued and checkpoint_file and os.path.exists(checkpoint_file):
            optimizer.load_state_dict(optimizer_state)
            scheduler.load_state_dict(scheduler_state)
        
        prev_loss = float('inf') if start_iter == 0 else history[-1] if history else float('inf')
        
        for it in range(start_iter, max_iters):
            total_loss = 0.0
            total_samples = 0
            
            # Shuffle grids
            indices = list(range(len(grids)))
            random.shuffle(indices)
            
            for grid_idx in indices:
                grid = grids[grid_idx]
                tokens, positions = self.extract_tokens(grid)
                if not tokens:
                    continue
                
                token_indices = self.token_to_indices(tokens)
                # Build list of (k, token_idx) for each window, but k is now a pair (i,j)
                # We'll store flattened position index for j calculation
                samples = []
                for idx, (pos, token_idx) in enumerate(zip(positions, token_indices)):
                    i, j = pos
                    j_idx = (i % self.Lh) * self.Lw + (j % self.Lw)
                    samples.append((j_idx, token_idx.item()))
                
                if not samples:
                    continue
                
                for batch_start in range(0, len(samples), batch_size):
                    batch_samples = samples[batch_start:batch_start + batch_size]
                    
                    optimizer.zero_grad()
                    
                    j_idx_list = [s[0] for s in batch_samples]
                    token_idx_list = [s[1] for s in batch_samples]
                    
                    j_tensor = torch.tensor(j_idx_list, device=self.device, dtype=torch.long)
                    token_tensor = torch.tensor(token_idx_list, device=self.device, dtype=torch.long)
                    
                    x = self.embedding(token_tensor)
                    B_rows = self.Bbasis[j_tensor]
                    scalar = torch.sum(B_rows * x, dim=1)
                    A_cols = self.Acoeff[:, j_tensor].t()
                    Nk_batch = A_cols * scalar.unsqueeze(1)
                    
                    targets = self.embedding(token_tensor)
                    loss = torch.mean(torch.sum((Nk_batch - targets) ** 2, dim=1))
                    loss.backward()
                    optimizer.step()
                    
                    batch_loss = loss.item() * len(batch_samples)
                    total_loss += batch_loss
                    total_samples += len(batch_samples)
                    
                    del j_tensor, token_tensor, Nk_batch, targets, loss, x, B_rows, A_cols, scalar
                    
                    if total_samples % 1000 == 0 and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                del token_indices, samples
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            avg_loss = total_loss / total_samples if total_samples else 0.0
            history.append(avg_loss)
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_model_state = copy.deepcopy(self.state_dict())
            
            if it % print_every == 0 or it == max_iters - 1:
                current_lr = scheduler.get_last_lr()[0]
                print(f"SelfTrain Iter {it:3d}: loss = {avg_loss:.6f}, LR = {current_lr:.6f}, Samples = {total_samples}")
            
            if checkpoint_file and (it % checkpoint_interval == 0 or it == max_iters - 1):
                self._save_checkpoint(checkpoint_file, it, history, optimizer, scheduler, best_loss)
            
            if abs(prev_loss - avg_loss) < tol:
                print(f"Converged after {it+1} iterations")
                if best_model_state is not None and avg_loss > prev_loss:
                    self.load_state_dict(best_model_state)
                    print(f"Restored best model state with loss = {best_loss:.6f}")
                    history[-1] = best_loss
                break
            prev_loss = avg_loss
            scheduler.step()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        if best_model_state is not None and best_loss < history[-1]:
            self.load_state_dict(best_model_state)
            print(f"Training ended. Restored best model state with loss = {best_loss:.6f}")
            history[-1] = best_loss
        
        self._compute_training_statistics(grids)
        self.trained = True
        return history

    def _save_checkpoint(self, checkpoint_file, iteration, history, optimizer, scheduler, best_loss):
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
                'bas_dim_h': self.Lh,
                'bas_dim_w': self.Lw,
                'rank': self.rank,
                'rank_mode': self.rank_mode,
                'mode': self.mode,
                'user_step': self.step
            },
            'training_stats': {
                'mean_t': self.mean_t if hasattr(self, 'mean_t') else None,
                'mean_window_count': self.mean_window_count if hasattr(self, 'mean_window_count') else None
            }
        }
        torch.save(checkpoint, checkpoint_file)
        print(f"Checkpoint saved at iteration {iteration} to {checkpoint_file}")

    def reconstruct_2d(self, H, W, tau=0.0):
        """
        Reconstruct a 2D grid of size H x W using temperature-controlled randomness.
        
        The method generates windows in row-major order, each window predicts a token
        (flattened pattern), and then the grid is assembled by voting: each cell
        receives votes from all windows covering it, and the most frequent character
        is chosen.
        
        Args:
            H: height of output grid
            W: width of output grid
            tau: temperature (0 = deterministic, >0 = stochastic)
            
        Returns:
            List of lists of chars: reconstructed 2D grid
        """
        assert self.trained, "Model must be trained first"
        if tau < 0:
            raise ValueError("Temperature must be non-negative")
        
        # Step size for sliding window during reconstruction (same as training step)
        step = self.step if self.mode == 'nonlinear' else 1
        if step is None:
            step = self.rank
        
        # Determine all window positions that would be used in training
        # (i.e., sliding over the entire grid)
        positions = []
        for i in range(0, H - self.rank + 1, step):
            for j in range(0, W - self.rank + 1, step):
                positions.append((i, j))
        # For pad mode, we would also need incomplete windows, but for simplicity we only handle full windows.
        # If pad mode is used, we need to also consider boundary windows with padding.
        # Here we assume pad mode is not used for reconstruction (or use drop mode).
        
        mean_t_tensor = torch.tensor(self.mean_t, dtype=torch.float32, device=self.device)
        
        # Precompute all token embeddings
        all_token_indices = torch.arange(self.vocab_size, device=self.device)
        all_embeddings = self.embedding(all_token_indices)  # [vocab_size, m]
        
        # For each window position, compute the best token
        tokens_for_positions = []
        for pos in positions:
            i, j = pos
            # Compute j index
            j_idx = (i % self.Lh) * self.Lw + (j % self.Lw)
            B_row = self.Bbasis[j_idx].unsqueeze(0)  # [1, m]
            
            # scalar = B_row · all_embeddings
            scalar = torch.sum(B_row * all_embeddings, dim=1)  # [vocab_size]
            
            A_col = self.Acoeff[:, j_idx]  # [m]
            Nk_all = A_col * scalar.unsqueeze(1)  # [vocab_size, m]
            
            errors = torch.sum((Nk_all - mean_t_tensor) ** 2, dim=1)
            scores = -errors
            
            if tau == 0:
                best_idx = torch.argmax(scores).item()
                token = self.idx_to_token[best_idx]
            else:
                probs = torch.softmax(scores / tau, dim=0).detach().cpu().numpy()
                chosen_idx = np.random.choice(self.vocab_size, p=probs)
                token = self.idx_to_token[chosen_idx]
            tokens_for_positions.append(token)
        
        # Now assemble grid by voting
        # Initialize vote count for each cell (H x W x len(charset))
        char_to_idx = {c: idx for idx, c in enumerate(self.charset)}
        vote_counts = np.zeros((H, W, len(self.charset)), dtype=int)
        
        for token, (i, j) in zip(tokens_for_positions, positions):
            # Token is a flattened rank*rank string
            # Convert to 2D pattern
            pattern = [token[k*self.rank:(k+1)*self.rank] for k in range(self.rank)]
            for di in range(self.rank):
                for dj in range(self.rank):
                    if i+di < H and j+dj < W:
                        ch = pattern[di][dj]
                        if ch in char_to_idx:
                            vote_counts[i+di, j+dj, char_to_idx[ch]] += 1
        
        # Choose the character with maximum votes for each cell
        grid = []
        for i in range(H):
            row = []
            for j in range(W):
                char_idx = np.argmax(vote_counts[i, j])
                row.append(self.charset[char_idx])
            grid.append(row)
        return grid

    def save(self, filename):
        """Save model state to file"""
        save_dict = {
            'state_dict': self.state_dict(),
            'mean_t': self.mean_t if hasattr(self, 'mean_t') else None,
            'mean_window_count': self.mean_window_count if hasattr(self, 'mean_window_count') else None,
            'trained': self.trained,
            'num_classes': self.num_classes,
            'num_labels': self.num_labels
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
        self.mean_window_count = save_dict.get('mean_window_count', None)
        self.trained = save_dict.get('trained', False)
        self.num_classes = save_dict.get('num_classes', None)
        self.num_labels = save_dict.get('num_labels', None)
        
        if self.num_classes is not None:
            self.classifier = nn.Linear(self.m, self.num_classes).to(self.device)
        if self.num_labels is not None:
            self.labeller = nn.Linear(self.m, self.num_labels).to(self.device)
            
        print(f"Model loaded from {filename}")
        return self


# === Example Usage ===
if __name__ == "__main__":
    from statistics import correlation
    
    print("="*50)
    print("Spatial Dual Descriptor AB2 - 2D Grid Version")
    print("Optimized with batch processing")
    print("="*50)
    
    # Set random seeds
    torch.manual_seed(11)
    random.seed(11)
    
    charset = ['A','C','G','T']
    vec_dim = 3
    bas_dim_h = 50
    bas_dim_w = 50
    rank = 2          # 2x2 windows
    grid_num = 50     # number of training grids
    
    # Generate random 2D grids (10x10 each) and random target vectors
    grids = []
    t_list = []
    for _ in range(grid_num):
        H = random.randint(8, 12)
        W = random.randint(8, 12)
        grid = [[random.choice(charset) for _ in range(W)] for _ in range(H)]
        grids.append(grid)
        t_list.append([random.uniform(-1,1) for _ in range(vec_dim)])
    
    # Create model
    dd2 = SpatialDualDescriptorAB2(
        charset, 
        vec_dim=vec_dim, 
        bas_dim_h=bas_dim_h,
        bas_dim_w=bas_dim_w,
        rank=rank,
        mode='linear',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    print(f"\nUsing device: {dd2.device}")
    print(f"Number of tokens: {len(dd2.tokens)}")
    
    # === Regression Training ===
    print("\n" + "="*50)
    print("Testing Gradient Descent Training (Regression)")
    print("="*50)
    
    reg_history = dd2.reg_train(
        grids, t_list,
        learning_rate=0.1,
        max_iters=100,
        tol=1e-66,
        print_every=10,
        decay_rate=0.99,
        batch_size=16
    )
    
    # Predict target for first grid
    pred_t = dd2.predict_t(grids[0])
    print(f"\nPredicted t for first grid: {[round(x.item(), 4) for x in pred_t]}")
    
    # Calculate prediction correlation
    pred_t_list = [dd2.predict_t(grid) for grid in grids]
    corr_sum = 0.0
    for i in range(dd2.m):
        actu_t = [t_vec[i] for t_vec in t_list]
        pred_t_vals = [t_vec[i] for t_vec in pred_t_list]
        corr = correlation(actu_t, pred_t_vals)
        print(f"Prediction correlation: {corr:.4f}")
        corr_sum += corr
    corr_avg = corr_sum / dd2.m
    print(f"Average correlation: {corr_avg:.4f}")   
    
    # === Classification Task ===
    print("\n" + "="*50)
    print("Classification Task")
    print("="*50)
    
    num_classes = 3
    class_grids = []
    class_labels = []
    
    for class_id in range(num_classes):
        for _ in range(40):
            H = random.randint(8, 12)
            W = random.randint(8, 12)
            if class_id == 0:
                # Class 0: High A content
                grid = [[random.choices(['A','C','G','T'], weights=[0.6,0.1,0.1,0.2])[0] for _ in range(W)] for _ in range(H)]
            elif class_id == 1:
                # Class 1: High GC content
                grid = [[random.choices(['A','C','G','T'], weights=[0.1,0.4,0.4,0.1])[0] for _ in range(W)] for _ in range(H)]
            else:
                # Class 2: Balanced
                grid = [[random.choice(charset) for _ in range(W)] for _ in range(H)]
            class_grids.append(grid)
            class_labels.append(class_id)
    
    dd_cls = SpatialDualDescriptorAB2(
        charset, vec_dim=vec_dim, bas_dim_h=bas_dim_h, bas_dim_w=bas_dim_w, rank=rank, mode='linear',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    print("\nStarting Classification Training")
    history = dd_cls.cls_train(class_grids, class_labels, num_classes, 
                              max_iters=50, tol=1e-8, learning_rate=0.05,
                              decay_rate=0.99, batch_size=16, print_every=10)
    
    correct = 0
    for grid, true_label in zip(class_grids, class_labels):
        pred_class, probs = dd_cls.predict_c(grid)
        if pred_class == true_label:
            correct += 1
    accuracy = correct / len(class_grids)
    print(f"Classification Accuracy: {accuracy:.4f} ({correct}/{len(class_grids)})")
    
    # === Multi-Label Classification Task ===
    print("\n" + "="*50)
    print("Multi-Label Classification Task")
    print("="*50)
    
    num_labels = 4
    label_grids = []
    labels = []
    for _ in range(100):
        H = random.randint(8, 12)
        W = random.randint(8, 12)
        grid = [[random.choice(charset) for _ in range(W)] for _ in range(H)]
        label_grids.append(grid)
        label_vec = [1.0 if random.random() > 0.7 else 0.0 for _ in range(num_labels)]
        labels.append(label_vec)
    
    dd_lbl = SpatialDualDescriptorAB2(
        charset, vec_dim=vec_dim, bas_dim_h=bas_dim_h, bas_dim_w=bas_dim_w, rank=rank, mode='linear',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    loss_history, acc_history = dd_lbl.lbl_train(
        label_grids, labels, num_labels,
        max_iters=50, tol=1e-16, learning_rate=0.05, decay_rate=0.99, print_every=10, batch_size=16
    )
    print(f"Multi-label final training loss: {loss_history[-1]:.6f}, accuracy: {acc_history[-1]:.4f}")
    
    # === Self-Training Example ===
    print("\n" + "="*50)
    print("Self-Training (Gap Mode)")
    print("="*50)
    
    self_grids = []
    for _ in range(30):
        H = random.randint(8, 12)
        W = random.randint(8, 12)
        grid = [[random.choice(charset) for _ in range(W)] for _ in range(H)]
        self_grids.append(grid)
    
    dd_self = SpatialDualDescriptorAB2(
        charset, vec_dim=vec_dim, bas_dim_h=bas_dim_h, bas_dim_w=bas_dim_w, rank=rank, mode='linear',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    self_history = dd_self.self_train(
        self_grids,
        max_iters=50,
        learning_rate=0.001,
        decay_rate=0.995,
        print_every=10,
        batch_size=256
    )
    
    # Reconstruct a 2D grid from self-trained model
    print("\nReconstructing a 10x10 grid from self-trained model (deterministic):")
    recon_grid = dd_self.reconstruct_2d(H=10, W=10, tau=0.0)
    print("First 5 rows:")
    for row in recon_grid[:5]:
        print(''.join(row))
    
    # Save and load model
    print("\n=== Model Persistence Test ===")
    dd_self.save("self_trained_2d.pkl")
    dd_loaded = SpatialDualDescriptorAB2(
        charset, vec_dim=vec_dim, bas_dim_h=bas_dim_h, bas_dim_w=bas_dim_w, rank=rank, mode='linear',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    ).load("self_trained_2d.pkl")
    print("Model loaded successfully. Reconstructing with loaded model:")
    recon_loaded = dd_loaded.reconstruct_2d(H=8, W=8, tau=0.0)
    for row in recon_loaded:
        print(''.join(row))
    
    print("\n=== All Tests Completed ===")
