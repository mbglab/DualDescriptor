# Copyright (C) 2005-2025, Bin-Guang Ma (mbg@mail.hzau.edu.cn); SPDX-License-Identifier: MIT
# The Spatial Dual Descriptor Vector class (P Matrix form) for 4D character arrays implemented with PyTorch
# This program is for the demonstration of methodology and not fully refined.
# Author: Bin-Guang Ma (assisted by DeepSeek); Date: 2025-7-29 ~ 2026-3-25

import math
import random
import itertools
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
import os
from typing import List, Tuple, Union, Optional

class SpatialDualDescriptorPM4(nn.Module):
    """
    Spatial Dual Descriptor for 4D character arrays (hypercubes) with GPU acceleration:
      - matrix P ∈ R^{m×m} of basis coefficients
      - embedding: hypercube token embeddings in R^m
      - indexed periods: period1..4[i,j] for four spatial dimensions
      - basis function phi_{i,j}(k1,k2,k3,k4) = ∏_{d=1}^4 cos(2π*kd/period_d)
      - supports 'linear' (step=1) or 'nonlinear' (step-by-window-size) hypercube extraction
    """
    def __init__(self, charset: List[str], window_size: int = 3, rank_mode: str = 'drop',
                 vec_dim: int = 2, mode: str = 'linear', user_step: Optional[int] = None,
                 device: str = 'cuda'):
        """
        Args:
            charset: List of characters that appear in the input hypercube.
            window_size: Size of the hypercubic sliding window (e.g., 3 for 3x3x3x3 hypercubes).
            rank_mode: 'pad' or 'drop' for handling incomplete windows at boundaries.
            vec_dim: Dimension of the embedding space (m).
            mode: 'linear' (step=1) or 'nonlinear' (step = user_step or window_size).
            user_step: Step size for nonlinear mode (same for all dimensions).
            device: 'cuda' or 'cpu'.
        """
        super().__init__()
        self.charset = list(charset)
        self.window_size = window_size          # hypercubic window side length
        self.rank_mode = rank_mode
        self.m = vec_dim
        assert mode in ('linear', 'nonlinear')
        self.mode = mode
        self.step = user_step if user_step is not None else window_size   # step for nonlinear mode
        self.trained = False
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Generate all possible hypercube tokens (window_size^4 characters)
        # Each token is a concatenation of all characters in row-major order across all dimensions.
        # The order is: depth1, depth2, ..., depth4 (with nested loops in order: dim1, dim2, dim3, dim4)
        # Total length = window_size^4
        chars_with_pad = self.charset + (['_'] if self.rank_mode == 'pad' else [])
        token_length = window_size ** 4
        toks = []
        for combo in itertools.product(chars_with_pad, repeat=token_length):
            tok = ''.join(combo)
            toks.append(tok)
        self.tokens = sorted(set(toks))
        self.token_to_idx = {token: idx for idx, token in enumerate(self.tokens)}
        self.idx_to_token = {idx: token for idx, token in enumerate(self.tokens)}
        
        # Embedding layer for tokens
        self.embedding = nn.Embedding(len(self.tokens), self.m)
        
        # Position-weight matrix P[i][j]
        self.P = nn.Parameter(torch.empty(self.m, self.m))
        
        # I matrix for vector sequence processing (unused but kept)
        self.I = nn.Parameter(torch.empty(self.m, self.m))
        
        # Precompute period matrices for four dimensions (fixed, not trainable)
        periods1 = torch.zeros(self.m, self.m, dtype=torch.float32)
        periods2 = torch.zeros(self.m, self.m, dtype=torch.float32)
        periods3 = torch.zeros(self.m, self.m, dtype=torch.float32)
        periods4 = torch.zeros(self.m, self.m, dtype=torch.float32)
        for i in range(self.m):
            for j in range(self.m):
                base = i * self.m + j + 2
                periods1[i, j] = base
                periods2[i, j] = base
                periods3[i, j] = base
                periods4[i, j] = base   # can be made different if needed
        self.register_buffer('periods1', periods1)
        self.register_buffer('periods2', periods2)
        self.register_buffer('periods3', periods3)
        self.register_buffer('periods4', periods4)

        # Class head (initialized later when num_classes is known)
        self.num_classes = None
        self.classifier = None

        # Label head (initialized later when num_labels is known)
        self.num_labels = None
        self.labeller = None

        # Initialize parameters
        self.reset_parameters()
        self.to(self.device)
        
    def reset_parameters(self):
        """Initialize model parameters"""
        nn.init.uniform_(self.embedding.weight, -0.5, 0.5)
        nn.init.uniform_(self.P, -0.1, 0.1)
        nn.init.uniform_(self.I, -0.1, 0.1)
        if self.classifier is not None:
            nn.init.normal_(self.classifier.weight, 0, 0.01)
            if self.classifier.bias is not None:
                nn.init.constant_(self.classifier.bias, 0)
        if self.labeller is not None:
            nn.init.xavier_uniform_(self.labeller.weight)
            nn.init.zeros_(self.labeller.bias)
        
    def token_to_indices(self, token_list: List[str]) -> torch.Tensor:
        """Convert list of tokens to tensor of indices"""
        return torch.tensor([self.token_to_idx[tok] for tok in token_list], device=self.device)
    
    def extract_hypercubes(self, hypercube: List) -> Tuple[List[str], List[Tuple[int, int, int, int]]]:
        """
        Extract hypercubic window tokens and their top-front-left-time coordinates from a 4D character array.
        
        The hypercube is represented as a nested list with 4 dimensions: d1 x d2 x d3 x d4.
        For simplicity, we assume input is a list of depth1, each depth1 is a list of depth2,
        each depth2 is a list of rows (depth3), and each row is a string of characters (depth4).
        So the structure is: hypercube[d1][d2][d3] is a string of length = dimension4.
        This matches the pattern used in lower dimensions.
        
        Args:
            hypercube: 4D list of characters. Access: hypercube[i][j][k][l] is a character,
                       but we accept a list-of-strings representation for the last dimension.
                       Actually, to avoid extreme nesting, we represent as:
                         hypercube[dim1][dim2][dim3] = string of length dim4
                       This is consistent with 2D (list of strings) and 3D (list of list of strings).
        
        Returns:
            tokens: List of hypercube tokens (strings of length window_size^4).
            coords: List of (d1, d2, d3, d4) coordinates (starting indices) of each hypercube.
        """
        # Determine dimensions
        D1 = len(hypercube)
        D2 = len(hypercube[0]) if D1 > 0 else 0
        D3 = len(hypercube[0][0]) if D2 > 0 else 0
        D4 = len(hypercube[0][0][0]) if D3 > 0 else 0
        step = 1 if self.mode == 'linear' else self.step
        
        tokens = []
        coords = []
        
        if self.rank_mode == 'drop':
            # Only extract fully contained hypercubes
            for d1 in range(0, D1 - self.window_size + 1, step):
                for d2 in range(0, D2 - self.window_size + 1, step):
                    for d3 in range(0, D3 - self.window_size + 1, step):
                        for d4 in range(0, D4 - self.window_size + 1, step):
                            # Build token: concatenate characters in order: d1, d2, d3, d4 (nested loops)
                            token_chars = []
                            for i in range(self.window_size):
                                for j in range(self.window_size):
                                    for k in range(self.window_size):
                                        for l in range(self.window_size):
                                            token_chars.append(hypercube[d1+i][d2+j][d3+k][d4+l])
                            token = ''.join(token_chars)
                            tokens.append(token)
                            coords.append((d1, d2, d3, d4))
        else:  # pad mode
            for d1 in range(0, D1, step):
                for d2 in range(0, D2, step):
                    for d3 in range(0, D3, step):
                        for d4 in range(0, D4, step):
                            # Build token with padding
                            token_chars = []
                            for i in range(self.window_size):
                                for j in range(self.window_size):
                                    for k in range(self.window_size):
                                        for l in range(self.window_size):
                                            d1_idx = d1 + i
                                            d2_idx = d2 + j
                                            d3_idx = d3 + k
                                            d4_idx = d4 + l
                                            if (d1_idx < D1 and d2_idx < D2 and 
                                                d3_idx < D3 and d4_idx < D4):
                                                token_chars.append(hypercube[d1_idx][d2_idx][d3_idx][d4_idx])
                                            else:
                                                token_chars.append('_')
                            token = ''.join(token_chars)
                            tokens.append(token)
                            coords.append((d1, d2, d3, d4))
        return tokens, coords

    def batch_compute_Nk(self, k1_tensor: torch.Tensor, k2_tensor: torch.Tensor,
                         k3_tensor: torch.Tensor, k4_tensor: torch.Tensor,
                         token_indices: torch.Tensor) -> torch.Tensor:
        """
        Vectorized computation of N(k1,k2,k3,k4) vectors for a batch of positions and tokens.
        
        Args:
            k1_tensor: Tensor of dimension1 indices [batch_size]
            k2_tensor: Tensor of dimension2 indices [batch_size]
            k3_tensor: Tensor of dimension3 indices [batch_size]
            k4_tensor: Tensor of dimension4 indices [batch_size]
            token_indices: Tensor of token indices [batch_size]
            
        Returns:
            Tensor of N(k) vectors [batch_size, m]
        """
        # Get embeddings for all tokens [batch_size, m]
        x = self.embedding(token_indices)
        
        # Expand dimensions for broadcasting
        k1_exp = k1_tensor.view(-1, 1, 1)
        k2_exp = k2_tensor.view(-1, 1, 1)
        k3_exp = k3_tensor.view(-1, 1, 1)
        k4_exp = k4_tensor.view(-1, 1, 1)
        
        # Compute basis function: product of cosines for four dimensions
        phi = (torch.cos(2 * math.pi * k1_exp / self.periods1) *
               torch.cos(2 * math.pi * k2_exp / self.periods2) *
               torch.cos(2 * math.pi * k3_exp / self.periods3) *
               torch.cos(2 * math.pi * k4_exp / self.periods4))  # [batch_size, m, m]
        
        # Optimized computation using einsum
        Nk = torch.einsum('bj,ij,bij->bi', x, self.P, phi)
        return Nk

    def compute_Nk(self, k1: int, k2: int, k3: int, k4: int, token_idx: int) -> torch.Tensor:
        """Compute N(k1,k2,k3,k4) for a single position and token (uses batch internally)."""
        k1_tensor = torch.tensor([k1], dtype=torch.float32, device=self.device)
        k2_tensor = torch.tensor([k2], dtype=torch.float32, device=self.device)
        k3_tensor = torch.tensor([k3], dtype=torch.float32, device=self.device)
        k4_tensor = torch.tensor([k4], dtype=torch.float32, device=self.device)
        idx_tensor = torch.tensor([token_idx], device=self.device)
        result = self.batch_compute_Nk(k1_tensor, k2_tensor, k3_tensor, k4_tensor, idx_tensor)
        return result[0]

    def describe(self, hypercube: List) -> np.ndarray:
        """
        Compute N(k1,k2,k3,k4) vectors for each hypercube in the 4D array.
        
        Args:
            hypercube: 4D list of characters.
            
        Returns:
            Numpy array of shape (num_hypercubes, m).
        """
        tokens, coords = self.extract_hypercubes(hypercube)
        if not tokens:
            return np.array([])
        
        token_indices = self.token_to_indices(tokens)
        k1_list = [c[0] for c in coords]
        k2_list = [c[1] for c in coords]
        k3_list = [c[2] for c in coords]
        k4_list = [c[3] for c in coords]
        k1_tensor = torch.tensor(k1_list, dtype=torch.float32, device=self.device)
        k2_tensor = torch.tensor(k2_list, dtype=torch.float32, device=self.device)
        k3_tensor = torch.tensor(k3_list, dtype=torch.float32, device=self.device)
        k4_tensor = torch.tensor(k4_list, dtype=torch.float32, device=self.device)
        
        N_batch = self.batch_compute_Nk(k1_tensor, k2_tensor, k3_tensor, k4_tensor, token_indices)
        return N_batch.detach().cpu().numpy()

    def S(self, hypercube: List) -> List[np.ndarray]:
        """
        Compute list of S(l)=sum(N(k1,k2,k3,k4)) for l=1..num_hypercubes (cumulative sum in extraction order).
        """
        tokens, coords = self.extract_hypercubes(hypercube)
        if not tokens:
            return []
            
        token_indices = self.token_to_indices(tokens)
        k1_list = [c[0] for c in coords]
        k2_list = [c[1] for c in coords]
        k3_list = [c[2] for c in coords]
        k4_list = [c[3] for c in coords]
        k1_tensor = torch.tensor(k1_list, dtype=torch.float32, device=self.device)
        k2_tensor = torch.tensor(k2_list, dtype=torch.float32, device=self.device)
        k3_tensor = torch.tensor(k3_list, dtype=torch.float32, device=self.device)
        k4_tensor = torch.tensor(k4_list, dtype=torch.float32, device=self.device)
        
        N_batch = self.batch_compute_Nk(k1_tensor, k2_tensor, k3_tensor, k4_tensor, token_indices)
        S_cum = torch.cumsum(N_batch, dim=0)
        return [s.detach().cpu().numpy() for s in S_cum]

    def D(self, hypercubes: List[List], t_list: List[List[float]]) -> float:
        """
        Compute mean squared deviation D across hypercubes:
        D = average over all positions of (N(k)-t_hypercube)^2
        """
        total_loss = 0.0
        total_positions = 0
        
        # Convert target vectors to tensor
        t_tensors = [torch.tensor(t, dtype=torch.float32, device=self.device) for t in t_list]
        
        for hcube, t in zip(hypercubes, t_tensors):
            tokens, coords = self.extract_hypercubes(hcube)
            if not tokens:
                continue
                
            token_indices = self.token_to_indices(tokens)
            k1_list = [c[0] for c in coords]
            k2_list = [c[1] for c in coords]
            k3_list = [c[2] for c in coords]
            k4_list = [c[3] for c in coords]
            k1_tensor = torch.tensor(k1_list, dtype=torch.float32, device=self.device)
            k2_tensor = torch.tensor(k2_list, dtype=torch.float32, device=self.device)
            k3_tensor = torch.tensor(k3_list, dtype=torch.float32, device=self.device)
            k4_tensor = torch.tensor(k4_list, dtype=torch.float32, device=self.device)
            
            N_batch = self.batch_compute_Nk(k1_tensor, k2_tensor, k3_tensor, k4_tensor, token_indices)
            losses = torch.sum((N_batch - t) ** 2, dim=1)
            total_loss += losses.sum().item()
            total_positions += len(tokens)
                
        return total_loss / total_positions if total_positions else 0.0

    def d(self, hypercube: List, t: List[float]) -> float:
        """Compute pattern deviation value for a single hypercube."""
        return self.D([hypercube], [t])

    def reg_train(self, hypercubes: List[List], t_list: List[List[float]],
                  max_iters: int = 1000, tol: float = 1e-8, learning_rate: float = 0.01,
                  continued: bool = False, decay_rate: float = 1.0, print_every: int = 10,
                  batch_size: int = 32, checkpoint_file: Optional[str] = None,
                  checkpoint_interval: int = 10) -> List[float]:
        """Train the model using gradient descent with hypercube-level batch processing."""
        if not continued:
            self.reset_parameters()
        
        t_tensors = [torch.tensor(t, dtype=torch.float32, device=self.device) for t in t_list]
        
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)
        
        history = []
        prev_loss = float('inf')
        best_loss = float('inf')
        best_model_state = None
        
        for it in range(max_iters):
            total_loss = 0.0
            total_hypercubes = 0
            
            indices = list(range(len(hypercubes)))
            random.shuffle(indices)
            
            for batch_start in range(0, len(indices), batch_size):
                batch_indices = indices[batch_start:batch_start + batch_size]
                batch_hcubes = [hypercubes[idx] for idx in batch_indices]
                batch_targets = [t_tensors[idx] for idx in batch_indices]
                
                optimizer.zero_grad()
                batch_loss = 0.0
                
                for hcube, target in zip(batch_hcubes, batch_targets):
                    tokens, coords = self.extract_hypercubes(hcube)
                    if not tokens:
                        continue
                        
                    token_indices = self.token_to_indices(tokens)
                    k1_list = [c[0] for c in coords]
                    k2_list = [c[1] for c in coords]
                    k3_list = [c[2] for c in coords]
                    k4_list = [c[3] for c in coords]
                    k1_tensor = torch.tensor(k1_list, dtype=torch.float32, device=self.device)
                    k2_tensor = torch.tensor(k2_list, dtype=torch.float32, device=self.device)
                    k3_tensor = torch.tensor(k3_list, dtype=torch.float32, device=self.device)
                    k4_tensor = torch.tensor(k4_list, dtype=torch.float32, device=self.device)
                    
                    N_batch = self.batch_compute_Nk(k1_tensor, k2_tensor, k3_tensor, k4_tensor, token_indices)
                    hcube_pred = torch.mean(N_batch, dim=0)
                    loss = torch.sum((hcube_pred - target) ** 2)
                    batch_loss += loss
                    
                    del N_batch, hcube_pred, token_indices, k1_tensor, k2_tensor, k3_tensor, k4_tensor
                
                if len(batch_hcubes) > 0:
                    batch_loss = batch_loss / len(batch_hcubes)
                    batch_loss.backward()
                    optimizer.step()
                    total_loss += batch_loss.item() * len(batch_hcubes)
                    total_hypercubes += len(batch_hcubes)
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            avg_loss = total_loss / total_hypercubes if total_hypercubes > 0 else 0.0
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
                if best_model_state is not None:
                    self.load_state_dict(best_model_state)
                break
                
            prev_loss = avg_loss
            scheduler.step()
        
        self._compute_training_statistics(hypercubes)
        self.trained = True
        return history

    def cls_train(self, hypercubes: List[List], labels: List[int], num_classes: int,
                  max_iters: int = 1000, tol: float = 1e-8, learning_rate: float = 0.01,
                  continued: bool = False, decay_rate: float = 1.0, print_every: int = 10,
                  batch_size: int = 32, checkpoint_file: Optional[str] = None,
                  checkpoint_interval: int = 10) -> List[float]:
        """Train the model for multi-class classification."""
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
            total_hypercubes = 0
            correct = 0
            
            indices = list(range(len(hypercubes)))
            random.shuffle(indices)
            
            for batch_start in range(0, len(indices), batch_size):
                batch_indices = indices[batch_start:batch_start + batch_size]
                batch_hcubes = [hypercubes[idx] for idx in batch_indices]
                batch_labels = label_tensors[batch_indices]
                
                optimizer.zero_grad()
                batch_logits = []
                
                for hcube in batch_hcubes:
                    tokens, coords = self.extract_hypercubes(hcube)
                    if not tokens:
                        vec = torch.zeros(self.m, device=self.device)
                    else:
                        token_indices = self.token_to_indices(tokens)
                        k1_list = [c[0] for c in coords]
                        k2_list = [c[1] for c in coords]
                        k3_list = [c[2] for c in coords]
                        k4_list = [c[3] for c in coords]
                        k1_tensor = torch.tensor(k1_list, dtype=torch.float32, device=self.device)
                        k2_tensor = torch.tensor(k2_list, dtype=torch.float32, device=self.device)
                        k3_tensor = torch.tensor(k3_list, dtype=torch.float32, device=self.device)
                        k4_tensor = torch.tensor(k4_list, dtype=torch.float32, device=self.device)
                        N_batch = self.batch_compute_Nk(k1_tensor, k2_tensor, k3_tensor, k4_tensor, token_indices)
                        vec = torch.mean(N_batch, dim=0)
                        del N_batch, token_indices, k1_tensor, k2_tensor, k3_tensor, k4_tensor
                    logits = self.classifier(vec.unsqueeze(0))
                    batch_logits.append(logits)
                
                if batch_logits:
                    all_logits = torch.cat(batch_logits, dim=0)
                    loss = criterion(all_logits, batch_labels)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item() * len(batch_hcubes)
                    total_hypercubes += len(batch_hcubes)
                    with torch.no_grad():
                        preds = torch.argmax(all_logits, dim=1)
                        correct += (preds == batch_labels).sum().item()
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            avg_loss = total_loss / total_hypercubes if total_hypercubes > 0 else 0.0
            acc = correct / total_hypercubes if total_hypercubes > 0 else 0.0
            history.append(avg_loss)
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_model_state = copy.deepcopy(self.state_dict())
            
            if it % print_every == 0 or it == max_iters - 1:
                current_lr = scheduler.get_last_lr()[0]
                print(f"CLS Iter {it:3d}: Loss = {avg_loss:.6e}, Acc = {acc:.4f}, LR = {current_lr:.6f}")
            
            if checkpoint_file and (it % checkpoint_interval == 0 or it == max_iters - 1):
                torch.save({
                    'iteration': it,
                    'model_state_dict': self.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'history': history,
                    'best_loss': best_loss,
                    'num_classes': self.num_classes
                }, checkpoint_file)
                print(f"Checkpoint saved at iteration {it}")
            
            if abs(prev_loss - avg_loss) < tol:
                print(f"Converged after {it+1} iterations.")
                if best_model_state is not None:
                    self.load_state_dict(best_model_state)
                break
                
            prev_loss = avg_loss
            scheduler.step()
        
        self.trained = True
        return history

    def lbl_train(self, hypercubes: List[List], labels: Union[List[List[float]], np.ndarray],
                  num_labels: int, max_iters: int = 1000, tol: float = 1e-8,
                  learning_rate: float = 0.01, continued: bool = False, decay_rate: float = 1.0,
                  print_every: int = 10, batch_size: int = 32, checkpoint_file: Optional[str] = None,
                  checkpoint_interval: int = 10, pos_weight: Optional[List[float]] = None) -> Tuple[List[float], List[float]]:
        """Train the model for multi-label classification."""
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
            total_hypercubes = 0
            
            indices = list(range(len(hypercubes)))
            random.shuffle(indices)
            
            for batch_start in range(0, len(indices), batch_size):
                batch_indices = indices[batch_start:batch_start + batch_size]
                batch_hcubes = [hypercubes[idx] for idx in batch_indices]
                batch_labels = labels_tensor[batch_indices]
                
                optimizer.zero_grad()
                batch_logits = []
                
                for hcube in batch_hcubes:
                    tokens, coords = self.extract_hypercubes(hcube)
                    if not tokens:
                        continue
                    token_indices = self.token_to_indices(tokens)
                    k1_list = [c[0] for c in coords]
                    k2_list = [c[1] for c in coords]
                    k3_list = [c[2] for c in coords]
                    k4_list = [c[3] for c in coords]
                    k1_tensor = torch.tensor(k1_list, dtype=torch.float32, device=self.device)
                    k2_tensor = torch.tensor(k2_list, dtype=torch.float32, device=self.device)
                    k3_tensor = torch.tensor(k3_list, dtype=torch.float32, device=self.device)
                    k4_tensor = torch.tensor(k4_list, dtype=torch.float32, device=self.device)
                    N_batch = self.batch_compute_Nk(k1_tensor, k2_tensor, k3_tensor, k4_tensor, token_indices)
                    vec = torch.mean(N_batch, dim=0)
                    logits = self.labeller(vec)
                    batch_logits.append(logits)
                    del N_batch, vec, token_indices, k1_tensor, k2_tensor, k3_tensor, k4_tensor
                
                if batch_logits:
                    all_logits = torch.stack(batch_logits, dim=0)
                    loss = criterion(all_logits, batch_labels)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item() * len(batch_hcubes)
                    total_hypercubes += len(batch_hcubes)
                    with torch.no_grad():
                        probs = torch.sigmoid(all_logits)
                        preds = (probs > 0.5).float()
                        total_correct += (preds == batch_labels).sum().item()
                        total_predictions += batch_labels.numel()
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            avg_loss = total_loss / total_hypercubes if total_hypercubes > 0 else 0.0
            avg_acc = total_correct / total_predictions if total_predictions > 0 else 0.0
            loss_history.append(avg_loss)
            acc_history.append(avg_acc)
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_model_state = copy.deepcopy(self.state_dict())
            
            if it % print_every == 0 or it == max_iters - 1:
                current_lr = scheduler.get_last_lr()[0]
                print(f"MLC Iter {it:3d}: Loss = {avg_loss:.6e}, Acc = {avg_acc:.4f}, LR = {current_lr:.6f}")
            
            if checkpoint_file and (it % checkpoint_interval == 0 or it == max_iters - 1):
                torch.save({
                    'iteration': it,
                    'model_state_dict': self.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss_history': loss_history,
                    'acc_history': acc_history,
                    'best_loss': best_loss
                }, checkpoint_file)
                print(f"Checkpoint saved at iteration {it}")
            
            if abs(prev_loss - avg_loss) < tol:
                print(f"Converged after {it+1} iterations.")
                if best_model_state is not None:
                    self.load_state_dict(best_model_state)
                break
                
            prev_loss = avg_loss
            scheduler.step()
        
        self.trained = True
        return loss_history, acc_history

    def self_train(self, hypercubes: List[List], max_iters: int = 100, tol: float = 1e-6,
                   learning_rate: float = 0.01, continued: bool = False, decay_rate: float = 1.0,
                   print_every: int = 10, batch_size: int = 32, checkpoint_file: Optional[str] = None,
                   checkpoint_interval: int = 5) -> List[float]:
        """Self-training using self-consistency: N(k) should match token embedding."""
        if not continued:
            self.reset_parameters()
        
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)
        
        history = []
        prev_loss = float('inf')
        best_loss = float('inf')
        best_model_state = None
        
        for it in range(max_iters):
            total_loss = 0.0
            total_hypercubes = 0
            
            indices = list(range(len(hypercubes)))
            random.shuffle(indices)
            
            for batch_start in range(0, len(indices), batch_size):
                batch_indices = indices[batch_start:batch_start + batch_size]
                batch_hcubes = [hypercubes[idx] for idx in batch_indices]
                
                optimizer.zero_grad()
                batch_loss = 0.0
                batch_count = 0
                
                for hcube in batch_hcubes:
                    tokens, coords = self.extract_hypercubes(hcube)
                    if not tokens:
                        continue
                    token_indices = self.token_to_indices(tokens)
                    k1_list = [c[0] for c in coords]
                    k2_list = [c[1] for c in coords]
                    k3_list = [c[2] for c in coords]
                    k4_list = [c[3] for c in coords]
                    k1_tensor = torch.tensor(k1_list, dtype=torch.float32, device=self.device)
                    k2_tensor = torch.tensor(k2_list, dtype=torch.float32, device=self.device)
                    k3_tensor = torch.tensor(k3_list, dtype=torch.float32, device=self.device)
                    k4_tensor = torch.tensor(k4_list, dtype=torch.float32, device=self.device)
                    
                    N_batch = self.batch_compute_Nk(k1_tensor, k2_tensor, k3_tensor, k4_tensor, token_indices)
                    embeddings = self.embedding(token_indices)
                    loss = torch.sum((N_batch - embeddings) ** 2) / len(tokens)
                    batch_loss += loss
                    batch_count += 1
                    del N_batch, embeddings, token_indices, k1_tensor, k2_tensor, k3_tensor, k4_tensor
                
                if batch_count > 0:
                    batch_loss = batch_loss / batch_count
                    batch_loss.backward()
                    optimizer.step()
                    total_loss += batch_loss.item() * batch_count
                    total_hypercubes += batch_count
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            avg_loss = total_loss / total_hypercubes if total_hypercubes > 0 else 0.0
            history.append(avg_loss)
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_model_state = copy.deepcopy(self.state_dict())
            
            if it % print_every == 0 or it == max_iters - 1:
                current_lr = scheduler.get_last_lr()[0]
                print(f"Self-Train Iter {it:3d}: Loss = {avg_loss:.6f}, LR = {current_lr:.6f}")
            
            if checkpoint_file and (it % checkpoint_interval == 0 or it == max_iters - 1):
                self._save_checkpoint(checkpoint_file, it, history, optimizer, scheduler, best_loss)
            
            if abs(prev_loss - avg_loss) < tol:
                print(f"Converged after {it+1} iterations")
                if best_model_state is not None:
                    self.load_state_dict(best_model_state)
                break
                
            prev_loss = avg_loss
            scheduler.step()
        
        self._compute_training_statistics(hypercubes)
        self.trained = True
        return history

    def _compute_training_statistics(self, hypercubes: List[List], batch_size: int = 50):
        """Compute average t vector from all windows in training data."""
        total_hypercubes = 0
        total_t = torch.zeros(self.m, device=self.device)
        
        with torch.no_grad():
            for i in range(0, len(hypercubes), batch_size):
                batch_hcubes = hypercubes[i:i+batch_size]
                for hcube in batch_hcubes:
                    tokens, coords = self.extract_hypercubes(hcube)
                    if not tokens:
                        continue
                    total_hypercubes += len(tokens)
                    token_indices = self.token_to_indices(tokens)
                    k1_list = [c[0] for c in coords]
                    k2_list = [c[1] for c in coords]
                    k3_list = [c[2] for c in coords]
                    k4_list = [c[3] for c in coords]
                    k1_tensor = torch.tensor(k1_list, dtype=torch.float32, device=self.device)
                    k2_tensor = torch.tensor(k2_list, dtype=torch.float32, device=self.device)
                    k3_tensor = torch.tensor(k3_list, dtype=torch.float32, device=self.device)
                    k4_tensor = torch.tensor(k4_list, dtype=torch.float32, device=self.device)
                    N_batch = self.batch_compute_Nk(k1_tensor, k2_tensor, k3_tensor, k4_tensor, token_indices)
                    total_t += N_batch.sum(dim=0)
                    del N_batch, token_indices, k1_tensor, k2_tensor, k3_tensor, k4_tensor
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        self.mean_hypercubes_per_sample = total_hypercubes / len(hypercubes) if hypercubes else 0
        self.mean_t = (total_t / total_hypercubes).cpu().numpy() if total_hypercubes > 0 else np.zeros(self.m)

    def _save_checkpoint(self, checkpoint_file: str, iteration: int, history: List[float],
                         optimizer, scheduler, best_loss: float):
        """Save training checkpoint."""
        checkpoint = {
            'iteration': iteration,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'history': history,
            'best_loss': best_loss,
            'training_stats': {
                'mean_t': self.mean_t,
                'mean_hypercubes_per_sample': self.mean_hypercubes_per_sample
            }
        }
        torch.save(checkpoint, checkpoint_file)
        print(f"Checkpoint saved at iteration {iteration}")

    def predict_t(self, hypercube: List) -> np.ndarray:
        """Predict target vector for a hypercube (average of all N(k1,k2,k3,k4) vectors)."""
        tokens, coords = self.extract_hypercubes(hypercube)
        if not tokens:
            return np.zeros(self.m)
            
        token_indices = self.token_to_indices(tokens)
        k1_list = [c[0] for c in coords]
        k2_list = [c[1] for c in coords]
        k3_list = [c[2] for c in coords]
        k4_list = [c[3] for c in coords]
        k1_tensor = torch.tensor(k1_list, dtype=torch.float32, device=self.device)
        k2_tensor = torch.tensor(k2_list, dtype=torch.float32, device=self.device)
        k3_tensor = torch.tensor(k3_list, dtype=torch.float32, device=self.device)
        k4_tensor = torch.tensor(k4_list, dtype=torch.float32, device=self.device)
        
        N_batch = self.batch_compute_Nk(k1_tensor, k2_tensor, k3_tensor, k4_tensor, token_indices)
        return (torch.mean(N_batch, dim=0)).detach().cpu().numpy()

    def predict_c(self, hypercube: List) -> Tuple[int, np.ndarray]:
        """Predict class label and probabilities for a hypercube."""
        if self.classifier is None:
            raise ValueError("Model must be trained first for classification")
        vec = self.predict_t(hypercube)
        vec_tensor = torch.tensor(vec, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits = self.classifier(vec_tensor.unsqueeze(0))
            probs = torch.softmax(logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()
        return pred, probs[0].cpu().numpy()

    def predict_l(self, hypercube: List, threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """Predict multi-label classification for a hypercube."""
        if self.labeller is None:
            raise ValueError("Model must be trained first for label prediction")
        tokens, coords = self.extract_hypercubes(hypercube)
        if not tokens:
            return np.zeros(self.num_labels), np.zeros(self.num_labels)
        token_indices = self.token_to_indices(tokens)
        k1_list = [c[0] for c in coords]
        k2_list = [c[1] for c in coords]
        k3_list = [c[2] for c in coords]
        k4_list = [c[3] for c in coords]
        k1_tensor = torch.tensor(k1_list, dtype=torch.float32, device=self.device)
        k2_tensor = torch.tensor(k2_list, dtype=torch.float32, device=self.device)
        k3_tensor = torch.tensor(k3_list, dtype=torch.float32, device=self.device)
        k4_tensor = torch.tensor(k4_list, dtype=torch.float32, device=self.device)
        N_batch = self.batch_compute_Nk(k1_tensor, k2_tensor, k3_tensor, k4_tensor, token_indices)
        vec = torch.mean(N_batch, dim=0)
        with torch.no_grad():
            logits = self.labeller(vec)
            probs = torch.sigmoid(logits).cpu().numpy()
        binary = (probs > threshold).astype(np.float32)
        return binary, probs

    def reconstruct(self, dim1: int, dim2: int, dim3: int, dim4: int, tau: float = 0.0) -> List:
        """
        Reconstruct a 4D hypercube of given dimensions by generating tiles of size window_size^4.
        All dimensions must be multiples of window_size.
        
        Returns:
            A 4D list representing the hypercube: outer list = dim1, each element is a dim2 list,
            each dim2 element is a dim3 list of strings (dim4). This matches input format.
        """
        assert self.trained, "Model must be trained first"
        if tau < 0:
            raise ValueError("Temperature must be non-negative")
        
        # Number of tiles in each dimension
        tiles1 = dim1 // self.window_size
        tiles2 = dim2 // self.window_size
        tiles3 = dim3 // self.window_size
        tiles4 = dim4 // self.window_size
        if any(d % self.window_size != 0 for d in [dim1, dim2, dim3, dim4]):
            print(f"Warning: dimensions ({dim1},{dim2},{dim3},{dim4}) not multiples of window_size ({self.window_size}). Cropping will occur.")
        
        mean_t_tensor = torch.tensor(self.mean_t, dtype=torch.float32, device=self.device)
        all_token_indices = torch.arange(len(self.tokens), device=self.device)
        
        # Generate tokens for each tile
        tile_tokens = []  # 4D list of tokens
        for d1 in range(tiles1):
            dim1_tokens = []
            for d2 in range(tiles2):
                dim2_tokens = []
                for d3 in range(tiles3):
                    dim3_tokens = []
                    for d4 in range(tiles4):
                        k1 = d1 * self.window_size
                        k2 = d2 * self.window_size
                        k3 = d3 * self.window_size
                        k4 = d4 * self.window_size
                        k1_tensor = torch.full((len(self.tokens),), k1, dtype=torch.float32, device=self.device)
                        k2_tensor = torch.full((len(self.tokens),), k2, dtype=torch.float32, device=self.device)
                        k3_tensor = torch.full((len(self.tokens),), k3, dtype=torch.float32, device=self.device)
                        k4_tensor = torch.full((len(self.tokens),), k4, dtype=torch.float32, device=self.device)
                        N_all = self.batch_compute_Nk(k1_tensor, k2_tensor, k3_tensor, k4_tensor, all_token_indices)
                        errors = torch.sum((N_all - mean_t_tensor) ** 2, dim=1)
                        scores = -errors
                        
                        if tau == 0:
                            best_idx = torch.argmax(scores).item()
                            token = self.idx_to_token[best_idx]
                        else:
                            probs = torch.softmax(scores / tau, dim=0).detach().cpu().numpy()
                            chosen_idx = random.choices(range(len(self.tokens)), weights=probs, k=1)[0]
                            token = self.idx_to_token[chosen_idx]
                        dim3_tokens.append(token)
                    dim2_tokens.append(dim3_tokens)
                dim1_tokens.append(dim2_tokens)
            tile_tokens.append(dim1_tokens)
        
        # Initialize a 4D character array with '_' to avoid None values
        char_cube = [[[['_' for _ in range(dim4)] for _ in range(dim3)] for _ in range(dim2)] for _ in range(dim1)]
        
        # Fill the character cube with token data
        for d1 in range(tiles1):
            for d2 in range(tiles2):
                for d3 in range(tiles3):
                    for d4 in range(tiles4):
                        token = tile_tokens[d1][d2][d3][d4]
                        idx = 0
                        for sub_d1 in range(self.window_size):
                            for sub_d2 in range(self.window_size):
                                for sub_d3 in range(self.window_size):
                                    for sub_d4 in range(self.window_size):
                                        target_d1 = d1 * self.window_size + sub_d1
                                        target_d2 = d2 * self.window_size + sub_d2
                                        target_d3 = d3 * self.window_size + sub_d3
                                        target_d4 = sub_d4
                                        # Only assign if within bounds; out-of-bounds already '_'
                                        if (target_d1 < dim1 and target_d2 < dim2 and 
                                            target_d3 < dim3 and target_d4 < dim4):
                                            char_cube[target_d1][target_d2][target_d3][target_d4] = token[idx]
                                        idx += 1
        
        # Convert to list-of-list-of-strings format: hypercube[d1][d2] = list of strings (each string is a row of length dim4)
        result = []
        for d1 in range(dim1):
            dim1_layer = []
            for d2 in range(dim2):
                rows = []
                for d3 in range(dim3):
                    row_chars = [char_cube[d1][d2][d3][d4] for d4 in range(dim4)]
                    rows.append(''.join(row_chars))
                dim1_layer.append(rows)
            result.append(dim1_layer)
        return result

    def save(self, filename: str):
        """Save model state to file."""
        torch.save(self.state_dict(), filename)
        print(f"Model saved to {filename}")

    def load(self, filename: str):
        """Load model state from file."""
        try:
            state_dict = torch.load(filename, map_location=self.device, weights_only=True)
        except TypeError:
            state_dict = torch.load(filename, map_location=self.device)
        self.load_state_dict(state_dict)
        print(f"Model loaded from {filename}")
        return self


# === Example Usage ===
if __name__ == "__main__":
    from statistics import correlation
    
    print("=" * 50)
    print("Spatial Dual Descriptor PM4 - 4D Hypercube Version")
    print("Optimized with batch processing")
    print("=" * 50)
    
    # Set random seeds
    torch.manual_seed(1)
    random.seed(1)
    np.random.seed(1)
    
    charset = ['A', 'C', 'G', 'T']
    window_size = 1
    vec_dim = 2
    user_step = 1   # step size for nonlinear mode (equal to window_size, so non-overlapping)
    
    # Generate 100 random 4D hypercubes with random target vectors
    # Dimensions: dim1=6, dim2=6, dim3=6, dim4=6 (so that each dimension is multiple of window_size=3)
    hypercubes = []
    t_list = []
    dim1 = 6
    dim2 = 6
    dim3 = 6
    dim4 = 6
    for _ in range(100):
        # Create a 4D hypercube: list of dim1, each is list of dim2, each is list of strings (dim3) of length dim4
        hypercube = []
        for d1 in range(dim1):
            dim1_layer = []
            for d2 in range(dim2):
                # Each d2 is a list of dim3 strings, each string of length dim4
                rows = [''.join(random.choices(charset, k=dim4)) for _ in range(dim3)]
                dim1_layer.append(rows)
            hypercube.append(dim1_layer)
        hypercubes.append(hypercube)
        # Random target vector
        t_list.append([random.uniform(-1.0, 1.0) for _ in range(vec_dim)])
    
    # Initialize the model
    dd = SpatialDualDescriptorPM4(
        charset,
        window_size=window_size,
        vec_dim=vec_dim,
        mode='nonlinear',
        user_step=user_step,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    print(f"\nUsing device: {dd.device}")
    print(f"Number of tokens: {len(dd.tokens)}")
    
    # Training regression model
    print("\n" + "=" * 50)
    print("Starting Gradient Descent Training (Regression)")
    print("=" * 50)
    dd.reg_train(hypercubes, t_list, max_iters=50, tol=1e-19, learning_rate=0.1, decay_rate=0.999, batch_size=2048)
    
    # Predict target vector of the first hypercube
    ahypercube = hypercubes[0]
    t_pred = dd.predict_t(ahypercube)
    print(f"\nPredicted t for first hypercube: {[round(x.item(), 4) for x in t_pred]}")
    
    # Calculate correlation between predicted and real targets
    pred_t_list = [dd.predict_t(hcube) for hcube in hypercubes]
    corr_sum = 0.0
    for i in range(dd.m):
        actu_t = [t_vec[i] for t_vec in t_list]
        pred_t = [t_vec[i] for t_vec in pred_t_list]
        corr = correlation(actu_t, pred_t)
        print(f"Dimension {i} prediction correlation: {corr:.4f}")
        corr_sum += corr
    corr_avg = corr_sum / dd.m
    print(f"Average correlation: {corr_avg:.4f}")
    
    # Reconstruct representative hypercubes (first layer of dim1 and dim2, first 2 rows)
    rec_det = dd.reconstruct(dim1=dim1, dim2=dim2, dim3=dim3, dim4=dim4, tau=0.0)
    rec_rand = dd.reconstruct(dim1=dim1, dim2=dim2, dim3=dim3, dim4=dim4, tau=0.5)
    print("\nDeterministic Reconstruction (first dim1 layer, first dim2 layer, first 2 rows):")
    for row in rec_det[0][0][:2]:
        print(row[:50] + "...")
    print("\nStochastic Reconstruction (tau=0.5, first dim1 layer, first dim2 layer, first 2 rows):")
    for row in rec_rand[0][0][:2]:
        print(row[:50] + "...")
    
    # === Classification Task ===
    print("\n" + "=" * 50)
    print("Classification Task")
    print("=" * 50)
    
    num_classes = 3
    class_hypercubes = []
    class_labels = []
    
    # Create hypercubes with different patterns
    for class_id in range(num_classes):
        for _ in range(50):  # 50 hypercubes per class
            if class_id == 0:
                # Class 0: High A content
                row_probs = [0.6, 0.1, 0.1, 0.2]
            elif class_id == 1:
                # Class 1: High GC content
                row_probs = [0.1, 0.4, 0.4, 0.1]
            else:
                # Class 2: Balanced
                row_probs = [0.25, 0.25, 0.25, 0.25]
            hypercube = []
            for d1 in range(dim1):
                dim1_layer = []
                for d2 in range(dim2):
                    rows = [''.join(random.choices(charset, weights=row_probs, k=dim4)) for _ in range(dim3)]
                    dim1_layer.append(rows)
                hypercube.append(dim1_layer)
            class_hypercubes.append(hypercube)
            class_labels.append(class_id)
    
    # Initialize new model for classification
    dd_cls = SpatialDualDescriptorPM4(
        charset,
        window_size=window_size,
        vec_dim=vec_dim,
        mode='nonlinear',
        user_step=user_step,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    print("\nStarting Classification Training")
    dd_cls.cls_train(class_hypercubes, class_labels, num_classes,
                     max_iters=50, tol=1e-8, learning_rate=0.05,
                     decay_rate=0.99, batch_size=32, print_every=1)
    
    # Evaluate on training set
    correct = 0
    for hcube, true_label in zip(class_hypercubes, class_labels):
        pred_class, _ = dd_cls.predict_c(hcube)
        if pred_class == true_label:
            correct += 1
    accuracy = correct / len(class_hypercubes)
    print(f"\nClassification Accuracy: {accuracy:.4f} ({correct}/{len(class_hypercubes)})")
    
    # === Multi-label Classification ===
    print("\n" + "=" * 50)
    print("Multi-Label Classification")
    print("=" * 50)
    
    num_labels = 4
    label_hypercubes = []
    labels = []
    for _ in range(100):
        hypercube = []
        for d1 in range(dim1):
            dim1_layer = []
            for d2 in range(dim2):
                rows = [''.join(random.choices(charset, k=dim4)) for _ in range(dim3)]
                dim1_layer.append(rows)
            hypercube.append(dim1_layer)
        label_hypercubes.append(hypercube)
        # Random binary labels
        label_vec = [1 if random.random() > 0.7 else 0 for _ in range(num_labels)]
        labels.append(label_vec)
    
    dd_lbl = SpatialDualDescriptorPM4(
        charset,
        window_size=window_size,
        vec_dim=vec_dim,
        mode='nonlinear',
        user_step=user_step,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    print("\nStarting Multi-Label Training")
    loss_hist, acc_hist = dd_lbl.lbl_train(
        label_hypercubes, labels, num_labels,
        max_iters=50, tol=1e-16, learning_rate=0.01,
        decay_rate=0.99, print_every=10, batch_size=32
    )
    print(f"\nFinal training loss: {loss_hist[-1]:.6f}")
    print(f"Final training accuracy: {acc_hist[-1]:.4f}")
    
    # Example prediction on a test hypercube
    test_hypercube = []
    for d1 in range(dim1):
        dim1_layer = []
        for d2 in range(dim2):
            rows = [''.join(random.choices(charset, k=dim4)) for _ in range(dim3)]
            dim1_layer.append(rows)
        test_hypercube.append(dim1_layer)
    binary_pred, prob_pred = dd_lbl.predict_l(test_hypercube, threshold=0.5)
    print("\nExample multi-label prediction:")
    print(f"Binary predictions: {binary_pred}")
    print(f"Probabilities: {[f'{p:.4f}' for p in prob_pred]}")
    
    # === Self-training ===
    print("\n" + "=" * 50)
    print("Self-Training Example")
    print("=" * 50)
    
    dd_self = SpatialDualDescriptorPM4(
        charset,
        window_size=window_size,
        vec_dim=vec_dim,
        mode='nonlinear',
        user_step=user_step,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Generate self-training hypercubes
    self_hypercubes = []
    for _ in range(10):
        hypercube = []
        for d1 in range(dim1):
            dim1_layer = []
            for d2 in range(dim2):
                rows = [''.join(random.choices(charset, k=dim4)) for _ in range(dim3)]
                dim1_layer.append(rows)
            hypercube.append(dim1_layer)
        self_hypercubes.append(hypercube)
    
    print("Training for self-consistency:")
    self_hist = dd_self.self_train(self_hypercubes, max_iters=50, tol=1e-8, learning_rate=0.01, batch_size=1024)
    
    rec_hypercube = dd_self.reconstruct(dim1=dim1, dim2=dim2, dim3=dim3, dim4=dim4, tau=0.2)
    print("\nReconstructed hypercube (first dim1 layer, first dim2 layer, first 2 rows):")
    for row in rec_hypercube[0][0][:2]:
        print(row[:50] + "...")
    
    print("\nAll tests completed successfully!")
