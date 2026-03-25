# Copyright (C) 2005-2025, Bin-Guang Ma (mbg@mail.hzau.edu.cn); SPDX-License-Identifier: MIT
# The Spatial Dual Descriptor Vector class (P Matrix form) for 3D character arrays implemented with PyTorch
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

class SpatialDualDescriptorPM3(nn.Module):
    """
    Spatial Dual Descriptor for 3D character arrays (cubes) with GPU acceleration:
      - matrix P ∈ R^{m×m} of basis coefficients
      - embedding: cube token embeddings in R^m
      - indexed periods: period1[i,j], period2[i,j], period3[i,j] for three spatial dimensions
      - basis function phi_{i,j}(k1,k2,k3) = cos(2π*k1/period1) * cos(2π*k2/period2) * cos(2π*k3/period3)
      - supports 'linear' (step=1) or 'nonlinear' (step-by-window-size) cube extraction
    """
    def __init__(self, charset: List[str], window_size: int = 3, rank_mode: str = 'drop',
                 vec_dim: int = 2, mode: str = 'linear', user_step: Optional[int] = None,
                 device: str = 'cuda'):
        """
        Args:
            charset: List of characters that appear in the input grid.
            window_size: Size of the cubic sliding window (e.g., 3 for 3x3x3 cubes).
            rank_mode: 'pad' or 'drop' for handling incomplete windows at boundaries.
            vec_dim: Dimension of the embedding space (m).
            mode: 'linear' (step=1) or 'nonlinear' (step = user_step or window_size).
            user_step: Step size for nonlinear mode (same for all dimensions).
            device: 'cuda' or 'cpu'.
        """
        super().__init__()
        self.charset = list(charset)
        self.window_size = window_size          # cubic window side length
        self.rank_mode = rank_mode
        self.m = vec_dim
        assert mode in ('linear', 'nonlinear')
        self.mode = mode
        self.step = user_step if user_step is not None else window_size   # step for nonlinear mode
        self.trained = False
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Generate all possible cube tokens (window_size x window_size x window_size characters)
        # Each token is a concatenation of all characters in row-major order across all layers.
        # Total length = window_size^3
        chars_with_pad = self.charset + (['_'] if self.rank_mode == 'pad' else [])
        # Generate all possible tokens (complete cube)
        token_length = window_size * window_size * window_size
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
        
        # Precompute period matrices for three dimensions (fixed, not trainable)
        periods1 = torch.zeros(self.m, self.m, dtype=torch.float32)
        periods2 = torch.zeros(self.m, self.m, dtype=torch.float32)
        periods3 = torch.zeros(self.m, self.m, dtype=torch.float32)
        for i in range(self.m):
            for j in range(self.m):
                base = i * self.m + j + 2
                periods1[i, j] = base
                periods2[i, j] = base
                periods3[i, j] = base   # can be made different if needed
        self.register_buffer('periods1', periods1)
        self.register_buffer('periods2', periods2)
        self.register_buffer('periods3', periods3)

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
    
    def extract_cubes(self, cube: List[List[List[str]]]) -> Tuple[List[str], List[Tuple[int, int, int]]]:
        """
        Extract cubic window tokens and their top-front-left coordinates from a 3D character cube.
        
        Args:
            cube: 3D list of characters, with dimensions D x H x W (depth, height, width).
                  Each element is a string of length 1? Actually we'll assume cube is a 3D list of characters,
                  but for simplicity we represent it as a list of layers, each layer is a list of strings (rows).
                  So cube[d][h][w] is the character at depth d, height h, width w.
                  All dimensions are assumed to be consistent.
        
        Returns:
            tokens: List of cube tokens (strings of length window_size^3).
            coords: List of (d, h, w) top-front-left coordinates of each cube.
        """
        D = len(cube)
        H = len(cube[0]) if D > 0 else 0
        W = len(cube[0][0]) if H > 0 else 0
        step = 1 if self.mode == 'linear' else self.step
        
        tokens = []
        coords = []
        
        if self.rank_mode == 'drop':
            # Only extract fully contained cubes
            for d in range(0, D - self.window_size + 1, step):
                for h in range(0, H - self.window_size + 1, step):
                    for w in range(0, W - self.window_size + 1, step):
                        # Build token: concatenate all characters in cube order (depth major, then row major)
                        token_chars = []
                        for i in range(self.window_size):
                            for j in range(self.window_size):
                                for k in range(self.window_size):
                                    token_chars.append(cube[d+i][h+j][w+k])
                        token = ''.join(token_chars)
                        tokens.append(token)
                        coords.append((d, h, w))
        else:  # pad mode
            for d in range(0, D, step):
                for h in range(0, H, step):
                    for w in range(0, W, step):
                        # Build token with padding
                        token_chars = []
                        for i in range(self.window_size):
                            for j in range(self.window_size):
                                for k in range(self.window_size):
                                    dd = d + i
                                    hh = h + j
                                    ww = w + k
                                    if dd < D and hh < H and ww < W:
                                        token_chars.append(cube[dd][hh][ww])
                                    else:
                                        token_chars.append('_')
                        token = ''.join(token_chars)
                        tokens.append(token)
                        coords.append((d, h, w))
        return tokens, coords

    def batch_compute_Nk(self, k1_tensor: torch.Tensor, k2_tensor: torch.Tensor,
                         k3_tensor: torch.Tensor, token_indices: torch.Tensor) -> torch.Tensor:
        """
        Vectorized computation of N(k1,k2,k3) vectors for a batch of positions and tokens.
        
        Args:
            k1_tensor: Tensor of depth indices [batch_size]
            k2_tensor: Tensor of height indices [batch_size]
            k3_tensor: Tensor of width indices [batch_size]
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
        
        # Compute basis function: product of cosines for three dimensions
        phi = (torch.cos(2 * math.pi * k1_exp / self.periods1) *
               torch.cos(2 * math.pi * k2_exp / self.periods2) *
               torch.cos(2 * math.pi * k3_exp / self.periods3))  # [batch_size, m, m]
        
        # Optimized computation using einsum
        Nk = torch.einsum('bj,ij,bij->bi', x, self.P, phi)
        return Nk

    def compute_Nk(self, k1: int, k2: int, k3: int, token_idx: int) -> torch.Tensor:
        """Compute N(k1,k2,k3) for a single position and token (uses batch internally)."""
        k1_tensor = torch.tensor([k1], dtype=torch.float32, device=self.device)
        k2_tensor = torch.tensor([k2], dtype=torch.float32, device=self.device)
        k3_tensor = torch.tensor([k3], dtype=torch.float32, device=self.device)
        idx_tensor = torch.tensor([token_idx], device=self.device)
        result = self.batch_compute_Nk(k1_tensor, k2_tensor, k3_tensor, idx_tensor)
        return result[0]

    def describe(self, cube: List[List[List[str]]]) -> np.ndarray:
        """
        Compute N(k1,k2,k3) vectors for each cube in the 3D array.
        
        Args:
            cube: 3D list of characters.
            
        Returns:
            Numpy array of shape (num_cubes, m).
        """
        tokens, coords = self.extract_cubes(cube)
        if not tokens:
            return np.array([])
        
        token_indices = self.token_to_indices(tokens)
        k1_list = [c[0] for c in coords]
        k2_list = [c[1] for c in coords]
        k3_list = [c[2] for c in coords]
        k1_tensor = torch.tensor(k1_list, dtype=torch.float32, device=self.device)
        k2_tensor = torch.tensor(k2_list, dtype=torch.float32, device=self.device)
        k3_tensor = torch.tensor(k3_list, dtype=torch.float32, device=self.device)
        
        N_batch = self.batch_compute_Nk(k1_tensor, k2_tensor, k3_tensor, token_indices)
        return N_batch.detach().cpu().numpy()

    def S(self, cube: List[List[List[str]]]) -> List[np.ndarray]:
        """
        Compute list of S(l)=sum(N(k1,k2,k3)) for l=1..num_cubes (cumulative sum in extraction order).
        """
        tokens, coords = self.extract_cubes(cube)
        if not tokens:
            return []
            
        token_indices = self.token_to_indices(tokens)
        k1_list = [c[0] for c in coords]
        k2_list = [c[1] for c in coords]
        k3_list = [c[2] for c in coords]
        k1_tensor = torch.tensor(k1_list, dtype=torch.float32, device=self.device)
        k2_tensor = torch.tensor(k2_list, dtype=torch.float32, device=self.device)
        k3_tensor = torch.tensor(k3_list, dtype=torch.float32, device=self.device)
        
        N_batch = self.batch_compute_Nk(k1_tensor, k2_tensor, k3_tensor, token_indices)
        S_cum = torch.cumsum(N_batch, dim=0)
        return [s.detach().cpu().numpy() for s in S_cum]

    def D(self, cubes: List[List[List[List[str]]]], t_list: List[List[float]]) -> float:
        """
        Compute mean squared deviation D across cubes:
        D = average over all positions of (N(k)-t_cube)^2
        """
        total_loss = 0.0
        total_positions = 0
        
        # Convert target vectors to tensor
        t_tensors = [torch.tensor(t, dtype=torch.float32, device=self.device) for t in t_list]
        
        for cube, t in zip(cubes, t_tensors):
            tokens, coords = self.extract_cubes(cube)
            if not tokens:
                continue
                
            token_indices = self.token_to_indices(tokens)
            k1_list = [c[0] for c in coords]
            k2_list = [c[1] for c in coords]
            k3_list = [c[2] for c in coords]
            k1_tensor = torch.tensor(k1_list, dtype=torch.float32, device=self.device)
            k2_tensor = torch.tensor(k2_list, dtype=torch.float32, device=self.device)
            k3_tensor = torch.tensor(k3_list, dtype=torch.float32, device=self.device)
            
            N_batch = self.batch_compute_Nk(k1_tensor, k2_tensor, k3_tensor, token_indices)
            losses = torch.sum((N_batch - t) ** 2, dim=1)
            total_loss += losses.sum().item()
            total_positions += len(tokens)
                
        return total_loss / total_positions if total_positions else 0.0

    def d(self, cube: List[List[List[str]]], t: List[float]) -> float:
        """Compute pattern deviation value for a single cube."""
        return self.D([cube], [t])

    def reg_train(self, cubes: List[List[List[List[str]]]], t_list: List[List[float]],
                  max_iters: int = 1000, tol: float = 1e-8, learning_rate: float = 0.01,
                  continued: bool = False, decay_rate: float = 1.0, print_every: int = 10,
                  batch_size: int = 32, checkpoint_file: Optional[str] = None,
                  checkpoint_interval: int = 10) -> List[float]:
        """Train the model using gradient descent with cube-level batch processing."""
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
            total_cubes = 0
            
            indices = list(range(len(cubes)))
            random.shuffle(indices)
            
            for batch_start in range(0, len(indices), batch_size):
                batch_indices = indices[batch_start:batch_start + batch_size]
                batch_cubes = [cubes[idx] for idx in batch_indices]
                batch_targets = [t_tensors[idx] for idx in batch_indices]
                
                optimizer.zero_grad()
                batch_loss = 0.0
                
                for cube, target in zip(batch_cubes, batch_targets):
                    tokens, coords = self.extract_cubes(cube)
                    if not tokens:
                        continue
                        
                    token_indices = self.token_to_indices(tokens)
                    k1_list = [c[0] for c in coords]
                    k2_list = [c[1] for c in coords]
                    k3_list = [c[2] for c in coords]
                    k1_tensor = torch.tensor(k1_list, dtype=torch.float32, device=self.device)
                    k2_tensor = torch.tensor(k2_list, dtype=torch.float32, device=self.device)
                    k3_tensor = torch.tensor(k3_list, dtype=torch.float32, device=self.device)
                    
                    N_batch = self.batch_compute_Nk(k1_tensor, k2_tensor, k3_tensor, token_indices)
                    cube_pred = torch.mean(N_batch, dim=0)
                    loss = torch.sum((cube_pred - target) ** 2)
                    batch_loss += loss
                    
                    del N_batch, cube_pred, token_indices, k1_tensor, k2_tensor, k3_tensor
                
                if len(batch_cubes) > 0:
                    batch_loss = batch_loss / len(batch_cubes)
                    batch_loss.backward()
                    optimizer.step()
                    total_loss += batch_loss.item() * len(batch_cubes)
                    total_cubes += len(batch_cubes)
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            avg_loss = total_loss / total_cubes if total_cubes > 0 else 0.0
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
        
        self._compute_training_statistics(cubes)
        self.trained = True
        return history

    def cls_train(self, cubes: List[List[List[List[str]]]], labels: List[int], num_classes: int,
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
            total_cubes = 0
            correct = 0
            
            indices = list(range(len(cubes)))
            random.shuffle(indices)
            
            for batch_start in range(0, len(indices), batch_size):
                batch_indices = indices[batch_start:batch_start + batch_size]
                batch_cubes = [cubes[idx] for idx in batch_indices]
                batch_labels = label_tensors[batch_indices]
                
                optimizer.zero_grad()
                batch_logits = []
                
                for cube in batch_cubes:
                    tokens, coords = self.extract_cubes(cube)
                    if not tokens:
                        vec = torch.zeros(self.m, device=self.device)
                    else:
                        token_indices = self.token_to_indices(tokens)
                        k1_list = [c[0] for c in coords]
                        k2_list = [c[1] for c in coords]
                        k3_list = [c[2] for c in coords]
                        k1_tensor = torch.tensor(k1_list, dtype=torch.float32, device=self.device)
                        k2_tensor = torch.tensor(k2_list, dtype=torch.float32, device=self.device)
                        k3_tensor = torch.tensor(k3_list, dtype=torch.float32, device=self.device)
                        N_batch = self.batch_compute_Nk(k1_tensor, k2_tensor, k3_tensor, token_indices)
                        vec = torch.mean(N_batch, dim=0)
                        del N_batch, token_indices, k1_tensor, k2_tensor, k3_tensor
                    logits = self.classifier(vec.unsqueeze(0))
                    batch_logits.append(logits)
                
                if batch_logits:
                    all_logits = torch.cat(batch_logits, dim=0)
                    loss = criterion(all_logits, batch_labels)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item() * len(batch_cubes)
                    total_cubes += len(batch_cubes)
                    with torch.no_grad():
                        preds = torch.argmax(all_logits, dim=1)
                        correct += (preds == batch_labels).sum().item()
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            avg_loss = total_loss / total_cubes if total_cubes > 0 else 0.0
            acc = correct / total_cubes if total_cubes > 0 else 0.0
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

    def lbl_train(self, cubes: List[List[List[List[str]]]], labels: Union[List[List[float]], np.ndarray],
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
            total_cubes = 0
            
            indices = list(range(len(cubes)))
            random.shuffle(indices)
            
            for batch_start in range(0, len(indices), batch_size):
                batch_indices = indices[batch_start:batch_start + batch_size]
                batch_cubes = [cubes[idx] for idx in batch_indices]
                batch_labels = labels_tensor[batch_indices]
                
                optimizer.zero_grad()
                batch_logits = []
                
                for cube in batch_cubes:
                    tokens, coords = self.extract_cubes(cube)
                    if not tokens:
                        continue
                    token_indices = self.token_to_indices(tokens)
                    k1_list = [c[0] for c in coords]
                    k2_list = [c[1] for c in coords]
                    k3_list = [c[2] for c in coords]
                    k1_tensor = torch.tensor(k1_list, dtype=torch.float32, device=self.device)
                    k2_tensor = torch.tensor(k2_list, dtype=torch.float32, device=self.device)
                    k3_tensor = torch.tensor(k3_list, dtype=torch.float32, device=self.device)
                    N_batch = self.batch_compute_Nk(k1_tensor, k2_tensor, k3_tensor, token_indices)
                    vec = torch.mean(N_batch, dim=0)
                    logits = self.labeller(vec)
                    batch_logits.append(logits)
                    del N_batch, vec, token_indices, k1_tensor, k2_tensor, k3_tensor
                
                if batch_logits:
                    all_logits = torch.stack(batch_logits, dim=0)
                    loss = criterion(all_logits, batch_labels)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item() * len(batch_cubes)
                    total_cubes += len(batch_cubes)
                    with torch.no_grad():
                        probs = torch.sigmoid(all_logits)
                        preds = (probs > 0.5).float()
                        total_correct += (preds == batch_labels).sum().item()
                        total_predictions += batch_labels.numel()
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            avg_loss = total_loss / total_cubes if total_cubes > 0 else 0.0
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

    def self_train(self, cubes: List[List[List[List[str]]]], max_iters: int = 100, tol: float = 1e-6,
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
            total_cubes = 0
            
            indices = list(range(len(cubes)))
            random.shuffle(indices)
            
            for batch_start in range(0, len(indices), batch_size):
                batch_indices = indices[batch_start:batch_start + batch_size]
                batch_cubes = [cubes[idx] for idx in batch_indices]
                
                optimizer.zero_grad()
                batch_loss = 0.0
                batch_count = 0
                
                for cube in batch_cubes:
                    tokens, coords = self.extract_cubes(cube)
                    if not tokens:
                        continue
                    token_indices = self.token_to_indices(tokens)
                    k1_list = [c[0] for c in coords]
                    k2_list = [c[1] for c in coords]
                    k3_list = [c[2] for c in coords]
                    k1_tensor = torch.tensor(k1_list, dtype=torch.float32, device=self.device)
                    k2_tensor = torch.tensor(k2_list, dtype=torch.float32, device=self.device)
                    k3_tensor = torch.tensor(k3_list, dtype=torch.float32, device=self.device)
                    
                    N_batch = self.batch_compute_Nk(k1_tensor, k2_tensor, k3_tensor, token_indices)
                    embeddings = self.embedding(token_indices)
                    loss = torch.sum((N_batch - embeddings) ** 2) / len(tokens)
                    batch_loss += loss
                    batch_count += 1
                    del N_batch, embeddings, token_indices, k1_tensor, k2_tensor, k3_tensor
                
                if batch_count > 0:
                    batch_loss = batch_loss / batch_count
                    batch_loss.backward()
                    optimizer.step()
                    total_loss += batch_loss.item() * batch_count
                    total_cubes += batch_count
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            avg_loss = total_loss / total_cubes if total_cubes > 0 else 0.0
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
        
        self._compute_training_statistics(cubes)
        self.trained = True
        return history

    def _compute_training_statistics(self, cubes: List[List[List[List[str]]]], batch_size: int = 50):
        """Compute average t vector from all windows in training data."""
        total_cubes = 0
        total_t = torch.zeros(self.m, device=self.device)
        
        with torch.no_grad():
            for i in range(0, len(cubes), batch_size):
                batch_cubes = cubes[i:i+batch_size]
                for cube in batch_cubes:
                    tokens, coords = self.extract_cubes(cube)
                    if not tokens:
                        continue
                    total_cubes += len(tokens)
                    token_indices = self.token_to_indices(tokens)
                    k1_list = [c[0] for c in coords]
                    k2_list = [c[1] for c in coords]
                    k3_list = [c[2] for c in coords]
                    k1_tensor = torch.tensor(k1_list, dtype=torch.float32, device=self.device)
                    k2_tensor = torch.tensor(k2_list, dtype=torch.float32, device=self.device)
                    k3_tensor = torch.tensor(k3_list, dtype=torch.float32, device=self.device)
                    N_batch = self.batch_compute_Nk(k1_tensor, k2_tensor, k3_tensor, token_indices)
                    total_t += N_batch.sum(dim=0)
                    del N_batch, token_indices, k1_tensor, k2_tensor, k3_tensor
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        self.mean_cubes_per_grid = total_cubes / len(cubes) if cubes else 0
        self.mean_t = (total_t / total_cubes).cpu().numpy() if total_cubes > 0 else np.zeros(self.m)

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
                'mean_cubes_per_grid': self.mean_cubes_per_grid
            }
        }
        torch.save(checkpoint, checkpoint_file)
        print(f"Checkpoint saved at iteration {iteration}")

    def predict_t(self, cube: List[List[List[str]]]) -> np.ndarray:
        """Predict target vector for a cube (average of all N(k1,k2,k3) vectors)."""
        tokens, coords = self.extract_cubes(cube)
        if not tokens:
            return np.zeros(self.m)
            
        token_indices = self.token_to_indices(tokens)
        k1_list = [c[0] for c in coords]
        k2_list = [c[1] for c in coords]
        k3_list = [c[2] for c in coords]
        k1_tensor = torch.tensor(k1_list, dtype=torch.float32, device=self.device)
        k2_tensor = torch.tensor(k2_list, dtype=torch.float32, device=self.device)
        k3_tensor = torch.tensor(k3_list, dtype=torch.float32, device=self.device)
        
        N_batch = self.batch_compute_Nk(k1_tensor, k2_tensor, k3_tensor, token_indices)
        return (torch.mean(N_batch, dim=0)).detach().cpu().numpy()

    def predict_c(self, cube: List[List[List[str]]]) -> Tuple[int, np.ndarray]:
        """Predict class label and probabilities for a cube."""
        if self.classifier is None:
            raise ValueError("Model must be trained first for classification")
        vec = self.predict_t(cube)
        vec_tensor = torch.tensor(vec, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits = self.classifier(vec_tensor.unsqueeze(0))
            probs = torch.softmax(logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()
        return pred, probs[0].cpu().numpy()

    def predict_l(self, cube: List[List[List[str]]], threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """Predict multi-label classification for a cube."""
        if self.labeller is None:
            raise ValueError("Model must be trained first for label prediction")
        tokens, coords = self.extract_cubes(cube)
        if not tokens:
            return np.zeros(self.num_labels), np.zeros(self.num_labels)
        token_indices = self.token_to_indices(tokens)
        k1_list = [c[0] for c in coords]
        k2_list = [c[1] for c in coords]
        k3_list = [c[2] for c in coords]
        k1_tensor = torch.tensor(k1_list, dtype=torch.float32, device=self.device)
        k2_tensor = torch.tensor(k2_list, dtype=torch.float32, device=self.device)
        k3_tensor = torch.tensor(k3_list, dtype=torch.float32, device=self.device)
        N_batch = self.batch_compute_Nk(k1_tensor, k2_tensor, k3_tensor, token_indices)
        vec = torch.mean(N_batch, dim=0)
        with torch.no_grad():
            logits = self.labeller(vec)
            probs = torch.sigmoid(logits).cpu().numpy()
        binary = (probs > threshold).astype(np.float32)
        return binary, probs

    def reconstruct(self, depth: int, height: int, width: int, tau: float = 0.0) -> List[List[List[str]]]:
        """
        Reconstruct a 3D cube of given dimensions by generating cubes of size window_size^3.
        All dimensions must be multiples of window_size.
        """
        assert self.trained, "Model must be trained first"
        if tau < 0:
            raise ValueError("Temperature must be non-negative")
        
        # Number of tiles in each dimension
        tiles_d = depth // self.window_size
        tiles_h = height // self.window_size
        tiles_w = width // self.window_size
        if depth % self.window_size != 0 or height % self.window_size != 0 or width % self.window_size != 0:
            print(f"Warning: dimensions ({depth}, {height}, {width}) not multiples of window_size ({self.window_size}). Cropping will occur.")
        
        mean_t_tensor = torch.tensor(self.mean_t, dtype=torch.float32, device=self.device)
        all_token_indices = torch.arange(len(self.tokens), device=self.device)
        
        # Generate tokens for each cube tile
        tile_tokens = []  # 3D list of tokens
        for d in range(tiles_d):
            layer_tokens = []
            for h in range(tiles_h):
                row_tokens = []
                for w in range(tiles_w):
                    k1 = d * self.window_size
                    k2 = h * self.window_size
                    k3 = w * self.window_size
                    k1_tensor = torch.full((len(self.tokens),), k1, dtype=torch.float32, device=self.device)
                    k2_tensor = torch.full((len(self.tokens),), k2, dtype=torch.float32, device=self.device)
                    k3_tensor = torch.full((len(self.tokens),), k3, dtype=torch.float32, device=self.device)
                    N_all = self.batch_compute_Nk(k1_tensor, k2_tensor, k3_tensor, all_token_indices)
                    errors = torch.sum((N_all - mean_t_tensor) ** 2, dim=1)
                    scores = -errors
                    
                    if tau == 0:
                        best_idx = torch.argmax(scores).item()
                        token = self.idx_to_token[best_idx]
                    else:
                        probs = torch.softmax(scores / tau, dim=0).detach().cpu().numpy()
                        chosen_idx = random.choices(range(len(self.tokens)), weights=probs, k=1)[0]
                        token = self.idx_to_token[chosen_idx]
                    row_tokens.append(token)
                layer_tokens.append(row_tokens)
            tile_tokens.append(layer_tokens)
        
        # Assemble the full cube from tokens
        # Token string length = window_size^3, arranged in depth-major order.
        # We need to reconstruct the cube layer by layer, row by row, cell by cell.
        full_cube = []
        for d in range(tiles_d):
            # For each tile in depth dimension, we have a list of tokens for that depth slice.
            # Actually tile_tokens[d][h][w] is a token for tile at depth d, height h, width w.
            # To build the output cube, we need to extract the characters from each token in order.
            # The token order: first window_size layers in depth, for each depth: window_size rows, for each row: window_size columns.
            # So we can iterate over sub-layers within the tile.
            for sub_d in range(self.window_size):
                layer = []
                for h in range(tiles_h):
                    for sub_h in range(self.window_size):
                        row_chars = []
                        for w in range(tiles_w):
                            token = tile_tokens[d][h][w]
                            # Compute index into token: (sub_d * window_size * window_size) + (sub_h * window_size) + column offset
                            base_idx = sub_d * self.window_size * self.window_size + sub_h * self.window_size
                            row_chars.extend(token[base_idx:base_idx + self.window_size])
                        # crop row to exact width
                        layer.append(''.join(row_chars)[:width])
                full_cube.extend(layer)
        # crop depth to exact depth
        full_cube = full_cube[:depth]
        # Convert list of strings to list of list of strings (list of rows, each row is a string) is sufficient.
        # We return a list of strings per depth layer? Actually the original cube was a 3D list of characters.
        # Here we return a list of layers, each layer is a list of strings (rows). This is compatible with input format.
        # To convert to 3D list of characters, we could split each string into list of chars, but we'll keep as strings for consistency.
        # The input format we used in examples is list of layers, each layer is list of strings.
        return full_cube

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
    print("Spatial Dual Descriptor PM3 - 3D Cube Version")
    print("Optimized with batch processing")
    print("=" * 50)
    
    # Set random seeds
    torch.manual_seed(1)
    random.seed(1)
    np.random.seed(1)
    
    charset = ['A', 'C', 'G', 'T']
    window_size = 2
    vec_dim = 3
    user_step = 2   # step size for nonlinear mode (equal to window_size, so non-overlapping)
    
    # Generate 100 random 3D cubes with random target vectors
    cubes = []
    t_list = []
    cube_depth = 10
    cube_height = 10
    cube_width = 10
    for _ in range(50):
        # Create a 3D cube: list of layers, each layer is list of strings (rows)
        cube = []
        for d in range(cube_depth):
            layer = [''.join(random.choices(charset, k=cube_width)) for _ in range(cube_height)]
            cube.append(layer)
        cubes.append(cube)
        # Random target vector
        t_list.append([random.uniform(-1.0, 1.0) for _ in range(vec_dim)])
    
    # Initialize the model
    dd = SpatialDualDescriptorPM3(
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
    dd.reg_train(cubes, t_list, max_iters=50, tol=1e-19, learning_rate=0.05, decay_rate=0.999, batch_size=2048)
    
    # Predict target vector of the first cube
    acube = cubes[0]
    t_pred = dd.predict_t(acube)
    print(f"\nPredicted t for first cube: {[round(x.item(), 4) for x in t_pred]}")
    
    # Calculate correlation between predicted and real targets
    pred_t_list = [dd.predict_t(cube) for cube in cubes]
    corr_sum = 0.0
    for i in range(dd.m):
        actu_t = [t_vec[i] for t_vec in t_list]
        pred_t = [t_vec[i] for t_vec in pred_t_list]
        corr = correlation(actu_t, pred_t)
        print(f"Dimension {i} prediction correlation: {corr:.4f}")
        corr_sum += corr
    corr_avg = corr_sum / dd.m
    print(f"Average correlation: {corr_avg:.4f}")
    
    # Reconstruct representative cubes
    rec_det = dd.reconstruct(depth=cube_depth, height=cube_height, width=cube_width, tau=0.0)
    rec_rand = dd.reconstruct(depth=cube_depth, height=cube_height, width=cube_width, tau=0.5)
    print("\nDeterministic Reconstruction (first depth layer, first 2 rows):")
    for row in rec_det[0][:2]:
        print(row[:50] + "...")
    print("\nStochastic Reconstruction (tau=0.5, first depth layer, first 2 rows):")
    for row in rec_rand[0][:2]:
        print(row[:50] + "...")
    
    # === Classification Task ===
    print("\n" + "=" * 50)
    print("Classification Task")
    print("=" * 50)
    
    num_classes = 3
    class_cubes = []
    class_labels = []
    
    # Create cubes with different patterns
    for class_id in range(num_classes):
        for _ in range(50):  # 50 cubes per class
            if class_id == 0:
                # Class 0: High A content
                row_probs = [0.6, 0.1, 0.1, 0.2]
            elif class_id == 1:
                # Class 1: High GC content
                row_probs = [0.1, 0.4, 0.4, 0.1]
            else:
                # Class 2: Balanced
                row_probs = [0.25, 0.25, 0.25, 0.25]
            cube = []
            for d in range(cube_depth):
                layer = [''.join(random.choices(charset, weights=row_probs, k=cube_width)) for _ in range(cube_height)]
                cube.append(layer)
            class_cubes.append(cube)
            class_labels.append(class_id)
    
    # Initialize new model for classification
    dd_cls = SpatialDualDescriptorPM3(
        charset,
        window_size=window_size,
        vec_dim=vec_dim,
        mode='nonlinear',
        user_step=user_step,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    print("\nStarting Classification Training")
    dd_cls.cls_train(class_cubes, class_labels, num_classes,
                     max_iters=10, tol=1e-8, learning_rate=0.05,
                     decay_rate=0.99, batch_size=32, print_every=1)
    
    # Evaluate on training set
    correct = 0
    for cube, true_label in zip(class_cubes, class_labels):
        pred_class, _ = dd_cls.predict_c(cube)
        if pred_class == true_label:
            correct += 1
    accuracy = correct / len(class_cubes)
    print(f"\nClassification Accuracy: {accuracy:.4f} ({correct}/{len(class_cubes)})")
    
    # === Multi-label Classification ===
    print("\n" + "=" * 50)
    print("Multi-Label Classification")
    print("=" * 50)
    
    num_labels = 4
    label_cubes = []
    labels = []
    for _ in range(50):
        cube = []
        for d in range(cube_depth):
            layer = [''.join(random.choices(charset, k=cube_width)) for _ in range(cube_height)]
            cube.append(layer)
        label_cubes.append(cube)
        # Random binary labels
        label_vec = [1 if random.random() > 0.7 else 0 for _ in range(num_labels)]
        labels.append(label_vec)
    
    dd_lbl = SpatialDualDescriptorPM3(
        charset,
        window_size=window_size,
        vec_dim=vec_dim,
        mode='nonlinear',
        user_step=user_step,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    print("\nStarting Multi-Label Training")
    loss_hist, acc_hist = dd_lbl.lbl_train(
        label_cubes, labels, num_labels,
        max_iters=50, tol=1e-16, learning_rate=0.05,
        decay_rate=0.99, print_every=10, batch_size=32
    )
    print(f"\nFinal training loss: {loss_hist[-1]:.6f}")
    print(f"Final training accuracy: {acc_hist[-1]:.4f}")
    
    # Example prediction on a test cube
    test_cube = []
    for d in range(cube_depth):
        layer = [''.join(random.choices(charset, k=cube_width)) for _ in range(cube_height)]
        test_cube.append(layer)
    binary_pred, prob_pred = dd_lbl.predict_l(test_cube, threshold=0.5)
    print("\nExample multi-label prediction:")
    print(f"Binary predictions: {binary_pred}")
    print(f"Probabilities: {[f'{p:.4f}' for p in prob_pred]}")
    
    # === Self-training ===
    print("\n" + "=" * 50)
    print("Self-Training Example")
    print("=" * 50)
    
    dd_self = SpatialDualDescriptorPM3(
        charset,
        window_size=window_size,
        vec_dim=vec_dim,
        mode='nonlinear',
        user_step=user_step,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Generate self-training cubes
    self_cubes = []
    for _ in range(10):
        cube = []
        for d in range(cube_depth):
            layer = [''.join(random.choices(charset, k=cube_width)) for _ in range(cube_height)]
            cube.append(layer)
        self_cubes.append(cube)
    
    print("Training for self-consistency:")
    self_hist = dd_self.self_train(self_cubes, max_iters=50, tol=1e-8, learning_rate=0.01, batch_size=1024)
    
    rec_cube = dd_self.reconstruct(depth=cube_depth, height=cube_height, width=cube_width, tau=0.2)
    print("\nReconstructed cube (first depth layer, first 2 rows):")
    for row in rec_cube[0][:2]:
        print(row[:50] + "...")
    
    print("\nAll tests completed successfully!")
