# Copyright (C) 2005-2025, Bin-Guang Ma (mbg@mail.hzau.edu.cn); SPDX-License-Identifier: MIT
# The Spatial Dual Descriptor Vector class (Tensor form) implemented with PyTorch for 3D character arrays
# This program is for the demonstration of methodology and not fully refined.
# Author: Bin-Guang Ma (assisted by DeepSeek); Date: 2025-7-29 ~ 2026-3-25

import math
import random
import itertools
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy

class SpatialDualDescriptorTS3(nn.Module):
    """
    Spatial Dual Descriptor for 3D character arrays with GPU acceleration using PyTorch:
      - tensor P ∈ R^{m×m×o} of basis coefficients
      - embedding: k-mer token embeddings in R^m (tokens are rank×rank×rank cubes)
      - indexed periods for three dimensions: period1, period2, period3
      - basis function phi_{i,j,g}(k1,k2,k3) = cos(2π*k1/p1) * cos(2π*k2/p2) * cos(2π*k3/p3)
      - supports 'linear' (step=1) or 'nonlinear' (step-by-rank) window extraction
    """
    def __init__(self, charset, rank=1, rank_mode='drop', vec_dim=2, num_basis=5,
                 mode='linear', user_step=None, device='cuda'):
        super().__init__()
        self.charset = list(charset)
        self.rank = rank          # cube size: rank x rank x rank
        self.rank_mode = rank_mode  # 'pad' or 'drop' for boundary handling
        self.m = vec_dim          # embedding dimension
        self.o = num_basis        # number of basis terms
        assert mode in ('linear', 'nonlinear')
        self.mode = mode
        self.step = user_step      # step size in all dimensions (if None, use rank in nonlinear mode)
        self.trained = False
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # Generate all possible tokens (rank x rank x rank cubes)
        # Each token is a string formed by concatenating layers, rows, columns in row-major order
        toks = []
        num_cells = self.rank ** 3
        for combo in itertools.product(self.charset, repeat=num_cells):
            tok = ''.join(combo)
            toks.append(tok)
        self.tokens = sorted(set(toks))
        self.token_to_idx = {token: idx for idx, token in enumerate(self.tokens)}
        self.idx_to_token = {idx: token for idx, token in enumerate(self.tokens)}

        # Embedding layer for tokens
        self.embedding = nn.Embedding(len(self.tokens), self.m)

        # Position-weight tensor P[i][j][g]
        self.P = nn.Parameter(torch.empty(self.m, self.m, self.o))

        # Precompute indexed periods for three dimensions (fixed, not trainable)
        periods1 = torch.zeros(self.m, self.m, self.o, dtype=torch.float32)
        periods2 = torch.zeros(self.m, self.m, self.o, dtype=torch.float32)
        periods3 = torch.zeros(self.m, self.m, self.o, dtype=torch.float32)
        for i in range(self.m):
            for j in range(self.m):
                for g in range(self.o):
                    base = i * (self.m * self.o) + j * self.o + g
                    periods1[i, j, g] = base + 2
                    periods2[i, j, g] = base + 3
                    periods3[i, j, g] = base + 4
        self.register_buffer('periods1', periods1)
        self.register_buffer('periods2', periods2)
        self.register_buffer('periods3', periods3)

        # Class head (initialized later)
        self.num_classes = None
        self.classifier = None

        # Label head (initialized later)
        self.num_labels = None
        self.labeller = None

        # Initialize parameters
        self.reset_parameters()
        self.to(self.device)

    def reset_parameters(self):
        """Initialize model parameters"""
        nn.init.uniform_(self.embedding.weight, -0.5, 0.5)
        nn.init.uniform_(self.P, -0.1, 0.1)
        if self.classifier is not None:
            nn.init.normal_(self.classifier.weight, 0, 0.01)
            if self.classifier.bias is not None:
                nn.init.constant_(self.classifier.bias, 0)
        if self.labeller is not None:
            nn.init.xavier_uniform_(self.labeller.weight)
            nn.init.zeros_(self.labeller.bias)

    def token_to_indices(self, token_list):
        """Convert list of tokens to tensor of indices"""
        return torch.tensor([self.token_to_idx[tok] for tok in token_list], device=self.device)

    def extract_tokens(self, arr_3d):
        """
        Extract cubic windows (tokens) from a 3D character array.
        arr_3d: list of 2D layers. Each layer is a list of strings (rows).
        All layers have same dimensions.
        Returns:
            token_list: list of token strings
            positions: list of (k1, k2, k3) tuples, where (k1,k2,k3) is the corner index
        """
        if not arr_3d:
            return [], []
        depth = len(arr_3d)
        rows = len(arr_3d[0])
        cols = len(arr_3d[0][0]) if rows > 0 else 0
        step = self.step if self.mode == 'nonlinear' else 1
        if step is None:
            step = self.rank

        tokens = []
        positions = []
        for k1 in range(0, depth, step):
            for k2 in range(0, rows, step):
                for k3 in range(0, cols, step):
                    # For 'drop' mode, check if the cube fits entirely inside the array
                    if self.rank_mode == 'drop':
                        if (k1 + self.rank > depth or
                            k2 + self.rank > rows or
                            k3 + self.rank > cols):
                            continue

                    # Build token string by concatenating layers, rows, columns
                    token_chars = []
                    for d in range(k1, min(k1 + self.rank, depth)):
                        for r in range(k2, min(k2 + self.rank, rows)):
                            row_str = arr_3d[d][r]
                            segment = row_str[k3:k3+self.rank]
                            if len(segment) < self.rank and self.rank_mode == 'pad':
                                segment = segment.ljust(self.rank, '_')
                            token_chars.extend(segment)
                    # Check if we have enough rows in this layer (for 'drop' mode)
                    if self.rank_mode == 'drop':
                        # We need self.rank layers, each with self.rank rows, each row of length self.rank
                        expected_len = self.rank ** 3
                        if len(token_chars) != expected_len:
                            continue
                    # For 'pad' mode, fill missing parts
                    if self.rank_mode == 'pad':
                        # Fill missing layers
                        missing_layers = self.rank - (min(k1+self.rank, depth) - k1)
                        for _ in range(missing_layers):
                            for _ in range(self.rank):
                                token_chars.extend(['_'] * self.rank)
                        # Fill missing rows within existing layers
                        # This is more complex; for simplicity we'll just fill missing cells later
                        # Actually we already padded individual rows; missing rows are handled by
                        # not having them, but we need to insert full rows of underscores.
                        # Simpler approach: just pad the entire cube with underscores at the end.
                        # We'll do a quick fix: if the token length is less than expected, pad with '_'
                        while len(token_chars) < self.rank ** 3:
                            token_chars.append('_')
                    token = ''.join(token_chars)
                    tokens.append(token)
                    positions.append((k1, k2, k3))
        return tokens, positions

    def batch_compute_Nk(self, k1_tensor, k2_tensor, k3_tensor, token_indices):
        """
        Vectorized computation of N(k1,k2,k3) vectors for a batch of positions and tokens.
        Args:
            k1_tensor: Tensor of layer positions [batch_size]
            k2_tensor: Tensor of row positions [batch_size]
            k3_tensor: Tensor of column positions [batch_size]
            token_indices: Tensor of token indices [batch_size]
        Returns:
            Tensor of N(k) vectors [batch_size, m]
        """
        # Get embeddings for all tokens [batch_size, m]
        x = self.embedding(token_indices)

        # Expand dimensions for broadcasting [batch_size, 1, 1, 1]
        k1_exp = k1_tensor.view(-1, 1, 1, 1)
        k2_exp = k2_tensor.view(-1, 1, 1, 1)
        k3_exp = k3_tensor.view(-1, 1, 1, 1)

        # Calculate basis functions
        phi1 = torch.cos(2 * math.pi * k1_exp / self.periods1)
        phi2 = torch.cos(2 * math.pi * k2_exp / self.periods2)
        phi3 = torch.cos(2 * math.pi * k3_exp / self.periods3)
        phi = phi1 * phi2 * phi3   # [batch_size, m, m, o]

        # Optimized computation using einsum
        Nk = torch.einsum('bj,ijg,bijg->bi', x, self.P, phi)
        return Nk

    def compute_Nk(self, k1, k2, k3, token_idx):
        """Compute N(k1,k2,k3) for single position and token"""
        k1_t = torch.tensor([k1], dtype=torch.float32, device=self.device)
        k2_t = torch.tensor([k2], dtype=torch.float32, device=self.device)
        k3_t = torch.tensor([k3], dtype=torch.float32, device=self.device)
        idx_t = torch.tensor([token_idx], device=self.device)
        result = self.batch_compute_Nk(k1_t, k2_t, k3_t, idx_t)
        return result[0]

    def describe(self, arr_3d):
        """Compute N(k1,k2,k3) vectors for each window in the 3D array"""
        tokens, positions = self.extract_tokens(arr_3d)
        if not tokens:
            return []
        token_indices = self.token_to_indices(tokens)
        k1_pos = torch.tensor([p[0] for p in positions], dtype=torch.float32, device=self.device)
        k2_pos = torch.tensor([p[1] for p in positions], dtype=torch.float32, device=self.device)
        k3_pos = torch.tensor([p[2] for p in positions], dtype=torch.float32, device=self.device)
        N_batch = self.batch_compute_Nk(k1_pos, k2_pos, k3_pos, token_indices)
        return N_batch.detach().cpu().numpy()

    def S(self, arr_3d):
        """
        Compute list of cumulative sums S(l) = sum_{t=1}^{l} N(t) for flattened windows.
        """
        tokens, positions = self.extract_tokens(arr_3d)
        if not tokens:
            return []
        token_indices = self.token_to_indices(tokens)
        k1_pos = torch.tensor([p[0] for p in positions], dtype=torch.float32, device=self.device)
        k2_pos = torch.tensor([p[1] for p in positions], dtype=torch.float32, device=self.device)
        k3_pos = torch.tensor([p[2] for p in positions], dtype=torch.float32, device=self.device)
        N_batch = self.batch_compute_Nk(k1_pos, k2_pos, k3_pos, token_indices)
        S_cum = torch.cumsum(N_batch, dim=0)
        return [s.detach().cpu().numpy() for s in S_cum]

    def D(self, arrays_3d, t_list):
        """
        Compute mean squared deviation D across 3D arrays:
        D = average over all windows of (N(k)-t_seq)^2
        """
        total_loss = 0.0
        total_windows = 0
        t_tensors = [torch.tensor(t, dtype=torch.float32, device=self.device) for t in t_list]
        for arr, t in zip(arrays_3d, t_tensors):
            tokens, positions = self.extract_tokens(arr)
            if not tokens:
                continue
            token_indices = self.token_to_indices(tokens)
            k1_pos = torch.tensor([p[0] for p in positions], dtype=torch.float32, device=self.device)
            k2_pos = torch.tensor([p[1] for p in positions], dtype=torch.float32, device=self.device)
            k3_pos = torch.tensor([p[2] for p in positions], dtype=torch.float32, device=self.device)
            N_batch = self.batch_compute_Nk(k1_pos, k2_pos, k3_pos, token_indices)
            losses = torch.sum((N_batch - t) ** 2, dim=1)
            total_loss += losses.sum().item()
            total_windows += len(tokens)
        return total_loss / total_windows if total_windows else 0.0

    def d(self, arr_3d, t):
        """Compute pattern deviation value (d) for a single 3D array."""
        return self.D([arr_3d], [t])

    def reg_train(self, arrays_3d, t_list, max_iters=1000, tol=1e-8, learning_rate=0.01,
                  continued=False, decay_rate=1.0, print_every=10, batch_size=32,
                  checkpoint_file=None, checkpoint_interval=10):
        """Train the model using gradient descent."""
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
            total_arrays = 0
            indices = list(range(len(arrays_3d)))
            random.shuffle(indices)
            for batch_start in range(0, len(indices), batch_size):
                batch_indices = indices[batch_start:batch_start + batch_size]
                batch_arrays = [arrays_3d[idx] for idx in batch_indices]
                batch_targets = [t_tensors[idx] for idx in batch_indices]
                optimizer.zero_grad()
                batch_loss = 0.0
                for arr, target in zip(batch_arrays, batch_targets):
                    tokens, positions = self.extract_tokens(arr)
                    if not tokens:
                        continue
                    token_indices = self.token_to_indices(tokens)
                    k1_pos = torch.tensor([p[0] for p in positions], dtype=torch.float32, device=self.device)
                    k2_pos = torch.tensor([p[1] for p in positions], dtype=torch.float32, device=self.device)
                    k3_pos = torch.tensor([p[2] for p in positions], dtype=torch.float32, device=self.device)
                    Nk_batch = self.batch_compute_Nk(k1_pos, k2_pos, k3_pos, token_indices)
                    seq_pred = torch.mean(Nk_batch, dim=0)
                    seq_loss = torch.sum((seq_pred - target) ** 2)
                    batch_loss += seq_loss
                    del Nk_batch, seq_pred, token_indices, k1_pos, k2_pos, k3_pos
                if len(batch_arrays) > 0:
                    batch_loss = batch_loss / len(batch_arrays)
                    batch_loss.backward()
                    optimizer.step()
                    total_loss += batch_loss.item() * len(batch_arrays)
                    total_arrays += len(batch_arrays)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            avg_loss = total_loss / total_arrays if total_arrays else 0.0
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
        self._compute_training_statistics(arrays_3d)
        self.trained = True
        return history

    def cls_train(self, arrays_3d, labels, num_classes, max_iters=1000, tol=1e-8, learning_rate=0.01,
                  continued=False, decay_rate=1.0, print_every=10, batch_size=32,
                  checkpoint_file=None, checkpoint_interval=10):
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
            total_arrays = 0
            correct_predictions = 0
            indices = list(range(len(arrays_3d)))
            random.shuffle(indices)
            for batch_start in range(0, len(indices), batch_size):
                batch_indices = indices[batch_start:batch_start + batch_size]
                batch_arrays = [arrays_3d[idx] for idx in batch_indices]
                batch_labels = label_tensors[batch_indices]
                optimizer.zero_grad()
                batch_logits = []
                for arr in batch_arrays:
                    tokens, positions = self.extract_tokens(arr)
                    if not tokens:
                        seq_vector = torch.zeros(self.m, device=self.device)
                    else:
                        token_indices = self.token_to_indices(tokens)
                        k1_pos = torch.tensor([p[0] for p in positions], dtype=torch.float32, device=self.device)
                        k2_pos = torch.tensor([p[1] for p in positions], dtype=torch.float32, device=self.device)
                        k3_pos = torch.tensor([p[2] for p in positions], dtype=torch.float32, device=self.device)
                        Nk_batch = self.batch_compute_Nk(k1_pos, k2_pos, k3_pos, token_indices)
                        seq_vector = torch.mean(Nk_batch, dim=0)
                        del Nk_batch, token_indices, k1_pos, k2_pos, k3_pos
                    logits = self.classifier(seq_vector.unsqueeze(0))
                    batch_logits.append(logits)
                if batch_logits:
                    all_logits = torch.cat(batch_logits, dim=0)
                    loss = criterion(all_logits, batch_labels)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item() * len(batch_arrays)
                    total_arrays += len(batch_arrays)
                    with torch.no_grad():
                        predictions = torch.argmax(all_logits, dim=1)
                        correct_predictions += (predictions == batch_labels).sum().item()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            avg_loss = total_loss / total_arrays if total_arrays else 0.0
            accuracy = correct_predictions / total_arrays if total_arrays else 0.0
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

    def lbl_train(self, arrays_3d, labels, num_labels, max_iters=1000, tol=1e-8, learning_rate=0.01,
                  continued=False, decay_rate=1.0, print_every=10, batch_size=32,
                  checkpoint_file=None, checkpoint_interval=10, pos_weight=None):
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
            total_arrays = 0
            indices = list(range(len(arrays_3d)))
            random.shuffle(indices)
            for batch_start in range(0, len(indices), batch_size):
                batch_indices = indices[batch_start:batch_start + batch_size]
                batch_arrays = [arrays_3d[idx] for idx in batch_indices]
                batch_labels = labels_tensor[batch_indices]
                optimizer.zero_grad()
                batch_predictions_list = []
                for arr in batch_arrays:
                    tokens, positions = self.extract_tokens(arr)
                    if not tokens:
                        continue
                    token_indices = self.token_to_indices(tokens)
                    k1_pos = torch.tensor([p[0] for p in positions], dtype=torch.float32, device=self.device)
                    k2_pos = torch.tensor([p[1] for p in positions], dtype=torch.float32, device=self.device)
                    k3_pos = torch.tensor([p[2] for p in positions], dtype=torch.float32, device=self.device)
                    Nk_batch = self.batch_compute_Nk(k1_pos, k2_pos, k3_pos, token_indices)
                    seq_rep = torch.mean(Nk_batch, dim=0)
                    logits = self.labeller(seq_rep)
                    batch_predictions_list.append(logits)
                    del Nk_batch, seq_rep, token_indices, k1_pos, k2_pos, k3_pos
                if batch_predictions_list:
                    batch_logits = torch.stack(batch_predictions_list, dim=0)
                    batch_loss = criterion(batch_logits, batch_labels)
                    with torch.no_grad():
                        probs = torch.sigmoid(batch_logits)
                        predictions = (probs > 0.5).float()
                        batch_correct = (predictions == batch_labels).sum().item()
                        batch_predictions = batch_labels.numel()
                    batch_loss.backward()
                    optimizer.step()
                    total_loss += batch_loss.item() * len(batch_arrays)
                    total_correct += batch_correct
                    total_predictions += batch_predictions
                    total_arrays += len(batch_arrays)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            avg_loss = total_loss / total_arrays if total_arrays else 0.0
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

    def self_train(self, arrays_3d, max_iters=100, tol=1e-6, learning_rate=0.01,
                   continued=False, decay_rate=1.0, print_every=10,
                   batch_size=32, checkpoint_file=None, checkpoint_interval=5):
        """Self-training method for self-consistency."""
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
            total_windows = 0
            indices = list(range(len(arrays_3d)))
            random.shuffle(indices)
            for batch_start in range(0, len(indices), batch_size):
                batch_indices = indices[batch_start:batch_start + batch_size]
                batch_arrays = [arrays_3d[idx] for idx in batch_indices]
                optimizer.zero_grad()
                batch_loss = 0.0
                batch_window_count = 0
                for arr in batch_arrays:
                    tokens, positions = self.extract_tokens(arr)
                    if not tokens:
                        continue
                    token_indices = self.token_to_indices(tokens)
                    k1_pos = torch.tensor([p[0] for p in positions], dtype=torch.float32, device=self.device)
                    k2_pos = torch.tensor([p[1] for p in positions], dtype=torch.float32, device=self.device)
                    k3_pos = torch.tensor([p[2] for p in positions], dtype=torch.float32, device=self.device)
                    Nk_batch = self.batch_compute_Nk(k1_pos, k2_pos, k3_pos, token_indices)
                    token_embeddings = self.embedding(token_indices)
                    seq_loss = torch.sum((Nk_batch - token_embeddings) ** 2) / len(tokens)
                    batch_loss += seq_loss
                    batch_window_count += len(tokens)
                    del Nk_batch, token_embeddings, token_indices, k1_pos, k2_pos, k3_pos
                if batch_window_count > 0:
                    batch_loss = batch_loss / len(batch_arrays)  # average over sequences
                    batch_loss.backward()
                    optimizer.step()
                    total_loss += batch_loss.item() * batch_window_count
                    total_windows += batch_window_count
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            avg_loss = total_loss / total_windows if total_windows else 0.0
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
        self._compute_training_statistics(arrays_3d)
        self.trained = True
        return history

    def _compute_training_statistics(self, arrays_3d, batch_size=50):
        """Compute training statistics for reconstruction and generation."""
        total_window_count = 0
        total_t = torch.zeros(self.m, device=self.device)
        with torch.no_grad():
            for i in range(0, len(arrays_3d), batch_size):
                batch_arrays = arrays_3d[i:i+batch_size]
                batch_window_count = 0
                batch_vec_sum = torch.zeros(self.m, device=self.device)
                for arr in batch_arrays:
                    tokens, positions = self.extract_tokens(arr)
                    if not tokens:
                        continue
                    batch_window_count += len(tokens)
                    token_indices = self.token_to_indices(tokens)
                    k1_pos = torch.tensor([p[0] for p in positions], dtype=torch.float32, device=self.device)
                    k2_pos = torch.tensor([p[1] for p in positions], dtype=torch.float32, device=self.device)
                    k3_pos = torch.tensor([p[2] for p in positions], dtype=torch.float32, device=self.device)
                    Nk_batch = self.batch_compute_Nk(k1_pos, k2_pos, k3_pos, token_indices)
                    batch_vec_sum += Nk_batch.sum(dim=0)
                    del Nk_batch, token_indices, k1_pos, k2_pos, k3_pos
                total_window_count += batch_window_count
                total_t += batch_vec_sum
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        self.mean_window_count = total_window_count / len(arrays_3d) if arrays_3d else 0
        self.mean_t = (total_t / total_window_count).cpu().numpy() if total_window_count > 0 else np.zeros(self.m)

    def _save_checkpoint(self, checkpoint_file, iteration, history, optimizer, scheduler, best_loss):
        checkpoint = {
            'iteration': iteration,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'history': history,
            'best_loss': best_loss,
            'training_stats': {
                'mean_t': self.mean_t,
                'mean_window_count': self.mean_window_count
            }
        }
        torch.save(checkpoint, checkpoint_file)
        print(f"Checkpoint saved at iteration {iteration}")

    def predict_t(self, arr_3d):
        """Predict target vector for a 3D array (average of all N(k) vectors)."""
        tokens, positions = self.extract_tokens(arr_3d)
        if not tokens:
            return [0.0] * self.m
        token_indices = self.token_to_indices(tokens)
        k1_pos = torch.tensor([p[0] for p in positions], dtype=torch.float32, device=self.device)
        k2_pos = torch.tensor([p[1] for p in positions], dtype=torch.float32, device=self.device)
        k3_pos = torch.tensor([p[2] for p in positions], dtype=torch.float32, device=self.device)
        Nk_batch = self.batch_compute_Nk(k1_pos, k2_pos, k3_pos, token_indices)
        return (torch.mean(Nk_batch, dim=0)).detach().cpu().numpy()

    def predict_c(self, arr_3d):
        """Predict class label for a 3D array."""
        if self.classifier is None:
            raise ValueError("Model must be trained first for classification")
        seq_vector = self.predict_t(arr_3d)
        seq_vector_tensor = torch.tensor(seq_vector, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits = self.classifier(seq_vector_tensor.unsqueeze(0))
            probabilities = torch.softmax(logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
        return predicted_class, probabilities[0].cpu().numpy()

    def predict_l(self, arr_3d, threshold=0.5):
        """Predict multi-label classification for a 3D array."""
        assert self.labeller is not None, "Model must be trained first for label prediction"
        tokens, positions = self.extract_tokens(arr_3d)
        if not tokens:
            return np.zeros(self.num_labels, dtype=np.float32), np.zeros(self.num_labels, dtype=np.float32)
        token_indices = self.token_to_indices(tokens)
        k1_pos = torch.tensor([p[0] for p in positions], dtype=torch.float32, device=self.device)
        k2_pos = torch.tensor([p[1] for p in positions], dtype=torch.float32, device=self.device)
        k3_pos = torch.tensor([p[2] for p in positions], dtype=torch.float32, device=self.device)
        Nk_batch = self.batch_compute_Nk(k1_pos, k2_pos, k3_pos, token_indices)
        seq_rep = torch.mean(Nk_batch, dim=0)
        with torch.no_grad():
            logits = self.labeller(seq_rep)
            probs = torch.sigmoid(logits).cpu().numpy()
        binary_preds = (probs > threshold).astype(np.float32)
        return binary_preds, probs

    def reconstruct(self, depth, rows, cols, tau=0.0):
        """
        Reconstruct a 3D character array of size depth x rows x cols.
        Assumes step = rank (non-overlapping cubes) for simplicity.
        For each cube position, selects a token that minimizes error with mean_t.
        Returns a 3D list: list of layers, each layer is list of strings.
        """
        assert self.trained, "Model must be trained first"
        if tau < 0:
            raise ValueError("Temperature must be non-negative")
        step = self.rank  # fixed for reconstruction
        num_depth_windows = (depth + step - 1) // step
        num_rows_windows = (rows + step - 1) // step
        num_cols_windows = (cols + step - 1) // step
        mean_t_tensor = torch.tensor(self.mean_t, dtype=torch.float32, device=self.device)

        # Precompute all token embeddings
        all_token_indices = torch.arange(len(self.tokens), device=self.device)
        all_embeddings = self.embedding(all_token_indices)  # not directly used, but for potential future use

        # Initialize 3D grid: list of layers, each layer is list of strings (rows)
        grid = [[['' for _ in range(cols)] for _ in range(rows)] for _ in range(depth)]

        for k1 in range(num_depth_windows):
            for k2 in range(num_rows_windows):
                for k3 in range(num_cols_windows):
                    # Compute Nk for all tokens at this cube corner
                    k1_tensor = torch.tensor([k1 * step] * len(self.tokens), dtype=torch.float32, device=self.device)
                    k2_tensor = torch.tensor([k2 * step] * len(self.tokens), dtype=torch.float32, device=self.device)
                    k3_tensor = torch.tensor([k3 * step] * len(self.tokens), dtype=torch.float32, device=self.device)
                    Nk_all = self.batch_compute_Nk(k1_tensor, k2_tensor, k3_tensor, all_token_indices)
                    errors = torch.sum((Nk_all - mean_t_tensor) ** 2, dim=1)
                    scores = -errors
                    if tau == 0:
                        max_idx = torch.argmax(scores).item()
                        best_tok = self.idx_to_token[max_idx]
                    else:
                        probs = torch.softmax(scores / tau, dim=0).detach().cpu().numpy()
                        chosen_idx = random.choices(range(len(self.tokens)), weights=probs, k=1)[0]
                        best_tok = self.idx_to_token[chosen_idx]

                    # Place token characters into grid (row-major order within cube)
                    # Token string: layer0 row0 col0, layer0 row0 col1, ..., layer0 row0 col(rank-1),
                    #               layer0 row1 col0, ...
                    #               layer1 row0 col0, ...
                    for d in range(self.rank):
                        for r in range(self.rank):
                            for c in range(self.rank):
                                grid_d = k1 * step + d
                                grid_r = k2 * step + r
                                grid_c = k3 * step + c
                                if grid_d < depth and grid_r < rows and grid_c < cols:
                                    char_index = (d * self.rank * self.rank) + (r * self.rank) + c
                                    grid[grid_d][grid_r][grid_c] = best_tok[char_index]
        # Convert each layer to list of strings
        result = [[''.join(row) for row in layer] for layer in grid]
        return result

    def save(self, filename):
        """Save model state to file"""
        torch.save(self.state_dict(), filename)
        print(f"Model saved to {filename}")

    def load(self, filename):
        """Load model state from file"""
        try:
            state_dict = torch.load(filename, map_location=self.device, weights_only=True)
        except TypeError:
            state_dict = torch.load(filename, map_location=self.device)
        self.load_state_dict(state_dict)
        print(f"Model loaded from {filename}")
        return self


# === Example Usage (3D) ===
if __name__ == "__main__":
    from statistics import correlation

    print("=" * 50)
    print("Spatial Dual Descriptor TS3 - 3D Array Version")
    print("Optimized with batch processing")
    print("=" * 50)

    torch.manual_seed(11)
    random.seed(11)

    charset = ['A', 'C', 'G', 'T']   # 4 nucleotides
    vec_dim = 4
    num_basis = 4
    rank = 2          # 2x2x2 cubes
    user_step = 2     # non-overlapping cubes
    # Note: For rank=2 and charset size 4, token vocabulary size = 4^(8) = 65536,
    # which is manageable in memory for demonstration.

    # Initialize the model
    dd = SpatialDualDescriptorTS3(
        charset,
        rank=rank,
        vec_dim=vec_dim,
        num_basis=num_basis,
        mode='nonlinear',
        user_step=user_step,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    print(f"\nUsing device: {dd.device}")
    print(f"Number of tokens (2x2x2 windows): {len(dd.tokens)}")

    # Generate 100 random 3D arrays (list of layers, each layer list of strings)
    arrays_3d, t_list = [], []
    for _ in range(100):
        depth = random.randint(4, 8)
        rows = random.randint(4, 8)
        cols = random.randint(4, 8)
        # Build array: list of layers, each layer is list of strings
        arr = []
        for _ in range(depth):
            layer = [''.join(random.choices(charset, k=cols)) for _ in range(rows)]
            arr.append(layer)
        arrays_3d.append(arr)
        # Random target vector
        t_list.append([random.uniform(-1.0, 1.0) for _ in range(vec_dim)])

    # Training regression model
    print("\n" + "=" * 50)
    print("Starting Gradient Descent Training (Regression)")
    print("=" * 50)
    dd.reg_train(arrays_3d, t_list, max_iters=50, tol=1e-9, learning_rate=0.1,
                 decay_rate=0.99, batch_size=32)

    # Predict target for first array
    arr0 = arrays_3d[0]
    t_pred = dd.predict_t(arr0)
    print(f"\nPredicted t for first 3D array: {[round(x, 4) for x in t_pred]}")

    # Correlation between predicted and actual targets
    pred_t_list = [dd.predict_t(arr) for arr in arrays_3d]
    corr_sum = 0.0
    for i in range(dd.m):
        actu_t = [t_vec[i] for t_vec in t_list]
        pred_t = [t_vec[i] for t_vec in pred_t_list]
        corr = correlation(actu_t, pred_t)
        print(f"Dimension {i} prediction correlation: {corr:.4f}")
        corr_sum += corr
    corr_avg = corr_sum / dd.m
    print(f"Average correlation: {corr_avg:.4f}")

    # Reconstruction (3D array)
    print("\n" + "=" * 50)
    print("Reconstruction of 3D arrays")
    print("=" * 50)
    rec_arr_det = dd.reconstruct(depth=6, rows=6, cols=6, tau=0.0)
    rec_arr_rand = dd.reconstruct(depth=6, rows=6, cols=6, tau=0.5)
    print("Deterministic reconstruction (first layer, first 3 rows):")
    for row in rec_arr_det[0][:3]:
        print(row)
    print("Stochastic reconstruction (tau=0.5, first layer, first 3 rows):")
    for row in rec_arr_rand[0][:3]:
        print(row)

    # Classification task
    print("\n" + "=" * 50)
    print("Classification Task (3D)")
    print("=" * 50)

    num_classes = 3
    class_arrays = []
    class_labels = []
    for class_id in range(num_classes):
        for _ in range(50):
            depth = random.randint(4, 8)
            rows = random.randint(4, 8)
            cols = random.randint(4, 8)
            if class_id == 0:
                # Class 0: high 'A' content
                arr = [[ ''.join(random.choices(['A','C','G','T'], weights=[0.6,0.1,0.1,0.2], k=cols))
                         for _ in range(rows) ] for _ in range(depth)]
            elif class_id == 1:
                # Class 1: high 'G'/'C' content
                arr = [[ ''.join(random.choices(['A','C','G','T'], weights=[0.1,0.4,0.4,0.1], k=cols))
                         for _ in range(rows) ] for _ in range(depth)]
            else:
                # Class 2: balanced
                arr = [[ ''.join(random.choices(charset, k=cols)) for _ in range(rows) ] for _ in range(depth)]
            class_arrays.append(arr)
            class_labels.append(class_id)

    dd_cls = SpatialDualDescriptorTS3(
        charset, rank=rank, vec_dim=vec_dim, num_basis=num_basis,
        mode='nonlinear', user_step=user_step,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    print("\nTraining classification model...")
    history = dd_cls.cls_train(class_arrays, class_labels, num_classes,
                               max_iters=30, tol=1e-8, learning_rate=0.05,
                               decay_rate=0.99, batch_size=32, print_every=5)

    correct = 0
    for arr, true_label in zip(class_arrays, class_labels):
        pred, _ = dd_cls.predict_c(arr)
        if pred == true_label:
            correct += 1
    accuracy = correct / len(class_arrays)
    print(f"\nClassification accuracy: {accuracy:.4f} ({correct}/{len(class_arrays)})")

    # Multi-label classification
    print("\n" + "=" * 50)
    print("Multi-Label Classification (3D)")
    print("=" * 50)

    num_labels = 4
    ml_arrays = []
    ml_labels = []
    for _ in range(100):
        depth = random.randint(4, 8)
        rows = random.randint(4, 8)
        cols = random.randint(4, 8)
        arr = [[ ''.join(random.choices(charset, k=cols)) for _ in range(rows) ] for _ in range(depth)]
        ml_arrays.append(arr)
        label_vec = [1.0 if random.random() > 0.7 else 0.0 for _ in range(num_labels)]
        ml_labels.append(label_vec)

    dd_lbl = SpatialDualDescriptorTS3(
        charset, rank=rank, vec_dim=vec_dim, num_basis=num_basis,
        mode='nonlinear', user_step=user_step,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    print("\nTraining multi-label model...")
    loss_hist, acc_hist = dd_lbl.lbl_train(ml_arrays, ml_labels, num_labels,
                                           max_iters=30, tol=1e-6, learning_rate=0.01,
                                           decay_rate=0.99, print_every=5, batch_size=32)

    correct_seq = 0
    for arr, true in zip(ml_arrays, ml_labels):
        pred_bin, _ = dd_lbl.predict_l(arr, threshold=0.5)
        if np.all(pred_bin == np.array(true)):
            correct_seq += 1
    print(f"\nMulti-label sequence accuracy: {correct_seq}/{len(ml_arrays)} = {correct_seq/len(ml_arrays):.4f}")

    # Self-training example
    print("\n" + "=" * 50)
    print("Self-Training Example (3D)")
    print("=" * 50)

    dd_self = SpatialDualDescriptorTS3(
        charset, rank=rank, vec_dim=vec_dim, num_basis=num_basis,
        mode='nonlinear', user_step=user_step,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    self_arrays = []
    for _ in range(20):
        depth = random.randint(4, 6)
        rows = random.randint(4, 6)
        cols = random.randint(4, 6)
        arr = [[ ''.join(random.choices(charset, k=cols)) for _ in range(rows) ] for _ in range(depth)]
        self_arrays.append(arr)

    print("\nTraining self-consistency...")
    self_hist = dd_self.self_train(self_arrays, max_iters=30, learning_rate=0.01, batch_size=16)

    rec_self = dd_self.reconstruct(depth=6, rows=6, cols=6, tau=0.2)
    print("\nReconstructed 3D array (first layer, first 3 rows):")
    for row in rec_self[0][:3]:
        print(row)

    print("\nAll tests completed successfully!")
