# Copyright (C) 2005-2025, Bin-Guang Ma (mbg@mail.hzau.edu.cn); SPDX-License-Identifier: MIT
# The Spatial Dual Descriptor Vector class (AB matrix form) for 4D character arrays implemented with PyTorch
# This program is for the demonstration of methodology and not fully refined.
# Author: Bin-Guang Ma (assisted by DeepSeek); Date: 2025-7-29 ~ 2026-3-25

import math
import itertools
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
import os

class SpatialDualDescriptorAB4(nn.Module):
    """
    Spatial Dual Descriptor for 4D character arrays (hypercube) with GPU acceleration:
      - learnable coefficient matrix Acoeff ∈ R^{m×L_total}, L_total = L1×L2×L3×L4
      - fixed basis matrix Bbasis ∈ R^{L_total×m},
        Bbasis[k1,k2,k3,k4,i] = ∏_{d=1}^4 cos(2π*(kd+1)/(i+2))
      - learnable token embeddings via nn.Embedding (tokens are flattened 4D windows)
      - Supports both linear and nonlinear tokenization in 4D
      - Batch processing for GPU acceleration
      - Supports regression, classification, and multi-label classification tasks
    """
    def __init__(self, charset, vec_dim=4, bas_dim_1=10, bas_dim_2=10, bas_dim_3=10, bas_dim_4=10,
                 rank=1, rank_mode='drop', mode='linear', user_step=None, device='cuda'):
        super().__init__()
        self.charset = list(charset)
        self.m = vec_dim
        self.L1 = bas_dim_1
        self.L2 = bas_dim_2
        self.L3 = bas_dim_3
        self.L4 = bas_dim_4
        self.L_total = self.L1 * self.L2 * self.L3 * self.L4
        self.rank = rank
        self.rank_mode = rank_mode
        assert mode in ('linear','nonlinear')
        self.mode = mode
        self.step = user_step
        self.trained = False
        self.mean_t = None
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # Generate tokens: all possible rank^4 patterns flattened row-major
        self.window_volume = rank ** 4
        if self.rank_mode == 'pad':
            extended_charset = self.charset + ['_']
            toks = [''.join(p) for p in itertools.product(extended_charset, repeat=self.window_volume)]
        else:
            toks = [''.join(p) for p in itertools.product(self.charset, repeat=self.window_volume)]
        self.tokens = sorted(set(toks))
        self.token_to_idx = {token: idx for idx, token in enumerate(self.tokens)}
        self.idx_to_token = {idx: token for idx, token in enumerate(self.tokens)}
        self.vocab_size = len(self.tokens)
        print(f"Vocabulary size (4D windows of size {rank}^4): {self.vocab_size}")

        # Token embeddings
        self.embedding = nn.Embedding(self.vocab_size, self.m)

        # Coefficient matrix A (m × L_total)
        self.Acoeff = nn.Parameter(torch.empty(self.m, self.L_total))

        # Fixed basis matrix B (L_total × m)
        Bbasis = torch.empty(self.L_total, self.m)
        for idx in range(self.L_total):
            # Decompose linear index into 4D coordinates
            k1 = idx // (self.L2 * self.L3 * self.L4)
            rem = idx % (self.L2 * self.L3 * self.L4)
            k2 = rem // (self.L3 * self.L4)
            rem2 = rem % (self.L3 * self.L4)
            k3 = rem2 // self.L4
            k4 = rem2 % self.L4
            for i in range(self.m):
                Bbasis[idx, i] = (math.cos(2 * math.pi * (k1+1) / (i+2)) *
                                  math.cos(2 * math.pi * (k2+1) / (i+2)) *
                                  math.cos(2 * math.pi * (k3+1) / (i+2)) *
                                  math.cos(2 * math.pi * (k4+1) / (i+2)))
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

    def extract_tokens(self, hypercube):
        """
        Extract 4D window tokens from a 4D character array.

        Args:
            hypercube: 4D list of chars, shape (D1, D2, D3, D4)
                       hypercube[d1][d2][d3][d4] is a char

        Returns:
            tokens: list of token strings (flattened window)
            positions: list of (d1, d2, d3, d4) tuples for the origin of each window
        """
        D1 = len(hypercube)
        if D1 == 0:
            return [], []
        D2 = len(hypercube[0])
        D3 = len(hypercube[0][0])
        D4 = len(hypercube[0][0][0])

        step = 1 if self.mode == 'linear' else (self.step or self.rank)

        tokens = []
        positions = []
        # Iterate over all possible window origins
        for d1 in range(0, D1 - self.rank + 1, step):
            for d2 in range(0, D2 - self.rank + 1, step):
                for d3 in range(0, D3 - self.rank + 1, step):
                    for d4 in range(0, D4 - self.rank + 1, step):
                        window = []
                        for i1 in range(self.rank):
                            for i2 in range(self.rank):
                                for i3 in range(self.rank):
                                    for i4 in range(self.rank):
                                        window.append(hypercube[d1+i1][d2+i2][d3+i3][d4+i4])
                        token = ''.join(window)
                        tokens.append(token)
                        positions.append((d1, d2, d3, d4))

        # Handle incomplete windows if rank_mode == 'pad'
        if self.rank_mode == 'pad':
            # For each possible origin (including those causing partial windows)
            for d1 in range(0, D1, step):
                for d2 in range(0, D2, step):
                    for d3 in range(0, D3, step):
                        for d4 in range(0, D4, step):
                            # Skip if it's already a full window
                            if (d1 + self.rank <= D1 and d2 + self.rank <= D2 and
                                d3 + self.rank <= D3 and d4 + self.rank <= D4):
                                continue
                            window = []
                            for i1 in range(self.rank):
                                for i2 in range(self.rank):
                                    for i3 in range(self.rank):
                                        for i4 in range(self.rank):
                                            if (d1+i1 < D1 and d2+i2 < D2 and
                                                d3+i3 < D3 and d4+i4 < D4):
                                                window.append(hypercube[d1+i1][d2+i2][d3+i3][d4+i4])
                                            else:
                                                window.append('_')
                            token = ''.join(window)
                            tokens.append(token)
                            positions.append((d1, d2, d3, d4))
        return tokens, positions

    def _compute_j_index(self, pos_tensor):
        """
        Compute j index for a batch of positions.
        pos_tensor: (N, 4) of integers (d1, d2, d3, d4)
        Returns: (N,) tensor of j indices in [0, L_total-1]
        """
        d1 = pos_tensor[:, 0] % self.L1
        d2 = pos_tensor[:, 1] % self.L2
        d3 = pos_tensor[:, 2] % self.L3
        d4 = pos_tensor[:, 3] % self.L4
        j_idx = (((d1 * self.L2 + d2) * self.L3 + d3) * self.L4 + d4).long()
        return j_idx

    def describe(self, hypercube):
        """Compute N(k) vectors for each window in the hypercube"""
        tokens, positions = self.extract_tokens(hypercube)
        if not tokens:
            return []

        token_indices = self.token_to_indices(tokens)
        pos_tensor = torch.tensor(positions, device=self.device, dtype=torch.long)
        j_indices = self._compute_j_index(pos_tensor)

        x = self.embedding(token_indices)          # [N, m]
        B_rows = self.Bbasis[j_indices]            # [N, m]
        scalar = torch.sum(B_rows * x, dim=1)      # [N]
        A_cols = self.Acoeff[:, j_indices].t()     # [N, m]
        Nk = A_cols * scalar.unsqueeze(1)          # [N, m]

        return Nk.detach().cpu().numpy()

    def S(self, hypercube):
        """Compute cumulative sum of N(k) vectors (ordered by window traversal)"""
        tokens, positions = self.extract_tokens(hypercube)
        if not tokens:
            return []

        token_indices = self.token_to_indices(tokens)
        pos_tensor = torch.tensor(positions, device=self.device, dtype=torch.long)
        j_indices = self._compute_j_index(pos_tensor)

        x = self.embedding(token_indices)
        B_rows = self.Bbasis[j_indices]
        scalar = torch.sum(B_rows * x, dim=1)
        A_cols = self.Acoeff[:, j_indices].t()
        Nk = A_cols * scalar.unsqueeze(1)

        S_cum = torch.cumsum(Nk, dim=0)
        return S_cum.detach().cpu().numpy()

    def D(self, hypercubes, t_list):
        """
        Compute mean squared deviation D across hypercubes:
        D = average over all windows of (N(k)-t_hypercube)^2

        Args:
            hypercubes: List of 4D character arrays
            t_list: List of target vectors

        Returns:
            float: Average mean squared deviation
        """
        total_loss = 0.0
        total_windows = 0

        t_tensors = [torch.tensor(t, dtype=torch.float32, device=self.device) for t in t_list]

        for hcube, t in zip(hypercubes, t_tensors):
            tokens, positions = self.extract_tokens(hcube)
            if not tokens:
                continue

            token_indices = self.token_to_indices(tokens)
            pos_tensor = torch.tensor(positions, device=self.device, dtype=torch.long)
            j_indices = self._compute_j_index(pos_tensor)

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

    def d(self, hypercube, t):
        """Compute pattern deviation value for a single hypercube."""
        return self.D([hypercube], [t])

    def _compute_training_statistics(self, hypercubes, batch_size=5):
        """Compute mean target vector and mean window count across all hypercubes."""
        total_window_count = 0
        total_t = torch.zeros(self.m, device=self.device)

        with torch.no_grad():
            for i in range(0, len(hypercubes), batch_size):
                batch_hcubes = hypercubes[i:i+batch_size]
                batch_window_count = 0
                batch_t_sum = torch.zeros(self.m, device=self.device)

                for hcube in batch_hcubes:
                    tokens, positions = self.extract_tokens(hcube)
                    batch_window_count += len(tokens)

                    if tokens:
                        token_indices = self.token_to_indices(tokens)
                        pos_tensor = torch.tensor(positions, device=self.device, dtype=torch.long)
                        j_indices = self._compute_j_index(pos_tensor)

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

        self.mean_window_count = total_window_count / len(hypercubes) if hypercubes else 0
        self.mean_t = (total_t / total_window_count).cpu().numpy() if total_window_count > 0 else np.zeros(self.m)

    def predict_t(self, hypercube):
        """Predict target vector as average of N(k) vectors over all windows"""
        tokens, positions = self.extract_tokens(hypercube)
        if not tokens:
            return np.zeros(self.m)

        Nk = self.describe(hypercube)
        return np.mean(Nk, axis=0)

    def predict_c(self, hypercube):
        """Predict class label using the classification head."""
        if self.classifier is None:
            raise ValueError("Model must be trained first for classification")

        vec = self.predict_t(hypercube)
        vec_tensor = torch.tensor(vec, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            logits = self.classifier(vec_tensor.unsqueeze(0))
            probabilities = torch.softmax(logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()

        return predicted_class, probabilities[0].cpu().numpy()

    def predict_l(self, hypercube, threshold=0.5):
        """Predict multi-label classification."""
        assert self.labeller is not None, "Model must be trained first for label prediction"

        tokens, positions = self.extract_tokens(hypercube)
        if not tokens:
            return np.zeros(self.num_labels, dtype=np.float32), np.zeros(self.num_labels, dtype=np.float32)

        token_indices = self.token_to_indices(tokens)
        pos_tensor = torch.tensor(positions, device=self.device, dtype=torch.long)
        j_indices = self._compute_j_index(pos_tensor)

        x = self.embedding(token_indices)
        B_rows = self.Bbasis[j_indices]
        scalar = torch.sum(B_rows * x, dim=1)
        A_cols = self.Acoeff[:, j_indices].t()
        Nk_batch = A_cols * scalar.unsqueeze(1)

        seq_rep = torch.mean(Nk_batch, dim=0)

        with torch.no_grad():
            logits = self.labeller(seq_rep)
            probs = torch.sigmoid(logits).cpu().numpy()

        binary_preds = (probs > threshold).astype(np.float32)
        return binary_preds, probs

    # ========== Training Methods ==========
    # Each training method is adapted to accept 4D data.
    # We'll implement reg_train, cls_train, lbl_train, self_train with the same pattern as before.

    def reg_train(self, hypercubes, t_list, max_iters=1000, tol=1e-8, learning_rate=0.01,
                  continued=False, decay_rate=1.0, print_every=10, batch_size=8,
                  checkpoint_file=None, checkpoint_interval=10):
        """Train for regression using gradient descent."""
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
            total_hcubes = 0

            indices = list(range(len(hypercubes)))
            random.shuffle(indices)

            for batch_start in range(0, len(indices), batch_size):
                batch_indices = indices[batch_start:batch_start + batch_size]
                batch_hcubes = [hypercubes[idx] for idx in batch_indices]
                batch_targets = [t_tensors[idx] for idx in batch_indices]

                optimizer.zero_grad()
                batch_loss = 0.0
                batch_count = 0

                for hcube, target in zip(batch_hcubes, batch_targets):
                    tokens, positions = self.extract_tokens(hcube)
                    if not tokens:
                        continue

                    token_indices = self.token_to_indices(tokens)
                    pos_tensor = torch.tensor(positions, device=self.device, dtype=torch.long)
                    j_indices = self._compute_j_index(pos_tensor)

                    x = self.embedding(token_indices)
                    B_rows = self.Bbasis[j_indices]
                    scalar = torch.sum(B_rows * x, dim=1)
                    A_cols = self.Acoeff[:, j_indices].t()
                    Nk_batch = A_cols * scalar.unsqueeze(1)

                    pred = torch.mean(Nk_batch, dim=0)
                    loss = torch.sum((pred - target) ** 2)
                    batch_loss += loss
                    batch_count += 1

                    del Nk_batch, pred, token_indices, pos_tensor, x, B_rows, A_cols, scalar

                if batch_count > 0:
                    batch_loss = batch_loss / batch_count
                    batch_loss.backward()
                    optimizer.step()

                    total_loss += batch_loss.item() * batch_count
                    total_hcubes += batch_count

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            avg_loss = total_loss / total_hcubes if total_hcubes else 0.0
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

        if best_model_state is not None and best_loss < history[-1]:
            self.load_state_dict(best_model_state)
            print(f"Training ended. Restored best model state with loss = {best_loss:.6e}")
            history[-1] = best_loss

        self._compute_training_statistics(hypercubes)
        self.trained = True
        return history

    def cls_train(self, hypercubes, labels, num_classes, max_iters=1000, tol=1e-8, learning_rate=0.01,
                  continued=False, decay_rate=1.0, print_every=10, batch_size=8,
                  checkpoint_file=None, checkpoint_interval=10):
        """Train for multi-class classification."""
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
            total_hcubes = 0
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
                    tokens, positions = self.extract_tokens(hcube)
                    if not tokens:
                        vec = torch.zeros(self.m, device=self.device)
                    else:
                        token_indices = self.token_to_indices(tokens)
                        pos_tensor = torch.tensor(positions, device=self.device, dtype=torch.long)
                        j_indices = self._compute_j_index(pos_tensor)

                        x = self.embedding(token_indices)
                        B_rows = self.Bbasis[j_indices]
                        scalar = torch.sum(B_rows * x, dim=1)
                        A_cols = self.Acoeff[:, j_indices].t()
                        Nk_batch = A_cols * scalar.unsqueeze(1)

                        vec = torch.mean(Nk_batch, dim=0)
                        del Nk_batch, token_indices, pos_tensor, x, B_rows, A_cols, scalar

                    logits = self.classifier(vec.unsqueeze(0))
                    batch_logits.append(logits)

                if batch_logits:
                    all_logits = torch.cat(batch_logits, dim=0)
                    loss = criterion(all_logits, batch_labels)
                    loss.backward()
                    optimizer.step()

                    batch_loss = loss.item()
                    total_loss += batch_loss * len(batch_hcubes)
                    total_hcubes += len(batch_hcubes)

                    with torch.no_grad():
                        preds = torch.argmax(all_logits, dim=1)
                        correct += (preds == batch_labels).sum().item()

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            avg_loss = total_loss / total_hcubes if total_hcubes else 0.0
            accuracy = correct / total_hcubes if total_hcubes else 0.0
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

    def lbl_train(self, hypercubes, labels, num_labels, max_iters=1000, tol=1e-8, learning_rate=0.01,
                  continued=False, decay_rate=1.0, print_every=10, batch_size=8,
                  checkpoint_file=None, checkpoint_interval=10, pos_weight=None):
        """Train for multi-label classification."""
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
            total_preds = 0
            total_hcubes = 0

            indices = list(range(len(hypercubes)))
            random.shuffle(indices)

            for batch_start in range(0, len(indices), batch_size):
                batch_indices = indices[batch_start:batch_start + batch_size]
                batch_hcubes = [hypercubes[idx] for idx in batch_indices]
                batch_labels = labels_tensor[batch_indices]

                optimizer.zero_grad()
                batch_logits_list = []

                for hcube in batch_hcubes:
                    tokens, positions = self.extract_tokens(hcube)
                    if not tokens:
                        continue

                    token_indices = self.token_to_indices(tokens)
                    pos_tensor = torch.tensor(positions, device=self.device, dtype=torch.long)
                    j_indices = self._compute_j_index(pos_tensor)

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
                    total_loss += batch_loss * len(batch_hcubes)
                    total_hcubes += len(batch_hcubes)

                    with torch.no_grad():
                        probs = torch.sigmoid(batch_logits)
                        preds = (probs > 0.5).float()
                        batch_correct = (preds == batch_labels).sum().item()
                        batch_preds = batch_labels.numel()
                        total_correct += batch_correct
                        total_preds += batch_preds

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            avg_loss = total_loss / total_hcubes if total_hcubes else 0.0
            avg_acc = total_correct / total_preds if total_preds else 0.0
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

    def self_train(self, hypercubes, max_iters=1000, tol=1e-8, learning_rate=0.01,
                   continued=False, decay_rate=1.0, print_every=10, batch_size=256,
                   checkpoint_file=None, checkpoint_interval=10):
        """Self-supervised training: predict each window from its own embedding."""
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

            indices = list(range(len(hypercubes)))
            random.shuffle(indices)

            for hcube_idx in indices:
                hcube = hypercubes[hcube_idx]
                tokens, positions = self.extract_tokens(hcube)
                if not tokens:
                    continue

                token_indices = self.token_to_indices(tokens)
                # Build list of (j_idx, token_idx) for each window
                samples = []
                for pos, token_idx in zip(positions, token_indices):
                    # Compute j_idx directly without tensor
                    d1, d2, d3, d4 = pos
                    j_idx = (((d1 % self.L1) * self.L2 + (d2 % self.L2)) * self.L3 +
                             (d3 % self.L3)) * self.L4 + (d4 % self.L4)
                    samples.append((j_idx, token_idx.item()))

                if not samples:
                    continue

                for batch_start in range(0, len(samples), batch_size):
                    batch_samples = samples[batch_start:batch_start + batch_size]

                    optimizer.zero_grad()

                    j_list = [s[0] for s in batch_samples]
                    tok_list = [s[1] for s in batch_samples]

                    j_tensor = torch.tensor(j_list, device=self.device, dtype=torch.long)
                    tok_tensor = torch.tensor(tok_list, device=self.device, dtype=torch.long)

                    x = self.embedding(tok_tensor)
                    B_rows = self.Bbasis[j_tensor]
                    scalar = torch.sum(B_rows * x, dim=1)
                    A_cols = self.Acoeff[:, j_tensor].t()
                    Nk_batch = A_cols * scalar.unsqueeze(1)

                    targets = self.embedding(tok_tensor)
                    loss = torch.mean(torch.sum((Nk_batch - targets) ** 2, dim=1))
                    loss.backward()
                    optimizer.step()

                    batch_loss = loss.item() * len(batch_samples)
                    total_loss += batch_loss
                    total_samples += len(batch_samples)

                    del j_tensor, tok_tensor, Nk_batch, targets, loss, x, B_rows, A_cols, scalar

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

        if best_model_state is not None and best_loss < history[-1]:
            self.load_state_dict(best_model_state)
            print(f"Training ended. Restored best model state with loss = {best_loss:.6f}")
            history[-1] = best_loss

        self._compute_training_statistics(hypercubes)
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
                'bas_dim_1': self.L1,
                'bas_dim_2': self.L2,
                'bas_dim_3': self.L3,
                'bas_dim_4': self.L4,
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

    def reconstruct_4d(self, D1, D2, D3, D4, tau=0.0):
        """
        Reconstruct a 4D hypercube of size D1×D2×D3×D4 using temperature-controlled randomness.

        The method slides a window over the target hypercube, predicts a token for each position,
        and then uses voting to assign each voxel a character.

        Args:
            D1, D2, D3, D4: dimensions of the hypercube
            tau: temperature (0 = deterministic)

        Returns:
            4D list of chars: hypercube[d1][d2][d3][d4]
        """
        assert self.trained, "Model must be trained first"
        if tau < 0:
            raise ValueError("Temperature must be non-negative")

        step = 1 if self.mode == 'linear' else (self.step or self.rank)

        # Determine all window origins (full windows only for simplicity)
        positions = []
        for d1 in range(0, D1 - self.rank + 1, step):
            for d2 in range(0, D2 - self.rank + 1, step):
                for d3 in range(0, D3 - self.rank + 1, step):
                    for d4 in range(0, D4 - self.rank + 1, step):
                        positions.append((d1, d2, d3, d4))

        mean_t_tensor = torch.tensor(self.mean_t, dtype=torch.float32, device=self.device)

        all_token_indices = torch.arange(self.vocab_size, device=self.device)
        all_embeddings = self.embedding(all_token_indices)

        # For each window origin, choose a token
        tokens_for_positions = []
        for pos in positions:
            d1, d2, d3, d4 = pos
            j_idx = (((d1 % self.L1) * self.L2 + (d2 % self.L2)) * self.L3 +
                     (d3 % self.L3)) * self.L4 + (d4 % self.L4)
            B_row = self.Bbasis[j_idx].unsqueeze(0)  # [1, m]
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

        # Voting: each window contributes its characters to overlapping voxels
        char_to_idx = {c: idx for idx, c in enumerate(self.charset)}
        vote_counts = np.zeros((D1, D2, D3, D4, len(self.charset)), dtype=int)

        for token, (d1, d2, d3, d4) in zip(tokens_for_positions, positions):
            # Token is a flattened rank^4 string; convert to 4D pattern
            # Flattening order: d1, d2, d3, d4 (row-major)
            pattern = []
            idx = 0
            for i1 in range(self.rank):
                dim1 = []
                for i2 in range(self.rank):
                    dim2 = []
                    for i3 in range(self.rank):
                        dim3 = []
                        for i4 in range(self.rank):
                            dim3.append(token[idx])
                            idx += 1
                        dim2.append(dim3)
                    dim1.append(dim2)
                pattern.append(dim1)

            for i1 in range(self.rank):
                for i2 in range(self.rank):
                    for i3 in range(self.rank):
                        for i4 in range(self.rank):
                            if (d1+i1 < D1 and d2+i2 < D2 and
                                d3+i3 < D3 and d4+i4 < D4):
                                ch = pattern[i1][i2][i3][i4]
                                if ch in char_to_idx:
                                    vote_counts[d1+i1, d2+i2, d3+i3, d4+i4, char_to_idx[ch]] += 1

        # Build final hypercube
        hypercube = []
        for d1 in range(D1):
            dim1 = []
            for d2 in range(D2):
                dim2 = []
                for d3 in range(D3):
                    dim3 = []
                    for d4 in range(D4):
                        char_idx = np.argmax(vote_counts[d1, d2, d3, d4])
                        dim3.append(self.charset[char_idx])
                    dim2.append(dim3)
                dim1.append(dim2)
            hypercube.append(dim1)
        return hypercube

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


# === Example Usage (4D) ===
if __name__ == "__main__":
    from statistics import correlation
    import sys

    print("="*60)
    print("Spatial Dual Descriptor AB4 - 4D Hypercube Version")
    print("Optimized with batch processing")
    print("="*60)

    # Set random seeds
    torch.manual_seed(11)
    random.seed(11)

    # Parameters (keep very small for demo)
    charset = ['A','C','G','T']
    vec_dim = 3
    bas_dim_1 = 5
    bas_dim_2 = 5
    bas_dim_3 = 5
    bas_dim_4 = 5
    rank = 1          # 1x1x1x1 windows (vocabulary size = 4^1 = 4)
    hypercube_num = 20  # number of training hypercubes

    print(f"\nGenerating {hypercube_num} random 4D hypercubes (size 4x4x4x4 each)...")
    # Generate random hypercubes of small size (4x4x4x4) to keep memory manageable
    hypercubes = []
    t_list = []
    for _ in range(hypercube_num):
        D1 = 4
        D2 = 4
        D3 = 4
        D4 = 4
        # Build 4D list: [d1][d2][d3][d4]
        hcube = [[[[random.choice(charset) for _ in range(D4)] for _ in range(D3)] for _ in range(D2)] for _ in range(D1)]
        hypercubes.append(hcube)
        t_list.append([random.uniform(-1,1) for _ in range(vec_dim)])

    # Create model
    dd4 = SpatialDualDescriptorAB4(
        charset,
        vec_dim=vec_dim,
        bas_dim_1=bas_dim_1,
        bas_dim_2=bas_dim_2,
        bas_dim_3=bas_dim_3,
        bas_dim_4=bas_dim_4,
        rank=rank,
        mode='linear',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    print(f"\nUsing device: {dd4.device}")
    print(f"Number of tokens: {len(dd4.tokens)}")

    # === Regression Training ===
    print("\n" + "="*50)
    print("Testing Gradient Descent Training (Regression)")
    print("="*50)

    reg_history = dd4.reg_train(
        hypercubes, t_list,
        learning_rate=0.1,
        max_iters=30,        # small for demo
        tol=1e-66,
        print_every=10,
        decay_rate=0.99,
        batch_size=4
    )

    # Predict target for first hypercube
    pred_t = dd4.predict_t(hypercubes[0])
    print(f"\nPredicted t for first hypercube: {[round(x.item(), 4) for x in pred_t]}")

    # Calculate prediction correlation (if enough data)
    if hypercube_num > 1:
        pred_t_list = [dd4.predict_t(hc) for hc in hypercubes]
        corr_sum = 0.0
        for i in range(dd4.m):
            actu_t = [t_vec[i] for t_vec in t_list]
            pred_t_vals = [t_vec[i] for t_vec in pred_t_list]
            corr = correlation(actu_t, pred_t_vals)
            print(f"Prediction correlation: {corr:.4f}")
            corr_sum += corr
        corr_avg = corr_sum / dd4.m
        print(f"Average correlation: {corr_avg:.4f}")

    # === Classification Task ===
    print("\n" + "="*50)
    print("Classification Task (3-class)")
    print("="*50)

    num_classes = 3
    class_hcubes = []
    class_labels = []
    for class_id in range(num_classes):
        for _ in range(15):
            D1 = D2 = D3 = D4 = 4
            if class_id == 0:
                # High A content
                hcube = [[[[random.choices(['A','C','G','T'], weights=[0.6,0.1,0.1,0.2])[0]
                           for _ in range(D4)] for _ in range(D3)] for _ in range(D2)] for _ in range(D1)]
            elif class_id == 1:
                # High GC content
                hcube = [[[[random.choices(['A','C','G','T'], weights=[0.1,0.4,0.4,0.1])[0]
                           for _ in range(D4)] for _ in range(D3)] for _ in range(D2)] for _ in range(D1)]
            else:
                # Balanced
                hcube = [[[[random.choice(charset) for _ in range(D4)] for _ in range(D3)] for _ in range(D2)] for _ in range(D1)]
            class_hcubes.append(hcube)
            class_labels.append(class_id)

    dd_cls = SpatialDualDescriptorAB4(
        charset, vec_dim=vec_dim, bas_dim_1=bas_dim_1, bas_dim_2=bas_dim_2,
        bas_dim_3=bas_dim_3, bas_dim_4=bas_dim_4, rank=rank, mode='linear',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    print("\nStarting Classification Training (max_iters=20)...")
    cls_history = dd_cls.cls_train(class_hcubes, class_labels, num_classes,
                                   max_iters=20, learning_rate=0.05,
                                   decay_rate=0.99, batch_size=4, print_every=5)

    correct = 0
    for hcube, true_label in zip(class_hcubes, class_labels):
        pred_class, _ = dd_cls.predict_c(hcube)
        if pred_class == true_label:
            correct += 1
    acc = correct / len(class_hcubes)
    print(f"Classification Accuracy: {acc:.4f} ({correct}/{len(class_hcubes)})")

    # === Multi-Label Classification Task ===
    print("\n" + "="*50)
    print("Multi-Label Classification Task (4 labels)")
    print("="*50)

    num_labels = 4
    label_hcubes = []
    labels = []
    for _ in range(20):
        D1 = D2 = D3 = D4 = 4
        hcube = [[[[random.choice(charset) for _ in range(D4)] for _ in range(D3)] for _ in range(D2)] for _ in range(D1)]
        label_hcubes.append(hcube)
        label_vec = [1.0 if random.random() > 0.7 else 0.0 for _ in range(num_labels)]
        labels.append(label_vec)

    dd_lbl = SpatialDualDescriptorAB4(
        charset, vec_dim=vec_dim, bas_dim_1=bas_dim_1, bas_dim_2=bas_dim_2,
        bas_dim_3=bas_dim_3, bas_dim_4=bas_dim_4, rank=rank, mode='linear',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    loss_hist, acc_hist = dd_lbl.lbl_train(
        label_hcubes, labels, num_labels,
        max_iters=20, learning_rate=0.05, decay_rate=0.99, batch_size=4, print_every=5
    )
    print(f"Multi-label final training loss: {loss_hist[-1]:.6f}, accuracy: {acc_hist[-1]:.4f}")

    # === Self-Training Example ===
    print("\n" + "="*50)
    print("Self-Training (Gap Mode)")
    print("="*50)

    self_hcubes = []
    for _ in range(15):
        D1 = D2 = D3 = D4 = 4
        hcube = [[[[random.choice(charset) for _ in range(D4)] for _ in range(D3)] for _ in range(D2)] for _ in range(D1)]
        self_hcubes.append(hcube)

    dd_self = SpatialDualDescriptorAB4(
        charset, vec_dim=vec_dim, bas_dim_1=bas_dim_1, bas_dim_2=bas_dim_2,
        bas_dim_3=bas_dim_3, bas_dim_4=bas_dim_4, rank=rank, mode='linear',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    self_history = dd_self.self_train(
        self_hcubes,
        max_iters=20,
        learning_rate=0.001,
        decay_rate=0.995,
        print_every=5,
        batch_size=64
    )

    # Reconstruct a small 4D hypercube from self-trained model
    print("\nReconstructing a 4x4x4x4 hypercube from self-trained model (deterministic):")
    recon_hcube = dd_self.reconstruct_4d(D1=4, D2=4, D3=4, D4=4, tau=0.0)
    print("First layer (d1=0, d2=0, d3=0):")
    # Print a 2D slice at d1=0, d2=0, d3=0
    slice_2d = [[recon_hcube[0][0][0][w] for w in range(4)] for _ in range(1)]  # Actually 1 row? Let's just print one row.
    print(''.join(recon_hcube[0][0][0]))
    print("...")

    # Save and load model
    print("\n=== Model Persistence Test ===")
    dd_self.save("self_trained_4d.pkl")
    dd_loaded = SpatialDualDescriptorAB4(
        charset, vec_dim=vec_dim, bas_dim_1=bas_dim_1, bas_dim_2=bas_dim_2,
        bas_dim_3=bas_dim_3, bas_dim_4=bas_dim_4, rank=rank, mode='linear',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    ).load("self_trained_4d.pkl")
    print("Model loaded successfully. Reconstructing with loaded model:")
    recon_loaded = dd_loaded.reconstruct_4d(D1=4, D2=4, D3=4, D4=4, tau=0.0)
    print("First voxel of reconstructed hypercube:", recon_loaded[0][0][0][0])

    print("\n=== All 4D Tests Completed ===")
