# Copyright (C) 2005-2025, Bin-Guang Ma (mbg@mail.hzau.edu.cn); SPDX-License-Identifier: MIT
# The Spatial Dual Descriptor Vector class (AB matrix form) for 3D character arrays implemented with PyTorch
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

class SpatialDualDescriptorAB3(nn.Module):
    """
    Spatial Dual Descriptor for 3D character arrays (voxel grids) with GPU acceleration:
      - learnable coefficient matrix Acoeff ∈ R^{m×(Lh×Lw×Ld)}
      - fixed basis matrix Bbasis ∈ R^{(Lh×Lw×Ld)×m}, 
        Bbasis[ki, kj, kk, i] = cos(2π*(ki+1)/(i+2)) * cos(2π*(kj+1)/(i+2)) * cos(2π*(kk+1)/(i+2))
      - learnable token embeddings via nn.Embedding (tokens are flattened 3D windows)
      - Supports both linear and nonlinear tokenization in 3D
      - Batch processing for GPU acceleration
      - Supports regression, classification, and multi-label classification tasks
    """
    def __init__(self, charset, vec_dim=4, bas_dim_h=50, bas_dim_w=50, bas_dim_d=50, 
                 rank=1, rank_mode='drop', mode='linear', user_step=None, device='cuda'):
        super().__init__()
        self.charset = list(charset)
        self.m = vec_dim
        self.Lh = bas_dim_h
        self.Lw = bas_dim_w
        self.Ld = bas_dim_d
        self.L_total = self.Lh * self.Lw * self.Ld
        self.rank = rank
        self.rank_mode = rank_mode
        assert mode in ('linear','nonlinear')
        self.mode = mode
        self.step = user_step
        self.trained = False
        self.mean_t = None  
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Generate tokens: all possible rank×rank×rank patterns flattened row-major
        self.window_volume = rank ** 3
        # For drop mode, tokens are exactly all patterns from charset
        # For pad mode, we also need '_' for padding (extended charset)
        if self.rank_mode == 'pad':
            extended_charset = self.charset + ['_']
            toks = [''.join(p) for p in itertools.product(extended_charset, repeat=self.window_volume)]
        else:
            toks = [''.join(p) for p in itertools.product(self.charset, repeat=self.window_volume)]
        self.tokens = sorted(set(toks))
        self.token_to_idx = {token: idx for idx, token in enumerate(self.tokens)}
        self.idx_to_token = {idx: token for idx, token in enumerate(self.tokens)}
        self.vocab_size = len(self.tokens)
        print(f"Vocabulary size (3D windows of size {rank}^3): {self.vocab_size}")
        
        # Token embeddings
        self.embedding = nn.Embedding(self.vocab_size, self.m)
        
        # Coefficient matrix A (m × L_total)
        self.Acoeff = nn.Parameter(torch.empty(self.m, self.L_total))
        
        # Fixed basis matrix B (L_total × m)
        Bbasis = torch.empty(self.L_total, self.m)
        for idx in range(self.L_total):
            ki = idx // (self.Lw * self.Ld)
            rem = idx % (self.Lw * self.Ld)
            kj = rem // self.Ld
            kk = rem % self.Ld
            for i in range(self.m):
                Bbasis[idx, i] = (math.cos(2 * math.pi * (ki+1) / (i+2)) *
                                  math.cos(2 * math.pi * (kj+1) / (i+2)) *
                                  math.cos(2 * math.pi * (kk+1) / (i+2)))
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
    
    def extract_tokens(self, volume):
        """
        Extract 3D window tokens from a 3D character array.
        
        Args:
            volume: 3D list of chars, shape (D, H, W) or (H, W, D)? We'll use (depth, height, width) for clarity.
                    Input format: list of list of list of chars: volume[d][h][w]
            
        Returns:
            tokens: list of token strings (flattened window)
            positions: list of (d, h, w) tuples for the origin (minimum corner) of each window
        """
        D = len(volume)
        if D == 0:
            return [], []
        H = len(volume[0])
        if H == 0:
            return [], []
        W = len(volume[0][0])
        
        # Linear mode: step=1; nonlinear: step = user_step or rank
        step = 1 if self.mode == 'linear' else (self.step or self.rank)
        
        tokens = []
        positions = []
        # Iterate over all possible window origins
        for d in range(0, D - self.rank + 1, step):
            for h in range(0, H - self.rank + 1, step):
                for w in range(0, W - self.rank + 1, step):
                    # Extract window: order is depth, height, width (row-major)
                    window = []
                    for di in range(self.rank):
                        for hi in range(self.rank):
                            for wi in range(self.rank):
                                window.append(volume[d+di][h+hi][w+wi])
                    token = ''.join(window)
                    tokens.append(token)
                    positions.append((d, h, w))
        
        # Handle incomplete windows if rank_mode == 'pad'
        if self.rank_mode == 'pad':
            # We need to consider windows that go beyond the volume boundaries.
            # We'll slide over all origins that are within the volume (including those that cause partial windows)
            # and pad with '_'.
            for d in range(0, D, step):
                for h in range(0, H, step):
                    for w in range(0, W, step):
                        # Skip already processed full windows
                        if d + self.rank <= D and h + self.rank <= H and w + self.rank <= W:
                            continue
                        # Extract with padding
                        window = []
                        for di in range(self.rank):
                            for hi in range(self.rank):
                                for wi in range(self.rank):
                                    if (d+di < D) and (h+hi < H) and (w+wi < W):
                                        window.append(volume[d+di][h+hi][w+wi])
                                    else:
                                        window.append('_')
                        token = ''.join(window)
                        tokens.append(token)
                        positions.append((d, h, w))
        return tokens, positions

    def describe(self, volume):
        """Compute N(k) vectors for each window in the volume"""
        tokens, positions = self.extract_tokens(volume)
        if not tokens:
            return []
        
        token_indices = self.token_to_indices(tokens)
        # positions tensor (num_windows, 3)
        pos_tensor = torch.tensor(positions, device=self.device, dtype=torch.long)
        
        # Compute j indices: j = (d % Lh)*Lw*Ld + (h % Lw)*Ld + (w % Ld)
        d_idx = pos_tensor[:, 0] % self.Lh
        h_idx = pos_tensor[:, 1] % self.Lw
        w_idx = pos_tensor[:, 2] % self.Ld
        j_indices = d_idx * (self.Lw * self.Ld) + h_idx * self.Ld + w_idx  # shape (num_windows,)
        
        # Get token embeddings
        x = self.embedding(token_indices)  # [num_windows, m]
        
        # Get B rows for each j
        B_rows = self.Bbasis[j_indices]  # [num_windows, m]
        
        # Compute scalar = B[j] • x
        scalar = torch.sum(B_rows * x, dim=1)  # [num_windows]
        
        # Get A columns for each j
        A_cols = self.Acoeff[:, j_indices].t()  # [num_windows, m]
        
        # Nk = scalar * A_cols
        Nk = A_cols * scalar.unsqueeze(1)  # [num_windows, m]
        
        return Nk.detach().cpu().numpy()
    
    def S(self, volume):
        """Compute cumulative sum of N(k) vectors (ordered by window traversal)"""
        tokens, positions = self.extract_tokens(volume)
        if not tokens:
            return []
        
        token_indices = self.token_to_indices(tokens)
        pos_tensor = torch.tensor(positions, device=self.device, dtype=torch.long)
        d_idx = pos_tensor[:, 0] % self.Lh
        h_idx = pos_tensor[:, 1] % self.Lw
        w_idx = pos_tensor[:, 2] % self.Ld
        j_indices = d_idx * (self.Lw * self.Ld) + h_idx * self.Ld + w_idx
        
        x = self.embedding(token_indices)
        B_rows = self.Bbasis[j_indices]
        scalar = torch.sum(B_rows * x, dim=1)
        A_cols = self.Acoeff[:, j_indices].t()
        Nk = A_cols * scalar.unsqueeze(1)
        
        S_cum = torch.cumsum(Nk, dim=0)
        return S_cum.detach().cpu().numpy()

    def D(self, volumes, t_list):
        """
        Compute mean squared deviation D across volumes:
        D = average over all windows of (N(k)-t_vol)^2
        
        Args:
            volumes: List of 3D character arrays
            t_list: List of target vectors corresponding to each volume
            
        Returns:
            float: Average mean squared deviation across all windows and volumes
        """
        total_loss = 0.0
        total_windows = 0
        
        t_tensors = [torch.tensor(t, dtype=torch.float32, device=self.device) for t in t_list]
        
        for vol, t in zip(volumes, t_tensors):
            tokens, positions = self.extract_tokens(vol)
            if not tokens:
                continue
                
            token_indices = self.token_to_indices(tokens)
            pos_tensor = torch.tensor(positions, device=self.device, dtype=torch.long)
            d_idx = pos_tensor[:, 0] % self.Lh
            h_idx = pos_tensor[:, 1] % self.Lw
            w_idx = pos_tensor[:, 2] % self.Ld
            j_indices = d_idx * (self.Lw * self.Ld) + h_idx * self.Ld + w_idx
            
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

    def d(self, volume, t):
        """Compute pattern deviation value for a single volume."""
        return self.D([volume], [t])

    def _compute_training_statistics(self, volumes, batch_size=5):
        """
        Compute and store statistics for reconstruction and generation.
        Calculates mean window count and mean target vector across all volumes.
        """
        total_window_count = 0
        total_t = torch.zeros(self.m, device=self.device)
        
        with torch.no_grad():
            for i in range(0, len(volumes), batch_size):
                batch_vols = volumes[i:i+batch_size]
                batch_window_count = 0
                batch_t_sum = torch.zeros(self.m, device=self.device)
                
                for vol in batch_vols:
                    tokens, positions = self.extract_tokens(vol)
                    batch_window_count += len(tokens)
                    
                    if tokens:
                        token_indices = self.token_to_indices(tokens)
                        pos_tensor = torch.tensor(positions, device=self.device, dtype=torch.long)
                        d_idx = pos_tensor[:, 0] % self.Lh
                        h_idx = pos_tensor[:, 1] % self.Lw
                        w_idx = pos_tensor[:, 2] % self.Ld
                        j_indices = d_idx * (self.Lw * self.Ld) + h_idx * self.Ld + w_idx
                        
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
        
        self.mean_window_count = total_window_count / len(volumes) if volumes else 0
        self.mean_t = (total_t / total_window_count).cpu().numpy() if total_window_count > 0 else np.zeros(self.m)
    
    def predict_t(self, volume):
        """Predict target vector as average of N(k) vectors over all windows"""
        tokens, positions = self.extract_tokens(volume)
        if not tokens:
            return np.zeros(self.m)
        
        Nk = self.describe(volume)
        return np.mean(Nk, axis=0)

    def predict_c(self, volume):
        """Predict class label for a volume using the classification head."""
        if self.classifier is None:
            raise ValueError("Model must be trained first for classification")
        
        vec = self.predict_t(volume)
        vec_tensor = torch.tensor(vec, dtype=torch.float32, device=self.device)
        
        with torch.no_grad():
            logits = self.classifier(vec_tensor.unsqueeze(0))
            probabilities = torch.softmax(logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            
        return predicted_class, probabilities[0].cpu().numpy()

    def predict_l(self, volume, threshold=0.5):
        """Predict multi-label classification for a volume."""
        assert self.labeller is not None, "Model must be trained first for label prediction"
        
        tokens, positions = self.extract_tokens(volume)
        if not tokens:
            return np.zeros(self.num_labels, dtype=np.float32), np.zeros(self.num_labels, dtype=np.float32)
        
        token_indices = self.token_to_indices(tokens)
        pos_tensor = torch.tensor(positions, device=self.device, dtype=torch.long)
        d_idx = pos_tensor[:, 0] % self.Lh
        h_idx = pos_tensor[:, 1] % self.Lw
        w_idx = pos_tensor[:, 2] % self.Ld
        j_indices = d_idx * (self.Lw * self.Ld) + h_idx * self.Ld + w_idx
        
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
    # All training methods follow the same structure as the 1D and 2D versions,
    # but using 3D data. They are adapted here for completeness.
    
    def reg_train(self, volumes, t_list, max_iters=1000, tol=1e-8, learning_rate=0.01, 
               continued=False, decay_rate=1.0, print_every=10, batch_size=16,
               checkpoint_file=None, checkpoint_interval=10):
        """
        Train model for regression using gradient descent with batch processing.
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
            total_volumes = 0
            
            indices = list(range(len(volumes)))
            random.shuffle(indices)
            
            for batch_start in range(0, len(indices), batch_size):
                batch_indices = indices[batch_start:batch_start + batch_size]
                batch_vols = [volumes[idx] for idx in batch_indices]
                batch_targets = [t_tensors[idx] for idx in batch_indices]
                
                optimizer.zero_grad()
                batch_loss = 0.0
                batch_count = 0
                
                for vol, target in zip(batch_vols, batch_targets):
                    tokens, positions = self.extract_tokens(vol)
                    if not tokens:
                        continue
                        
                    token_indices = self.token_to_indices(tokens)
                    pos_tensor = torch.tensor(positions, device=self.device, dtype=torch.long)
                    d_idx = pos_tensor[:, 0] % self.Lh
                    h_idx = pos_tensor[:, 1] % self.Lw
                    w_idx = pos_tensor[:, 2] % self.Ld
                    j_indices = d_idx * (self.Lw * self.Ld) + h_idx * self.Ld + w_idx
                    
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
                    total_volumes += batch_count
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            avg_loss = total_loss / total_volumes if total_volumes else 0.0
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
        
        self._compute_training_statistics(volumes)
        self.trained = True
        return history

    def cls_train(self, volumes, labels, num_classes, max_iters=1000, tol=1e-8, learning_rate=0.01,
                  continued=False, decay_rate=1.0, print_every=10, batch_size=16,
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
            total_vols = 0
            correct = 0
            
            indices = list(range(len(volumes)))
            random.shuffle(indices)
            
            for batch_start in range(0, len(indices), batch_size):
                batch_indices = indices[batch_start:batch_start + batch_size]
                batch_vols = [volumes[idx] for idx in batch_indices]
                batch_labels = label_tensors[batch_indices]
                
                optimizer.zero_grad()
                batch_logits = []
                
                for vol in batch_vols:
                    tokens, positions = self.extract_tokens(vol)
                    if not tokens:
                        vec = torch.zeros(self.m, device=self.device)
                    else:
                        token_indices = self.token_to_indices(tokens)
                        pos_tensor = torch.tensor(positions, device=self.device, dtype=torch.long)
                        d_idx = pos_tensor[:, 0] % self.Lh
                        h_idx = pos_tensor[:, 1] % self.Lw
                        w_idx = pos_tensor[:, 2] % self.Ld
                        j_indices = d_idx * (self.Lw * self.Ld) + h_idx * self.Ld + w_idx
                        
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
                    total_loss += batch_loss * len(batch_vols)
                    total_vols += len(batch_vols)
                    
                    with torch.no_grad():
                        preds = torch.argmax(all_logits, dim=1)
                        correct += (preds == batch_labels).sum().item()
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            avg_loss = total_loss / total_vols if total_vols else 0.0
            accuracy = correct / total_vols if total_vols else 0.0
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

    def lbl_train(self, volumes, labels, num_labels, max_iters=1000, tol=1e-8, learning_rate=0.01, 
                 continued=False, decay_rate=1.0, print_every=10, batch_size=16,
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
            total_vols = 0
            
            indices = list(range(len(volumes)))
            random.shuffle(indices)
            
            for batch_start in range(0, len(indices), batch_size):
                batch_indices = indices[batch_start:batch_start + batch_size]
                batch_vols = [volumes[idx] for idx in batch_indices]
                batch_labels = labels_tensor[batch_indices]
                
                optimizer.zero_grad()
                batch_logits_list = []
                
                for vol in batch_vols:
                    tokens, positions = self.extract_tokens(vol)
                    if not tokens:
                        continue
                    
                    token_indices = self.token_to_indices(tokens)
                    pos_tensor = torch.tensor(positions, device=self.device, dtype=torch.long)
                    d_idx = pos_tensor[:, 0] % self.Lh
                    h_idx = pos_tensor[:, 1] % self.Lw
                    w_idx = pos_tensor[:, 2] % self.Ld
                    j_indices = d_idx * (self.Lw * self.Ld) + h_idx * self.Ld + w_idx
                    
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
                    total_loss += batch_loss * len(batch_vols)
                    total_vols += len(batch_vols)
                    
                    with torch.no_grad():
                        probs = torch.sigmoid(batch_logits)
                        preds = (probs > 0.5).float()
                        batch_correct = (preds == batch_labels).sum().item()
                        batch_preds = batch_labels.numel()
                        total_correct += batch_correct
                        total_preds += batch_preds
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            avg_loss = total_loss / total_vols if total_vols else 0.0
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

    def self_train(self, volumes, max_iters=1000, tol=1e-8, learning_rate=0.01, 
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
            
            indices = list(range(len(volumes)))
            random.shuffle(indices)
            
            for vol_idx in indices:
                vol = volumes[vol_idx]
                tokens, positions = self.extract_tokens(vol)
                if not tokens:
                    continue
                
                token_indices = self.token_to_indices(tokens)
                samples = []
                for pos, token_idx in zip(positions, token_indices):
                    d, h, w = pos
                    j_idx = (d % self.Lh) * (self.Lw * self.Ld) + (h % self.Lw) * self.Ld + (w % self.Ld)
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
        
        self._compute_training_statistics(volumes)
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
                'bas_dim_d': self.Ld,
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

    def reconstruct_3d(self, D, H, W, tau=0.0):
        """
        Reconstruct a 3D grid of size D x H x W using temperature-controlled randomness.
        
        The method slides a window over the target grid, predicts a token for each position,
        and then uses voting to assign each voxel a character.
        
        Args:
            D: depth
            H: height
            W: width
            tau: temperature (0 = deterministic)
            
        Returns:
            3D list of chars: volume[d][h][w]
        """
        assert self.trained, "Model must be trained first"
        if tau < 0:
            raise ValueError("Temperature must be non-negative")
        
        step = 1 if self.mode == 'linear' else (self.step or self.rank)
        
        # Determine all window origins that would be used during training (full windows only for simplicity)
        positions = []
        for d in range(0, D - self.rank + 1, step):
            for h in range(0, H - self.rank + 1, step):
                for w in range(0, W - self.rank + 1, step):
                    positions.append((d, h, w))
        # For pad mode, we would also include partial windows, but we skip for simplicity.
        
        mean_t_tensor = torch.tensor(self.mean_t, dtype=torch.float32, device=self.device)
        
        all_token_indices = torch.arange(self.vocab_size, device=self.device)
        all_embeddings = self.embedding(all_token_indices)
        
        # For each window origin, choose a token
        tokens_for_positions = []
        for pos in positions:
            d, h, w = pos
            j_idx = (d % self.Lh) * (self.Lw * self.Ld) + (h % self.Lw) * self.Ld + (w % self.Ld)
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
        
        # Voting: each window contributes its characters to the overlapping voxels
        char_to_idx = {c: idx for idx, c in enumerate(self.charset)}
        vote_counts = np.zeros((D, H, W, len(self.charset)), dtype=int)
        
        for token, (d, h, w) in zip(tokens_for_positions, positions):
            # Token is a flattened rank^3 string; convert to 3D pattern
            # The flattening order is depth first, then height, then width (as used in extraction)
            pattern = []
            for di in range(self.rank):
                layer = []
                for hi in range(self.rank):
                    row = []
                    for wi in range(self.rank):
                        row.append(token[di*self.rank*self.rank + hi*self.rank + wi])
                    layer.append(row)
                pattern.append(layer)
            for di in range(self.rank):
                for hi in range(self.rank):
                    for wi in range(self.rank):
                        if d+di < D and h+hi < H and w+wi < W:
                            ch = pattern[di][hi][wi]
                            if ch in char_to_idx:
                                vote_counts[d+di, h+hi, w+wi, char_to_idx[ch]] += 1
        
        # Build final volume
        volume = []
        for d in range(D):
            depth_layer = []
            for h in range(H):
                row = []
                for w in range(W):
                    char_idx = np.argmax(vote_counts[d, h, w])
                    row.append(self.charset[char_idx])
                depth_layer.append(row)
            volume.append(depth_layer)
        return volume

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


# === Example Usage (3D) ===
if __name__ == "__main__":
    from statistics import correlation
    import sys
    
    print("="*60)
    print("Spatial Dual Descriptor AB3 - 3D Voxel Version")
    print("Optimized with batch processing")
    print("="*60)
    
    # Set random seeds
    torch.manual_seed(11)
    random.seed(11)
    
    # Parameters (keep small for demo)
    charset = ['A','C','G','T']
    vec_dim = 3
    bas_dim_h = 4
    bas_dim_w = 4
    bas_dim_d = 4
    rank = 2          # 2x2x2 windows (vocabulary size = 4^8 = 65536)
    volume_num = 50   # number of training volumes
    
    print(f"\nGenerating {volume_num} random 3D volumes (size 6x6x6 each)...")
    # Generate random volumes of small size (6x6x6) to keep memory manageable
    volumes = []
    t_list = []
    for _ in range(volume_num):
        D = random.randint(6, 8)
        H = random.randint(6, 8)
        W = random.randint(6, 8)
        vol = [[[random.choice(charset) for _ in range(W)] for _ in range(H)] for _ in range(D)]
        volumes.append(vol)
        t_list.append([random.uniform(-1,1) for _ in range(vec_dim)])
    
    # Create model
    dd3 = SpatialDualDescriptorAB3(
        charset, 
        vec_dim=vec_dim, 
        bas_dim_h=bas_dim_h,
        bas_dim_w=bas_dim_w,
        bas_dim_d=bas_dim_d,
        rank=rank,
        mode='linear',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    print(f"\nUsing device: {dd3.device}")
    print(f"Number of tokens: {len(dd3.tokens)}")
    
    # === Regression Training ===
    print("\n" + "="*50)
    print("Testing Gradient Descent Training (Regression)")
    print("="*50)
    
    reg_history = dd3.reg_train(
        volumes, t_list,
        learning_rate=0.1,
        max_iters=100,      # small for demo
        tol=1e-66,
        print_every=10,
        decay_rate=0.99,
        batch_size=8
    )
    
    # Predict target for first volume
    pred_t = dd3.predict_t(volumes[0])
    print(f"\nPredicted t for first volume: {[round(x.item(), 4) for x in pred_t]}")
    
    # Calculate prediction correlation (if enough data)
    if volume_num > 1:
        pred_t_list = [dd3.predict_t(vol) for vol in volumes]
        corr_sum = 0.0
        for i in range(dd3.m):
            actu_t = [t_vec[i] for t_vec in t_list]
            pred_t_vals = [t_vec[i] for t_vec in pred_t_list]
            corr = correlation(actu_t, pred_t_vals)
            print(f"Prediction correlation: {corr:.4f}")
            corr_sum += corr
        corr_avg = corr_sum / dd3.m
        print(f"Average correlation: {corr_avg:.4f}")
    
    # === Classification Task ===
    print("\n" + "="*50)
    print("Classification Task (3-class)")
    print("="*50)
    
    num_classes = 3
    class_vols = []
    class_labels = []
    for class_id in range(num_classes):
        for _ in range(50):
            D = random.randint(6, 8)
            H = random.randint(6, 8)
            W = random.randint(6, 8)
            if class_id == 0:
                # High A content
                vol = [[[random.choices(['A','C','G','T'], weights=[0.6,0.1,0.1,0.2])[0] 
                         for _ in range(W)] for _ in range(H)] for _ in range(D)]
            elif class_id == 1:
                # High GC content
                vol = [[[random.choices(['A','C','G','T'], weights=[0.1,0.4,0.4,0.1])[0] 
                         for _ in range(W)] for _ in range(H)] for _ in range(D)]
            else:
                # Balanced
                vol = [[[random.choice(charset) for _ in range(W)] for _ in range(H)] for _ in range(D)]
            class_vols.append(vol)
            class_labels.append(class_id)
    
    dd_cls = SpatialDualDescriptorAB3(
        charset, vec_dim=vec_dim, bas_dim_h=bas_dim_h, bas_dim_w=bas_dim_w, bas_dim_d=bas_dim_d,
        rank=rank, mode='linear', device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    print("\nStarting Classification Training (max_iters=30)...")
    cls_history = dd_cls.cls_train(class_vols, class_labels, num_classes, 
                                   max_iters=20, learning_rate=0.05,
                                   decay_rate=0.99, batch_size=8, print_every=10)
    
    correct = 0
    for vol, true_label in zip(class_vols, class_labels):
        pred_class, _ = dd_cls.predict_c(vol)
        if pred_class == true_label:
            correct += 1
    acc = correct / len(class_vols)
    print(f"Classification Accuracy: {acc:.4f} ({correct}/{len(class_vols)})")
    
    # === Multi-Label Classification Task ===
    print("\n" + "="*50)
    print("Multi-Label Classification Task (4 labels)")
    print("="*50)
    
    num_labels = 4
    label_vols = []
    labels = []
    for _ in range(50):
        D = random.randint(6, 8)
        H = random.randint(6, 8)
        W = random.randint(6, 8)
        vol = [[[random.choice(charset) for _ in range(W)] for _ in range(H)] for _ in range(D)]
        label_vols.append(vol)
        label_vec = [1.0 if random.random() > 0.7 else 0.0 for _ in range(num_labels)]
        labels.append(label_vec)
    
    dd_lbl = SpatialDualDescriptorAB3(
        charset, vec_dim=vec_dim, bas_dim_h=bas_dim_h, bas_dim_w=bas_dim_w, bas_dim_d=bas_dim_d,
        rank=rank, mode='linear', device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    loss_hist, acc_hist = dd_lbl.lbl_train(
        label_vols, labels, num_labels,
        max_iters=50, learning_rate=0.05, decay_rate=0.99, batch_size=8, print_every=10
    )
    print(f"Multi-label final training loss: {loss_hist[-1]:.6f}, accuracy: {acc_hist[-1]:.4f}")
    
    # === Self-Training Example ===
    print("\n" + "="*50)
    print("Self-Training (Gap Mode)")
    print("="*50)
    
    self_vols = []
    for _ in range(20):
        D = random.randint(6, 8)
        H = random.randint(6, 8)
        W = random.randint(6, 8)
        vol = [[[random.choice(charset) for _ in range(W)] for _ in range(H)] for _ in range(D)]
        self_vols.append(vol)
    
    dd_self = SpatialDualDescriptorAB3(
        charset, vec_dim=vec_dim, bas_dim_h=bas_dim_h, bas_dim_w=bas_dim_w, bas_dim_d=bas_dim_d,
        rank=rank, mode='linear', device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    self_history = dd_self.self_train(
        self_vols,
        max_iters=30,
        learning_rate=0.001,
        decay_rate=0.995,
        print_every=10,
        batch_size=128
    )
    
    # Reconstruct a small 3D volume from self-trained model
    print("\nReconstructing a 6x6x6 volume from self-trained model (deterministic):")
    recon_vol = dd_self.reconstruct_3d(D=6, H=6, W=6, tau=0.0)
    print("First depth layer (first 3 rows):")
    for row in recon_vol[0][:3]:
        print(''.join(row))
    print("...")
    
    # Save and load model
    print("\n=== Model Persistence Test ===")
    dd_self.save("self_trained_3d.pkl")
    dd_loaded = SpatialDualDescriptorAB3(
        charset, vec_dim=vec_dim, bas_dim_h=bas_dim_h, bas_dim_w=bas_dim_w, bas_dim_d=bas_dim_d,
        rank=rank, mode='linear', device='cuda' if torch.cuda.is_available() else 'cpu'
    ).load("self_trained_3d.pkl")
    print("Model loaded successfully. Reconstructing with loaded model:")
    recon_loaded = dd_loaded.reconstruct_3d(D=5, H=5, W=5, tau=0.0)
    print("First depth layer:")
    for row in recon_loaded[0][:3]:
        print(''.join(row))
    
    print("\n=== All 3D Tests Completed ===")
