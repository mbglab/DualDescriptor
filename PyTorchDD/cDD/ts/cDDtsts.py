# Copyright (C) 2005-2025, Bin-Guang Ma (mbg@mail.hzau.edu.cn); SPDX-License-Identifier: MIT
# Coupled Dual Descriptor Vector (Tensor‑Tensor) model – Optimized version
# Character‑level DualDescriptor (DDvTS) → Numerical‑level NumDualDescriptor (nDDts)
# Author: Bin-Guang Ma (assisted by DeepSeek); Date: 2026-05-06

import math
import random
import itertools
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy

class CoupledDualDescriptorTsTs(nn.Module):
    """
    Coupled Dual Descriptor: Character layer (DDvTS) + Numerical layer (nDDts)
    Fully optimised for GPU throughput while preserving exact mathematical equivalence.
    """
    def __init__(self,
                 charset,
                 # --- Character‑layer configuration ---
                 char_rank=1, char_rank_mode='drop', char_vec_dim=2, char_num_basis=5,
                 char_mode='linear', char_user_step=None,
                 # --- Numerical‑layer configuration ---
                 num_vec_dim=None,
                 num_rank=1, num_rank_op='avg', num_rank_mode='drop', num_num_basis=5,
                 num_mode='linear', num_user_step=None,
                 device='cuda'):
        super().__init__()
        self._device_str = device
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # ----- Character layer parameters -----
        self.charset = list(charset)
        self.char_rank = char_rank
        self.char_rank_mode = char_rank_mode
        assert char_rank_mode in ('pad', 'drop')
        self.m_char = char_vec_dim
        self.o_char = char_num_basis
        self.char_mode = char_mode
        assert char_mode in ('linear', 'nonlinear')
        self.char_step = char_user_step

        # ----- Numerical layer parameters -----
        self.num_rank = num_rank
        self.num_rank_op = num_rank_op
        self.num_rank_mode = num_rank_mode
        assert num_rank_op in ('avg', 'sum', 'pick', 'user_func')
        assert num_rank_mode in ('pad', 'drop')
        self.num_mode = num_mode
        assert num_mode in ('linear', 'nonlinear')
        self.num_step = num_user_step
        self.m_num = num_vec_dim if num_vec_dim is not None else self.m_char
        self.o_num = num_num_basis

        # ----- Projection (if dimensions differ) -----
        if self.m_char != self.m_num:
            self.projection = nn.Linear(self.m_char, self.m_num, bias=False)
        else:
            self.projection = None

        # ----- Character tokens & embedding -----
        toks = []
        if self.char_rank_mode == 'pad':
            for r in range(1, self.char_rank + 1):
                for prefix in itertools.product(self.charset, repeat=r):
                    tok = ''.join(prefix).ljust(self.char_rank, '_')
                    toks.append(tok)
        else:
            toks = [''.join(p) for p in itertools.product(self.charset, repeat=self.char_rank)]
        self.tokens = sorted(set(toks))
        self.token_to_idx = {t: i for i, t in enumerate(self.tokens)}
        self.idx_to_token = {i: t for i, t in enumerate(self.tokens)}
        self.embedding = nn.Embedding(len(self.tokens), self.m_char)

        # ----- Character layer basis tensor P_char -----
        self.P_char = nn.Parameter(torch.empty(self.m_char, self.m_char, self.o_char))

        # ----- Numerical layer matrix M and basis tensor P_num -----
        self.M = nn.Linear(self.m_num, self.m_num, bias=False)
        self.P_num = nn.Parameter(torch.empty(self.m_num, self.m_num, self.o_num))

        # ----- Precomputed periods and their inverses (fixed) -----
        periods_char = torch.zeros(self.m_char, self.m_char, self.o_char, dtype=torch.float32)
        periods_num = torch.zeros(self.m_num, self.m_num, self.o_num, dtype=torch.float32)
        for i in range(self.m_char):
            for j in range(self.m_char):
                for g in range(self.o_char):
                    periods_char[i, j, g] = i * (self.m_char * self.o_char) + j * self.o_char + g + 2
        for i in range(self.m_num):
            for j in range(self.m_num):
                for g in range(self.o_num):
                    periods_num[i, j, g] = i * (self.m_num * self.o_num) + j * self.o_num + g + 2
        self.register_buffer('periods_char', periods_char)
        self.register_buffer('periods_num', periods_num)
        self.register_buffer('inv_periods_char', 1.0 / periods_char)
        self.register_buffer('inv_periods_num', 1.0 / periods_num)

        # ----- Classification / Label heads -----
        self.num_classes = None
        self.classifier = None
        self.num_labels = None
        self.labeller = None

        # User function for numerical rank_op
        self.user_func = None

        # Training statistics
        self.trained = False
        self.mean_t_char = None
        self.mean_t_num = None
        self.mean_token_count = 0.0

        # Initialise parameters
        self.reset_parameters()
        self.to(self.device)

        # ---- torch.compile with safe fallback for older GPUs ----
        self._compile_enabled = False
        if hasattr(torch, 'compile'):
            if self.device.type == 'cuda':
                major, _ = torch.cuda.get_device_capability(self.device)
                if major >= 7:
                    self._compile_enabled = True
            else:
                self._compile_enabled = True
            if self._compile_enabled:
                try:
                    dummy = nn.Linear(1, 1).to(self.device)
                    torch.compile(dummy, mode="reduce-overhead")(torch.randn(1, 1, device=self.device))
                except Exception:
                    self._compile_enabled = False

        if self._compile_enabled:
            self._forward_single_seq_fn = torch.compile(self._forward_single_seq, mode="reduce-overhead")
            self._char_Nk_batched_fn = torch.compile(self._char_Nk_batched, mode="reduce-overhead")
        else:
            self._forward_single_seq_fn = self._forward_single_seq
            self._char_Nk_batched_fn = self._char_Nk_batched

    # ---------------------------------------------------------------------
    # Parameter initialisation
    # ---------------------------------------------------------------------
    def reset_parameters(self):
        nn.init.uniform_(self.embedding.weight, -0.5, 0.5)
        nn.init.uniform_(self.P_char, -0.1, 0.1)
        nn.init.uniform_(self.M.weight, -0.5, 0.5)
        nn.init.uniform_(self.P_num, -0.1, 0.1)
        if self.projection is not None:
            nn.init.uniform_(self.projection.weight, -0.5, 0.5)
        if self.classifier is not None:
            nn.init.normal_(self.classifier.weight, 0, 0.01)
            if self.classifier.bias is not None:
                nn.init.constant_(self.classifier.bias, 0)
        if self.labeller is not None:
            nn.init.xavier_uniform_(self.labeller.weight)
            nn.init.zeros_(self.labeller.bias)

    def set_user_func(self, func):
        if callable(func):
            self.user_func = func
        else:
            raise ValueError("User function must be callable")

    # ---------------------------------------------------------------------
    # Character‑layer tokenisation
    # ---------------------------------------------------------------------
    def _extract_tokens(self, seq):
        L = len(seq)
        if self.char_mode == 'linear':
            return [seq[i:i+self.char_rank] for i in range(L - self.char_rank + 1)]
        toks = []
        step = self.char_step or self.char_rank
        for i in range(0, L, step):
            frag = seq[i:i+self.char_rank]
            frag_len = len(frag)
            if self.char_rank_mode == 'pad':
                toks.append(frag if frag_len == self.char_rank else frag.ljust(self.char_rank, '_'))
            else:  # drop
                if frag_len == self.char_rank:
                    toks.append(frag)
        return toks

    def _token_list_to_indices(self, token_list):
        return torch.tensor([self.token_to_idx[t] for t in token_list], device=self.device)

    # ---------------------------------------------------------------------
    # Character‑layer N(k) – core computation (batched version)
    # ---------------------------------------------------------------------
    def _char_Nk_batched(self, k_flat, indices_flat):
        x = self.embedding(indices_flat)                     # (N, m_char)
        k_exp = k_flat.view(-1, 1, 1, 1)
        arg = 2 * math.pi * k_exp * self.inv_periods_char    # (N, m_char, m_char, o_char)
        phi = torch.cos(arg)
        Nk = torch.einsum('bj,ijg,bijg->bi', x, self.P_char, phi)
        return Nk

    def _char_Nk(self, k_tensor, token_indices):
        return self._char_Nk_batched(k_tensor, token_indices)

    # ---------------------------------------------------------------------
    # Numerical layer – vectorised window extraction and rank operation
    # ---------------------------------------------------------------------
    def _extract_num_vectors(self, vectors):
        L = vectors.shape[0]
        if L == 0:
            return torch.empty(0, self.m_num, device=self.device)

        def apply_op_vectorized(windows):
            if self.num_rank_op == 'sum':
                return windows.sum(dim=1)
            elif self.num_rank_op == 'avg':
                return windows.mean(dim=1)
            elif self.num_rank_op == 'pick':
                idx = torch.randint(0, self.num_rank, (windows.shape[0],), device=self.device)
                return windows[torch.arange(windows.shape[0]), idx, :]
            elif self.num_rank_op == 'user_func':
                if self.user_func is not None:
                    return torch.stack([self.user_func(windows[i]) for i in range(windows.shape[0])])
                else:
                    return torch.sigmoid(windows.mean(dim=1))

        if self.num_mode == 'linear':
            windows = vectors.unfold(0, self.num_rank, 1)   # (L-num_rank+1, m_num, num_rank)
            windows = windows.transpose(1, 2)               # (L-num_rank+1, num_rank, m_num)
            return apply_op_vectorized(windows)

        step = self.num_step or self.num_rank
        if step == self.num_rank and self.num_rank_mode == 'drop':
            valid_len = (L // self.num_rank) * self.num_rank
            if valid_len == 0:
                return torch.empty(0, self.m_num, device=self.device)
            windows = vectors[:valid_len].view(-1, self.num_rank, self.m_num)
            return apply_op_vectorized(windows)

        windows_list = []
        for i in range(0, L, step):
            frag = vectors[i:i+self.num_rank]
            frag_len = frag.shape[0]
            if self.num_rank_mode == 'pad' and frag_len < self.num_rank:
                padding = torch.zeros(self.num_rank - frag_len, self.m_num, device=self.device)
                frag = torch.cat([frag, padding], dim=0)
                windows_list.append(frag)
            elif frag_len == self.num_rank:
                windows_list.append(frag)
        if not windows_list:
            return torch.empty(0, self.m_num, device=self.device)
        windows = torch.stack(windows_list)
        return apply_op_vectorized(windows)

    # ---------------------------------------------------------------------
    # Numerical layer N(k) computation
    # ---------------------------------------------------------------------
    def _num_Nk(self, k_tensor, vectors):
        x = self.M(vectors)
        k_exp = k_tensor.view(-1, 1, 1, 1)
        arg = 2 * math.pi * k_exp * self.inv_periods_num
        phi = torch.cos(arg)
        Nk = torch.einsum('bj,ijg,bijg->bi', x, self.P_num, phi)
        return Nk

    # ---------------------------------------------------------------------
    # Single‑sequence forward (used for inference only)
    # ---------------------------------------------------------------------
    def _forward_single_seq(self, seq):
        toks = self._extract_tokens(seq)
        if not toks:
            return torch.empty(0, self.m_num, device=self.device)
        indices = self._token_list_to_indices(toks)
        k = torch.arange(len(toks), dtype=torch.float32, device=self.device)
        Nk_char = self._char_Nk(k, indices)
        if self.projection is not None:
            vecs = self.projection(Nk_char)
        else:
            vecs = Nk_char
        extracted = self._extract_num_vectors(vecs)
        if extracted.shape[0] == 0:
            return torch.empty(0, self.m_num, device=self.device)
        k_num = torch.arange(extracted.shape[0], dtype=torch.float32, device=self.device)
        Nk_num = self._num_Nk(k_num, extracted)
        return Nk_num

    def _single_seq_N(self, seq):
        """Returns detached tensor for inference."""
        return self._forward_single_seq_fn(seq).detach()

    # ---------------------------------------------------------------------
    # Batched forward – used in training loops
    # ---------------------------------------------------------------------
    def _forward_batch(self, seqs):
        all_tokens = [self._extract_tokens(s) for s in seqs]
        lengths = [len(t) for t in all_tokens]
        valid_mask = [l > 0 for l in lengths]
        if not any(valid_mask):
            return [], []
        flat_indices, flat_k = [], []
        for tokens in all_tokens:
            for pos, tok in enumerate(tokens):
                flat_indices.append(self.token_to_idx[tok])
                flat_k.append(pos)
        if not flat_indices:
            return [], []
        indices_tensor = torch.tensor(flat_indices, device=self.device)
        k_tensor = torch.tensor(flat_k, dtype=torch.float32, device=self.device)

        Nk_char_all = self._char_Nk_batched_fn(k_tensor, indices_tensor)

        cum_lengths = [0] + list(torch.tensor(lengths).cumsum(0).tolist())
        char_splits = [Nk_char_all[cum_lengths[i]:cum_lengths[i+1]] for i in range(len(seqs))]

        valid_idx, Nk_num_list = [], []
        for i, char_vecs in enumerate(char_splits):
            if not valid_mask[i] or char_vecs.shape[0] == 0:
                continue
            if self.projection is not None:
                vecs = self.projection(char_vecs)
            else:
                vecs = char_vecs
            extracted = self._extract_num_vectors(vecs)
            if extracted.shape[0] == 0:
                continue
            k_num = torch.arange(extracted.shape[0], dtype=torch.float32, device=self.device)
            Nk_num = self._num_Nk(k_num, extracted)
            Nk_num_list.append(Nk_num)
            valid_idx.append(i)
        return Nk_num_list, valid_idx

    # ---------------------------------------------------------------------
    # Public methods (inference, rely on detached tensors)
    # ---------------------------------------------------------------------
    def describe(self, seq):
        return self._single_seq_N(seq).cpu().numpy()

    def S(self, seq):
        Nk = self._single_seq_N(seq)
        if Nk.shape[0] == 0:
            return []
        S_cum = torch.cumsum(Nk, dim=0)
        return [s.cpu().numpy() for s in S_cum]

    def D(self, seqs, t_list):
        total_loss, total_pos = 0.0, 0
        t_tensors = [torch.tensor(t, dtype=torch.float32, device=self.device) for t in t_list]
        for seq, t in zip(seqs, t_tensors):
            Nk = self._single_seq_N(seq)
            if Nk.shape[0] == 0:
                continue
            total_loss += torch.sum((Nk - t) ** 2).item()
            total_pos += Nk.shape[0]
        return total_loss / total_pos if total_pos else 0.0

    def d(self, seq, t):
        return self.D([seq], [t])

    def predict_t(self, seq):
        Nk = self._single_seq_N(seq)
        if Nk.shape[0] == 0:
            return np.zeros(self.m_num, dtype=np.float32)
        return torch.mean(Nk, dim=0).cpu().numpy()

    def predict_c(self, seq):
        if self.classifier is None:
            raise ValueError("Model must be trained for classification first.")
        vec = torch.tensor(self.predict_t(seq), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits = self.classifier(vec.unsqueeze(0))
            probs = torch.softmax(logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()
        return pred, probs[0].cpu().numpy()

    def predict_l(self, seq, threshold=0.5):
        if self.labeller is None:
            raise ValueError("Model must be trained for multi‑label prediction first.")
        Nk = self._single_seq_N(seq)
        if Nk.shape[0] == 0:
            return np.zeros(self.num_labels, dtype=np.float32), np.zeros(self.num_labels, dtype=np.float32)
        seq_repr = torch.mean(Nk, dim=0)
        with torch.no_grad():
            logits = self.labeller(seq_repr)
            probs = torch.sigmoid(logits).cpu().numpy()
        binary = (probs > threshold).astype(np.float32)
        return binary, probs

    # ---------------------------------------------------------------------
    # Optimised training loops
    # ---------------------------------------------------------------------
    def reg_train(self, seqs, t_list, max_iters=1000, tol=1e-8, learning_rate=0.01,
                  continued=False, decay_rate=1.0, print_every=10, batch_size=32,
                  checkpoint_file=None, checkpoint_interval=10):
        if not continued:
            self.reset_parameters()
        t_tensors = [torch.tensor(t, dtype=torch.float32, device=self.device) for t in t_list]
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)
        history = []
        prev_loss = float('inf')
        best_loss, best_state = float('inf'), None

        for it in range(max_iters):
            total_loss, total_seqs = 0.0, 0
            indices = list(range(len(seqs)))
            random.shuffle(indices)
            for start in range(0, len(indices), batch_size):
                batch_idx = indices[start:start+batch_size]
                batch_seqs = [seqs[i] for i in batch_idx]
                batch_t = [t_tensors[i] for i in batch_idx]
                optimizer.zero_grad()
                Nk_list, valid_idx = self._forward_batch(batch_seqs)
                if not Nk_list:
                    continue
                batch_loss = 0.0
                for j, Nk in enumerate(Nk_list):
                    target = batch_t[valid_idx[j]]
                    pred = torch.mean(Nk, dim=0)
                    batch_loss += torch.sum((pred - target) ** 2)
                batch_loss = batch_loss / len(Nk_list)
                batch_loss.backward()
                optimizer.step()
                total_loss += batch_loss.item() * len(Nk_list)
                total_seqs += len(Nk_list)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            avg_loss = total_loss / total_seqs if total_seqs else 0.0
            history.append(avg_loss)
            if avg_loss < best_loss:
                best_loss, best_state = avg_loss, copy.deepcopy(self.state_dict())
            if it % print_every == 0 or it == max_iters-1:
                print(f"GD Iter {it:3d}: Loss = {avg_loss:.6e}, LR = {scheduler.get_last_lr()[0]:.6f}")
            if checkpoint_file and (it % checkpoint_interval == 0 or it == max_iters-1):
                self._save_checkpoint(checkpoint_file, it, history, optimizer, scheduler, best_loss)
            if abs(prev_loss - avg_loss) < tol:
                print(f"Converged after {it+1} iterations.")
                if best_state:
                    self.load_state_dict(best_state)
                break
            prev_loss = avg_loss
            scheduler.step()
        self._compute_training_statistics(seqs)
        self.trained = True
        return history

    def cls_train(self, seqs, labels, num_classes, max_iters=1000, tol=1e-8, learning_rate=0.01,
                  continued=False, decay_rate=1.0, print_every=10, batch_size=32,
                  checkpoint_file=None, checkpoint_interval=10):
        if self.classifier is None or self.num_classes != num_classes:
            self.classifier = nn.Linear(self.m_num, num_classes).to(self.device)
            self.num_classes = num_classes
        if not continued:
            self.reset_parameters()
        label_tensors = torch.tensor(labels, dtype=torch.long, device=self.device)
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)
        criterion = nn.CrossEntropyLoss()
        history = []
        prev_loss = float('inf')
        best_loss, best_state = float('inf'), None

        for it in range(max_iters):
            total_loss, total_seqs, correct = 0.0, 0, 0
            indices = list(range(len(seqs)))
            random.shuffle(indices)
            for start in range(0, len(indices), batch_size):
                batch_idx = indices[start:start+batch_size]
                batch_seqs = [seqs[i] for i in batch_idx]
                batch_labels = label_tensors[batch_idx]
                optimizer.zero_grad()
                Nk_list, valid_idx = self._forward_batch(batch_seqs)
                if not Nk_list:
                    continue
                logits_list = []
                for Nk in Nk_list:
                    seq_vec = torch.mean(Nk, dim=0)
                    logits_list.append(self.classifier(seq_vec.unsqueeze(0)))
                all_logits = torch.cat(logits_list, dim=0)
                loss = criterion(all_logits, batch_labels[valid_idx])
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(valid_idx)
                total_seqs += len(valid_idx)
                with torch.no_grad():
                    preds = torch.argmax(all_logits, dim=1)
                    correct += (preds == batch_labels[valid_idx]).sum().item()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            avg_loss = total_loss / total_seqs if total_seqs else 0.0
            acc = correct / total_seqs if total_seqs else 0.0
            history.append(avg_loss)
            if avg_loss < best_loss:
                best_loss, best_state = avg_loss, copy.deepcopy(self.state_dict())
            if it % print_every == 0 or it == max_iters-1:
                print(f"CLS-Train Iter {it:3d}: Loss = {avg_loss:.6e}, Acc = {acc:.4f}, LR = {scheduler.get_last_lr()[0]:.6f}")
            if checkpoint_file and (it % checkpoint_interval == 0 or it == max_iters-1):
                self._save_checkpoint(checkpoint_file, it, history, optimizer, scheduler, best_loss)
            if abs(prev_loss - avg_loss) < tol:
                print(f"Converged after {it+1} iterations.")
                if best_state:
                    self.load_state_dict(best_state)
                break
            prev_loss = avg_loss
            scheduler.step()
        self.trained = True
        return history

    def lbl_train(self, seqs, labels, num_labels, max_iters=1000, tol=1e-8, learning_rate=0.01,
                  continued=False, decay_rate=1.0, print_every=10, batch_size=32,
                  checkpoint_file=None, checkpoint_interval=10, pos_weight=None):
        if self.labeller is None or self.num_labels != num_labels:
            self.labeller = nn.Linear(self.m_num, num_labels).to(self.device)
            self.num_labels = num_labels
        if not continued:
            self.reset_parameters()
        labels_tensor = torch.tensor(labels, dtype=torch.float32, device=self.device) if isinstance(labels, list) else torch.as_tensor(labels, dtype=torch.float32, device=self.device)
        if pos_weight is not None:
            pos_w = torch.tensor(pos_weight, dtype=torch.float32, device=self.device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
        else:
            criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)
        loss_hist, acc_hist = [], []
        prev_loss, best_loss, best_state = float('inf'), float('inf'), None

        for it in range(max_iters):
            total_loss, total_correct, total_preds, total_seqs = 0.0, 0, 0, 0
            indices = list(range(len(seqs)))
            random.shuffle(indices)
            for start in range(0, len(indices), batch_size):
                batch_idx = indices[start:start+batch_size]
                batch_seqs = [seqs[i] for i in batch_idx]
                batch_labels = labels_tensor[batch_idx]
                optimizer.zero_grad()
                Nk_list, valid_idx = self._forward_batch(batch_seqs)
                if not Nk_list:
                    continue
                logits_list = [self.labeller(torch.mean(Nk, dim=0)) for Nk in Nk_list]
                batch_logits = torch.stack(logits_list, dim=0)
                loss = criterion(batch_logits, batch_labels[valid_idx])
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(valid_idx)
                total_seqs += len(valid_idx)
                with torch.no_grad():
                    probs = torch.sigmoid(batch_logits)
                    preds = (probs > 0.5).float()
                    total_correct += (preds == batch_labels[valid_idx]).sum().item()
                    total_preds += batch_labels[valid_idx].numel()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            avg_loss = total_loss / total_seqs if total_seqs else 0.0
            avg_acc = total_correct / total_preds if total_preds else 0.0
            loss_hist.append(avg_loss)
            acc_hist.append(avg_acc)
            if avg_loss < best_loss:
                best_loss, best_state = avg_loss, copy.deepcopy(self.state_dict())
            if it % print_every == 0 or it == max_iters-1:
                print(f"MLC-Train Iter {it:3d}: Loss = {avg_loss:.6e}, Acc = {avg_acc:.4f}, LR = {scheduler.get_last_lr()[0]:.6f}")
            if checkpoint_file and (it % checkpoint_interval == 0 or it == max_iters-1):
                self._save_checkpoint(checkpoint_file, it, loss_hist, optimizer, scheduler, best_loss)
            if abs(prev_loss - avg_loss) < tol:
                print(f"Converged after {it+1} iterations.")
                if best_state:
                    self.load_state_dict(best_state)
                break
            prev_loss = avg_loss
            scheduler.step()
        self.trained = True
        return loss_hist, acc_hist

    def self_train(self, seqs, max_iters=100, tol=1e-6, learning_rate=0.01,
                   continued=False, decay_rate=1.0, print_every=10,
                   batch_size=32, checkpoint_file=None, checkpoint_interval=5):
        if not continued:
            self.reset_parameters()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)
        history = []
        prev_loss, best_loss, best_state = float('inf'), float('inf'), None

        for it in range(max_iters):
            total_loss, total_samples = 0.0, 0
            indices = list(range(len(seqs)))
            random.shuffle(indices)
            for start in range(0, len(indices), batch_size):
                batch_idx = indices[start:start+batch_size]
                batch_seqs = [seqs[i] for i in batch_idx]
                optimizer.zero_grad()
                batch_loss, cnt = 0.0, 0
                for seq in batch_seqs:
                    toks = self._extract_tokens(seq)
                    if not toks: continue
                    indices_t = self._token_list_to_indices(toks)
                    k_char = torch.arange(len(toks), dtype=torch.float32, device=self.device)
                    Nk_char = self._char_Nk(k_char, indices_t)
                    if self.projection is not None:
                        vecs = self.projection(Nk_char)
                    else:
                        vecs = Nk_char
                    extracted = self._extract_num_vectors(vecs)
                    if extracted.shape[0] == 0: continue
                    k_num = torch.arange(extracted.shape[0], dtype=torch.float32, device=self.device)
                    Nk_num = self._num_Nk(k_num, extracted)
                    target = self.M(extracted)
                    seq_loss = torch.sum((Nk_num - target) ** 2) / extracted.shape[0]
                    batch_loss += seq_loss
                    cnt += 1
                if cnt > 0:
                    batch_loss = batch_loss / cnt
                    batch_loss.backward()
                    optimizer.step()
                    total_loss += batch_loss.item() * cnt
                    total_samples += cnt
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            avg_loss = total_loss / total_samples if total_samples else 0.0
            history.append(avg_loss)
            if avg_loss < best_loss:
                best_loss, best_state = avg_loss, copy.deepcopy(self.state_dict())
            if it % print_every == 0 or it == max_iters-1:
                print(f"Self-Train Iter {it:3d}: Loss = {avg_loss:.6f}, LR = {scheduler.get_last_lr()[0]:.6f}")
            if checkpoint_file and (it % checkpoint_interval == 0 or it == max_iters-1):
                self._save_checkpoint(checkpoint_file, it, history, optimizer, scheduler, best_loss)
            if abs(prev_loss - avg_loss) < tol:
                print(f"Converged after {it+1} iterations")
                if best_state:
                    self.load_state_dict(best_state)
                break
            prev_loss = avg_loss
            scheduler.step()
        self._compute_training_statistics(seqs)
        self.trained = True
        return history

    # ---------------------------------------------------------------------
    # Training statistics
    # ---------------------------------------------------------------------
    def _compute_training_statistics(self, seqs, batch_size=50):
        total_tokens = 0
        total_char = torch.zeros(self.m_char, device=self.device)
        total_num = torch.zeros(self.m_num, device=self.device)
        total_num_pos = 0
        with torch.no_grad():
            for i in range(0, len(seqs), batch_size):
                batch_seqs = seqs[i:i+batch_size]
                for seq in batch_seqs:
                    toks = self._extract_tokens(seq)
                    if not toks: continue
                    indices = self._token_list_to_indices(toks)
                    k = torch.arange(len(toks), dtype=torch.float32, device=self.device)
                    Nk_char = self._char_Nk(k, indices)
                    total_char += Nk_char.sum(dim=0)
                    total_tokens += len(toks)
                    if self.projection is not None:
                        vecs = self.projection(Nk_char)
                    else:
                        vecs = Nk_char
                    extracted = self._extract_num_vectors(vecs)
                    if extracted.shape[0] == 0: continue
                    k_num = torch.arange(extracted.shape[0], dtype=torch.float32, device=self.device)
                    Nk_num = self._num_Nk(k_num, extracted)
                    total_num += Nk_num.sum(dim=0)
                    total_num_pos += extracted.shape[0]
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        self.mean_token_count = total_tokens / len(seqs) if seqs else 0
        self.mean_t_char = (total_char / total_tokens).cpu().numpy() if total_tokens else np.zeros(self.m_char)
        self.mean_t_num = (total_num / total_num_pos).cpu().numpy() if total_num_pos else np.zeros(self.m_num)

    def _save_checkpoint(self, path, iteration, history, optimizer, scheduler, best_loss):
        checkpoint = {
            'iteration': iteration,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'history': history,
            'best_loss': best_loss,
            'mean_t_char': self.mean_t_char,
            'mean_t_num': self.mean_t_num,
            'mean_token_count': self.mean_token_count
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved at iteration {iteration}")

    # ---------------------------------------------------------------------
    # Reconstruction
    # ---------------------------------------------------------------------
    def reconstruct(self, L, tau=0.0):
        assert self.trained and self.mean_t_char is not None, "Model must be trained first"
        if tau < 0: raise ValueError("Temperature must be non-negative")
        num_blocks = (L + self.char_rank - 1) // self.char_rank
        mean_t = torch.tensor(self.mean_t_char, dtype=torch.float32, device=self.device)
        all_idx = torch.arange(len(self.tokens), device=self.device)
        all_emb = self.embedding(all_idx)
        generated = []
        for k in range(num_blocks):
            k_t = torch.tensor([k]*len(self.tokens), dtype=torch.float32, device=self.device)
            arg = 2*math.pi * k_t.view(-1,1,1,1) * self.inv_periods_char
            phi = torch.cos(arg)
            Nk_all = torch.einsum('bj,ijg,bijg->bi', all_emb, self.P_char, phi)
            scores = -torch.sum((Nk_all - mean_t) ** 2, dim=1)
            if tau == 0:
                best = torch.argmax(scores).item()
            else:
                probs = torch.softmax(scores / tau, dim=0).detach().cpu().numpy()
                best = random.choices(range(len(self.tokens)), weights=probs, k=1)[0]
            generated.append(self.idx_to_token[best])
        return ''.join(generated)[:L]

    def reconstruct_num(self, L, tau=0.0, num_candidates=100):
        assert self.trained and self.mean_t_num is not None, "Model must be trained first"
        if tau < 0: raise ValueError("Temperature must be non-negative")
        num_win = (L + self.num_rank - 1) // self.num_rank
        mean_t = torch.tensor(self.mean_t_num, dtype=torch.float32, device=self.device)
        cand = torch.randn(num_candidates, self.m_num, device=self.device)
        gen = []
        for k in range(num_win):
            k_t = torch.tensor([k]*num_candidates, dtype=torch.float32, device=self.device)
            arg = 2*math.pi * k_t.view(-1,1,1,1) * self.inv_periods_num
            x = self.M(cand)
            phi = torch.cos(arg)
            Nk = torch.einsum('bj,ijg,bijg->bi', x, self.P_num, phi)
            scores = -torch.sum((Nk - mean_t) ** 2, dim=1)
            if tau == 0:
                best = cand[torch.argmax(scores)]
            else:
                probs = torch.softmax(scores / tau, dim=0).cpu().numpy()
                best = cand[random.choices(range(num_candidates), weights=probs, k=1)[0]]
            gen.append(best)
        return torch.stack(gen)[:L].cpu().numpy()

    def save(self, filename):
        torch.save(self.state_dict(), filename)
        print(f"Model saved to {filename}")

    def load(self, filename):
        try:
            sd = torch.load(filename, map_location=self.device, weights_only=True)
        except TypeError:
            sd = torch.load(filename, map_location=self.device)
        self.load_state_dict(sd)
        print(f"Model loaded from {filename}")
        return self


# =========================================================================
# Example usage – detailed printout as in original DDvTS.py style
# =========================================================================
if __name__ == "__main__":
    from statistics import correlation

    print("="*60)
    print("Coupled Dual Descriptor TsTs – Optimized Demo")
    print("="*60)

    torch.manual_seed(11)
    random.seed(11)
    np.random.seed(11)

    charset = ['A', 'C', 'G', 'T']
    char_rank = 4
    char_vec_dim = 8
    char_num_basis = 6
    char_mode = 'nonlinear'
    char_user_step = 3

    num_vec_dim = 8
    num_rank = 2
    num_num_basis = 4
    num_mode = 'nonlinear'
    num_user_step = 2

    dd = CoupledDualDescriptorTsTs(
        charset,
        char_rank=char_rank, char_rank_mode='drop',
        char_vec_dim=char_vec_dim, char_num_basis=char_num_basis,
        char_mode=char_mode, char_user_step=char_user_step,
        num_vec_dim=num_vec_dim,
        num_rank=num_rank, num_rank_op='avg', num_rank_mode='drop',
        num_num_basis=num_num_basis,
        num_mode=num_mode, num_user_step=num_user_step,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    print(f"\nUsing device: {dd.device}")
    print(f"Character layer: rank={dd.char_rank}, vec_dim={dd.m_char}, num_basis={dd.o_char}, mode={dd.char_mode}, step={dd.char_step}")
    print(f"Numerical layer:  rank={dd.num_rank}, vec_dim={dd.m_num}, num_basis={dd.o_num}, mode={dd.num_mode}, step={dd.num_step}")
    print(f"Number of character tokens: {len(dd.tokens)}")

    # ----------------------------------------------------------------
    # 1. REGRESSION TASK
    # ----------------------------------------------------------------
    print("\n" + "="*60)
    print("1. Regression Training")
    print("="*60)

    seqs, t_list = [], []
    for _ in range(100):
        L = random.randint(200, 300)
        seq = ''.join(random.choices(charset, k=L))
        seqs.append(seq)
        t_list.append([random.uniform(-1.0, 1.0) for _ in range(dd.m_num)])

    print(f"Generated {len(seqs)} sequences with {dd.m_num}-dim targets.")

    print("\n" + "="*60)
    print("Starting Gradient Descent Training")
    print("="*60)
    dd.reg_train(seqs, t_list, max_iters=100, tol=1e-9, learning_rate=0.1,
                 decay_rate=0.99, batch_size=128, print_every=10)

    # Evaluate regression performance
    print("\nEvaluating regression performance...")
    pred_t_list = [dd.predict_t(seq) for seq in seqs]

    corr_sum = 0.0
    for dim in range(dd.m_num):
        actual = [t[dim] for t in t_list]
        pred = [p[dim] for p in pred_t_list]
        c = correlation(actual, pred)
        print(f"  Dimension {dim} prediction correlation: {c:.4f}")
        corr_sum += c
    print(f"  Average correlation: {corr_sum / dd.m_num:.4f}")

    # Reconstruction
    print("\nCharacter sequence reconstruction:")
    seq_det = dd.reconstruct(L=100, tau=0.0)
    seq_rand = dd.reconstruct(L=100, tau=0.5)
    print("  Deterministic reconstruction (first 50):", seq_det[:50] + "...")
    print("  Stochastic reconstruction (first 50):   ", seq_rand[:50] + "...")

    # ----------------------------------------------------------------
    # 2. MULTI‑CLASS CLASSIFICATION
    # ----------------------------------------------------------------
    print("\n" + "="*60)
    print("2. Multi‑Class Classification")
    print("="*60)

    num_classes = 3
    class_seqs, class_labels = [], []
    for class_id in range(num_classes):
        for _ in range(50):
            L = random.randint(150, 250)
            if class_id == 0:
                seq = ''.join(random.choices(charset, weights=[0.6, 0.1, 0.1, 0.2], k=L))
            elif class_id == 1:
                seq = ''.join(random.choices(charset, weights=[0.1, 0.4, 0.4, 0.1], k=L))
            else:
                seq = ''.join(random.choices(charset, k=L))
            class_seqs.append(seq)
            class_labels.append(class_id)

    dd_cls = CoupledDualDescriptorTsTs(
        charset,
        char_rank=char_rank, char_rank_mode='drop',
        char_vec_dim=char_vec_dim, char_num_basis=char_num_basis,
        char_mode=char_mode, char_user_step=char_user_step,
        num_vec_dim=num_vec_dim,
        num_rank=num_rank, num_rank_op='avg', num_rank_mode='drop',
        num_num_basis=num_num_basis,
        num_mode=num_mode, num_user_step=num_user_step,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    print(f"Training on {len(class_seqs)} sequences, {num_classes} classes.")
    print("\n" + "="*60)
    print("Starting Classification Training")
    print("="*60)
    cls_hist = dd_cls.cls_train(class_seqs, class_labels, num_classes,
                                max_iters=50, tol=1e-8, learning_rate=0.05,
                                decay_rate=0.99, batch_size=32, print_every=5)

    # Evaluate classification
    correct = 0
    all_preds = []
    for seq, true_label in zip(class_seqs, class_labels):
        pred_class, probs = dd_cls.predict_c(seq)
        all_preds.append(pred_class)
        if pred_class == true_label:
            correct += 1
    acc = correct / len(class_seqs)
    print(f"\nClassification accuracy: {acc:.4f} ({correct}/{len(class_seqs)})")

    # Example predictions
    print("\nExample predictions:")
    for i in range(min(5, len(class_seqs))):
        pred_class, probs = dd_cls.predict_c(class_seqs[i])
        print(f"  Seq {i+1}: True={class_labels[i]}, Pred={pred_class}, Probs={[f'{p:.3f}' for p in probs]}")

    # ----------------------------------------------------------------
    # 3. MULTI‑LABEL CLASSIFICATION
    # ----------------------------------------------------------------
    print("\n" + "="*60)
    print("3. Multi‑Label Classification")
    print("="*60)

    num_labels = 4
    label_seqs, labels = [], []
    for _ in range(100):
        L = random.randint(200, 300)
        seq = ''.join(random.choices(charset, k=L))
        label_seqs.append(seq)
        label_vec = [1.0 if random.random() > 0.7 else 0.0 for _ in range(num_labels)]
        labels.append(label_vec)

    dd_lbl = CoupledDualDescriptorTsTs(
        charset,
        char_rank=char_rank, char_rank_mode='drop',
        char_vec_dim=char_vec_dim, char_num_basis=char_num_basis,
        char_mode=char_mode, char_user_step=char_user_step,
        num_vec_dim=num_vec_dim,
        num_rank=num_rank, num_rank_op='avg', num_rank_mode='drop',
        num_num_basis=num_num_basis,
        num_mode=num_mode, num_user_step=num_user_step,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    print(f"Training multi‑label model: {num_labels} labels on {len(label_seqs)} sequences.")
    print("\n" + "="*60)
    print("Starting Multi‑Label Classification Training")
    print("="*60)
    loss_hist, acc_hist = dd_lbl.lbl_train(
        label_seqs, labels, num_labels,
        max_iters=50, tol=1e-16, learning_rate=0.05,
        decay_rate=0.99, print_every=10, batch_size=32
    )

    print(f"\nFinal training loss: {loss_hist[-1]:.6f}")
    print(f"Final training accuracy: {acc_hist[-1]:.4f}")

    # Detailed evaluation
    all_correct = 0
    total = 0
    label_names = ["Function_A", "Function_B", "Function_C", "Function_D"]

    for seq, true_labels in zip(label_seqs, labels):
        pred_bin, pred_prob = dd_lbl.predict_l(seq, threshold=0.5)
        true_np = np.array(true_labels)
        exact_match = np.all(pred_bin == true_np)
        all_correct += exact_match
        total += 1
        if total <= 3:
            print(f"\nSequence {total}:")
            print(f"  True labels:      {true_np}")
            print(f"  Predicted binary: {pred_bin}")
            print(f"  Probabilities:    {[f'{p:.3f}' for p in pred_prob]}")
            print(f"  Exact match: {exact_match}")

    print(f"\nExact‑match accuracy: {all_correct/total:.4f} ({all_correct}/{total})")

    # New sequence prediction
    test_seq = ''.join(random.choices(charset, k=250))
    print(f"\nTest sequence (first 50): {test_seq[:50]}...")
    bin_pred, prob_pred = dd_lbl.predict_l(test_seq, threshold=0.5)
    print(f"Predicted binary labels: {bin_pred}")
    print(f"Predicted probabilities: {[f'{p:.4f}' for p in prob_pred]}")
    print("Label interpretation:")
    for i, (b, p) in enumerate(zip(bin_pred, prob_pred)):
        status = "ACTIVE" if b > 0.5 else "INACTIVE"
        print(f"  {label_names[i]}: {status} (confidence: {p:.4f})")

    # ----------------------------------------------------------------
    # 4. SELF‑TRAINING
    # ----------------------------------------------------------------
    print("\n" + "="*60)
    print("4. Self‑Training (Self‑Consistency)")
    print("="*60)

    self_seqs = [''.join(random.choices(charset, k=random.randint(200, 300))) for _ in range(10)]
    dd_self = CoupledDualDescriptorTsTs(
        charset,
        char_rank=char_rank, char_rank_mode='drop',
        char_vec_dim=char_vec_dim, char_num_basis=char_num_basis,
        char_mode=char_mode, char_user_step=char_user_step,
        num_vec_dim=num_vec_dim,
        num_rank=num_rank, num_rank_op='avg', num_rank_mode='drop',
        num_num_basis=num_num_basis,
        num_mode=num_mode, num_user_step=num_user_step,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    print(f"Self‑training on {len(self_seqs)} sequences.")
    print("\nStarting Self‑Training...")
    self_hist = dd_self.self_train(self_seqs, max_iters=50, tol=1e-8,
                                   learning_rate=0.01, batch_size=4, print_every=10)

    # Reconstruct from self‑trained model
    print("\nReconstructed sequences from self‑trained model:")
    for i in range(2):
        s = dd_self.reconstruct(L=100, tau=0.2)
        print(f"  Sequence {i+1} (first 50): {s[:50]}...")

    print("\n" + "="*60)
    print("All tests completed successfully!")
    print("="*60)
