# Copyright (C) 2005-2026, Bin-Guang Ma (mbg@mail.hzau.edu.cn); SPDX-License-Identifier: MIT
# The Numerical Dual Descriptor Vector class (P Matrix form) implemented with PyTorch
# This program is for the demonstration of methodology and not fully refined.
# Author: Bin-Guang Ma (assisted by DeepSeek); Date: 2025-8-28 ~ 2026-10-9
#
# Two interchangeable regression schemes are provided:
#   (a) grad_train + predict_t / t_generate : the m-dimensional model output is fitted directly
#   (b) reg_train  + predict_r / r_generate : a trainable linear regression head maps m -> target_dim
#
# Sequence generation methods (all assume rank_mode != 'pad' and a trained model):
#   * generate(L, tau)         : after self_train  – target = global mean_t
#   * t_generate(L, tau, t)    : after grad_train  – target vector t (default: mean_t)
#   * r_generate(L, r, tau)    : after reg_train   – target vector r (required)
#   * c_generate(L, c, tau)    : after cls_train   – target class c (required)
#   * l_generate(L, l, tau)    : after lbl_train   – target multi-label vector l (required)
#
# Generation of a vector sequence means producing L vectors of dimension vec_dim. It is
# performed by gradient-optimizing the window tensors so that N(k) matches the target at
# each window position, then stitching the (possibly overlapping) windows by averaging
# the contributions of all windows that cover a given position.

import math
import random
import itertools
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy


class NumDualDescriptorPM(nn.Module):
    """
    Numerical (vector-sequence) Dual Descriptor (P Matrix form) with GPU acceleration:
      - input is a sequence of real m-dimensional vectors instead of characters
      - tensor P ∈ R^{m×m} of basis coefficients (2D matrix form; num_basis fixed to 1)
      - a trainable square linear map M ∈ R^{m×m} replaces the token embedding;
        it is applied to each extracted window vector before the basis expansion
      - indexed periods: period[i,j] = i*m + j + 2
      - basis function phi_{i,j}(k) = cos(2π * k / period[i,j])
      - supports 'linear' or 'nonlinear' (step-by-rank) window extraction
      - rank_op reduces each rank-length window of vectors into a single vector:
        'avg', 'sum', 'max', or a user-supplied callable
      - two interchangeable regression schemes, multi-class classification, and
        multi-label classification, in one-to-one correspondence with DDvTS.py
    """

    # Maximum number of windows handled by a single chunk of batch_compute_Nk.
    # The intermediate phi tensor has shape [windows, m, m]; chunking keeps its
    # memory bounded independently of the training batch size.
    NK_CHUNK = 16384

    def __init__(self, vec_dim, rank=1, rank_op='avg', rank_mode='drop',
                 mode='linear', user_step=None, device='cuda'):
        """
        Initialize the Numerical Dual Descriptor model (P Matrix form).

        Args:
            vec_dim (int): Dimension of input vectors and internal representation (m)
            rank (int): Length of the vector window (r-per / k-mer length)
            rank_op (str): 'avg', 'sum', 'max', or 'user_func' – how to reduce a window
            rank_mode (str): 'pad' or 'drop' – how to handle incomplete fragments
            mode (str): 'linear' or 'nonlinear' – window extraction mode
            user_step (int, optional): Step size for nonlinear extraction
            device (str): 'cuda' or 'cpu'
        """
        super().__init__()
        self.vec_dim = vec_dim
        self.rank = rank
        self.rank_op = rank_op
        self.rank_mode = rank_mode
        self.m = vec_dim
        assert mode in ('linear', 'nonlinear')
        self.mode = mode
        self.step = user_step

        # User function for the 'user_func' rank operation (set via set_user_func)
        self.user_func = None

        # Training statistics and the trained flag are buffers, so save() / load()
        # preserves them and a reloaded model can reconstruct immediately.
        self.register_buffer('_trained', torch.zeros(1, dtype=torch.bool))
        self.register_buffer('_mean_t', torch.zeros(self.m))
        self.register_buffer('_mean_vector_count', torch.zeros(1, dtype=torch.float64))
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # Trainable square linear map applied to each extracted window vector.
        # This replaces the character-based token embedding.
        self.M = nn.Linear(self.vec_dim, self.m, bias=False)

        # Position-weight matrix P[i][j]
        self.P = nn.Parameter(torch.empty(self.m, self.m))

        # Indexed periods[i][j] (fixed, not trainable): a pure function of m,
        # rebuilt in the constructor and never saved as part of state_dict.
        periods = torch.zeros(self.m, self.m, dtype=torch.float32)
        for i in range(self.m):
            for j in range(self.m):
                periods[i, j] = i * self.m + j + 2
        self.register_buffer('periods', periods, persistent=False)

        # Pre-scaled angular frequency omega[i,j] = 2*pi / period[i,j], so the basis
        # function becomes cos(k * omega) with a single multiply inside cos().
        # Computed in float64 and rounded once; also derived state, non-persistent.
        self.register_buffer('omega', ((2 * math.pi) / periods.double()).float(),
                             persistent=False)

        # Lazily created prediction heads
        self.num_classes = None
        self.classifier = None
        self.num_labels = None
        self.labeller = None
        self.target_dim = None
        self.regresser = None

        self.reset_parameters()
        self.to(self.device)

    # ---------- buffer accessors ----------
    @property
    def trained(self):
        """True once a training method has fitted this model (persisted)."""
        return bool(self._trained.item())

    @trained.setter
    def trained(self, value):
        self._trained.fill_(bool(value))

    @property
    def mean_t(self):
        """Mean of N(k) over all training windows, as a numpy array (persisted)."""
        return self._mean_t.detach().cpu().numpy()

    @mean_t.setter
    def mean_t(self, value):
        flat = torch.as_tensor(value, dtype=torch.float32).detach().flatten()
        if flat.numel() != self.m:
            raise ValueError(f"mean_t must have {self.m} elements, got {flat.numel()}")
        self._mean_t.copy_(flat.to(self._mean_t.device))

    @property
    def mean_vector_count(self):
        """Average number of extracted windows per training sequence (persisted)."""
        return float(self._mean_vector_count.item())

    @mean_vector_count.setter
    def mean_vector_count(self, value):
        self._mean_vector_count.fill_(float(value))

    def set_user_func(self, func):
        """Set a custom user function for the 'user_func' rank operation."""
        if callable(func):
            self.user_func = func
        else:
            raise ValueError("User function must be callable")

    def reset_parameters(self):
        """Initialize model parameters with appropriate distributions."""
        nn.init.uniform_(self.M.weight, -0.5, 0.5)
        nn.init.uniform_(self.P, -0.1, 0.1)
        if self.classifier is not None:
            nn.init.normal_(self.classifier.weight, 0, 0.01)
            if self.classifier.bias is not None:
                nn.init.constant_(self.classifier.bias, 0)
        if self.labeller is not None:
            nn.init.xavier_uniform_(self.labeller.weight)
            nn.init.zeros_(self.labeller.bias)
        if self.regresser is not None:
            nn.init.xavier_uniform_(self.regresser.weight)
            nn.init.zeros_(self.regresser.bias)

    # ---------- window extraction & N(k) ----------
    def _apply_op(self, windows):
        """
        Apply rank_op to a batch of vector windows.

        Args:
            windows (Tensor): shape [..., rank, vec_dim]

        Returns:
            Tensor of shape [..., vec_dim]
        """
        if self.rank_op == 'sum':
            return windows.sum(dim=-2)
        elif self.rank_op == 'avg':
            return windows.mean(dim=-2)
        elif self.rank_op == 'max':
            return windows.max(dim=-2).values
        elif self.rank_op == 'user_func':
            if self.user_func is not None and callable(self.user_func):
                shape = windows.shape[:-2]
                flat = windows.reshape(-1, windows.shape[-2], self.vec_dim)
                outs = [self.user_func(flat[i]) for i in range(flat.shape[0])]
                return torch.stack(outs).reshape(*shape, self.vec_dim)
            else:
                return torch.sigmoid(windows.mean(dim=-2))
        else:
            raise ValueError(f"Unknown rank_op: {self.rank_op}")

    def extract_vectors(self, seq_vectors):
        """
        Extract window vectors from a vector sequence according to processing mode and
        rank operation.

        - 'linear': slide a window of length rank by step 1.
        - 'nonlinear': slide by custom step (or rank if step is not specified).

        For nonlinear mode, incomplete trailing fragments are handled as:
        - 'pad': pad with zero vectors to maintain window length
        - 'drop': discard incomplete fragments

        Args:
            seq_vectors (list or tensor): Input vector sequence, shape [L, vec_dim]

        Returns:
            Tensor of shape [num_windows, vec_dim] on self.device.
        """
        if not isinstance(seq_vectors, torch.Tensor):
            seq_vectors = torch.tensor(seq_vectors, dtype=torch.float32, device=self.device)
        else:
            seq_vectors = seq_vectors.to(self.device)
            if seq_vectors.dtype != torch.float32:
                seq_vectors = seq_vectors.float()

        L = seq_vectors.shape[0]
        if L == 0:
            return torch.empty(0, self.vec_dim, device=self.device)

        if self.mode == 'linear':
            if L < self.rank:
                return torch.empty(0, self.vec_dim, device=self.device)
            windows = seq_vectors.unfold(0, self.rank, 1).transpose(1, 2)
            return self._apply_op(windows)
        else:
            step = self.step if self.step is not None else self.rank
            windows_list = []
            for i in range(0, L, step):
                frag = seq_vectors[i:i + self.rank]
                fl = frag.shape[0]
                if fl < self.rank:
                    if self.rank_mode == 'pad':
                        padding = torch.zeros(self.rank - fl, self.vec_dim, device=self.device)
                        frag = torch.cat([frag, padding], dim=0)
                        windows_list.append(frag)
                    # 'drop': discard incomplete fragment
                else:
                    windows_list.append(frag)
            if not windows_list:
                return torch.empty(0, self.vec_dim, device=self.device)
            windows = torch.stack(windows_list, dim=0)
            return self._apply_op(windows)

    def batch_compute_Nk(self, k_tensor, vectors):
        """
        Vectorized computation of N(k) vectors for a batch of window positions and
        window representations.

        Batches larger than NK_CHUNK are split into chunks: the intermediate tensor phi
        has shape [windows, m, m], so chunking bounds its memory without touching the
        reduction (over j), i.e. without changing the result.

        Args:
            k_tensor (Tensor): Position indices [batch_size]
            vectors (Tensor): Window representations [batch_size, vec_dim]

        Returns:
            Tensor of N(k) vectors [batch_size, m]
        """
        n = k_tensor.numel()
        if n <= self.NK_CHUNK:
            return self._compute_Nk_chunk(k_tensor, vectors)
        return torch.cat([self._compute_Nk_chunk(k_tensor[i:i + self.NK_CHUNK],
                                                 vectors[i:i + self.NK_CHUNK])
                          for i in range(0, n, self.NK_CHUNK)], dim=0)

    def _compute_Nk_chunk(self, k_tensor, vectors):
        """Compute N(k) for one chunk of windows (see batch_compute_Nk)."""
        x = self.M(vectors)                            # [chunk, m]
        k_expanded = k_tensor.view(-1, 1, 1)           # [chunk, 1, 1]
        phi = torch.cos(k_expanded * self.omega)       # [chunk, m, m]
        return torch.einsum('bj,ij,bij->bi', x, self.P, phi)

    def compute_Nk(self, k, vector):
        """Compute N(k) for a single position and a single window vector."""
        if not isinstance(vector, torch.Tensor):
            vector = torch.tensor(vector, dtype=torch.float32, device=self.device)
        else:
            vector = vector.to(self.device).float()
        k_tensor = torch.tensor([k], dtype=torch.float32, device=self.device)
        return self.batch_compute_Nk(k_tensor, vector.unsqueeze(0))[0]

    def describe(self, seq_vectors):
        """Compute N(k) vectors for each window in a vector sequence."""
        ex = self.extract_vectors(seq_vectors)
        if ex.shape[0] == 0:
            return []
        k_positions = torch.arange(ex.shape[0], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            Nk_batch = self.batch_compute_Nk(k_positions, ex)
        return Nk_batch.detach().cpu().numpy()

    def S(self, seq_vectors):
        """List of S(l) = sum(N(k)) for k=1..l, l=1..L for a given vector sequence."""
        ex = self.extract_vectors(seq_vectors)
        if ex.shape[0] == 0:
            return []
        k_positions = torch.arange(ex.shape[0], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            N_batch = self.batch_compute_Nk(k_positions, ex)
            S_cum = torch.cumsum(N_batch, dim=0)
        return [s.detach().cpu().numpy() for s in S_cum]

    def D(self, vector_seqs, t_list):
        """
        Compute mean squared deviation D across vector sequences:
        D = average over all window positions of (N(k) - t_seq)^2.
        All sequences are processed in a single vectorized pass; batch_compute_Nk
        chunks the work internally.
        """
        extracted_list, target_list = [], []
        for vs, t in zip(vector_seqs, t_list):
            ex = self.extract_vectors(vs)
            if ex.shape[0] == 0:
                continue
            extracted_list.append(ex)
            target_list.append(torch.tensor(t, dtype=torch.float32, device=self.device))
        if not extracted_list:
            return 0.0
        counts = torch.tensor([e.shape[0] for e in extracted_list],
                              dtype=torch.long, device=self.device)
        flat_vecs = torch.cat(extracted_list, dim=0)
        starts = torch.cumsum(counts, dim=0) - counts
        flat_k = (torch.arange(flat_vecs.shape[0], dtype=torch.float32, device=self.device)
                  - torch.repeat_interleave(starts.float(), counts))
        seq_indices = torch.repeat_interleave(
            torch.arange(len(extracted_list), dtype=torch.long, device=self.device), counts)
        targets = torch.stack(target_list)
        with torch.no_grad():
            Nk_batch = self.batch_compute_Nk(flat_k, flat_vecs)
            per_position = torch.sum((Nk_batch - targets[seq_indices]) ** 2, dim=1)
        return per_position.mean().item()

    def d(self, seq_vectors, t):
        """Compute pattern deviation value (d) for a single vector sequence."""
        return self.D([seq_vectors], [t])

    # ---------- shared training engine ----------
    def _sequence_vectors(self, Nk_flat, seq_indices, counts, B):
        """Average N(k) over the windows of each sequence -> (B, m)."""
        seq_sums = torch.zeros(B, self.m, device=self.device)
        seq_sums.scatter_add_(0, seq_indices.unsqueeze(1).expand(-1, self.m), Nk_flat)
        return seq_sums / counts.unsqueeze(1)

    def _train(self, seqs, step, *, max_iters=1000, tol=1e-8, learning_rate=0.01,
               continued=False, decay_rate=1.0, print_every=10, batch_size=32,
               checkpoint_file=None, checkpoint_interval=10, tag='Train',
               report=None, checkpoint_extra=None):
        """
        Shared training engine; every training method is a thin wrapper around this.

        A task supplies only the two things that actually differ between tasks:

          * ``step(Nk_flat, flat_vecs, seq_indices, counts, B, batch_indices)`` turns one
            batch into ``(loss, metrics)``, where metrics is a dict of plain numbers that
            is summed over the batches of an epoch (use {} when there is nothing to count);
          * ``report(it, avg_loss, current_lr, stats)`` formats the progress line, where
            ``stats`` holds the summed metrics plus '_n', the sequences seen this epoch.

        Everything else -- the per-sequence window cache, the vectorized batch scaffolding,
        the optimizer and its schedule, best-state tracking, checkpointing and early
        stopping -- lives here and is therefore identical for every task.

        Returns ``(loss_history, epoch_metrics)``: the mean loss per iteration and the list
        of per-iteration metric dicts.
        """
        if not continued:
            self.reset_parameters()

        # Pre-extract window vectors for all sequences (avoid repeated extraction).
        all_extracted = [self.extract_vectors(seq) for seq in seqs]

        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)

        if report is None:
            def report(it, avg_loss, current_lr, stats):
                return f"{tag} Iter {it:3d}: Loss = {avg_loss:.6e}, LR = {current_lr:.6f}"

        history, epoch_metrics = [], []
        prev_loss = float('inf')
        best_loss = float('inf')
        best_model_state = None

        for it in range(max_iters):
            total_loss = 0.0
            total_sequences = 0
            stats = {}
            indices = list(range(len(seqs)))
            random.shuffle(indices)

            for batch_start in range(0, len(indices), batch_size):
                batch_indices = indices[batch_start:batch_start + batch_size]
                B = len(batch_indices)
                batch_extracted = [all_extracted[i] for i in batch_indices]
                flat_vecs = torch.cat(batch_extracted, dim=0)
                if flat_vecs.shape[0] == 0:
                    continue
                # Vectorized per-window bookkeeping (no python loop over sequences)
                batch_counts = torch.tensor([e.shape[0] for e in batch_extracted],
                                            dtype=torch.long, device=self.device)
                seq_indices = torch.repeat_interleave(
                    torch.arange(B, dtype=torch.long, device=self.device), batch_counts)
                starts = torch.cumsum(batch_counts, dim=0) - batch_counts
                flat_k = (torch.arange(flat_vecs.shape[0], dtype=torch.float32, device=self.device)
                          - torch.repeat_interleave(starts.float(), batch_counts))
                counts = torch.clamp(batch_counts, min=1).float()
                Nk_flat = self.batch_compute_Nk(flat_k, flat_vecs)

                loss, batch_stats = step(Nk_flat, flat_vecs, seq_indices, counts,
                                         B, batch_indices)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * B
                total_sequences += B
                for name, value in batch_stats.items():
                    stats[name] = stats.get(name, 0.0) + value

            avg_loss = total_loss / total_sequences if total_sequences else 0.0
            history.append(avg_loss)
            stats['_n'] = total_sequences
            epoch_metrics.append(stats)

            if avg_loss < best_loss:
                best_loss = avg_loss
                best_model_state = copy.deepcopy(self.state_dict())

            if it % print_every == 0 or it == max_iters - 1:
                current_lr = scheduler.get_last_lr()[0]
                print(report(it, avg_loss, current_lr, stats))

            if checkpoint_file and (it % checkpoint_interval == 0 or it == max_iters - 1):
                self._save_checkpoint(checkpoint_file, it, history, optimizer, scheduler,
                                      best_loss, checkpoint_extra, epoch_metrics)

            if abs(prev_loss - avg_loss) < tol:
                print(f"Converged after {it+1} iterations.")
                if best_model_state is not None:
                    self.load_state_dict(best_model_state)
                break

            prev_loss = avg_loss
            scheduler.step()

        self._finish_training(seqs, history, optimizer, scheduler, best_loss,
                              checkpoint_file, checkpoint_extra, epoch_metrics)
        return history, epoch_metrics

    # ---------- specific training methods ----------
    def grad_train(self, seqs, t_list, max_iters=1000, tol=1e-8, learning_rate=0.01,
                   continued=False, decay_rate=1.0, print_every=10, batch_size=32,
                   checkpoint_file=None, checkpoint_interval=10):
        """
        Train the model for regression without any regression head, using gradient descent
        with fully vectorized batch processing.

        The m-dimensional model output (the mean of N(k) over the windows of a sequence)
        is fitted directly against the target vectors, so every target vector must have
        exactly m = vec_dim components. Meant to be paired with predict_t / t_generate;
        any regression head created earlier by reg_train is not used here.

        Returns:
            list: Training loss history
        """
        t_tensors = [torch.tensor(t, dtype=torch.float32, device=self.device) for t in t_list]

        def step(Nk_flat, flat_vecs, seq_indices, counts, B, batch_indices):
            """Sequence-mean of N(k) regressed on the target vectors (dimension m)."""
            seq_preds = self._sequence_vectors(Nk_flat, seq_indices, counts, B)
            batch_targets = torch.stack([t_tensors[i] for i in batch_indices], dim=0)
            return torch.mean((seq_preds - batch_targets) ** 2), {}

        history, _ = self._train(seqs, step, tag='GD', max_iters=max_iters, tol=tol,
                                 learning_rate=learning_rate, continued=continued,
                                 decay_rate=decay_rate, print_every=print_every,
                                 batch_size=batch_size, checkpoint_file=checkpoint_file,
                                 checkpoint_interval=checkpoint_interval)
        return history

    def reg_train(self, seqs, t_list, target_dim=None, max_iters=1000, tol=1e-8,
                  learning_rate=0.01, continued=False, decay_rate=1.0, print_every=10,
                  batch_size=32, checkpoint_file=None, checkpoint_interval=10):
        """
        Train the model for regression using gradient descent with fully vectorized batch
        processing.

        A trainable regression head (regresser) is created if not already present, mapping
        from model dimension (self.m) to the target dimension (target_dim). If target_dim
        is not provided, it is inferred from t_list. Meant to be paired with
        predict_r / r_generate.

        Returns:
            list: Training loss history
        """
        if target_dim is None:
            target_dim = len(t_list[0]) if t_list else self.m
        if self.regresser is None or self.target_dim != target_dim:
            self.regresser = nn.Linear(self.m, target_dim).to(self.device)
            self.target_dim = target_dim
            nn.init.xavier_uniform_(self.regresser.weight)
            nn.init.zeros_(self.regresser.bias)
        t_tensors = [torch.tensor(t, dtype=torch.float32, device=self.device) for t in t_list]

        def step(Nk_flat, flat_vecs, seq_indices, counts, B, batch_indices):
            """The objective of grad_train, pushed through the regression head."""
            seq_vectors = self._sequence_vectors(Nk_flat, seq_indices, counts, B)
            pred_targets = self.regresser(seq_vectors)
            batch_targets = torch.stack([t_tensors[i] for i in batch_indices], dim=0)
            return torch.mean((pred_targets - batch_targets) ** 2), {}

        history, _ = self._train(seqs, step, tag='GD', max_iters=max_iters, tol=tol,
                                 learning_rate=learning_rate, continued=continued,
                                 decay_rate=decay_rate, print_every=print_every,
                                 batch_size=batch_size, checkpoint_file=checkpoint_file,
                                 checkpoint_interval=checkpoint_interval)
        return history

    def cls_train(self, seqs, labels, num_classes, max_iters=1000, tol=1e-8,
                  learning_rate=0.01, continued=False, decay_rate=1.0, print_every=10,
                  batch_size=32, checkpoint_file=None, checkpoint_interval=10):
        """
        Train for multi-class classification with fully vectorized batch processing.
        Meant to be paired with predict_c / c_generate.
        """
        if self.classifier is None or self.num_classes != num_classes:
            self.classifier = nn.Linear(self.m, num_classes).to(self.device)
            self.num_classes = num_classes
        label_tensors = torch.tensor(labels, dtype=torch.long, device=self.device)
        criterion = nn.CrossEntropyLoss()

        def step(Nk_flat, flat_vecs, seq_indices, counts, B, batch_indices):
            """Cross-entropy on the class logits; also counts the correct sequences."""
            logits = self.classifier(self._sequence_vectors(Nk_flat, seq_indices, counts, B))
            batch_labels = label_tensors[batch_indices]
            loss = criterion(logits, batch_labels)
            with torch.no_grad():
                correct = (torch.argmax(logits, dim=1) == batch_labels).sum().item()
            return loss, {'correct': correct}

        def report(it, avg_loss, current_lr, stats):
            acc = stats['correct'] / stats['_n'] if stats['_n'] else 0.0
            return (f"CLS-Train Iter {it:3d}: Loss = {avg_loss:.6e}, Acc = {acc:.4f}, "
                    f"LR = {current_lr:.6f}")

        history, _ = self._train(seqs, step, tag='CLS-Train', report=report,
                                 checkpoint_extra={'num_classes': self.num_classes},
                                 max_iters=max_iters, tol=tol,
                                 learning_rate=learning_rate, continued=continued,
                                 decay_rate=decay_rate, print_every=print_every,
                                 batch_size=batch_size, checkpoint_file=checkpoint_file,
                                 checkpoint_interval=checkpoint_interval)
        return history

    def lbl_train(self, seqs, labels, num_labels, max_iters=1000, tol=1e-8,
                  learning_rate=0.01, continued=False, decay_rate=1.0, print_every=10,
                  batch_size=32, checkpoint_file=None, checkpoint_interval=10,
                  pos_weight=None):
        """
        Train for multi-label classification with fully vectorized batch processing.
        Meant to be paired with predict_l / l_generate.
        """
        if self.labeller is None or self.num_labels != num_labels:
            self.labeller = nn.Linear(self.m, num_labels).to(self.device)
            self.num_labels = num_labels
        if isinstance(labels, list):
            labels_tensor = torch.tensor(labels, dtype=torch.float32, device=self.device)
        else:
            labels_tensor = torch.as_tensor(labels, dtype=torch.float32, device=self.device)
        if pos_weight is not None:
            pos_weight_tensor = torch.tensor(pos_weight, dtype=torch.float32, device=self.device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        else:
            criterion = nn.BCEWithLogitsLoss()

        def step(Nk_flat, flat_vecs, seq_indices, counts, B, batch_indices):
            """BCE-with-logits on the label logits; also counts the correct label entries."""
            logits = self.labeller(self._sequence_vectors(Nk_flat, seq_indices, counts, B))
            batch_labels = labels_tensor[batch_indices]
            loss = criterion(logits, batch_labels)
            with torch.no_grad():
                preds = (torch.sigmoid(logits) > 0.5).float()
                correct = (preds == batch_labels).sum().item()
            return loss, {'correct': correct, 'predictions': batch_labels.numel()}

        def report(it, avg_loss, current_lr, stats):
            acc = stats['correct'] / stats['predictions'] if stats['predictions'] else 0.0
            return (f"MLC-Train Iter {it:3d}: Loss = {avg_loss:.6e}, Acc = {acc:.4f}, "
                    f"LR = {current_lr:.6f}")

        loss_history, epoch_stats = self._train(seqs, step, tag='MLC-Train', report=report,
                                                max_iters=max_iters, tol=tol,
                                                learning_rate=learning_rate, continued=continued,
                                                decay_rate=decay_rate, print_every=print_every,
                                                batch_size=batch_size, checkpoint_file=checkpoint_file,
                                                checkpoint_interval=checkpoint_interval)
        acc_history = [s['correct'] / s['predictions'] if s['predictions'] else 0.0
                       for s in epoch_stats]
        return loss_history, acc_history

    def self_train(self, seqs, max_iters=100, tol=1e-6, learning_rate=0.01,
                   continued=False, decay_rate=1.0, print_every=10,
                   batch_size=32, checkpoint_file=None, checkpoint_interval=5):
        """
        Self-training for self-consistency with fully vectorized batch processing:
        N(k) is pulled towards the transformed window vector M(v) at each position.
        Meant to be paired with generate().
        """
        def step(Nk_flat, flat_vecs, seq_indices, counts, B, batch_indices):
            """Per-window squared error between N(k) and the transformed window vector."""
            target_flat = self.M(flat_vecs)                             # [total, m]
            pos_loss = torch.sum((Nk_flat - target_flat) ** 2, dim=1)   # (total_windows,)
            seq_loss_sums = torch.zeros(B, device=self.device)
            seq_loss_sums.scatter_add_(0, seq_indices, pos_loss)
            return torch.mean(seq_loss_sums / counts), {}

        def report(it, avg_loss, current_lr, stats):
            return f"Self-Train Iter {it:3d}: Loss = {avg_loss:.6f}, LR = {current_lr:.6f}"

        history, _ = self._train(seqs, step, tag='Self-Train', report=report,
                                 max_iters=max_iters, tol=tol,
                                 learning_rate=learning_rate, continued=continued,
                                 decay_rate=decay_rate, print_every=print_every,
                                 batch_size=batch_size, checkpoint_file=checkpoint_file,
                                 checkpoint_interval=checkpoint_interval)
        return history

    # ---------- statistics & checkpoints ----------
    def _compute_training_statistics(self, seqs):
        """
        Compute training statistics for reconstruction and generation.

        The mean of N(k) over all windows of all sequences is accumulated in a single
        vectorized pass; chunking is handled by NK_CHUNK inside batch_compute_Nk.
        """
        extracted_list = []
        for seq in seqs:
            ex = self.extract_vectors(seq)
            if ex.shape[0] > 0:
                extracted_list.append(ex)
        total_window_count = sum(e.shape[0] for e in extracted_list)
        self.mean_vector_count = total_window_count / len(seqs) if seqs else 0
        if total_window_count == 0:
            self.mean_t = np.zeros(self.m)
            return
        counts = torch.tensor([e.shape[0] for e in extracted_list],
                              dtype=torch.long, device=self.device)
        flat_vecs = torch.cat(extracted_list, dim=0)
        starts = torch.cumsum(counts, dim=0) - counts
        flat_k = (torch.arange(flat_vecs.shape[0], dtype=torch.float32, device=self.device)
                  - torch.repeat_interleave(starts.float(), counts))
        with torch.no_grad():
            total_t = self.batch_compute_Nk(flat_k, flat_vecs).sum(dim=0)
        self.mean_t = (total_t / total_window_count).cpu().numpy()

    def _save_checkpoint(self, checkpoint_file, iteration, history, optimizer, scheduler,
                         best_loss, extra=None, metrics=None):
        """
        Write a training checkpoint that torch.load(..., weights_only=True) can read.

        Everything stored is a tensor or a plain scalar, so the file stays safe to load
        from an untrusted source; the training statistics need no separate entry because
        trained / mean_t / mean_vector_count are persistent buffers inside state_dict.
        ``extra`` carries method-specific bookkeeping (such as num_classes) and
        ``metrics`` the per-iteration metrics accumulated by the engine.
        """
        checkpoint = {
            'iteration': iteration,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'history': history,
            'best_loss': best_loss
        }
        if metrics is not None:
            checkpoint['metrics'] = metrics
        if extra:
            checkpoint.update(extra)
        torch.save(checkpoint, checkpoint_file)
        print(f"Checkpoint saved at iteration {iteration}")

    def _finish_training(self, seqs, history, optimizer, scheduler, best_loss,
                         checkpoint_file, extra=None, metrics=None):
        """
        Close a training run: compute the training statistics, mark the model as trained
        and, when checkpointing is on, rewrite the checkpoint file one last time.
        """
        self._compute_training_statistics(seqs)
        self.trained = True
        if checkpoint_file:
            self._save_checkpoint(checkpoint_file, len(history) - 1, history, optimizer,
                                  scheduler, best_loss, extra, metrics)
            print("Final checkpoint saved with the complete, reconstructable state")

    # ---------- predictors ----------
    def predict_t(self, seq_vectors):
        """
        Predict target vector for a vector sequence as the mean of N(k) over all its
        windows. This is the m-dimensional model output produced directly by the learned
        M map and P matrix, with no regression head involved; paired with
        grad_train / t_generate. If the sequence yields no window, a zero vector of
        length m is returned.
        """
        ex = self.extract_vectors(seq_vectors)
        if ex.shape[0] == 0:
            return np.zeros(self.m, dtype=np.float32)
        k_positions = torch.arange(ex.shape[0], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            Nk_batch = self.batch_compute_Nk(k_positions, ex)
        return Nk_batch.mean(dim=0).detach().cpu().numpy()

    def predict_r(self, seq_vectors):
        """
        Predict target vector for a vector sequence through the regression head.
        If a regression head (regresser) exists, the m-dimensional model output is mapped
        to the target dimension; otherwise the original m-dimensional model vector is
        returned. Paired with reg_train / r_generate.
        """
        ex = self.extract_vectors(seq_vectors)
        if ex.shape[0] == 0:
            if self.regresser is not None:
                return np.zeros(self.target_dim, dtype=np.float32)
            else:
                return np.zeros(self.m, dtype=np.float32)
        k_positions = torch.arange(ex.shape[0], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            Nk_batch = self.batch_compute_Nk(k_positions, ex)
            seq_rep = Nk_batch.mean(dim=0)
            if self.regresser is not None:
                out = self.regresser(seq_rep.unsqueeze(0)).squeeze(0)
                return out.detach().cpu().numpy()
            else:
                return seq_rep.detach().cpu().numpy()

    def predict_c(self, seq_vectors):
        """
        Predict class label for a vector sequence using the classification head.
        Paired with cls_train / c_generate.
        """
        if self.classifier is None:
            raise ValueError("Model must be trained first for classification")
        ex = self.extract_vectors(seq_vectors)
        if ex.shape[0] == 0:
            raise ValueError("Empty vector sequence")
        k_positions = torch.arange(ex.shape[0], dtype=torch.float32, device=self.device)
        Nk_batch = self.batch_compute_Nk(k_positions, ex)
        seq_rep = Nk_batch.mean(dim=0)
        with torch.no_grad():
            logits = self.classifier(seq_rep.unsqueeze(0))
            probabilities = torch.softmax(logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
        return predicted_class, probabilities[0].cpu().numpy()

    def predict_l(self, seq_vectors, threshold=0.5):
        """
        Predict multi-label classification for a vector sequence.
        Paired with lbl_train / l_generate.
        """
        assert self.labeller is not None, "Model must be trained first for label prediction"
        ex = self.extract_vectors(seq_vectors)
        if ex.shape[0] == 0:
            return (np.zeros(self.num_labels, dtype=np.float32),
                    np.zeros(self.num_labels, dtype=np.float32))
        k_positions = torch.arange(ex.shape[0], dtype=torch.float32, device=self.device)
        Nk_batch = self.batch_compute_Nk(k_positions, ex)
        seq_rep = Nk_batch.mean(dim=0)
        with torch.no_grad():
            logits = self.labeller(seq_rep.unsqueeze(0))
            probs = torch.sigmoid(logits).cpu().numpy()[0]
        binary_preds = (probs > threshold).astype(np.float32)
        return binary_preds, probs

    # ==================================================================
    # Vector-sequence generation
    # ==================================================================
    def _resolve_step(self):
        """Return the window step used by extract_vectors, with sanity checks."""
        if self.mode == 'linear':
            step = 1
        else:
            step = self.step if self.step is not None else self.rank
        if step <= 0:
            raise ValueError("step must be positive")
        return step

    def _num_windows_for_length(self, L, step):
        """Number of windows extract_vectors would produce for a sequence of length L."""
        if L < self.rank:
            return 0
        if self.mode == 'linear':
            return L - self.rank + 1
        return (L - self.rank) // step + 1

    def _Nk_from_windows(self, k_tensor, windows):
        """
        Apply rank_op to raw window tensors and evaluate N(k) for each window.

        Args:
            k_tensor (Tensor): [T] position indices
            windows (Tensor): [T, rank, vec_dim] raw window tensors

        Returns:
            N(k) tensor of shape [T, m]
        """
        applied = self._apply_op(windows)
        return self.batch_compute_Nk(k_tensor, applied)

    def _Nk_scorer(self, k_tensor, windows, target_tensor):
        """Negative squared error between N(k) of each window and a fixed target vector."""
        Nk = self._Nk_from_windows(k_tensor, windows)
        return -torch.sum((Nk - target_tensor) ** 2, dim=1)

    def _generate_windows(self, L, scorer, tau=0.0, num_steps=300, opt_lr=0.05):
        """
        Shared generation engine used by all *_generate methods.

        The length-L output vector sequence is produced by gradient-optimizing the
        T window tensors (shape [T, rank, vec_dim]) so that the score returned by
        ``scorer(k_tensor, windows)`` is maximized at each window position. The
        optimized windows are then stitched into a full [L, vec_dim] sequence: each
        window contributes its rank vectors to positions k*step .. k*step+rank-1,
        and overlapping positions are averaged across all covering windows.

        Stochastic sampling is achieved by adding Gaussian noise scaled by tau to the
        optimized windows, so tau=0 is deterministic and tau>0 yields random variants.

        Args:
            L (int): desired sequence length (number of output vectors).
            scorer (callable): scorer(k_tensor, windows) -> 1-D tensor of scores,
                one per window; higher is better.
            tau (float): temperature for stochastic sampling; tau=0 is deterministic.
            num_steps (int): number of gradient steps per generation.
            opt_lr (float): learning rate of the Adam optimizer used for generation.

        Returns:
            numpy.ndarray of shape [L, vec_dim].
        """
        assert self.trained, "Model must be trained first"
        assert self.rank_mode != 'pad', "generation is not applicable to rank_mode='pad'"
        if tau < 0:
            raise ValueError("Temperature must be non-negative")
        if L <= 0:
            return np.zeros((0, self.vec_dim), dtype=np.float32)

        step = self._resolve_step()
        T = self._num_windows_for_length(L, step)
        if T <= 0:
            return (0.1 * np.random.randn(L, self.vec_dim)).astype(np.float32)

        init = torch.randn(T, self.rank, self.vec_dim, device=self.device) * 0.1
        v = nn.Parameter(init)
        opt = torch.optim.Adam([v], lr=opt_lr)
        k_tensor = torch.arange(T, dtype=torch.float32, device=self.device)

        for _ in range(num_steps):
            opt.zero_grad()
            scores = scorer(k_tensor, v)
            loss = -scores.mean()
            loss.backward()
            opt.step()

        with torch.no_grad():
            v_final = v.detach()
            if tau > 0:
                v_final = v_final + tau * torch.randn_like(v_final)

            positions = (torch.arange(T, device=self.device).unsqueeze(1) * step
                         + torch.arange(self.rank, device=self.device).unsqueeze(0))
            mask = positions < L
            flat_positions = positions[mask]
            flat_v = v_final[mask]

            out = torch.zeros(L, self.vec_dim, device=self.device)
            cnt = torch.zeros(L, 1, device=self.device)
            out.index_add_(0, flat_positions, flat_v)
            cnt.index_add_(0, flat_positions,
                           torch.ones(flat_positions.shape[0], 1, device=self.device))
            out = out / cnt.clamp(min=1)

        return out.cpu().numpy()

    def generate(self, L, tau=0.0):
        """
        Generate a length-L vector sequence after self_train.
        The global training mean_t is used as the reconstruction target for every window
        position. rank_mode must not be 'pad'.
        """
        assert self.trained, "Model must be trained first"
        target_tensor = torch.tensor(self.mean_t, dtype=torch.float32, device=self.device)

        def scorer(k_tensor, windows):
            return self._Nk_scorer(k_tensor, windows, target_tensor)

        return self._generate_windows(L, scorer, tau=tau)

    def t_generate(self, L, tau=0.0, t=None):
        """
        Generate a length-L vector sequence after grad_train by matching the target
        vector t. If t is None, the global mean_t is used (equivalent to generate).
        """
        assert self.trained, "Model must be trained first"
        if t is None:
            target = self.mean_t
        else:
            target = np.asarray(t, dtype=np.float32).flatten()
        if target.size != self.m:
            raise ValueError(f"Target vector must have {self.m} elements, got {target.size}")
        target_tensor = torch.tensor(target, dtype=torch.float32, device=self.device)

        def scorer(k_tensor, windows):
            return self._Nk_scorer(k_tensor, windows, target_tensor)

        return self._generate_windows(L, scorer, tau=tau)

    def r_generate(self, L, r, tau=0.0):
        """
        Generate a length-L vector sequence after reg_train by matching the target
        vector r through the regression head. r must have length self.target_dim.
        """
        assert self.trained, "Model must be trained first"
        if self.regresser is None:
            raise ValueError("No regression head found; train with reg_train first")
        target = np.asarray(r, dtype=np.float32).flatten()
        if target.size != self.target_dim:
            raise ValueError(
                f"Target r must have {self.target_dim} elements, got {target.size}")
        target_tensor = torch.tensor(target, dtype=torch.float32, device=self.device)

        def scorer(k_tensor, windows):
            Nk = self._Nk_from_windows(k_tensor, windows)
            preds = self.regresser(Nk)
            return -torch.sum((preds - target_tensor) ** 2, dim=1)

        return self._generate_windows(L, scorer, tau=tau)

    def c_generate(self, L, c, tau=0.0):
        """
        Generate a length-L vector sequence after cls_train for a given class c.
        """
        assert self.trained, "Model must be trained first"
        if self.classifier is None:
            raise ValueError("No classifier found; train with cls_train first")
        if not (0 <= c < self.num_classes):
            raise ValueError(f"Class c must be in [0, {self.num_classes}), got {c}")

        def scorer(k_tensor, windows):
            Nk = self._Nk_from_windows(k_tensor, windows)
            logits = self.classifier(Nk)
            return logits[:, c]

        return self._generate_windows(L, scorer, tau=tau)

    def l_generate(self, L, l, tau=0.0):
        """
        Generate a length-L vector sequence after lbl_train for a target multi-label
        vector l. l must have length self.num_labels.
        """
        assert self.trained, "Model must be trained first"
        if self.labeller is None:
            raise ValueError("No labeller found; train with lbl_train first")
        target = np.asarray(l, dtype=np.float32).flatten()
        if target.size != self.num_labels:
            raise ValueError(
                f"Target l must have {self.num_labels} elements, got {target.size}")
        target_tensor = torch.tensor(target, dtype=torch.float32, device=self.device)

        def scorer(k_tensor, windows):
            Nk = self._Nk_from_windows(k_tensor, windows)
            probs = torch.sigmoid(self.labeller(Nk))
            return -torch.sum((probs - target_tensor) ** 2, dim=1)

        return self._generate_windows(L, scorer, tau=tau)

    # ---------- persistence ----------
    def save(self, filename):
        """Save model state to file."""
        torch.save(self.state_dict(), filename)
        print(f"Model saved to {filename}")

    def load(self, filename):
        """
        Load a model from file, accepting either format this class writes:

          * a plain state dict, as written by save();
          * a training checkpoint, as written by the training methods through
            checkpoint_file, whose state is taken from its 'model_state_dict'.

        The three prediction heads (regresser / classifier / labeller) are created
        lazily, so any head carried by the file but missing here is rebuilt first.
        The file is always read with weights_only=True: everything written is a tensor
        or a plain scalar, which keeps loading an untrusted file safe.
        """
        state_dict = torch.load(filename, map_location=self.device, weights_only=True)
        if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']

        for head, dim_attr in (('regresser', 'target_dim'),
                               ('classifier', 'num_classes'),
                               ('labeller', 'num_labels')):
            weight = state_dict.get(f'{head}.weight')
            if weight is None:
                continue
            out_dim = weight.shape[0]
            if getattr(self, head) is None or getattr(self, dim_attr) != out_dim:
                setattr(self, head, nn.Linear(self.m, out_dim).to(self.device))
                setattr(self, dim_attr, out_dim)

        self.load_state_dict(state_dict)
        print(f"Model loaded from {filename}")
        return self


# === Example Usage ===
if __name__ == "__main__":
    torch.manual_seed(11)
    random.seed(11)
    np.random.seed(11)

    # ----- global settings -----
    vec_dim = 15
    rank = 3
    user_step = 2
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ----- shared hyper-parameters for training -----
    # Larger batches and fewer iterations: the same gradient signal with fewer steps.
    BATCH_SIZE = 128
    MAX_ITERS = 100

    print("=" * 60)
    print("Numerical Dual Descriptor PM - PyTorch GPU Accelerated Version")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"vec_dim = {vec_dim}, rank = {rank}, "
          f"step = {user_step}, mode = nonlinear, rank_op = avg")
    print(f"Shared training settings: batch_size = {BATCH_SIZE}, max_iters = {MAX_ITERS}")
    print()
    print("Synthetic data carries real signal:")
    print("  * regression : target = fixed linear projection of the sequence mean")
    print("  * multi-label: labels  = sign of the first four mean components")
    print("  * classif.   : class-specific mean offsets")
    print("So the models can actually be seen to learn.")

    # ----- helpers -----
    def corr(a, b):
        """Pearson correlation via np.corrcoef; 0.0 if either input is constant."""
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        if a.std() == 0 or b.std() == 0:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    def show(seq):
        return (f"shape={tuple(seq.shape)}, "
                f"mean={float(np.mean(seq)):+.4f}, std={float(np.std(seq)):.4f}")

    # ----- data generators with real signal -----
    def make_latent_seqs(n_seqs, seed, latent_dim=4, noise_scale=0.2):
        """
        Generate vector sequences with a low-dimensional latent code.

        Each sequence is (h @ H) repeated across all time steps plus i.i.d. Gaussian
        noise, where h ∈ R^latent_dim is drawn fresh per sequence and H ∈ R^{latent_dim × m}
        is a fixed basis. Thus the sequence mean ≈ h @ H is a learnable, non-trivial
        target that the model can actually recover.
        """
        rng = np.random.RandomState(seed)
        H = rng.randn(latent_dim, vec_dim).astype(np.float32) * 0.7
        seqs = []
        for _ in range(n_seqs):
            L = rng.randint(200, 300)
            h = rng.randn(latent_dim).astype(np.float32)
            seq = (h @ H)[None, :] + rng.randn(L, vec_dim).astype(np.float32) * noise_scale
            seqs.append(seq.astype(np.float32))
        return seqs

    # =====================================================================
    # (1) grad_train + predict_t + t_generate (no regression head)
    # =====================================================================
    print("\n" + "=" * 60)
    print("(1) grad_train + predict_t + t_generate (no regression head)")
    print("=" * 60)

    seqs_grad = make_latent_seqs(100, seed=1)
    # Target = the sequence mean itself; that is exactly what predict_t returns
    # and is a fully learnable signal.
    t_list_grad = [s.mean(axis=0).astype(np.float32).tolist() for s in seqs_grad]

    dd_grad = NumDualDescriptorPM(vec_dim, rank=rank, rank_op='avg', rank_mode='drop',
                                  mode='nonlinear',
                                  user_step=user_step, device=device)

    print("\n" + "-" * 60)
    print("Starting Gradient Descent Training (grad_train, no head)")
    print("-" * 60)
    dd_grad.grad_train(seqs_grad, t_list_grad,
                       max_iters=MAX_ITERS, tol=1e-12,
                       learning_rate=0.02, decay_rate=0.999,
                       batch_size=BATCH_SIZE, print_every=20)

    pred_t_arr = np.array([dd_grad.predict_t(seq) for seq in seqs_grad])   # (N, m)
    true_t_arr = np.array(t_list_grad)                                      # (N, m)
    corrs = [corr(true_t_arr[:, i], pred_t_arr[:, i]) for i in range(vec_dim)]
    print(f"\nAverage prediction correlation: {np.mean(corrs):.4f} "
          f"(min {np.min(corrs):.4f}, max {np.max(corrs):.4f})")

    print("\n--- t_generate (default target = mean_t) ---")
    seq_def = dd_grad.t_generate(L=60, tau=0.0)
    print("Deterministic (tau=0):   ", show(seq_def))
    seq_rand = dd_grad.t_generate(L=60, tau=0.5)
    print("Stochastic  (tau=0.5):   ", show(seq_rand))

    print("\n--- t_generate (explicit target = mean of sequence #0) ---")
    my_target = seqs_grad[0].mean(axis=0).astype(np.float32)
    seq_tgt = dd_grad.t_generate(L=60, tau=0.0, t=my_target)
    print("Deterministic (tau=0):   ", show(seq_tgt))
    print("Verification: correlation between mean(N(k)) of the generated sequence")
    print(f"  and the target vector: "
          f"{corr(dd_grad.predict_t(seq_tgt), my_target):.4f}")

    # =====================================================================
    # (2) reg_train + predict_r + r_generate (with a regression head)
    # =====================================================================
    print("\n" + "=" * 60)
    print("(2) reg_train + predict_r + r_generate (with a regression head)")
    print("=" * 60)

    seqs_reg = make_latent_seqs(100, seed=2)
    target_dim_reg = 8
    W_reg = np.random.RandomState(99).randn(target_dim_reg, vec_dim).astype(np.float32) * 0.5
    t_list_reg = [(W_reg @ s.mean(axis=0)).astype(np.float32).tolist() for s in seqs_reg]

    dd = NumDualDescriptorPM(vec_dim, rank=rank, rank_op='avg', rank_mode='drop',
                             mode='nonlinear',
                             user_step=user_step, device=device)

    print("\n" + "-" * 60)
    print(f"Starting reg_train (target_dim = {target_dim_reg})")
    print("-" * 60)
    dd.reg_train(seqs_reg, t_list_reg, target_dim=target_dim_reg,
                 max_iters=MAX_ITERS, tol=1e-12,
                 learning_rate=0.02, decay_rate=0.999,
                 batch_size=BATCH_SIZE, print_every=20)

    pred_r_arr = np.array([dd.predict_r(seq) for seq in seqs_reg])   # (N, target_dim)
    true_r_arr = np.array(t_list_reg)
    corrs = [corr(true_r_arr[:, i], pred_r_arr[:, i]) for i in range(target_dim_reg)]
    print(f"\nAverage prediction correlation (target_dim={target_dim_reg}): "
          f"{np.mean(corrs):.4f} (min {np.min(corrs):.4f}, max {np.max(corrs):.4f})")

    print("\n--- r_generate (target r must be provided) ---")
    my_r = np.random.uniform(-1.0, 1.0, target_dim_reg).astype(np.float32)
    seq_r = dd.r_generate(L=60, r=my_r, tau=0.0)
    print("Deterministic (tau=0):   ", show(seq_r))
    seq_r_rand = dd.r_generate(L=60, r=my_r, tau=0.5)
    print("Stochastic  (tau=0.5):   ", show(seq_r_rand))
    print(f"Verification: predicted r on the generated sequence vs target -> "
          f"r-pred={np.round(dd.predict_r(seq_r), 3)}, r-target={np.round(my_r, 3)}")

    # =====================================================================
    # (3) Classification + c_generate
    # =====================================================================
    print("\n" + "=" * 60)
    print("(3) Classification Task")
    print("=" * 60)

    num_classes = 3
    class_seqs, class_labels = [], []
    rng = np.random.RandomState(7)
    for class_id in range(num_classes):
        for _ in range(50):
            L = rng.randint(150, 250)
            if class_id == 0:
                seq = rng.randn(L, vec_dim) + 1.0
            elif class_id == 1:
                seq = rng.randn(L, vec_dim) - 1.0
            else:
                seq = rng.randn(L, vec_dim)
            class_seqs.append(seq.astype(np.float32))
            class_labels.append(class_id)

    dd_cls = NumDualDescriptorPM(vec_dim, rank=rank, rank_op='avg', rank_mode='drop',
                                 mode='nonlinear',
                                 user_step=user_step, device=device)

    print("\n" + "-" * 60)
    print("Starting Classification Training")
    print("-" * 60)
    dd_cls.cls_train(class_seqs, class_labels, num_classes,
                     max_iters=MAX_ITERS, tol=1e-10,
                     learning_rate=0.05, decay_rate=0.995,
                     batch_size=BATCH_SIZE, print_every=20)

    correct = 0
    for seq, true_label in zip(class_seqs, class_labels):
        pred_class, _ = dd_cls.predict_c(seq)
        if pred_class == true_label:
            correct += 1
    print(f"\nTraining accuracy: {correct / len(class_seqs):.4f} "
          f"({correct}/{len(class_seqs)})")

    print("\n--- c_generate: one sequence per class ---")
    for c in range(num_classes):
        seq_c = dd_cls.c_generate(L=60, c=c, tau=0.0)
        pred_c, probs_c = dd_cls.predict_c(seq_c)
        print(f"Class {c} (tau=0):   {show(seq_c)}  ->  predicted={pred_c}, "
              f"probs={[f'{p:.3f}' for p in probs_c]}")

    # =====================================================================
    # (4) Multi-label classification + l_generate
    # =====================================================================
    print("\n" + "=" * 60)
    print("(4) Multi-Label Classification Model")
    print("=" * 60)

    num_labels = 4
    label_seqs, labels = [], []
    rng = np.random.RandomState(8)
    H_lbl = rng.randn(4, vec_dim).astype(np.float32) * 0.7
    for _ in range(100):
        L = rng.randint(200, 300)
        h = rng.randn(4).astype(np.float32)
        seq = (h @ H_lbl)[None, :] + rng.randn(L, vec_dim).astype(np.float32) * 0.2
        m = seq.mean(axis=0)
        # Labels are deterministic functions of the sequence mean -> learnable.
        label_vec = [
            1.0 if m[0] > 0 else 0.0,
            1.0 if m[1] > 0 else 0.0,
            1.0 if m[2] > 0 else 0.0,
            1.0 if m[3] > 0 else 0.0,
        ]
        label_seqs.append(seq.astype(np.float32))
        labels.append(label_vec)

    dd_lbl = NumDualDescriptorPM(vec_dim, rank=rank, rank_op='avg', rank_mode='drop',
                                 mode='nonlinear',
                                 user_step=user_step, device=device)

    print("\n" + "-" * 60)
    print("Starting Multi-Label Training (lbl_train)")
    print("-" * 60)
    loss_history, acc_history = dd_lbl.lbl_train(
        label_seqs, labels, num_labels,
        max_iters=MAX_ITERS, tol=1e-14,
        learning_rate=0.02, decay_rate=0.995,
        print_every=20, batch_size=BATCH_SIZE
    )
    print(f"\nFinal training loss: {loss_history[-1]:.6f}")
    print(f"Final training accuracy (per-label): {acc_history[-1]:.4f}")

    exact = 0
    for seq, true_labels in zip(label_seqs, labels):
        pred_binary, _ = dd_lbl.predict_l(seq, threshold=0.5)
        if np.all(pred_binary == np.array(true_labels)):
            exact += 1
    print(f"Sequence-level exact-match accuracy: "
          f"{exact / len(label_seqs):.4f} ({exact}/{len(label_seqs)})")

    print("\n--- a couple of example predictions ---")
    for i in range(3):
        pred_bin, pred_probs = dd_lbl.predict_l(label_seqs[i], threshold=0.5)
        print(f"Seq {i+1}: true={labels[i]}  pred={pred_bin.tolist()}  "
              f"probs={[f'{p:.3f}' for p in pred_probs]}")

    print("\n--- l_generate (target multi-label vector is required) ---")
    for target_l in ([1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]):
        seq_l = dd_lbl.l_generate(L=60, l=target_l, tau=0.0)
        pred_bin, pred_probs = dd_lbl.predict_l(seq_l, threshold=0.5)
        print(f"Target l={target_l} (tau=0):   {show(seq_l)}")
        print(f"  -> predicted labels: {pred_bin.tolist()}  "
              f"probs={[f'{p:.4f}' for p in pred_probs]}")
        seq_l_rand = dd_lbl.l_generate(L=60, l=target_l, tau=0.5)
        pred_bin_r, _ = dd_lbl.predict_l(seq_l_rand, threshold=0.5)
        print(f"Target l={target_l} (tau=0.5): {show(seq_l_rand)}")
        print(f"  -> predicted labels: {pred_bin_r.tolist()}")

    # =====================================================================
    # (5) Self-training + generate
    # =====================================================================
    print("\n" + "=" * 60)
    print("(5) Self-Training + generate")
    print("=" * 60)

    dd_self = NumDualDescriptorPM(vec_dim, rank=rank, rank_op='avg', rank_mode='drop',
                                  mode='nonlinear',
                                  user_step=user_step, device=device)

    self_seqs = make_latent_seqs(20, seed=5)

    print("\n" + "-" * 60)
    print("Self-training for self-consistency")
    print("-" * 60)
    dd_self.self_train(self_seqs, max_iters=MAX_ITERS, tol=1e-10,
                       learning_rate=0.02, decay_rate=0.999,
                       batch_size=16, print_every=20)

    print("\n--- generate ---")
    for i in range(2):
        gen_seq = dd_self.generate(L=60, tau=0.2)
        print(f"Sequence {i+1} (tau=0.2): ", show(gen_seq))

    # =====================================================================
    # (6) reg_train with a target dimension different from m + r_generate
    # =====================================================================
    print("\n" + "=" * 60)
    print("(6) reg_train + r_generate with a different target dimension")
    print("=" * 60)

    seqs_diff = make_latent_seqs(100, seed=3)
    target_dim_diff = 5
    W_diff = np.random.RandomState(21).randn(target_dim_diff, vec_dim).astype(np.float32) * 0.5
    t_list_diff = [(W_diff @ s.mean(axis=0)).astype(np.float32).tolist() for s in seqs_diff]

    print(f"Model dimension m = {vec_dim}, target dimension = {target_dim_diff}")
    dd_reg_diff = NumDualDescriptorPM(vec_dim, rank=rank, rank_op='avg', rank_mode='drop',
                                      mode='nonlinear',
                                      user_step=user_step, device=device)

    print("\n" + "-" * 60)
    print("Training regression (target_dim different from m)")
    print("-" * 60)
    dd_reg_diff.reg_train(seqs_diff, t_list_diff, target_dim=target_dim_diff,
                          max_iters=MAX_ITERS, tol=1e-12,
                          learning_rate=0.02, decay_rate=0.999,
                          batch_size=BATCH_SIZE, print_every=20)

    pred_diff = np.array([dd_reg_diff.predict_r(seq) for seq in seqs_diff])
    true_diff = np.array(t_list_diff)
    corrs = [corr(true_diff[:, i], pred_diff[:, i]) for i in range(target_dim_diff)]
    print(f"\nAverage correlation (target_dim={target_dim_diff}): "
          f"{np.mean(corrs):.4f} (min {np.min(corrs):.4f}, max {np.max(corrs):.4f})")

    print("\n--- r_generate with a target_dim_diff-dimensional r ---")
    my_r_diff = np.random.uniform(-1.0, 1.0, target_dim_diff).astype(np.float32)
    seq_r_diff = dd_reg_diff.r_generate(L=60, r=my_r_diff, tau=0.0)
    print("Deterministic (tau=0):   ", show(seq_r_diff))
    print(f"Verification: predicted r vs target -> "
          f"r-pred={np.round(dd_reg_diff.predict_r(seq_r_diff), 3)}, "
          f"r-target={np.round(my_r_diff, 3)}")

    print("\nAll tests completed successfully!")
