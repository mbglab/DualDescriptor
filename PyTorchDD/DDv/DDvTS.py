# Copyright (C) 2005-2025, Bin-Guang Ma (mbg@mail.hzau.edu.cn); SPDX-License-Identifier: MIT
# The Dual Descriptor Vector class (Tensor form) implemented with PyTorch
# This program is for the demonstration of methodology and not fully refined.
# Author: Bin-Guang Ma (assisted by DeepSeek); Date: 2025-7-29 ~ 2026-9-29
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

import math
import random
import itertools
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy


class DualDescriptorTS(nn.Module):
    """
    Vector Dual Descriptor with GPU acceleration using PyTorch:
      - tensor P ∈ R^{m×m×o} of basis coefficients
      - embedding: k-mer token embeddings in R^m
      - indexed periods: period[i,j,g] = i*(m*o) + j*o + g + 2
      - basis function phi_{i,j,g}(k) = cos(2π * k / period[i,j,g])
      - supports 'linear' or 'nonlinear' (step-by-rank) k-mer extraction
      - two interchangeable regression schemes:
          * grad_train + predict_t / t_generate : no regression head; the m-dimensional
            model output is fitted directly, so every target vector must have m components
          * reg_train  + predict_r / r_generate : a trainable linear regression head maps
            the m-dimensional model output to an arbitrary target dimension
      - also supports multi-class classification (cls_train / predict_c / c_generate)
        and multi-label classification (lbl_train / predict_l / l_generate)
    """

    # Maximum number of tokens handled by a single chunk of batch_compute_Nk.
    # The intermediate phi tensor has shape [tokens, m, m, o]; chunking keeps its memory
    # bounded independently of the training batch size. Override per instance if needed,
    # e.g. model.NK_CHUNK = 4096 for a smaller card.
    NK_CHUNK = 16384

    def __init__(self, charset, rank=1, rank_mode='drop', vec_dim=2, num_basis=5,
                 mode='linear', user_step=None, device='cuda'):
        """
        Initialize the Dual Descriptor model.

        Args:
            charset (list): List of characters in the alphabet (e.g., ['A','C','G','T'])
            rank (int): Length of k-mers
            rank_mode (str): 'pad' or 'drop' – how to handle incomplete fragments
            vec_dim (int): Embedding dimension (model dimension m)
            num_basis (int): Number of basis functions per pair (o)
            mode (str): 'linear' or 'nonlinear' – token extraction mode
            user_step (int, optional): Step size for nonlinear extraction
            device (str): 'cuda' or 'cpu'
        """
        super().__init__()
        self.charset = list(charset)
        self.rank = rank
        self.rank_mode = rank_mode
        self.m = vec_dim
        self.o = num_basis
        assert mode in ('linear', 'nonlinear')
        self.mode = mode
        self.step = user_step

        # Training statistics and the trained flag are buffers, so save() / load()
        # preserves them and a reloaded model can reconstruct immediately.
        self.register_buffer('_trained', torch.zeros(1, dtype=torch.bool))
        self.register_buffer('_mean_t', torch.zeros(self.m))
        self.register_buffer('_mean_token_count', torch.zeros(1, dtype=torch.float64))
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # All possible tokens (k-mers, optionally right-padded with '_')
        toks = []
        if self.rank_mode == 'pad':
            for r in range(1, self.rank + 1):
                for prefix in itertools.product(self.charset, repeat=r):
                    tok = ''.join(prefix).ljust(self.rank, '_')
                    toks.append(tok)
        else:
            toks = [''.join(p) for p in itertools.product(self.charset, repeat=self.rank)]
        self.tokens = sorted(set(toks))
        self.token_to_idx = {token: idx for idx, token in enumerate(self.tokens)}
        self.idx_to_token = {idx: token for idx, token in enumerate(self.tokens)}

        # Embedding layer for tokens
        self.embedding = nn.Embedding(len(self.tokens), self.m)

        # Position-weight tensor P[i][j][g]
        self.P = nn.Parameter(torch.empty(self.m, self.m, self.o))

        # Indexed periods[i][j][g] (fixed, not trainable): a pure function of (m, o),
        # rebuilt in the constructor and never saved as part of state_dict.
        periods = torch.zeros(self.m, self.m, self.o, dtype=torch.float32)
        for i in range(self.m):
            for j in range(self.m):
                for g in range(self.o):
                    periods[i, j, g] = i * (self.m * self.o) + j * self.o + g + 2
        self.register_buffer('periods', periods, persistent=False)

        # Pre-scaled angular frequency omega[i,j,g] = 2*pi / period[i,j,g], so the basis
        # function becomes cos(k * omega) with a single multiply inside cos().
        # Computed in float64 and rounded once; also derived state, kept non-persistent.
        self.register_buffer('omega', ((2 * math.pi) / periods.double()).float(), persistent=False)

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
        """Mean of N(k) over all training tokens, as a numpy array (persisted)."""
        return self._mean_t.detach().cpu().numpy()

    @mean_t.setter
    def mean_t(self, value):
        flat = torch.as_tensor(value, dtype=torch.float32).detach().flatten()
        if flat.numel() != self.m:
            raise ValueError(f"mean_t must have {self.m} elements, got {flat.numel()}")
        self._mean_t = flat.clone().to(self._mean_t.device)

    @property
    def mean_token_count(self):
        """Average number of tokens per training sequence (persisted, float64)."""
        return float(self._mean_token_count.item())

    @mean_token_count.setter
    def mean_token_count(self, value):
        self._mean_token_count.fill_(float(value))

    def reset_parameters(self):
        """Initialize model parameters with appropriate distributions."""
        nn.init.uniform_(self.embedding.weight, -0.5, 0.5)
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

    # ---------- tokenization & N(k) ----------
    def token_to_indices(self, token_list):
        """Convert list of tokens to tensor of indices."""
        return torch.tensor([self.token_to_idx[tok] for tok in token_list], device=self.device)

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
        if self.mode == 'linear':
            return [seq[i:i + self.rank] for i in range(L - self.rank + 1)]
        toks = []
        step = self.step or self.rank
        for i in range(0, L, step):
            frag = seq[i:i + self.rank]
            if self.rank_mode == 'pad':
                toks.append(frag if len(frag) == self.rank else frag.ljust(self.rank, '_'))
            elif self.rank_mode == 'drop' and len(frag) == self.rank:
                toks.append(frag)
        return toks

    def batch_compute_Nk(self, k_tensor, token_indices):
        """
        Vectorized computation of N(k) vectors for a batch of positions and tokens.

        Batches larger than NK_CHUNK tokens are split into chunks: the intermediate
        tensor phi has shape [tokens, m, m, o], so chunking bounds its memory without
        touching the reduction (over j and g), i.e. without changing the result.

        Args:
            k_tensor: Tensor of position indices [batch_size]
            token_indices: Tensor of token indices [batch_size]

        Returns:
            Tensor of N(k) vectors [batch_size, m]
        """
        n = k_tensor.numel()
        if n <= self.NK_CHUNK:
            return self._compute_Nk_chunk(k_tensor, token_indices)
        return torch.cat([self._compute_Nk_chunk(k_tensor[i:i + self.NK_CHUNK],
                                                 token_indices[i:i + self.NK_CHUNK])
                          for i in range(0, n, self.NK_CHUNK)], dim=0)

    def _compute_Nk_chunk(self, k_tensor, token_indices):
        """Compute N(k) for one chunk of tokens (see batch_compute_Nk)."""
        x = self.embedding(token_indices)                 # [chunk, m]
        k_expanded = k_tensor.view(-1, 1, 1, 1)           # [chunk, 1, 1, 1]
        phi = torch.cos(k_expanded * self.omega)          # [chunk, m, m, o]
        return torch.einsum('bj,ijg,bijg->bi', x, self.P, phi)

    def compute_Nk(self, k, token_idx):
        """Compute N(k) for a single position and token (uses batch internally)."""
        k_tensor = torch.tensor([k], dtype=torch.float32, device=self.device)
        idx_tensor = torch.tensor([token_idx], device=self.device)
        return self.batch_compute_Nk(k_tensor, idx_tensor)[0]

    def describe(self, seq):
        """Compute N(k) vectors for each k-mer in the sequence."""
        toks = self.extract_tokens(seq)
        if not toks:
            return []
        token_indices = self.token_to_indices(toks)
        k_positions = torch.arange(len(toks), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            Nk_batch = self.batch_compute_Nk(k_positions, token_indices)
        return Nk_batch.detach().cpu().numpy()

    def S(self, seq):
        """List of S(l) = sum(N(k)) for k=1..l, l=1..L for a given sequence."""
        toks = self.extract_tokens(seq)
        if not toks:
            return []
        token_indices = self.token_to_indices(toks)
        k_positions = torch.arange(len(toks), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            N_batch = self.batch_compute_Nk(k_positions, token_indices)
            S_cum = torch.cumsum(N_batch, dim=0)
        return [s.detach().cpu().numpy() for s in S_cum]

    def D(self, seqs, t_list):
        """
        Compute mean squared deviation D across sequences:
        D = average over all positions of (N(k) - t_seq)^2

        All sequences are processed in a single vectorized pass; batch_compute_Nk
        chunks the work internally.
        """
        token_index_list, target_list = [], []
        for seq, t in zip(seqs, t_list):
            toks = self.extract_tokens(seq)
            if not toks:
                continue
            token_index_list.append(self.token_to_indices(toks))
            target_list.append(torch.tensor(t, dtype=torch.float32, device=self.device))
        if not token_index_list:
            return 0.0
        counts = torch.tensor([len(ids) for ids in token_index_list],
                              dtype=torch.long, device=self.device)
        flat_ids = torch.cat(token_index_list)
        starts = torch.cumsum(counts, dim=0) - counts
        flat_k = (torch.arange(flat_ids.numel(), dtype=torch.float32, device=self.device)
                  - torch.repeat_interleave(starts.float(), counts))
        seq_indices = torch.repeat_interleave(
            torch.arange(len(token_index_list), dtype=torch.long, device=self.device), counts)
        targets = torch.stack(target_list)
        with torch.no_grad():
            Nk_batch = self.batch_compute_Nk(flat_k, flat_ids)
            per_position = torch.sum((Nk_batch - targets[seq_indices]) ** 2, dim=1)
        return per_position.mean().item()

    def d(self, seq, t):
        """Compute pattern deviation value (d) for a single sequence."""
        return self.D([seq], [t])

    # ---------- shared training engine ----------
    def _sequence_vectors(self, Nk_flat, seq_indices, counts, B):
        """Average N(k) over the tokens of each sequence -> (B, m)."""
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

          * ``step(Nk_flat, flat_ids, seq_indices, counts, B, batch_indices)`` turns one
            batch into ``(loss, metrics)``, where metrics is a dict of plain numbers that
            is summed over the batches of an epoch (use {} when there is nothing to count);
          * ``report(it, avg_loss, current_lr, stats)`` formats the progress line, where
            ``stats`` holds the summed metrics plus '_n', the sequences seen this epoch.

        Everything else -- the token-index cache, the vectorized batch scaffolding, the
        optimizer and its schedule, best-state tracking, checkpointing and early stopping
        -- lives here and is therefore identical for every task.

        Returns ``(loss_history, epoch_metrics)``: the mean loss per iteration and the list
        of per-iteration metric dicts.
        """
        if not continued:
            self.reset_parameters()

        # Pre-extract token indices for all sequences (avoid repeated extraction)
        all_token_indices = []
        for seq in seqs:
            toks = self.extract_tokens(seq)
            if toks:
                idxs = torch.tensor([self.token_to_idx[tok] for tok in toks],
                                    dtype=torch.long, device=self.device)
            else:
                idxs = torch.tensor([], dtype=torch.long, device=self.device)
            all_token_indices.append(idxs)

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
                batch_token_ids = [all_token_indices[i] for i in batch_indices]
                flat_ids = torch.cat(batch_token_ids)
                if flat_ids.numel() == 0:
                    continue
                # Vectorized per-token bookkeeping (no python loop over sequences)
                batch_counts = torch.tensor([len(ids) for ids in batch_token_ids],
                                            dtype=torch.long, device=self.device)
                seq_indices = torch.repeat_interleave(
                    torch.arange(B, dtype=torch.long, device=self.device), batch_counts)
                starts = torch.cumsum(batch_counts, dim=0) - batch_counts
                flat_k = (torch.arange(flat_ids.numel(), dtype=torch.float32, device=self.device)
                          - torch.repeat_interleave(starts.float(), batch_counts))
                counts = torch.clamp(batch_counts, min=1).float()
                Nk_flat = self.batch_compute_Nk(flat_k, flat_ids)

                loss, batch_stats = step(Nk_flat, flat_ids, seq_indices, counts, B, batch_indices)

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

        The m-dimensional model output (the mean of N(k) over the tokens of a sequence) is
        fitted directly against the target vectors, so every target vector must have exactly
        m = vec_dim components. Meant to be paired with predict_t / t_generate; any regression
        head created earlier by reg_train is not used here.

        Args:
            seqs (list): List of sequences
            t_list (list): List of target vectors, each of length m = vec_dim
            max_iters (int): Maximum number of training iterations
            tol (float): Convergence tolerance (stop if loss change < tol)
            learning_rate (float): Initial learning rate
            continued (bool): If True, continue training from current parameters
            decay_rate (float): Exponential decay rate for learning rate scheduler
            print_every (int): Print progress every this many iterations
            batch_size (int): Batch size for training
            checkpoint_file (str, optional): File to save checkpoints
            checkpoint_interval (int): Save checkpoint every this many iterations

        Returns:
            list: Training loss history
        """
        t_tensors = [torch.tensor(t, dtype=torch.float32, device=self.device) for t in t_list]

        def step(Nk_flat, flat_ids, seq_indices, counts, B, batch_indices):
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

    def reg_train(self, seqs, t_list, target_dim=None, max_iters=1000, tol=1e-8, learning_rate=0.01,
                  continued=False, decay_rate=1.0, print_every=10, batch_size=32,
                  checkpoint_file=None, checkpoint_interval=10):
        """
        Train the model for regression using gradient descent with fully vectorized batch
        processing.

        A trainable regression head (regresser) is created if not already present, mapping
        from model dimension (self.m) to the target dimension (target_dim). If target_dim is
        not provided, it is inferred from t_list. This allows the target dimension to differ
        from the model dimension. Meant to be paired with predict_r / r_generate.

        Args:
            seqs (list): List of sequences
            t_list (list): List of target vectors (each a list of floats)
            target_dim (int, optional): Desired output dimension. If None, inferred from t_list[0].
            max_iters (int): Maximum number of training iterations
            tol (float): Convergence tolerance (stop if loss change < tol)
            learning_rate (float): Initial learning rate
            continued (bool): If True, continue training from current parameters
            decay_rate (float): Exponential decay rate for learning rate scheduler
            print_every (int): Print progress every this many iterations
            batch_size (int): Batch size for training
            checkpoint_file (str, optional): File to save checkpoints
            checkpoint_interval (int): Save checkpoint every this many iterations

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

        def step(Nk_flat, flat_ids, seq_indices, counts, B, batch_indices):
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

    def cls_train(self, seqs, labels, num_classes, max_iters=1000, tol=1e-8, learning_rate=0.01,
                  continued=False, decay_rate=1.0, print_every=10, batch_size=32,
                  checkpoint_file=None, checkpoint_interval=10):
        """
        Train for multi-class classification with fully vectorized batch processing.
        Meant to be paired with predict_c / c_generate.
        """
        if self.classifier is None or self.num_classes != num_classes:
            self.classifier = nn.Linear(self.m, num_classes).to(self.device)
            self.num_classes = num_classes
        label_tensors = torch.tensor(labels, dtype=torch.long, device=self.device)
        criterion = nn.CrossEntropyLoss()

        def step(Nk_flat, flat_ids, seq_indices, counts, B, batch_indices):
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

    def lbl_train(self, seqs, labels, num_labels, max_iters=1000, tol=1e-8, learning_rate=0.01,
                  continued=False, decay_rate=1.0, print_every=10, batch_size=32,
                  checkpoint_file=None, checkpoint_interval=10, pos_weight=None):
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

        def step(Nk_flat, flat_ids, seq_indices, counts, B, batch_indices):
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
        Self-training for self-consistency with fully vectorized batch processing.
        Meant to be paired with generate().
        """
        def step(Nk_flat, flat_ids, seq_indices, counts, B, batch_indices):
            """Per-token squared error between N(k) and the token embedding."""
            emb_flat = self.embedding(flat_ids)
            pos_loss = torch.sum((Nk_flat - emb_flat) ** 2, dim=1)     # (total_tokens,)
            seq_loss_sums = torch.zeros(B, device=self.device)
            seq_loss_sums.scatter_add_(0, seq_indices, pos_loss)
            return torch.mean(seq_loss_sums / counts), {}

        def report(it, avg_loss, current_lr, stats):
            return f"Self-Train Iter {it:3d}: Loss = {avg_loss:.6f}, LR = {current_lr:.6f}"

        history, _ = self._train(seqs, step, tag='Self-Train', report=report, max_iters=max_iters, tol=tol,
                                 learning_rate=learning_rate, continued=continued,
                                 decay_rate=decay_rate, print_every=print_every,
                                 batch_size=batch_size, checkpoint_file=checkpoint_file,
                                 checkpoint_interval=checkpoint_interval)
        return history

    # ---------- statistics & checkpoints ----------
    def _compute_training_statistics(self, seqs):
        """
        Compute training statistics for reconstruction and generation.

        The mean of N(k) over all tokens of all sequences is accumulated in a single
        vectorized pass; chunking is handled by NK_CHUNK inside batch_compute_Nk.

        Memory scales with the total number of tokens in seqs; batch_compute_Nk bounds
        only the intermediate phi tensor.
        """
        token_index_list = []
        for seq in seqs:
            toks = self.extract_tokens(seq)
            if toks:
                token_index_list.append(self.token_to_indices(toks))
        total_token_count = sum(len(ids) for ids in token_index_list)
        self.mean_token_count = total_token_count / len(seqs) if seqs else 0
        if total_token_count == 0:
            self.mean_t = np.zeros(self.m)
            return
        counts = torch.tensor([len(ids) for ids in token_index_list],
                              dtype=torch.long, device=self.device)
        flat_ids = torch.cat(token_index_list)
        starts = torch.cumsum(counts, dim=0) - counts
        flat_k = (torch.arange(flat_ids.numel(), dtype=torch.float32, device=self.device)
                  - torch.repeat_interleave(starts.float(), counts))
        with torch.no_grad():
            total_t = self.batch_compute_Nk(flat_k, flat_ids).sum(dim=0)
        self.mean_t = (total_t / total_token_count).cpu().numpy()

    def _save_checkpoint(self, checkpoint_file, iteration, history, optimizer, scheduler,
                         best_loss, extra=None, metrics=None):
        """
        Write a training checkpoint that torch.load(..., weights_only=True) can read.

        Everything stored is a tensor or a plain scalar, so the file stays safe to load
        from an untrusted source; the training statistics need no separate entry because
        trained / mean_t / mean_token_count are persistent buffers inside state_dict.
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

        The training statistics only become meaningful once training has finished, so
        the interval checkpoints taken inside the loop are deliberately lightweight.
        The final write replaces the file with a complete, self-consistent state that a
        reloaded model can reconstruct from. Nothing is written when checkpoint_file is None.
        """
        self._compute_training_statistics(seqs)
        self.trained = True
        if checkpoint_file:
            # the iteration comes from the history length, so an early stop stays exact
            self._save_checkpoint(checkpoint_file, len(history) - 1, history, optimizer,
                                  scheduler, best_loss, extra, metrics)
            print("Final checkpoint saved with the complete, reconstructable state")

    # ---------- predictors ----------
    def predict_t(self, seq):
        """
        Predict target vector for a sequence as the mean of N(k) over all its tokens.
        This is the m-dimensional model output produced directly by the learned embedding
        and P tensor, with no regression head involved; paired with grad_train / t_generate.
        If the sequence yields no token, a zero vector of length m is returned.
        """
        toks = self.extract_tokens(seq)
        if not toks:
            return [0.0] * self.m
        token_indices = self.token_to_indices(toks)
        k_positions = torch.arange(len(toks), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            Nk_batch = self.batch_compute_Nk(k_positions, token_indices)
        return (Nk_batch.mean(dim=0)).detach().cpu().numpy()

    def predict_r(self, seq):
        """
        Predict target vector for a sequence through the regression head.
        If a regression head (regresser) exists, the m-dimensional model output is mapped
        to the target dimension; otherwise the original m-dimensional model vector is
        returned. Paired with reg_train / r_generate.
        """
        toks = self.extract_tokens(seq)
        if not toks:
            if self.regresser is not None:
                return [0.0] * self.target_dim
            else:
                return [0.0] * self.m
        token_indices = self.token_to_indices(toks)
        k_positions = torch.arange(len(toks), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            Nk_batch = self.batch_compute_Nk(k_positions, token_indices)
            seq_rep = Nk_batch.mean(dim=0)  # (m,)
            if self.regresser is not None:
                out = self.regresser(seq_rep.unsqueeze(0)).squeeze(0)
                return out.detach().cpu().numpy()
            else:
                return seq_rep.detach().cpu().numpy()

    def predict_c(self, seq):
        """
        Predict class label for a sequence using the classification head.
        Paired with cls_train / c_generate.
        """
        if self.classifier is None:
            raise ValueError("Model must be trained first for classification")
        toks = self.extract_tokens(seq)
        if not toks:
            raise ValueError("Empty sequence")
        token_indices = self.token_to_indices(toks)
        k_positions = torch.arange(len(toks), dtype=torch.float32, device=self.device)
        Nk_batch = self.batch_compute_Nk(k_positions, token_indices)
        seq_rep = Nk_batch.mean(dim=0)
        with torch.no_grad():
            logits = self.classifier(seq_rep.unsqueeze(0))
            probabilities = torch.softmax(logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
        return predicted_class, probabilities[0].cpu().numpy()

    def predict_l(self, seq, threshold=0.5):
        """
        Predict multi-label classification for a sequence.
        Paired with lbl_train / l_generate.
        """
        assert self.labeller is not None, "Model must be trained first for label prediction"
        toks = self.extract_tokens(seq)
        if not toks:
            return np.zeros(self.num_labels, dtype=np.float32), np.zeros(self.num_labels, dtype=np.float32)
        token_indices = self.token_to_indices(toks)
        k_positions = torch.arange(len(toks), dtype=torch.float32, device=self.device)
        Nk_batch = self.batch_compute_Nk(k_positions, token_indices)
        seq_rep = Nk_batch.mean(dim=0)
        with torch.no_grad():
            logits = self.labeller(seq_rep.unsqueeze(0))
            probs = torch.sigmoid(logits).cpu().numpy()[0]
        binary_preds = (probs > threshold).astype(np.float32)
        return binary_preds, probs

    # ==================================================================
    # Sequence generation
    # ==================================================================
    def _resolve_step(self):
        """Return the token step used by extract_tokens, with sanity checks."""
        if self.mode == 'linear':
            step = 1
        else:
            step = self.step if self.step is not None else self.rank
        if step <= 0:
            raise ValueError("step must be positive")
        return step

    def _num_tokens_for_length(self, L, step):
        """Number of tokens extract_tokens would produce for a sequence of length L."""
        if L < self.rank:
            return 0
        if self.mode == 'linear':
            return L - self.rank + 1
        return (L - self.rank) // step + 1

    def _generate_sequence(self, L, scorer, tau=0.0):
        """
        Shared generation engine used by all *_generate methods.

        The sequence is built token by token using the same tokenization rule as
        extract_tokens. Overlap between adjacent tokens (when step < rank) is
        respected: the first rank - step characters of each new token are forced
        to match the tail of the previous token.

        Positions that remain None at the end (possible when step > rank, i.e.
        tokens do not tile the sequence) are filled with random characters from
        the charset.

        Args:
            L (int): desired sequence length.
            scorer (callable): scorer(k, cand_indices) -> 1-D tensor of scores,
                one per candidate token; higher is better. Usually derived from
                a squared-error criterion between N(k) and a target.
            tau (float): temperature for stochastic sampling; tau=0 is deterministic
                (argmax); tau>0 samples from softmax(scores / tau).

        Returns:
            str: the generated sequence of length L.
        """
        assert self.trained, "Model must be trained first"
        assert self.rank_mode != 'pad', "generation is not applicable to rank_mode='pad'"
        if tau < 0:
            raise ValueError("Temperature must be non-negative")
        if L <= 0:
            return ""

        step = self._resolve_step()
        T = self._num_tokens_for_length(L, step)
        if T <= 0:
            # Sequence shorter than rank: no full token can be formed.
            return ''.join(random.choices(self.charset, k=L))

        chars = [None] * L

        with torch.no_grad():
            for k in range(T):
                # Starting character index of the k-th token.
                start = k * step
                # Overlap with the previous token: the first rank - step characters of
                # this token are already fixed by the previous token. If step >= rank,
                # there is no overlap (prefix_len = 0) and every token is independent.
                prefix_len = 0 if k == 0 else max(0, self.rank - step)

                if prefix_len > 0:
                    prefix = ''.join(chars[start:start + prefix_len])
                    if any(ch is None for ch in chars[start:start + prefix_len]):
                        raise RuntimeError("Internal error: overlapping characters are not set")
                    candidates = [tok for tok in self.tokens if tok.startswith(prefix)]
                else:
                    candidates = self.tokens

                if not candidates:
                    raise RuntimeError("No candidate token found for the given prefix")

                cand_indices = torch.tensor([self.token_to_idx[tok] for tok in candidates],
                                            device=self.device)
                scores = scorer(k, cand_indices)

                if tau == 0:
                    best_idx = torch.argmax(scores).item()
                    chosen_tok = candidates[best_idx]
                else:
                    # Numerical stability: subtract the max before softmax.
                    adj = scores - scores.max()
                    probs = torch.softmax(adj / tau, dim=0).detach().cpu().numpy()
                    chosen_idx = random.choices(range(len(candidates)), weights=probs, k=1)[0]
                    chosen_tok = candidates[chosen_idx]

                for i, ch in enumerate(chosen_tok):
                    pos = start + i
                    if pos < L:
                        chars[pos] = ch

        # Fill any remaining unknown characters (e.g. gaps when step > rank).
        for i in range(L):
            if chars[i] is None:
                chars[i] = random.choice(self.charset)

        return ''.join(chars)

    def _Nk_scorer(self, k, cand_indices, target_tensor):
        """Return negative squared error between N(k) and a fixed target vector."""
        k_tensor = torch.full((cand_indices.numel(),), float(k),
                              dtype=torch.float32, device=self.device)
        Nk = self.batch_compute_Nk(k_tensor, cand_indices)
        return -torch.sum((Nk - target_tensor) ** 2, dim=1)

    def generate(self, L, tau=0.0):
        """
        Reconstruct / generate a sequence of length L after self_train.

        The global training mean_t is used as the reconstruction target for every
        token position. The generated sequence respects the tokenization rule and
        the overlap between adjacent tokens. rank_mode must not be 'pad'.

        Args:
            L (int): desired sequence length.
            tau (float): temperature; tau=0 deterministic, tau>0 stochastic.

        Returns:
            str: generated sequence of length L.
        """
        assert self.trained, "Model must be trained first"
        target_tensor = torch.tensor(self.mean_t, dtype=torch.float32, device=self.device)

        def scorer(k, cand_indices):
            return self._Nk_scorer(k, cand_indices, target_tensor)

        return self._generate_sequence(L, scorer, tau=tau)

    def t_generate(self, L, tau=0.0, t=None):
        """
        Generate a sequence of length L after grad_train by matching the target vector t.

        For each token position k, a token is selected whose N(k) is close to t. If t is
        None, the global mean_t is used (equivalent to generate). This is a per-token
        approximation to the sequence-mean objective optimized by grad_train.

        Args:
            L (int): desired sequence length.
            tau (float): temperature; tau=0 deterministic, tau>0 stochastic.
            t (array-like, optional): target vector of length self.m; default mean_t.

        Returns:
            str: generated sequence of length L.
        """
        assert self.trained, "Model must be trained first"
        if t is None:
            target = self.mean_t
        else:
            target = np.asarray(t, dtype=np.float32).flatten()
        if target.size != self.m:
            raise ValueError(f"Target vector must have {self.m} elements, got {target.size}")
        target_tensor = torch.tensor(target, dtype=torch.float32, device=self.device)

        def scorer(k, cand_indices):
            return self._Nk_scorer(k, cand_indices, target_tensor)

        return self._generate_sequence(L, scorer, tau=tau)

    def r_generate(self, L, r, tau=0.0):
        """
        Generate a sequence of length L after reg_train by matching the target vector r.

        The target r is required (no default), keeping it clearly distinct from
        t_generate. r must have length self.target_dim; the regression head is applied
        to each candidate N(k), and the token minimizing ||regresser(N(k)) - r||^2 is
        chosen.

        Args:
            L (int): desired sequence length.
            r (array-like): target vector of length self.target_dim (required).
            tau (float): temperature; tau=0 deterministic, tau>0 stochastic.

        Returns:
            str: generated sequence of length L.
        """
        assert self.trained, "Model must be trained first"
        if self.regresser is None:
            raise ValueError("No regression head found; train with reg_train first")
        target = np.asarray(r, dtype=np.float32).flatten()
        if target.size != self.target_dim:
            raise ValueError(
                f"Target r must have {self.target_dim} elements, got {target.size}")
        target_tensor = torch.tensor(target, dtype=torch.float32, device=self.device)

        def scorer(k, cand_indices):
            k_tensor = torch.full((cand_indices.numel(),), float(k),
                                  dtype=torch.float32, device=self.device)
            Nk = self.batch_compute_Nk(k_tensor, cand_indices)
            preds = self.regresser(Nk)
            return -torch.sum((preds - target_tensor) ** 2, dim=1)

        return self._generate_sequence(L, scorer, tau=tau)

    def c_generate(self, L, c, tau=0.0):
        """
        Generate a sequence of length L after cls_train for a given class c.

        For each token position k, the token whose N(k) gives the highest class-c logit
        through the classifier is chosen. No default target: c is required.

        Args:
            L (int): desired sequence length.
            c (int): target class index in [0, self.num_classes).
            tau (float): temperature; tau=0 deterministic, tau>0 stochastic.

        Returns:
            str: generated sequence of length L.
        """
        assert self.trained, "Model must be trained first"
        if self.classifier is None:
            raise ValueError("No classifier found; train with cls_train first")
        if not (0 <= c < self.num_classes):
            raise ValueError(f"Class c must be in [0, {self.num_classes}), got {c}")

        def scorer(k, cand_indices):
            k_tensor = torch.full((cand_indices.numel(),), float(k),
                                  dtype=torch.float32, device=self.device)
            Nk = self.batch_compute_Nk(k_tensor, cand_indices)
            logits = self.classifier(Nk)
            return logits[:, c]

        return self._generate_sequence(L, scorer, tau=tau)

    def l_generate(self, L, l, tau=0.0):
        """
        Generate a sequence of length L after lbl_train for a target multi-label vector l.

        For each token position k, the token whose N(k) gives sigmoid probabilities closest
        (in squared error) to l through the labeller is chosen. No default target: l is
        required.

        Args:
            L (int): desired sequence length.
            l (array-like): target multi-label vector of length self.num_labels (required).
            tau (float): temperature; tau=0 deterministic, tau>0 stochastic.

        Returns:
            str: generated sequence of length L.
        """
        assert self.trained, "Model must be trained first"
        if self.labeller is None:
            raise ValueError("No labeller found; train with lbl_train first")
        target = np.asarray(l, dtype=np.float32).flatten()
        if target.size != self.num_labels:
            raise ValueError(
                f"Target l must have {self.num_labels} elements, got {target.size}")
        target_tensor = torch.tensor(target, dtype=torch.float32, device=self.device)

        def scorer(k, cand_indices):
            k_tensor = torch.full((cand_indices.numel(),), float(k),
                                  dtype=torch.float32, device=self.device)
            Nk = self.batch_compute_Nk(k_tensor, cand_indices)
            probs = torch.sigmoid(self.labeller(Nk))
            return -torch.sum((probs - target_tensor) ** 2, dim=1)

        return self._generate_sequence(L, scorer, tau=tau)

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

        # Rebuild every lazily created head whose parameters the file carries
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
    from statistics import correlation

    print("=" * 50)
    print("Dual Descriptor TS - PyTorch GPU Accelerated Version")
    print("Two interchangeable regression schemes:")
    print("  (a) grad_train + predict_t / t_generate  (no regression head)")
    print("  (b) reg_train  + predict_r / r_generate  (with a regression head)")
    print("=" * 50)

    torch.manual_seed(11)
    random.seed(11)

    charset = ['A', 'C', 'G', 'T']
    vec_dim = 10
    num_basis = 10
    rank = 6
    user_step = 3

    # =====================================================================
    # grad_train + predict_t + t_generate (no regression head)
    # =====================================================================
    print("\n" + "=" * 50)
    print("grad_train + predict_t + t_generate (no regression head)")
    print("=" * 50)

    dd_grad = DualDescriptorTS(
        charset,
        rank=rank,
        vec_dim=vec_dim,
        num_basis=num_basis,
        mode='nonlinear',
        user_step=user_step,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    print(f"\nUsing device: {dd_grad.device}")
    print(f"Number of tokens: {len(dd_grad.tokens)}")

    # grad_train fits the m-dimensional model output directly, so each target
    # vector must have exactly m = vec_dim components.
    seqs_grad, t_list_grad = [], []
    for _ in range(100):
        L = random.randint(200, 300)
        seqs_grad.append(''.join(random.choices(charset, k=L)))
        t_list_grad.append([random.uniform(-1.0, 1.0) for _ in range(vec_dim)])

    print("\n" + "=" * 50)
    print("Starting Gradient Descent Training (no regression head, grad_train)")
    print("=" * 50)
    dd_grad.grad_train(seqs_grad, t_list_grad, max_iters=100, tol=1e-9,
                       learning_rate=0.1, decay_rate=0.99, batch_size=2048)

    t_pred_grad = dd_grad.predict_t(seqs_grad[0])
    print(f"\nPredicted t for first sequence (predict_t): "
          f"{[round(float(x), 4) for x in t_pred_grad]}")

    pred_t_list_grad = [dd_grad.predict_t(seq) for seq in seqs_grad]
    corr_sum_grad = 0.0
    for i in range(dd_grad.m):
        actu_t = [t_vec[i] for t_vec in t_list_grad]
        pred_t = [t_vec[i] for t_vec in pred_t_list_grad]
        corr = correlation(actu_t, pred_t)
        print(f"Dimension {i} prediction correlation: {corr:.4f}")
        corr_sum_grad += corr
    corr_avg_grad = corr_sum_grad / dd_grad.m
    print(f"Average correlation: {corr_avg_grad:.4f}")

    # t_generate: default target is the global mean_t; also demo an explicit target vector
    print("\n--- t_generate with the global mean_t (default) ---")
    seq_t_default = dd_grad.t_generate(L=100, tau=0.0)
    print("Deterministic (tau=0):   ", seq_t_default[:50] + "...")
    seq_t_rand = dd_grad.t_generate(L=100, tau=0.5)
    print("Stochastic  (tau=0.5):   ", seq_t_rand[:50] + "...")

    print("\n--- t_generate with an explicit target vector ---")
    my_target = [random.uniform(-1.0, 1.0) for _ in range(vec_dim)]
    seq_t_target = dd_grad.t_generate(L=100, tau=0.0, t=my_target)
    print("Deterministic (tau=0):   ", seq_t_target[:50] + "...")
    seq_t_target_rand = dd_grad.t_generate(L=100, tau=0.5, t=my_target)
    print("Stochastic  (tau=0.5):   ", seq_t_target_rand[:50] + "...")

    # =====================================================================
    # reg_train + predict_r + r_generate (with a linear regression head)
    # =====================================================================
    print("\n" + "=" * 50)
    print("reg_train + predict_r + r_generate (with a regression head)")
    print("=" * 50)

    dd = DualDescriptorTS(
        charset,
        rank=rank,
        vec_dim=vec_dim,
        num_basis=num_basis,
        mode='nonlinear',
        user_step=user_step,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    print(f"\nUsing device: {dd.device}")
    print(f"Number of tokens: {len(dd.tokens)}")

    seqs, t_list = [], []
    for _ in range(100):
        L = random.randint(200, 300)
        seq = ''.join(random.choices(charset, k=L))
        seqs.append(seq)
        t_list.append([random.uniform(-1.0, 1.0) for _ in range(vec_dim)])

    print("\n" + "=" * 50)
    print("Starting Gradient Descent Training (regression head, reg_train)")
    print("=" * 50)
    dd.reg_train(seqs, t_list, max_iters=100, tol=1e-9, learning_rate=0.1,
                 decay_rate=0.99, batch_size=2048)

    aseq = seqs[0]
    t_pred = dd.predict_r(aseq)
    print(f"\nPredicted t for first sequence (predict_r): "
          f"{[round(float(x), 4) for x in t_pred]}")

    pred_t_list = [dd.predict_r(seq) for seq in seqs]
    corr_sum = 0.0
    for i in range(dd.m):
        actu_t = [t_vec[i] for t_vec in t_list]
        pred_t = [t_vec[i] for t_vec in pred_t_list]
        corr = correlation(actu_t, pred_t)
        print(f"Dimension {i} prediction correlation: {corr:.4f}")
        corr_sum += corr
    corr_avg = corr_sum / dd.m
    print(f"Average correlation: {corr_avg:.4f}")

    # r_generate: the target r is a required argument, distinct from t_generate.
    # The regression head maps N(k) (dimension m) to the target dimension, and the
    # token whose mapped vector is closest to r is chosen at each position.
    print("\n--- r_generate with an explicit target r (target_dim-dimensional) ---")
    my_r = [random.uniform(-1.0, 1.0) for _ in range(dd.target_dim)]
    seq_r_target = dd.r_generate(L=100, r=my_r, tau=0.0)
    print("Deterministic (tau=0):   ", seq_r_target[:50] + "...")
    seq_r_target_rand = dd.r_generate(L=100, r=my_r, tau=0.5)
    print("Stochastic  (tau=0.5):   ", seq_r_target_rand[:50] + "...")

    # =====================================================================
    # Classification task + c_generate
    # =====================================================================
    print("\n" + "=" * 50)
    print("Classification Task")
    print("=" * 50)

    num_classes = 3
    class_seqs = []
    class_labels = []
    for class_id in range(num_classes):
        for _ in range(50):
            L = random.randint(150, 250)
            if class_id == 0:
                seq = ''.join(random.choices(['A', 'C', 'G', 'T'], weights=[0.6, 0.1, 0.1, 0.2], k=L))
            elif class_id == 1:
                seq = ''.join(random.choices(['A', 'C', 'G', 'T'], weights=[0.1, 0.4, 0.4, 0.1], k=L))
            else:
                seq = ''.join(random.choices(['A', 'C', 'G', 'T'], k=L))
            class_seqs.append(seq)
            class_labels.append(class_id)

    dd_cls = DualDescriptorTS(
        charset,
        rank=rank,
        vec_dim=vec_dim,
        num_basis=num_basis,
        mode='nonlinear',
        user_step=user_step,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    print("\n" + "=" * 50)
    print("Starting Classification Training")
    print("=" * 50)
    dd_cls.cls_train(class_seqs, class_labels, num_classes,
                     max_iters=50, tol=1e-8, learning_rate=0.05,
                     decay_rate=0.99, batch_size=32, print_every=1)

    print("\n" + "=" * 50)
    print("Prediction results")
    print("=" * 50)

    correct = 0
    for seq, true_label in zip(class_seqs, class_labels):
        pred_class, _ = dd_cls.predict_c(seq)
        if pred_class == true_label:
            correct += 1
    accuracy = correct / len(class_seqs)
    print(f"Accuracy: {accuracy:.4f} ({correct}/{len(class_seqs)})")

    print("\nExample predictions:")
    for i in range(min(5, len(class_seqs))):
        pred_class, probs = dd_cls.predict_c(class_seqs[i])
        print(f"Seq {i+1}: True={class_labels[i]}, Pred={pred_class}, "
              f"Probs={[f'{p:.3f}' for p in probs]}")

    # c_generate: one sequence per class; the target class is required.
    print("\n" + "=" * 50)
    print("c_generate: generate sequences for each class")
    print("=" * 50)
    for c in range(num_classes):
        seq_c = dd_cls.c_generate(L=120, c=c, tau=0.0)
        print(f"Class {c} (tau=0):   ", seq_c[:50] + "...")
        seq_c_rand = dd_cls.c_generate(L=120, c=c, tau=0.5)
        print(f"Class {c} (tau=0.5): ", seq_c_rand[:50] + "...")

    # =====================================================================
    # Multi-label classification + l_generate
    # =====================================================================
    print("\n\n" + "=" * 50)
    print("Multi-Label Classification Model")
    print("=" * 50)

    num_labels = 4
    label_seqs = []
    labels = []
    for _ in range(100):
        L = random.randint(200, 300)
        seq = ''.join(random.choices(charset, k=L))
        label_seqs.append(seq)
        label_vec = [random.random() > 0.7 for _ in range(num_labels)]
        labels.append([1.0 if x else 0.0 for x in label_vec])

    dd_lbl = DualDescriptorTS(
        charset,
        rank=rank,
        vec_dim=vec_dim,
        num_basis=num_basis,
        mode='nonlinear',
        user_step=user_step,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    print("\n" + "=" * 50)
    print("Starting Gradient Descent Training for Multi-Label Classification")
    print("=" * 50)

    loss_history, acc_history = dd_lbl.lbl_train(
        label_seqs, labels, num_labels,
        max_iters=50, tol=1e-16, learning_rate=0.01,
        decay_rate=0.99, print_every=10, batch_size=32
    )

    print(f"\nFinal training loss: {loss_history[-1]:.6f}")
    print(f"Final training accuracy: {acc_history[-1]:.4f}")

    print("\n" + "=" * 50)
    print("Prediction Results")
    print("=" * 50)

    all_correct = 0
    total = 0
    for seq, true_labels in zip(label_seqs, labels):
        pred_binary, pred_probs = dd_lbl.predict_l(seq, threshold=0.5)
        true_labels_np = np.array(true_labels)
        correct = np.all(pred_binary == true_labels_np)
        all_correct += correct
        total += 1
        if total <= 3:
            print(f"\nSequence {total}:")
            print(f"True labels: {true_labels_np}")
            print(f"Predicted binary: {pred_binary}")
            print(f"Predicted probabilities: {[f'{p:.4f}' for p in pred_probs]}")
            print(f"Correct: {correct}")

    accuracy = all_correct / total if total > 0 else 0.0
    print(f"\nOverall prediction accuracy: {accuracy:.4f} ({all_correct}/{total} sequences)")

    print("\n" + "=" * 50)
    print("Label Prediction Example")
    print("=" * 50)

    test_seq = "".join(random.choices(charset, k=250))
    print(f"Test sequence (first 50 chars): {test_seq[:50]}...")
    binary_pred, probs_pred = dd_lbl.predict_l(test_seq, threshold=0.5)
    print(f"\nPredicted binary labels: {binary_pred}")
    print(f"Predicted probabilities: {[f'{p:.4f}' for p in probs_pred]}")

    label_names = ["Function_A", "Function_B", "Function_C", "Function_D"]
    print("\nLabel interpretation:")
    for i, (binary, prob) in enumerate(zip(binary_pred, probs_pred)):
        status = "ACTIVE" if binary > 0.5 else "INACTIVE"
        print(f"  {label_names[i]}: {status} (confidence: {prob:.4f})")

    # l_generate: the target multi-label vector is required.
    print("\n" + "=" * 50)
    print("l_generate: generate a sequence for a target multi-label vector")
    print("=" * 50)
    target_l_1 = [1.0, 0.0, 1.0, 0.0]
    target_l_2 = [0.0, 1.0, 0.0, 1.0]
    for target_l in (target_l_1, target_l_2):
        seq_l = dd_lbl.l_generate(L=150, l=target_l, tau=0.0)
        print(f"Target l={target_l} (tau=0):   ", seq_l[:50] + "...")
        seq_l_rand = dd_lbl.l_generate(L=150, l=target_l, tau=0.5)
        print(f"Target l={target_l} (tau=0.5): ", seq_l_rand[:50] + "...")
        # Verify that the generated sequence is classified towards the target
        pred_binary, pred_probs = dd_lbl.predict_l(seq_l, threshold=0.5)
        print(f"  -> predicted labels on generated seq: {pred_binary}")
        print(f"  -> predicted probs:                    "
              f"{[f'{p:.4f}' for p in pred_probs]}")

    # =====================================================================
    # Self-Training + generate
    # =====================================================================
    print("\n" + "=" * 50)
    print("Self-Training Example")
    print("=" * 50)

    dd_self = DualDescriptorTS(
        charset,
        rank=rank,
        vec_dim=vec_dim,
        num_basis=num_basis,
        mode='nonlinear',
        user_step=user_step,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    self_seqs = []
    for _ in range(10):
        L = random.randint(200, 300)
        self_seqs.append(''.join(random.choices(charset, k=L)))

    print("\nTraining for self-consistency:")
    dd_self.self_train(self_seqs, max_iters=50, tol=1e-8, learning_rate=0.01, batch_size=1024)

    print("\nGenerated sequences from model (generate method):")
    for i in range(2):
        gen_seq = dd_self.generate(100, tau=0.2)
        print(f"Sequence {i+1}: {gen_seq[:50]}...")

    # =====================================================================
    # reg_train with a target dimension different from m, plus r_generate
    # =====================================================================
    print("\n" + "=" * 50)
    print("reg_train + r_generate with a different target dimension")
    print("=" * 50)

    target_dim_diff = 5
    print(f"Model dimension (m) = {vec_dim}, target dimension = {target_dim_diff}")

    dd_reg_diff = DualDescriptorTS(
        charset,
        rank=rank,
        vec_dim=vec_dim,
        num_basis=num_basis,
        mode='nonlinear',
        user_step=user_step,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    t_list_diff = []
    for _ in range(100):
        t_list_diff.append([random.uniform(-1.0, 1.0) for _ in range(target_dim_diff)])

    print("\nTraining regression with target_dim =", target_dim_diff)
    dd_reg_diff.reg_train(seqs, t_list_diff, target_dim=target_dim_diff,
                          max_iters=100, tol=1e-9, learning_rate=0.1,
                          decay_rate=0.99, batch_size=2048, print_every=20)

    pred_t_diff = [dd_reg_diff.predict_r(seq) for seq in seqs]
    corr_sum_diff = 0.0
    for i in range(target_dim_diff):
        actu = [t[i] for t in t_list_diff]
        pred = [p[i] for p in pred_t_diff]
        corr = correlation(actu, pred)
        print(f"Dimension {i} prediction correlation: {corr:.4f}")
        corr_sum_diff += corr
    corr_avg_diff = corr_sum_diff / target_dim_diff
    print(f"Average correlation (target_dim={target_dim_diff}): {corr_avg_diff:.4f}")

    # r_generate with a target dimension different from m
    print("\n--- r_generate with a target_dim_diff-dimensional r ---")
    my_r_diff = [random.uniform(-1.0, 1.0) for _ in range(target_dim_diff)]
    seq_r_diff = dd_reg_diff.r_generate(L=120, r=my_r_diff, tau=0.0)
    print("Deterministic (tau=0):   ", seq_r_diff[:50] + "...")
    seq_r_diff_rand = dd_reg_diff.r_generate(L=120, r=my_r_diff, tau=0.5)
    print("Stochastic  (tau=0.5):   ", seq_r_diff_rand[:50] + "...")

    print("\nAll tests completed successfully!")
