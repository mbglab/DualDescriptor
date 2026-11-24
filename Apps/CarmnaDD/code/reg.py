# -*- coding: utf-8 -*-
"""
DDvTS  -- regression on nitrogenase multi-subunit sequences.

Pipeline:
  1. Read nifD + nifH + nifK sequences from Excel, concatenate and clean;
  2. Read log-transformed regression labels from CSV;
  3. Run 5-fold cross-validation with early stopping on the training set;
  4. Select the fold with the highest validation R² and evaluate it on the held-out test set.

No additional normalization is applied to the labels.
"""

import random
import copy
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from sklearn.model_selection import KFold
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from DDvTS import DualDescriptorTS


# ========================= Sequence processing =========================

STANDARD_AA = set('ACDEFGHIKLMNPQRSTVWY')


def clean_sequence(seq):
    """Keep only standard amino acid characters."""
    return ''.join(ch for ch in seq.upper() if ch in STANDARD_AA)


def load_sequences_from_excel(filepath, sheet_name='Sheet2'):
    """
    Read nifD / nifH / nifK sequences from an Excel file,
    concatenate them into a single sequence, and clean it.

    Returns
    -------
    sequences : list[str]
        Sequences aligned with the row order of dataset.csv.
    """
    df = pd.read_excel(filepath, sheet_name=sheet_name, header=1)
    sequences = []
    for _, row in df.iterrows():
        seq_d = str(row['Sequence_nifD']).strip()
        seq_h = str(row['Sequence_nifH']).strip()
        seq_k = str(row['Sequence_nifK']).strip()

        # Skip missing or too-short sequences
        if 'nan' in (seq_d, seq_h, seq_k):
            continue
        if min(len(seq_d), len(seq_h), len(seq_k)) < 10:
            continue

        full_seq = clean_sequence(seq_d + seq_h + seq_k)
        if full_seq:
            sequences.append(full_seq)
    return sequences


def get_charset(sequences):
    """Return the sorted list of amino acid characters present in the dataset."""
    charset = set()
    for seq in sequences:
        charset.update(seq)
    return sorted(charset)


# ========================= Metrics =========================

def compute_regression_metrics(y_true, y_pred):
    """Compute Pearson correlation, R², RMSE, MSE and MAE."""
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    if len(y_true) < 2:
        return {k: np.nan for k in ('corr', 'r2', 'rmse', 'mse', 'mae')}

    corr, _ = pearsonr(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    return {
        'corr': corr,
        'r2': r2_score(y_true, y_pred),
        'rmse': np.sqrt(mse),
        'mse': mse,
        'mae': mean_absolute_error(y_true, y_pred),
    }


def print_metrics(metrics, title="Regression Metrics"):
    print(f"\n{title}:")
    print(f"  Correlation (Pearson): {metrics['corr']:.4f}")
    print(f"  R²:   {metrics['r2']:.4f}")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  MSE:  {metrics['mse']:.4f}")
    print(f"  MAE:  {metrics['mae']:.4f}")


# ========================= Prediction =========================

def predict_scalar(model, sequence):
    """Return a scalar prediction (mean of the descriptor vector, in log scale)."""
    model.eval()
    with torch.no_grad():
        vec = model.predict_t(sequence)
    return float(np.mean(vec))


# ========================= Training with early stopping =========================

def train_with_early_stopping(model, train_seqs, train_targets,
                              val_seqs, val_targets,
                              max_iters=100, learning_rate=0.01,
                              decay_rate=0.99, batch_size=32,
                              print_every=10, patience=10, min_delta=1e-6):
    """
    Train the model and stop early based on validation loss.

    Returns
    -------
    best_model_state : dict | None
        Model state_dict at the lowest validation loss.
    """
    device = model.device
    train_tensors = [torch.tensor(t, dtype=torch.float32, device=device)
                     for t in train_targets]
    val_tensors = [torch.tensor(t, dtype=torch.float32, device=device)
                   for t in val_targets]

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)

    best_val_loss = float('inf')
    best_model_state = None
    wait = 0

    for it in range(max_iters):
        # ---------- Training ----------
        model.train()
        total_loss, total_seqs = 0.0, 0

        indices = list(range(len(train_seqs)))
        random.shuffle(indices)

        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start:start + batch_size]
            batch_seqs = [train_seqs[i] for i in batch_idx]
            batch_targets = [train_tensors[i] for i in batch_idx]

            optimizer.zero_grad()
            batch_loss = 0.0

            for seq, target in zip(batch_seqs, batch_targets):
                tokens = model.extract_tokens(seq)
                if not tokens:
                    continue
                token_indices = model.token_to_indices(tokens)
                k_positions = torch.arange(len(tokens),
                                           dtype=torch.float32, device=device)
                Nk_batch = model.batch_compute_Nk(k_positions, token_indices)
                seq_pred = torch.mean(Nk_batch, dim=0)
                batch_loss = batch_loss + torch.sum((seq_pred - target) ** 2)
                del Nk_batch, seq_pred, token_indices, k_positions

            if len(batch_seqs) > 0:
                batch_loss = batch_loss / len(batch_seqs)
                batch_loss.backward()
                optimizer.step()
                total_loss += batch_loss.item() * len(batch_seqs)
                total_seqs += len(batch_seqs)

        avg_train_loss = total_loss / total_seqs if total_seqs else 0.0

        # ---------- Validation ----------
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for seq, target in zip(val_seqs, val_tensors):
                tokens = model.extract_tokens(seq)
                if not tokens:
                    continue
                token_indices = model.token_to_indices(tokens)
                k_positions = torch.arange(len(tokens),
                                           dtype=torch.float32, device=device)
                Nk = model.batch_compute_Nk(k_positions, token_indices)
                pred = torch.mean(Nk, dim=0)
                val_loss += torch.sum((pred - target) ** 2).item()
        val_loss /= len(val_seqs)

        # ---------- Early stopping ----------
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at iteration {it} "
                      f"(val_loss={val_loss:.6f})")
                break

        if it % print_every == 0 or it == max_iters - 1:
            current_lr = scheduler.get_last_lr()[0]
            print(f"Iter {it:3d}: Train Loss = {avg_train_loss:.6e}, "
                  f"Val Loss = {val_loss:.6e}, LR = {current_lr:.6f}")

        scheduler.step()

    return best_model_state


# ========================= Main =========================

def main():
    # ---------- Random seeds ----------
    torch.manual_seed(1)
    random.seed(2)
    np.random.seed(3)

    # ---------- Paths and hyperparameters ----------
    excel_path       = r"Data/Data_literature.xlsx"
    dataset_csv_path = r"Data/dataset.csv"
    train_index_path = r"Data/train_index.csv"
    test_index_path  = r"Data/test_index.csv"

    rank      = 3
    vec_dim   = 26
    num_basis = 18
    mode      = 'linear'
    user_step = None
    rank_mode = 'drop'
    device    = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Cross-validation and early-stopping settings
    n_folds       = 5
    learning_rate = 0.01
    batch_size    = 32
    max_iters     = 100
    patience      = 10
    min_delta     = 1e-6

    # ---------- 1. Load data ----------
    print("Loading sequences from Excel...")
    sequences = load_sequences_from_excel(excel_path, sheet_name='Sheet2')
    print(f"Total sequences loaded: {len(sequences)}")

    charset = get_charset(sequences)
    print(f"Alphabet size: {len(charset)}")
    print(f"Characters: {''.join(charset)}")

    df_dataset = pd.read_csv(dataset_csv_path, encoding='ISO-8859-1')
    labels_log = df_dataset["label_regression"].values.astype(np.float32)
    print(f"Labels loaded (log scale), shape: {labels_log.shape}")

    # ---------- 2. Split train / test ----------
    train_idx = pd.read_csv(train_index_path)['train_index'].values
    test_idx  = pd.read_csv(test_index_path)['test_index'].values

    train_seqs_all = [sequences[i] for i in train_idx]
    test_seqs      = [sequences[i] for i in test_idx]
    train_labels   = labels_log[train_idx]
    test_labels    = labels_log[test_idx]

    print(f"Training set size: {len(train_seqs_all)}")
    print(f"Test set size: {len(test_seqs)}")

    # Broadcast scalar labels to vec_dim-dimensional vectors
    # to match the model's descriptor dimension
    train_targets_all = np.tile(
        train_labels.reshape(-1, 1), (1, vec_dim)
    ).tolist()

    # ---------- 3. 5-fold cross-validation ----------
    print("\n" + "=" * 60)
    print("PART 1: 5-fold CV with Early Stopping")
    print("=" * 60)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_r2s = []
    best_overall_r2 = -np.inf
    best_overall_state = None

    for fold, (tr_idx, val_idx) in enumerate(kf.split(train_seqs_all)):
        print(f"\n--- Fold {fold + 1}/{n_folds} ---")

        # Independent random seed per fold for reproducibility
        seed = rank * 100000 + vec_dim * 1000 + num_basis * 100 + fold + 42
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        train_seqs    = [train_seqs_all[i] for i in tr_idx]
        val_seqs      = [train_seqs_all[i] for i in val_idx]
        train_targets = [train_targets_all[i] for i in tr_idx]
        val_targets   = [train_targets_all[i] for i in val_idx]

        model = DualDescriptorTS(
            charset=charset, rank=rank, rank_mode=rank_mode,
            vec_dim=vec_dim, num_basis=num_basis,
            mode=mode, user_step=user_step, device=device,
        )

        print(f"Training with early stopping (patience={patience})...")
        best_state = train_with_early_stopping(
            model=model,
            train_seqs=train_seqs, train_targets=train_targets,
            val_seqs=val_seqs,     val_targets=val_targets,
            max_iters=max_iters,   learning_rate=learning_rate,
            batch_size=batch_size, patience=patience,
            min_delta=min_delta,   print_every=10,
        )

        # Evaluate the validation set using this fold's best weights
        model.load_state_dict(best_state)
        val_pred = [predict_scalar(model, s) for s in val_seqs]
        val_true = [train_targets_all[i][0] for i in val_idx]
        r2_fold  = r2_score(val_true, val_pred)

        fold_r2s.append(r2_fold)
        print(f"Fold {fold + 1} Val R² = {r2_fold:.4f}")

        if r2_fold > best_overall_r2:
            best_overall_r2    = r2_fold
            best_overall_state = best_state
            print(f"    *** New best fold! R² = {r2_fold:.4f} ***")

    print(f"\n5-fold CV Avg R² = {np.mean(fold_r2s):.4f} "
          f"± {np.std(fold_r2s):.4f}")
    print(f"Best single-fold Val R² = {best_overall_r2:.4f}")

    # ---------- Save the best model ----------
    model_save_path = f"best_model_weights_rank{rank}_{mode}.pth"
    torch.save(best_overall_state, model_save_path)
    print(f"Best model weights saved to {model_save_path}")

    # ---------- 4. Test-set evaluation ----------
    print("\n" + "=" * 60)
    print("PART 2: Evaluate Best Fold Model on Test Set")
    print("=" * 60)

    final_model = DualDescriptorTS(
        charset=charset, rank=rank, rank_mode=rank_mode,
        vec_dim=vec_dim, num_basis=num_basis,
        mode=mode, user_step=user_step, device=device,
    )
    final_model.load_state_dict(best_overall_state)

    test_pred = np.array([predict_scalar(final_model, s) for s in test_seqs])
    test_metrics = compute_regression_metrics(test_labels, test_pred)
    print_metrics(test_metrics, "Test Set Performance (log scale)")

    print("\nAll tasks completed successfully!")


if __name__ == "__main__":
    main()