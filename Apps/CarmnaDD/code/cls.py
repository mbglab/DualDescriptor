# -*- coding: utf-8 -*-
"""
Classification on nitrogenase multi-subunit sequences.

Pipeline:
  1. Read nifD + nifH + nifK sequences from Excel, concatenate and clean;
  2. Read classification labels from CSV;
  3. Run 5-fold stratified cross-validation with early stopping
     (based on validation loss), and select the fold with the highest validation AUC;
  4. Evaluate the selected model on the held-out test set;
  5. Save training / validation / test metrics to a CSV file.
"""

import os
import random
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score

from DDvTS import DualDescriptorTS

# ========================= Hyperparameters =========================
RANK = 3
VEC_DIM = 30
NUM_BASIS = 12
MODE = 'linear'
USER_STEP = None
RANK_MODE = 'drop'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Training settings
N_FOLDS = 5
LEARNING_RATE = 0.01
BATCH_SIZE = 32
MAX_ITERS = 100
PATIENCE = 10
MIN_DELTA = 1e-6
THRESHOLD = 0.5   # Unified classification threshold

# Data paths
EXCEL_PATH = r"Data/Data_literature.xlsx"
DATASET_CSV = r"Data/dataset.csv"
TRAIN_INDEX = r"Data/train_index.csv"
TEST_INDEX = r"Data/test_index.csv"


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
def compute_classification_metrics(y_true, y_pred, y_proba):
    """Compute AUC, F1, accuracy, precision and recall."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_proba = np.asarray(y_proba)
    if y_proba.ndim == 2:
        y_proba = y_proba[:, 1]

    if len(np.unique(y_true)) < 2:
        auc_val = 0.0
    else:
        auc_val = roc_auc_score(y_true, y_proba)

    return {
        'auc': auc_val,
        'f1': f1_score(y_true, y_pred),
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred),
    }


def print_cls_metrics(metrics, title="Classification Metrics"):
    print(f"\n{title}:")
    print(f"  AUC:       {metrics['auc']:.4f}")
    print(f"  F1:        {metrics['f1']:.4f}")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")


# ========================= Prediction =========================
def predict_proba(model, sequence):
    """Return the class probability vector for a single sequence."""
    model.eval()
    with torch.no_grad():
        _, probs = model.predict_c(sequence)
    return probs


def apply_threshold(probs, threshold=THRESHOLD):
    """Convert class probabilities to a binary label using the given threshold."""
    return 1 if probs[1] >= threshold else 0


# ========================= Training with early stopping =========================
def train_cls_with_early_stopping(model, train_seqs, train_labels, val_seqs, val_labels,
                                  max_iters=100, learning_rate=0.01, decay_rate=0.99,
                                  batch_size=32, print_every=10, patience=10, min_delta=1e-6,
                                  device='cuda', threshold=0.5):
    """
    Train the classifier and stop early based on validation loss.

    Returns
    -------
    best_model_state : dict
        Model state_dict at the lowest validation loss.
    best_val_metrics : dict
        Validation metrics computed at the best checkpoint.
    """
    num_classes = 2
    if model.classifier is None or model.num_classes != num_classes:
        model.classifier = nn.Linear(model.m, num_classes).to(device)
        model.num_classes = num_classes
        model.reset_parameters()

    train_labels_t = torch.tensor(train_labels, dtype=torch.long, device=device)
    val_labels_t = torch.tensor(val_labels, dtype=torch.long, device=device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    best_model_state = None
    best_val_metrics = None
    wait = 0

    for epoch in range(max_iters):
        # ---------- Training ----------
        model.train()
        total_loss, total_seqs = 0.0, 0

        indices = list(range(len(train_seqs)))
        random.shuffle(indices)

        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start:start + batch_size]
            batch_seqs = [train_seqs[i] for i in batch_idx]
            batch_labels = train_labels_t[batch_idx]

            optimizer.zero_grad()
            batch_loss = 0.0

            for seq, label in zip(batch_seqs, batch_labels):
                tokens = model.extract_tokens(seq)
                if not tokens:
                    seq_vector = torch.zeros(model.m, device=device)
                else:
                    token_indices = model.token_to_indices(tokens)
                    k_positions = torch.arange(len(tokens),
                                               dtype=torch.float32, device=device)
                    Nk_batch = model.batch_compute_Nk(k_positions, token_indices)
                    seq_vector = torch.mean(Nk_batch, dim=0)

                logits = model.classifier(seq_vector.unsqueeze(0))
                loss = criterion(logits, label.unsqueeze(0))
                batch_loss = batch_loss + loss
                del seq_vector, logits, loss

            batch_loss = batch_loss / len(batch_seqs)
            batch_loss.backward()
            optimizer.step()

            total_loss += batch_loss.item() * len(batch_seqs)
            total_seqs += len(batch_seqs)

        avg_train_loss = total_loss / total_seqs if total_seqs else 0.0

        # ---------- Validation ----------
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_probs = []
        with torch.no_grad():
            for seq, label in zip(val_seqs, val_labels_t):
                tokens = model.extract_tokens(seq)
                if not tokens:
                    seq_vector = torch.zeros(model.m, device=device)
                else:
                    token_indices = model.token_to_indices(tokens)
                    k_positions = torch.arange(len(tokens),
                                               dtype=torch.float32, device=device)
                    Nk_batch = model.batch_compute_Nk(k_positions, token_indices)
                    seq_vector = torch.mean(Nk_batch, dim=0)

                logits = model.classifier(seq_vector.unsqueeze(0))
                loss = criterion(logits, label.unsqueeze(0))
                val_loss += loss.item()
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                val_probs.append(probs)
                val_preds.append(apply_threshold(probs, threshold))

        val_loss /= len(val_seqs)
        val_metrics = compute_classification_metrics(
            val_labels_t.cpu().numpy(), val_preds, val_probs
        )

        # ---------- Early stopping ----------
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            best_val_metrics = val_metrics
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch} (val_loss={val_loss:.6f})")
                break

        if epoch % print_every == 0 or epoch == max_iters - 1:
            current_lr = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch:3d}: Train Loss = {avg_train_loss:.6f}, "
                  f"Val Loss = {val_loss:.6f}, Val AUC = {val_metrics['auc']:.4f}, "
                  f"LR = {current_lr:.6f}")

        scheduler.step()

    # Fallback when no improvement was ever recorded
    if best_model_state is None:
        best_model_state = copy.deepcopy(model.state_dict())
        model.eval()
        val_preds = []
        val_probs = []
        with torch.no_grad():
            for seq in val_seqs:
                _, probs = model.predict_c(seq)
                val_probs.append(probs)
                val_preds.append(apply_threshold(probs, threshold))
        best_val_metrics = compute_classification_metrics(
            val_labels_t.cpu().numpy(), val_preds, val_probs
        )

    return best_model_state, best_val_metrics


# ========================= Main =========================
def main():
    # ---------- Random seeds ----------
    torch.manual_seed(1)
    random.seed(2)
    np.random.seed(3)

    # ---------- 1. Load data ----------
    print("Loading sequences from Excel...")
    sequences = load_sequences_from_excel(EXCEL_PATH, sheet_name='Sheet2')
    print(f"Total sequences loaded: {len(sequences)}")

    charset = get_charset(sequences)
    print(f"Alphabet size: {len(charset)}")
    print(f"Characters: {''.join(charset)}")

    df_dataset = pd.read_csv(DATASET_CSV, encoding='ISO-8859-1')
    labels = df_dataset["label_classification"].values.astype(np.int32)
    print(f"Labels loaded, shape: {labels.shape}")

    train_idx = pd.read_csv(TRAIN_INDEX)['train_index'].values
    test_idx  = pd.read_csv(TEST_INDEX)['test_index'].values

    train_seqs_all = [sequences[i] for i in train_idx]
    test_seqs      = [sequences[i] for i in test_idx]
    train_labels   = labels[train_idx]
    test_labels    = labels[test_idx]

    print(f"Training set size: {len(train_seqs_all)}")
    print(f"Test set size: {len(test_seqs)}")
    print(f"Train label distribution: {np.bincount(train_labels)}")
    print(f"Test label distribution: {np.bincount(test_labels)}")

    print("\n" + "=" * 80)
    print(f"TRAINING SINGLE COMBO: rank={RANK}, vec_dim={VEC_DIM}, num_basis={NUM_BASIS}")
    print(f"Threshold = {THRESHOLD}")
    print("=" * 80)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    fold_train_metrics_list = []
    fold_val_metrics_list = []
    best_overall_auc = -np.inf
    best_overall_state = None

    model = None  # last-fold model, reused for test-set evaluation

    for fold, (tr_idx, val_idx) in enumerate(skf.split(train_seqs_all, train_labels)):
        print(f"\n--- Fold {fold + 1}/{N_FOLDS} ---")

        # Independent random seed per fold for reproducibility
        seed = RANK * 100000 + VEC_DIM * 1000 + NUM_BASIS * 100 + fold + 42
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        train_seqs        = [train_seqs_all[i] for i in tr_idx]
        val_seqs          = [train_seqs_all[i] for i in val_idx]
        train_labels_fold = [train_labels[i] for i in tr_idx]
        val_labels_fold   = [train_labels[i] for i in val_idx]

        model = DualDescriptorTS(
            charset=charset, rank=RANK, rank_mode=RANK_MODE,
            vec_dim=VEC_DIM, num_basis=NUM_BASIS,
            mode=MODE, user_step=USER_STEP, device=DEVICE,
        )

        best_state_fold, val_metrics = train_cls_with_early_stopping(
            model=model,
            train_seqs=train_seqs, train_labels=train_labels_fold,
            val_seqs=val_seqs,     val_labels=val_labels_fold,
            max_iters=MAX_ITERS,   learning_rate=LEARNING_RATE,
            batch_size=BATCH_SIZE, patience=PATIENCE,
            min_delta=MIN_DELTA,   print_every=10,
            device=DEVICE,         threshold=THRESHOLD,
        )

        # Evaluate the training set with this fold's best weights
        model.load_state_dict(best_state_fold)
        train_preds = []
        train_probs = []
        for seq in train_seqs:
            probs = predict_proba(model, seq)
            train_probs.append(probs)
            train_preds.append(apply_threshold(probs, THRESHOLD))
        train_metrics = compute_classification_metrics(
            train_labels_fold, train_preds, train_probs
        )

        fold_train_metrics_list.append(train_metrics)
        fold_val_metrics_list.append(val_metrics)

        print(f"  Fold {fold + 1}: Train AUC={train_metrics['auc']:.4f}, "
              f"Val AUC={val_metrics['auc']:.4f}")

        if val_metrics['auc'] > best_overall_auc:
            best_overall_auc   = val_metrics['auc']
            best_overall_state = best_state_fold

        del best_state_fold
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---------- Cross-validation summary ----------
    metric_keys = ['auc', 'f1', 'accuracy', 'precision', 'recall']
    avg_train_metrics = {
        k: float(np.mean([m[k] for m in fold_train_metrics_list]))
        for k in metric_keys
    }
    avg_val_metrics = {
        k: float(np.mean([m[k] for m in fold_val_metrics_list]))
        for k in metric_keys
    }
    std_val_metrics = {
        k: float(np.std([m[k] for m in fold_val_metrics_list]))
        for k in metric_keys
    }

    print(f"\nAvg Val AUC = {avg_val_metrics['auc']:.4f} "
          f"± {std_val_metrics['auc']:.4f}")

    # ---------- 4. Test-set evaluation ----------
    # Reuse the last-fold model structure and overwrite weights with the best fold
    model.load_state_dict(best_overall_state)

    test_preds = []
    test_probs = []
    for seq in test_seqs:
        probs = predict_proba(model, seq)
        test_probs.append(probs)
        test_preds.append(apply_threshold(probs, THRESHOLD))
    test_metrics = compute_classification_metrics(test_labels, test_preds, test_probs)
    print_cls_metrics(test_metrics, "Test Set Performance")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---------- Save results ----------
    print("\n" + "=" * 80)
    print("TRAINING COMPLETED")
    print("=" * 80)

    output_dir = "./cls1_results"
    os.makedirs(output_dir, exist_ok=True)

    row = {
        'rank': RANK,
        'vec_dim': VEC_DIM,
        'num_basis': NUM_BASIS,
        'threshold': THRESHOLD,
        'train_auc':       avg_train_metrics['auc'],
        'train_f1':        avg_train_metrics['f1'],
        'train_accuracy':  avg_train_metrics['accuracy'],
        'train_precision': avg_train_metrics['precision'],
        'train_recall':    avg_train_metrics['recall'],
        'val_auc':         avg_val_metrics['auc'],
        'val_f1':          avg_val_metrics['f1'],
        'val_accuracy':    avg_val_metrics['accuracy'],
        'val_precision':   avg_val_metrics['precision'],
        'val_recall':      avg_val_metrics['recall'],
        'std_auc':         std_val_metrics['auc'],
        'std_f1':          std_val_metrics['f1'],
        'std_accuracy':    std_val_metrics['accuracy'],
        'std_precision':   std_val_metrics['precision'],
        'std_recall':      std_val_metrics['recall'],
        'test_auc':        test_metrics['auc'],
        'test_f1':         test_metrics['f1'],
        'test_accuracy':   test_metrics['accuracy'],
        'test_precision':  test_metrics['precision'],
        'test_recall':     test_metrics['recall'],
        'best_val_auc':    best_overall_auc,
    }

    df_result = pd.DataFrame([row])
    csv_path = os.path.join(output_dir, "33012single_combo_results.csv")
    df_result.to_csv(csv_path, index=False)
    print(f"\nAll results saved to {csv_path}")

    print("\nAll tasks completed successfully!")


if __name__ == "__main__":
    main()