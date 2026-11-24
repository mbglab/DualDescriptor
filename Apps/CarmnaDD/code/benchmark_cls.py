# -*- coding: utf-8 -*-
"""
Multi-model classification benchmark on nitrogenase multi-subunit sequences.

Evaluation strategy aligned with the CarmnaDD classification setup:
  - Train/test split taken from train_index.csv / test_index.csv;
  - Features: 3-mer frequency vectors (L1-normalized);
  - Six classifiers with fixed hyperparameters;
  - Stratified 5-fold cross-validation; the fold with the highest validation AUC
    is selected as the best model;
  - Report train / validation (5-fold average) / test (best fold) metrics.
"""

import os
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score,
    precision_score, recall_score, average_precision_score,
)
from sklearn.base import clone

# Optional dependencies (skipped if not installed)
try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None
try:
    from lightgbm import LGBMClassifier
    import lightgbm as lgb
except ImportError:
    LGBMClassifier = None
    lgb = None


# ========================= 1. Sequence processing =========================

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


# ========================= 2. 3-mer frequency features =========================

def kmer_features(seqs, k=3, fit=False, vectorizer=None):
    """
    Compute L1-normalized k-mer frequency features.

    When ``fit=True`` a new CountVectorizer is fitted on ``seqs``;
    otherwise the provided ``vectorizer`` is reused to transform ``seqs``.
    """
    def kmer_split(seq):
        return [seq[i:i + k] for i in range(len(seq) - k + 1)]

    if fit:
        vectorizer = CountVectorizer(analyzer=kmer_split,
                                     dtype=np.float32, lowercase=False)
        X = vectorizer.fit_transform(seqs)
    else:
        X = vectorizer.transform(seqs)
    X_norm = normalize(X, norm='l1', axis=1)
    return X_norm, vectorizer


# ========================= 3. Classification metrics =========================

def compute_classification_metrics(y_true, y_pred, y_proba):
    """
    Compute AUC, F1, accuracy, precision, recall and AUPRC.

    ``y_proba`` may be a 1D array of positive-class probabilities
    or a 2D array of shape (n_samples, 2).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_proba = np.asarray(y_proba)
    if y_proba.ndim == 2:
        y_proba = y_proba[:, 1]

    if len(np.unique(y_true)) < 2:
        auc_val = 0.0
        auprc_val = 0.0
    else:
        auc_val = roc_auc_score(y_true, y_proba)
        auprc_val = average_precision_score(y_true, y_proba)

    return {
        'auc': auc_val,
        'f1': f1_score(y_true, y_pred),
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'auprc': auprc_val,
    }


# ========================= 4. Main =========================

def main():
    seed = 42
    random.seed(seed)
    np.random.seed(seed)

    # -------------------- Paths and settings --------------------
    excel_path       = r"Data/Data_literature.xlsx"
    dataset_csv_path = r"Data/dataset.csv"
    train_index_path = r"Data/train_index.csv"
    test_index_path  = r"Data/test_index.csv"

    n_folds = 5

    # -------------------- Load data --------------------
    print("Loading sequences from Excel...")
    sequences = load_sequences_from_excel(excel_path, sheet_name='Sheet2')
    print(f"Total sequences loaded: {len(sequences)}")

    charset = get_charset(sequences)
    print(f"Alphabet size: {len(charset)}")

    df_dataset = pd.read_csv(dataset_csv_path, encoding='ISO-8859-1')
    labels = df_dataset["label_classification"].values.astype(np.int32)
    print(f"Labels loaded, shape: {labels.shape}")

    train_idx = pd.read_csv(train_index_path)['train_index'].values
    test_idx  = pd.read_csv(test_index_path)['test_index'].values

    train_seqs_all = [sequences[i] for i in train_idx]
    test_seqs      = [sequences[i] for i in test_idx]
    train_labels   = labels[train_idx]
    test_labels    = labels[test_idx]

    print(f"Training set size: {len(train_seqs_all)}")
    print(f"Test set size: {len(test_seqs)}")
    print(f"Train label distribution: {np.bincount(train_labels)}")
    print(f"Test label distribution: {np.bincount(test_labels)}")

    # -------------------- 3-mer features --------------------
    print("\nExtracting 3-mer frequency features...")
    X_train, vectorizer = kmer_features(train_seqs_all, fit=True)
    X_test, _ = kmer_features(test_seqs, fit=False, vectorizer=vectorizer)
    print(f"Feature dimension: {X_train.shape[1]}")

    # Convert to dense matrices (dimension is manageable)
    X_train_dense = X_train.toarray()
    X_test_dense = X_test.toarray()

    # -------------------- Classifiers (fixed hyperparameters) --------------------
    models = {}

    # SVM (with standardization)
    models['SVM'] = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', SVC(probability=True, random_state=seed)),
    ])

    # Random Forest
    models['RandomForest'] = RandomForestClassifier(
        max_features='sqrt',
        random_state=seed,
        n_jobs=-1,
    )

    # Gradient Boosting
    models['GBDT'] = GradientBoostingClassifier(
        n_iter_no_change=10,
        validation_fraction=0.1,
        random_state=seed,
    )

    # MLP (with standardization)
    models['MLP'] = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', MLPClassifier(
            max_iter=500,
            early_stopping=True,
            random_state=seed,
        )),
    ])

    # XGBoost (optional)
    if XGBClassifier is not None:
        models['XGBoost'] = XGBClassifier(
            early_stopping_rounds=10,
            random_state=seed,
            eval_metric='logloss',
            verbosity=0,
        )
    else:
        print("Warning: XGBoost not installed, skipping.")

    # LightGBM (optional)
    if LGBMClassifier is not None:
        models['LGBM'] = LGBMClassifier(
            learning_rate=0.5,
            random_state=seed,
            verbose=-1,
        )
    else:
        print("Warning: LightGBM not installed, skipping.")

    # -------------------- Stratified cross-validation --------------------
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    results = []
    test_preds_baseline = {}  # best-fold test predictions per model (for ROC / Venn downstream)

    for name, model in models.items():
        print(f"\n{'=' * 60}")
        print(f"Training {name} ...")
        print('=' * 60)

        fold_train_metrics = []
        fold_val_metrics = []
        best_val_auc = -np.inf
        best_test_proba = None
        best_test_pred = None

        for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train_dense, train_labels)):
            X_tr  = X_train_dense[tr_idx]
            X_val = X_train_dense[val_idx]
            y_tr  = train_labels[tr_idx]
            y_val = train_labels[val_idx]

            model_fold = clone(model)

            # Pass the validation set to XGBoost / LightGBM for early stopping
            if name == 'XGBoost':
                model_fold.fit(X_tr, y_tr,
                               eval_set=[(X_val, y_val)], verbose=False)
            elif name == 'LGBM':
                callbacks = [lgb.early_stopping(10)]
                model_fold.fit(X_tr, y_tr,
                               eval_set=[(X_val, y_val)], callbacks=callbacks)
            else:
                model_fold.fit(X_tr, y_tr)

            # Training-set metrics
            y_tr_pred = model_fold.predict(X_tr)
            y_tr_proba = model_fold.predict_proba(X_tr)
            train_metrics = compute_classification_metrics(
                y_tr, y_tr_pred, y_tr_proba
            )

            # Validation-set metrics
            y_val_pred = model_fold.predict(X_val)
            y_val_proba = model_fold.predict_proba(X_val)
            val_metrics = compute_classification_metrics(
                y_val, y_val_pred, y_val_proba
            )

            fold_train_metrics.append(train_metrics)
            fold_val_metrics.append(val_metrics)

            print(f"  Fold {fold + 1}: Train AUC={train_metrics['auc']:.4f}, "
                  f"Val AUC={val_metrics['auc']:.4f}")

            # Update the best fold (highest validation AUC)
            if val_metrics['auc'] > best_val_auc:
                best_val_auc = val_metrics['auc']
                y_test_pred = model_fold.predict(X_test_dense)
                y_test_proba = model_fold.predict_proba(X_test_dense)
                best_test_proba = y_test_proba[:, 1]
                best_test_pred = y_test_pred

        # Save the best-fold test predictions for this model
        test_preds_baseline[name] = {'proba': best_test_proba, 'pred': best_test_pred}

        # Average across folds
        avg_train = {
            k: float(np.mean([m[k] for m in fold_train_metrics]))
            for k in fold_train_metrics[0]
        }
        avg_val = {
            k: float(np.mean([m[k] for m in fold_val_metrics]))
            for k in fold_val_metrics[0]
        }

        results.append({
            'model':           name,
            'train_auc':       avg_train['auc'],
            'train_f1':        avg_train['f1'],
            'train_accuracy':  avg_train['accuracy'],
            'train_precision': avg_train['precision'],
            'train_recall':    avg_train['recall'],
            'train_auprc':     avg_train['auprc'],
            'val_auc':         avg_val['auc'],
            'val_f1':          avg_val['f1'],
            'val_accuracy':    avg_val['accuracy'],
            'val_precision':   avg_val['precision'],
            'val_recall':      avg_val['recall'],
            'val_auprc':       avg_val['auprc'],
            'test_auc':        roc_auc_score(test_labels, best_test_proba),
            'test_f1':         f1_score(test_labels, best_test_pred),
            'test_accuracy':   accuracy_score(test_labels, best_test_pred),
            'test_precision':  precision_score(test_labels, best_test_pred, zero_division=0),
            'test_recall':     recall_score(test_labels, best_test_pred, zero_division=0),
            'test_auprc':      average_precision_score(test_labels, best_test_proba),
        })

        print(f"  => Best Val AUC = {best_val_auc:.4f}, "
              f"Test AUC = {results[-1]['test_auc']:.4f}")

    # -------------------- Summary --------------------
    print("\n" + "=" * 80)
    print("SUMMARY OF ALL MODELS (sorted by test AUC)")
    print("=" * 80)
    df_results = pd.DataFrame(results)
    df_sorted = df_results.sort_values('test_auc', ascending=False).round(4)
    display_cols = ['model', 'test_auc', 'test_auprc', 'test_f1',
                    'test_accuracy', 'test_precision', 'test_recall']
    print(df_sorted[display_cols].to_string(index=False))

    output_dir = "./ml_classification_results"
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "six_models_classification_metrics.csv")
    df_sorted.to_csv(csv_path, index=False)
    print(f"\nFull results saved to {csv_path}")

    # Save best-fold test predictions (for downstream ROC / Venn plots)
    pred_df = pd.DataFrame({'true_label': test_labels})
    for name in models.keys():
        pred_df[f'{name}_proba'] = test_preds_baseline[name]['proba']
        pred_df[f'{name}_pred'] = test_preds_baseline[name]['pred']
    pred_df.to_csv(
        os.path.join(output_dir, "baseline_cls_test_predictions.csv"),
        index=False,
    )
    print("Baseline test predictions saved.")

    print("\nAll tasks completed successfully!")


if __name__ == "__main__":
    main()