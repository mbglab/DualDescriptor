# -*- coding: utf-8 -*-
"""
Multi-model regression benchmark on nitrogenase multi-subunit sequences.

Evaluation strategy aligned with the CarmnaDD regression setup:
  - Train/test split taken from train_index.csv / test_index.csv;
  - Features: 3-mer frequency vectors (L1-normalized);
  - Six regressors with fixed hyperparameters;
  - 5-fold cross-validation; the fold with the highest validation R²
    is selected as the best model;
  - Report train / validation (5-fold average) / test (best fold) metrics
    (R², RMSE, MSE, MAE).
"""

import os
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.base import clone
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import lightgbm as lgb


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


# ========================= 3. Regression metrics =========================

def compute_regression_metrics(y_true, y_pred):
    """Compute R², RMSE, MSE and MAE."""
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    if len(y_true) < 2:
        return {'r2': np.nan, 'rmse': np.nan, 'mse': np.nan, 'mae': np.nan}

    mse = mean_squared_error(y_true, y_pred)
    return {
        'r2': r2_score(y_true, y_pred),
        'rmse': np.sqrt(mse),
        'mse': mse,
        'mae': mean_absolute_error(y_true, y_pred),
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
    labels_log = df_dataset["label_regression"].values.astype(np.float32)
    print(f"Labels (log scale) loaded, shape: {labels_log.shape}")

    train_idx = pd.read_csv(train_index_path)['train_index'].values
    test_idx  = pd.read_csv(test_index_path)['test_index'].values

    train_seqs_all = [sequences[i] for i in train_idx]
    test_seqs      = [sequences[i] for i in test_idx]
    train_labels   = labels_log[train_idx]
    test_labels    = labels_log[test_idx]

    print(f"Training set size: {len(train_seqs_all)}")
    print(f"Test set size: {len(test_seqs)}")

    # -------------------- 3-mer features --------------------
    print("\nExtracting 3-mer frequency features...")
    X_train, vectorizer = kmer_features(train_seqs_all, fit=True)
    X_test, _ = kmer_features(test_seqs, fit=False, vectorizer=vectorizer)
    print(f"Feature dimension: {X_train.shape[1]}")

    X_train_dense = X_train.toarray()
    X_test_dense = X_test.toarray()

    # -------------------- Regressors (fixed hyperparameters) --------------------
    models = {}

    # SVR (with standardization)
    models['SVR'] = Pipeline([
        ('scaler', StandardScaler()),
        ('reg', SVR()),
    ])

    # Random Forest
    models['RandomForest'] = RandomForestRegressor(
        max_features='sqrt',
        random_state=seed,
        n_jobs=-1,
    )

    # Gradient Boosting
    models['GBDT'] = GradientBoostingRegressor(
        n_iter_no_change=10,
        validation_fraction=0.1,
        random_state=seed,
    )

    # MLP (with standardization)
    models['MLP'] = Pipeline([
        ('scaler', StandardScaler()),
        ('reg', MLPRegressor(
            hidden_layer_sizes=(100,),
            activation='tanh',
            solver='adam',
            alpha=0.0001,
            learning_rate_init=0.05,
            random_state=seed,
        )),
    ])

    # XGBoost
    models['XGBoost'] = XGBRegressor(
        early_stopping_rounds=10,
        random_state=seed,
        eval_metric='rmse',
        verbosity=0,
    )

    # LightGBM
    models['LGBM'] = LGBMRegressor(
        learning_rate=0.5,
        random_state=seed,
        verbose=-1,
    )

    # -------------------- Plain KFold cross-validation --------------------
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

    results = []

    for name, model in models.items():
        print(f"\n{'=' * 60}")
        print(f"Training {name} ...")
        print('=' * 60)

        fold_train_metrics = []
        fold_val_metrics = []
        best_val_r2 = -np.inf
        best_test_metrics = None

        for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train_dense)):
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
            train_metrics = compute_regression_metrics(y_tr, y_tr_pred)

            # Validation-set metrics
            y_val_pred = model_fold.predict(X_val)
            val_metrics = compute_regression_metrics(y_val, y_val_pred)

            fold_train_metrics.append(train_metrics)
            fold_val_metrics.append(val_metrics)

            print(f"  Fold {fold + 1}: Train R²={train_metrics['r2']:.4f}, "
                  f"Val R²={val_metrics['r2']:.4f}")

            # Update the best fold (highest validation R²)
            if val_metrics['r2'] > best_val_r2:
                best_val_r2 = val_metrics['r2']
                y_test_pred = model_fold.predict(X_test_dense)
                best_test_metrics = compute_regression_metrics(
                    test_labels, y_test_pred
                )

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
            'model':         name,
            'train_r2':      avg_train['r2'],
            'train_rmse':    avg_train['rmse'],
            'train_mse':     avg_train['mse'],
            'train_mae':     avg_train['mae'],
            'val_r2':        avg_val['r2'],
            'val_rmse':      avg_val['rmse'],
            'val_mse':       avg_val['mse'],
            'val_mae':       avg_val['mae'],
            'test_r2':       best_test_metrics['r2'],
            'test_rmse':     best_test_metrics['rmse'],
            'test_mse':      best_test_metrics['mse'],
            'test_mae':      best_test_metrics['mae'],
        })

        print(f"  => Best Val R² = {best_val_r2:.4f}, "
              f"Test R² = {best_test_metrics['r2']:.4f}")

    # -------------------- Summary --------------------
    print("\n" + "=" * 80)
    print("SUMMARY OF ALL MODELS (sorted by test R²)")
    print("=" * 80)
    df_results = pd.DataFrame(results)
    df_sorted = df_results.sort_values('test_r2', ascending=False).round(4)
    display_cols = ['model', 'test_r2', 'test_rmse', 'test_mae']
    print(df_sorted[display_cols].to_string(index=False))

    output_dir = "./ml_regression_results"
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "six_models_regression_metrics.csv")
    df_sorted.to_csv(csv_path, index=False)
    print(f"\nFull results saved to {csv_path}")

    print("\nAll tasks completed successfully!")


if __name__ == "__main__":
    main()