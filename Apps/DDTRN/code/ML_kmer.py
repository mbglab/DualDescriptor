#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Transcriptional regulatory network prediction - k-mer frequency + machine learning classifiers (benchmark)

import os
import json
import random
import argparse
import numpy as np
from itertools import product
from collections import Counter
from Bio import SeqIO
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


def load_sequences(fasta_file):
    """
    Load gene sequences
    
    Args:
        fasta_file: Path to the FASTA-formatted gene sequence file
        
    Returns:
        sequences: dict mapping gene IDs to sequences
    """
    print("\nLoading data...")
    sequences = {}
    for record in SeqIO.parse(fasta_file, "fasta"):
        gene_id = record.id.split("|")[0] if "|" in record.id else record.id
        sequences[gene_id] = str(record.seq).upper()
    print(f"  Loaded {len(sequences)} gene sequences")
    return sequences


def load_positive_pairs(tsv_file, sequences):
    """
    Load labeled gene pairs from a TSV file.
    
    Args:
        tsv_file: Path to the TSV-formatted regulatory relationship file
        sequences: Gene sequence dictionary
        
    Returns:
        all_positive_pairs: Valid positive sample list
        tf_to_targets: Mapping from TF to its regulated target genes
        labeled_negative_pairs: Valid negative sample list from the TSV file
    """
    all_positive_pairs = []
    labeled_negative_pairs = []
    skipped_missing_sequence = 0
    skipped_bad_label = 0

    with open(tsv_file, 'r') as f:
        for line_number, line in enumerate(f, start=1):
            parts = line.strip().split('\t')
            if len(parts) < 2:
                continue

            gene1, gene2 = parts[0], parts[1]
            label = None
            if len(parts) >= 3:
                try:
                    label = int(parts[2])
                except ValueError:
                    if line_number == 1 and parts[2].lower() in {'label', 'class'}:
                        continue
                    skipped_bad_label += 1
                    continue

                if label not in {0, 1}:
                    skipped_bad_label += 1
                    continue

            if gene1 not in sequences or gene2 not in sequences:
                skipped_missing_sequence += 1
                continue

            if label == 0:
                labeled_negative_pairs.append((gene1, gene2))
            else:
                all_positive_pairs.append((gene1, gene2))

    print(f"  Valid positive samples: {len(all_positive_pairs)}")

    tf_to_targets = {}
    for tf, target in all_positive_pairs:
        tf_to_targets.setdefault(tf, set()).add(target)
    
    return all_positive_pairs, tf_to_targets, labeled_negative_pairs


def generate_tf_based_negative_samples(all_positive_pairs, tf_to_targets, gene_list):
    """
    Generate TF-based negative samples (same number as positives)
    
    Args:
        all_positive_pairs: All positive pairs
        tf_to_targets: Mapping from TF to target genes
        gene_list: List of all genes
        
    Returns:
        negative_pairs: List of negative samples
    """
    positive_set = set(all_positive_pairs)
    gene_set = set(gene_list)
    tf_list = sorted(tf_to_targets.keys())
    
    negative_set = set()
    num_needed = len(all_positive_pairs)
    attempts = 0
    max_attempts = num_needed * 10

    while len(negative_set) < num_needed and attempts < max_attempts:
        tf = random.choice(tf_list)
        regulated_targets = tf_to_targets[tf]
        candidate_targets = gene_set - regulated_targets - {tf}
        
        if not candidate_targets:
            attempts += 1
            continue

        neg_target = random.choice(sorted(candidate_targets))
        neg_pair = (tf, neg_target)
        
        if neg_pair not in positive_set:
            negative_set.add(neg_pair)
        attempts += 1

    return sorted(negative_set)


def build_kmer_vocabulary(rank=6):
    """
    Build the full k-mer vocabulary
    
    Args:
        rank: k-mer length (default: 6)
        
    Returns:
        kmer_to_idx: Mapping from k-mer to index
        vocab_size: Vocabulary size (4^rank)
    """
    charset = ['A', 'C', 'G', 'T']
    all_kmers = [''.join(p) for p in product(charset, repeat=rank)]
    kmer_to_idx = {kmer: idx for idx, kmer in enumerate(all_kmers)}
    vocab_size = len(all_kmers)  # 4^rank
    print(f"  k-mer vocabulary size: {vocab_size} (rank={rank})")
    return kmer_to_idx, vocab_size


def extract_kmer_features(seq, kmer_to_idx, vocab_size, rank=6, mode='linear'):
    """
    Extract k-mer frequency features for a single sequence
    
    Args:
        seq: DNA sequence string
        kmer_to_idx: Mapping from k-mer to index
        vocab_size: Vocabulary size
        rank: k-mer length
        mode: 'linear' (overlapping, step=1) or 'nonlinear' (non-overlapping, step=rank)
        
    Returns:
        freq_vector: Normalized k-mer frequency vector
    """
    if mode == 'nonlinear':
        kmers = [seq[i:i+rank] for i in range(0, len(seq) - rank + 1, rank)]
    else:
        kmers = [seq[i:i+rank] for i in range(len(seq) - rank + 1)]
    
    # Keep only valid k-mers (composed solely of ACGT)
    valid_kmers = [k for k in kmers if k in kmer_to_idx]
    
    # Count frequencies
    freq_vector = np.zeros(vocab_size, dtype=np.float32)
    if valid_kmers:
        kmer_counts = Counter(valid_kmers)
        for kmer, count in kmer_counts.items():
            freq_vector[kmer_to_idx[kmer]] = count
        # Normalize
        freq_vector = freq_vector / len(valid_kmers)
    
    return freq_vector


def extract_features_batch(seq_pairs, kmer_to_idx, vocab_size, rank=6, mode='linear'):
    """
    Extract k-mer frequency features in batch
    
    Args:
        seq_pairs: List of sequence pairs (after concatenating TF_seq + Target_seq)
        kmer_to_idx: Mapping from k-mer to index
        vocab_size: Vocabulary size
        rank: k-mer length
        mode: 'linear' or 'nonlinear'
        
    Returns:
        X: Feature matrix (N, vocab_size)
    """
    print(f"\nExtracting {rank}-mer frequency features ({vocab_size} dims, mode={mode})...")
    features = []
    
    for i, seq in enumerate(seq_pairs):
        feat = extract_kmer_features(seq, kmer_to_idx, vocab_size, rank, mode)
        features.append(feat)
        
        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1}/{len(seq_pairs)} pairs...")
    
    X = np.array(features)
    print(f"  Feature matrix shape: {X.shape}")
    return X


def build_seq_pairs(pairs, sequences):
    """
    Build sequence pairs: seq_pair = TF_seq + Target_seq
    
    Args:
        pairs: List of gene pairs [(tf, target, label), ...]
        sequences: Gene sequence dictionary
        
    Returns:
        seq_pairs: List of sequence pairs
        labels: List of labels
    """
    seq_pairs = [sequences[tf] + sequences[target] for tf, target, _ in pairs]
    labels = [label for _, _, label in pairs]
    return seq_pairs, labels

def get_classifiers(seed=42):
    """
    Get all machine learning classifiers (using default parameters)
    
    Args:
        seed: Random seed
        
    Returns:
        classifiers: Classifier dictionary
    """
    all_classifiers = {
        'SVM': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', SVC(
                probability=True,  
                random_state=seed
            ))
        ]),
        'RandomForest': RandomForestClassifier(
            max_features='sqrt',  
            random_state=seed, 
            n_jobs=-1
        ),
        'GBDT': GradientBoostingClassifier(
            n_iter_no_change=10, 
            validation_fraction=0.1,
            random_state=seed
        ),
        'MLP': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', MLPClassifier(
                max_iter=500,
                early_stopping=True,  
                random_state=seed
            ))
        ]),
        'XGBoost': XGBClassifier(
            early_stopping_rounds=10,  
            random_state=seed,
            eval_metric='logloss',
            verbosity=0
        ),
        'LGBM': LGBMClassifier(
            learning_rate=0.5,  
            random_state=seed,
            verbose=-1
        )
    }
    return all_classifiers


def get_scores(clf, X):
    """
    Get predicted scores from the classifier, compatible with predict_proba and decision_function
    
    Args:
        clf: classifier
        X: Feature matrix
        
    Returns:
        scores: Predicted scores
    """
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(X)[:, 1]
    if hasattr(clf, "decision_function"):
        return clf.decision_function(X)
    # fallback
    return clf.predict(X)


def evaluate_classifier(clf, X_train, y_train, X_test, y_test, clf_name, X_val=None, y_val=None):
    """
    Train and evaluate a classifier
    
    Args:
        clf: classifier
        X_train, y_train: Training data
        X_test, y_test: Test data
        clf_name: Classifier name
        X_val, y_val: Validation data (optional, for early stopping)
        
    Returns:
        metrics: Evaluation metrics dictionary
    """
    print(f"\nTraining {clf_name}...")
    
    # Use a validation set for classifiers that support early stopping
    if X_val is not None and y_val is not None:
        if clf_name == 'XGBoost':
            clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            # Check whether early stopping was actually used
            if hasattr(clf, 'best_iteration') and clf.get_booster().best_iteration > 0:
                print(f"  XGBoost early-stopped at iteration {clf.best_iteration}")
            else:
                print(f"  XGBoost training complete")
        elif clf_name == 'LGBM':
            clf.fit(X_train, y_train, eval_set=[(X_val, y_val)])
            if hasattr(clf, 'best_iteration_') and clf.best_iteration_ > 0:
                print(f"  LGBM early-stopped at iteration {clf.best_iteration_}")
            else:
                print(f"  LGBM training complete")
        else:
            clf.fit(X_train, y_train)
    else:
        clf.fit(X_train, y_train)
    
    # Predict
    y_pred = clf.predict(X_test)
    y_score = get_scores(clf, X_test)
    
    # Compute metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'auroc': roc_auc_score(y_test, y_score) if len(set(y_test)) > 1 else 0,
        'aupr': average_precision_score(y_test, y_score) if len(set(y_test)) > 1 else 0
    }
    
    print(f"  {clf_name} results:")
    for metric_name, value in metrics.items():
        print(f"    {metric_name.upper():10}: {value:.4f}")
    
    # Return metrics and ROC data
    return metrics, y_test.tolist(), y_score.tolist()


def run_experiment(args, sequences, all_positive_pairs, tf_to_targets, labeled_negative_pairs=None):
    """
    Run the k-mer frequency + ML comparison experiment
    
    Workflow:
    1. Generate negative samples and build the dataset
    2. Split train/test first
    3. Build seq_pair = TF + Target
    4. Extract k-mer frequency features
    5. Train ML classifiers and evaluate
    
    Args:
        args: Command-line arguments
        sequences: Gene sequence dictionary
        all_positive_pairs: List of positive pairs
        tf_to_targets: Mapping from TF to target genes
        labeled_negative_pairs: Optional negative pairs loaded from TSV
        
    Returns:
        all_results: Results for all classifiers
    """
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    labeled_negative_pairs = labeled_negative_pairs or []
    if labeled_negative_pairs:
        print("\nUsing negative samples from TSV...")
        num_samples = min(len(all_positive_pairs), len(labeled_negative_pairs))
        positive_pairs = random.sample(all_positive_pairs, num_samples)
        negative_pairs = random.sample(labeled_negative_pairs, num_samples)
        sampling_strategy = 'labeled TSV negative samples + simple train/test split'
        print(f"  Balanced positives: {len(positive_pairs)}, negatives: {len(negative_pairs)}")
    else:
        gene_list = sorted(sequences.keys())
        random.shuffle(gene_list)

        # 1. Generate negative samples
        print("\nGenerating TF-based negative samples...")
        negative_pairs = generate_tf_based_negative_samples(
            all_positive_pairs, tf_to_targets, gene_list
        )
        positive_pairs = all_positive_pairs
        sampling_strategy = 'TF-based negative sampling + simple train/test split'
        print(f"  Positives: {len(positive_pairs)}, Negatives: {len(negative_pairs)}")

    # 2. Build a balanced dataset
    all_pairs = [(p[0], p[1], 1) for p in positive_pairs] + \
                [(p[0], p[1], 0) for p in negative_pairs]
    random.shuffle(all_pairs)

    # 3. Split train/test first
    test_ratio = 0.2
    all_labels = [p[2] for p in all_pairs]
    train_pairs, test_pairs, _, _ = train_test_split(
        all_pairs, all_labels, 
        test_size=test_ratio, 
        stratify=all_labels, 
        random_state=args.seed
    )
    print(f"\nDataset split:")
    print(f"  Train set: {len(train_pairs)}, Test set: {len(test_pairs)}")

    # 4. Build sequence pairs (TF_seq + Target_seq)
    print(f"\nBuilding sequence pairs (TF_seq + Target_seq)...")
    seq_pair_train_full, y_train_full = build_seq_pairs(train_pairs, sequences)
    seq_pair_test, y_test = build_seq_pairs(test_pairs, sequences)
    y_test = np.array(y_test)

    # 5. Split a validation set from the training set
    val_ratio = 0.2
    val_size = int(len(seq_pair_train_full) * val_ratio)
    indices = list(range(len(seq_pair_train_full)))
    random.shuffle(indices)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    
    # Actual training set (excluding validation set)
    seq_pair_train = [seq_pair_train_full[i] for i in train_indices]
    y_train = np.array([y_train_full[i] for i in train_indices])
    # Validation set (for XGBoost/LGBM early stopping)
    seq_pair_val = [seq_pair_train_full[i] for i in val_indices]
    y_val = np.array([y_train_full[i] for i in val_indices])
    
    print(f"  Train set (actual): {len(seq_pair_train)}, Validation set: {len(seq_pair_val)}")

    # 6. Build k-mer vocabulary
    kmer_to_idx, vocab_size = build_kmer_vocabulary(args.rank)

    # 7. Extract k-mer frequency features
    X_train = extract_features_batch(seq_pair_train, kmer_to_idx, vocab_size, args.rank, args.mode)
    X_val = extract_features_batch(seq_pair_val, kmer_to_idx, vocab_size, args.rank, args.mode)
    X_test = extract_features_batch(seq_pair_test, kmer_to_idx, vocab_size, args.rank, args.mode)

    # 8. Get classifiers
    classifiers = get_classifiers(seed=args.seed)

    # 9. Evaluate all classifiers (pass validation set for early stopping)
    all_results = {}
    roc_data = {}  # store data needed for ROC curves
    for clf_name, clf in classifiers.items():
        metrics, y_true, y_score = evaluate_classifier(clf, X_train, y_train, X_test, y_test, clf_name, X_val, y_val)
        all_results[clf_name] = metrics
        roc_data[clf_name] = {'y_true': y_true, 'y_score': y_score}

    return all_results, roc_data, sampling_strategy


def main():
    """Main function - k-mer frequency + machine learning classifiers"""
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Transcriptional regulatory network prediction - k-mer frequency + ML classifiers')
    parser.add_argument('--fasta', type=str, required=True, help='Path to FASTA file')
    parser.add_argument('--tsv', type=str, required=True, help='Path to TSV file')
    parser.add_argument('--rank', type=int, default=6, help='k-mer length (default: 6, i.e., 4096-dim features)')
    parser.add_argument('--mode', type=str, default='linear', choices=['linear', 'nonlinear'],
                        help='k-mer extraction mode: linear  or nonlinear  (default: linear)')
    parser.add_argument('--seed', type=int, default=88, help='Random seed (default: 88)')
    
    args = parser.parse_args()
    
    # Automatically extract species name from filename
    fasta_basename = os.path.splitext(os.path.basename(args.fasta))[0]
    species_name = fasta_basename.replace('.', '_').replace(' ', '_')

    # Print configuration
    print("=" * 70)
    print("Transcriptional Regulatory Network Prediction - k-mer frequency + machine learning classifiers")
    print(f"Features: {args.rank}-mer frequency vector (dim: {4**args.rank})")
    print("=" * 70)
    print(f"Species: {species_name}")
    print(f"k-mer length: {args.rank} (dim: {4**args.rank}), mode: {args.mode}")
    print(f"Classifiers: SVM, RandomForest, GBDT, MLP, XGBoost, LGBM")
    print(f"seed: {args.seed}")
    print("=" * 70)

    # 1. Load data
    sequences = load_sequences(args.fasta)
    all_positive_pairs, tf_to_targets, labeled_negative_pairs = load_positive_pairs(args.tsv, sequences)

    # 2. Run experiment
    all_results, roc_data, sampling_strategy = run_experiment(
        args, sequences, all_positive_pairs, tf_to_targets, labeled_negative_pairs
    )

    # 3. Output summary results
    print("\n" + "=" * 70)
    print("Summary results")
    print("=" * 70)
    print(f"{'Classifier':<20} {'AUROC':>8} {'AUPR':>8} {'F1':>8} {'Accuracy':>10}")
    print("-" * 70)
    for clf_name, metrics in all_results.items():
        print(f"{clf_name:<20} {metrics['auroc']:>8.4f} {metrics['aupr']:>8.4f} "
              f"{metrics['f1']:>8.4f} {metrics['accuracy']:>10.4f}")

    # 4. Save results
    output_file = f"{species_name}_kmer_ML.json"
    with open(output_file, 'w') as f:
        json.dump({
            'method': f'{args.rank}-mer frequency + ML classifiers',
            'feature_dim': 4**args.rank,
            'rank': args.rank,
            'mode': args.mode,
            'seed': args.seed,
            'sampling_strategy': sampling_strategy,
            'results': all_results
        }, f, indent=2)
    print(f"\nResults saved to: {output_file}")
    
    # 5. Save ROC data (for plotting ROC curves)
    roc_file = f"{species_name}_roc_data.json"
    with open(roc_file, 'w') as f:
        json.dump(roc_data, f)
    print(f"ROC data saved to: {roc_file}")


if __name__ == "__main__":
    main()
