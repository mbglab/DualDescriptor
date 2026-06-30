#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Transcriptional regulatory network prediction

import os
import json
import copy
import random
import argparse
import traceback
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from Bio import SeqIO
from DDvTS import DualDescriptorTS
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split


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
        sequences[gene_id] = str(record.seq)
    print(f"  Loaded {len(sequences)} gene sequences")
    return sequences


def load_positive_pairs(tsv_file, sequences):
    """
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


def create_model(rank, rank_mode, vec_dim, num_basis, mode):
    """
    Create the DualDescriptorTS model
    
    Args:
        rank: Rank parameter
        rank_mode: Rank mode ('drop' or other)
        vec_dim: Vector dimension
        num_basis: Number of basis functions
        mode: Model mode ('linear' or other)
        
    Returns:
        dd_model: Initialized model
    """
    charset = ['A', 'C', 'G', 'T']
    dd_model = DualDescriptorTS(
        charset, 
        rank=rank, 
        rank_mode=rank_mode, 
        vec_dim=vec_dim,
        num_basis=num_basis, 
        mode=mode,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    dd_model.classifier = nn.Linear(dd_model.m, 2).to(dd_model.device)
    dd_model.num_classes = 2
    dd_model.reset_parameters()
    return dd_model


def encode_sequence(dd_model, seq):
    """
    Encode a nucleotide sequence into a fixed-length feature vector.
    
    Args:
        dd_model: Model
        seq: Nucleotide sequence string
        
    Returns:
        seq_vector: Feature vector
    """
    tokens = dd_model.extract_tokens(seq)
    if not tokens:
        return torch.zeros(dd_model.m, device=dd_model.device)
    token_indices = dd_model.token_to_indices(tokens)
    k_positions = torch.arange(len(tokens), dtype=torch.float32, device=dd_model.device)
    Nk_batch = dd_model.batch_compute_Nk(k_positions, token_indices)
    return torch.mean(Nk_batch, dim=0)


def train_model(dd_model, train_seqs, train_labels, val_seqs, val_labels, 
                epochs, lr, batch_size, patience, gamma=0.99, print_every=10):
    """
    Train the model (with early stopping)
    
    Args:
        dd_model: Model
        train_seqs: Training sequence list
        train_labels: Training labels list
        val_seqs: Validation sequence list
        val_labels: Validation labels list
        epochs: Maximum number of epochs
        lr: Learning rate
        batch_size: Batch size
        patience: Early-stopping patience
        gamma: Learning rate decay factor
        print_every: Print progress every N epochs
        
    Returns:
        dd_model: Trained model
    """
    optimizer = optim.Adam(dd_model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    criterion = nn.CrossEntropyLoss()
    
    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    label_tensors = torch.tensor(train_labels, dtype=torch.long, device=dd_model.device)

    for epoch in range(epochs):
        dd_model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        indices = list(range(len(train_seqs)))
        random.shuffle(indices)

        for batch_start in range(0, len(indices), batch_size):
            batch_indices = indices[batch_start:batch_start + batch_size]
            batch_seqs = [train_seqs[idx] for idx in batch_indices]
            batch_labels = label_tensors[batch_indices]

            optimizer.zero_grad()
            batch_logits = []
            for seq in batch_seqs:
                seq_vector = encode_sequence(dd_model, seq)
                batch_logits.append(dd_model.classifier(seq_vector.unsqueeze(0)))

            if batch_logits:
                all_logits = torch.cat(batch_logits, dim=0)
                loss = criterion(all_logits, batch_labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item() * len(batch_seqs)
                total_samples += len(batch_seqs)
                with torch.no_grad():
                    predictions = torch.argmax(all_logits, dim=1)
                    total_correct += (predictions == batch_labels).sum().item()

        # Compute training metrics
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        train_acc = total_correct / total_samples if total_samples > 0 else 0.0
        
        # Validation
        val_loss, val_acc = validate_model(dd_model, val_seqs, val_labels, criterion)
        
        # Print progress
        if epoch % print_every == 0 or epoch == epochs - 1:
            current_lr = scheduler.get_last_lr()[0]
            print(f"  Epoch {epoch:3d}: Train Loss={avg_loss:.4f}, Train Acc={train_acc:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}, LR={current_lr:.6f}")

        # Early stopping
        if val_loss < best_val_loss - 0.001:
            best_val_loss = val_loss
            best_state = copy.deepcopy(dd_model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch + 1}, best validation loss: {best_val_loss:.4f}")
                break
        scheduler.step()

    if best_state:
        dd_model.load_state_dict(best_state)
    
    return dd_model


def validate_model(dd_model, val_seqs, val_labels, criterion):
    """
    Evaluate the model on the validation set
    
    Args:
        dd_model: Model
        val_seqs: Validation sequence list
        val_labels: Validation labels list
        criterion: Loss function
        
    Returns:
        val_loss: Validation loss
        val_acc: Validation accuracy
    """
    dd_model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for seq, label in zip(val_seqs, val_labels):
            seq_vector = encode_sequence(dd_model, seq)
            logits = dd_model.classifier(seq_vector.unsqueeze(0))
            val_loss += criterion(logits, torch.tensor([label], device=dd_model.device)).item()
            pred = torch.argmax(logits, dim=1).item()
            if pred == label:
                correct += 1
            total += 1
    avg_loss = val_loss / total if total > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0
    return avg_loss, accuracy


def evaluate_model(dd_model, test_pairs, sequences):
    """
    Evaluate model performance
    
    Args:
        dd_model: Model
        test_pairs: Test data pairs [(reg, tgt, label), ...]
        sequences: Gene sequence dictionary
        
    Returns:
        metrics: Evaluation metrics dictionary
    """
    dd_model.eval()
    predictions, probabilities, labels = [], [], []
    
    with torch.no_grad():
        for reg, tgt, label in test_pairs:
            seq = sequences[reg] + sequences[tgt]
            seq_vector = encode_sequence(dd_model, seq)
            logits = dd_model.classifier(seq_vector.unsqueeze(0))
            probs = torch.softmax(logits, dim=1)
            predictions.append(torch.argmax(probs, dim=1).item())
            probabilities.append(probs[0, 1].item())
            labels.append(label)

    metrics = {
        'accuracy': accuracy_score(labels, predictions),
        'precision': precision_score(labels, predictions, zero_division=0),
        'recall': recall_score(labels, predictions, zero_division=0),
        'f1': f1_score(labels, predictions, zero_division=0),
        'auroc': roc_auc_score(labels, probabilities) if len(set(labels)) > 1 else 0,
        'aupr': average_precision_score(labels, probabilities) if len(set(labels)) > 1 else 0
    }
    # Return metrics and ROC data
    return metrics, labels, probabilities


def run_training(args, sequences, all_positive_pairs, tf_to_targets, labeled_negative_pairs=None):
    """
    Run training and evaluation
    
    Args:
        args: Command-line arguments
        sequences: Gene sequence dictionary
        all_positive_pairs: List of positive pairs
        tf_to_targets: Mapping from TF to target genes
        labeled_negative_pairs: Optional negative pairs loaded from TSV
        
    Returns:
        metrics: Evaluation metrics
    """

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
  
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

        # Generate negative samples
        print("\nGenerating TF-based negative samples...")
        negative_pairs = generate_tf_based_negative_samples(
            all_positive_pairs, tf_to_targets, gene_list
        )
        positive_pairs = all_positive_pairs
        sampling_strategy = 'TF-based negative sampling + simple train/test split'
        print(f"  Positives: {len(positive_pairs)}, Negatives: {len(negative_pairs)}")

    # Build a balanced dataset
    all_pairs = [(p[0], p[1], 1) for p in positive_pairs] + \
                [(p[0], p[1], 0) for p in negative_pairs]
    random.shuffle(all_pairs)

    # Simple train/test split
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

    try:
        # Create model
        dd_model = create_model(args.rank, args.rank_mode, args.vec_dim, args.num_basis, args.mode)
        
        val_ratio = 0.2
        epochs = 300
        lr = 0.00015
        batch_size = 32
        patience = 10
        gamma = 0.99

        # Prepare training data
        train_seqs = [sequences[p[0]] + sequences[p[1]] for p in train_pairs]
        train_labels = [p[2] for p in train_pairs]
        
        # Randomly split validation set
        val_size = int(len(train_seqs) * val_ratio)
        indices = list(range(len(train_seqs)))
        random.shuffle(indices)
        val_indices = indices[:val_size]
        train_indices = indices[val_size:]
        val_seqs = [train_seqs[i] for i in val_indices]
        val_labels = [train_labels[i] for i in val_indices]
        train_seqs = [train_seqs[i] for i in train_indices]
        train_labels = [train_labels[i] for i in train_indices]
        print(f"  Train set (actual): {len(train_seqs)}, Validation set: {len(val_seqs)}")

        # Train model
        print("\nStarting training...")
        dd_model = train_model(
            dd_model, train_seqs, train_labels, val_seqs, val_labels,
            epochs, lr, batch_size, patience, gamma
        )

        # Evaluate model
        print("\nEvaluating model...")
        metrics, y_true, y_score = evaluate_model(dd_model, test_pairs, sequences)
        
        return dd_model, metrics, y_true, y_score, sampling_strategy

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return None, None, None, None, None


def main():

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Transcriptional regulatory network prediction')
    parser.add_argument('--fasta', type=str, required=True, help='Path to FASTA file')
    parser.add_argument('--tsv', type=str, required=True, help='Path to TSV file')
    parser.add_argument('--rank', type=int, default=6, help='Rank parameter (default: 6)')
    parser.add_argument('--rank-mode', type=str, default='drop', help='Rank mode (default: drop)')
    parser.add_argument('--vec-dim', type=int, default=10, help='Vector dimension (default: 10)')
    parser.add_argument('--num-basis', type=int, default=10, help='Number of basis functions (default: 10)')
    parser.add_argument('--mode', type=str, default='linear', help='Model mode (default: linear)')
    parser.add_argument('--seed', type=int, default=88, help='Random seed (default: 88)')
    
    args = parser.parse_args()
    
    # Automatically extract species name from filename
    fasta_basename = os.path.splitext(os.path.basename(args.fasta))[0]
    species_name = fasta_basename.replace('.', '_').replace(' ', '_')

    # Print configuration
    print("=" * 70)
    print(" Transcriptional regulatory network prediction ")
    print("=" * 70)
    print(f"Species: {species_name}")
    print(f"Model config: rank={args.rank}, rank_mode={args.rank_mode}, vec_dim={args.vec_dim}, num_basis={args.num_basis}, mode={args.mode}")
    print(f"seed: {args.seed}")
    print("=" * 70)

    # 1. Load data
    sequences = load_sequences(args.fasta)
    all_positive_pairs, tf_to_targets, labeled_negative_pairs = load_positive_pairs(args.tsv, sequences)

    # 2. Run training and evaluation
    dd_model, metrics, y_true, y_score, sampling_strategy = run_training(
        args, sequences, all_positive_pairs, tf_to_targets, labeled_negative_pairs
    )

    # 3. Output results
    if metrics:
        print("\n" + "=" * 70)
        print("Final results")
        print("=" * 70)
        for metric_name, value in metrics.items():
            print(f"  {metric_name.upper():10}: {value:.4f}")
        
        # Save results
        output_file = f"{species_name}_TFbased_simple.json"
        with open(output_file, 'w') as f:
            json.dump({
                'strategy': sampling_strategy,
                'model_config': {
                    'rank': args.rank,
                    'rank_mode': args.rank_mode,
                    'vec_dim': args.vec_dim,
                    'num_basis': args.num_basis,
                    'mode': args.mode
                },
                'seed': args.seed,
                'metrics': metrics
            }, f, indent=2)
        print(f"\nResults saved to: {output_file}")
        
        # Save model
        model_file = f"{species_name}_trained_model.pth"
        torch.save(dd_model.state_dict(), model_file)
        print(f"Model saved to: {model_file}")
    else:
        print("Training failed!")


if __name__ == "__main__":
    main()
