# Copyright (C) 2005-2025, Bin-Guang Ma (mbg@mail.hzau.edu.cn); SPDX-License-Identifier: MIT
# The Numeric Dual Descriptor Vector class (Random AB matrix form) implemented with PyTorch
# This program is for the demonstration of methodology and not fully refined.
# Author: Bin-Guang Ma (assisted by DeepSeek); Date: 2025-8-28

import math
import random
import itertools
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class NumDualDescriptorRN(nn.Module):
    """
    Numeric Dual Descriptor with GPU acceleration using PyTorch:
      - Learnable coefficient matrix Acoeff ∈ R^{m×L}
      - Learnable basis matrix Bbasis ∈ R^{L×m}
      - Learnable transformation matrix M: R^m → R^m
      - Optimized with batch processing for GPU acceleration
      - Modified to handle m-dimensional vector sequences
    """
    def __init__(self, vec_dim=4, bas_dim=50, rank=1, rank_op='avg', rank_mode='drop', 
                 mode='linear', user_step=None, device='cuda'):
        """
        Initialize the Dual Descriptor for vector sequences.
        
        Args:
            vec_dim (int): Dimension m of input vectors
            bas_dim (int): Basis dimension L
            rank (int): Window size for vector aggregation
            rank_op (str): 'avg', 'sum', 'pick', or 'user_func'
            rank_mode (str): 'pad' or 'drop' for handling incomplete windows
            mode (str): 'linear' (sliding window) or 'nonlinear' (stepped window)
            user_step (int): Custom step size for nonlinear mode
            device (str): 'cuda' or 'cpu'
        """
        super().__init__()
        self.m = vec_dim
        self.L = bas_dim
        self.rank = rank
        self.rank_op = rank_op
        self.rank_mode = rank_mode
        assert mode in ('linear','nonlinear')
        self.mode = mode
        self.step = user_step
        self.trained = False
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Learnable transformation matrix M (m×m)
        self.M = nn.Parameter(torch.empty(self.m, self.m))
        
        # Coefficient matrix Acoeff: m×L
        self.Acoeff = nn.Parameter(torch.empty(self.m, self.L))
        
        # Basis matrix Bbasis: L×m
        self.Bbasis = nn.Parameter(torch.empty(self.L, self.m))
        
        # Initialize parameters
        self.reset_parameters()
        self.to(self.device)
        
    def reset_parameters(self):
        """Initialize model parameters"""
        nn.init.uniform_(self.M, -0.5, 0.5)
        nn.init.uniform_(self.Acoeff, -0.1, 0.1)
        nn.init.uniform_(self.Bbasis, -0.1, 0.1)

    def extract_vectors(self, vec_seq):
        """
        Extract vector groups from a vector sequence based on vectorization mode
        and return the aggregated vectors.
        
        - 'linear': Slide window by 1 step, extracting contiguous vectors of length = rank
        - 'nonlinear': Slide window by custom step (or rank length if step not specified)
        
        For nonlinear mode, handles incomplete trailing fragments using:
        - 'pad': Pads with zero vectors to maintain group length
        - 'drop': Discards incomplete fragments
        
        Args:
            vec_seq (list or tensor):  Input sequence of m-dimensional vectors
            
        Returns:
            list: List of vectors aggregated from extracted vector groups
        """
        # Convert to tensor if needed
        if not isinstance(vec_seq, torch.Tensor):
            vec_seq = torch.tensor(np.array(vec_seq), dtype=torch.float32, device=self.device)        
        
        L = vec_seq.shape[0]  # Sequence length

        # Helper function to apply vector operations
        def apply_op(vectors):
            """Apply rank operation to a list of vectors"""
            if self.rank_op == 'sum':
                return torch.sum(vectors, dim=0)
            elif self.rank_op == 'pick':
                idx = random.randint(0, vectors.size(0)-1)
                return vectors[idx]
            elif self.rank_op == 'user_func':
                # Default: average + sigmoid
                avg = torch.mean(vectors, dim=0)
                return torch.sigmoid(avg)
            else:  # 'avg' is default
                return torch.mean(vectors, dim=0)
        
        # Linear mode: sliding window with step=1
        if self.mode == 'linear':
            return [apply_op(vec_seq[i:i+self.rank]) for i in range(L - self.rank + 1)]            
        
        # Nonlinear mode: stepping with custom step size
        vectors = []
        step = self.step or self.rank  # Use custom step if defined, else use rank length
        
        for i in range(0, L, step):
            frag = vec_seq[i:i+self.rank]
            frag_len = frag.shape[0]
            
            # Pad or drop based on rank_mode setting
            if self.rank_mode == 'pad':
                # Pad fragment with zero vectors if shorter than rank
                if frag_len < self.rank:
                    padding = torch.zeros(self.rank - frag_len, self.m, device=self.device)
                    frag = torch.cat([frag, padding], dim=0)
                vectors.append(apply_op(frag)) 
            elif self.rank_mode == 'drop':
                # Only add fragments that match full rank length
                if frag_len == self.rank:
                    vectors.append(apply_op(frag)) 
        return vectors

    def batch_compute_Nk(self, k_tensor, vector_groups):
        """
        Vectorized computation of N(k) vectors for a batch of positions and vector groups
        Args:
            k_tensor: Tensor of position indices [batch_size]
            vector_groups: Tensor of vector groups [batch_size, m]
        Returns:
            Tensor of N(k) vectors [batch_size, m]
        """
        # Apply transformation matrix M to all vector groups [batch_size, m]
        x = torch.matmul(vector_groups, self.M)
        
        # Calculate basis indices j = k % L [batch_size]
        j_indices = (k_tensor % self.L).long()
        
        # Get Bbasis vectors: [batch_size, m]
        B_j = self.Bbasis[j_indices]
        
        # Compute scalar projection: B_j • x [batch_size]
        scalar = torch.sum(B_j * x, dim=1, keepdim=True)
        
        # Get Acoeff vectors: [batch_size, m]
        # Note: Acoeff is [m, L] -> permute to [L, m] then index with j_indices
        A_j = self.Acoeff.permute(1, 0)[j_indices]
        
        # Compute Nk = scalar * A_j [batch_size, m]
        Nk = scalar * A_j
            
        return Nk

    def describe(self, vec_seq):
        """Compute N(k) vectors for each vector group in sequence"""
        groups = self.extract_vectors(vec_seq)
        if not groups:
            return []
        
        vector_tensor = torch.stack(groups)
        k_positions = torch.arange(len(groups), dtype=torch.float32, device=self.device)
        
        # Batch compute all Nk vectors
        N_batch = self.batch_compute_Nk(k_positions, vector_tensor)
        return N_batch.detach().cpu().numpy()

    def S(self, vec_seq):
        """
        Compute list of S(l)=sum(N(k)) (k=1,...,l; l=1,...,L) for a given sequence.        
        """
        groups = self.extract_vectors(vec_seq)
        if not groups:
            return []
            
        vector_tensor = torch.stack(groups)
        k_positions = torch.arange(len(groups), dtype=torch.float32, device=self.device)
        
        # Batch compute all Nk vectors
        N_batch = self.batch_compute_Nk(k_positions, vector_tensor)
        
        # Compute cumulative sum (S vectors)
        S_cum = torch.cumsum(N_batch, dim=0)
        return [s.detach().cpu().numpy() for s in S_cum]

    def D(self, vec_seqs, t_list):
        """
        Compute mean squared deviation D across sequences:
        D = average over all positions of (N(k)-t_seq)^2
        """
        total_loss = 0.0
        total_positions = 0
        
        # Convert target vectors to tensor
        t_tensors = [torch.tensor(t, dtype=torch.float32, device=self.device) for t in t_list]
        
        for vec_seq, t in zip(vec_seqs, t_tensors):
            groups = self.extract_vectors(vec_seq)
            if not groups:
                continue
                
            vector_tensor = torch.stack(groups)
            k_positions = torch.arange(len(groups), dtype=torch.float32, device=self.device)
            
            # Batch compute all Nk vectors
            N_batch = self.batch_compute_Nk(k_positions, vector_tensor)
            
            # Compute loss for each position
            losses = torch.sum((N_batch - t) ** 2, dim=1)
            total_loss += losses.sum().item()
            total_positions += len(groups)
                
        return total_loss / total_positions if total_positions else 0.0

    def grad_train(self, vec_seqs, t_list, max_iters=1000, tol=1e-8, learning_rate=0.01, 
                   continued=False, decay_rate=1.0, print_every=10, batch_size=1024):
        """
        Train the model using gradient descent with batch processing
        GPU-accelerated with PyTorch.
        """
        if not continued:
            self.reset_parameters()
            
        # Precompute vector groups for all sequences
        all_groups = []
        for vec_seq in vec_seqs:
            groups = self.extract_vectors(vec_seq)
            all_groups.append(groups)
        
        # Convert target vectors to tensor
        t_tensors = [torch.tensor(t, dtype=torch.float32, device=self.device) for t in t_list]
        
        # Prepare all training samples (position, seq_idx)
        all_samples = []
        for seq_idx, groups in enumerate(all_groups):
            if not groups:
                continue
            for pos in range(len(groups)):
                # Store as tuple: (position, seq_idx)
                all_samples.append((pos, seq_idx))
        
        total_samples = len(all_samples)
        if total_samples == 0:
            print("Warning: No training samples found")
            return []
        
        # Set up optimizer
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)
        
        history = []
        prev_loss = float('inf')
        
        for it in range(max_iters):
            epoch_loss = 0.0
            # Shuffle samples
            random.shuffle(all_samples)
            
            # Process in batches
            for batch_start in range(0, total_samples, batch_size):
                batch_end = min(batch_start + batch_size, total_samples)
                batch_samples = all_samples[batch_start:batch_end]
                
                optimizer.zero_grad()
                
                # Prepare batch data directly as tensors
                k_list = []
                target_list = []
                vector_list = []
                
                for pos, seq_idx in batch_samples:
                    groups = all_groups[seq_idx]
                    k_list.append(pos)
                    vector_list.append(groups[pos])
                    target_list.append(t_tensors[seq_idx])
                
                # Create tensors directly on GPU
                k_tensor = torch.tensor(k_list, dtype=torch.float32, device=self.device)
                vector_tensor = torch.stack(vector_list)
                targets = torch.stack(target_list)
                
                # Compute Nk in batch
                Nk_batch = self.batch_compute_Nk(k_tensor, vector_tensor)
                
                # Compute loss
                loss = torch.mean(torch.sum((Nk_batch - targets) ** 2, dim=1))
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item() * len(batch_samples)
            
            # Average loss over epoch
            avg_loss = epoch_loss / total_samples
            history.append(avg_loss)
            scheduler.step()
            
            if it % print_every == 0 or it == max_iters - 1:
                print(f"GD Iter {it:2d}: D = {avg_loss:.6e}, LR = {scheduler.get_last_lr()[0]:.6f}")
            
            # Check convergence
            if abs(prev_loss - avg_loss) < tol:
                print(f"Converged after {it+1} iterations.")
                break
            prev_loss = avg_loss
        
        # Calculate statistics for reconstruction/generation
        total_group_count = 0
        total_t = torch.zeros(self.m, device=self.device)
        for vec_seq, t_vec in zip(vec_seqs, t_tensors):
            groups = self.extract_vectors(vec_seq)
            total_group_count += len(groups)
            total_t += t_vec * len(groups)
            
        self.mean_group_count = total_group_count / len(vec_seqs)
        self.mean_t = (total_t / total_group_count).detach().cpu().numpy()
        self.trained = True
        return history

    def auto_train(self, vec_seqs, auto_mode='gap', max_iters=1000, tol=1e-8, 
                   learning_rate=0.01, continued=False, decay_rate=1.0, 
                   print_every=10, batch_size=1024):
        """
        Train the model using self-supervised learning with two modes:
          - 'gap': Gap filling (mask current vector group and predict it from context)
          - 'reg': Auto-regressive (predict next vector group from previous groups)
        
        This method updates all learnable components with GPU acceleration.
        """
        if auto_mode not in ('gap', 'reg'):
            raise ValueError("auto_mode must be either 'gap' or 'reg'")

        if not continued:
            self.reset_parameters()
            
        # Precompute vector groups for all sequences
        all_sequences = []
        for vec_seq in vec_seqs:
            groups = self.extract_vectors(vec_seq)
            all_sequences.append(groups)
        
        # Prepare all training samples
        all_samples = []
        for seq_idx, groups in enumerate(all_sequences):
            if not groups:
                continue
                
            if auto_mode == 'gap':
                # Each group is a sample (position k)
                for k in range(len(groups)):
                    all_samples.append((k, seq_idx))
            else:  # 'reg' mode
                # Each group except last is a sample (position k, next group)
                for k in range(len(groups) - 1):
                    all_samples.append((k, seq_idx))
        
        if not all_samples:
            raise ValueError("No training samples found")
            
        # Set up optimizer
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)
        
        history = []
        prev_loss = float('inf')
        total_samples = len(all_samples)
        
        for it in range(max_iters):
            epoch_loss = 0.0
            # Shuffle samples
            random.shuffle(all_samples)
            
            # Process in batches
            for batch_start in range(0, total_samples, batch_size):
                batch_end = min(batch_start + batch_size, total_samples)
                batch_samples = all_samples[batch_start:batch_end]
                
                optimizer.zero_grad()
                
                # Prepare batch data directly as tensors
                k_list = []
                current_vector_list = []
                target_vector_list = []
                
                for sample in batch_samples:
                    k, seq_idx = sample
                    groups = all_sequences[seq_idx]
                    
                    if auto_mode == 'reg':
                        # Current group at position k, target is next group at k+1
                        current_vector_list.append(groups[k])
                        target_vector_list.append(groups[k+1])
                        k_list.append(k)
                    else:  # 'gap' mode
                        # Current group at position k is the target
                        current_vector_list.append(groups[k])
                        target_vector_list.append(groups[k])
                        k_list.append(k)
                
                # Create tensors directly on GPU
                k_tensor = torch.tensor(k_list, dtype=torch.float32, device=self.device)
                current_vector_tensor = torch.stack(current_vector_list)
                target_vector_tensor = torch.stack(target_vector_list)
                
                # Compute Nk for current vectors
                Nk_batch = self.batch_compute_Nk(k_tensor, current_vector_tensor)
                
                # Compute loss
                loss = torch.mean(torch.sum((Nk_batch - target_vector_tensor) ** 2, dim=1))
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item() * len(batch_samples)
            
            # Average loss over epoch
            avg_loss = epoch_loss / total_samples
            history.append(avg_loss)
            scheduler.step()
            
            if it % print_every == 0 or it == max_iters - 1:
                mode_display = "Gap" if auto_mode == 'gap' else "Reg"
                print(f"AutoTrain({mode_display}) Iter {it:3d}: loss = {avg_loss:.6f}, LR = {scheduler.get_last_lr()[0]:.6f}")
            
            # Check convergence
            if abs(prev_loss - avg_loss) < tol:
                print(f"Converged after {it+1} iterations")
                break
            prev_loss = avg_loss
        
        # Compute and store statistics for reconstruction/generation
        total_t = torch.zeros(self.m, device=self.device)
        total_group_count = 0
        for groups in all_sequences:
            group_tensor = torch.stack(groups)
            total_group_count += len(groups)
            total_t += group_tensor.sum(dim=0)
        
        self.mean_group_count = total_group_count / len(vec_seqs)
        self.mean_t = (total_t / total_group_count).detach().cpu().numpy()
        self.trained = True
        
        return history

    def predict_t(self, vec_seq):
        """
        Predict target vector for a sequence
        Returns the average of all N(k) vectors in the sequence
        """
        groups = self.extract_vectors(vec_seq)
        if not groups:
            return [0.0] * self.m
            
        vector_tensor = torch.stack(groups)
        k_positions = torch.arange(len(groups), dtype=torch.float32, device=self.device)
        
        # Batch compute all Nk vectors
        Nk_batch = self.batch_compute_Nk(k_positions, vector_tensor)
        Nk_sum = torch.sum(Nk_batch, dim=0)
        
        return (Nk_sum / len(groups)).detach().cpu().numpy()
    
    def reconstruct(self):
        """Reconstruct representative vector sequence by minimizing error"""
        assert self.trained, "Model must be trained first"
        n_groups = round(self.mean_group_count)
        mean_t_tensor = torch.tensor(self.mean_t, dtype=torch.float32, device=self.device)
        reconstructed_vectors = []
        
        # Generate random vectors as candidates
        candidate_vectors = torch.randn(1000, self.m, device=self.device)  # 1000 random vectors
        
        for k in range(n_groups):
            # Compute Nk for all candidate vectors at position k
            k_tensor = torch.tensor([k] * len(candidate_vectors), dtype=torch.float32, device=self.device)
            Nk_all = self.batch_compute_Nk(k_tensor, candidate_vectors)
            
            # Compute errors
            errors = torch.sum((Nk_all - mean_t_tensor) ** 2, dim=1)
            min_idx = torch.argmin(errors).item()
            best_vector = candidate_vectors[min_idx]
            
            reconstructed_vectors.append(best_vector.detach().cpu().numpy().tolist())
            
        return reconstructed_vectors

    def generate(self, n_groups, tau=0.0):
        """Generate sequence of n_groups vectors with temperature-controlled randomness"""
        assert self.trained, "Model must be trained first"
        if tau < 0:
            raise ValueError("Temperature must be non-negative")
            
        mean_t_tensor = torch.tensor(self.mean_t, dtype=torch.float32, device=self.device)
        generated_vectors = []
        
        # Generate random vectors as candidates
        candidate_vectors = torch.randn(1000, self.m, device=self.device)  # 1000 random vectors
        
        for k in range(n_groups):
            # Compute Nk for all candidate vectors at position k
            k_tensor = torch.tensor([k] * len(candidate_vectors), dtype=torch.float32, device=self.device)
            Nk_all = self.batch_compute_Nk(k_tensor, candidate_vectors)
            
            # Compute scores
            errors = torch.sum((Nk_all - mean_t_tensor) ** 2, dim=1)
            scores = -errors  # Convert to score (higher = better)
            
            if tau == 0:  # Deterministic selection
                max_idx = torch.argmax(scores).item()
                best_vector = candidate_vectors[max_idx]
                generated_vectors.append(best_vector.detach().cpu().numpy())
            else:  # Stochastic selection
                probs = torch.softmax(scores / tau, dim=0).detach().cpu().numpy()
                chosen_idx = random.choices(range(len(candidate_vectors)), weights=probs, k=1)[0]
                chosen_vector = candidate_vectors[chosen_idx]
                generated_vectors.append(chosen_vector.detach().cpu().numpy().tolist())
                
        return generated_vectors

    def save(self, filename):
        """Save model state to file"""
        save_dict = {
            'state_dict': self.state_dict(),
            'mean_t': self.mean_t if hasattr(self, 'mean_t') else None,
            'mean_group_count': self.mean_group_count if hasattr(self, 'mean_group_count') else None,
            'trained': self.trained
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
        self.mean_group_count = save_dict.get('mean_group_count', None)
        self.trained = save_dict.get('trained', False)
        print(f"Model loaded from {filename}")
        return self


# === Example Usage ===
if __name__ == "__main__":

    from statistics import correlation
    
    print("="*50)
    print("Dual Descriptor RN - Modified for Vector Sequences")
    print("Optimized with batch processing")
    print("="*50)
    
    # Set random seeds to ensure reproducibility
    torch.manual_seed(1)
    random.seed(1)
    np.random.seed(1)
    
    vec_dim = 3
    bas_dim = 250
    seq_num = 100

    # Generate sequences of m-dimensional vectors and random targets
    vec_seqs, t_list = [], []
    for _ in range(seq_num):
        # Generate a sequence of vectors
        L = random.randint(200, 300)
        vec_seq = [np.random.uniform(-1, 1, vec_dim) for _ in range(L)]
        vec_seqs.append(vec_seq)
        t_list.append(np.random.uniform(-1, 1, vec_dim))

    # === Gradient Descent Training Example ===
    print("\n" + "="*50)
    print("Testing Gradient Descent Training with Vector Sequences")
    print("="*50)

    # Create new model instance with GPU acceleration
    dd = NumDualDescriptorRN(
        vec_dim=vec_dim, 
        bas_dim=bas_dim, 
        rank=1,
        rank_op='avg',
        mode='linear', 
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Display device information
    print(f"\nUsing device: {dd.device}")
    print(f"Vector dimension: {dd.m}")

    # Train using gradient descent
    print("\nStarting gradient descent training...")
    grad_history = dd.grad_train(
        vec_seqs, 
        t_list,
        learning_rate=0.01,
        max_iters=100,
        tol=1e-8,
        decay_rate=0.99,
        print_every=5,
        batch_size=2048
    )

    # Predict target vector for first sequence
    aseq = vec_seqs[0]
    t_pred = dd.predict_t(aseq)
    print(f"\nPredicted t for first sequence: {[round(x.item(), 4) for x in t_pred]}")    
    
    # Calculate correlation between predicted and actual targets
    pred_t_list = [dd.predict_t(seq) for seq in vec_seqs]
    
    # Calculate correlations for each dimension
    corr_sum = 0.0
    for i in range(dd.m):
        actu_t = [t_vec[i] for t_vec in t_list]
        pred_t = [t_vec[i] for t_vec in pred_t_list]
        corr = correlation(actu_t, pred_t)
        print(f"Prediction correlation for dimension {i}: {corr:.4f}")
        corr_sum += corr
    corr_avg = corr_sum / dd.m
    print(f"Average correlation: {corr_avg:.4f}")

    # Reconstruct representative sequence
    repr_seq = dd.reconstruct()
    print(f"\nReconstructed representative sequence (first 5 vectors):")
    for i, vec in enumerate(repr_seq[:5]):
        print(f"Vector {i}: {[round(x, 4) for x in vec]}")
    
    # Generate sequences
    print("\nGenerated sequences:")
    seq_det = dd.generate(n_groups=5, tau=0.0)
    print(f"Deterministic (tau=0) generation:")
    for i, vec in enumerate(seq_det):
        print(f"Vector {i}: {[round(x.item(), 4) for x in vec]}")

    # === Auto-Training Example ===
    # Set random seed for reproducible results
    random.seed(1)
    np.random.seed(1)
    
    # Define vector dimension and model parameters
    vec_dim = 3   # Vector dimension
    bas_dim = 100 # Base matrix dimension
    seq_num = 30  # Number of sequences

    # Generate training sequences of vectors
    print("\n=== Generating Training Sequences ===")
    vec_seqs = []
    for i in range(seq_num):
        # Generate a sequence of vectors
        L = random.randint(50, 100)
        vec_seq = [np.random.uniform(-1, 1, vec_dim) for _ in range(L)]
        vec_seqs.append(vec_seq)
        print(f"Generated sequence {i+1}: length={len(vec_seq)}, first vector: {[round(x.item(), 4) for x in vec_seq[0]]}")
    
    print("=== Creating Dual Descriptor Model ===")
    dd_auto_gap = NumDualDescriptorRN(
        vec_dim=vec_dim,
        bas_dim=bas_dim,
        rank=1,
        mode='linear',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )    
    
    # Run self-supervised training (Gap Filling mode)
    print("\n=== Starting Gap Filling Training ===")
    gap_history = dd_auto_gap.auto_train(
        vec_seqs,
        auto_mode='gap',
        max_iters=100,
        learning_rate=0.001,
        decay_rate=0.995,
        print_every=10,
        batch_size=2048
    )

    print("=== Creating Dual Descriptor Model ===")
    dd_auto_reg = NumDualDescriptorRN(
        vec_dim=vec_dim,
        bas_dim=bas_dim,
        rank=1,
        mode='linear',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Run self-supervised training (Auto-Regressive mode)
    print("\n=== Starting Auto-Regressive Training ===")
    reg_history = dd_auto_reg.auto_train(
        vec_seqs,
        auto_mode='reg',
        max_iters=100,
        learning_rate=0.001,
        decay_rate=0.99,
        print_every=10,
        batch_size=2048
    )
    
    # Generate new vector sequences
    print("\n=== Generating New Vector Sequences ===")
    
    # Temperature=0 (deterministic generation)
    seq_det = dd_auto_gap.generate(n_groups=5, tau=0.0)
    print(f"Deterministic Generation (tau=0.0):")
    for i, vec in enumerate(seq_det):
        print(f"Vector {i}: {[round(x.item(), 4) for x in vec]}")
    
    # Temperature=0.5 (moderate randomness)
    seq_sto = dd_auto_reg.generate(n_groups=5, tau=0.5)
    print(f"\nStochastic Generation (tau=0.5):")
    for i, vec in enumerate(seq_sto):
        print(f"Vector {i}: {[round(x, 4) for x in vec]}")
    
    # Save and load model
    print("\n=== Model Persistence Test ===")
    dd_auto_reg.save("auto_trained_model_rn.pt")
    
    # Load model
    dd_loaded = NumDualDescriptorRN(
        vec_dim=vec_dim,
        bas_dim=bas_dim,
        rank=1,
        mode='linear',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    dd_loaded.load("auto_trained_model_rn.pt")
    print("Model loaded successfully. Generating with loaded model:")
    generated = dd_loaded.generate(n_groups=3, tau=0.0)
    for i, vec in enumerate(generated):
        print(f"Vector {i}: {[round(x.item(), 4) for x in vec]}")
    
    print("\n=== Auto-Training Demo Completed ===")
