# Copyright (C) 2005-2025, Bin-Guang Ma (mbg@mail.hzau.edu.cn); SPDX-License-Identifier: MIT
# The Numeric Dual Descriptor Vector class (AB matrix form) implemented with PyTorch
# This program is for the demonstration of methodology and not fully refined.
# Author: Bin-Guang Ma (assisted by DeepSeek); Date: 2025-8-28

import math
import itertools
import random
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class NumDualDescriptorAB(nn.Module):
    """
    Numeric Dual Descriptor with GPU acceleration using PyTorch:
      - learnable coefficient matrix Acoeff ∈ R^{m×L}
      - fixed basis matrix Bbasis ∈ R^{L×m}, Bbasis[k][i] = cos(2π*(k+1)/(i+1))
      - learnable mapping matrix M ∈ R^{m×m} for vector transformation
      - Supports both linear and nonlinear vectorization
      - Batch processing for GPU acceleration
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
        
        # Learnable mapping matrix M (m x m)
        self.M = nn.Parameter(torch.empty(self.m, self.m))
        
        # Coefficient matrix A (m x L)
        self.Acoeff = nn.Parameter(torch.empty(self.m, self.L))
        
        # Fixed basis matrix B (L x m)
        Bbasis = torch.empty(self.L, self.m)
        for k in range(self.L):
            for i in range(self.m):
                Bbasis[k, i] = math.cos(2 * math.pi * (k+1) / (i+1))
        self.register_buffer('Bbasis', Bbasis)

        # Initialize parameters
        self.reset_parameters()
        self.to(self.device)
    
    def reset_parameters(self):
        """Initialize model parameters"""        
        nn.init.uniform_(self.M, -0.5, 0.5)
        nn.init.uniform_(self.Acoeff, -0.1, 0.1)
    
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
            vec_seq (list or tensor): Input vector sequence to vectorize
            
        Returns:
            list: List of vectors aggregated from extracted vector groups
        """
        # Convert to tensor if needed
        if not isinstance(vec_seq, torch.Tensor):
            vec_seq = torch.tensor(vec_seq, dtype=torch.float32, device=self.device)
        
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

    def describe(self, vec_seq):
        """Compute N(k) vectors for each vector group in sequence"""
        avg_vectors = self.extract_vectors(vec_seq)
        if not avg_vectors:
            return torch.empty(0, self.m, device=self.device)
        
        # Stack average vectors
        avg_vectors_tensor = torch.stack(avg_vectors)  # [seq_len, m]
        
        # Apply mapping matrix M
        x = torch.mm(avg_vectors_tensor, self.M)  # [seq_len, m]
        
        k_positions = torch.arange(len(avg_vectors), dtype=torch.float32, device=self.device)
        
        # Compute basis indices (k mod L)
        j_indices = (k_positions % self.L).long()
        
        # Get corresponding B basis rows
        B_rows = self.Bbasis[j_indices]  # [seq_len, m]
        
        # Compute scalar = B[j] • x for each position
        scalar = torch.sum(B_rows * x, dim=1)  # [seq_len]
        
        # Get A columns for each position
        A_cols = self.Acoeff[:, j_indices].t()  # [seq_len, m]
        
        # Compute Nk = scalar * A_cols
        Nk = A_cols * scalar.unsqueeze(1)  # [seq_len, m]
        
        return Nk
    
    def S(self, vec_seq):
        """Compute cumulative sum of N(k) vectors"""
        Nk = self.describe(vec_seq)
        if Nk.shape[0] == 0:
            return torch.empty(0, self.m, device=self.device)
        
        # Compute cumulative sum
        S_cum = torch.cumsum(Nk, dim=0)
        return S_cum

    def grad_train(self, vec_seqs, t_list, max_iters=1000, tol=1e-8, learning_rate=0.01, 
                  continued=False, decay_rate=1.0, print_every=10, batch_size=1024):
        """
        Train model using gradient descent with batch processing
        GPU-accelerated with PyTorch
        """
        if not continued:
            self.reset_parameters()
            
        # Precompute average vectors for all sequences
        all_sequences = []
        for vec_seq in vec_seqs:
            avg_vectors = self.extract_vectors(vec_seq)
            all_sequences.append(avg_vectors)
        
        # Convert targets to tensor
        t_tensors = [torch.tensor(t, dtype=torch.float32, device=self.device) for t in t_list]
        
        # Prepare training samples (position, seq_idx)
        all_samples = []
        for seq_idx, avg_vectors in enumerate(all_sequences):
            for pos in range(len(avg_vectors)):
                all_samples.append((pos, seq_idx))
        
        total_samples = len(all_samples)
        if total_samples == 0:
            raise ValueError("No training samples found")
        
        # Set up optimizer
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)
        
        history = []
        prev_loss = float('inf')
        
        for it in range(max_iters):
            epoch_loss = 0.0
            random.shuffle(all_samples)
            
            # Process in batches
            for batch_start in range(0, total_samples, batch_size):
                batch_end = min(batch_start + batch_size, total_samples)
                batch_samples = all_samples[batch_start:batch_end]
                
                optimizer.zero_grad()
                
                # Prepare batch data
                k_list = []
                seq_idx_list = []
                avg_vectors_list = []
                
                for pos, seq_idx in batch_samples:
                    avg_vector = all_sequences[seq_idx][pos]
                    
                    k_list.append(pos)
                    seq_idx_list.append(seq_idx)
                    avg_vectors_list.append(avg_vector)
                
                k_tensor = torch.tensor(k_list, dtype=torch.float32, device=self.device)
                avg_vectors_tensor = torch.stack(avg_vectors_list)
                
                # Get corresponding targets
                targets = torch.stack([t_tensors[idx] for idx in seq_idx_list])
                
                # Compute Nk vectors
                x = torch.mm(avg_vectors_tensor, self.M)  # Apply mapping matrix M
                j_indices = (k_tensor % self.L).long()
                B_rows = self.Bbasis[j_indices]
                scalar = torch.sum(B_rows * x, dim=1)
                A_cols = self.Acoeff[:, j_indices].t()
                Nk = A_cols * scalar.unsqueeze(1)
                
                # Compute loss (MSE per position)
                loss = torch.mean(torch.sum((Nk - targets) ** 2, dim=1))
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item() * len(batch_samples)
            
            # Average loss and update scheduler
            avg_loss = epoch_loss / total_samples
            history.append(avg_loss)            
            
            # Print progress
            if it % print_every == 0 or it == max_iters - 1:
                print(f"GD Iter {it:3d}: D = {avg_loss:.6e}, LR = {scheduler.get_last_lr()[0]:.6f}")

            scheduler.step()
            
            # Check convergence
            if abs(prev_loss - avg_loss) < tol:
                print(f"Converged after {it+1} iterations")
                break
            prev_loss = avg_loss
        
        # Compute statistics for reconstruction
        total_positions = 0
        total_t = torch.zeros(self.m, device=self.device)
        for avg_vectors, t in zip(all_sequences, t_tensors):
            n_groups = len(avg_vectors)
            total_positions += n_groups
            total_t += t * n_groups
        
        self.mean_L = total_positions / len(vec_seqs)
        self.mean_t = (total_t / total_positions).detach().cpu().numpy()
        self.trained = True
        
        return history

    def predict_t(self, vec_seq):
        """Predict target vector as average of N(k) vectors"""
        Nk = self.describe(vec_seq)
        if Nk.shape[0] == 0:
            return np.zeros(self.m)
        return torch.mean(Nk, dim=0).detach().cpu().numpy()

    def auto_train(self, vec_seqs, auto_mode='gap', max_iters=1000, tol=1e-8, 
                   learning_rate=0.01, continued=False, decay_rate=1.0, 
                   print_every=10, batch_size=1024):
        """
        Self-supervised training with GPU acceleration
        Supports 'gap' and 'reg' modes
        """
        if not continued:
            self.reset_parameters()
            
        # Precompute average vectors for all sequences
        all_sequences = []
        for vec_seq in vec_seqs:
            avg_vectors = self.extract_vectors(vec_seq)
            all_sequences.append(avg_vectors)
        
        # Prepare training samples
        all_samples = []
        for seq_idx, avg_vectors in enumerate(all_sequences):
            n = len(avg_vectors)
            if n == 0:
                continue
                
            if auto_mode == 'gap':
                # Predict current vector group
                for pos in range(n):
                    all_samples.append((pos, seq_idx, pos))
            else:  # 'reg' mode
                # Predict next vector group
                for pos in range(n - 1):
                    all_samples.append((pos, seq_idx, pos+1))
        
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
            random.shuffle(all_samples)
            
            # Process in batches
            for batch_start in range(0, total_samples, batch_size):
                batch_end = min(batch_start + batch_size, total_samples)
                batch_samples = all_samples[batch_start:batch_end]
                
                optimizer.zero_grad()
                
                # Prepare batch data
                k_list = []
                input_idx_list = []
                target_idx_list = []
                input_vectors_list = []
                target_vectors_list = []
                
                for k, seq_idx, target_pos in batch_samples:
                    input_vector = all_sequences[seq_idx][k]
                    target_vector = all_sequences[seq_idx][target_pos]
                    
                    k_list.append(k)
                    input_idx_list.append(seq_idx)
                    target_idx_list.append(seq_idx)
                    input_vectors_list.append(input_vector)
                    target_vectors_list.append(target_vector)
                
                k_tensor = torch.tensor(k_list, dtype=torch.float32, device=self.device)
                input_vectors_tensor = torch.stack(input_vectors_list)
                target_vectors_tensor = torch.stack(target_vectors_list)
                
                # Compute Nk vectors
                x_input = torch.mm(input_vectors_tensor, self.M)  # Apply mapping matrix M
                j_indices = (k_tensor % self.L).long()
                B_rows = self.Bbasis[j_indices]
                scalar = torch.sum(B_rows * x_input, dim=1)
                A_cols = self.Acoeff[:, j_indices].t()
                Nk = A_cols * scalar.unsqueeze(1)

                # Transform target vectors using M matrix (?)
                target_vectors_tensor = torch.matmul(target_vectors_tensor, self.M.t())
                
                # Compute loss (MSE between Nk and target vector)
                loss = torch.mean(torch.sum((Nk - target_vectors_tensor) ** 2, dim=1))
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item() * len(batch_samples)
            
            # Average loss and update scheduler
            avg_loss = epoch_loss / total_samples
            history.append(avg_loss)            
            
            # Print progress
            mode_name = "Gap" if auto_mode == 'gap' else "Reg"
            if it % print_every == 0 or it == max_iters - 1:
                print(f"AutoTrain({mode_name}) Iter {it:3d}: loss = {avg_loss:.6e}, LR = {scheduler.get_last_lr()[0]:.6f}")

            scheduler.step()
            
            # Check convergence
            if abs(prev_loss - avg_loss) < tol:
                print(f"Converged after {it+1} iterations")
                break
            prev_loss = avg_loss
        
        # Compute statistics for reconstruction
        total_vec = torch.zeros(self.m, device=self.device)
        total_groups = 0
        for avg_vectors in all_sequences:
            for avg_vector in avg_vectors: 
                total_vec += avg_vector 
                total_groups += 1
        
        self.mean_L = total_groups / len(vec_seqs) if len(vec_seqs) > 0 else 0
        self.mean_t = (total_vec / total_groups).detach().cpu().numpy() if total_groups > 0 else np.zeros(self.m)
        self.trained = True
        
        return history

    def reconstruct(self):
        """Reconstruct representative vector sequence"""
        assert self.trained, "Model must be trained first"
        n_groups = round(self.mean_L)
        mean_t_tensor = torch.tensor(self.mean_t, dtype=torch.float32, device=self.device)
        seq_vectors = []
        
        for k in range(n_groups):
            # Compute j index for basis
            j = k % self.L
            B_row = self.Bbasis[j].unsqueeze(0)  # [1, m]
            
            # We need to find a vector v such that M*v minimizes the error
            # This is an optimization problem, we'll use gradient descent
            
            # Initialize a random vector
            v = nn.Parameter(torch.randn(self.m, device=self.device))
            
            # Set up optimizer for this vector
            optimizer = optim.Adam([v], lr=0.1)
            
            # Optimize for a few steps
            for opt_step in range(10):
                optimizer.zero_grad()
                
                # Apply mapping matrix M
                x = torch.mv(self.M, v)  # [m]
                
                # Compute scalar = B[j] • x
                scalar = torch.sum(B_row * x)
                
                # Compute Nk = scalar * A[:,j]
                A_col = self.Acoeff[:, j]  # [m]
                Nk = A_col * scalar
                
                # Compute error to mean target
                error = torch.sum((Nk - mean_t_tensor) ** 2)
                error.backward()
                optimizer.step()
            
            # Add the optimized vector to the sequence
            seq_vectors.append(v.detach().cpu().numpy())
        
        return np.array(seq_vectors)

    def generate(self, L, tau=0.0):
        """Generate vector sequence with temperature-controlled randomness"""
        assert self.trained, "Model must be trained first"
        if tau < 0:
            raise ValueError("Temperature must be non-negative")
            
        num_groups = (L + self.rank - 1) // self.rank
        mean_t_tensor = torch.tensor(self.mean_t, dtype=torch.float32, device=self.device)
        generated_vectors = []
        
        for k in range(num_groups):
            # Compute j index for basis
            j = k % self.L
            B_row = self.Bbasis[j].unsqueeze(0)  # [1, m]
            
            # We'll generate multiple candidate vectors and select the best one
            num_candidates = 1000 if tau > 0 else 1
            candidates = torch.randn(num_candidates, self.m, device=self.device)
            scores = torch.zeros(num_candidates, device=self.device)
            
            for i, candidate in enumerate(candidates):
                # Apply mapping matrix M
                x = torch.mv(self.M, candidate)  # [m]
                
                # Compute scalar = B[j] • x
                scalar = torch.sum(B_row * x)
                
                # Compute Nk = scalar * A[:,j]
                A_col = self.Acoeff[:, j]  # [m]
                Nk = A_col * scalar
                
                # Compute score (negative MSE)
                scores[i] = -torch.sum((Nk - mean_t_tensor) ** 2)
            
            # Select vector
            if tau == 0:  # Deterministic
                best_idx = torch.argmax(scores).item()
                best_vec = candidates[best_idx]
                generated_vectors.append(best_vec.cpu().numpy())
            else:  # Stochastic
                probs = torch.softmax(scores / tau, dim=0).detach().cpu().numpy()
                chosen_idx = np.random.choice(num_candidates, p=probs)
                chosen_vec = candidates[chosen_idx]
                generated_vectors.append(chosen_vec.cpu().numpy())
        
        # Trim to exact length
        full_seq = np.vstack(generated_vectors)
        return full_seq[:L]

    def save(self, filename):
        """Save model state to file"""
        save_dict = {
            'state_dict': self.state_dict(),
            'mean_t': self.mean_t if hasattr(self, 'mean_t') else None,
            'mean_L': self.mean_L if hasattr(self, 'mean_L') else None,
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
        self.mean_L = save_dict.get('mean_L', None)
        self.trained = save_dict.get('trained', False)
        print(f"Model loaded from {filename}")
        return self

# === Example Usage ===
if __name__ == "__main__":

    from statistics import correlation
    
    print("="*50)
    print("Dual Descriptor Vector AB - PyTorch GPU Accelerated Version")
    print("Modified for m-dimensional real vector sequences")
    print("="*50)
    
    # Set random seeds for reproducibility
    torch.manual_seed(1)
    random.seed(1)
    np.random.seed(1)
    
    vec_dim = 3
    bas_dim = 100
    seq_num = 10    
    
    # Generate vector sequences and random targets
    vec_seqs, t_list = [], []
    for _ in range(seq_num):
        # Generate random vector sequence
        L = random.randint(200, 300)
        vec_seq = np.random.randn(L, vec_dim)
        vec_seqs.append(vec_seq)
        # Generate random target vector
        t_list.append(np.random.uniform(-1, 1, vec_dim))
    
    # Create model
    dd = NumDualDescriptorAB(
        vec_dim=vec_dim, 
        bas_dim=bas_dim, 
        rank=1,
        rank_op='avg',
        rank_mode='drop',
        mode='linear',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Display device information
    print(f"\nUsing device: {dd.device}")
    print(f"Vector dimension: {dd.m}")
    
    # === Gradient Descent Training ===
    print("\n" + "="*50)
    print("Testing Gradient Descent Training")
    print("="*50)
    
    # Train using gradient descent
    print("\nStarting gradient descent training...")
    grad_history = dd.grad_train(
        vec_seqs, 
        t_list,
        learning_rate=0.1,
        max_iters=50,
        tol=1e-6,
        print_every=5,
        decay_rate=0.99,
        batch_size=1024
    )
    
    # Predict target for first sequence
    aseq = vec_seqs[0]
    t_pred = dd.predict_t(aseq)
    print(f"\nPredicted t for first sequence: {[round(x, 4) for x in t_pred]}")
    print(f"Actual t for first sequence: {[round(x, 4) for x in t_list[0]]}")
    
    # Calculate prediction correlation
    pred_t_list = [dd.predict_t(seq) for seq in vec_seqs]
    
    corr_sum = 0.0
    for i in range(dd.m):
        actu_t = [t_vec[i] for t_vec in t_list]
        pred_t = [t_vec[i] for t_vec in pred_t_list]
        try:
            corr = correlation(actu_t, pred_t)
            print(f"Prediction correlation for dimension {i}: {corr:.4f}")
            corr_sum += corr
        except:
            print(f"Cannot calculate correlation for dimension {i} (possibly constant values)")
    corr_avg = corr_sum / dd.m
    print(f"Average correlation: {corr_avg:.4f}")
    
    # Reconstruct representative sequence
    repr_seq = dd.reconstruct()
    print(f"\nReconstructed representative sequence shape: {repr_seq.shape}")
    print(f"First few vectors:\n{repr_seq[:3]}")
    
    # Generate sequences
    print("\nGenerated sequences:")
    seq_det = dd.generate(L=10, tau=0.0)
    print(f"Deterministic (tau=0) shape: {seq_det.shape}")
    print(f"First few vectors:\n{seq_det[:3]}")
    
    # === Auto-Training Example ===
    # Set random seeds
    torch.manual_seed(2)
    random.seed(2)
    np.random.seed(2)
    
    # Define parameters
    vec_dim = 3
    bas_dim = 100
    
    # Generate training sequences
    print("\n=== Generating Training Sequences ===")
    vec_seqs = []
    for i in range(30):
        L = random.randint(100, 200)
        vec_seq = np.random.randn(L, vec_dim)
        vec_seqs.append(vec_seq)
        print(f"Generated sequence {i+1}: length={L}, shape={vec_seq.shape}")
    
    # Create model for gap filling
    print("\n=== Creating Dual Descriptor Model for Gap Filling ===")
    dd_gap = NumDualDescriptorAB(
        vec_dim=vec_dim,
        bas_dim=bas_dim,
        rank=2,
        rank_mode='drop',
        mode='linear',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Train in gap filling mode
    print("\n=== Starting Gap Filling Training ===")
    gap_history = dd_gap.auto_train(
        vec_seqs,
        auto_mode='gap',
        max_iters=100,
        learning_rate=0.01,
        decay_rate=0.995,
        print_every=10,
        batch_size=1024
    )
    
    # Create model for auto-regressive training
    print("\n=== Creating Dual Descriptor Model for Auto-Regressive Training ===")
    dd_reg = NumDualDescriptorAB(
        vec_dim=vec_dim,
        bas_dim=bas_dim,
        rank=2,
        rank_mode='drop',
        mode='linear',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Train in auto-regressive mode
    print("\n=== Starting Auto-Regressive Training ===")
    reg_history = dd_reg.auto_train(
        vec_seqs,
        auto_mode='reg',
        max_iters=100,
        learning_rate=0.01,
        decay_rate=0.99,
        print_every=10,
        batch_size=1024
    )
    
    # Generate new sequences
    print("\n=== Generating New Sequences ===")
    
    # From gap model
    gap_seq = dd_gap.generate(L=10, tau=0.0)
    print(f"Gap model generation shape: {gap_seq.shape}")
    print(f"First few vectors:\n{gap_seq[:3]}")
    
    # From reg model
    reg_seq = dd_reg.generate(L=10, tau=0.0)
    print(f"Reg model generation shape: {reg_seq.shape}")
    print(f"First few vectors:\n{reg_seq[:3]}")
    
    # Save and load model
    print("\n=== Model Persistence Test ===")
    dd_reg.save("auto_trained_vector_model.pkl")
    
    # Load model
    dd_loaded = NumDualDescriptorAB(
        vec_dim=vec_dim,
        bas_dim=bas_dim,
        rank=2,
        rank_mode='drop',
        mode='linear',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    ).load("auto_trained_vector_model.pkl")
    
    print("Model loaded successfully. Generating with loaded model:")
    generated = dd_loaded.generate(L=5, tau=0.0)
    print(f"Generated shape: {generated.shape}")
    print(f"Vectors:\n{generated}")
    
    print("\n=== All Tests Completed ===")