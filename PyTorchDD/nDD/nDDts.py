# Copyright (C) 2005-2025, Bin-Guang Ma (mbg@mail.hzau.edu.cn); SPDX-License-Identifier: MIT
# The Numeric Dual Descriptor Vector class (Tensor form) implemented with PyTorch
# This program is for the demonstration of methodology and not fully refined.
# Author: Bin-Guang Ma (assisted by DeepSeek); Date: 2025-8-28

import math
import random
import itertools
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from statistics import correlation

class DualDescriptorVectorTS(nn.Module):
    """
    Vector Dual Descriptor with GPU acceleration using PyTorch:
      - tensor P ∈ R^{m×m×o} of basis coefficients
      - mapping matrix M ∈ R^{m×m} for vector transformation
      - indexed periods: period[i,j,g] = i*(m*o) + j*o + g + 2
      - basis function phi_{i,j,g}(k) = cos(2π * k / period[i,j,g])
      - supports 'linear' or 'nonlinear' (step-by-rank) vector extraction
      - Added Layer Normalization for stable training
    """
    def __init__(self, vec_dim=2, rank=1, rank_op='avg', rank_mode='drop', num_basis=5, mode='linear', user_step=None, device='cuda', use_norm=True):
        super().__init__()
        self.m = vec_dim    # vector dimension
        self.o = num_basis  # number of basis terms        
        self.rank = rank    # r-per/k-mer length
        self.rank_op = rank_op  # 'avg', 'sum', 'pick', 'user_func'
        self.rank_mode = rank_mode # 'pad' or 'drop'         
        assert mode in ('linear','nonlinear')
        self.mode = mode
        self.step = user_step
        self.trained = False
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.use_norm = use_norm  # whether to use layer normalization
        
        # Mapping matrix M (m×m) for vector transformation
        self.M = nn.Parameter(torch.empty(self.m, self.m))
        
        # Position-weight tensor P[i][j][g]
        self.P = nn.Parameter(torch.empty(self.m, self.m, self.o))
        
        # Precompute indexed periods[i][j][g] (fixed, not trainable)
        periods = torch.zeros(self.m, self.m, self.o, dtype=torch.float32)
        for i in range(self.m):
            for j in range(self.m):
                for g in range(self.o):
                    periods[i, j, g] = i*(self.m*self.o) + j*self.o + g + 2
        self.register_buffer('periods', periods)

        # Add Layer Normalization for stable training
        if self.use_norm:
            self.norm = nn.LayerNorm(self.m)
        
        # Initialize parameters
        self.reset_parameters()
        self.to(self.device)
        
    def reset_parameters(self):
        """Initialize model parameters"""
        nn.init.uniform_(self.M, -0.5, 0.5)
        nn.init.uniform_(self.P, -0.1, 0.1)
        if self.use_norm:
            nn.init.constant_(self.norm.weight, 1.0)
            nn.init.constant_(self.norm.bias, 0.0)  
    
    def extract_vectors(self, seq_vectors):
        """
        Extract window vectors from sequence based on processing mode and rank operation.
        
        - 'linear': Slide window by 1 step, extracting contiguous vectors of length = rank
        - 'nonlinear': Slide window by custom step (or rank length if step not specified)
        
        For nonlinear mode, handles incomplete trailing fragments using:
        - 'pad': Pads with zero vectors to maintain group length
        - 'drop': Discards incomplete fragments
        
        Args:
            seq_vectors (list or tensor): Input vector sequence
            
        Returns:
            list: List of vectors after applying rank operation to each extracted vector group
        """
        L = len(seq_vectors)
        # Convert to tensor if needed
        if not isinstance(seq_vectors, torch.Tensor):
            seq_vectors = torch.tensor(seq_vectors, dtype=torch.float32, device=self.device)        
        
        # Linear mode: sliding window with step=1
        if self.mode == 'linear':
            vector_groups = [seq_vectors[i:i+self.rank] for i in range(L - self.rank + 1)]
        
        # Nonlinear mode: stepping with custom step size
        else:
            vector_groups = []
            step = self.step or self.rank  # Use custom step if defined, else use rank length
            
            for i in range(0, L, step):
                frag = seq_vectors[i:i+self.rank]
                frag_len = len(frag)
                
                # Pad or drop based on rank_mode setting
                if self.rank_mode == 'pad' and frag_len < self.rank:
                    # Pad fragment with zero vectors if shorter than rank
                    padding = torch.zeros(self.rank - frag_len, self.m, device=self.device)
                    frag = torch.cat([frag, padding], dim=0)
                    vector_groups.append(frag)
                elif frag_len == self.rank:
                    # Only add fragments that match full rank length
                    vector_groups.append(frag)

        def apply_op(vec_tensor):
            """Apply rank operation (avg/sum/pick/user_func) to a list of vectors"""
            
            if self.rank_op == 'sum':
                return torch.sum(vec_tensor, dim=0)
                
            elif self.rank_op == 'pick':
                idx = random.randint(0, len(vec_tensor)-1)
                return vec_tensor[idx]
                
            elif self.rank_op == 'user_func':
                # Use custom function if provided, else default behavior
                if hasattr(self, 'user_func') and callable(self.user_func):
                    return self.user_func(vec_tensor)
                else:
                    # Default: average + sigmoid
                    avg = torch.mean(vec_tensor, dim=0)
                    return torch.sigmoid(avg)
                    
            else:  # 'avg' is default
                return torch.mean(vec_tensor, dim=0)
        
        # Apply rank operation for each group
        vectors = [apply_op(group) for group in vector_groups]
        return vectors    

    def batch_compute_Nk(self, k_tensor, vectors):
        """
        Vectorized computation of N(k) vectors for a batch of positions and vectors
        Optimized using einsum for better performance
        Args:
            k_tensor: Tensor of position indices [batch_size]
            vectors: Tensor of vectors [batch_size, m]
        Returns:
            Tensor of N(k) vectors [batch_size, m]
        """
        # Transform vectors using matrix M [batch_size, m]
        x = torch.matmul(vectors, self.M.t())
        
        # Expand dimensions for broadcasting [batch_size, 1, 1, 1]
        k_expanded = k_tensor.view(-1, 1, 1, 1)
        
        # Calculate basis functions: cos(2π*k/periods) [batch_size, m, m, o]
        phi = torch.cos(2 * math.pi * k_expanded / self.periods)
        
        # Optimized computation using einsum
        Nk = torch.einsum('bj,ijg,bijg->bi', x, self.P, phi)
        
        # Apply Layer Normalization if enabled
        if self.use_norm:
            Nk = self.norm(Nk)
            
        return Nk

    def compute_Nk(self, k, vector):
        """Compute N(k) for single position and vector (uses batch internally)"""
        # Convert to tensors
        k_tensor = torch.tensor([k], dtype=torch.float32, device=self.device)
        vector_tensor = torch.tensor([vector], dtype=torch.float32, device=self.device)
        
        # Use batch computation
        result = self.batch_compute_Nk(k_tensor, vector_tensor)
        return result[0]  # Return first element

    def describe(self, seq_vectors):
        """Compute N(k) vectors for each vector group in sequence"""
        vector_groups = self.extract_vectors(seq_vectors)
        if not vector_groups:
            return []
        
        vectors_tensor = torch.stack(vector_groups)
        k_positions = torch.arange(len(vector_groups), dtype=torch.float32, device=self.device)
        
        # Batch compute all Nk vectors
        N_batch = self.batch_compute_Nk(k_positions, vectors_tensor)
        return N_batch.detach().cpu().numpy()

    def S(self, seq_vectors):
        """
        Compute list of S(l)=sum(N(k)) (k=1,...,l; l=1,...,L) for a given vector sequence.        
        """
        vector_groups = self.extract_vectors(seq_vectors)
        if not vector_groups:
            return []
            
        vectors_tensor = torch.stack(vector_groups)
        k_positions = torch.arange(len(vector_groups), dtype=torch.float32, device=self.device)
        
        # Batch compute all Nk vectors
        N_batch = self.batch_compute_Nk(k_positions, vectors_tensor)
        
        # Compute cumulative sum (S vectors)
        S_cum = torch.cumsum(N_batch, dim=0)
        return [s.detach().cpu().numpy() for s in S_cum]

    def D(self, seqs_vectors, t_list):
        """
        Compute mean squared deviation D across vector sequences:
        D = average over all positions of (N(k)-t_seq)^2
        """
        total_loss = 0.0
        total_positions = 0
        
        # Convert target vectors to tensor
        t_tensors = [torch.tensor(t, dtype=torch.float32, device=self.device) for t in t_list]
        
        for seq_vectors, t in zip(seqs_vectors, t_tensors):
            vector_groups = self.extract_vectors(seq_vectors)
            if not vector_groups:
                continue
                
            vectors_tensor = torch.stack(vector_groups)
            k_positions = torch.arange(len(vector_groups), dtype=torch.float32, device=self.device)
            
            # Batch compute all Nk vectors
            N_batch = self.batch_compute_Nk(k_positions, vectors_tensor)
            
            # Compute loss for each position
            losses = torch.sum((N_batch - t) ** 2, dim=1)
            total_loss += losses.sum().item()
            total_positions += len(vector_groups)
                
        return total_loss / total_positions if total_positions else 0.0

    def grad_train(self, seqs_vectors, t_list, max_iters=1000, tol=1e-8, learning_rate=0.01, continued=False, decay_rate=1.0, print_every=10, batch_size=1024):
        """
        Train the model using gradient descent with batch processing
        GPU-accelerated with PyTorch.
        """
        if not continued:
            self.reset_parameters()
            
        # Precompute vector groups for all sequences
        all_vector_groups = []
        for seq_vectors in seqs_vectors:
            groups = self.extract_vectors(seq_vectors)
            all_vector_groups.append(groups)
        
        # Convert target vectors to tensor
        t_tensors = [torch.tensor(t, dtype=torch.float32, device=self.device) for t in t_list]
        
        # Prepare all training samples (position, vector, target)
        all_samples = []
        for seq_idx, groups in enumerate(all_vector_groups):
            if not groups:
                continue
            for pos, vector in enumerate(groups):
                # Store as tuple: (position, seq_idx, vector)
                all_samples.append((pos, seq_idx, vector))
        
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
                vector_list = []
                target_list = []
                
                for pos, seq_idx, vector in batch_samples:
                    k_list.append(pos)
                    vector_list.append(vector)
                    target_list.append(t_tensors[seq_idx])
                
                # Create tensors directly on GPU
                k_tensor = torch.tensor(k_list, dtype=torch.float32, device=self.device)
                vectors_tensor = torch.stack(vector_list).to(self.device)
                targets = torch.stack(target_list)
                
                # Compute Nk in batch
                Nk_batch = self.batch_compute_Nk(k_tensor, vectors_tensor)
                
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
        for groups, t_vec in zip(all_vector_groups, t_tensors):
            total_group_count += len(groups)
            total_t += t_vec * len(groups)
            
        self.mean_group_count = total_group_count / len(seqs_vectors)
        self.mean_t = (total_t / total_group_count).detach().cpu().numpy()
        self.trained = True
        return history

    def auto_train(self, seqs_vectors, max_iters=100, tol=1e-6, learning_rate=0.01, continued=False, auto_mode='reg', decay_rate=1.0, print_every=10, batch_size=1024):
        """
        Self-training method using gradient descent with batch processing
        GPU-accelerated with PyTorch.
        """
        if auto_mode not in ('gap', 'reg'):
            raise ValueError("auto_mode must be either 'gap' or 'reg'")

        if not continued:
            self.reset_parameters()
            
        # Precompute vector groups for all sequences
        all_sequences = []
        for seq_vectors in seqs_vectors:
            groups = self.extract_vectors(seq_vectors)
            all_sequences.append(groups)
        
        # Prepare all training samples
        all_samples = []
        for seq_idx, groups in enumerate(all_sequences):
            if not groups:
                continue
                
            if auto_mode == 'gap':
                # Each vector group is a sample (position k, vector)
                for k, vector in enumerate(groups):
                    all_samples.append((k, vector))
            else:  # 'reg' mode
                # Each vector group except last is a sample (position k, current vector, next vector)
                for k in range(len(groups) - 1):
                    all_samples.append((k, groups[k], groups[k+1]))
        
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
                current_vectors_list = []
                target_vectors_list = [] if auto_mode == 'reg' else None
                
                for sample in batch_samples:
                    if auto_mode == 'reg':
                        k, current_vec, next_vec = sample
                        k_list.append(k)
                        current_vectors_list.append(current_vec)
                        target_vectors_list.append(next_vec)
                    else:  # 'gap' mode
                        k, vector = sample
                        k_list.append(k)
                        current_vectors_list.append(vector)
                
                # Create tensors directly on GPU
                k_tensor = torch.tensor(k_list, dtype=torch.float32, device=self.device)
                current_vectors_tensor = torch.stack(current_vectors_list).to(self.device)
                
                # Compute Nk for current vectors
                Nk_batch = self.batch_compute_Nk(k_tensor, current_vectors_tensor)
                
                # Get target vectors
                if auto_mode == 'gap':
                    targets = torch.mm(current_vectors_tensor, self.M) #current_vectors_tensor
                else:  # 'reg' mode
                    target_vectors_tensor = torch.stack(target_vectors_list).to(self.device)
                    targets = torch.mm(target_vectors_tensor,  self.M) #target_vectors_tensor
                
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
            vectors_tensor = torch.stack(groups).to(self.device) if groups else torch.empty(0, self.m, device=self.device)
            total_group_count += len(groups)
            total_t += vectors_tensor.sum(dim=0)
        
        self.mean_group_count = total_group_count / len(seqs_vectors)
        self.mean_t = (total_t / total_group_count).detach().cpu().numpy()
        self.trained = True
        
        return history

    def predict_t(self, seq_vectors):
        """
        Predict target vector for a vector sequence
        Returns the average of all N(k) vectors in the sequence
        """
        vector_groups = self.extract_vectors(seq_vectors)
        if not vector_groups:
            return [0.0] * self.m
            
        vectors_tensor = torch.stack(vector_groups)
        k_positions = torch.arange(len(vector_groups), dtype=torch.float32, device=self.device)
        
        # Batch compute all Nk vectors
        Nk_batch = self.batch_compute_Nk(k_positions, vectors_tensor)
        Nk_sum = torch.sum(Nk_batch, dim=0)
        
        return (Nk_sum / len(vector_groups)).detach().cpu().numpy()
    
    def reconstruct(self):
        """Reconstruct representative vector sequence by minimizing error"""
        assert self.trained, "Model must be trained first"
        n_groups = round(self.mean_group_count)
        mean_t_tensor = torch.tensor(self.mean_t, dtype=torch.float32, device=self.device)
        reconstructed_vectors = []
        
        # Generate random vectors as candidates (in practice, you might want to use a more sophisticated approach)
        candidate_vectors = torch.randn(100, self.m, device=self.device)  # 100 random vectors as candidates
        
        for k in range(n_groups):
            # Compute Nk for all candidate vectors at position k
            k_tensor = torch.tensor([k] * len(candidate_vectors), dtype=torch.float32, device=self.device)
            Nk_all = self.batch_compute_Nk(k_tensor, candidate_vectors)
            
            # Compute errors
            errors = torch.sum((Nk_all - mean_t_tensor) ** 2, dim=1)
            min_idx = torch.argmin(errors).item()
            best_vector = candidate_vectors[min_idx]
            reconstructed_vectors.append(best_vector.detach().cpu().numpy())
            
        return reconstructed_vectors

    def generate(self, L, tau=0.0):
        """Generate vector sequence of length L with temperature-controlled randomness"""
        assert self.trained, "Model must be trained first"
        if tau < 0:
            raise ValueError("Temperature must be non-negative")
            
        num_groups = (L + self.rank - 1) // self.rank
        mean_t_tensor = torch.tensor(self.mean_t, dtype=torch.float32, device=self.device)
        generated_vectors = []
        
        # Generate random vectors as candidates
        candidate_vectors = torch.randn(100, self.m, device=self.device)  # 100 random vectors as candidates
        
        for k in range(num_groups):
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
                generated_vectors.append(chosen_vector.detach().cpu().numpy())
                
        return generated_vectors[:L] if L <= num_groups else generated_vectors + [np.zeros(self.m) for _ in range(L - num_groups)]

    def save(self, filename):
        """Save model state to file"""
        torch.save(self.state_dict(), filename)
        print(f"Model saved to {filename}")

    def load(self, filename):
        """Load model state from file"""        
        # Use weights only=True to avoid security warnings
        try:
            # PyTorch 1.13+ supports the "weights only" parameter
            state_dict = torch.load(filename, map_location=self.device, weights_only=True)
        except TypeError:
            # Rollback solution for old versions of PyTorch
            state_dict = torch.load(filename, map_location=self.device)
        
        self.load_state_dict(state_dict)
        print(f"Model loaded from {filename}")
        return self

# === Example Usage ===
if __name__=="__main__":
    print("="*50)
    print("Dual Descriptor Vector TS - PyTorch GPU Accelerated Version")
    print("Modified for processing m-dimensional real vector sequences")
    print("="*50)
    
    # Set random seeds to ensure reproducibility
    torch.manual_seed(1)
    random.seed(1)
    np.random.seed(1)
    
    vec_dim = 5 # Dimension of input vectors    
    num_basis = 5 # Number of basis functions
    rank = 1 # Window size for vector aggregation
    rank_op = 'avg' # Window aggregation operation
    user_step = 1   # Step size for nonlinear mode
    
    # Generate 1000 vector sequences with random target vectors
    seqs_vectors, t_list = [], []
    for _ in range(10):
        L = random.randint(100, 200)  # Sequence length
        # Generate random vector sequence
        seq_vectors = np.random.randn(L, vec_dim).astype(np.float32)
        seqs_vectors.append(seq_vectors)
        # Create a random vector target
        t_list.append(np.random.uniform(-1.0, 1.0, vec_dim).astype(np.float32))

    # Initialize the model
    dd = DualDescriptorVectorTS(
        vec_dim=vec_dim, 
        rank=rank,
        rank_op=rank_op,
        num_basis=num_basis, 
        mode='nonlinear', 
        user_step=user_step,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        use_norm=False  # Enable layer normalization
    )
    
    # Display device information
    print(f"\nUsing device: {dd.device}")
    print(f"Vector dimension: {dd.m}")
    print(f"Using Layer Normalization: {dd.use_norm}")

    # Training model
    print("\n" + "="*50)
    print("Starting Gradient Descent Training with LayerNorm")
    print("="*50)
    dd.grad_train(seqs_vectors, t_list, max_iters=100, tol=1e-6, learning_rate=0.1, decay_rate=0.99, batch_size=1024)  
   
    # Predict the target vector of the first sequence
    aseq_vectors = seqs_vectors[0]
    t_pred = dd.predict_t(aseq_vectors)
    print(f"\nPredicted t for first sequence: {[round(x, 4) for x in t_pred]}")    
    
    # Calculate the correlation between the predicted and the real target
    pred_t_list = [dd.predict_t(seq) for seq in seqs_vectors]
    
    # The predicted values and actual values used for correlation calculation
    corr_sum = 0.0
    for i in range(dd.m):
        actu_t = [t_vec[i] for t_vec in t_list]
        pred_t = [t_vec[i] for t_vec in pred_t_list]
        corr = correlation(actu_t, pred_t)
        print(f"Dimension {i} prediction correlation: {corr:.4f}")
        corr_sum += corr
    corr_avg = corr_sum / dd.m
    print(f"Average correlation: {corr_avg:.4f}")       
  
    # Reconstruct the representative vector sequence
    repr_vectors = dd.reconstruct()
    print(f"\nRepresentative vector sequence (length={len(repr_vectors)}):")
    for i, vec in enumerate(repr_vectors[:3]):  # Show first 3 vectors
        print(f"Vector {i}: {[round(x, 4) for x in vec]}")
    
    # Generate new vector sequences
    vecs_det = dd.generate(L=5, tau=0.0)
    vecs_rand = dd.generate(L=5, tau=0.5)
    print("\nDeterministic generation (first 3 vectors):")
    for i, vec in enumerate(vecs_det[:3]):
        print(f"Vector {i}: {[round(x, 4) for x in vec]}")
    print("Stochastic generation (tau=0.5, first 3 vectors):")
    for i, vec in enumerate(vecs_rand[:3]):
        print(f"Vector {i}: {[round(x, 4) for x in vec]}")
   
    # === Combined self-training examples ===
    print("\n" + "="*50)
    print("Combined Auto-Training Example with LayerNorm")
    print("="*50)
    
    # Create a new model with normalization
    dd_gap = DualDescriptorVectorTS(
        vec_dim=vec_dim, 
        rank=rank, 
        num_basis=num_basis, 
        mode='nonlinear', 
        user_step=user_step,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        use_norm=True  # Enable layer normalization
    )
    
    # Generate sample vector sequences
    auto_seqs = []
    for _ in range(10):
        L = random.randint(20, 30)
        auto_seqs.append(np.random.randn(L, vec_dim).astype(np.float32))
    
    # Conduct self-consistenty training (gap mode)
    print("\nTraining in 'gap' mode (self-consistency) with LayerNorm:")
    gap_history = dd_gap.auto_train(
        auto_seqs, 
        max_iters=50, 
        tol=1e-6, 
        learning_rate=0.1, 
        auto_mode='gap',
        batch_size=512
    )
    
    # Generate vector sequences
    print("\nGenerated vector sequences from 'gap' model with LayerNorm:")
    for i in range(2):
        gen_vectors = dd_gap.generate(5, tau=0.2)
        print(f"Sequence {i+1} (first 3 vectors):")
        for j, vec in enumerate(gen_vectors[:3]):
            print(f"  Vector {j}: {[round(x, 4) for x in vec]}")
    
    print("\nAll tests completed successfully with Vector Sequence Processing!")
