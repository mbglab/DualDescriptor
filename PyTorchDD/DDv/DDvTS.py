# Copyright (C) 2005-2025, Bin-Guang Ma (mbg@mail.hzau.edu.cn); SPDX-License-Identifier: MIT
# The Dual Descriptor Vector class (Tensor form) implemented with PyTorch
# This program is for the demonstration of methodology and not fully refined.
# Author: Bin-Guang Ma (assisted by DeepSeek); Date: 2025-7-29

import math
import random
import itertools
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class DualDescriptorTS(nn.Module):
    """
    Vector Dual Descriptor with GPU acceleration using PyTorch:
      - tensor P ∈ R^{m×m×o} of basis coefficients
      - embedding: k-mer token embeddings in R^m
      - indexed periods: period[i,j,g] = i*(m*o) + j*o + g + 2
      - basis function phi_{i,j,g}(k) = cos(2π * k / period[i,j,g])
      - supports 'linear' or 'nonlinear' (step-by-rank) k-mer extraction
    """
    def __init__(self, charset, rank=1, rank_mode='drop', vec_dim=2, num_basis=5, mode='linear', user_step=None, device='cuda'):
        super().__init__()
        self.charset = list(charset)
        self.rank = rank    # r-per/k-mer length
        self.rank_mode = rank_mode # 'pad' or 'drop'
        self.m = vec_dim    # embedding dimension
        self.o = num_basis  # number of basis terms         
        assert mode in ('linear','nonlinear')
        self.mode = mode
        self.step = user_step
        self.trained = False
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # 1) Generate all possible tokens (k-mers + right-padded with '_')
        toks = []
        if self.rank_mode=='pad':
            for r in range(1, self.rank+1):
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
        
        # Precompute indexed periods[i][j][g] (fixed, not trainable)
        periods = torch.zeros(self.m, self.m, self.o, dtype=torch.float32)
        for i in range(self.m):
            for j in range(self.m):
                for g in range(self.o):
                    periods[i, j, g] = i*(self.m*self.o) + j*self.o + g + 2
        self.register_buffer('periods', periods)

        # Initialize parameters
        self.reset_parameters()
        self.to(self.device)
        
    def reset_parameters(self):
        """Initialize model parameters"""
        nn.init.uniform_(self.embedding.weight, -0.5, 0.5)
        nn.init.uniform_(self.P, -0.1, 0.1)
        
    def token_to_indices(self, token_list):
        """Convert list of tokens to tensor of indices"""
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
        # Linear mode: sliding window with step=1
        if self.mode == 'linear':
            return [seq[i:i+self.rank] for i in range(L - self.rank + 1)]
        
        # Nonlinear mode: stepping with custom step size
        toks = []
        step = self.step or self.rank  # Use custom step if defined, else use rank length
        
        for i in range(0, L, step):
            frag = seq[i:i+self.rank]
            frag_len = len(frag)
            
            # Pad or drop based on rank_mode setting
            if self.rank_mode == 'pad':
                # Pad fragment with '_' if shorter than rank
                toks.append(frag if frag_len == self.rank else frag.ljust(self.rank, '_'))
            elif self.rank_mode == 'drop':
                # Only add fragments that match full rank length
                if frag_len == self.rank:
                    toks.append(frag)
        return toks

    def batch_compute_Nk(self, k_tensor, token_indices):
        """
        Vectorized computation of N(k) vectors for a batch of positions and tokens
        Optimized using einsum for better performance
        Args:
            k_tensor: Tensor of position indices [batch_size]
            token_indices: Tensor of token indices [batch_size]
        Returns:
            Tensor of N(k) vectors [batch_size, m]
        """
        # Get embeddings for all tokens [batch_size, m]
        x = self.embedding(token_indices)
        
        # Expand dimensions for broadcasting [batch_size, 1, 1, 1]
        k_expanded = k_tensor.view(-1, 1, 1, 1)
        
        # Calculate basis functions: cos(2π*k/periods) [batch_size, m, m, o]
        phi = torch.cos(2 * math.pi * k_expanded / self.periods)
        
        # Optimized computation using einsum
        # Original: term = x.unsqueeze(1).unsqueeze(3) * self.P * phi
        # Original: return term.sum(dim=(2, 3))
        Nk = torch.einsum('bj,ijg,bijg->bi', x, self.P, phi)
            
        return Nk

    def compute_Nk(self, k, token_idx):
        """Compute N(k) for single position and token (uses batch internally)"""
        # Convert to tensors
        k_tensor = torch.tensor([k], dtype=torch.float32, device=self.device)
        idx_tensor = torch.tensor([token_idx], device=self.device)
        
        # Use batch computation
        result = self.batch_compute_Nk(k_tensor, idx_tensor)
        return result[0]  # Return first element

    def describe(self, seq):
        """Compute N(k) vectors for each k-mer in sequence"""
        toks = self.extract_tokens(seq)
        if not toks:
            return []
        
        token_indices = self.token_to_indices(toks)
        k_positions = torch.arange(len(toks), dtype=torch.float32, device=self.device)
        
        # Batch compute all Nk vectors
        N_batch = self.batch_compute_Nk(k_positions, token_indices)
        return N_batch.detach().cpu().numpy()

    def S(self, seq):
        """
        Compute list of S(l)=sum(N(k)) (k=1,...,l; l=1,...,L) for a given sequence.        
        """
        toks = self.extract_tokens(seq)
        if not toks:
            return []
            
        token_indices = self.token_to_indices(toks)
        k_positions = torch.arange(len(toks), dtype=torch.float32, device=self.device)
        
        # Batch compute all Nk vectors
        N_batch = self.batch_compute_Nk(k_positions, token_indices)
        
        # Compute cumulative sum (S vectors)
        S_cum = torch.cumsum(N_batch, dim=0)
        return [s.detach().cpu().numpy() for s in S_cum]

    def D(self, seqs, t_list):
        """
        Compute mean squared deviation D across sequences:
        D = average over all positions of (N(k)-t_seq)^2
        """
        total_loss = 0.0
        total_positions = 0
        
        # Convert target vectors to tensor
        t_tensors = [torch.tensor(t, dtype=torch.float32, device=self.device) for t in t_list]
        
        for seq, t in zip(seqs, t_tensors):
            toks = self.extract_tokens(seq)
            if not toks:
                continue
                
            token_indices = self.token_to_indices(toks)
            k_positions = torch.arange(len(toks), dtype=torch.float32, device=self.device)
            
            # Batch compute all Nk vectors
            N_batch = self.batch_compute_Nk(k_positions, token_indices)
            
            # Compute loss for each position
            losses = torch.sum((N_batch - t) ** 2, dim=1)
            total_loss += losses.sum().item()
            total_positions += len(toks)
                
        return total_loss / total_positions if total_positions else 0.0

    def grad_train(self, seqs, t_list, max_iters=1000, tol=1e-8, learning_rate=0.01, continued=False, decay_rate=1.0, print_every=10, batch_size=1024):
        """
        Train the model using gradient descent with batch processing
        GPU-accelerated with PyTorch.
        """
        if not continued:
            self.reset_parameters()
            
        # Precompute token indices for all sequences
        all_tokens = []
        for seq in seqs:
            toks = self.extract_tokens(seq)
            token_indices = self.token_to_indices(toks) if toks else torch.tensor([], dtype=torch.long, device=self.device)
            all_tokens.append((toks, token_indices))
        
        # Convert target vectors to tensor
        t_tensors = [torch.tensor(t, dtype=torch.float32, device=self.device) for t in t_list]
        
        # Prepare all training samples (position, token_idx, target)
        all_samples = []
        for seq_idx, (toks, token_indices) in enumerate(all_tokens):
            if not toks:
                continue
            for pos in range(len(toks)):
                # Store as tuple: (position, seq_idx, token_index)
                all_samples.append((pos, seq_idx, token_indices[pos].item()))
        
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
                token_idx_list = []
                target_list = []
                
                for pos, seq_idx, token_idx in batch_samples:
                    k_list.append(pos)
                    token_idx_list.append(token_idx)
                    target_list.append(t_tensors[seq_idx])
                
                # Create tensors directly on GPU
                k_tensor = torch.tensor(k_list, dtype=torch.float32, device=self.device)
                token_indices_tensor = torch.tensor(token_idx_list, device=self.device)
                targets = torch.stack(target_list)
                
                # Compute Nk in batch
                Nk_batch = self.batch_compute_Nk(k_tensor, token_indices_tensor)
                
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
        total_token_count = 0
        total_t = torch.zeros(self.m, device=self.device)
        for seq, t_vec in zip(seqs, t_tensors):
            tokens = self.extract_tokens(seq)
            total_token_count += len(tokens)
            total_t += t_vec * len(tokens)
            
        self.mean_token_count = total_token_count / len(seqs)
        self.mean_t = (total_t / total_token_count).detach().cpu().numpy()
        self.trained = True
        return history

    def auto_train(self, seqs, max_iters=100, tol=1e-6, learning_rate=0.01, continued=False, auto_mode='reg', decay_rate=1.0, print_every=10, batch_size=1024):
        """
        Self-training method using gradient descent with batch processing
        GPU-accelerated with PyTorch.
        """
        if auto_mode not in ('gap', 'reg'):
            raise ValueError("auto_mode must be either 'gap' or 'reg'")

        if not continued:
            self.reset_parameters()
            
        # Precompute token indices for all sequences
        all_sequences = []
        for seq in seqs:
            tokens = self.extract_tokens(seq)
            token_indices = self.token_to_indices(tokens) if tokens else torch.tensor([], dtype=torch.long, device=self.device)
            all_sequences.append((tokens, token_indices))
        
        # Prepare all training samples
        all_samples = []
        for seq_idx, (tokens, token_indices) in enumerate(all_sequences):
            if not tokens:
                continue
                
            if auto_mode == 'gap':
                # Each token is a sample (position k, token_idx)
                for k, token_idx in enumerate(token_indices):
                    all_samples.append((k, token_idx.item()))
            else:  # 'reg' mode
                # Each token except last is a sample (position k, current token, next token)
                for k in range(len(tokens) - 1):
                    all_samples.append((k, token_indices[k].item(), token_indices[k+1].item()))
        
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
                current_indices_list = []
                target_indices_list = [] if auto_mode == 'reg' else None
                
                for sample in batch_samples:
                    if auto_mode == 'reg':
                        k, current_idx, next_idx = sample
                        k_list.append(k)
                        current_indices_list.append(current_idx)
                        target_indices_list.append(next_idx)
                    else:  # 'gap' mode
                        k, token_idx = sample
                        k_list.append(k)
                        current_indices_list.append(token_idx)
                
                # Create tensors directly on GPU
                k_tensor = torch.tensor(k_list, dtype=torch.float32, device=self.device)
                current_indices_tensor = torch.tensor(current_indices_list, device=self.device)
                
                # Compute Nk for current tokens
                Nk_batch = self.batch_compute_Nk(k_tensor, current_indices_tensor)
                
                # Get target embeddings
                if auto_mode == 'gap':
                    targets = self.embedding(current_indices_tensor)
                else:  # 'reg' mode
                    target_indices_tensor = torch.tensor(target_indices_list, device=self.device)
                    targets = self.embedding(target_indices_tensor)
                
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
        total_token_count = 0
        for tokens, token_indices in all_sequences:
            embeds = self.embedding(token_indices)
            total_token_count += len(tokens)
            total_t += embeds.sum(dim=0)
        
        self.mean_token_count = total_token_count / len(seqs)
        self.mean_t = (total_t / total_token_count).detach().cpu().numpy()
        self.trained = True
        
        return history

    def predict_t(self, seq):
        """
        Predict target vector for a sequence
        Returns the average of all N(k) vectors in the sequence
        """
        toks = self.extract_tokens(seq)
        if not toks:
            return [0.0] * self.m
            
        token_indices = self.token_to_indices(toks)
        k_positions = torch.arange(len(toks), dtype=torch.float32, device=self.device)
        
        # Batch compute all Nk vectors
        Nk_batch = self.batch_compute_Nk(k_positions, token_indices)
        Nk_sum = torch.sum(Nk_batch, dim=0)
        
        return (Nk_sum / len(toks)).detach().cpu().numpy()
    
    def reconstruct(self):
        """Reconstruct representative sequence by minimizing error"""
        assert self.trained, "Model must be trained first"
        n_tokens = round(self.mean_token_count)
        mean_t_tensor = torch.tensor(self.mean_t, dtype=torch.float32, device=self.device)
        seq_tokens = []
        
        # Precompute all token embeddings
        all_token_indices = torch.arange(len(self.tokens), device=self.device)
        all_embeddings = self.embedding(all_token_indices)
        
        for k in range(n_tokens):
            # Compute Nk for all tokens at position k
            k_tensor = torch.tensor([k] * len(self.tokens), dtype=torch.float32, device=self.device)
            Nk_all = self.batch_compute_Nk(k_tensor, all_token_indices)
            
            # Compute errors
            errors = torch.sum((Nk_all - mean_t_tensor) ** 2, dim=1)
            min_idx = torch.argmin(errors).item()
            best_tok = self.idx_to_token[min_idx]
            seq_tokens.append(best_tok)
            
        return ''.join(seq_tokens)

    def generate(self, L, tau=0.0):
        """Generate sequence of length L with temperature-controlled randomness"""
        assert self.trained, "Model must be trained first"
        if tau < 0:
            raise ValueError("Temperature must be non-negative")
            
        num_blocks = (L + self.rank - 1) // self.rank
        mean_t_tensor = torch.tensor(self.mean_t, dtype=torch.float32, device=self.device)
        generated_tokens = []
        
        # Precompute all token embeddings
        all_token_indices = torch.arange(len(self.tokens), device=self.device)
        all_embeddings = self.embedding(all_token_indices)
        
        for k in range(num_blocks):
            # Compute Nk for all tokens at position k
            k_tensor = torch.tensor([k] * len(self.tokens), dtype=torch.float32, device=self.device)
            Nk_all = self.batch_compute_Nk(k_tensor, all_token_indices)
            
            # Compute scores
            errors = torch.sum((Nk_all - mean_t_tensor) ** 2, dim=1)
            scores = -errors  # Convert to score (higher = better)
            
            if tau == 0:  # Deterministic selection
                max_idx = torch.argmax(scores).item()
                best_tok = self.idx_to_token[max_idx]
                generated_tokens.append(best_tok)
            else:  # Stochastic selection
                probs = torch.softmax(scores / tau, dim=0).detach().cpu().numpy()
                chosen_idx = random.choices(range(len(self.tokens)), weights=probs, k=1)[0]
                chosen_tok = self.idx_to_token[chosen_idx]
                generated_tokens.append(chosen_tok)
                
        full_seq = ''.join(generated_tokens)
        return full_seq[:L]      

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

    from statistics import correlation
    
    print("="*50)
    print("Dual Descriptor TS - PyTorch GPU Accelerated Version")
    print("Optimized with batch processing")
    print("="*50)
    
    # Set random seeds to ensure reproducibility
    torch.manual_seed(11)
    random.seed(11)
    
    charset = ['A','C','G','T']
    vec_dim = 10
    num_basis = 10
    rank = 6
    user_step = 3
    
    # Generate 1000 sequences with random target vectors
    seqs, t_list = [], []
    for _ in range(100):
        L = random.randint(200, 300)
        seq = ''.join(random.choices(charset, k=L))
        seqs.append(seq)
        # Create a random vector target
        t_list.append([random.uniform(-1.0, 1.0) for _ in range(vec_dim)])

    # Initialize the model
    dd = DualDescriptorTS(
        charset, 
        rank=rank, 
        vec_dim=vec_dim, 
        num_basis=num_basis, 
        mode='nonlinear', 
        user_step=user_step,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Display device information
    print(f"\nUsing device: {dd.device}")
    print(f"Number of tokens: {len(dd.tokens)}")

    # Training model
    print("\n" + "="*50)
    print("Starting Gradient Descent Training")
    print("="*50)
    dd.grad_train(seqs, t_list, max_iters=200, tol=1e-199, learning_rate=0.1, decay_rate = 0.99, batch_size=2048)  
   
    # Predict the target vector of the first sequence
    aseq = seqs[0]
    t_pred = dd.predict_t(aseq)
    print(f"\nPredicted t for first sequence: {[round(x, 4) for x in t_pred]}")    
    
    # Calculate the correlation between the predicted and the real target
    pred_t_list = [dd.predict_t(seq) for seq in seqs]
    
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
  
    # Reconstruct the representative sequence
    repr_seq = dd.reconstruct()
    print(f"\nRepresentative sequence (len={len(repr_seq)}): {repr_seq[:50]}...")
    
    # Generate new sequences
    seq_det = dd.generate(L=100, tau=0.0)
    seq_rand = dd.generate(L=100, tau=0.5)
    print("\nDeterministic generation:", seq_det[:50] + "...")
    print("Stochastic generation (tau=0.5):", seq_rand[:50] + "...")    
   
    # === Combined self-training examples ===
    print("\n" + "="*50)
    print("Combined Auto-Training Example")
    print("="*50)
    
    # Create a new model
    dd_gap = DualDescriptorTS(
        charset, 
        rank=rank, 
        vec_dim=vec_dim, 
        num_basis=num_basis, 
        mode='nonlinear', 
        user_step=user_step,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Generate sample sequences
    auto_seqs = []
    for _ in range(10):
        L = random.randint(200, 300)
        auto_seqs.append(''.join(random.choices(charset, k=L)))
    
    # Conduct self-consistenty training (gap mode)
    print("\nTraining in 'gap' mode (self-consistency):")
    gap_history = dd_gap.auto_train(
        auto_seqs, 
        max_iters=50, 
        tol=1e-8, 
        learning_rate=0.5, 
        auto_mode='gap',
        batch_size=1024
    )
    
    # Generate sequences
    print("\nGenerated sequences from 'gap' model:")
    for i in range(2):
        gen_seq = dd_gap.generate(100, tau=0.2)
        print(f"Sequence {i+1}: {gen_seq[:50]}...") 
    
    print("\nAll tests completed successfully!")