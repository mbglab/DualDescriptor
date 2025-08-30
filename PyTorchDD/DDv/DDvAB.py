# Copyright (C) 2005-2025, Bin-Guang Ma (mbg@mail.hzau.edu.cn); SPDX-License-Identifier: MIT
# The Dual Descriptor Vector class (AB matrix form) implemented with PyTorch
# This program is for the demonstration of methodology and not fully refined.
# Author: Bin-Guang Ma (assisted by DeepSeek); Date: 2025-7-29

import math
import itertools
import random
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class DualDescriptorAB(nn.Module):
    """
    Dual Descriptor with GPU acceleration using PyTorch:
      - learnable coefficient matrix Acoeff ∈ R^{m×L}
      - fixed basis matrix Bbasis ∈ R^{L×m}, Bbasis[k][i] = cos(2π*(k+1)/(i+2))
      - learnable token embeddings via nn.Embedding
      - Supports both linear and nonlinear tokenization
      - Batch processing for GPU acceleration
    """
    def __init__(self, charset, vec_dim=4, bas_dim=50, rank=1, rank_mode='drop', 
                 mode='linear', user_step=None, device='cuda'):
        super().__init__()
        self.charset = list(charset)
        self.m = vec_dim
        self.L = bas_dim
        self.rank = rank
        self.rank_mode = rank_mode
        assert mode in ('linear','nonlinear')
        self.mode = mode
        self.step = user_step
        self.trained = False
        self.mean_t = None  
        self.mean_L = None
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Generate tokens
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
        self.vocab_size = len(self.tokens)
        
        # Token embeddings
        self.embedding = nn.Embedding(self.vocab_size, self.m)
        
        # Coefficient matrix A (m x L)
        self.Acoeff = nn.Parameter(torch.empty(self.m, self.L))
        
        # Fixed basis matrix B (L x m)
        Bbasis = torch.empty(self.L, self.m)
        for k in range(self.L):
            for i in range(self.m):
                Bbasis[k, i] = math.cos(2 * math.pi * (k+1) / (i+2))
        self.register_buffer('Bbasis', Bbasis)
        
        # Initialize parameters
        self.reset_parameters()
        self.to(self.device)
    
    def reset_parameters(self):
        """Initialize model parameters"""
        nn.init.uniform_(self.embedding.weight, -0.5, 0.5)
        nn.init.uniform_(self.Acoeff, -0.1, 0.1)
        
    def token_to_indices(self, token_list):
        """Convert list of tokens to tensor of indices"""
        return torch.tensor([self.token_to_idx[tok] for tok in token_list], 
                           device=self.device, dtype=torch.long)
    
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

    def describe(self, seq):
        """Compute N(k) vectors for each token in sequence"""
        toks = self.extract_tokens(seq)
        if not toks:
            return []
        
        token_indices = self.token_to_indices(toks)
        k_positions = torch.arange(len(toks), dtype=torch.float32, device=self.device)
        
        # Get token embeddings
        x = self.embedding(token_indices)  # [seq_len, m]
        
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
        
        return Nk.detach().cpu().numpy()
    
    def S(self, seq):
        """Compute cumulative sum of N(k) vectors"""
        toks = self.extract_tokens(seq)
        if not toks:
            return []
        
        token_indices = self.token_to_indices(toks)
        k_positions = torch.arange(len(toks), dtype=torch.float32, device=self.device)
        
        # Compute Nk vectors
        x = self.embedding(token_indices)
        j_indices = (k_positions % self.L).long()
        B_rows = self.Bbasis[j_indices]
        scalar = torch.sum(B_rows * x, dim=1)
        A_cols = self.Acoeff[:, j_indices].t()
        Nk = A_cols * scalar.unsqueeze(1)
        
        # Compute cumulative sum
        S_cum = torch.cumsum(Nk, dim=0)
        return S_cum.detach().cpu().numpy()

    def grad_train(self, seqs, t_list, max_iters=1000, tol=1e-8, learning_rate=0.01, 
                  continued=False, decay_rate=1.0, print_every=10, batch_size=1024):
        """
        Train model using gradient descent with batch processing
        GPU-accelerated with PyTorch
        """
        if not continued:
            self.reset_parameters()
            
        # Precompute token indices for all sequences
        all_sequences = []
        for seq in seqs:
            tokens = self.extract_tokens(seq)
            token_indices = self.token_to_indices(tokens) if tokens else torch.tensor([], dtype=torch.long, device=self.device)
            all_sequences.append((tokens, token_indices))
        
        # Convert targets to tensor
        t_tensors = [torch.tensor(t, dtype=torch.float32, device=self.device) for t in t_list]
        
        # Prepare training samples (position, seq_idx, token_idx)
        all_samples = []
        for seq_idx, (tokens, token_indices) in enumerate(all_sequences):
            for pos in range(len(tokens)):
                all_samples.append((pos, seq_idx, token_indices[pos].item()))
        
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
                
                # Prepare batch tensors
                k_list = []
                seq_idx_list = []
                token_idx_list = []
                
                for pos, seq_idx, token_idx in batch_samples:
                    k_list.append(pos)
                    seq_idx_list.append(seq_idx)
                    token_idx_list.append(token_idx)
                
                k_tensor = torch.tensor(k_list, dtype=torch.float32, device=self.device)
                token_indices_tensor = torch.tensor(token_idx_list, device=self.device)
                
                # Get corresponding targets
                targets = torch.stack([t_tensors[idx] for idx in seq_idx_list])
                
                # Compute Nk vectors
                x = self.embedding(token_indices_tensor)
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
            scheduler.step()
            
            # Print progress
            if it % print_every == 0 or it == max_iters - 1:
                print(f"GD Iter {it:3d}: D = {avg_loss:.6e}, LR = {scheduler.get_last_lr()[0]:.6f}")
            
            # Check convergence
            if abs(prev_loss - avg_loss) < tol:
                print(f"Converged after {it+1} iterations")
                break
            prev_loss = avg_loss
        
        # Compute statistics for reconstruction
        total_positions = 0
        total_t = torch.zeros(self.m, device=self.device)
        for (tokens, _), t in zip(all_sequences, t_tensors):
            n_tokens = len(tokens)
            total_positions += n_tokens
            total_t += t * n_tokens
        
        self.mean_L = total_positions / len(seqs)
        self.mean_t = (total_t / total_positions).detach().cpu().numpy()
        self.trained = True
        
        return history

    def auto_train(self, seqs, auto_mode='gap', max_iters=1000, tol=1e-8, 
                   learning_rate=0.01, continued=False, decay_rate=1.0, 
                   print_every=10, batch_size=1024):
        """
        Self-supervised training with GPU acceleration
        Supports 'gap' and 'reg' modes
        """
        if not continued:
            self.reset_parameters()
            
        # Precompute token indices
        all_sequences = []
        for seq in seqs:
            tokens = self.extract_tokens(seq)
            token_indices = self.token_to_indices(tokens) if tokens else torch.tensor([], dtype=torch.long, device=self.device)
            all_sequences.append((tokens, token_indices))
        
        # Prepare training samples
        all_samples = []
        for seq_idx, (tokens, token_indices) in enumerate(all_sequences):
            n = len(tokens)
            if n == 0:
                continue
                
            if auto_mode == 'gap':
                # Predict current token
                for pos in range(n):
                    all_samples.append((pos, token_indices[pos].item(), token_indices[pos].item()))
            else:  # 'reg' mode
                # Predict next token
                for pos in range(n - 1):
                    all_samples.append((pos, token_indices[pos].item(), token_indices[pos+1].item()))
        
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
                
                # Prepare batch tensors
                k_list = []
                input_idx_list = []
                target_idx_list = []
                
                for k, input_idx, target_idx in batch_samples:
                    k_list.append(k)
                    input_idx_list.append(input_idx)
                    target_idx_list.append(target_idx)
                
                k_tensor = torch.tensor(k_list, dtype=torch.float32, device=self.device)
                input_indices = torch.tensor(input_idx_list, device=self.device)
                target_indices = torch.tensor(target_idx_list, device=self.device)
                
                # Compute Nk vectors
                x_input = self.embedding(input_indices)
                j_indices = (k_tensor % self.L).long()
                B_rows = self.Bbasis[j_indices]
                scalar = torch.sum(B_rows * x_input, dim=1)
                A_cols = self.Acoeff[:, j_indices].t()
                Nk = A_cols * scalar.unsqueeze(1)
                
                # Get target embeddings
                target_emb = self.embedding(target_indices)
                
                # Compute loss (MSE between Nk and target embedding)
                loss = torch.mean(torch.sum((Nk - target_emb) ** 2, dim=1))
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item() * len(batch_samples)
            
            # Average loss and update scheduler
            avg_loss = epoch_loss / total_samples
            history.append(avg_loss)
            scheduler.step()
            
            # Print progress
            mode_name = "Gap" if auto_mode == 'gap' else "Reg"
            if it % print_every == 0 or it == max_iters - 1:
                print(f"AutoTrain({mode_name}) Iter {it:3d}: loss = {avg_loss:.6e}, LR = {scheduler.get_last_lr()[0]:.6f}")
            
            # Check convergence
            if abs(prev_loss - avg_loss) < tol:
                print(f"Converged after {it+1} iterations")
                break
            prev_loss = avg_loss
        
        # Compute statistics for reconstruction
        total_emb = torch.zeros(self.m, device=self.device)
        total_tokens = 0
        for tokens, token_indices in all_sequences:
            if len(token_indices) > 0:
                emb = self.embedding(token_indices).sum(dim=0)
                total_emb += emb
                total_tokens += len(token_indices)
        
        self.mean_L = total_tokens / len(seqs) if len(seqs) > 0 else 0
        self.mean_t = (total_emb / total_tokens).detach().cpu().numpy() if total_tokens > 0 else np.zeros(self.m)
        self.trained = True
        
        return history

    def predict_t(self, seq):
        """Predict target vector as average of N(k) vectors"""
        toks = self.extract_tokens(seq)
        if not toks:
            return np.zeros(self.m)
        
        # Compute all Nk vectors
        Nk = self.describe(seq)
        return np.mean(Nk, axis=0)

    def reconstruct(self):
        """Reconstruct representative sequence"""
        assert self.trained, "Model must be trained first"
        n_tokens = round(self.mean_L)
        mean_t_tensor = torch.tensor(self.mean_t, dtype=torch.float32, device=self.device)
        seq_tokens = []
        
        # Precompute all token embeddings
        all_token_indices = torch.arange(self.vocab_size, device=self.device)
        all_embeddings = self.embedding(all_token_indices)
        
        for k in range(n_tokens):
            # Compute j index for basis
            j = k % self.L
            B_row = self.Bbasis[j].unsqueeze(0)  # [1, m]
            
            # Compute scalar = B[j] • x for all tokens
            scalar = torch.sum(B_row * all_embeddings, dim=1)  # [vocab_size]
            
            # Compute Nk = scalar * A[:,j]
            A_col = self.Acoeff[:, j]  # [m]
            Nk_all = A_col * scalar.unsqueeze(1)  # [vocab_size, m]
            
            # Compute error to mean target
            errors = torch.sum((Nk_all - mean_t_tensor) ** 2, dim=1)
            min_idx = torch.argmin(errors).item()
            seq_tokens.append(self.idx_to_token[min_idx])
        
        return ''.join(seq_tokens)

    def generate(self, L, tau=0.0):
        """Generate sequence with temperature-controlled randomness"""
        assert self.trained, "Model must be trained first"
        if tau < 0:
            raise ValueError("Temperature must be non-negative")
            
        num_blocks = (L + self.rank - 1) // self.rank
        mean_t_tensor = torch.tensor(self.mean_t, dtype=torch.float32, device=self.device)
        generated_tokens = []
        
        # Precompute all token embeddings
        all_token_indices = torch.arange(self.vocab_size, device=self.device)
        all_embeddings = self.embedding(all_token_indices)
        
        for k in range(num_blocks):
            # Compute j index for basis
            j = k % self.L
            B_row = self.Bbasis[j].unsqueeze(0)  # [1, m]
            
            # Compute scalar = B[j] • x for all tokens
            scalar = torch.sum(B_row * all_embeddings, dim=1)  # [vocab_size]
            
            # Compute Nk = scalar * A[:,j]
            A_col = self.Acoeff[:, j]  # [m]
            Nk_all = A_col * scalar.unsqueeze(1)  # [vocab_size, m]
            
            # Compute scores (negative MSE)
            errors = torch.sum((Nk_all - mean_t_tensor) ** 2, dim=1)
            scores = -errors
            
            # Select token
            if tau == 0:  # Deterministic
                best_idx = torch.argmax(scores).item()
                best_tok = self.idx_to_token[best_idx]
                generated_tokens.append(best_tok)
            else:  # Stochastic
                probs = torch.softmax(scores / tau, dim=0).detach().cpu().numpy()
                chosen_idx = np.random.choice(self.vocab_size, p=probs)
                chosen_tok = self.idx_to_token[chosen_idx]
                generated_tokens.append(chosen_tok)
        
        # Trim to exact length
        full_seq = ''.join(generated_tokens)
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
    print("Dual Descriptor AB - PyTorch GPU Accelerated Version")
    print("Optimized with batch processing")
    print("="*50)
    
    # Set random seeds for reproducibility
    torch.manual_seed(11)
    random.seed(11)
    
    charset = ['A','C','G','T']
    vec_dim = 3
    bas_dim = 300
    seq_num = 100
    
    # Generate sequences and random targets
    seqs, t_list = [], []
    for _ in range(seq_num):
        L = random.randint(200, 300)
        seq = ''.join(random.choices(charset, k=L))
        seqs.append(seq)
        t_list.append([random.uniform(-1,1) for _ in range(vec_dim)])
    
    # Create model
    dd = DualDescriptorAB(
        charset, 
        vec_dim=vec_dim, 
        bas_dim=bas_dim, 
        rank=3, 
        mode='linear',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Display device information
    print(f"\nUsing device: {dd.device}")
    print(f"Number of tokens: {len(dd.tokens)}")
    
    # === Gradient Descent Training ===
    print("\n" + "="*50)
    print("Testing Gradient Descent Training")
    print("="*50)
    
    # Train using gradient descent
    print("\nStarting gradient descent training...")
    grad_history = dd.grad_train(
        seqs, 
        t_list,
        learning_rate=0.1,
        max_iters=100,
        tol=1e-66,
        print_every=5,
        decay_rate=0.99,
        batch_size=1024
    )
    
    # Predict target for first sequence
    aseq = seqs[0]
    t_pred = dd.predict_t(aseq)
    print(f"\nPredicted t for first sequence: {[round(x.item(), 4) for x in t_pred]}")
    
    # Calculate prediction correlation
    pred_t_list = [dd.predict_t(seq) for seq in seqs]
    
    corr_sum = 0.0
    for i in range(dd.m):
        actu_t = [t_vec[i] for t_vec in t_list]
        pred_t = [t_vec[i] for t_vec in pred_t_list]
        corr = correlation(actu_t, pred_t)
        print(f"Prediction correlation: {corr:.4f}")
        corr_sum += corr
    corr_avg = corr_sum / dd.m
    print(f"Average correlation: {corr_avg:.4f}")
    
    # Reconstruct representative sequence
    repr_seq = dd.reconstruct()
    print(f"\nReconstructed representative sequence: {repr_seq[:50]}...")
    
    # Generate sequences
    print("\nGenerated sequences:")
    seq_det = dd.generate(L=100, tau=0.0)
    print(f"Deterministic (tau=0): {seq_det[:50]}...")
    
    # === Auto-Training Example ===
    # Set random seeds
    torch.manual_seed(2)
    random.seed(2)
    
    # Define parameters
    charset = ['A','C','G','T']
    vec_dim = 3
    bas_dim = 100
    
    # Generate training sequences
    print("\n=== Generating Training Sequences ===")
    seqs = []
    for i in range(30):
        L = random.randint(100, 200)
        seq = ''.join(random.choices(charset, k=L))
        seqs.append(seq)
        print(f"Generated sequence {i+1}: length={L}, first 10 chars: {seq[:10]}...")
    
    # Create model for gap filling
    print("\n=== Creating Dual Descriptor Model for Gap Filling ===")
    dd_gap = DualDescriptorAB(
        charset,
        vec_dim=vec_dim,
        bas_dim=bas_dim,
        rank=1,
        mode='linear',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Train in gap filling mode
    print("\n=== Starting Gap Filling Training ===")
    gap_history = dd_gap.auto_train(
        seqs,
        auto_mode='gap',
        max_iters=300,
        learning_rate=0.001,
        decay_rate=0.995,
        print_every=20,
        batch_size=1024
    )
    
    # Create model for auto-regressive training
    print("\n=== Creating Dual Descriptor Model for Auto-Regressive Training ===")
    dd_reg = DualDescriptorAB(
        charset,
        vec_dim=vec_dim,
        bas_dim=bas_dim,
        rank=1,
        mode='linear',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Train in auto-regressive mode
    print("\n=== Starting Auto-Regressive Training ===")
    reg_history = dd_reg.auto_train(
        seqs,
        auto_mode='reg',
        max_iters=300,
        learning_rate=0.01,
        decay_rate=0.99,
        print_every=20,
        batch_size=1024
    )
    
    # Generate new sequences
    print("\n=== Generating New Sequences ===")
    
    # From gap model
    gap_seq = dd_gap.generate(L=40, tau=0.0)
    print(f"Gap model generation: {gap_seq}")
    
    # From reg model
    reg_seq = dd_reg.generate(L=40, tau=0.0)
    print(f"Reg model generation: {reg_seq}")
    
    # Save and load model
    print("\n=== Model Persistence Test ===")
    dd_reg.save("auto_trained_model.pkl")
    
    # Load model
    dd_loaded = DualDescriptorAB(
        charset,
        vec_dim=vec_dim,
        bas_dim=bas_dim,
        rank=1,
        mode='linear',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    ).load("auto_trained_model.pkl")
    
    print("Model loaded successfully. Generating with loaded model:")
    print(dd_loaded.generate(L=20, tau=0.0))
    
    print("\n=== All Tests Completed ===")
