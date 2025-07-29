# Copyright (C) Bin-Guang Ma (mbg@mail.hzau.edu.cn). All rights reserved.
# The Dual Descriptor Vector class (Random AB matrix form) - PyTorch GPU Version
# Author: Bin-Guang Ma (assisted by DeepSeek); Date: 2025-6-6
# Optimized for GPU acceleration with batch processing and layer normalization

import math
import random
import itertools
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class DualDescriptorRN(nn.Module):
    """
    Dual Descriptor with GPU acceleration using PyTorch:
      - Learnable coefficient matrix Acoeff ∈ R^{m×L}
      - Learnable basis matrix Bbasis ∈ R^{L×m}
      - Token embeddings M: token → R^m
      - Added Layer Normalization for stable training
      - Optimized with batch processing for GPU acceleration
    """
    def __init__(self, charset, vec_dim=4, bas_dim=50, rank=1, rank_mode='drop', 
                 mode='linear', user_step=None, device='cuda', use_norm=True):
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
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.use_norm = use_norm
        
        # Generate all possible tokens (k-mers + right-padded with '_')
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
        
        # Token embeddings
        self.embedding = nn.Embedding(len(self.tokens), self.m)
        
        # Coefficient matrix Acoeff: m×L
        self.Acoeff = nn.Parameter(torch.empty(self.m, self.L))
        
        # Basis matrix Bbasis: L×m
        self.Bbasis = nn.Parameter(torch.empty(self.L, self.m))
        
        # Layer normalization
        if self.use_norm:
            self.norm = nn.LayerNorm(self.m)
        
        # Initialize parameters
        self.reset_parameters()
        self.to(self.device)
        
    def reset_parameters(self):
        """Initialize model parameters"""
        nn.init.uniform_(self.embedding.weight, -0.5, 0.5)
        nn.init.uniform_(self.Acoeff, -0.1, 0.1)
        nn.init.uniform_(self.Bbasis, -0.1, 0.1)
        if self.use_norm:
            nn.init.constant_(self.norm.weight, 1.0)
            nn.init.constant_(self.norm.bias, 0.0)
    
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
        Args:
            k_tensor: Tensor of position indices [batch_size]
            token_indices: Tensor of token indices [batch_size]
        Returns:
            Tensor of N(k) vectors [batch_size, m]
        """
        # Get embeddings for all tokens [batch_size, m]
        x = self.embedding(token_indices)
        
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
        
        # Apply Layer Normalization if enabled
        if self.use_norm:
            Nk = self.norm(Nk)
            
        return Nk

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

    def grad_train(self, seqs, t_list, max_iters=1000, tol=1e-8, learning_rate=0.01, 
                   continued=False, decay_rate=1.0, print_every=10, batch_size=1024):
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
        
        # Prepare all training samples (position, seq_idx, token_idx)
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

    def auto_train(self, seqs, auto_mode='gap', max_iters=1000, tol=1e-8, 
                   learning_rate=0.01, continued=False, decay_rate=1.0, 
                   print_every=10, batch_size=1024):
        """
        Train the model using self-supervised learning with two modes:
          - 'gap': Gap filling (mask current token and predict it from context)
          - 'reg': Auto-regressive (predict next token from previous tokens)
        
        This method updates all learnable components with GPU acceleration.
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
    print("Dual Descriptor RN - PyTorch GPU Accelerated Version")
    print("Optimized with batch processing and Layer Normalization")
    print("="*50)
    
    # Set random seeds to ensure reproducibility
    torch.manual_seed(11)
    random.seed(11)
    
    charset = ['A','C','G','T']
    vec_dim = 3
    bas_dim = 300
    seq_num = 100

    # Generate sequences of length 100–200 and random targets
    seqs, t_list = [], []
    for _ in range(seq_num):
        L = random.randint(200, 300)
        seq = ''.join(random.choices(charset, k=L))
        seqs.append(seq)
        t_list.append([random.uniform(-1,1) for _ in range(vec_dim)])

    # === Gradient Descent Training Example ===
    print("\n" + "="*50)
    print("Testing Gradient Descent Training")
    print("="*50)

    # Create new model instance with GPU acceleration and normalization
    dd = DualDescriptorRN(
        charset, 
        rank=3, 
        vec_dim=vec_dim, 
        bas_dim=bas_dim, 
        mode='linear', 
        device='cuda' if torch.cuda.is_available() else 'cpu',
        use_norm=False
    )
    
    # Display device information
    print(f"\nUsing device: {dd.device}")
    print(f"Number of tokens: {len(dd.tokens)}")
    print(f"Using Layer Normalization: {dd.use_norm}")

    # Train using gradient descent
    print("\nStarting gradient descent training...")
    grad_history = dd.grad_train(
        seqs, 
        t_list,
        learning_rate=0.01,
        max_iters=100,
        tol=1e-86,
        decay_rate=0.99,
        print_every=5,
        batch_size=2048
    )

    # Predict target vector for first sequence
    aseq = seqs[0]
    t_pred = dd.predict_t(aseq)
    print(f"\nPredicted t for first sequence: {[round(x, 4) for x in t_pred]}")    
    
    # Calculate flattened correlation between predicted and actual targets
    pred_t_list = [dd.predict_t(seq) for seq in seqs]
    
    # Predictions and actuals for correlation calculation
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
    print(f"\nReconstructed representative sequence: {repr_seq[:50]}...")
    
    # Generate sequences
    print("\nGenerated sequences:")
    seq_det = dd.generate(L=100, tau=0.0)
    print(f"Deterministic (tau=0): {seq_det[:50]}...")

    # === Auto-Training Example ===
    # Set random seed for reproducible results
    random.seed(1)
    
    # Define character set and model parameters
    charset = ['A','C','G','T']
    vec_dim = 3   # Vector dimension
    bas_dim = 100 # Base matrix dimension

    # Generate training sequences
    print("\n=== Generating Training Sequences ===")
    seqs = []
    for i in range(30):  # Generate 30 sequences
        L = random.randint(100, 200)
        seq = ''.join(random.choices(charset, k=L))
        seqs.append(seq)
        print(f"Generated sequence {i+1}: length={L}, first 10 chars: {seq[:10]}...")
    
    print("=== Creating Dual Descriptor Model ===")
    dd_auto_gap = DualDescriptorRN(
        charset, 
        rank=1,
        vec_dim=vec_dim,
        bas_dim=bas_dim,
        mode='linear',
        device='cuda' if torch.cuda.is_available() else 'cpu',
        use_norm=True
    )    
    
    # Run self-supervised training (Gap Filling mode)
    print("\n=== Starting Gap Filling Training ===")
    gap_history = dd_auto_gap.auto_train(
        seqs,
        auto_mode='gap',
        max_iters=300,
        learning_rate=0.001,
        decay_rate=0.995,
        print_every=20,
        batch_size=2048
    )

    print("=== Creating Dual Descriptor Model ===")
    dd_auto_reg = DualDescriptorRN(
        charset, 
        rank=1,
        vec_dim=vec_dim,
        bas_dim=bas_dim,
        mode='linear',
        device='cuda' if torch.cuda.is_available() else 'cpu',
        use_norm=True
    )
    
    # Run self-supervised training (Auto-Regressive mode)
    print("\n=== Starting Auto-Regressive Training ===")
    reg_history = dd_auto_reg.auto_train(
        seqs,
        auto_mode='reg',
        max_iters=300,
        learning_rate=0.001,
        decay_rate=0.99,
        print_every=20,
        batch_size=2048
    )
    
    # Generate new sequences
    print("\n=== Generating New Sequences ===")
    
    # Temperature=0 (deterministic generation)
    seq_det = dd_auto_gap.generate(L=40, tau=0.0)
    print(f"Deterministic Generation (tau=0.0):\n{seq_det}")
    
    # Temperature=0.5 (moderate randomness)
    seq_sto = dd_auto_reg.generate(L=40, tau=0.5)
    print(f"\nStochastic Generation (tau=0.5):\n{seq_sto}")
    
    # Save and load model
    print("\n=== Model Persistence Test ===")
    dd_auto_reg.save("auto_trained_model_rn.pt")
    
    # Load model
    dd_loaded = DualDescriptorRN(
        charset, 
        rank=1,
        vec_dim=vec_dim,
        bas_dim=bas_dim,
        mode='linear',
        device='cuda' if torch.cuda.is_available() else 'cpu',
        use_norm=True
    )
    dd_loaded.load("auto_trained_model_rn.pt")
    print("Model loaded successfully. Generating with loaded model:")
    print(dd_loaded.generate(L=20, tau=0.0))
    
    print("\n=== Auto-Training Demo Completed ===")
