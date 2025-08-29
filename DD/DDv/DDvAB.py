# Copyright (C) 2005-2025, Bin-Guang Ma (mbg@mail.hzau.edu.cn); SPDX-License-Identifier: MIT
# The Dual Descriptor Vector class (AB matrix form) implemented with pure Python
# This program is for the demonstration of methodology and not fully refined.
# Author: Bin-Guang Ma (assisted by DeepSeek); Date: 2025-6-5

import math
import itertools
import random
import pickle

class DualDescriptorAB:
    """
    Dual Descriptor with:
      - learnable coefficient matrix Acoeff ∈ R^{m×L}
      - fixed basis matrix Bbasis ∈ R^{L×m}, Bbasis[k][i] = cos(2π*(k+1)/(i+1))
      - learnable token embeddings x_map: token → R^m
    """
    def __init__(self, charset, vec_dim=4, bas_dim=50, rank=1, rank_mode='drop', mode='linear', user_step=None):
        self.charset = list(charset)       
        self.m = vec_dim
        self.L = bas_dim
        self.rank = rank
        self.rank_mode = rank_mode
        assert mode in ('linear','nonlinear')
        self.mode = mode
        self.step = user_step
        self.trained = False
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

        # token embeddings
        self.x_map = {
            tok: [random.uniform(-0.5, 0.5) for _ in range(self.m)]
            for tok in self.tokens
        }

        # initialize Acoeff: m×L
        self.Acoeff = [[random.uniform(-0.1,0.1) for _ in range(self.L)]
                       for _ in range(self.m)]
        # build fixed basis Bbasis: L×m
        self.Bbasis = [[math.cos(2*math.pi*(k+1)/(i+1))
                        for i in range(self.m)]
                       for k in range(self.L)]        
        # cache its transpose: m×L
        self.B_t = self._transpose(self.Bbasis)

    # ---- linear algebra helpers ----
    def _transpose(self, M):
        return [list(col) for col in zip(*M)]

    def _mat_mul(self, A, B):
        # A: p×q, B: q×r → C: p×r
        p, q = len(A), len(A[0])
        r    = len(B[0])
        C = [[0.0]*r for _ in range(p)]
        for i in range(p):
            for k in range(q):
                aik = A[i][k]
                for j in range(r):
                    C[i][j] += aik * B[k][j]
        return C

    def _mat_vec(self, M, v):
        # M: p×q, v: length q → length p
        return [sum(M[i][j]*v[j] for j in range(len(v))) for i in range(len(M))]

    def _invert(self, A):
        # Gauss-Jordan invert n×n matrix A
        n = len(A)
        M = [A[i][:] + [1.0 if i==j else 0.0 for j in range(n)]
             for i in range(n)]
        for i in range(n):
            piv = M[i][i]
            if abs(piv)<1e-12:
                continue
            M[i] = [x/piv for x in M[i]]
            for r in range(n):
                if r==i: continue
                fac = M[r][i]
                M[r] = [M[r][c] - fac*M[i][c] for c in range(2*n)]
        return [row[n:] for row in M]

    # ---- tokenization ----
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

    # ---- describe sequence ----
    def describe(self, seq):
        toks = self.extract_tokens(seq)
        N = []
        for k, tok in enumerate(toks):
            A = [[self.Acoeff[i][k%self.L]] for i in range(self.m)]
            B = [[self.Bbasis[k%self.L][i] for i in range(self.m)]]
            P = self._mat_mul(A, B)
            x = self.x_map[tok]  # m-vector
            Nk = self._mat_vec(P, x)
            N.append(Nk)
        return N

    def S(self, seq):
        """
        Compute list of S(l)=sum(N(k)) (k=1,...,l; l=1,...,L) for a given sequence.        
        """
        Nk_list = self.describe(seq)        
        S = [0.0] * self.m; S_list = []
        for l in range(len(Nk_list)):
            for i in range(self.m):
                S[i] += Nk_list[l][i]
            S_list.append(list(S))
        return S_list   

    # ---- compute deviation ----
    def deviation(self, seqs, t_list):
        tot = 0.0
        cnt = 0
        for seq, t in zip(seqs, t_list):
            for Nk in self.describe(seq):
                for i in range(self.m):
                    diff = Nk[i] - t[i]
                    tot += diff*diff
                cnt += 1
        return tot/cnt if cnt else 0.0

    # ---- update Acoeff ----
    def update_Acoeff(self, seqs, t_list):
        """
        Update coefficient matrix Acoeff using position-specific projections.
        Matches the computation method used in describe().
        
        For each token at position k:
          idx = k % L  (basis index)
          scalar = dot(Bbasis[idx], x)  (projection)
          Nk = scalar * Acoeff[:, idx]  (m-dimensional vector)
        
        We minimize ||scalar * Acoeff[:, idx] - t||^2 for each position.
        """
        m, L = self.m, self.L
        # Initialize accumulators
        numerator = [[0.0] * L for _ in range(m)]  # m x L
        denominator = [0.0] * L                    # L-dimensional
        
        for seq, t in zip(seqs, t_list):
            toks = self.extract_tokens(seq)
            for k, tok in enumerate(toks):
                idx = k % L  # Basis index for this position
                x = self.x_map[tok]
                
                # Compute projection scalar: Bbasis[idx] • x
                scalar = sum(self.Bbasis[idx][i] * x[i] for i in range(m))
                
                # Update accumulators
                for i in range(m):
                    numerator[i][idx] += scalar * t[i]
                denominator[idx] += scalar * scalar  # scalar²
        
        # Update Acoeff column-wise
        for j in range(L):
            if abs(denominator[j]) > 1e-12:  # Avoid division by zero
                inv_denom = 1.0 / denominator[j]
                for i in range(m):
                    self.Acoeff[i][j] = numerator[i][j] * inv_denom

    # ---- update x_map ----
    def update_x(self, seqs, t_list):
        """
        Update token embeddings (x_map) using position-specific projections.
        Matches the computation method used in describe().
        
        For each token at position k:
          idx = k % L  (basis index)
          scalar = dot(Bbasis[idx], x)  (projection)
          Nk = scalar * Acoeff[:, idx]  (m-dimensional vector)
        
        Solve linear system: Mx = b for each token using correct formulation:
          M = Σ [ (A_col ⊗ basis_vec)^T (A_col ⊗ basis_vec) ]
          b = Σ [ <A_col, t> * basis_vec ]
        """
        m, L = self.m, self.L
        # Precompute token occurrence data: {token: [(idx, target)]}
        token_data = {tok: [] for tok in self.tokens}
        
        # Collect position data for each token
        for seq, t in zip(seqs, t_list):
            toks = self.extract_tokens(seq)
            for k, tok in enumerate(toks):
                idx = k % L  # Basis index
                token_data[tok].append((idx, t))
        
        # Update each token embedding
        for tok, occurrences in token_data.items():
            if not occurrences:
                continue
                
            # Build linear system: M (m x m) and b (m-dimensional)
            M = [[0.0] * m for _ in range(m)]
            b = [0.0] * m
            
            for (idx, t) in occurrences:
                # Get basis vector and Acoeff column
                basis_vec = self.Bbasis[idx]
                A_col = [self.Acoeff[i][idx] for i in range(m)]
                
                # Compute ||A_col||² = A_col · A_col
                a_norm_sq = sum(a*a for a in A_col)
                
                # Correctly compute M = Σ(||A_col||² * (basis_vec ⊗ basis_vec))
                for i in range(m):
                    for j in range(m):
                        M[i][j] += a_norm_sq * basis_vec[i] * basis_vec[j]
                
                # Compute b contribution: <A_col, t> * basis_vec
                a_dot_t = sum(A_col[i] * t[i] for i in range(m))
                for j in range(m):
                    b[j] += a_dot_t * basis_vec[j]
            
            # Solve linear system: Mx = b
            try:
                M_inv = self._invert(M)
                self.x_map[tok] = self._mat_vec(M_inv, b)
            except:  # Fallback to identity on singular matrix
                pass

    # ---- training loop ----
    def train(self, seqs, t_list, max_iters=10, tol=1e-8, print_every=10):        
        D_prev = float('inf')
        history = []
        for it in range(max_iters):
            self.update_Acoeff(seqs, t_list)
            self.update_x(seqs, t_list)
            D = self.deviation(seqs, t_list)
            history.append(D)
            if it % print_every == 0 or it == max_iters - 1:
                print(f"Iter {it:2d}: D = {D:.6e}")
            if D >= D_prev - tol:
                print("Converged.")
                break
            D_prev = D
        self.mean_L = int(sum(len(self.extract_tokens(seq)) for seq in seqs) / len(seqs))
        self.mean_t = [0.0] * self.m
        for t in t_list:
            for i in range(self.m):
                self.mean_t[i] += t[i]
        self.mean_t = [x / len(t_list) for x in self.mean_t]        
        self.trained = True
        return history

    def grad_train(self, seqs, t_list, max_iters=1000, tol=1e-8, learning_rate=0.01, continued=False, decay_rate=1.0, print_every=10):
        """
        Train the DualDescriptorAB model using gradient descent optimization with position-level normalization.
        
        This implementation uses the total number of tokens across all sequences for normalization,
        which provides more stable gradients and better handling of variable-length sequences.
        
        Steps:
        1. Compute total number of tokens (positions) in the dataset
        2. For each iteration:
            a. Initialize gradients for Acoeff and x_map
            b. For each sequence and each position:
                - Compute predicted Nk vector
                - Compute loss and gradients using total_positions normalization
            c. Update parameters using learning rate
            d. Apply learning rate decay
            e. Track convergence using mean squared deviation
        
        Args:
            seqs: List of input sequences
            t_list: List of target vectors (each is m-dimensional)
            max_iters: Maximum number of training iterations
            tol: Tolerance for convergence detection
            learning_rate: Initial learning rate for gradient descent
            continued: Whether to continue training from current parameters
            decay_rate: Learning rate decay factor per iteration
            print_every: Print progress every N iterations
            
        Returns:
            list: Training loss history
        """
        # Reinitialize parameters if not continuing training
        if not continued:
            # Initialize token embeddings
            self.x_map = {
                tok: [random.uniform(-0.5, 0.5) for _ in range(self.m)]
                for tok in self.tokens
            }
            # Initialize coefficient matrix
            self.Acoeff = [[random.uniform(-0.1, 0.1) for _ in range(self.L)]
                           for _ in range(self.m)]
        
        # Compute total number of tokens (positions) in the dataset
        total_positions = sum(len(self.extract_tokens(seq)) for seq in seqs)
        if total_positions == 0:
            raise ValueError("No tokens found in sequences")
        
        n_seqs = len(seqs)
        history = []
        D_prev = float('inf')
        
        for it in range(max_iters):
            # Initialize gradients
            grad_A = [[0.0] * self.L for _ in range(self.m)]  # Gradient for Acoeff
            grad_x = {tok: [0.0] * self.m for tok in self.tokens}  # Gradient for x_map
            
            total_loss = 0.0  # Track average loss per token
            
            # Process each sequence and each position
            for seq, t in zip(seqs, t_list):
                toks = self.extract_tokens(seq)
                T = len(toks)
                if T == 0:  # Skip empty sequences
                    continue
                    
                # Compute position-wise descriptor vectors
                for k, tok in enumerate(toks):
                    j = k % self.L
                    # Compute projection scalar: Bbasis[j] • x
                    scalar = sum(self.Bbasis[j][i] * self.x_map[tok][i] for i in range(self.m))
                    # Compute Nk = scalar * Acoeff[:, j]
                    Nk = [self.Acoeff[i][j] * scalar for i in range(self.m)]
                    
                    # Compute position loss (using position-level targets)
                    pos_loss = 0.0
                    for i in range(self.m):
                        diff = Nk[i] - t[i]
                        pos_loss += diff * diff
                    
                    # Add to total loss (normalized by dimensions)
                    total_loss += pos_loss / self.m
                    
                    # Compute gradients for this position
                    for i in range(self.m):
                        # Normalize by total positions (like grad_train.py)
                        error_i = 2 * (Nk[i] - t[i]) / (self.m * total_positions)
                        
                        # Gradient for Acoeff[i][j]
                        grad_A[i][j] += error_i * scalar
                        
                        # Gradient for x_map[tok]
                        for l in range(self.m):
                            grad_x[tok][l] += error_i * self.Acoeff[i][j] * self.Bbasis[j][l]
            
            # Average loss per token (for tracking)
            avg_loss = total_loss / total_positions
            history.append(avg_loss)
            
            # Update parameters using gradients
            for i in range(self.m):
                for j in range(self.L):
                    self.Acoeff[i][j] -= learning_rate * grad_A[i][j]
                    
            for tok in self.tokens:
                for l in range(self.m):
                    self.x_map[tok][l] -= learning_rate * grad_x[tok][l]
            
            # Apply learning rate decay
            learning_rate *= decay_rate
            
            # Print progress and check convergence
            if it % print_every == 0 or it == max_iters - 1:
                print(f"Iter {it:3d}: D = {avg_loss:.6e}, LR = {learning_rate:.6f}")
                
            # Check convergence based on loss improvement
            if D_prev - avg_loss < tol:
                print(f"Converged after {it+1} iterations.")
                break
            D_prev = avg_loss
        
        # Final setup after training
        self.mean_L = int(sum(len(self.extract_tokens(seq)) for seq in seqs) / len(seqs))
        self.mean_t = [0.0] * self.m
        for t in t_list:
            for i in range(self.m):
                self.mean_t[i] += t[i]
        self.mean_t = [x / len(t_list) for x in self.mean_t]
        self.trained = True
        
        return history

    def auto_train(self, seqs, auto_mode='gap', max_iters=1000, tol=1e-8, 
                   learning_rate=0.01, continued=False, decay_rate=1.0, 
                   print_every=10):
        """
        Train the model using self-supervised learning with two modes:
          - 'gap': Gap filling (mask current token and predict it from context)
          - 'reg': Auto-regressive (predict next token from previous tokens)
        
        Args:
            seqs: List of input sequences
            auto_mode: Training mode ('gap' or 'reg')
            max_iters: Maximum training iterations
            tol: Convergence tolerance
            learning_rate: Initial learning rate
            continued: Continue training from current parameters
            decay_rate: Learning rate decay factor
            print_every: Print frequency            
            
        Returns:
            list: Training loss history
        """
        # Reinitialize if not continuing
        if not continued:
            self.x_map = {
                tok: [random.uniform(-0.5, 0.5) for _ in range(self.m)]
                for tok in self.tokens
            }
            self.Acoeff = [[random.uniform(-0.1, 0.1) for _ in range(self.L)] 
                           for _ in range(self.m)]
        
        # Prepare training samples
        samples = []  # (sequence, position, target_token)
        for seq in seqs:
            toks = self.extract_tokens(seq)
            if not toks:
                continue
                
            if auto_mode == 'gap':
                # Gap filling: predict current token from context
                for k in range(len(toks)):
                    samples.append((seq, k, toks[k]))
                    
            elif auto_mode == 'reg':
                # Auto-regressive: predict next token from previous
                for k in range(len(toks) - 1):
                    samples.append((seq, k, toks[k + 1]))
        
        if not samples:
            print("Warning: No training samples generated")
            return []
            
        # Initialize training variables
        history = []
        prev_loss = float('inf')
        current_lr = learning_rate
        
        # Main training loop
        for it in range(max_iters):
            total_loss = 0.0
            grad_A = [[0.0] * self.L for _ in range(self.m)]
            grad_x = {tok: [0.0] * self.m for tok in self.tokens}
            
            # Process each sample
            for seq, k, target_tok in samples:
                # Get input token (for 'reg' this is current token at position k)
                context_toks = self.extract_tokens(seq)
                if k >= len(context_toks):
                    continue
                input_tok = context_toks[k]
                
                # Compute Nk descriptor at position k
                j = k % self.L
                scalar = sum(self.Bbasis[j][i] * self.x_map[input_tok][i] 
                          for i in range(self.m))
                Nk = [self.Acoeff[i][j] * scalar for i in range(self.m)]
                
                # Get target vector
                target_vec = self.x_map[target_tok]
                
                # Compute loss and gradients
                loss_val = 0.0
                dNk = [0.0] * self.m
                for i in range(self.m):
                    error = Nk[i] - target_vec[i]
                    loss_val += error * error
                    dNk[i] = 2 * error / self.m  # Gradient of loss w.r.t Nk[i]
                total_loss += loss_val / self.m
                
                # Compute gradients for Acoeff and input token embedding
                for i in range(self.m):
                    grad_A[i][j] += dNk[i] * scalar
                    for l in range(self.m):
                        grad_x[input_tok][l] += (
                            dNk[i] * self.Acoeff[i][j] * self.Bbasis[j][l]
                        )
                
                # Compute gradient for target token embedding
                for i in range(self.m):
                    grad_x[target_tok][i] -= dNk[i]  # Negative gradient
            
            # Update parameters            
            for i in range(self.m):
                for j in range(self.L):
                    self.Acoeff[i][j] -= current_lr * grad_A[i][j]
            for tok in self.tokens:
                for i in range(self.m):
                    self.x_map[tok][i] -= current_lr * grad_x[tok][i]
            
            # Record and print progress
            avg_loss = total_loss / len(samples)
            history.append(avg_loss)
            if it % print_every == 0 or it == max_iters - 1:
                print(f"Iter {it:4d}: loss = {avg_loss:.6e}, lr = {current_lr:.4f}")
            
            # Check convergence
            if prev_loss - avg_loss < tol:
                print(f"Converged after {it+1} iterations")
                break
            prev_loss = avg_loss
            
            # Update learning rate
            current_lr *= decay_rate
        
        # Finalize training
        self.trained = True
        self.mean_L = sum(len(self.extract_tokens(seq)) for seq in seqs) / len(seqs)
        self.mean_t = [0.0] * self.m
        count = 0
        for tok in self.tokens:
            vec = self.x_map[tok]
            for i in range(self.m):
                self.mean_t[i] += vec[i]
            count += 1
        if count > 0:
            self.mean_t = [x / count for x in self.mean_t]
            
        return history

    def predict_t(self, seq):
        """
        Predict target vector for a sequence
        Returns the average of all N(k) vectors in the sequence
        """
        N_list = self.describe(seq)
        if not N_list:
            return [0.0] * self.m            
        # Average all N(k) vectors component-wise
        sum_t = [0.0] * self.m
        for Nk in N_list:
            for d in range(self.m):
                sum_t[d] += Nk[d]
        N = len(N_list)
        return [ti / N for ti in sum_t]

    def reconstruct(self):
        """
        Reconstructs a representative sequence by selecting tokens that minimize 
        the deviation from the mean target vector at each position.
        
        Steps:
        1. Uses the average token count (mean_L) rounded to nearest integer
        2. For each position k, evaluates all possible tokens
        3. Computes Nk vector for token at position k
        4. Selects token minimizing MSE between Nk and mean_t
        5. Returns concatenated token sequence
        
        Requires prior training (sets mean_L and mean_t)
        """
        assert self.trained, "Model must be trained first"
        n_tokens = round(self.mean_L)
        seq_tokens = []
        
        for k in range(n_tokens):
            best_tok = None
            min_error = float('inf')
            j = k % self.L  # Basis matrix index
            
            for tok in self.tokens:
                # Compute Nk = (A[:,j] * scalar) where scalar = B[j]·x
                scalar = sum(self.Bbasis[j][i] * self.x_map[tok][i] for i in range(self.m))
                Nk = [self.Acoeff[i][j] * scalar for i in range(self.m)]
                
                # Calculate MSE between Nk and mean target
                error = 0.0
                for d in range(self.m):
                    diff = Nk[d] - self.mean_t[d]
                    error += diff * diff
                
                # Track minimum error token
                if error < min_error:
                    min_error = error
                    best_tok = tok
            
            seq_tokens.append(best_tok)
        
        return ''.join(seq_tokens)

    def generate(self, L, tau=0.0):
        """
        Generates a sequence of length L using temperature-controlled sampling.
        
        Args:
            L: Desired sequence length
            tau: Temperature parameter (0=deterministic, >0=stochastic)
            
        Process:
        1. Calculates required token count based on rank
        2. For each position k:
            a. Computes Nk for all tokens
            b. Converts deviation to selection score
            c. Samples token (deterministic if tau=0)
        3. Truncates to exact length L
        
        Requires prior training (uses mean_t)
        """
        assert self.trained, "Model must be trained first"
        if tau < 0:
            raise ValueError("Temperature must be non-negative")
        
        num_blocks = (L + self.rank - 1) // self.rank
        generated_tokens = []

        # Generate tokens block by block
        for k in range(num_blocks):
            scores = {}
            j = k % self.L  # Basis matrix index

            # Compute scores for all tokens            
            for tok in self.tokens:
                # Compute Nk vector for token
                scalar = sum(self.Bbasis[j][i] * self.x_map[tok][i] for i in range(self.m))
                Nk = [self.Acoeff[i][j] * scalar for i in range(self.m)]
                
                # Calculate deviation score (negative MSE)
                error = 0.0
                for d in range(self.m):
                    diff = Nk[d] - self.mean_t[d]
                    error += diff * diff
                scores[tok] = -error  # Higher score = better match
            
            # Token selection logic
            if tau == 0:  # Deterministic
                best_tok = max(scores.keys(), key=lambda t: scores[t])
                generated_tokens.append(best_tok)
            else:  # Stochastic
                tokens = list(scores.keys())
                exp_scores = [math.exp(scores[t] / tau) for t in tokens]
                sum_exp = sum(exp_scores)
                # Handle near-zero sum case
                if sum_exp < 1e-18:
                    probs = [1.0 / len(tokens)] * len(tokens)
                else:
                    probs = [es / sum_exp for es in exp_scores]
                chosen_tok = random.choices(tokens, weights=probs, k=1)[0]
                generated_tokens.append(chosen_tok)
        
        # Trim to exact requested length
        full_seq = ''.join(generated_tokens)
        return full_seq[:L]

    # ---- feature extraction ----
    def dd_features(self, seq, t=None):
        """
        Extract feature vector for a sequence in DualDescriptorAB format.
        Features include:
          'd' : Deviation value (scalar)
          'pwc': Flattened PWC coefficients (Acoeff ⊗ Bbasis)
          'cwf': Flattened token embeddings (x_map values)
          'frq': Position-weighted token frequencies
          'pdv': Partial dual variables
          'all': Concatenation of all features
        """
        # Predict target vector if not provided
        tg = t or self.predict_t(seq)
        feats = {}
        # 1. Deviation value (d)
        feats['d'] = [self.deviation([seq], [tg])]
        # 2. PWC coefficients (pwc): Flattened Acoeff
        A_backup = self.Acoeff.copy()        
        self.update_Acoeff([seq], [tg])        
        p_flat = []        
        for i in range(self.m):
            p_flat.extend(self.Acoeff[i])                
        feats['pwc'] = p_flat
        self.Acoeff = A_backup
        # 3. Token embeddings (cwf): Flattened x_map values
        x_backup = self.x_map.copy()        
        self.update_x([seq], [tg])  
        x_flat = []
        for tok in sorted(self.tokens):  # Consistent ordering
            x_flat.extend(self.x_map[tok])
        feats['cwf'] = x_flat
        self.x_map = x_backup
        # 4. Position-weighted frequencies (frq) and partial dual variables (pdv)
        frqs = []; pdvs = []
        toks = self.extract_tokens(seq)
        L = len(toks)
        for tok in sorted(self.tokens):  # Consistent ordering
            for g in range(self.L):      # Basis index
                for i in range(self.m):  # Vector dimension
                    s_frq = 0.0  # Frequency accumulator
                    s_pdv = 0.0  # Dual variable accumulator
                    # Scan all positions in sequence
                    for k, token in enumerate(toks):
                        if token == tok and (k % self.L) == g:
                            basis_val = self.Bbasis[g][i]
                            s_frq += basis_val
                            s_pdv += basis_val * self.x_map[tok][i]
                    # Normalize by sequence length
                    frqs.append(s_frq / L if L > 0 else 0.0)
                    pdvs.append(s_pdv / L if L > 0 else 0.0)        
        # Remove last element because of dependency
        feats['frq'] = frqs[:-1]
        feats['pdv'] = pdvs[:-1]
        # 5. Concatenate all features
        feats['all'] = feats['d'] + feats['pwc'] + feats['cwf'] + feats['frq'] + feats['pdv']        
        return feats

    # ---- show state ----
    def show(self, what=None, first_num=5):
        """
        Display detailed information about the DualDescriptorAB model.
        
        Args:
            what (str): Specific attribute to display. Options include:                
                - None or 'config': Model configuration parameters
                - 'Acoeff': Coefficient matrix (partial view)
                - 'Bbasis': Basis matrix (partial view)
                - 'x_map': Token embeddings (partial view)
                - 'I': Interaction matrix (if exists)
                - 'mean_t': Mean target vector
                - 'mean_L': Mean sequence length
                - 'tokens': Token vocabulary (partial view)
                - 'all': Display all available attributes
            first_num (int): Number of initial elements to display for large attributes
        """        
        # Default attribute views
        if what is None or what == 'config' or what == 'all':
            print("\n=== DualDescriptorAB Model Configuration ===")
            print(f" Vector dimension (m): {self.m}")
            print(f" Basis dimension (L): {self.L}")
            print(f" K-mer rank: {self.rank}")
            print(f" Rank mode: {self.rank_mode}")
            print(f" Description mode: {self.mode}")
            print(f" User step size: {self.step}")
            print(f" Trained status: {self.trained}")
        
        if what == 'Acoeff' or what == 'all':
            print("\n=== Coefficient Matrix (Acoeff) ===")
            print(f" Shape: {len(self.Acoeff)}x{len(self.Acoeff[0])}")
            print(" First few elements:")
            for i in range(min(first_num, len(self.Acoeff))):
                print(f" Row {i}: {self.Acoeff[i][:min(first_num, len(self.Acoeff[i]))]}")
        
        if what == 'Bbasis' or what == 'all':
            print("\n=== Basis Matrix (Bbasis) ===")
            print(f" Shape: {len(self.Bbasis)}x{len(self.Bbasis[0])}")
            print(" First few elements:")
            for i in range(min(first_num, len(self.Bbasis))):
                print(f" Row {i}: {[round(x, 4) for x in self.Bbasis[i][:min(first_num, len(self.Bbasis[i]))]]}")
        
        if what == 'x_map' or what == 'all':
            print("\n=== Token Embeddings (x_map) ===")
            print(f" Vocabulary size: {len(self.tokens)}")
            print(" First few tokens and embeddings:")
            for i, tok in enumerate(list(self.tokens)[:first_num]):
                vec_preview = self.x_map[tok][:min(first_num, self.m)]
                print(f" '{tok}': {[round(x, 4) for x in vec_preview]}")
        
        if (what == 'I' or what == 'all') and hasattr(self, 'I'):
            print("\n=== Interaction Matrix (I) ===")
            print(f" Shape: {len(self.I)}x{len(self.I[0])}")
            print(" First few elements:")
            for i in range(min(first_num, len(self.I))):
                print(f" Row {i}: {[round(x, 4) for x in self.I[i][:min(first_num, len(self.I[i]))]]}")
        
        if (what == 'mean_t' or what == 'all') and hasattr(self, 'mean_t'):
            print("\n=== Mean Target Vector ===")
            preview = self.mean_t[:min(first_num, len(self.mean_t))]
            print(f" Values: {[round(x, 4) for x in preview]}")
            print(f" Full length: {len(self.mean_t)}")
        
        if (what == 'mean_L' or what == 'all') and hasattr(self, 'mean_L'):
            print("\n=== Mean Sequence Length ===")
            print(f" Mean tokens per sequence: {round(self.mean_L, 2)}")
        
        if what == 'tokens' or what == 'all':
            print("\n=== Token Vocabulary ===")
            print(f" Total tokens: {len(self.tokens)}")
            print(f" First {first_num} tokens: {self.tokens[:first_num]}")
        
        # Handle invalid attribute requests
        if what not in [None, 'config', 'Acoeff', 'Bbasis', 'x_map', 'I', 
                       'mean_t', 'mean_L', 'tokens', 'all']:
            print(f"Invalid display option: '{what}'. Valid options are:")
            print("'config', 'Acoeff', 'Bbasis', 'x_map', 'I', 'mean_t', 'mean_L', 'tokens', 'all'")

    def part_train(self, vec_seqs, max_iters=100, tol=1e-6, learning_rate=0.01, 
                   continued=False, auto_mode='reg', decay_rate=1.0, print_every=10):
        """
        Trains the interaction matrix I on vector sequences using gradient descent.
        Supports two training modes:
          - 'gap': Predicts current vector (self-consistency)
          - 'reg': Predicts next vector (auto-regressive)
        
        Parameters:
            vec_seqs (list): List of vector sequences (each sequence is a list of m-dimensional vectors)
            learning_rate (float): Step size for gradient updates
            max_iters (int): Maximum training iterations
            tol (float): Convergence tolerance
            auto_mode (str): Training mode - 'gap' or 'reg'
            continued (bool): Continue training existing I matrix
            decay_rate (float): Learning rate decay factor (1.0 = no decay)
            
        Returns:
            list: Training history (loss values per iteration)
        """
        if auto_mode not in ('gap', 'reg'):
            raise ValueError("auto_mode must be either 'gap' or 'reg'")

        # Initialize I matrix if needed
        if not continued or not hasattr(self, 'I'):
            self.I = [[random.uniform(-0.1, 0.1) for _ in range(self.m)] 
                      for _ in range(self.m)]
        
        # Calculate total training samples
        total_samples = 0
        for seq in vec_seqs:
            if auto_mode == 'gap':
                total_samples += len(seq)  # All vectors are samples
            else:  # 'reg' mode
                total_samples += max(0, len(seq) - 1)  # Vectors except last
                
        if total_samples == 0:
            raise ValueError("No training samples found")
            
        history = []  # Store loss per iteration
        prev_loss = float('inf')
        
        for it in range(max_iters):
            # Initialize gradient matrix for I
            grad_I = [[0.0] * self.m for _ in range(self.m)]  # m x m matrix
            total_loss = 0.0
            
            # Process all vector sequences
            for seq in vec_seqs:
                if not seq:
                    continue
                    
                # Process vectors based on mode
                for k in range(len(seq)):
                    # Skip last vector in 'reg' mode (no next vector)
                    if auto_mode == 'reg' and k == len(seq) - 1:
                        continue
                        
                    current_vec = seq[k]
                    j = k % self.L  # Basis index
                    basis_vec = self.Bbasis[j]  # Get basis vector for position k
                    
                    # Compute N(k) for current vector at position k
                    Nk = [0.0] * self.m
                    for i in range(self.m):
                        for j_dim in range(self.m):
                            # Basis modulation: I[i][j_dim] * current_vec[j_dim] * basis_vec[j_dim]
                            Nk[i] += self.I[i][j_dim] * current_vec[j_dim] * basis_vec[j_dim]
                    
                    # Determine target based on mode
                    if auto_mode == 'gap':
                        target = current_vec  # Self-consistency
                    else:  # 'reg' mode
                        target = seq[k + 1]  # Next vector prediction
                    
                    # Compute loss and gradients
                    for i in range(self.m):
                        error = Nk[i] - target[i]
                        total_loss += error * error
                        
                        # Compute gradients (2 * error from derivative of squared loss)
                        grad_coeff = 2 * error / total_samples
                        
                        # Gradient for I[i][j_dim]
                        for j_dim in range(self.m):
                            grad_I[i][j_dim] += grad_coeff * current_vec[j_dim] * basis_vec[j_dim]
            
            # Update I matrix
            for i in range(self.m):
                for j in range(self.m):
                    self.I[i][j] -= learning_rate * grad_I[i][j]
            
            # Record and print progress
            avg_loss = total_loss / total_samples
            history.append(avg_loss)
            if it % print_every == 0 or it == max_iters - 1:
                mode_display = "Gap" if auto_mode == 'gap' else "Reg"
                print(f"PartTrain({mode_display}) Iter {it:3d}: loss = {avg_loss:.6e}, LR = {learning_rate}")
            
            # Check convergence
            if prev_loss - avg_loss < tol:
                print(f"Converged after {it+1} iterations")
                break
            prev_loss = avg_loss
            
            # Apply learning rate decay
            learning_rate *= decay_rate
        
        # Compute and store mean vector for generation
        total_vectors = 0
        total_vec_sum = [0.0] * self.m
        for seq in vec_seqs:
            for vec in seq:
                total_vectors += 1
                for d in range(self.m):
                    total_vec_sum[d] += vec[d]
        
        self.mean_vector = [v_sum / total_vectors for v_sum in total_vec_sum]
        
        return history

    def part_generate(self, L, tau=0.0, mode='reg'):
        """
        Generates a sequence of vectors using the trained I matrix
        
        Parameters:
            L (int): Length of sequence to generate
            tau (float): Temperature for randomness (0 = deterministic)
            mode (str): Generation mode - 'gap' or 'reg' (must match training)
                
        Returns:
            list: Generated sequence of m-dimensional vectors
        """
        if not hasattr(self, 'I'):
            raise RuntimeError("I matrix not initialized - train first")
            
        if tau < 0:
            raise ValueError("Temperature must be non-negative")
            
        sequence = []
        
        if mode == 'gap':
            # Gap mode: Generate independent reconstructions
            for k in range(L):
                j = k % self.L  # Basis index
                basis_vec = self.Bbasis[j]  # Basis vector for position
                
                # Start with mean vector
                current_vec = self.mean_vector[:]
                
                # Compute reconstruction at position k
                reconstructed_vec = [0.0] * self.m
                for i in range(self.m):
                    for j_dim in range(self.m):
                        reconstructed_vec[i] += self.I[i][j_dim] * current_vec[j_dim] * basis_vec[j_dim]
                
                # Add temperature-controlled noise
                if tau > 0:
                    reconstructed_vec = [reconstructed_vec[i] + random.gauss(0, tau) 
                                        for i in range(self.m)]
                    
                sequence.append(reconstructed_vec)
                
        else:  # 'reg' mode
            # Reg mode: Auto-regressive generation
            current_vec = self.mean_vector[:]  # Start with mean vector
            
            for k in range(L):
                j = k % self.L  # Basis index
                basis_vec = self.Bbasis[j]  # Basis vector for position
                
                # Compute prediction for next vector
                next_vec_pred = [0.0] * self.m
                for i in range(self.m):
                    for j_dim in range(self.m):
                        next_vec_pred[i] += self.I[i][j_dim] * current_vec[j_dim] * basis_vec[j_dim]
                
                # Add temperature-controlled noise
                if tau > 0:
                    next_vec = [next_vec_pred[i] + random.gauss(0, tau) 
                                for i in range(self.m)]
                else:
                    next_vec = next_vec_pred
                    
                sequence.append(next_vec)
                current_vec = next_vec  # Use prediction as next input
                
        return sequence

    def double_train(self, seqs, auto_mode='reg', part_mode='reg', 
                    auto_params=None, part_params=None):
        """
        Two-stage training method: 
          1. First train on character sequences using auto_train (unsupervised)
          2. Then convert sequences to vector sequences using S(l) and train I matrix
        
        Parameters:
            seqs (list): Input character sequences
            auto_mode (str): Training mode for auto_train - 'gap' or 'reg'
            part_mode (str): Training mode for part_train - 'gap' or 'reg'
            auto_params (dict): Parameters for auto_train (max_iters, tol, learning_rate)
            part_params (dict): Parameters for part_train (max_iters, tol, learning_rate)
            
        Returns:
            tuple: (auto_history, part_history) training histories
        """
        # Set default parameters if not provided
        auto_params = auto_params or {'max_iters': 100, 'tol': 1e-6, 'learning_rate': 0.01}
        part_params = part_params or {'max_iters': 100, 'tol': 1e-6, 'learning_rate': 0.01}
        
        # Stage 1: Train character model with auto_train
        print("="*50)
        print("Stage 1: Auto-training on character sequences")
        print("="*50)
        auto_history = self.auto_train(
            seqs, 
            auto_mode=auto_mode,
            max_iters=auto_params['max_iters'],
            tol=auto_params['tol'],
            learning_rate=auto_params['learning_rate']
        )
        
        # Convert sequences to vector sequences using S(l)
        print("\n" + "="*50)
        print("Converting sequences to vector representations")
        print("="*50)
        vec_seqs = []
        for i, seq in enumerate(seqs):
            # Get cumulative S(l) vectors for the sequence
            s_vectors = self.S(seq)
            vec_seqs.append(s_vectors)
            if i < 3:  # Show sample conversion for first 3 sequences
                print(f"Sequence {i+1} (len={len(seq)}) -> {len(s_vectors)} vectors")
                print(f"  First vector: {[round(x, 4) for x in s_vectors[0]]}")
                print(f"  Last vector: {[round(x, 4) for x in s_vectors[-1]]}")
        
        # Stage 2: Train I matrix on vector sequences
        print("\n" + "="*50)
        print("Stage 2: Training I matrix on vector sequences")
        print("="*50)
        part_history = self.part_train(
            vec_seqs,
            max_iters=part_params['max_iters'],
            tol=part_params['tol'],
            learning_rate=part_params['learning_rate'],
            auto_mode=part_mode
        )
        
        return auto_history, part_history

    def double_generate(self, L, tau=0.0, mode='reg'):
        """
        Generate character sequences using a two-stage approach that combines:
          1. Character-level model (auto-trained) for token probabilities
          2. Vector-sequence model (part-trained) for structural coherence
        
        Steps:
          a. Generate initial sequence with character model
          b. Compute cumulative vectors S(l) for initial sequence
          c. Use I-matrix to refine vector sequence
          d. Select tokens that best match the refined vectors
        
        Parameters:
            L (int): Length of sequence to generate
            tau (float): Temperature for stochastic sampling (0=deterministic)
        
        Returns:
            str: Generated character sequence
        """
        # Helper function to compute Nk for a token at position k
        def compute_Nk(k, token):
            j = k % self.L
            scalar = sum(self.Bbasis[j][i] * self.x_map[token][i] for i in range(self.m))
            return [self.Acoeff[i][j] * scalar for i in range(self.m)]
        
        # Stage 1: Generate initial sequence with character model
        init_seq = self.generate(L, tau=tau)
        
        # Stage 2: Compute S(l) vectors for initial sequence
        s_vectors = self.S(init_seq)
        
        # Stage 3: Refine vectors using I-matrix with specified mode
        refined_vectors = self.part_generate(len(s_vectors), mode=mode, tau=tau)
        
        # Stage 4: Reconstruct character sequence using both models
        generated_tokens = []
        current_s = [0.0] * self.m  # Initialize cumulative vector
        
        for k in range(L):
            # Get target vector for current position
            if k < len(refined_vectors):
                target_vec = refined_vectors[k]
            else:
                # If beyond refined vectors, use character model prediction
                target_vec = self.mean_t
            
            # Calculate required N(k) vector: ΔS = S(k) - S(k-1)
            required_nk = [target_vec[i] - current_s[i] for i in range(self.m)]
            
            # Find best matching token
            best_token = None
            min_error = float('inf')
            token_scores = []
            
            for token in self.tokens:
                # Predict N(k) for this token at position k
                predicted_nk = compute_Nk(k, token)
                
                # Calculate matching error
                error = 0.0
                for d in range(self.m):
                    diff = predicted_nk[d] - required_nk[d]
                    error += diff * diff
                
                token_scores.append((token, error))
                
                # Track best token
                if error < min_error:
                    min_error = error
                    best_token = token
            
            # Select token (deterministic or stochastic)
            if tau == 0:
                chosen_token = best_token
            else:
                # Convert errors to probabilities
                tokens, errors = zip(*token_scores)
                weights = [math.exp(-err/tau) for err in errors]
                total_weight = sum(weights)
                if total_weight > 0:
                    probs = [w/total_weight for w in weights]
                    chosen_token = random.choices(tokens, weights=probs, k=1)[0]
                else:
                    chosen_token = random.choice(tokens)
            
            # Update sequence and cumulative vector
            generated_tokens.append(chosen_token)
            actual_nk = compute_Nk(k, chosen_token)
            current_s = [current_s[i] + actual_nk[i] for i in range(self.m)]
        
        return ''.join(generated_tokens)

    def save(self, filename):
        """
        Save the current state of the DualDescriptorAB object to a binary file.
        """
        with open(filename, 'wb') as f:
            pickle.dump(self.__dict__, f)
        print(f"Model saved to {filename}")

    @classmethod
    def load(cls, filename):
        """
        Load a saved DualDescriptorAB model from a binary file.
        """
        with open(filename, 'rb') as f:
            state = pickle.load(f)
        
        # Create new instance without calling __init__
        obj = cls.__new__(cls)
        # Restore saved state
        obj.__dict__.update(state)
        print(f"Model loaded from {filename}")
        return obj

# === Example Usage ===
if __name__ == "__main__":

    from statistics import correlation
    
    random.seed(55)
    charset = ['A','C','G','T']
    dd = DualDescriptorAB(charset, rank=1, vec_dim=2, bas_dim=150, mode='linear', user_step=None)

    # generate 10 sequences of length 100–200 and random targets
    seqs, t_list = [], []
    for _ in range(30):
        L = random.randint(100, 200)
        seq = ''.join(random.choices(charset, k=L))
        seqs.append(seq)
        t_list.append([random.uniform(-1,1) for _ in range(dd.m)])

    dd.train(seqs, t_list, max_iters=500)
    dd.show()

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
        print(f"Prediction correlation: {corr:.4f}")
        corr_sum += corr
    corr_avg = corr_sum / dd.m
    print(f"Average correlation: {corr_avg:.4f}")   

    feats = dd.dd_features(seqs[0])
    print("\nFeature vector length:", len(feats))
    print("First 10 features:", feats['all'][:10])

    # Reconstruct representative sequence
    repr_seq = dd.reconstruct()
    print(f"\nReconstructed representative sequence: {repr_seq}")
    
    # Generate sequences
    print("\nGenerated sequences:")
    seq_det = dd.generate(L=100, tau=0.0)
    print(f"Deterministic (tau=0): {seq_det[:50]}...")

    # === Gradient Descent Training Example ===
    print("\n" + "="*50)
    print("Testing Gradient Descent Training")
    print("="*50)

    # Create new model instance
    dd_grad = DualDescriptorAB(charset, rank=1, vec_dim=2, bas_dim=150, mode='linear', user_step=None)

    # Train using gradient descent
    print("\nStarting gradient descent training...")
    grad_history = dd_grad.grad_train(
        seqs, 
        t_list,
        learning_rate=1.0,
        max_iters=200,
        tol=1e-6,
        print_every=5
    )

    # Evaluate predictions
    pred_t_list_grad = [dd_grad.predict_t(seq) for seq in seqs]

    # Calculate prediction correlations
    corr_sum = 0.0
    for i in range(dd_grad.m):
        actu_t = [t_vec[i] for t_vec in t_list]
        pred_t = [t_vec[i] for t_vec in pred_t_list_grad]
        corr = correlation(actu_t, pred_t)
        print(f"GD Prediction correlation dim-{i}: {corr:.4f}")
        corr_sum += corr
    corr_avg = corr_sum / dd_grad.m
    print(f"GD Average correlation: {corr_avg:.4f}")

    # Generate sequence using trained model
    print("\nGenerated sequence with gradient-trained model:")
    seq_grad = dd_grad.generate(L=50, tau=0.1)
    print(seq_grad)

    # === Auto-Training Example ===
    # Set random seed for reproducible results
    random.seed(42)
    
    # Define character set and model parameters
    charset = ['A','C','G','T']
    vec_dim = 3   # Vector dimension
    bas_dim = 100 # Base matrix dimension

    # Generate training sequences
    print("\n=== Generating Training Sequences ===")
    seqs = []
    for _ in range(30):  # Generate 30 sequences
        L = random.randint(100, 200)
        seq = ''.join(random.choices(charset, k=L))
        seqs.append(seq)
        print(f"Generated sequence {_+1}: length={L}, first 10 chars: {seq[:10]}...")
    
    print("=== Creating Dual Descriptor Model ===")
    dd_auto_gap = DualDescriptorAB(
        charset, 
        rank=1,           # Use 2-mers
        vec_dim=vec_dim,
        bas_dim=bas_dim,
        mode='linear',    # Linear sliding window
        user_step=None
    )    
    
    # Run self-supervised training (Gap Filling mode)
    print("\n=== Starting Gap Filling Training ===")
    gap_history = dd_auto_gap.auto_train(
        seqs,
        auto_mode='gap',     # Gap filling mode
        max_iters=300,       # Maximum iterations
        learning_rate=0.001, # Learning rate
        decay_rate=0.995,    # Learning rate decay
        print_every=1       # Print progress every 20 iterations
    )

    print("=== Creating Dual Descriptor Model ===")
    dd_auto_reg = DualDescriptorAB(
        charset, 
        rank=1,           # Use 2-mers
        vec_dim=vec_dim,
        bas_dim=bas_dim,
        mode='linear',    # Linear sliding window
        user_step=None
    )
    
    # Run self-supervised training (Auto-Regressive mode)
    print("\n=== Starting Auto-Regressive Training ===")
    reg_history = dd_auto_reg.auto_train(
        seqs,
        auto_mode='reg',     # Auto-regressive mode
        max_iters=300,       # Maximum iterations
        learning_rate=0.001, # Learning rate
        decay_rate=0.99,    # Learning rate decay
        print_every=1,      # Print progress every 20 iterations
        continued=True       # Continue training (don't reset parameters)
    )
    
    # Generate new sequences
    print("\n=== Generating New Sequences ===")
    
    # Temperature=0 (deterministic generation)
    seq_det = dd_auto_gap.generate(L=40, tau=0.0)
    print(f"Deterministic Generation (tau=0.0):\n{seq_det}")
    
    # Temperature=0.5 (moderate randomness)
    seq_sto = dd_auto_reg.generate(L=40, tau=0.0)
    print(f"\nStochastic Generation (tau=0.5):\n{seq_sto}")
    
    # 4. Save and load model
    print("\n=== Model Persistence Test ===")
    dd_auto_reg.save("auto_trained_model.pkl")
    
    # Load model
    dd_loaded = DualDescriptorAB.load("auto_trained_model.pkl")
    print("Model loaded successfully. Generating with loaded model:")
    print(dd_loaded.generate(L=20, tau=0.0))
    
    print("\n=== Auto-Training Demo Completed ===")

    print("\n" + "="*50)
    print("Part Train/Generate Example")
    print("="*50)
    
    # Create new model
    dd_part = DualDescriptorAB(charset="", rank=1, vec_dim=2)
    
    # Generate sample vector sequences (2D vectors)
    vec_seqs = []
    for _ in range(5):  # 5 sequences
        seq_len = random.randint(100, 150)
        seq = []
        for _ in range(seq_len):
            # Generate random 2D vector
            vec = [random.uniform(-1, 1), random.uniform(-1, 1)]
            seq.append(vec)
        vec_seqs.append(seq)
    
    # Train in self-consistency (gap) mode
    print("\nTraining in 'gap' mode (self-consistency):")
    gap_history = dd_part.part_train(vec_seqs, max_iters=100, 
                                     learning_rate=0.1, auto_mode='gap')
    
    # Generate new vector sequence
    print("\nGenerated vector sequence (gap mode):")
    gen_seq = dd_part.part_generate(10, mode='gap', tau=0.0)
    for i, vec in enumerate(gen_seq):
        print(f"Vec {i+1}: [{vec[0]:.10f}, {vec[1]:.10f}]")
    
    # Train in auto-regressive (reg) mode
    print("\nTraining in 'reg' mode (next-vector prediction):")
    reg_history = dd_part.part_train(vec_seqs, max_iters=100, 
                                     learning_rate=0.1, auto_mode='reg')
    
    # Generate new vector sequence with randomness
    print("\nGenerated vector sequence with temperature (reg mode):")
    gen_seq = dd_part.part_generate(10, mode='reg', tau=0.1)
    for i, vec in enumerate(gen_seq):
        print(f"Vec {i+1}: [{vec[0]:.10f}, {vec[1]:.10f}]")

    print("\n" + "="*50)
    print("Double Train/Generate Example")
    print("="*50)
    
    # === Double Generation Example ===
    print("\n" + "="*50)
    print("Double Generation Example")
    print("="*50)    

    # Create and train model using double_train
    dd_double = DualDescriptorAB(
        charset=['A','C','G','T'], 
        rank=1, 
        vec_dim=2,
        mode='nonlinear',
        user_step=1
    )

    # Generate sample DNA sequences
    dna_seqs = []
    for _ in range(10):  # 10 sequences
        seq_len = random.randint(100, 200)
        dna_seqs.append(''.join(random.choices(['A','C','G','T'], k=seq_len)))
    
    # Configure training parameters
    auto_config = {
        'max_iters': 50,
        'tol': 1e-6,
        'learning_rate': 0.001
    }
    
    part_config = {
        'max_iters': 50,
        'tol': 1e-20,
        'learning_rate': 0.1
    }

    # Train with double_train (as in previous example)
    auto_hist, part_hist = dd_double.double_train(
        dna_seqs,  # Sample DNA sequences
        auto_mode='reg',
        part_mode='reg',
        auto_params=auto_config,
        part_params=part_config
    )

    # Generate sequences using different methods for comparison
    print("\n1. Character-only generation:")
    char_seq = dd_double.generate(100, tau=0.3)
    print(char_seq)

    print("\n2. Vector-only generation:")
    vec_seq = dd_double.part_generate(10, mode='reg', tau=0.1)
    for i, vec in enumerate(vec_seq):
        print(f"Position {i}: [{vec[0]:.4f}, {vec[1]:.4f}]")

    print("\n3. Double-generation (combined models):")
    double_seq = dd_double.double_generate(100, tau=0.2)
    print(double_seq)

    # Extract features from different generation methods
    char_features = dd_double.dd_features(char_seq)['all'][:5]
    double_features = dd_double.dd_features(double_seq)['all'][:5]
    print("\nFeature comparison (first 5 values):")
    print(f"Char-only: {[round(f, 4) for f in char_features]}")
    print(f"Double:    {[round(f, 4) for f in double_features]}")
