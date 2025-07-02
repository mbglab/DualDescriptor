# Copyright (C) Bin-Guang Ma (mbg@mail.hzau.edu.cn). All rights reserved.
# Simplified Dual Descriptor Vector class (P Matrix form)
# Author: Bin-Guang Ma; Date: 2025-6-4

import math
import random
import itertools
import pickle

class DualDescriptorPM:
    """
    Vector Dual Descriptor with:
      - matrix P ∈ R^{m×m} of basis coefficients
      - M: k-mer token → R^m character embedding
      - indexed periods: period[i,j] = i*m + j + 2
      - basis function phi_{i,j}(k) = cos(2π * k / period[i,j])
      - supports 'linear' or 'nonlinear' (step-by-rank) k-mer extraction    
    """
    def __init__(self, charset, rank=1, rank_mode='drop', vec_dim=2, mode='linear', user_step=None):
        self.charset = list(charset)
        self.rank = rank    # r-per/k-mer length
        self.rank_mode = rank_mode # 'pad' or 'drop'
        self.m = vec_dim    # embedding dimension
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

        # character/token embeddings M[token] ∈ R^m
        self.M = {tok: [random.uniform(-0.5, 0.5) for _ in range(self.m)]
                      for tok in self.tokens}

        # position-weight matrix P[i][j]
        self.P = [[random.uniform(-0.1, 0.1) for _ in range(self.m)]
                   for _ in range(self.m)]

        # precompute indexed periods[i][j]
        self.periods = [[i*self.m + j + 2 for j in range(self.m)]
                          for i in range(self.m)]

    def _invert(self, A):
        n = len(A)
        M = [row[:] + [1.0 if i==j else 0.0 for j in range(n)]
             for i, row in enumerate(A)]
        for i in range(n):
            piv = M[i][i]
            if abs(piv) < 1e-12: continue
            M[i] = [x/piv for x in M[i]]
            for r in range(n):
                if r==i: continue
                fac = M[r][i]
                M[r] = [M[r][c] - fac*M[i][c] for c in range(2*n)]
        return [row[n:] for row in M]

    def _mat_vec(self, M, v):
        return [sum(M[i][j]*v[j] for j in range(len(v))) for i in range(len(M))]

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

    def compute_Nk(self, k, tok):
        """Compute N(k) vector for a single position and token"""
        x = self.M[tok]
        Nk = [0.0] * self.m
        for i in range(self.m):
            for j in range(self.m):
                period = self.periods[i][j]
                phi = math.cos(2 * math.pi * k / period)
                Nk[i] += self.P[i][j] * x[j] * phi
        return Nk

    def describe(self, seq):
        """Compute N(k) vectors for each k-mer in sequence"""
        toks = self.extract_tokens(seq)
        N = []
        for k, tok in enumerate(toks):
            N.append(self.compute_Nk(k, tok))
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

    def D(self, seqs, t_list):
        """
        Compute mean squared deviation D across sequences:
        D = average over all positions of (N(k)-t_seq)^2
        """
        tot=0.0; cnt=0
        for seq, t in zip(seqs, t_list):
            for Nk in self.describe(seq):
                for i in range(self.m):
                    d = Nk[i] - t[i]
                    tot += d*d
                cnt += 1
        return tot/cnt if cnt else 0.0

    def update_P(self, seqs, t_list):
        """
        Closed-form update of P matrix by solving independent systems for each i-dimension.        
        Steps:
        1. For each i in range(m):
            a. Build linear system U_i (size: m x m) and V_i (size: m)
            b. Solve P_i = U_i^{-1} V_i
            c. Update P[i][j] for all j
        Mathematical formulation:
            min_{P_i} Σ_seq Σ_k (N_k[i] - t_i)^2
            Where:
                N_k[i] = Σ_j P[i][j] * x_j * φ_{ij}(k)            
        """
        # Iterate over each i-dimension independently
        for i in range(self.m):
            dim = self.m  # Size of subsystem for this i
            U_i = [[0.0] * dim for _ in range(dim)]  # m x m matrix
            V_i = [0.0] * dim  # m vector
            
            # Accumulate data from all sequences and positions
            for seq, t in zip(seqs, t_list):
                toks = self.extract_tokens(seq)
                for k, tok in enumerate(toks):
                    x = self.M[tok]
                    # Build indices and basis values for current (i,k)
                    for j in range(self.m):
                        idx1 = j  # Linear index in subsystem
                        period = self.periods[i][j]
                        phi = math.cos(2 * math.pi * k / period)
                        a = x[j] * phi
                        
                        # Right-hand side vector component
                        V_i[idx1] += t[i] * a  # Note: only t[i] used
                        
                        # Left-hand side matrix components
                        for j2 in range(self.m):
                            idx2 = j2
                            period2 = self.periods[i][j2]  # Same i!
                            phi2 = math.cos(2 * math.pi * k / period2)
                            b = x[j2] * phi2
                            U_i[idx1][idx2] += a * b
            
            # Solve subsystem for current i
            try:
                U_inv = self._invert(U_i)
                sol = [sum(U_inv[r][c] * V_i[c] for c in range(dim)) for r in range(dim)]
            except Exception as e:
                print(f"Warning: inversion failed for i={i}, using identity. Error: {str(e)}")
                sol = V_i[:]  # Fallback to identity solution
            
            # Update P matrix for current i
            for j in range(self.m):
                self.P[i][j] = sol[j]

    def update_M(self, seqs, t_list):
        """
        Closed-form update of token embeddings (M).        
        Mathematical formulation for token c:
            min_{x_c} Σ_seq Σ_{k: token_k = c} Σ_i (N_k[i] - t_i)^2
            Where:
                N_k[i] = Σ_j P[i][j] * x_j * φ_{ij}(k)            
            This yields linear system for x_c:
                M * x_c = R
            Where:
                M[d1][d2] = Σ_seq Σ_{k: token_k = c} Σ_i [P[i][d1] * φ_{i,d1}(k) * P[i][d2] * φ_{i,d2}(k)]
                R[d1] = Σ_seq Σ_{k: token_k = c} Σ_i [t_i * P[i][d1] * φ_{i,d1}(k)]
        """
        # Precompute basis products to avoid redundant calculations
        basis_cache = {}
        for i in range(self.m):
            for j in range(self.m):
                period = self.periods[i][j]
                basis_cache[(i, j)] = period
        
        # Update each token independently
        for c in self.tokens:
            M = [[0.0] * self.m for _ in range(self.m)]  # m x m matrix
            R = [0.0] * self.m  # m-dim vector
            token_count = 0  # Track occurrences of this token
            
            # Accumulate data from all sequences and positions
            for seq, t in zip(seqs, t_list):
                toks = self.extract_tokens(seq)
                for k, tok in enumerate(toks):
                    if tok != c:
                        continue
                        
                    token_count += 1
                    # Precompute basis-parameter products
                    A = [[0.0]*self.m for _ in range(self.m)]
                    for i in range(self.m):
                        for j in range(self.m):
                            period = basis_cache[(i, j)]
                            phi = math.cos(2 * math.pi * k / period)
                            A[i][j] = self.P[i][j] * phi
                    
                    # Update M matrix (Hessian approximation)
                    for d1 in range(self.m):
                        for d2 in range(self.m):
                            # M[d1][d2] += Σ_i A[i][d1] * A[i][d2]
                            s = 0.0
                            for i in range(self.m):
                                s += A[i][d1] * A[i][d2]
                            M[d1][d2] += s
                    
                    # Update R vector
                    for d1 in range(self.m):
                        s = 0.0
                        for i in range(self.m):
                            s += t[i] * A[i][d1]
                        R[d1] += s
            
            # Only solve if token appeared in data
            if token_count > 0:
                try:
                    M_inv = self._invert(M)
                    self.M[c] = self._mat_vec(M_inv, R)
                except Exception as e:
                    print(f"Warning: M update failed for token '{c}', keeping previous value. Error: {str(e)}") 

    def train(self, seqs, t_list, max_iters=10, tol=1e-8, print_every=10):
        """Alternate closed-form updates with convergence tracking"""        
        D_prev = float('inf')
        history = []
        for it in range(max_iters):
            self.update_P(seqs, t_list)
            self.update_M(seqs, t_list)
            D = self.D(seqs, t_list)
            history.append(D)
            if it % print_every == 0 or it == max_iters - 1:
                print(f"Iter {it:2d}: D = {D:.6e}")
            if D >= D_prev - tol:
                print("Converged.")
                break
            D_prev = D            
        # Calculate statistics for reconstruction/generation
        total_token_count = 0
        total_t = [0.0] * self.m        
        for seq, t_vec in zip(seqs, t_list):
            tokens = self.extract_tokens(seq)
            total_token_count += len(tokens)
            for d in range(self.m):
                total_t[d] += t_vec[d]                
        self.mean_token_count = total_token_count / len(seqs)
        self.mean_t = [total_t[d] / len(seqs) for d in range(self.m)]
        self.trained = True        
        return history

    def grad_train(self, seqs, t_list, max_iters=1000, tol=1e-8, learning_rate=0.01, continued=False, decay_rate=1.0, print_every=10):
        """
        Train the model using gradient descent with alternating updates for P and M.
        
        Steps:
        1. For each iteration:
            a. Compute gradients for P and M
            b. Update P matrix using gradients
            c. Update M embeddings using gradients
        2. Track convergence using mean squared deviation (D)
        3. Stop when improvement falls below tolerance
        4. Apply learning rate decay at end of each iteration
        
        Parameters:
            seqs (list): Input sequences
            t_list (list): Target vectors for each sequence
            learning_rate (float): Step size for gradient descent
            max_iters (int): Maximum training iterations
            tol (float): Convergence tolerance
            decay_rate (float): Learning rate decay factor (1.0 = no decay)
        
        Returns:
            list: Training history (D values per iteration)
        """
        if not continued:
            # Use small random values to initialize CWM and PWC.
            self.M = {tok: [random.uniform(-0.5, 0.5) for _ in range(self.m)]
                      for tok in self.tokens}
            self.P = [[random.uniform(-0.1, 0.1) for _ in range(self.m)]
                  for _ in range(self.m)]
            
        history = []
        D_prev = float('inf')
        total_positions = sum(len(self.extract_tokens(seq)) for seq in seqs)  # Total tokens in dataset
        
        for it in range(max_iters):
            # Initialize gradients
            grad_P = [[0.0 for _ in range(self.m)] for _ in range(self.m)]
            grad_x = {tok: [0.0] * self.m for tok in self.tokens}
            
            # Accumulate gradients over all sequences and positions
            for seq, t_vec in zip(seqs, t_list):
                toks = self.extract_tokens(seq)
                for pos, token in enumerate(toks):
                    x = self.M[token]
                    Nk = [0.0] * self.m
                    
                    # Precompute basis functions and Nk
                    phi_vals = {}
                    for i in range(self.m):
                        for j in range(self.m):
                            period = self.periods[i][j]
                            phi = math.cos(2 * math.pi * pos / period)
                            phi_vals[(i, j)] = phi
                            Nk[i] += self.P[i][j] * x[j] * phi
                    
                    # Compute gradients
                    for i in range(self.m):
                        residual = 2 * (Nk[i] - t_vec[i]) / total_positions
                        for j in range(self.m):
                            phi = phi_vals[(i, j)]
                            # Gradient for P[i][j]
                            grad_P[i][j] += residual * x[j] * phi
                            # Gradient for M[token][j]
                            grad_x[token][j] += residual * self.P[i][j] * phi
            
            # Update P matrix
            for i in range(self.m):
                for j in range(self.m):
                    self.P[i][j] -= learning_rate * grad_P[i][j]
            
            # Update M embeddings
            for token in self.tokens:
                for j in range(self.m):
                    self.M[token][j] -= learning_rate * grad_x[token][j]
            
            # Calculate current loss and print progress
            current_D = self.D(seqs, t_list)
            history.append(current_D)
            if it % print_every == 0 or it == max_iters - 1:
                print(f"GD Iter {it:2d}: D = {current_D:.6e}, LR = {learning_rate}")
            
            # Check convergence
            if D_prev - current_D < tol:
                print(f"Converged after {it+1} iterations.")
                break
            D_prev = current_D
            
            # Apply learning rate decay
            learning_rate *= decay_rate
        
        # Calculate statistics for reconstruction/generation
        total_token_count = 0
        total_t = [0.0] * self.m
        for seq, t_vec in zip(seqs, t_list):
            tokens = self.extract_tokens(seq)
            total_token_count += len(tokens)
            for d in range(self.m):
                total_t[d] += t_vec[d]
        self.mean_token_count = total_token_count / len(seqs)
        self.mean_t = [total_t[d] / len(seqs) for d in range(self.m)]
        self.trained = True
        return history

    def auto_train(self, seqs, max_iters=100, tol=1e-6, learning_rate=0.01, continued=False, auto_mode='reg', decay_rate=1.0, print_every=10):
        """
        Self-training method using gradient descent with two modes:
          - 'gap': Predicts current token's embedding (self-consistency)
          - 'reg': Predicts next token's embedding (auto-regressive)
        
        Parameters:
            seqs (list): Input sequences for training
            learning_rate (float): Step size for gradient updates
            max_iters (int): Maximum training iterations
            tol (float): Convergence tolerance
            auto_mode (str): Training mode - 'gap' or 'reg' (default: 'reg')
            continued (bool): Continue training existing I tensor
            decay_rate (float): Learning rate decay factor (1.0 = no decay)
            
        Returns:
            list: Training history (loss values per iteration)
        """
        # Validate auto_mode parameter
        if auto_mode not in ('gap', 'reg'):
            raise ValueError("auto_mode must be either 'gap' or 'reg'")

        if not continued:
            # Use small random values to initialize CWM and PWC.
            self.M = {tok: [random.uniform(-0.5, 0.5) for _ in range(self.m)]
                      for tok in self.tokens}
            self.P = [[random.uniform(-0.1, 0.1) for _ in range(self.m)]
                  for _ in range(self.m)]
        
        # Calculate total training samples
        total_samples = 0
        for seq in seqs:
            tokens = self.extract_tokens(seq)
            if auto_mode == 'gap':
                total_samples += len(tokens)  # All tokens are samples
            else:  # 'reg' mode
                total_samples += max(0, len(tokens) - 1)  # Tokens except last
            
        if total_samples == 0:
            raise ValueError("No training samples found")
            
        history = []  # Store loss per iteration
        prev_loss = float('inf')
        
        for it in range(max_iters):
            # Initialize gradients
            grad_P = [[0.0] * self.m for _ in range(self.m)]  # m x m matrix
            grad_x = {token: [0.0] * self.m for token in self.tokens}
            total_loss = 0.0
            
            # Process all sequences
            for seq in seqs:
                tokens = self.extract_tokens(seq)
                if not tokens:
                    continue
                    
                # Process tokens based on mode
                for k in range(len(tokens)):
                    # Skip last token in 'reg' mode as it has no next token
                    if auto_mode == 'reg' and k == len(tokens) - 1:
                        continue
                        
                    current_token = tokens[k]
                    x_current = self.M[current_token]
                    
                    # Compute N(k) for current token at position k
                    Nk = [0.0] * self.m
                    phi_vals = {}
                    for i in range(self.m):
                        for j in range(self.m):
                            period = self.periods[i][j]
                            phi = math.cos(2 * math.pi * k / period)
                            phi_vals[(i, j)] = phi
                            Nk[i] += self.P[i][j] * x_current[j] * phi
                    
                    # Determine target based on mode
                    if auto_mode == 'gap':
                        # Target is current token's embedding (self-consistency)
                        target = x_current
                    else:  # 'reg' mode
                        # Target is next token's embedding (auto-regressive)
                        next_token = tokens[k + 1]
                        target = self.M[next_token]
                    
                    # Compute loss and gradients
                    for i in range(self.m):
                        error = Nk[i] - target[i]
                        total_loss += error * error
                        
                        # Compute gradients (2 * error from derivative of squared loss)
                        grad_coeff = 2 * error / total_samples
                        
                        # Gradient for P[i][j]
                        for j in range(self.m):
                            phi = phi_vals[(i, j)]
                            grad_P[i][j] += grad_coeff * x_current[j] * phi
                            
                        # Gradient for current token's embedding
                        for j in range(self.m):
                            phi = phi_vals[(i, j)]
                            grad_x[current_token][j] += grad_coeff * self.P[i][j] * phi
                        
                        # Additional gradient for next token in 'reg' mode
                        if auto_mode == 'reg':
                            # Gradient for next token's embedding (target)
                            grad_x[next_token][i] -= grad_coeff
                    # Additional gradient for current token in 'gap' mode
                    if auto_mode == 'gap':
                        for i in range(self.m):
                            error = Nk[i] - target[i]
                            grad_coeff = 2 * error / total_samples
                            # Gradient for current token as target
                            grad_x[current_token][i] -= grad_coeff
            
            # Update parameters
            for i in range(self.m):
                for j in range(self.m):
                    self.P[i][j] -= learning_rate * grad_P[i][j]
            
            for token in self.tokens:
                for j in range(self.m):
                    self.M[token][j] -= learning_rate * grad_x[token][j]
            
            # Record and print progress
            avg_loss = total_loss / total_samples
            history.append(avg_loss)
            if it % print_every == 0 or it == max_iters - 1:
                mode_display = "Gap" if auto_mode == 'gap' else "Reg"
                print(f"AutoTrain({mode_display}) Iter {it:3d}: loss = {avg_loss:.6f}, LR = {learning_rate}")
            
            # Check convergence
            if prev_loss - avg_loss < tol:
                print(f"Converged after {it+1} iterations")
                break
            prev_loss = avg_loss
            
            # Apply learning rate decay
            learning_rate *= decay_rate
        
        # Compute and store statistics for reconstruction/generation
        total_t = [0.0] * self.m
        total_token_count = 0
        for seq in seqs:
            tokens = self.extract_tokens(seq)
            total_token_count += len(tokens)
            for token in tokens:
                x_val = self.M[token]
                for d in range(self.m):
                    total_t[d] += x_val[d]
        
        self.mean_token_count = total_token_count / len(seqs)
        self.mean_t = [t_val / total_token_count for t_val in total_t]
        self.trained = True
        
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
        """Reconstruct representative sequence by minimizing error"""
        assert self.trained, "Model must be trained first"
        # Round average token count to nearest integer
        n_tokens = round(self.mean_token_count)
        seq_tokens = []        
        for k in range(n_tokens):
            best_tok = None
            min_error = float('inf')            
            for tok in self.tokens:
                Nk = self.compute_Nk(k, tok)
                error = 0.0
                for d in range(self.m):
                    diff = Nk[d] - self.mean_t[d]
                    error += diff * diff                    
                if error < min_error:
                    min_error = error
                    best_tok = tok                    
            seq_tokens.append(best_tok)            
        return ''.join(seq_tokens)

    def generate(self, L, tau=0.0):
        """Generate sequence of length L with temperature-controlled randomness"""
        assert self.trained, "Model must be trained first"
        if tau < 0:
            raise ValueError("Temperature must be non-negative")            
        num_blocks = (L + self.rank - 1) // self.rank
        generated_tokens = []        
        for k in range(num_blocks):
            scores = {}
            for tok in self.tokens:
                Nk = self.compute_Nk(k, tok)
                error = 0.0
                for d in range(self.m):
                    diff = Nk[d] - self.mean_t[d]
                    error += diff * diff
                scores[tok] = -error  # Convert to score (higher = better)
                
            if tau == 0:  # Deterministic selection
                best_tok = max(scores.keys(), key=lambda t: scores[t])
                generated_tokens.append(best_tok)
            else:  # Stochastic selection
                tokens = list(scores.keys())
                exp_scores = [math.exp(scores[t] / tau) for t in tokens]
                sum_exp = sum(exp_scores)
                probs = [es / sum_exp for es in exp_scores]
                chosen_tok = random.choices(tokens, weights=probs, k=1)[0]
                generated_tokens.append(chosen_tok)                
        full_seq = ''.join(generated_tokens)
        return full_seq[:L]    

    def dd_features(self, seq, t=None):
        """
        Extract feature vector for a sequence:
        { 'd' : [deviation value],
         'pwc': [PWC coefficients], 
         'cwf': [CWF values for all tokens sorted],
         'frq': [Position-weighted frequency for each token],
         'pdv': [Partial Dual Variable for each token],         
         'all': [concatenate the above]}
        """
        tg = t or self.predict_t(seq)
        feats = {}
        # Deviation value 
        feats['d'] = [self.D([seq], [tg])]
        # Flatten P matrix (PWC)
        P_backup = [row[:] for row in self.P]  # Shallow copy
        self.update_P([seq], [tg])        
        p_flat = []
        for i in range(self.m):
            p_flat.extend(self.P[i])
        feats['pwc'] = p_flat
        self.P = P_backup                
        # Flatten M (CWF)
        x_backup = self.M.copy()        
        self.update_M([seq], [tg])  
        x_flat = []
        for tok in self.tokens:
            x_flat.extend(self.M[tok])
        feats['cwf'] = x_flat
        self.M = x_backup              
        # Basis-weighted frequencies and partial dual variables
        frqs = []; pdvs = []
        toks = self.extract_tokens(seq)
        L = len(toks)
        for tok in self.tokens:
            for i in range(self.m):
                for j in range(self.m):
                    s = 0.0; v =0.0
                    for k, ch in enumerate(toks):
                        if ch == tok:
                            period = self.periods[i][j]
                            phi = math.cos(2 * math.pi * k / period)
                            s += phi
                            v += phi * self.M[tok][j]
                    frqs.append(s/L if L>0 else 0.0) 
                    pdvs.append(v/L if L>0 else 0.0)
        feats['frq'] = frqs
        feats['pdv'] = pdvs
        feats['all'] = feats['d'] + feats['pwc'] + feats['cwf'] + feats['frq'] + feats['pdv']        
        return feats

    def show(self, what='status', first_num=5):
        """
        Display detailed information about the model state, controlled by the 'what' parameter.
        
        Parameters:
            what (str): Specifies which part of the model to display. Options:
                'status' : Basic configuration (default)
                'tokens' : Token vocabulary
                'M'      : Character embedding matrix
                'P'      : Basis coefficient matrix
                'periods': Positional period matrix
                'stats'  : Training statistics
                'all'    : All available information
            first_num (int): Number of initial items to display for long lists (default: 5)
        """
        # Always display basic status information
        print("="*50)
        print("DualDescriptorPM Model Status")
        print("="*50)
        print(f"• m (vec_dim)    : {self.m}")
        print(f"• rank (kmer_len): {self.rank}")
        print(f"• mode           : {self.mode}")
        print(f"• rank_mode      : {self.rank_mode}")
        print(f"• step           : {self.step}")
        print(f"• trained        : {self.trained}")
        print(f"• charset        : {self.charset}")
        print(f"• # tokens       : {len(self.tokens)}")
        
        # Display token vocabulary
        if what in ('tokens', 'all'):
            print("\n" + "-"*20 + " Token Vocabulary " + "-"*20)
            print(f"Showing first {min(first_num, len(self.tokens))} of {len(self.tokens)} tokens:")
            for i, tok in enumerate(self.tokens[:first_num]):
                print(f"  {i+1:2d}. '{tok}'")
        
        # Display character embeddings
        if what in ('M', 'all'):
            print("\n" + "-"*20 + " Character Embeddings (M) " + "-"*20)
            print(f"Showing first {min(first_num, len(self.tokens))} of {len(self.tokens)} embeddings:")
            for i, tok in enumerate(self.tokens[:first_num]):
                emb = [f"{x:.6f}" for x in self.M[tok]]
                print(f"  '{tok}': [{', '.join(emb)}]")
        
        # Display basis coefficient matrix
        if what in ('P', 'all'):
            print("\n" + "-"*20 + " Basis Coefficient Matrix (P) " + "-"*20)
            print(f"Matrix dimension: {self.m}×{self.m}")
            print(f"First {min(first_num, self.m)} rows:")
            for i, row in enumerate(self.P[:first_num]):
                vals = [f"{x:.6f}" for x in row[:first_num]]
                print(f"  Row {i}: [{', '.join(vals)}" + 
                      (", ...]" if len(row) > first_num else "]"))
        
        # Display period matrix
        if what in ('periods', 'all'):
            print("\n" + "-"*20 + " Period Matrix " + "-"*20)
            print(f"Matrix dimension: {self.m}×{self.m}")
            print("First 5×5 elements:")
            for i, row in enumerate(self.periods[:5]):
                print(f"  Row {i}: {row[:5]}" + 
                      (", ..." if len(row) > 5 else ""))
        
        # Display training statistics
        if what in ('stats', 'all') and self.trained:
            print("\n" + "-"*20 + " Training Statistics " + "-"*20)
            print(f"• Avg tokens/seq : {self.mean_token_count:.2f}")
            print("• Mean target (t): [" + 
                  ", ".join(f"{x:.6f}" for x in self.mean_t) + "]")
        
        # Footer for complete display
        if what == 'all':
            print("\n" + "="*50)
            print(f"Displayed first {first_num} items for long lists")
            print("Use show('specific_property') for focused view")

    def part_train(self, vec_seqs, max_iters=100, tol=1e-6, learning_rate=0.01, 
               continued=False, auto_mode='reg', decay_rate=1.0, print_every=10):
        """
        Train the I matrix on vector sequences using gradient descent.
        Supports two modes:
          - 'gap': Predicts current vector (self-consistency)
          - 'reg': Predicts next vector (auto-regressive)
        
        Parameters:
            vec_seqs (list): List of vector sequences (each sequence is list of m-dim vectors)
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
                    
                    # Compute N(k) for current vector at position k
                    Nk = [0.0] * self.m
                    phi_vals = {}
                    for i in range(self.m):
                        for j in range(self.m):
                            period = self.periods[i][j]
                            phi = math.cos(2 * math.pi * k / period)
                            phi_vals[(i, j)] = phi
                            Nk[i] += self.I[i][j] * current_vec[j] * phi
                    
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
                        
                        # Gradient for I[i][j]
                        for j in range(self.m):
                            phi = phi_vals[(i, j)]
                            grad_I[i][j] += grad_coeff * current_vec[j] * phi
            
            # Update I matrix
            for i in range(self.m):
                for j in range(self.m):
                    self.I[i][j] -= learning_rate * grad_I[i][j]
            
            # Record and print progress
            avg_loss = total_loss / total_samples
            history.append(avg_loss)
            if it % print_every == 0 or it == max_iters - 1:
                mode_display = "Gap" if auto_mode == 'gap' else "Reg"
                print(f"PartTrain({mode_display}) Iter {it:3d}: loss = {avg_loss:.6f}, LR = {learning_rate}")
            
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
        Generate a sequence of vectors using the trained I matrix
        
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
            
        if mode == 'gap':
            # Gap mode: Generate independent reconstructions at each position
            sequence = []
            for k in range(L):
                # Start with mean vector
                current_vec = self.mean_vector[:]
                
                # Compute reconstruction at position k
                reconstructed_vec = [0.0] * self.m
                for i in range(self.m):
                    for j in range(self.m):
                        period = self.periods[i][j]
                        phi = math.cos(2 * math.pi * k / period)
                        reconstructed_vec[i] += self.I[i][j] * current_vec[j] * phi
                
                # Add temperature-controlled noise
                if tau > 0:
                    reconstructed_vec = [reconstructed_vec[i] + random.gauss(0, tau) 
                                        for i in range(self.m)]
                    
                sequence.append(reconstructed_vec)
            return sequence
            
        else:  # 'reg' mode
            # Reg mode: Auto-regressive generation
            sequence = []
            current_vec = self.mean_vector[:]  # Start with mean vector
            
            for k in range(L):
                # Compute prediction for next vector
                next_vec_pred = [0.0] * self.m
                for i in range(self.m):
                    for j in range(self.m):
                        period = self.periods[i][j]
                        phi = math.cos(2 * math.pi * k / period)
                        next_vec_pred[i] += self.I[i][j] * current_vec[j] * phi
                
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
                predicted_nk = self.compute_Nk(k, token)
                
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
            actual_nk = self.compute_Nk(k, chosen_token)
            current_s = [current_s[i] + actual_nk[i] for i in range(self.m)]
        
        return ''.join(generated_tokens)

    def save(self, filename):
        """
        Save the current state of the DualDescriptorScalar object to a binary file.
        """
        with open(filename, 'wb') as f:
            pickle.dump(self.__dict__, f)
        print(f"Model saved to {filename}")

    @classmethod
    def load(cls, filename):
        """
        Load a saved DualDescriptorScalar model from a binary file.
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
if __name__=="__main__":

    from statistics import correlation
    
    #random.seed(0)
    charset = ['A','C','G','T']
    # Create matrix descriptor for vector targets
    dd = DualDescriptorPM(charset, rank=3, vec_dim=10, mode='nonlinear', user_step=2)

    # Generate 10 sequences with random vector targets
    seqs, t_list = [], []
    for _ in range(10):
        L = random.randint(200, 300)
        seq = ''.join(random.choices(charset, k=L))
        seqs.append(seq)
        # Create random vector target
        t_list.append([random.uniform(-1.0, 1.0) for _ in range(dd.m)])

    # Train the model
    dd.train(seqs, t_list, max_iters=100)
    
    # Show model status
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
  
    # Reconstruct representative sequence
    repr_seq = dd.reconstruct()
    print(f"\nRepresentative sequence (len={len(repr_seq)}): {repr_seq[:50]}...")
    
    # Generate new sequences
    seq_det = dd.generate(L=100, tau=0.0)
    seq_rand = dd.generate(L=100, tau=0.5)
    print("\nDeterministic generation:", seq_det[:50] + "...")
    print("Stochastic generation (tau=0.5):", seq_rand[:50] + "...")
    
    # Extract features for first sequence
    features = dd.dd_features(seqs[0])['all']
    print(f"\nFeature vector length: {len(features)}")
    print(f"First 10 features: {[round(x, 4) for x in features[:10]]}")

    # Initialize model
    dd_grad = DualDescriptorPM(charset, rank=4, vec_dim=2, mode='nonlinear', user_step=4)
    
    # Gradient Descent Training
    print("\n" + "="*50)
    print("Training with Gradient Descent")
    print("="*50)
    grad_history = dd_grad.grad_train(seqs, t_list, max_iters=500, tol=1e-6, learning_rate=0.5, continued=False)

    # Evaluation
    pred_t_list = [dd_grad.predict_t(seq) for seq in seqs]
    print("\nEvaluation Results:")
    for i in range(dd_grad.m):
        actu_t = [t_vec[i] for t_vec in t_list]
        pred_t = [t_vec[i] for t_vec in pred_t_list]
        corr = correlation(actu_t, pred_t)
        print(f"Dimension {i} correlation: {corr:.4f}")
    
    # Reconstruction and Generation
    print("\nReconstructed Sequence (first 50 bases):")
    print(dd_grad.reconstruct()[:50])
    
    print("\nGenerated Sequence (tau=0.3):")
    print(dd_grad.generate(100, tau=0.3)[:50])
    
# === Combined Auto-Training Example ===
    print("\n" + "="*50)
    print("Combined Auto-Training Example")
    print("="*50)
    
    # Create new models
    dd_gap = DualDescriptorPM(charset, rank=3, vec_dim=2, mode='nonlinear', user_step=2)
    dd_reg = DualDescriptorPM(charset, rank=3, vec_dim=2, mode='nonlinear', user_step=2)
    
    # Generate sample sequences
    auto_seqs = []
    for _ in range(10):
        L = random.randint(200, 300)
        auto_seqs.append(''.join(random.choices(charset, k=L)))
    
    # Perform self-consistency training (gap mode)
    print("\nTraining in 'gap' mode (self-consistency):")
    gap_history = dd_gap.auto_train(auto_seqs, max_iters=50, tol=1e-8, learning_rate=0.5, continued=False, auto_mode='gap')
    
    # Perform auto-regressive training (reg mode)
    print("\nTraining in 'reg' mode (next-token prediction):")
    reg_history = dd_reg.auto_train(auto_seqs, max_iters=50, tol=1e-8, learning_rate=0.5, continued=False, auto_mode='reg')
     
    # Generate sequences from both models
    print("\nGenerated sequences from 'gap' model:")
    for i in range(2):
        gen_seq = dd_gap.generate(100, tau=0.2)
        print(f"Sequence {i+1}: {gen_seq[:50]}...")
    
    print("\nGenerated sequences from 'reg' model:")
    for i in range(2):
        gen_seq = dd_reg.generate(100, tau=0.3)
        print(f"Sequence {i+1}: {gen_seq[:50]}...")

    # Save trained model
    dd.save("trained_model.pkl")
    
    # Load saved model
    dd_loaded = DualDescriptorPM.load("trained_model.pkl")
    
    # Verify predictions work
    t_pred = dd_loaded.predict_t(seqs[0])
    print("Predicted t using loaded model: ", t_pred)
    
    # Continue training
    dd_loaded.grad_train(seqs, t_list, max_iters=50)
    
    # Test other functionality
    features = dd_loaded.dd_features(seqs[0])
    print("Feature vector length:", len(features))
    
    repr_seq = dd_loaded.reconstruct()
    print(f"Representative sequence: {repr_seq}")
    
    gen_seq = dd_loaded.generate(L=100, tau=0.5)
    print(f"Generated sequence: {gen_seq}")

    print("\n" + "="*50)
    print("Part Train/Generate Example")
    print("="*50)
    
    # Create new model
    dd_part = DualDescriptorPM(charset="", rank=3, vec_dim=2)
    
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

    # === Double Generation Example ===
    print("\n" + "="*50)
    print("Double Generation Example")
    print("="*50)

    # Create and train model using double_train
    dd_double = DualDescriptorPM(
        charset=['A','C','G','T'], 
        rank=3, 
        vec_dim=2,
        mode='nonlinear',
        user_step=2
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
        'learning_rate': 0.1
    }
    
    part_config = {
        'max_iters': 50,
        'tol': 1e-10,
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
