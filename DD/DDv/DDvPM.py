# Copyright (C) Bin-Guang Ma (mbg@mail.hzau.edu.cn). All rights reserved.
# The Dual Descriptor Vector class (P matrix form)
# Author: Bin-Guang Ma; Date: 2025-6-17

# The program is provided as it is and without warranty of any kind,
# either expressed or implied.

import math
import random
import itertools
import pickle

class DualDescriptorPM:
    """
    Purely randomized vector Dual Descriptor with optional k-mer (rank) extension.
    - charset: list of base characters
    - m: dimension of composition vectors and target vectors
    - rank: k-mer length for tokenization
    - mode: 'linear' (slide by 1) or 'nonlinear' (slide by rank)
    Each sequence may have its own target vector t_j.
    """

    def __init__(self, charset, vec_dim=4, rank=1, rank_mode='drop', mode='linear', user_step=None):
        self.charset = list(charset)
        self.m = vec_dim
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

        # 2) initialize Composition Weight Map: token -> m-dimensional vector
        self.M = {
            tok: [random.uniform(-0.5, 0.5) for _ in range(m)]
            for tok in self.tokens
        }

        # 3) initialize Position Weight matrix P (m×m)
        self.P = [[random.uniform(-0.1, 0.1) for _ in range(m)] for _ in range(m)]

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

    # ---- linear algebra helpers ----
    def mat_vec(self, M, v):
        """Multiply m×m matrix M by m-vector v."""
        return [sum(M[i][j] * v[j] for j in range(self.m)) for i in range(self.m)]

    def vec_sub(self, u, v):
        """Subtract two m-vectors."""
        return [u[i] - v[i] for i in range(self.m)]

    def dot(self, u, v):
        """Dot product of two m-vectors."""
        return sum(u[i] * v[i] for i in range(self.m))

    def mat_mul(self, A, B):
        """Multiply p×q matrix A by q×r matrix B → p×r matrix."""
        p, q = len(A), len(A[0])
        r = len(B[0])
        C = [[0.0]*r for _ in range(p)]
        for i in range(p):
            for k in range(q):
                aik = A[i][k]
                for j in range(r):
                    C[i][j] += aik * B[k][j]
        return C

    def transpose(self, M):
        """Transpose an m×m matrix."""
        return [list(col) for col in zip(*M)]

    def describe(self, seq):
        """
        Compute list of N(k)=P·M[token_k] for a given sequence.
        Each N(k) is an m-dimensional vector obtained by matrix-vector multiplication.
        
        Args:
            seq (str): Input sequence to describe
            
        Returns:
            list: List of m-dimensional vectors (N(k) values) for each token position
        """
        toks = self.extract_tokens(seq)
        # For each token, compute N(k) = P · M[token]
        return [self.mat_vec(self.P, self.M[tok]) for tok in toks]

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

    def deviation(self, seqs, t_list):
        """
        Compute mean squared deviation D:
        D = average over all positions and sequences of ||P x_token - t_j||^2.
        """
        total = 0.0
        count = 0
        for seq, t in zip(seqs, t_list):
            toks = self.extract_tokens(seq)
            for tok in toks:
                xk = self.M[tok]
                Nk = self.mat_vec(self.P, xk)
                err = self.vec_sub(Nk, t)
                total += self.dot(err, err)
                count += 1
        return total / count if count else 0.0

    def _invert_matrix(self, A):
        """
        Invert an m×m matrix A by Gauss-Jordan elimination.
        Returns A_inv.
        """
        n = len(A)
        # build augmented [A | I]
        M = [row[:] + [1.0 if i==j else 0.0 for j in range(n)] for i, row in enumerate(A)]
        for i in range(n):
            piv = M[i][i]
            if abs(piv) < 1e-12:
                continue
            # normalize row
            M[i] = [mij / piv for mij in M[i]]
            # eliminate others
            for r in range(n):
                if r == i: continue
                fac = M[r][i]
                M[r] = [M[r][c] - fac*M[i][c] for c in range(2*n)]
        # extract inverse
        A_inv = [row[n:] for row in M]
        return A_inv    

    def update_P(self, seqs, t_list):
        """
        Update P by closed-form:
          U = sum_{j,k} x_{jk} x_{jk}^T
          V = sum_{j,k} t_j x_{jk}^T
          P = V U^{-1}
        """
        m = self.m
        # zero matrices
        U = [[0.0]*m for _ in range(m)]
        V = [[0.0]*m for _ in range(m)]
        for seq, t in zip(seqs, t_list):
            toks = self.extract_tokens(seq)
            for tok in toks:
                xk = self.M[tok]
                # accumulate U += xk xk^T
                for i in range(m):
                    for j in range(m):
                        U[i][j] += xk[i] * xk[j]
                # accumulate V += t xk^T
                for i in range(m):
                    for j in range(m):
                        V[i][j] += t[i] * xk[j]
        # invert U
        U_inv = self._invert_matrix(U)
        # P = V * U^{-1}
        self.P = self.mat_mul(V, U_inv)

    def update_M(self, seqs, t_list):
        """
        Update each M[token] by closed-form:
          For token c, let O_c = occurrences,
          R_c = sum P^T t_j over O_c,
          x_c = (P^T P)^{-1} (R_c / |O_c|)
        """
        m = self.m
        Pt = self.transpose(self.P)
        # precompute M = (P^T P)^{-1}
        Mmat = [[sum(Pt[i][k]*self.P[k][j] for k in range(m)) for j in range(m)] for i in range(m)]
        M_inv = self._invert_matrix(Mmat)
        # accumulate R_c and counts
        R = {tok: [0.0]*m for tok in self.tokens}
        Cnt = {tok: 0 for tok in self.tokens}
        for seq, t in zip(seqs, t_list):
            toks = self.extract_tokens(seq)
            for tok in toks:
                # R_c += P^T t
                for i in range(m):
                    R[tok][i] += sum(Pt[i][j]*t[j] for j in range(m))
                Cnt[tok] += 1
        # update M
        for tok in self.tokens:
            cnt = Cnt[tok]
            if cnt > 0:
                # avg R
                Ravg = [ri/cnt for ri in R[tok]]
                # x = M_inv * Ravg
                self.M[tok] = [sum(M_inv[i][j]*Ravg[j] for j in range(m)) for i in range(m)]

    def train(self, seqs, t_list, max_iters=20, tol=1e-8):
        """
        Alternate training of P and M until deviation converges.
        Returns list of deviation history.
        """
        D_prev = float('inf')
        history = []
        for it in range(max_iters):
            self.update_P(seqs, t_list)
            self.update_M(seqs, t_list)
            D = self.deviation(seqs, t_list)
            history.append(D)
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

    def grad_train(self, seqs, t_list, max_iters=20, tol=1e-8, 
                   learning_rate=0.01, decay_rate=0.95):
        """
        Train the model using gradient descent instead of closed-form solutions.
        Alternates between updating P and M using gradients of the loss function.
        
        Args:
            seqs (list): List of input sequences
            t_list (list): List of target vectors for each sequence
            max_iters (int): Maximum number of training iterations
            tol (float): Tolerance for convergence detection
            learning_rate (float): Initial learning rate for gradient descent
            decay_rate (float): Multiplicative decay factor for learning rate after each iteration
            
        Returns:
            list: History of mean squared deviations over iterations
        """
        history = []
        D_prev = float('inf')
        
        for it in range(max_iters):
            total_tokens = 0
            m = self.m
            
            # 1. Update P using gradient descent
            grad_P = [[0.0] * m for _ in range(m)]
            
            for seq, t in zip(seqs, t_list):
                toks = self.extract_tokens(seq)
                for tok in toks:
                    x = self.M[tok]
                    Nk = self.mat_vec(self.P, x)  # P * x
                    error = self.vec_sub(Nk, t)    # (P*x - t)
                    
                    # Compute gradient contributions: dL/dP = 2 * error * x^T
                    for i in range(m):
                        for j in range(m):
                            grad_P[i][j] += 2 * error[i] * x[j]
                
                total_tokens += len(toks)
            
            # Average gradient and update P
            if total_tokens > 0:
                for i in range(m):
                    for j in range(m):
                        self.P[i][j] -= learning_rate * grad_P[i][j] / total_tokens
            
            # 2. Update M using gradient descent
            grad_M = {tok: [0.0] * m for tok in self.tokens}
            token_counts = {tok: 0 for tok in self.tokens}
            total_tokens = 0
            
            for seq, t in zip(seqs, t_list):
                toks = self.extract_tokens(seq)
                for tok in toks:
                    x = self.M[tok]
                    Nk = self.mat_vec(self.P, x)  # P * x
                    error = self.vec_sub(Nk, t)    # (P*x - t)
                    
                    # Compute gradient: dL/dx = 2 * P^T * error
                    Pt_error = self.mat_vec(self.transpose(self.P), error)
                    for i in range(m):
                        grad_M[tok][i] += 2 * Pt_error[i]
                    token_counts[tok] += 1
                
                total_tokens += len(toks)
            
            # Update M vectors
            for tok in self.tokens:
                if token_counts[tok] > 0:
                    for i in range(m):
                        self.M[tok][i] -= learning_rate * grad_M[tok][i] / total_tokens
            
            # 3. Compute current deviation and record history
            D = self.deviation(seqs, t_list)
            history.append(D)
            print(f"Iter {it:2d}: D = {D:.6e} (lr={learning_rate:.6f})")
            
            # 4. Check for convergence
            if D >= D_prev - tol:
                print("Converged.")
                break
            D_prev = D
            
            # 5. Decay learning rate
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

    def auto_train(self, seqs, max_iters=100, tol=1e-6, learning_rate=0.01, continued=False, auto_mode='reg', decay_rate=0.95):
        """
        Self-training method using gradient descent with two modes:
          - 'gap': Predicts current token's embedding (self-consistency)
          - 'reg': Predicts next token's embedding (auto-regressive)
        
        Parameters:
            seqs (list): Input sequences for training
            learning_rate (float): Initial learning rate
            max_iters (int): Maximum training iterations
            tol (float): Convergence tolerance
            continued (bool): Continue training with existing parameters
            auto_mode (str): Training mode - 'gap' or 'reg'
            decay_rate (float): Multiplicative decay for learning rate
        
        Returns:
            list: Training history (loss values per iteration)
        """
        # Validate auto_mode
        if auto_mode not in ('gap', 'reg'):
            raise ValueError("auto_mode must be either 'gap' or 'reg'")
        
        # Initialize M and P if not continuing
        if not continued:
            self.M = {
                tok: [random.uniform(-0.5, 0.5) for _ in range(self.m)]
                for tok in self.tokens
            }
            self.P = [[random.uniform(-0.1, 0.1) for _ in range(self.m)] 
                      for _ in range(self.m)]
        
        # Calculate total training samples
        total_samples = 0
        for seq in seqs:
            tokens = self.extract_tokens(seq)
            if auto_mode == 'gap':
                total_samples += len(tokens)
            else:  # 'reg' mode
                total_samples += max(0, len(tokens) - 1)
        
        if total_samples == 0:
            raise ValueError("No training samples found")
        
        history = []  # Store loss history
        prev_loss = float('inf')
        
        for it in range(max_iters):
            # Initialize gradients
            grad_P = [[0.0] * self.m for _ in range(self.m)]
            grad_M = {tok: [0.0] * self.m for tok in self.tokens}
            total_loss = 0.0
            
            # Process all sequences
            for seq in seqs:
                tokens = self.extract_tokens(seq)
                if not tokens:
                    continue
                
                # Process tokens based on mode
                for k in range(len(tokens)):
                    # Skip last token in 'reg' mode
                    if auto_mode == 'reg' and k == len(tokens) - 1:
                        continue
                    
                    current_token = tokens[k]
                    x_current = self.M[current_token]
                    
                    # Compute N(k) = P · x_current
                    Nk = self.mat_vec(self.P, x_current)
                    
                    # Set target based on mode
                    if auto_mode == 'gap':
                        target = x_current
                    else:  # 'reg'
                        next_token = tokens[k + 1]
                        target = self.M[next_token]
                    
                    # Compute error and loss
                    error = [Nk[i] - target[i] for i in range(self.m)]
                    sq_error = sum(e ** 2 for e in error)
                    total_loss += sq_error
                    
                    # Compute gradient coefficient (2 * error)
                    grad_coeff = 2.0
                    
                    # Gradient for P: outer product of error and x_current
                    for i in range(self.m):
                        for j in range(self.m):
                            grad_P[i][j] += grad_coeff * error[i] * x_current[j]
                    
                    # Gradient for current token
                    if auto_mode == 'gap':
                        # Compute (P^T · error) - error
                        Pt_error = self.mat_vec(self.transpose(self.P), error)
                        for i in range(self.m):
                            grad_term = Pt_error[i] - error[i]
                            grad_M[current_token][i] += grad_coeff * grad_term
                    else:  # 'reg'
                        # Compute P^T · error
                        Pt_error = self.mat_vec(self.transpose(self.P), error)
                        for i in range(self.m):
                            grad_M[current_token][i] += grad_coeff * Pt_error[i]
                        # Gradient for next token: -2 * error
                        for i in range(self.m):
                            grad_M[next_token][i] -= grad_coeff * error[i]
            
            # Average gradients and update parameters
            if total_samples > 0:
                # Average gradients
                for i in range(self.m):
                    for j in range(self.m):
                        grad_P[i][j] /= total_samples
                for tok in self.tokens:
                    for i in range(self.m):
                        grad_M[tok][i] /= total_samples
                
                # Update P and M
                for i in range(self.m):
                    for j in range(self.m):
                        self.P[i][j] -= learning_rate * grad_P[i][j]
                for tok in self.tokens:
                    for i in range(self.m):
                        self.M[tok][i] -= learning_rate * grad_M[tok][i]
                
                # Compute average loss
                avg_loss = total_loss / total_samples
                history.append(avg_loss)
            else:
                avg_loss = 0.0
                history.append(avg_loss)
            
            # Print progress
            mode_display = "Gap" if auto_mode == 'gap' else "Reg"
            print(f"AutoTrain({mode_display}) Iter {it:3d}: loss = {avg_loss:.6f}, lr = {learning_rate:.6f}")
            
            # Check convergence
            if prev_loss - avg_loss < tol:
                print(f"Converged after {it+1} iterations")
                break
            prev_loss = avg_loss
            
            # Apply learning rate decay
            learning_rate *= decay_rate
        
        # Compute mean token embeddings and token count
        total_token_count = 0
        total_emb = [0.0] * self.m
        for seq in seqs:
            tokens = self.extract_tokens(seq)
            total_token_count += len(tokens)
            for token in tokens:
                emb = self.M[token]
                for d in range(self.m):
                    total_emb[d] += emb[d]
        
        self.mean_token_count = total_token_count / len(seqs)
        self.mean_t = [total_emb[d] / total_token_count for d in range(self.m)]
        self.trained = True
        
        return history

    def dd_features(self, seq, t):
        """
        Extract features for one sequence:
        [flattened P matrix] + [flattened M vectors sorted by token] +
        [mean deviations per token]
        """
        feats = []
        # 1) flatten P
        for row in self.P:
            feats.extend(row)
        # 2) flatten M
        for tok in sorted(self.tokens):
            feats.extend(self.M[tok])
        # 3) mean deviation per token in seq
        toks = self.extract_tokens(seq)
        for tok in sorted(self.tokens):
            devs = []
            for _ in toks:
                Nk = self.mat_vec(self.P, self.M[tok])
                diff = self.vec_sub(Nk, t)
                devs.append(self.dot(diff, diff))
            feats.append(sum(devs)/len(devs) if devs else 0.0)
        return feats

    def predict_t(self, seq):
        """
        Predict the target vector t for a given sequence.
        The optimal t is the mean of all N(k) vectors in the sequence.
        
        Args:
            seq (str): Input sequence to predict
            
        Returns:
            list: m-dimensional vector (predicted target value t)
        """
        # Get all N(k) vectors for the sequence
        Nk_list = self.describe(seq)
        if not Nk_list:
            return [0.0] * self.m  # Return zero vector if no tokens
        
        # Calculate mean vector: t_pred = mean(N(k))
        t_pred = [0.0] * self.m
        for vec in Nk_list:
            for i in range(self.m):
                t_pred[i] += vec[i]
        return [x / len(Nk_list) for x in t_pred]

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
                Nk = self.mat_vec(self.P, self.M[tok])
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
                Nk = self.mat_vec(self.P, self.M[tok])
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

    def show(self):
        """
        Display model status:
        m, rank, mode, #tokens, sample P, and sample M.
        """
        print("DualDescriptorRandomVector Status:")
        print(f"  m = {self.m}, rank = {self.rank}, mode = {self.mode}")
        print(f"  # tokens = {len(self.tokens)}")
        print("  Sample P matrix:")
        for row in self.P:
            print("   ", row)
        print("  Sample M (first 3 tokens):")
        for tok in sorted(self.tokens)[:3]:
            print(f"    {tok}: {self.M[tok]}")

    def count_parameters(self):
        """
        Calculate, print, and return the total number of learnable parameters in the model.
        
        The model has two sets of parameters:
        1. Composition Weight Map (M): Each token has an m-dimensional vector.
        2. Position Weight Matrix (P): An m×m matrix.
        
        Prints a detailed breakdown of parameter counts.
        Returns:
            int: Total number of learnable parameters
        """
        # Calculate parameter counts
        num_tokens = len(self.tokens)
        M_params = num_tokens * self.m
        p_matrix_params = self.m * self.m
        total_params = M_params + p_matrix_params
        
        # Print detailed breakdown
        print(f"Parameter Count Breakdown:")
        print(f"1. Composition Weight Map (M):")
        print(f"   - Number of tokens: {num_tokens}")
        print(f"   - Vector dimension (m): {self.m}")
        print(f"   - Parameters: {num_tokens} × {self.m} = {M_params}")
        print(f"2. Position Weight Matrix (P):")
        print(f"   - Matrix dimension: {self.m}×{self.m}")
        print(f"   - Parameters: {self.m} × {self.m} = {p_matrix_params}")
        print(f"Total Learnable Parameters: {total_params}")
        
        return total_params

    def part_train(self, vec_seqs, max_iters=100, tol=1e-6, learning_rate=0.01, 
                   continued=False, auto_mode='reg', decay_rate=0.95):
        """
        Train the I matrix (transformation matrix) on vector sequences using gradient descent.
        Only updates self.I, leaves P and M unchanged.
        
        Parameters:
            vec_seqs (list): List of vector sequences (each sequence is list of m-dim vectors)
            max_iters (int): Maximum training iterations
            tol (float): Convergence tolerance
            learning_rate (float): Initial learning rate
            continued (bool): Continue training existing I matrix
            auto_mode (str): Training mode - 'gap' (self-consistency) or 'reg' (auto-regressive)
            decay_rate (float): Multiplicative decay factor for learning rate
            
        Returns:
            list: Training history (loss values per iteration)
        """
        if auto_mode not in ('gap', 'reg'):
            raise ValueError("auto_mode must be either 'gap' or 'reg'")

        # Initialize I matrix if needed
        if not continued or not hasattr(self, 'I'):
            # Initialize I as identity matrix
            self.I = [[1.0 if i == j else 0.0 for j in range(self.m)] 
                      for i in range(self.m)]
        
        # Calculate total training samples
        total_samples = 0
        for seq in vec_seqs:
            if auto_mode == 'gap':
                total_samples += len(seq)  # All vectors are samples
            else:  # 'reg' mode
                total_samples += max(0, len(seq) - 1)  # Vectors except last
                
        if total_samples == 0:
            raise ValueError("No training samples found")
            
        history = []  # Store loss history
        prev_loss = float('inf')
        original_lr = learning_rate  # Save initial learning rate
        
        # Compute mean vector for generation
        total_vectors = 0
        total_vec_sum = [0.0] * self.m
        for seq in vec_seqs:
            for vec in seq:
                total_vectors += 1
                for d in range(self.m):
                    total_vec_sum[d] += vec[d]
        self.mean_vector = [v_sum / total_vectors for v_sum in total_vec_sum] if total_vectors > 0 else [0.0]*self.m
        
        for it in range(max_iters):
            # Initialize gradient matrix for I
            grad_I = [[0.0] * self.m for _ in range(self.m)]
            total_loss = 0.0
            
            # Process all vector sequences
            for seq in vec_seqs:
                # Process vectors based on mode
                for k in range(len(seq)):
                    # Skip last vector in 'reg' mode
                    if auto_mode == 'reg' and k == len(seq) - 1:
                        continue
                    
                    current_vec = seq[k]
                    
                    # Compute transformed vector: I · current_vec
                    transformed = self.mat_vec(self.I, current_vec)
                    
                    # Set target based on mode
                    if auto_mode == 'gap':
                        target = current_vec  # Self-consistency
                    else:  # 'reg'
                        target = seq[k + 1]   # Next vector
                    
                    # Compute error and loss
                    error = [transformed[i] - target[i] for i in range(self.m)]
                    sq_error = sum(e ** 2 for e in error)
                    total_loss += sq_error
                    
                    # Compute gradients (2 * error from derivative of squared loss)
                    grad_coeff = 2.0
                    
                    # Gradient for I: outer product of error and current_vec
                    for i in range(self.m):
                        for j in range(self.m):
                            grad_I[i][j] += grad_coeff * error[i] * current_vec[j]
            
            # Average gradients and update I matrix
            if total_samples > 0:
                # Average gradients
                for i in range(self.m):
                    for j in range(self.m):
                        grad_I[i][j] /= total_samples
                
                # Update I matrix
                for i in range(self.m):
                    for j in range(self.m):
                        self.I[i][j] -= learning_rate * grad_I[i][j]
                
                # Compute average loss
                avg_loss = total_loss / total_samples
                history.append(avg_loss)
            else:
                avg_loss = 0.0
                history.append(avg_loss)
            
            # Print progress
            mode_display = "Gap" if auto_mode == 'gap' else "Reg"
            print(f"PartTrain({mode_display}) Iter {it:3d}: loss = {avg_loss:.6f}, lr = {learning_rate:.6f}")
            
            # Check convergence
            if prev_loss - avg_loss < tol:
                print(f"Converged after {it+1} iterations")
                break
            prev_loss = avg_loss
            
            # Apply learning rate decay
            learning_rate *= decay_rate
        
        return history

    def part_generate(self, L, tau=0.0, mode='reg'):
        """
        Generate a sequence of vectors using the trained I matrix.
        
        Parameters:
            L (int): Length of sequence to generate
            tau (float): Temperature for randomness (0 = deterministic)
            mode (str): Generation mode - 'gap' (self-consistency) or 'reg' (auto-regressive)
                
        Returns:
            list: Generated sequence of m-dimensional vectors
        """
        if not hasattr(self, 'I'):
            raise RuntimeError("I matrix not initialized - train first")
            
        if tau < 0:
            raise ValueError("Temperature must be non-negative")
            
        sequence = []
        
        if mode == 'gap':
            # Gap mode: Generate independent vectors
            for _ in range(L):
                # Start with mean vector
                vec = self.mean_vector[:]
                
                # Apply transformation: I · vec
                transformed = self.mat_vec(self.I, vec)
                
                # Add temperature-controlled noise
                if tau > 0:
                    transformed = [v + random.gauss(0, tau) for v in transformed]
                    
                sequence.append(transformed)
                
        else:  # 'reg' mode
            # Auto-regressive generation
            # Start with mean vector
            current_vec = self.mean_vector[:]
            
            for _ in range(L):
                # Apply transformation: I · current_vec
                transformed = self.mat_vec(self.I, current_vec)
                
                # Add temperature-controlled noise
                if tau > 0:
                    transformed = [v + random.gauss(0, tau) for v in transformed]
                
                sequence.append(transformed)
                current_vec = transformed  # Set as input for next step
                
        return sequence

    def double_train(self, seqs, auto_mode='reg', part_mode='reg', 
                    auto_params=None, part_params=None):
        """
        Two-stage training method: 
          1. First train on character sequences using auto_train (unsupervised)
          2. Then convert sequences to vector sequences using S(l) and train I tensor
        
        Parameters:
            seqs (list): Input character sequences
            auto_mode (str): Training mode for auto_train - 'gap' or 'reg'
            part_mode (str): Training mode for part_train - 'gap' or 'reg'
            auto_params (dict): Parameters for auto_train (max_iters, tol, learning_rate, decay_rate)
            part_params (dict): Parameters for part_train (max_iters, tol, learning_rate, decay_rate)
            
        Returns:
            tuple: (auto_history, part_history) training histories
        """
        # Set default parameters if not provided
        auto_params = auto_params or {
            'max_iters': 100, 
            'tol': 1e-6, 
            'learning_rate': 0.01,
            'decay_rate': 0.95  # Added learning rate decay parameter
        }
        part_params = part_params or {
            'max_iters': 100, 
            'tol': 1e-6, 
            'learning_rate': 0.01,
            'decay_rate': 0.95  # Added learning rate decay parameter
        }
        
        # Stage 1: Train character model with auto_train
        print("="*50)
        print("Stage 1: Auto-training on character sequences")
        print("="*50)
        auto_history = self.auto_train(
            seqs, 
            auto_mode=auto_mode,
            max_iters=auto_params['max_iters'],
            tol=auto_params['tol'],
            learning_rate=auto_params['learning_rate'],
            decay_rate=auto_params['decay_rate']  # Pass decay rate
        )
        
        # Stage 2: Convert sequences to vector sequences using S(l)
        print("\n" + "="*50)
        print("Stage 2: Converting sequences to vector representations")
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
        
        # Train I tensor on vector sequences
        print("\n" + "="*50)
        print("Stage 3: Training I tensor on vector sequences")
        print("="*50)
        part_history = self.part_train(
            vec_seqs,
            max_iters=part_params['max_iters'],
            tol=part_params['tol'],
            learning_rate=part_params['learning_rate'],
            decay_rate=part_params['decay_rate'],  # Pass decay rate
            auto_mode=part_mode
        )
        
        return auto_history, part_history

    def compute_Nk(self, token):
        """
        Compute the Nk vector for a given token.
        Nk = P · M[token]
        """
        return self.mat_vec(self.P, self.M[token])

    def double_generate(self, L, tau=0.0):
        """
        Generate character sequences using a two-stage approach that combines:
          1. Character-level model (auto-trained) for token probabilities
          2. Vector-sequence model (part-trained) for structural coherence
        
        Steps:
          a. Generate initial sequence with character model
          b. Compute cumulative vectors S(l) for initial sequence
          c. Use I-tensor to refine vector sequence
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
        
        # Stage 3: Refine vectors using I-tensor (part_generate in 'reg' mode)
        refined_vectors = self.part_generate(len(s_vectors), mode='reg', tau=tau)
        
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
                # Predict N(k) for this token
                predicted_nk = self.compute_Nk(token)
                
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
            actual_nk = self.compute_Nk(chosen_token)
            current_s = [current_s[i] + actual_nk[i] for i in range(self.m)]
        
        return ''.join(generated_tokens)

    def save(self, filename):
        """
        Save the current state of the DualDescriptorPM object to a binary file.
        """
        with open(filename, 'wb') as f:
            pickle.dump(self.__dict__, f)
        print(f"Model saved to {filename}")

    @classmethod
    def load(cls, filename):
        """
        Load a saved DualDescriptorPM model from a binary file.
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
    
    charset = ['A','C','G','T']
    m = 3
    rank = 3
    mode = 'nonlinear'

    dd = DualDescriptorPM(charset, vec_dim=m, rank=rank, mode=mode)

    random.seed(2)
    seqs = []
    t_list = []
    for _ in range(10):
        L = random.randint(300, 500)
        seq = ''.join(random.choices(charset, k=L))
        seqs.append(seq)
        t_list.append([random.uniform(-1,1) for _ in range(m)])

    history = dd.train(seqs, t_list)

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

    dd.show()

    dd.count_parameters()

    feats = dd.dd_features(seqs[0], t_list[0])
    print("\nFeature vector length:", len(feats))
    print("First 10 features:", feats[:10])

    # reconstruct a representative sequence after training
    repr_seq = dd.reconstruct()
    print(f"Representative sequence: {repr_seq}")

    # generate sequences after training
    seq_deterministic = dd.generate(L=100, tau=0.0)
    print('\nSeq_deterministic: ', seq_deterministic)
    seq_random = dd.generate(L=100, tau=1.0)
    print('Seq_random: ', seq_random)

     # Create model instance
    dd = DualDescriptorPM(charset, vec_dim=m, rank=rank, mode=mode)
    
    # Generate synthetic training data
    random.seed(2)
    seqs = []
    t_list = []
    for _ in range(10):  # Smaller dataset for demonstration
        L = random.randint(200, 300)  # Shorter sequences
        seq = ''.join(random.choices(charset, k=L))
        seqs.append(seq)
        t_list.append([random.uniform(-1,1) for _ in range(m)])
    
    # Train using gradient descent
    print("Training with gradient descent...")
    gd_history = dd.grad_train(
        seqs, 
        t_list,
        max_iters=100,
        learning_rate=1.1,
        decay_rate=0.99
    )
    
    # Train using closed-form for comparison
    print("\nTraining with closed-form updates...")
    cf_dd = DualDescriptorPM(charset, vec_dim=m, rank=rank, mode=mode)
    cf_history = cf_dd.train(seqs, t_list, max_iters=15)
    
    # Compare convergence
    print("\nConvergence comparison:")
    print("Iter |  GD Deviation  |  CF Deviation")
    for i, (gd, cf) in enumerate(zip(gd_history, cf_history)):
        print(f"{i:4d} | {gd:12.6e} | {cf:12.6e}")
    
    # Evaluate predictions
    pred_t_gd = [dd.predict_t(seq) for seq in seqs]
    pred_t_cf = [cf_dd.predict_t(seq) for seq in seqs]
    
    # Calculate correlation coefficients
    print("\nPrediction correlations (GD model):")
    corr_sum = 0.0
    for i in range(m):
        actual = [t[i] for t in t_list]
        pred = [t[i] for t in pred_t_gd]
        corr = correlation(actual, pred)
        print(f"Dim {i}: {corr:.4f}")
        corr_sum += corr
    print(f"Average correlation: {corr_sum/m:.4f}")
    
    print("\nPrediction correlations (CF model):")
    corr_sum = 0.0
    for i in range(m):
        actual = [t[i] for t in t_list]
        pred = [t[i] for t in pred_t_cf]
        corr = correlation(actual, pred)
        print(f"Dim {i}: {corr:.4f}")
        corr_sum += corr
    print(f"Average correlation: {corr_sum/m:.4f}")
    
    # Generate sequences
    print("\nGenerated sequence (GD model):")
    print(dd.generate(L=20, tau=0.5))
    
    print("\nGenerated sequence (CF model):")
    print(cf_dd.generate(L=20, tau=0.5))

    # Create model
    dd_auto = DualDescriptorPM(charset, vec_dim=m, rank=rank, mode=mode)
    
    # Generate synthetic sequences
    seqs_auto = []
    for _ in range(50):  # 50 sequences
        L = random.randint(100, 200)
        seq = ''.join(random.choices(charset, k=L))
        seqs_auto.append(seq)
    
    # Train in auto-regressive mode
    print("Training in auto-regressive mode...")
    history_reg = dd_auto.auto_train(
        seqs_auto,
        auto_mode='reg',
        learning_rate=0.1,
        decay_rate=0.98,
        max_iters=50
    )
    
    # Train in self-consistency mode
    print("\nTraining in self-consistency mode...")
    history_gap = dd_auto.auto_train(
        seqs_auto,
        auto_mode='gap',
        learning_rate=0.05,
        decay_rate=0.97,
        continued=True  # Continue with existing parameters
    )
    
    # Generate sequence using trained model
    print("\nGenerated sequence (auto-regressive):")
    print(dd_auto.generate(L=30, tau=0.5))

    # Initialize model
    charset = ['A','C','G','T']
    m = 3
    rank = 2
    mode = 'linear'
    dd = DualDescriptorPM(charset, vec_dim=m, rank=rank, mode=mode)
    
    # Create synthetic vector sequences for training
    num_sequences = 5
    sequence_length = 10
    vec_seqs = []
    
    # Create sequences with linear relationship: v_{k+1} ≈ 0.5 * v_k
    for _ in range(num_sequences):
        seq = []
        start_vec = [random.uniform(-1, 1) for _ in range(m)]
        seq.append(start_vec)
        for i in range(1, sequence_length):
            next_vec = [0.5 * val for val in seq[i-1]]
            # Add small noise
            next_vec = [v + random.gauss(0, 0.05) for v in next_vec]
            seq.append(next_vec)
        vec_seqs.append(seq)
    
    print("\nTraining in auto-regressive mode...")
    history_reg = dd.part_train(
        vec_seqs,
        auto_mode='reg',
        learning_rate=0.1,
        decay_rate=0.95,
        max_iters=50
    )
    
    print("\nGenerating sequence in auto-regressive mode...")
    generated_seq_reg = dd.part_generate(L=8, tau=0.05, mode='reg')
    print("Generated sequence (reg mode):")
    for i, vec in enumerate(generated_seq_reg):
        print(f"Position {i}: {[round(x, 4) for x in vec]}")
    
    print("\nTraining in self-consistency mode...")
    history_gap = dd.part_train(
        vec_seqs,
        auto_mode='gap',
        learning_rate=0.05,
        decay_rate=0.97,
        continued=True,  # Continue with existing parameters
        max_iters=30
    )
    
    print("\nGenerating sequence in self-consistency mode...")
    generated_seq_gap = dd.part_generate(L=6, tau=0.1, mode='gap')
    print("Generated sequence (gap mode):")
    for i, vec in enumerate(generated_seq_gap):
        print(f"Position {i}: {[round(x, 4) for x in vec]}")
    
    # Show learned I matrix
    print("\nLearned transformation matrix I:")
    for row in dd.I:
        print(f"  {[round(x, 4) for x in row]}")

    charset = ['A', 'C', 'G', 'T']
    m = 4
    rank = 2
    mode = 'linear'
    
    # Create model instance
    dd = DualDescriptorPM(charset, vec_dim=m, rank=rank, mode=mode)
    
    # Generate synthetic DNA sequences for training
    seqs = []
    for _ in range(50):
        L = random.randint(100, 200)
        seq = ''.join(random.choices(charset, k=L))
        seqs.append(seq)
    
    # Configure training parameters with decay rates
    auto_params = {
        'max_iters': 30,
        'learning_rate': 0.1,
        'decay_rate': 0.97,  # Learning rate decay for auto_train
        'tol': 1e-5
    }
    
    part_params = {
        'max_iters': 20,
        'learning_rate': 0.05,
        'decay_rate': 0.95,  # Learning rate decay for part_train
        'tol': 1e-5
    }
    
    print("Starting two-stage training...")
    auto_hist, part_hist = dd.double_train(
        seqs,
        auto_mode='reg',
        part_mode='reg',
        auto_params=auto_params,
        part_params=part_params
    )
    
    print("\nTraining completed!")
    print(f"Auto-train history: {auto_hist[-5:]}")
    print(f"Part-train history: {part_hist[-5:]}")
    
    # Generate new sequences using the trained model
    print("\nGenerating sequences with two-stage approach...")
    for i in range(3):
        seq = dd.double_generate(L=100, tau=0.5)
        print(f"Generated sequence {i+1}: {seq[:50]}... (len={len(seq)})")
        print(f"  GC content: {sum(1 for c in seq if c in ['G','C'])/len(seq):.2%}")

