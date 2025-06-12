# Copyright (C) 2005-2025, Bin-Guang Ma (mbg@mail.hzau.edu.cn). All rights reserved.
# The Dual Descriptor Vector class (AB Matrix form)
# Author: Bin-Guang Ma; Date: 2025-6-5

# The program is provided as it is and without warranty of any kind,
# either expressed or implied.

import math
import itertools
import random
import pickle

class DualDescriptorABM:
    """
    Dual Descriptor with:
      - learnable coefficient matrix Acoeff ∈ R^{m×L}
      - fixed basis matrix Bbasis ∈ R^{L×m}, Bbasis[k][i] = cos(2π*(k+1)/(i+1))
      - learnable token embeddings x_map: token → R^m
    """
    def __init__(self, charset, rank=1, vec_dim=4, mode='linear', user_step=None):
        self.charset = list(charset)
        self.rank = rank
        self.m = vec_dim        
        assert mode in ('linear','nonlinear')
        self.mode = mode
        self.step = user_step
        self.trained = False
        # build k-mer vocabulary
        toks = []
        for r in range(1, rank+1):
            for p in itertools.product(self.charset, repeat=r):
                toks.append(''.join(p).ljust(rank, '_'))
        self.tokens = sorted(set(toks))

        # token embeddings
        self.x_map = {
            tok: [random.uniform(-0.5, 0.5) for _ in range(self.m)]
            for tok in self.tokens
        }

        # to be initialized in train()
        self.L = None
        self.Acoeff = None   # m×L
        self.Bbasis = None   # L×m
        self.B_t = None      # m×L (transpose of Bbasis)

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
        L = len(seq)
        if self.mode == 'linear':
            return [seq[i:i+self.rank] for i in range(L-self.rank+1)]
        toks = []
        step = self.step or self.rank
        for i in range(0, L, step):
            frag = seq[i:i+self.rank]
            toks.append(frag.ljust(self.rank, '_'))
        return toks

    # ---- initialization ----
    def initialize(self, seqs):
        if self.L is None:
            self.L = max(len(s) for s in seqs)
        # initialize Acoeff: m×L
        self.Acoeff = [[random.uniform(-0.1,0.1) for _ in range(self.L)]
                       for _ in range(self.m)]
        # build fixed basis Bbasis: L×m
        self.Bbasis = [[math.cos(2*math.pi*(k+1)/(i+1))
                        for i in range(self.m)]
                       for k in range(self.L)]
        # cache its transpose: m×L
        self.B_t = self._transpose(self.Bbasis)

    # ---- describe sequence ----
    def describe(self, seq):
        toks = self.extract_tokens(seq)
        N = []
        for k, tok in enumerate(toks):
            if k >= self.L:
                break
            x = self.x_map[tok]                 # m-vector
            # z = Bbasis * x  => length L
            z = [sum(self.Bbasis[j][i]*x[i] for i in range(self.m))
                 for j in range(self.L)]
            # Nk = Acoeff * z => m-vector
            Nk = [sum(self.Acoeff[i][j]*z[j] for j in range(self.L))
                  for i in range(self.m)]
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
        L, m = self.L, self.m
        U = [[0.0]*L for _ in range(L)]
        V = [[0.0]*L for _ in range(m)]
        for seq, t in zip(seqs, t_list):
            toks = self.extract_tokens(seq)
            for tok in toks[:L]:
                x = self.x_map[tok]
                # z = Bbasis * x
                z = [sum(self.Bbasis[j][i]*x[i] for i in range(m)) for j in range(L)]
                # accumulate U, V
                for i in range(L):
                    for j in range(L):
                        U[i][j] += z[i]*z[j]
                for i in range(m):
                    for j in range(L):
                        V[i][j] += t[i]*z[j]
        U_inv = self._invert(U)
        self.Acoeff = self._mat_mul(V, U_inv)

    # ---- update x_map ----
    def update_x(self, seqs, t_list):
        L, m = self.L, self.m
        # precompute M_base = (Acoeff * Bbasis) (m×L * L×m = m×m) squared
        AB = self._mat_mul(self.Acoeff, self.Bbasis)   # m×m
        M_base = self._mat_mul(AB, self._transpose(AB))# m×m
        M_inv = self._invert(M_base)

        # accumulate R_c = sum B_t * (Acoeff^T * t)
        R = {tok: [0.0]*m for tok in self.tokens}
        cnt = {tok: 0 for tok in self.tokens}
        At = self._transpose(self.Acoeff)  # L×m
        for seq, t in zip(seqs, t_list):
            toks = self.extract_tokens(seq)
            for tok in toks[:L]:
                # v = B_t * (A^T * t)
                At_t = self._mat_vec(At, t)      # length m? Actually At is L×m => At_t length L
                # But here At (L×m) times t (m) → L correct, so At_t length L
                # v = B_t (m×L) * At_t (L) → m
                v = self._mat_vec(self.B_t, At_t)
                for i in range(m):
                    R[tok][i] += v[i]
                cnt[tok] += 1

        for tok in self.tokens:
            c = cnt[tok]
            if c > 0:
                avg = [ri/c for ri in R[tok]]
                self.x_map[tok] = self._mat_vec(M_inv, avg)

    # ---- training loop ----
    def train(self, seqs, t_list, max_iters=10, tol=1e-8):
        self.initialize(seqs)
        D_prev = float('inf')
        history = []
        for it in range(max_iters):
            self.update_Acoeff(seqs, t_list)
            self.update_x(seqs, t_list)
            D = self.deviation(seqs, t_list)
            history.append(D)
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

    def grad_train(self, seqs, t_list, learning_rate=0.01, max_iters=100, tol=1e-6, print_every=10):
        """
        Train the model using gradient descent optimization.
        
        Steps:
        1. Initialize model parameters
        2. For each iteration:
            a. Compute gradients for Acoeff and token embeddings
            b. Update parameters using gradient descent
            c. Calculate current loss
            d. Check convergence
        
        Args:
            seqs: List of training sequences
            t_list: List of target vectors corresponding to sequences
            learning_rate: Step size for gradient updates (default=0.01)
            max_iters: Maximum number of iterations (default=100)
            tol: Convergence tolerance (default=1e-6)
            print_every: Print progress every n iterations (default=10)
        
        Returns:
            list: Loss history during training
        """
        self.initialize(seqs)
        history = []
        prev_loss = float('inf')
        
        for it in range(max_iters):
            # Initialize gradients
            grad_A = [[0.0] * self.L for _ in range(self.m)]  # Gradient for Acoeff (m x L)
            grad_x = {tok: [0.0] * self.m for tok in self.tokens}  # Gradient for token embeddings
            
            total_loss = 0.0
            total_positions = 0
            
            # Process all sequences
            for seq, t in zip(seqs, t_list):
                toks = self.extract_tokens(seq)[:self.L]
                total_positions += len(toks)
                
                for k, tok in enumerate(toks):
                    x = self.x_map[tok]
                    
                    # Forward pass: Compute z = Bbasis @ x (L-dimensional)
                    z = [0.0] * self.L
                    for j in range(self.L):
                        for i in range(self.m):
                            z[j] += self.Bbasis[j][i] * x[i]
                    
                    # Forward pass: Compute N = Acoeff @ z (m-dimensional)
                    N = [0.0] * self.m
                    for i in range(self.m):
                        for j in range(self.L):
                            N[i] += self.Acoeff[i][j] * z[j]
                    
                    # Compute error = N - t
                    error = [N_i - t_i for N_i, t_i in zip(N, t)]
                    
                    # Accumulate loss (MSE)
                    total_loss += sum(e*e for e in error)
                    
                    # --- Backpropagation ---
                    # Gradient for Acoeff: dL/dAcoeff = error * z^T
                    for i in range(self.m):
                        for j in range(self.L):
                            grad_A[i][j] += 2 * error[i] * z[j]  # 2 comes from dL/dN
                    
                    # Gradient for token embedding:
                    # dL/dx = Bbasis^T @ (Acoeff^T @ error)
                    
                    # Step 1: Compute v = Acoeff^T @ error (L-dimensional)
                    v = [0.0] * self.L
                    for j in range(self.L):
                        for i in range(self.m):
                            v[j] += self.Acoeff[i][j] * error[i]
                    
                    # Step 2: Compute w = Bbasis^T @ v (m-dimensional)
                    w = [0.0] * self.m
                    for i in range(self.m):
                        for j in range(self.L):
                            w[i] += self.B_t[i][j] * v[j]
                    
                    # Accumulate gradient (multiply by 2 from dL/dN)
                    for i in range(self.m):
                        grad_x[tok][i] += 2 * w[i]
            
            # Calculate average loss
            avg_loss = total_loss / total_positions if total_positions else 0
            history.append(avg_loss)
            
            # Update parameters using gradients
            # Update Acoeff: Acoeff -= lr * grad_A / num_positions
            for i in range(self.m):
                for j in range(self.L):
                    self.Acoeff[i][j] -= learning_rate * grad_A[i][j] / total_positions
            
            # Update token embeddings
            for tok in self.tokens:
                for i in range(self.m):
                    self.x_map[tok][i] -= learning_rate * grad_x[tok][i] / total_positions
            
            # Print progress
            if it % print_every == 0 or it == max_iters - 1:
                print(f"Iter {it:3d}: Loss = {avg_loss:.6f}")
            
            # Check convergence
            if abs(prev_loss - avg_loss) < tol:
                print(f"Converged after {it} iterations")
                break
                
            prev_loss = avg_loss
        
        # Final setup after training
        self.mean_L = int(sum(len(self.extract_tokens(seq)) for seq in seqs) / len(seqs))
        self.mean_t = [0.0] * self.m
        for t in t_list:
            for i in range(self.m):
                self.mean_t[i] += t[i]
        self.mean_t = [x / len(t_list) for x in self.mean_t]
        self.trained = True
        
        return history

    def auto_train(self, seqs, learning_rate=0.01, max_iters=100, tol=1e-6, 
                   print_every=10, auto_mode='reg'):
        """
        Self-supervised training with two modes:
        - 'gap': Predict current token's embedding (autoencoder)
        - 'reg': Predict next token's embedding (sequential prediction)
        
        Args:
            seqs: List of training sequences
            learning_rate: Step size for gradient updates (default=0.01)
            max_iters: Maximum number of iterations (default=100)
            tol: Convergence tolerance (default=1e-6)
            print_every: Print progress every n iterations (default=10)
            auto_mode: Training objective ('gap' or 'reg', default='reg')
            
        Returns:
            list: Loss history during training
        """
        assert auto_mode in ('gap', 'reg'), "auto_mode must be 'gap' or 'reg'"
        
        self.initialize(seqs)
        history = []
        prev_loss = float('inf')
        
        # Precompute total token count for mean_L calculation
        total_tokens = 0
        for seq in seqs:
            total_tokens += len(self.extract_tokens(seq))
        
        for it in range(max_iters):
            # Initialize gradients
            grad_A = [[0.0] * self.L for _ in range(self.m)]  # Gradient for Acoeff (m x L)
            grad_x = {tok: [0.0] * self.m for tok in self.tokens}  # Gradient for token embeddings
            
            total_loss = 0.0
            total_instances = 0  # Count of training instances
            
            # Process all sequences
            for seq in seqs:
                toks = self.extract_tokens(seq)[:self.L]
                
                # Skip sequences with less than 2 tokens in 'reg' mode
                if auto_mode == 'reg' and len(toks) < 2:
                    continue
                
                # Determine token range based on mode
                token_range = range(len(toks))
                if auto_mode == 'reg':
                    token_range = range(len(toks) - 1)  # Skip last token
                
                for k in token_range:
                    current_tok = toks[k]
                    
                    # Determine target based on mode
                    if auto_mode == 'gap':
                        target_tok = current_tok
                    else:  # 'reg'
                        target_tok = toks[k+1]
                    
                    # Get embeddings
                    x_current = self.x_map[current_tok]
                    x_target = self.x_map[target_tok]
                    
                    # Forward pass: Compute z = Bbasis @ x_current (L-dimensional)
                    z = [0.0] * self.L
                    for j in range(self.L):
                        for i in range(self.m):
                            z[j] += self.Bbasis[j][i] * x_current[i]
                    
                    # Forward pass: Compute N = Acoeff @ z (m-dimensional)
                    N = [0.0] * self.m
                    for i in range(self.m):
                        for j in range(self.L):
                            N[i] += self.Acoeff[i][j] * z[j]
                    
                    # Compute error = N - x_target
                    error = [N_i - x_target_i for N_i, x_target_i in zip(N, x_target)]
                    
                    # Accumulate loss (MSE)
                    total_loss += sum(e*e for e in error)
                    total_instances += 1
                    
                    # --- Backpropagation ---
                    # Gradient for Acoeff: dL/dAcoeff = error * z^T
                    for i in range(self.m):
                        for j in range(self.L):
                            grad_A[i][j] += 2 * error[i] * z[j]  # 2 comes from dL/dN
                    
                    # Gradient for current token embedding:
                    # dL/dx_current = Bbasis^T @ (Acoeff^T @ error)
                    
                    # Step 1: Compute v = Acoeff^T @ error (L-dimensional)
                    v = [0.0] * self.L
                    for j in range(self.L):
                        for i in range(self.m):
                            v[j] += self.Acoeff[i][j] * error[i]
                    
                    # Step 2: Compute w = Bbasis^T @ v (m-dimensional)
                    w = [0.0] * self.m
                    for i in range(self.m):
                        for j in range(self.L):
                            w[i] += self.B_t[i][j] * v[j]
                    
                    # Accumulate gradient for current token
                    for i in range(self.m):
                        grad_x[current_tok][i] += 2 * w[i]
            
            # Calculate average loss
            avg_loss = total_loss / total_instances if total_instances else 0
            history.append(avg_loss)
            
            # Update parameters using gradients
            if total_instances > 0:
                # Update Acoeff: Acoeff -= lr * grad_A / num_instances
                for i in range(self.m):
                    for j in range(self.L):
                        self.Acoeff[i][j] -= learning_rate * grad_A[i][j] / total_instances
                
                # Update token embeddings
                for tok in self.tokens:
                    for i in range(self.m):
                        self.x_map[tok][i] -= learning_rate * grad_x[tok][i] / total_instances
            
            # Print progress
            if it % print_every == 0 or it == max_iters - 1:
                mode_str = "autoencoder" if auto_mode == 'gap' else "next-token prediction"
                print(f"Iter {it:3d} ({mode_str}): Loss = {avg_loss:.6f}")
            
            # Check convergence
            if abs(prev_loss - avg_loss) < tol:
                print(f"Converged after {it} iterations")
                break
                
            prev_loss = avg_loss
        
        # Calculate and store mean target and average sequence length
        self.mean_L = total_tokens // len(seqs)  # Average tokens per sequence
        
        # Calculate mean_t: average descriptor vector across all tokens
        total_vec = [0.0] * self.m
        token_count = 0
        
        for tok in self.tokens:
            # Compute descriptor vector for each token
            vec = self._compute_token_descriptor(tok)
            for i in range(self.m):
                total_vec[i] += vec[i]
            token_count += 1
        
        self.mean_t = [x / token_count for x in total_vec]
        self.trained = True
        
        return history    

    def predict_t(self, seq):
        """
        Predict the target vector t for an input sequence.        
        Steps:
        1. Compute position-wise descriptor vectors using describe()
        2. Average these vectors across all positions in the sequence
        3. Return the averaged vector as the predicted t-value        
        Args:
            seq: Input sequence string        
        Returns:
            list: Predicted t-value (m-dimensional vector)
        """
        # Get position-wise descriptor vectors
        N_list = self.describe(seq)
        
        # Handle empty sequence case
        if not N_list:
            return [0.0] * self.m
        
        # Initialize accumulator for m-dimensional vector
        t_pred = [0.0] * self.m
        
        # Sum all position vectors
        for Nk in N_list:
            for i in range(self.m):
                t_pred[i] += Nk[i]
        
        # Compute average
        n_positions = len(N_list)
        return [x / n_positions for x in t_pred]

    def _compute_token_descriptor(self, tok):
        """
        Compute the descriptor vector for a token.
        Args:
            tok: Token string
        Returns:
            list: m-dimensional descriptor vector
        """
        x = self.x_map[tok]
        # Compute z = Bbasis * x (L-dimensional vector)
        z = [0.0] * self.L
        for j in range(self.L):
            for i in range(self.m):
                z[j] += self.Bbasis[j][i] * x[i]
        # Compute Nk_vec = Acoeff * z (m-dimensional vector)
        Nk_vec = [0.0] * self.m
        for i in range(self.m):
            for j in range(self.L):
                Nk_vec[i] += self.Acoeff[i][j] * z[j]
        return Nk_vec

    def _get_token_descriptor(self, tok):
        """
        Retrieve cached descriptor vector for a token (compute if not cached).
        Args:
            tok: Token string
        Returns:
            list: m-dimensional descriptor vector
        """
        if not hasattr(self, '_tok_vec_cache'):
            self._tok_vec_cache = {}
        if tok not in self._tok_vec_cache:
            self._tok_vec_cache[tok] = self._compute_token_descriptor(tok)
        return self._tok_vec_cache[tok]
    
    def reconstruct(self):
        """
        Reconstruct a representative sequence by selecting tokens that minimize 
        the squared error between their descriptor vector and the mean target vector.
        Returns:
            str: Reconstructed sequence truncated to average token count length
        """
        assert self.trained, "Model must be trained before reconstruction."
        # Precompute descriptor vectors for all tokens
        tok_vecs = {}
        for tok in self.tokens:
            tok_vecs[tok] = self._get_token_descriptor(tok)
        
        tokens = []
        for k in range(self.mean_L):  # For each token position
            best_tok = None
            min_error = float('inf')
            # Find token minimizing error to mean target
            for tok in self.tokens:
                vec = tok_vecs[tok]
                error = sum((vec[i] - self.mean_t[i]) ** 2 for i in range(self.m))
                if error < min_error:
                    min_error = error
                    best_tok = tok
            # Remove padding underscores
            clean_tok = best_tok.rstrip('_')
            tokens.append(clean_tok)
        
        full_seq = ''.join(tokens)
        return full_seq[:self.mean_L]  # Truncate to average token count

    def generate(self, L, tau=0.0):
        """
        Generate a sequence of length L using descriptor vectors.
        Args:
            L (int): Target sequence length
            tau (float): Temperature (≥0); 0=deterministic, >0=stochastic
        Returns:
            str: Generated sequence truncated to length L
        """
        assert self.trained, "Model must be trained before generation."
        if tau < 0:
            raise ValueError("Temperature must be non-negative.")
        
        # Filter valid full-length tokens
        valid_tokens = [tok for tok in self.tokens 
                       if len(tok) == self.rank and '_' not in tok]
        if not valid_tokens:
            raise ValueError("No valid full-length tokens found.")
        
        # Precompute descriptor vectors for valid tokens
        tok_vecs = {}
        for tok in valid_tokens:
            tok_vecs[tok] = self._get_token_descriptor(tok)
        
        num_blocks = (L + self.rank - 1) // self.rank  # Blocks needed
        generated_tokens = []
        
        for k in range(num_blocks):
            scores = {}
            # Calculate score = -error for each token
            for tok in valid_tokens:
                vec = tok_vecs[tok]
                error = sum((vec[i] - self.mean_t[i]) ** 2 for i in range(self.m))
                scores[tok] = -error
            
            if tau == 0.0:  # Deterministic selection
                best_tok = max(scores.keys(), key=lambda t: scores[t])
                generated_tokens.append(best_tok)
            else:  # Stochastic selection
                token_list = list(scores.keys())
                exp_scores = [math.exp(scores[t] / tau) for t in token_list]
                sum_exp = sum(exp_scores)
                probs = [es / sum_exp for es in exp_scores]
                chosen_tok = random.choices(token_list, weights=probs, k=1)[0]
                generated_tokens.append(chosen_tok)
        
        full_seq = ''.join(generated_tokens)
        return full_seq[:L]  # Truncate to target length

    # ---- feature extraction ----
    def dd_features(self, seq): 
        feats = []
        # flatten Acoeff
        for row in self.Acoeff:
            feats.extend(row)
        # flatten x_map
        for tok in self.tokens:
            feats.extend(self.x_map[tok])
        # basis-weighted frequencies
        toks = self.extract_tokens(seq)
        for tok in self.tokens:
            for i in range(self.m):
                s = 0.0
                for j, tk in enumerate(toks[:self.L]):
                    if tk == tok:
                        s += self.Bbasis[j][i]
                feats.append(s)  
        return feats

    # ---- show state ----
    def show(self):
        print("DualDescriptorABM status:")
        print(f" L={self.L}, m={self.m}, rank={self.rank}, mode={self.mode}")
        print(" Sample Acoeff[0][:5]:", self.Acoeff[0][:5])
        print(" Sample Bbasis[0][:5]:", self.Bbasis[0][:5])
        tok0 = self.tokens[0]
        print(" Sample x_map first token:", tok0, self.x_map[tok0][:5])

    def part_train(self, vec_seqs, learning_rate=0.01, max_iters=100, tol=1e-6, 
                   print_every=10, auto_mode='reg'):
        """
        Train on sequences of real-valued vectors (instead of character sequences).
        Creates and updates only AcoeffI (m×L), while BbasisI (L×m) is fixed using sine basis.
        
        Args:
            vec_seqs: List of sequences, each sequence is a list of m-dimensional vectors
            learning_rate: Step size for gradient updates
            max_iters: Maximum number of iterations
            tol: Convergence tolerance
            print_every: Print progress every n iterations
            auto_mode: Training objective ('gap' or 'reg')
                - 'gap': Autoencoder (reconstruct current vector)
                - 'reg': Predict next vector in sequence
        """
        assert auto_mode in ('gap', 'reg'), "auto_mode must be 'gap' or 'reg'"
        
        # Determine maximum sequence length (L)
        self.L = max(len(seq) for seq in vec_seqs) if vec_seqs else 10
        
        # Initialize AcoeffI (m×L)
        self.AcoeffI = [[random.uniform(-0.1, 0.1) for _ in range(self.L)] 
                        for _ in range(self.m)]
        
        # Create fixed BbasisI using sine functions: BbasisI[j][i] = sin(2π*(j+1)/(i+1))
        self.BbasisI = [[math.sin(2 * math.pi * (j+1) / (i+1)) 
                         for i in range(self.m)] 
                        for j in range(self.L)]
        
        # Precompute transpose of BbasisI for efficient calculations
        self.BbasisI_t = self._transpose(self.BbasisI)
        
        # Calculate mean vector across all training vectors
        self.mean_vec = [0.0] * self.m
        total_vectors = 0
        for seq in vec_seqs:
            for vec in seq:
                for i in range(self.m):
                    self.mean_vec[i] += vec[i]
                total_vectors += 1
        if total_vectors > 0:
            self.mean_vec = [x / total_vectors for x in self.mean_vec]
        
        history = []
        prev_loss = float('inf')
        
        for it in range(max_iters):
            # Initialize gradient only for AcoeffI
            grad_AcoeffI = [[0.0] * self.L for _ in range(self.m)]
            
            total_loss = 0.0
            total_instances = 0
            
            # Process all vector sequences
            for seq in vec_seqs:
                # Skip sequences with less than 2 vectors in 'reg' mode
                if auto_mode == 'reg' and len(seq) < 2:
                    continue
                
                # Determine vector range based on mode
                vec_range = range(len(seq))
                if auto_mode == 'reg':
                    vec_range = range(len(seq) - 1)  # Skip last vector
                
                for k in vec_range:
                    current_vec = seq[k]
                    
                    # Determine target based on mode
                    if auto_mode == 'gap':
                        target_vec = current_vec
                    else:  # 'reg'
                        target_vec = seq[k+1]
                    
                    # --- Forward pass ---
                    # Compute z = BbasisI @ current_vec (L-dimensional)
                    z = [0.0] * self.L
                    for j in range(self.L):
                        for i in range(self.m):
                            z[j] += self.BbasisI[j][i] * current_vec[i]
                    
                    # Compute N = AcoeffI @ z (m-dimensional)
                    N = [0.0] * self.m
                    for i in range(self.m):
                        for j in range(self.L):
                            N[i] += self.AcoeffI[i][j] * z[j]
                    
                    # Compute error = N - target_vec
                    error = [N_i - target_vec_i for N_i, target_vec_i in zip(N, target_vec)]
                    
                    # Accumulate loss (MSE)
                    total_loss += sum(e*e for e in error)
                    total_instances += 1
                    
                    # --- Backpropagation ---
                    # Gradient for AcoeffI: dL/dAcoeffI = error * z^T
                    for i in range(self.m):
                        for j in range(self.L):
                            grad_AcoeffI[i][j] += 2 * error[i] * z[j]
            
            # Calculate average loss
            avg_loss = total_loss / total_instances if total_instances else 0
            history.append(avg_loss)
            
            # Update AcoeffI using gradient
            if total_instances > 0:
                for i in range(self.m):
                    for j in range(self.L):
                        self.AcoeffI[i][j] -= learning_rate * grad_AcoeffI[i][j] / total_instances
            
            # Print progress
            if it % print_every == 0 or it == max_iters - 1:
                mode_str = "autoencoder" if auto_mode == 'gap' else "next-vector prediction"
                print(f"Iter {it:3d} ({mode_str}): Loss = {avg_loss:.6f}")
            
            # Check convergence
            if abs(prev_loss - avg_loss) < tol:
                print(f"Converged after {it} iterations")
                break
                
            prev_loss = avg_loss
        
        return history

    def part_generate(self, length, tau=0.0):
        """
        Generate a sequence of vectors using the trained AcoeffI and fixed BbasisI.
        
        Args:
            length: Number of vectors to generate
            tau: Temperature parameter (≥0)
                0.0 = deterministic generation
                >0.0 = stochastic generation (standard deviation of Gaussian noise)
        
        Returns:
            list: Generated sequence of m-dimensional vectors
        """
        if not hasattr(self, 'AcoeffI') or not hasattr(self, 'BbasisI'):
            raise RuntimeError("Model not trained for vector generation. Call part_train first.")
        
        generated_seq = []
        
        # Start with the mean vector
        current_vec = self.mean_vec[:]  # Make a copy
        
        for _ in range(length):
            # Compute z = BbasisI @ current_vec
            z = [0.0] * self.L
            for j in range(self.L):
                for i in range(self.m):
                    z[j] += self.BbasisI[j][i] * current_vec[i]
            
            # Compute next_vec = AcoeffI @ z
            next_vec = [0.0] * self.m
            for i in range(self.m):
                for j in range(self.L):
                    next_vec[i] += self.AcoeffI[i][j] * z[j]
            
            # Apply stochasticity if tau > 0
            if tau > 0:
                next_vec = [val + random.gauss(0, tau) for val in next_vec]
            
            generated_seq.append(next_vec)
            current_vec = next_vec  # Set current to generated for next iteration
        
        return generated_seq

    def double_train(self, seqs, 
                     auto_mode='reg', auto_learning_rate=0.01, auto_max_iters=100, 
                     part_mode='reg', part_learning_rate=0.01, part_max_iters=100,
                     tol=1e-6, print_every=10):
        """
        Two-stage training process:
        1. First stage: Train on character sequences using auto_train (self-supervised)
        2. Second stage: 
            a. Convert sequences to vector representations using trained model
            b. Train on vector sequences using part_train (self-supervised)
        
        Args:
            seqs: List of training sequences (strings)
            auto_mode: Training objective for first stage ('gap' or 'reg')
            auto_learning_rate: Learning rate for first stage
            auto_max_iters: Max iterations for first stage
            part_mode: Training objective for second stage ('gap' or 'reg')
            part_learning_rate: Learning rate for second stage
            part_max_iters: Max iterations for second stage
            tol: Convergence tolerance
            print_every: Print progress every n iterations
            
        Returns:
            tuple: (auto_history, part_history) - loss histories from both stages
        """
        # Stage 1: Train on character sequences
        print("=== Starting Stage 1: Character sequence training ===")
        auto_history = self.auto_train(
            seqs,
            learning_rate=auto_learning_rate,
            max_iters=auto_max_iters,
            tol=tol,
            print_every=print_every,
            auto_mode=auto_mode
        )
        
        # Stage 2: Convert sequences to vectors and train
        print("\n=== Starting Stage 2: Vector sequence training ===")
        print("Converting character sequences to vector representations...")
        vec_seqs = []
        for seq in seqs:
            # Get position-wise descriptor vectors
            N_list = self.describe(seq)
            vec_seqs.append(N_list)
        
        # Train on vector sequences
        part_history = self.part_train(
            vec_seqs,
            learning_rate=part_learning_rate,
            max_iters=part_max_iters,
            tol=tol,
            print_every=print_every,
            auto_mode=part_mode
        )
        
        print("\nDouble training complete!")
        return auto_history, part_history

    def double_generate(self, L, tau=0.0):
        """
        Generate a character sequence using both stages of the trained model.
        
        Steps:
        1. Generate an initial character sequence using stage 1 parameters
        2. Convert it to descriptor vectors using stage 1 model
        3. Extend the vector sequence using stage 2 parameters
        4. Convert new vectors back to characters using stage 1 token descriptors
        5. Repeat until target length is reached
        
        Args:
            L (int): Target character sequence length
            tau (float): Temperature for stochastic selection (0=deterministic)
            
        Returns:
            str: Generated character sequence
        """
        assert self.trained, "Model must be trained before generation"
        assert hasattr(self, 'AcoeffI'), "Stage 2 parameters missing - run double_train first"
        
        # Step 1: Generate initial character sequence using stage 1
        init_chars = min(L, max(10, L//2))  # Initial character length
        char_seq = self.generate(init_chars, tau)
        
        # Continue generating until we reach target length
        while len(char_seq) < L:
            # Step 2: Convert current chars to descriptor vectors
            vec_seq = self.describe(char_seq)
            
            # Step 3: Extend vector sequence using stage 2 parameters
            next_vec = self._predict_next_vector(vec_seq)
            
            # Step 4: Find best matching token for the new vector
            next_token = self._vector_to_token(next_vec, tau)
            
            # Step 5: Append token (remove padding underscores)
            clean_token = next_token.rstrip('_')
            char_seq += clean_token
        
        return char_seq[:L]  # Trim to exact length

    def _predict_next_vector(self, vec_seq):
        """Predict next vector using stage 2 parameters"""
        # Use last vector in sequence as current state
        current_vec = vec_seq[-1] if vec_seq else self.mean_vec
        
        # Compute z = BbasisI @ current_vec
        z = [0.0] * self.L
        for j in range(self.L):
            for i in range(self.m):
                z[j] += self.BbasisI[j][i] * current_vec[i]
        
        # Compute next_vec = AcoeffI @ z
        next_vec = [0.0] * self.m
        for i in range(self.m):
            for j in range(self.L):
                next_vec[i] += self.AcoeffI[i][j] * z[j]
        
        return next_vec

    def _vector_to_token(self, target_vec, tau):
        """Find token whose descriptor best matches target vector"""
        best_tok, best_score = None, -float('inf')
        scores = []
        
        # Calculate matching score for each token
        for tok in self.tokens:
            # Get token's descriptor vector from stage 1
            tok_vec = self._get_token_descriptor(tok)
            
            # Calculate similarity (negative squared error)
            score = -sum((t - v)**2 for t, v in zip(target_vec, tok_vec))
            
            # For deterministic selection
            if tau == 0 and score > best_score:
                best_score = score
                best_tok = tok
            
            # For stochastic selection
            scores.append((tok, score))
        
        # Deterministic return
        if tau == 0:
            return best_tok
        
        # Stochastic selection
        tokens, raw_scores = zip(*scores)
        exp_scores = [math.exp(s/tau) for s in raw_scores]
        total = sum(exp_scores)
        probs = [s/total for s in exp_scores]
        return random.choices(tokens, weights=probs, k=1)[0]

    def save(self, filename):
        """
        Save the current state of the DualDescriptorABM object to a binary file.
        """
        with open(filename, 'wb') as f:
            pickle.dump(self.__dict__, f)
        print(f"Model saved to {filename}")

    @classmethod
    def load(cls, filename):
        """
        Load a saved DualDescriptorABM model from a binary file.
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
    
    #random.seed(0)
    charset = ['A','C','G','T']
    dd = DualDescriptorABM(charset, rank=3, vec_dim=2, mode='nonlinear', user_step=2)

    # generate 10 sequences of length 100–200 and random targets
    seqs, t_list = [], []
    for _ in range(10):
        L = random.randint(100, 200)
        seq = ''.join(random.choices(charset, k=L))
        seqs.append(seq)
        t_list.append([random.uniform(-1,1) for _ in range(dd.m)])

    dd.train(seqs, t_list, max_iters=50)
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
    print("First 10 features:", feats[:10])

    # Demonstrate prediction
    print("\n=== Prediction Demo ===")
    test_seq = "ACGTACGT"  # Sample test sequence
    true_t = [0.42, -0.17, 0.89]  # Example true value
    
    # Make prediction
    pred_t = dd.predict_t(test_seq)
    print(f"Test Sequence: {test_seq}")
    print(f"Predicted t: {[f'{x:.4f}' for x in pred_t]}")
    print(f"True t: {true_t}")    

    # Show prediction error
    error = sum((p - t)**2 for p, t in zip(pred_t, true_t))
    print(f"Prediction MSE: {error:.6f}")

    # Reconstruct representative sequence
    repr_seq = dd.reconstruct()
    print(f"\nReconstructed representative sequence: {repr_seq}")
    
    # Generate sequences
    print("\nGenerated sequences:")
    seq_det = dd.generate(L=100, tau=0.0)
    print(f"Deterministic (tau=0): {seq_det[:50]}...")
##    seq_stoch = dd.generate(L=100, tau=0.1)
##    print(f"Stochastic (tau=0.5): {seq_stoch[:50]}...")

    # Create model with smaller parameters for faster demonstration
    dd_grad = DualDescriptorABM(charset, rank=3, vec_dim=2, mode='nonlinear', user_step=2)
    
    print("Training with gradient descent...")
    grad_history = dd_grad.grad_train(
        seqs,
        t_list,
        learning_rate=0.05,  # Higher learning rate for demonstration
        max_iters=300,
        tol=1e-5,
        print_every=5
    )
    
    # Test on a new sequence
    test_seq = "ACGTACGTACGT"
    pred_t = dd_grad.predict_t(test_seq)
    print(f"\nTest sequence: {test_seq}")
    print(f"Predicted t: {[f'{x:.4f}' for x in pred_t]}")

    # Predict target vector for first sequence
    aseq = seqs[0]
    t_pred = dd_grad.predict_t(aseq)
    print(f"\nPredicted t for first sequence: {[round(x, 4) for x in t_pred]}")    
    
    # Calculate flattened correlation between predicted and actual targets
    pred_t_list = [dd_grad.predict_t(seq) for seq in seqs]
    
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
    
    # Generate a new sequence
    generated = dd_grad.generate(L=20, tau=0.3)
    print(f"\nGenerated sequence: {generated}")

    # Create models
    dd_gap = DualDescriptorABM(charset, rank=2, vec_dim=3, mode='linear')
    dd_reg = DualDescriptorABM(charset, rank=2, vec_dim=3, mode='linear')
    
    # Generate training data
    train_seqs = []
    for _ in range(20):  # 20 sequences
        L = random.randint(50, 80)
        seq = ''.join(random.choices(charset, k=L))
        train_seqs.append(seq)
    
    # Autoencoder training (gap mode)
    print("Training with autoencoder objective (gap mode)...")
    gap_history = dd_gap.auto_train(
        train_seqs,
        learning_rate=0.1,
        max_iters=50,
        tol=1e-5,
        print_every=5,
        auto_mode='gap'
    )
    
    # Next-token prediction training (reg mode)
    print("\nTraining with next-token prediction objective (reg mode)...")
    reg_history = dd_reg.auto_train(
        train_seqs,
        learning_rate=0.1,
        max_iters=50,
        tol=1e-5,
        print_every=5,
        auto_mode='reg'
    )
    
    # Generate sequences from both models
    print("\n=== Generated Sequences ===")
    print("Autoencoder (gap) deterministic:")
    print(dd_gap.generate(L=30, tau=0.0))
    
    print("\nNext-token (reg) deterministic:")
    print(dd_reg.generate(L=30, tau=0.0))
    
    print("\nNext-token (reg) stochastic:")
    print(dd_reg.generate(L=30, tau=0.3))

    # Create a DualDescriptorABM instance
    dd = DualDescriptorABM(charset="ACGT", rank=3, vec_dim=3)

    # Create synthetic vector sequences (3-dimensional vectors)
    vector_sequences = [
        [[0.1, -0.2, 0.3], [0.4, -0.5, 0.6], [0.7, -0.8, 0.9]],
        [[-0.3, 0.2, -0.1], [-0.6, 0.5, -0.4]],
        [[0.9, -0.8, 0.7], [0.6, -0.5, 0.4], [0.3, -0.2, 0.1]]
    ]

    # Train using autoencoder (gap) mode with fixed sine basis
    print("Training in autoencoder mode with fixed sine basis...")
    gap_history = dd.part_train(
        vector_sequences,
        learning_rate=0.1,
        max_iters=50,
        auto_mode='gap',
        print_every=5
    )

    # Train using next-vector prediction (reg) mode with fixed sine basis
    print("\nTraining in next-vector prediction mode with fixed sine basis...")
    reg_history = dd.part_train(
        vector_sequences,
        learning_rate=0.1,
        max_iters=50,
        auto_mode='reg',
        print_every=5
    )

    # Generate new sequences
    print("\nGenerating deterministic sequence...")
    det_seq = dd.part_generate(length=5, tau=0.0)
    print("Deterministic sequence:")
    for i, vec in enumerate(det_seq):
        print(f"Vector {i+1}: {[round(v, 4) for v in vec]}")

    print("\nGenerating stochastic sequence...")
    stoch_seq = dd.part_generate(length=5, tau=0.1)
    print("Stochastic sequence:")
    for i, vec in enumerate(stoch_seq):
        print(f"Vector {i+1}: {[round(v, 4) for v in vec]}")

    # Show the fixed sine basis matrix
    print("\nFixed sine basis matrix (first 3 rows):")
    for j in range(3):
        print(f"Row {j+1}: {[round(dd.BbasisI[j][i], 4) for i in range(dd.m)]}")

     # Generate training sequences
    train_seqs = []
    for _ in range(15):
        length = random.randint(50, 100)
        train_seqs.append(''.join(random.choices(charset, k=length)))
    
    print("Starting double training...")
    auto_hist, part_hist = dd.double_train(
        train_seqs,
        auto_mode='reg',          # Next-token prediction for characters
        auto_learning_rate=0.1,
        auto_max_iters=50,
        part_mode='reg',         # Next-vector prediction for descriptors
        part_learning_rate=0.05,
        part_max_iters=30,
        tol=1e-5,
        print_every=5
    )
    
    # Generate new character sequence
    print("\nGenerating new character sequence:")
    char_seq = dd.generate(L=40, tau=0.2)
    print(char_seq)
    
    # Generate new vector sequence
    print("\nGenerating new vector sequence:")
    vec_seq = dd.part_generate(length=5, tau=0.1)
    for i, vec in enumerate(vec_seq):
        print(f"Vector {i+1}: {[round(v, 4) for v in vec]}")

    # After double_train as shown in the original example
    print("\n=== Double Generation Demo ===")
    print("Generating sequence with double_generate (tau=0):")
    seq_double_det = dd.double_generate(L=100, tau=0.0)
    print(seq_double_det[:100])

    print("\nGenerating sequence with double_generate (tau=0.2):")
    seq_double_stoch = dd.double_generate(L=100, tau=0.2)
    print(seq_double_stoch[:100])

    # Compare with basic generation
    print("\nBasic generate (tau=0):")
    seq_basic = dd.generate(L=100, tau=0.0)
    print(seq_basic[:100])

    # Save trained model
    dd_grad.save("trained_model.pkl")
    
    # Load saved model
    dd_loaded = DualDescriptorABM.load("trained_model.pkl")
    
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

