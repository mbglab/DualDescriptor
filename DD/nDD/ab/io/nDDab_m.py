# Copyright (C) 2005-2025, Bin-Guang Ma (mbg@mail.hzau.edu.cn); SPDX-License-Identifier: MIT
# The Numeric Dual Descriptor Vector class (AB matrix form) implemented with pure Python
# Modified to support n-dimensional input and m-dimensional output
# This program is for the demonstration of methodology and not fully refined.
# Author: Bin-Guang Ma (assisted by DeepSeek); Date: 2025-6-20

import math
import random
import pickle

class NumDualDescriptorAB:
    """
    Numeric Dual Descriptor for vector sequences with:
      - learnable coefficient matrix Acoeff ∈ R^{m×L}
      - fixed basis matrix Bbasis ∈ R^{L×m} (computed with cosine formula)
      - learnable transformation matrix M ∈ R^{m×n}
    """
    def __init__(self, input_dim=4, output_dim=4, basis_dim=50, rank=1, rank_op='avg', rank_mode='drop', mode='linear', user_step=None):
        """
        Initialize the Dual Descriptor for vector sequences.
        
        Args:
            input_dim (int): Dimension n of input vectors
            output_dim (int): Dimension m of output vectors
            bas_dim (int): Basis dimension L
            rank (int): Window size for vector aggregation
            rank_op (str or callable): Rank operation for vector aggregation
            rank_mode (str): 'pad' or 'drop' for handling incomplete windows
            mode (str): 'linear' (sliding window) or 'nonlinear' (stepped window)
            user_step (int): Custom step size for nonlinear mode
        """
        self.n = input_dim  # Input vector dimension
        self.m = output_dim  # Output vector dimension
        self.L = basis_dim # Basis dimension (L)
        self.rank = rank
        self.rank_op = rank_op
        self.rank_mode = rank_mode
        assert mode in ('linear','nonlinear')
        self.mode = mode
        self.step = user_step
        self.trained = False
        
        # Initialize transformation matrix M (m×n)
        self.M = [[random.uniform(-0.1, 0.1) for _ in range(self.n)]
                 for _ in range(self.m)]
        
        # Initialize coefficient matrix Acoeff: m×L
        self.Acoeff = [[random.uniform(-0.1, 0.1) for _ in range(self.L)]
                       for _ in range(self.m)]
        
        # Fixed basis matrix Bbasis: L×m using cosine formula
        self.Bbasis = [[0.0] * self.m for _ in range(self.L)]
        for k in range(self.L):
            for i in range(self.m):
                # Fixed cosine basis: Bbasis[k][i] = cos(2π*(k+1)/(i+1))
                self.Bbasis[k][i] = math.cos(2 * math.pi * (k+1) / (i+1))
        
        # Cache transpose of Bbasis: m×L
        self.B_t = self._transpose(self.Bbasis)

    # ---- linear algebra helpers ----
    def _transpose(self, M):
        """Transpose a matrix"""
        return [list(col) for col in zip(*M)]

    def _mat_mul(self, A, B):
        """Matrix multiplication: A (p×q) * B (q×r) → C (p×r)"""
        p, q = len(A), len(A[0])
        r = len(B[0])
        C = [[0.0]*r for _ in range(p)]
        for i in range(p):
            for k in range(q):
                aik = A[i][k]
                for j in range(r):
                    C[i][j] += aik * B[k][j]
        return C

    def _mat_vec(self, M, v):
        """Matrix-vector multiplication: M (p×q) * v (q) → result (p)"""
        return [sum(M[i][j]*v[j] for j in range(len(v))) for i in range(len(M))]

    def _invert(self, A):
        """Gauss-Jordan inversion for square matrix"""
        n = len(A)
        M = [A[i][:] + [1.0 if i==j else 0.0 for j in range(n)]
             for i in range(n)]
        for i in range(n):
            piv = M[i][i]
            if abs(piv) < 1e-12:
                continue
            M[i] = [x/piv for x in M[i]]
            for r in range(n):
                if r == i: 
                    continue
                fac = M[r][i]
                M[r] = [M[r][c] - fac*M[i][c] for c in range(2*n)]
        return [row[n:] for row in M]

    # ---- sequence processing ----
    def extract_windows(self, seq):
        """
        Extract window vectors from sequence based on processing mode and rank operation.
        
        For linear mode:
        - Slide window by 1 step, extracting contiguous vectors of length = rank
        - Apply rank operation (avg/sum/pick/user_func) to each window
        
        For nonlinear mode:
        - Slide window by custom step (or rank length if step not specified)
        - Handle incomplete windows using rank_mode:
            • 'pad': Pad with zero vectors to maintain rank length
            • 'drop': Discard incomplete windows
        - Apply rank operation to each complete window
        
        Rank operations:
        - 'avg': Average vectors in window (default)
        - 'sum': Sum vectors in window
        - 'pick': Randomly select one vector in window
        - 'user_func': Apply custom function to window
            • Default behavior: Apply sigmoid to average vector
        
        Args:
            seq: List of m-dimensional vectors
            
        Returns:
            list: Processed vectors based on mode and operations
        """
        
        # Helper function to apply vector operations
        def apply_op(vectors):
            """Apply rank operation to a list of vectors"""
            # Get vector dimension from first vector
            d = len(vectors[0]) if vectors else 0
            
            if self.rank_op == 'sum':
                return [sum(v[j] for v in vectors) for j in range(d)]
                
            elif self.rank_op == 'pick':
                return random.choice(vectors)
                
            elif self.rank_op == 'user_func':
                # Use custom function if provided, else default behavior
                if hasattr(self, 'user_func') and callable(self.user_func):
                    return self.user_func(vectors)
                else:
                    # Default: average + sigmoid
                    avg = [sum(v[j] for v in vectors) / len(vectors) for j in range(d)]
                    return [1 / (1 + math.exp(-x)) for x in avg]
                    
            else:  # 'avg' is default
                return [sum(v[j] for v in vectors) / len(vectors) for j in range(d)]
        
        # Handle empty sequence case
        if not seq:
            return []
        
        # Linear mode: sliding window with step=1
        if self.mode == 'linear':
            L = len(seq)
            # Only process if sequence is long enough
            if L < self.rank:
                return []
                
            return [
                apply_op(seq[i:i + self.rank])
                for i in range(L - self.rank + 1)
            ]
        
        # Nonlinear mode: stepping with custom step size
        step = self.step or self.rank
        results = []
        vector_dim = len(seq[0])  # Dimension of vectors
        
        for i in range(0, len(seq), step):
            frag = seq[i:i + self.rank]
            
            if self.rank_mode == 'pad':
                # Pad fragment with zero vectors if shorter than rank
                if len(frag) < self.rank:
                    padding = [[0] * vector_dim] * (self.rank - len(frag))
                    frag += padding
                results.append(apply_op(frag))
                
            elif self.rank_mode == 'drop':
                # Only process fragments that match full rank length
                if len(frag) == self.rank:
                    results.append(apply_op(frag))
                    
        return results  

    # ---- describe sequence ----
    def describe(self, vec_seq):
        """Compute descriptor vectors for each window position"""
        windows = self.extract_windows(vec_seq)
        N = []
        for k, window_vec in enumerate(windows):
            # Apply transformation: x' = M * window_vec
            transformed_vec = self._mat_vec(self.M, window_vec)
            
            j = k % self.L  # Basis index
            # A: m×1 column vector from Acoeff
            A_col = [[self.Acoeff[i][j]] for i in range(self.m)]
            # B: 1×m row vector from Bbasis
            B_row = [self.Bbasis[j]]
            # P = A_col * B_row: m×m matrix
            P = self._mat_mul(A_col, B_row)
            # Nk = P * transformed_vec: m-vector
            Nk = self._mat_vec(P, transformed_vec)
            N.append(Nk)
        return N

    def S(self, vec_seq):
        """Compute cumulative descriptor vectors S(l) = ΣN(k) for k=1 to l"""
        Nk_list = self.describe(vec_seq)        
        S = [0.0] * self.m
        S_list = []
        for Nk in Nk_list:
            for i in range(self.m):
                S[i] += Nk[i]
            S_list.append(list(S))
        return S_list

    # ---- compute deviation ----
    def deviation(self, seqs, t_list):
        """Compute mean squared error between descriptors and targets"""
        tot = 0.0
        cnt = 0
        for seq, t in zip(seqs, t_list):
            for Nk in self.describe(seq):
                for i in range(self.m):
                    diff = Nk[i] - t[i]
                    tot += diff * diff
                cnt += 1
        return tot / cnt if cnt else 0.0

    # ---- update Acoeff ----
    def update_Acoeff(self, seqs, t_list):
        """Update coefficient matrix using position-specific projections"""
        m, L = self.m, self.L
        # Initialize accumulators
        numerator = [[0.0] * L for _ in range(m)]  # m x L
        denominator = [0.0] * L  # L-dimensional
        
        for seq, t in zip(seqs, t_list):
            windows = self.extract_windows(seq)
            for k, window_vec in enumerate(windows):
                idx = k % L  # Basis index
                # Apply transformation: x' = M * window_vec
                transformed_vec = self._mat_vec(self.M, window_vec)
                
                # Compute scalar: Bbasis[idx] • transformed_vec
                scalar = sum(self.Bbasis[idx][i] * transformed_vec[i] for i in range(m))
                
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

    # ---- update M ----
    def update_M(self, seqs, t_list):
        """Update transformation matrix using position-specific least squares"""
        m, n = self.m, self.n
        # Initialize m×n × m×n matrix and m×n vector
        size = m * n
        M_mat = [[0.0] * size for _ in range(size)]
        v_vec = [0.0] * size

        # Process each sequence and window
        for seq, t in zip(seqs, t_list):
            windows = self.extract_windows(seq)
            for k, x_k in enumerate(windows):
                j = k % self.L # Basis index
                A_col = [self.Acoeff[i][j] for i in range(m)]
                b_j = self.Bbasis[j]
                
                # Compute scalar constants
                a_norm_sq = sum(a*a for a in A_col)
                a_dot_t = sum(A_col[i] * t[i] for i in range(m))
                
                # Compute outer product matrix: x_k ⊗ b_j
                for r in range(m):
                    for s in range(n):
                        idx1 = r * n + s  # Row-major indexing
                        val = a_norm_sq * x_k[s] * b_j[r]
                        for p in range(m):
                            for q in range(n):
                                idx2 = p * n + q
                                M_mat[idx1][idx2] += val * x_k[q] * b_j[p]
                        v_vec[idx1] += a_dot_t * x_k[s] * b_j[r]
        
        try:
            M_inv = self._invert(M_mat)
            M_flat = self._mat_vec(M_inv, v_vec)
            # Reshape flattened solution to m × n matrix
            self.M = [[M_flat[i*n + j] for j in range(n)] for i in range(m)]
        except:
            pass  # Fallback to current M on singular matrix

    # ---- training loop ----
    def train(self, seqs, t_list, max_iters=10, tol=1e-8, print_every=1):
        """
        Train using alternating least squares updates.
        
        Args:
            seqs: List of training vector sequences
            t_list: List of target vectors
            max_iters: Maximum iterations
            tol: Convergence tolerance
            print_every: Print frequency
            
        Returns:
            list: Deviation history
        """
        D_prev = float('inf')
        history = []
        for it in range(max_iters):
            self.update_Acoeff(seqs, t_list)
            self.update_M(seqs, t_list)
            D = self.deviation(seqs, t_list)
            history.append(D)
            if it % print_every == 0 or it == max_iters - 1:
                print(f"Iter {it:2d}: D = {D:.6e}")
            if D - D_prev >= tol:
                print("Converged.")
                break
            D_prev = D
        # Compute training statistics
        self.mean_L = int(sum(len(seq) for seq in seqs) / len(seqs))
        self.mean_t = [0.0] * self.m
        for t in t_list:
            for i in range(self.m):
                self.mean_t[i] += t[i]
        self.mean_t = [x / len(t_list) for x in self.mean_t]
        self.trained = True
        return history

    def grad_train(self, seqs, t_list, max_iters=1000, tol=1e-8, 
                   learning_rate=0.01, continued=False, 
                   decay_rate=1.0, print_every=10):
        """
        Train using gradient descent optimization.
        
        Args:
            seqs: List of training vector sequences
            t_list: List of target vectors
            max_iters: Maximum iterations
            tol: Convergence tolerance
            learning_rate: Initial learning rate
            continued: Continue from current parameters
            decay_rate: Learning rate decay
            print_every: Print frequency
            
        Returns:
            list: Loss history
        """
        if not continued:
            # Reinitialize parameters
            self.M = [[random.uniform(-0.1, 0.1) for _ in range(self.n)]
                     for _ in range(self.m)]
            self.Acoeff = [[random.uniform(-0.1,0.1) for _ in range(self.L)]
                           for _ in range(self.m)]
        
        total_positions = sum(len(self.extract_windows(seq)) for seq in seqs)
        if total_positions == 0:
            raise ValueError("No valid windows in sequences")
        
        history = []
        D_prev = float('inf')
        current_lr = learning_rate
        
        for it in range(max_iters):
            # Initialize gradients
            grad_A = [[0.0] * self.L for _ in range(self.m)]
            grad_M = [[0.0] * self.n for _ in range(self.m)]
            total_loss = 0.0
            
            # Process each sequence
            for seq, t in zip(seqs, t_list):
                windows = self.extract_windows(seq)
                for k, window_vec in enumerate(windows):
                    j = k % self.L  # Basis index
                    
                    # Apply transformation: x' = M * window_vec
                    transformed_vec = self._mat_vec(self.M, window_vec)
                    
                    # Compute scalar: Bbasis[j] • x'
                    scalar = sum(self.Bbasis[j][i] * transformed_vec[i] for i in range(self.m))
                    
                    # Compute Nk = scalar * A_j
                    A_j = [self.Acoeff[i][j] for i in range(self.m)]
                    Nk = [A_j[i] * scalar for i in range(self.m)]
                    
                    # Compute position loss
                    pos_loss = 0.0
                    dNk = [0.0] * self.m
                    for i in range(self.m):
                        error = Nk[i] - t[i]
                        pos_loss += error * error
                        dNk[i] = 2 * error / self.m  # Gradient w.r.t Nk
                    total_loss += pos_loss / self.m
                    
                    # Compute gradients
                    for i in range(self.m):
                        # Gradient for Acoeff
                        grad_A[i][j] += dNk[i] * scalar
                        
                        # Common factor
                        factor = dNk[i] * A_j[i]
                        
                        # Gradient for M (through transformed_vec)
                        for d in range(self.m):
                            for e in range(self.n):
                                # ∂(transformed_vec[d])/∂M[d][e] = window_vec[e]
                                grad_M[d][e] += factor * self.Bbasis[j][d] * window_vec[e]
            
            # Update parameters with gradients
            for i in range(self.m):
                for j in range(self.L):
                    self.Acoeff[i][j] -= current_lr * grad_A[i][j]
            
            for d in range(self.m):
                for e in range(self.n):
                    self.M[d][e] -= current_lr * grad_M[d][e]
            
            # Calculate average loss
            avg_loss = total_loss / total_positions
            history.append(avg_loss)
            current_lr *= decay_rate  # Decay learning rate
            
            # Print progress
            if it % print_every == 0 or it == max_iters - 1:
                print(f"Iter {it:3d}: Loss = {avg_loss:.6e}, LR = {current_lr:.6f}")
                
            # Check convergence
            if D_prev - avg_loss < tol:
                print(f"Converged after {it+1} iterations")
                break
            D_prev = avg_loss
        
        # Finalize training
        self.mean_L = int(sum(len(seq) for seq in seqs) / len(seqs))
        self.mean_t = [0.0] * self.m
        for t in t_list:
            for i in range(self.m):
                self.mean_t[i] += t[i]
        self.mean_t = [x / len(t_list) for x in self.mean_t]
        self.trained = True
        return history

    def auto_train(self, seqs, auto_mode='reg', max_iters=1000, tol=1e-8,
                   learning_rate=0.01, continued=False, decay_rate=0.99,
                   print_every=10):
        """
        Self-supervised training for vector sequences.
        
        Args:
            seqs: List of training vector sequences
            auto_mode: 'reg' (predict next descriptor) only
            max_iters: Maximum iterations
            tol: Convergence tolerance
            learning_rate: Initial learning rate
            continued: Continue from current parameters
            decay_rate: Learning rate decay
            print_every: Print frequency
            
        Returns:
            list: Loss history
        """
        if not continued:
            # Reinitialize parameters
            self.M = [[random.uniform(-0.1, 0.1) for _ in range(self.n)]
                     for _ in range(self.m)]
            self.Acoeff = [[random.uniform(-0.1,0.1) for _ in range(self.L)]
                           for _ in range(self.m)]
        
        history = []
        prev_loss = float('inf')
        current_lr = learning_rate
        
        for it in range(max_iters):
            total_loss = 0.0
            grad_A = [[0.0] * self.L for _ in range(self.m)]
            grad_M = [[0.0] * self.n for _ in range(self.m)]
            
            # Prepare training samples: (sequence, window_index, next_descriptor)
            samples = []
            for seq in seqs:
                # Calculate descriptors for the entire sequence
                descriptors = self.describe(seq)
                windows = self.extract_windows(seq)
                
                if len(windows) < 2:
                    continue
                    
                if auto_mode == 'reg':
                    # Predict next descriptor from current window
                    for k in range(len(windows) - 1):
                        # Target is the next descriptor in the sequence
                        next_descriptor = descriptors[k+1]
                        samples.append((seq, k, next_descriptor))
            
            if not samples:
                if it == 0:
                    print("Warning: No training samples generated")
                continue
            
            for seq, k, next_descriptor in samples:
                windows = self.extract_windows(seq)
                if k >= len(windows):
                    continue
                    
                current_window = windows[k]
                j = k % self.L  # Basis index
                
                # Apply transformation: x' = M * current_window
                transformed_vec = self._mat_vec(self.M, current_window)
                
                # Compute scalar: Bbasis[j] • x'
                scalar = sum(self.Bbasis[j][i] * transformed_vec[i] for i in range(self.m))
                
                # Compute Nk = scalar * A_j
                A_j = [self.Acoeff[i][j] for i in range(self.m)]
                Nk = [A_j[i] * scalar for i in range(self.m)]
                
                # Compute loss to next descriptor
                loss_val = 0.0
                dNk = [0.0] * self.m
                for i in range(self.m):
                    error = Nk[i] - next_descriptor[i]
                    loss_val += error * error
                    dNk[i] = 2 * error / self.m
                total_loss += loss_val / self.m
                
                # Compute gradients
                for i in range(self.m):
                    # Gradient for Acoeff
                    grad_A[i][j] += dNk[i] * scalar
                    
                    # Common factor
                    factor = dNk[i] * A_j[i]
                    
                    # Gradient for M
                    for d in range(self.m):
                        for e in range(self.n):
                            grad_M[d][e] += factor * self.Bbasis[j][d] * current_window[e]
            
            # Update parameters
            for i in range(self.m):
                for j in range(self.L):
                    self.Acoeff[i][j] -= current_lr * grad_A[i][j]
            
            for d in range(self.m):
                for e in range(self.n):
                    self.M[d][e] -= current_lr * grad_M[d][e]
            
            # Calculate average loss
            avg_loss = total_loss / len(samples) if samples else 0.0
            history.append(avg_loss)
            current_lr *= decay_rate
            
            # Print progress
            if it % print_every == 0 or it == max_iters - 1:
                if samples:
                    print(f"Iter {it:4d}: loss = {avg_loss:.6e}, lr = {current_lr:.6f}")
                else:
                    print(f"Iter {it:4d}: skipped (no samples)")
            
            # Check convergence
            if samples and abs(prev_loss - avg_loss) < tol:
                print(f"Converged after {it+1} iterations")
                break
            prev_loss = avg_loss
        
        # Finalize training
        self.trained = True
        self.mean_L = sum(len(seq) for seq in seqs) / len(seqs) if seqs else 0
        self.mean_t = [0.0] * self.m
        count = 0
        for seq in seqs:
            descriptors = self.describe(seq)
            for desc in descriptors:
                for i in range(self.m):
                    self.mean_t[i] += desc[i]
                count += 1
        if count > 0:
            self.mean_t = [x / count for x in self.mean_t]
        
        return history

    def predict_t(self, vec_seq):
        """Predict target vector as average of all N(k) vectors"""
        N_list = self.describe(vec_seq)
        if not N_list:
            return [0.0] * self.m
        sum_t = [0.0] * self.m
        for Nk in N_list:
            for d in range(self.m):
                sum_t[d] += Nk[d]
        N = len(N_list)
        return [ti / N for ti in sum_t]    

    def generate(self, length, tau=0.0):
        """
        Generate a vector sequence using temperature-controlled sampling.
        
        Args:
            length (int): Desired sequence length
            tau (float): Temperature parameter
            
        Returns:
            list: Generated m-dimensional vector sequence
        """
        assert self.trained, "Model must be trained first"
        seq = []
        prev_vec = [random.uniform(-1, 1) for _ in range(self.n)]  # Initial input vector
        
        for k in range(length):
            j = k % self.L  # Basis index
            candidates = []
            scores = []
            
            # Generate candidate next vectors
            for _ in range(100):
                # Simple random walk generation for input vectors
                candidate = [prev_vec[d] + random.uniform(-0.2, 0.2) for d in range(self.n)]
                candidates.append(candidate)
                
                # Apply transformation: x' = M * candidate
                transformed = self._mat_vec(self.M, candidate)
                
                # Compute scalar: Bbasis[j] • transformed
                scalar = sum(self.Bbasis[j][i] * transformed[i] for i in range(self.m))
                
                # Compute Nk (m-dimensional descriptor)
                Nk = [self.Acoeff[i][j] * scalar for i in range(self.m)]
                
                # Calculate error to mean target
                error = 0.0
                for d in range(self.m):
                    diff = Nk[d] - self.mean_t[d]
                    error += diff * diff
                scores.append(-error)  # Higher score is better
            
            # Select next vector
            if tau == 0:  # Deterministic selection
                best_idx = scores.index(max(scores))
                next_vec = candidates[best_idx]
            else:  # Stochastic selection
                exp_scores = [math.exp(s / tau) for s in scores]
                sum_exp = sum(exp_scores)
                probs = [s / sum_exp for s in exp_scores]
                next_vec = random.choices(candidates, weights=probs, k=1)[0]
            
            # Store the descriptor Nk of the selected input vector
            transformed_next = self._mat_vec(self.M, next_vec)
            scalar_next = sum(self.Bbasis[j][i] * transformed_next[i] for i in range(self.m))
            Nk_next = [self.Acoeff[i][j] * scalar_next for i in range(self.m)]
            seq.append(Nk_next)
            
            prev_vec = next_vec  # Update for next iteration
        
        return seq

    # ---- feature extraction ----
    def dd_features(self, vec_seq, t=None):
        """
        Extract feature vector for a vector sequence.
        
        Features include:
          'd' : Deviation value
          'pwc': Flattened PWC coefficients
          'cwf': Flattened transformation matrix
          'frq': Position-weighted frequencies
          'pdv': Partial dual variables
          'all': Concatenated features
        """
        tg = t or self.predict_t(vec_seq)
        feats = {}
        
        # 1. Deviation value
        feats['d'] = [self.deviation([vec_seq], [tg])]
        
        # 2. PWC coefficients (Acoeff ⊗ Bbasis)
        A_backup = [row[:] for row in self.Acoeff]
        self.update_Acoeff([vec_seq], [tg])
        p_flat = []
        for i in range(self.m):
            for j in range(self.m):
                for g in range(self.L):
                    p_flat.append(self.Acoeff[i][g] * self.Bbasis[g][j])
        feats['pwc'] = p_flat
        self.Acoeff = A_backup
        
        # 3. Transformation matrix (flattened)
        M_backup = [row[:] for row in self.M]
        self.update_M([vec_seq], [tg])
        feats['cwf'] = [item for row in self.M for item in row]
        self.M = M_backup
        
        # 4. Position-weighted frequencies and partial dual variables
        frqs = []
        pdvs = []
        windows = self.extract_windows(vec_seq)
        L = len(windows)
        for g in range(self.L):      # Basis index
            for i in range(self.m):  # Vector dimension
                s_frq = 0.0
                s_pdv = 0.0
                for k, window_vec in enumerate(windows):
                    if (k % self.L) == g:
                        # Apply transformation: x' = M * window_vec
                        transformed = self._mat_vec(self.M, window_vec)
                        basis_val = self.Bbasis[g][i]
                        s_frq += basis_val
                        s_pdv += basis_val * transformed[i]
                frqs.append(s_frq / L if L > 0 else 0.0)
                pdvs.append(s_pdv / L if L > 0 else 0.0)
        feats['frq'] = frqs
        feats['pdv'] = pdvs
        
        # 5. Concatenated features
        feats['all'] = feats['d'] + feats['pwc'] + feats['cwf'] + feats['frq'] + feats['pdv']
        return feats

    # ---- show state ----
    def show(self, what=None, first_num=5):
        """Display model state information"""
        if what is None:
            what = ['params', 'stats']
        elif isinstance(what, str):
            what = [what]
        
        if 'all' in what:
            what = ['params', 'Acoeff', 'Bbasis', 'M', 'stats']
        
        print("NumDualDescriptorAB Model Status:")
        print("-" * 50)
        
        # 1. Configuration parameters
        if 'params' in what:
            print("[Configuration Parameters]")
            print(f"  Input dim (n)   : {self.n}")
            print(f"  Output dim (m)  : {self.m}")
            print(f"  L (basis dim)   : {self.L}")
            print(f"  rank (window)   : {self.rank}")
            print(f"  rank_mode       : {self.rank_mode}")
            print(f"  mode (processing): {self.mode}")
            print(f"  user_step       : {self.step}")
            print(f"  trained         : {self.trained}")
            print("-" * 50)
        
        # 2. Coefficient matrix
        if 'Acoeff' in what:
            print("[Coefficient Matrix Acoeff (partial)]")
            print(f"  Shape: {len(self.Acoeff)}x{len(self.Acoeff[0])} (m x L)")
            rows = min(first_num, len(self.Acoeff))
            cols = min(first_num, len(self.Acoeff[0]))
            for i in range(rows):
                row_preview = [f"{self.Acoeff[i][j]:.4f}" for j in range(cols)]
                print(f"  Row {i}: {row_preview}" + 
                      ("..." if len(self.Acoeff[0]) > cols else ""))
            print("-" * 50)
        
        # 3. Basis matrix
        if 'Bbasis' in what:
            print("[Basis Matrix Bbasis (partial)]")
            print(f"  Shape: {len(self.Bbasis)}x{len(self.Bbasis[0])} (L x m)")
            rows = min(first_num, len(self.Bbasis))
            cols = min(first_num, len(self.Bbasis[0]))
            for i in range(rows):
                row_preview = [f"{self.Bbasis[i][j]:.4f}" for j in range(cols)]
                print(f"  Row {i}: {row_preview}" + 
                      ("..." if len(self.Bbasis[0]) > cols else ""))
            print("-" * 50)
        
        # 4. Transformation matrix
        if 'M' in what:
            print("[Transformation Matrix M (partial)]")
            print(f"  Shape: {len(self.M)}x{len(self.M[0])} (m x n)")
            rows = min(first_num, len(self.M))
            cols = min(first_num, len(self.M[0]))
            for i in range(rows):
                row_preview = [f"{self.M[i][j]:.4f}" for j in range(cols)]
                print(f"  Row {i}: {row_preview}" + 
                      ("..." if len(self.M[0]) > cols else ""))
            print("-" * 50)
        
        # 5. Training statistics
        if 'stats' in what and self.trained:
            print("[Training Statistics]")
            print(f"  mean_L (avg seq len): {self.mean_L}")
            if hasattr(self, 'mean_t'):
                vec_preview = [f"{x:.4f}" for x in self.mean_t[:min(first_num, len(self.mean_t))]]
                print(f"  mean_t (avg target): {vec_preview}" + 
                      ("..." if len(self.mean_t) > first_num else ""))
            print("-" * 50)

    def count_parameters(self):
        """
        Calculate and print the number of learnable parameters in the model.        
        The model has three sets of learnable parameters:
        1. Acoeff matrix: m × L parameters
        2. M matrix: m × n parameters        
        Returns:
            int: Total number of learnable parameters
        """
        if self.L is None:
            print("Model not initialized. Please train or initialize first.")
            return 0            
        # Calculate parameter counts for each component
        a_params = self.m * self.L
        m_params = self.m * self.n        
        total_params = a_params + m_params        
        # Print parameter information
        print(f"Model Parameter Counts:")
        print(f"- Acoeff ({self.m}×{self.L}): {a_params:,} parameters")
        print(f"- M ({self.m}×{self.n}): {m_params:,} parameters")
        print(f"Total learnable parameters: {total_params:,}")        
        return total_params

    def save(self, filename):
        """Save model to file"""
        with open(filename, 'wb') as f:
            pickle.dump(self.__dict__, f)
        print(f"Model saved to {filename}")

    @classmethod
    def load(cls, filename):
        """Load model from file"""
        with open(filename, 'rb') as f:
            state = pickle.load(f)
        obj = cls.__new__(cls)
        obj.__dict__.update(state)
        print(f"Model loaded from {filename}")
        return obj

# === Example Usage ===
if __name__ == "__main__":

    from statistics import correlation, mean

    # Set random seed for reproducibility
    random.seed(3)
    
    # Configuration
    input_dim = 3    # Dimension of input vectors (n)
    output_dim = 5   # Dimension of output vectors (m)
    seq_count = 10   # Number of training sequences
    min_len = 100    # Minimum sequence length
    max_len = 200    # Maximum sequence length
    
    # Generate training data
    seqs = []  # List of n-dimensional vector sequences
    t_list = []  # List of m-dimensional target vectors
    
    print("Generating training data...")
    print(f"Input dimension: {input_dim}, Output dimension: {output_dim}")
    for i in range(seq_count):
        # Random sequence length
        length = random.randint(min_len, max_len)
        # Generate vector sequence: list of n-dimensional vectors
        sequence = []
        for _ in range(length):
            vector = [random.uniform(-1, 1) for _ in range(input_dim)]
            sequence.append(vector)
        seqs.append(sequence)
        
        # Generate random target vector (m-dimensional)
        target = [random.uniform(-1, 1) for _ in range(output_dim)]
        t_list.append(target)
        print(f"Sequence {i+1}: length={length}, target={[round(t,2) for t in target]}")
    
    # Initialize model
    print("\nInitializing NumDualDescriptorAB model...")
    dd = NumDualDescriptorAB(
        input_dim=input_dim,
        output_dim=output_dim,
        basis_dim=150,  # Basis dimension
        rank=1,       # Window size
        rank_mode='drop',
        mode='linear'
    )
    
    # Train model with alternating least squares
    print("\nTraining with ALS...")
    als_history = dd.train(
        seqs, 
        t_list,
        max_iters=20,
        tol=1e-12,
        print_every=2
    )
    dd.show(['params', 'stats'])
    dd.count_parameters()
    
    # Evaluate predictions
    print("\nEvaluating predictions...")
    pred_t_list = [dd.predict_t(seq) for seq in seqs]
    
    # Calculate correlations per dimension
    print("Prediction correlations per dimension:")
    dim_corrs = []
    for d in range(output_dim):
        actuals = [t[d] for t in t_list]
        preds = [p[d] for p in pred_t_list]
        corr = correlation(actuals, preds)
        dim_corrs.append(corr)
        print(f"  Dim {d}: correlation = {corr:.4f}")
    
    print(f"Average correlation: {mean(dim_corrs):.4f}")
    
    # Train with gradient descent
    print("\nTraining with Gradient Descent...")
    dd_grad = NumDualDescriptorAB(
        input_dim=input_dim,
        output_dim=output_dim,
        basis_dim=150,
        rank=1,
        rank_mode='drop',
        mode='linear'
    )
    
    gd_history = dd_grad.grad_train(
        seqs,
        t_list,
        learning_rate=0.01,
        max_iters=100,
        decay_rate=0.995,
        print_every=10
    )
    
    # Evaluate GD predictions
    pred_t_list_gd = [dd_grad.predict_t(seq) for seq in seqs]
    print("\nGD Prediction correlations per dimension:")
    gd_corrs = []
    for d in range(output_dim):
        actuals = [t[d] for t in t_list]
        preds = [p[d] for p in pred_t_list_gd]
        corr = correlation(actuals, preds)
        gd_corrs.append(corr)
        print(f"  Dim {d}: correlation = {corr:.4f}")
    
    print(f"GD Average correlation: {mean(gd_corrs):.4f}")
    
    # Self-supervised training (fixed)
    print("\nSelf-supervised training (auto-regressive)...")
    dd_auto = NumDualDescriptorAB(
        input_dim=input_dim,
        output_dim=output_dim,
        basis_dim=100,
        rank=1,
        rank_mode='drop',
        mode='linear'
    )
    
    auto_history = dd_auto.auto_train(
        seqs,
        auto_mode='reg',
        learning_rate=0.05,
        max_iters=100,
        decay_rate=0.99,
        print_every=10
    )
    
    # Generate new sequence (m-dimensional)
    print("\nGenerating new vector sequence (output dimension m)...")
    gen_seq = dd_auto.generate(length=20, tau=0.1)
    print("Generated sequence (first 5 vectors):")
    for i, vec in enumerate(gen_seq[:5]):
        print(f"  Vec {i}: {[round(v, 3) for v in vec]}")
    
    # Feature extraction
    print("\nExtracting features for first sequence...")
    feats = dd_auto.dd_features(seqs[0])
    print(f"Feature vector length: {len(feats['all'])}")
    print(f"First 10 features: {[round(f, 4) for f in feats['all'][:10]]}")
    
    # Save and load model
    print("\nTesting model persistence...")
    dd_auto.save("vector_model.pkl")
    dd_loaded = NumDualDescriptorAB.load("vector_model.pkl")
    print("Loaded model prediction for first sequence:")
    pred = dd_loaded.predict_t(seqs[0])
    print(f"  Predicted target: {[round(p, 4) for p in pred]}")
    
    print("\n=== Vector Sequence Processing Demo Completed ===")