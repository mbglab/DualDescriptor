# Copyright (C) 2005-2025, Bin-Guang Ma (mbg@mail.hzau.edu.cn); SPDX-License-Identifier: MIT
# The Numeric Dual Descriptor Vector class (Random AB matrix form) implemented with pure Python
# Modified to support n-dimensional input and l-dimensional output
# This program is for the demonstration of methodology and not fully refined.
# Author: Bin-Guang Ma (assisted by DeepSeek); Date: 2025-6-20

import math
import random
import pickle

class NumDualDescriptorAB:
    """
    Numeric Dual Descriptor for n-dimensional vector sequences with m-dimensional internal representation.
    Supports n-dimensional input and l-dimensional output.
    
    Model parameters:
      - M: model_dim × input_dim matrix (input transformation)
      - Acoeff: output_dim × bas_dim matrix (output coefficients)
      - Bbasis: bas_dim × model_dim matrix (internal basis)
    """
    
    def __init__(self, input_dim=4, output_dim=4, model_dim=4, basis_dim=50, rank=1, rank_op='avg', 
                 rank_mode='drop', mode='linear', user_step=None):
        """
        Initialize the Dual Descriptor for vector sequences.
        
        Args:
            input_dim (int): Dimension n of input vectors
            output_dim (int): Dimension l of output vectors
            model_dim (int): Dimension m of internal representation
            bas_dim (int): Basis dimension L
            rank (int): Window size for vector aggregation
            rank_mode (str): 'pad' or 'drop' for handling incomplete windows
            mode (str): 'linear' (sliding window) or 'nonlinear' (stepped window)
            user_step (int): Custom step size for nonlinear mode
        """
        self.n = input_dim    # Input dimension (n)
        self.l = output_dim   # Output dimension (l)
        self.m = model_dim    # Internal dimension (m)
        self.L = basis_dim      # Basis dimension (L)
        self.rank = rank
        self.rank_op = rank_op
        self.rank_mode = rank_mode
        assert mode in ('linear','nonlinear')
        self.mode = mode
        self.step = user_step
        self.trained = False
        
        # Initialize transformation matrix M (m×n): maps n-dim input to m-dim internal space
        self.M = [[random.uniform(-0.1, 0.1) for _ in range(self.n)] 
                  for _ in range(self.m)]
        
        # Initialize coefficient matrix Acoeff: l×L (maps to output dimension)
        self.Acoeff = [[random.uniform(-0.1, 0.1) for _ in range(self.L)]
                       for _ in range(self.l)]
        
        # Initialize basis matrix Bbasis: L×m
        self.Bbasis = [[random.uniform(-0.1, 0.1) for _ in range(self.m)]
                       for _ in range(self.L)]
        
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
        # Ensure dimensions match
        if len(M[0]) != len(v):
            raise ValueError(f"Matrix columns ({len(M[0])}) must match vector length ({len(v)})")
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
        """Compute l-dimensional descriptor vectors for each window position"""
        windows = self.extract_windows(vec_seq)
        N = []
        for k, window_vec in enumerate(windows):
            # Apply transformation: x' = M * window_vec (m-dim)
            transformed_vec = self._mat_vec(self.M, window_vec)
            
            j = k % self.L  # Basis index
            # Compute scalar: Bbasis[j] • transformed_vec
            scalar = sum(self.Bbasis[j][i] * transformed_vec[i] for i in range(self.m))
            
            # Compute Nk = scalar * Acoeff[:, j] (l-dim)
            Nk = [self.Acoeff[i][j] * scalar for i in range(self.l)]
            N.append(Nk)
        return N

    def S(self, vec_seq):
        """Compute cumulative descriptor vectors S(l) = ΣN(k) for k=1 to l (l-dim)"""
        Nk_list = self.describe(vec_seq)        
        S = [0.0] * self.l
        S_list = []
        for Nk in Nk_list:
            for i in range(self.l):
                S[i] += Nk[i]
            S_list.append(list(S))
        return S_list

    # ---- compute deviation ----
    def deviation(self, seqs, t_list):
        """Compute mean squared error between descriptors and l-dimensional targets"""
        tot = 0.0
        cnt = 0
        for seq, t in zip(seqs, t_list):
            for Nk in self.describe(seq):
                for i in range(self.l):
                    diff = Nk[i] - t[i]
                    tot += diff * diff
                cnt += 1
        return tot / cnt if cnt else 0.0

    # ---- update Acoeff ----
    def update_Acoeff(self, seqs, t_list):
        """Update coefficient matrix using position-specific projections"""
        l, L = self.l, self.L
        # Initialize accumulators
        numerator = [[0.0] * L for _ in range(l)]  # l x L
        denominator = [0.0] * L  # L-dimensional
        
        for seq, t in zip(seqs, t_list):
            windows = self.extract_windows(seq)
            for k, window_vec in enumerate(windows):
                idx = k % L  # Basis index
                # Apply transformation: x' = M * window_vec (m-dim)
                transformed_vec = self._mat_vec(self.M, window_vec)
                
                # Compute scalar: Bbasis[idx] • transformed_vec
                scalar = sum(self.Bbasis[idx][i] * transformed_vec[i] for i in range(self.m))
                
                # Update accumulators
                for i in range(l):
                    numerator[i][idx] += scalar * t[i]
                denominator[idx] += scalar * scalar  # scalar²
        
        # Update Acoeff column-wise
        for j in range(L):
            if abs(denominator[j]) > 1e-12:  # Avoid division by zero
                inv_denom = 1.0 / denominator[j]
                for i in range(l):
                    self.Acoeff[i][j] = numerator[i][j] * inv_denom

    def update_Bbasis(self, seqs, t_list):
        """Update basis matrix using position-specific least squares"""
        l, m, L = self.l, self.m, self.L        
        # Initialize accumulators for each basis index
        M_list = [[[0.0] * m for _ in range(m)] for _ in range(L)]  # L x m x m matrices
        v_list = [[0.0] * m for _ in range(L)]  # L x m vectors
        
        # Collect data grouped by basis index j
        for seq, t in zip(seqs, t_list):
            windows = self.extract_windows(seq)
            for k, window_vec in enumerate(windows):
                j = k % L  # Basis index
                # Apply transformation: x' = M * window_vec (m-dim)
                transformed_vec = self._mat_vec(self.M, window_vec)
                A_j = [self.Acoeff[i][j] for i in range(l)]  # A_j: j-th column of Acoeff
                
                # Compute ||A_j||^2 = A_j • A_j
                a_norm_sq = sum(a*a for a in A_j)
                
                # Update M_j: Σ [x' x'^T * ||A_j||^2]
                for r in range(m):
                    for s in range(m):
                        M_list[j][r][s] += a_norm_sq * transformed_vec[r] * transformed_vec[s]
                
                # Update v_j: Σ [<A_j, t> * x']
                a_dot_t = sum(A_j[i] * t[i] for i in range(l))
                for i in range(m):
                    v_list[j][i] += a_dot_t * transformed_vec[i]
        
        # Update each basis vector b_j (j-th row of Bbasis)
        for j in range(L):
            M = M_list[j]
            v = v_list[j]
            
            try:
                # Solve linear system: M * b_j = v
                M_inv = self._invert(M)
                b_j = self._mat_vec(M_inv, v)
                
                # Update j-th row of Bbasis
                for i in range(m):
                    self.Bbasis[j][i] = b_j[i]
            except:  # Fallback to current values on singular matrix
                pass
        
        # Update transpose cache
        self.B_t = self._transpose(self.Bbasis)

    # ---- update M ----
    def update_M(self, seqs, t_list):
        """Update transformation matrix using position-specific least squares"""
        l, m, n = self.l, self.m, self.n
        # Initialize (m*n) × (m*n) matrix and (m*n) vector
        M_mat = [[0.0] * (m*n) for _ in range(m*n)]
        v_vec = [0.0] * (m*n)

        # Process each sequence and window
        for seq, t in zip(seqs, t_list):
            windows = self.extract_windows(seq)
            for k, x_k in enumerate(windows):
                j = k % self.L # Basis index
                A_col = [self.Acoeff[i][j] for i in range(l)]
                b_j = self.Bbasis[j]
                
                # Compute scalar constants
                a_norm_sq = sum(a*a for a in A_col)
                a_dot_t = sum(A_col[i] * t[i] for i in range(l))
                
                # Compute outer product matrix: x_k ⊗ b_j
                # Corrected index calculation
                for r in range(m):
                    for s in range(n):
                        idx1 = r * n + s  # Row index in M_mat
                        # Compute contribution for each element in M matrix
                        for p in range(m):
                            for q in range(n):
                                idx2 = p * n + q  # Column index in M_mat
                                # Only add when the row and column correspond to the same position
                                if r == p:  # Only when the row indices match
                                    M_mat[idx1][idx2] += a_norm_sq * b_j[r] * b_j[p] * x_k[s] * x_k[q]
                        # Update v_vec
                        v_vec[idx1] += a_dot_t * b_j[r] * x_k[s]
        
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
            seqs: List of training vector sequences (n-dim)
            t_list: List of target vectors (l-dim)
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
            self.update_Bbasis(seqs, t_list)
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
        self.mean_t = [0.0] * self.l
        for t in t_list:
            for i in range(self.l):
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
            seqs: List of training vector sequences (n-dim)
            t_list: List of target vectors (l-dim)
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
                           for _ in range(self.l)]
            self.Bbasis = [[random.uniform(-0.1, 0.1) for _ in range(self.m)]
                           for _ in range(self.L)]
        
        total_positions = sum(len(self.extract_windows(seq)) for seq in seqs)
        if total_positions == 0:
            raise ValueError("No valid windows in sequences")
        
        history = []
        D_prev = float('inf')
        current_lr = learning_rate
        
        for it in range(max_iters):
            # Initialize gradients
            grad_A = [[0.0] * self.L for _ in range(self.l)]
            grad_B = [[0.0] * self.m for _ in range(self.L)]
            grad_M = [[0.0] * self.n for _ in range(self.m)]
            total_loss = 0.0
            
            # Process each sequence
            for seq, t in zip(seqs, t_list):
                windows = self.extract_windows(seq)
                for k, window_vec in enumerate(windows):
                    j = k % self.L  # Basis index
                    
                    # Apply transformation: x' = M * window_vec (m-dim)
                    transformed_vec = self._mat_vec(self.M, window_vec)
                    
                    # Compute scalar: B_j • x'
                    scalar = sum(self.Bbasis[j][i] * transformed_vec[i] for i in range(self.m))
                    
                    # Compute Nk = scalar * A_j (l-dim)
                    A_j = [self.Acoeff[i][j] for i in range(self.l)]
                    Nk = [A_j[i] * scalar for i in range(self.l)]
                    
                    # Compute position loss
                    pos_loss = 0.0
                    dNk = [0.0] * self.l
                    for i in range(self.l):
                        error = Nk[i] - t[i]
                        pos_loss += error * error
                        dNk[i] = 2 * error / self.l  # Gradient w.r.t Nk
                    total_loss += pos_loss / self.l
                    
                    # Compute gradients
                    for i in range(self.l):
                        # Gradient for Acoeff
                        grad_A[i][j] += dNk[i] * scalar
                        
                        # Common factor
                        factor = dNk[i] * A_j[i]
                        
                        # Gradient for Bbasis
                        for d in range(self.m):
                            grad_B[j][d] += factor * transformed_vec[d]
                            
                        # Gradient for M (through transformed_vec)
                        for d in range(self.m):
                            for e in range(self.n):
                                # ∂(transformed_vec[d])/∂M[d][e] = window_vec[e]
                                grad_M[d][e] += factor * self.Bbasis[j][d] * window_vec[e]
            
            # Update parameters with gradients
            for i in range(self.l):
                for j in range(self.L):
                    self.Acoeff[i][j] -= current_lr * grad_A[i][j]
            
            for j in range(self.L):
                for d in range(self.m):
                    self.Bbasis[j][d] -= current_lr * grad_B[j][d]
            
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
        self.mean_t = [0.0] * self.l
        for t in t_list:
            for i in range(self.l):
                self.mean_t[i] += t[i]
        self.mean_t = [x / len(t_list) for x in self.mean_t]
        self.trained = True
        return history

    def auto_train(self, seqs, auto_mode='reg', max_iters=1000, tol=1e-8,
                   learning_rate=0.01, continued=False, decay_rate=0.99,
                   print_every=10):
        """
        Self-supervised training for n-dimensional vector sequences.
        
        Args:
            seqs: List of training vector sequences (n-dim)
            auto_mode: 'reg' (predict next window) only
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
                           for _ in range(self.l)]
            self.Bbasis = [[random.uniform(-0.1, 0.1) for _ in range(self.m)]
                           for _ in range(self.L)]
        
        # Prepare training samples: (sequence, window_index, next_window_vector)
        samples = []
        for seq in seqs:
            windows = self.extract_windows(seq)
            if len(windows) < 2:
                continue
                
            if auto_mode == 'reg':
                # Predict next window from current window
                for k in range(len(windows) - 1):
                    samples.append((seq, k, windows[k+1]))
        
        if not samples:
            print("Warning: No training samples generated")
            return []
        
        history = []
        prev_loss = float('inf')
        current_lr = learning_rate
        
        for it in range(max_iters):
            total_loss = 0.0
            grad_A = [[0.0] * self.L for _ in range(self.l)]
            grad_B = [[0.0] * self.m for _ in range(self.L)]
            grad_M = [[0.0] * self.n for _ in range(self.m)]
            
            for seq, k, next_window in samples:
                windows = self.extract_windows(seq)
                if k >= len(windows):
                    continue
                    
                current_window = windows[k]
                j = k % self.L  # Basis index
                
                # Apply transformation: x' = M * current_window (m-dim)
                transformed_vec = self._mat_vec(self.M, current_window)
                
                # Compute scalar: B_j • x'
                scalar = sum(self.Bbasis[j][i] * transformed_vec[i] for i in range(self.m))
                
                # Compute Nk = scalar * A_j (l-dim)
                A_j = [self.Acoeff[i][j] for i in range(self.l)]
                Nk = [A_j[i] * scalar for i in range(self.l)]
                
                # Compute loss to next window vector (n-dim)
                loss_val = 0.0
                dNk = [0.0] * self.n
                for i in range(min(self.l, self.n)):
                    error = Nk[i] - next_window[i]
                    loss_val += error * error
                    dNk[i] = 2 * error / self.n
                total_loss += loss_val / self.n
                
                # Compute gradients
                for i in range(min(self.l, self.n)):
                    # Gradient for Acoeff
                    grad_A[i][j] += dNk[i] * scalar
                    
                    # Common factor
                    factor = dNk[i] * A_j[i]
                    
                    # Gradient for Bbasis
                    for d in range(self.m):
                        grad_B[j][d] += factor * transformed_vec[d]
                        
                    # Gradient for M
                    for d in range(self.m):
                        for e in range(self.n):
                            grad_M[d][e] += factor * self.Bbasis[j][d] * current_window[e]
            
            # Update parameters
            for i in range(self.l):
                for j in range(self.L):
                    self.Acoeff[i][j] -= current_lr * grad_A[i][j]
            
            for j in range(self.L):
                for d in range(self.m):
                    self.Bbasis[j][d] -= current_lr * grad_B[j][d]
            
            for d in range(self.m):
                for e in range(self.n):
                    self.M[d][e] -= current_lr * grad_M[d][e]
            
            # Calculate average loss
            avg_loss = total_loss / len(samples)
            history.append(avg_loss)
            current_lr *= decay_rate
            
            # Print progress
            if it % print_every == 0 or it == max_iters - 1:
                print(f"Iter {it:4d}: loss = {avg_loss:.6e}, lr = {current_lr:.6f}")
            
            # Check convergence
            if prev_loss - avg_loss < tol:
                print(f"Converged after {it+1} iterations")
                break
            prev_loss = avg_loss
        
        # Finalize training
        self.trained = True
        self.mean_L = sum(len(seq) for seq in seqs) / len(seqs)
        self.mean_t = [0.0] * self.l
        count = 0
        for seq in seqs:
            windows = self.extract_windows(seq)
            for window in windows:
                for i in range(min(self.l, len(window))):
                    self.mean_t[i] += window[i]
                count += 1
        if count > 0:
            self.mean_t = [x / count for x in self.mean_t]
        
        return history

    def predict_t(self, vec_seq):
        """Predict target vector as average of all N(k) vectors (l-dimensional)"""
        N_list = self.describe(vec_seq)
        if not N_list:
            return [0.0] * self.l
        sum_t = [0.0] * self.l
        for Nk in N_list:
            for d in range(self.l):
                sum_t[d] += Nk[d]
        N = len(N_list)
        return [ti / N for ti in sum_t]    

    def generate(self, length, init_vec=None, tau=0.0):
        """
        Generate l-dimensional vector sequence using temperature-controlled sampling.
        
        Args:
            length (int): Desired sequence length
            init_vec (list): Starting vector (n-dimensional, random if None)
            tau (float): Temperature for randomness
            
        Returns:
            list: Generated vector sequence (each vector l-dimensional)
        """
        assert self.trained, "Model must be trained first"
        # Initialize with n-dimensional input
        if init_vec is None:
            current_input = [random.uniform(-1, 1) for _ in range(self.n)]
        else:
            # Ensure input is n-dimensional
            current_input = init_vec[:self.n]  
            if len(current_input) < self.n:
                current_input += [0.0] * (self.n - len(current_input))
        
        seq = []
        
        # Generate sequence of l-dimensional vectors
        for k in range(length):
            # Get position index (mod L)
            j = k % self.L
            
            # Apply transformation: M * current_input → m-dim
            transformed = self._mat_vec(self.M, current_input)
            
            # Compute scalar: Bbasis[j] • transformed
            scalar = sum(self.Bbasis[j][i] * transformed[i] for i in range(self.m))
            
            # Compute base next vector: scalar * Acoeff[:,j] → l-dim
            base_next = [scalar * self.Acoeff[i][j] for i in range(self.l)]
            
            # Apply temperature-controlled noise
            if tau > 0:
                output_vec = [v + random.gauss(0, tau) for v in base_next]
            else:
                output_vec = base_next
                
            seq.append(output_vec)
            
            # For next iteration, use the n-dimensional input again
            # (we don't feed output back as input since dimensions differ)
            # Option: Add small variation to input for sequence progression
            if k > 0:
                current_input = [x + random.gauss(0, 0.01) for x in current_input]
        
        return seq

    # ---- feature extraction ----
    def dd_features(self, vec_seq, t=None):
        """
        Extract feature vector for a vector sequence.
        
        Features include:
          'd' : Deviation value
          'pwc': Flattened PWC coefficients (Acoeff ⊗ Bbasis)
          'cwf': Flattened transformation matrix M
          'frq': Position-weighted frequencies
          'pdv': Partial dual variables
          'all': Concatenated features
        
        Args:
            vec_seq (list): Sequence of n-dimensional vectors
            t (list, optional): Target vector. If None, uses predicted target.
            
        Returns:
            dict: Dictionary of feature vectors
        """
        tg = t or self.predict_t(vec_seq)
        feats = {}
        
        # 1. Deviation value
        feats['d'] = [self.deviation([vec_seq], [tg])]
        
        # 2. PWC coefficients (Acoeff ⊗ Bbasis)
        A_backup = [row[:] for row in self.Acoeff]
        B_backup = [row[:] for row in self.Bbasis]
        self.update_Acoeff([vec_seq], [tg])
        self.update_Bbasis([vec_seq], [tg])
        p_flat = []
        for i in range(self.l):
            for j in range(self.m):
                for g in range(self.L):
                    p_flat.append(self.Acoeff[i][g] * self.Bbasis[g][j])
        feats['pwc'] = p_flat
        self.Acoeff = A_backup
        self.Bbasis = B_backup
        
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
        
        # Corrected: iterate over basis index and internal dimension (m)
        for g in range(self.L):      # Basis index (0 to L-1)
            for i in range(self.m):  # Internal dimension (0 to m-1)
                s_frq = 0.0
                s_pdv = 0.0
                for k, window_vec in enumerate(windows):
                    if (k % self.L) == g:
                        # Apply transformation: x' = M * window_vec (m-dim)
                        transformed = self._mat_vec(self.M, window_vec)
                        basis_val = self.Bbasis[g][i]  # Basis value for dimension i
                        s_frq += basis_val
                        s_pdv += basis_val * transformed[i]  # Transformed value at dimension i
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
            print(f"  Input dimension (n): {self.n}")
            print(f"  Output dimension (l): {self.l}")
            print(f"  Internal dimension (m): {self.m}")
            print(f"  Basis dimension (L): {self.L}")
            print(f"  rank (window)    : {self.rank}")
            print(f"  rank_mode        : {self.rank_mode}")
            print(f"  mode (processing): {self.mode}")
            print(f"  user_step        : {self.step}")
            print(f"  trained          : {self.trained}")
            print("-" * 50)
        
        # 2. Coefficient matrix
        if 'Acoeff' in what:
            print("[Coefficient Matrix Acoeff (l×L) (partial)]")
            print(f"  Shape: {len(self.Acoeff)}x{len(self.Acoeff[0])}")
            rows = min(first_num, len(self.Acoeff))
            cols = min(first_num, len(self.Acoeff[0]))
            for i in range(rows):
                row_preview = [f"{self.Acoeff[i][j]:.4f}" for j in range(cols)]
                print(f"  Row {i}: {row_preview}" + 
                      ("..." if len(self.Acoeff[0]) > cols else ""))
            print("-" * 50)
        
        # 3. Basis matrix
        if 'Bbasis' in what:
            print("[Basis Matrix Bbasis (L×m) (partial)]")
            print(f"  Shape: {len(self.Bbasis)}x{len(self.Bbasis[0])}")
            rows = min(first_num, len(self.Bbasis))
            cols = min(first_num, len(self.Bbasis[0]))
            for i in range(rows):
                row_preview = [f"{self.Bbasis[i][j]:.4f}" for j in range(cols)]
                print(f"  Row {i}: {row_preview}" + 
                      ("..." if len(self.Bbasis[0]) > cols else ""))
            print("-" * 50)
        
        # 4. Transformation matrix
        if 'M' in what:
            print("[Transformation Matrix M (m×n) (partial)]")
            print(f"  Shape: {len(self.M)}x{len(self.M[0])}")
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
        1. M matrix: m × n parameters
        2. Acoeff matrix: l × L parameters
        3. Bbasis matrix: L × m parameters
        Returns:
            int: Total number of learnable parameters
        """
        if not hasattr(self, 'n'):
            print("Model not initialized. Please train or initialize first.")
            return 0            
        # Calculate parameter counts for each component
        m_params = self.m * self.n
        a_params = self.l * self.L
        b_params = self.L * self.m        
        total_params = m_params + a_params + b_params        
        # Print parameter information
        print(f"Model Parameter Counts:")
        print(f"- M ({self.m}×{self.n}): {m_params:,} parameters")
        print(f"- Acoeff ({self.l}×{self.L}): {a_params:,} parameters")
        print(f"- Bbasis ({self.L}×{self.m}): {b_params:,} parameters")
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
    random.seed(1)
    
    # Configuration: n=5 input dim, l=3 output dim, m=10 internal dim
    n_dim = 5   # Input vector dimension
    l_dim = 3   # Output vector dimension
    m_dim = 10  # Internal model dimension
    num_seqs = 10
    min_len, max_len = 100, 200

    # Generate synthetic training data (n-dimensional input, l-dimensional targets)
    print("Generating training data...")
    seqs = []   # List of vector sequences (n-dim)
    t_list = [] # List of target vectors (l-dim)
    
    for i in range(num_seqs):
        # Random sequence length
        length = random.randint(min_len, max_len)
        # Generate vector sequence: list of n-dimensional vectors
        sequence = []
        for _ in range(length):
            vector = [random.uniform(-1, 1) for _ in range(n_dim)]
            sequence.append(vector)
        seqs.append(sequence)
        
        # Generate random target vector (l-dim)
        target = [random.uniform(-1, 1) for _ in range(l_dim)]
        t_list.append(target)
        print(f"Sequence {i+1}: length={length}, input_dim={n_dim}, output_dim={l_dim}")

    # Initialize model with input_dim=5, output_dim=3, model_dim=10, L=150
    print("\nInitializing NumDualDescriptorAB model...")
    dd = NumDualDescriptorAB(
        input_dim=n_dim,
        output_dim=l_dim,
        model_dim=m_dim,
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
        tol=1e-8,
        print_every=1
    )
    dd.show(['params', 'stats'])
    dd.count_parameters()
    
    # Evaluate predictions
    print("\nEvaluating predictions...")
    pred_t_list = [dd.predict_t(seq) for seq in seqs]
    
    # Calculate correlations per output dimension
    print("Prediction correlations per output dimension:")
    dim_corrs = []
    for d in range(l_dim):
        actuals = [t[d] for t in t_list]
        preds = [p[d] for p in pred_t_list]
        corr = correlation(actuals, preds) if len(actuals) > 1 else 0.0
        dim_corrs.append(corr)
        print(f"  Dim {d}: correlation = {corr:.4f}")
    
    print(f"Average correlation: {mean(dim_corrs):.4f}")
    
    # Train with gradient descent
    print("\nTraining with Gradient Descent...")
    dd_grad = NumDualDescriptorAB(
        input_dim=n_dim,
        output_dim=l_dim,
        model_dim=m_dim,
        basis_dim=150,
        rank=1,
        rank_mode='drop',
        mode='linear'
    )
    
    gd_history = dd_grad.grad_train(
        seqs,
        t_list,
        learning_rate=0.1,
        max_iters=150,
        decay_rate=0.999,
        print_every=10
    )
    
    # Evaluate GD predictions
    pred_t_list_gd = [dd_grad.predict_t(seq) for seq in seqs]
    print("\nGD Prediction correlations per output dimension:")
    gd_corrs = []
    for d in range(l_dim):
        actuals = [t[d] for t in t_list]
        preds = [p[d] for p in pred_t_list_gd]
        corr = correlation(actuals, preds) if len(actuals) > 1 else 0.0
        gd_corrs.append(corr)
        print(f"  Dim {d}: correlation = {corr:.4f}")
    
    print(f"GD Average correlation: {mean(gd_corrs):.4f}")
    
    # Self-supervised training
    print("\nSelf-supervised training (auto-regressive)...")
    dd_auto = NumDualDescriptorAB(
        input_dim=n_dim,
        output_dim=l_dim,
        model_dim=m_dim,
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
    
    # Generate new sequence (l-dimensional output)
    print("\nGenerating new vector sequence...")
    gen_seq = dd_auto.generate(
        length=10,
        init_vec=seqs[0][0],  # Start with first vector of first sequence
        tau=0.05  # Small randomness
    )
    print("Generated sequence (first 3 vectors):")
    for i, vec in enumerate(gen_seq[:3]):
        print(f"  Vec {i}: {[round(v, 3) for v in vec]}")
    
    # Feature extraction
    print("\nExtracting features for first sequence...")
    feats = dd_auto.dd_features(seqs[0])
    print(f"Feature vector length: {len(feats['all'])}")
    print(f"First 5 features: {feats['all'][:5]}")
    
    # Save and load model
    print("\nTesting model persistence...")
    dd_auto.save("nDDAB_model.pkl")
    dd_loaded = NumDualDescriptorAB.load("nDDAB_model.pkl")
    print("Loaded model prediction for first sequence:")
    pred = dd_loaded.predict_t(seqs[0])
    print(f"  Predicted target: {[round(p, 4) for p in pred]}")
    
    print("\n=== n-Dimensional Input to l-Dimensional Output Processing Demo Completed ===")
