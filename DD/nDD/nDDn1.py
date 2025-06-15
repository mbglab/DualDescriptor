# Copyright (C) 2005-2025, Bin-Guang Ma (mbg@mail.hzau.edu.cn). All rights reserved.
# The Dual Descriptor Vector class
# Author: Bin-Guang Ma; Date: 2025-6-12
# Modified for n-dimensional output and target transformation

import math
import random
import pickle

class NumDualDescriptor:
    """
    Numeric Dual Descriptor for vector sequences with:
      - Learnable coefficient matrix Acoeff ∈ R^{m×L}
      - Learnable basis matrix Bbasis ∈ R^{L×m}
      - Learnable transformation matrix M ∈ R^{m×n} (maps n-dim inputs to m-dim space)
    """
    def __init__(self, model_dim=4, input_dim=4, rank=1, rank_mode='drop', rank_op='avg', mode='linear', user_step=None):
        """
        Initialize the Dual Descriptor for vector sequences.
        
        Args:
            model_dim (int): Internal dimensionality of vectors (m)
            input_dim (int): Dimensionality of input vectors (n)
            rank (int): Parameter controlling step size in nonlinear mode
            mode (str): 'linear' or 'nonlinear' processing mode
            user_step (int): Custom step size for nonlinear mode
        """
        self.rank = rank
        self.rank_mode = rank_mode
        self.rank_op = rank_op
        self.m = model_dim  # Internal dimension
        self.n = input_dim  # Input dimension
        assert mode in ('linear', 'nonlinear')
        self.mode = mode
        self.step = user_step
        self.trained = False
        
        # Initialize transformation matrix M (m×n)
        self.M = [[random.uniform(-0.5, 0.5) for _ in range(self.n)] 
                  for _ in range(self.m)]
        
        # To be initialized in train()
        self.L = None      # Maximum sequence length
        self.Acoeff = None  # m×L coefficient matrix
        self.Bbasis = None  # L×m basis matrix
        self.B_t = None     # m×L (transpose of Bbasis)

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
        """Gauss-Jordan inversion of n×n matrix"""
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
    def extract_vectors(self, seq):
        """
        Extract vectors from sequence based on processing mode and rank operation.
        
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

    # ---- initialization ----
    def initialize(self, seqs):
        """Initialize model parameters based on training sequences"""
        if self.L is None:
            self.L = max(len(s) for s in seqs)
        # Initialize Acoeff (m×L)
        self.Acoeff = [[random.uniform(-0.1, 0.1) for _ in range(self.L)]
                       for _ in range(self.m)]
        # Initialize Bbasis (L×m)
        self.Bbasis = [[random.uniform(-0.1, 0.1) for _ in range(self.m)]
                       for _ in range(self.L)]
        # Cache transpose
        self.B_t = self._transpose(self.Bbasis)

    # ---- describe sequence ----
    def describe(self, seq):
        """
        Compute descriptor vectors for each position in the sequence.
        
        Args:
            seq: List of n-dimensional vectors
            
        Returns:
            list: Descriptor vectors (N) for each position (m-dimensional)
        """
        vecs = self.extract_vectors(seq)
        N = []
        for k, v in enumerate(vecs):
            if k >= self.L:
                break
            # Transform input vector: x = M * v (map n-dim → m-dim)
            x = self._mat_vec(self.M, v)
            # Compute z = Bbasis * x
            z = [sum(self.Bbasis[j][i] * x[i] for i in range(self.m))
                 for j in range(self.L)]
            # Compute Nk = Acoeff * z
            Nk = [sum(self.Acoeff[i][j] * z[j] for j in range(self.L))
                  for i in range(self.m)]
            N.append(Nk)
        return N

    def S(self, seq):
        """
        Compute cumulative descriptor vectors S(l) = ΣN(k) for k=1 to l.
        
        Args:
            seq: List of n-dimensional vectors
            
        Returns:
            list: Cumulative descriptor vectors (m-dimensional)
        """
        Nk_list = self.describe(seq)
        S = [0.0] * self.m
        S_list = []
        for l in range(len(Nk_list)):
            for i in range(self.m):
                S[i] += Nk_list[l][i]
            S_list.append(S[:])  # Append a copy
        return S_list

    # ---- compute deviation ----
    def deviation(self, seqs, t_list):
        """
        Compute mean squared error between descriptors and targets.
        
        Args:
            seqs: List of vector sequences (each vector n-dimensional)
            t_list: List of target vectors (n-dimensional)
            
        Returns:
            float: Mean squared error
        """
        total_error = 0.0
        position_count = 0
        
        for seq, t in zip(seqs, t_list):
            # Transform target vector to model space: t_m = M * t
            t_m = self._mat_vec(self.M, t)
            
            for Nk in self.describe(seq):
                for i in range(self.m):
                    error = Nk[i] - t_m[i]
                    total_error += error * error
                position_count += 1
                
        return total_error / position_count if position_count else 0.0

    # ---- update Acoeff ----
    def update_Acoeff(self, seqs, t_list):
        """
        Update Acoeff using closed-form least squares solution.
        
        Args:
            seqs: List of training sequences (n-dimensional vectors)
            t_list: List of target vectors (n-dimensional)
        """
        L, m = self.L, self.m
        U = [[0.0] * L for _ in range(L)]  # L×L
        V = [[0.0] * L for _ in range(m)]  # m×L
        
        for seq, t in zip(seqs, t_list):
            # Transform target vector to model space: t_m = M * t
            t_m = self._mat_vec(self.M, t)
            
            vecs = self.extract_vectors(seq)
            for v in vecs[:L]:
                # Transform vector: x = M * v (n-dim → m-dim)
                x = self._mat_vec(self.M, v)
                # Compute z = Bbasis * x
                z = [sum(self.Bbasis[j][i] * x[i] for i in range(m))
                     for j in range(L)]
                # Accumulate U and V
                for i in range(L):
                    for j in range(L):
                        U[i][j] += z[i] * z[j]
                for i in range(m):
                    for j in range(L):
                        V[i][j] += t_m[i] * z[j]
        
        U_inv = self._invert(U)
        self.Acoeff = self._mat_mul(V, U_inv)

    def update_Bbasis(self, seqs, t_list):
        """
        Update Bbasis using closed-form least squares solution.
        
        Args:
            seqs: List of training sequences (n-dimensional vectors)
            t_list: List of target vectors (n-dimensional)
        """
        L, m = self.L, self.m
        reg = 1e-8  # Regularization factor
        
        # Initialize accumulation matrices
        B_mat = [[0.0] * m for _ in range(m)]  # m×m
        TXT = [[0.0] * m for _ in range(m)]    # m×m
        
        for seq, t in zip(seqs, t_list):
            # Transform target vector to model space: t_m = M * t
            t_m = self._mat_vec(self.M, t)
            
            vecs = self.extract_vectors(seq)
            for v in vecs[:L]:
                # Transform vector: x = M * v (n-dim → m-dim)
                x = self._mat_vec(self.M, v)
                # Update B matrix: x * x^T
                for i in range(m):
                    for j in range(m):
                        B_mat[i][j] += x[i] * x[j]
                # Update TXT matrix: t_m * x^T
                for i in range(m):
                    for j in range(m):
                        TXT[i][j] += t_m[i] * x[j]
        
        # Compute A = Acoeff^T * Acoeff (L×L)
        Acoeff_t = self._transpose(self.Acoeff)
        A = self._mat_mul(Acoeff_t, self.Acoeff)
        
        # Compute C = Acoeff^T * TXT (L×m)
        C = self._mat_mul(Acoeff_t, TXT)
        
        # Apply regularization
        for i in range(L):
            A[i][i] += reg
        for i in range(m):
            B_mat[i][i] += reg
        
        # Invert matrices
        A_inv = self._invert(A)
        B_inv = self._invert(B_mat)
        
        # Compute new Bbasis = A_inv * C * B_inv
        CB_inv = self._mat_mul(C, B_inv)
        Bbasis_new = self._mat_mul(A_inv, CB_inv)
        
        # Update parameters
        self.Bbasis = Bbasis_new
        self.B_t = self._transpose(self.Bbasis)

    # ---- update transformation matrix M ----
    def update_M(self, seqs, t_list):
        """
        Update transformation matrix M using least squares.
        Handles n-dimensional inputs and m-dimensional targets.
        
        Args:
            seqs: List of training sequences (n-dimensional vectors)
            t_list: List of target vectors (n-dimensional)
        """
        L, m, n = self.L, self.m, self.n
        reg = 1e-8
        
        # Precompute C = Acoeff * Bbasis (m×m)
        C = self._mat_mul(self.Acoeff, self.Bbasis)
        C_t = self._transpose(C)
        
        # Compute left side: C^T * C (m×m)
        left = self._mat_mul(C_t, C)
        
        # Initialize accumulation matrices
        sum_vvT = [[0.0] * n for _ in range(n)]  # n×n
        sum_tvT = [[0.0] * n for _ in range(m)]   # m×n
        
        for seq, t in zip(seqs, t_list):
            # Transform target vector to model space: t_m = M * t
            t_m = self._mat_vec(self.M, t)
            
            vecs = self.extract_vectors(seq)
            for v in vecs[:L]:
                # Compute v * v^T (n×n)
                for i in range(n):
                    for j in range(n):
                        sum_vvT[i][j] += v[i] * v[j]
                # Compute t_m * v^T (m×n)
                for i in range(m):
                    for j in range(n):
                        sum_tvT[i][j] += t_m[i] * v[j]
        
        # Apply regularization
        for i in range(n):
            sum_vvT[i][i] += reg
        
        # Compute right side: C^T * (sum_tvT) * sum_vvT^{-1}
        sum_vvT_inv = self._invert(sum_vvT)
        Ct_sum_tvT = self._mat_mul(C_t, sum_tvT)
        right = self._mat_mul(Ct_sum_tvT, sum_vvT_inv)
        
        # Solve left * M = right
        left_inv = self._invert(left)
        self.M = self._mat_mul(left_inv, right)

    # ---- training loop ----
    def train(self, seqs, t_list, max_iters=10, tol=1e-8):
        """
        Train model using alternating least squares.
        
        Args:
            seqs: List of training sequences (n-dimensional vectors)
            t_list: List of target vectors (n-dimensional)
            max_iters: Maximum iterations
            tol: Convergence tolerance
            
        Returns:
            list: Training history (MSE values)
        """
        self.initialize(seqs)
        prev_dev = float('inf')
        history = []
        
        for it in range(max_iters):
            self.update_Acoeff(seqs, t_list)
            self.update_Bbasis(seqs, t_list)
            self.update_M(seqs, t_list)
            
            dev = self.deviation(seqs, t_list)
            history.append(dev)
            print(f"Iter {it:2d}: MSE = {dev:.6e}")
            
            if prev_dev - dev < tol:
                print("Converged.")
                break
            prev_dev = dev
        
        # Compute statistics
        self.mean_L = int(sum(len(seq) for seq in seqs) / len(seqs))
        self.mean_t = [0.0] * self.n
        for t in t_list:
            for i in range(self.n):
                self.mean_t[i] += t[i]
        self.mean_t = [x / len(t_list) for x in self.mean_t]
        self.trained = True
        
        return history

    def grad_train(self, seqs, t_list, learning_rate=0.01, max_iters=100, 
                  tol=1e-9, print_every=10):
        """
        Train model using gradient descent.
        
        Args:
            seqs: List of training sequences (n-dimensional vectors)
            t_list: List of target vectors (n-dimensional)
            learning_rate: Step size
            max_iters: Maximum iterations
            tol: Convergence tolerance
            print_every: Print interval
            
        Returns:
            list: Training loss history
        """
        self.initialize(seqs)
        history = []
        prev_loss = float('inf')
        
        for it in range(max_iters):
            # Initialize gradients
            grad_A = [[0.0] * self.L for _ in range(self.m)]
            grad_B = [[0.0] * self.m for _ in range(self.L)]
            grad_M = [[0.0] * self.n for _ in range(self.m)]  # m×n gradient
            
            total_loss = 0.0
            total_positions = 0
            
            # Process all sequences
            for seq, t in zip(seqs, t_list):
                # Transform target vector to model space: t_m = M * t
                t_m = self._mat_vec(self.M, t)
                
                vecs = self.extract_vectors(seq)[:self.L]
                total_positions += len(vecs)
                
                for v in vecs:
                    # Forward pass
                    x = self._mat_vec(self.M, v)          # M (m×n) * v (n) → x (m)
                    z = self._mat_vec(self.Bbasis, x)     # Bbasis (L×m) * x (m) → z (L)
                    N = self._mat_vec(self.Acoeff, z)     # Acoeff (m×L) * z (L) → N (m)
                    
                    # Compute error (compare to transformed target)
                    error = [N_i - t_m_i for N_i, t_m_i in zip(N, t_m)]
                    total_loss += sum(e*e for e in error)
                    
                    # Backpropagation
                    # Gradient w.r.t. Acoeff: dL/dAcoeff = error * z^T
                    for i in range(self.m):
                        for j in range(self.L):
                            grad_A[i][j] += 2 * error[i] * z[j]
                    
                    # Gradient w.r.t. Bbasis: dL/dBbasis = (Acoeff^T @ error) * x^T
                    Acoeff_t_error = self._mat_vec(self._transpose(self.Acoeff), error)
                    for j in range(self.L):
                        for i in range(self.m):
                            grad_B[j][i] += 2 * Acoeff_t_error[j] * x[i]
                    
                    # Gradient w.r.t. M: dL/dM = (Bbasis^T @ Acoeff^T @ error) * v^T
                    B_t_Acoeff_t_error = self._mat_vec(
                        self.B_t, Acoeff_t_error
                    )
                    for i in range(self.m):
                        for j in range(self.n):  # j indexes input dimension
                            grad_M[i][j] += 2 * B_t_Acoeff_t_error[i] * v[j]
            
            # Average gradients
            if total_positions > 0:
                norm = 1 / total_positions
                for i in range(self.m):
                    for j in range(self.L):
                        grad_A[i][j] *= norm
                for j in range(self.L):
                    for i in range(self.m):
                        grad_B[j][i] *= norm
                for i in range(self.m):
                    for j in range(self.n):
                        grad_M[i][j] *= norm
            
            # Update parameters
            for i in range(self.m):
                for j in range(self.L):
                    self.Acoeff[i][j] -= learning_rate * grad_A[i][j]
            
            for j in range(self.L):
                for i in range(self.m):
                    self.Bbasis[j][i] -= learning_rate * grad_B[j][i]
            
            for i in range(self.m):
                for j in range(self.n):
                    self.M[i][j] -= learning_rate * grad_M[i][j]
            
            # Update B_t
            self.B_t = self._transpose(self.Bbasis)
            
            # Calculate and record loss
            avg_loss = total_loss / total_positions if total_positions else 0
            history.append(avg_loss)
            
            # Print progress
            if it % print_every == 0 or it == max_iters - 1:
                print(f"Iter {it:3d}: Loss = {avg_loss:.6f}")
            
            # Check convergence
            if abs(prev_loss - avg_loss) < tol:
                print(f"Converged after {it} iterations")
                break
            prev_loss = avg_loss
        
        # Final statistics
        self.mean_L = int(sum(len(seq) for seq in seqs) / len(seqs))
        self.mean_t = [0.0] * self.n
        for t in t_list:
            for i in range(self.n):
                self.mean_t[i] += t[i]
        self.mean_t = [x / len(t_list) for x in self.mean_t]
        self.trained = True
        
        return history

    def auto_train(self, seqs, learning_rate=0.01, max_iters=100, tol=1e-6,
                   print_every=10, auto_mode='reg'):
        """
        Self-supervised training for vector sequences.
        
        Args:
            seqs: List of training sequences (n-dimensional vectors)
            learning_rate: Step size
            max_iters: Maximum iterations
            tol: Convergence tolerance
            print_every: Print interval
            auto_mode: 'gap' (autoencoder) or 'reg' (next-vector prediction)
            
        Returns:
            list: Training loss history
        """
        assert auto_mode in ('gap', 'reg')
        self.initialize(seqs)
        history = []
        prev_loss = float('inf')
        
        for it in range(max_iters):
            # Initialize gradients
            grad_A = [[0.0] * self.L for _ in range(self.m)]
            grad_B = [[0.0] * self.m for _ in range(self.L)]
            grad_M = [[0.0] * self.n for _ in range(self.m)]  # m×n gradient
            
            total_loss = 0.0
            total_instances = 0
            
            # Process all sequences
            for seq in seqs:
                vecs = self.extract_vectors(seq)[:self.L]
                n = len(vecs)
                
                if auto_mode == 'reg' and n < 2:
                    continue  # Skip short sequences
                
                # Determine processing range
                indices = range(n) if auto_mode == 'gap' else range(n - 1)
                
                for k in indices:
                    current_vec = vecs[k]  # n-dimensional
                    target_vec = vecs[k] if auto_mode == 'gap' else vecs[k + 1]  # n-dimensional
                    
                    # Transform target vector to model space: t_m = M * target_vec
                    t_m = self._mat_vec(self.M, target_vec)
                    
                    # Forward pass
                    x = self._mat_vec(self.M, current_vec)  # M (m×n) * v → x (m)
                    z = self._mat_vec(self.Bbasis, x)       # Bbasis (L×m) * x → z (L)
                    N = self._mat_vec(self.Acoeff, z)       # Acoeff (m×L) * z → N (m)
                    
                    # Compute error against transformed target
                    error = [N_i - t_m_i for N_i, t_m_i in zip(N, t_m)]
                    total_loss += sum(e*e for e in error)
                    total_instances += 1
                    
                    # Backpropagation
                    # Gradient w.r.t. Acoeff: dL/dAcoeff = error * z^T
                    for i in range(self.m):
                        for j in range(self.L):
                            grad_A[i][j] += 2 * error[i] * z[j]
                    
                    # Gradient w.r.t. Bbasis: dL/dBbasis = (Acoeff^T @ error) * x^T
                    Acoeff_t_error = self._mat_vec(self._transpose(self.Acoeff), error)
                    for j in range(self.L):
                        for i in range(self.m):
                            grad_B[j][i] += 2 * Acoeff_t_error[j] * x[i]
                    
                    # Gradient w.r.t. M: dL/dM = (Bbasis^T @ Acoeff^T @ error) * current_vec^T
                    B_t_Acoeff_t_error = self._mat_vec(
                        self.B_t, Acoeff_t_error
                    )
                    for i in range(self.m):
                        for j in range(self.n):
                            grad_M[i][j] += 2 * B_t_Acoeff_t_error[i] * current_vec[j]
            
            # Average gradients
            if total_instances > 0:
                norm = 1 / total_instances
                for i in range(self.m):
                    for j in range(self.L):
                        grad_A[i][j] *= norm
                for j in range(self.L):
                    for i in range(self.m):
                        grad_B[j][i] *= norm
                for i in range(self.m):
                    for j in range(self.n):
                        grad_M[i][j] *= norm
            
            # Update parameters
            for i in range(self.m):
                for j in range(self.L):
                    self.Acoeff[i][j] -= learning_rate * grad_A[i][j]
            
            for j in range(self.L):
                for i in range(self.m):
                    self.Bbasis[j][i] -= learning_rate * grad_B[j][i]
            
            for i in range(self.m):
                for j in range(self.n):
                    self.M[i][j] -= learning_rate * grad_M[i][j]
            
            # Update B_t
            self.B_t = self._transpose(self.Bbasis)
            
            # Calculate and record loss
            avg_loss = total_loss / total_instances if total_instances else 0
            history.append(avg_loss)
            
            # Print progress
            if it % print_every == 0 or it == max_iters - 1:
                mode_str = "Autoencoder" if auto_mode == 'gap' else "Next-vector"
                print(f"Iter {it:3d} ({mode_str}): Loss = {avg_loss:.6f}")
            
            # Check convergence
            if abs(prev_loss - avg_loss) < tol:
                print(f"Converged after {it} iterations")
                break
            prev_loss = avg_loss
        
        # Final statistics
        self.mean_L = int(sum(len(seq) for seq in seqs) / len(seqs))
        # Compute mean descriptor vector (n-dimensional)
        self.mean_t = [0.0] * self.n
        count = 0
        for seq in seqs:
            for Nk in self.describe(seq):
                # Transform descriptor back to n-dim space
                t_n = self._mat_vec(self._transpose(self.M), Nk)
                for i in range(self.n):
                    self.mean_t[i] += t_n[i]
                count += 1
        self.mean_t = [x / count for x in self.mean_t] if count > 0 else self.mean_t
        self.trained = True
        
        return history

    def predict_t(self, seq):
        """
        Predict target vector for a sequence (output n-dimensional).
        
        Args:
            seq: List of n-dimensional vectors
            
        Returns:
            list: Predicted target vector (n-dimensional)
        """
        N_list = self.describe(seq)
        if not N_list:
            return [0.0] * self.n
        
        # Average all position vectors in model space
        t_pred_m = [0.0] * self.m
        for Nk in N_list:
            for i in range(self.m):
                t_pred_m[i] += Nk[i]
        
        n_pos = len(N_list)
        t_pred_m = [x / n_pos for x in t_pred_m]
        
        # Transform back to original n-dimensional space
        return self._mat_vec(self._transpose(self.M), t_pred_m)

    def generate(self, length, init_vec=None, tau=0.0):
        """
        Generate a sequence of vectors (output n-dimensional).
        
        Args:
            length: Number of vectors to generate
            init_vec: Starting vector (n-dimensional, random if None)
            tau: Temperature (0=deterministic, >0=stochastic)
            
        Returns:
            list: Generated sequence of vectors (n-dimensional)
        """
        assert self.trained, "Model must be trained before generation"
        if tau < 0:
            raise ValueError("Temperature must be non-negative")
        
        # Initialize sequence
        if init_vec is None:
            init_vec = [random.gauss(0, 1) for _ in range(self.n)]  # n-dim
        sequence = [init_vec]
        current_vec = init_vec
        
        for _ in range(length - 1):
            # Transform current vector to model space
            x = self._mat_vec(self.M, current_vec)  # Map to m-dim
            z = self._mat_vec(self.Bbasis, x)
            next_pred_m = self._mat_vec(self.Acoeff, z)  # m-dim prediction
            
            # Apply temperature to model-space vector
            if tau > 0.0:
                next_pred_m = [random.gauss(p, tau) for p in next_pred_m]
            
            # Transform prediction back to n-dim space
            next_vec = self._mat_vec(self._transpose(self.M), next_pred_m)
            
            sequence.append(next_vec)
            current_vec = next_vec
        
        return sequence

    # ---- feature extraction ----
    def dd_features(self, seq):
        """Extract features from a sequence"""
        feats = []
        
        # Flatten Acoeff
        for row in self.Acoeff:
            feats.extend(row)
        
        # Flatten M
        for row in self.M:
            feats.extend(row)
        
        # Basis-weighted features
        vecs = self.extract_vectors(seq)[:self.L]
        weighted_sum = [0.0] * self.m
        for j, v in enumerate(vecs):
            # Transform vector (n-dim → m-dim)
            x = self._mat_vec(self.M, v)
            for i in range(self.m):
                weighted_sum[i] += self.Bbasis[j][i] * x[i]
        
        feats.extend(weighted_sum)
        return feats

    # ---- show state ----
    def show(self):
        """Display model state"""
        print("DualDescriptorVectorRNM status:")
        print(f" L={self.L}, m={self.m}, n={self.n}, mode={self.mode}")
        print(" Sample Acoeff[0][:5]:", self.Acoeff[0][:5])
        print(" Sample Bbasis[0][:5]:", self.Bbasis[0][:5])
        print(" Sample M[0]:", self.M[0])

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
        
        # Create instance without __init__
        obj = cls.__new__(cls)
        obj.__dict__.update(state)
        print(f"Model loaded from {filename}")
        return obj


# === Example Usage ===
if __name__ == "__main__":
    
    from statistics import correlation, mean
    
    # Configuration   
    input_dim = 10   # Input vector dimension (n)
    model_dim = 20   # Internal model dimension (m)
    n_seqs = 10      # Number of training sequences

    # Generate sequences with n-dimensional vectors and n-dimensional targets
    seqs, t_list = [], []
    for _ in range(n_seqs):
        L = random.randint(100, 200)
        # Create sequence of n-dimensional vectors
        seq = [[random.uniform(-1,1) for _ in range(input_dim)] for __ in range(L)]        
        seqs.append(seq)
        # Create n-dimensional target vector
        t_list.append([random.uniform(-1,1) for _ in range(input_dim)])
    
    # Create model with input_dim and model_dim
    dd = NumDualDescriptor(model_dim=model_dim, input_dim=input_dim, 
                           rank=3, mode='nonlinear', user_step=2)
    
    # Train with ALS
    print("Training with Alternating Least Squares:")
    als_history = dd.train(seqs, t_list, max_iters=20)
    
    # Predict targets (now n-dimensional)
    print("\nPredictions (n-dimensional output):")
    for i, seq in enumerate(seqs[:3]):  # First 3 sequences
        t_pred = dd.predict_t(seq)
        print(f"Seq {i+1}: Target={[f'{x:.4f}' for x in t_list[i]]}")
        print(f"         Predicted={[f'{x:.4f}' for x in t_pred]}")
    
    # Calculate prediction correlations (n-dimensional)
    preds = [dd.predict_t(seq) for seq in seqs]
    correlations = []
    for i in range(input_dim):  # Use input_dim instead of model_dim
        actual = [t[i] for t in t_list]
        predicted = [p[i] for p in preds]
        corr = correlation(actual, predicted)
        correlations.append(corr)
        print(f"Dim {i} correlation: {corr:.4f}")
    print(f"Average correlation: {mean(correlations):.4f}")
    
    # Generate sequence (n-dimensional output)
    print("\nGenerated sequence (n-dimensional):")
    generated_seq = dd.generate(length=5)
    for i, vec in enumerate(generated_seq):
        print(f"Step {i}: {[f'{x:.4f}' for x in vec]}")
    
    # Train with Gradient Descent
    print("\nTraining with Gradient Descent:")
    dd_grad = NumDualDescriptor(model_dim=model_dim, input_dim=input_dim,
                               rank=3, mode='nonlinear', user_step=2)
    grad_history = dd_grad.grad_train(
        seqs, 
        t_list, 
        learning_rate=0.1,
        max_iters=300,
        print_every=10
    )

    # Calculate prediction correlations (n-dimensional)
    preds = [dd_grad.predict_t(seq) for seq in seqs]
    correlations = []
    for i in range(input_dim):  # Use input_dim instead of model_dim
        actual = [t[i] for t in t_list]
        predicted = [p[i] for p in preds]
        corr = correlation(actual, predicted)
        correlations.append(corr)
        print(f"Dim {i} correlation: {corr:.4f}")
    print(f"Average correlation: {mean(correlations):.4f}")
    
    # Generate sequence with gradient-trained model
    print("\nGenerated sequence from gradient-trained model:")
    generated_seq = dd_grad.generate(length=5)
    for i, vec in enumerate(generated_seq):
        print(f"Step {i}: {[f'{x:.4f}' for x in vec]}")
