# Copyright (C) Bin-Guang Ma (mbg@mail.hzau.edu.cn). All rights reserved.
# The Dual Descriptor Vector class (P matrix form) for vector sequences
# Modified to support n-dimensional input sequences with m-dimensional internal representation
# Author: Bin-Guang Ma; Date: 2025-6-17

# The program is provided as it is and without warranty of any kind,
# either expressed or implied.

import math
import random
import pickle

class NumDualDescriptor:
    """
    Dual Descriptor for sequences of n-dimensional real vectors with m-dimensional internal representation.
    - input_dim: dimension of input vectors (n)
    - model_dim: dimension of internal representation (m)
    Model: 
        For each vector x_k in sequence: 
            N(k) = P * M * x_k  (output is n-dimensional)
        S(l) = cumulative sum of N(k) from k=1 to l
    """

    def __init__(self, model_dim=4, input_dim=4, rank=1, rank_mode='drop', rank_op='avg', mode='linear', user_step=None):
        """
        Initialize the Dual Descriptor for vector sequences.
        
        Args:
            input_dim (int): Dimensionality of input vectors (n)
            model_dim (int): Dimensionality of internal representation (m)
            rank (int): Parameter controlling step size in nonlinear mode
            rank_mode (str): 'pad' or 'drop' for handling incomplete windows
            rank_op (str): 'avg', 'sum', 'pick', or 'user_func' for window processing
            mode (str): 'linear' or 'nonlinear' processing mode
            user_step (int): Custom step size for nonlinear mode
        """
        self.rank = rank
        self.rank_mode = rank_mode
        self.rank_op = rank_op
        self.n = input_dim  # Input dimension (n)
        self.m = model_dim  # Internal dimension (m)
        assert mode in ('linear', 'nonlinear')
        self.mode = mode
        self.step = user_step
        self.trained = False

        # Initialize M matrix (m×n): maps n-dim input to m-dim internal space
        self.M = [[random.uniform(-0.5, 0.5) for _ in range(self.n)] 
                  for _ in range(self.m)]
        
        # Initialize P matrix (n×m): maps m-dim internal to n-dim output
        self.P = [[random.uniform(-0.1, 0.1) for _ in range(self.m)] 
                  for _ in range(self.n)]

    # ---- sequence processing ----
    def extract_vectors(self, seq):
        """
        Extract vectors from sequence based on processing mode and rank operation.
        Returns n-dimensional vectors.
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

    # ---- linear algebra helpers ----
    def mat_vec(self, M, v):
        """Multiply matrix M by vector v. Supports any compatible dimensions."""
        rows = len(M)
        cols = len(M[0])
        if cols != len(v):
            raise ValueError(f"Matrix columns ({cols}) must match vector length ({len(v)})")
        return [sum(M[i][j] * v[j] for j in range(cols)) for i in range(rows)]

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
        """Transpose a matrix."""
        return [list(col) for col in zip(*M)]

    def vec_sub(self, u, v):
        """Subtract two vectors of same dimension."""
        if len(u) != len(v):
            raise ValueError("Vectors must have same length for subtraction")
        return [u[i] - v[i] for i in range(len(u))]

    def dot(self, u, v):
        """Dot product of two vectors of same dimension."""
        if len(u) != len(v):
            raise ValueError("Vectors must have same length for dot product")
        return sum(u[i] * v[i] for i in range(len(u)))

    def describe(self, seq):
        """
        Compute list of N(k)=P·M·x_k for a given vector sequence.
        Each N(k) is an n-dimensional vector.
        """
        vecs = self.extract_vectors(seq)
        Nk_list = []
        for xk in vecs:
            # Transform input vector: M_xk = M · xk (m-dim)
            M_xk = self.mat_vec(self.M, xk)
            # Compute N(k) = P · (M · xk) (n-dim)
            Nk = self.mat_vec(self.P, M_xk)
            Nk_list.append(Nk)
        return Nk_list

    def S(self, seq):
        """
        Compute cumulative sum vectors S(l)=sum_{k=1}^{l} N(k) 
        for l = 1,...,L. Returns n-dimensional vectors.
        """
        Nk_list = self.describe(seq)
        if not Nk_list:
            return []
        # Initialize S as zero vector of output dimension (n)
        S = [0.0] * len(Nk_list[0])
        S_list = []
        for Nk in Nk_list:
            S = [S[i] + Nk[i] for i in range(len(S))]
            S_list.append(S.copy())
        return S_list

    def deviation(self, seqs, t_list):
        """
        Compute mean squared deviation D:
        D = average over all positions and sequences of ||P·M·x_k - t_j||^2.
        t_j are n-dimensional targets.
        """
        total = 0.0
        count = 0
        for seq, t in zip(seqs, t_list):
            for Nk in self.describe(seq):                
                # Compute error: Nk - t (both n-dimensional)
                err = self.vec_sub(Nk, t)
                total += self.dot(err, err)
                count += 1
        return total / count if count else 0.0

    def _invert_matrix(self, A):
        """Invert a square matrix A using Gauss-Jordan elimination."""
        n = len(A)
        # Build augmented matrix [A | I]
        M = [row[:] + [1.0 if i==j else 0.0 for j in range(n)] 
             for i, row in enumerate(A)]
        
        for i in range(n):
            # Find pivot row
            max_row = i
            for r in range(i+1, n):
                if abs(M[r][i]) > abs(M[max_row][i]):
                    max_row = r
            M[i], M[max_row] = M[max_row], M[i]
            
            piv = M[i][i]
            if abs(piv) < 1e-12:
                continue
                
            # Normalize pivot row
            M[i] = [mij / piv for mij in M[i]]
            
            # Eliminate other rows
            for r in range(n):
                if r == i: 
                    continue
                fac = M[r][i]
                M[r] = [M[r][c] - fac * M[i][c] for c in range(2*n)]
                
        # Extract inverse
        A_inv = [row[n:] for row in M]
        return A_inv

    def update_P(self, seqs, t_list):
        """
        Update P by closed-form solution:
          U = sum_{j,k} (M·x_k)(M·x_k)^T  (m×m matrix)
          V = sum_{j,k} t_j (M·x_k)^T      (n×m matrix)
          P = V U^{-1}  (n×m matrix)
        """
        m = self.m
        n = self.n
        U = [[0.0]*m for _ in range(m)]
        V = [[0.0]*m for _ in range(n)]
        
        for seq, t in zip(seqs, t_list):
            for xk in seq:
                # Compute M_xk = M · xk (m-dim vector)
                M_xk = self.mat_vec(self.M, xk)
                
                # Accumulate U += M_xk · M_xk^T (m×m)
                for i in range(m):
                    for j in range(m):
                        U[i][j] += M_xk[i] * M_xk[j]
                
                # Accumulate V += t · M_xk^T (n×m)
                for i in range(n):
                    for j in range(m):
                        V[i][j] += t[i] * M_xk[j]
        
        # Invert U and update P
        U_inv = self._invert_matrix(U)
        self.P = self.mat_mul(V, U_inv)  # (n×m) = (n×m) × (m×m)

    def update_M(self, seqs, t_list):
        """
        Update M by closed-form solution:
          R = sum_{k} P^T t_j x_k^T  (m×n matrix)
          M = (P^T P)^{-1} R  (m×n matrix)
        """
        m = self.m
        n = self.n
        Pt = self.transpose(self.P)  # m×n -> n×m transpose → m×n? Actually: P is n×m → Pt is m×n
        
        # Precompute (P^T P)^{-1} (m×m matrix)
        # Pt is m×n, P is n×m → PtP is m×m
        PtP = self.mat_mul(Pt, self.P)  # (m×n) × (n×m) → (m×m)
        PtP_inv = self._invert_matrix(PtP)
        
        # Accumulate R = sum(P^T t_j x_k^T) (m×n matrix)
        R = [[0.0]*n for _ in range(m)]
        for seq, t in zip(seqs, t_list):
            for xk in seq:
                # Compute P^T t (m-dim vector)
                Pt_t = self.mat_vec(Pt, t)  # (m×n) × (n×1) → m×1
                # Accumulate R += (P^T t) · xk^T (m×n)
                for i in range(m):
                    for j in range(n):
                        R[i][j] += Pt_t[i] * xk[j]
        
        # Update M: M = PtP_inv * R (m×n) = (m×m) × (m×n)
        self.M = self.mat_mul(PtP_inv, R)

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
            if abs(D - D_prev) < tol or D >= D_prev:
                print("Converged.")
                break
            D_prev = D
        
        # Calculate statistics
        total_vectors = sum(len(seq) for seq in seqs)
        total_t = [0.0] * self.n  # n-dimensional mean
        for t in t_list:
            for d in range(self.n):
                total_t[d] += t[d]
        self.mean_vector_count = total_vectors / len(seqs)
        self.mean_t = [t_val / len(seqs) for t_val in total_t]
        self.trained = True
        return history

    def grad_train(self, seqs, t_list, max_iters=100, tol=1e-8, 
                   learning_rate=0.01, decay_rate=0.95):
        """
        Train using gradient descent.
        Alternates between updating P and M using gradients.
        """
        history = []
        D_prev = float('inf')
        m = self.m
        n = self.n
        
        for it in range(max_iters):
            total_vectors = 0
            grad_P = [[0.0]*m for _ in range(n)]  # n×m
            grad_M = [[0.0]*n for _ in range(m)]  # m×n
            
            # Compute gradients
            for seq, t in zip(seqs, t_list):
                for xk in seq:
                    total_vectors += 1
                    # Forward pass
                    M_xk = self.mat_vec(self.M, xk)  # m-dim
                    Nk = self.mat_vec(self.P, M_xk)   # n-dim
                    error = self.vec_sub(Nk, t)       # n-dim
                    
                    # Gradient for P: dD/dP = 2 * error * (M_xk)^T (n×m)
                    for i in range(n):
                        for j in range(m):
                            grad_P[i][j] += 2 * error[i] * M_xk[j]
                    
                    # Gradient for M: dD/dM = 2 * P^T error * xk^T (m×n)
                    Pt_error = self.mat_vec(self.transpose(self.P), error)  # m-dim
                    for i in range(m):
                        for j in range(n):
                            grad_M[i][j] += 2 * Pt_error[i] * xk[j]
            
            # Apply updates
            if total_vectors > 0:
                lr = learning_rate / total_vectors
                # Update P (n×m)
                for i in range(n):
                    for j in range(m):
                        self.P[i][j] -= lr * grad_P[i][j]
                # Update M (m×n)
                for i in range(m):
                    for j in range(n):
                        self.M[i][j] -= lr * grad_M[i][j]
            
            # Compute current deviation
            D = self.deviation(seqs, t_list)
            history.append(D)
            print(f"Iter {it:3d}: D = {D:.6e}, lr = {learning_rate:.6f}")
            
            # Check convergence
            if abs(D - D_prev) < tol:
                print("Converged.")
                break
            D_prev = D
            
            # Update learning rate
            learning_rate *= decay_rate
        
        # Calculate statistics
        total_vectors = sum(len(seq) for seq in seqs)
        total_t = [0.0] * self.n  # n-dimensional
        for t in t_list:
            for d in range(self.n):
                total_t[d] += t[d]
        self.mean_vector_count = total_vectors / len(seqs)
        self.mean_t = [t_val / len(seqs) for t_val in total_t]
        self.trained = True
        return history

    def auto_train(self, seqs, max_iters=100, tol=1e-6, learning_rate=0.01, 
                   continued=False, auto_mode='reg', decay_rate=0.95):
        """
        Self-training with two modes:
          'gap': Predict current vector (self-consistency)
          'reg': Predict next vector (auto-regressive)
        """
        if auto_mode not in ('gap', 'reg'):
            raise ValueError("auto_mode must be 'gap' or 'reg'")
            
        if not continued:
            # Initialize with proper dimensions
            self.M = [[random.uniform(-0.5, 0.5) for _ in range(self.n)] 
                      for _ in range(self.m)]
            self.P = [[random.uniform(-0.1, 0.1) for _ in range(self.m)] 
                      for _ in range(self.n)]
        
        # Calculate total training samples
        total_samples = 0
        for seq in seqs:
            if auto_mode == 'gap':
                total_samples += len(seq)
            else:  # reg mode
                total_samples += max(0, len(seq) - 1)
        
        if total_samples == 0:
            raise ValueError("No training samples found")
        
        history = []
        prev_loss = float('inf')
        
        for it in range(max_iters):
            grad_P = [[0.0]*self.m for _ in range(self.n)]  # n×m
            grad_M = [[0.0]*self.n for _ in range(self.m)]  # m×n
            total_loss = 0.0
            
            for seq in seqs:
                L = len(seq)
                if L == 0:
                    continue
                    
                for k in range(L):
                    if auto_mode == 'reg' and k == L-1:
                        continue
                    
                    xk = seq[k]
                    # Forward pass: Nk = P·M·xk (n-dim)
                    M_xk = self.mat_vec(self.M, xk)  # m-dim
                    Nk = self.mat_vec(self.P, M_xk)   # n-dim
                    
                    # Set target
                    if auto_mode == 'gap':
                        target = xk  # n-dim
                    else:  # reg
                        target = seq[k+1]  # n-dim
                    
                    # Compute error and loss
                    error = [Nk[i] - target[i] for i in range(self.n)]
                    sq_error = sum(e**2 for e in error)
                    total_loss += sq_error
                    
                    # Compute gradients
                    # Gradient w.r.t P: 2 * error * (M_xk)^T (n×m)
                    for i in range(self.n):
                        for j in range(self.m):
                            grad_P[i][j] += 2 * error[i] * M_xk[j]
                    
                    # Gradient w.r.t M: 2 * (P^T error) * xk^T (m×n)
                    Pt_error = self.mat_vec(self.transpose(self.P), error)  # m-dim
                    for i in range(self.m):
                        for j in range(self.n):
                            grad_M[i][j] += 2 * Pt_error[i] * xk[j]
            
            # Apply updates
            avg_loss = total_loss / total_samples
            lr = learning_rate
            # Update P (n×m)
            for i in range(self.n):
                for j in range(self.m):
                    self.P[i][j] -= lr * grad_P[i][j] / total_samples
            # Update M (m×n)
            for i in range(self.m):
                for j in range(self.n):
                    self.M[i][j] -= lr * grad_M[i][j] / total_samples
            
            history.append(avg_loss)
            mode_str = "Gap" if auto_mode == 'gap' else "Reg"
            print(f"AutoTrain({mode_str}) Iter {it:3d}: loss = {avg_loss:.6f}")
            
            # Check convergence
            if prev_loss - avg_loss < tol:
                print("Converged.")
                break
            prev_loss = avg_loss
            
            # Update learning rate
            learning_rate *= decay_rate
        
        # Compute mean statistics
        total_vectors = sum(len(seq) for seq in seqs)
        self.mean_vector_count = total_vectors / len(seqs)
        self.trained = True
        return history

    def predict_t(self, seq):
        """
        Predict target vector t for a sequence (n-dimensional).
        Optimal t is the mean of all N(k) vectors (n-dimensional).
        """
        Nk_list = self.describe(seq)
        if not Nk_list:
            return [0.0] * self.n
            
        t_pred = [0.0] * self.n
        for vec in Nk_list:
            for i in range(self.n):
                t_pred[i] += vec[i]
        return [x / len(Nk_list) for x in t_pred]

    def generate(self, L, init_vec=None, tau=0.0):
        """
        Generate n-dimensional vector sequence of length L.
        init_vec: starting vector (n-dimensional, random if None)
        tau: temperature for randomness
        """
        assert self.trained, "Model must be trained first"
        if tau < 0:
            raise ValueError("Temperature must be non-negative")
        
        # Initialize sequence
        seq = []
        if init_vec is None:
            current = [random.gauss(0, 1) for _ in range(self.n)]  # n-dim
        else:
            if len(init_vec) != self.n:
                raise ValueError(f"Initial vector must be {self.n}-dimensional")
            current = init_vec.copy()
        seq.append(current)
        
        # Generate subsequent vectors
        for _ in range(1, L):
            # Compute next_vec = P·M·current (n-dim)
            M_current = self.mat_vec(self.M, current)  # m-dim
            next_vec = self.mat_vec(self.P, M_current)  # n-dim
            
            # Apply temperature-controlled noise
            if tau > 0:
                next_vec = [v + random.gauss(0, tau) for v in next_vec]
            
            seq.append(next_vec)
            current = next_vec
        
        return seq

    def dd_features(self, seq):
        """
        Extract features for a sequence:
        [flattened P] + [flattened M] (length = n*m + m*n)
        """
        feats = []
        # Flatten P matrix (n×m)
        for row in self.P:
            feats.extend(row)
        # Flatten M matrix (m×n)
        for row in self.M:
            feats.extend(row)
        return feats

    def show(self):
        """Display model status."""
        print("NumDualDescriptor Status:")
        print(f"  Input dimension n = {self.n}")
        print(f"  Internal dimension m = {self.m}")
        print("  M matrix (m×n):")
        for row in self.M:
            print("   ", [f"{x:.4f}" for x in row])
        print("  P matrix (n×m):")
        for row in self.P:
            print("   ", [f"{x:.4f}" for x in row])

    def count_parameters(self):
        """Count learnable parameters (M and P matrices)."""
        total_params = self.n * self.m + self.m * self.n
        print(f"Parameter Count:")
        print(f"  M matrix: {self.m}×{self.n} = {self.m*self.n} parameters")
        print(f"  P matrix: {self.n}×{self.m} = {self.n*self.m} parameters")
        print(f"Total parameters: {total_params}")
        return total_params

    def save(self, filename):
        """Save model state to file."""
        with open(filename, 'wb') as f:
            pickle.dump(self.__dict__, f)
        print(f"Model saved to {filename}")

    @classmethod
    def load(cls, filename):
        """Load model state from file."""
        with open(filename, 'rb') as f:
            state = pickle.load(f)
        obj = cls.__new__(cls)
        obj.__dict__.update(state)
        print(f"Model loaded from {filename}")
        return obj


# === Example Usage ===
if __name__ == "__main__":
    from statistics import correlation, mean

    random.seed(11)
    # Configuration: n=20 input dim, m=10 internal dim
    n_dim = 20  # Input vector dimension (n)
    m_dim = 10  # Internal model dimension (m)
    num_seqs = 10
    min_len, max_len = 100, 200

    # Generate synthetic training data (n-dimensional)
    seqs = []
    t_list = []
    for _ in range(num_seqs):
        L = random.randint(min_len, max_len)
        seq = [[random.uniform(-1,1) for _ in range(n_dim)] for __ in range(L)]
        seqs.append(seq)
        t_list.append([random.uniform(-1,1) for _ in range(n_dim)])  # n-dim targets

    # Create model with n=20, m=10
    dd = NumDualDescriptor(input_dim=n_dim, model_dim=m_dim, 
                          rank=5, rank_mode='drop', rank_op='avg', 
                          mode='nonlinear', user_step=5)
    
    # Train with ALS
    print("Training with Alternating Least Squares:")
    als_history = dd.train(seqs, t_list, max_iters=20)
    
    # Predict targets (n-dimensional)
    print("\nPredictions:")
    for i, seq in enumerate(seqs[:3]):  # First 3 sequences
        t_pred = dd.predict_t(seq)
        # Show first 5 dimensions for readability
        print(f"Seq {i+1}: Target={[f'{x:.4f}' for x in t_list[i][:5]]}...")
        print(f"         Predicted={[f'{x:.4f}' for x in t_pred[:5]]}...")
    
    # Calculate prediction correlations
    preds = [dd.predict_t(seq) for seq in seqs]
    correlations = []
    for i in range(n_dim):
        actual = [t[i] for t in t_list]
        predicted = [p[i] for p in preds]
        corr = correlation(actual, predicted)
        correlations.append(corr)
        if i < 5:  # Show first 5 dimensions
            print(f"Dim {i} correlation: {corr:.4f}")
    print(f"Average correlation: {mean(correlations):.4f}")
    
    # Train with Gradient Descent
    print("\nTraining with Gradient Descent:")
    dd_grad = NumDualDescriptor(input_dim=n_dim, model_dim=m_dim,
                               rank=5, rank_mode='drop', rank_op='avg', 
                               mode='nonlinear', user_step=5)
    grad_history = dd_grad.grad_train(
        seqs, 
        t_list, 
        learning_rate=0.3,
        max_iters=100,
        decay_rate=0.98
    )

    # Calculate prediction correlations
    preds = [dd_grad.predict_t(seq) for seq in seqs]
    correlations = []
    for i in range(n_dim):
        actual = [t[i] for t in t_list]
        predicted = [p[i] for p in preds]
        corr = correlation(actual, predicted)
        correlations.append(corr)
        if i < 5:  # Show first 5 dimensions
            print(f"Dim {i} correlation: {corr:.4f}")
    print(f"Average correlation: {mean(correlations):.4f}")

    # Parameter count
    print("\nParameter count:")
    param_count = dd.count_parameters()
    print(f"Total parameters: {param_count} (n*m + m*n = {n_dim*m_dim + m_dim*n_dim})")

    # Create a new model for auto_train examples
    dd_auto = NumDualDescriptor(input_dim=n_dim, model_dim=m_dim,
                               rank=3, mode='nonlinear', user_step=2)

    # Example of auto_train in 'gap' mode (autoencoder)
    print("\nAuto-train in 'gap' mode (autoencoder):")
    gap_history = dd_auto.auto_train(
        seqs, 
        learning_rate=0.1,
        max_iters=50,
        auto_mode='gap',
        decay_rate=0.98
    )
    print(f"Final autoencoder loss: {gap_history[-1]:.6f}")

    # Example of auto_train in 'reg' mode (next-vector prediction)
    print("\nAuto-train in 'reg' mode (next-vector prediction):")
    reg_history = dd_auto.auto_train(
        seqs, 
        learning_rate=0.1,
        max_iters=50,
        auto_mode='reg',
        decay_rate=0.98
    )
    print(f"Final next-vector prediction loss: {reg_history[-1]:.6f}")

    # Example of sequence generation after auto_train
    print("\nGenerating sequence with 'reg' trained model:")
    generated_seq = dd_auto.generate(
        L=10,
        init_vec=seqs[0][0],  # Start with first vector of first sequence (n-dim)
        tau=0.05  # Small randomness
    )
    print(f"First 3 generated vectors (n-dim):")
    for i, vec in enumerate(generated_seq[:3]):
        # Show first 5 dimensions
        print(f"Vec {i+1}: {[f'{x:.4f}' for x in vec[:5]]}...")

    # Example of feature extraction
    print("\nFeature extraction example:")
    sample_seq = seqs[0][:20]  # First 20 vectors of first sequence
    features = dd_auto.dd_features(sample_seq)
    print(f"Extracted {len(features)} features (n*m + m*n = {n_dim*m_dim + m_dim*n_dim})")
    print(f"First 5 features: {[f'{x:.6f}' for x in features[:5]]}")
    print(f"Last 5 features: {[f'{x:.6f}' for x in features[-5:]]}")

    # Example of show method
    print("\nModel state visualization:")
    dd_auto.show()

    # Example of save/load
    print("\nModel saving/loading example:")
    dd_auto.save("dd_auto_model.pkl")
    loaded_model = NumDualDescriptor.load("dd_auto_model.pkl")
    print("Loaded model show:")
    loaded_model.show()
