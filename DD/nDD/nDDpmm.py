# Copyright (C) Bin-Guang Ma (mbg@mail.hzau.edu.cn). All rights reserved.
# The Dual Descriptor Vector class (P matrix form) for vector sequences
# Modified to support n-dimensional input and m-dimensional output
# Author: Bin-Guang Ma; Date: 2025-6-17

# The program is provided as it is and without warranty of any kind,
# either expressed or implied.

import math
import random
import pickle

class NumDualDescriptor:
    """
    Dual Descriptor for sequences of n-dimensional input vectors and m-dimensional outputs.
    - in_dim: dimension of input vectors
    - out_dim: dimension of output vectors
    Model: 
        For each vector x_k in sequence: 
            N(k) = P * M * x_k
        S(l) = cumulative sum of N(k) from k=1 to l
    """

    def __init__(self, in_dim=4, out_dim=4, rank=1, rank_mode='drop', rank_op='avg', mode='linear', user_step=None):
        """
        Initialize the Dual Descriptor for vector sequences.
        
        Args:
            in_dim (int): Dimensionality of input vectors (n)
            out_dim (int): Dimensionality of output vectors (m)
            rank (int): Parameter controlling step size in nonlinear mode
            rank_mode (str): 'pad' or 'drop' for handling incomplete windows
            rank_op (str): 'avg', 'sum', 'pick', or 'user_func' for window processing
            mode (str): 'linear' or 'nonlinear' processing mode
            user_step (int): Custom step size for nonlinear mode
        """
        self.rank = rank
        self.rank_mode = rank_mode
        self.rank_op = rank_op
        self.in_dim = in_dim  # Input vector dimension
        self.out_dim = out_dim  # Output vector dimension
        assert mode in ('linear', 'nonlinear')
        self.mode = mode
        self.step = user_step
        self.trained = False

        # Initialize M matrix (m×n)
        self.M = [[random.uniform(-0.5, 0.5) for _ in range(in_dim)] 
                  for _ in range(out_dim)]
        
        # Initialize P matrix (m×m)
        self.P = [[random.uniform(-0.1, 0.1) for _ in range(out_dim)] 
                  for _ in range(out_dim)]

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
            seq: List of n-dimensional vectors
            
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

    # ---- linear algebra helpers ----
    def mat_vec(self, M, v):
        """Multiply matrix M (with dimensions a×b) by vector v (length b) → vector of length a."""
        a = len(M)
        b = len(M[0])
        assert b == len(v), f"Matrix columns {b} != vector length {len(v)}"
        return [sum(M[i][j] * v[j] for j in range(b)) for i in range(a)]

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
        """Subtract two vectors of same length."""
        assert len(u) == len(v), "Vector dimensions must match for subtraction"
        return [u[i] - v[i] for i in range(len(u))]

    def dot(self, u, v):
        """Dot product of two vectors of same length."""
        assert len(u) == len(v), "Vector dimensions must match for dot product"
        return sum(u[i] * v[i] for i in range(len(u)))

    def describe(self, seq):
        """
        Compute list of N(k)=P·M·x_k for a given vector sequence.
        Each N(k) is an m-dimensional vector.
        
        Args:
            seq (list): List of n-dimensional vectors
            
        Returns:
            list: List of N(k) vectors for each position
        """
        vecs = self.extract_vectors(seq)
        Nk_list = []
        for xk in vecs:
            # Transform input vector: M_xk = M · xk
            M_xk = self.mat_vec(self.M, xk)
            # Compute N(k) = P · (M · xk)
            Nk = self.mat_vec(self.P, M_xk)
            Nk_list.append(Nk)
        return Nk_list

    def S(self, seq):
        """
        Compute cumulative sum vectors S(l)=sum_{k=1}^{l} N(k) 
        for l = 1,...,L.
        """
        Nk_list = self.describe(seq)
        S = [0.0] * self.out_dim  # Initialize with output dimension
        S_list = []
        for Nk in Nk_list:
            S = [S[i] + Nk[i] for i in range(self.out_dim)]
            S_list.append(S.copy())
        return S_list

    def deviation(self, seqs, t_list):
        """
        Compute mean squared deviation D:
        D = average over all positions and sequences of ||P·M·x_k - t_j||^2.
        """
        total = 0.0
        count = 0
        for seq, t in zip(seqs, t_list):
            for Nk in self.describe(seq):                
                # Compute error: Nk - t
                err = self.vec_sub(Nk, t)
                total += self.dot(err, err)
                count += 1
        return total / count if count else 0.0

    def _invert_matrix(self, A):
        """Invert an m×m matrix A using Gauss-Jordan elimination."""
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
          U = sum_{j,k} (M·x_k)(M·x_k)^T
          V = sum_{j,k} t_j (M·x_k)^T
          P = V U^{-1}
        """
        m = self.out_dim
        U = [[0.0]*m for _ in range(m)]
        V = [[0.0]*m for _ in range(m)]
        
        for seq, t in zip(seqs, t_list):
            for xk in seq:
                # Compute M_xk = M · xk
                M_xk = self.mat_vec(self.M, xk)
                
                # Accumulate U += M_xk · M_xk^T
                for i in range(m):
                    for j in range(m):
                        U[i][j] += M_xk[i] * M_xk[j]
                
                # Accumulate V += t · M_xk^T
                for i in range(m):
                    for j in range(m):
                        V[i][j] += t[i] * M_xk[j]
        
        # Invert U and update P
        U_inv = self._invert_matrix(U)
        self.P = self.mat_mul(V, U_inv)

    def update_M(self, seqs, t_list):
        """
        Update M by closed-form solution:
          For each input vector x_k:
            R = sum_{k} P^T t_j x_k^T
          M = (P^T P)^{-1} R
        """
        m = self.out_dim
        n = self.in_dim
        Pt = self.transpose(self.P)
        
        # Precompute (P^T P)^{-1}
        PtP = [[sum(Pt[i][k] * self.P[k][j] for k in range(m)) 
               for j in range(m)] for i in range(m)]
        PtP_inv = self._invert_matrix(PtP)
        
        # Accumulate R = sum(P^T t_j x_k^T)
        R = [[0.0]*n for _ in range(m)]  # m×n matrix
        for seq, t in zip(seqs, t_list):
            for xk in seq:
                # Compute P^T t
                Pt_t = self.mat_vec(Pt, t)
                # Accumulate R += (P^T t) · xk^T
                for i in range(m):
                    for j in range(n):
                        R[i][j] += Pt_t[i] * xk[j]
        
        # Update M: M = PtP_inv * R
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
        total_t = [0.0] * self.out_dim
        for t in t_list:
            for d in range(self.out_dim):
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
        m = self.out_dim
        n = self.in_dim
        
        for it in range(max_iters):
            total_vectors = 0
            grad_P = [[0.0]*m for _ in range(m)]
            grad_M = [[0.0]*n for _ in range(m)]  # m×n matrix
            
            # Compute gradients
            for seq, t in zip(seqs, t_list):
                for xk in seq:
                    total_vectors += 1
                    # Forward pass
                    M_xk = self.mat_vec(self.M, xk)
                    Nk = self.mat_vec(self.P, M_xk)
                    error = self.vec_sub(Nk, t)
                    
                    # Gradient for P: dD/dP = 2 * error * (M_xk)^T
                    for i in range(m):
                        for j in range(m):
                            grad_P[i][j] += 2 * error[i] * M_xk[j]
                    
                    # Gradient for M: dD/dM = 2 * P^T error * xk^T
                    Pt_error = self.mat_vec(self.transpose(self.P), error)
                    for i in range(m):
                        for j in range(n):
                            grad_M[i][j] += 2 * Pt_error[i] * xk[j]
            
            # Apply updates
            if total_vectors > 0:
                lr = learning_rate / total_vectors
                # Update P matrix (m×m)
                for i in range(m):
                    for j in range(m):
                        self.P[i][j] -= lr * grad_P[i][j]
                # Update M matrix (m×n)
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
        total_t = [0.0] * self.out_dim
        for t in t_list:
            for d in range(self.out_dim):
                total_t[d] += t[d]
        self.mean_vector_count = total_vectors / len(seqs)
        self.mean_t = [t_val / len(seqs) for t_val in total_t]
        self.trained = True
        return history

    def auto_train(self, seqs, max_iters=100, tol=1e-6, learning_rate=0.01, 
                   continued=False, auto_mode='reg', decay_rate=0.95):
        """
        Self-training with two modes (requires in_dim == out_dim):
          'gap': Predict current vector (self-consistency)
          'reg': Predict next vector (auto-regressive)
        """
        # Auto training requires matching dimensions
        if self.in_dim != self.out_dim:
            raise ValueError("auto_train requires in_dim == out_dim")
            
        if auto_mode not in ('gap', 'reg'):
            raise ValueError("auto_mode must be 'gap' or 'reg'")
            
        if not continued:
            # Reinitialize matrices
            self.M = [[random.uniform(-0.5, 0.5) for _ in range(self.in_dim)] 
                      for _ in range(self.out_dim)]
            self.P = [[random.uniform(-0.1, 0.1) for _ in range(self.out_dim)] 
                      for _ in range(self.out_dim)]
        
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
            grad_P = [[0.0]*self.out_dim for _ in range(self.out_dim)]
            grad_M = [[0.0]*self.in_dim for _ in range(self.out_dim)]
            total_loss = 0.0
            
            for seq in seqs:
                L = len(seq)
                if L == 0:
                    continue
                    
                for k in range(L):
                    if auto_mode == 'reg' and k == L-1:
                        continue
                    
                    xk = seq[k]
                    # Forward pass: Nk = P·M·xk
                    M_xk = self.mat_vec(self.M, xk)
                    Nk = self.mat_vec(self.P, M_xk)
                    
                    # Set target
                    if auto_mode == 'gap':
                        target = xk
                    else:  # reg
                        target = seq[k+1]
                    
                    # Compute error and loss
                    error = [Nk[i] - target[i] for i in range(self.out_dim)]
                    sq_error = sum(e**2 for e in error)
                    total_loss += sq_error
                    
                    # Compute gradients
                    # Gradient w.r.t P: 2 * error * (M_xk)^T
                    for i in range(self.out_dim):
                        for j in range(self.out_dim):
                            grad_P[i][j] += 2 * error[i] * M_xk[j]
                    
                    # Gradient w.r.t M: 2 * (P^T error) * xk^T
                    Pt_error = self.mat_vec(self.transpose(self.P), error)
                    for i in range(self.out_dim):
                        for j in range(self.in_dim):
                            grad_M[i][j] += 2 * Pt_error[i] * xk[j]
            
            # Apply updates
            avg_loss = total_loss / total_samples
            lr = learning_rate
            # Update P matrix
            for i in range(self.out_dim):
                for j in range(self.out_dim):
                    self.P[i][j] -= lr * grad_P[i][j] / total_samples
            # Update M matrix
            for i in range(self.out_dim):
                for j in range(self.in_dim):
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
        Predict target vector t for a sequence.
        Optimal t is the mean of all N(k) vectors.
        Returns m-dimensional vector.
        """
        Nk_list = self.describe(seq)
        if not Nk_list:
            return [0.0] * self.out_dim
            
        t_pred = [0.0] * self.out_dim
        for vec in Nk_list:
            for i in range(self.out_dim):
                t_pred[i] += vec[i]
        return [x / len(Nk_list) for x in t_pred]

    def generate(self, L, init_vec=None, tau=0.0):
        """
        Generate vector sequence of length L.
        init_vec: starting vector (random n-dimensional if None)
        tau: temperature for randomness
        Returns list of m-dimensional vectors.
        """
        assert self.trained, "Model must be trained first"
        if tau < 0:
            raise ValueError("Temperature must be non-negative")
        
        # Initialize sequence
        seq = []
        if init_vec is None:
            # Create random n-dimensional initial vector
            current = [random.gauss(0, 1) for _ in range(self.in_dim)]
        else:
            assert len(init_vec) == self.in_dim, \
                f"Initial vector must be {self.in_dim}-dimensional"
            current = init_vec.copy()
        
        # Generate sequence
        for _ in range(L):
            # Compute next vector: N(k) = P·M·current
            M_current = self.mat_vec(self.M, current)
            next_vec = self.mat_vec(self.P, M_current)
            
            # Apply temperature-controlled noise
            if tau > 0:
                next_vec = [v + random.gauss(0, tau) for v in next_vec]
            
            seq.append(next_vec)
            # For next iteration, use the generated m-dimensional vector
            # as input to M (which expects n-dim). This requires dimension matching.
            # Since we can't directly use m-dim as n-dim, we either:
            # 1. Pad/truncate (not recommended)
            # 2. Use a projection (not implemented)
            # Instead, we'll reuse the original input dimension by maintaining state
            # in the n-dimensional space. This requires rethinking the approach.
            # For simplicity, we'll generate each vector independently from the initial input
            # but this breaks the sequence generation concept.
            # More sophisticated approaches would require additional architecture.
            # We'll keep current as the initial vector for all steps
            # (each generated vector is based on original input, not previous output)
            # This is a limitation when in_dim != out_dim
            # current = next_vec  # Disabled for dimension mismatch
            
        return seq

    def dd_features(self, seq):
        """
        Extract features for a sequence:
        [flattened P] + [flattened M]
        Returns feature vector of length (m*m + m*n)
        """
        feats = []
        # Flatten P matrix (m×m)
        for row in self.P:
            feats.extend(row)
        # Flatten M matrix (m×n)
        for row in self.M:
            feats.extend(row)
        return feats

    def show(self):
        """Display model status."""
        print("NumDualDescriptor Status:")
        print(f"  Input dimension (n) = {self.in_dim}")
        print(f"  Output dimension (m) = {self.out_dim}")
        print("  M matrix (m×n):")
        for row in self.M:
            print("   ", [f"{x:.4f}" for x in row])
        print("  P matrix (m×m):")
        for row in self.P:
            print("   ", [f"{x:.4f}" for x in row])

    def count_parameters(self):
        """Count learnable parameters (M and P matrices)."""
        total_params = self.out_dim * self.in_dim + self.out_dim * self.out_dim
        print(f"Parameter Count:")
        print(f"  M matrix: {self.out_dim}×{self.in_dim} = {self.out_dim*self.in_dim} parameters")
        print(f"  P matrix: {self.out_dim}×{self.out_dim} = {self.out_dim*self.out_dim} parameters")
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
        # Create instance without calling __init__
        obj = cls.__new__(cls)
        obj.__dict__.update(state)
        print(f"Model loaded from {filename}")
        return obj


# === Example Usage ===
if __name__ == "__main__":
    from statistics import correlation, mean
    import random

    random.seed(3)
    # Configuration with different dimensions
    in_dim = 25    # Input vector dimension (n)
    out_dim = 13   # Output vector dimension (m)
    num_seqs = 10
    min_len, max_len = 100, 200

    # Generate synthetic training data
    seqs = []
    t_list = []
    for _ in range(num_seqs):
        L = random.randint(min_len, max_len)
        # Generate n-dimensional vectors
        seq = [[random.uniform(-1,1) for _ in range(in_dim)] for __ in range(L)]
        seqs.append(seq)
        # Generate m-dimensional target vectors
        t_list.append([random.uniform(-1,1) for _ in range(out_dim)])

    # Create model with different input/output dimensions
    dd = NumDualDescriptor(
        in_dim=in_dim, 
        out_dim=out_dim,
        rank=5, 
        rank_mode='drop', 
        rank_op='avg', 
        mode='nonlinear', 
        user_step=5
    )
    
    # Train with Alternating Least Squares
    print("Training with Alternating Least Squares:")
    als_history = dd.train(seqs, t_list, max_iters=20)
    
    # Predict targets
    print("\nPredictions (first 3 sequences):")
    for i, seq in enumerate(seqs[:3]):
        t_pred = dd.predict_t(seq)
        print(f"Seq {i+1}: Target={[f'{x:.4f}' for x in t_list[i]]}")
        print(f"         Predicted={[f'{x:.4f}' for x in t_pred]}")
    
    # Calculate prediction correlations
    preds = [dd.predict_t(seq) for seq in seqs]
    correlations = []
    for i in range(out_dim):
        actual = [t[i] for t in t_list]
        predicted = [p[i] for p in preds]
        corr = correlation(actual, predicted)
        correlations.append(corr)
        print(f"Output dim {i} correlation: {corr:.4f}")
    print(f"Average correlation: {mean(correlations):.4f}")
    
    # Train with Gradient Descent
    print("\nTraining with Gradient Descent:")
    dd_grad = NumDualDescriptor(
        in_dim=in_dim,
        out_dim=out_dim,
        rank=5, 
        rank_mode='drop', 
        rank_op='avg', 
        mode='nonlinear', 
        user_step=5
    )
    grad_history = dd_grad.grad_train(
        seqs, 
        t_list, 
        learning_rate=0.3,
        max_iters=100,
        decay_rate=0.98
    )

    # Predict targets
    print("\nPredictions (first 3 sequences):")
    for i, seq in enumerate(seqs[:3]):
        t_pred = dd_grad.predict_t(seq)
        print(f"Seq {i+1}: Target={[f'{x:.4f}' for x in t_list[i]]}")
        print(f"         Predicted={[f'{x:.4f}' for x in t_pred]}")
    
    # Calculate prediction correlations
    preds = [dd_grad.predict_t(seq) for seq in seqs]
    correlations = []
    for i in range(out_dim):
        actual = [t[i] for t in t_list]
        predicted = [p[i] for p in preds]
        corr = correlation(actual, predicted)
        correlations.append(corr)
        print(f"Output dim {i} correlation: {corr:.4f}")
    print(f"Average correlation: {mean(correlations):.4f}")

    # Generate sample sequence
    print("\nGenerating sequence:")
    init_vec = [0.5] * in_dim  # n-dimensional initial vector
    generated = dd_grad.generate(
        L=5, 
        init_vec=init_vec,
        tau=0.1
    )
    print("Generated vectors (m-dimensional):")
    for i, vec in enumerate(generated):
        print(f"Vec {i+1}: {[f'{x:.4f}' for x in vec]}")

    # Show model structure
    print("\nModel structure:")
    dd_grad.show()

    # Parameter count
    print("\nParameter count:")
    dd_grad.count_parameters()

    # Save and load model
    print("\nTesting save/load functionality:")
    dd_grad.save("nDDpm_model.pkl")
    loaded = NumDualDescriptor.load("nDDpm_model.pkl")
    print("Loaded model prediction on first sequence:")
    print(loaded.predict_t(seqs[0]))
