# Copyright (C) Bin-Guang Ma (mbg@mail.hzau.edu.cn). All rights reserved.
# The Dual Descriptor Vector class (P matrix form) for vector sequences
# Author: Bin-Guang Ma; Date: 2025-6-17

# The program is provided as it is and without warranty of any kind,
# either expressed or implied.

import math
import random
import pickle

class NumDualDescriptor:
    """
    Dual Descriptor for sequences of vectors with different input/output dimensions.
    - input_dim: dimension of input vectors (n)
    - output_dim: dimension of target vectors (l)
    - m: internal representation dimension
    
    Model: 
        For each vector x_k in sequence: 
            N(k) = P * M * x_k
            where M: m×n matrix, P: l×m matrix
        S(l) = cumulative sum of N(k) from k=1 to l
    """

    def __init__(self, input_dim=4, output_dim=4, m=4, 
                 rank=1, rank_mode='drop', rank_op='avg', 
                 mode='linear', user_step=None):
        """
        Initialize the Dual Descriptor for vector sequences.
        
        Args:
            input_dim (int): Dimensionality of input vectors (n)
            output_dim (int): Dimensionality of target vectors (l)
            m (int): Internal representation dimension
            rank (int): Parameter controlling step size in nonlinear mode
            rank_mode (str): 'pad' or 'drop' for handling incomplete windows
            rank_op (str): 'avg', 'sum', 'pick', or 'user_func' for window processing
            mode (str): 'linear' or 'nonlinear' processing mode
            user_step (int): Custom step size for nonlinear mode
        """
        self.rank = rank
        self.rank_mode = rank_mode
        self.rank_op = rank_op
        self.n = input_dim   # Input vector dimension
        self.l = output_dim  # Output vector dimension
        self.m = m           # Internal representation dimension
        assert mode in ('linear', 'nonlinear')
        self.mode = mode
        self.step = user_step
        self.trained = False

        # Initialize M matrix (m×n)
        self.M = [[random.uniform(-0.5, 0.5) for _ in range(self.n)] 
                  for _ in range(self.m)]
        
        # Initialize P matrix (l×m)
        self.P = [[random.uniform(-0.1, 0.1) for _ in range(self.m)] 
                  for _ in range(self.l)]

    # ---- sequence processing ----
    def extract_vectors(self, seq):
        """
        Extract vectors from sequence based on processing mode and rank operation.
        Returns list of vectors of dimension n (input_dim).
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
        """Multiply matrix M (a×b) by vector v (b-dimensional) → a-dimensional vector"""
        a = len(M)
        b = len(v)
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
        """Subtract two vectors of same dimension."""
        dim = len(u)
        return [u[i] - v[i] for i in range(dim)]

    def dot(self, u, v):
        """Dot product of two vectors of same dimension."""
        dim = len(u)
        return sum(u[i] * v[i] for i in range(dim))

    def describe(self, seq):
        """
        Compute list of N(k)=P·M·x_k for a given vector sequence.
        Each N(k) is an l-dimensional vector.
        """
        vecs = self.extract_vectors(seq)  # Returns n-dimensional vectors
        Nk_list = []
        for xk in vecs:
            # Transform input vector: M_xk = M · xk (m×n * n×1 → m×1)
            M_xk = self.mat_vec(self.M, xk)
            # Compute N(k) = P · (M · xk) (l×m * m×1 → l×1)
            Nk = self.mat_vec(self.P, M_xk)
            Nk_list.append(Nk)
        return Nk_list

    def S(self, seq):
        """
        Compute cumulative sum vectors S(l)=sum_{k=1}^{l} N(k) 
        for l = 1,...,L. Returns l-dimensional vectors.
        """
        Nk_list = self.describe(seq)
        S = [0.0] * self.l
        S_list = []
        for Nk in Nk_list:
            S = [S[i] + Nk[i] for i in range(self.l)]
            S_list.append(S.copy())
        return S_list

    def deviation(self, seqs, t_list):
        """
        Compute mean squared deviation D:
        D = average over all positions and sequences of ||P·M·x_k - t_j||^2.
        t_j are l-dimensional target vectors.
        """
        total = 0.0
        count = 0
        for seq, t in zip(seqs, t_list):
            for Nk in self.describe(seq):                
                # Compute error: Nk - t (both l-dimensional)
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
          U = sum_{j,k} (M·x_k)(M·x_k)^T  (m×m matrix)
          V = sum_{j,k} t_j (M·x_k)^T     (l×m matrix)
          P = V U^{-1} (l×m matrix)
        """
        U = [[0.0]*self.m for _ in range(self.m)]
        V = [[0.0]*self.m for _ in range(self.l)]
        
        for seq, t in zip(seqs, t_list):
            for xk in seq:
                # Compute M_xk = M · xk (m-dimensional)
                M_xk = self.mat_vec(self.M, xk)
                
                # Accumulate U += M_xk · M_xk^T
                for i in range(self.m):
                    for j in range(self.m):
                        U[i][j] += M_xk[i] * M_xk[j]
                
                # Accumulate V += t · M_xk^T
                for i in range(self.l):
                    for j in range(self.m):
                        V[i][j] += t[i] * M_xk[j]
        
        # Invert U and update P
        U_inv = self._invert_matrix(U)
        self.P = self.mat_mul(V, U_inv)

    def update_M(self, seqs, t_list):
        """
        Update M by closed-form solution:
          R = sum_{j,k} P^T t_j x_k^T  (m×n matrix)
          M = (P^T P)^{-1} R           (m×n matrix)
        """
        Pt = self.transpose(self.P)  # m×l matrix
        
        # Precompute (P^T P)^{-1} (m×m matrix)
        PtP = self.mat_mul(Pt, self.P)  # m×m
        PtP_inv = self._invert_matrix(PtP)
        
        # Accumulate R = sum(P^T t_j x_k^T)
        R = [[0.0]*self.n for _ in range(self.m)]
        for seq, t in zip(seqs, t_list):
            for xk in seq:
                # Compute P^T t (m-dimensional)
                Pt_t = self.mat_vec(Pt, t)
                # Accumulate R += (P^T t) · xk^T
                for i in range(self.m):
                    for j in range(self.n):
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
        self.mean_vector_count = total_vectors / len(seqs)
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
        
        for it in range(max_iters):
            total_vectors = 0
            grad_P = [[0.0]*self.m for _ in range(self.l)]
            grad_M = [[0.0]*self.n for _ in range(self.m)]
            
            # Compute gradients
            for seq, t in zip(seqs, t_list):
                for xk in seq:
                    total_vectors += 1
                    # Forward pass
                    M_xk = self.mat_vec(self.M, xk)        # m-dim
                    Nk = self.mat_vec(self.P, M_xk)         # l-dim
                    error = self.vec_sub(Nk, t)             # l-dim
                    
                    # Gradient for P: dD/dP = 2 * error * (M_xk)^T
                    for i in range(self.l):
                        for j in range(self.m):
                            grad_P[i][j] += 2 * error[i] * M_xk[j]
                    
                    # Gradient for M: dD/dM = 2 * P^T error * xk^T
                    Pt_error = self.mat_vec(self.transpose(self.P), error)  # m-dim
                    for i in range(self.m):
                        for j in range(self.n):
                            grad_M[i][j] += 2 * Pt_error[i] * xk[j]
            
            # Apply updates
            if total_vectors > 0:
                lr = learning_rate / total_vectors
                for i in range(self.l):
                    for j in range(self.m):
                        self.P[i][j] -= lr * grad_P[i][j]
                for i in range(self.m):
                    for j in range(self.n):
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
        self.mean_vector_count = total_vectors / len(seqs)
        self.trained = True
        return history

    def auto_train(self, seqs, max_iters=100, tol=1e-6, learning_rate=0.01, 
                   continued=False, auto_mode='reg', decay_rate=0.95):
        """
        Self-training with two modes (requires n = l):
          'gap': Predict current vector (self-consistency)
          'reg': Predict next vector (auto-regressive)
        """
        if self.n != self.l:
            raise ValueError("auto_train requires input_dim = output_dim")
            
        if auto_mode not in ('gap', 'reg'):
            raise ValueError("auto_mode must be 'gap' or 'reg'")
            
        if not continued:
            self.M = [[random.uniform(-0.5, 0.5) for _ in range(self.n)] 
                      for _ in range(self.m)]
            self.P = [[random.uniform(-0.1, 0.1) for _ in range(self.m)] 
                      for _ in range(self.l)]
        
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
            grad_P = [[0.0]*self.m for _ in range(self.l)]
            grad_M = [[0.0]*self.n for _ in range(self.m)]
            total_loss = 0.0
            
            for seq in seqs:
                L = len(seq)
                if L == 0:
                    continue
                    
                for k in range(L):
                    if auto_mode == 'reg' and k == L-1:
                        continue
                    
                    xk = seq[k]
                    # Forward pass: Nk = P·M·xk (l-dim)
                    M_xk = self.mat_vec(self.M, xk)
                    Nk = self.mat_vec(self.P, M_xk)
                    
                    # Set target (n-dim, but n=l)
                    if auto_mode == 'gap':
                        target = xk
                    else:  # reg
                        target = seq[k+1]
                    
                    # Compute error and loss
                    error = [Nk[i] - target[i] for i in range(self.l)]
                    sq_error = sum(e**2 for e in error)
                    total_loss += sq_error
                    
                    # Compute gradients
                    # Gradient w.r.t P: 2 * error * (M_xk)^T
                    for i in range(self.l):
                        for j in range(self.m):
                            grad_P[i][j] += 2 * error[i] * M_xk[j]
                    
                    # Gradient w.r.t M: 2 * (P^T error) * xk^T
                    Pt_error = self.mat_vec(self.transpose(self.P), error)
                    for i in range(self.m):
                        for j in range(self.n):
                            grad_M[i][j] += 2 * Pt_error[i] * xk[j]
            
            # Apply updates
            avg_loss = total_loss / total_samples
            lr = learning_rate
            for i in range(self.l):
                for j in range(self.m):
                    self.P[i][j] -= lr * grad_P[i][j] / total_samples
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
        Predict target vector t for a sequence.
        Optimal t is the mean of all N(k) vectors (l-dimensional).
        """
        Nk_list = self.describe(seq)
        if not Nk_list:
            return [0.0] * self.l
            
        t_pred = [0.0] * self.l
        for vec in Nk_list:
            for i in range(self.l):
                t_pred[i] += vec[i]
        return [x / len(Nk_list) for x in t_pred]

    def generate(self, L, init_vec=None, tau=0.0):
        """
        Generate vector sequence of length L (l-dimensional vectors).
        init_vec: starting vector (n-dimensional)
        tau: temperature for randomness
        """
        assert self.trained, "Model must be trained first"
        if tau < 0:
            raise ValueError("Temperature must be non-negative")
        
        # Initialize sequence
        seq = []
        if init_vec is None:
            current = [random.gauss(0, 1) for _ in range(self.n)]
        else:
            current = init_vec.copy()
        seq.append(current)
        
        # Generate subsequent vectors
        for _ in range(1, L):
            # Compute N(k) = P·M·current (l-dimensional output)
            M_current = self.mat_vec(self.M, current)  # m-dim
            next_vec = self.mat_vec(self.P, M_current)  # l-dim
            
            # Apply temperature-controlled noise
            if tau > 0:
                next_vec = [v + random.gauss(0, tau) for v in next_vec]
            
            seq.append(next_vec)
            current = next_vec
        
        return seq

    def dd_features(self, seq):
        """
        Extract features for a sequence:
        [flattened P] + [flattened M]
        """
        feats = []
        # Flatten P matrix (l×m)
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
        print(f"  Output dimension l = {self.l}")
        print(f"  Internal dimension m = {self.m}")
        print("  M matrix (m×n):")
        for row in self.M:
            print("   ", [f"{x:.4f}" for x in row])
        print("  P matrix (l×m):")
        for row in self.P:
            print("   ", [f"{x:.4f}" for x in row])

    def count_parameters(self):
        """Count learnable parameters (M and P matrices)."""
        total_params = self.m * self.n + self.l * self.m
        print(f"Parameter Count:")
        print(f"  M matrix: {self.m}×{self.n} = {self.m*self.n} parameters")
        print(f"  P matrix: {self.l}×{self.m} = {self.l*self.m} parameters")
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
    import random

    random.seed(3)
    # Configuration with different input/output dimensions
    input_dim = 10   # Input vector dimension (n)
    output_dim = 8   # Target vector dimension (l)
    m = 20           # Internal representation dimension
    num_seqs = 10
    min_len, max_len = 100, 200

    # Generate synthetic training data
    seqs = []    # List of sequences (n-dimensional vectors)
    t_list = []  # List of target vectors (l-dimensional)
    for _ in range(num_seqs):
        L = random.randint(min_len, max_len)
        seq = [[random.uniform(-1,1) for _ in range(input_dim)] for __ in range(L)]
        seqs.append(seq)
        t_list.append([random.uniform(-1,1) for _ in range(output_dim)])

    # Create model
    dd = NumDualDescriptor(
        input_dim=input_dim,
        output_dim=output_dim,
        m=m,
        rank=5,
        rank_mode='drop',
        rank_op='avg',
        mode='nonlinear',
        user_step=5
    )
    
    # Train with ALS
    print("Training with Alternating Least Squares:")
    als_history = dd.train(seqs, t_list, max_iters=20)
    
    # Predict targets
    print("\nPredictions:")
    for i, seq in enumerate(seqs[:3]):  # First 3 sequences
        t_pred = dd.predict_t(seq)
        print(f"Seq {i+1}: Target={[f'{x:.4f}' for x in t_list[i]]}")
        print(f"         Predicted={[f'{x:.4f}' for x in t_pred]}")
    
    # Calculate prediction correlations
    preds = [dd.predict_t(seq) for seq in seqs]
    correlations = []
    for i in range(output_dim):
        actual = [t[i] for t in t_list]
        predicted = [p[i] for p in preds]
        corr = correlation(actual, predicted)
        correlations.append(corr)
        print(f"Dim {i} correlation: {corr:.4f}")
    print(f"Average correlation: {mean(correlations):.4f}")
    
    # Train with Gradient Descent
    print("\nTraining with Gradient Descent:")
    dd_grad = NumDualDescriptor(
        input_dim=input_dim,
        output_dim=output_dim,
        m=m,
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

    # Calculate prediction correlations
    preds = [dd_grad.predict_t(seq) for seq in seqs]
    correlations = []
    for i in range(output_dim):
        actual = [t[i] for t in t_list]
        predicted = [p[i] for p in preds]
        corr = correlation(actual, predicted)
        correlations.append(corr)
        print(f"Dim {i} correlation: {corr:.4f}")
    print(f"Average correlation: {mean(correlations):.4f}")

    # Parameter count
    print("\nParameter count:")
    dd.count_parameters()

    # Auto-train example (requires n=l)
    print("\nAuto-train example (requires input_dim=output_dim):")
    auto_dim = 8  # Must be same for input and output
    auto_seqs = [
        [[random.uniform(-1,1) for _ in range(auto_dim)] 
        for _ in range(random.randint(50,100))]
        for _ in range(5)
    ]
    
    dd_auto = NumDualDescriptor(
        input_dim=auto_dim,
        output_dim=auto_dim,
        m=12,
        rank=3,
        mode='nonlinear',
        user_step=2
    )
    
    # Auto-train in 'gap' mode (autoencoder)
    print("\nAuto-train in 'gap' mode (autoencoder):")
    gap_history = dd_auto.auto_train(
        auto_seqs, 
        learning_rate=0.1,
        max_iters=50,
        auto_mode='gap',
        decay_rate=0.98
    )
    print(f"Final autoencoder loss: {gap_history[-1]:.6f}")

    # Auto-train in 'reg' mode (next-vector prediction)
    print("\nAuto-train in 'reg' mode (next-vector prediction):")
    reg_history = dd_auto.auto_train(
        auto_seqs, 
        learning_rate=0.1,
        max_iters=50,
        auto_mode='reg',
        decay_rate=0.98
    )
    print(f"Final next-vector prediction loss: {reg_history[-1]:.6f}")

    # Sequence generation example
    print("\nGenerating sequence:")
    generated_seq = dd_auto.generate(
        L=5,
        init_vec=auto_seqs[0][0],  # Start with first vector of first sequence
        tau=0.05  # Small randomness
    )
    print("Generated sequence (l-dimensional vectors):")
    for i, vec in enumerate(generated_seq):
        print(f"Vec {i+1}: {[f'{x:.4f}' for x in vec]}")

    # Feature extraction example
    print("\nFeature extraction example:")
    sample_seq = auto_seqs[0][:10]  # First 10 vectors of first sequence
    features = dd_auto.dd_features(sample_seq)
    print(f"Extracted {len(features)} features")
    print(f"First 5 features: {[f'{x:.6f}' for x in features[:5]]}")
    print(f"Last 5 features: {[f'{x:.6f}' for x in features[-5:]]}")

    # Show model parameters
    print("\nModel state visualization:")
    dd_auto.show()

    # Save/load example
    print("\nModel saving/loading example:")
    dd_auto.save("dd_auto_model.pkl")
    loaded_model = NumDualDescriptor.load("dd_auto_model.pkl")
    print("Loaded model show:")
    loaded_model.show()
