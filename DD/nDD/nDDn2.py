# Copyright (C) 2005-2025, Bin-Guang Ma (mbg@mail.hzau.edu.cn). All rights reserved.
# The Dual Descriptor Vector class - Modified for n-dim Input/Output
# Author: Bin-Guang Ma; Date: 2025-6-12
# Modifications: Added support for n-dimensional input/output vectors

import math
import random
import pickle

class NumDualDescriptor:
    """
    Numeric Dual Descriptor for vector sequences with:
      - Learnable coefficient matrix Acoeff ∈ R^{m×L}
      - Learnable basis matrix Bbasis ∈ R^{L×m}
      - Learnable transformation matrix M ∈ R^{m×n} (input transformation)
      - Output transformation using M^T ∈ R^{n×m}
    """
    def __init__(self, input_dim, model_dim=4, rank=1, mode='linear', user_step=None):
        """
        Initialize the Dual Descriptor for vector sequences.
        
        Args:
            input_dim (int): Dimensionality of input vectors (n)
            model_dim (int): Internal model dimensionality (m)
            rank (int): Parameter controlling step size in nonlinear mode
            mode (str): 'linear' or 'nonlinear' processing mode
            user_step (int): Custom step size for nonlinear mode
        """
        self.rank = rank
        self.n = input_dim   # Input/output dimension
        self.m = model_dim   # Internal model dimension
        assert mode in ('linear', 'nonlinear')
        self.mode = mode
        self.step = user_step
        self.trained = False
        
        # Initialize transformation matrix M (m×n)
        self.M = [[random.uniform(-0.5, 0.5) for _ in range(self.n)] 
                  for _ in range(self.m)]
        
        # To be initialized in train()
        self.L = None        # Maximum sequence length
        self.Acoeff = None   # m×L coefficient matrix
        self.Bbasis = None   # L×m basis matrix
        self.B_t = None      # m×L (transpose of Bbasis)

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
        Extract vectors from sequence based on processing mode.
        
        Args:
            seq: List of n-dimensional vectors
            
        Returns:
            list: Selected vectors based on processing mode
        """
        if self.mode == 'linear':
            return seq
        step = self.step or self.rank
        return seq[::step]

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
        Returns internal m-dimensional descriptors.
        
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
            # Transform input vector: x = M * v (n→m)
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
        Returns internal m-dimensional descriptors.
        
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
        Compute mean squared error between n-dimensional outputs and targets.
        
        Args:
            seqs: List of vector sequences
            t_list: List of n-dimensional target vectors
            
        Returns:
            float: Mean squared error
        """
        total_error = 0.0
        position_count = 0
        for seq, t in zip(seqs, t_list):
            # Get predicted n-dimensional outputs
            outputs = []
            for Nk in self.describe(seq):
                # Transform to n-dim: y = M^T * Nk
                y = self._mat_vec(self._transpose(self.M), Nk)
                outputs.append(y)
                
            # Calculate MSE
            for y in outputs:
                for i in range(self.n):
                    error = y[i] - t[i]
                    total_error += error * error
                position_count += 1
        return total_error / position_count if position_count else 0.0

    # ---- training loop (gradient-based only) ----
    def grad_train(self, seqs, t_list, learning_rate=0.01, max_iters=100, 
                  tol=1e-9, print_every=10):
        """
        Train model using gradient descent with n-dim inputs/outputs.
        
        Args:
            seqs: List of training sequences (n-dim vectors)
            t_list: List of n-dimensional target vectors
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
            grad_M = [[0.0] * self.n for _ in range(self.m)]  # m x n matrix
            
            total_loss = 0.0
            total_positions = 0
            
            # Process all sequences
            for seq, t in zip(seqs, t_list):
                vecs = self.extract_vectors(seq)[:self.L]
                total_positions += len(vecs)
                
                for v in vecs:
                    # Forward pass (n→m→n)
                    # Input transformation: x = M * v (n→m)
                    x = self._mat_vec(self.M, v)
                    # Basis projection: z = Bbasis * x
                    z = self._mat_vec(self.Bbasis, x)
                    # Coefficient application: N = Acoeff * z
                    N = self._mat_vec(self.Acoeff, z)
                    # Output transformation: y = M^T * N (m→n)
                    y = self._mat_vec(self._transpose(self.M), N)
                    
                    # Compute error (n-dimensional)
                    error = [y_i - t_i for y_i, t_i in zip(y, t)]
                    total_loss += sum(e*e for e in error)
                    
                    # Backpropagation through output transformation
                    # dL/dN = M * error (n→m)
                    dN = self._mat_vec(self.M, error)
                    
                    # Backpropagation through coefficient matrix
                    # dL/dAcoeff = dN * z^T
                    for i in range(self.m):
                        for j in range(self.L):
                            grad_A[i][j] += 2 * dN[i] * z[j]
                    
                    # Backpropagation through basis matrix
                    # dL/dBbasis = (Acoeff^T @ dN) * x^T
                    Acoeff_t_dN = self._mat_vec(self._transpose(self.Acoeff), dN)
                    for j in range(self.L):
                        for i in range(self.m):
                            grad_B[j][i] += 2 * Acoeff_t_dN[j] * x[i]
                    
                    # Backpropagation through input transformation
                    # dL/dM_input = (Bbasis^T @ Acoeff^T @ dN) * v^T
                    B_t_Acoeff_t_dN = self._mat_vec(self.B_t, Acoeff_t_dN)
                    for i in range(self.m):
                        for j in range(self.n):
                            grad_M[i][j] += 2 * B_t_Acoeff_t_dN[i] * v[j]
                    
                    # Additional gradient from output transformation
                    # dL/dM_output = N * error^T
                    for i in range(self.m):
                        for j in range(self.n):
                            grad_M[i][j] += 2 * N[i] * error[j]
            
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
        self.mean_t = [0.0] * self.n  # Now n-dimensional
        for t in t_list:
            for i in range(self.n):
                self.mean_t[i] += t[i]
        self.mean_t = [x / len(t_list) for x in self.mean_t]
        self.trained = True
        
        return history

    def auto_train(self, seqs, learning_rate=0.01, max_iters=100, tol=1e-6,
                   print_every=10, auto_mode='reg'):
        """
        Self-supervised training for n-dimensional vector sequences.
        
        Args:
            seqs: List of training sequences (n-dim vectors)
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
            grad_M = [[0.0] * self.n for _ in range(self.m)]  # m x n matrix
            
            total_loss = 0.0
            total_instances = 0
            
            # Process all sequences
            for seq in seqs:
                vecs = self.extract_vectors(seq)[:self.L]
                n_vectors = len(vecs)
                
                if auto_mode == 'reg' and n_vectors < 2:
                    continue  # Skip short sequences
                
                # Determine processing range
                indices = range(n_vectors) if auto_mode == 'gap' else range(n_vectors - 1)
                
                for k in indices:
                    current_vec = vecs[k]
                    target_vec = current_vec if auto_mode == 'gap' else vecs[k + 1]
                    
                    # Forward pass (n→m→n)
                    x = self._mat_vec(self.M, current_vec)  # n→m
                    z = self._mat_vec(self.Bbasis, x)
                    N = self._mat_vec(self.Acoeff, z)
                    y = self._mat_vec(self._transpose(self.M), N)  # m→n
                    
                    # Compute error
                    error = [y_i - t_i for y_i, t_i in zip(y, target_vec)]
                    total_loss += sum(e*e for e in error)
                    total_instances += 1
                    
                    # Backpropagation through output transformation
                    dN = self._mat_vec(self.M, error)  # n→m
                    
                    # Backpropagation through coefficient matrix
                    for i in range(self.m):
                        for j in range(self.L):
                            grad_A[i][j] += 2 * dN[i] * z[j]
                    
                    # Backpropagation through basis matrix
                    Acoeff_t_dN = self._mat_vec(self._transpose(self.Acoeff), dN)
                    for j in range(self.L):
                        for i in range(self.m):
                            grad_B[j][i] += 2 * Acoeff_t_dN[j] * x[i]
                    
                    # Backpropagation through input transformation
                    B_t_Acoeff_t_dN = self._mat_vec(self.B_t, Acoeff_t_dN)
                    for i in range(self.m):
                        for j in range(self.n):
                            grad_M[i][j] += 2 * B_t_Acoeff_t_dN[i] * current_vec[j]
                    
                    # Additional gradient from output transformation
                    for i in range(self.m):
                        for j in range(self.n):
                            grad_M[i][j] += 2 * N[i] * error[j]
            
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
        # Compute mean output vector (n-dimensional)
        self.mean_t = [0.0] * self.n
        count = 0
        for seq in seqs:
            # Get n-dimensional outputs
            outputs = []
            for Nk in self.describe(seq):
                y = self._mat_vec(self._transpose(self.M), Nk)
                outputs.append(y)
            # Accumulate outputs
            for y in outputs:
                for i in range(self.n):
                    self.mean_t[i] += y[i]
                count += 1
        self.mean_t = [x / count for x in self.mean_t] if count > 0 else self.mean_t
        self.trained = True
        
        return history

    def predict_t(self, seq):
        """
        Predict n-dimensional target vector for a sequence.
        
        Args:
            seq: List of n-dimensional vectors
            
        Returns:
            list: Predicted target vector (n-dimensional)
        """
        N_list = self.describe(seq)
        if not N_list:
            return [0.0] * self.n
        
        # Average all position vectors (m-dimensional)
        N_avg = [0.0] * self.m
        for Nk in N_list:
            for i in range(self.m):
                N_avg[i] += Nk[i]
        n_pos = len(N_list)
        N_avg = [x / n_pos for x in N_avg]
        
        # Transform to n-dim: y = M^T * N_avg
        return self._mat_vec(self._transpose(self.M), N_avg)

    def generate(self, length, init_vec=None, tau=0.0):
        """
        Generate a sequence of n-dimensional vectors.
        
        Args:
            length: Number of vectors to generate
            init_vec: n-dimensional starting vector (random if None)
            tau: Temperature (0=deterministic, >0=stochastic)
            
        Returns:
            list: Generated sequence of n-dimensional vectors
        """
        assert self.trained, "Model must be trained before generation"
        if tau < 0:
            raise ValueError("Temperature must be non-negative")
        
        # Initialize sequence
        if init_vec is None:
            init_vec = [random.gauss(0, 1) for _ in range(self.n)]
        sequence = [init_vec]
        current_vec = init_vec
        
        for _ in range(length - 1):
            # Transform current vector: x = M * current_vec (n→m)
            x = self._mat_vec(self.M, current_vec)
            # Compute basis projection: z = Bbasis * x
            z = self._mat_vec(self.Bbasis, x)
            # Compute descriptor: N = Acoeff * z
            N = self._mat_vec(self.Acoeff, z)
            # Transform to n-dim: next_pred = M^T * N
            next_pred = self._mat_vec(self._transpose(self.M), N)
            
            # Apply temperature
            if tau == 0.0:
                next_vec = next_pred
            else:
                next_vec = [random.gauss(p, tau) for p in next_pred]
            
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
            # Transform vector (n→m)
            x = self._mat_vec(self.M, v)
            for i in range(self.m):
                weighted_sum[i] += self.Bbasis[j][i] * x[i]
        
        feats.extend(weighted_sum)
        return feats

    # ---- show state ----
    def show(self):
        """Display model state"""
        print("DualDescriptorVectorRNM status:")
        print(f" n={self.n}, m={self.m}, L={self.L}, mode={self.mode}")
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
    n = 5   # Input/output dimension
    m = 10  # Internal model dimension
    n_seqs = 10  # Number of training sequences

    # Generate sequences (n-dimensional vectors) and targets (n-dimensional)
    seqs, t_list = [], []
    for _ in range(n_seqs):
        L = random.randint(100, 200)
        seq = [[random.uniform(-1,1) for _ in range(n)] for __ in range(L)]        
        seqs.append(seq)
        t_list.append([random.uniform(-1,1) for _ in range(n)])  # n-dim targets
    
    # Create model with n=5, m=10
    dd = NumDualDescriptor(input_dim=n, model_dim=m, mode='nonlinear', user_step=2)
    
    # Train with Gradient Descent
    print("Training with Gradient Descent (n-dim input/output):")
    grad_history = dd.grad_train(
        seqs, 
        t_list, 
        learning_rate=0.1,
        max_iters=200,
        print_every=20
    )

    # Predict targets (n-dimensional)
    print("\nPredictions (n-dimensional outputs):")
    for i, seq in enumerate(seqs[:3]):  # First 3 sequences
        t_pred = dd.predict_t(seq)
        print(f"Seq {i+1}: Target={[f'{x:.4f}' for x in t_list[i]]}")
        print(f"         Predicted={[f'{x:.4f}' for x in t_pred]}")

    # Calculate prediction correlations
    preds = [dd.predict_t(seq) for seq in seqs]
    correlations = []
    for i in range(n):  # n-dimensional outputs
        actual = [t[i] for t in t_list]
        predicted = [p[i] for p in preds]
        corr = correlation(actual, predicted)
        correlations.append(corr)
        print(f"Dim {i} correlation: {corr:.4f}")
    print(f"Average correlation: {mean(correlations):.4f}")

    # Generate sequence (n-dimensional vectors)
    print("\nGenerating n-dimensional sequence:")
    generated_seq = dd.generate(length=5, tau=0.05)
    for i, vec in enumerate(generated_seq):
        print(f"Step {i}: {[f'{x:.4f}' for x in vec]}")

    # Self-supervised training example
    print("\nSelf-supervised training (next-vector prediction):")
    dd_auto = NumDualDescriptor(input_dim=n, model_dim=m)
    auto_history = dd_auto.auto_train(
        seqs,
        learning_rate=0.05,
        auto_mode='reg',
        max_iters=150,
        print_every=25
    )
