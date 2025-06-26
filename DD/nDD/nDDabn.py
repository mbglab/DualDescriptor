# Copyright (C) 2005-2025, Bin-Guang Ma (mbg@mail.hzau.edu.cn). All rights reserved.
# The Dual Descriptor Vector class
# Author: Bin-Guang Ma; Date: 2025-6-10

# The program is provided as it is and without warranty of any kind,
# either expressed or implied.

import math
import random
import pickle

class NumDualDescriptor:
    """
    Numeric Dual Descriptor for vector sequences with:
      - Learnable coefficient matrix Acoeff ∈ R^{m×L}
      - Learnable basis matrix Bbasis ∈ R^{L×m}
      - Learnable transformation matrix M ∈ R^{m×m} (internal transformation)
      - Learnable input projection matrix M_in ∈ R^{m×n} (input to internal)
      - Learnable output projection matrix M_out ∈ R^{n×m} (internal to output)
    """
    def __init__(self, in_dim=10, model_dim=10, out_dim=10, basis_dim=50, rank=1, rank_mode='drop', rank_op='avg', mode='linear', user_step=None):
        """
        Initialize the Dual Descriptor for vector sequences.
        
        Args:
            in_dim (int): Dimensionality of input vectors (n)
            model_dim (int): Internal representation dimensionality (m)
            out_dim (int): Dimensionality of output vectors (n)
            basis_dim (int): Dimensionality of basis vectors (L)
            rank (int): Parameter controlling step size in nonlinear mode
            mode (str): 'linear' or 'nonlinear' processing mode
            user_step (int): Custom step size for nonlinear mode
        """
        self.n = in_dim          # Input dimension
        self.m = model_dim    # Internal dimension
        self.out_dim = out_dim   # Output dimension
        self.L = basis_dim
        self.rank = rank
        self.rank_mode = rank_mode
        self.rank_op = rank_op        
        assert mode in ('linear', 'nonlinear')
        self.mode = mode
        self.step = user_step
        self.trained = False
        
        # Initialize input projection matrix M_in (m×n)
        self.M_in = [[random.uniform(-0.5, 0.5) for _ in range(self.n)] 
                     for _ in range(self.m)]
        
        # Initialize internal transformation matrix M (m×m)
        self.M = [[random.uniform(-0.5, 0.5) for _ in range(self.m)] 
                  for _ in range(self.m)]
        
        # Initialize output projection matrix M_out (n×m)
        self.M_out = [[random.uniform(-0.5, 0.5) for _ in range(self.m)] 
                      for _ in range(self.out_dim)]
        
        # Initialize Acoeff (m×L)
        self.Acoeff = [[random.uniform(-0.1, 0.1) for _ in range(self.L)]
                       for _ in range(self.m)]
        
        # Initialize Bbasis (L×m)
        self.Bbasis = [[random.uniform(-0.1, 0.1) for _ in range(self.m)]
                       for _ in range(self.L)]
        
        # Cache transpose
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
    

    # ---- describe sequence ----
    def describe(self, seq):
        """
        Compute descriptor vectors for each position in the sequence.
        
        Args:
            seq: List of n-dimensional vectors
            
        Returns:
            list: Descriptor vectors (N) for each position (internal m-dim)
        """
        vecs = self.extract_vectors(seq)
        N = []
        for k, v in enumerate(vecs):
            # Project input to internal space: x_in = M_in * v
            x_in = self._mat_vec(self.M_in, v)
            # Apply internal transformation: x = M * x_in
            x = self._mat_vec(self.M, x_in)
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
            list: Cumulative descriptor vectors (internal m-dim)
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
        Compute mean squared error between projected descriptors and targets.
        
        Args:
            seqs: List of vector sequences (each vector n-dim)
            t_list: List of target vectors (n-dim)
            
        Returns:
            float: Mean squared error
        """
        total_error = 0.0
        position_count = 0
        
        for seq, t in zip(seqs, t_list):
            for Nk in self.describe(seq):
                # Project internal descriptor to output space: t_pred = M_out * Nk
                t_pred = self._mat_vec(self.M_out, Nk)
                # Compute error in output space
                for i in range(self.out_dim):
                    error = t_pred[i] - t[i]
                    total_error += error * error
                position_count += 1
        
        return total_error / position_count if position_count else 0.0

    def grad_train(self, seqs, t_list, learning_rate=0.01, max_iters=100, 
                  tol=1e-9, print_every=10, lr_decay=1.0):
        """
        Train model using gradient descent with learning rate decay.
        
        Args:
            seqs: List of training sequences (n-dim vectors)
            t_list: List of target vectors (n-dim)
            learning_rate: Initial step size
            max_iters: Maximum iterations
            tol: Convergence tolerance
            print_every: Print interval
            lr_decay: Learning rate decay factor per iteration (default=1.0, no decay)
            
        Returns:
            list: Training loss history
        """        
        history = []
        prev_loss = float('inf')
        original_lr = learning_rate  # Store original learning rate
        
        for it in range(max_iters):
            # Initialize gradients
            grad_A = [[0.0] * self.L for _ in range(self.m)]
            grad_B = [[0.0] * self.m for _ in range(self.L)]
            grad_M = [[0.0] * self.m for _ in range(self.m)]
            grad_M_in = [[0.0] * self.n for _ in range(self.m)]
            grad_M_out = [[0.0] * self.m for _ in range(self.out_dim)]
            
            total_loss = 0.0
            total_positions = 0
            
            # Process all sequences
            for seq, t in zip(seqs, t_list):
                vecs = self.extract_vectors(seq)
                total_positions += len(vecs)
                
                for v in vecs:
                    # Forward pass
                    x_in = self._mat_vec(self.M_in, v)   # M_in * v (n→m)
                    x = self._mat_vec(self.M, x_in)       # M * x_in (m→m)
                    z = self._mat_vec(self.Bbasis, x)     # Bbasis * x (m→L)
                    N = self._mat_vec(self.Acoeff, z)     # Acoeff * z (L→m)
                    t_pred = self._mat_vec(self.M_out, N) # M_out * N (m→n)
                    
                    # Compute error in output space
                    error = [t_pred_i - t_i for t_pred_i, t_i in zip(t_pred, t)]
                    total_loss += sum(e*e for e in error)
                    
                    # Backpropagation
                    # Gradient at output: dL/dt_pred = 2*error (n-dim)
                    d_output = [2 * e for e in error]
                    
                    # Backprop through M_out: dL/dM_out = d_output * N^T
                    for i in range(self.out_dim):
                        for j in range(self.m):
                            grad_M_out[i][j] += d_output[i] * N[j]
                    
                    # Backprop to N: dL/dN = M_out^T * d_output
                    d_N = self._mat_vec(self._transpose(self.M_out), d_output)
                    
                    # Backprop through Acoeff: dL/dAcoeff = d_N * z^T
                    for i in range(self.m):
                        for j in range(self.L):
                            grad_A[i][j] += d_N[i] * z[j]
                    
                    # Backprop to z: dL/dz = Acoeff^T * d_N
                    d_z = self._mat_vec(self._transpose(self.Acoeff), d_N)
                    
                    # Backprop through Bbasis: dL/dBbasis = d_z * x^T
                    for i in range(self.L):
                        for j in range(self.m):
                            grad_B[i][j] += d_z[i] * x[j]
                    
                    # Backprop to x: dL/dx = Bbasis^T * d_z
                    d_x = self._mat_vec(self.B_t, d_z)
                    
                    # Backprop through M: dL/dM = d_x * x_in^T
                    for i in range(self.m):
                        for j in range(self.m):
                            grad_M[i][j] += d_x[i] * x_in[j]
                    
                    # Backprop to x_in: dL/dx_in = M^T * d_x
                    d_x_in = self._mat_vec(self._transpose(self.M), d_x)
                    
                    # Backprop through M_in: dL/dM_in = d_x_in * v^T
                    for i in range(self.m):
                        for j in range(self.n):
                            grad_M_in[i][j] += d_x_in[i] * v[j]
            
            # Average gradients
            if total_positions > 0:
                norm = 1 / total_positions
                # Acoeff gradient
                for i in range(self.m):
                    for j in range(self.L):
                        grad_A[i][j] *= norm
                # Bbasis gradient
                for j in range(self.L):
                    for i in range(self.m):
                        grad_B[j][i] *= norm
                # M gradient
                for i in range(self.m):
                    for j in range(self.m):
                        grad_M[i][j] *= norm
                # M_in gradient
                for i in range(self.m):
                    for j in range(self.n):
                        grad_M_in[i][j] *= norm
                # M_out gradient
                for i in range(self.out_dim):
                    for j in range(self.m):
                        grad_M_out[i][j] *= norm
            
            # Update parameters
            for i in range(self.m):
                for j in range(self.L):
                    self.Acoeff[i][j] -= learning_rate * grad_A[i][j]
            
            for j in range(self.L):
                for i in range(self.m):
                    self.Bbasis[j][i] -= learning_rate * grad_B[j][i]
            
            for i in range(self.m):
                for j in range(self.m):
                    self.M[i][j] -= learning_rate * grad_M[i][j]
            
            for i in range(self.m):
                for j in range(self.n):
                    self.M_in[i][j] -= learning_rate * grad_M_in[i][j]
            
            for i in range(self.out_dim):
                for j in range(self.m):
                    self.M_out[i][j] -= learning_rate * grad_M_out[i][j]
            
            # Update B_t cache
            self.B_t = self._transpose(self.Bbasis)
            
            # Calculate and record loss
            avg_loss = total_loss / total_positions if total_positions else 0
            history.append(avg_loss)
            
            # Apply learning rate decay
            learning_rate *= lr_decay
            
            # Print progress
            if it % print_every == 0 or it == max_iters - 1:
                print(f"Iter {it:3d}: Loss = {avg_loss:.6f}, LR = {learning_rate:.6f}")
            
            # Check convergence
            if abs(prev_loss - avg_loss) < tol:
                print(f"Converged after {it} iterations")
                break
            prev_loss = avg_loss
        
        # Final statistics
        self.mean_L = int(sum(len(seq) for seq in seqs) / len(seqs))
        self.mean_t = [0.0] * self.out_dim
        for t in t_list:
            for i in range(self.out_dim):
                self.mean_t[i] += t[i]
        self.mean_t = [x / len(t_list) for x in self.mean_t]
        self.trained = True
        
        return history

    def auto_train(self, seqs, learning_rate=0.01, max_iters=100, tol=1e-6,
                   print_every=10, auto_mode='reg', lr_decay=1.0):
        """
        Self-supervised training for vector sequences with learning rate decay.
        
        Args:
            seqs: List of training sequences (n-dim vectors)
            learning_rate: Initial step size
            max_iters: Maximum iterations
            tol: Convergence tolerance
            print_every: Print interval
            auto_mode: 'gap' (autoencoder) or 'reg' (next-vector prediction)
            lr_decay: Learning rate decay factor per iteration (default=1.0, no decay)
            
        Returns:
            list: Training loss history
        """
        assert auto_mode in ('gap', 'reg')        
        history = []
        prev_loss = float('inf')
        original_lr = learning_rate  # Store original learning rate
        
        for it in range(max_iters):
            # Initialize gradients
            grad_A = [[0.0] * self.L for _ in range(self.m)]
            grad_B = [[0.0] * self.m for _ in range(self.L)]
            grad_M = [[0.0] * self.m for _ in range(self.m)]
            grad_M_in = [[0.0] * self.n for _ in range(self.m)]
            grad_M_out = [[0.0] * self.m for _ in range(self.out_dim)]
            
            total_loss = 0.0
            total_instances = 0
            
            # Process all sequences
            for seq in seqs:
                vecs = self.extract_vectors(seq)
                n = len(vecs)
                
                if auto_mode == 'reg' and n < 2:
                    continue  # Skip short sequences
                
                # Determine processing range
                indices = range(n) if auto_mode == 'gap' else range(n - 1)
                
                for k in indices:
                    current_vec = vecs[k]
                    # Target is same vector for autoencoder, next vector for regression
                    target_vec = vecs[k] if auto_mode == 'gap' else vecs[k + 1]
                    
                    # Forward pass
                    x_in = self._mat_vec(self.M_in, current_vec)   # n→m
                    x = self._mat_vec(self.M, x_in)                # m→m
                    z = self._mat_vec(self.Bbasis, x)              # m→L
                    N = self._mat_vec(self.Acoeff, z)              # L→m
                    pred_vec = self._mat_vec(self.M_out, N)        # m→n
                    
                    # Compute error in output space
                    error = [pred_vec_i - target_vec_i 
                             for pred_vec_i, target_vec_i in zip(pred_vec, target_vec)]
                    total_loss += sum(e*e for e in error)
                    total_instances += 1
                    
                    # Backpropagation
                    # Gradient at output: dL/dpred = 2*error (n-dim)
                    d_output = [2 * e for e in error]
                    
                    # Backprop through M_out: dL/dM_out = d_output * N^T
                    for i in range(self.out_dim):
                        for j in range(self.m):
                            grad_M_out[i][j] += d_output[i] * N[j]
                    
                    # Backprop to N: dL/dN = M_out^T * d_output
                    d_N = self._mat_vec(self._transpose(self.M_out), d_output)
                    
                    # Backprop through Acoeff: dL/dAcoeff = d_N * z^T
                    for i in range(self.m):
                        for j in range(self.L):
                            grad_A[i][j] += d_N[i] * z[j]
                    
                    # Backprop to z: dL/dz = Acoeff^T * d_N
                    d_z = self._mat_vec(self._transpose(self.Acoeff), d_N)
                    
                    # Backprop through Bbasis: dL/dBbasis = d_z * x^T
                    for i in range(self.L):
                        for j in range(self.m):
                            grad_B[i][j] += d_z[i] * x[j]
                    
                    # Backprop to x: dL/dx = Bbasis^T * d_z
                    d_x = self._mat_vec(self.B_t, d_z)
                    
                    # Backprop through M: dL/dM = d_x * x_in^T
                    for i in range(self.m):
                        for j in range(self.m):
                            grad_M[i][j] += d_x[i] * x_in[j]
                    
                    # Backprop to x_in: dL/dx_in = M^T * d_x
                    d_x_in = self._mat_vec(self._transpose(self.M), d_x)
                    
                    # Backprop through M_in: dL/dM_in = d_x_in * current_vec^T
                    for i in range(self.m):
                        for j in range(self.n):
                            grad_M_in[i][j] += d_x_in[i] * current_vec[j]
            
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
                    for j in range(self.m):
                        grad_M[i][j] *= norm
                for i in range(self.m):
                    for j in range(self.n):
                        grad_M_in[i][j] *= norm
                for i in range(self.out_dim):
                    for j in range(self.m):
                        grad_M_out[i][j] *= norm
            
            # Update parameters
            for i in range(self.m):
                for j in range(self.L):
                    self.Acoeff[i][j] -= learning_rate * grad_A[i][j]
            
            for j in range(self.L):
                for i in range(self.m):
                    self.Bbasis[j][i] -= learning_rate * grad_B[j][i]
            
            for i in range(self.m):
                for j in range(self.m):
                    self.M[i][j] -= learning_rate * grad_M[i][j]
            
            for i in range(self.m):
                for j in range(self.n):
                    self.M_in[i][j] -= learning_rate * grad_M_in[i][j]
            
            for i in range(self.out_dim):
                for j in range(self.m):
                    self.M_out[i][j] -= learning_rate * grad_M_out[i][j]
            
            # Update B_t cache
            self.B_t = self._transpose(self.Bbasis)
            
            # Apply learning rate decay
            learning_rate *= lr_decay
            
            # Calculate and record loss
            avg_loss = total_loss / total_instances if total_instances else 0
            history.append(avg_loss)
            
            # Print progress
            if it % print_every == 0 or it == max_iters - 1:
                mode_str = "Autoencoder" if auto_mode == 'gap' else "Next-vector"
                print(f"Iter {it:3d} ({mode_str}): Loss = {avg_loss:.6f}, LR = {learning_rate:.6f}")
            
            # Check convergence
            if abs(prev_loss - avg_loss) < tol:
                print(f"Converged after {it} iterations")
                break
            prev_loss = avg_loss
        
        # Final statistics
        self.mean_L = int(sum(len(seq) for seq in seqs) / len(seqs))
        # Compute mean descriptor vector in output space
        self.mean_t = [0.0] * self.out_dim
        count = 0
        for seq in seqs:
            # Get predicted target for each sequence
            t_pred = self.predict_t(seq)
            for i in range(self.out_dim):
                self.mean_t[i] += t_pred[i]
            count += 1
        self.mean_t = [x / count for x in self.mean_t] if count > 0 else self.mean_t
        self.trained = True
        
        return history

    def predict_t(self, seq):
        """
        Predict target vector for a sequence (outputs n-dim vector).
        
        Args:
            seq: List of n-dimensional vectors
            
        Returns:
            list: Predicted target vector (n-dimensional)
        """
        N_list = self.describe(seq)
        if not N_list:
            return [0.0] * self.out_dim
        
        # Average all position vectors in internal space
        avg_N = [0.0] * self.m
        for Nk in N_list:
            for i in range(self.m):
                avg_N[i] += Nk[i]
        
        n_pos = len(N_list)
        avg_N = [x / n_pos for x in avg_N]
        
        # Project to output space: t_pred = M_out * avg_N
        return self._mat_vec(self.M_out, avg_N)

    def generate(self, length, init_vec=None, tau=0.0):
        """
        Generate a sequence of n-dimensional vectors.
        
        Args:
            length: Number of vectors to generate
            init_vec: Starting n-dim vector (random if None)
            tau: Temperature (0=deterministic, >0=stochastic)
            
        Returns:
            list: Generated sequence of n-dimensional vectors
        """
        assert self.trained, "Model must be trained before generation"
        if tau < 0:
            raise ValueError("Temperature must be non-negative")
        
        # Initialize sequence with n-dim vector
        if init_vec is None:
            init_vec = [random.gauss(0, 1) for _ in range(self.n)]
        sequence = [init_vec]
        current_vec = init_vec
        
        for _ in range(length - 1):
            # Project current vector to internal space
            x_in = self._mat_vec(self.M_in, current_vec)
            # Apply internal transformation
            x = self._mat_vec(self.M, x_in)
            # Compute basis activation
            z = self._mat_vec(self.Bbasis, x)
            # Predict next internal vector
            next_internal = self._mat_vec(self.Acoeff, z)
            
            # Apply temperature for stochasticity
            if tau > 0:
                next_internal = [random.gauss(p, tau) for p in next_internal]
            
            # Project back to output space
            next_vec = self._mat_vec(self.M_out, next_internal)
            sequence.append(next_vec)
            current_vec = next_vec
        
        return sequence

    # ---- feature extraction ----
    def dd_features(self, seq):
        """Extract features from a sequence (unchanged)"""
        feats = []
        
        # Flatten Acoeff
        for row in self.Acoeff:
            feats.extend(row)
        
        # Flatten M
        for row in self.M:
            feats.extend(row)
        
        # Basis-weighted features
        vecs = self.extract_vectors(seq)
        weighted_sum = [0.0] * self.m
        for j, v in enumerate(vecs):
            # Transform vector
            x_in = self._mat_vec(self.M_in, v)
            x = self._mat_vec(self.M, x_in)
            for i in range(self.m):
                weighted_sum[i] += self.Bbasis[j][i] * x[i]
        
        feats.extend(weighted_sum)
        return feats

    # ---- show state ----
    def show(self):
        """Display model state"""
        print("DualDescriptorVector status:")
        print(f" Input dim (n)={self.n}, Internal dim (m)={self.m}, Output dim={self.out_dim}")
        print(f" L={self.L}, mode={self.mode}")
        print(" Sample M_in[0][:5]:", self.M_in[0][:5])
        print(" Sample M[0][:5]:", self.M[0][:5])
        print(" Sample M_out[0][:5]:", self.M_out[0][:5])
        print(" Sample Acoeff[0][:5]:", self.Acoeff[0][:5])
        print(" Sample Bbasis[0][:5]:", self.Bbasis[0][:5])

    def count_parameters(self):
        """
        Calculate and print the number of learnable parameters in the model.        
        Returns:
            int: Total number of learnable parameters
        """
        if self.L is None:
            print("Model not initialized. Please train or initialize first.")
            return 0            
        # Calculate parameter counts
        m_in_params = self.m * self.n
        m_params = self.m * self.m
        m_out_params = self.out_dim * self.m
        a_params = self.m * self.L
        b_params = self.L * self.m
        
        total_params = (m_in_params + m_params + m_out_params + 
                        a_params + b_params)
        
        # Print parameter information
        print(f"Model Parameter Counts:")
        print(f"- M_in ({self.m}×{self.n}): {m_in_params:,} parameters")
        print(f"- M ({self.m}×{self.m}): {m_params:,} parameters")
        print(f"- M_out ({self.out_dim}×{self.m}): {m_out_params:,} parameters")
        print(f"- Acoeff ({self.m}×{self.L}): {a_params:,} parameters")
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
        
        # Create instance without __init__
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
    in_dim = 20    # Input vector dimension (n)
    model_dim = 30  # Internal representation dimension (m)
    out_dim = 10    # Output vector dimension (n)
    basis_dim = 50    # Basis dimension (L)
    n_seqs = 10     # Number of training sequences

    # Generate 10 sequences of length 100-200 with n-dim vectors
    # and random n-dim target vectors
    seqs, t_list = [], []
    for _ in range(n_seqs):
        length = random.randint(100, 200)
        seq = [[random.uniform(-1, 1) for _ in range(in_dim)] 
               for __ in range(length)]
        seqs.append(seq)
        t_list.append([random.uniform(-1, 1) for _ in range(out_dim)])
    
    # Create model with different dimensions
    dd = NumDualDescriptor(
        in_dim=in_dim,
        model_dim=model_dim,
        out_dim=out_dim,
        basis_dim=basis_dim,
        rank=1,
        rank_mode='drop',
        rank_op='avg',
        mode='nonlinear',
        user_step=1
    )
    
    # Train with Gradient Descent (ALS requires m==out_dim)
    print("\nTraining with Gradient Descent:")
    grad_history = dd.grad_train(
        seqs, 
        t_list, 
        learning_rate=0.1,
        max_iters=100,
        print_every=10,
        lr_decay=0.99  # Add learning rate decay
    )

    # Predict targets (now outputs n-dim vectors)
    print("\nPredictions:")
    for i, seq in enumerate(seqs[:3]):  # First 3 sequences
        t_pred = dd.predict_t(seq)
        print(f"Seq {i+1}: Target={[f'{x:.4f}' for x in t_list[i][:3]]}...")
        print(f"         Predicted={[f'{x:.4f}' for x in t_pred[:3]]}...")
    
    # Calculate prediction correlations
    preds = [dd.predict_t(seq) for seq in seqs]
    correlations = []
    for i in range(out_dim):
        actual = [t[i] for t in t_list]
        predicted = [p[i] for p in preds]
        corr = correlation(actual, predicted)
        correlations.append(corr)
        if i < 3:  # Print first 3 dimensions
            print(f"Dim {i} correlation: {corr:.4f}")
    print(f"Average correlation: {mean(correlations):.4f}")

    # Parameter count
    print("\nParameter count:")
    dd.count_parameters()
    
    # Self-supervised training example
    print("\nSelf-supervised training (next-vector prediction):")
    dd_auto = NumDualDescriptor(
        in_dim=in_dim,
        model_dim=model_dim,
        out_dim=out_dim,
        basis_dim=basis_dim,
        rank=3,
        mode='nonlinear',
        user_step=2
    )
    reg_history = dd_auto.auto_train(
        seqs, 
        learning_rate=0.05,
        max_iters=50,
        auto_mode='reg',
        print_every=5,
        lr_decay=0.97  # Add learning rate decay
    )
    print(f"Final next-vector prediction loss: {reg_history[-1]:.6f}")

    # Sequence generation example
    print("\nGenerating sequence:")
    init_vec = seqs[0][0]  # n-dim vector
    generated_seq = dd_auto.generate(
        length=5,
        init_vec=init_vec,
        tau=0.02
    )
    print(f"First 3 generated vectors (n-dim):")
    for i, vec in enumerate(generated_seq[:3]):
        print(f"Vec {i+1}: {[f'{x:.4f}' for x in vec[:3]]}...")

    # Feature extraction example
    print("\nFeature extraction:")
    sample_seq = seqs[0][:20]  # First 20 vectors
    features = dd_auto.dd_features(sample_seq)
    print(f"Extracted {len(features)} features")
    print(f"First 5 features: {[f'{x:.6f}' for x in features[:5]]}")
    print(f"Last 5 features: {[f'{x:.6f}' for x in features[-5:]]}")

    # Save/load example
    print("\nSaving and loading model:")
    dd_auto.save("nDD_model.pkl")
    loaded_model = NumDualDescriptor.load("nDD_model.pkl")
    print("Loaded model prediction on first sequence:")
    t_pred_loaded = loaded_model.predict_t(seqs[0])
    print(f"First 3 dims: {[f'{x:.4f}' for x in t_pred_loaded[:3]]}...")
