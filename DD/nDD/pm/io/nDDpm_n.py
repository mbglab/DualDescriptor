# Copyright (C) 2005-2025, Bin-Guang Ma (mbg@mail.hzau.edu.cn); SPDX-License-Identifier: MIT
# The Numeric Dual Descriptor Vector class (P Matrix form) implemented with pure Python
# Modified to support n-dimensional input and output
# This program is for the demonstration of methodology and not fully refined.
# Author: Bin-Guang Ma (assisted by DeepSeek); Date: 2025-6-20

import math
import random
import pickle

class NumDualDescriptorPM:
    """
    Numeric Dual Descriptor for sequences of n-dimensional vectors with m-dimensional internal representation.
    - input_dim: dimension of input vectors (n)
    - model_dim: dimension of internal representation (m)
    Model:
        For each vector x_k in sequence:
            z_k = M * x_k (m-dimensional internal representation)
            N(k)[i] = Σ_j P[i][j] * z_k[j] * cos(2πk / period[i][j])
            S(l) = cumulative sum of N(k) from k=1 to l
    """
    def __init__(self, input_dim=2, model_dim=2, rank=1, rank_op='avg', rank_mode='drop', mode='linear', user_step=None):
        """
        Initialize the Dual Descriptor for n-dimensional vector sequences.
        
        Args:
            input_dim (int): Dimension of input vectors (n)
            model_dim (int): Dimension of internal representation (m)
            rank (int): Window size for vector aggregation
            rank_mode (str): 'pad' or 'drop' for handling incomplete windows
            mode (str): 'linear' or 'nonlinear' processing mode
            user_step (int): Custom step size for nonlinear mode
        """        
        self.n = input_dim    # Input dimension (n)
        self.m = model_dim    # Internal dimension (m)
        self.rank = rank
        self.rank_op = rank_op
        self.rank_mode = rank_mode
        assert mode in ('linear','nonlinear')
        self.mode = mode
        self.step = user_step
        self.trained = False

        # Linear transformation matrix M ∈ R^{m×n}
        self.M = [[random.uniform(-0.5, 0.5) for _ in range(self.n)] 
                  for _ in range(self.m)]
        
        # Position-weight matrix P[i][j] ∈ R^{n×m} (simplified to 2D matrix)
        self.P = [[random.uniform(-0.1, 0.1) for _ in range(self.m)]
                  for _ in range(self.n)]

        # Precompute indexed periods[i][j]
        self.periods = [[i * self.m + j + 2  # Unique period for each (i,j) pair
                         for j in range(self.m)]
                        for i in range(self.n)]

    def _invert(self, A):
        """Matrix inversion helper function"""
        n = len(A)
        M = [row[:] + [1.0 if i==j else 0.0 for j in range(n)]
             for i, row in enumerate(A)]
        for i in range(n):
            piv = M[i][i]
            if abs(piv) < 1e-12: continue
            M[i] = [x/piv for x in M[i]]
            for r in range(n):
                if r==i: continue
                fac = M[r][i]
                M[r] = [M[r][c] - fac*M[i][c] for c in range(2*n)]
        return [row[n:] for row in M]

    def _mat_vec(self, M, v):
        """Matrix-vector multiplication helper"""
        return [sum(M[i][j]*v[j] for j in range(len(v))) for i in range(len(M))]

    def _mat_mult(self, A, B):
        """Matrix multiplication helper"""
        n = len(A)
        p = len(B[0])
        m = len(B)
        C = [[0.0] * p for _ in range(n)]
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    C[i][j] += A[i][k] * B[k][j]
        return C

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

    def compute_Nk(self, k, vec):
        """Compute N(k) vector (n-dimensional) for a single position and vector"""
        # Apply linear transformation: z = M * vec (m-dimensional)
        z = self._mat_vec(self.M, vec)
        Nk = [0.0] * self.n  # n-dimensional output
        
        for i in range(self.n):      # Output dimension
            for j in range(self.m):  # Internal dimension
                period = self.periods[i][j]
                phi = math.cos(2 * math.pi * k / period)
                Nk[i] += self.P[i][j] * z[j] * phi
        return Nk

    def describe(self, seq):
        """Compute N(k) vectors (n-dimensional) for each window in sequence"""
        windows = self.extract_windows(seq)
        if not windows:
            return []
            
        N = []
        for pos, vec in enumerate(windows):
            N.append(self.compute_Nk(pos, vec))
        return N

    def S(self, seq):
        """Compute cumulative sum vectors S(l) = Σ N(k) (n-dimensional)"""
        Nk_list = self.describe(seq)
        if not Nk_list:
            return []
            
        S = [0.0] * self.n  # n-dimensional cumulative sum
        S_list = []
        for Nk in Nk_list:
            S = [S[i] + Nk[i] for i in range(self.n)]
            S_list.append(S.copy())
        return S_list 

    def D(self, seqs, t_list):
        """Compute mean squared deviation (n-dimensional targets)"""
        tot = 0.0
        cnt = 0
        for seq, t in zip(seqs, t_list):
            for Nk in self.describe(seq):
                for i in range(self.n):
                    d = Nk[i] - t[i]
                    tot += d * d
                cnt += 1
        return tot / cnt if cnt else 0.0

    def update_P(self, seqs, t_list):
        """Closed-form update of P matrix for n-dimensional vectors"""
        # Iterate over each output dimension independently
        for i in range(self.n):
            dim = self.m  # Size of subsystem for this i
            U_i = [[0.0] * dim for _ in range(dim)]  # (m) x (m) matrix
            V_i = [0.0] * dim  # (m) vector
            
            # Accumulate data from all sequences and positions
            for seq, t in zip(seqs, t_list):
                windows = self.extract_windows(seq)
                for pos, vec in enumerate(windows):
                    # Apply linear transformation: z = M * vec
                    z = self._mat_vec(self.M, vec)
                    # Build indices and basis values for current (i,pos)
                    for j in range(self.m):
                        period = self.periods[i][j]
                        phi = math.cos(2 * math.pi * pos / period)
                        a = z[j] * phi
                        
                        # Right-hand side vector component
                        V_i[j] += t[i] * a
                        
                        # Left-hand side matrix components
                        for j2 in range(self.m):
                            period2 = self.periods[i][j2]
                            phi2 = math.cos(2 * math.pi * pos / period2)
                            b = z[j2] * phi2
                            U_i[j][j2] += a * b
            
            # Solve subsystem for current i
            try:
                U_inv = self._invert(U_i)
                sol = [sum(U_inv[r][c] * V_i[c] for c in range(dim)) for r in range(dim)]
            except Exception as e:
                print(f"Warning: inversion failed for i={i}, using fallback. Error: {str(e)}")
                sol = [0.0] * dim  # Fallback to zero solution
            
            # Update P matrix for current i
            for j in range(self.m):
                self.P[i][j] = sol[j]

    def update_M(self, seqs, t_list):
        """Closed-form update of transformation matrix M (m×n)"""
        # Initialize gradient accumulation
        M_grad = [[0.0] * self.n for _ in range(self.m)]
        total_positions = 0
        
        # Accumulate data from all sequences and positions
        for seq, t in zip(seqs, t_list):
            windows = self.extract_windows(seq)
            total_positions += len(windows)
            for pos, vec in enumerate(windows):
                # Precompute basis-parameter products
                psi = [0.0] * self.m  # ψ_j = P[i][j] * φ_{ij}(pos) summed over i
                for j in range(self.m):
                    s = 0.0
                    for i in range(self.n):
                        period = self.periods[i][j]
                        phi = math.cos(2 * math.pi * pos / period)
                        s += self.P[i][j] * phi
                    psi[j] = s
                
                # Compute current transformed vector: z = M * vec
                z = self._mat_vec(self.M, vec)
                
                # Compute error terms (n-dimensional)
                error = [0.0] * self.n
                for i in range(self.n):
                    # Compute predicted Nk[i]
                    pred = 0.0
                    for j in range(self.m):
                        period = self.periods[i][j]
                        phi = math.cos(2 * math.pi * pos / period)
                        pred += self.P[i][j] * phi * z[j]
                    error[i] = pred - t[i]
                
                # Update gradient for M
                for row in range(self.m):    # M row index
                    for col in range(self.n):  # M column index
                        grad = 0.0
                        for i in range(self.n):
                            period = self.periods[i][row]
                            phi = math.cos(2 * math.pi * pos / period)
                            grad += error[i] * self.P[i][row] * phi * vec[col]
                        M_grad[row][col] += grad
        
        # Normalize gradient and update M
        lr = 0.01  # Learning rate
        if total_positions > 0:
            for i in range(self.m):
                for j in range(self.n):
                    self.M[i][j] -= lr * M_grad[i][j] / total_positions

    def train(self, seqs, t_list, max_iters=10, tol=1e-8, print_every=1):
        """Alternate closed-form updates for P and M"""        
        D_prev = float('inf')
        history = []
        for it in range(max_iters):
            self.update_P(seqs, t_list)
            self.update_M(seqs, t_list)
            D = self.D(seqs, t_list)
            history.append(D)
            if it % print_every == 0 or it == max_iters - 1:
                print(f"Iter {it:2d}: D = {D:.6e}")
            if D >= D_prev - tol:
                print("Converged.")
                break
            D_prev = D
            
        # Calculate statistics
        total_window_count = 0
        total_t = [0.0] * self.n        
        for seq, t_vec in zip(seqs, t_list):
            windows = self.extract_windows(seq)
            total_window_count += len(windows)
            for d in range(self.n):
                total_t[d] += t_vec[d]                
        self.mean_window_count = total_window_count / len(seqs)
        self.mean_t = [total_t[d] / len(seqs) for d in range(self.n)]
        self.trained = True        
        return history

    def grad_train(self, seqs, t_list, max_iters=1000, tol=1e-8, learning_rate=0.01, 
                  continued=False, decay_rate=1.0, print_every=10):
        """Gradient descent training for n-dimensional vectors"""
        if not continued:
            # Initialize with small random values
            self.M = [[random.uniform(-0.5, 0.5) for _ in range(self.n)] 
                      for _ in range(self.m)]
            self.P = [[random.uniform(-0.1, 0.1) for _ in range(self.m)]
                      for _ in range(self.n)]
            
        history = []
        D_prev = float('inf')
        total_positions = sum(len(self.extract_windows(seq)) for seq in seqs)
        
        for it in range(max_iters):
            # Initialize gradients
            grad_P = [[0.0 for _ in range(self.m)] for _ in range(self.n)]
            grad_M = [[0.0 for _ in range(self.n)] for _ in range(self.m)]
            
            # Accumulate gradients over all sequences and positions
            for seq, t in zip(seqs, t_list):
                windows = self.extract_windows(seq)
                for pos, vec in enumerate(windows):
                    # Apply linear transformation: z = M * vec
                    z = self._mat_vec(self.M, vec)
                    Nk = [0.0] * self.n
                    
                    # Precompute basis functions and Nk
                    phi_vals = {}
                    for i in range(self.n):
                        for j in range(self.m):
                            period = self.periods[i][j]
                            phi = math.cos(2 * math.pi * pos / period)
                            phi_vals[(i, j)] = phi
                            Nk[i] += self.P[i][j] * z[j] * phi
                    
                    # Compute gradients
                    for i in range(self.n):
                        residual = 2 * (Nk[i] - t[i]) / total_positions
                        for j in range(self.m):
                            phi = phi_vals[(i, j)]
                            # Gradient for P[i][j]
                            grad_P[i][j] += residual * z[j] * phi
                            
                            # Gradient for M (through chain rule)
                            grad_M_term = residual * self.P[i][j] * phi
                            for d in range(self.n):
                                grad_M[j][d] += grad_M_term * vec[d]
            
            # Update P matrix
            for i in range(self.n):
                for j in range(self.m):
                    self.P[i][j] -= learning_rate * grad_P[i][j]
            
            # Update M matrix
            for i in range(self.m):
                for j in range(self.n):
                    self.M[i][j] -= learning_rate * grad_M[i][j]
            
            # Calculate current loss
            current_D = self.D(seqs, t_list)
            history.append(current_D)
            if it % print_every == 0 or it == max_iters - 1:
                print(f"GD Iter {it:4d}: D = {current_D:.6e}, LR = {learning_rate:.4f}")
            
            # Check convergence
            if abs(D_prev - current_D) < tol:
                print(f"Converged after {it+1} iterations.")
                break
            D_prev = current_D
            
            # Apply learning rate decay
            learning_rate *= decay_rate
        
        # Calculate statistics
        total_window_count = 0
        total_t = [0.0] * self.n
        for seq, t in zip(seqs, t_list):
            windows = self.extract_windows(seq)
            total_window_count += len(windows)
            for d in range(self.n):
                total_t[d] += t[d]
        self.mean_window_count = total_window_count / len(seqs)
        self.mean_t = [total_t[d] / len(seqs) for d in range(self.n)]
        self.trained = True
        return history

    def auto_train(self, seqs, max_iters=100, tol=1e-6, learning_rate=0.01, 
                  continued=False, auto_mode='reg', decay_rate=1.0, print_every=10):
        """
        Self-training for n-dimensional vector sequences:
          - 'gap': Predicts current window's vector
          - 'reg': Predicts next window's vector
        """
        if auto_mode not in ('gap', 'reg'):
            raise ValueError("auto_mode must be either 'gap' or 'reg'")

        if not continued:
            # Initialize with small random values
            self.M = [[random.uniform(-0.5, 0.5) for _ in range(self.n)] 
                      for _ in range(self.m)]
            self.P = [[random.uniform(-0.1, 0.1) for _ in range(self.m)]
                      for _ in range(self.n)]
        
        # Calculate total training samples
        total_samples = 0
        for seq in seqs:
            windows = self.extract_windows(seq)
            if auto_mode == 'gap':
                total_samples += len(windows)
            else:  # 'reg' mode
                total_samples += max(0, len(windows) - 1)
            
        if total_samples == 0:
            raise ValueError("No training samples found")
            
        history = []
        prev_loss = float('inf')
        
        for it in range(max_iters):
            # Initialize gradients
            grad_P = [[0.0] * self.m for _ in range(self.n)]
            grad_M = [[0.0] * self.n for _ in range(self.m)]
            total_loss = 0.0
            
            # Process all sequences
            for seq in seqs:
                windows = self.extract_windows(seq)
                if not windows:
                    continue
                    
                # Process windows based on mode
                for k in range(len(windows)):
                    if auto_mode == 'reg' and k == len(windows) - 1:
                        continue
                        
                    current_vec = windows[k]
                    
                    # Compute N(k) for current window at position k
                    z = self._mat_vec(self.M, current_vec)
                    Nk = [0.0] * self.n
                    phi_vals = {}
                    for i in range(self.n):
                        for j in range(self.m):
                            period = self.periods[i][j]
                            phi = math.cos(2 * math.pi * k / period)
                            phi_vals[(i, j)] = phi
                            Nk[i] += self.P[i][j] * z[j] * phi
                    
                    # Determine target based on mode
                    if auto_mode == 'gap':
                        target = current_vec  # n-dimensional
                    else:  # 'reg' mode
                        target = windows[k + 1]  # n-dimensional
                    
                    # Compute loss and gradients
                    for i in range(self.n):
                        error = Nk[i] - target[i]
                        total_loss += error * error
                        
                        # Compute gradients (2 * error from derivative of squared loss)
                        grad_coeff = 2 * error / total_samples
                        
                        # Gradient for P[i][j]
                        for j in range(self.m):
                            phi = phi_vals[(i, j)]
                            grad_P[i][j] += grad_coeff * z[j] * phi
                                
                        # Gradient for M
                        for j in range(self.m):
                            phi = phi_vals[(i, j)]
                            grad_M_term = grad_coeff * self.P[i][j] * phi
                            for d in range(self.n):
                                grad_M[j][d] += grad_M_term * current_vec[d]
            
            # Update parameters
            for i in range(self.n):
                for j in range(self.m):
                    self.P[i][j] -= learning_rate * grad_P[i][j]
            
            for i in range(self.m):
                for j in range(self.n):
                    self.M[i][j] -= learning_rate * grad_M[i][j]
            
            # Record and print progress
            avg_loss = total_loss / total_samples
            history.append(avg_loss)
            if it % print_every == 0 or it == max_iters - 1:
                mode_display = "Gap" if auto_mode == 'gap' else "Reg"
                print(f"AutoTrain({mode_display}) Iter {it:3d}: loss = {avg_loss:.6f}, LR = {learning_rate:.4f}")
            
            # Check convergence
            if abs(prev_loss - avg_loss) < tol:
                print(f"Converged after {it+1} iterations")
                break
            prev_loss = avg_loss
            
            # Apply learning rate decay
            learning_rate *= decay_rate
        
        # Compute and store statistics
        total_t = [0.0] * self.n
        total_window_count = 0
        for seq in seqs:
            windows = self.extract_windows(seq)
            total_window_count += len(windows)
            for window in windows:
                for d in range(self.n):
                    total_t[d] += window[d]
        
        self.mean_window_count = total_window_count / len(seqs)
        self.mean_t = [t_val / total_window_count for t_val in total_t]
        self.trained = True
        
        return history

    def predict_t(self, seq):
        """Predict target vector (n-dimensional) for a sequence"""
        N_list = self.describe(seq)
        if not N_list:
            return [0.0] * self.n
            
        # Average all N(k) vectors component-wise
        sum_t = [0.0] * self.n
        for Nk in N_list:
            for d in range(self.n):
                sum_t[d] += Nk[d]
        N = len(N_list)
        return [ti / N for ti in sum_t]   

    def generate(self, num_windows, tau=0.0):
        """Generate sequence of n-dimensional vector windows"""
        assert self.trained, "Model must be trained first"
        if tau < 0:
            raise ValueError("Temperature must be non-negative")            
        generated_windows = []        
        for k in range(num_windows):
            # Generate candidate vectors (n-dimensional)
            candidates = []
            for _ in range(100):
                candidate = [random.gauss(self.mean_t[d], 0.1) for d in range(self.n)]
                candidates.append(candidate)
            
            # Score candidates
            scores = []
            for candidate in candidates:
                Nk = self.compute_Nk(k, candidate)
                error = 0.0
                for d in range(self.n):
                    diff = Nk[d] - self.mean_t[d]
                    error += diff * diff
                scores.append(-error)  # Convert to score (higher = better)
                
            if tau == 0:  # Deterministic selection
                best_idx = scores.index(max(scores))
                generated_windows.append(candidates[best_idx])
            else:  # Stochastic selection
                exp_scores = [math.exp(score / tau) for score in scores]
                sum_exp = sum(exp_scores)
                probs = [es / sum_exp for es in exp_scores]
                chosen_idx = random.choices(range(len(candidates)), weights=probs, k=1)[0]
                generated_windows.append(candidates[chosen_idx])                
        return generated_windows

    def dd_features(self, seq):
        """Extract feature vector (flattened P and M matrices)"""
        features = []
        # Flatten P matrix (n×m)
        for row in self.P:
            features.extend(row)
        # Flatten M matrix (m×n)
        for row in self.M:
            features.extend(row)
        return features

    def show(self):
        """Display model status"""
        print("NumDualDescriptorPM Status:")
        print(f"  Input dimension n = {self.n}")
        print(f"  Internal dimension m = {self.m}")
        print("  M matrix (m×n) sample:")
        for i in range(min(3, self.m)):
            print(f"    Row {i}: {[f'{x:.4f}' for x in self.M[i][:min(3, self.n)]]}...")
        if self.m > 3:
            print("    ...")
        print("  P matrix (n×m) sample:")
        for i in range(min(3, self.n)):
            print(f"    Row {i}: {[f'{x:.4f}' for x in self.P[i][:min(3, self.m)]]}...")

    def count_parameters(self):
        """Count learnable parameters (P matrix and M matrix)"""
        total_params = self.n * self.m + self.m * self.n
        print(f"Parameter Count:")
        print(f"  P matrix: {self.n}×{self.m} = {self.n*self.m}")
        print(f"  M matrix: {self.m}×{self.n} = {self.m*self.n}")
        print(f"Total parameters: {total_params}")
        return total_params

    def save(self, filename):
        """Save model state to file"""
        with open(filename, 'wb') as f:
            pickle.dump(self.__dict__, f)
        print(f"Model saved to {filename}")

    @classmethod
    def load(cls, filename):
        """Load model state from file"""
        with open(filename, 'rb') as f:
            state = pickle.load(f)
        
        # Create new instance without calling __init__
        obj = cls.__new__(cls)
        # Restore saved state
        obj.__dict__.update(state)
        print(f"Model loaded from {filename}")
        return obj


# === Example Usage ===
if __name__=="__main__":

    from statistics import correlation, mean
    
    random.seed(2)
    
    # Configuration: n=5 input dim, m=3 internal dim
    n_dim = 5   # Input vector dimension
    m_dim = 3   # Internal model dimension
    num_seqs = 10
    min_len, max_len = 50, 100

    # Generate synthetic training data (n-dimensional)
    seqs = []
    t_list = []
    for _ in range(num_seqs):
        L = random.randint(min_len, max_len)
        seq = [[random.uniform(-1,1) for _ in range(n_dim)] for __ in range(L)]
        seqs.append(seq)
        t_list.append([random.uniform(-1,1) for _ in range(n_dim)])  # n-dim targets

    # Create model with n=5, m=3
    dd = NumDualDescriptorPM(input_dim=n_dim, model_dim=m_dim, 
                            rank=3, 
                            mode='nonlinear', user_step=2)
    
    # Train the model
    print("Training model with closed-form updates...")
    history = dd.train(seqs, t_list, max_iters=15, print_every=1)
    
    # Show model status
    dd.show()
    dd.count_parameters()
    
    # Predict targets (n-dimensional)
    print("\nPredictions for first 3 sequences:")
    for i, seq in enumerate(seqs[:3]):
        t_pred = dd.predict_t(seq)
        # Show first 3 dimensions for readability
        print(f"Seq {i+1}: Target={[f'{x:.4f}' for x in t_list[i][:3]]}...")
        print(f"         Predicted={[f'{x:.4f}' for x in t_pred[:3]]}...")
    
    # Calculate prediction correlations
    preds = [dd.predict_t(seq) for seq in seqs]
    correlations = []
    for i in range(n_dim):
        actual = [t[i] for t in t_list]
        predicted = [p[i] for p in preds]
        corr = correlation(actual, predicted) if len(actual) > 1 else 0.0
        correlations.append(corr)
        print(f"Dim {i} correlation: {corr:.4f}")
    print(f"Average correlation: {mean(correlations):.4f}")
    
    # Gradient Descent Training
    print("\nTraining with Gradient Descent...")
    dd_grad = NumDualDescriptorPM(input_dim=n_dim, model_dim=m_dim,
                                 rank=3)
    grad_history = dd_grad.grad_train(
        seqs, 
        t_list, 
        learning_rate=0.1,
        max_iters=200,
        decay_rate=0.98,
        print_every=20
    )
    
    # Calculate prediction correlations
    preds = [dd_grad.predict_t(seq) for seq in seqs]
    correlations = []
    for i in range(n_dim):
        actual = [t[i] for t in t_list]
        predicted = [p[i] for p in preds]
        corr = correlation(actual, predicted) if len(actual) > 1 else 0.0
        correlations.append(corr)
        print(f"Dim {i} correlation: {corr:.4f}")
    print(f"Average correlation: {mean(correlations):.4f}")
    
    # Auto-Training Example (next-vector prediction)
    print("\nAuto-Training in 'reg' mode (next-vector prediction)...")
    dd_auto = NumDualDescriptorPM(input_dim=n_dim, model_dim=m_dim,
                                 rank=3)
    auto_history = dd_auto.auto_train(
        seqs, 
        auto_mode='reg', 
        learning_rate=0.05,
        max_iters=30
    )
    
    # Generate sequence
    print("\nGenerating new sequence (n-dimensional)...")
    gen_seq = dd_auto.generate(num_windows=5, tau=0.1)
    print("Generated windows (first 3 dimensions):")
    for i, vec in enumerate(gen_seq):
        print(f"Window {i+1}: {[f'{x:.4f}' for x in vec[:3]]}...")
    
    # Extract features
    print("\nExtracting features for first sequence...")
    features = dd_auto.dd_features(seqs[0])
    print(f"Feature vector length: {len(features)}")
    print(f"First 5 features: {[f'{x:.6f}' for x in features[:5]]}")
    
    # Save and load model
    print("\nSaving and loading model...")
    dd_auto.save("n_vector_model.pkl")
    dd_loaded = NumDualDescriptorPM.load("n_vector_model.pkl")
    
    # Verify loaded model
    t_pred_loaded = dd_loaded.predict_t(seqs[0])
    print(f"Prediction from loaded model: {[f'{x:.4f}' for x in t_pred_loaded[:3]]}...")
