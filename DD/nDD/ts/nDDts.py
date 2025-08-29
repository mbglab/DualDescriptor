# Copyright (C) 2005-2025, Bin-Guang Ma (mbg@mail.hzau.edu.cn); SPDX-License-Identifier: MIT
# The Numeric Dual Descriptor Vector class (Tensor form) implemented with pure Python
# This program is for the demonstration of methodology and not fully refined.
# Author: Bin-Guang Ma (assisted by DeepSeek); Date: 2025-6-20

import math
import random
import pickle

class NumDualDescriptorPM:
    """
    Numeric Dual Descriptor for vector sequences with:
      - tensor P ∈ R^{m×m×o} of basis coefficients
      - Matrix M ∈ R^{m×m} for linear transformation
      - rank: window size for vector aggregation
      - indexed periods: period[i,j,g] = i*(m*o) + j*o + g + 2
      - basis function phi_{i,j,g}(k) = cos(2π * k / period[i,j,g])
      - supports 'linear' or 'nonlinear' (step-by-rank) window extraction    
    """
    def __init__(self, vec_dim=2, num_basis=5, rank=1, rank_op='avg', rank_mode='drop', mode='linear', user_step=None):
        self.m = vec_dim    # vector dimension
        self.o = num_basis  # number of basis terms  
        self.rank = rank    # window size for vector aggregation
        self.rank_op = rank_op
        self.rank_mode = rank_mode # 'pad' or 'drop'               
        assert mode in ('linear','nonlinear')
        self.mode = mode
        self.step = user_step
        self.trained = False

        # Linear transformation matrix M ∈ R^{m×m}
        self.M = [[random.uniform(-0.5, 0.5) for _ in range(self.m)] 
                  for _ in range(self.m)]
        
        # position-weight tensor P[i][j][g]
        self.P = [[[random.uniform(-0.1, 0.1) for _ in range(self.o)]
                   for _ in range(self.m)]
                  for _ in range(self.m)]

        # precompute indexed periods[i][j][g]
        self.periods = [[[ i*(self.m*self.o) + j*self.o + g + 2
                            for g in range(self.o)]
                           for j in range(self.m)]
                          for i in range(self.m)]

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
        """Compute N(k) vector for a single position and vector"""
        # Apply linear transformation: x = M * vec
        x = self._mat_vec(self.M, vec)
        Nk = [0.0] * self.m
        for i in range(self.m):
            for j in range(self.m):
                for g in range(self.o):
                    period = self.periods[i][j][g]
                    phi = math.cos(2 * math.pi * k / period)
                    Nk[i] += self.P[i][j][g] * x[j] * phi
        return Nk

    def describe(self, seq):
        """Compute N(k) vectors for each window in sequence"""
        windows = self.extract_windows(seq)
        N = []
        for k, vec in enumerate(windows):
            N.append(self.compute_Nk(k, vec))
        return N

    def S(self, seq):
        """
        Compute list of S(l)=sum(N(k)) (k=1,...,l; l=1,...,L) for a given sequence.        
        """
        Nk_list = self.describe(seq)        
        S = [0.0] * self.m
        S_list = []
        for l in range(len(Nk_list)):
            for i in range(self.m):
                S[i] += Nk_list[l][i]
            S_list.append(list(S))
        return S_list 

    def D(self, seqs, t_list):
        """
        Compute mean squared deviation D across sequences:
        D = average over all positions of (N(k)-t_seq)^2
        """
        tot = 0.0
        cnt = 0
        for seq, t in zip(seqs, t_list):
            for Nk in self.describe(seq):
                for i in range(self.m):
                    d = Nk[i] - t[i]
                    tot += d * d
                cnt += 1
        return tot / cnt if cnt else 0.0

    def update_P(self, seqs, t_list):
        """
        Closed-form update of P tensor for vector sequences.
        """
        # Iterate over each i-dimension independently
        for i in range(self.m):
            dim = self.m * self.o  # Size of subsystem for this i
            U_i = [[0.0] * dim for _ in range(dim)]  # (m*o) x (m*o) matrix
            V_i = [0.0] * dim  # (m*o) vector
            
            # Accumulate data from all sequences and positions
            for seq, t in zip(seqs, t_list):
                windows = self.extract_windows(seq)
                for k, vec in enumerate(windows):
                    # Apply linear transformation: x = M * vec
                    x = self._mat_vec(self.M, vec)
                    # Build indices and basis values for current (i,k)
                    for j in range(self.m):
                        for g in range(self.o):
                            idx1 = j * self.o + g  # Linear index in subsystem
                            period = self.periods[i][j][g]
                            phi = math.cos(2 * math.pi * k / period)
                            a = x[j] * phi
                            
                            # Right-hand side vector component
                            V_i[idx1] += t[i] * a
                            
                            # Left-hand side matrix components
                            for j2 in range(self.m):
                                for h in range(self.o):
                                    idx2 = j2 * self.o + h
                                    period2 = self.periods[i][j2][h]  # Same i!
                                    phi2 = math.cos(2 * math.pi * k / period2)
                                    b = x[j2] * phi2
                                    U_i[idx1][idx2] += a * b
            
            # Solve subsystem for current i
            try:
                U_inv = self._invert(U_i)
                sol = [sum(U_inv[r][c] * V_i[c] for c in range(dim)) for r in range(dim)]
            except Exception as e:
                print(f"Warning: inversion failed for i={i}, using identity. Error: {str(e)}")
                sol = V_i[:]  # Fallback to identity solution
            
            # Update P tensor for current i
            for j in range(self.m):
                for g in range(self.o):
                    idx = j * self.o + g
                    self.P[i][j][g] = sol[idx]

    def update_M(self, seqs, t_list):
        """
        Closed-form update of transformation matrix M.
        """
        # Precompute basis products to avoid redundant calculations
        basis_cache = {}
        for i in range(self.m):
            for j in range(self.m):
                for g in range(self.o):
                    period = self.periods[i][j][g]
                    basis_cache[(i, j, g)] = period
        
        # Initialize gradient accumulation structures
        M_grad = [[0.0] * self.m for _ in range(self.m)]
        
        # Accumulate data from all sequences and positions
        total_positions = 0
        for seq, t in zip(seqs, t_list):
            windows = self.extract_windows(seq)
            total_positions += len(windows)
            for k, vec in enumerate(windows):
                # Precompute basis-parameter products
                psi = [[0.0] * self.m for _ in range(self.m)]  # ψ_{j,d} = Σ_g P[i][j][g] * φ_{ijg}(k)
                for i in range(self.m):
                    for j in range(self.m):
                        s = 0.0
                        for g in range(self.o):
                            period = basis_cache[(i, j, g)]
                            phi = math.cos(2 * math.pi * k / period)
                            s += self.P[i][j][g] * phi
                        psi[i][j] = s
                
                # Compute current transformed vector: x = M * vec
                x = self._mat_vec(self.M, vec)
                
                # Compute error terms
                error = [0.0] * self.m
                for i in range(self.m):
                    # Compute predicted Nk[i]
                    pred = 0.0
                    for j in range(self.m):
                        pred += psi[i][j] * x[j]
                    error[i] = pred - t[i]
                
                # Update gradient for M
                for d1 in range(self.m):  # Row in M
                    for d2 in range(self.m):  # Column in M
                        grad = 0.0
                        for i in range(self.m):
                            grad += error[i] * psi[i][d1] * vec[d2]
                        M_grad[d1][d2] += grad
        
        # Normalize gradient by number of positions
        if total_positions > 0:
            for i in range(self.m):
                for j in range(self.m):
                    M_grad[i][j] /= total_positions
        
        # Update M using gradient (simple gradient descent)
        lr = 0.01  # Learning rate
        for i in range(self.m):
            for j in range(self.m):
                self.M[i][j] -= lr * M_grad[i][j]

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
            
        # Calculate statistics for reconstruction/generation
        total_window_count = 0
        total_t = [0.0] * self.m        
        for seq, t_vec in zip(seqs, t_list):
            windows = self.extract_windows(seq)
            total_window_count += len(windows)
            for d in range(self.m):
                total_t[d] += t_vec[d]                
        self.mean_window_count = total_window_count / len(seqs)
        self.mean_t = [total_t[d] / len(seqs) for d in range(self.m)]
        self.trained = True        
        return history

    def grad_train(self, seqs, t_list, max_iters=1000, tol=1e-8, learning_rate=0.01, continued=False, decay_rate=1.0, print_every=10):
        """
        Train using gradient descent for vector sequences.
        """
        if not continued:
            # Initialize with small random values
            self.M = [[random.uniform(-0.5, 0.5) for _ in range(self.m)] 
                      for _ in range(self.m)]
            self.P = [[[random.uniform(-0.1, 0.1) for _ in range(self.o)]
                   for _ in range(self.m)]
                  for _ in range(self.m)]
            
        history = []
        D_prev = float('inf')
        total_positions = sum(len(self.extract_windows(seq)) for seq in seqs)
        
        for it in range(max_iters):
            # Initialize gradients
            grad_P = [[[0.0 for _ in range(self.o)] for _ in range(self.m)] for _ in range(self.m)]
            grad_M = [[0.0 for _ in range(self.m)] for _ in range(self.m)]
            
            # Accumulate gradients over all sequences and positions
            for seq, t_vec in zip(seqs, t_list):
                windows = self.extract_windows(seq)
                for pos, vec in enumerate(windows):
                    # Apply linear transformation: x = M * vec
                    x = self._mat_vec(self.M, vec)
                    Nk = [0.0] * self.m
                    
                    # Precompute basis functions and Nk
                    phi_vals = {}
                    for i in range(self.m):
                        for j in range(self.m):
                            for g in range(self.o):
                                period = self.periods[i][j][g]
                                phi = math.cos(2 * math.pi * pos / period)
                                phi_vals[(i, j, g)] = phi
                                Nk[i] += self.P[i][j][g] * x[j] * phi
                    
                    # Compute gradients
                    for i in range(self.m):
                        residual = 2 * (Nk[i] - t_vec[i]) / total_positions
                        for j in range(self.m):
                            for g in range(self.o):
                                phi = phi_vals[(i, j, g)]
                                # Gradient for P[i][j][g]
                                grad_P[i][j][g] += residual * x[j] * phi
                                
                                # Gradient for M (through chain rule)
                                grad_M_term = residual * self.P[i][j][g] * phi
                                for d in range(self.m):
                                    grad_M[j][d] += grad_M_term * vec[d]
            
            # Update P tensor
            for i in range(self.m):
                for j in range(self.m):
                    for g in range(self.o):
                        self.P[i][j][g] -= learning_rate * grad_P[i][j][g]
            
            # Update M matrix
            for i in range(self.m):
                for j in range(self.m):
                    self.M[i][j] -= learning_rate * grad_M[i][j]
            
            # Calculate current loss and print progress
            current_D = self.D(seqs, t_list)
            history.append(current_D)
            if it % print_every == 0 or it == max_iters - 1:
                print(f"GD Iter {it:2d}: D = {current_D:.6e}, LR = {learning_rate}")
            
            # Check convergence
            if D_prev - current_D < tol:
                print(f"Converged after {it+1} iterations.")
                break
            D_prev = current_D
            
            # Apply learning rate decay
            learning_rate *= decay_rate
        
        # Calculate statistics for reconstruction/generation
        total_window_count = 0
        total_t = [0.0] * self.m
        for seq, t_vec in zip(seqs, t_list):
            windows = self.extract_windows(seq)
            total_window_count += len(windows)
            for d in range(self.m):
                total_t[d] += t_vec[d]
        self.mean_window_count = total_window_count / len(seqs)
        self.mean_t = [total_t[d] / len(seqs) for d in range(self.m)]
        self.trained = True
        return history

    def auto_train(self, seqs, max_iters=100, tol=1e-6, learning_rate=0.01, continued=False, auto_mode='reg', decay_rate=1.0, print_every=10):
        """
        Self-training for vector sequences with two modes:
          - 'gap': Predicts current window's vector (self-consistency)
          - 'reg': Predicts next window's vector (auto-regressive)
        """
        if auto_mode not in ('gap', 'reg'):
            raise ValueError("auto_mode must be either 'gap' or 'reg'")

        if not continued:
            # Initialize with small random values
            self.M = [[random.uniform(-0.5, 0.5) for _ in range(self.m)] 
                      for _ in range(self.m)]
            self.P = [[[random.uniform(-0.1, 0.1) for _ in range(self.o)]
                   for _ in range(self.m)]
                  for _ in range(self.m)]
        
        # Calculate total training samples
        total_samples = 0
        for seq in seqs:
            windows = self.extract_windows(seq)
            if auto_mode == 'gap':
                total_samples += len(windows)  # All windows are samples
            else:  # 'reg' mode
                total_samples += max(0, len(windows) - 1)  # Windows except last
            
        if total_samples == 0:
            raise ValueError("No training samples found")
            
        history = []  # Store loss per iteration
        prev_loss = float('inf')
        
        for it in range(max_iters):
            # Initialize gradients
            grad_P = [[[0.0] * self.o for _ in range(self.m)] for _ in range(self.m)]
            grad_M = [[0.0] * self.m for _ in range(self.m)]
            total_loss = 0.0
            
            # Process all sequences
            for seq in seqs:
                windows = self.extract_windows(seq)
                if not windows:
                    continue
                    
                # Process windows based on mode
                for k in range(len(windows)):
                    # Skip last window in 'reg' mode
                    if auto_mode == 'reg' and k == len(windows) - 1:
                        continue
                        
                    current_vec = windows[k]
                    
                    # Compute N(k) for current window at position k
                    x = self._mat_vec(self.M, current_vec)
                    Nk = [0.0] * self.m
                    phi_vals = {}
                    for i in range(self.m):
                        for j in range(self.m):
                            for g in range(self.o):
                                period = self.periods[i][j][g]
                                phi = math.cos(2 * math.pi * k / period)
                                phi_vals[(i, j, g)] = phi
                                Nk[i] += self.P[i][j][g] * x[j] * phi
                    
                    # Determine target based on mode
                    if auto_mode == 'gap':
                        target = current_vec  # Current window vector
                    else:  # 'reg' mode
                        target = windows[k + 1]  # Next window vector
                    
                    # Compute loss and gradients
                    for i in range(self.m):
                        error = Nk[i] - target[i]
                        total_loss += error * error
                        
                        # Compute gradients (2 * error from derivative of squared loss)
                        grad_coeff = 2 * error / total_samples
                        
                        # Gradient for P[i][j][g]
                        for j in range(self.m):
                            for g in range(self.o):
                                phi = phi_vals[(i, j, g)]
                                grad_P[i][j][g] += grad_coeff * x[j] * phi
                                
                        # Gradient for M
                        for j in range(self.m):
                            for g in range(self.o):
                                phi = phi_vals[(i, j, g)]
                                grad_M_term = grad_coeff * self.P[i][j][g] * phi
                                for d in range(self.m):
                                    grad_M[j][d] += grad_M_term * current_vec[d]
            
            # Update parameters
            for i in range(self.m):
                for j in range(self.m):
                    for g in range(self.o):
                        self.P[i][j][g] -= learning_rate * grad_P[i][j][g]
            
            for i in range(self.m):
                for j in range(self.m):
                    self.M[i][j] -= learning_rate * grad_M[i][j]
            
            # Record and print progress
            avg_loss = total_loss / total_samples
            history.append(avg_loss)
            if it % print_every == 0 or it == max_iters - 1:
                mode_display = "Gap" if auto_mode == 'gap' else "Reg"
                print(f"AutoTrain({mode_display}) Iter {it:3d}: loss = {avg_loss:.6f}, LR = {learning_rate}")
            
            # Check convergence
            if prev_loss - avg_loss < tol:
                print(f"Converged after {it+1} iterations")
                break
            prev_loss = avg_loss
            
            # Apply learning rate decay
            learning_rate *= decay_rate
        
        # Compute and store statistics for reconstruction/generation
        total_t = [0.0] * self.m
        total_window_count = 0
        for seq in seqs:
            windows = self.extract_windows(seq)
            total_window_count += len(windows)
            for window in windows:
                for d in range(self.m):
                    total_t[d] += window[d]
        
        self.mean_window_count = total_window_count / len(seqs)
        self.mean_t = [t_val / total_window_count for t_val in total_t]
        self.trained = True
        
        return history

    def predict_t(self, seq):
        """
        Predict target vector for a sequence
        Returns the average of all N(k) vectors in the sequence
        """
        N_list = self.describe(seq)
        if not N_list:
            return [0.0] * self.m            
        # Average all N(k) vectors component-wise
        sum_t = [0.0] * self.m
        for Nk in N_list:
            for d in range(self.m):
                sum_t[d] += Nk[d]
        N = len(N_list)
        return [ti / N for ti in sum_t]
    
    def reconstruct(self):
        """Reconstruct representative window sequence"""
        assert self.trained, "Model must be trained first"
        # Round average window count to nearest integer
        n_windows = round(self.mean_window_count)
        seq_windows = []        
        for k in range(n_windows):
            best_vec = [0.0] * self.m
            min_error = float('inf')            
            # Generate candidate vectors around mean_t
            for _ in range(100):  # Generate 100 candidate vectors
                candidate = [random.gauss(self.mean_t[d], 0.1) for d in range(self.m)]
                Nk = self.compute_Nk(k, candidate)
                error = 0.0
                for d in range(self.m):
                    diff = Nk[d] - self.mean_t[d]
                    error += diff * diff                    
                if error < min_error:
                    min_error = error
                    best_vec = candidate                    
            seq_windows.append(best_vec)            
        return seq_windows

    def generate(self, num_windows, tau=0.0):
        """Generate sequence of vector windows with temperature-controlled randomness"""
        assert self.trained, "Model must be trained first"
        if tau < 0:
            raise ValueError("Temperature must be non-negative")            
        generated_windows = []        
        for k in range(num_windows):
            # Generate candidate vectors
            candidates = []
            for _ in range(100):
                candidate = [random.gauss(self.mean_t[d], 0.1) for d in range(self.m)]
                candidates.append(candidate)
            
            # Score candidates
            scores = []
            for candidate in candidates:
                Nk = self.compute_Nk(k, candidate)
                error = 0.0
                for d in range(self.m):
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

    def dd_features(self, seq, t=None):
        """
        Extract feature vector for a vector sequence
        """
        tg = t or self.predict_t(seq)
        feats = {}
        # Deviation value 
        feats['d'] = [self.D([seq], [tg])]
        # Flatten P tensor (PWC)
        p_flat = []
        for i in range(self.m):
            for j in range(self.m):
                p_flat.extend(self.P[i][j])
        feats['pwc'] = p_flat
        # Flatten M matrix
        m_flat = []
        for row in self.M:
            m_flat.extend(row)
        feats['mtx'] = m_flat
        # Position-based features
        windows = self.extract_windows(seq)
        L = len(windows)
        pos_feats = []
        for k, window in enumerate(windows):
            for i in range(self.m):
                for j in range(self.m):
                    for g in range(self.o):
                        period = self.periods[i][j][g]
                        phi = math.cos(2 * math.pi * k / period)
                        pos_feats.append(phi)
        feats['pos'] = pos_feats
        feats['all'] = feats['d'] + feats['pwc'] + feats['mtx'] + feats['pos']       
        return feats

    def show(self, what=None, first_num=5):
        """
        Display model status.
        """
        # Default attributes to show
        default_attrs = ['config', 'M', 'P', 'periods']
        
        # Handle different what parameter types
        if what is None:
            attrs = default_attrs
        elif what == 'all':
            attrs = ['config', 'M', 'P', 'periods']
        elif isinstance(what, str):
            attrs = [what]
        else:
            attrs = what
            
        print("NumDualDescriptorPM Status")
        print("=" * 50)
        
        # Display each requested attribute
        for attr in attrs:
            if attr == 'config':
                # Display basic configuration
                print("\n[Configuration]")
                print(f"{'Vector dim:':<15} {self.m}")
                print(f"{'Rank:':<15} {self.rank} ({self.rank_mode} mode)")
                print(f"{'Basis count:':<15} {self.o}")
                print(f"{'Window mode:':<15} {self.mode}")
                print(f"{'Trained:':<15} {self.trained}")
                if self.mode == 'nonlinear':
                    print(f"{'Step size:':<15} {self.step or self.rank}")
            
            elif attr == 'M':
                # Display transformation matrix
                print("\n[Transformation Matrix (M)]")
                print("Matrix values:")
                for i in range(min(first_num, len(self.M))):
                    vals = self.M[i][:min(first_num, len(self.M[i]))]
                    print(f"  Row {i}: [{', '.join(f'{x:.4f}' for x in vals)}" + 
                          (f", ...]" if len(self.M[i]) > first_num else "]"))
            
            elif attr == 'P':
                # Display weight tensor
                print("\n[Weight Tensor (P)]")
                print(f"Shape: {len(self.P)}x{len(self.P[0])}x{len(self.P[0][0])}")
                print("Sample slices:")
                for i in range(min(first_num, len(self.P))):
                    for j in range(min(first_num, len(self.P[i]))):
                        vals = self.P[i][j][:min(first_num, len(self.P[i][j]))]
                        print(f"  P[{i}][{j}][:]: [{', '.join(f'{v:.6f}' for v in vals)}" + 
                              (f", ...]" if len(self.P[i][j]) > first_num else "]"))
            
            elif attr == 'periods':
                # Display basis function periods
                print("\n[Basis Periods]")
                print(f"Shape: {len(self.periods)}x{len(self.periods[0])}x{len(self.periods[0][0])}")
                print("Sample values:")
                for i in range(min(first_num, len(self.periods))):
                    for j in range(min(first_num, len(self.periods[i]))):
                        vals = self.periods[i][j][:min(first_num, len(self.periods[i][j]))]
                        print(f"  periods[{i}][{j}][:]: {vals}" + 
                              (f", ..." if len(self.periods[i][j]) > first_num else ""))
            
            else:
                print(f"\n[Unknown attribute: {attr}]")
        
        print("=" * 50)

    def count_parameters(self):
        """Count learnable parameters (P tensor and M matrix)."""
        m = self.m
        o = self.o
        total_params = m * m * (o + 1)
        print(f"Parameter Count:")
        print(f"  P tensor: {m}×{m}×{o} = {m*m*o} parameters")
        print(f"  M matrix: {m}×{m} = {m*m} parameters")
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

    #random.seed(3)
    
    # Configuration
    vec_dim = 4  # Dimension of input vectors
    rank = 1     # Window size for aggregation
    num_basis = 7 # Number of basis functions
    
    # Create descriptor for vector sequences
    dd = NumDualDescriptorPM(vec_dim=vec_dim, rank=rank, rank_op='avg', 
                            num_basis=num_basis, mode='linear')
    
    # Generate synthetic training data
    m = vec_dim
    num_seqs = 30
    min_len, max_len = 100, 200
    seqs = []
    t_list = []
    for _ in range(num_seqs):
        L = random.randint(min_len, max_len)
        seq = [[random.uniform(-1,1) for _ in range(m)] for __ in range(L)]
        seqs.append(seq)
        t_list.append([random.uniform(-1,1) for _ in range(m)])
    
    # Train the model
    print("Training model...")
    history = dd.train(seqs, t_list, max_iters=20, print_every=1)
    
    # Show model status
    dd.show(['config', 'M'])

    # Count learnable parameters
    dd.count_parameters()
    
    # Predict targets
    print("\nPredictions:")
    for i, seq in enumerate(seqs[:3]):  # First 3 sequences
        t_pred = dd.predict_t(seq)
        print(f"Seq {i+1}: Target={[f'{x:.4f}' for x in t_list[i]]}")
        print(f"         Predicted={[f'{x:.4f}' for x in t_pred]}")
    
    # Calculate prediction correlations
    preds = [dd.predict_t(seq) for seq in seqs]
    correlations = []
    for i in range(m):
        actual = [t[i] for t in t_list]
        predicted = [p[i] for p in preds]
        corr = correlation(actual, predicted)
        correlations.append(corr)
        print(f"Dim {i} correlation: {corr:.4f}")
    print(f"Average correlation: {mean(correlations):.4f}")
    
    # Reconstruct representative sequence
    print("\nReconstructing representative sequence...")
    repr_seq = dd.reconstruct()
    print(f"First 3 windows: {repr_seq[:3]}")
    
    # Generate new sequence
    print("\nGenerating new sequence...")
    gen_seq = dd.generate(num_windows=5, tau=0.2)
    print(f"Generated windows: {gen_seq}")
    
    # Extract features
    print("\nExtracting features for first sequence...")
    features = dd.dd_features(seqs[0])
    print(f"Feature vector length: {len(features['all'])}")
    print(f"First 5 features: {features['all'][:5]}")
    
    # Gradient Descent Training
    print("\nTraining with Gradient Descent...")
    dd_grad = NumDualDescriptorPM(vec_dim=vec_dim, num_basis=num_basis, rank=rank, rank_op='avg')
    grad_history = dd_grad.grad_train(seqs, t_list, max_iters=300, learning_rate=1.0)

    # Calculate prediction correlations
    preds = [dd_grad.predict_t(seq) for seq in seqs]
    correlations = []
    for i in range(m):
        actual = [t[i] for t in t_list]
        predicted = [p[i] for p in preds]
        corr = correlation(actual, predicted)
        correlations.append(corr)
        print(f"Dim {i} correlation: {corr:.4f}")
    print(f"Average correlation: {mean(correlations):.4f}")
    
    # Auto-Training Example
    print("\nAuto-Training in 'reg' mode...")
    dd_auto = NumDualDescriptorPM(vec_dim=vec_dim, rank=rank, num_basis=num_basis)
    auto_history = dd_auto.auto_train(seqs, auto_mode='reg', max_iters=20, learning_rate=0.05)
    
    # Save and load model
    print("\nSaving and loading model...")
    dd.save("vector_model.pkl")
    dd_loaded = NumDualDescriptorPM.load("vector_model.pkl")
    
    # Verify loaded model
    t_pred_loaded = dd_loaded.predict_t(seqs[0])
    print(f"Prediction from loaded model: {[round(x, 4) for x in t_pred_loaded]}")
