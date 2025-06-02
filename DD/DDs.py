# Copyright (C) 2005-2025, Bin-Guang Ma (mbg@mail.hzau.edu.cn). All rights reserved.
# The Dual Descriptor Scalar class
# Author: Bin-Guang Ma; Date: 2025-5-25

# The program is provided as it is and without warranty of any kind,
# either expressed or implied.

import math
import random
import itertools
import pickle

# Define picklable basis function classes
class CosineBasis:
    def __init__(self, period):
        self.period = period
    def __call__(self, k):
        return math.cos(2 * math.pi * k / self.period)

class SineBasis:
    def __init__(self, period):
        self.period = period
    def __call__(self, k):
        return math.sin(2 * math.pi * k / self.period)
    
class DualDescriptorScalar:
    """
    Scalar Dual Descriptor with rank (r-per/k-mer) extension.
    - charset: list of base characters.
    - rank: the r-per/k-mer length (r-per means permutations of length r).    
    - num_basis: number of default basis functions for PWF.
    - user_bases: the basis functions for PWF given by user,
      a list of callables like b(k) where k is integer for position index. 
    - mode: 'linear' or 'nonlinear'; for nonlinear model, a user_step can be given.
    After training, you can predict the target value t for a sequence via predict_t(),
    extract features per sequence via dd_features(), and inspect current model via show().
    """
    def __init__(self, charset, rank=1, num_basis=5, user_bases=None, mode='linear', user_step=None):
        self.charset = list(charset)
        self.rank = rank
        self.o = num_basis        
        assert mode in ('linear','nonlinear')
        self.mode = mode
        self.step = user_step
        self.trained = False
        # 1) Generate all possible tokens (k-mers + right-padded with '_')
        tokens = []
        for r in range(1, rank+1):
            for prefix in itertools.product(self.charset, repeat=r):
                tok = ''.join(prefix).ljust(rank, '_')
                tokens.append(tok)
        self.tokens = list(set(tokens))
        
        # 2) Initialize Composition Weight Map (CWF): token → scalar weight
        self.x = {tok: i for i, tok in enumerate(self.tokens)}
        
        # 3) Set basis functions; default basis functions: cosines of periods 2..2+o-1
        if user_bases:
            self.basis = user_bases
            self.o = len(user_bases)
        else:
            self.basis = [CosineBasis(period) for period in range(2, 2 + self.o)]
            
        # 4) Initialize Position Weight Coefficients (PWC): a_i for each basis
        self.a = [1.0]*self.o

    def extract_tokens(self, seq):
        """
        Extract k-mer tokens from sequence.
        'linear': slide window by 1
        'nonlinear': slide by rank or user step, pad right with '_'        
        """
        L = len(seq)
        if self.mode == 'linear':
            return [seq[i:i+self.rank] for i in range(L - self.rank + 1)]
        else:
            toks = []
            step = self.step or self.rank
            for i in range(0, L, step):
                frag = seq[i:i+self.rank]
                toks.append(frag.ljust(self.rank, '_'))
            return toks

    def P(self, k):
        """
        Compute scalar P(k) = sum_i a[i] * basis[i](k).
        """
        return sum(self.a[i] * self.basis[i](k) for i in range(self.o))

    def describe(self, seq):
        """
        Compute list of N(k)=P(k)*x[token_k] for a given sequence.
        """
        toks = self.extract_tokens(seq)
        return [self.P(k) * self.x[tok] for k, tok in enumerate(toks)]

    def S(self, seq):
        """
        Compute list of S(l)=sum(P(k)*x[token_k]) (k=1,...,l; l=1,...,L) for a given sequence.        
        """
        toks = self.extract_tokens(seq)
        S = 0.0; S_list = []
        for k, tok in enumerate(toks):
            S += self.P(k) * self.x[tok]
            S_list.append(S)
        return S_list        

    def d(self, seq, t):
        """
        Compute mean squared deviation d for a single sequence:
        d = average over positions of (N(k)-t)^2
        """
        total = 0.0
        Nk_list = self.describe(seq)
        L = len(Nk_list)
        for Nk in Nk_list:
            diff = Nk - t
            total += diff * diff
        return total / L if L > 0 else 0.0

    def D(self, seqs, t_list):
        """
        Compute mean squared deviation D across sequences:
        D = average over all positions of (N(k)-t_seq)^2
        """
        total = 0.0
        count = 0
        for seq, t in zip(seqs, t_list):
            for Nk in self.describe(seq):
                diff = Nk - t
                total += diff*diff
                count += 1
        return total/count if count else 0.0    

    def _solve_linear(self, U, v):
        """
        Solve linear system U · x = v by Gauss-Jordan elimination.
        U: n×n matrix, v: length-n vector
        Returns solution x of length n.
        """
        n = len(U)
        M = [U[i][:] + [v[i]] for i in range(n)]
        for i in range(n):
            piv = M[i][i]
            if abs(piv)<1e-12:
                continue
            M[i] = [mij/piv for mij in M[i]]
            for r in range(n):
                if r==i: continue
                fac = M[r][i]
                M[r] = [M[r][c] - fac*M[i][c] for c in range(n+1)]
        return [M[i][-1] for i in range(n)]

    def update_PWC(self, seqs, t_list):
        """
        Update PWC coefficients a via solving u·a = v:
        u[i][j] = mean basis[i]*basis[j]*x_k^2
        v[i]    = mean basis[i]*x_k*t_seq
        """
        o = self.o
        u = [[0.0]*o for _ in range(o)]
        v = [0.0]*o
        N = 0
        for seq, t in zip(seqs, t_list):
            toks = self.extract_tokens(seq)
            for k, tok in enumerate(toks):
                xk = self.x[tok]
                b = [self.basis[i](k) for i in range(o)]
                for i in range(o):
                    v[i] += b[i] * xk * t
                    for j in range(o):
                        u[i][j] += b[i]*b[j]*xk*xk
                N += 1
        # normalize
        for i in range(o):
            v[i] /= N
            for j in range(o):
                u[i][j] /= N
        # solve
        self.a = self._solve_linear(u, v)

    def update_CWM(self, seqs, t_list):
        """
        Update CWM x[token] via closed-form:
        x = (sum P(k)*t_seq) / (sum P(k)^2)
        """
        sumP = {tok:0.0 for tok in self.tokens}
        sumP2 = {tok:0.0 for tok in self.tokens}
        for seq, t in zip(seqs, t_list):
            toks = self.extract_tokens(seq)
            for k, tok in enumerate(toks):
                pk = self.P(k)
                sumP[tok] += pk * t
                sumP2[tok] += pk*pk
        for tok in self.tokens:
            denom = sumP2[tok]
            if denom>0:
                self.x[tok] = sumP[tok]/denom

    def train(self, seqs, t_list, max_iters=1000, tol=1e-8):
        """
        Alternate optimization until D converges.
        Returns history of D values.
        """
        D_prev = float('inf')
        history = []
        for it in range(max_iters):
            self.update_PWC(seqs, t_list)
            self.update_CWM(seqs, t_list)
            D = self.D(seqs, t_list)
            history.append(D)
            print(f"Iter {it:2d}: D = {D:.6e}")
            if D >= D_prev - tol:
                print("Converged.")
                break
            D_prev = D
        self.mean_L = int(sum(len(self.extract_tokens(seq)) for seq in seqs) / len(seqs))
        self.mean_t = sum(t_list) / len(t_list)
        self.trained = True
        return history

    def grad_train(self, seqs, t_list, max_iters=1000, tol=1e-8, learning_rate=0.01, continued=False):
        """
        Train using gradient descent optimization instead of closed-form solutions.
        Alternates between updating PWC (a) and CWM (x) using gradients.
        
        Args:
            seqs: List of training sequences
            t_list: List of target values corresponding to sequences
            max_iters: Maximum number of training iterations
            tol: Tolerance for convergence detection
            learning_rate: Step size for gradient updates
            
        Returns:
            history: List of D values during training
        """
        if not continued:
            # Use small random values to initialize CWM and PWC.
            self.x = {tok: random.uniform(-0.1, 0.1) for tok in self.tokens}
            self.a = [random.uniform(-0.1, 0.1) for _ in range(self.o)]
            
        D_prev = float('inf')
        history = []
        
        for it in range(max_iters):
            # Initialize gradients
            grad_a = [0.0] * self.o  # Gradient for PWC coefficients (a)
            grad_x = {tok: 0.0 for tok in self.tokens}  # Gradient for CWM weights (x)
            total_positions = 0  # Total number of positions across all sequences
            
            # Compute gradients over all sequences and positions
            for seq, t_target in zip(seqs, t_list):
                toks = self.extract_tokens(seq)
                positions = len(toks)
                total_positions += positions
                
                # Precompute P(k) for all positions in this sequence
                P_vals = [self.P(k) for k in range(positions)]
                
                for k, tok in enumerate(toks):
                    # Compute prediction and error
                    Nk = P_vals[k] * self.x[tok]
                    error = Nk - t_target
                    
                    # Compute gradient for PWC coefficients (a)
                    for i in range(self.o):
                        grad_a[i] += error * self.basis[i](k) * self.x[tok]
                    
                    # Compute gradient for CWM weight (x[tok])
                    grad_x[tok] += error * P_vals[k]
            
            # Normalize gradients by total number of positions
            if total_positions > 0:
                grad_a = [g / total_positions for g in grad_a]
                for tok in self.tokens:
                    grad_x[tok] /= total_positions
            
            # Update parameters using gradient descent
            # Update PWC coefficients (a)
            for i in range(self.o):
                self.a[i] -= learning_rate * grad_a[i]
                
            # Update CWM weights (x)
            for tok in self.tokens:
                self.x[tok] -= learning_rate * grad_x[tok]
            
            # Compute current loss
            D_current = self.D(seqs, t_list)
            history.append(D_current)
            print(f"Iter {it:2d}: D = {D_current:.6e}")
            
            # Check convergence
            if D_prev - D_current < tol:
                print("Converged.")
                break
                
            D_prev = D_current
        
        # Store training statistics
        self.mean_L = int(sum(len(self.extract_tokens(seq)) for seq in seqs) / len(seqs))
        self.mean_t = sum(t_list) / len(t_list)
        self.trained = True
        return history

    def auto_train(self, seqs, max_iters=1000, tol=1e-8, learning_rate=0.01, 
                   continued=False, auto_mode='reg'):
        """
        Self-training method using gradient descent with two modes:
        1. 'gap' mode: Predicts current token's value (self-supervised gap filling)
        2. 'reg' mode: Predicts next token's value (autoregressive prediction)
        
        Args:
            seqs: List of training sequences
            max_iters: Maximum number of training iterations
            tol: Tolerance for convergence detection
            learning_rate: Step size for gradient updates
            continued: Whether to continue from existing parameters
            auto_mode: 'gap' or 'reg' (default 'reg')
            
        Returns:
            history: List of loss values during training
        """
        if auto_mode not in ('gap', 'reg'):
            raise ValueError("auto_mode must be 'gap' or 'reg'")
            
        if not continued:
            # Initialize CWM and PWC with small random values
            self.x = {tok: random.uniform(-0.1, 0.1) for tok in self.tokens}
            self.a = [random.uniform(-0.1, 0.1) for _ in range(self.o)]
            
        prev_loss = float('inf')
        history = []
        
        for it in range(max_iters):
            # Initialize gradients and loss
            grad_a = [0.0] * self.o
            grad_x = {tok: 0.0 for tok in self.tokens}
            total_loss = 0.0
            total_positions = 0
            
            # Process all sequences
            for seq in seqs:
                tokens = self.extract_tokens(seq)
                positions = len(tokens)
                
                # Skip sequences with insufficient tokens based on mode
                if auto_mode == 'reg' and positions < 2:
                    continue
                if auto_mode == 'gap' and positions < 1:
                    continue
                    
                # Precompute P(k) for all positions
                P_vals = [self.P(k) for k in range(positions)]
                
                if auto_mode == 'gap':
                    # Self-supervised gap filling: predict current token's value
                    for k, token in enumerate(tokens):
                        target = self.x[token]
                        prediction = P_vals[k] * self.x[token]
                        error = prediction - target
                        total_loss += error * error
                        total_positions += 1
                        
                        # Compute gradients
                        for i in range(self.o):
                            grad_a[i] += error * self.basis[i](k) * self.x[token]
                        grad_x[token] += error * P_vals[k]
                        
                else:  # auto_mode == 'reg'
                    # Autoregressive: predict next token's value
                    total_positions += positions - 1
                    for k in range(positions - 1):
                        current_token = tokens[k]
                        next_token = tokens[k+1]
                        
                        # Predict current token's representation
                        prediction = P_vals[k] * self.x[current_token]
                        # Target is the representation of the next token
                        target = self.x[next_token]
                        error = prediction - target
                        total_loss += error * error
                        
                        # Compute gradients
                        # For PWC coefficients
                        for i in range(self.o):
                            grad_a[i] += error * self.basis[i](k) * self.x[current_token]
                        # For current token weight
                        grad_x[current_token] += error * P_vals[k]
                        # For next token weight
                        grad_x[next_token] += error * (-1)  # Derivative w.r.t target
            
            # Skip iteration if no valid positions
            if total_positions == 0:
                print(f"Iter {it:3d}: No valid positions - skipping")
                history.append(prev_loss)
                continue
                
            # Calculate average loss
            total_loss /= total_positions
            
            # Normalize gradients
            grad_a = [g / total_positions for g in grad_a]
            for token in self.tokens:
                if token in grad_x:  # Only tokens that appeared in sequences
                    grad_x[token] /= total_positions
            
            # Update parameters
            for i in range(self.o):
                self.a[i] -= learning_rate * grad_a[i]
            for token in self.tokens:
                if token in grad_x:  # Only update tokens that appeared
                    self.x[token] -= learning_rate * grad_x[token]
            
            # Record and print progress
            history.append(total_loss)
            print(f"Iter {it:3d}: Loss = {total_loss:.6e}")
            
            # Check convergence
            if prev_loss - total_loss < tol:
                print("Converged.")
                break
                
            prev_loss = total_loss
        
        # Calculate average token weight (x) for sequence generation
        total_weight = 0.0
        token_count = 0
        for seq in seqs:
            tokens = self.extract_tokens(seq)
            for token in tokens:
                total_weight += self.x[token]
                token_count += 1
        self.mean_t = total_weight / token_count if token_count > 0 else 0.0
        
        # Update sequence length statistics
        self.mean_L = int(sum(len(self.extract_tokens(seq)) for seq in seqs) / len(seqs)) if seqs else 0
        self.trained = True
        return history

    def show(self):
        """
        Display current DualDescriptor status:
        rank, mode, number of tokens, PWC and sample CWF.
        """
        print("DualDescriptorcalar Status:")
        print(f"  rank = {self.rank}, mode = {self.mode}, bases = {self.o}")
        print(f"  # tokens = {len(self.tokens)}")
        print("  PWC a coefficients:")
        print("   ", self.a)
        print("  Sample CWM (first 5 tokens):")
        for tok in sorted(self.tokens)[:5]:
            print(f"    {tok}: {self.x[tok]:.4f}")       

    def dd_features(self, seq):
        """
        Extract feature dict for one sequence:
        {'pwc': [PWC coefficients], 
         'cwf': [CWF values for all tokens sorted],
         'frq': [Position-weighted frequency for each token],
         'pdv': [Partial Dual Variable for each token],
         'all': [concatenate the above]}
        """
        feats = {}
        # PWC
        feats['pwc'] = list(self.a)
        # CWF sorted by token
        cwfs = []
        for tok in sorted(self.tokens):
            cwfs.append(self.x[tok])
        feats['cwf'] = cwfs    
        # Position-weighted frequencies and partial dual variables
        frqd = {}; pdvd = {}
        for tok in sorted(self.tokens):
            frqd[tok] = 0.0
            pdvd[tok] = 0.0
        toks = self.extract_tokens(seq)
        L = len(toks)
        for k, tok in enumerate(toks):            
            for i in range(self.o):
                Pki = self.a[i] * self.basis[i](k)            
                frqd[tok] += Pki
                pdvd[tok] += Pki * self.x[tok]
        frqs = []; pdvs = []
        for tok in sorted(self.tokens):
            frqd[tok] = frqd[tok] / L if L>0 else 0.0
            frqs.append(frqd[tok])
            pdvd[tok] = pdvd[tok] / L if L>0 else 0.0
            pdvs.append(pdvd[tok])        
        feats['frq'] = frqs[:-1] #remove the last one because of dependency
        feats['pdv'] = pdvs[:-1] #remove the last one because of dependency
        feats['all'] = feats['pwc'] + feats['cwf'] + feats['frq'] + feats['pdv']        
        return feats

    def predict_t(self, seq):
        """
        Predict the target value t for a given sequence.
        The optimal t is the mean of all N(k) values in the sequence.        
        Arg: seq (str), input sequence to predict.            
        Return (float): predicted target value t.
        """
        # Get all N(k) values for the sequence
        Nk_list = self.describe(seq)
        if not Nk_list:
            return 0.0  # Default value if no tokens are extracted
        # Optimal t minimizes MSE: t = mean(N(k))
        t_pred = sum(Nk_list) / len(Nk_list)
        return t_pred
    
    def reconstruct(self):
        """
        Reconstruct a representative sequence by minimizing the squared error 
        between N(k) and the average target value of the training set.
        Return a reconstructed sequence of average length.
        """
        # Step 1: get average sequence length and average target t_avg
        assert self.trained, "DD must be trained first to reconstruct a representative sequence"        
        avg_length = self.mean_L; t_avg = self.mean_t
        # Step 2: Generate optimal k-mer for each position
        repr_seq = []
        for k in range(avg_length):
            min_error = float('inf')
            best_tok = '_' * self.rank  # default padding            
            # Step 3: Iterate through all possible tokens
            for tok in self.tokens:
                # Trim or pad token to match rank (for edge cases)
                trimmed_tok = tok[:self.rank].ljust(self.rank, '_')
                x_tok = self.x.get(trimmed_tok, 0.0)
                Nk = x_tok * self.P(k)
                error = (Nk - t_avg) ** 2  # Squared error                
                # Update best token if error is smaller
                if error < min_error:
                    min_error = error
                    best_tok = trimmed_tok            
            # Step 4: Append token based on mode
            repr_seq.append(best_tok.replace('_', ''))    
        # Step 5: Join and truncate to average length
        final_seq = ''.join(repr_seq)[:avg_length]
        return final_seq    

    def generate(self, L, tau=0.0):
        """
        Generate a sequence of length L using trained parameters.
        - Each token is a full k-mer of length `rank` (no padding).
        - Tau (temperature) controls randomness: higher values increase diversity;
        - tau=0: Always choose the best token (deterministic).        
        Args: L (int): target sequence length; tau (float): randomness control (≥ 0.0). Default 0.0 (deterministic).            
        Return (str): Generated sequence, truncated to length L.
        """
        if tau < 0:
            raise ValueError("Temperature must be non-negative.")
        assert self.trained, "DD must be trained first to generate a sequence"        
        # Step 1: Get average target value
        t_avg = self.mean_t        
        # Step 2: Filter valid tokens (exact length=rank, no padding)
        valid_tokens = [
            tok for tok in self.tokens 
            if len(tok) == self.rank and '_' not in tok
        ]
        if not valid_tokens:
            raise ValueError("No valid full-length tokens found.")        
        # Step 3: Calculate number of blocks needed
        num_blocks = (L + self.rank - 1) // self.rank  # Ceiling division        
        generated_sequence = []
        for k in range(num_blocks):
            # Step 4: Compute scores for all valid tokens
            scores = {}
            for tok in valid_tokens:
                x_tok = self.x[tok]
                Nk = x_tok * self.P(k)  # P(k) is position weight for block k
                error = (Nk - t_avg) ** 2
                scores[tok] = -error  # Higher score = better alignment            
            # Step 5: Select token based on temperature
            if tau == 0:
                # Deterministic: choose token with highest score
                best_tok = max(scores.keys(), key=lambda t: scores[t])
                generated_sequence.append(best_tok)                
            else:
                # Stochastic: apply softmax with temperature
                tokens = list(scores.keys())
                exp_scores = [math.exp(scores[t] / tau) for t in tokens]
                sum_exp = sum(exp_scores)
                probs = [es / sum_exp for es in exp_scores]
                chosen_tok = random.choices(tokens, weights=probs, k=1)[0]
                generated_sequence.append(chosen_tok)        
        # Step 6: Concatenate and truncate to L characters
        full_seq = ''.join(generated_sequence)
        return full_seq[:L]

    def part_train(self, num_seqs, max_iters=1000, tol=1e-8, learning_rate=0.01, 
               continued=False, auto_mode='reg'):
        """
        Train position weight coefficients (a) on real-valued sequences.
        This method learns a position weighting function I(k) = Σ a_i * basis_i(k)
        without token mapping (x weights). Two training modes are supported:
        1. 'gap': Predict current value at position k
        2. 'reg': Predict next value at position k+1
        
        Args:
            num_seqs: List of real-valued sequences (each sequence is a list of floats)
            max_iters: Maximum number of training iterations
            tol: Tolerance for convergence detection
            learning_rate: Step size for gradient updates
            continued: Whether to continue training from current coefficients
            auto_mode: 'gap' (predict current value) or 'reg' (predict next value)
            
        Returns:
            history: List of loss values during training
        """
        if auto_mode not in ('gap', 'reg'):
            raise ValueError("auto_mode must be 'gap' or 'reg'")
        
        if not continued:
            # Initialize PWC coefficients with small random values
            self.a = [random.uniform(-0.1, 0.1) for _ in range(self.o)]
        
        prev_loss = float('inf')
        history = []
        
        for it in range(max_iters):
            # Initialize gradients and loss
            grad_a = [0.0] * self.o
            total_loss = 0.0
            total_positions = 0
            
            # Process all sequences
            for seq in num_seqs:
                positions = len(seq)
                
                # Skip sequences with insufficient positions based on mode
                if auto_mode == 'reg' and positions < 2:
                    continue
                if auto_mode == 'gap' and positions < 1:
                    continue
                    
                if auto_mode == 'gap':
                    # Self-supervised gap filling: predict current position's value
                    for k in range(positions):
                        target = seq[k]
                        prediction = self.P(k)  # I(k) = P(k) = Σ a_i * basis_i(k)
                        error = prediction - target
                        total_loss += error * error
                        total_positions += 1
                        
                        # Compute gradients for each basis function
                        for i in range(self.o):
                            grad_a[i] += error * self.basis[i](k)
                            
                else:  # auto_mode == 'reg'
                    # Autoregressive: predict next position's value
                    total_positions += positions - 1
                    for k in range(positions - 1):
                        target = seq[k+1]
                        prediction = self.P(k)  # Use I(k) to predict value at k+1
                        error = prediction - target
                        total_loss += error * error
                        
                        # Compute gradients for each basis function
                        for i in range(self.o):
                            grad_a[i] += error * self.basis[i](k)
            
            # Skip iteration if no valid positions
            if total_positions == 0:
                print(f"Iter {it:3d}: No valid positions - skipping")
                history.append(prev_loss)
                continue
                
            # Calculate average loss
            total_loss /= total_positions
            
            # Normalize gradients
            grad_a = [g / total_positions for g in grad_a]
            
            # Update parameters
            for i in range(self.o):
                self.a[i] -= learning_rate * grad_a[i]
            
            # Record and print progress
            history.append(total_loss)
            print(f"Iter {it:3d}: Loss = {total_loss:.6e}")
            
            # Check convergence
            if prev_loss - total_loss < tol:
                print("Converged.")
                break
                
            prev_loss = total_loss
        
        self.trained = True
        return history    

    def part_generate(self, L, mode='gap', initial_value=0.0, tau=0.0):
        """
        Generate a real-valued sequence of length L using the trained position weight function.
        
        Args:
            L: Length of sequence to generate
            mode: 'gap' or 'reg' (should match training mode)
            initial_value: Starting value for 'reg' mode sequences
            tau: Temperature controlling randomness (standard deviation of Gaussian noise)
            
        Returns:
            List of generated real values
        """
        if tau < 0:
            raise ValueError("Temperature must be non-negative.")
        
        sequence = []
        
        if mode == 'gap':
            # Gap mode: generate each position independently
            for k in range(L):
                base_value = self.P(k)  # Base value from position weight function
                # Add Gaussian noise scaled by temperature
                noise = random.gauss(0, tau) if tau > 0 else 0
                sequence.append(base_value + noise)
        
        elif mode == 'reg':
            # Reg mode: generate sequence autoregressively
            sequence = [initial_value]
            # Generate remaining values
            for k in range(L-1):
                # Predict next value using current position's weight function
                base_value = self.P(k)
                # Add Gaussian noise scaled by temperature
                noise = random.gauss(0, tau) if tau > 0 else 0
                next_value = base_value + noise
                sequence.append(next_value)
        
        else:
            raise ValueError("Invalid mode. Must be 'gap' or 'reg'")
        
        return sequence

    def double_train(self, char_seqs, auto_mode='gap', part_mode='gap', 
                 max_iters1=1000, max_iters2=1000, 
                 learning_rate1=0.01, learning_rate2=0.01,
                 tol1=1e-8, tol2=1e-8, continued=False):
        """
        Enhanced two-stage training method:
        1. First trains on character sequences using auto_train
        2. Converts character sequences to real-valued S-sequences
        3. Creates new PWC for second stage to capture higher-order patterns
        4. Trains on S-sequences using part_train with new PWC
        
        Args:
            char_seqs: List of character sequences for initial training
            auto_mode: Training mode for first stage ('gap' or 'reg')
            part_mode: Training mode for second stage ('gap' or 'reg')
            max_iters1: Max iterations for first training stage
            max_iters2: Max iterations for second training stage
            learning_rate1: Learning rate for first training stage
            learning_rate2: Learning rate for second training stage
            tol1: Tolerance for convergence in first stage
            tol2: Tolerance for convergence in second stage
            continued: Whether to continue from existing parameters
            
        Returns:
            Tuple of (auto_history, part_history) training histories
        """
        # Preserve original state for later restoration
        original_state = {
            'a': self.a.copy(),
            'x': self.x.copy(),
            'trained': self.trained
        }
        
        # Stage 1: Train on character sequences
        print("=== Stage 1: Training on character sequences ===")
        auto_history = self.auto_train(
            char_seqs, 
            max_iters=max_iters1,
            learning_rate=learning_rate1,
            tol=tol1,
            continued=continued,
            auto_mode=auto_mode
        )
        
        # Store stage 1 parameters
        self.stage1_a = self.a.copy()
        self.stage1_x = self.x.copy()
        
        # Stage 2: Convert to real-valued S-sequences
        print("\n=== Stage 2: Converting to S-sequences ===")
        s_seqs = []
        for seq in char_seqs:
            s_sequence = self.S(seq)
            s_seqs.append(s_sequence)
            print(f"Converted sequence (length {len(s_sequence)}): {s_sequence[:5]}...")
        
        # Create new PWC for stage 2 using CosineBasis
        print("\n=== Stage 2: Initializing new PWC for S-sequences ===")
        self.stage2_a = [random.uniform(-0.1, 0.1) for _ in range(self.o)]
        
        # Create I(k) function using CosineBasis (serializable)
        self.stage2_basis = [CosineBasis(period) for period in range(2, 2 + self.o)]
        
        # Restore original parameters for character processing
        self.a = original_state['a']
        self.x = original_state['x']
        self.trained = original_state['trained']
        
        # Train on S-sequences using part_train with new PWC
        print("\n=== Stage 2: Training on S-sequences ===")
        # Temporarily swap parameters for training
        current_a, current_basis = self.a, self.basis
        self.a, self.basis = self.stage2_a, self.stage2_basis
        
        part_history = self.part_train(
            s_seqs,
            max_iters=max_iters2,
            learning_rate=learning_rate2,
            tol=tol2,
            continued=True,
            auto_mode=part_mode
        )
        
        # Update and restore parameters
        self.stage2_a = self.a.copy()
        self.a, self.basis = current_a, current_basis
        
        # Final restoration
        self.trained = True
        return auto_history, part_history

    def I(self, k):
        """Serializable position weight function for S-sequences"""
        if hasattr(self, 'stage2_a') and hasattr(self, 'stage2_basis'):
            return sum(a * basis(k) for a, basis in zip(self.stage2_a, self.stage2_basis))
        return 0.0
    
    def double_generate(self, L, tau1=0.0, tau2=0.0):
        """
        Generate a character sequence that integrates patterns from both stages:
        1. Uses stage 1 parameters for token selection
        2. Uses stage 2 parameters to guide cumulative S-values
        
        Args:
            L: Length of sequence to generate
            tau1: Temperature for token selection randomness
            tau2: Temperature for S-value guidance randomness
            
        Returns:
            Generated character sequence
        """
        # Save current parameters
        current_a, current_x = self.a, self.x
        
        # Use stage 1 parameters for token processing
        self.a, self.x = self.stage1_a, self.stage1_x
        
        # Initialize sequence and cumulative S
        seq = []
        cumulative_s = 0.0
        valid_tokens = [tok for tok in self.tokens if '_' not in tok and len(tok) == self.rank]
        
        for k in range(0, L, self.rank):
            # Calculate target S-value from stage 2
            target_s = self.I(k) + (random.gauss(0, tau2) if tau2 > 0 else 0)
            
            # Score tokens based on both stages
            scores = {}
            for token in valid_tokens:
                token_value = self.P(k) * self.x[token]
                new_s = cumulative_s + token_value
                s_error = abs(new_s - target_s)
                scores[token] = -s_error
            
            # Select token
            if tau1 == 0:
                token = max(scores, key=scores.get)
            else:
                tokens = list(scores.keys())
                exp_scores = [math.exp(scores[t] / tau1) for t in tokens]
                token = random.choices(tokens, weights=exp_scores, k=1)[0]
            
            # Update sequence and cumulative S
            seq.append(token)
            cumulative_s += self.P(k) * self.x[token]
        
        # Restore original parameters
        self.a, self.x = current_a, current_x
        
        # Combine tokens and truncate to length L
        return ''.join(seq)[:L]
    
    def save(self, filename):
        """
        Save the current state of the DualDescriptorScalar object to a binary file.
        """
        with open(filename, 'wb') as f:
            pickle.dump(self.__dict__, f)
        print(f"Model saved to {filename}")

    @classmethod
    def load(cls, filename):
        """
        Load a saved DualDescriptorScalar model from a binary file.
        """
        with open(filename, 'rb') as f:
            state = pickle.load(f)
        
        # Create new instance without calling __init__
        obj = cls.__new__(cls)
        # Restore saved state
        obj.__dict__.update(state)
        print(f"Model loaded from {filename}")
        return obj


# ===== Example Usage =====
if __name__ == "__main__":
    
    from statistics import correlation
    
    # initialize charset
    charset = ['A','C','G','T']
    
    # define user basis functions sin() instead of the default cos()   
    user_bases = [SineBasis(period) for period in range(2, 2 + 7)]
    
    # create Scalar Dual Descriptor object
    dd = DualDescriptorScalar(charset, rank=3, num_basis=6, user_bases=user_bases, mode='nonlinear', user_step=2) 

    # generate 10 sequences length 200-300 and random target scalars
    random.seed(0)
    seqs, t_list = [], []
    for _ in range(10):
        L = random.randint(200,300)
        seq = ''.join(random.choices(charset, k=L))
        seqs.append(seq)
        t_list.append(random.uniform(-1.0,1.0))

    # train
    history = dd.train(seqs, t_list)

    # show status
    dd.show()

    # predict t for a sequence after training the model
    aseq = seqs[0]
    t_pred = dd.predict_t(aseq)
    print(f"Predicted t: {t_pred:.4f}")

    # correlation between the predicted and original t values
    pred_t_list = [dd.predict_t(seq) for seq in seqs]
    print("\nCorrelation:", correlation(t_list, pred_t_list))    

    # extract features for first sequence
    features = dd.dd_features(seqs[0])['all']
    print("\nFeature vector length:", len(features))
    print("First 10 features:", features[:10])

    # reconstruct a representative sequence after training
    repr_seq = dd.reconstruct()
    print(f"Representative sequence: {repr_seq}")

    # generate sequences after training
    seq_deterministic = dd.generate(L=100, tau=0.0)
    print('\nSeq_deterministic: ', seq_deterministic)
    seq_random = dd.generate(L=100, tau=1.0)
    print('Seq_random: ', seq_random)

    # create Scalar Dual Descriptor object
    dd = DualDescriptorScalar(charset, rank=3, num_basis=6, user_bases=user_bases, mode='nonlinear', user_step=2)
    # Train with gradient descent method
    print("Training with gradient descent:")
    history_grad = dd.grad_train(seqs, t_list, learning_rate=1.0, max_iters=1000)
    
    # show status
    dd.show()

    # predict t for a sequence after training the model
    aseq = seqs[0]
    t_pred = dd.predict_t(aseq)
    print(f"Predicted t: {t_pred:.4f}")

    # correlation between the predicted and original t values
    pred_t_list = [dd.predict_t(seq) for seq in seqs]
    print("\nCorrelation:", correlation(t_list, pred_t_list))    

    # extract features for first sequence
    features = dd.dd_features(seqs[0])['all']
    print("\nFeature vector length:", len(features))
    print("First 10 features:", features[:10])

    # reconstruct a representative sequence after training
    repr_seq = dd.reconstruct()
    print(f"Representative sequence: {repr_seq}")

    # generate sequences after training
    seq_deterministic = dd.generate(L=100, tau=0.0)
    print('\nSeq_deterministic: ', seq_deterministic)
    seq_random = dd.generate(L=100, tau=1.0)
    print('Seq_random: ', seq_random)

    # Self-train using auto_train method (auto_mode='gap')
    print("\n=== Starting Self-Training with auto_train (auto_mode='gap') ===")
    loss_history = dd.auto_train(seqs, learning_rate=0.5, max_iters=1000, auto_mode='gap')
    
    # Show model status after self-training
    print("\nModel after self-training auto_mode='gap':")
    dd.show()
   
    # Extract features for first sequence
    features = dd.dd_features(seqs[0])['all']
    print("\nFeature vector length:", len(features))
    print("First 10 features:", features[:10])

    # reconstruct a representative sequence after training
    repr_seq = dd.reconstruct()
    print(f"Representative sequence: {repr_seq}")
    
    # Generate sequences using self-trained model
    print("\nGenerated sequence (deterministic):")
    print(dd.generate(L=100, tau=0.0))
    
    print("\nGenerated sequence (stochastic):")
    print(dd.generate(L=100, tau=1.0))
    
    # Self-train using auto_train_reg method (auto_mode='reg')
    print("\n=== Starting Self-Training with auto_train (auto_mode='reg') ===")
    loss_history = dd.auto_train(seqs, learning_rate=0.5, max_iters=1000, auto_mode='reg')
    
    # Show model status after self-training
    print("\nModel after self-training (auto_mode='reg'):")
    dd.show()
   
    # Extract features for first sequence
    features = dd.dd_features(seqs[0])['all']
    print("\nFeature vector length:", len(features))
    print("First 10 features:", features[:10])

    # reconstruct a representative sequence after training
    repr_seq = dd.reconstruct()
    print(f"Representative sequence: {repr_seq}")
    
    # Generate sequences using self-trained model
    print("\nGenerated sequence (deterministic):")
    print(dd.generate(L=100, tau=0.0))
    
    print("\nGenerated sequence (stochastic):")
    print(dd.generate(L=100, tau=1.0))

    # Save trained model
    dd.save("trained_model.pkl")
    
    # Load saved model
    dd_loaded = DualDescriptorScalar.load("trained_model.pkl")
    
    # Verify predictions work
    t_pred = dd_loaded.predict_t(seqs[0])
    print(f"Predicted t using loaded model: {t_pred:.4f}")
    
    # Continue training
    dd_loaded.grad_train(seqs, t_list, max_iters=50)
    
    # Test other functionality
    features = dd_loaded.dd_features(seqs[0])
    print("Feature vector length:", len(features['all']))
    
    repr_seq = dd_loaded.reconstruct()
    print(f"Representative sequence: {repr_seq}")
    
    gen_seq = dd_loaded.generate(L=100, tau=0.5)
    print(f"Generated sequence: {gen_seq}")
    
    # Create DualDescriptorScalar object
    dd = DualDescriptorScalar(charset=['A','C','G','T'], rank=2, num_basis=5)

    # Create sample real-valued sequences
    num_seqs = [
        [0.1, 0.3, 0.5, 0.7, 0.9],  # Increasing sequence
        [1.0, 0.8, 0.6, 0.4, 0.2],  # Decreasing sequence
        [0.5, 0.2, 0.8, 0.3, 0.9]   # Random sequence
    ]

    print("=== Training in 'gap' mode (predict current value) ===")
    gap_history = dd.part_train(num_seqs, auto_mode='gap', learning_rate=0.1, max_iters=500)
    print("\nFinal PWC coefficients:", dd.a)

    print("\n=== Training in 'reg' mode (predict next value) ===")
    reg_history = dd.part_train(num_seqs, auto_mode='reg', learning_rate=0.1, max_iters=500, continued=True)
    print("\nFinal PWC coefficients:", dd.a)

    # Evaluate on new sequence
    test_seq = [0.4, 0.6, 0.8, 1.0]
    print("\n=== Evaluation ===")
    print("Position weights for test sequence:")
    for k in range(len(test_seq)):
        print(f"Position {k}: I(k) = {dd.P(k):.4f}")

    # Create sample real-valued sequences
    num_seqs = [
        [0.1, 0.2, 0.3, 0.4, 0.5],  # Linear sequence
        [0.5, 0.4, 0.3, 0.2, 0.1],  # Inverse sequence
        [0.3, 0.1, 0.4, 0.2, 0.5]   # Random sequence
    ]

    # Train in gap mode (predict current value)
    print("Training in 'gap' mode...")
    dd.part_train(num_seqs, auto_mode='gap', learning_rate=0.1, max_iters=100)

    # Generate sequences using gap mode
    print("\nGenerating gap mode sequences:")
    print("Deterministic (tau=0):", dd.part_generate(L=5, mode='gap', tau=0))
    print("Low randomness (tau=0.1):", dd.part_generate(L=5, mode='gap', tau=0.1))
    print("High randomness (tau=0.5):", dd.part_generate(L=5, mode='gap', tau=0.5))

    # Train in reg mode (predict next value)
    print("\nTraining in 'reg' mode...")
    dd.part_train(num_seqs, auto_mode='reg', learning_rate=0.1, max_iters=100)

    # Generate sequences using reg mode
    print("\nGenerating reg mode sequences:")
    print("Deterministic (tau=0):", 
          dd.part_generate(L=5, mode='reg', initial_value=0.0, tau=0))
    print("Low randomness (tau=0.1):", 
          dd.part_generate(L=5, mode='reg', initial_value=0.0, tau=0.1))
    print("High randomness (tau=0.5):", 
          dd.part_generate(L=5, mode='reg', initial_value=0.0, tau=0.5))


    print("=== Starting Double Training ===")
    auto_history, part_history = dd.double_train(
        seqs,
        auto_mode='reg',    # First stage: predict next token
        part_mode='reg',    # Second stage: predict next S-value
        max_iters1=1000,
        max_iters2=1000,
        learning_rate1=0.1,
        learning_rate2=0.1
    )

    # Inspect model parameters
    print("\n=== Model Parameters ===")
    print("Stage 1 PWC:", dd.stage1_a)
    print("Stage 2 PWC:", dd.stage2_a)

    # Generate sequences with different randomness levels
    print("\n=== Double Generation Results ===")
    print("Deterministic (tau1=0, tau2=0):")
    print(dd.double_generate(L=20, tau1=0, tau2=0))

    print("\nToken-random (tau1=0.5, tau2=0):")
    print(dd.double_generate(L=20, tau1=0.5, tau2=0))

    print("\nS-guided (tau1=0, tau2=0.5):")
    print(dd.double_generate(L=20, tau1=0, tau2=0.5))

    print("\nFull-random (tau1=0.5, tau2=0.5):")
    print(dd.double_generate(L=20, tau1=0.5, tau2=0.5))

    # Generate and analyze S-sequence
    def analyze_generated_sequence(dd, seq):
        """Analyze generated sequence with both stages"""
        # Calculate actual S-sequence
        actual_s = dd.S(seq)
        
        # Calculate ideal S-sequence from stage 2
        ideal_s = [dd.I(k) for k in range(len(actual_s))]
        
        # Calculate correlation
        corr = correlation(actual_s, ideal_s) if len(actual_s) > 1 else 0
        
        print(f"\nGenerated sequence: {seq}")
        print(f"S-sequence correlation: {corr:.4f}")
        print(f"First 5 S-values: Actual {actual_s[:5]}, Ideal {ideal_s[:5]}")

    print("\n=== Sequence Analysis ===")
    seq = dd.double_generate(L=20, tau1=0.1, tau2=0.1)
    analyze_generated_sequence(dd, seq)
    
    # Save trained model
    dd.save("trained_model.pkl")
    
    # Load saved model
    dd_loaded = DualDescriptorScalar.load("trained_model.pkl")
    
    # Verify predictions work
    t_pred = dd_loaded.predict_t(seqs[0])
    print(f"Predicted t using loaded model: {t_pred:.4f}")
    
    # Continue training
    dd_loaded.grad_train(seqs, t_list, max_iters=50)
    
    # Test other functionality
    features = dd_loaded.dd_features(seqs[0])
    print("Feature vector length:", len(features['all']))
    
    repr_seq = dd_loaded.reconstruct()
    print(f"Representative sequence: {repr_seq}")
    
    gen_seq = dd_loaded.generate(L=100, tau=0.5)
    print(f"Generated sequence: {gen_seq}")

    # Generate sequence with loaded model
    print("\nGenerated sequence with loaded model:")
    print(dd_loaded.double_generate(L=20, tau1=0.1, tau2=0.1))

    # Verify I() function works after loading
    print("\nTesting I() function after loading:")
    for k in range(5):
        print(f"I({k}) = {dd_loaded.I(k):.4f}")
