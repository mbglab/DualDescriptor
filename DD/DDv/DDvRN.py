# Copyright (C) 2005-2025, Bin-Guang Ma (mbg@mail.hzau.edu.cn). All rights reserved.
# The Dual Descriptor Vector class (Random AB Matrix form)
# Author: Bin-Guang Ma; Date: 2025-6-6

# The program is provided as it is and without warranty of any kind,
# either expressed or implied.

import math
import itertools
import random
import pickle

class DualDescriptorRN:
    """
    Dual Descriptor with:
      - learnable coefficient matrix Acoeff ∈ R^{m×L}
      - learnable basis matrix Bbasis ∈ R^{L×m} (now trainable with random initialization)
      - learnable token embeddings M: token → R^m
    """
    def __init__(self, charset, vec_dim=4, bas_dim=50, rank=1, rank_mode='drop', mode='linear', user_step=None):
        self.charset = list(charset)        
        self.m = vec_dim
        self.L = bas_dim
        self.rank = rank
        self.rank_mode = rank_mode
        assert mode in ('linear','nonlinear')
        self.mode = mode
        self.step = user_step
        self.trained = False
        
        # Generate all possible tokens (k-mers + right-padded with '_')
        toks = []
        if self.rank_mode=='pad':
            for r in range(1, self.rank+1):
                for prefix in itertools.product(self.charset, repeat=r):
                    tok = ''.join(prefix).ljust(self.rank, '_')
                    toks.append(tok)
        else:
            toks = [''.join(p) for p in itertools.product(self.charset, repeat=self.rank)]
        self.tokens = sorted(set(toks))

        # token embeddings
        self.M = {
            tok: [random.uniform(-0.5, 0.5) for _ in range(self.m)]
            for tok in self.tokens
        }

        # initialize Acoeff: m×L
        self.Acoeff = [[random.uniform(-0.1,0.1) for _ in range(self.L)]
                       for _ in range(self.m)]
        # initialize trainable basis Bbasis: L×m with random values
        self.Bbasis = [[random.uniform(-0.1, 0.1) for _ in range(self.m)]
                       for _ in range(self.L)]
        # cache its transpose: m×L
        self.B_t = self._transpose(self.Bbasis)

    # ---- linear algebra helpers ----
    def _transpose(self, M):
        return [list(col) for col in zip(*M)]

    def _mat_mul(self, A, B):
        # A: p×q, B: q×r → C: p×r
        p, q = len(A), len(A[0])
        r    = len(B[0])
        C = [[0.0]*r for _ in range(p)]
        for i in range(p):
            for k in range(q):
                aik = A[i][k]
                for j in range(r):
                    C[i][j] += aik * B[k][j]
        return C

    def _mat_vec(self, M, v):
        # M: p×q, v: length q → length p
        return [sum(M[i][j]*v[j] for j in range(len(v))) for i in range(len(M))]

    def _invert(self, A):
        # Gauss-Jordan invert n×n matrix A
        n = len(A)
        M = [A[i][:] + [1.0 if i==j else 0.0 for j in range(n)]
             for i in range(n)]
        for i in range(n):
            piv = M[i][i]
            if abs(piv)<1e-12:
                continue
            M[i] = [x/piv for x in M[i]]
            for r in range(n):
                if r==i: continue
                fac = M[r][i]
                M[r] = [M[r][c] - fac*M[i][c] for c in range(2*n)]
        return [row[n:] for row in M]

    # ---- tokenization ----
    def extract_tokens(self, seq):
        """
        Extract k-mer tokens from a character sequence based on tokenization mode.
        
        - 'linear': Slide window by 1 step, extracting contiguous kmers of length = rank
        - 'nonlinear': Slide window by custom step (or rank length if step not specified)
        
        For nonlinear mode, handles incomplete trailing fragments using:
        - 'pad': Pads with '_' to maintain kmer length
        - 'drop': Discards incomplete fragments
        
        Args:
            seq (str): Input character sequence to tokenize
            
        Returns:
            list: List of extracted kmer tokens
        """
        L = len(seq)
        # Linear mode: sliding window with step=1
        if self.mode == 'linear':
            return [seq[i:i+self.rank] for i in range(L - self.rank + 1)]
        
        # Nonlinear mode: stepping with custom step size
        toks = []
        step = self.step or self.rank  # Use custom step if defined, else use rank length
        
        for i in range(0, L, step):
            frag = seq[i:i+self.rank]
            frag_len = len(frag)
            
            # Pad or drop based on rank_mode setting
            if self.rank_mode == 'pad':
                # Pad fragment with '_' if shorter than rank
                toks.append(frag if frag_len == self.rank else frag.ljust(self.rank, '_'))
            elif self.rank_mode == 'drop':
                # Only add fragments that match full rank length
                if frag_len == self.rank:
                    toks.append(frag)
        return toks

    # ---- describe sequence ----
    def describe(self, seq):
        toks = self.extract_tokens(seq)
        N = []
        for k, tok in enumerate(toks):
            A = [[self.Acoeff[i][k%self.L]] for i in range(self.m)]
            B = [[self.Bbasis[k%self.L][i] for i in range(self.m)]]
            P = self._mat_mul(A, B)
            x = self.M[tok]  # m-vector
            Nk = self._mat_vec(P, x)
            N.append(Nk)
        return N

    def S(self, seq):
        """
        Compute list of S(l)=sum(N(k)) (k=1,...,l; l=1,...,L) for a given sequence.        
        """
        Nk_list = self.describe(seq)        
        S = [0.0] * self.m; S_list = []
        for l in range(len(Nk_list)):
            for i in range(self.m):
                S[i] += Nk_list[l][i]
            S_list.append(list(S))
        return S_list   

    # ---- compute deviation ----
    def deviation(self, seqs, t_list):
        tot = 0.0
        cnt = 0
        for seq, t in zip(seqs, t_list):
            for Nk in self.describe(seq):
                for i in range(self.m):
                    diff = Nk[i] - t[i]
                    tot += diff*diff
                cnt += 1
        return tot/cnt if cnt else 0.0

    # ---- update Acoeff ----
    def update_Acoeff(self, seqs, t_list):
        """
        Update coefficient matrix Acoeff using position-specific projections.
        Matches the computation method used in describe().
        
        For each token at position k:
          idx = k % L  (basis index)
          scalar = dot(Bbasis[idx], x)  (projection)
          Nk = scalar * Acoeff[:, idx]  (m-dimensional vector)
        
        We minimize ||scalar * Acoeff[:, idx] - t||^2 for each position.
        """
        m, L = self.m, self.L
        # Initialize accumulators
        numerator = [[0.0] * L for _ in range(m)]  # m x L
        denominator = [0.0] * L                    # L-dimensional
        
        for seq, t in zip(seqs, t_list):
            toks = self.extract_tokens(seq)
            for k, tok in enumerate(toks):
                idx = k % L  # Basis index for this position
                x = self.M[tok]
                
                # Compute projection scalar: Bbasis[idx] • x
                scalar = sum(self.Bbasis[idx][i] * x[i] for i in range(m))
                
                # Update accumulators
                for i in range(m):
                    numerator[i][idx] += scalar * t[i]
                denominator[idx] += scalar * scalar  # scalar²
        
        # Update Acoeff column-wise
        for j in range(L):
            if abs(denominator[j]) > 1e-12:  # Avoid division by zero
                inv_denom = 1.0 / denominator[j]
                for i in range(m):
                    self.Acoeff[i][j] = numerator[i][j] * inv_denom

    def update_Bbasis(self, seqs, t_list):
        """
        Update the basis matrix Bbasis using position-specific least squares.
        Ensures consistency with the computation in describe().
        
        Steps:
        1. For each basis index j (0 ≤ j < L):
            a. Collect all (token, target) pairs where position index k % L == j
            b. Build linear system: M_j * b_j = v_j
            c. Solve for basis vector b_j (j-th row of Bbasis)
        
        For each position with index j:
            Nk = (B_j • x) * A_j
            Minimize ||(B_j • x) * A_j - t||^2
            Solution: b_j = (Σ [x x^T * ||A_j||^2])⁻¹ * (Σ [<A_j, t> * x])
        
        Args:
            seqs: List of training sequences
            t_list: List of target vectors corresponding to sequences
        """
        m, L = self.m, self.L        
        
        # Initialize accumulators for each basis index j
        # M_j: m x m matrix, v_j: m-dimensional vector
        M_list = [[[0.0] * m for _ in range(m)] for _ in range(L)]
        v_list = [[0.0] * m for _ in range(L)]
        
        # Collect data grouped by basis index j
        for seq, t in zip(seqs, t_list):
            toks = self.extract_tokens(seq)
            for k, tok in enumerate(toks):
                j = k % L  # Basis index for this position
                x = self.M[tok]
                A_j = [self.Acoeff[i][j] for i in range(m)]  # A_j: j-th column of Acoeff
                
                # Compute ||A_j||^2 = A_j • A_j
                a_norm_sq = sum(a*a for a in A_j)
                
                # Update M_j: Σ [x x^T * ||A_j||^2]
                for r in range(m):
                    for s in range(m):
                        M_list[j][r][s] += a_norm_sq * x[r] * x[s]
                
                # Update v_j: Σ [<A_j, t> * x]
                a_dot_t = sum(A_j[i] * t[i] for i in range(m))
                for i in range(m):
                    v_list[j][i] += a_dot_t * x[i]
        
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
        """
        Update token embeddings (M) using position-specific projections.
        Matches the computation method used in describe().
        
        For each token at position k:
          idx = k % L  (basis index)
          scalar = dot(Bbasis[idx], x)  (projection)
          Nk = scalar * Acoeff[:, idx]  (m-dimensional vector)
        
        Solve linear system: Mx = b for each token using correct formulation:
          M = Σ [ (A_col ⊗ basis_vec)^T (A_col ⊗ basis_vec) ]
          b = Σ [ <A_col, t> * basis_vec ]
        """
        m, L = self.m, self.L
        # Precompute token occurrence data: {token: [(idx, target)]}
        token_data = {tok: [] for tok in self.tokens}
        
        # Collect position data for each token
        for seq, t in zip(seqs, t_list):
            toks = self.extract_tokens(seq)
            for k, tok in enumerate(toks):
                idx = k % L  # Basis index
                token_data[tok].append((idx, t))
        
        # Update each token embedding
        for tok, occurrences in token_data.items():
            if not occurrences:
                continue
                
            # Build linear system: M (m x m) and b (m-dimensional)
            M = [[0.0] * m for _ in range(m)]
            b = [0.0] * m
            
            for (idx, t) in occurrences:
                # Get basis vector and Acoeff column
                basis_vec = self.Bbasis[idx]
                A_col = [self.Acoeff[i][idx] for i in range(m)]
                
                # Compute ||A_col||² = A_col · A_col
                a_norm_sq = sum(a*a for a in A_col)
                
                # Correctly compute M = Σ(||A_col||² * (basis_vec ⊗ basis_vec))
                for i in range(m):
                    for j in range(m):
                        M[i][j] += a_norm_sq * basis_vec[i] * basis_vec[j]
                
                # Compute b contribution: <A_col, t> * basis_vec
                a_dot_t = sum(A_col[i] * t[i] for i in range(m))
                for j in range(m):
                    b[j] += a_dot_t * basis_vec[j]
            
            # Solve linear system: Mx = b
            try:
                M_inv = self._invert(M)
                self.M[tok] = self._mat_vec(M_inv, b)
            except:  # Fallback to identity on singular matrix
                pass                

    # ---- training loop ----
    def train(self, seqs, t_list, max_iters=10, tol=1e-8, print_every=10):
        """
        Train the model using alternating least squares updates.
        
        Update sequence per iteration:
        1. Update coefficient matrix Acoeff
        2. Update basis matrix Bbasis
        3. Update token embeddings M
        
        Args:
            seqs: List of training sequences
            t_list: List of target vectors corresponding to sequences
            max_iters: Maximum number of iterations
            tol: Convergence tolerance
            
        Returns:
            list: History of deviation values during training
        """
        D_prev = float('inf')
        history = []
        for it in range(max_iters):
            self.update_Acoeff(seqs, t_list)
            self.update_Bbasis(seqs, t_list)  # Added basis matrix update
            self.update_M(seqs, t_list)
            D = self.deviation(seqs, t_list)
            history.append(D)
            if it % print_every == 0 or it == max_iters - 1:
                print(f"Iter {it:2d}: D = {D:.6e}")
            if D >= D_prev - tol:
                print("Converged.")
                break
            D_prev = D
        self.mean_L = int(sum(len(self.extract_tokens(seq)) for seq in seqs) / len(seqs))
        self.mean_t = [0.0] * self.m
        for t in t_list:
            for i in range(self.m):
                self.mean_t[i] += t[i]
        self.mean_t = [x / len(t_list) for x in self.mean_t]        
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

    def _compute_token_descriptor(self, tok):
        """
        Compute the descriptor vector for a token.
        Args:
            tok: Token string
        Returns:
            list: m-dimensional descriptor vector
        """
        x = self.M[tok]
        # Compute z = Bbasis * x (L-dimensional vector)
        z = [0.0] * self.L
        for j in range(self.L):
            for i in range(self.m):
                z[j] += self.Bbasis[j][i] * x[i]
        # Compute Nk_vec = Acoeff * z (m-dimensional vector)
        Nk_vec = [0.0] * self.m
        for i in range(self.m):
            for j in range(self.L):
                Nk_vec[i] += self.Acoeff[i][j] * z[j]
        return Nk_vec

    def _get_token_descriptor(self, tok):
        """
        Retrieve cached descriptor vector for a token (compute if not cached).
        Args:
            tok: Token string
        Returns:
            list: m-dimensional descriptor vector
        """
        if not hasattr(self, '_tok_vec_cache'):
            self._tok_vec_cache = {}
        if tok not in self._tok_vec_cache:
            self._tok_vec_cache[tok] = self._compute_token_descriptor(tok)
        return self._tok_vec_cache[tok]
    
    def reconstruct(self):
        """
        Reconstruct a representative sequence by selecting tokens that minimize 
        the squared error between their descriptor vector and the mean target vector.
        Returns:
            str: Reconstructed sequence truncated to average token count length
        """
        assert self.trained, "Model must be trained before reconstruction."
        # Precompute descriptor vectors for all tokens
        tok_vecs = {}
        for tok in self.tokens:
            tok_vecs[tok] = self._get_token_descriptor(tok)
        
        tokens = []
        for k in range(self.mean_L):  # For each token position
            best_tok = None
            min_error = float('inf')
            # Find token minimizing error to mean target
            for tok in self.tokens:
                vec = tok_vecs[tok]
                error = sum((vec[i] - self.mean_t[i]) ** 2 for i in range(self.m))
                if error < min_error:
                    min_error = error
                    best_tok = tok
            # Remove padding underscores
            clean_tok = best_tok.rstrip('_')
            tokens.append(clean_tok)
        
        full_seq = ''.join(tokens)
        return full_seq[:self.mean_L]  # Truncate to average token count

    def generate(self, L, tau=0.0):
        """
        Generate a sequence of length L using descriptor vectors.
        Args:
            L (int): Target sequence length
            tau (float): Temperature (≥0); 0=deterministic, >0=stochastic
        Returns:
            str: Generated sequence truncated to length L
        """
        assert self.trained, "Model must be trained before generation."
        if tau < 0:
            raise ValueError("Temperature must be non-negative.")
        
        # Filter valid full-length tokens
        valid_tokens = [tok for tok in self.tokens 
                       if len(tok) == self.rank and '_' not in tok]
        if not valid_tokens:
            raise ValueError("No valid full-length tokens found.")
        
        # Precompute descriptor vectors for valid tokens
        tok_vecs = {}
        for tok in valid_tokens:
            tok_vecs[tok] = self._get_token_descriptor(tok)
        
        num_blocks = (L + self.rank - 1) // self.rank  # Blocks needed
        generated_tokens = []
        
        for k in range(num_blocks):
            scores = {}
            # Calculate score = -error for each token
            for tok in valid_tokens:
                vec = tok_vecs[tok]
                error = sum((vec[i] - self.mean_t[i]) ** 2 for i in range(self.m))
                scores[tok] = -error
            
            if tau == 0.0:  # Deterministic selection
                best_tok = max(scores.keys(), key=lambda t: scores[t])
                generated_tokens.append(best_tok)
            else:  # Stochastic selection
                token_list = list(scores.keys())
                exp_scores = [math.exp(scores[t] / tau) for t in token_list]
                sum_exp = sum(exp_scores)
                probs = [es / sum_exp for es in exp_scores]
                chosen_tok = random.choices(token_list, weights=probs, k=1)[0]
                generated_tokens.append(chosen_tok)
        
        full_seq = ''.join(generated_tokens)
        return full_seq[:L]  # Truncate to target length

    # ---- feature extraction ----
    def dd_features(self, seq): 
        feats = []
        # flatten Acoeff
        for row in self.Acoeff:
            feats.extend(row)
        # flatten M
        for tok in self.tokens:
            feats.extend(self.M[tok])
##        # basis-weighted frequencies
##        toks = self.extract_tokens(seq)
##        for tok in self.tokens:
##            for i in range(self.m):
##                s = 0.0
##                for j, tk in enumerate(toks[:self.L]):
##                    if tk == tok:
##                        s += self.Bbasis[j][i]
##                feats.append(s)  
        return feats

    # ---- show state ----
    def show(self):
        print("DualDescriptorRN status:")
        print(f" L={self.L}, m={self.m}, rank={self.rank}, mode={self.mode}")
        print(" Sample Acoeff[0][:5]:", self.Acoeff[0][:5])
        print(" Sample Bbasis[0][:5]:", self.Bbasis[0][:5])
        tok0 = self.tokens[0]
        print(" Sample M first token:", tok0, self.M[tok0][:5])
    

    def save(self, filename):
        """
        Save the current state of the DualDescriptorRN object to a binary file.
        """
        with open(filename, 'wb') as f:
            pickle.dump(self.__dict__, f)
        print(f"Model saved to {filename}")

    @classmethod
    def load(cls, filename):
        """
        Load a saved DualDescriptorRN model from a binary file.
        """
        with open(filename, 'rb') as f:
            state = pickle.load(f)
        
        # Create new instance without calling __init__
        obj = cls.__new__(cls)
        # Restore saved state
        obj.__dict__.update(state)
        print(f"Model loaded from {filename}")
        return obj

# === Example Usage ===
if __name__ == "__main__":

    from statistics import correlation
    
    #random.seed(0)
    charset = ['A','C','G','T']
    dd = DualDescriptorRN(charset, rank=1, vec_dim=2, bas_dim=150, mode='nonlinear', user_step=1)

    # generate 10 sequences of length 100–200 and random targets
    seqs, t_list = [], []
    for _ in range(10):
        L = random.randint(100, 200)
        seq = ''.join(random.choices(charset, k=L))
        seqs.append(seq)
        t_list.append([random.uniform(-1,1) for _ in range(dd.m)])

    dd.train(seqs, t_list, max_iters=200)
    dd.show()

    # Predict target vector for first sequence
    aseq = seqs[0]
    t_pred = dd.predict_t(aseq)
    print(f"\nPredicted t for first sequence: {[round(x, 4) for x in t_pred]}")    
    
    # Calculate flattened correlation between predicted and actual targets
    pred_t_list = [dd.predict_t(seq) for seq in seqs]
    
    # Predictions and actuals for correlation calculation
    corr_sum = 0.0
    for i in range(dd.m):
        actu_t = [t_vec[i] for t_vec in t_list]
        pred_t = [t_vec[i] for t_vec in pred_t_list]
        corr = correlation(actu_t, pred_t)
        print(f"Prediction correlation: {corr:.4f}")
        corr_sum += corr
    corr_avg = corr_sum / dd.m
    print(f"Average correlation: {corr_avg:.4f}")   

    feats = dd.dd_features(seqs[0])
    print("\nFeature vector length:", len(feats))
    print("First 10 features:", feats[:10])

    # Demonstrate prediction
    print("\n=== Prediction Demo ===")
    test_seq = "ACGTACGT"  # Sample test sequence
    true_t = [0.42, -0.17, 0.89]  # Example true value
    
    # Make prediction
    pred_t = dd.predict_t(test_seq)
    print(f"Test Sequence: {test_seq}")
    print(f"Predicted t: {[f'{x:.4f}' for x in pred_t]}")
    print(f"True t: {true_t}")    

    # Show prediction error
    error = sum((p - t)**2 for p, t in zip(pred_t, true_t))
    print(f"Prediction MSE: {error:.6f}")

    # Reconstruct representative sequence
    repr_seq = dd.reconstruct()
    print(f"\nReconstructed representative sequence: {repr_seq}")
    
    # Generate sequences
    print("\nGenerated sequences:")
    seq_det = dd.generate(L=100, tau=0.0)
    print(f"Deterministic (tau=0): {seq_det[:50]}...")


