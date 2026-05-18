# Copyright (C) 2005-2025, Bin-Guang Ma (mbg@mail.hzau.edu.cn). All rights reserved.
# The Dual Descriptor Scalar class
# Author: Bin-Guang Ma; Date: 2025-5-25

# The program is provided as it is and without warranty of any kind,
# either expressed or implied.

import math
import random
import itertools

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
            self.basis = [
                (lambda p: (lambda k: math.cos(2*math.pi*k/p)))(period)
                for period in range(2, 2 + self.o)
            ]
            
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
        x = (sum P(k)*t_seq) / (sum P(k)^2
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

# ===== Example Usage =====
if __name__ == "__main__":
    from statistics import correlation    
    # initialize charset
    charset = ['A','C','G','T']
    
    # define user basis functions tan() instead of the default cos()
    #user_bases = None
    user_bases = [
                (lambda p: (lambda k: math.sin(2*math.pi*k/p)))(period)
                for period in range(2, 2 + 7)]
    
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


    
    
    
    
