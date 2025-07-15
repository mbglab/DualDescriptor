# Copyright (C) 2005-2025, Bin-Guang Ma (mbg@mail.hzau.edu.cn). All rights reserved.
# The Hierarchical Numeric Dual Descriptor class (Random AB matrix form) for Vector Sequences
# Modified to support hierarchical structure with multiple layers
# Author: Bin-Guang Ma; Date: 2025-7-7

import math
import random
import pickle

class HierDDab:
    """
    Hierarchical Numeric Dual Descriptor for vector sequences with:
      - input_dim: dimension of input vectors
      - model_dims: list of output dimensions for each layer
      - basis_dims: list of basis dimensions for each layer
    Each layer contains:
      - Matrix M ∈ R^{m_i×m_{i-1}} for linear transformation
      - Matrix Acoeff ∈ R^{m_i×L_i} of coefficients
      - Matrix Bbasis ∈ R^{L_i×m_i} of basis functions
    """
    def __init__(self, input_dim=4, model_dims=[4], basis_dims=[50]):
        """
        Initialize the hierarchical Dual Descriptor.
        
        Args:
            input_dim (int): Input vector dimension
            model_dims (list): List of output dimensions for each layer
            basis_dims (list): List of basis dimensions for each layer
        """
        self.input_dim = input_dim
        self.model_dims = model_dims
        self.basis_dims = basis_dims
        self.num_layers = len(model_dims)
        self.trained = False
        
        # Validate dimensions
        if len(basis_dims) != self.num_layers:
            raise ValueError("basis_dims length must match model_dims length")
        
        self.layers = []
        
        # Initialize each layer
        for i, out_dim in enumerate(model_dims):
            # Determine input dimension for this layer
            if i == 0:
                in_dim = input_dim
            else:
                in_dim = model_dims[i-1]
                
            L_i = basis_dims[i]  # Basis dimension for this layer
            
            # Initialize transformation matrix M (out_dim × in_dim)
            M = [[random.uniform(-0.1, 0.1) for _ in range(in_dim)]
                 for _ in range(out_dim)]
            
            # Initialize coefficient matrix Acoeff (out_dim × L_i)
            Acoeff = [[random.uniform(-0.1, 0.1) for _ in range(L_i)]
                       for _ in range(out_dim)]
            
            # Initialize basis matrix Bbasis (L_i × out_dim)
            Bbasis = [[random.uniform(-0.1, 0.1) for _ in range(out_dim)]
                       for _ in range(L_i)]
            
            self.layers.append({
                'M': M,
                'Acoeff': Acoeff,
                'Bbasis': Bbasis
            })

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
    
    def _vec_sub(self, u, v):
        """Vector subtraction"""
        return [u[i] - v[i] for i in range(len(u))]
    
    def _dot(self, u, v):
        """Dot product of two vectors"""
        return sum(u[i] * v[i] for i in range(len(u)))

    # ---- describe sequence ----
    def describe(self, vec_seq):
        """Compute descriptor vectors for each position in all layers"""
        current = vec_seq
        # Process through each layer
        for layer in self.layers:
            M = layer['M']
            Acoeff = layer['Acoeff']
            Bbasis = layer['Bbasis']
            L_i = len(Bbasis)  # Basis dimension for this layer
            out_dim = len(Acoeff)  # Output dimension for this layer
            
            new_current = []
            for k, vec in enumerate(current):
                # Apply transformation: x' = M * vec
                transformed_vec = self._mat_vec(M, vec)
                
                # Compute basis index for this position
                j = k % L_i
                
                # Compute scalar: B_j • x'
                scalar = self._dot(Bbasis[j], transformed_vec)
                
                # Get A_j vector
                A_j = [Acoeff[i][j] for i in range(out_dim)]
                
                # Compute Nk = scalar * A_j
                Nk = [a * scalar for a in A_j]
                new_current.append(Nk)
            
            current = new_current  # Output becomes input for next layer
        
        return current

    # ---- compute deviation ----
    def deviation(self, seqs, t_list):
        """Compute mean squared error between descriptors and targets"""
        tot = 0.0
        cnt = 0
        for seq, t in zip(seqs, t_list):
            # Get final layer outputs
            outputs = self.describe(seq)
            for Nk in outputs:
                # t dimension should match final layer output dimension
                for i in range(len(Nk)):
                    diff = Nk[i] - t[i]
                    tot += diff * diff
                cnt += 1
        return tot / cnt if cnt else 0.0

    def grad_train(self, seqs, t_list, max_iters=1000, tol=1e-8, 
               learning_rate=0.01, continued=False, 
               decay_rate=1.0, print_every=10, epsilon=1e-8):
        """
        Train using gradient descent with backpropagation through layers.
        Modified to use per-layer output dimension for normalization.
        Added Layer Normalization after M transformation for better gradient scaling.
        
        Args:
            seqs: List of input sequences
            t_list: List of target vectors
            max_iters: Maximum training iterations
            tol: Convergence tolerance
            learning_rate: Initial learning rate
            continued: Continue training from current state
            decay_rate: Learning rate decay factor
            print_every: Print progress every N iterations
            epsilon: Small constant for numerical stability in normalization
        """        
        
        if not continued:
            # Reinitialize parameters
            self.__init__(self.input_dim, self.model_dims, self.basis_dims)
        
        total_positions = sum(len(seq) for seq in seqs)
        if total_positions == 0:
            raise ValueError("No valid vectors in sequences")
        
        history = []
        D_prev = float('inf')
        current_lr = learning_rate
        
        for it in range(max_iters):
            # Save current parameters (deep copy) for potential rollback
            current_params = []
            for layer in self.layers:
                # Create deep copies of all parameter matrices
                Acoeff_copy = [row[:] for row in layer['Acoeff']]
                Bbasis_copy = [row[:] for row in layer['Bbasis']]
                M_copy = [row[:] for row in layer['M']]
                current_params.append({
                    'Acoeff': Acoeff_copy,
                    'Bbasis': Bbasis_copy,
                    'M': M_copy
                })
            
            # Initialize gradients for all layers
            grad_A_list = []
            grad_B_list = []
            grad_M_list = []
            for layer in self.layers:
                out_dim = len(layer['Acoeff'])
                L_i = len(layer['Bbasis'])
                in_dim = len(layer['M'][0])
                
                # Gradient for Acoeff matrix
                grad_A = [[0.0] * L_i for _ in range(out_dim)]
                # Gradient for Bbasis matrix
                grad_B = [[0.0] * out_dim for _ in range(L_i)]
                # Gradient for M matrix
                grad_M = [[0.0] * in_dim for _ in range(out_dim)]
                
                grad_A_list.append(grad_A)
                grad_B_list.append(grad_B)
                grad_M_list.append(grad_M)
            
            total_loss = 0.0
            
            # Process each sequence
            for seq, t in zip(seqs, t_list):
                # Forward pass: store intermediate results for each layer
                intermediates = []
                current = seq
                
                # Process through each layer and save intermediate results
                for layer_idx, layer in enumerate(self.layers):
                    M = layer['M']
                    Acoeff = layer['Acoeff']
                    Bbasis = layer['Bbasis']
                    L_i = len(Bbasis)  # Basis dimension for this layer
                    out_dim = len(Acoeff)  # Output dimension for this layer
                    
                    layer_intermediate = []
                    new_current = []
                    for k, vec in enumerate(current):
                        # Apply transformation
                        transformed_vec = self._mat_vec(M, vec)
                        
                        # NEW: Apply Layer Normalization to transformed vector
                        # Calculate mean and variance
                        d = len(transformed_vec)  # Feature dimension
                        vec_mean = sum(transformed_vec) / d
                        vec_var = sum((x - vec_mean) ** 2 for x in transformed_vec) / d
                        
                        # Normalize with epsilon for numerical stability
                        normalized_vec = [(x - vec_mean) / math.sqrt(vec_var + epsilon) 
                                        for x in transformed_vec]
                        
                        # Compute basis index
                        j = k % L_i
                        
                        # Compute scalar using normalized vector
                        scalar = self._dot(Bbasis[j], normalized_vec)
                        
                        # Get A_j vector
                        A_j = [Acoeff[i][j] for i in range(out_dim)]
                        
                        # Compute output vector
                        Nk = [a * scalar for a in A_j]
                        
                        # Save intermediate values including normalization statistics
                        layer_intermediate.append({
                            'vec': vec,
                            'transformed_vec': transformed_vec,  # Original vector before normalization
                            'normalized_vec': normalized_vec,    # Vector after normalization
                            'j': j,
                            'scalar': scalar,
                            'A_j': A_j,
                            'mean': vec_mean,  # Saved for backprop
                            'var': vec_var     # Saved for backprop
                        })
                        new_current.append(Nk)
                    
                    intermediates.append(layer_intermediate)
                    current = new_current
                
                # Backward pass: calculate gradients for each position
                for pos in range(len(seq)):
                    # Start with error from final layer output
                    final_output = intermediates[-1][pos]
                    Nk = [a * final_output['scalar'] for a in final_output['A_j']]
                    
                    # Get final layer output dimension
                    final_out_dim = self.model_dims[-1]
                    
                    # MODIFIED: Use current layer's output dimension for normalization
                    dNk = [2 * (Nk[i] - t[i]) / final_out_dim for i in range(final_out_dim)]
                    
                    # Propagate error backward through layers
                    for l_idx in range(self.num_layers-1, -1, -1):
                        layer = self.layers[l_idx]
                        inter = intermediates[l_idx][pos]
                        M = layer['M']
                        Acoeff = layer['Acoeff']
                        Bbasis = layer['Bbasis']
                        L_i = len(Bbasis)
                        out_dim = len(Acoeff)  # Current layer output dimension
                        in_dim = len(M[0])
                        
                        j = inter['j']
                        transformed_vec = inter['transformed_vec']  # Original vector before normalization
                        normalized_vec = inter['normalized_vec']    # Vector after normalization
                        scalar = inter['scalar']
                        A_j = inter['A_j']
                        vec = inter['vec']
                        vec_mean = inter['mean']
                        vec_var = inter['var']
                        
                        # Compute dL/dscalar
                        dscalar = sum(dNk[i] * A_j[i] for i in range(len(dNk)))
                        
                        # Update gradients for current layer
                        # Gradient for Acoeff
                        for i in range(len(Acoeff)):
                            grad_A_list[l_idx][i][j] += dNk[i] * scalar
                        
                        # Gradient for Bbasis
                        for d in range(len(Bbasis[j])):
                            # Uses normalized vector for gradient calculation
                            grad_B_list[l_idx][j][d] += dscalar * normalized_vec[d]
                        
                        # NEW: Backpropagation through normalization layer
                        # Gradient w.r.t normalized vector
                        d_normalized = [dscalar * Bbasis[j][d] for d in range(len(Bbasis[j]))]
                        
                        # Gradient w.r.t original transformed vector (before normalization)
                        d_transformed = [0.0] * len(transformed_vec)
                        n = len(transformed_vec)
                        
                        # Compute intermediate gradients
                        d_var = 0.0
                        d_mean = 0.0
                        
                        # Gradient w.r.t variance
                        for i in range(n):
                            d_var += d_normalized[i] * (transformed_vec[i] - vec_mean) * \
                                     (-0.5) * (vec_var + epsilon) ** (-1.5)
                        
                        # Gradient w.r.t mean
                        for i in range(n):
                            # From normalized value
                            d_mean += d_normalized[i] * (-1.0 / math.sqrt(vec_var + epsilon))
                            # From variance
                            d_mean += d_var * (2.0 / n) * (transformed_vec[i] - vec_mean) * (-1.0)
                        
                        # Gradient w.r.t original transformed vector
                        for i in range(n):
                            term1 = d_normalized[i] / math.sqrt(vec_var + epsilon)
                            term2 = d_mean * (1.0 / n)
                            term3 = d_var * (2.0 / n) * (transformed_vec[i] - vec_mean)
                            d_transformed[i] = term1 + term2 + term3
                        
                        # Gradient for M (through transformed_vec)
                        for d in range(len(d_transformed)):
                            for e in range(in_dim):
                                grad_M_list[l_idx][d][e] += d_transformed[d] * vec[e]
                        
                        # Propagate gradient to previous layer (if not first layer)
                        if l_idx > 0:
                            # Compute dL/dvec for previous layer
                            d_prev = [0.0] * in_dim
                            for e in range(in_dim):
                                for d in range(len(d_transformed)):
                                    d_prev[e] += d_transformed[d] * M[d][e]
                            dNk = d_prev
                        else:
                            dNk = None  # No need to propagate further
                    
                    # MODIFIED: Use current layer's output dimension for loss normalization
                    pos_loss = sum((Nk[i] - t[i])**2 for i in range(final_out_dim)) / final_out_dim
                    total_loss += pos_loss
            
            # Update parameters with gradients
            for l_idx in range(self.num_layers):
                layer = self.layers[l_idx]
                M = layer['M']
                Acoeff = layer['Acoeff']
                Bbasis = layer['Bbasis']
                
                # Update Acoeff matrix
                for i in range(len(Acoeff)):
                    for j in range(len(Acoeff[i])):
                        Acoeff[i][j] -= current_lr * grad_A_list[l_idx][i][j]
                
                # Update Bbasis matrix
                for j in range(len(Bbasis)):
                    for d in range(len(Bbasis[j])):
                        Bbasis[j][d] -= current_lr * grad_B_list[l_idx][j][d]
                
                # Update M matrix
                for d in range(len(M)):
                    for e in range(len(M[d])):
                        M[d][e] -= current_lr * grad_M_list[l_idx][d][e]
            
            # Calculate average loss
            D = total_loss / total_positions
            history.append(D)
            
            # Print progress
            if it % print_every == 0 or it == max_iters - 1:
                print(f"Iter {it:3d}: Loss = {D:.36f}, LR = {current_lr:.6f}")
            
            # Check convergence: if loss increases (D >= D_prev - tol)
            if D >= D_prev - tol:
                # Roll back parameters to state before update
                for l_idx, layer in enumerate(self.layers):
                    layer['Acoeff'] = current_params[l_idx]['Acoeff']
                    layer['Bbasis'] = current_params[l_idx]['Bbasis']
                    layer['M'] = current_params[l_idx]['M']
                # Replace last history entry with previous loss value
                history[-1] = D_prev
                print(f"Converged after {it+1} iterations (rolled back to previous parameters)")
                break
                
            # Update previous loss and decay learning rate
            D_prev = D 
            current_lr *= decay_rate  # Decay learning rate        
        self.trained = True
        return history

    def predict_t(self, vec_seq):
        """Predict target vector as average of all final layer outputs"""
        # Get final layer outputs
        outputs = self.describe(vec_seq)
        if not outputs:
            return [0.0] * self.model_dims[-1]
        
        # Sum all output vectors
        sum_t = [0.0] * self.model_dims[-1]
        for Nk in outputs:
            for i in range(len(Nk)):
                sum_t[i] += Nk[i]        
        # Return average
        N = len(outputs)
        return [x / N for x in sum_t]

    # ---- show state ----
    def show(self, what=None, first_num=5):
        """Display hierarchical model state information"""
        if what is None:
            what = ['params', 'stats']
        elif isinstance(what, str):
            what = ['params', 'stats'] if what == 'all' else [what]
        
        print("Hierarchical HierDDab Model Status:")
        print("=" * 50)
        
        # Configuration parameters
        if 'params' in what:
            print("[Configuration Parameters]")
            print(f"  Input dim       : {self.input_dim}")
            print(f"  Layer dims      : {self.model_dims}")
            print(f"  Basis dims      : {self.basis_dims}")
            print(f"  Number of layers: {self.num_layers}")
            print(f"  Trained         : {self.trained}")
            print("=" * 50)
        
        # Layer details
        if 'layers' in what or 'all' in what:
            for l_idx, layer in enumerate(self.layers):
                print(f"\nLayer {l_idx}:")
                in_dim = self.input_dim if l_idx == 0 else self.model_dims[l_idx-1]
                out_dim = self.model_dims[l_idx]
                L_i = self.basis_dims[l_idx]
                
                # Show M matrix sample
                M = layer['M']
                print(f"  M matrix ({len(M)}x{len(M[0])}):")
                for i in range(min(first_num, len(M))):
                    row = M[i][:min(first_num, len(M[0]))]
                    print(f"    Row {i}: [{', '.join(f'{x:.4f}' for x in row)}" + 
                          (", ...]" if len(M[0]) > first_num else "]"))
                
                # Show Acoeff matrix sample
                Acoeff = layer['Acoeff']
                print(f"  Acoeff matrix ({len(Acoeff)}x{len(Acoeff[0])}):")
                for i in range(min(first_num, len(Acoeff))):
                    row = Acoeff[i][:min(first_num, len(Acoeff[0]))]
                    print(f"    Row {i}: [{', '.join(f'{x:.4f}' for x in row)}" + 
                          (", ...]" if len(Acoeff[0]) > first_num else "]"))
                
                # Show Bbasis matrix sample
                Bbasis = layer['Bbasis']
                print(f"  Bbasis matrix ({len(Bbasis)}x{len(Bbasis[0])}):")
                for i in range(min(first_num, len(Bbasis))):
                    row = Bbasis[i][:min(first_num, len(Bbasis[0]))]
                    print(f"    Row {i}: [{', '.join(f'{x:.4f}' for x in row)}" + 
                          (", ...]" if len(Bbasis[0]) > first_num else "]"))       

            print("=" * 50)

    def count_parameters(self):
        """
        Calculate and print the number of learnable parameters.
        """
        total_params = 0
        print("Model Parameter Counts:")
        
        for l_idx, layer in enumerate(self.layers):
            M = layer['M']
            Acoeff = layer['Acoeff']
            Bbasis = layer['Bbasis']
            
            in_dim = self.input_dim if l_idx == 0 else self.model_dims[l_idx-1]
            out_dim = self.model_dims[l_idx]
            L_i = self.basis_dims[l_idx]
            
            m_params = len(M) * len(M[0])
            a_params = len(Acoeff) * len(Acoeff[0])
            b_params = len(Bbasis) * len(Bbasis[0])
            layer_params = m_params + a_params + b_params
            total_params += layer_params
            
            print(f"  Layer {l_idx} (in: {in_dim}, out: {out_dim}, L: {L_i}):")
            print(f"    M: {len(M)}×{len(M[0])} = {m_params} params")
            print(f"    Acoeff: {len(Acoeff)}×{len(Acoeff[0])} = {a_params} params")
            print(f"    Bbasis: {len(Bbasis)}×{len(Bbasis[0])} = {b_params} params")
            print(f"    Layer total: {layer_params}")
        
        print(f"Total parameters: {total_params}")        
        return total_params

    def save(self, filename):
        """Save hierarchical model to file"""
        with open(filename, 'wb') as f:
            pickle.dump(self.__dict__, f)
        print(f"Model saved to {filename}")

    @classmethod
    def load(cls, filename):
        """Load hierarchical model from file"""
        with open(filename, 'rb') as f:
            state = pickle.load(f)
        obj = cls.__new__(cls)
        obj.__dict__.update(state)
        print(f"Model loaded from {filename}")
        return obj

# === Example Usage ===
if __name__ == "__main__":

    from statistics import correlation, mean
    
    # Set random seed for reproducibility
    #random.seed(222)
    
    # Hierarchical configuration
    input_dim = 10     # Input vector dimension
    model_dims = [5, 2]  # Output dimensions for each layer
    basis_dims = [150, 150]   # Basis dimensions for each layer
    seq_count = 10    # Number of training sequences
    min_len = 100     # Minimum sequence length
    max_len = 200     # Maximum sequence length
    
    # Generate training data
    print("Generating training data...")
    print(f"Input dimension: {input_dim}, Layer dims: {model_dims}, Basis dims: {basis_dims}")
    seqs = []  # List of sequences
    t_list = []  # List of target vectors (dimension = last layer output)
    
    for i in range(seq_count):
        # Random sequence length
        length = random.randint(min_len, max_len)
        # Generate vector sequence: list of n-dimensional vectors
        sequence = []
        for _ in range(length):
            vector = [random.uniform(-1, 1) for _ in range(input_dim)]
            sequence.append(vector)
        seqs.append(sequence)
        
        # Generate random target vector (dimension = last layer output)
        target = [random.uniform(-1, 1) for _ in range(model_dims[-1])]
        t_list.append(target)
        print(f"Sequence {i+1}: length={length}, target={[round(t,2) for t in target]}")
    
    # Create and train the hierarchical model
    print("\nCreating Hierarchical HierDDab...")
    hdd = HierDDab(
        input_dim=input_dim,
        model_dims=model_dims,
        basis_dims=basis_dims
    )
    
    # Show initial model structure
    print("\nInitial model structure:")
    hdd.show('params')
    hdd.count_parameters()
    
    # Train with gradient descent
    print("\nTraining with Gradient Descent...")
    gd_history = hdd.grad_train(
        seqs,
        t_list,
        learning_rate=0.02,
        max_iters=300,
        tol=1e-88,
        decay_rate=0.98,
        print_every=10
    )
    
    # Evaluate predictions
    pred_t_list = [hdd.predict_t(seq) for seq in seqs]
    print("\nPrediction correlations per dimension:")
    corrs = []
    for d in range(model_dims[-1]):
        actuals = [t[d] for t in t_list]
        preds = [p[d] for p in pred_t_list]
        corr = correlation(actuals, preds)
        corrs.append(corr)
        print(f"  Dim {d}: correlation = {corr:.4f}")
    
    print(f"Average correlation: {mean(corrs):.4f}")
    
    # Show final model
    print("\nTrained model structure:")
    hdd.show('all')
    hdd.count_parameters()
    
    # Save and load model
    print("\nTesting model persistence...")
    hdd.save("hierarchical_vector_model.pkl")
    loaded = HierDDab.load("hierarchical_vector_model.pkl")
    print("Loaded model prediction for first sequence:")
    pred = loaded.predict_t(seqs[0])
    print(f"  Predicted target: {[round(p, 4) for p in pred]}")
    
    print("\n=== Hierarchical Vector Sequence Processing Demo Completed ===")
