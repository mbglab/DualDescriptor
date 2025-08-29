# Copyright (C) 2005-2025, Bin-Guang Ma (mbg@mail.hzau.edu.cn); SPDX-License-Identifier: MIT
# The Hierarchical Numeric Dual Descriptor (P Matrix form) implemented with pure Python
# Added layer normalization and residual connections
# This program is for the demonstration of methodology and not fully refined.
# Author: Bin-Guang Ma (assisted by DeepSeek); Date: 2025-6-24

import math
import random
import pickle

class HierDDpm:
    """
    Hierarchical Numeric Dual Descriptor for vector sequences with:
      - Multiple layers with linear transformation M and matrix P
      - Each layer: P ∈ R^{m×m} of basis coefficients, M ∈ R^{m×n}
      - Periods: period[i,j] = i*m + j + 2
      - Basis function: phi_{i,j}(k) = cos(2π * k / period[i,j])
      - Layer normalization after linear transformation
      - Residual connections with configurable modes
    """
    def __init__(self, input_dim=2, model_dims=[2], use_residual_list=None):
        """
        Initialize hierarchical HierDDpm
        
        Args:
            input_dim (int): Input vector dimension
            model_dims (list): Output dimensions for each layer
            use_residual_list (list): Residual connection type for each layer. 
                Options: 'separate', 'shared', None. Defaults to ['separate']*n_layers.
        """
        self.input_dim = input_dim
        self.model_dims = model_dims
        self.num_layers = len(model_dims)
        self.trained = False
        
        # Handle residual connection defaults
        if use_residual_list is None:
            self.use_residual_list = ['separate'] * self.num_layers
        else:
            self.use_residual_list = use_residual_list
            
        # Initialize layers
        self.layers = []
        for l, out_dim in enumerate(model_dims):
            # Determine input dimension for this layer
            if l == 0:
                in_dim = input_dim
            else:
                in_dim = model_dims[l-1]
                
            # Get residual mode for this layer
            use_residual = self.use_residual_list[l]
                
            # Initialize M matrix (out_dim x in_dim)
            M = [[random.uniform(-0.5, 0.5) for _ in range(in_dim)] 
                 for _ in range(out_dim)]
            
            # Initialize P matrix (out_dim x out_dim)
            P = [[random.uniform(-0.1, 0.1) for _ in range(out_dim)]
                  for _ in range(out_dim)]
            
            # Precompute periods matrix
            periods = [[i * out_dim + j + 2 for j in range(out_dim)]
                       for i in range(out_dim)]
            
            # Initialize layer normalization parameters
            gamma = [1.0] * out_dim  # Scaling factors (init to 1)
            beta = [0.0] * out_dim   # Bias terms (init to 0)
            
            # Initialize residual projection matrix if needed
            M_res = None
            if use_residual == 'separate' and in_dim != out_dim:
                M_res = [[random.uniform(-0.1, 0.1) for _ in range(in_dim)] 
                         for _ in range(out_dim)]
            
            self.layers.append({
                'M': M,
                'P': P,
                'periods': periods,
                'gamma': gamma,
                'beta': beta,
                'M_res': M_res,
                'use_residual': use_residual,
                'eps': 1e-5  # Small constant for numerical stability
            })

    # ---- Helper functions ----
    def _mat_vec(self, M, v):
        """Matrix-vector multiplication"""
        return [sum(M[i][j]*v[j] for j in range(len(v))) for i in range(len(M))]
    
    def _transpose(self, M):
        """Matrix transpose"""
        return [list(col) for col in zip(*M)]
    
    def _vec_sub(self, u, v):
        """Vector subtraction"""
        return [u_i - v_i for u_i, v_i in zip(u, v)]
    
    def _dot(self, u, v):
        """Vector dot product"""
        return sum(u_i * v_i for u_i, v_i in zip(u, v))
    
    def _vec_add(self, u, v):
        """Vector addition"""
        return [u_i + v_i for u_i, v_i in zip(u, v)]
    
    def _vec_scale(self, v, s):
        """Vector scaling"""
        return [x * s for x in v]
    
    def _layer_norm(self, x, gamma, beta, eps):
        """
        Apply layer normalization to vector x
        y = gamma * (x - mean) / sqrt(var + eps) + beta
        """
        n = len(x)
        mean = sum(x) / n
        var = sum((xi - mean)**2 for xi in x) / n
        std = math.sqrt(var + eps)
        x_norm = [(xi - mean) / std for xi in x]
        return [gamma[i] * x_norm[i] + beta[i] for i in range(n)]

    def describe(self, seq):
        """Compute N(k) vectors for each position and vector in sequence"""
        current = seq
        for layer in self.layers:
            new_seq = []
            for k, vec in enumerate(current):
                # Linear transformation: x_linear = M * vec
                x_linear = self._mat_vec(layer['M'], vec)
                
                # Apply layer normalization
                x_norm = self._layer_norm(
                    x_linear, 
                    layer['gamma'], 
                    layer['beta'], 
                    layer['eps']
                )
                
                out_dim = len(layer['P'])
                Nk = [0.0] * out_dim
                
                # Apply position-dependent transformation
                for i in range(out_dim):
                    for j in range(out_dim):
                        period = layer['periods'][i][j]
                        phi = math.cos(2 * math.pi * k / period)
                        Nk[i] += layer['P'][i][j] * x_norm[j] * phi
                
                # Apply residual connection
                if layer['use_residual'] == 'separate':
                    if layer['M_res'] is not None:
                        residual = self._mat_vec(layer['M_res'], vec)
                    else:
                        residual = vec  # Identity mapping when dims match
                elif layer['use_residual'] == 'shared':
                    residual = x_linear  # Use linear output (pre-norm)
                else:  # None or other
                    if len(vec) == out_dim:
                        residual = vec  # Identity mapping
                    else:
                        residual = [0.0] * out_dim  # Zero residual
                
                # Add residual
                Nk = self._vec_add(Nk, residual)
                new_seq.append(Nk)
            current = new_seq
        return current

    def deviation(self, seqs, t_list):
        """
        Compute mean squared deviation D across sequences:
        D = average over all positions of (N(k)-t_seq)^2
        """
        total = 0.0
        count = 0
        for seq, t in zip(seqs, t_list):
            outputs = self.describe(seq)
            for vec in outputs:
                err = self._vec_sub(vec, t)
                total += self._dot(err, err)
                count += 1
        return total / count if count else 0.0

    def grad_train(self, seqs, t_list, max_iters=1000, tol=1e-8, 
                  learning_rate=0.01, continued=False, decay_rate=1.0, 
                  print_every=10):
        """
        Train using gradient descent with backpropagation through layers
        Supports layer normalization and residual connections
        """
        if not continued:
            # Reinitialize with small random values
            self.__init__(self.input_dim, self.model_dims, self.use_residual_list)
            
        history = []
        D_prev = float('inf')
        total_positions = sum(len(seq) for seq in seqs)
        
        for it in range(max_iters):
            # Initialize gradients for all layers
            grad_P_list = []
            grad_M_list = []
            grad_gamma_list = []
            grad_beta_list = []
            grad_M_res_list = []
            
            for l in range(self.num_layers):
                out_dim = self.model_dims[l]
                
                # Gradients for P matrix
                grad_P = [[0.0] * out_dim for _ in range(out_dim)]
                grad_P_list.append(grad_P)
                
                # Gradients for M matrix
                if l == 0:
                    in_dim = self.input_dim
                else:
                    in_dim = self.model_dims[l-1]
                grad_M = [[0.0] * in_dim for _ in range(out_dim)]
                grad_M_list.append(grad_M)
                
                # Gradients for layer norm parameters
                grad_gamma = [0.0] * out_dim
                grad_beta = [0.0] * out_dim
                grad_gamma_list.append(grad_gamma)
                grad_beta_list.append(grad_beta)
                
                # Gradients for residual projection
                layer = self.layers[l]
                if layer['use_residual'] == 'separate' and layer['M_res'] is not None:
                    grad_M_res = [[0.0] * in_dim for _ in range(out_dim)]
                else:
                    grad_M_res = None
                grad_M_res_list.append(grad_M_res)
            
            # Forward pass: store intermediate results
            for seq, t_vec in zip(seqs, t_list):
                intermediates = []  # Store layer intermediates for backprop
                current = seq
                
                # Forward pass through layers
                for l, layer in enumerate(self.layers):
                    layer_intermediate = []
                    next_seq = []
                    
                    for k, vec in enumerate(current):
                        # Apply linear transformation
                        x_linear = self._mat_vec(layer['M'], vec)
                        
                        # Apply layer normalization
                        x_norm = self._layer_norm(
                            x_linear, 
                            layer['gamma'], 
                            layer['beta'], 
                            layer['eps']
                        )
                        
                        out_dim = len(layer['P'])
                        Nk = [0.0] * out_dim
                        phi_vals = [[0.0] * out_dim for _ in range(out_dim)]
                        
                        # Compute output and store basis values
                        for i in range(out_dim):
                            for j in range(out_dim):
                                period = layer['periods'][i][j]
                                phi = math.cos(2 * math.pi * k / period)
                                phi_vals[i][j] = phi
                                Nk[i] += layer['P'][i][j] * x_norm[j] * phi
                        
                        # Compute residual
                        residual = None
                        if layer['use_residual'] == 'separate':
                            if layer['M_res'] is not None:
                                residual = self._mat_vec(layer['M_res'], vec)
                            else:
                                residual = vec
                        elif layer['use_residual'] == 'shared':
                            residual = x_linear
                        else:  # None
                            if len(vec) == out_dim:
                                residual = vec
                            else:
                                residual = [0.0] * out_dim
                        
                        # Add residual
                        Nk = self._vec_add(Nk, residual)
                        
                        # Store intermediate values for backprop
                        layer_intermediate.append({
                            'input_vec': vec,
                            'x_linear': x_linear,
                            'x_norm': x_norm,
                            'phi_vals': phi_vals,
                            'output_vec': Nk,
                            'residual': residual
                        })
                        next_seq.append(Nk)
                    
                    intermediates.append(layer_intermediate)
                    current = next_seq
                
                # Backward pass through layers
                for k in range(len(seq)):
                    # Start with last layer
                    d_out = None
                    for l in range(self.num_layers-1, -1, -1):
                        layer = self.layers[l]
                        interm = intermediates[l][k]
                        vec = interm['input_vec']
                        x_linear = interm['x_linear']
                        x_norm = interm['x_norm']
                        phi_vals = interm['phi_vals']
                        output_vec = interm['output_vec']
                        residual = interm['residual']
                        out_dim = len(output_vec)
                        
                        # Initialize error for this layer
                        if l == self.num_layers-1:
                            # Final layer error: 2*(output - target)
                            d_out = [2 * (output_vec[i] - t_vec[i]) 
                                     for i in range(len(output_vec))]
                        else:
                            # Propagate error from next layer
                            d_out = d_next
                        
                        # Compute gradient for P matrix
                        for i in range(len(d_out)):
                            for j in range(len(layer['P'][i])):
                                grad_P_list[l][i][j] += (
                                    d_out[i] * x_norm[j] * phi_vals[i][j]
                                ) / total_positions
                        
                        # Compute gradient for normalized input (x_norm)
                        d_x_norm = [0.0] * len(x_norm)
                        for j in range(len(x_norm)):
                            for i in range(len(d_out)):
                                term = d_out[i] * layer['P'][i][j] * phi_vals[i][j]
                                d_x_norm[j] += term
                        
                        # Compute gradient for layer norm parameters
                        n = len(x_linear)
                        mean = sum(x_linear) / n
                        var = sum((xi - mean)**2 for xi in x_linear) / n
                        std = math.sqrt(var + layer['eps'])
                        
                        # Gradient for gamma and beta
                        for j in range(n):
                            # d_beta = d_x_norm * 1
                            grad_beta_list[l][j] += d_x_norm[j] / total_positions
                            
                            # d_gamma = d_x_norm * normalized_x
                            grad_gamma_list[l][j] += (
                                d_x_norm[j] * (x_linear[j] - mean) / std
                            ) / total_positions
                        
                        # Gradient for x_linear (before normalization)
                        d_x_linear = [0.0] * n
                        for j in range(n):
                            # Gradient from normalized value
                            term1 = d_x_norm[j] * layer['gamma'][j] / std
                            
                            # Gradient from mean and variance
                            term2 = sum(d_x_norm[i] * layer['gamma'][i] * 
                                       (x_linear[j] - mean) / (n * std**3)
                                       for i in range(n))
                            
                            d_x_linear[j] = term1 - term2 - sum(
                                d_x_norm[i] * layer['gamma'][i] for i in range(n)
                            ) / (n * std)
                        
                        # Compute gradient for M matrix
                        for i in range(len(d_x_linear)):
                            for d in range(len(vec)):
                                grad_M_list[l][i][d] += (
                                    d_x_linear[i] * vec[d]
                                ) / total_positions
                        
                        # Compute gradient for residual projection
                        if layer['use_residual'] == 'separate' and layer['M_res'] is not None:
                            for i in range(len(d_out)):
                                for d in range(len(vec)):
                                    grad_M_res_list[l][i][d] += (
                                        d_out[i] * vec[d]
                                    ) / total_positions
                        
                        # Propagate error to previous layer
                        d_prev = [0.0] * len(vec)
                        
                        # Gradient from linear path (M)
                        for d in range(len(vec)):
                            for i in range(len(d_x_linear)):
                                d_prev[d] += d_x_linear[i] * layer['M'][i][d]
                        
                        # Gradient from residual path
                        if layer['use_residual'] == 'separate':
                            if layer['M_res'] is not None:
                                for d in range(len(vec)):
                                    for i in range(len(d_out)):
                                        d_prev[d] += d_out[i] * layer['M_res'][i][d]
                            else:  # Identity residual
                                for i in range(min(len(d_out), len(vec))):
                                    d_prev[i] += d_out[i]
                        elif layer['use_residual'] == 'shared':
                            for d in range(len(vec)):
                                for i in range(len(d_out)):
                                    d_prev[d] += d_out[i] * layer['M'][i][d]
                        else:  # None
                            if len(vec) == len(d_out):  # Identity
                                for i in range(len(d_out)):
                                    d_prev[i] += d_out[i]
                        
                        d_next = d_prev
            
            # Update parameters
            for l in range(self.num_layers):
                layer = self.layers[l]
                # Update P matrix
                for i in range(len(layer['P'])):
                    for j in range(len(layer['P'][i])):
                        layer['P'][i][j] -= learning_rate * grad_P_list[l][i][j]
                
                # Update M matrix
                for i in range(len(layer['M'])):
                    for j in range(len(layer['M'][i])):
                        layer['M'][i][j] -= learning_rate * grad_M_list[l][i][j]
                
                # Update layer norm parameters
                for i in range(len(layer['gamma'])):
                    layer['gamma'][i] -= learning_rate * grad_gamma_list[l][i]
                    layer['beta'][i] -= learning_rate * grad_beta_list[l][i]
                
                # Update residual projection matrix
                if grad_M_res_list[l] is not None:
                    for i in range(len(layer['M_res'])):
                        for j in range(len(layer['M_res'][i])):
                            layer['M_res'][i][j] -= learning_rate * grad_M_res_list[l][i][j]
            
            # Calculate current loss
            current_D = self.deviation(seqs, t_list)
            history.append(current_D)
            
            # Print progress
            if it % print_every == 0 or it == max_iters - 1:
                print(f"GD Iter {it:3d}: D = {current_D:.6e}, LR = {learning_rate:.6f}")
            
            # Check convergence
            if current_D >= D_prev - tol:
                print(f"Converged after {it+1} iterations.")
                break
            D_prev = current_D
            
            # Apply learning rate decay
            learning_rate *= decay_rate
        
        self.trained = True
        return history

    def predict_t(self, seq):
        """
        Predict target vector for a sequence
        Returns the average of all N(k) vectors in the sequence
        """
        outputs = self.describe(seq)
        if not outputs:
            return [0.0] * self.model_dims[-1]
        
        # Average all output vectors
        t_pred = [0.0] * self.model_dims[-1]
        for vec in outputs:
            for i in range(len(vec)):
                t_pred[i] += vec[i]
        return [x / len(outputs) for x in t_pred]

    def show(self, what=None, first_num=5):
        """
        Display model status with hierarchical support
        Includes layer normalization and residual parameters
        """
        # Default attributes to show
        default_attrs = ['config', 'M', 'P', 'gamma', 'beta', 'M_res', 'periods']
        
        # Handle different what parameter types
        if what is None:
            attrs = default_attrs
        elif what == 'all':
            attrs = default_attrs
        elif isinstance(what, str):
            attrs = [what]
        else:
            attrs = what
            
        print("Hierarchical HierDDpm Status")
        print("=" * 50)
        
        # Display each requested attribute
        for attr in attrs:
            if attr == 'config':
                print("\n[Configuration]")
                print(f"{'Input dim:':<20} {self.input_dim}")
                print(f"{'Layer dims:':<20} {self.model_dims}")
                print(f"{'Residual modes:':<20} {self.use_residual_list}")
                print(f"{'Trained:':<20} {self.trained}")
            
            elif attr in ['M', 'P', 'gamma', 'beta', 'M_res', 'periods']:
                print(f"\n[{attr.upper()} Matrices]")
                for l in range(self.num_layers):
                    layer = self.layers[l]
                    print(f"  Layer {l} ({attr}):")
                    
                    if attr == 'gamma' or attr == 'beta':
                        # Vector display
                        vec = layer[attr]
                        print(f"    Values: [{', '.join(f'{x:.4f}' for x in vec[:min(first_num, len(vec))])}" +
                              (', ...]' if len(vec) > first_num else ']'))
                    
                    elif attr == 'M_res' and layer[attr] is None:
                        print("    No residual projection matrix (identity or shared)")
                    
                    else:
                        matrix = layer[attr]
                        if matrix is None:
                            print("    None")
                            continue
                            
                        print(f"    Shape: {len(matrix)}x{len(matrix[0])}")
                        print("    Sample rows:")
                        for i in range(min(first_num, len(matrix))):
                            vals = matrix[i][:min(first_num, len(matrix[i]))]
                            if attr in ['M', 'M_res']:
                                formatted = [f"{v:.4f}" for v in vals]
                            elif attr == 'P':
                                formatted = [f"{v:.6f}" for v in vals]
                            else:  # periods
                                formatted = [str(int(v)) for v in vals]
                            print(f"      {attr}[{i}][:]: [{', '.join(formatted)}" + 
                                  (f", ...]" if len(matrix[i]) > first_num else "]"))
            
            else:
                print(f"\n[Unknown attribute: {attr}]")
        
        print("=" * 50)

    def count_parameters(self):
        """Count learnable parameters including layer norm and residual projections"""
        total_params = 0
        print("Parameter Count:")
        
        for l in range(self.num_layers):
            layer = self.layers[l]
            M = layer['M']
            P = layer['P']
            gamma = layer['gamma']
            beta = layer['beta']
            
            M_params = len(M) * len(M[0])
            P_params = len(P) * len(P[0])
            gamma_params = len(gamma)
            beta_params = len(beta)
            
            M_res_params = 0
            if layer['M_res'] is not None:
                M_res_params = len(layer['M_res']) * len(layer['M_res'][0])
            
            layer_params = M_params + P_params + gamma_params + beta_params + M_res_params
            total_params += layer_params
            
            print(f"  Layer {l}:")
            print(f"    M matrix: {len(M)}×{len(M[0])} = {M_params}")
            print(f"    P matrix: {len(P)}×{len(P[0])} = {P_params}")
            print(f"    Gamma: {len(gamma)} = {gamma_params}")
            print(f"    Beta: {len(beta)} = {beta_params}")
            if layer['M_res'] is not None:
                print(f"    M_res: {len(layer['M_res'])}×{len(layer['M_res'][0])} = {M_res_params}")
            print(f"    Layer total: {layer_params}")
        
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
    
    random.seed(3)
    
    # Hierarchical configuration
    input_dim = 10      # Input vector dimension
    model_dims = [8, 6, 3]  # Output dimensions for each layer
    use_residual_list = ['separate', 'shared', None]  # Residual modes
    num_seqs = 10
    min_len, max_len = 100, 200

    # Generate synthetic training data
    seqs = []     # List of sequences (each sequence: list of n-dim vectors)
    t_list = []   # List of target vectors (m-dim)
    
    for _ in range(num_seqs):
        L = random.randint(min_len, max_len)
        # Generate n-dimensional input sequence
        seq = [[random.uniform(-1,1) for _ in range(input_dim)] for __ in range(L)]
        seqs.append(seq)
        # Generate m-dimensional target vector
        t_list.append([random.uniform(-1,1) for _ in range(model_dims[-1])])

    # Create and train hierarchical model
    print("\nTraining Hierarchical HierDDpm with layer norm and residual connections...")
    hndd = HierDDpm(
        input_dim=input_dim,
        model_dims=model_dims,
        use_residual_list=use_residual_list
    )
    
    # Display initial model configuration
    hndd.show(['config'])
    
    # Gradient Descent Training
    grad_history = hndd.grad_train(
        seqs, 
        t_list, 
        max_iters=50,
        learning_rate=0.1,
        decay_rate=0.99,
        print_every=5
    )

    # Predict targets
    print("\nPredictions (first 3 sequences):")
    for i, seq in enumerate(seqs[:3]):
        t_pred = hndd.predict_t(seq)
        print(f"Seq {i+1}: Target={[f'{x:.4f}' for x in t_list[i]]}")
        print(f"         Predicted={[f'{x:.4f}' for x in t_pred]}")

    # Calculate prediction correlations
    preds = [hndd.predict_t(seq) for seq in seqs]
    correlations = []
    for i in range(model_dims[-1]):
        actual = [t[i] for t in t_list]
        predicted = [p[i] for p in preds]
        corr = correlation(actual, predicted)
        correlations.append(corr)
        print(f"Output dim {i} correlation: {corr:.4f}")
    print(f"Average correlation: {mean(correlations):.4f}")

    # Show model status
    print("\nModel status after training:")
    hndd.show(['config', 'M', 'P', 'gamma', 'beta'])

    # Count learnable parameters
    print("\nParameter count:")
    hndd.count_parameters()

    # Save and load model
    print("\nTesting save/load functionality:")
    hndd.save("hierarchical_ndd_model.pkl")
    loaded = HierDDpm.load("hierarchical_ndd_model.pkl")
    
    # Verify loaded model
    t_pred_loaded = loaded.predict_t(seqs[0])
    print(f"Prediction from loaded model: {[f'{x:.4f}' for x in t_pred_loaded]}")
