# Copyright (C) 2005-2025, Bin-Guang Ma (mbg@mail.hzau.edu.cn); SPDX-License-Identifier: MIT
# The Hierarchical Numeric Dual Descriptor (Tensor form) implemented with pure Python
# Added layer normalization and residual connections
# This program is for the demonstration of methodology and not fully refined.
# Author: Bin-Guang Ma (assisted by DeepSeek); Date: 2025-6-24

import math
import random
import pickle

class HierDDpm:
    """
    Hierarchical Numeric Dual Descriptor for vector sequences with:
      - Multiple layers with linear transformation M and tensor P
      - Layer normalization after linear transformation
      - Residual connection options
      - Each layer: P ∈ R^{m×m×o} of basis coefficients, M ∈ R^{m×n}
      - Periods: period[i,j,g] = i*(m*o) + j*o + g + 2
      - Basis function: phi_{i,j,g}(k) = cos(2π * k / period[i,j,g])
    """
    def __init__(self, input_dim=2, model_dims=[2], num_basis_list=[5], use_residual_list=None):
        """
        Initialize hierarchical HierDDpm
        
        Args:
            input_dim (int): Input vector dimension
            model_dims (list): Output dimensions for each layer
            num_basis_list (list): Number of basis functions for each layer
            use_residual_list (list): Residual connection types for each layer
                Options: 'separate', 'shared', None
        """
        self.input_dim = input_dim
        self.model_dims = model_dims
        self.num_basis_list = num_basis_list
        self.num_layers = len(model_dims)
        self.trained = False
        
        # Handle residual connection list
        if use_residual_list is None:
            self.use_residual_list = ['separate'] * len(model_dims)
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
            
            # Get residual connection type for this layer
            use_residual = self.use_residual_list[l]
                
            # Initialize M matrix (out_dim x in_dim)
            M = [[random.uniform(-0.5, 0.5) for _ in range(in_dim)] 
                 for _ in range(out_dim)]
            
            # Initialize P tensor (out_dim x out_dim x num_basis)
            P = [[[random.uniform(-0.1, 0.1) for _ in range(num_basis_list[l])]
                  for _ in range(out_dim)]
                 for _ in range(out_dim)]
            
            # Precompute periods tensor
            periods = [[[ i*(out_dim*num_basis_list[l]) + j*num_basis_list[l] + g + 2
                         for g in range(num_basis_list[l])]
                        for j in range(out_dim)]
                       for i in range(out_dim)]
            
            # Initialize residual projection if needed
            residual_M = None
            if use_residual == 'separate' and in_dim != out_dim:
                # Create separate projection matrix for residual
                residual_M = [[random.uniform(-0.5, 0.5) for _ in range(in_dim)]
                             for _ in range(out_dim)]
            
            # Initialize layer normalization parameters
            # Gamma (scale) and beta (shift) parameters
            gamma = [1.0] * out_dim  # Initialize scale to 1
            beta = [0.0] * out_dim   # Initialize shift to 0
            
            self.layers.append({
                'M': M,
                'P': P,
                'periods': periods,
                'use_residual': use_residual,
                'residual_M': residual_M,
                'gamma': gamma,
                'beta': beta
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
    
    def _vec_div(self, v, s):
        """Vector division by scalar"""
        return [x / s for x in v]
    
    def _layer_norm(self, x, gamma, beta, eps=1e-5):
        """
        Apply layer normalization to a vector
        
        Args:
            x (list): Input vector
            gamma (list): Scale parameters
            beta (list): Shift parameters
            eps (float): Small value to avoid division by zero
            
        Returns:
            list: Normalized vector
        """
        # Calculate mean
        mean = sum(x) / len(x)
        
        # Calculate variance
        variance = sum((xi - mean) ** 2 for xi in x) / len(x)
        
        # Normalize
        x_norm = [(xi - mean) / math.sqrt(variance + eps) for xi in x]
        
        # Apply scale and shift
        return [gamma_i * x_norm_i + beta_i 
                for gamma_i, x_norm_i, beta_i in zip(gamma, x_norm, beta)]

    def describe(self, seq):
        """Compute N(k) vectors for each position and vector in sequence"""
        current = seq
        for layer in self.layers:
            new_seq = []
            for k, vec in enumerate(current):
                # Linear transformation: x = M * vec
                x = self._mat_vec(layer['M'], vec)
                
                # Apply layer normalization
                x = self._layer_norm(x, layer['gamma'], layer['beta'])
                
                out_dim = len(layer['P'])
                Nk = [0.0] * out_dim
                
                # Apply position-dependent transformation
                for i in range(out_dim):
                    for j in range(out_dim):
                        for g in range(len(layer['P'][i][j])):
                            period = layer['periods'][i][j][g]
                            phi = math.cos(2 * math.pi * k / period)
                            Nk[i] += layer['P'][i][j][g] * x[j] * phi
                
                # Apply residual connection
                use_residual = layer['use_residual']
                if use_residual == 'separate':
                    if layer['residual_M'] is not None:
                        # Use separate projection matrix
                        residual = self._mat_vec(layer['residual_M'], vec)
                    else:
                        # Identity residual (when dimensions match)
                        residual = vec
                    Nk = self._vec_add(Nk, residual)
                elif use_residual == 'shared':        
                    # Use the same M matrix for residual
                    residual = self._mat_vec(layer['M'], vec)
                    Nk = self._vec_add(Nk, residual)
                else:
                    # None or other: use identity residual only if dimensions match
                    if len(vec) == out_dim and use_residual is not None:
                        Nk = self._vec_add(Nk, vec)
                
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
        """
        if not continued:
            # Reinitialize with small random values
            self.__init__(self.input_dim, self.model_dims, self.num_basis_list, self.use_residual_list)
            
        history = []
        D_prev = float('inf')
        total_positions = sum(len(seq) for seq in seqs)
        
        for it in range(max_iters):
            # Initialize gradients for all layers
            grad_P_list = []
            grad_M_list = []
            grad_residual_M_list = []
            grad_gamma_list = []
            grad_beta_list = []
            
            for l in range(self.num_layers):
                out_dim = self.model_dims[l]
                num_basis = self.num_basis_list[l]
                
                # Gradients for P tensor
                grad_P = [[[0.0] * num_basis for _ in range(out_dim)] 
                          for _ in range(out_dim)]
                grad_P_list.append(grad_P)
                
                # Gradients for M matrix
                if l == 0:
                    in_dim = self.input_dim
                else:
                    in_dim = self.model_dims[l-1]
                grad_M = [[0.0] * in_dim for _ in range(out_dim)]
                grad_M_list.append(grad_M)
                
                # Gradients for residual projection (if exists)
                use_residual = self.layers[l]['use_residual']
                if use_residual == 'separate' and in_dim != out_dim:
                    grad_residual_M = [[0.0] * in_dim for _ in range(out_dim)]
                else:
                    grad_residual_M = None
                grad_residual_M_list.append(grad_residual_M)
                
                # Gradients for layer normalization parameters
                grad_gamma = [0.0] * out_dim
                grad_beta = [0.0] * out_dim
                grad_gamma_list.append(grad_gamma)
                grad_beta_list.append(grad_beta)
            
            # Forward pass: store intermediate results
            for seq, t_vec in zip(seqs, t_list):
                intermediates = []  # Store layer intermediates for backprop
                current = seq
                
                # Forward pass through layers
                for l, layer in enumerate(self.layers):
                    layer_intermediate = []
                    next_seq = []
                    use_residual = layer['use_residual']
                    
                    for k, vec in enumerate(current):
                        # Apply linear transformation
                        x = self._mat_vec(layer['M'], vec)
                        
                        # Apply layer normalization
                        gamma = layer['gamma']
                        beta = layer['beta']
                        x_norm = self._layer_norm(x, gamma, beta)
                        
                        out_dim = len(layer['P'])
                        Nk = [0.0] * out_dim
                        phi_vals = {}
                        
                        # Compute output and store basis values
                        for i in range(out_dim):
                            for j in range(out_dim):
                                for g in range(len(layer['P'][i][j])):
                                    period = layer['periods'][i][j][g]
                                    phi = math.cos(2 * math.pi * k / period)
                                    phi_vals[(i, j, g)] = phi
                                    Nk[i] += layer['P'][i][j][g] * x_norm[j] * phi
                        
                        # Apply residual connection
                        if use_residual == 'separate':
                            if layer['residual_M'] is not None:
                                residual = self._mat_vec(layer['residual_M'], vec)
                            else:
                                residual = vec
                            output_vec = self._vec_add(Nk, residual)
                        elif use_residual == 'shared':        
                            residual = self._mat_vec(layer['M'], vec)
                            output_vec = self._vec_add(Nk, residual)
                        else:
                            if len(vec) == out_dim and use_residual is not None:
                                output_vec = self._vec_add(Nk, vec)
                            else:
                                output_vec = Nk
                        
                        layer_intermediate.append({
                            'input_vec': vec,
                            'x': x,          # Before normalization
                            'x_norm': x_norm, # After normalization
                            'phi_vals': phi_vals,
                            'output_vec': output_vec
                        })
                        next_seq.append(output_vec)
                    
                    intermediates.append(layer_intermediate)
                    current = next_seq
                
                # Backward pass through layers
                for k in range(len(seq)):
                    # Start with last layer
                    d_out = None
                    for l in range(self.num_layers-1, -1, -1):
                        layer = self.layers[l]
                        interm = intermediates[l][k]
                        x = interm['x']              # Linear output (before norm)
                        x_norm = interm['x_norm']    # After normalization
                        phi_vals = interm['phi_vals']
                        input_vec = interm['input_vec']
                        output_vec = interm['output_vec']
                        use_residual = layer['use_residual']
                        
                        # Initialize error for this layer
                        if l == self.num_layers-1:
                            # Final layer error: 2*(output - target)
                            d_out = [2 * (output_vec[i] - t_vec[i]) 
                                     for i in range(len(output_vec))]
                        else:
                            # Propagate error from next layer
                            d_out = d_next
                        
                        # Compute gradient for P tensor
                        for i in range(len(d_out)):
                            for j in range(len(layer['P'][i])):
                                for g in range(len(layer['P'][i][j])):
                                    grad_P_list[l][i][j][g] += (
                                        d_out[i] * x_norm[j] * phi_vals[(i, j, g)]
                                    ) / total_positions
                        
                        # Compute gradient for layer normalization parameters
                        # (Simplified implementation - actual gradients would be more complex)
                        for i in range(len(d_out)):
                            # Gamma gradient: dL/dgamma_i = dL/dy_i * x_norm_i
                            grad_gamma_list[l][i] += d_out[i] * x_norm[i] / total_positions
                            
                            # Beta gradient: dL/dbeta_i = dL/dy_i
                            grad_beta_list[l][i] += d_out[i] / total_positions
                        
                        # Compute gradient for M matrix and input
                        d_x_norm = [0.0] * len(x_norm)
                        for j in range(len(x_norm)):
                            for i in range(len(d_out)):
                                for g in range(len(layer['P'][i][j])):
                                    term = d_out[i] * layer['P'][i][j][g] * phi_vals[(i, j, g)]
                                    d_x_norm[j] += term
                        
                        # Compute gradient for M matrix
                        for j in range(len(x)):
                            for d in range(len(input_vec)):
                                # Simplified gradient calculation (actual would involve norm derivative)
                                grad_M_list[l][j][d] += d_x_norm[j] * input_vec[d] / total_positions
                        
                        # Compute gradient for residual projection (if exists)
                        if use_residual == 'separate' and layer['residual_M'] is not None:
                            for i in range(len(d_out)):
                                for d in range(len(input_vec)):
                                    grad_residual_M_list[l][i][d] += d_out[i] * input_vec[d] / total_positions
                        
                        # Propagate error to previous layer
                        if l > 0:
                            # d_input = M^T * d_x (simplified)
                            M_T = self._transpose(layer['M'])
                            d_next = self._mat_vec(M_T, d_x_norm)
                        else:
                            d_next = None
            
            # Update parameters
            for l in range(self.num_layers):
                layer = self.layers[l]
                
                # Update P tensor
                for i in range(len(layer['P'])):
                    for j in range(len(layer['P'][i])):
                        for g in range(len(layer['P'][i][j])):
                            layer['P'][i][j][g] -= learning_rate * grad_P_list[l][i][j][g]
                
                # Update M matrix
                for i in range(len(layer['M'])):
                    for j in range(len(layer['M'][i])):
                        layer['M'][i][j] -= learning_rate * grad_M_list[l][i][j]
                
                # Update residual projection matrix (if exists)
                if grad_residual_M_list[l] is not None:
                    for i in range(len(layer['residual_M'])):
                        for j in range(len(layer['residual_M'][i])):
                            layer['residual_M'][i][j] -= learning_rate * grad_residual_M_list[l][i][j]
                
                # Update layer normalization parameters
                for i in range(len(layer['gamma'])):
                    layer['gamma'][i] -= learning_rate * grad_gamma_list[l][i]
                    layer['beta'][i] -= learning_rate * grad_beta_list[l][i]
            
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
        """
        # Default attributes to show
        default_attrs = ['config', 'M', 'P', 'periods']
        
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
                print(f"{'Num basis:':<20} {self.num_basis_list}")
                print(f"{'Residual types:':<20} {self.use_residual_list}")
                print(f"{'Trained:':<20} {self.trained}")
            
            elif attr in ['M', 'P', 'periods', 'residual_M', 'gamma', 'beta']:
                print(f"\n[{attr.upper()} Matrices/Tensors]")
                for l in range(self.num_layers):
                    print(f"  Layer {l}:")
                    if attr in self.layers[l]:
                        tensor = self.layers[l][attr]
                        if tensor is None:
                            print(f"    {attr} is None")
                            continue
                            
                        # Handle different tensor types
                        if isinstance(tensor[0], list) and isinstance(tensor[0][0], list):
                            # 3D tensor (P, periods)
                            print(f"    Shape: {len(tensor)}x{len(tensor[0])}x{len(tensor[0][0])}")
                            print("    Sample slices:")
                            for i in range(min(first_num, len(tensor))):
                                for j in range(min(first_num, len(tensor[i]))):
                                    vals = tensor[i][j][:min(first_num, len(tensor[i][j]))]
                                    if attr == 'P':
                                        formatted = [f"{v:.6f}" for v in vals]
                                    elif attr == 'periods':
                                        formatted = [str(int(v)) for v in vals]
                                    else:
                                        formatted = [f"{v:.6f}" for v in vals]
                                    print(f"      {attr}[{i}][{j}][:]: [{', '.join(formatted)}" + 
                                          (f", ...]" if len(tensor[i][j]) > first_num else "]"))
                        elif isinstance(tensor[0], list):
                            # 2D matrix (M, residual_M)
                            print(f"    Shape: {len(tensor)}x{len(tensor[0])}")
                            print("    Sample values:")
                            for i in range(min(first_num, len(tensor))):
                                vals = tensor[i][:min(first_num, len(tensor[i]))]
                                print(f"      Row {i}: [{', '.join(f'{x:.4f}' for x in vals)}" + 
                                      (f", ...]" if len(tensor[i]) > first_num else "]"))
                        else:
                            # 1D vector (gamma, beta)
                            print(f"    Values: [{', '.join(f'{x:.6f}' for x in tensor[:first_num])}" + 
                                  (", ..." if len(tensor) > first_num else ""))
                    else:
                        print(f"    {attr} not present in layer")
            
            else:
                print(f"\n[Unknown attribute: {attr}]")
        
        print("=" * 50)

    def count_parameters(self):
        """Count learnable parameters (P tensors, M matrices, residual projections, and normalization parameters for all layers)"""
        total_params = 0
        print("Parameter Count:")
        
        for l in range(self.num_layers):
            M = self.layers[l]['M']
            P = self.layers[l]['P']
            residual_M = self.layers[l]['residual_M']
            gamma = self.layers[l]['gamma']
            beta = self.layers[l]['beta']
            
            M_params = len(M) * len(M[0])
            P_params = len(P) * len(P[0]) * len(P[0][0])
            residual_params = len(residual_M) * len(residual_M[0]) if residual_M is not None else 0
            norm_params = len(gamma) + len(beta)  # Gamma and beta parameters
            
            layer_params = M_params + P_params + residual_params + norm_params
            total_params += layer_params
            
            print(f"  Layer {l}:")
            print(f"    M matrix: {len(M)}×{len(M[0])} = {M_params} parameters")
            print(f"    P tensor: {len(P)}×{len(P[0])}×{len(P[0][0])} = {P_params} parameters")
            if residual_M is not None:
                print(f"    Residual projection: {len(residual_M)}×{len(residual_M[0])} = {residual_params} parameters")
            print(f"    Layer norm params: gamma={len(gamma)}, beta={len(beta)} = {norm_params} parameters")
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
    num_basis_list = [5, 4, 3]  # Basis functions per layer
    # Define residual connection types for each layer
    use_residual_list = ['separate', 'shared', None]
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
    print("\nTraining Hierarchical HierDDpm with LayerNorm and Residual Connections...")
    hndd = HierDDpm(
        input_dim=input_dim,
        model_dims=model_dims,
        num_basis_list=num_basis_list,
        use_residual_list=use_residual_list
    )
    
    # Show model configuration before training
    print("\nModel configuration before training:")
    hndd.show(['config'])
    
    # Gradient Descent Training
    grad_history = hndd.grad_train(
        seqs, 
        t_list, 
        max_iters=50,
        learning_rate=0.2,  # Adjusted for better convergence
        decay_rate=0.95,
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
    hndd.show(['config', 'M', 'gamma', 'beta'])

    # Count learnable parameters
    print("\nParameter count:")
    total_params = hndd.count_parameters()
    print(f"Total parameters: {total_params}")

    # Save and load model
    print("\nTesting save/load functionality:")
    hndd.save("hierarchical_ndd_model.pkl")
    loaded = HierDDpm.load("hierarchical_ndd_model.pkl")
    
    # Verify loaded model
    t_pred_loaded = loaded.predict_t(seqs[0])
    print(f"Prediction from loaded model: {[f'{x:.4f}' for x in t_pred_loaded]}")
