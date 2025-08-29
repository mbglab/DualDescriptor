# Copyright (C) 2005-2025, Bin-Guang Ma (mbg@mail.hzau.edu.cn); SPDX-License-Identifier: MIT
# The Hierarchical Numeric Dual Descriptor (Tensor form) with Linker matrices in pure Python
# This program is for the demonstration of methodology and not fully refined.
# Author: Bin-Guang Ma (assisted by DeepSeek); Date: 2025-6-22

import math
import random
import pickle

class HierDDLpm:
    """
    Hierarchical Numeric Dual Descriptor for vector sequences with:
      - Multiple layers with linear transformation M and tensor P
      - Each layer: P ∈ R^{m×m×o} of basis coefficients, M ∈ R^{m×n}
      - Periods: period[i,j,g] = i*(m*o) + j*o + g + 2
      - Basis function: phi_{i,j,g}(k) = cos(2π * k / period[i,j,g])
    Key modifications:
    - Added Linker matrices for sequence length transformation
    - Layer operations: T = position-dependent transform, U = T * Linker
    - Added linker_trainable parameter to control Linker matrix training
    """
    def __init__(self, input_dim=2, model_dims=[2], num_basis_list=[5],
                 input_seq_len=100, linker_dims=[50], linker_trainable=False):
        """
        Initialize hierarchical HierDDLpm with sequence length transformation
        
        Args:
            input_dim (int): Input vector dimension
            model_dims (list): Output dimensions for each layer
            num_basis_list (list): Number of basis functions for each layer
            input_seq_len (int): Fixed input sequence length
            linker_dims (list): Output sequence lengths for each layer
            linker_trainable (bool or list): Trainability of Linker matrices
        """
        self.input_dim = input_dim
        self.model_dims = model_dims
        self.num_basis_list = num_basis_list
        self.input_seq_len = input_seq_len
        self.linker_dims = linker_dims
        self.num_layers = len(model_dims)
        self.trained = False
        
        # Validate dimensions
        if len(linker_dims) != self.num_layers:
            raise ValueError("linker_dims must have same length as model_dims")
        if len(num_basis_list) != self.num_layers:
            raise ValueError("num_basis_list must have same length as model_dims")
        
        # Process linker_trainable parameter
        if isinstance(linker_trainable, bool):
            self.linker_trainable = [linker_trainable] * self.num_layers
        elif isinstance(linker_trainable, list):
            if len(linker_trainable) != self.num_layers:
                raise ValueError("linker_trainable list length must match number of layers")
            self.linker_trainable = linker_trainable
        else:
            raise TypeError("linker_trainable must be bool or list of bools")
        
        # Initialize layers with M, P, periods, and Linker matrices
        self.layers = []
        for l in range(self.num_layers):
            # Determine input dimensions for this layer
            if l == 0:
                in_dim = input_dim
                in_seq = input_seq_len
            else:
                in_dim = model_dims[l-1]
                in_seq = linker_dims[l-1]
                
            out_dim = model_dims[l]
            out_seq = linker_dims[l]
            
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
            
            # Initialize Linker matrix (in_seq x out_seq)
            Linker = [[random.uniform(-0.1, 0.1) for _ in range(out_seq)] 
                      for _ in range(in_seq)]
            
            self.layers.append({
                'M': M,
                'P': P,
                'periods': periods,
                'Linker': Linker,
                'linker_trainable': self.linker_trainable[l]
            })

    # ---- Matrix and vector operations ----
    def _mat_mul(self, A, B):
        """Matrix multiplication: A (p×q) * B (q×r) → (p×r) matrix"""
        p = len(A)
        q = len(A[0])
        r = len(B[0])
        C = [[0.0] * r for _ in range(p)]
        for i in range(p):
            for k in range(q):
                aik = A[i][k]
                for j in range(r):
                    C[i][j] += aik * B[k][j]
        return C
    
    def _mat_vec(self, M, v):
        """Matrix-vector multiplication: M (a×b) * v (b) → vector (a)"""
        return [sum(M[i][j] * v[j] for j in range(len(v))) for i in range(len(M))]
    
    def _transpose(self, M):
        """Matrix transpose"""
        return [list(col) for col in zip(*M)]
    
    def _vec_sub(self, u, v):
        """Vector subtraction"""
        return [u_i - v_i for u_i, v_i in zip(u, v)]
    
    def _dot(self, u, v):
        """Vector dot product"""
        return sum(u_i * v_i for u_i, v_i in zip(u, v))
    
    def _vec_avg(self, vectors):
        """Element-wise average of vectors"""
        if not vectors:
            return []
        dim = len(vectors[0])
        avg = [0.0] * dim
        for vec in vectors:
            for i in range(dim):
                avg[i] += vec[i]
        return [x / len(vectors) for x in avg]

    def describe(self, seq):
        """Compute output vectors for each position in the sequence"""
        # Validate input sequence length
        if len(seq) != self.input_seq_len:
            raise ValueError(f"Input sequence length must be {self.input_seq_len}")
        
        current = seq
        for l, layer in enumerate(self.layers):
            # Apply linear transformation to each vector
            linear_out = [self._mat_vec(layer['M'], vec) for vec in current]
            
            # Apply position-dependent transformation
            T_matrix = []
            for k, vec in enumerate(linear_out):
                out_dim = len(layer['P'])
                Tk = [0.0] * out_dim
                for i in range(out_dim):
                    for j in range(out_dim):
                        for g in range(len(layer['P'][i][j])):
                            period = layer['periods'][i][j][g]
                            phi = math.cos(2 * math.pi * k / period)
                            Tk[i] += layer['P'][i][j][g] * vec[j] * phi
                T_matrix.append(Tk)
            
            # Convert to matrix form: rows=features, columns=time steps
            T_transposed = [list(col) for col in zip(*T_matrix)]
            
            # Apply sequence length transformation: U = T * Linker
            U_matrix = self._mat_mul(T_transposed, layer['Linker'])
            
            # Convert back to sequence of vectors
            current = []
            for j in range(len(U_matrix[0])):
                vec = [U_matrix[i][j] for i in range(len(U_matrix))]
                current.append(vec)
        
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
        with added Layer Normalization after linear transformation for better gradient scaling.
        """
        if not continued:
            # Reinitialize with small random values
            self.__init__(self.input_dim, self.model_dims, self.num_basis_list,
                          self.input_seq_len, self.linker_dims, self.linker_trainable)
            
        history = []
        D_prev = float('inf')
        total_positions = sum(len(seq) for seq in seqs)
        epsilon = 1e-88  # Small constant for numerical stability in normalization
        
        # Initialize gradient storage for all layers
        grad_P_list = []
        grad_M_list = []
        grad_Linker_list = []
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
            
            # Gradients for Linker matrix
            if l == 0:
                in_seq = self.input_seq_len
            else:
                in_seq = self.linker_dims[l-1]
            out_seq = self.linker_dims[l]
            grad_Linker = [[0.0] * out_seq for _ in range(in_seq)]
            grad_Linker_list.append(grad_Linker)
        
        for it in range(max_iters):
            # Reset gradients
            for l in range(self.num_layers):
                # Reset P gradients
                for i in range(len(grad_P_list[l])):
                    for j in range(len(grad_P_list[l][i])):
                        for g in range(len(grad_P_list[l][i][j])):
                            grad_P_list[l][i][j][g] = 0.0
                
                # Reset M gradients
                for i in range(len(grad_M_list[l])):
                    for j in range(len(grad_M_list[l][i])):
                        grad_M_list[l][i][j] = 0.0
                
                # Reset Linker gradients
                for i in range(len(grad_Linker_list[l])):
                    for j in range(len(grad_Linker_list[l][i])):
                        grad_Linker_list[l][i][j] = 0.0
            
            total_loss = 0.0
            
            # Process each sequence
            for seq, t_vec in zip(seqs, t_list):
                intermediates = []  # Store layer intermediates for backprop
                current = seq
                
                # Forward pass through layers
                for l, layer in enumerate(self.layers):
                    layer_intermediate = {}
                    
                    # Apply linear transformation
                    linear_out = [self._mat_vec(layer['M'], vec) for vec in current]
                    
                    # Apply Layer Normalization to each position's vector
                    normalized_linear_out = []
                    mean_cache = []
                    var_cache = []
                    orig_cache = []
                    for vec in linear_out:
                        # Compute mean and variance
                        vec_mean = sum(vec) / len(vec)
                        vec_var = sum((x - vec_mean) ** 2 for x in vec) / len(vec)
                        
                        # Normalize with epsilon for numerical stability
                        normalized_vec = [(x - vec_mean) / math.sqrt(vec_var + epsilon) 
                                         for x in vec]
                        
                        normalized_linear_out.append(normalized_vec)
                        mean_cache.append(vec_mean)
                        var_cache.append(vec_var)
                        orig_cache.append(vec)  # Save original for backprop
                    
                    layer_intermediate['linear_out'] = linear_out
                    layer_intermediate['normalized_linear_out'] = normalized_linear_out
                    layer_intermediate['mean_cache'] = mean_cache
                    layer_intermediate['var_cache'] = var_cache
                    layer_intermediate['orig_cache'] = orig_cache
                    
                    # Apply position-dependent transformation to normalized output
                    T_matrix = []
                    phi_vals_list = []
                    for k, vec in enumerate(normalized_linear_out):
                        out_dim = len(layer['P'])
                        Tk = [0.0] * out_dim
                        phi_vals = {}
                        for i in range(out_dim):
                            for j in range(out_dim):
                                for g in range(len(layer['P'][i][j])):
                                    period = layer['periods'][i][j][g]
                                    phi = math.cos(2 * math.pi * k / period)
                                    phi_vals[(i, j, g)] = phi
                                    Tk[i] += layer['P'][i][j][g] * vec[j] * phi
                        T_matrix.append(Tk)
                        phi_vals_list.append(phi_vals)
                    
                    layer_intermediate['T_matrix'] = T_matrix
                    layer_intermediate['phi_vals'] = phi_vals_list
                    
                    # Convert to matrix form: rows=features, columns=time steps
                    T_transposed = [list(col) for col in zip(*T_matrix)]
                    layer_intermediate['T_transposed'] = T_transposed
                    
                    # Apply sequence length transformation: U = T * Linker
                    U_matrix = self._mat_mul(T_transposed, layer['Linker'])
                    layer_intermediate['U_matrix'] = U_matrix
                    
                    # Convert back to sequence of vectors for next layer
                    current = []
                    for j in range(len(U_matrix[0])):
                        vec = [U_matrix[i][j] for i in range(len(U_matrix))]
                        current.append(vec)
                    
                    intermediates.append(layer_intermediate)
                
                # Backward pass through layers
                # Start with last layer's output gradient
                d_out = None
                for l in range(self.num_layers-1, -1, -1):
                    layer = self.layers[l]
                    interm = intermediates[l]
                    T_matrix = interm['T_matrix']
                    T_transposed = interm['T_transposed']
                    phi_vals_list = interm['phi_vals']
                    linear_out = interm['linear_out']
                    normalized_linear_out = interm['normalized_linear_out']
                    mean_cache = interm['mean_cache']
                    var_cache = interm['var_cache']
                    orig_cache = interm['orig_cache']
                    
                    # For last layer: error is 2*(output - target) for each position
                    if l == self.num_layers-1:
                        # Convert output sequence to matrix
                        output_matrix = interm['U_matrix']
                        dU = []
                        for i in range(len(output_matrix)):
                            row = []
                            for j in range(len(output_matrix[0])):
                                # Gradient: dL/dU_ij = 2*(U_ij - t_j[i]) / num_positions
                                # But note: t_vec is the same for all positions
                                error = 2 * (output_matrix[i][j] - t_vec[i]) / total_positions
                                row.append(error)
                                total_loss += (output_matrix[i][j] - t_vec[i])**2
                            dU.append(row)
                    else:
                        # Propagate gradient from next layer
                        dU = d_next
                    
                    # Gradient for Linker matrix: dL/dLinker = T^T * dU
                    dLinker = self._mat_mul(self._transpose(T_transposed), dU)
                    
                    # Gradient for T: dL/dT = dU * Linker^T
                    Linker_T = self._transpose(layer['Linker'])
                    dT_mat = self._mat_mul(dU, Linker_T)
                    
                    # Convert dT from matrix form to list of vectors
                    dT = [list(col) for col in zip(*dT_mat)]
                    
                    # Gradient for position-dependent transformation
                    d_normalized = [[0.0] * len(normalized_linear_out[0]) for _ in range(len(normalized_linear_out))]
                    for k, vec in enumerate(normalized_linear_out):
                        dTk = dT[k]
                        phi_vals = phi_vals_list[k]
                        
                        # Compute gradients for P and normalized linear output
                        for i in range(len(dTk)):
                            for j in range(len(layer['P'][i])):
                                for g in range(len(layer['P'][i][j])):
                                    # Gradient for P[i][j][g]
                                    grad_P_list[l][i][j][g] += (
                                        dTk[i] * vec[j] * phi_vals[(i, j, g)]
                                    )
                                    
                                    # Gradient for normalized linear output
                                    d_normalized[k][j] += (
                                        dTk[i] * layer['P'][i][j][g] * phi_vals[(i, j, g)]
                                    )
                    
                    # Backpropagate through Layer Normalization
                    d_linear = [[0.0] * len(linear_out[0]) for _ in range(len(linear_out))]
                    for k in range(len(d_normalized)):
                        d_normalized_vec = d_normalized[k]
                        orig_vec = orig_cache[k]
                        vec_mean = mean_cache[k]
                        vec_var = var_cache[k]
                        n = len(orig_vec)  # Number of features
                        
                        # Gradient w.r.t variance
                        d_var = 0.0
                        for i in range(n):
                            # dL/dvar = dL/dnormalized_i * dnormalized_i/dvar
                            d_var += d_normalized_vec[i] * (orig_vec[i] - vec_mean) * \
                                     (-0.5) * (vec_var + epsilon) ** (-1.5)
                        
                        # Gradient w.r.t mean
                        d_mean = 0.0
                        for i in range(n):
                            # dL/dmean = dL/dnormalized_i * dnormalized_i/dmean
                            d_mean += d_normalized_vec[i] * \
                                      (-1.0 / math.sqrt(vec_var + epsilon))
                            
                            # Additional term from variance gradient
                            d_mean += d_var * (2.0 / n) * (orig_vec[i] - vec_mean) * (-1.0)
                        
                        # Gradient w.r.t original linear output
                        for i in range(n):
                            # Term 1: from normalized value
                            term1 = d_normalized_vec[i] / math.sqrt(vec_var + epsilon)
                            
                            # Term 2: from mean gradient
                            term2 = d_mean * (1.0 / n)
                            
                            # Term 3: from variance gradient
                            term3 = d_var * (2.0 / n) * (orig_vec[i] - vec_mean)
                            
                            d_linear[k][i] = term1 + term2 + term3
                    
                    # Gradient for M matrix
                    for k, vec in enumerate(current if l > 0 else seq):
                        for i in range(len(d_linear[k])):
                            for j in range(len(vec)):
                                grad_M_list[l][i][j] += d_linear[k][i] * vec[j] / total_positions
                    
                    # Propagate gradient to previous layer
                    if l > 0:
                        # Gradient for input to linear transformation
                        d_input = []
                        for k, d_lin_vec in enumerate(d_linear):
                            d_vec = self._mat_vec(self._transpose(layer['M']), d_lin_vec)
                            d_input.append(d_vec)
                        
                        # Convert to matrix form for next lower layer
                        d_next = [list(col) for col in zip(*d_input)]
                    
                    # Accumulate Linker gradient if trainable
                    if layer['linker_trainable']:
                        for i in range(len(grad_Linker_list[l])):
                            for j in range(len(grad_Linker_list[l][i])):
                                grad_Linker_list[l][i][j] += dLinker[i][j]
            
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
                
                # Update Linker matrix if trainable
                if layer['linker_trainable']:
                    for i in range(len(layer['Linker'])):
                        for j in range(len(layer['Linker'][i])):
                            layer['Linker'][i][j] -= learning_rate * grad_Linker_list[l][i][j]
            
            # Calculate current loss
            current_D = total_loss / total_positions if total_positions else 0.0
            history.append(current_D)
            
            # Print progress
            if it % print_every == 0 or it == max_iters - 1:
                print(f"GD Iter {it:3d}: D = {current_D:.6e}, LR = {learning_rate:.6f}")
            
            # Check convergence
            if abs(D_prev - current_D) < tol:
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
        Returns the average of all output vectors in the sequence
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
        default_attrs = ['config', 'M', 'P', 'periods', 'Linker']
        
        # Handle different what parameter types
        if what is None:
            attrs = default_attrs
        elif what == 'all':
            attrs = default_attrs
        elif isinstance(what, str):
            attrs = [what]
        else:
            attrs = what
            
        print("Hierarchical HierDDLpm Status")
        print("=" * 50)
        
        # Display each requested attribute
        for attr in attrs:
            if attr == 'config':
                print("\n[Configuration]")
                print(f"{'Input dim:':<20} {self.input_dim}")
                print(f"{'Input seq len:':<20} {self.input_seq_len}")
                print(f"{'Layer dims:':<20} {self.model_dims}")
                print(f"{'Num basis:':<20} {self.num_basis_list}")
                print(f"{'Linker dims:':<20} {self.linker_dims}")
                print(f"{'Linker trainable:':<20} {self.linker_trainable}")
                print(f"{'Trained:':<20} {self.trained}")
            
            elif attr in ['M', 'P', 'periods', 'Linker']:
                print(f"\n[{attr.upper()} Matrices/Tensors]")
                for l in range(self.num_layers):
                    print(f"  Layer {l}:")
                    if attr == 'M':
                        M = self.layers[l]['M']
                        print(f"    Shape: {len(M)}×{len(M[0])}")
                        print("    Sample values:")
                        for i in range(min(first_num, len(M))):
                            vals = M[i][:min(first_num, len(M[i]))]
                            print(f"      Row {i}: [{', '.join(f'{x:.4f}' for x in vals)}" + 
                                  (f", ...]" if len(M[i]) > first_num else "]"))
                    
                    elif attr in ['P', 'periods']:
                        tensor = self.layers[l][attr]
                        print(f"    Shape: {len(tensor)}×{len(tensor[0])}×{len(tensor[0][0])}")
                        print("    Sample slices:")
                        for i in range(min(first_num, len(tensor))):
                            for j in range(min(first_num, len(tensor[i]))):
                                vals = tensor[i][j][:min(first_num, len(tensor[i][j]))]
                                if attr == 'P':
                                    formatted = [f"{v:.6f}" for v in vals]
                                else:
                                    formatted = [str(int(v)) for v in vals]
                                print(f"      {attr}[{i}][{j}][:]: [{', '.join(formatted)}" + 
                                      (f", ...]" if len(tensor[i][j]) > first_num else "]"))
                    
                    elif attr == 'Linker':
                        Linker = self.layers[l]['Linker']
                        print(f"    Shape: {len(Linker)}×{len(Linker[0])}")
                        print(f"    Trainable: {self.layers[l]['linker_trainable']}")
                        print("    Sample values:")
                        for i in range(min(first_num, len(Linker))):
                            vals = Linker[i][:min(first_num, len(Linker[i]))]
                            print(f"      Row {i}: [{', '.join(f'{x:.4f}' for x in vals)}" + 
                                  (f", ...]" if len(Linker[i]) > first_num else "]"))
            
            else:
                print(f"\n[Unknown attribute: {attr}]")
        
        print("=" * 50)

    def count_parameters(self):
        """Count learnable parameters (P tensors, M matrices, and Linker matrices)"""
        total_params = 0
        trainable_params = 0
        print("Parameter Count:")
        
        for l in range(self.num_layers):
            M = self.layers[l]['M']
            P = self.layers[l]['P']
            Linker = self.layers[l]['Linker']
            
            M_params = len(M) * len(M[0])
            P_params = len(P) * len(P[0]) * len(P[0][0])
            Linker_params = len(Linker) * len(Linker[0])
            
            layer_params = M_params + P_params + Linker_params
            total_params += layer_params
            
            # Count trainable parameters
            layer_trainable = M_params + P_params
            if self.layers[l]['linker_trainable']:
                layer_trainable += Linker_params
            trainable_params += layer_trainable
            
            print(f"  Layer {l}:")
            print(f"    M matrix: {len(M)}×{len(M[0])} = {M_params} parameters")
            print(f"    P tensor: {len(P)}×{len(P[0])}×{len(P[0][0])} = {P_params} parameters")
            print(f"    Linker matrix: {len(Linker)}×{len(Linker[0])} = {Linker_params} parameters")
            print(f"    Linker trainable: {self.layers[l]['linker_trainable']}")
            print(f"    Layer total: {layer_params} (trainable: {layer_trainable})")
        
        print(f"Total parameters: {total_params}")
        print(f"Trainable parameters: {trainable_params}")
        return total_params, trainable_params

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
    input_dim = 10          # Input vector dimension
    model_dims = [8, 6, 3]  # Output dimensions for each layer
    num_basis_list = [5, 4, 3]  # Basis functions per layer
    input_seq_len = 100     # Fixed input sequence length
    linker_dims = [50, 20, 10]  # Output sequence lengths for each layer
    num_seqs = 10           # Number of training sequences
    
    # Generate synthetic training data
    print(f"Generating {num_seqs} sequences with length {input_seq_len}...")
    seqs = []     # List of sequences (each sequence: list of n-dim vectors)
    t_list = []   # List of target vectors (m-dim)
    
    for _ in range(num_seqs):
        # Generate input_dim-dimensional input sequence (fixed length)
        seq = [[random.uniform(-1,1) for _ in range(input_dim)] 
               for __ in range(input_seq_len)]
        seqs.append(seq)
        # Generate model_dims[-1]-dimensional target vector
        t_list.append([random.uniform(-1,1) for _ in range(model_dims[-1])])

    # Test Case 1: All Linker matrices trainable
    print("\n=== Test Case 1: All Linker Matrices Trainable ===")
    hndd_trainable = HierDDLpm(
        input_dim=input_dim,
        model_dims=model_dims,
        num_basis_list=num_basis_list,
        input_seq_len=input_seq_len,
        linker_dims=linker_dims,
        linker_trainable=True  # All Linker matrices trainable
    )
    
    # Gradient Descent Training
    print("\nTraining with Gradient Descent:")
    grad_history = hndd_trainable.grad_train(
        seqs, 
        t_list, 
        max_iters=200,
        learning_rate=2.0,
        decay_rate=0.99,
        print_every=5
    )
    
    # Predict targets
    print("\nPredictions (first 3 sequences):")
    for i, seq in enumerate(seqs[:3]):
        t_pred = hndd_trainable.predict_t(seq)
        print(f"Seq {i+1}: Target={[f'{x:.4f}' for x in t_list[i]]}")
        print(f"         Predicted={[f'{x:.4f}' for x in t_pred]}")
    
    # Calculate prediction correlations
    preds = [hndd_trainable.predict_t(seq) for seq in seqs]
    correlations = []
    for i in range(model_dims[-1]):
        actual = [t[i] for t in t_list]
        predicted = [p[i] for p in preds]
        corr = correlation(actual, predicted)
        correlations.append(corr)
        print(f"Output dim {i} correlation: {corr:.4f}")
    print(f"Average correlation: {mean(correlations):.4f}")

    # Parameter count
    print("\nParameter count:")
    total_params, trainable_params = hndd_trainable.count_parameters()
    
    # Test Case 2: Mixed trainability (first layer trainable, others not)
    print("\n\n=== Test Case 2: Mixed Linker Trainability ===")
    hndd_mixed = HierDDLpm(
        input_dim=input_dim,
        model_dims=model_dims,
        num_basis_list=num_basis_list,
        input_seq_len=input_seq_len,
        linker_dims=linker_dims,
        linker_trainable=[False, True, False]  # Per-layer control
    )
    
    print("\nTraining with Gradient Descent:")
    grad_history_mixed = hndd_mixed.grad_train(
        seqs, 
        t_list, 
        max_iters=200,
        learning_rate=2.0,
        decay_rate=0.99,
        print_every=5
    )
    
    # Calculate prediction correlations for mixed case
    preds_mixed = [hndd_mixed.predict_t(seq) for seq in seqs]
    correlations_mixed = []
    for i in range(model_dims[-1]):
        actual = [t[i] for t in t_list]
        predicted = [p[i] for p in preds_mixed]
        corr = correlation(actual, predicted)
        correlations_mixed.append(corr)
        print(f"Output dim {i} correlation: {corr:.4f}")
    print(f"Average correlation: {mean(correlations_mixed):.4f}")

    # Parameter count for mixed case
    print("\nParameter count (mixed trainability):")
    total_params_mixed, trainable_params_mixed = hndd_mixed.count_parameters()
    
    # Test Case 3: No Linker matrices trainable
    print("\n\n=== Test Case 3: No Linker Matrices Trainable ===")
    hndd_fixed = HierDDLpm(
        input_dim=input_dim,
        model_dims=model_dims,
        num_basis_list=num_basis_list,
        input_seq_len=input_seq_len,
        linker_dims=linker_dims,
        linker_trainable=False  # No Linker matrices trainable
    )
    
    print("\nTraining with Gradient Descent:")
    grad_history_fixed = hndd_fixed.grad_train(
        seqs, 
        t_list, 
        max_iters=200,
        learning_rate=2.0,
        decay_rate=0.99,
        print_every=5
    )
    
    # Calculate prediction correlations for fixed Linker case
    preds_fixed = [hndd_fixed.predict_t(seq) for seq in seqs]
    correlations_fixed = []
    for i in range(model_dims[-1]):
        actual = [t[i] for t in t_list]
        predicted = [p[i] for p in preds_fixed]
        corr = correlation(actual, predicted)
        correlations_fixed.append(corr)
        print(f"Output dim {i} correlation: {corr:.4f}")
    print(f"Average correlation: {mean(correlations_fixed):.4f}")

    # Parameter count for fixed case
    print("\nParameter count (fixed Linkers):")
    total_params_fixed, trainable_params_fixed = hndd_fixed.count_parameters()
    
    # Save and load model
    print("\nTesting save/load functionality:")
    hndd_trainable.save("hierarchical_ndd_model.pkl")
    loaded = HierDDLpm.load("hierarchical_ndd_model.pkl")
    
    # Verify loaded model
    t_pred_loaded = loaded.predict_t(seqs[0])
    print(f"Prediction from loaded model: {[f'{x:.4f}' for x in t_pred_loaded]}")
