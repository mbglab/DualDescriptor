# Copyright (C) 2005-2025, Bin-Guang Ma (mbg@mail.hzau.edu.cn). All rights reserved.
# The Hierarchical Numeric Dual Descriptor class with Linker Matrices for Sequence Length Transformation
# Modified to support hierarchical structure with multiple layers and sequence length transformation
# Author: Bin-Guang Ma; Date: 2025-7-7

import math
import random
import pickle

class HierDDLab:
    """
    Hierarchical Numeric Dual Descriptor with Linker for vector sequences with:
      - input_dim: dimension of input vectors
      - model_dims: list of output dimensions for each layer
      - basis_dims: list of basis dimensions for each layer
      - linker_dims: list of output sequence lengths for each layer
      - linker_trainable: controls whether Linker matrices are trainable
    Each layer contains:
      - Matrix M ∈ R^{m_i×m_{i-1}} for linear transformation
      - Matrix Acoeff ∈ R^{m_i×L_i} of coefficients
      - Matrix Bbasis ∈ R^{L_i×m_i} of basis functions
      - Linker matrix for sequence length transformation
    """
    def __init__(self, input_dim=4, model_dims=[4], basis_dims=[50], 
                 linker_dims=[50], input_seq_len=100, linker_trainable=False):
        """
        Initialize the hierarchical Dual Descriptor with Linker matrices.
        
        Args:
            input_dim (int): Input vector dimension
            model_dims (list): List of output dimensions for each layer
            basis_dims (list): List of basis dimensions for each layer
            linker_dims (list): List of output sequence lengths for each layer
            input_seq_len (int): Fixed input sequence length
            linker_trainable (bool|list): Controls if Linker matrices are trainable
        """
        self.input_dim = input_dim
        self.model_dims = model_dims
        self.basis_dims = basis_dims
        self.linker_dims = linker_dims
        self.input_seq_len = input_seq_len
        self.num_layers = len(model_dims)
        self.trained = False
        
        # Validate dimensions
        if len(basis_dims) != self.num_layers:
            raise ValueError("basis_dims length must match model_dims length")
        if len(linker_dims) != self.num_layers:
            raise ValueError("linker_dims length must match model_dims length")
        
        # Process linker_trainable parameter
        if isinstance(linker_trainable, bool):
            self.linker_trainable = [linker_trainable] * self.num_layers
        elif isinstance(linker_trainable, list):
            if len(linker_trainable) != self.num_layers:
                raise ValueError("linker_trainable list length must match number of layers")
            self.linker_trainable = linker_trainable
        else:
            raise TypeError("linker_trainable must be bool or list of bools")
        
        self.layers = []
        
        # Initialize each layer
        for i, out_dim in enumerate(model_dims):
            # Determine input dimensions for this layer
            if i == 0:
                in_dim = input_dim
                in_seq = input_seq_len
            else:
                in_dim = model_dims[i-1]
                in_seq = linker_dims[i-1]
                
            L_i = basis_dims[i]  # Basis dimension for this layer
            out_seq = linker_dims[i]  # Output sequence length for this layer
            
            # Initialize transformation matrix M (out_dim × in_dim)
            M = [[random.uniform(-0.1, 0.1) for _ in range(in_dim)]
                 for _ in range(out_dim)]
            
            # Initialize coefficient matrix Acoeff (out_dim × L_i)
            Acoeff = [[random.uniform(-0.1, 0.1) for _ in range(L_i)]
                       for _ in range(out_dim)]
            
            # Initialize basis matrix Bbasis (L_i × out_dim)
            Bbasis = [[random.uniform(-0.1, 0.1) for _ in range(out_dim)]
                       for _ in range(L_i)]
            
            # Initialize Linker matrix (in_seq × out_seq)
            Linker = [[random.uniform(-0.1, 0.1) for _ in range(out_seq)]
                      for _ in range(in_seq)]
            
            self.layers.append({
                'M': M,
                'Acoeff': Acoeff,
                'Bbasis': Bbasis,
                'Linker': Linker,
                'linker_trainable': self.linker_trainable[i]
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
    
    def _vec_avg(self, vectors):
        """Compute element-wise average of vectors"""
        if not vectors:
            return []
        dim = len(vectors[0])
        avg = [0.0] * dim
        for vec in vectors:
            for i in range(dim):
                avg[i] += vec[i]
        return [x / len(vectors) for x in avg]

    # ---- describe sequence ----
    def describe(self, vec_seq):
        """Compute descriptor vectors for each position in all layers with sequence length transformation"""
        # Convert sequence to matrix: input_dim × input_seq_len
        # Each column is a feature vector
        X = [list(col) for col in zip(*vec_seq)]  # Transpose: input_dim rows, input_seq_len columns
        
        # Process through each layer
        for l_idx, layer in enumerate(self.layers):
            M = layer['M']
            Acoeff = layer['Acoeff']
            Bbasis = layer['Bbasis']
            Linker = layer['Linker']
            L_i = len(Bbasis)  # Basis dimension for this layer
            out_dim = len(Acoeff)  # Output dimension for this layer
            
            # Step 1: Apply M transformation to input matrix
            T = self._mat_mul(M, X)  # Feature transformation: out_dim × in_seq
            
            # Step 2: Compute intermediate matrix U (out_dim × in_seq)
            U = []
            for k in range(len(T[0])):  # For each column in T
                vec = [T[i][k] for i in range(len(T))]  # Get column vector
                j = k % L_i  # Basis index
                scalar = self._dot(Bbasis[j], vec)
                U_col = [Acoeff[i][j] * scalar for i in range(out_dim)]
                U.append(U_col)
            
            # Convert U to matrix: out_dim × in_seq (each column is a vector)
            U_mat = [list(col) for col in zip(*U)]
            
            # Step 3: Apply sequence length transformation using Linker matrix
            V = self._mat_mul(U_mat, Linker)  # out_dim × out_seq
            
            # Output becomes input for next layer
            X = V
        
        # Convert final output matrix to list of vectors
        output_vectors = []
        for j in range(len(X[0])):
            vec = [X[i][j] for i in range(len(X))]
            output_vectors.append(vec)
            
        return output_vectors

    # ---- compute deviation ----
    def deviation(self, seqs, t_list):
        """Compute mean squared error between descriptors and targets"""
        tot = 0.0
        cnt = 0
        for seq, t in zip(seqs, t_list):
            # Get final layer outputs
            outputs = self.describe(seq)
            # Compute average vector
            avg_vec = self._vec_avg(outputs)
            # Calculate error
            err = self._vec_sub(avg_vec, t)
            tot += self._dot(err, err)
            cnt += 1
        return tot / cnt if cnt else 0.0

    def grad_train(self, seqs, t_list, max_iters=1000, tol=1e-8, 
               learning_rate=0.01, continued=False, 
               decay_rate=1.0, print_every=10, epsilon=1e-88):
        """
        Train using gradient descent with backpropagation through layers.
        Supports learning M, Acoeff, Bbasis, and Linker matrices.
        Rolls back parameters when loss increases during training.
        Added Layer Normalization after M transformation for better gradient scaling.
        """
        if not continued:
            # Reinitialize parameters
            self.__init__(self.input_dim, self.model_dims, self.basis_dims,
                          self.linker_dims, self.input_seq_len, self.linker_trainable)
        
        total_sequences = len(seqs)
        if total_sequences == 0:
            raise ValueError("No sequences provided")
        
        history = []
        D_prev = float('inf')
        current_lr = learning_rate
        
        # Initialize gradient storage for all layers
        grad_A_list = []
        grad_B_list = []
        grad_M_list = []
        grad_Linker_list = []
        for layer in self.layers:
            out_dim = len(layer['Acoeff'])
            L_i = len(layer['Bbasis'])
            in_dim = len(layer['M'][0])
            in_seq = len(layer['Linker'])
            out_seq = len(layer['Linker'][0])
            
            grad_A = [[0.0] * L_i for _ in range(out_dim)]
            grad_B = [[0.0] * out_dim for _ in range(L_i)]
            grad_M = [[0.0] * in_dim for _ in range(out_dim)]
            grad_Linker = [[0.0] * out_seq for _ in range(in_seq)]
            
            grad_A_list.append(grad_A)
            grad_B_list.append(grad_B)
            grad_M_list.append(grad_M)
            grad_Linker_list.append(grad_Linker)
        
        # Variable to store previous parameters for rollback
        prev_params = None
        
        for it in range(max_iters):
            # Save current parameters before update
            prev_params = []
            for layer in self.layers:
                layer_params = {
                    'Acoeff': [row[:] for row in layer['Acoeff']],
                    'Bbasis': [row[:] for row in layer['Bbasis']],
                    'M': [row[:] for row in layer['M']],
                    'Linker': [row[:] for row in layer['Linker']]
                }
                prev_params.append(layer_params)
            
            # Reset gradients
            for l_idx in range(self.num_layers):
                for i in range(len(grad_A_list[l_idx])):
                    grad_A_list[l_idx][i] = [0.0] * len(grad_A_list[l_idx][i])
                for i in range(len(grad_B_list[l_idx])):
                    grad_B_list[l_idx][i] = [0.0] * len(grad_B_list[l_idx][i])
                for i in range(len(grad_M_list[l_idx])):
                    grad_M_list[l_idx][i] = [0.0] * len(grad_M_list[l_idx][i])
                for i in range(len(grad_Linker_list[l_idx])):
                    grad_Linker_list[l_idx][i] = [0.0] * len(grad_Linker_list[l_idx][i])
            
            total_loss = 0.0
            
            # Process each sequence
            for seq, t in zip(seqs, t_list):
                # Forward pass: store intermediate results
                forward_cache = []
                X = [list(col) for col in zip(*seq)]  # Convert to matrix: input_dim × input_seq_len
                
                for l_idx, layer in enumerate(self.layers):
                    M = layer['M']
                    Acoeff = layer['Acoeff']
                    Bbasis = layer['Bbasis']
                    Linker = layer['Linker']
                    L_i = len(Bbasis)
                    out_dim = len(Acoeff)
                    
                    # Apply M transformation
                    T = self._mat_mul(M, X)  # out_dim × in_seq
                    
                    # NEW: Apply Layer Normalization to T
                    # For each position (column) in the sequence
                    T_normalized = []
                    mean_cache = []
                    var_cache = []
                    for col_idx in range(len(T[0])):
                        # Extract feature vector for this position
                        vec = [T[i][col_idx] for i in range(out_dim)]
                        
                        # Compute mean and variance
                        vec_mean = mean(vec)
                        vec_var = sum((x - vec_mean) ** 2 for x in vec) / len(vec)
                        
                        # Normalize with epsilon for numerical stability
                        normalized_vec = [(x - vec_mean) / math.sqrt(vec_var + epsilon) 
                                         for x in vec]
                        
                        T_normalized.append(normalized_vec)
                        mean_cache.append(vec_mean)
                        var_cache.append(vec_var)
                    
                    # Convert back to matrix format: out_dim × in_seq
                    T_norm_mat = [list(col) for col in zip(*T_normalized)]
                    
                    # Compute intermediate U matrix using normalized T
                    U = []
                    intermediate_vals = []
                    for k in range(len(T_norm_mat[0])):
                        vec = [T_norm_mat[i][k] for i in range(len(T_norm_mat))]
                        j = k % L_i
                        scalar = self._dot(Bbasis[j], vec)
                        U_col = [Acoeff[i][j] * scalar for i in range(out_dim)]
                        U.append(U_col)
                        intermediate_vals.append({
                            'vec': vec,          # Normalized input vector
                            'original_vec': [T[i][k] for i in range(len(T))],  # Original vector for backprop
                            'j': j,              # Basis index
                            'scalar': scalar,
                            'U_col': U_col,       # Output vector
                            'mean': mean_cache[k], # Saved mean for backprop
                            'var': var_cache[k]    # Saved variance for backprop
                        })
                    
                    # Convert U to matrix: out_dim × in_seq
                    U_mat = [list(col) for col in zip(*U)]
                    
                    # Apply Linker transformation
                    V = self._mat_mul(U_mat, Linker)  # out_dim × out_seq
                    
                    # Save intermediate results
                    forward_cache.append({
                        'X': X,             # Input matrix
                        'T': T,             # After M transformation (unnormalized)
                        'T_norm': T_norm_mat, # After normalization
                        'intermediate_vals': intermediate_vals,  # Position-wise values
                        'U_mat': U_mat,      # After position processing
                        'V': V              # After Linker transformation
                    })
                    
                    # Output becomes input for next layer
                    X = V
                
                # Final output matrix
                final_output = forward_cache[-1]['V']
                
                # Convert to list of vectors and compute average
                output_vectors = []
                for j in range(len(final_output[0])):
                    vec = [final_output[i][j] for i in range(len(final_output))]
                    output_vectors.append(vec)
                avg_vec = self._vec_avg(output_vectors)
                
                # Compute error and loss
                err = self._vec_sub(avg_vec, t)
                loss = self._dot(err, err)
                total_loss += loss
                
                # Backward pass: start from final layer
                # Gradient of loss w.r.t avg_vec is 2*(avg_vec - t)
                d_avg = [2 * e for e in err]
                
                # Each output vector contributes equally to the average
                d_output_vectors = [d_avg[:] for _ in output_vectors]
                
                # Convert to matrix gradient (dL/dV)
                dV = [[0.0] * len(final_output[0]) for _ in range(len(final_output))]
                for j in range(len(d_output_vectors)):
                    for i in range(len(d_output_vectors[j])):
                        dV[i][j] = d_output_vectors[j][i]
                
                # Backpropagate through layers
                for l_idx in range(self.num_layers-1, -1, -1):
                    layer_cache = forward_cache[l_idx]
                    layer = self.layers[l_idx]
                    M = layer['M']
                    Acoeff = layer['Acoeff']
                    Bbasis = layer['Bbasis']
                    Linker = layer['Linker']
                    L_i = len(Bbasis)
                    out_dim = len(Acoeff)
                    
                    X = layer_cache['X']         # Input matrix
                    T = layer_cache['T']         # After M transformation (unnormalized)
                    T_norm = layer_cache['T_norm'] # Normalized T
                    U_mat = layer_cache['U_mat']  # After position processing
                    V = layer_cache['V']         # After Linker transformation
                    intermediate_vals = layer_cache['intermediate_vals']
                    
                    # Gradient w.r.t Linker: dL/dLinker = U_mat^T * dV
                    dLinker = self._mat_mul(self._transpose(U_mat), dV)
                    
                    # Gradient w.r.t U_mat: dL/dU_mat = dV * Linker^T
                    dU_mat = self._mat_mul(dV, self._transpose(Linker))
                    
                    # Convert dU_mat to list of column gradients
                    dU_cols = []
                    for j in range(len(dU_mat[0])):
                        col = [dU_mat[i][j] for i in range(len(dU_mat))]
                        dU_cols.append(col)
                    
                    # Initialize gradients for position processing
                    dT_norm = [[0.0] * len(T_norm[0]) for _ in range(len(T_norm))]  # dL/dT_norm
                    dAcoeff = [[0.0] * len(Acoeff[0]) for _ in range(len(Acoeff))]
                    dBbasis = [[0.0] * len(Bbasis[0]) for _ in range(len(Bbasis))]
                    dM = [[0.0] * len(M[0]) for _ in range(len(M))]
                    
                    # Backpropagate through position processing
                    for k in range(len(intermediate_vals)):
                        vals = intermediate_vals[k]
                        j = vals['j']
                        vec = vals['vec']        # Normalized input vector
                        orig_vec = vals['original_vec']  # Original unnormalized vector
                        scalar = vals['scalar']
                        U_col = vals['U_col']    # Output vector for this position
                        vec_mean = vals['mean']
                        vec_var = vals['var']
                        
                        dU_col = dU_cols[k]      # Gradient from next layer
                        
                        # Gradient w.r.t Acoeff
                        for i in range(len(Acoeff)):
                            dAcoeff[i][j] += dU_col[i] * scalar
                        
                        # Gradient w.r.t scalar
                        d_scalar = sum(dU_col[i] * Acoeff[i][j] for i in range(len(Acoeff)))
                        
                        # Gradient w.r.t Bbasis
                        for d in range(len(Bbasis[j])):
                            dBbasis[j][d] += d_scalar * vec[d]
                        
                        # Gradient w.r.t normalized vec (input to this position)
                        d_vec = [d_scalar * Bbasis[j][d] for d in range(len(Bbasis[j]))]
                        
                        # Accumulate gradient for normalized T matrix
                        for d in range(len(d_vec)):
                            dT_norm[d][k] += d_vec[d]
                    
                    # NEW: Backpropagate through Layer Normalization
                    dT = [[0.0] * len(T[0]) for _ in range(len(T))]  # dL/dT (unnormalized)
                    
                    # For each position (column) in the sequence
                    for k in range(len(intermediate_vals)):
                        vals = intermediate_vals[k]
                        vec_mean = vals['mean']
                        vec_var = vals['var']
                        orig_vec = vals['original_vec']
                        n = len(orig_vec)  # Number of features
                        
                        # Gradient w.r.t normalized vector for this position
                        d_normalized = [dT_norm[i][k] for i in range(len(dT_norm))]
                        
                        # Gradient w.r.t variance
                        d_var = 0.0
                        for i in range(n):
                            # dL/dvar = dL/dnormalized_i * dnormalized_i/dvar
                            d_var += d_normalized[i] * (orig_vec[i] - vec_mean) * \
                                     (-0.5) * (vec_var + epsilon) ** (-1.5)
                        
                        # Gradient w.r.t mean
                        d_mean = 0.0
                        for i in range(n):
                            # dL/dmean = dL/dnormalized_i * dnormalized_i/dmean
                            d_mean += d_normalized[i] * \
                                      (-1.0 / math.sqrt(vec_var + epsilon))
                            
                            # Additional term from variance gradient
                            d_mean += d_var * (2.0 / n) * (orig_vec[i] - vec_mean) * (-1.0)
                        
                        # Gradient w.r.t original vector
                        for i in range(n):
                            # Term 1: from normalized value
                            term1 = d_normalized[i] / math.sqrt(vec_var + epsilon)
                            
                            # Term 2: from mean gradient
                            term2 = d_mean * (1.0 / n)
                            
                            # Term 3: from variance gradient
                            term3 = d_var * (2.0 / n) * (orig_vec[i] - vec_mean)
                            
                            dT[i][k] = term1 + term2 + term3
                    
                    # Gradient w.r.t M: dL/dM = dT * X^T
                    dM_inc = self._mat_mul(dT, self._transpose(X))
                    for i in range(len(dM)):
                        for j in range(len(dM[i])):
                            dM[i][j] += dM_inc[i][j]
                    
                    # Gradient w.r.t X (for previous layer): dL/dX = M^T * dT
                    dX_prev = self._mat_mul(self._transpose(M), dT)
                    
                    # Accumulate gradients
                    for i in range(len(grad_A_list[l_idx])):
                        for j in range(len(grad_A_list[l_idx][i])):
                            grad_A_list[l_idx][i][j] += dAcoeff[i][j]
                    
                    for i in range(len(grad_B_list[l_idx])):
                        for j in range(len(grad_B_list[l_idx][i])):
                            grad_B_list[l_idx][i][j] += dBbasis[i][j]
                    
                    for i in range(len(grad_M_list[l_idx])):
                        for j in range(len(grad_M_list[l_idx][i])):
                            grad_M_list[l_idx][i][j] += dM[i][j]
                    
                    # Only accumulate Linker gradient if trainable
                    if layer['linker_trainable']:
                        for i in range(len(grad_Linker_list[l_idx])):
                            for j in range(len(grad_Linker_list[l_idx][i])):
                                grad_Linker_list[l_idx][i][j] += dLinker[i][j]
                    
                    # Prepare for previous layer
                    dV = dX_prev  # dV for next lower layer
            
            # Gradient clipping to prevent explosion
            max_grad_norm = 1.0
            for l_idx in range(self.num_layers):
                # Clip Acoeff gradients
                for i in range(len(grad_A_list[l_idx])):
                    for j in range(len(grad_A_list[l_idx][i])):
                        grad = grad_A_list[l_idx][i][j]
                        if abs(grad) > max_grad_norm:
                            grad = max_grad_norm if grad > 0 else -max_grad_norm
                        grad_A_list[l_idx][i][j] = grad / total_sequences
                
                # Clip Bbasis gradients
                for i in range(len(grad_B_list[l_idx])):
                    for j in range(len(grad_B_list[l_idx][i])):
                        grad = grad_B_list[l_idx][i][j]
                        if abs(grad) > max_grad_norm:
                            grad = max_grad_norm if grad > 0 else -max_grad_norm
                        grad_B_list[l_idx][i][j] = grad / total_sequences
                
                # Clip M gradients
                for i in range(len(grad_M_list[l_idx])):
                    for j in range(len(grad_M_list[l_idx][i])):
                        grad = grad_M_list[l_idx][i][j]
                        if abs(grad) > max_grad_norm:
                            grad = max_grad_norm if grad > 0 else -max_grad_norm
                        grad_M_list[l_idx][i][j] = grad / total_sequences
                
                # Clip Linker gradients (if trainable)
                if self.layers[l_idx]['linker_trainable']:
                    for i in range(len(grad_Linker_list[l_idx])):
                        for j in range(len(grad_Linker_list[l_idx][i])):
                            grad = grad_Linker_list[l_idx][i][j]
                            if abs(grad) > max_grad_norm:
                                grad = max_grad_norm if grad > 0 else -max_grad_norm
                            grad_Linker_list[l_idx][i][j] = grad / total_sequences
            
            # Calculate effective learning rate
            effective_lr = current_lr / total_sequences
            
            # Update parameters with effective learning rate
            for l_idx in range(self.num_layers):
                layer = self.layers[l_idx]
                
                # Update Acoeff
                for i in range(len(layer['Acoeff'])):
                    for j in range(len(layer['Acoeff'][i])):
                        layer['Acoeff'][i][j] -= effective_lr * grad_A_list[l_idx][i][j]
                
                # Update Bbasis
                for i in range(len(layer['Bbasis'])):
                    for j in range(len(layer['Bbasis'][i])):
                        layer['Bbasis'][i][j] -= effective_lr * grad_B_list[l_idx][i][j]
                
                # Update M
                for i in range(len(layer['M'])):
                    for j in range(len(layer['M'][i])):
                        layer['M'][i][j] -= effective_lr * grad_M_list[l_idx][i][j]
                
                # Update Linker if trainable
                if layer['linker_trainable']:
                    for i in range(len(layer['Linker'])):
                        for j in range(len(layer['Linker'][i])):
                            layer['Linker'][i][j] -= effective_lr * grad_Linker_list[l_idx][i][j]
            
            # Calculate average loss
            D = total_loss / total_sequences
            history.append(D)
            
            # Print progress
            if it % print_every == 0 or it == max_iters - 1:
                print(f"Iter {it:3d}: Loss = {D:.36f}, LR = {current_lr:.6f}")
            
            # Check convergence - roll back if loss increases
            if D >= D_prev - tol:
                # Roll back to previous parameters
                for l_idx, layer in enumerate(self.layers):
                    layer['Acoeff'] = [row[:] for row in prev_params[l_idx]['Acoeff']]
                    layer['Bbasis'] = [row[:] for row in prev_params[l_idx]['Bbasis']]
                    layer['M'] = [row[:] for row in prev_params[l_idx]['M']]
                    layer['Linker'] = [row[:] for row in prev_params[l_idx]['Linker']]
                
                print(f"Converged after {it+1} iterations (loss increased)")
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
        
        # Return average vector
        return self._vec_avg(outputs)

    # ---- show state ----
    def show(self, what=None, first_num=5):
        """Display hierarchical model state information"""
        if what is None:
            what = ['params', 'stats']
        elif isinstance(what, str):
            what = ['params', 'stats'] if what == 'all' else [what]
        
        print("Hierarchical HierDDLab with Linker Matrices - Model Status:")
        print("=" * 70)
        
        # Configuration parameters
        if 'params' in what:
            print("[Configuration Parameters]")
            print(f"  Input dim        : {self.input_dim}")
            print(f"  Input seq len    : {self.input_seq_len}")
            print(f"  Layer dims       : {self.model_dims}")
            print(f"  Basis dims       : {self.basis_dims}")
            print(f"  Linker dims      : {self.linker_dims}")
            print(f"  Linker trainable : {self.linker_trainable}")
            print(f"  Number of layers : {self.num_layers}")
            print(f"  Trained          : {self.trained}")
            print("=" * 70)
        
        # Layer details
        if 'layers' in what or 'all' in what:
            for l_idx, layer in enumerate(self.layers):
                print(f"\nLayer {l_idx}:")
                in_dim = self.input_dim if l_idx == 0 else self.model_dims[l_idx-1]
                in_seq = self.input_seq_len if l_idx == 0 else self.linker_dims[l_idx-1]
                out_dim = self.model_dims[l_idx]
                L_i = self.basis_dims[l_idx]
                out_seq = self.linker_dims[l_idx]
                
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
                
                # Show Linker matrix sample
                Linker = layer['Linker']
                print(f"  Linker matrix ({len(Linker)}x{len(Linker[0])}) [Trainable: {layer['linker_trainable']}]:")
                for i in range(min(first_num, len(Linker))):
                    row = Linker[i][:min(first_num, len(Linker[0]))]
                    print(f"    Row {i}: [{', '.join(f'{x:.4f}' for x in row)}" + 
                          (", ...]" if len(Linker[0]) > first_num else "]"))      

            print("=" * 70)

    def count_parameters(self):
        """
        Calculate and print the number of learnable parameters.
        """
        total_params = 0
        trainable_params = 0
        print("Model Parameter Counts:")
        
        for l_idx, layer in enumerate(self.layers):
            M = layer['M']
            Acoeff = layer['Acoeff']
            Bbasis = layer['Bbasis']
            Linker = layer['Linker']
            
            in_dim = self.input_dim if l_idx == 0 else self.model_dims[l_idx-1]
            in_seq = self.input_seq_len if l_idx == 0 else self.linker_dims[l_idx-1]
            out_dim = self.model_dims[l_idx]
            L_i = self.basis_dims[l_idx]
            out_seq = self.linker_dims[l_idx]
            
            m_params = len(M) * len(M[0])
            a_params = len(Acoeff) * len(Acoeff[0])
            b_params = len(Bbasis) * len(Bbasis[0])
            linker_params = len(Linker) * len(Linker[0])
            
            layer_params = m_params + a_params + b_params + linker_params
            total_params += layer_params
            
            layer_trainable = m_params + a_params + b_params
            if layer['linker_trainable']:
                layer_trainable += linker_params
            trainable_params += layer_trainable
            
            print(f"  Layer {l_idx} (in_dim: {in_dim}, out_dim: {out_dim}, L: {L_i}, in_seq: {in_seq}, out_seq: {out_seq}):")
            print(f"    M: {len(M)}×{len(M[0])} = {m_params} params")
            print(f"    Acoeff: {len(Acoeff)}×{len(Acoeff[0])} = {a_params} params")
            print(f"    Bbasis: {len(Bbasis)}×{len(Bbasis[0])} = {b_params} params")
            print(f"    Linker: {len(Linker)}×{len(Linker[0])} = {linker_params} params [Trainable: {layer['linker_trainable']}]")
            print(f"    Layer total: {layer_params} (Trainable: {layer_trainable})")
        
        print(f"Total parameters: {total_params}")
        print(f"Trainable parameters: {trainable_params}")        
        return total_params, trainable_params

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
    random.seed(2)
    
    # Hierarchical configuration with sequence length transformation
    input_dim = 9        # Input vector dimension
    input_seq_len = 100   # Fixed input sequence length
    model_dims = [6, 3]   # Output dimensions for each layer
    basis_dims = [100, 50]   # Basis dimensions for each layer
    linker_dims = [50, 20]  # Output sequence lengths for each layer
    seq_count = 30        # Number of training sequences
    
    # Generate training data with fixed sequence length
    print("Generating training data with fixed sequence length...")
    print(f"Input dimension: {input_dim}, Input seq length: {input_seq_len}")
    print(f"Layer dims: {model_dims}, Basis dims: {basis_dims}, Linker dims: {linker_dims}")
    
    seqs = []  # List of sequences
    t_list = []  # List of target vectors (dimension = last layer output)
    
    for i in range(seq_count):
        # Generate vector sequence with fixed length
        sequence = []
        for _ in range(input_seq_len):
            vector = [random.uniform(-1, 1) for _ in range(input_dim)]
            sequence.append(vector)
        seqs.append(sequence)
        
        # Generate random target vector (dimension = last layer output)
        target = [random.uniform(-1, 1) for _ in range(model_dims[-1])]
        t_list.append(target)
    
    # Create models with different Linker trainability settings
    print("\n=== Test Case 1: All Linker Matrices Trainable ===")
    hdd_trainable = HierDDLab(
        input_dim=input_dim,
        model_dims=model_dims,
        basis_dims=basis_dims,
        linker_dims=linker_dims,
        input_seq_len=input_seq_len,
        linker_trainable=True  # All Linker matrices trainable
    )
    
    # Show initial model structure
    print("\nInitial model structure (Trainable Linkers):")
    hdd_trainable.show('params')
    total_params, trainable_params = hdd_trainable.count_parameters()
    
    # Train with gradient descent
    print("\nTraining with Gradient Descent...")
    gd_history = hdd_trainable.grad_train(
        seqs,
        t_list,
        learning_rate=5,
        max_iters=200,
        tol=1e-36,
        decay_rate=0.99,
        print_every=10
    )
    
    # Evaluate predictions
    pred_t_list = [hdd_trainable.predict_t(seq) for seq in seqs]
    print("\nPrediction correlations per dimension:")
    corrs = []
    for d in range(model_dims[-1]):
        actuals = [t[d] for t in t_list]
        preds = [p[d] for p in pred_t_list]
        corr = correlation(actuals, preds)
        corrs.append(corr)
        print(f"  Dim {d}: correlation = {corr:.4f}")
    
    print(f"Average correlation: {mean(corrs):.4f}")
    
    # Test Case 2: Mixed Linker trainability
    print("\n\n=== Test Case 2: Mixed Linker Trainability ===")
    hdd_mixed = HierDDLab(
        input_dim=input_dim,
        model_dims=model_dims,
        basis_dims=basis_dims,
        linker_dims=linker_dims,
        input_seq_len=input_seq_len,
        linker_trainable=[True, False]  # First layer trainable, second not
    )
    
    print("\nTraining with Gradient Descent...")
    gd_history_mixed = hdd_mixed.grad_train(
        seqs,
        t_list,
        learning_rate=5,
        max_iters=200,
        tol=1e-36,
        decay_rate=0.99,
        print_every=10
    )
    
    # Evaluate predictions for mixed case
    pred_t_list_mixed = [hdd_mixed.predict_t(seq) for seq in seqs]
    print("\nPrediction correlations per dimension (Mixed):")
    corrs_mixed = []
    for d in range(model_dims[-1]):
        actuals = [t[d] for t in t_list]
        preds = [p[d] for p in pred_t_list_mixed]
        corr = correlation(actuals, preds)
        corrs_mixed.append(corr)
        print(f"  Dim {d}: correlation = {corr:.4f}")
    
    print(f"Average correlation: {mean(corrs_mixed):.4f}")
    
    # Test Case 3: No Linker matrices trainable
    print("\n\n=== Test Case 3: No Linker Matrices Trainable ===")
    hdd_fixed = HierDDLab(
        input_dim=input_dim,
        model_dims=model_dims,
        basis_dims=basis_dims,
        linker_dims=linker_dims,
        input_seq_len=input_seq_len,
        linker_trainable=False  # No Linker matrices trainable
    )
    
    print("\nTraining with Gradient Descent...")
    gd_history_fixed = hdd_fixed.grad_train(
        seqs,
        t_list,
        learning_rate=5,
        max_iters=200,
        tol=1e-36,
        decay_rate=0.99,
        print_every=10
    )
    
    # Evaluate predictions for fixed case
    pred_t_list_fixed = [hdd_fixed.predict_t(seq) for seq in seqs]
    print("\nPrediction correlations per dimension (Fixed Linkers):")
    corrs_fixed = []
    for d in range(model_dims[-1]):
        actuals = [t[d] for t in t_list]
        preds = [p[d] for p in pred_t_list_fixed]
        corr = correlation(actuals, preds)
        corrs_fixed.append(corr)
        print(f"  Dim {d}: correlation = {corr:.4f}")
    
    print(f"Average correlation: {mean(corrs_fixed):.4f}")
    
    # Save and load model
    print("\nTesting model persistence...")
    hdd_trainable.save("hierarchical_vector_model_linker.pkl")
    loaded = HierDDLab.load("hierarchical_vector_model_linker.pkl")
    print("Loaded model prediction for first sequence:")
    pred = loaded.predict_t(seqs[0])
    print(f"  Predicted target: {[round(p, 4) for p in pred]}")
    
    print("\n=== Hierarchical Vector Sequence Processing with Linker Matrices Demo Completed ===")
