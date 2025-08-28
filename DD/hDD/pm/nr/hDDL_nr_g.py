# Copyright (C) Bin-Guang Ma (mbg@mail.hzau.edu.cn). All rights reserved.
# The Numeric Dual Descriptor class (Tensor form) with hierarchical structure
# Modified to support sequence length transformation between layers using Linker matrices
# Added linker_trainable parameter to control Linker matrix training
# Added layer normalization and residual connections
# Added auto-training, reconstruction and generation methods
# Author: Bin-Guang Ma; Date: 2025-6-4

import math
import random
import pickle

class HierDDLpm:
    """
    Hierarchical Numeric Dual Descriptor for vector sequences with:
      - Multiple layers with linear transformation M and tensor P
      - Each layer: P ∈ R^{m×m} of basis coefficients, M ∈ R^{m×n}
      - Periods: period[i,j] = i*m + j + 2
      - Basis function: phi_{i,j}(k) = cos(2π * k / period[i,j])
    Key modifications:
    - Added Linker matrices for sequence length transformation
    - Layer operations: T = position-dependent transform, U = T * Linker
    - Added linker_trainable parameter to control Linker matrix training
    - Added layer normalization after linear transformation
    - Added residual connections with three modes: 'separate', 'shared', and None
    - Simplified P tensor from 3D to 2D matrix
    - Added auto-training, reconstruction and generation methods
    """
    def __init__(self, input_dim=2, model_dims=[2],
                 input_seq_len=100, linker_dims=[50], linker_trainable=False,
                 use_residual_list=None, epsilon=1e-8):
        """
        Initialize hierarchical HierDDLpm with sequence length transformation
        
        Args:
            input_dim (int): Input vector dimension
            model_dims (list): Output dimensions for each layer
            input_seq_len (int): Fixed input sequence length
            linker_dims (list): Output sequence lengths for each layer
            linker_trainable (bool or list): Trainability of Linker matrices
            use_residual_list (list): Residual connection type for each layer
            epsilon (float): Small constant for numerical stability in layer normalization
        """
        self.input_dim = input_dim
        self.model_dims = model_dims
        self.input_seq_len = input_seq_len
        self.linker_dims = linker_dims
        self.num_layers = len(model_dims)
        self.trained = False
        
        # Validate dimensions
        if len(linker_dims) != self.num_layers:
            raise ValueError("linker_dims must have same length as model_dims")
        
        # Process linker_trainable parameter
        if isinstance(linker_trainable, bool):
            self.linker_trainable = [linker_trainable] * self.num_layers
        elif isinstance(linker_trainable, list):
            if len(linker_trainable) != self.num_layers:
                raise ValueError("linker_trainable list length must match number of layers")
            self.linker_trainable = linker_trainable
        else:
            raise TypeError("linker_trainable must be bool or list of bools")
        
        # Process use_residual_list
        if use_residual_list is None:
            self.use_residual_list = [None] * self.num_layers
        elif isinstance(use_residual_list, list):
            if len(use_residual_list) != self.num_layers:
                raise ValueError("use_residual_list length must match number of layers")
            self.use_residual_list = use_residual_list
        else:
            raise TypeError("use_residual_list must be list or None")
        
        # Initialize layers with M, P, periods, Linker and epsilon
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
            use_residual = self.use_residual_list[l]
            
            # Initialize M matrix (out_dim x in_dim)
            M = [[random.uniform(-0.5, 0.5) for _ in range(in_dim)] 
                 for _ in range(out_dim)]
            
            # Initialize P matrix (out_dim x out_dim) - simplified to 2D
            P = [[random.uniform(-0.1, 0.1) for _ in range(out_dim)]
                 for _ in range(out_dim)]
            
            # Precompute periods matrix (2D instead of 3D)
            periods = [[i * out_dim + j + 2 for j in range(out_dim)]
                       for i in range(out_dim)]
            
            # Initialize Linker matrix (in_seq x out_seq)
            Linker = [[random.uniform(-0.1, 0.1) for _ in range(out_seq)] 
                      for _ in range(in_seq)]
            
            layer_dict = {
                'M': M,
                'P': P,
                'periods': periods,
                'Linker': Linker,
                'linker_trainable': self.linker_trainable[l],
                'use_residual': use_residual,
                'epsilon': epsilon  # Register epsilon as layer member
            }
            
            # Initialize separate residual parameters if using 'separate' mode
            if use_residual == 'separate':
                # Initialize separate residual projection matrix
                residual_proj = [[random.uniform(-0.5, 0.5) for _ in range(in_dim)] 
                                for _ in range(out_dim)]
                # Initialize separate residual Linker matrix
                residual_linker = [[random.uniform(-0.1, 0.1) for _ in range(out_seq)] 
                                  for _ in range(in_seq)]
                layer_dict['residual_proj'] = residual_proj
                layer_dict['residual_linker'] = residual_linker
            
            self.layers.append(layer_dict)

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
    
    def _vec_add(self, u, v):
        """Vector addition"""
        return [u_i + v_i for u_i, v_i in zip(u, v)]
    
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
    
    def _layer_norm(self, vec, epsilon):
        """Apply layer normalization to a vector with given epsilon"""
        # Compute mean and variance
        mean = sum(vec) / len(vec)
        variance = sum((x - mean) ** 2 for x in vec) / len(vec)
        # Normalize
        std_dev = math.sqrt(variance + epsilon)
        return [(x - mean) / std_dev for x in vec], mean, variance

    def _normalize_layer(self, vectors, epsilon):
        """
        Apply layer normalization to a sequence of vectors
        Returns normalized vectors, mean cache, and variance cache
        """
        normalized = []
        mean_cache = []
        var_cache = []
        
        for vec in vectors:
            norm_vec, mean_val, var_val = self._layer_norm(vec, epsilon)
            normalized.append(norm_vec)
            mean_cache.append(mean_val)
            var_cache.append(var_val)
            
        return normalized, mean_cache, var_cache

    def _clip_gradients(self, grad_matrix, threshold):
        """
        Clip gradients to prevent explosion
        grad_matrix: can be 2D matrix or 3D tensor
        threshold: maximum allowed gradient norm
        """
        if not grad_matrix:
            return
        
        # Flatten the gradients to compute norm
        if isinstance(grad_matrix[0], list):
            if isinstance(grad_matrix[0][0], list):  # 3D tensor
                flat_grads = []
                for i in range(len(grad_matrix)):
                    for j in range(len(grad_matrix[i])):
                        for g in range(len(grad_matrix[i][j])):
                            flat_grads.append(grad_matrix[i][j][g])
            else:  # 2D matrix
                flat_grads = [item for row in grad_matrix for item in row]
        else:
            flat_grads = grad_matrix
        
        # Compute norm
        grad_norm = math.sqrt(sum(g * g for g in flat_grads))
        
        # Clip if norm exceeds threshold
        if grad_norm > threshold:
            scale = threshold / (grad_norm + 1e-10)
            if isinstance(grad_matrix[0], list):
                if isinstance(grad_matrix[0][0], list):  # 3D tensor
                    for i in range(len(grad_matrix)):
                        for j in range(len(grad_matrix[i])):
                            for g in range(len(grad_matrix[i][j])):
                                grad_matrix[i][j][g] *= scale
                else:  # 2D matrix
                    for i in range(len(grad_matrix)):
                        for j in range(len(grad_matrix[i])):
                            grad_matrix[i][j] *= scale
            else:
                for i in range(len(grad_matrix)):
                    grad_matrix[i] *= scale

    def describe(self, seq):
        """Compute output vectors for each position in the sequence"""
        # Validate input sequence length
        if len(seq) != self.input_seq_len:
            raise ValueError(f"Input sequence length must be {self.input_seq_len}")
        
        current = seq
        for l, layer in enumerate(self.layers):
            # Store current input for residual connection
            layer_input = current
            
            # Apply linear transformation to each vector
            linear_out = [self._mat_vec(layer['M'], vec) for vec in layer_input]
            
            # Apply Layer Normalization to each position's vector
            normalized_out = []
            for vec in linear_out:
                # Use layer's epsilon parameter
                norm_vec, _, _ = self._layer_norm(vec, layer['epsilon'])
                normalized_out.append(norm_vec)
            
            # Apply position-dependent transformation with simplified P matrix
            T_matrix = []
            for k, vec in enumerate(normalized_out):
                out_dim = len(layer['P'])
                Tk = [0.0] * out_dim
                for i in range(out_dim):
                    for j in range(out_dim):
                        period = layer['periods'][i][j]
                        phi = math.cos(2 * math.pi * k / period)
                        Tk[i] += layer['P'][i][j] * vec[j] * phi
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
            
            # Get input/output dimensions for residual connection
            in_dim = len(layer_input[0])
            in_seq = len(layer_input)
            out_dim = self.model_dims[l]
            out_seq = self.linker_dims[l]
            
            # Handle residual connection
            if layer['use_residual'] == 'separate':
                # Separate projection and Linker for residual
                residual_feat = [self._mat_vec(layer['residual_proj'], vec) for vec in layer_input]
                residual_feat_transposed = [list(col) for col in zip(*residual_feat)]
                residual_matrix = self._mat_mul(residual_feat_transposed, layer['residual_linker'])
                # Convert residual to sequence of vectors and add to output
                residual_vectors = []
                for j in range(len(residual_matrix[0])):
                    vec = [residual_matrix[i][j] for i in range(len(residual_matrix))]
                    residual_vectors.append(vec)
                # Add residual to current output
                current = [self._vec_add(current[i], residual_vectors[i]) 
                          for i in range(len(current))]
            
            elif layer['use_residual'] == 'shared':
                # Shared M and Linker for residual
                residual_feat = [self._mat_vec(layer['M'], vec) for vec in layer_input]
                residual_feat_transposed = [list(col) for col in zip(*residual_feat)]
                residual_matrix = self._mat_mul(residual_feat_transposed, layer['Linker'])
                # Convert residual to sequence of vectors and add to output
                residual_vectors = []
                for j in range(len(residual_matrix[0])):
                    vec = [residual_matrix[i][j] for i in range(len(residual_matrix))]
                    residual_vectors.append(vec)
                # Add residual to current output
                current = [self._vec_add(current[i], residual_vectors[i]) 
                          for i in range(len(current))]
            
            else:
                # Identity residual only if dimensions match
                if in_dim == out_dim and in_seq == out_seq:
                    layer_input_transposed = [list(col) for col in zip(*layer_input)]
                    residual_matrix = self._mat_mul(layer_input_transposed, layer['Linker'])
                    # Convert residual to sequence of vectors and add to output
                    residual_vectors = []
                    for j in range(len(residual_matrix[0])):
                        vec = [residual_matrix[i][j] for i in range(len(residual_matrix))]
                        residual_vectors.append(vec)
                    # Add residual to current output
                    current = [self._vec_add(current[i], residual_vectors[i]) 
                              for i in range(len(current))]
        
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
            self.__init__(self.input_dim, self.model_dims,
                          self.input_seq_len, self.linker_dims, 
                          self.linker_trainable, self.use_residual_list)
            
        history = []
        D_prev = float('inf')
        total_positions = sum(len(seq) for seq in seqs)
        # Removed fixed epsilon - now using layer's epsilon
        
        # Initialize gradient storage for all layers
        grad_P_list = []
        grad_M_list = []
        grad_Linker_list = []
        for l in range(self.num_layers):
            out_dim = self.model_dims[l]
            
            # Gradients for P matrix (2D instead of 3D)
            grad_P = [[0.0] * out_dim for _ in range(out_dim)]
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
            
            # Initialize gradients for separate residual parameters if needed
            if self.layers[l]['use_residual'] == 'separate':
                # Gradient for residual projection matrix
                grad_residual_proj = [[0.0] * in_dim for _ in range(out_dim)]
                self.layers[l]['grad_residual_proj'] = grad_residual_proj
                
                # Gradient for residual Linker matrix
                grad_residual_linker = [[0.0] * out_seq for _ in range(in_seq)]
                self.layers[l]['grad_residual_linker'] = grad_residual_linker
        
        for it in range(max_iters):
            # Reset gradients
            for l in range(self.num_layers):
                # Reset P gradients (2D matrix)
                for i in range(len(grad_P_list[l])):
                    for j in range(len(grad_P_list[l][i])):
                        grad_P_list[l][i][j] = 0.0
                
                # Reset M gradients
                for i in range(len(grad_M_list[l])):
                    for j in range(len(grad_M_list[l][i])):
                        grad_M_list[l][i][j] = 0.0
                
                # Reset Linker gradients
                for i in range(len(grad_Linker_list[l])):
                    for j in range(len(grad_Linker_list[l][i])):
                        grad_Linker_list[l][i][j] = 0.0
                
                # Reset separate residual gradients if present
                if self.layers[l]['use_residual'] == 'separate':
                    grad_residual_proj = self.layers[l]['grad_residual_proj']
                    for i in range(len(grad_residual_proj)):
                        for j in range(len(grad_residual_proj[i])):
                            grad_residual_proj[i][j] = 0.0
                    
                    grad_residual_linker = self.layers[l]['grad_residual_linker']
                    for i in range(len(grad_residual_linker)):
                        for j in range(len(grad_residual_linker[i])):
                            grad_residual_linker[i][j] = 0.0
            
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
                    
                    # Apply Layer Normalization using layer's epsilon
                    normalized_linear_out = []
                    mean_cache = []
                    var_cache = []
                    orig_cache = []
                    for vec in linear_out:
                        # Use layer's epsilon parameter
                        norm_vec, vec_mean, vec_var = self._layer_norm(vec, layer['epsilon'])
                        normalized_linear_out.append(norm_vec)
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
                                period = layer['periods'][i][j]
                                phi = math.cos(2 * math.pi * k / period)
                                phi_vals[(i, j)] = phi
                                Tk[i] += layer['P'][i][j] * vec[j] * phi
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
                    
                    # Handle residual connection
                    if layer['use_residual'] == 'separate':
                        # Separate projection and Linker for residual
                        residual_feat = [self._mat_vec(layer['residual_proj'], vec) for vec in current]
                        residual_feat_transposed = [list(col) for col in zip(*residual_feat)]
                        residual_matrix = self._mat_mul(residual_feat_transposed, layer['residual_linker'])
                        # Store residual intermediates for backprop
                        layer_intermediate['residual_feat'] = residual_feat
                        layer_intermediate['residual_matrix'] = residual_matrix
                        # Add residual to output
                        for i in range(len(U_matrix)):
                            for j in range(len(U_matrix[i])):
                                U_matrix[i][j] += residual_matrix[i][j]
                    
                    elif layer['use_residual'] == 'shared':
                        # Shared M and Linker for residual
                        residual_feat = [self._mat_vec(layer['M'], vec) for vec in current]
                        residual_feat_transposed = [list(col) for col in zip(*residual_feat)]
                        residual_matrix = self._mat_mul(residual_feat_transposed, layer['Linker'])
                        # Store residual intermediates for backprop
                        layer_intermediate['residual_feat'] = residual_feat
                        layer_intermediate['residual_matrix'] = residual_matrix
                        # Add residual to output
                        for i in range(len(U_matrix)):
                            for j in range(len(U_matrix[i])):
                                U_matrix[i][j] += residual_matrix[i][j]
                    
                    else:
                        # Identity residual only if dimensions match
                        if (l == 0 and self.input_dim == self.model_dims[l] and 
                            self.input_seq_len == self.linker_dims[l]):
                            seq_transposed = [list(col) for col in zip(*current)]
                            residual_matrix = self._mat_mul(seq_transposed, layer['Linker'])
                            # Add residual to output
                            for i in range(len(U_matrix)):
                                for j in range(len(U_matrix[i])):
                                    U_matrix[i][j] += residual_matrix[i][j]
                    
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
                                # Gradient for P[i][j] (simplified to 2D)
                                grad_P_list[l][i][j] += (
                                    dTk[i] * vec[j] * phi_vals[(i, j)]
                                )
                                
                                # Gradient for normalized linear output
                                d_normalized[k][j] += (
                                    dTk[i] * layer['P'][i][j] * phi_vals[(i, j)]
                                )
                    
                    # Backpropagate through Layer Normalization
                    d_linear = [[0.0] * len(linear_out[0]) for _ in range(len(linear_out))]
                    for k in range(len(d_normalized)):
                        d_normalized_vec = d_normalized[k]
                        orig_vec = orig_cache[k]
                        vec_mean = mean_cache[k]
                        vec_var = var_cache[k]
                        n = len(orig_vec)  # Number of features
                        
                        # Use layer's epsilon for backprop
                        epsilon_layer = layer['epsilon']
                        
                        # Gradient w.r.t variance
                        d_var = 0.0
                        for i in range(n):
                            # dL/dvar = dL/dnormalized_i * dnormalized_i/dvar
                            d_var += d_normalized_vec[i] * (orig_vec[i] - vec_mean) * \
                                     (-0.5) * (vec_var + epsilon_layer) ** (-1.5)
                        
                        # Gradient w.r.t mean
                        d_mean = 0.0
                        for i in range(n):
                            # dL/dmean = dL/dnormalized_i * dnormalized_i/dmean
                            d_mean += d_normalized_vec[i] * \
                                      (-1.0 / math.sqrt(vec_var + epsilon_layer))
                            
                            # Additional term from variance gradient
                            d_mean += d_var * (2.0 / n) * (orig_vec[i] - vec_mean) * (-1.0)
                        
                        # Gradient w.r.t original linear output
                        for i in range(n):
                            # Term 1: from normalized value
                            term1 = d_normalized_vec[i] / math.sqrt(vec_var + epsilon_layer)
                            
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
                    else:
                        d_next = None
            
            # Update parameters
            for l in range(self.num_layers):
                layer = self.layers[l]
                
                # Update P matrix (2D)
                for i in range(len(layer['P'])):
                    for j in range(len(layer['P'][i])):
                        layer['P'][i][j] -= learning_rate * grad_P_list[l][i][j]
                
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

    def auto_train(self, seqs, max_iters=100, tol=1e-6, learning_rate=0.01, 
                  continued=False, auto_mode='gap', decay_rate=1.0, print_every=10):
        """
        Self-training for the ENTIRE hierarchical model with two modes:
          - 'gap': Predicts current input vector (self-consistency)
          - 'reg': Predicts next input vector (auto-regressive) with causal masking
        
        Now trains ALL layers of the hierarchical model, not just the first layer.
        Stores statistical information for reconstruction and generation.
        
        Key improvements for reg mode:
        - Added causal masking to prevent information leakage
        - Ensures strict auto-regressive property
        - Uses only historical information for next-step prediction
        """
        if auto_mode not in ('gap', 'reg'):
            raise ValueError("auto_mode must be either 'gap' or 'reg'")
        
        # Validate input sequences
        for seq in seqs:
            if len(seq) != self.input_seq_len:
                raise ValueError(f"All sequences must have length {self.input_seq_len}")
        
        # Initialize all layers if not continuing training
        if not continued:
            for l in range(self.num_layers):
                layer = self.layers[l]
                
                # Determine input dimensions for this layer
                if l == 0:
                    in_dim = self.input_dim
                else:
                    in_dim = self.model_dims[l-1]
                    
                out_dim = self.model_dims[l]
                
                # Reinitialize M and P with small random values
                layer['M'] = [[random.uniform(-0.5, 0.5) for _ in range(in_dim)] 
                               for _ in range(out_dim)]
                layer['P'] = [[random.uniform(-0.1, 0.1) for _ in range(out_dim)]
                               for _ in range(out_dim)]
                
                # Reinitialize Linker if trainable
                if l == 0:
                    in_seq = self.input_seq_len
                else:
                    in_seq = self.linker_dims[l-1]
                out_seq = self.linker_dims[l]
                
                if layer['linker_trainable']:
                    layer['Linker'] = [[random.uniform(-0.1, 0.1) for _ in range(out_seq)] 
                                      for _ in range(in_seq)]
                
                # Reinitialize residual components if using 'separate' mode
                if layer['use_residual'] == 'separate':
                    layer['residual_proj'] = [[random.uniform(-0.1, 0.1) for _ in range(in_dim)]
                                             for _ in range(out_dim)]
                    layer['residual_linker'] = [[random.uniform(-0.1, 0.1) for _ in range(out_seq)] 
                                              for _ in range(in_seq)]
            
            # Calculate global mean vector of input sequences
            total = [0.0] * self.input_dim
            total_windows = 0
            for seq in seqs:
                for vec in seq:
                    for d in range(self.input_dim):
                        total[d] += vec[d]
                    total_windows += 1
            self.mean_t = [t / total_windows for t in total]
            self.mean_window_count = total_windows / len(seqs)
        
        # Calculate total training samples
        total_samples = 0
        for seq in seqs:
            if auto_mode == 'gap':
                total_samples += len(seq)  # All positions are samples
            else:  # 'reg' mode
                total_samples += max(0, len(seq) - 1)  # All except last position
        
        if total_samples == 0:
            raise ValueError("No training samples found")
        
        history = []
        prev_loss = float('inf')
        
        # Initialize gradient storage for all layers
        grad_P_list = []
        grad_M_list = []
        grad_Linker_list = []
        grad_residual_proj_list = []
        grad_residual_linker_list = []
        
        for l in range(self.num_layers):
            out_dim = self.model_dims[l]
            
            # Gradients for P matrix (2D)
            grad_P = [[0.0] * out_dim for _ in range(out_dim)]
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
            
            # Gradients for residual components (if using 'separate' mode)
            if self.layers[l]['use_residual'] == 'separate':
                grad_residual_proj = [[0.0] * in_dim for _ in range(out_dim)]
                grad_residual_linker = [[0.0] * out_seq for _ in range(in_seq)]
            else:
                grad_residual_proj = None
                grad_residual_linker = None
                
            grad_residual_proj_list.append(grad_residual_proj)
            grad_residual_linker_list.append(grad_residual_linker)
        
        # Gradient clipping threshold
        grad_clip_threshold = 1.0
        
        for it in range(max_iters):
            # Reset gradients
            for l in range(self.num_layers):
                # Reset P gradients
                if grad_P_list[l] is not None:
                    for i in range(len(grad_P_list[l])):
                        for j in range(len(grad_P_list[l][i])):
                            grad_P_list[l][i][j] = 0.0
                
                # Reset M gradients
                if grad_M_list[l] is not None:
                    for i in range(len(grad_M_list[l])):
                        for j in range(len(grad_M_list[l][i])):
                            grad_M_list[l][i][j] = 0.0
                
                # Reset Linker gradients
                if grad_Linker_list[l] is not None:
                    for i in range(len(grad_Linker_list[l])):
                        for j in range(len(grad_Linker_list[l][i])):
                            grad_Linker_list[l][i][j] = 0.0
                
                # Reset residual gradients if using 'separate' mode
                if self.layers[l]['use_residual'] == 'separate':
                    for i in range(len(grad_residual_proj_list[l])):
                        for j in range(len(grad_residual_proj_list[l][i])):
                            grad_residual_proj_list[l][i][j] = 0.0
                    for i in range(len(grad_residual_linker_list[l])):
                        for j in range(len(grad_residual_linker_list[l][i])):
                            grad_residual_linker_list[l][i][j] = 0.0
            
            total_loss = 0.0
            sequence_count = 0
            
            # Process each sequence
            for seq in seqs:
                sequence_count += 1
                intermediates = []  # Store layer intermediates for backprop
                current = seq
                
                # Forward pass through ALL layers
                for l, layer in enumerate(self.layers):
                    layer_intermediate = {}
                    
                    # Store input for backprop
                    layer_intermediate['input_vec'] = current
                    
                    # Apply linear transformation
                    linear_out = [self._mat_vec(layer['M'], vec) for vec in current]
                    
                    # Apply unified layer normalization - with causal masking for reg mode
                    if auto_mode == 'reg':
                        # For reg mode, normalize only using historical information at each step
                        normalized_linear_out = []
                        mean_cache = []
                        var_cache = []
                        
                        for k in range(len(linear_out)):
                            # Use only positions 0 to k for normalization (causal)
                            historical_linear_out = linear_out[:k+1]
                            
                            # Compute mean and variance from historical data only
                            mean_val = sum(sum(row) for row in historical_linear_out) / (len(historical_linear_out) * len(historical_linear_out[0]))
                            var_val = sum((x - mean_val) ** 2 for row in historical_linear_out for x in row) / (len(historical_linear_out) * len(historical_linear_out[0]))
                            
                            # Normalize current position using historical statistics
                            norm_vec = [(x - mean_val) / math.sqrt(var_val + layer['epsilon']) 
                                       for x in linear_out[k]]
                            
                            normalized_linear_out.append(norm_vec)
                            mean_cache.append(mean_val)
                            var_cache.append(var_val)
                    else:
                        # For gap mode, use full sequence normalization
                        normalized_linear_out, mean_cache, var_cache = self._normalize_layer(
                            linear_out, layer['epsilon']
                        )
                    
                    # Save intermediate values for backprop
                    layer_intermediate['linear_out'] = linear_out
                    layer_intermediate['normalized_linear_out'] = normalized_linear_out
                    layer_intermediate['mean_cache'] = mean_cache
                    layer_intermediate['var_cache'] = var_cache
                    
                    # Apply position-dependent transformation to normalized output
                    T_matrix = []
                    phi_vals_list = []
                    
                    for k, vec in enumerate(normalized_linear_out):
                        out_dim = len(layer['P'])
                        Tk = [0.0] * out_dim
                        phi_vals = {}
                        
                        for i in range(out_dim):
                            for j in range(out_dim):
                                period = layer['periods'][i][j]
                                phi = math.cos(2 * math.pi * k / period)
                                phi_vals[(i, j)] = phi
                                Tk[i] += layer['P'][i][j] * vec[j] * phi
                        
                        T_matrix.append(Tk)
                        phi_vals_list.append(phi_vals)
                    
                    layer_intermediate['T_matrix'] = T_matrix
                    layer_intermediate['phi_vals'] = phi_vals_list
                    
                    # Convert to matrix form: rows=features, columns=time steps
                    T_transposed = [list(col) for col in zip(*T_matrix)]
                    layer_intermediate['T_transposed'] = T_transposed
                    
                    # Apply causal masking for Linker matrix in reg mode
                    if auto_mode == 'reg' and layer['linker_trainable']:
                        # Create causal Linker mask: only allow past-to-future information flow
                        causal_linker = []
                        for i in range(len(layer['Linker'])):
                            causal_row = []
                            for j in range(len(layer['Linker'][i])):
                                # Allow connection only if output position >= input position
                                # This ensures causal information flow
                                if j >= i:  # Output position j receives info from input position i only if j >= i
                                    causal_row.append(layer['Linker'][i][j])
                                else:
                                    causal_row.append(0.0)
                            causal_linker.append(causal_row)
                        used_linker = causal_linker
                    else:
                        used_linker = layer['Linker']
                    
                    # Apply sequence length transformation: U = T * Linker
                    U_matrix = self._mat_mul(T_transposed, used_linker)
                    layer_intermediate['U_matrix'] = U_matrix
                    layer_intermediate['used_linker'] = used_linker
                    
                    # Convert back to sequence of vectors for next layer
                    current_U = []
                    for j in range(len(U_matrix[0])):
                        vec = [U_matrix[i][j] for i in range(len(U_matrix))]
                        current_U.append(vec)
                    
                    # Handle residual connections with causal masking for reg mode
                    residual = None
                    if layer['use_residual'] == 'separate':
                        residual_feat = [self._mat_vec(layer['residual_proj'], vec) for vec in current]
                        residual_transposed = [list(col) for col in zip(*residual_feat)]
                        
                        # Apply causal masking to residual linker
                        if auto_mode == 'reg':
                            causal_residual_linker = []
                            for i in range(len(layer['residual_linker'])):
                                causal_row = []
                                for j in range(len(layer['residual_linker'][i])):
                                    if j >= i:
                                        causal_row.append(layer['residual_linker'][i][j])
                                    else:
                                        causal_row.append(0.0)
                                causal_residual_linker.append(causal_row)
                            used_residual_linker = causal_residual_linker
                        else:
                            used_residual_linker = layer['residual_linker']
                        
                        residual_U_matrix = self._mat_mul(residual_transposed, used_residual_linker)
                        residual = []
                        for j in range(len(residual_U_matrix[0])):
                            vec = [residual_U_matrix[i][j] for i in range(len(residual_U_matrix))]
                            residual.append(vec)
                        
                        layer_intermediate['residual_feat'] = residual_feat
                        layer_intermediate['residual_transposed'] = residual_transposed
                        layer_intermediate['residual'] = residual
                        layer_intermediate['used_residual_linker'] = used_residual_linker
                    
                    elif layer['use_residual'] == 'shared':
                        residual_feat = [self._mat_vec(layer['M'], vec) for vec in current]
                        residual_transposed = [list(col) for col in zip(*residual_feat)]
                        residual_U_matrix = self._mat_mul(residual_transposed, used_linker)  # Use same causal linker
                        residual = []
                        for j in range(len(residual_U_matrix[0])):
                            vec = [residual_U_matrix[i][j] for i in range(len(residual_U_matrix))]
                            residual.append(vec)
                        
                        layer_intermediate['residual_feat'] = residual_feat
                        layer_intermediate['residual_transposed'] = residual_transposed
                        layer_intermediate['residual'] = residual
                    
                    else:
                        if len(current[0]) == len(current_U[0]) and len(current) == len(current_U):
                            residual_transposed = [list(col) for col in zip(*current)]
                            residual_U_matrix = self._mat_mul(residual_transposed, used_linker)  # Use causal linker
                            residual = []
                            for j in range(len(residual_U_matrix[0])):
                                vec = [residual_U_matrix[i][j] for i in range(len(residual_U_matrix))]
                                residual.append(vec)
                            layer_intermediate['residual'] = residual
                    
                    # Apply residual if available
                    if residual is not None:
                        current = [self._vec_add(u_vec, r_vec) for u_vec, r_vec in zip(current_U, residual)]
                    else:
                        current = current_U
                    
                    intermediates.append(layer_intermediate)
                
                # Backward pass through ALL layers
                # Start with output gradient
                d_next_layer_grad = None
                
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
                    used_linker = interm.get('used_linker', layer['Linker'])
                    
                    # Determine gradient source
                    if l == self.num_layers-1:
                        # Output layer gradient: dL/doutput
                        d_output = []
                        valid_positions = 0
                        
                        for k in range(len(current)):
                            # Skip last position in 'reg' mode
                            if auto_mode == 'reg' and k == len(seq) - 1:
                                continue
                                
                            # Determine target based on mode
                            if auto_mode == 'gap':
                                target = seq[k]  # Current input vector
                            else:  # 'reg' mode
                                target = seq[k + 1]  # Next input vector
                            
                            # Calculate gradient for output vectors
                            d_output_k = []
                            for i in range(len(current[k])):
                                # Gradient: dL/dU_ij = 2*(U_ij - target[i]) / total_samples
                                error = 2 * (current[k][i] - target[i]) / total_samples
                                d_output_k.append(error)
                                total_loss += (current[k][i] - target[i])**2
                            d_output.append(d_output_k)
                            valid_positions += 1
                        
                        if valid_positions == 0:
                            continue
                            
                        # Convert to matrix form for backpropagation (features × time)
                        d_output_mat = [list(col) for col in zip(*d_output)]
                        
                    else:
                        # Middle layer gradient: from next layer
                        d_output_mat = d_next_layer_grad
                    
                    # Handle residual connection gradient
                    if 'residual' in interm and interm['residual'] is not None:
                        # Gradient flows through both main and residual paths
                        d_output_mat = [[d_output_mat[i][j] * 0.5 for j in range(len(d_output_mat[0]))] 
                                      for i in range(len(d_output_mat))]
                    
                    # Backpropagate through sequence length transformation
                    # Gradient for Linker: dL/dLinker = T^T * d_output_mat
                    dLinker = self._mat_mul(self._transpose(T_transposed), d_output_mat)
                    
                    # Apply causal masking to Linker gradient in reg mode
                    if auto_mode == 'reg' and layer['linker_trainable']:
                        for i in range(len(dLinker)):
                            for j in range(len(dLinker[i])):
                                if j < i:  # Zero out gradients for non-causal connections
                                    dLinker[i][j] = 0.0
                    
                    # Gradient for T: dL/dT = d_output_mat * Linker^T
                    Linker_T = self._transpose(used_linker)
                    dT_mat = self._mat_mul(d_output_mat, Linker_T)
                    
                    # Convert dT from matrix form to list of vectors (time × features)
                    dT = [list(col) for col in zip(*dT_mat)]
                    
                    # Gradient for position-dependent transformation
                    d_normalized = [[0.0] * len(normalized_linear_out[0]) for _ in range(len(normalized_linear_out))]
                    
                    for k, vec in enumerate(normalized_linear_out):
                        if k < len(dT):  # Ensure we don't go out of bounds
                            dTk = dT[k]
                            phi_vals = phi_vals_list[k]
                            
                            # Compute gradients for P and normalized linear output
                            for i in range(len(dTk)):
                                for j in range(len(layer['P'][i])):
                                    # Gradient for P[i][j]
                                    grad_P_list[l][i][j] += (
                                        dTk[i] * vec[j] * phi_vals[(i, j)]
                                    )
                                    
                                    # Gradient for normalized linear output
                                    d_normalized[k][j] += (
                                        dTk[i] * layer['P'][i][j] * phi_vals[(i, j)]
                                    )
                    
                    # Backpropagate through Layer Normalization
                    d_linear = [[0.0] * len(linear_out[0]) for _ in range(len(linear_out))]
                    
                    for k in range(len(d_normalized)):
                        d_normalized_vec = d_normalized[k]
                        orig_vec = linear_out[k]
                        vec_mean = mean_cache[k]
                        vec_var = var_cache[k]
                        n = len(orig_vec)  # Number of features
                        
                        # Gradient w.r.t variance
                        d_var = 0.0
                        for i in range(n):
                            # dL/dvar = dL/dnormalized_i * dnormalized_i/dvar
                            d_var += d_normalized_vec[i] * (orig_vec[i] - vec_mean) * \
                                     (-0.5) * (vec_var + layer['epsilon']) ** (-1.5)
                        
                        # Gradient w.r.t mean
                        d_mean = 0.0
                        for i in range(n):
                            # dL/dmean = dL/dnormalized_i * dnormalized_i/dmean
                            d_mean += d_normalized_vec[i] * \
                                      (-1.0 / math.sqrt(vec_var + layer['epsilon']))
                            
                            # Additional term from variance gradient
                            d_mean += d_var * (2.0 / n) * (orig_vec[i] - vec_mean) * (-1.0)
                        
                        # Gradient w.r.t original linear output
                        for i in range(n):
                            # Term 1: from normalized value
                            term1 = d_normalized_vec[i] / math.sqrt(vec_var + layer['epsilon'])
                            
                            # Term 2: from mean gradient
                            term2 = d_mean * (1.0 / n)
                            
                            # Term 3: from variance gradient
                            term3 = d_var * (2.0 / n) * (orig_vec[i] - vec_mean)
                            
                            d_linear[k][i] = term1 + term2 + term3
                    
                    # Gradient for M matrix
                    input_vecs = interm['input_vec']
                    for k, vec in enumerate(input_vecs):
                        if k < len(d_linear):  # Ensure we don't go out of bounds
                            for i in range(len(d_linear[k])):
                                for j in range(len(vec)):
                                    grad_M_list[l][i][j] += d_linear[k][i] * vec[j]
                    
                    # Accumulate Linker gradient if trainable
                    if layer['linker_trainable']:
                        for i in range(len(dLinker)):
                            for j in range(len(dLinker[i])):
                                grad_Linker_list[l][i][j] += dLinker[i][j]
                    
                    # Handle residual linker gradients for separate mode
                    if layer['use_residual'] == 'separate' and 'used_residual_linker' in interm:
                        used_residual_linker = interm['used_residual_linker']
                        residual_transposed = interm['residual_transposed']
                        
                        # Gradient for residual linker: dL/dResidualLinker = residual_feat^T * d_output_mat
                        dResidualLinker = self._mat_mul(self._transpose(residual_transposed), d_output_mat)
                        
                        # Apply causal masking to residual linker gradient
                        if auto_mode == 'reg':
                            for i in range(len(dResidualLinker)):
                                for j in range(len(dResidualLinker[i])):
                                    if j < i:  # Zero out non-causal gradients
                                        dResidualLinker[i][j] = 0.0
                        
                        for i in range(len(dResidualLinker)):
                            for j in range(len(dResidualLinker[i])):
                                grad_residual_linker_list[l][i][j] += dResidualLinker[i][j]
                    
                    # Prepare gradient for next lower layer
                    if l > 0:
                        # Convert d_linear to matrix form for next layer
                        d_linear_mat = [list(col) for col in zip(*d_linear)]
                        d_next_layer_grad = d_linear_mat
                
                # End of backward pass for this sequence
            
            # Average gradients across all sequences
            scale = 1.0 # / sequence_count if sequence_count > 0 else 1.0
            for l in range(self.num_layers):
                # Average P gradients
                for i in range(len(grad_P_list[l])):
                    for j in range(len(grad_P_list[l][i])):
                        grad_P_list[l][i][j] *= scale
                
                # Average M gradients
                for i in range(len(grad_M_list[l])):
                    for j in range(len(grad_M_list[l][i])):
                        grad_M_list[l][i][j] *= scale
                
                # Average Linker gradients
                for i in range(len(grad_Linker_list[l])):
                    for j in range(len(grad_Linker_list[l][i])):
                        grad_Linker_list[l][i][j] *= scale
                
                # Average residual gradients if using 'separate' mode
                if self.layers[l]['use_residual'] == 'separate':
                    for i in range(len(grad_residual_proj_list[l])):
                        for j in range(len(grad_residual_proj_list[l][i])):
                            grad_residual_proj_list[l][i][j] *= scale
                    for i in range(len(grad_residual_linker_list[l])):
                        for j in range(len(grad_residual_linker_list[l][i])):
                            grad_residual_linker_list[l][i][j] *= scale
            
            # Apply gradient clipping to prevent explosion
            for l in range(self.num_layers):
                self._clip_gradients(grad_P_list[l], grad_clip_threshold)
                self._clip_gradients(grad_M_list[l], grad_clip_threshold)
                if self.layers[l]['linker_trainable']:
                    self._clip_gradients(grad_Linker_list[l], grad_clip_threshold)
                if self.layers[l]['use_residual'] == 'separate':
                    self._clip_gradients(grad_residual_proj_list[l], grad_clip_threshold)
                    self._clip_gradients(grad_residual_linker_list[l], grad_clip_threshold)
            
            # Update parameters for ALL layers
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
                
                # Update Linker matrix if trainable
                if layer['linker_trainable']:
                    for i in range(len(layer['Linker'])):
                        for j in range(len(layer['Linker'][i])):
                            layer['Linker'][i][j] -= learning_rate * grad_Linker_list[l][i][j]
                
                # Update residual components if using 'separate' mode
                if layer['use_residual'] == 'separate' and grad_residual_proj_list[l] is not None:
                    for i in range(len(layer['residual_proj'])):
                        for j in range(len(layer['residual_proj'][i])):
                            layer['residual_proj'][i][j] -= learning_rate * grad_residual_proj_list[l][i][j]
                    
                    for i in range(len(layer['residual_linker'])):
                        for j in range(len(layer['residual_linker'][i])):
                            layer['residual_linker'][i][j] -= learning_rate * grad_residual_linker_list[l][i][j]
            
            # Record and print progress
            avg_loss = total_loss / total_samples
            history.append(avg_loss)
            if it % print_every == 0 or it == max_iters - 1:
                mode_display = "Gap" if auto_mode == 'gap' else "Reg"
                print(f"AutoTrain({mode_display}) Iter {it:3d}: loss = {avg_loss:.6f}, LR = {learning_rate:.6f}")
            
            # Check convergence
            if prev_loss - avg_loss < tol:
                print(f"Converged after {it+1} iterations")
                break
            prev_loss = avg_loss
            
            # Apply learning rate decay (only every few iterations to avoid premature stopping)
            if it % 5 == 0:  # Decay every 5 iterations
                learning_rate *= decay_rate
        
        self.trained = True
        return history

    def reconstruct(self):
        """
        Reconstruct representative sequence using the ENTIRE trained hierarchical model.
        Generates sequence with length equal to mean window count.
        Uses full backpropagation to find input sequence that minimizes
        the difference between model output and target.
        """
        assert self.trained and hasattr(self, 'mean_t'), "Model must be auto-trained first"
        
        # Round average window count to nearest integer
        n_windows = round(self.mean_window_count)
        
        # Initialize a random sequence
        seq = [[random.gauss(self.mean_t[d], 0.1) for d in range(self.input_dim)] 
               for _ in range(n_windows)]
        
        # Optimization parameters
        max_iters = 100
        learning_rate = 0.01
        tol = 1e-6
        
        prev_loss = float('inf')
        
        for it in range(max_iters):
            # Forward pass through the entire model, storing intermediates for backprop
            intermediates = []
            current = seq
            
            for l, layer in enumerate(self.layers):
                layer_intermediate = {}
                
                # Apply linear transformation
                linear_out = [self._mat_vec(layer['M'], vec) for vec in current]
                
                # Apply unified layer normalization
                normalized_out, mean_cache, var_cache = self._normalize_layer(linear_out, layer['epsilon'])
                
                # Save intermediate values for backprop
                layer_intermediate['linear_out'] = linear_out
                layer_intermediate['normalized_linear_out'] = normalized_out
                layer_intermediate['mean_cache'] = mean_cache
                layer_intermediate['var_cache'] = var_cache
                
                # Apply position-dependent transformation
                T_matrix = []
                phi_vals_list = []
                for k, vec in enumerate(normalized_out):
                    out_dim = len(layer['P'])
                    Tk = [0.0] * out_dim
                    phi_vals = {}
                    for i in range(out_dim):
                        for j in range(out_dim):
                            period = layer['periods'][i][j]
                            phi = math.cos(2 * math.pi * k / period)
                            phi_vals[(i, j)] = phi
                            Tk[i] += layer['P'][i][j] * vec[j] * phi
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
                
                # Convert back to sequence of vectors
                current_U = []
                for j in range(len(U_matrix[0])):
                    vec = [U_matrix[i][j] for i in range(len(U_matrix))]
                    current_U.append(vec)
                
                # Handle residual connections
                residual = None
                if layer['use_residual'] == 'separate':
                    residual_feat = [self._mat_vec(layer['residual_proj'], vec) for vec in current]
                    residual_transposed = [list(col) for col in zip(*residual_feat)]
                    residual_U_matrix = self._mat_mul(residual_transposed, layer['residual_linker'])
                    residual = []
                    for j in range(len(residual_U_matrix[0])):
                        vec = [residual_U_matrix[i][j] for i in range(len(residual_U_matrix))]
                        residual.append(vec)
                    layer_intermediate['residual_feat'] = residual_feat
                    layer_intermediate['residual_transposed'] = residual_transposed
                    layer_intermediate['residual'] = residual
                
                elif layer['use_residual'] == 'shared':
                    residual_feat = [self._mat_vec(layer['M'], vec) for vec in current]
                    residual_transposed = [list(col) for col in zip(*residual_feat)]
                    residual_U_matrix = self._mat_mul(residual_transposed, layer['Linker'])
                    residual = []
                    for j in range(len(residual_U_matrix[0])):
                        vec = [residual_U_matrix[i][j] for i in range(len(residual_U_matrix))]
                        residual.append(vec)
                    layer_intermediate['residual_feat'] = residual_feat
                    layer_intermediate['residual_transposed'] = residual_transposed
                    layer_intermediate['residual'] = residual
                
                else:
                    if len(current[0]) == len(current_U[0]) and len(current) == len(current_U):
                        residual_transposed = [list(col) for col in zip(*current)]
                        residual_U_matrix = self._mat_mul(residual_transposed, layer['Linker'])
                        residual = []
                        for j in range(len(residual_U_matrix[0])):
                            vec = [residual_U_matrix[i][j] for i in range(len(residual_U_matrix))]
                            residual.append(vec)
                        layer_intermediate['residual'] = residual
                
                # Apply residual if available
                if residual is not None:
                    current = [self._vec_add(u_vec, r_vec) for u_vec, r_vec in zip(current_U, residual)]
                else:
                    current = current_U
                
                intermediates.append(layer_intermediate)
            
            # Calculate loss (difference between output and mean_t)
            loss = 0.0
            for vec in current:
                for i in range(len(vec)):
                    diff = vec[i] - self.mean_t[i]
                    loss += diff * diff
            loss /= len(current)
            
            # Backward pass to compute gradients with respect to input sequence
            d_next = None
            grad_seq = [[0.0] * self.input_dim for _ in range(n_windows)]
            
            for l in range(self.num_layers-1, -1, -1):
                layer = self.layers[l]
                interm = intermediates[l]
                T_matrix = interm['T_matrix']
                T_transposed = interm['T_transposed']
                phi_vals_list = interm['phi_vals']
                linear_out = interm['linear_out']
                normalized_out = interm['normalized_linear_out']
                mean_cache = interm['mean_cache']
                var_cache = interm['var_cache']
                
                if l == self.num_layers-1:
                    # Gradient of loss with respect to output
                    d_out = []
                    for vec in current:
                        d_vec = [2 * (vec[i] - self.mean_t[i]) / len(current) for i in range(len(vec))]
                        d_out.append(d_vec)
                else:
                    d_out = d_next
                
                # Convert gradient to matrix form for backpropagation
                d_out_mat = [list(col) for col in zip(*d_out)]
                
                # Backpropagate through residual connection if present
                if 'residual' in interm and interm['residual'] is not None:
                    # Gradient flows through both main path and residual path
                    # For simplicity, we assume gradient is split equally
                    # In a more precise implementation, we would need to handle this properly
                    d_out_mat = [[d_out_mat[i][j] * 0.5 for j in range(len(d_out_mat[0]))] for i in range(len(d_out_mat))]
                
                # Backpropagate through sequence length transformation
                Linker_T = self._transpose(layer['Linker'])
                dT_mat = self._mat_mul(d_out_mat, Linker_T)
                
                # Convert gradient to list of vectors
                dT = [list(col) for col in zip(*dT_mat)]
                
                # Backpropagate through position-dependent transformation
                d_normalized = [[0.0] * len(normalized_out[0]) for _ in range(len(normalized_out))]
                for k, vec in enumerate(normalized_out):
                    dTk = dT[k]
                    phi_vals = phi_vals_list[k]
                    
                    for i in range(len(dTk)):
                        for j in range(len(layer['P'][i])):
                            d_normalized[k][j] += (
                                dTk[i] * layer['P'][i][j] * phi_vals[(i, j)]
                            )
                
                # Backpropagate through layer normalization
                d_linear = [[0.0] * len(linear_out[0]) for _ in range(len(linear_out))]
                for k in range(len(d_normalized)):
                    d_normalized_vec = d_normalized[k]
                    orig_vec = linear_out[k]
                    vec_mean = mean_cache[k]
                    vec_var = var_cache[k]
                    n = len(orig_vec)
                    
                    # Gradient with respect to variance
                    d_var = 0.0
                    for i in range(n):
                        d_var += d_normalized_vec[i] * (orig_vec[i] - vec_mean) * \
                                 (-0.5) * (vec_var + layer['epsilon']) ** (-1.5)
                    
                    # Gradient with respect to mean
                    d_mean = 0.0
                    for i in range(n):
                        d_mean += d_normalized_vec[i] * (-1.0 / math.sqrt(vec_var + layer['epsilon']))
                        d_mean += d_var * (2.0 / n) * (orig_vec[i] - vec_mean) * (-1.0)
                    
                    # Gradient with respect to original linear output
                    for i in range(n):
                        term1 = d_normalized_vec[i] / math.sqrt(vec_var + layer['epsilon'])
                        term2 = d_mean * (1.0 / n)
                        term3 = d_var * (2.0 / n) * (orig_vec[i] - vec_mean)
                        d_linear[k][i] = term1 + term2 + term3
                
                # Backpropagate through linear transformation
                if l == 0:
                    # Gradient with respect to input sequence
                    for k, d_lin_vec in enumerate(d_linear):
                        d_vec = self._mat_vec(self._transpose(layer['M']), d_lin_vec)
                        for d in range(self.input_dim):
                            grad_seq[k][d] += d_vec[d]
                else:
                    # Prepare gradient for next lower layer
                    d_next = [list(col) for col in zip(*d_linear)]
            
            # Update sequence using gradient descent
            for k in range(n_windows):
                for d in range(self.input_dim):
                    seq[k][d] -= learning_rate * grad_seq[k][d]
            
            # Check convergence
            if abs(prev_loss - loss) < tol:
                break
            prev_loss = loss
        
        return seq

    def generate(self, num_windows, tau=0.0, discrete_mode=False, vocab_size=None):
        """
        Generate sequence of vectors with temperature-controlled randomness.
        Supports both continuous and discrete generation modes.
        
        Args:
            num_windows (int): Number of vectors to generate
            tau (float): Temperature parameter
                - Continuous mode: controls noise amplitude (0=deterministic)
                - Discrete mode: controls sampling randomness (0=greedy, 1=normal, >1=more random)
            discrete_mode (bool): If True, use discrete sampling; if False, use continuous generation
            vocab_size (int): Required for discrete mode - size of vocabulary for each dimension
        
        Returns:
            list: Generated sequence of vectors
        """
        assert self.trained and hasattr(self, 'mean_t'), "Model must be auto-trained first"
        if tau < 0:
            raise ValueError("Temperature must be non-negative")
        if discrete_mode and vocab_size is None:
            raise ValueError("vocab_size must be specified for discrete mode")
        
        generated = []
        current_seq = [[random.gauss(self.mean_t[d], 0.1) for d in range(self.input_dim)] 
                       for _ in range(self.input_seq_len)]
        
        for _ in range(num_windows):
            # Forward pass through the entire model
            current = current_seq
            for layer in self.layers:
                # Apply linear transformation
                linear_out = [self._mat_vec(layer['M'], vec) for vec in current]
                
                # Apply unified layer normalization
                normalized_out, _, _ = self._normalize_layer(linear_out, layer['epsilon'])
                
                # Apply position-dependent transformation
                T_matrix = []
                for k, vec in enumerate(normalized_out):
                    out_dim = len(layer['P'])
                    Tk = [0.0] * out_dim
                    for i in range(out_dim):
                        for j in range(out_dim):
                            period = layer['periods'][i][j]
                            phi = math.cos(2 * math.pi * k / period)
                            Tk[i] += layer['P'][i][j] * vec[j] * phi
                    T_matrix.append(Tk)
                
                # Convert to matrix form: rows=features, columns=time steps
                T_transposed = [list(col) for col in zip(*T_matrix)]
                
                # Apply sequence length transformation: U = T * Linker
                U_matrix = self._mat_mul(T_transposed, layer['Linker'])
                
                # Convert back to sequence of vectors
                current_U = []
                for j in range(len(U_matrix[0])):
                    vec = [U_matrix[i][j] for i in range(len(U_matrix))]
                    current_U.append(vec)
                
                # Handle residual connections
                if layer['use_residual'] == 'separate':
                    residual_feat = [self._mat_vec(layer['residual_proj'], vec) for vec in current]
                    residual_transposed = [list(col) for col in zip(*residual_feat)]
                    residual_U_matrix = self._mat_mul(residual_transposed, layer['residual_linker'])
                    residual = []
                    for j in range(len(residual_U_matrix[0])):
                        vec = [residual_U_matrix[i][j] for i in range(len(residual_U_matrix))]
                        residual.append(vec)
                    current = [self._vec_add(u_vec, r_vec) for u_vec, r_vec in zip(current_U, residual)]
                
                elif layer['use_residual'] == 'shared':
                    residual_feat = [self._mat_vec(layer['M'], vec) for vec in current]
                    residual_transposed = [list(col) for col in zip(*residual_feat)]
                    residual_U_matrix = self._mat_mul(residual_transposed, layer['Linker'])
                    residual = []
                    for j in range(len(residual_U_matrix[0])):
                        vec = [residual_U_matrix[i][j] for i in range(len(residual_U_matrix))]
                        residual.append(vec)
                    current = [self._vec_add(u_vec, r_vec) for u_vec, r_vec in zip(current_U, residual)]
                
                else:
                    if len(current[0]) == len(current_U[0]) and len(current) == len(current_U):
                        residual_transposed = [list(col) for col in zip(*current)]
                        residual_U_matrix = self._mat_mul(residual_transposed, layer['Linker'])
                        residual = []
                        for j in range(len(residual_U_matrix[0])):
                            vec = [residual_U_matrix[i][j] for i in range(len(residual_U_matrix))]
                            residual.append(vec)
                        current = [self._vec_add(u_vec, r_vec) for u_vec, r_vec in zip(current_U, residual)]
                    else:
                        current = current_U
            
            # Get the last output vector
            output_vector = current[-1]
            
            if discrete_mode:
                # === DISCRETE GENERATION MODE ===
                # Apply temperature-controlled discrete sampling
                discrete_vector = []
                
                for i, value in enumerate(output_vector):
                    # Create logits from the continuous value
                    # Assuming value represents a score for each vocabulary item
                    # We need to map the single value to a probability distribution over vocab_size
                    
                    # Method 1: Use value as bias for uniform distribution
                    # This is a simple approach that maintains the value's influence
                    base_logits = [value] * vocab_size
                    
                    # Add some randomness based on temperature
                    if tau > 0:
                        # Add temperature-scaled random noise to logits
                        noisy_logits = [logit + random.gauss(0, tau) for logit in base_logits]
                    else:
                        noisy_logits = base_logits
                    
                    # Apply softmax to get probabilities
                    max_logit = max(noisy_logits)
                    exp_logits = [math.exp(logit - max_logit) for logit in noisy_logits]
                    sum_exp = sum(exp_logits)
                    probabilities = [exp_logit / sum_exp for exp_logit in exp_logits]
                    
                    # Sample from the distribution
                    if tau == 0:
                        # Greedy sampling: choose maximum probability
                        sampled_index = probabilities.index(max(probabilities))
                    else:
                        # Probabilistic sampling
                        rand_val = random.random()
                        cumulative_prob = 0.0
                        sampled_index = 0
                        for j, prob in enumerate(probabilities):
                            cumulative_prob += prob
                            if rand_val <= cumulative_prob:
                                sampled_index = j
                                break
                    
                    # Convert index back to a discrete value (0 to vocab_size-1)
                    discrete_vector.append(float(sampled_index))
                
                output_vector = discrete_vector
                
            else:
                # === CONTINUOUS GENERATION MODE (original behavior) ===
                if tau > 0:
                    # For continuous outputs, apply temperature-scaled Gaussian noise
                    noisy_vector = []
                    for value in output_vector:
                        # Add noise with standard deviation proportional to temperature
                        noise = random.gauss(0, tau * abs(value) + 0.01)
                        noisy_vector.append(value + noise)
                    output_vector = noisy_vector
            
            generated.append(output_vector)
            
            # Update current sequence for next iteration (shift window)
            current_seq = current_seq[1:] + [output_vector]
        
        return generated

    def _softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        e_x = [math.exp(i - max(x)) for i in x]
        return [i / sum(e_x) for i in e_x]

    def _sample_from_distribution(self, probabilities, temperature=1.0):
        """
        Sample from a probability distribution with temperature control.
        
        Args:
            probabilities (list): Probability distribution
            temperature (float): Controls sampling randomness
            
        Returns:
            int: Sampled index
        """
        if temperature != 1.0:
            # Apply temperature scaling
            logits = [math.log(p + 1e-10) for p in probabilities]
            scaled_logits = [logit / temperature for logit in logits]
            exp_logits = [math.exp(logit) for logit in scaled_logits]
            sum_exp = sum(exp_logits)
            probabilities = [exp_logit / sum_exp for exp_logit in exp_logits]
        
        if temperature == 0:
            # Greedy sampling
            return probabilities.index(max(probabilities))
        else:
            # Probabilistic sampling
            rand_val = random.random()
            cumulative_prob = 0.0
            for i, prob in enumerate(probabilities):
                cumulative_prob += prob
                if rand_val <= cumulative_prob:
                    return i
            return len(probabilities) - 1  # Fallback

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
                print(f"{'Linker dims:':<20} {self.linker_dims}")
                print(f"{'Linker trainable:':<20} {self.linker_trainable}")
                print(f"{'Residual types:':<20} {self.use_residual_list}")
                print(f"{'Trained:':<20} {self.trained}")
                # Display epsilon for the first layer (all layers have same epsilon)
                if self.layers:
                    print(f"{'Layer epsilon:':<20} {self.layers[0]['epsilon']}")
            
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
                        matrix = self.layers[l][attr]
                        print(f"    Shape: {len(matrix)}×{len(matrix[0])}")
                        print("    Sample values:")
                        for i in range(min(first_num, len(matrix))):
                            vals = matrix[i][:min(first_num, len(matrix[i]))]
                            if attr == 'P':
                                formatted = [f"{v:.6f}" for v in vals]
                            else:
                                formatted = [str(int(v)) for v in vals]
                            print(f"      Row {i}: [{', '.join(formatted)}" + 
                                  (f", ...]" if len(matrix[i]) > first_num else "]"))
                    
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
        """Count learnable parameters (P matrices, M matrices, and Linker matrices)"""
        total_params = 0
        trainable_params = 0
        print("Parameter Count:")
        
        for l in range(self.num_layers):
            M = self.layers[l]['M']
            P = self.layers[l]['P']
            Linker = self.layers[l]['Linker']
            use_residual = self.layers[l]['use_residual']
            
            M_params = len(M) * len(M[0])
            P_params = len(P) * len(P[0])  # Now 2D matrix
            Linker_params = len(Linker) * len(Linker[0])
            
            layer_params = M_params + P_params + Linker_params
            total_params += layer_params
            
            # Count trainable parameters
            layer_trainable = M_params + P_params
            if self.layers[l]['linker_trainable']:
                layer_trainable += Linker_params
                
            # Add residual parameters if using separate residual
            if use_residual == 'separate':
                residual_proj = self.layers[l]['residual_proj']
                residual_linker = self.layers[l]['residual_linker']
                residual_proj_params = len(residual_proj) * len(residual_proj[0])
                residual_linker_params = len(residual_linker) * len(residual_linker[0])
                layer_params += residual_proj_params + residual_linker_params
                layer_trainable += residual_proj_params
                # Residual linker is always trainable in separate mode
                layer_trainable += residual_linker_params

            trainable_params += layer_trainable
            
            print(f"  Layer {l}:")
            print(f"    M matrix: {len(M)}×{len(M[0])} = {M_params} parameters")
            print(f"    P matrix: {len(P)}×{len(P[0])} = {P_params} parameters")
            print(f"    Linker matrix: {len(Linker)}×{len(Linker[0])} = {Linker_params} parameters")
            if use_residual == 'separate':
                print(f"    Residual proj: {len(residual_proj)}×{len(residual_proj[0])} = {residual_proj_params} parameters")
                print(f"    Residual linker: {len(residual_linker)}×{len(residual_linker[0])} = {residual_linker_params} parameters")
            print(f"    Residual type: {use_residual}")
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

    # Test Case 1: Mixed residual strategies
    print("\n=== Test Case 1: Mixed Residual Strategies ===")
    hndd_mixed = HierDDLpm(
        input_dim=input_dim,
        model_dims=model_dims,
        input_seq_len=input_seq_len,
        linker_dims=linker_dims,
        linker_trainable=[True, False, True],
        use_residual_list=['separate', 'shared', None],  # Different residual for each layer
        epsilon=1e-8  # Set custom epsilon for layer normalization
    )
    
    # Show configuration
    print("\nModel configuration:")
    hndd_mixed.show('config')
    
    # Parameter count
    print("\nParameter count before training:")
    total_params, trainable_params = hndd_mixed.count_parameters()
    
    # Gradient Descent Training
    print("\nTraining with Gradient Descent:")
    grad_history = hndd_mixed.grad_train(
        seqs, 
        t_list, 
        max_iters=100,
        learning_rate=1.0,
        decay_rate=0.999,
        print_every=5
    )
    
    # Predict targets
    print("\nPredictions (first 3 sequences):")
    for i, seq in enumerate(seqs[:3]):
        t_pred = hndd_mixed.predict_t(seq)
        print(f"Seq {i+1}: Target={[f'{x:.4f}' for x in t_list[i]]}")
        print(f"         Predicted={[f'{x:.4f}' for x in t_pred]}")
    
    # Calculate prediction correlations
    preds = [hndd_mixed.predict_t(seq) for seq in seqs]
    correlations = []
    for i in range(model_dims[-1]):
        actual = [t[i] for t in t_list]
        predicted = [p[i] for p in preds]
        corr = correlation(actual, predicted)
        correlations.append(corr)
        print(f"Output dim {i} correlation: {corr:.4f}")
    print(f"Average correlation: {mean(correlations):.4f}")

    # Test Case 2: Save and load model
    print("\nTesting save/load functionality:")
    hndd_mixed.save("hierarchical_ndd_residual_model.pkl")
    loaded = HierDDLpm.load("hierarchical_ndd_residual_model.pkl")
    
    # Verify loaded model
    t_pred_loaded = loaded.predict_t(seqs[0])
    print(f"Prediction from loaded model: {[f'{x:.4f}' for x in t_pred_loaded]}")

    # Test Case 3: Auto-training and sequence generation
    print("\n=== Test Case 3: Auto-training and Sequence Generation ===")
    
    # Create a new model with compatible dimensions for auto-training
    hndd_auto = HierDDLpm(
        input_dim=input_dim,
        model_dims=[input_dim],  # Output dim must match input dim for reconstruction
        input_seq_len=input_seq_len,
        linker_dims=[input_seq_len],  # Sequence length must match for reconstruction
        linker_trainable=False,
        use_residual_list=[None]
    )
    
    # Auto-train in 'gap' mode (self-consistency)
    print("\nAuto-training in 'gap' mode:")
    auto_history = hndd_auto.auto_train(
        seqs,
        auto_mode='gap',
        max_iters=50,
        learning_rate=1.0,
        decay_rate=0.98,
        print_every=5
    )
    
    # Reconstruct representative sequence
    print("\nReconstructing representative sequence:")
    recon_seq = hndd_auto.reconstruct()
    print(f"Reconstructed sequence length: {len(recon_seq)}")
    print(f"First vector: {[f'{x:.4f}' for x in recon_seq[0][:min(5, input_dim)]]}...")
    
    # Generate new sequence with temperature
    print("\nGenerating new sequences:")
    print("Deterministic generation (tau=0):")
    gen_det = hndd_auto.generate(num_windows=10, tau=0)
    print(f"  Generated {len(gen_det)} vectors")
    print(f"  First vector: {[f'{x:.4f}' for x in gen_det[0][:min(5, input_dim)]]}...")
    
    print("\nStochastic generation (tau=0.5):")
    gen_stoch = hndd_auto.generate(num_windows=10, tau=0.5)
    print(f"  Generated {len(gen_stoch)} vectors")
    print(f"  First vector: {[f'{x:.4f}' for x in gen_stoch[0][:min(5, input_dim)]]}...")
    
    # Test Case 4: Auto-training in 'reg' mode (auto-regressive)
    print("\n=== Test Case 4: Auto-regressive Training ===")
    
    # Auto-train in 'reg' mode (next-step prediction)
    print("\nAuto-training in 'reg' mode:")
    auto_history_reg = hndd_auto.auto_train(
        seqs,
        auto_mode='reg',
        max_iters=50,
        learning_rate=0.5,
        decay_rate=0.98,
        print_every=5,
        continued=True  # Continue training existing model
    )
    
    # Generate sequence using auto-regressive model
    print("\nGenerating sequence with auto-regressive model:")
    gen_reg = hndd_auto.generate(num_windows=15, tau=0.3)
    print(f"Generated {len(gen_reg)} vectors")
    print(f"First vector: {[f'{x:.4f}' for x in gen_reg[0][:min(5, input_dim)]]}...")
    print(f"Last vector: {[f'{x:.4f}' for x in gen_reg[-1][:min(5, input_dim)]]}...")