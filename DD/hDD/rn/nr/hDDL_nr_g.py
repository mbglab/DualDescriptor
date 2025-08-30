# Copyright (C) 2005-2025, Bin-Guang Ma (mbg@mail.hzau.edu.cn); SPDX-License-Identifier: MIT
# The Hierarchical Numeric Dual Descriptor (Random AB matrix form) with Linker matrices in pure Python
# Added layer normalization and residual connections and generation capability
# This program is for the demonstration of methodology and not fully refined.
# Author: Bin-Guang Ma (assisted by DeepSeek); Date: 2025-6-24

import math
import random
import pickle

class HierDDrn:
    """
    Hierarchical Numeric Dual Descriptor with Linker for vector sequences with:
      - input_dim: dimension of input vectors
      - model_dims: list of output dimensions for each layer
      - basis_dims: list of basis dimensions for each layer
      - linker_dims: list of output sequence lengths for each layer
      - linker_trainable: controls whether Linker matrices are trainable
      - use_residual_list: list of residual connection types for each layer
    Each layer contains:
      - Matrix M ∈ R^{m_i×m_{i-1}} for linear transformation
      - Matrix Acoeff ∈ R^{m_i×L_i} of coefficients
      - Matrix Bbasis ∈ R^{L_i×m_i} of basis functions
      - Linker matrix for sequence length transformation
      - Layer normalization after linear transformation
      - Residual connection options
    """
    def __init__(self, input_dim=4, model_dims=[4], basis_dims=[50], 
                 linker_dims=[50], input_seq_len=100, linker_trainable=False,
                 use_residual_list=None):
        """
        Initialize the hierarchical Dual Descriptor with Linker matrices.
        
        Args:
            input_dim (int): Input vector dimension
            model_dims (list): List of output dimensions for each layer
            basis_dims (list): List of basis dimensions for each layer
            linker_dims (list): List of output sequence lengths for each layer
            input_seq_len (int): Fixed input sequence length
            linker_trainable (bool|list): Controls if Linker matrices are trainable
            use_residual_list (list|None): Residual connection types for each layer
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
        
        # Process use_residual_list
        if use_residual_list is None:
            self.use_residual_list = [None] * self.num_layers
        elif isinstance(use_residual_list, list):
            if len(use_residual_list) != self.num_layers:
                raise ValueError("use_residual_list length must match number of layers")
            self.use_residual_list = use_residual_list
        else:
            raise TypeError("use_residual_list must be list or None")
        
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
            use_residual = self.use_residual_list[i]
            
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
            
            # Initialize residual parameters if needed
            residual_proj = None
            residual_linker = None
            if use_residual == 'separate':
                # Separate projection for residual path
                residual_proj = [[random.uniform(-0.1, 0.1) for _ in range(in_dim)]
                                 for _ in range(out_dim)]
                residual_linker = [[random.uniform(-0.1, 0.1) for _ in range(out_seq)]
                                   for _ in range(in_seq)]
            
            self.layers.append({
                'M': M,
                'Acoeff': Acoeff,
                'Bbasis': Bbasis,
                'Linker': Linker,
                'linker_trainable': self.linker_trainable[i],
                'use_residual': use_residual,
                'residual_proj': residual_proj,
                'residual_linker': residual_linker,
                'epsilon': 1e-8  # For layer normalization stability
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
    
    def _layer_norm(self, vec, epsilon):
        """
        Apply layer normalization to a vector
        Args:
            vec: input vector
            epsilon: small value for numerical stability
            
        Returns:
            normalized vector, mean, variance
        """
        n = len(vec)
        # Compute mean and variance
        vec_mean = sum(vec) / n
        vec_var = sum((x - vec_mean) ** 2 for x in vec) / n
        
        # Normalize
        std = math.sqrt(vec_var + epsilon)
        normalized = [(x - vec_mean) / std for x in vec]
        return normalized, vec_mean, vec_var

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
            use_residual = layer['use_residual']
            residual_proj = layer['residual_proj']
            residual_linker = layer['residual_linker']
            L_i = len(Bbasis)  # Basis dimension for this layer
            out_dim = len(Acoeff)  # Output dimension for this layer
            
            # Step 1: Apply M transformation to input matrix
            T = self._mat_mul(M, X)  # Feature transformation: out_dim × in_seq
            
            # Step 1.5: Apply layer normalization to each position (column)
            T_normalized = []
            for col_idx in range(len(T[0])):
                # Extract feature vector for this position
                vec = [T[i][col_idx] for i in range(out_dim)]
                
                # Apply layer normalization
                normalized_vec, _, _ = self._layer_norm(vec, layer['epsilon'])
                T_normalized.append(normalized_vec)
            
            # Convert back to matrix format: out_dim × in_seq
            T_norm_mat = [list(col) for col in zip(*T_normalized)]
            
            # Step 2: Compute intermediate matrix U (out_dim × in_seq)
            U = []
            for k in range(len(T_norm_mat[0])):
                vec = [T_norm_mat[i][k] for i in range(len(T_norm_mat))]
                j = k % L_i
                scalar = self._dot(Bbasis[j], vec)
                U_col = [Acoeff[i][j] * scalar for i in range(out_dim)]
                U.append(U_col)
            
            # Convert U to matrix: out_dim × in_seq
            U_mat = [list(col) for col in zip(*U)]
            
            # Step 3: Apply sequence length transformation using Linker matrix
            V = self._mat_mul(U_mat, Linker)  # out_dim × out_seq
            
            # Step 4: Apply residual connection
            if use_residual == 'separate':
                # Separate projection and linker for residual path
                residual_feat = self._mat_mul(residual_proj, X)
                residual = self._mat_mul(residual_feat, residual_linker)
                # Add residual to main output
                for i in range(len(V)):
                    for j in range(len(V[0])):
                        V[i][j] += residual[i][j]
            
            elif use_residual == 'shared':
                # Shared M and Linker for residual
                residual_feat = self._mat_mul(M, X)
                residual = self._mat_mul(residual_feat, Linker)
                # Add residual to main output
                for i in range(len(V)):
                    for j in range(len(V[0])):
                        V[i][j] += residual[i][j]
            
            else:  # None or automatic
                # Identity residual if dimensions match
                if len(X) == len(V) and len(X[0]) == len(V[0]):
                    # Use Linker for sequence length transformation
                    residual = self._mat_mul(X, Linker)
                    # Add residual to main output
                    for i in range(len(V)):
                        for j in range(len(V[0])):
                            V[i][j] += residual[i][j]
            
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
               decay_rate=1.0, print_every=10):
        """
        Train using gradient descent with backpropagation through layers.
        Supports learning M, Acoeff, Bbasis, and Linker matrices.
        Rolls back parameters when loss increases during training.
        Added Layer Normalization after M transformation for better gradient scaling.
        Added support for residual connections.
        """
        if not continued:
            # Reinitialize parameters
            self.__init__(self.input_dim, self.model_dims, self.basis_dims,
                          self.linker_dims, self.input_seq_len, self.linker_trainable,
                          self.use_residual_list)
        
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
        grad_residual_proj_list = []
        grad_residual_linker_list = []
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
            
            # Initialize gradients for residual parameters if they exist
            grad_residual_proj = None
            grad_residual_linker = None
            if layer['use_residual'] == 'separate':
                grad_residual_proj = [[0.0] * in_dim for _ in range(out_dim)]
                grad_residual_linker = [[0.0] * out_seq for _ in range(in_seq)]
            
            grad_A_list.append(grad_A)
            grad_B_list.append(grad_B)
            grad_M_list.append(grad_M)
            grad_Linker_list.append(grad_Linker)
            grad_residual_proj_list.append(grad_residual_proj)
            grad_residual_linker_list.append(grad_residual_linker)
        
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
                if layer['use_residual'] == 'separate':
                    layer_params['residual_proj'] = [row[:] for row in layer['residual_proj']]
                    layer_params['residual_linker'] = [row[:] for row in layer['residual_linker']]
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
                
                # Reset residual gradients if they exist
                if self.layers[l_idx]['use_residual'] == 'separate':
                    for i in range(len(grad_residual_proj_list[l_idx])):
                        grad_residual_proj_list[l_idx][i] = [0.0] * len(grad_residual_proj_list[l_idx][i])
                    for i in range(len(grad_residual_linker_list[l_idx])):
                        grad_residual_linker_list[l_idx][i] = [0.0] * len(grad_residual_linker_list[l_idx][i])
            
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
                    use_residual = layer['use_residual']
                    residual_proj = layer['residual_proj']
                    residual_linker = layer['residual_linker']
                    L_i = len(Bbasis)
                    out_dim = len(Acoeff)
                    
                    # Apply M transformation
                    T = self._mat_mul(M, X)  # out_dim × in_seq
                    
                    # Apply Layer Normalization to T
                    T_normalized = []
                    mean_cache = []
                    var_cache = []
                    for col_idx in range(len(T[0])):
                        # Extract feature vector for this position
                        vec = [T[i][col_idx] for i in range(out_dim)]
                        
                        # Apply layer normalization
                        normalized_vec, vec_mean, vec_var = self._layer_norm(vec, layer['epsilon'])
                        
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
                    
                    # Apply residual connection
                    if use_residual == 'separate':
                        # Separate projection and linker for residual path
                        residual_feat = self._mat_mul(residual_proj, X)
                        residual = self._mat_mul(residual_feat, residual_linker)
                        # Add residual to main output
                        for i in range(len(V)):
                            for j in range(len(V[0])):
                                V[i][j] += residual[i][j]
                    
                    elif use_residual == 'shared':
                        # Shared M and Linker for residual
                        residual_feat = self._mat_mul(M, X)
                        residual = self._mat_mul(residual_feat, Linker)
                        # Add residual to main output
                        for i in range(len(V)):
                            for j in range(len(V[0])):
                                V[i][j] += residual[i][j]
                    
                    else:  # None or automatic
                        # Identity residual if dimensions match
                        if len(X) == len(V) and len(X[0]) == len(V[0]):
                            residual = self._mat_mul(X, Linker)
                            # Add residual to main output
                            for i in range(len(V)):
                                for j in range(len(V[0])):
                                    V[i][j] += residual[i][j]
                    
                    # Save intermediate results
                    forward_cache.append({
                        'X': X,             # Input matrix
                        'T': T,             # After M transformation (unnormalized)
                        'T_norm': T_norm_mat, # After normalization
                        'intermediate_vals': intermediate_vals,  # Position-wise values
                        'U_mat': U_mat,      # After position processing
                        'V': V,             # After Linker transformation
                        'residual': use_residual  # Residual connection type
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
                    use_residual = layer['use_residual']
                    residual_proj = layer['residual_proj']
                    residual_linker = layer['residual_linker']
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
                    
                    # Backpropagate through Layer Normalization
                    dT = [[0.0] * len(T[0]) for _ in range(len(T))]  # dL/dT (unnormalized)
                    
                    # For each position (column) in the sequence
                    for k in range(len(intermediate_vals)):
                        vals = intermediate_vals[k]
                        vec_mean = vals['mean']
                        vec_var = vals['var']
                        orig_vec = vals['original_vec']
                        n = len(orig_vec)  # Number of features
                        epsilon = layer['epsilon']
                        
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
                    
                    # Backpropagate through residual connection
                    if use_residual == 'separate':
                        # Gradient through residual_linker: dL/dresidual_linker = residual_feat^T * dV
                        d_residual_linker = self._mat_mul(self._transpose(residual_feat), dV)
                        
                        # Gradient through residual_feat: dL/dresidual_feat = dV * residual_linker^T
                        d_residual_feat = self._mat_mul(dV, self._transpose(residual_linker))
                        
                        # Gradient through residual_proj: dL/dresidual_proj = d_residual_feat * X^T
                        d_residual_proj = self._mat_mul(d_residual_feat, self._transpose(X))
                        
                        # Gradient through X from residual: dL/dX += residual_proj^T * d_residual_feat
                        dX_res = self._mat_mul(self._transpose(residual_proj), d_residual_feat)
                        for i in range(len(dX_prev)):
                            for j in range(len(dX_prev[0])):
                                dX_prev[i][j] += dX_res[i][j]
                        
                        # Accumulate residual gradients
                        for i in range(len(d_residual_proj)):
                            for j in range(len(d_residual_proj[i])):
                                grad_residual_proj_list[l_idx][i][j] += d_residual_proj[i][j]
                        
                        for i in range(len(d_residual_linker)):
                            for j in range(len(d_residual_linker[i])):
                                grad_residual_linker_list[l_idx][i][j] += d_residual_linker[i][j]
                    
                    elif use_residual == 'shared':
                        # Gradient through shared Linker: dL/dLinker += residual_feat^T * dV
                        dLinker_shared = self._mat_mul(self._transpose(residual_feat), dV)
                        for i in range(len(dLinker)):
                            for j in range(len(dLinker[0])):
                                dLinker[i][j] += dLinker_shared[i][j]
                        
                        # Gradient through shared M: dL/dM += d_residual_feat * X^T
                        d_residual_feat = self._mat_mul(dV, self._transpose(Linker))
                        dM_shared = self._mat_mul(d_residual_feat, self._transpose(X))
                        for i in range(len(dM)):
                            for j in range(len(dM[0])):
                                dM[i][j] += dM_shared[i][j]
                        
                        # Gradient through X: dL/dX += M^T * d_residual_feat
                        dX_res = self._mat_mul(self._transpose(M), d_residual_feat)
                        for i in range(len(dX_prev)):
                            for j in range(len(dX_prev[0])):
                                dX_prev[i][j] += dX_res[i][j]
                    
                    elif use_residual is None:
                        # For identity residual (if dimensions matched)
                        if len(X) == len(V) and len(X[0]) == len(V[0]):
                            # Gradient through Linker: dL/dLinker += X^T * dV
                            dLinker_identity = self._mat_mul(self._transpose(X), dV)
                            for i in range(len(dLinker)):
                                for j in range(len(dLinker[0])):
                                    dLinker[i][j] += dLinker_identity[i][j]
                            
                            # Gradient through X: dL/dX += dV * Linker^T
                            dX_res = self._mat_mul(dV, self._transpose(Linker))
                            for i in range(len(dX_prev)):
                                for j in range(len(dX_prev[0])):
                                    dX_prev[i][j] += dX_res[i][j]
                    
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
                
                # Clip residual gradients if they exist
                if self.layers[l_idx]['use_residual'] == 'separate':
                    for i in range(len(grad_residual_proj_list[l_idx])):
                        for j in range(len(grad_residual_proj_list[l_idx][i])):
                            grad = grad_residual_proj_list[l_idx][i][j]
                            if abs(grad) > max_grad_norm:
                                grad = max_grad_norm if grad > 0 else -max_grad_norm
                            grad_residual_proj_list[l_idx][i][j] = grad / total_sequences
                    
                    for i in range(len(grad_residual_linker_list[l_idx])):
                        for j in range(len(grad_residual_linker_list[l_idx][i])):
                            grad = grad_residual_linker_list[l_idx][i][j]
                            if abs(grad) > max_grad_norm:
                                grad = max_grad_norm if grad > 0 else -max_grad_norm
                            grad_residual_linker_list[l_idx][i][j] = grad / total_sequences
            
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
                
                # Update residual parameters if they exist
                if layer['use_residual'] == 'separate':
                    for i in range(len(layer['residual_proj'])):
                        for j in range(len(layer['residual_proj'][i])):
                            layer['residual_proj'][i][j] -= effective_lr * grad_residual_proj_list[l_idx][i][j]
                    
                    for i in range(len(layer['residual_linker'])):
                        for j in range(len(layer['residual_linker'][i])):
                            layer['residual_linker'][i][j] -= effective_lr * grad_residual_linker_list[l_idx][i][j]
            
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
                    if layer['use_residual'] == 'separate' and 'residual_proj' in prev_params[l_idx]:
                        layer['residual_proj'] = [row[:] for row in prev_params[l_idx]['residual_proj']]
                        layer['residual_linker'] = [row[:] for row in prev_params[l_idx]['residual_linker']]
                
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
                L_i = self.basis_dims[l]
                
                # Reinitialize Acoeff and Bbasis with small random values
                layer['Acoeff'] = [[random.uniform(-0.5, 0.5) for _ in range(L_i)] 
                                   for _ in range(out_dim)]
                layer['Bbasis'] = [[random.uniform(-0.1, 0.1) for _ in range(out_dim)]
                                   for _ in range(L_i)]
                
                # Reinitialize M with small random values
                layer['M'] = [[random.uniform(-0.5, 0.5) for _ in range(in_dim)] 
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
        grad_A_list = []
        grad_B_list = []
        grad_M_list = []
        grad_Linker_list = []
        grad_residual_proj_list = []
        grad_residual_linker_list = []
        
        for l in range(self.num_layers):
            out_dim = self.model_dims[l]
            L_i = self.basis_dims[l]
            
            # Gradients for Acoeff matrix
            grad_A = [[0.0] * L_i for _ in range(out_dim)]
            grad_A_list.append(grad_A)
            
            # Gradients for Bbasis matrix
            grad_B = [[0.0] * out_dim for _ in range(L_i)]
            grad_B_list.append(grad_B)
            
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
                # Reset A gradients
                for i in range(len(grad_A_list[l])):
                    for j in range(len(grad_A_list[l][i])):
                        grad_A_list[l][i][j] = 0.0
                
                # Reset B gradients
                for i in range(len(grad_B_list[l])):
                    for j in range(len(grad_B_list[l][i])):
                        grad_B_list[l][i][j] = 0.0
                
                # Reset M gradients
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
                    
                    # Convert to matrix form for processing
                    X = [list(col) for col in zip(*current)]
                    
                    # Apply M transformation
                    T = self._mat_mul(layer['M'], X)
                    
                    # Apply Layer Normalization with causal masking for reg mode
                    T_normalized = []
                    mean_cache = []
                    var_cache = []
                    
                    for col_idx in range(len(T[0])):
                        # Extract feature vector for this position
                        vec = [T[i][col_idx] for i in range(len(T))]
                        
                        # For reg mode, use only historical information for normalization
                        if auto_mode == 'reg':
                            # Use only positions 0 to col_idx for normalization (causal)
                            historical_vecs = []
                            for i in range(col_idx + 1):
                                historical_vec = [T[j][i] for j in range(len(T))]
                                historical_vecs.append(historical_vec)
                            
                            # Flatten historical vectors for statistics calculation
                            flat_historical = [x for vec in historical_vecs for x in vec]
                            vec_mean = sum(flat_historical) / len(flat_historical)
                            vec_var = sum((x - vec_mean) ** 2 for x in flat_historical) / len(flat_historical)
                            
                            # Normalize current position using historical statistics
                            normalized_vec = [(x - vec_mean) / math.sqrt(vec_var + layer['epsilon']) 
                                            for x in vec]
                        else:
                            # For gap mode, use full sequence normalization
                            normalized_vec, vec_mean, vec_var = self._layer_norm(vec, layer['epsilon'])
                        
                        T_normalized.append(normalized_vec)
                        mean_cache.append(vec_mean)
                        var_cache.append(vec_var)
                    
                    # Convert back to matrix format
                    T_norm_mat = [list(col) for col in zip(*T_normalized)]
                    
                    # Save intermediate values for backprop
                    layer_intermediate['T'] = T
                    layer_intermediate['T_norm'] = T_norm_mat
                    layer_intermediate['mean_cache'] = mean_cache
                    layer_intermediate['var_cache'] = var_cache
                    
                    # Compute intermediate U matrix using normalized T
                    U = []
                    intermediate_vals = []
                    for k in range(len(T_norm_mat[0])):
                        vec = [T_norm_mat[i][k] for i in range(len(T_norm_mat))]
                        j = k % len(layer['Bbasis'])  # Basis index
                        scalar = self._dot(layer['Bbasis'][j], vec)
                        U_col = [layer['Acoeff'][i][j] * scalar for i in range(len(layer['Acoeff']))]
                        U.append(U_col)
                        intermediate_vals.append({
                            'vec': vec,
                            'original_vec': [T[i][k] for i in range(len(T))],
                            'j': j,
                            'scalar': scalar,
                            'U_col': U_col,
                            'mean': mean_cache[k],
                            'var': var_cache[k]
                        })
                    
                    # Convert U to matrix: out_dim × in_seq
                    U_mat = [list(col) for col in zip(*U)]
                    layer_intermediate['U_mat'] = U_mat
                    layer_intermediate['intermediate_vals'] = intermediate_vals
                    
                    # Apply causal masking for Linker matrix in reg mode
                    if auto_mode == 'reg' and layer['linker_trainable']:
                        # Create causal Linker mask: only allow past-to-future information flow
                        causal_linker = []
                        for i in range(len(layer['Linker'])):
                            causal_row = []
                            for j in range(len(layer['Linker'][i])):
                                # Allow connection only if output position >= input position
                                if j >= i:  # Output position j receives info from input position i only if j >= i
                                    causal_row.append(layer['Linker'][i][j])
                                else:
                                    causal_row.append(0.0)
                            causal_linker.append(causal_row)
                        used_linker = causal_linker
                    else:
                        used_linker = layer['Linker']
                    
                    # Apply sequence length transformation: V = U_mat * Linker
                    V = self._mat_mul(U_mat, used_linker)
                    layer_intermediate['V'] = V
                    layer_intermediate['used_linker'] = used_linker
                    
                    # Handle residual connections with causal masking for reg mode
                    residual = None
                    if layer['use_residual'] == 'separate':
                        residual_feat = self._mat_mul(layer['residual_proj'], X)
                        
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
                        
                        residual = self._mat_mul(residual_feat, used_residual_linker)
                        layer_intermediate['residual_feat'] = residual_feat
                        layer_intermediate['residual'] = residual
                        layer_intermediate['used_residual_linker'] = used_residual_linker
                    
                    elif layer['use_residual'] == 'shared':
                        residual_feat = self._mat_mul(layer['M'], X)
                        residual = self._mat_mul(residual_feat, used_linker)  # Use same causal linker
                        layer_intermediate['residual_feat'] = residual_feat
                        layer_intermediate['residual'] = residual
                    
                    else:
                        if len(X) == len(V) and len(X[0]) == len(V[0]):
                            residual = self._mat_mul(X, used_linker)  # Use causal linker
                            layer_intermediate['residual'] = residual
                    
                    # Apply residual if available
                    if residual is not None:
                        for i in range(len(V)):
                            for j in range(len(V[0])):
                                V[i][j] += residual[i][j]
                    
                    # Convert back to sequence of vectors for next layer
                    current = []
                    for j in range(len(V[0])):
                        vec = [V[i][j] for i in range(len(V))]
                        current.append(vec)
                    
                    intermediates.append(layer_intermediate)
                
                # Backward pass through ALL layers
                # Start with output gradient
                d_next_layer_grad = None
                
                for l in range(self.num_layers-1, -1, -1):
                    layer = self.layers[l]
                    interm = intermediates[l]
                    T = interm['T']
                    T_norm = interm['T_norm']
                    U_mat = interm['U_mat']
                    intermediate_vals = interm['intermediate_vals']
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
                                # Gradient: dL/dV_ij = 2*(V_ij - target[i]) / total_samples
                                error = 2 * (current[k][i] - target[i]) / total_samples
                                d_output_k.append(error)
                                total_loss += (current[k][i] - target[i])**2
                            d_output.append(d_output_k)
                            valid_positions += 1
                        
                        if valid_positions == 0:
                            continue
                            
                        # Convert to matrix form for backpropagation (features × time)
                        dV = [list(col) for col in zip(*d_output)]
                        
                    else:
                        # Middle layer gradient: from next layer
                        dV = d_next_layer_grad
                    
                    # Handle residual connection gradient
                    if 'residual' in interm and interm['residual'] is not None:
                        # Gradient flows through both main and residual paths
                        dV = [[dV[i][j] * 0.5 for j in range(len(dV[0]))] 
                              for i in range(len(dV))]
                    
                    # Backpropagate through sequence length transformation
                    # Gradient for Linker: dL/dLinker = U_mat^T * dV
                    dLinker = self._mat_mul(self._transpose(U_mat), dV)
                    
                    # Apply causal masking to Linker gradient in reg mode
                    if auto_mode == 'reg' and layer['linker_trainable']:
                        for i in range(len(dLinker)):
                            for j in range(len(dLinker[i])):
                                if j < i:  # Zero out gradients for non-causal connections
                                    dLinker[i][j] = 0.0
                    
                    # Gradient for U_mat: dL/dU_mat = dV * Linker^T
                    Linker_T = self._transpose(used_linker)
                    dU_mat = self._mat_mul(dV, Linker_T)
                    
                    # Convert dU_mat to list of column gradients
                    dU_cols = []
                    for j in range(len(dU_mat[0])):
                        col = [dU_mat[i][j] for i in range(len(dU_mat))]
                        dU_cols.append(col)
                    
                    # Initialize gradients for position processing
                    dT_norm = [[0.0] * len(T_norm[0]) for _ in range(len(T_norm))]  # dL/dT_norm
                    dAcoeff = [[0.0] * len(layer['Acoeff'][0]) for _ in range(len(layer['Acoeff']))]
                    dBbasis = [[0.0] * len(layer['Bbasis'][0]) for _ in range(len(layer['Bbasis']))]
                    
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
                        for i in range(len(layer['Acoeff'])):
                            dAcoeff[i][j] += dU_col[i] * scalar
                        
                        # Gradient w.r.t scalar
                        d_scalar = sum(dU_col[i] * layer['Acoeff'][i][j] for i in range(len(layer['Acoeff'])))
                        
                        # Gradient w.r.t Bbasis
                        for d in range(len(layer['Bbasis'][j])):
                            dBbasis[j][d] += d_scalar * vec[d]
                        
                        # Gradient w.r.t normalized vec (input to this position)
                        d_vec = [d_scalar * layer['Bbasis'][j][d] for d in range(len(layer['Bbasis'][j]))]
                        
                        # Accumulate gradient for normalized T matrix
                        for d in range(len(d_vec)):
                            dT_norm[d][k] += d_vec[d]
                    
                    # Backpropagate through Layer Normalization
                    dT = [[0.0] * len(T[0]) for _ in range(len(T))]  # dL/dT (unnormalized)
                    
                    # For each position (column) in the sequence
                    for k in range(len(intermediate_vals)):
                        vals = intermediate_vals[k]
                        vec_mean = vals['mean']
                        vec_var = vals['var']
                        orig_vec = vals['original_vec']
                        n = len(orig_vec)  # Number of features
                        epsilon = layer['epsilon']
                        
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
                    X = [list(col) for col in zip(*interm['input_vec'])]  # Input matrix
                    dM_inc = self._mat_mul(dT, self._transpose(X))
                    
                    # Accumulate gradients
                    for i in range(len(dAcoeff)):
                        for j in range(len(dAcoeff[i])):
                            grad_A_list[l][i][j] += dAcoeff[i][j]
                    
                    for i in range(len(dBbasis)):
                        for j in range(len(dBbasis[i])):
                            grad_B_list[l][i][j] += dBbasis[i][j]
                    
                    for i in range(len(dM_inc)):
                        for j in range(len(dM_inc[i])):
                            grad_M_list[l][i][j] += dM_inc[i][j]
                    
                    # Accumulate Linker gradient if trainable
                    if layer['linker_trainable']:
                        for i in range(len(dLinker)):
                            for j in range(len(dLinker[i])):
                                grad_Linker_list[l][i][j] += dLinker[i][j]
                    
                    # Handle residual linker gradients for separate mode
                    if layer['use_residual'] == 'separate' and 'used_residual_linker' in interm:
                        used_residual_linker = interm['used_residual_linker']
                        residual_feat = interm['residual_feat']
                        
                        # Gradient for residual linker: dL/dResidualLinker = residual_feat^T * dV
                        dResidualLinker = self._mat_mul(self._transpose(residual_feat), dV)
                        
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
                        # Gradient w.r.t X (for previous layer): dL/dX = M^T * dT
                        dX_prev = self._mat_mul(self._transpose(layer['M']), dT)
                        
                        # Convert to sequence format for next layer
                        d_next_layer_grad = dX_prev
                
                # End of backward pass for this sequence
            
            # Average gradients across all sequences
            scale = 1.0 / sequence_count if sequence_count > 0 else 1.0
            for l in range(self.num_layers):
                # Average A gradients
                for i in range(len(grad_A_list[l])):
                    for j in range(len(grad_A_list[l][i])):
                        grad_A_list[l][i][j] *= scale
                
                # Average B gradients
                for i in range(len(grad_B_list[l])):
                    for j in range(len(grad_B_list[l][i])):
                        grad_B_list[l][i][j] *= scale
                
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
                self._clip_gradients(grad_A_list[l], grad_clip_threshold)
                self._clip_gradients(grad_B_list[l], grad_clip_threshold)
                self._clip_gradients(grad_M_list[l], grad_clip_threshold)
                if self.layers[l]['linker_trainable']:
                    self._clip_gradients(grad_Linker_list[l], grad_clip_threshold)
                if self.layers[l]['use_residual'] == 'separate':
                    self._clip_gradients(grad_residual_proj_list[l], grad_clip_threshold)
                    self._clip_gradients(grad_residual_linker_list[l], grad_clip_threshold)
            
            # Update parameters for ALL layers
            for l in range(self.num_layers):
                layer = self.layers[l]
                
                # Update Acoeff matrix
                for i in range(len(layer['Acoeff'])):
                    for j in range(len(layer['Acoeff'][i])):
                        layer['Acoeff'][i][j] -= learning_rate * grad_A_list[l][i][j]
                
                # Update Bbasis matrix
                for i in range(len(layer['Bbasis'])):
                    for j in range(len(layer['Bbasis'][i])):
                        layer['Bbasis'][i][j] -= learning_rate * grad_B_list[l][i][j]
                
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
                
                # Store input for backprop
                layer_intermediate['input_vec'] = current
                
                # Convert to matrix form for processing
                X = [list(col) for col in zip(*current)]
                
                # Apply M transformation
                T = self._mat_mul(layer['M'], X)
                
                # Apply Layer Normalization
                T_normalized = []
                mean_cache = []
                var_cache = []
                for col_idx in range(len(T[0])):
                    # Extract feature vector for this position
                    vec = [T[i][col_idx] for i in range(len(T))]
                    
                    # Apply layer normalization
                    normalized_vec, vec_mean, vec_var = self._layer_norm(vec, layer['epsilon'])
                    
                    T_normalized.append(normalized_vec)
                    mean_cache.append(vec_mean)
                    var_cache.append(vec_var)
                
                # Convert back to matrix format
                T_norm_mat = [list(col) for col in zip(*T_normalized)]
                
                # Save intermediate values for backprop
                layer_intermediate['T'] = T
                layer_intermediate['T_norm'] = T_norm_mat
                layer_intermediate['mean_cache'] = mean_cache
                layer_intermediate['var_cache'] = var_cache
                
                # Compute intermediate U matrix using normalized T
                U = []
                intermediate_vals = []
                for k in range(len(T_norm_mat[0])):
                    vec = [T_norm_mat[i][k] for i in range(len(T_norm_mat))]
                    j = k % len(layer['Bbasis'])  # Basis index
                    scalar = self._dot(layer['Bbasis'][j], vec)
                    U_col = [layer['Acoeff'][i][j] * scalar for i in range(len(layer['Acoeff']))]
                    U.append(U_col)
                    intermediate_vals.append({
                        'vec': vec,
                        'original_vec': [T[i][k] for i in range(len(T))],
                        'j': j,
                        'scalar': scalar,
                        'U_col': U_col,
                        'mean': mean_cache[k],
                        'var': var_cache[k]
                    })
                
                # Convert U to matrix: out_dim × in_seq
                U_mat = [list(col) for col in zip(*U)]
                layer_intermediate['U_mat'] = U_mat
                layer_intermediate['intermediate_vals'] = intermediate_vals
                
                # Apply sequence length transformation: V = U_mat * Linker
                V = self._mat_mul(U_mat, layer['Linker'])
                layer_intermediate['V'] = V
                
                # Handle residual connections
                residual = None
                if layer['use_residual'] == 'separate':
                    residual_feat = self._mat_mul(layer['residual_proj'], X)
                    residual = self._mat_mul(residual_feat, layer['residual_linker'])
                    layer_intermediate['residual_feat'] = residual_feat
                    layer_intermediate['residual'] = residual
                
                elif layer['use_residual'] == 'shared':
                    residual_feat = self._mat_mul(layer['M'], X)
                    residual = self._mat_mul(residual_feat, layer['Linker'])
                    layer_intermediate['residual_feat'] = residual_feat
                    layer_intermediate['residual'] = residual
                
                else:
                    if len(X) == len(V) and len(X[0]) == len(V[0]):
                        residual = self._mat_mul(X, layer['Linker'])
                        layer_intermediate['residual'] = residual
                
                # Apply residual if available
                if residual is not None:
                    for i in range(len(V)):
                        for j in range(len(V[0])):
                            V[i][j] += residual[i][j]
                
                # Convert back to sequence of vectors for next layer
                current = []
                for j in range(len(V[0])):
                    vec = [V[i][j] for i in range(len(V))]
                    current.append(vec)
                
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
                T = interm['T']
                T_norm = interm['T_norm']
                U_mat = interm['U_mat']
                intermediate_vals = interm['intermediate_vals']
                mean_cache = interm['mean_cache']
                var_cache = interm['var_cache']
                V = interm['V']
                
                if l == self.num_layers-1:
                    # Gradient of loss with respect to output
                    d_out = []
                    for vec in current:
                        d_vec = [2 * (vec[i] - self.mean_t[i]) / len(current) for i in range(len(vec))]
                        d_out.append(d_vec)
                    # Convert to matrix form
                    dV = [list(col) for col in zip(*d_out)]
                else:
                    dV = d_next
                
                # Handle residual connection gradient
                if 'residual' in interm and interm['residual'] is not None:
                    # Gradient flows through both main and residual paths
                    dV = [[dV[i][j] * 0.5 for j in range(len(dV[0]))] 
                          for i in range(len(dV))]
                
                # Backpropagate through sequence length transformation
                # Gradient for Linker: dL/dLinker = U_mat^T * dV
                dLinker = self._mat_mul(self._transpose(U_mat), dV)
                
                # Gradient for U_mat: dL/dU_mat = dV * Linker^T
                Linker_T = self._transpose(layer['Linker'])
                dU_mat = self._mat_mul(dV, Linker_T)
                
                # Convert dU_mat to list of column gradients
                dU_cols = []
                for j in range(len(dU_mat[0])):
                    col = [dU_mat[i][j] for i in range(len(dU_mat))]
                    dU_cols.append(col)
                
                # Initialize gradients for position processing
                dT_norm = [[0.0] * len(T_norm[0]) for _ in range(len(T_norm))]  # dL/dT_norm
                
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
                    
                    # Gradient w.r.t normalized vec (input to this position)
                    d_scalar = sum(dU_col[i] * layer['Acoeff'][i][j] for i in range(len(layer['Acoeff'])))
                    d_vec = [d_scalar * layer['Bbasis'][j][d] for d in range(len(layer['Bbasis'][j]))]
                    
                    # Accumulate gradient for normalized T matrix
                    for d in range(len(d_vec)):
                        dT_norm[d][k] += d_vec[d]
                
                # Backpropagate through Layer Normalization
                dT = [[0.0] * len(T[0]) for _ in range(len(T))]  # dL/dT (unnormalized)
                
                # For each position (column) in the sequence
                for k in range(len(intermediate_vals)):
                    vals = intermediate_vals[k]
                    vec_mean = vals['mean']
                    vec_var = vals['var']
                    orig_vec = vals['original_vec']
                    n = len(orig_vec)  # Number of features
                    epsilon = layer['epsilon']
                    
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
                X = [list(col) for col in zip(*interm['input_vec'])]  # Input matrix
                dM_inc = self._mat_mul(dT, self._transpose(X))
                
                # Gradient w.r.t X (for previous layer): dL/dX = M^T * dT
                dX_prev = self._mat_mul(self._transpose(layer['M']), dT)
                
                # Convert to sequence format for input gradient
                if l == 0:
                    for k in range(len(dX_prev[0])):
                        d_vec = [dX_prev[i][k] for i in range(len(dX_prev))]
                        for d in range(self.input_dim):
                            grad_seq[k][d] += d_vec[d]
                else:
                    # Prepare gradient for next lower layer
                    d_next = dX_prev
            
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
            # Forward pass through the entire hierarchical model
            current = current_seq
            
            for layer in self.layers:
                # Convert sequence to matrix form for processing
                X = [list(col) for col in zip(*current)]
                
                # Apply M transformation
                T = self._mat_mul(layer['M'], X)
                
                # Apply Layer Normalization
                T_normalized = []
                for col_idx in range(len(T[0])):
                    # Extract feature vector for this position
                    vec = [T[i][col_idx] for i in range(len(T))]
                    
                    # Apply layer normalization
                    normalized_vec, _, _ = self._layer_norm(vec, layer['epsilon'])
                    T_normalized.append(normalized_vec)
                
                # Convert back to matrix format
                T_norm_mat = [list(col) for col in zip(*T_normalized)]
                
                # Compute intermediate U matrix using normalized T
                U = []
                for k in range(len(T_norm_mat[0])):
                    vec = [T_norm_mat[i][k] for i in range(len(T_norm_mat))]
                    j = k % len(layer['Bbasis'])  # Basis index
                    scalar = self._dot(layer['Bbasis'][j], vec)
                    U_col = [layer['Acoeff'][i][j] * scalar for i in range(len(layer['Acoeff']))]
                    U.append(U_col)
                
                # Convert U to matrix: out_dim × in_seq
                U_mat = [list(col) for col in zip(*U)]
                
                # Apply sequence length transformation: V = U_mat * Linker
                V = self._mat_mul(U_mat, layer['Linker'])
                
                # Handle residual connections
                if layer['use_residual'] == 'separate':
                    residual_feat = self._mat_mul(layer['residual_proj'], X)
                    residual = self._mat_mul(residual_feat, layer['residual_linker'])
                    # Add residual to main output
                    for i in range(len(V)):
                        for j in range(len(V[0])):
                            V[i][j] += residual[i][j]
                
                elif layer['use_residual'] == 'shared':
                    residual_feat = self._mat_mul(layer['M'], X)
                    residual = self._mat_mul(residual_feat, layer['Linker'])
                    # Add residual to main output
                    for i in range(len(V)):
                        for j in range(len(V[0])):
                            V[i][j] += residual[i][j]
                
                else:
                    if len(X) == len(V) and len(X[0]) == len(V[0]):
                        residual = self._mat_mul(X, layer['Linker'])
                        # Add residual to main output
                        for i in range(len(V)):
                            for j in range(len(V[0])):
                                V[i][j] += residual[i][j]
                
                # Convert back to sequence of vectors for next layer
                current = []
                for j in range(len(V[0])):
                    vec = [V[i][j] for i in range(len(V))]
                    current.append(vec)
            
            # Get the last output vector from final layer
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
                # === CONTINUOUS GENERATION MODE ===
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

    # Add helper methods for discrete generation
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

    # ---- show state ----
    def show(self, what=None, first_num=5):
        """Display hierarchical model state information"""
        if what is None:
            what = ['params', 'stats']
        elif isinstance(what, str):
            what = ['params', 'stats'] if what == 'all' else [what]
        
        print("Hierarchical HierDDrn with Linker Matrices - Model Status:")
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
            print(f"  Residual types   : {self.use_residual_list}")
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
                use_residual = layer['use_residual']
                
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
                
                # Show residual parameters if they exist
                if use_residual == 'separate':
                    residual_proj = layer['residual_proj']
                    print(f"  Residual projection matrix ({len(residual_proj)}x{len(residual_proj[0])}):")
                    for i in range(min(first_num, len(residual_proj))):
                        row = residual_proj[i][:min(first_num, len(residual_proj[0]))]
                        print(f"    Row {i}: [{', '.join(f'{x:.4f}' for x in row)}" + 
                              (", ...]" if len(residual_proj[0]) > first_num else "]"))
                    
                    residual_linker = layer['residual_linker']
                    print(f"  Residual linker matrix ({len(residual_linker)}x{len(residual_linker[0])}):")
                    for i in range(min(first_num, len(residual_linker))):
                        row = residual_linker[i][:min(first_num, len(residual_linker[0]))]
                        print(f"    Row {i}: [{', '.join(f'{x:.4f}' for x in row)}" + 
                              (", ...]" if len(residual_linker[0]) > first_num else "]"))
                
                print(f"  Residual type: {use_residual}")      

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
            use_residual = layer['use_residual']
            residual_proj = layer['residual_proj']
            residual_linker = layer['residual_linker']
            
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
            
            # Add residual parameters if they exist
            res_proj_params = 0
            res_linker_params = 0
            if use_residual == 'separate':
                res_proj_params = len(residual_proj) * len(residual_proj[0])
                res_linker_params = len(residual_linker) * len(residual_linker[0])
                layer_params += res_proj_params + res_linker_params
                layer_trainable += res_proj_params + res_linker_params
            
            print(f"  Layer {l_idx} (in_dim: {in_dim}, out_dim: {out_dim}, L: {L_i}, in_seq: {in_seq}, out_seq: {out_seq}):")
            print(f"    M: {len(M)}×{len(M[0])} = {m_params} params")
            print(f"    Acoeff: {len(Acoeff)}×{len(Acoeff[0])} = {a_params} params")
            print(f"    Bbasis: {len(Bbasis)}×{len(Bbasis[0])} = {b_params} params")
            print(f"    Linker: {len(Linker)}×{len(Linker[0])} = {linker_params} params [Trainable: {layer['linker_trainable']}]")
            if use_residual == 'separate':
                print(f"    Residual proj: {len(residual_proj)}×{len(residual_proj[0])} = {res_proj_params} params")
                print(f"    Residual linker: {len(residual_linker)}×{len(residual_linker[0])} = {res_linker_params} params")
            print(f"    Residual type: {use_residual}")
            print(f"    Layer total: {layer_params} (Trainable: {layer_trainable})")
            
            trainable_params += layer_trainable
        
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
    residual_types = ['separate', 'shared']  # Residual connection types
    seq_count = 30        # Number of training sequences
    
    # Generate training data with fixed sequence length
    print("Generating training data with fixed sequence length...")
    print(f"Input dimension: {input_dim}, Input seq length: {input_seq_len}")
    print(f"Layer dims: {model_dims}, Basis dims: {basis_dims}, Linker dims: {linker_dims}")
    print(f"Residual types: {residual_types}")
    
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
    
    # Create model with residual connections
    print("\n=== Test Case: Residual Connections ===")
    hdd_residual = HierDDrn(
        input_dim=input_dim,
        model_dims=model_dims,
        basis_dims=basis_dims,
        linker_dims=linker_dims,
        input_seq_len=input_seq_len,
        linker_trainable=True,
        use_residual_list=residual_types
    )
    
    # Show initial model structure
    print("\nInitial model structure:")
    hdd_residual.show('params')
    total_params, trainable_params = hdd_residual.count_parameters()
    
    # Train with gradient descent
    print("\nTraining with Gradient Descent...")
    gd_history = hdd_residual.grad_train(
        seqs,
        t_list,
        learning_rate=5,
        max_iters=50,
        tol=1e-36,
        decay_rate=0.99,
        print_every=10
    )
    
    # Evaluate predictions
    pred_t_list = [hdd_residual.predict_t(seq) for seq in seqs]
    print("\nPrediction correlations per dimension:")
    corrs = []
    for d in range(model_dims[-1]):
        actuals = [t[d] for t in t_list]
        preds = [p[d] for p in pred_t_list]
        corr = correlation(actuals, preds)
        corrs.append(corr)
        print(f"  Dim {d}: correlation = {corr:.4f}")
    
    print(f"Average correlation: {mean(corrs):.4f}")
    
    # Save and load model
    print("\nTesting model persistence...")
    hdd_residual.save("hierarchical_vector_model_linker_residual.pkl")
    loaded = HierDDrn.load("hierarchical_vector_model_linker_residual.pkl")
    print("Loaded model prediction for first sequence:")
    pred = loaded.predict_t(seqs[0])
    print(f"  Predicted target: {[round(p, 4) for p in pred]}")
    
    print("\n=== Hierarchical Vector Sequence Processing with Residual Connections Demo Completed ===")

    # Test Case 3: Auto-training and sequence generation
    print("\n=== Test Case 3: Auto-training and Sequence Generation ===")
    
    # Create a new model with compatible dimensions for auto-training
    hdd_auto = HierDDrn(
        input_dim=input_dim,
        model_dims=[input_dim],  # Output dim must match input dim for reconstruction
        basis_dims=[50],
        input_seq_len=input_seq_len,
        linker_dims=[input_seq_len],  # Sequence length must match for reconstruction
        linker_trainable=False,
        use_residual_list=[None]
    )
    
    # Auto-train in 'gap' mode (self-consistency)
    print("\nAuto-training in 'gap' mode:")
    auto_history = hdd_auto.auto_train(
        seqs,
        auto_mode='gap',
        max_iters=50,
        learning_rate=1.0,
        decay_rate=0.98,
        print_every=5
    )
    
    # Reconstruct representative sequence
    print("\nReconstructing representative sequence:")
    recon_seq = hdd_auto.reconstruct()
    print(f"Reconstructed sequence length: {len(recon_seq)}")
    print(f"First vector: {[f'{x:.4f}' for x in recon_seq[0][:min(5, input_dim)]]}...")
    
    # Generate new sequence with temperature
    print("\nGenerating new sequences:")
    print("Deterministic generation (tau=0):")
    gen_det = hdd_auto.generate(num_windows=10, tau=0)
    print(f"  Generated {len(gen_det)} vectors")
    print(f"  First vector: {[f'{x:.4f}' for x in gen_det[0][:min(5, input_dim)]]}...")
    
    print("\nStochastic generation (tau=0.5):")
    gen_stoch = hdd_auto.generate(num_windows=10, tau=0.5)
    print(f"  Generated {len(gen_stoch)} vectors")
    print(f"  First vector: {[f'{x:.4f}' for x in gen_stoch[0][:min(5, input_dim)]]}...")
    
    # Test Case 4: Auto-training in 'reg' mode (auto-regressive)
    print("\n=== Test Case 4: Auto-regressive Training ===")
    
    # Auto-train in 'reg' mode (next-step prediction)
    print("\nAuto-training in 'reg' mode:")
    auto_history_reg = hdd_auto.auto_train(
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
    gen_reg = hdd_auto.generate(num_windows=15, tau=0.3)
    print(f"Generated {len(gen_reg)} vectors")
    print(f"First vector: {[f'{x:.4f}' for x in gen_reg[0][:min(5, input_dim)]]}...")
    print(f"Last vector: {[f'{x:.4f}' for x in gen_reg[-1][:min(5, input_dim)]]}...")
    
    print("\n=== All Auto-Training Methods Tested Successfully ===")

