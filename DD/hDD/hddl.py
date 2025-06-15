# Copyright (C) 2005-2025, Bin-Guang Ma (mbg@mail.hzau.edu.cn). All rights reserved.
# Hierarchical Dual Descriptor with Linker Matrices
# Author: Bin-Guang Ma; Date: 2025-6-14

import math
import random
import pickle

class LinkedHierarchicalDD:
    """
    Enhanced Hierarchical Dual Descriptor with Linker matrices between layers.
    Each layer transforms both feature dimensions and sequence lengths.
    
    Architecture:
      Layer 0: Input dim = n, Input seq len = L0_in, Output dim = model_dims[0], Output seq len = linker_dims[0]
      Layer 1: Input dim = model_dims[0], Input seq len = linker_dims[0], Output dim = model_dims[1], Output seq len = linker_dims[1]
      ...
      Layer k: Input dim = model_dims[k-1], Input seq len = linker_dims[k-1], Output dim = model_dims[k], Output seq len = linker_dims[k]
    """
    def __init__(self, num_layers, model_dims, linker_dims, input_dim, linker_trainable=False):
        """
        Initialize hierarchical model with linker matrices.
        
        Args:
            num_layers (int): Number of layers in the hierarchy
            model_dims (list): List of output feature dimensions for each layer
            linker_dims (list): List of output sequence lengths for each layer
            input_dim (int): Dimensionality of input vectors
            linker_trainable (bool or list): Controls whether Linker matrices are trainable. 
                If bool: applies to all layers. If list: must match num_layers length.
        """
        if num_layers != len(model_dims) or num_layers != len(linker_dims):
            raise ValueError("num_layers must match lengths of model_dims and linker_dims")
            
        self.num_layers = num_layers
        self.model_dims = model_dims
        self.linker_dims = linker_dims
        self.input_dim = input_dim
        self.trained = False
        
        # Process linker_trainable parameter
        if isinstance(linker_trainable, bool):
            self.linker_trainable = [linker_trainable] * num_layers
        elif isinstance(linker_trainable, list):
            if len(linker_trainable) != num_layers:
                raise ValueError("linker_trainable list must match num_layers length")
            self.linker_trainable = linker_trainable
        else:
            raise TypeError("linker_trainable must be bool or list of bools")
        
        # Initialize layers
        self.layers = []
        for i in range(num_layers):
            layer = {}
            # Determine input dimension for this layer
            in_dim = input_dim if i == 0 else model_dims[i-1]
            out_dim = model_dims[i]
            out_seq_len = linker_dims[i]
            
            layer['m'] = out_dim  # Output feature dimension
            layer['n'] = in_dim   # Input feature dimension
            layer['linker_dim'] = out_seq_len  # Output sequence length
            layer['input_seq_len'] = None  # Will be set during training
            layer['linker_trainable'] = self.linker_trainable[i]  # Store per-layer flag
            
            # Initialize M matrix (out_dim x in_dim)
            layer['M'] = [[random.uniform(-0.5, 0.5) for _ in range(in_dim)] 
                          for _ in range(out_dim)]
            
            # Placeholders for other parameters
            layer['Acoeff'] = None
            layer['Bbasis'] = None
            layer['Linker'] = None  # Linker matrix (input_seq_len x linker_dim)
            
            # Cache transposed matrices for efficiency
            layer['Acoeff_t'] = None
            layer['Bbasis_t'] = None
            layer['Linker_t'] = None
            
            self.layers.append(layer)
    
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
    
    def _vec_outer(self, u, v):
        """Outer product of two vectors: u (m) and v (n) → matrix (m×n)"""
        return [[u_i * v_j for v_j in v] for u_i in u]
    
    # ---- initialization ----
    def initialize(self, seqs):
        """Initialize model parameters based on training sequences"""
        L_max = max(len(seq) for seq in seqs)  # Maximum input sequence length
        
        for i, layer in enumerate(self.layers):
            # Set input sequence length for this layer
            if i == 0:
                layer['input_seq_len'] = L_max  # First layer uses max sequence length
            else:
                # Subsequent layers use previous layer's output sequence length
                layer['input_seq_len'] = self.layers[i-1]['linker_dim']
                
            m = layer['m']
            n = layer['n']
            in_seq_len = layer['input_seq_len']
            out_seq_len = layer['linker_dim']
            
            # Initialize Acoeff (m×out_seq_len)
            layer['Acoeff'] = [[random.uniform(-0.1, 0.1) for _ in range(out_seq_len)]
                              for _ in range(m)]
            layer['Acoeff_t'] = self._transpose(layer['Acoeff'])
            
            # Initialize Bbasis (out_seq_len×m)
            layer['Bbasis'] = [[random.uniform(-0.1, 0.1) for _ in range(m)]
                              for _ in range(out_seq_len)]
            layer['Bbasis_t'] = self._transpose(layer['Bbasis'])
            
            # Initialize Linker matrix (in_seq_len×out_seq_len)
            layer['Linker'] = [[random.uniform(-0.1, 0.1) for _ in range(out_seq_len)]
                              for _ in range(in_seq_len)]
            layer['Linker_t'] = self._transpose(layer['Linker'])
    
    # ---- forward pass for a single layer ----
    def _forward_layer(self, layer, input_seq):
        """
        Forward pass for a single layer with Linker matrix.
        
        Args:
            layer: Layer dictionary with parameters
            input_seq: Input sequence (list of vectors)
            
        Returns:
            output_seq: Output sequence (list of vectors)
            cache: Intermediate values for backpropagation
        """
        in_seq_len = layer['input_seq_len']
        out_seq_len = layer['linker_dim']
        n = layer['n']
        m = layer['m']
        T = len(input_seq)  # Actual sequence length
        
        # Pad or truncate input sequence to required length
        if T < in_seq_len:
            padded_seq = input_seq + [[0.0]*n for _ in range(in_seq_len - T)]
        else:
            padded_seq = input_seq[:in_seq_len]
        
        # Convert sequence to matrix: H (n×in_seq_len)
        H = self._transpose(padded_seq)  # Transpose to get n×in_seq_len
        
        # Step 1: Transform features: X = M * H (m×in_seq_len)
        X = self._mat_mul(layer['M'], H)
        
        # Step 2: Transform sequence length: X_link = X * Linker (m×out_seq_len)
        X_link = self._mat_mul(X, layer['Linker'])
        
        # Transpose to sequence format: (out_seq_len vectors of m dimensions)
        X_link_transposed = self._transpose(X_link)
        
        # Step 3: Process each position in the new sequence
        output_seq = []
        z_list = []
        for t in range(out_seq_len):
            x_t = X_link_transposed[t]  # Feature vector at position t
            
            # Compute basis: z_t = Bbasis * x_t (out_seq_len dimensions)
            z_t = self._mat_vec(layer['Bbasis'], x_t)
            
            # Compute output: o_t = Acoeff * z_t (m dimensions)
            o_t = self._mat_vec(layer['Acoeff'], z_t)
            
            output_seq.append(o_t)
            z_list.append(z_t)
        
        # Cache intermediate values for backpropagation
        cache = {
            'H': H,
            'X': X,
            'X_link': X_link,
            'X_link_transposed': X_link_transposed,
            'z_list': z_list,
            'padded_seq': padded_seq,
            'original_len': T
        }
        
        return output_seq, cache
    
    # ---- full forward pass ----
    def forward(self, seq):
        """
        Full forward pass through all layers.
        
        Args:
            seq: Input sequence (list of n-dimensional vectors)
            
        Returns:
            output_seq: Output sequence from last layer
            all_caches: Intermediate values from all layers
        """
        current_seq = seq
        all_caches = []
        
        for layer in self.layers:
            current_seq, cache = self._forward_layer(layer, current_seq)
            all_caches.append(cache)
            
        return current_seq, all_caches
    
    # ---- predict target vector ----
    def predict_t(self, seq):
        """
        Predict target vector for a sequence by averaging last layer outputs.
        
        Args:
            seq: Input sequence (list of vectors)
            
        Returns:
            list: Predicted target vector
        """
        output_seq, _ = self.forward(seq)
        if not output_seq:
            return [0.0] * self.model_dims[-1]
        
        # Average all position vectors
        t_pred = [0.0] * self.model_dims[-1]
        for vec in output_seq:
            for i in range(self.model_dims[-1]):
                t_pred[i] += vec[i]
        
        n_pos = len(output_seq)
        return [x / n_pos for x in t_pred]
    
    # ---- gradient-based training ----
    def grad_train(self, seqs, t_list, learning_rate=0.01, max_iters=100, 
                  tol=1e-9, print_every=10):
        """
        Train hierarchical model with Linker matrices using gradient descent.
        
        Args:
            seqs: List of training sequences
            t_list: List of target vectors
            learning_rate: Step size
            max_iters: Maximum iterations
            tol: Convergence tolerance
            print_every: Print interval
            
        Returns:
            list: Training loss history
        """
        self.initialize(seqs)
        history = []
        prev_loss = float('inf')
        
        for it in range(max_iters):
            # Initialize gradients for all layers
            grads = []
            for layer in self.layers:
                m = layer['m']
                n = layer['n']
                in_seq_len = layer['input_seq_len']
                out_seq_len = layer['linker_dim']
                
                grad_A = [[0.0]*out_seq_len for _ in range(m)]
                grad_B = [[0.0]*m for _ in range(out_seq_len)]
                grad_M = [[0.0]*n for _ in range(m)]
                grad_Linker = [[0.0]*out_seq_len for _ in range(in_seq_len)] if layer['linker_trainable'] else None
                
                grads.append({
                    'grad_A': grad_A,
                    'grad_B': grad_B,
                    'grad_M': grad_M,
                    'grad_Linker': grad_Linker
                })
            
            total_loss = 0.0
            total_sequences = len(seqs)
            
            # Process all sequences
            for seq_idx, (seq, t) in enumerate(zip(seqs, t_list)):
                # Forward pass (store all intermediate values)
                output_seq, all_caches = self.forward(seq)
                T_out = len(output_seq)  # Output sequence length
                
                # Calculate error for each position in output sequence
                position_errors = []
                for pos in range(T_out):
                    error_vec = [output_seq[pos][i] - t[i] 
                                for i in range(self.model_dims[-1])]
                    position_errors.append(error_vec)
                    total_loss += sum(e*e for e in error_vec)
                
                # Backpropagation through layers (last to first)
                # Start with output errors from the last layer
                layer_errors = position_errors
                
                for layer_idx in range(self.num_layers-1, -1, -1):
                    layer = self.layers[layer_idx]
                    cache = all_caches[layer_idx]
                    grad_layer = grads[layer_idx]
                    m = layer['m']
                    n = layer['n']
                    out_seq_len = layer['linker_dim']
                    linker_trainable = layer['linker_trainable']
                    
                    # Initialize errors for previous layer
                    prev_layer_errors = []
                    
                    # Unpack cached values
                    X_link_transposed = cache['X_link_transposed']
                    z_list = cache['z_list']
                    
                    # Compute dL/dX_link (m×out_seq_len)
                    dL_dX_link = [[0.0]*out_seq_len for _ in range(m)]
                    
                    # Process each position in the sequence
                    for pos in range(len(layer_errors)):
                        # Unpack current position data
                        error_t = layer_errors[pos]  # dL/do_t (m-dimensional)
                        z_t = z_list[pos]            # z_t at position t
                        x_t = X_link_transposed[pos]  # x_t at position t
                        
                        # Compute gradient for Acoeff: dL/dAcoeff += error_t * z_t^T
                        for i in range(m):
                            for j in range(out_seq_len):
                                grad_layer['grad_A'][i][j] += 2 * error_t[i] * z_t[j]
                        
                        # Compute dL/dz_t = Acoeff^T * error_t
                        dL_dz_t = self._mat_vec(layer['Acoeff_t'], error_t)
                        
                        # Compute gradient for Bbasis: dL/dBbasis += dL/dz_t * x_t^T
                        for i in range(out_seq_len):
                            for j in range(m):
                                grad_layer['grad_B'][i][j] += 2 * dL_dz_t[i] * x_t[j]
                        
                        # Compute dL/dx_t = Bbasis^T * dL/dz_t
                        dL_dx_t = self._mat_vec(layer['Bbasis_t'], dL_dz_t)
                        
                        # Accumulate dL/dX_link (transposed position)
                        for j in range(m):
                            dL_dX_link[j][pos] += 2 * dL_dx_t[j]
                    
                    # Compute gradient for Linker matrix only if trainable
                    if linker_trainable:
                        # dL/dLinker = X^T * dL/dX_link
                        X = cache['X']  # m×in_seq_len
                        dL_dX_link_mat = dL_dX_link  # m×out_seq_len
                        
                        # Compute dL/dLinker = X^T * dL/dX_link (in_seq_len×out_seq_len)
                        for i in range(len(X[0])):  # in_seq_len
                            for j in range(len(dL_dX_link_mat[0])):  # out_seq_len
                                for k in range(len(X)):  # m
                                    grad_layer['grad_Linker'][i][j] += 2 * X[k][i] * dL_dX_link_mat[k][j]
                    
                    # Compute dL/dX = dL/dX_link * Linker^T
                    Linker_t = layer['Linker_t']  # out_seq_len×in_seq_len
                    dL_dX = self._mat_mul(dL_dX_link, Linker_t)  # m×in_seq_len
                    
                    # Compute gradient for M matrix: dL/dM = dL/dX * H^T
                    H = cache['H']  # n×in_seq_len
                    H_t = self._transpose(H)  # in_seq_len×n
                    
                    for i in range(m):
                        for j in range(n):
                            for k in range(len(H_t)):
                                grad_layer['grad_M'][i][j] += 2 * dL_dX[i][k] * H_t[k][j]
                    
                    # Compute dL/dH = M^T * dL/dX (for previous layer)
                    M_t = self._transpose(layer['M'])  # n×m
                    dL_dH = self._mat_mul(M_t, dL_dX)  # n×in_seq_len
                    
                    # Convert to sequence format for previous layer
                    dL_dH_seq = self._transpose(dL_dH)
                    
                    # Adjust for padding: only propagate errors for actual positions
                    original_len = cache['original_len']
                    if original_len < len(dL_dH_seq):
                        prev_layer_errors = dL_dH_seq[:original_len]
                    else:
                        prev_layer_errors = dL_dH_seq
                    
                    # Set errors for next layer (towards input)
                    layer_errors = prev_layer_errors
            
            # Average gradients
            if total_sequences > 0:
                norm = 1.0 / total_sequences
                for grad_dict in grads:
                    for grad_name, grad_matrix in grad_dict.items():
                        if grad_matrix is None:  # Skip None gradients (untrainable Linker)
                            continue
                        for row in grad_matrix:
                            for j in range(len(row)):
                                row[j] *= norm
            
            # Update parameters
            for layer_idx, layer in enumerate(self.layers):
                grad_layer = grads[layer_idx]
                m = layer['m']
                n = layer['n']
                in_seq_len = layer['input_seq_len']
                out_seq_len = layer['linker_dim']
                linker_trainable = layer['linker_trainable']
                
                # Update Acoeff
                for i in range(m):
                    for j in range(out_seq_len):
                        layer['Acoeff'][i][j] -= learning_rate * grad_layer['grad_A'][i][j]
                layer['Acoeff_t'] = self._transpose(layer['Acoeff'])
                
                # Update Bbasis
                for i in range(out_seq_len):
                    for j in range(m):
                        layer['Bbasis'][i][j] -= learning_rate * grad_layer['grad_B'][i][j]
                layer['Bbasis_t'] = self._transpose(layer['Bbasis'])
                
                # Update Linker matrix only if trainable
                if linker_trainable and grad_layer['grad_Linker'] is not None:
                    for i in range(in_seq_len):
                        for j in range(out_seq_len):
                            layer['Linker'][i][j] -= learning_rate * grad_layer['grad_Linker'][i][j]
                    layer['Linker_t'] = self._transpose(layer['Linker'])
                
                # Update M
                for i in range(m):
                    for j in range(n):
                        layer['M'][i][j] -= learning_rate * grad_layer['grad_M'][i][j]
            
            # Calculate average loss
            avg_loss = total_loss / (len(seqs) * self.linker_dims[-1]) if seqs else 0.0
            history.append(avg_loss)
            
            # Print progress
            if it % print_every == 0 or it == max_iters - 1:
                print(f"Iter {it:3d}: Loss = {avg_loss:.9f}")
            
            # Check convergence
            if abs(prev_loss - avg_loss) < tol:
                print(f"Converged after {it} iterations")
                break
            prev_loss = avg_loss
        
        self.trained = True
        return history
    
    # ---- show model state ----
    def show(self):
        """Display model state"""
        print("LinkedHierarchicalDD status:")
        print(f" Layers: {self.num_layers}, Model dims: {self.model_dims}")
        print(f" Linker dims: {self.linker_dims}, Input dim: {self.input_dim}")
        print(f" Linker trainable: {self.linker_trainable}")
        for i, layer in enumerate(self.layers):
            print(f" Layer {i}: in_features={layer['n']}, out_features={layer['m']}")
            print(f"  in_seq_len={layer['input_seq_len']}, out_seq_len={layer['linker_dim']}")
            print(f"  Linker trainable: {layer['linker_trainable']}")
            print(f"  Acoeff[0][:3]: {layer['Acoeff'][0][:3]}")
            print(f"  Bbasis[0][:3]: {layer['Bbasis'][0][:3]}")
            print(f"  Linker[0][:3]: {layer['Linker'][0][:3]}")
            print(f"  M[0][:3]: {layer['M'][0][:3]}")

    def count_parameters(self):
        """Calculate total number of trainable parameters"""
        total = 0
        for i, layer in enumerate(self.layers):
            m = layer['m']
            n = layer['n']
            in_seq_len = layer['input_seq_len']
            out_seq_len = layer['linker_dim']
            linker_trainable = layer['linker_trainable']
            
            params_M = m * n
            params_Linker = in_seq_len * out_seq_len if linker_trainable else 0
            params_Bbasis = out_seq_len * m
            params_Acoeff = m * out_seq_len
            
            layer_params = params_M + params_Linker + params_Bbasis + params_Acoeff
            total += layer_params
            
            print(f"Layer {i}: {layer_params} params "
                  f"(M:{params_M}, Linker:{params_Linker if linker_trainable else 'FIXED'}, "
                  f"B:{params_Bbasis}, A:{params_Acoeff})")
        
        print(f"TOTAL PARAMETERS: {total}")
        return total
    
    # ---- save/load model ----
    def save(self, filename):
        """Save model to file"""
        with open(filename, 'wb') as f:
            pickle.dump(self.__dict__, f)
        print(f"Model saved to {filename}")
    
    @classmethod
    def load(cls, filename):
        """Load model from file"""
        with open(filename, 'rb') as f:
            state = pickle.load(f)
        
        # Create instance without __init__
        obj = cls.__new__(cls)
        obj.__dict__.update(state)
        print(f"Model loaded from {filename}")
        return obj


# === Example Usage ===
if __name__ == "__main__":

    from statistics import correlation, mean
    
    # Configuration
    input_dim = 30          # Input vector dimension
    model_dims = [20, 10, 5]  # Output feature dimensions for each layer
    linker_dims = [40, 20, 10]  # Output sequence lengths for each layer
    n_layers = len(model_dims)
    n_seqs = 10             # Number of training sequences
    
    # Generate training data
    seqs, t_list = [], []
    for _ in range(n_seqs):
        L = random.randint(80, 100)  # Variable sequence length
        seq = [[random.uniform(-1, 1) for _ in range(input_dim)] 
              for _ in range(L)]
        seqs.append(seq)
        # Target vector (dimension = last layer's output dim)
        t_list.append([random.uniform(-1, 1) for _ in range(model_dims[-1])])
    
    # Case 1: All Linker matrices trainable
    print("\nTraining with ALL Linker matrices trainable:")
    lhdd_trainable = LinkedHierarchicalDD(
        num_layers=n_layers,
        model_dims=model_dims,
        linker_dims=linker_dims,
        input_dim=input_dim,
        linker_trainable=True  # All layers trainable
    )
    
    history_trainable = lhdd_trainable.grad_train(
        seqs, 
        t_list,
        learning_rate=0.3,
        max_iters=200,
        print_every=20
    )
    
    # Calculate prediction correlations
    preds_trainable = [lhdd_trainable.predict_t(seq) for seq in seqs]
    correlations_trainable = []
    for i in range(model_dims[-1]):
        actual = [t[i] for t in t_list]
        predicted = [p[i] for p in preds_trainable]
        if len(set(actual)) > 1:  # Ensure variance for correlation
            corr = correlation(actual, predicted)
            correlations_trainable.append(corr)
            print(f"Dim {i} correlation: {corr:.4f}")
        else:
            print(f"Dim {i} has no variance, skipping correlation")
    
    if correlations_trainable:
        print(f"Average correlation (trainable Linkers): {mean(correlations_trainable):.4f}")
    else:
        print("No valid correlations calculated")
    
    # Case 2: Mixed trainable status (first layer fixed, others trainable)
    print("\nTraining with MIXED Linker trainability:")
    lhdd_mixed = LinkedHierarchicalDD(
        num_layers=n_layers,
        model_dims=model_dims,
        linker_dims=linker_dims,
        input_dim=input_dim,
        linker_trainable=[False, True, True]  # Layer-specific control
    )
    
    history_mixed = lhdd_mixed.grad_train(
        seqs, 
        t_list,
        learning_rate=0.3,
        max_iters=200,
        print_every=20
    )
    
    # Calculate prediction correlations
    preds_mixed = [lhdd_mixed.predict_t(seq) for seq in seqs]
    correlations_mixed = []
    for i in range(model_dims[-1]):
        actual = [t[i] for t in t_list]
        predicted = [p[i] for p in preds_mixed]
        if len(set(actual)) > 1:
            corr = correlation(actual, predicted)
            correlations_mixed.append(corr)
            print(f"Dim {i} correlation: {corr:.4f}")
        else:
            print(f"Dim {i} has no variance, skipping correlation")
    
    if correlations_mixed:
        print(f"Average correlation (mixed Linkers): {mean(correlations_mixed):.4f}")
    else:
        print("No valid correlations calculated")
    
    # Case 3: All Linker matrices fixed
    print("\nTraining with NO Linker matrices trainable:")
    lhdd_fixed = LinkedHierarchicalDD(
        num_layers=n_layers,
        model_dims=model_dims,
        linker_dims=linker_dims,
        input_dim=input_dim,
        linker_trainable=False  # All layers fixed
    )
    
    history_fixed = lhdd_fixed.grad_train(
        seqs, 
        t_list,
        learning_rate=0.3,
        max_iters=200,
        print_every=20
    )
    
    # Calculate prediction correlations
    preds_fixed = [lhdd_fixed.predict_t(seq) for seq in seqs]
    correlations_fixed = []
    for i in range(model_dims[-1]):
        actual = [t[i] for t in t_list]
        predicted = [p[i] for p in preds_fixed]
        if len(set(actual)) > 1:
            corr = correlation(actual, predicted)
            correlations_fixed.append(corr)
            print(f"Dim {i} correlation: {corr:.4f}")
        else:
            print(f"Dim {i} has no variance, skipping correlation")
    
    if correlations_fixed:
        print(f"Average correlation (fixed Linkers): {mean(correlations_fixed):.4f}")
    else:
        print("No valid correlations calculated")
    
    # Compare parameter counts
    print("\nParameter counts comparison:")
    print("Trainable Linkers model:")
    lhdd_trainable.count_parameters()
    print("\nMixed Linkers model:")
    lhdd_mixed.count_parameters()
    print("\nFixed Linkers model:")
    lhdd_fixed.count_parameters()
