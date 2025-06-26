# Copyright (C) 2005-2025, Bin-Guang Ma (mbg@mail.hzau.edu.cn). All rights reserved.
# Linked Hierarchical Dual Descriptor class with inter-layer Linker matrices
# Author: Bin-Guang Ma; Date: 2025-6-22

import math
import random
import pickle
from statistics import correlation, mean

class LinkedHierarchicalDD:
    """
    Hierarchical Dual Descriptor with inter-layer Linker matrices.
    Each layer transforms input dimensions and sequence lengths.
    
    Architecture:
      Layer 0: Input dim = n, Input len = L0, Output dim = model_dims[0], Output len = linker_dims[0]
      Layer 1: Input dim = model_dims[0], Input len = linker_dims[0], Output dim = model_dims[1], Output len = linker_dims[1]
      ...
    """
    def __init__(self, input_dim=10, input_seq_len=100, model_dims=[10], basis_dims=[50], linker_dims=[50], 
                 linker_trainable=False):
        """
        Initialize linked hierarchical model.
        
        Args:
            input_dim (int): Dimensionality of input vectors (n)
            input_seq_len (int): Length of input sequences (L0)
            model_dims (list): List of model dimensions for each layer
            basis_dims (list): List of basis dimensions for each layer
            linker_dims (list): List of output sequence lengths for each layer
            linker_trainable (bool or list): Controls if Linker matrices are trainable. 
                - If bool: applies to all layers
                - If list: per-layer control (must match num_layers)
        """
        self.n = input_dim
        self.L0 = input_seq_len
        self.model_dims = model_dims
        self.basis_dims = basis_dims
        self.linker_dims = linker_dims
        self.num_layers = len(model_dims)
        self.trained = False
        
        # Validate dimensions
        if len(basis_dims) != self.num_layers or len(linker_dims) != self.num_layers:
            raise ValueError("model_dims, basis_dims, and linker_dims must have same length")
        
        # Process linker_trainable parameter
        if isinstance(linker_trainable, bool):
            self.linker_trainable = [linker_trainable] * self.num_layers
        elif isinstance(linker_trainable, list) and len(linker_trainable) == self.num_layers:
            self.linker_trainable = linker_trainable
        else:
            raise ValueError("linker_trainable must be bool or list of bools matching num_layers")
        
        # Initialize layers
        self.layers = []
        for i in range(self.num_layers):
            # Determine input dim and sequence length for this layer
            if i == 0:
                layer_input_dim = input_dim
                layer_input_len = input_seq_len
            else:
                layer_input_dim = model_dims[i-1]
                layer_input_len = linker_dims[i-1]
                
            layer_output_len = linker_dims[i]
                
            layer = {
                'input_dim': layer_input_dim,
                'input_len': layer_input_len,
                'model_dim': model_dims[i],
                'basis_dim': basis_dims[i],
                'output_len': layer_output_len,
                'linker_trainable': self.linker_trainable[i],  # Store trainable flag
                'M': self._init_matrix(model_dims[i], layer_input_dim),           # Left multiplication matrix
                'Acoeff': self._init_matrix(model_dims[i], basis_dims[i]),        # Coefficient matrix
                'Bbasis': self._init_matrix(basis_dims[i], model_dims[i]),        # Basis matrix
                'Linker': self._init_matrix(layer_input_len, layer_output_len)    # Right multiplication matrix
            }
            # Precompute transposes for efficiency
            layer['B_t'] = self._transpose(layer['Bbasis'])
            layer['Linker_t'] = self._transpose(layer['Linker'])
            self.layers.append(layer)
    
    def _init_matrix(self, rows, cols):
        """Initialize matrix with uniform random values"""
        return [[random.uniform(-0.5, 0.5) for _ in range(cols)] 
                for _ in range(rows)]
    
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
    
    def _vec_to_mat(self, vec_list):
        """Convert list of vectors to column matrix (d x L)"""
        d = len(vec_list[0])
        L = len(vec_list)
        mat = [[0.0]*L for _ in range(d)]
        for j in range(L):
            for i in range(d):
                mat[i][j] = vec_list[j][i]
        return mat
    
    def _mat_to_vec(self, mat):
        """Convert column matrix (d x L) to list of vectors"""
        d = len(mat)
        L = len(mat[0]) if d > 0 else 0
        vec_list = []
        for j in range(L):
            vec = [mat[i][j] for i in range(d)]
            vec_list.append(vec)
        return vec_list
    
    # ---- forward propagation ----
    def forward_layer(self, layer_idx, input_seq):
        """
        Forward pass for a single layer with Linker matrix.
        
        Args:
            layer_idx: Index of layer to process
            input_seq: List of input vectors (length must match layer's input_len)
            
        Returns:
            List of output descriptor vectors (length = layer's output_len)
        """
        layer = self.layers[layer_idx]
        
        # Step 1: Convert input to matrix (d_in x L_in)
        X = self._vec_to_mat(input_seq)
        
        # Step 2: Apply Linker matrix (right multiplication): T1 = X * Linker (d_in x L_out)
        T1 = self._mat_mul(X, layer['Linker'])
        
        # Step 3: Apply M matrix (left multiplication): T2 = M * T1 (d_out x L_out)
        T2 = self._mat_mul(layer['M'], T1)
        
        # Step 4: Convert to vector sequence
        intermediate_seq = self._mat_to_vec(T2)
        
        # Step 5: Apply dual descriptor transformation
        output_seq = []
        for v in intermediate_seq:
            # Compute z = Bbasis * v
            z = self._mat_vec(layer['Bbasis'], v)
            # Compute Nk = Acoeff * z
            Nk = self._mat_vec(layer['Acoeff'], z)
            output_seq.append(Nk)
            
        return output_seq
    
    def predict_t(self, seq):
        """
        Predict target vector by propagating through all layers.
        
        Args:
            seq: Input sequence of n-dimensional vectors (length must be L0)
            
        Returns:
            Predicted target vector (m-dimensional)
        """
        if len(seq) != self.L0:
            raise ValueError(f"Input sequence length must be {self.L0}, got {len(seq)}")
        
        # Propagate through all layers
        current_seq = seq
        for i in range(self.num_layers):
            current_seq = self.forward_layer(i, current_seq)
        
        # Average output of last layer
        if not current_seq:
            return [0.0] * self.model_dims[-1]
        
        t_pred = [0.0] * self.model_dims[-1]
        for vec in current_seq:
            for i in range(self.model_dims[-1]):
                t_pred[i] += vec[i]
        
        n_pos = len(current_seq)
        return [x / n_pos for x in t_pred]
    
    # ---- training ----
    def grad_train(self, seqs, t_list, learning_rate=0.01, max_iters=100, 
                  tol=1e-9, print_every=10, lr_decay=1.0):
        """
        Train linked hierarchical model using gradient descent.
        
        Args:
            seqs: List of training sequences (n-dim, each of length L0)
            t_list: List of target vectors (m-dim)
            learning_rate: Initial step size
            max_iters: Maximum iterations
            tol: Convergence tolerance
            print_every: Print interval
            lr_decay: Learning rate decay factor per iteration
            
        Returns:
            list: Training loss history
        """
        history = []
        prev_loss = float('inf')
        current_lr = learning_rate
        
        for it in range(max_iters):
            # Initialize gradient accumulators for all layers
            layer_grads = []
            for i in range(self.num_layers):
                layer = self.layers[i]
                grad = {
                    'Acoeff': [[0.0]*layer['basis_dim'] for _ in range(layer['model_dim'])],
                    'Bbasis': [[0.0]*layer['model_dim'] for _ in range(layer['basis_dim'])],
                    'M': [[0.0]*layer['input_dim'] for _ in range(layer['model_dim'])],
                    'Linker': [[0.0]*layer['output_len'] for _ in range(layer['input_len'])]
                }
                layer_grads.append(grad)
            
            total_loss = 0.0
            total_seqs = len(seqs)
            
            # Process all sequences
            for seq, t in zip(seqs, t_list):
                # Forward pass: store outputs and intermediates for each layer
                layer_outputs = []
                layer_intermediates = []  # Store (X, T1, T2, intermediate_seq, z_list) for backprop
                current_seq = seq
                
                # Propagate through all layers
                for i in range(self.num_layers):
                    layer = self.layers[i]
                    
                    # Convert input to matrix (d_in x L_in)
                    X = self._vec_to_mat(current_seq)
                    
                    # Apply Linker: T1 = X * Linker (d_in x L_out)
                    T1 = self._mat_mul(X, layer['Linker'])
                    
                    # Apply M: T2 = M * T1 (d_out x L_out)
                    T2 = self._mat_mul(layer['M'], T1)
                    
                    # Convert to vector sequence
                    intermediate_seq = self._mat_to_vec(T2)
                    
                    # Apply dual descriptor transformation and store intermediate z values
                    output_seq = []
                    z_list = []  # Store intermediate z values for backprop
                    for v in intermediate_seq:
                        z = self._mat_vec(layer['Bbasis'], v)
                        z_list.append(z)
                        Nk = self._mat_vec(layer['Acoeff'], z)
                        output_seq.append(Nk)
                    
                    # Store for backprop
                    layer_intermediates.append((X, T1, T2, intermediate_seq, z_list))
                    layer_outputs.append(output_seq)
                    current_seq = output_seq
                
                # Final prediction (average of last layer outputs)
                last_layer_out = layer_outputs[-1]
                t_pred = [0.0] * self.model_dims[-1]
                for vec in last_layer_out:
                    for i in range(self.model_dims[-1]):
                        t_pred[i] += vec[i]
                n_pos = len(last_layer_out)
                t_pred = [x / n_pos for x in t_pred]
                
                # Compute error and loss
                error = [t_pred_i - t_i for t_pred_i, t_i in zip(t_pred, t)]
                total_loss += sum(e*e for e in error)
                
                # Backpropagation starts from last layer
                # Gradient for last layer output (dL/dNk)
                grad_output = [[2*e/n_pos for e in error] for _ in last_layer_out]
                
                # Backpropagate through layers in reverse order
                for i in range(self.num_layers-1, -1, -1):
                    layer = self.layers[i]
                    grad_layer = layer_grads[i]
                    X, T1, T2, intermediate_seq, z_list = layer_intermediates[i]
                    
                    # Initialize gradient accumulators for this sequence
                    grad_Acoeff = [[0.0]*layer['basis_dim'] for _ in range(layer['model_dim'])]
                    grad_Bbasis = [[0.0]*layer['model_dim'] for _ in range(layer['basis_dim'])]
                    grad_M = [[0.0]*layer['input_dim'] for _ in range(layer['model_dim'])]
                    grad_Linker = [[0.0]*layer['output_len'] for _ in range(layer['input_len'])]
                    grad_input = [[0.0]*len(seq[0]) for _ in range(len(seq))] if i > 0 else None
                    
                    # Compute dL/dY (gradient before dual descriptor transform)
                    grad_Y = []
                    for pos, (v, grad_Nk) in enumerate(zip(intermediate_seq, grad_output)):
                        # Backprop through dual descriptor
                        Acoeff_T_grad = self._mat_vec(
                            self._transpose(layer['Acoeff']), grad_Nk
                        )
                        grad_z = Acoeff_T_grad
                        grad_v = self._mat_vec(layer['B_t'], grad_z)
                        grad_Y.append(grad_v)
                        
                        # Compute gradients for Acoeff and Bbasis
                        for mi in range(layer['model_dim']):
                            for bj in range(layer['basis_dim']):
                                grad_Acoeff[mi][bj] += grad_Nk[mi] * z_list[pos][bj]
                        
                        for bj in range(layer['basis_dim']):
                            for mi in range(layer['model_dim']):
                                grad_Bbasis[bj][mi] += grad_z[bj] * v[mi]
                    
                    # Convert grad_Y to matrix (d_out x L_out)
                    grad_Y_mat = self._vec_to_mat(grad_Y)
                    
                    # Backprop through matrix transformations
                    # dL/dM = grad_Y_mat * T1^T
                    T1_T = self._transpose(T1)
                    dL_dM = self._mat_mul(grad_Y_mat, T1_T)
                    
                    # dL/dLinker = X^T * (M^T * grad_Y_mat)
                    M_T = self._transpose(layer['M'])
                    M_T_grad_Y = self._mat_mul(M_T, grad_Y_mat)
                    dL_dLinker = self._mat_mul(self._transpose(X), M_T_grad_Y)  # X^T * (M^T * grad_Y_mat)
                    
                    # dL/dT1 = M^T * grad_Y_mat
                    dL_dT1 = M_T_grad_Y
                    
                    # dL/dX = dL_dT1 * Linker^T
                    dL_dX = self._mat_mul(dL_dT1, layer['Linker_t'])
                    
                    # Accumulate gradients
                    for mi in range(layer['model_dim']):
                        for ij in range(layer['input_dim']):
                            grad_M[mi][ij] += dL_dM[mi][ij]
                            
                    # Only accumulate Linker gradients if trainable
                    if layer['linker_trainable']:
                        for ii in range(layer['input_len']):
                            for jj in range(layer['output_len']):
                                grad_Linker[ii][jj] += dL_dLinker[ii][jj]
                    
                    # Propagate gradient to previous layer
                    if i > 0:
                        grad_output = self._mat_to_vec(dL_dX)
                    
                    # Update layer gradients
                    for mi in range(layer['model_dim']):
                        for bj in range(layer['basis_dim']):
                            grad_layer['Acoeff'][mi][bj] += grad_Acoeff[mi][bj]
                            
                    for bj in range(layer['basis_dim']):
                        for mi in range(layer['model_dim']):
                            grad_layer['Bbasis'][bj][mi] += grad_Bbasis[bj][mi]
                            
                    for mi in range(layer['model_dim']):
                        for ij in range(layer['input_dim']):
                            grad_layer['M'][mi][ij] += grad_M[mi][ij]
                    
                    # Only accumulate Linker gradients if trainable
                    if layer['linker_trainable']:
                        for ii in range(layer['input_len']):
                            for jj in range(layer['output_len']):
                                grad_layer['Linker'][ii][jj] += grad_Linker[ii][jj]
            
            # Average gradients
            if total_seqs > 0:
                for layer_grad in layer_grads:
                    for matrix in layer_grad.values():
                        for row in matrix:
                            for j in range(len(row)):
                                row[j] /= total_seqs
            
            # Update parameters
            for i in range(self.num_layers):
                layer = self.layers[i]
                grad = layer_grads[i]
                
                # Update Acoeff
                for mi in range(layer['model_dim']):
                    for bj in range(layer['basis_dim']):
                        layer['Acoeff'][mi][bj] -= current_lr * grad['Acoeff'][mi][bj]
                
                # Update Bbasis
                for bj in range(layer['basis_dim']):
                    for mi in range(layer['model_dim']):
                        layer['Bbasis'][bj][mi] -= current_lr * grad['Bbasis'][bj][mi]
                
                # Update M
                for mi in range(layer['model_dim']):
                    for ij in range(layer['input_dim']):
                        layer['M'][mi][ij] -= current_lr * grad['M'][mi][ij]
                
                # Update Linker only if trainable
                if layer['linker_trainable']:
                    for ii in range(layer['input_len']):
                        for jj in range(layer['output_len']):
                            layer['Linker'][ii][jj] -= current_lr * grad['Linker'][ii][jj]
                
                # Update transpose caches
                layer['B_t'] = self._transpose(layer['Bbasis'])
                layer['Linker_t'] = self._transpose(layer['Linker'])
            
            # Calculate and record loss
            avg_loss = total_loss / total_seqs if total_seqs else 0
            history.append(avg_loss)
            
            # Print progress
            if it % print_every == 0 or it == max_iters - 1:
                print(f"Iter {it:3d}: Loss = {avg_loss:.6f}, LR = {current_lr:.6f}")
            
            # Apply learning rate decay
            current_lr *= lr_decay
            
            # Check convergence
            if abs(prev_loss - avg_loss) < tol:
                print(f"Converged after {it} iterations")
                break
            prev_loss = avg_loss
        
        self.trained = True
        return history
    
    # ---- utilities ----
    def count_parameters(self):
        """Count total learnable parameters"""
        total = 0
        trainable = 0
        for i, layer in enumerate(self.layers):
            m = layer['model_dim']
            L_basis = layer['basis_dim']
            n_in = layer['input_dim']
            L_in = layer['input_len']
            L_out = layer['output_len']
            
            layer_params = m*L_basis + L_basis*m + m*n_in
            linker_params = L_in*L_out
            
            total += layer_params + linker_params
            trainable += layer_params + (linker_params if layer['linker_trainable'] else 0)
            
            print(f"Layer {i} ({n_in}×{L_in} → {m}×{L_out}):")
            print(f"  Acoeff ({m}×{L_basis}): {m*L_basis:,} params")
            print(f"  Bbasis ({L_basis}×{m}): {L_basis*m:,} params")
            print(f"  M ({m}×{n_in}): {m*n_in:,} params")
            print(f"  Linker ({L_in}×{L_out}): {linker_params:,} params {'(trainable)' if layer['linker_trainable'] else '(fixed)'}")
        
        print(f"Total parameters: {total:,}")
        print(f"Trainable parameters: {trainable:,}")
        return total, trainable
    
    def save(self, filename):
        """Save model to file"""
        state = {
            'n': self.n,
            'L0': self.L0,
            'model_dims': self.model_dims,
            'basis_dims': self.basis_dims,
            'linker_dims': self.linker_dims,
            'linker_trainable': self.linker_trainable,
            'layers': self.layers
        }
        with open(filename, 'wb') as f:
            pickle.dump(state, f)
        print(f"Model saved to {filename}")
    
    @classmethod
    def load(cls, filename):
        """Load model from file"""
        with open(filename, 'rb') as f:
            state = pickle.load(f)
        
        # Create instance
        obj = cls(
            input_dim=state['n'],
            input_seq_len=state['L0'],
            model_dims=state['model_dims'],
            basis_dims=state['basis_dims'],
            linker_dims=state['linker_dims'],
            linker_trainable=state['linker_trainable']
        )
        obj.layers = state['layers']
        print(f"Model loaded from {filename}")
        return obj


# === Example Usage ===
if __name__ == "__main__":
    random.seed(3)
    
    # Configuration
    n = 20           # Input vector dimension
    L0 = 100         # Input sequence length
    m = 10           # Target vector dimension
    n_seqs = 10      # Number of training sequences
    
    # Hierarchical model configuration
    model_dims = [25, m]     # Model dimensions for each layer
    basis_dims = [50, 20]    # Basis dimensions for each layer
    linker_dims = [80, 50]   # Output sequence lengths for each layer
    
    # Generate training data
    seqs, t_list = [], []
    for _ in range(n_seqs):
        # All sequences have fixed length L0
        seq = [[random.uniform(-0.1,0.1) for _ in range(n)] for __ in range(L0)]        
        seqs.append(seq)
        t_list.append([random.uniform(-0.1,0.1) for _ in range(m)])
    
    # Test case 1: All Linkers trainable
    print("\nTest Case 1: All Linkers Trainable")
    lhdd_all_trainable = LinkedHierarchicalDD(
        input_dim=n,
        input_seq_len=L0,
        model_dims=model_dims,
        basis_dims=basis_dims,
        linker_dims=linker_dims,
        linker_trainable=True  # All layers trainable
    )
    
    # Show parameter count
    print("\nModel parameter count:")
    lhdd_all_trainable.count_parameters()
    
    # Train model
    print("\nTraining linked hierarchical model (all linkers trainable)...")
    history = lhdd_all_trainable.grad_train(
        seqs, 
        t_list, 
        learning_rate=0.015,
        max_iters=500,
        print_every=5,
        lr_decay=1.0
    )
    
    # Evaluate predictions
    print("\nPrediction evaluation (all linkers trainable):")
    preds = [lhdd_all_trainable.predict_t(seq) for seq in seqs]
    
    # Calculate dimension-wise correlations
    correlations = []
    for i in range(m):
        actual = [t[i] for t in t_list]
        predicted = [p[i] for p in preds]
        corr = correlation(actual, predicted)
        correlations.append(corr)
        
        # Print first 5 dimensions
        if i < 5:
            print(f"Dim {i} correlation: {corr:.4f}")
    
    print(f"Average correlation: {mean(correlations):.4f}")
    
    # Test case 2: Mixed trainability (first layer fixed, second layer trainable)
    print("\n\nTest Case 2: Mixed Linker Trainability")
    lhdd_mixed = LinkedHierarchicalDD(
        input_dim=n,
        input_seq_len=L0,
        model_dims=model_dims,
        basis_dims=basis_dims,
        linker_dims=linker_dims,
        linker_trainable=[False, True]  # Layer0: fixed, Layer1: trainable
    )
    
    # Show parameter count
    print("\nModel parameter count (mixed trainability):")
    lhdd_mixed.count_parameters()
    
    # Train model
    print("\nTraining linked hierarchical model (mixed trainability)...")
    history = lhdd_mixed.grad_train(
        seqs, 
        t_list, 
        learning_rate=0.015,
        max_iters=500,
        print_every=5,
        lr_decay=1.0
    )
    
    # Evaluate predictions
    print("\nPrediction evaluation (mixed trainability):")
    preds = [lhdd_mixed.predict_t(seq) for seq in seqs]
    
    # Calculate dimension-wise correlations
    correlations = []
    for i in range(m):
        actual = [t[i] for t in t_list]
        predicted = [p[i] for p in preds]
        corr = correlation(actual, predicted)
        correlations.append(corr)
        
        # Print first 5 dimensions
        if i < 5:
            print(f"Dim {i} correlation: {corr:.4f}")
    
    print(f"Average correlation: {mean(correlations):.4f}")
    
    # Test case 3: All Linkers fixed
    print("\n\nTest Case 3: All Linkers Fixed")
    lhdd_fixed = LinkedHierarchicalDD(
        input_dim=n,
        input_seq_len=L0,
        model_dims=model_dims,
        basis_dims=basis_dims,
        linker_dims=linker_dims,
        linker_trainable=False  # All layers fixed
    )
    
    # Show parameter count
    print("\nModel parameter count (all linkers fixed):")
    lhdd_fixed.count_parameters()
    
    # Train model
    print("\nTraining linked hierarchical model (all linkers fixed)...")
    history = lhdd_fixed.grad_train(
        seqs, 
        t_list, 
        learning_rate=0.015,
        max_iters=500,
        print_every=5,
        lr_decay=1.0
    )
    
    # Evaluate predictions
    print("\nPrediction evaluation (all linkers fixed):")
    preds = [lhdd_fixed.predict_t(seq) for seq in seqs]
    
    # Calculate dimension-wise correlations
    correlations = []
    for i in range(m):
        actual = [t[i] for t in t_list]
        predicted = [p[i] for p in preds]
        corr = correlation(actual, predicted)
        correlations.append(corr)
        
        # Print first 5 dimensions
        if i < 5:
            print(f"Dim {i} correlation: {corr:.4f}")
    
    print(f"Average correlation: {mean(correlations):.4f}")
