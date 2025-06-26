# Copyright (C) 2005-2025, Bin-Guang Ma (mbg@mail.hzau.edu.cn). All rights reserved.
# Linked Hierarchical Dual Descriptor class with inter-layer Linker matrices
# Author: Bin-Guang Ma; Date: 2025-6-22
# Modified for NumPy acceleration

import math
import random
import numpy as np
import pickle

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
        
        # Initialize layers with NumPy arrays
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
                'linker_trainable': self.linker_trainable[i],
                'M': self._init_matrix(model_dims[i], layer_input_dim),           # Left multiplication matrix
                'Acoeff': self._init_matrix(model_dims[i], basis_dims[i]),        # Coefficient matrix
                'Bbasis': self._init_matrix(basis_dims[i], model_dims[i]),        # Basis matrix
                'Linker': self._init_matrix(layer_input_len, layer_output_len)    # Right multiplication matrix
            }
            self.layers.append(layer)
    
    def _init_matrix(self, rows, cols):
        """Initialize matrix with uniform random values using NumPy"""
        return np.random.uniform(-0.5, 0.5, size=(rows, cols))
    
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
        
        # Convert input to NumPy array (L_in x d_in)
        X = np.array(input_seq).T  # Transpose to (d_in x L_in)
        
        # Step 2: Apply Linker matrix (right multiplication): T1 = X @ Linker (d_in x L_out)
        T1 = X @ layer['Linker']
        
        # Step 3: Apply M matrix (left multiplication): T2 = M @ T1 (d_out x L_out)
        T2 = layer['M'] @ T1
        
        # Step 4: Convert to vector sequence (L_out x d_out)
        intermediate_seq = T2.T
        
        # Step 5: Apply dual descriptor transformation
        # Vectorized computation: z = intermediate_seq @ Bbasis.T, then output = z @ Acoeff.T
        z = intermediate_seq @ layer['Bbasis'].T
        output_seq = z @ layer['Acoeff'].T
        
        return output_seq.tolist()
    
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
        
        # Convert to NumPy array for efficient averaging
        arr = np.array(current_seq)
        return np.mean(arr, axis=0).tolist()
    
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
        
        # Convert targets to NumPy array
        t_array = np.array(t_list)
        
        for it in range(max_iters):
            # Initialize gradient accumulators for all layers
            layer_grads = []
            for i in range(self.num_layers):
                layer = self.layers[i]
                grad = {
                    'Acoeff': np.zeros((layer['model_dim'], layer['basis_dim'])),
                    'Bbasis': np.zeros((layer['basis_dim'], layer['model_dim'])),
                    'M': np.zeros((layer['model_dim'], layer['input_dim'])),
                    'Linker': np.zeros((layer['input_len'], layer['output_len']))
                }
                layer_grads.append(grad)
            
            total_loss = 0.0
            total_seqs = len(seqs)
            
            # Process all sequences
            for seq_idx, (seq, t) in enumerate(zip(seqs, t_array)):
                # Forward pass: store outputs and intermediates for each layer
                layer_outputs = []
                layer_intermediates = []  # Store (X, T1, T2, intermediate_seq, z) for backprop
                current_seq = seq
                
                # Propagate through all layers
                for i in range(self.num_layers):
                    layer = self.layers[i]
                    
                    # Convert input to NumPy array (L_in x d_in) and transpose to (d_in x L_in)
                    X = np.array(current_seq).T
                    
                    # Apply Linker: T1 = X @ Linker (d_in x L_out)
                    T1 = X @ layer['Linker']
                    
                    # Apply M: T2 = M @ T1 (d_out x L_out)
                    T2 = layer['M'] @ T1
                    
                    # Transpose to (L_out x d_out)
                    intermediate_seq = T2.T
                    
                    # Apply dual descriptor transformation
                    z = intermediate_seq @ layer['Bbasis'].T
                    output_seq = z @ layer['Acoeff'].T
                    
                    # Store for backprop
                    layer_intermediates.append((X, T1, T2, intermediate_seq, z))
                    layer_outputs.append(output_seq.tolist())
                    current_seq = output_seq.tolist()
                
                # Final prediction (average of last layer outputs)
                last_layer_out = np.array(layer_outputs[-1])
                t_pred = np.mean(last_layer_out, axis=0)
                
                # Compute error and loss
                error = t_pred - t
                total_loss += np.sum(error**2)
                
                # Backpropagation starts from last layer
                # Gradient for last layer output (dL/dNk)
                grad_output = np.tile(2 * error / len(last_layer_out), (len(last_layer_out), 1))
                
                # Backpropagate through layers in reverse order
                for i in range(self.num_layers-1, -1, -1):
                    layer = self.layers[i]
                    grad_layer = layer_grads[i]
                    X, T1, T2, intermediate_seq, z = layer_intermediates[i]
                    
                    # Compute dL/dY (gradient before dual descriptor transform)
                    # dL/dz = grad_output @ Acoeff
                    grad_z = grad_output @ layer['Acoeff']
                    
                    # dL/dv = grad_z @ Bbasis
                    grad_Y = grad_z @ layer['Bbasis']
                    
                    # Convert to matrix form (d_out x L_out)
                    grad_Y_mat = grad_Y.T
                    
                    # Compute gradients for Acoeff and Bbasis
                    grad_Acoeff = grad_output.T @ z / total_seqs
                    grad_Bbasis = grad_z.T @ intermediate_seq / total_seqs
                    
                    # Backprop through matrix transformations
                    # dL/dM = grad_Y_mat @ T1.T
                    grad_M = grad_Y_mat @ T1.T
                    
                    # dL/dLinker = X.T @ (M.T @ grad_Y_mat)
                    grad_Linker = X.T @ (layer['M'].T @ grad_Y_mat)
                    
                    # Propagate gradient to previous layer: dL/dX = (M.T @ grad_Y_mat) @ Linker.T
                    if i > 0:
                        dL_dX = (layer['M'].T @ grad_Y_mat) @ layer['Linker'].T
                        grad_output = dL_dX.T  # Convert to (L_in x d_in) for next layer
                    
                    # Accumulate gradients
                    grad_layer['Acoeff'] += grad_Acoeff
                    grad_layer['Bbasis'] += grad_Bbasis
                    grad_layer['M'] += grad_M
                    if layer['linker_trainable']:
                        grad_layer['Linker'] += grad_Linker
            
            # Update parameters
            for i in range(self.num_layers):
                layer = self.layers[i]
                grad = layer_grads[i]
                
                layer['Acoeff'] -= current_lr * grad['Acoeff']
                layer['Bbasis'] -= current_lr * grad['Bbasis']
                layer['M'] -= current_lr * grad['M']
                if layer['linker_trainable']:
                    layer['Linker'] -= current_lr * grad['Linker']
            
            # Calculate and record loss
            avg_loss = total_loss / total_seqs
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

    from statistics import correlation, mean
    
    np.random.seed(3)
    random.seed(3)
    
    # Configuration
    n = 20           # Input vector dimension
    L0 = 100         # Input sequence length
    m = 10           # Target vector dimension
    n_seqs = 30      # Number of training sequences
    
    # Hierarchical model configuration
    model_dims = [25, m]     # Model dimensions for each layer
    basis_dims = [50, 20]    # Basis dimensions for each layer
    linker_dims = [80, 50]   # Output sequence lengths for each layer
    
    # Generate training data
    seqs, t_list = [], []
    for _ in range(n_seqs):
        # All sequences have fixed length L0
        seq = [list(np.random.uniform(-0.1, 0.1, size=n)) for __ in range(L0)]        
        seqs.append(seq)
        t_list.append(list(np.random.uniform(-0.1, 0.1, size=m)))
    
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
        learning_rate=0.02,
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
        learning_rate=0.02,
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
        learning_rate=0.02,
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
