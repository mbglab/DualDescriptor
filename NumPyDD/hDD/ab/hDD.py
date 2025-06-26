# Copyright (C) 2005-2025, Bin-Guang Ma (mbg@mail.hzau.edu.cn). All rights reserved.
# Hierarchical Dual Descriptor class with NumPy acceleration
# Author: Bin-Guang Ma; Date: 2025-6-22

import numpy as np
import pickle

class HierarchicalDD:
    """
    Hierarchical Dual Descriptor with multiple layers using NumPy acceleration.
    Each layer maintains its own model parameters (Acoeff, Bbasis, M).
    
    Architecture:
      Layer 0: Input dim = n, Output dim = model_dims[0]
      Layer 1: Input dim = model_dims[0], Output dim = model_dims[1]
      ...
      Last layer: Output dim = model_dims[-1] (must match target vector dim)
    """
    def __init__(self, input_dim=10, model_dims=[10], basis_dims=[50]):
        """
        Initialize hierarchical model with NumPy arrays.
        
        Args:
            input_dim (int): Dimensionality of input vectors (n)
            model_dims (list): List of model dimensions for each layer
            basis_dims (list): List of basis dimensions for each layer
        """
        self.n = input_dim
        self.model_dims = model_dims
        self.basis_dims = basis_dims
        self.num_layers = len(model_dims)
        self.trained = False
        
        # Validate dimensions
        if len(basis_dims) != self.num_layers:
            raise ValueError("model_dims and basis_dims must have same length")
        
        # Initialize layers with NumPy arrays
        self.layers = []
        for i in range(self.num_layers):
            # Determine input dim for this layer
            if i == 0:
                layer_input_dim = input_dim
            else:
                layer_input_dim = model_dims[i-1]
                
            layer = {
                'input_dim': layer_input_dim,
                'model_dim': model_dims[i],
                'basis_dim': basis_dims[i],
                'M': self._init_matrix(model_dims[i], layer_input_dim),
                'Acoeff': self._init_matrix(model_dims[i], basis_dims[i]),
                'Bbasis': self._init_matrix(basis_dims[i], model_dims[i]),
            }
            # Precompute transpose for efficiency
            layer['B_t'] = layer['Bbasis'].T
            self.layers.append(layer)
    
    def _init_matrix(self, rows, cols):
        """Initialize matrix with uniform random values using NumPy"""
        return np.random.uniform(-0.5, 0.5, size=(rows, cols))
    
    # ---- forward propagation ----
    def forward_layer(self, layer_idx, input_seq):
        """
        Forward pass for a single layer using NumPy operations.
        
        Args:
            layer_idx: Index of layer to process
            input_seq: List of input vectors (each as NumPy array)
            
        Returns:
            List of output descriptor vectors (each as NumPy array)
        """
        layer = self.layers[layer_idx]
        output_seq = []
        
        for v in input_seq:
            # Transform input: x = M @ v
            x = layer['M'] @ v
            # Compute z = Bbasis @ x
            z = layer['Bbasis'] @ x
            # Compute Nk = Acoeff @ z
            Nk = layer['Acoeff'] @ z
            output_seq.append(Nk)
            
        return output_seq
    
    def predict_t(self, seq):
        """
        Predict target vector by propagating through all layers.
        
        Args:
            seq: Input sequence of n-dimensional vectors (each as NumPy array)
            
        Returns:
            Predicted target vector (m-dimensional NumPy array)
        """
        # Propagate through all layers
        current_seq = seq
        for i in range(self.num_layers):
            current_seq = self.forward_layer(i, current_seq)
        
        # Average output of last layer using NumPy mean
        if not current_seq:
            return np.zeros(self.model_dims[-1])
        
        # Stack all output vectors into a matrix and compute mean along axis 0
        return np.mean(np.vstack(current_seq), axis=0)
    
    # ---- training ----
    def grad_train(self, seqs, t_list, learning_rate=0.01, max_iters=100, 
                  tol=1e-9, print_every=10, lr_decay=1.0):
        """
        Train hierarchical model using gradient descent with NumPy acceleration.
        
        Args:
            seqs: List of training sequences (n-dim NumPy arrays)
            t_list: List of target vectors (m-dim NumPy arrays)
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
        
        # Convert targets to NumPy arrays if needed
        t_list = [np.array(t) for t in t_list]
        
        for it in range(max_iters):
            # Initialize gradient accumulators for all layers
            layer_grads = []
            for i in range(self.num_layers):
                layer = self.layers[i]
                grad = {
                    'Acoeff': np.zeros((layer['model_dim'], layer['basis_dim'])),
                    'Bbasis': np.zeros((layer['basis_dim'], layer['model_dim'])),
                    'M': np.zeros((layer['model_dim'], layer['input_dim']))
                }
                layer_grads.append(grad)
            
            total_loss = 0.0
            total_positions = 0
            
            # Process all sequences
            for seq, t in zip(seqs, t_list):
                # Convert sequence to list of NumPy arrays if needed
                if not isinstance(seq[0], np.ndarray):
                    seq = [np.array(v) for v in seq]
                    
                # Forward pass: store outputs for each layer
                layer_outputs = []
                current_seq = seq
                
                # Propagate through all layers
                for i in range(self.num_layers):
                    current_seq = self.forward_layer(i, current_seq)
                    layer_outputs.append(current_seq)
                
                # Final prediction (average of last layer outputs)
                last_layer_out = layer_outputs[-1]
                t_pred = np.mean(np.vstack(last_layer_out), axis=0)
                
                # Compute error and loss
                error = t_pred - t
                total_loss += np.sum(error**2)
                total_positions += 1  # Count sequences for averaging
                
                # Backpropagation starts from last layer
                # Gradient for last layer output
                grad_output = [2 * error / len(last_layer_out)] * len(last_layer_out)
                
                # Backpropagate through layers in reverse order
                for i in range(self.num_layers-1, -1, -1):
                    layer = self.layers[i]
                    grad_layer = layer_grads[i]
                    
                    # Get layer input
                    if i == 0:
                        input_seq = seq
                    else:
                        input_seq = layer_outputs[i-1]
                    
                    # Get layer output
                    output_seq = layer_outputs[i]
                    
                    # Prepare gradient for previous layer
                    grad_output_prev = [np.zeros(layer['input_dim']) for _ in input_seq]
                    
                    # Process each position in sequence
                    for pos, (v, out) in enumerate(zip(input_seq, output_seq)):
                        # Compute intermediate values
                        x = layer['M'] @ v
                        z = layer['Bbasis'] @ x
                        
                        # Get current gradient
                        grad_Nk = grad_output[pos]
                        
                        # Compute gradients for parameters using NumPy operations
                        # dL/dAcoeff = grad_Nk * z^T
                        grad_layer['Acoeff'] += 2 * np.outer(grad_Nk, z)
                        
                        # dL/dBbasis = (Acoeff^T @ grad_Nk) * x^T
                        Acoeff_T_grad = layer['Acoeff'].T @ grad_Nk
                        grad_layer['Bbasis'] += 2 * np.outer(Acoeff_T_grad, x)
                        
                        # dL/dM = (Bbasis^T @ Acoeff^T @ grad_Nk) * v^T
                        Bt_At_grad = layer['B_t'] @ Acoeff_T_grad
                        grad_layer['M'] += 2 * np.outer(Bt_At_grad, v)
                        
                        # Propagate gradient to previous layer
                        if i > 0:  # Not the first layer
                            # dL/dx = Bbasis^T @ (Acoeff^T @ grad_Nk)
                            grad_x = Bt_At_grad
                            # Transform to input space: dL/dv = M^T @ grad_x
                            grad_v = layer['M'].T @ grad_x
                            grad_output_prev[pos] += grad_v
                    
                    # Prepare gradients for next (lower) layer
                    if i > 0:
                        grad_output = grad_output_prev
            
            # Average gradients
            if total_positions > 0:
                for layer_grad in layer_grads:
                    for param in layer_grad:
                        layer_grad[param] /= total_positions
            
            # Update parameters
            for i in range(self.num_layers):
                layer = self.layers[i]
                grad = layer_grads[i]
                
                # Update parameters using NumPy operations
                layer['Acoeff'] -= current_lr * grad['Acoeff']
                layer['Bbasis'] -= current_lr * grad['Bbasis']
                layer['M'] -= current_lr * grad['M']
                
                # Update transpose cache
                layer['B_t'] = layer['Bbasis'].T
            
            # Calculate and record loss
            avg_loss = total_loss / total_positions if total_positions else 0
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
        for i, layer in enumerate(self.layers):
            m = layer['model_dim']
            L = layer['basis_dim']
            n_in = layer['input_dim']
            
            layer_params = m*L + L*m + m*n_in
            total += layer_params
            
            print(f"Layer {i} ({n_in}→{m}):")
            print(f"  Acoeff ({m}×{L}): {m*L:,} params")
            print(f"  Bbasis ({L}×{m}): {L*m:,} params")
            print(f"  M ({m}×{n_in}): {m*n_in:,} params")
        
        print(f"Total parameters: {total:,}")
        return total
    
    def save(self, filename):
        """Save model to file"""
        state = {
            'n': self.n,
            'model_dims': self.model_dims,
            'basis_dims': self.basis_dims,
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
            model_dims=state['model_dims'],
            basis_dims=state['basis_dims']
        )
        obj.layers = state['layers']
        print(f"Model loaded from {filename}")
        return obj


# === Example Usage ===
if __name__ == "__main__":

    from statistics import correlation, mean
    
    np.random.seed(3)
    
    # Configuration
    n = 30       # Input vector dimension
    m = 10       # Target vector dimension
    n_seqs = 30  # Number of training sequences
    
    # Hierarchical model configuration
    model_dims = [25, 15, m]  # Dimensions for each layer
    basis_dims = [50, 30, 20]   # Basis dimensions for each layer
    
    # Generate training data
    seqs, t_list = [], []
    for _ in range(n_seqs):
        length = np.random.randint(100, 200)
        seq = [np.random.uniform(-1, 1, size=n) for _ in range(length)]        
        seqs.append(seq)
        t_list.append(np.random.uniform(-1, 1, size=m))
    
    # Create hierarchical model
    print("\nCreating HierarchicalDD model...")
    hdd = HierarchicalDD(
        input_dim=n,
        model_dims=model_dims,
        basis_dims=basis_dims
    )
    
    # Show parameter count
    print("\nModel parameter count:")
    hdd.count_parameters()
    
    # Train model
    print("\nTraining hierarchical model...")
    history = hdd.grad_train(
        seqs, 
        t_list, 
        learning_rate=0.05,
        max_iters=1000,
        print_every=5,
        lr_decay=1.0
    )
    
    # Evaluate predictions
    print("\nPrediction evaluation:")
    preds = [hdd.predict_t(seq) for seq in seqs]
    
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
    
    # Test with a new sequence
    print("\nTesting with new sequence...")
    test_seq = [np.random.uniform(-1, 1, size=n) for _ in range(150)]
    test_t = np.random.uniform(-1, 1, size=m)
    
    pred_t = hdd.predict_t(test_seq)
    mse = np.mean((pred_t - test_t)**2)
    print(f"Test MSE: {mse:.6f}")
