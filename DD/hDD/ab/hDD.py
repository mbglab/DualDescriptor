# Copyright (C) 2005-2025, Bin-Guang Ma (mbg@mail.hzau.edu.cn). All rights reserved.
# Hierarchical Dual Descriptor class
# Author: Bin-Guang Ma; Date: 2025-6-22

import math
import random
import pickle
from statistics import correlation, mean

class HierarchicalDD:
    """
    Hierarchical Dual Descriptor with multiple layers.
    Each layer maintains its own model parameters (Acoeff, Bbasis, M).
    
    Architecture:
      Layer 0: Input dim = n, Output dim = model_dims[0]
      Layer 1: Input dim = model_dims[0], Output dim = model_dims[1]
      ...
      Last layer: Output dim = model_dims[-1] (must match target vector dim)
    """
    def __init__(self, input_dim=10, model_dims=[10], basis_dims=[50]):
        """
        Initialize hierarchical model.
        
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
        
        # Initialize layers
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
            layer['B_t'] = self._transpose(layer['Bbasis'])
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
    
    # ---- forward propagation ----
    def forward_layer(self, layer_idx, input_seq):
        """
        Forward pass for a single layer.
        
        Args:
            layer_idx: Index of layer to process
            input_seq: List of input vectors
            
        Returns:
            List of output descriptor vectors
        """
        layer = self.layers[layer_idx]
        output_seq = []
        
        for v in input_seq:
            # Transform input: x = M * v
            x = self._mat_vec(layer['M'], v)
            # Compute z = Bbasis * x
            z = self._mat_vec(layer['Bbasis'], x)
            # Compute Nk = Acoeff * z
            Nk = self._mat_vec(layer['Acoeff'], z)
            output_seq.append(Nk)
            
        return output_seq
    
    def predict_t(self, seq):
        """
        Predict target vector by propagating through all layers.
        
        Args:
            seq: Input sequence of n-dimensional vectors
            
        Returns:
            Predicted target vector (m-dimensional)
        """
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
        Train hierarchical model using gradient descent.
        
        Args:
            seqs: List of training sequences (n-dim)
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
                    'M': [[0.0]*layer['input_dim'] for _ in range(layer['model_dim'])]
                }
                layer_grads.append(grad)
            
            total_loss = 0.0
            total_positions = 0
            
            # Process all sequences
            for seq, t in zip(seqs, t_list):
                # Forward pass: store outputs for each layer
                layer_outputs = []
                current_seq = seq
                
                # Propagate through all layers
                for i in range(self.num_layers):
                    current_seq = self.forward_layer(i, current_seq)
                    layer_outputs.append(current_seq)
                
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
                total_positions += 1  # Count sequences for averaging
                
                # Backpropagation starts from last layer
                # Gradient for last layer output
                grad_output = [[2*e/n_pos for e in error] for _ in last_layer_out]
                
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
                    
                    # Process each position in sequence
                    for pos, (v, out) in enumerate(zip(input_seq, output_seq)):
                        # Compute intermediate values
                        x = self._mat_vec(layer['M'], v)
                        z = self._mat_vec(layer['Bbasis'], x)
                        
                        # Get current gradient
                        grad_Nk = grad_output[pos]
                        
                        # Compute gradients for parameters
                        # dL/dAcoeff = grad_Nk * z^T
                        for mi in range(layer['model_dim']):
                            for bj in range(layer['basis_dim']):
                                grad_layer['Acoeff'][mi][bj] += 2 * grad_Nk[mi] * z[bj]
                        
                        # dL/dBbasis = (Acoeff^T @ grad_Nk) * x^T
                        Acoeff_T_grad = self._mat_vec(
                            self._transpose(layer['Acoeff']), grad_Nk
                        )
                        for bj in range(layer['basis_dim']):
                            for mi in range(layer['model_dim']):
                                grad_layer['Bbasis'][bj][mi] += 2 * Acoeff_T_grad[bj] * x[mi]
                        
                        # dL/dM = (Bbasis^T @ Acoeff^T @ grad_Nk) * v^T
                        Bt_At_grad = self._mat_vec(
                            layer['B_t'], Acoeff_T_grad
                        )
                        for mi in range(layer['model_dim']):
                            for ij in range(layer['input_dim']):
                                grad_layer['M'][mi][ij] += 2 * Bt_At_grad[mi] * v[ij]
                        
                        # Propagate gradient to previous layer
                        if i > 0:  # Not the first layer
                            # dL/dx = Bbasis^T @ (Acoeff^T @ grad_Nk)
                            grad_x = Bt_At_grad
                            # Transform to input space: dL/dv = M^T @ grad_x
                            grad_v = self._mat_vec(
                                self._transpose(layer['M']), grad_x
                            )
                            # Initialize gradient for previous layer if needed                            
                            grad_output_prev = [[0.0]*layer['input_dim'] for _ in input_seq]
                            
                            # Accumulate gradient
                            for idx in range(len(grad_v)):
                                grad_output_prev[pos][idx] += grad_v[idx]
                    
                    # Prepare gradients for next (lower) layer
                    if i > 0:
                        grad_output = grad_output_prev
            
            # Average gradients
            if total_positions > 0:
                norm = 1 / total_positions
                for layer_grad in layer_grads:
                    for matrix in layer_grad.values():
                        for row in matrix:
                            for j in range(len(row)):
                                row[j] *= norm
            
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
                
                # Update transpose cache
                layer['B_t'] = self._transpose(layer['Bbasis'])
            
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
    random.seed(3)
    
    # Configuration
    n = 20       # Input vector dimension
    m = 10       # Target vector dimension
    n_seqs = 10  # Number of training sequences
    
    # Hierarchical model configuration
    model_dims = [25, m]  # Dimensions for each layer
    basis_dims = [50, 20]   # Basis dimensions for each layer
    
    # Generate training data
    seqs, t_list = [], []
    for _ in range(n_seqs):
        length = random.randint(100, 200)
        seq = [[random.uniform(-1,1) for _ in range(n)] for __ in range(length)]        
        seqs.append(seq)
        t_list.append([random.uniform(-1,1) for _ in range(m)])
    
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
        learning_rate=0.3,
        max_iters=20,
        print_every=5,
        lr_decay=0.98
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
    test_seq = [[random.uniform(-1,1) for _ in range(n)] for _ in range(150)]
    test_t = [random.uniform(-1,1) for _ in range(m)]
    
    pred_t = hdd.predict_t(test_seq)
    mse = sum((p - t)**2 for p, t in zip(pred_t, test_t)) / m
    print(f"Test MSE: {mse:.6f}")
