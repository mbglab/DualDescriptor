# Copyright (C) 2005-2025, Bin-Guang Ma (mbg@mail.hzau.edu.cn). All rights reserved.
# Hierarchical Dual Descriptor class
# Author: Bin-Guang Ma; Date: 2025-6-14

import math
import random
import pickle

class HierarchicalDD:
    """
    Hierarchical Dual Descriptor with multiple layers.
    Each layer has its own Acoeff, Bbasis, and M matrices.
    
    Architecture:
      Layer 0: Input dim = n, Output dim = model_dims[0]
      Layer 1: Input dim = model_dims[0], Output dim = model_dims[1]
      ...
      Layer k: Input dim = model_dims[k-1], Output dim = model_dims[k]
    """
    def __init__(self, num_layers, model_dims, input_dim):
        """
        Initialize hierarchical model.
        
        Args:
            num_layers (int): Number of layers in the hierarchy
            model_dims (list): List of output dimensions for each layer
            input_dim (int): Dimensionality of input vectors
        """
        self.num_layers = num_layers
        self.model_dims = model_dims
        self.input_dim = input_dim
        self.trained = False
        
        # Initialize layers
        self.layers = []
        for i in range(num_layers):
            layer = {}
            # Determine input dimension for this layer
            in_dim = input_dim if i == 0 else model_dims[i-1]
            out_dim = model_dims[i]
            
            layer['m'] = out_dim
            layer['n'] = in_dim
            layer['L'] = None  # Will be set during training
            
            # Initialize M matrix (out_dim x in_dim)
            layer['M'] = [[random.uniform(-0.5, 0.5) for _ in range(in_dim)] 
                          for _ in range(out_dim)]
            
            # Placeholders for other parameters
            layer['Acoeff'] = None
            layer['Bbasis'] = None
            layer['B_t'] = None  # Transpose of Bbasis
            
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
    
    # ---- initialization ----
    def initialize(self, seqs):
        """Initialize model parameters based on training sequences"""
        L_max = max(len(seq) for seq in seqs)  # Maximum sequence length
        
        for layer in self.layers:
            layer['L'] = L_max
            m = layer['m']
            n = layer['n']
            
            # Initialize Acoeff (m×L)
            layer['Acoeff'] = [[random.uniform(-0.1, 0.1) for _ in range(L_max)]
                              for _ in range(m)]
            
            # Initialize Bbasis (L×m)
            layer['Bbasis'] = [[random.uniform(-0.1, 0.1) for _ in range(m)]
                              for _ in range(L_max)]
            
            # Cache transpose
            layer['B_t'] = self._transpose(layer['Bbasis'])
    
    # ---- forward pass for a single layer ----
    def _forward_layer(self, layer, input_seq):
        """
        Forward pass for a single layer.
        
        Args:
            layer: Layer dictionary with parameters
            input_seq: Input sequence (list of vectors)
            
        Returns:
            output_seq: Output sequence (list of vectors)
            cache: Intermediate values for backpropagation
        """
        L_max = layer['L']
        T = len(input_seq)  # Actual sequence length
        output_seq = []
        cache = []  # Store (h_t, x_t, z_t, o_t) for each position
        
        for t in range(min(T, L_max)):
            h_t = input_seq[t]  # Input vector
            
            # Transform input: x_t = M * h_t
            x_t = self._mat_vec(layer['M'], h_t)
            
            # Compute basis: z_t = Bbasis * x_t
            z_t = self._mat_vec(layer['Bbasis'], x_t)
            
            # Compute output: o_t = Acoeff * z_t
            o_t = self._mat_vec(layer['Acoeff'], z_t)
            
            output_seq.append(o_t)
            cache.append((h_t, x_t, z_t, o_t))
            
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
        Train hierarchical model using gradient descent.
        
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
                L = layer['L']
                n = layer['n']
                
                grad_A = [[0.0]*L for _ in range(m)]
                grad_B = [[0.0]*m for _ in range(L)]
                grad_M = [[0.0]*n for _ in range(m)]
                
                grads.append({
                    'grad_A': grad_A,
                    'grad_B': grad_B,
                    'grad_M': grad_M
                })
            
            total_loss = 0.0
            total_positions = 0
            
            # Process all sequences
            for seq, t in zip(seqs, t_list):
                # Forward pass (store all intermediate values)
                output_seq, all_caches = self.forward(seq)
                T = len(output_seq)  # Actual sequence length
                total_positions += T
                
                # Calculate error for each position
                position_errors = []
                for pos in range(T):
                    error_vec = [output_seq[pos][i] - t[i] 
                                for i in range(self.model_dims[-1])]
                    position_errors.append(error_vec)
                    total_loss += sum(e*e for e in error_vec)
                
                # Backpropagation through layers (last to first)
                layer_errors = position_errors  # Start with output errors
                for layer_idx in range(self.num_layers-1, -1, -1):
                    layer = self.layers[layer_idx]
                    cache = all_caches[layer_idx]
                    grad_layer = grads[layer_idx]
                    m = layer['m']
                    L = layer['L']
                    n = layer['n']
                    
                    # Initialize errors for previous layer
                    prev_layer_errors = []
                    
                    # Process each position
                    for pos in range(len(cache)):
                        h_t, x_t, z_t, o_t = cache[pos]
                        error_t = layer_errors[pos]
                        
                        # Gradients for Acoeff (dL/dA = error_t * z_t^T)
                        for i in range(m):
                            for j in range(L):
                                grad_layer['grad_A'][i][j] += 2 * error_t[i] * z_t[j]
                        
                        # Compute intermediate: Acoeff^T * error_t
                        Acoeff_t_error = [0.0] * L
                        for j in range(L):
                            for i in range(m):
                                Acoeff_t_error[j] += layer['Acoeff'][i][j] * error_t[i]
                        
                        # Gradients for Bbasis (dL/dB = (Acoeff^T @ error_t) * x_t^T)
                        for j in range(L):
                            for i in range(m):
                                grad_layer['grad_B'][j][i] += 2 * Acoeff_t_error[j] * x_t[i]
                        
                        # Compute intermediate: Bbasis^T * (Acoeff^T @ error_t)
                        B_t_Acoeff_t_error = self._mat_vec(layer['B_t'], Acoeff_t_error)
                        
                        # Gradients for M (dL/dM = [Bbasis^T @ Acoeff^T @ error_t] * h_t^T)
                        for i in range(m):
                            for j in range(n):
                                grad_layer['grad_M'][i][j] += 2 * B_t_Acoeff_t_error[i] * h_t[j]
                        
                        # Propagate error to previous layer (dL/dh_t = M^T @ [Bbasis^T @ Acoeff^T @ error_t])
                        if layer_idx > 0:  # Only propagate if not the first layer
                            M_t = self._transpose(layer['M'])
                            prev_error = self._mat_vec(M_t, B_t_Acoeff_t_error)
                            prev_layer_errors.append(prev_error)
                    
                    # Set errors for next layer (towards input)
                    layer_errors = prev_layer_errors
            
            # Average gradients
            if total_positions > 0:
                norm = 1.0 / total_positions
                for grad_dict in grads:
                    for grad_matrix in grad_dict.values():
                        for row in grad_matrix:
                            for j in range(len(row)):
                                row[j] *= norm
            
            # Update parameters
            for layer_idx, layer in enumerate(self.layers):
                grad_layer = grads[layer_idx]
                m = layer['m']
                L = layer['L']
                n = layer['n']
                
                # Update Acoeff
                for i in range(m):
                    for j in range(L):
                        layer['Acoeff'][i][j] -= learning_rate * grad_layer['grad_A'][i][j]
                
                # Update Bbasis
                for j in range(L):
                    for i in range(m):
                        layer['Bbasis'][j][i] -= learning_rate * grad_layer['grad_B'][j][i]
                
                # Update M
                for i in range(m):
                    for j in range(n):
                        layer['M'][i][j] -= learning_rate * grad_layer['grad_M'][i][j]
                
                # Update B_t
                layer['B_t'] = self._transpose(layer['Bbasis'])
            
            # Calculate average loss
            avg_loss = total_loss / total_positions if total_positions else 0.0
            history.append(avg_loss)
            
            # Print progress
            if it % print_every == 0 or it == max_iters - 1:
                print(f"Iter {it:3d}: Loss = {avg_loss:.6f}")
            
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
        print("HierarchicalDD status:")
        print(f" Layers: {self.num_layers}, Model dims: {self.model_dims}, Input dim: {self.input_dim}")
        for i, layer in enumerate(self.layers):
            print(f" Layer {i}: m={layer['m']}, n={layer['n']}, L={layer['L']}")
            print(f"  Acoeff[0][:3]: {layer['Acoeff'][0][:3]}")
            print(f"  Bbasis[0][:3]: {layer['Bbasis'][0][:3]}")
            print(f"  M[0][:3]: {layer['M'][0][:3]}")

    def count_parameters(self):
        """
        Calculate and return the total number of trainable parameters in the model.
        
        Returns:
            int: Total number of parameters in the model
        """
        total_params = 0
        for layer in self.layers:
            m = layer['m']        # Output dimension
            n = layer['n']        # Input dimension
            L = layer['L'] if layer['L'] is not None else 0  # Handle uninitialized case
            
            # Calculate parameters per layer:
            #   M matrix: m × n parameters
            #   Acoeff matrix: m × L parameters
            #   Bbasis matrix: L × m parameters
            layer_params = m * n + m * L + L * m
            total_params += layer_params
            
            # Print layer-wise statistics
            print(f"Layer (m={m}, n={n}, L={L}): {layer_params} parameters")
        
        print(f"TOTAL PARAMETERS: {total_params}")
        return total_params
    
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
    input_dim = 30     # Input vector dimension
    model_dims = [20, 10, 1]  # Output dimensions for each layer
    n_layers = len(model_dims)
    n_seqs = 10        # Number of training sequences
    
    # Generate training data
    seqs, t_list = [], []
    for _ in range(n_seqs):
        L = random.randint(80, 100)  # Variable sequence length
        seq = [[random.uniform(-1, 1) for _ in range(input_dim)] 
              for _ in range(L)]
        seqs.append(seq)
        # Target vector (dimension = last layer's output dim)
        t_list.append([random.uniform(-1, 1) for _ in range(model_dims[-1])])
    
    # Create and train hierarchical model
    print("\nTraining Hierarchical Dual Descriptor:")
    hdd = HierarchicalDD(
        num_layers=n_layers,
        model_dims=model_dims,
        input_dim=input_dim
    )
    
    history = hdd.grad_train(
        seqs, 
        t_list,
        learning_rate=1.0,
        max_iters=300,
        print_every=20
    )
    
    # Calculate prediction correlations
    preds = [hdd.predict_t(seq) for seq in seqs]
    correlations = []
    for i in range(model_dims[-1]):
        actual = [t[i] for t in t_list]
        predicted = [p[i] for p in preds]
        corr = correlation(actual, predicted) if len(actual) > 1 else 1.0
        correlations.append(corr)
        print(f"Dim {i} correlation: {corr:.4f}")
    
    print(f"Average correlation: {mean(correlations):.4f}")
    print("Final model state:")
    hdd.show()

    print("\nModel parameters count:")
    hdd.count_parameters()

