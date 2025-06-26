# Copyright (C) Bin-Guang Ma (mbg@mail.hzau.edu.cn). All rights reserved.
# Hierarchical Dual Descriptor class for vector sequences
# Modified to support n-dimensional input and m-dimensional output through multiple layers
# Author: Bin-Guang Ma; Date: 2025-6-20
# Optimized with NumPy for performance

import math
import random
import pickle
import numpy as np

class HierarchicalDD:
    """
    Hierarchical Dual Descriptor for sequences of n-dimensional input vectors and m-dimensional outputs.
    - input_dim: dimension of input vectors
    - model_dims: list of dimensions for each layer's output
    - num_layers: number of layers in the hierarchy
    Model: 
        For each vector x_k in sequence: 
            Layer 0: N0(k) = P0 * M0 * x_k
            Layer 1: N1(k) = P1 * M1 * N0(k)
            ...
            Layer L-1: N_{L-1}(k) = P_{L-1} * M_{L-1} * N_{L-2}(k)
        S(l) = cumulative sum of N(k) from k=1 to l
    """

    def __init__(self, input_dim=4, model_dims=[4]):
        """
        Initialize the Hierarchical Dual Descriptor for vector sequences.
        
        Args:
            input_dim (int): Dimensionality of input vectors (n)
            model_dims (list): List of output dimensions for each layer
        """
        self.input_dim = input_dim  # Input vector dimension
        self.model_dims = model_dims  # Output dimensions for each layer
        self.num_layers = len(model_dims)  # Number of layers
        self.trained = False
        
        # Initialize layers using NumPy arrays
        self.layers = []
        for i, dim in enumerate(model_dims):
            # Determine input dimension for this layer
            if i == 0:
                layer_input_dim = input_dim
            else:
                layer_input_dim = model_dims[i-1]
                
            # Create a new layer with NumPy arrays
            layer = {
                'M': np.random.uniform(-0.5, 0.5, size=(dim, layer_input_dim)),
                'P': np.random.uniform(-0.1, 0.1, size=(dim, dim))
            }
            self.layers.append(layer)

    def describe(self, seq):
        """
        Compute the output of the hierarchical model for a given vector sequence.
        
        Args:
            seq (list): List of n-dimensional vectors or NumPy array
            
        Returns:
            list: Output from the final layer for each position
        """
        # Convert input to NumPy array if needed
        if not isinstance(seq, np.ndarray):
            current = np.array(seq)
        else:
            current = seq.copy()
        
        # Process through each layer
        for layer in self.layers:
            M = layer['M']
            P = layer['P']
            # Compute M @ x for each vector in the sequence
            M_x = current @ M.T  # Equivalent to M @ x for each row vector
            # Compute P @ (M_x) for each vector
            output = M_x @ P.T  # Matrix multiplication for all vectors
            current = output
            
        return current.tolist()

    def deviation(self, seqs, t_list):
        """
        Compute mean squared deviation D:
        D = average over all positions and sequences of ||Final_layer_output - t_j||^2.
        """
        total = 0.0
        count = 0
        
        # Convert targets to NumPy array
        t_array = np.array(t_list)
        
        for i, seq in enumerate(seqs):
            # Get final layer outputs as NumPy array
            outputs = np.array(self.describe(seq))
            # Get corresponding target
            t = t_array[i]
            # Compute squared errors
            errors = outputs - t
            sq_errors = np.sum(errors**2, axis=1)
            total += np.sum(sq_errors)
            count += len(seq)
            
        return total / count if count else 0.0

    def grad_train(self, seqs, t_list, max_iters=100, tol=1e-8, 
                   learning_rate=0.01, decay_rate=0.95):
        """
        Train using gradient descent with backpropagation through layers.
        Optimized with NumPy operations.
        """
        history = []
        D_prev = float('inf')
        
        # Convert targets to NumPy array
        t_array = np.array(t_list)
        
        # Initialize gradient storage for all layers
        grad_P_list = [np.zeros_like(layer['P']) for layer in self.layers]
        grad_M_list = [np.zeros_like(layer['M']) for layer in self.layers]
        
        for it in range(max_iters):
            total_vectors = 0
            
            # Reset gradients
            for grad_P in grad_P_list:
                grad_P.fill(0.0)
            for grad_M in grad_M_list:
                grad_M.fill(0.0)
            
            # Process each sequence
            for seq_idx, seq in enumerate(seqs):
                # Convert sequence to NumPy array
                if not isinstance(seq, np.ndarray):
                    current = np.array(seq)
                else:
                    current = seq.copy()
                
                # Forward pass: store intermediate results for each layer
                intermediates = []
                
                # Process through each layer and save intermediate results
                for layer in self.layers:
                    M = layer['M']
                    P = layer['P']
                    # Compute M @ x for each vector in the sequence
                    M_x = current @ M.T
                    # Compute P @ (M_x) for each vector
                    output = M_x @ P.T
                    # Save intermediate results (input and M_x)
                    intermediates.append((current.copy(), M_x.copy()))
                    current = output
                
                total_vectors += len(seq)
                t = t_array[seq_idx]
                
                # Backward pass: calculate gradients
                # Start with error from final layer
                error = 2 * (current - t)  # Gradient of MSE loss
                
                # Propagate error backward through layers
                for l in range(self.num_layers-1, -1, -1):
                    x, M_x = intermediates[l]
                    layer = self.layers[l]
                    P = layer['P']
                    M = layer['M']
                    
                    # Compute gradient for P matrix
                    # dD/dP = error * M_x^T (for each vector, then sum)
                    grad_P = error.T @ M_x  # Matrix multiplication
                    grad_P_list[l] += grad_P
                    
                    # Compute gradient for M matrix
                    # First: backpropagate through P
                    Pt_error = error @ P  # P^T @ error
                    # Then: dD/dM = (P^T @ error) * x^T (for each vector, then sum)
                    grad_M = Pt_error.T @ x  # Matrix multiplication
                    grad_M_list[l] += grad_M
                    
                    # Propagate error to previous layer
                    if l > 0:
                        # Backpropagate through M: M^T @ (P^T @ error)
                        error = Pt_error @ M
                    else:
                        error = None
            
            # Apply updates to all layers
            if total_vectors > 0:
                lr = learning_rate / total_vectors
                for l in range(self.num_layers):
                    self.layers[l]['P'] -= lr * grad_P_list[l]
                    self.layers[l]['M'] -= lr * grad_M_list[l]
            
            # Compute current deviation
            D = self.deviation(seqs, t_list)
            history.append(D)
            print(f"Iter {it:3d}: D = {D:.6e}, lr = {learning_rate:.6f}")
            
            # Check convergence
            if abs(D - D_prev) < tol:
                print("Converged.")
                break
            D_prev = D
            
            # Update learning rate
            learning_rate *= decay_rate
        
        # Calculate statistics
        total_vectors = sum(len(seq) for seq in seqs)
        total_t = np.sum(t_array, axis=0)
        self.mean_vector_count = total_vectors / len(seqs)
        self.mean_t = total_t / len(seqs)
        self.trained = True
        return history

    def predict_t(self, seq):
        """
        Predict target vector t for a sequence.
        Optimal t is the mean of all final layer outputs.
        Returns m-dimensional vector.
        """
        outputs = np.array(self.describe(seq))
        if outputs.size == 0:
            return [0.0] * self.model_dims[-1]
        return np.mean(outputs, axis=0).tolist()

    def show(self):
        """Display model status."""
        print("HierarchicalDD Status:")
        print(f"  Input dimension (n) = {self.input_dim}")
        print(f"  Number of layers = {self.num_layers}")
        print(f"  Layer dimensions = {self.model_dims}")
        
        for i, layer in enumerate(self.layers):
            print(f"\n  Layer {i}:")
            print(f"    Input dim: {self.input_dim if i==0 else self.model_dims[i-1]}")
            print(f"    Output dim: {self.model_dims[i]}")
            print("    M matrix:")
            print(layer['M'])
            print("    P matrix:")
            print(layer['P'])

    def count_parameters(self):
        """Count learnable parameters (M and P matrices for all layers)."""
        total_params = 0
        print("Parameter Count:")
        
        for i, layer in enumerate(self.layers):
            M = layer['M']
            P = layer['P']
            M_params = M.size
            P_params = P.size
            layer_params = M_params + P_params
            total_params += layer_params
            
            print(f"  Layer {i}:")
            print(f"    M matrix: {M.shape} = {M_params} parameters")
            print(f"    P matrix: {P.shape} = {P_params} parameters")
            print(f"    Layer total: {layer_params}")
        
        print(f"Total parameters: {total_params}")
        return total_params

    def save(self, filename):
        """Save model state to file."""
        with open(filename, 'wb') as f:
            pickle.dump(self.__dict__, f)
        print(f"Model saved to {filename}")

    @classmethod
    def load(cls, filename):
        """Load model state from file."""
        with open(filename, 'rb') as f:
            state = pickle.load(f)
        # Create instance without calling __init__
        obj = cls.__new__(cls)
        obj.__dict__.update(state)
        print(f"Model loaded from {filename}")
        return obj


# === Example Usage ===
if __name__ == "__main__":

    from statistics import correlation, mean
    
    random.seed(22)
    np.random.seed(22)
    
    # Configuration with hierarchical dimensions
    input_dim = 20      # Input vector dimension (n)
    model_dims = [15, 10, 6]  # Output dimensions for each layer
    num_seqs = 10
    min_len, max_len = 100, 200

    # Generate synthetic training data
    seqs = []
    t_list = []
    for _ in range(num_seqs):
        L = random.randint(min_len, max_len)
        # Generate n-dimensional vectors
        seq = np.random.uniform(-1, 1, size=(L, input_dim))
        seqs.append(seq)
        # Generate target vectors with dimension matching final layer
        t_list.append(np.random.uniform(-1, 1, size=model_dims[-1]))
    
    # Train with Gradient Descent
    print("\nTraining HierarchicalDD with Gradient Descent:")
    hdd = HierarchicalDD(input_dim=input_dim, model_dims=model_dims)
    grad_history = hdd.grad_train(
        seqs, 
        t_list, 
        learning_rate=5.3,
        max_iters=500,
        decay_rate=1.0
    )

    # Predict targets
    print("\nPredictions (first 3 sequences):")
    for i, seq in enumerate(seqs[:3]):
        t_pred = hdd.predict_t(seq)
        print(f"Seq {i+1}: Target={[f'{x:.4f}' for x in t_list[i]]}")
        print(f"         Predicted={[f'{x:.4f}' for x in t_pred]}")
    
    # Calculate prediction correlations
    preds = [hdd.predict_t(seq) for seq in seqs]
    correlations = []
    for i in range(model_dims[-1]):
        actual = [t[i] for t in t_list]
        predicted = [p[i] for p in preds]
        corr = correlation(actual, predicted)
        correlations.append(corr)
        print(f"Output dim {i} correlation: {corr:.4f}")
    print(f"Average correlation: {mean(correlations):.4f}")

##    # Show model structure
##    print("\nModel structure:")
##    hdd.show()

    # Parameter count
    print("\nParameter count:")
    hdd.count_parameters()

    # Save and load model
    print("\nTesting save/load functionality:")
    hdd.save("hierarchical_model.pkl")
    loaded = HierarchicalDD.load("hierarchical_model.pkl")
    print("Loaded model prediction on first sequence:")
    print([f"{x:.4f}" for x in loaded.predict_t(seqs[0])])
