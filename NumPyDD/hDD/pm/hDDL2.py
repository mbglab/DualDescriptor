# Copyright (C) Bin-Guang Ma (mbg@mail.hzau.edu.cn). All rights reserved.
# Linked Hierarchical Dual Descriptor class for vector sequences
# Modified to support sequence length transformation between layers using Linker matrices
# Added linker_trainable parameter to control Linker matrix training
# Optimized with NumPy for computational efficiency
# Author: Bin-Guang Ma; Date: 2025-6-20

import math
import random
import pickle
import numpy as np  # Import NumPy for efficient array operations

class LinkedHierarchicalDD:
    """
    Linked Hierarchical Dual Descriptor for sequences of n-dimensional input vectors and m-dimensional outputs.
    Key modifications:
    - Each layer now has a Linker matrix for sequence length transformation
    - Layer operations: V = P @ (M @ X @ Linker) using NumPy for efficiency
    - Supports changing both vector dimensions and sequence lengths between layers
    - Added linker_trainable parameter to control Linker matrix training
    
    Args:
        input_dim (int): Dimensionality of input vectors (n)
        model_dims (list): Output dimensions for each layer (m_i)
        linker_dims (list): Output sequence lengths for each layer (L_i)
        input_seq_len (int): Fixed length of input sequences (L)
        linker_trainable (bool or list): Controls if Linker matrices are trainable.
            If bool: applies to all layers.
            If list: must match num_layers length, specifies per-layer trainability.
            Default: False (Linker matrices are not trained)
    """
    
    def __init__(self, input_dim=4, model_dims=[4], linker_dims=[4], input_seq_len=100, linker_trainable=False):
        self.input_dim = input_dim          # Input vector dimension (n)
        self.model_dims = model_dims        # Output dimensions for each layer (m_i)
        self.linker_dims = linker_dims      # Output sequence lengths for each layer (L_i)
        self.input_seq_len = input_seq_len  # Fixed input sequence length (L)
        self.num_layers = len(model_dims)   # Number of layers
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
        
        # Initialize layers with M, P, and Linker matrices as NumPy arrays
        self.layers = []
        for i in range(self.num_layers):
            # Determine input dimensions for this layer
            if i == 0:
                in_feat = input_dim        # Input feature dimension
                in_seq = input_seq_len      # Input sequence length
            else:
                in_feat = model_dims[i-1]  # Previous layer's output dimension
                in_seq = linker_dims[i-1]   # Previous layer's output sequence length
                
            out_feat = model_dims[i]        # Output feature dimension
            out_seq = linker_dims[i]        # Output sequence length
            
            # Initialize matrices with appropriate dimensions using NumPy
            layer = {
                # Feature transformation matrix: out_feat × in_feat
                'M': np.random.uniform(-0.5, 0.5, size=(out_feat, in_feat)),
                
                # Feature transformation matrix: out_feat × out_feat
                'P': np.random.uniform(-0.1, 0.1, size=(out_feat, out_feat)),
                
                # Sequence length transformation matrix: in_seq × out_seq
                'Linker': np.random.uniform(-0.1, 0.1, size=(in_seq, out_seq)),
                
                # Trainability flag for Linker matrix
                'linker_trainable': self.linker_trainable[i]
            }
            self.layers.append(layer)

    def describe(self, seq):
        """
        Compute the hierarchical representation for a given vector sequence.
        Returns the average vector from the final layer output.
        
        Args:
            seq (list): List of n-dimensional vectors (length must match input_seq_len)
            
        Returns:
            np.array: Average vector from the final layer output (m-dimensional)
        """
        # Convert sequence to matrix: input_dim × input_seq_len
        X = np.array(seq).T  # Transpose to get input_dim rows, input_seq_len columns
        
        # Process through each layer using NumPy matrix operations
        for i, layer in enumerate(self.layers):
            M = layer['M']
            Linker = layer['Linker']
            P = layer['P']
            
            # Apply transformations: V = P @ (M @ X @ Linker)
            T = M @ X            # Feature transformation: out_feat × in_seq
            U = T @ Linker       # Sequence length transformation: out_feat × out_seq
            V = P @ U            # Feature transformation: out_feat × out_seq
            
            # Output becomes input for next layer
            X = V
        
        # Final output matrix: model_dims[-1] × linker_dims[-1]
        # Compute average of all output vectors along the sequence dimension
        return np.mean(X, axis=1)
    
    def deviation(self, seqs, t_list):
        """
        Compute mean squared deviation D:
        D = average over sequences of ||final_layer_avg_vector - t_j||^2.
        """
        total = 0.0
        count = 0
        t_arr = np.array(t_list)
        
        for idx, seq in enumerate(seqs):
            # Get final layer average vector
            avg_vec = self.describe(seq)
            err = avg_vec - t_arr[idx]
            total += np.dot(err, err)
            count += 1
            
        return total / count if count else 0.0

    def grad_train(self, seqs, t_list, max_iters=100, tol=1e-8, 
                   learning_rate=0.01, decay_rate=0.95):
        """
        Train using gradient descent with backpropagation through layers.
        Supports learning M, P, and Linker matrices (with Linker trainability control).
        Uses NumPy for efficient matrix operations.
        """
        history = []
        D_prev = float('inf')
        t_arr = np.array(t_list)
        num_seqs = len(seqs)
        
        # Initialize gradient storage for all layers
        grad_M_list = [np.zeros_like(layer['M']) for layer in self.layers]
        grad_P_list = [np.zeros_like(layer['P']) for layer in self.layers]
        grad_Linker_list = [np.zeros_like(layer['Linker']) for layer in self.layers]
        
        for it in range(max_iters):
            total_loss = 0.0
            
            # Reset gradients to zero
            for grad_M, grad_P, grad_Linker in zip(grad_M_list, grad_P_list, grad_Linker_list):
                grad_M.fill(0)
                grad_P.fill(0)
                grad_Linker.fill(0)
            
            # Process each sequence
            for seq_idx, seq in enumerate(seqs):
                # Forward pass: store intermediate results
                forward_cache = []
                X = np.array(seq).T  # Convert to matrix (input_dim x input_seq_len)
                
                for l, layer in enumerate(self.layers):
                    M = layer['M']
                    Linker = layer['Linker']
                    P = layer['P']
                    
                    # Forward computations
                    T = M @ X           # Feature transformation
                    U = T @ Linker      # Sequence length transformation
                    V = P @ U           # Feature transformation
                    
                    # Save intermediate results for backpropagation
                    forward_cache.append((X.copy(), T.copy(), U.copy(), V.copy()))
                    X = V  # Output becomes input for next layer
                
                # Final output matrix
                final_output = forward_cache[-1][3]  # V from last layer
                
                # Compute average vector
                avg_vec = np.mean(final_output, axis=1)
                
                # Compute error and loss
                err = avg_vec - t_arr[seq_idx]
                loss = np.dot(err, err)
                total_loss += loss
                
                # Backward pass: start from final layer
                # Gradient of loss w.r.t avg_vec is 2*(avg_vec - t)
                d_avg = 2 * err
                
                # Gradient w.r.t final output vectors
                num_vectors = final_output.shape[1]
                d_output_vectors = np.outer(d_avg, np.ones(num_vectors)) / num_vectors
                
                # Gradient w.r.t V (final layer output)
                dV = d_output_vectors
                
                # Backpropagate through layers
                for l in range(self.num_layers-1, -1, -1):
                    X, T, U, V = forward_cache[l]
                    M = self.layers[l]['M']
                    Linker = self.layers[l]['Linker']
                    P = self.layers[l]['P']
                    
                    # Gradient w.r.t P: dL/dP = dL/dV @ U^T
                    dP = dV @ U.T
                    
                    # Gradient w.r.t U: dL/dU = P^T @ dL/dV
                    dU = P.T @ dV
                    
                    # Gradient w.r.t Linker: dL/dLinker = T^T @ dL/dU
                    dLinker = T.T @ dU
                    
                    # Gradient w.r.t T: dL/dT = dL/dU @ Linker^T
                    dT = dU @ Linker.T
                    
                    # Gradient w.r.t M: dL/dM = dL/dT @ X^T
                    dM = dT @ X.T
                    
                    # Gradient w.r.t X (for previous layer): dL/dX = M^T @ dL/dT
                    dX_prev = M.T @ dT
                    
                    # Accumulate gradients
                    grad_M_list[l] += dM
                    grad_P_list[l] += dP
                    
                    # Only accumulate Linker gradient if trainable
                    if self.layers[l]['linker_trainable']:
                        grad_Linker_list[l] += dLinker
                    
                    # Prepare for previous layer
                    dV = dX_prev  # dV for next lower layer
            
            # Update parameters
            lr = learning_rate / num_seqs if num_seqs else learning_rate
            for l in range(self.num_layers):
                self.layers[l]['M'] -= lr * grad_M_list[l]
                self.layers[l]['P'] -= lr * grad_P_list[l]
                
                # Only update Linker if trainable
                if self.layers[l]['linker_trainable']:
                    self.layers[l]['Linker'] -= lr * grad_Linker_list[l]
            
            # Compute and record deviation
            D = total_loss / num_seqs if num_seqs else 0.0
            history.append(D)
            print(f"Iter {it:3d}: D = {D:.6e}, lr = {learning_rate:.6f}")
            
            # Check convergence
            if abs(D - D_prev) < tol:
                print("Converged.")
                break
            D_prev = D
            
            # Update learning rate
            learning_rate *= decay_rate
        
        self.trained = True
        return history

    def predict_t(self, seq):
        """Predict target vector t for a sequence (same as describe)."""
        return self.describe(seq)
    
    def show(self):
        """Display model status."""
        print("LinkedHierarchicalDD Status:")
        print(f"  Input dimension (n) = {self.input_dim}")
        print(f"  Input sequence length = {self.input_seq_len}")
        print(f"  Number of layers = {self.num_layers}")
        print(f"  Layer dimensions (features) = {self.model_dims}")
        print(f"  Layer dimensions (sequence) = {self.linker_dims}")
        
        for i, layer in enumerate(self.layers):
            print(f"\n  Layer {i}:")
            in_feat = self.input_dim if i == 0 else self.model_dims[i-1]
            in_seq = self.input_seq_len if i == 0 else self.linker_dims[i-1]
            print(f"    Input features: {in_feat}, Input sequence: {in_seq}")
            print(f"    Output features: {self.model_dims[i]}, Output sequence: {self.linker_dims[i]}")
            print(f"    Linker trainable: {layer['linker_trainable']}")
            print("    M matrix:")
            print(layer['M'])
            print("    P matrix:")
            print(layer['P'])
            print("    Linker matrix:")
            print(layer['Linker'])
    
    def count_parameters(self):
        """Count learnable parameters (M, P, and Linker matrices for all layers)."""
        total_params = 0
        trainable_params = 0
        print("Parameter Count:")
        
        for i, layer in enumerate(self.layers):
            # Determine input dimensions
            if i == 0:
                in_feat = self.input_dim
                in_seq = self.input_seq_len
            else:
                in_feat = self.model_dims[i-1]
                in_seq = self.linker_dims[i-1]
                
            out_feat = self.model_dims[i]
            out_seq = self.linker_dims[i]
            
            M_params = out_feat * in_feat
            P_params = out_feat * out_feat
            Linker_params = in_seq * out_seq
            layer_params = M_params + P_params + Linker_params
            total_params += layer_params
            
            # Count trainable parameters
            layer_trainable = M_params + P_params
            if layer['linker_trainable']:
                layer_trainable += Linker_params
            
            trainable_params += layer_trainable
            
            print(f"  Layer {i}:")
            print(f"    M matrix: {out_feat}×{in_feat} = {M_params} parameters")
            print(f"    P matrix: {out_feat}×{out_feat} = {P_params} parameters")
            print(f"    Linker matrix: {in_seq}×{out_seq} = {Linker_params} parameters")
            print(f"    Linker trainable: {layer['linker_trainable']}")
            print(f"    Layer total: {layer_params} (trainable: {layer_trainable})")
        
        print(f"Total parameters: {total_params}")
        print(f"Trainable parameters: {trainable_params}")
        return total_params, trainable_params

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
    np.random.seed(13)
    random.seed(13)
    
    # Configuration with hierarchical dimensions
    input_dim = 20          # Input vector dimension (n)
    input_seq_len = 100     # Fixed input sequence length
    model_dims = [15, 10]  # Feature dimensions for each layer
    linker_dims = [50, 20]  # Sequence lengths for each layer
    num_seqs = 20           # Number of training sequences
    
    # Generate synthetic training data
    print(f"Generating {num_seqs} sequences with length {input_seq_len}...")
    seqs = []
    t_list = []
    for _ in range(num_seqs):
        # Generate n-dimensional vectors (fixed length)
        seq = np.random.uniform(-1, 1, size=(input_seq_len, input_dim))
        seqs.append(seq)
        # Generate target vectors with dimension matching final layer
        t_list.append(np.random.uniform(-1, 1, size=model_dims[-1]))
    
    # Test Case 1: All Linker matrices trainable
    print("\n=== Test Case 1: All Linker Matrices Trainable ===")
    lhdd_trainable = LinkedHierarchicalDD(
        input_dim=input_dim,
        model_dims=model_dims,
        linker_dims=linker_dims,
        input_seq_len=input_seq_len,
        linker_trainable=True  # All Linker matrices trainable
    )
    
    print("\nTraining with Gradient Descent:")
    grad_history = lhdd_trainable.grad_train(
        seqs, 
        t_list, 
        learning_rate=1.0,
        max_iters=500,
        decay_rate=0.96
    )
    
    # Predict targets
    print("\nPredictions (first 3 sequences):")
    for i, seq in enumerate(seqs[:3]):
        t_pred = lhdd_trainable.predict_t(seq)
        print(f"Seq {i+1}: Target={np.array2string(t_list[i], precision=4)}")
        print(f"         Predicted={np.array2string(t_pred, precision=4)}")
    
    # Calculate prediction correlations
    preds = [lhdd_trainable.predict_t(seq) for seq in seqs]
    correlations = []
    for i in range(model_dims[-1]):
        actual = [t[i] for t in t_list]
        predicted = [p[i] for p in preds]
        corr_matrix = np.corrcoef(actual, predicted)
        corr = corr_matrix[0, 1]
        correlations.append(corr)
        print(f"Output dim {i} correlation: {corr:.4f}")
    print(f"Average correlation: {np.mean(correlations):.4f}")

    # Parameter count
    print("\nParameter count:")
    total_params, trainable_params = lhdd_trainable.count_parameters()
    
    # Test Case 2: Mixed trainability (first layer trainable, others not)
    print("\n\n=== Test Case 2: Mixed Linker Trainability ===")
    lhdd_mixed = LinkedHierarchicalDD(
        input_dim=input_dim,
        model_dims=model_dims,
        linker_dims=linker_dims,
        input_seq_len=input_seq_len,
        linker_trainable=[False, True]  # Per-layer control
    )
    
    print("\nTraining with Gradient Descent:")
    grad_history_mixed = lhdd_mixed.grad_train(
        seqs, 
        t_list, 
        learning_rate=1.0,
        max_iters=500,
        decay_rate=0.96
    )
    
    # Calculate prediction correlations for mixed case
    preds_mixed = [lhdd_mixed.predict_t(seq) for seq in seqs]
    correlations_mixed = []
    for i in range(model_dims[-1]):
        actual = [t[i] for t in t_list]
        predicted = [p[i] for p in preds_mixed]
        corr_matrix = np.corrcoef(actual, predicted)
        corr = corr_matrix[0, 1]
        correlations_mixed.append(corr)
        print(f"Output dim {i} correlation: {corr:.4f}")
    print(f"Average correlation: {np.mean(correlations_mixed):.4f}")

    # Parameter count for mixed case
    print("\nParameter count (mixed trainability):")
    total_params_mixed, trainable_params_mixed = lhdd_mixed.count_parameters()
    
    # Test Case 3: No Linker matrices trainable
    print("\n\n=== Test Case 3: No Linker Matrices Trainable ===")
    lhdd_fixed = LinkedHierarchicalDD(
        input_dim=input_dim,
        model_dims=model_dims,
        linker_dims=linker_dims,
        input_seq_len=input_seq_len,
        linker_trainable=False  # No Linker matrices trainable
    )
    
    print("\nTraining with Gradient Descent:")
    grad_history_fixed = lhdd_fixed.grad_train(
        seqs, 
        t_list, 
        learning_rate=1.0,
        max_iters=500,
        decay_rate=0.96
    )
    
    # Calculate prediction correlations for fixed Linker case
    preds_fixed = [lhdd_fixed.predict_t(seq) for seq in seqs]
    correlations_fixed = []
    for i in range(model_dims[-1]):
        actual = [t[i] for t in t_list]
        predicted = [p[i] for p in preds_fixed]
        corr_matrix = np.corrcoef(actual, predicted)
        corr = corr_matrix[0, 1]
        correlations_fixed.append(corr)
        print(f"Output dim {i} correlation: {corr:.4f}")
    print(f"Average correlation: {np.mean(correlations_fixed):.4f}")

    # Parameter count for fixed case
    print("\nParameter count (fixed Linkers):")
    total_params_fixed, trainable_params_fixed = lhdd_fixed.count_parameters()
    
    # Save and load model
    print("\nTesting save/load functionality:")
    lhdd_trainable.save("linked_hierarchical_model.pkl")
    loaded = LinkedHierarchicalDD.load("linked_hierarchical_model.pkl")
    print("Loaded model prediction on first sequence:")
    print(loaded.predict_t(seqs[0]))
