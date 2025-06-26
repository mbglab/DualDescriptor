# Copyright (C) Bin-Guang Ma (mbg@mail.hzau.edu.cn). All rights reserved.
# Linked Hierarchical Dual Descriptor class for vector sequences
# Modified to support sequence length transformation between layers using Linker matrices
# Added linker_trainable parameter to control Linker matrix training
# Author: Bin-Guang Ma; Date: 2025-6-20

import math
import random
import pickle
from statistics import correlation, mean

class LinkedHierarchicalDD:
    """
    Linked Hierarchical Dual Descriptor for sequences of n-dimensional input vectors and m-dimensional outputs.
    Key modifications:
    - Each layer now has a Linker matrix for sequence length transformation
    - Layer operations: V = P * (M * X * Linker)
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
        
        # Initialize layers with M, P, and Linker matrices
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
            
            # Initialize matrices with appropriate dimensions
            layer = {
                # Feature transformation matrix: out_feat × in_feat
                'M': [[random.uniform(-0.5, 0.5) for _ in range(in_feat)] 
                      for _ in range(out_feat)],
                
                # Feature transformation matrix: out_feat × out_feat
                'P': [[random.uniform(-0.1, 0.1) for _ in range(out_feat)] 
                      for _ in range(out_feat)],
                
                # Sequence length transformation matrix: in_seq × out_seq
                'Linker': [[random.uniform(-0.1, 0.1) for _ in range(out_seq)] 
                           for _ in range(in_seq)],
                
                # Trainability flag for Linker matrix
                'linker_trainable': self.linker_trainable[i]
            }
            self.layers.append(layer)

    # ---- linear algebra helpers ----
    def mat_vec(self, M, v):
        """Multiply matrix M (a×b) by vector v (length b) → vector of length a."""
        a = len(M)
        b = len(M[0])
        assert b == len(v), f"Matrix columns {b} != vector length {len(v)}"
        return [sum(M[i][j] * v[j] for j in range(b)) for i in range(a)]
    
    def mat_mul(self, A, B):
        """Multiply p×q matrix A by q×r matrix B → p×r matrix."""
        p, q = len(A), len(A[0])
        r = len(B[0])
        C = [[0.0]*r for _ in range(p)]
        for i in range(p):
            for k in range(q):
                aik = A[i][k]
                for j in range(r):
                    C[i][j] += aik * B[k][j]
        return C

    def transpose(self, M):
        """Transpose a matrix."""
        return [list(col) for col in zip(*M)]
    
    def vec_sub(self, u, v):
        """Subtract two vectors of same length."""
        assert len(u) == len(v), "Vector dimensions must match for subtraction"
        return [u[i] - v[i] for i in range(len(u))]
    
    def dot(self, u, v):
        """Dot product of two vectors of same length."""
        assert len(u) == len(v), "Vector dimensions must match for dot product"
        return sum(u[i] * v[i] for i in range(len(u)))
    
    def vec_avg(self, vectors):
        """Compute element-wise average of a list of vectors."""
        if not vectors:
            return []
        dim = len(vectors[0])
        avg = [0.0] * dim
        for vec in vectors:
            for i in range(dim):
                avg[i] += vec[i]
        return [x / len(vectors) for x in avg]
    
    def describe(self, seq):
        """
        Compute the hierarchical representation for a given vector sequence.
        Returns the average vector from the final layer output.
        
        Args:
            seq (list): List of n-dimensional vectors (length must match input_seq_len)
            
        Returns:
            list: Average vector from the final layer output (m-dimensional)
        """
        # Convert sequence to matrix: input_dim × input_seq_len
        # Each column is a feature vector
        X = [list(col) for col in zip(*seq)]  # Transpose: input_dim rows, input_seq_len columns
        
        # Process through each layer
        for i, layer in enumerate(self.layers):
            M = layer['M']
            Linker = layer['Linker']
            P = layer['P']
            
            # Apply transformations: V = P * (M * X * Linker)
            T = self.mat_mul(M, X)           # Feature transformation: out_feat × in_seq
            U = self.mat_mul(T, Linker)       # Sequence length transformation: out_feat × out_seq
            V = self.mat_mul(P, U)            # Feature transformation: out_feat × out_seq
            
            # Output becomes input for next layer
            X = V
        
        # Final output matrix: model_dims[-1] × linker_dims[-1]
        # Convert matrix back to list of vectors (each column is a vector)
        output_vectors = []
        for j in range(len(X[0])):  # Iterate through columns
            vec = [X[i][j] for i in range(len(X))]
            output_vectors.append(vec)
            
        # Return average of all output vectors
        return self.vec_avg(output_vectors)
    
    def deviation(self, seqs, t_list):
        """
        Compute mean squared deviation D:
        D = average over sequences of ||final_layer_avg_vector - t_j||^2.
        """
        total = 0.0
        count = 0
        for seq, t in zip(seqs, t_list):
            # Get final layer average vector
            avg_vec = self.describe(seq)
            err = self.vec_sub(avg_vec, t)
            total += self.dot(err, err)
            count += 1
        return total / count if count else 0.0

    def grad_train(self, seqs, t_list, max_iters=100, tol=1e-8, 
                   learning_rate=0.01, decay_rate=0.95):
        """
        Train using gradient descent with backpropagation through layers.
        Supports learning M, P, and Linker matrices (with Linker trainability control).
        """
        history = []
        D_prev = float('inf')
        
        # Initialize gradient storage for all layers
        grad_M_list = []
        grad_P_list = []
        grad_Linker_list = []
        
        for i in range(self.num_layers):
            # Determine layer dimensions
            if i == 0:
                in_feat = self.input_dim
                in_seq = self.input_seq_len
            else:
                in_feat = self.model_dims[i-1]
                in_seq = self.linker_dims[i-1]
                
            out_feat = self.model_dims[i]
            out_seq = self.linker_dims[i]
            
            # Initialize gradient matrices
            grad_M_list.append([[0.0]*in_feat for _ in range(out_feat)])
            grad_P_list.append([[0.0]*out_feat for _ in range(out_feat)])
            grad_Linker_list.append([[0.0]*out_seq for _ in range(in_seq)])
        
        for it in range(max_iters):
            total_sequences = len(seqs)
            
            # Reset gradients
            for l in range(self.num_layers):
                for i in range(len(grad_M_list[l])):
                    grad_M_list[l][i] = [0.0] * len(grad_M_list[l][i])
                for i in range(len(grad_P_list[l])):
                    grad_P_list[l][i] = [0.0] * len(grad_P_list[l][i])
                for i in range(len(grad_Linker_list[l])):
                    grad_Linker_list[l][i] = [0.0] * len(grad_Linker_list[l][i])
            
            total_loss = 0.0
            
            # Process each sequence
            for seq, t in zip(seqs, t_list):
                # Forward pass: store intermediate results
                forward_cache = []
                X = [list(col) for col in zip(*seq)]  # Convert to matrix
                
                for l, layer in enumerate(self.layers):
                    M = layer['M']
                    Linker = layer['Linker']
                    P = layer['P']
                    
                    # Forward computations
                    T = self.mat_mul(M, X)           # Feature transformation
                    U = self.mat_mul(T, Linker)       # Sequence length transformation
                    V = self.mat_mul(P, U)            # Feature transformation
                    
                    # Save intermediate results for backpropagation
                    forward_cache.append((X, T, U, V))
                    X = V  # Output becomes input for next layer
                
                # Final output matrix
                final_output = forward_cache[-1][3]
                
                # Convert matrix to list of vectors and compute average
                output_vectors = []
                for j in range(len(final_output[0])):
                    vec = [final_output[i][j] for i in range(len(final_output))]
                    output_vectors.append(vec)
                avg_vec = self.vec_avg(output_vectors)
                
                # Compute error and loss
                err = self.vec_sub(avg_vec, t)
                loss = self.dot(err, err)
                total_loss += loss
                
                # Backward pass: start from final layer
                # Gradient of loss w.r.t avg_vec is 2*(avg_vec - t)
                d_avg = [2 * e for e in err]
                
                # Gradient w.r.t final output vectors
                num_vectors = len(output_vectors)
                d_output_vectors = [[d_avg[i] / num_vectors for i in range(len(d_avg))] 
                                    for _ in range(num_vectors)]
                
                # Convert to matrix gradient (dL/dV)
                dV = [[0.0] * len(final_output[0]) for _ in range(len(final_output))]
                for j in range(len(d_output_vectors)):
                    for i in range(len(d_output_vectors[j])):
                        dV[i][j] = d_output_vectors[j][i]
                
                # Backpropagate through layers
                for l in range(self.num_layers-1, -1, -1):
                    X, T, U, V = forward_cache[l]
                    M = self.layers[l]['M']
                    Linker = self.layers[l]['Linker']
                    P = self.layers[l]['P']
                    
                    # Gradient w.r.t P: dL/dP = dL/dV * U^T
                    dP = self.mat_mul(dV, self.transpose(U))
                    
                    # Gradient w.r.t U: dL/dU = P^T * dL/dV
                    dU = self.mat_mul(self.transpose(P), dV)
                    
                    # Gradient w.r.t Linker: dL/dLinker = T^T * dL/dU
                    dLinker = self.mat_mul(self.transpose(T), dU)
                    
                    # Gradient w.r.t T: dL/dT = dL/dU * Linker^T
                    dT = self.mat_mul(dU, self.transpose(Linker))
                    
                    # Gradient w.r.t M: dL/dM = dL/dT * X^T
                    dM = self.mat_mul(dT, self.transpose(X))
                    
                    # Gradient w.r.t X (for previous layer): dL/dX = M^T * dL/dT
                    dX_prev = self.mat_mul(self.transpose(M), dT)
                    
                    # Accumulate gradients
                    for i in range(len(grad_M_list[l])):
                        for j in range(len(grad_M_list[l][i])):
                            grad_M_list[l][i][j] += dM[i][j]
                    
                    for i in range(len(grad_P_list[l])):
                        for j in range(len(grad_P_list[l][i])):
                            grad_P_list[l][i][j] += dP[i][j]
                    
                    # Only accumulate Linker gradient if trainable
                    if self.layers[l]['linker_trainable']:
                        for i in range(len(grad_Linker_list[l])):
                            for j in range(len(grad_Linker_list[l][i])):
                                grad_Linker_list[l][i][j] += dLinker[i][j]
                    
                    # Prepare for previous layer
                    dV = dX_prev  # dV for next lower layer
            
            # Update parameters
            lr = learning_rate / total_sequences if total_sequences else learning_rate
            for l in range(self.num_layers):
                for i in range(len(self.layers[l]['M'])):
                    for j in range(len(self.layers[l]['M'][i])):
                        self.layers[l]['M'][i][j] -= lr * grad_M_list[l][i][j]
                
                for i in range(len(self.layers[l]['P'])):
                    for j in range(len(self.layers[l]['P'][i])):
                        self.layers[l]['P'][i][j] -= lr * grad_P_list[l][i][j]
                
                # Only update Linker if trainable
                if self.layers[l]['linker_trainable']:
                    for i in range(len(self.layers[l]['Linker'])):
                        for j in range(len(self.layers[l]['Linker'][i])):
                            self.layers[l]['Linker'][i][j] -= lr * grad_Linker_list[l][i][j]
            
            # Compute and record deviation
            D = total_loss / total_sequences if total_sequences else 0.0
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
            for row in layer['M']:
                print("      ", [f"{x:.4f}" for x in row])
            print("    P matrix:")
            for row in layer['P']:
                print("      ", [f"{x:.4f}" for x in row])
            print("    Linker matrix:")
            for row in layer['Linker']:
                print("      ", [f"{x:.4f}" for x in row])
    
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
    random.seed(11)
    
    # Configuration with hierarchical dimensions
    input_dim = 20          # Input vector dimension (n)
    input_seq_len = 100     # Fixed input sequence length
    model_dims = [15, 10, 5]  # Feature dimensions for each layer
    linker_dims = [50, 20, 10]  # Sequence lengths for each layer
    num_seqs = 10           # Number of training sequences
    
    # Generate synthetic training data
    print(f"Generating {num_seqs} sequences with length {input_seq_len}...")
    seqs = []
    t_list = []
    for _ in range(num_seqs):
        # Generate n-dimensional vectors (fixed length)
        seq = [[random.uniform(-1, 1) for _ in range(input_dim)] 
               for __ in range(input_seq_len)]
        seqs.append(seq)
        # Generate target vectors with dimension matching final layer
        t_list.append([random.uniform(-1, 1) for _ in range(model_dims[-1])])
    
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
        learning_rate=5.0,
        max_iters=200,
        decay_rate=0.95
    )
    
    # Predict targets
    print("\nPredictions (first 3 sequences):")
    for i, seq in enumerate(seqs[:3]):
        t_pred = lhdd_trainable.predict_t(seq)
        print(f"Seq {i+1}: Target={[f'{x:.4f}' for x in t_list[i]]}")
        print(f"         Predicted={[f'{x:.4f}' for x in t_pred]}")
    
    # Calculate prediction correlations
    preds = [lhdd_trainable.predict_t(seq) for seq in seqs]
    correlations = []
    for i in range(model_dims[-1]):
        actual = [t[i] for t in t_list]
        predicted = [p[i] for p in preds]
        corr = correlation(actual, predicted)
        correlations.append(corr)
        print(f"Output dim {i} correlation: {corr:.4f}")
    print(f"Average correlation: {mean(correlations):.4f}")

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
        linker_trainable=[False, True, False]  # Per-layer control
    )
    
    print("\nTraining with Gradient Descent:")
    grad_history_mixed = lhdd_mixed.grad_train(
        seqs, 
        t_list, 
        learning_rate=5.0,
        max_iters=200,
        decay_rate=0.95
    )
    
    # Calculate prediction correlations for mixed case
    preds_mixed = [lhdd_mixed.predict_t(seq) for seq in seqs]
    correlations_mixed = []
    for i in range(model_dims[-1]):
        actual = [t[i] for t in t_list]
        predicted = [p[i] for p in preds_mixed]
        corr = correlation(actual, predicted)
        correlations_mixed.append(corr)
        print(f"Output dim {i} correlation: {corr:.4f}")
    print(f"Average correlation: {mean(correlations_mixed):.4f}")

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
        learning_rate=5.0,
        max_iters=200,
        decay_rate=0.95
    )
    
    # Calculate prediction correlations for fixed Linker case
    preds_fixed = [lhdd_fixed.predict_t(seq) for seq in seqs]
    correlations_fixed = []
    for i in range(model_dims[-1]):
        actual = [t[i] for t in t_list]
        predicted = [p[i] for p in preds_fixed]
        corr = correlation(actual, predicted)
        correlations_fixed.append(corr)
        print(f"Output dim {i} correlation: {corr:.4f}")
    print(f"Average correlation: {mean(correlations_fixed):.4f}")

    # Parameter count for fixed case
    print("\nParameter count (fixed Linkers):")
    total_params_fixed, trainable_params_fixed = lhdd_fixed.count_parameters()
    
    # Save and load model
    print("\nTesting save/load functionality:")
    lhdd_trainable.save("linked_hierarchical_model.pkl")
    loaded = LinkedHierarchicalDD.load("linked_hierarchical_model.pkl")
    print("Loaded model prediction on first sequence:")
    print([f"{x:.4f}" for x in loaded.predict_t(seqs[0])])
