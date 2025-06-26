# Copyright (C) Bin-Guang Ma (mbg@mail.hzau.edu.cn). All rights reserved.
# Hierarchical Dual Descriptor class for vector sequences
# Modified to support n-dimensional input and m-dimensional output through multiple layers
# Author: Bin-Guang Ma; Date: 2025-6-20

import math
import random
import pickle
from statistics import correlation, mean

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
        
        # Initialize layers
        self.layers = []
        for i, dim in enumerate(model_dims):
            # Determine input dimension for this layer
            if i == 0:
                layer_input_dim = input_dim
            else:
                layer_input_dim = model_dims[i-1]
                
            # Create a new layer
            layer = {
                'M': [[random.uniform(-0.5, 0.5) for _ in range(layer_input_dim)] 
                      for _ in range(dim)],
                'P': [[random.uniform(-0.1, 0.1) for _ in range(dim)] 
                      for _ in range(dim)]
            }
            self.layers.append(layer)

    # ---- linear algebra helpers ----
    def mat_vec(self, M, v):
        """Multiply matrix M (with dimensions a×b) by vector v (length b) → vector of length a."""
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
    
    def describe(self, seq):
        """
        Compute the output of the hierarchical model for a given vector sequence.
        
        Args:
            seq (list): List of n-dimensional vectors
            
        Returns:
            list: Output from the final layer for each position
        """
        current = seq
        # Process through each layer
        for layer in self.layers:
            output = []
            M = layer['M']
            P = layer['P']
            for xk in current:
                M_xk = self.mat_vec(M, xk)
                Nk = self.mat_vec(P, M_xk)
                output.append(Nk)
            current = output
        return current

    def deviation(self, seqs, t_list):
        """
        Compute mean squared deviation D:
        D = average over all positions and sequences of ||Final_layer_output - t_j||^2.
        """
        total = 0.0
        count = 0
        for seq, t in zip(seqs, t_list):
            # Get final layer outputs
            outputs = self.describe(seq)
            for vec in outputs:
                # Compute error: vec - t
                err = self.vec_sub(vec, t)
                total += self.dot(err, err)
                count += 1
        return total / count if count else 0.0

    def grad_train(self, seqs, t_list, max_iters=100, tol=1e-8, 
                   learning_rate=0.01, decay_rate=0.95):
        """
        Train using gradient descent with backpropagation through layers.
        """
        history = []
        D_prev = float('inf')
        
        # Initialize gradients for all layers
        grad_P_list = []
        grad_M_list = []
        for layer in self.layers:
            m = len(layer['P'])
            grad_P_list.append([[0.0] * m for _ in range(m)])
            
            if self.layers.index(layer) == 0:
                input_dim = self.input_dim
            else:
                input_dim = self.model_dims[self.layers.index(layer)-1]
            grad_M_list.append([[0.0] * input_dim for _ in range(m)])
        
        for it in range(max_iters):
            total_vectors = 0
            
            # Reset gradients
            for i in range(self.num_layers):
                for j in range(len(grad_P_list[i])):
                    grad_P_list[i][j] = [0.0] * len(grad_P_list[i][j])
                for j in range(len(grad_M_list[i])):
                    grad_M_list[i][j] = [0.0] * len(grad_M_list[i][j])
            
            # Process each sequence
            for seq, t in zip(seqs, t_list):
                # Forward pass: store intermediate results for each layer
                intermediates = []
                current = seq
                
                # Process through each layer and save intermediate results
                for layer in self.layers:
                    M = layer['M']
                    P = layer['P']
                    layer_intermediate = []
                    output = []
                    for xk in current:
                        M_xk = self.mat_vec(M, xk)
                        Nk = self.mat_vec(P, M_xk)
                        layer_intermediate.append((xk, M_xk))
                        output.append(Nk)
                    intermediates.append(layer_intermediate)
                    current = output
                
                total_vectors += len(seq)
                
                # Backward pass: calculate gradients
                for k in range(len(seq)):
                    # Start with error from final layer
                    error = [2 * (x - t_val) for x, t_val in zip(current[k], t)]
                    
                    # Propagate error backward through layers
                    for l in range(self.num_layers-1, -1, -1):
                        x, M_x = intermediates[l][k]
                        layer = self.layers[l]
                        P = layer['P']
                        M = layer['M']
                        
                        # Compute gradient for P matrix
                        for i in range(len(error)):
                            for j in range(len(M_x)):
                                grad_P_list[l][i][j] += error[i] * M_x[j]
                        
                        # Compute gradient for M matrix
                        Pt_error = self.mat_vec(self.transpose(P), error)
                        for i in range(len(Pt_error)):
                            for j in range(len(x)):
                                grad_M_list[l][i][j] += Pt_error[i] * x[j]
                        
                        # Propagate error to previous layer
                        if l > 0:
                            Mt_error = self.mat_vec(self.transpose(M), Pt_error)
                            error = Mt_error
                        else:
                            error = None
            
            # Apply updates to all layers
            if total_vectors > 0:
                lr = learning_rate / total_vectors
                for l in range(self.num_layers):
                    P = self.layers[l]['P']
                    M = self.layers[l]['M']
                    for i in range(len(P)):
                        for j in range(len(P[i])):
                            P[i][j] -= lr * grad_P_list[l][i][j]
                    for i in range(len(M)):
                        for j in range(len(M[i])):
                            M[i][j] -= lr * grad_M_list[l][i][j]
            
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
        total_t = [0.0] * self.model_dims[-1]
        for t in t_list:
            for d in range(self.model_dims[-1]):
                total_t[d] += t[d]
        self.mean_vector_count = total_vectors / len(seqs)
        self.mean_t = [t_val / len(seqs) for t_val in total_t]
        self.trained = True
        return history

    def predict_t(self, seq):
        """
        Predict target vector t for a sequence.
        Optimal t is the mean of all final layer outputs.
        Returns m-dimensional vector.
        """
        outputs = self.describe(seq)
        if not outputs:
            return [0.0] * self.model_dims[-1]
            
        t_pred = [0.0] * self.model_dims[-1]
        for vec in outputs:
            for i in range(len(vec)):
                t_pred[i] += vec[i]
        return [x / len(outputs) for x in t_pred]

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
            for row in layer['M']:
                print("      ", [f"{x:.4f}" for x in row])
            print("    P matrix:")
            for row in layer['P']:
                print("      ", [f"{x:.4f}" for x in row])

    def count_parameters(self):
        """Count learnable parameters (M and P matrices for all layers)."""
        total_params = 0
        print("Parameter Count:")
        
        for i, layer in enumerate(self.layers):
            M = layer['M']
            P = layer['P']
            M_params = len(M) * len(M[0])
            P_params = len(P) * len(P[0])
            layer_params = M_params + P_params
            total_params += layer_params
            
            print(f"  Layer {i}:")
            print(f"    M matrix: {len(M)}×{len(M[0])} = {M_params} parameters")
            print(f"    P matrix: {len(P)}×{len(P[0])} = {P_params} parameters")
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
    random.seed(12)
    
    # Configuration with hierarchical dimensions
    input_dim = 20      # Input vector dimension (n)
    model_dims = [15, 10, 5]  # Output dimensions for each layer
    num_seqs = 10
    min_len, max_len = 100, 200

    # Generate synthetic training data
    seqs = []
    t_list = []
    for _ in range(num_seqs):
        L = random.randint(min_len, max_len)
        # Generate n-dimensional vectors
        seq = [[random.uniform(-1,1) for _ in range(input_dim)] for __ in range(L)]
        seqs.append(seq)
        # Generate target vectors with dimension matching final layer
        t_list.append([random.uniform(-1,1) for _ in range(model_dims[-1])])
    
    # Train with Gradient Descent
    print("\nTraining HierarchicalDD with Gradient Descent:")
    hdd = HierarchicalDD(input_dim=input_dim, model_dims=model_dims)
    grad_history = hdd.grad_train(
        seqs, 
        t_list, 
        learning_rate=10.3,
        max_iters=50,
        decay_rate=0.98
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
