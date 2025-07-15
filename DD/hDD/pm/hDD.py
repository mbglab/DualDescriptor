# Copyright (C) Bin-Guang Ma (mbg@mail.hzau.edu.cn). All rights reserved.
# The Numeric Dual Descriptor class (Tensor form) with hierarchical structure
# Author: Bin-Guang Ma; Date: 2025-6-4

import math
import random
import pickle

class HierDDpm:
    """
    Hierarchical Numeric Dual Descriptor for vector sequences with:
      - Multiple layers with linear transformation M and tensor P
      - Each layer: P ∈ R^{m×m×o} of basis coefficients, M ∈ R^{m×n}
      - Periods: period[i,j,g] = i*(m*o) + j*o + g + 2
      - Basis function: phi_{i,j,g}(k) = cos(2π * k / period[i,j,g])
    """
    def __init__(self, input_dim=2, model_dims=[2], num_basis_list=[5]):
        """
        Initialize hierarchical HierDDpm
        
        Args:
            input_dim (int): Input vector dimension
            model_dims (list): Output dimensions for each layer
            num_basis_list (list): Number of basis functions for each layer
        """
        self.input_dim = input_dim
        self.model_dims = model_dims
        self.num_basis_list = num_basis_list
        self.num_layers = len(model_dims)
        self.trained = False
        
        # Initialize layers
        self.layers = []
        for l, out_dim in enumerate(model_dims):
            # Determine input dimension for this layer
            if l == 0:
                in_dim = input_dim
            else:
                in_dim = model_dims[l-1]
                
            # Initialize M matrix (out_dim x in_dim)
            M = [[random.uniform(-0.5, 0.5) for _ in range(in_dim)] 
                 for _ in range(out_dim)]
            
            # Initialize P tensor (out_dim x out_dim x num_basis)
            P = [[[random.uniform(-0.1, 0.1) for _ in range(num_basis_list[l])]
                  for _ in range(out_dim)]
                 for _ in range(out_dim)]
            
            # Precompute periods tensor
            periods = [[[ i*(out_dim*num_basis_list[l]) + j*num_basis_list[l] + g + 2
                         for g in range(num_basis_list[l])]
                        for j in range(out_dim)]
                       for i in range(out_dim)]
            
            self.layers.append({
                'M': M,
                'P': P,
                'periods': periods
            })

    # ---- Helper functions ----
    def _mat_vec(self, M, v):
        """Matrix-vector multiplication"""
        return [sum(M[i][j]*v[j] for j in range(len(v))) for i in range(len(M))]
    
    def _transpose(self, M):
        """Matrix transpose"""
        return [list(col) for col in zip(*M)]
    
    def _vec_sub(self, u, v):
        """Vector subtraction"""
        return [u_i - v_i for u_i, v_i in zip(u, v)]
    
    def _dot(self, u, v):
        """Vector dot product"""
        return sum(u_i * v_i for u_i, v_i in zip(u, v))

    def describe(self, seq):
        """Compute N(k) vectors for each position and vector in sequence"""
        current = seq
        for layer in self.layers:
            new_seq = []
            for k, vec in enumerate(current):
                # Linear transformation: x = M * vec
                x = self._mat_vec(layer['M'], vec)
                out_dim = len(layer['P'])
                Nk = [0.0] * out_dim
                
                # Apply position-dependent transformation
                for i in range(out_dim):
                    for j in range(out_dim):
                        for g in range(len(layer['P'][i][j])):
                            period = layer['periods'][i][j][g]
                            phi = math.cos(2 * math.pi * k / period)
                            Nk[i] += layer['P'][i][j][g] * x[j] * phi
                new_seq.append(Nk)
            current = new_seq
        return current

    def deviation(self, seqs, t_list):
        """
        Compute mean squared deviation D across sequences:
        D = average over all positions of (N(k)-t_seq)^2
        """
        total = 0.0
        count = 0
        for seq, t in zip(seqs, t_list):
            outputs = self.describe(seq)
            for vec in outputs:
                err = self._vec_sub(vec, t)
                total += self._dot(err, err)
                count += 1
        return total / count if count else 0.0

    def grad_train(self, seqs, t_list, max_iters=1000, tol=1e-8, 
                  learning_rate=0.01, continued=False, decay_rate=1.0, 
                  print_every=10):
        """
        Train using gradient descent with backpropagation through layers
        """
        if not continued:
            # Reinitialize with small random values
            self.__init__(self.input_dim, self.model_dims, self.num_basis_list)
            
        history = []
        D_prev = float('inf')
        total_positions = sum(len(seq) for seq in seqs)
        
        for it in range(max_iters):
            # Initialize gradients for all layers
            grad_P_list = []
            grad_M_list = []
            for l in range(self.num_layers):
                out_dim = self.model_dims[l]
                num_basis = self.num_basis_list[l]
                
                # Gradients for P tensor
                grad_P = [[[0.0] * num_basis for _ in range(out_dim)] 
                          for _ in range(out_dim)]
                grad_P_list.append(grad_P)
                
                # Gradients for M matrix
                if l == 0:
                    in_dim = self.input_dim
                else:
                    in_dim = self.model_dims[l-1]
                grad_M = [[0.0] * in_dim for _ in range(out_dim)]
                grad_M_list.append(grad_M)
            
            # Forward pass: store intermediate results
            for seq, t_vec in zip(seqs, t_list):
                intermediates = []  # Store layer intermediates for backprop
                current = seq
                
                # Forward pass through layers
                for l, layer in enumerate(self.layers):
                    layer_intermediate = []
                    next_seq = []
                    
                    for k, vec in enumerate(current):
                        # Apply linear transformation
                        x = self._mat_vec(layer['M'], vec)
                        out_dim = len(layer['P'])
                        Nk = [0.0] * out_dim
                        phi_vals = {}
                        
                        # Compute output and store basis values
                        for i in range(out_dim):
                            for j in range(out_dim):
                                for g in range(len(layer['P'][i][j])):
                                    period = layer['periods'][i][j][g]
                                    phi = math.cos(2 * math.pi * k / period)
                                    phi_vals[(i, j, g)] = phi
                                    Nk[i] += layer['P'][i][j][g] * x[j] * phi
                        
                        layer_intermediate.append({
                            'input_vec': vec,
                            'x': x,
                            'phi_vals': phi_vals,
                            'output_vec': Nk
                        })
                        next_seq.append(Nk)
                    
                    intermediates.append(layer_intermediate)
                    current = next_seq
                
                # Backward pass through layers
                for k in range(len(seq)):
                    # Start with last layer
                    d_out = None
                    for l in range(self.num_layers-1, -1, -1):
                        layer = self.layers[l]
                        interm = intermediates[l][k]
                        x = interm['x']
                        phi_vals = interm['phi_vals']
                        input_vec = interm['input_vec']
                        output_vec = interm['output_vec']
                        
                        # Initialize error for this layer
                        if l == self.num_layers-1:
                            # Final layer error: 2*(output - target)
                            d_out = [2 * (output_vec[i] - t_vec[i]) 
                                     for i in range(len(output_vec))]
                        else:
                            # Propagate error from next layer
                            d_out = d_next
                        
                        # Compute gradient for P tensor
                        for i in range(len(d_out)):
                            for j in range(len(layer['P'][i])):
                                for g in range(len(layer['P'][i][j])):
                                    grad_P_list[l][i][j][g] += (
                                        d_out[i] * x[j] * phi_vals[(i, j, g)]
                                    ) / total_positions
                        
                        # Compute gradient for M matrix
                        d_x = [0.0] * len(x)
                        for j in range(len(x)):
                            for i in range(len(d_out)):
                                for g in range(len(layer['P'][i][j])):
                                    term = d_out[i] * layer['P'][i][j][g] * phi_vals[(i, j, g)]
                                    d_x[j] += term
                            
                            # Propagate to M gradients
                            for d in range(len(input_vec)):
                                grad_M_list[l][j][d] += d_x[j] * input_vec[d] / total_positions
                        
                        # Propagate error to previous layer
                        if l > 0:
                            # d_input = M^T * d_x
                            M_T = self._transpose(layer['M'])
                            d_next = self._mat_vec(M_T, d_x)
                        else:
                            d_next = None
            
            # Update parameters
            for l in range(self.num_layers):
                layer = self.layers[l]
                # Update P tensor
                for i in range(len(layer['P'])):
                    for j in range(len(layer['P'][i])):
                        for g in range(len(layer['P'][i][j])):
                            layer['P'][i][j][g] -= learning_rate * grad_P_list[l][i][j][g]
                
                # Update M matrix
                for i in range(len(layer['M'])):
                    for j in range(len(layer['M'][i])):
                        layer['M'][i][j] -= learning_rate * grad_M_list[l][i][j]
            
            # Calculate current loss
            current_D = self.deviation(seqs, t_list)
            history.append(current_D)
            
            # Print progress
            if it % print_every == 0 or it == max_iters - 1:
                print(f"GD Iter {it:3d}: D = {current_D:.6e}, LR = {learning_rate:.6f}")
            
            # Check convergence
            if current_D >= D_prev - tol:
                print(f"Converged after {it+1} iterations.")
                break
            D_prev = current_D
            
            # Apply learning rate decay
            learning_rate *= decay_rate
        
        self.trained = True
        return history

    def predict_t(self, seq):
        """
        Predict target vector for a sequence
        Returns the average of all N(k) vectors in the sequence
        """
        outputs = self.describe(seq)
        if not outputs:
            return [0.0] * self.model_dims[-1]
        
        # Average all output vectors
        t_pred = [0.0] * self.model_dims[-1]
        for vec in outputs:
            for i in range(len(vec)):
                t_pred[i] += vec[i]
        return [x / len(outputs) for x in t_pred]

    def show(self, what=None, first_num=5):
        """
        Display model status with hierarchical support
        """
        # Default attributes to show
        default_attrs = ['config', 'M', 'P', 'periods']
        
        # Handle different what parameter types
        if what is None:
            attrs = default_attrs
        elif what == 'all':
            attrs = default_attrs
        elif isinstance(what, str):
            attrs = [what]
        else:
            attrs = what
            
        print("Hierarchical HierDDpm Status")
        print("=" * 50)
        
        # Display each requested attribute
        for attr in attrs:
            if attr == 'config':
                print("\n[Configuration]")
                print(f"{'Input dim:':<20} {self.input_dim}")
                print(f"{'Layer dims:':<20} {self.model_dims}")
                print(f"{'Num basis:':<20} {self.num_basis_list}")
                print(f"{'Trained:':<20} {self.trained}")
            
            elif attr in ['M', 'P', 'periods']:
                print(f"\n[{attr.upper()} Matrices/Tensors]")
                for l in range(self.num_layers):
                    print(f"  Layer {l}:")
                    if attr == 'M':
                        M = self.layers[l]['M']
                        print(f"    Shape: {len(M)}x{len(M[0])}")
                        print("    Sample values:")
                        for i in range(min(first_num, len(M))):
                            vals = M[i][:min(first_num, len(M[i]))]
                            print(f"      Row {i}: [{', '.join(f'{x:.4f}' for x in vals)}" + 
                                  (f", ...]" if len(M[i]) > first_num else "]"))
                    
                    elif attr in ['P', 'periods']:
                        tensor = self.layers[l][attr]
                        print(f"    Shape: {len(tensor)}x{len(tensor[0])}x{len(tensor[0][0])}")
                        print("    Sample slices:")
                        for i in range(min(first_num, len(tensor))):
                            for j in range(min(first_num, len(tensor[i]))):
                                vals = tensor[i][j][:min(first_num, len(tensor[i][j]))]
                                if attr == 'P':
                                    formatted = [f"{v:.6f}" for v in vals]
                                else:
                                    formatted = [str(int(v)) for v in vals]
                                print(f"      {attr}[{i}][{j}][:]: [{', '.join(formatted)}" + 
                                      (f", ...]" if len(tensor[i][j]) > first_num else "]"))
            
            else:
                print(f"\n[Unknown attribute: {attr}]")
        
        print("=" * 50)

    def count_parameters(self):
        """Count learnable parameters (P tensors and M matrices for all layers)"""
        total_params = 0
        print("Parameter Count:")
        
        for l in range(self.num_layers):
            M = self.layers[l]['M']
            P = self.layers[l]['P']
            M_params = len(M) * len(M[0])
            P_params = len(P) * len(P[0]) * len(P[0][0])
            layer_params = M_params + P_params
            total_params += layer_params
            
            print(f"  Layer {l}:")
            print(f"    M matrix: {len(M)}×{len(M[0])} = {M_params} parameters")
            print(f"    P tensor: {len(P)}×{len(P[0])}×{len(P[0][0])} = {P_params} parameters")
            print(f"    Layer total: {layer_params}")
        
        print(f"Total parameters: {total_params}")
        return total_params

    def save(self, filename):
        """Save model state to file"""
        with open(filename, 'wb') as f:
            pickle.dump(self.__dict__, f)
        print(f"Model saved to {filename}")

    @classmethod
    def load(cls, filename):
        """Load model state from file"""
        with open(filename, 'rb') as f:
            state = pickle.load(f)
        
        # Create new instance without calling __init__
        obj = cls.__new__(cls)
        # Restore saved state
        obj.__dict__.update(state)
        print(f"Model loaded from {filename}")
        return obj


# === Example Usage ===
if __name__=="__main__":

    from statistics import correlation, mean
    
    #random.seed(3)
    
    # Hierarchical configuration
    input_dim = 10      # Input vector dimension
    model_dims = [8, 6, 3]  # Output dimensions for each layer
    num_basis_list = [5, 4, 3]  # Basis functions per layer
    num_seqs = 10
    min_len, max_len = 100, 200

    # Generate synthetic training data
    seqs = []     # List of sequences (each sequence: list of n-dim vectors)
    t_list = []   # List of target vectors (m-dim)
    
    for _ in range(num_seqs):
        L = random.randint(min_len, max_len)
        # Generate n-dimensional input sequence
        seq = [[random.uniform(-1,1) for _ in range(input_dim)] for __ in range(L)]
        seqs.append(seq)
        # Generate m-dimensional target vector
        t_list.append([random.uniform(-1,1) for _ in range(model_dims[-1])])

    # Create and train hierarchical model
    print("\nTraining Hierarchical HierDDpm...")
    hndd = HierDDpm(
        input_dim=input_dim,
        model_dims=model_dims,
        num_basis_list=num_basis_list
    )
    
    # Gradient Descent Training
    grad_history = hndd.grad_train(
        seqs, 
        t_list, 
        max_iters=50,
        learning_rate=5.0,
        decay_rate=0.9999,
        print_every=5
    )

    # Predict targets
    print("\nPredictions (first 3 sequences):")
    for i, seq in enumerate(seqs[:3]):
        t_pred = hndd.predict_t(seq)
        print(f"Seq {i+1}: Target={[f'{x:.4f}' for x in t_list[i]]}")
        print(f"         Predicted={[f'{x:.4f}' for x in t_pred]}")

    # Calculate prediction correlations
    preds = [hndd.predict_t(seq) for seq in seqs]
    correlations = []
    for i in range(model_dims[-1]):
        actual = [t[i] for t in t_list]
        predicted = [p[i] for p in preds]
        corr = correlation(actual, predicted)
        correlations.append(corr)
        print(f"Output dim {i} correlation: {corr:.4f}")
    print(f"Average correlation: {mean(correlations):.4f}")

    # Show model status
    print("\nModel status:")
    hndd.show(['config', 'M'])

    # Count learnable parameters
    print("\nParameter count:")
    hndd.count_parameters()

    # Save and load model
    print("\nTesting save/load functionality:")
    hndd.save("hierarchical_ndd_model.pkl")
    loaded = HierDDpm.load("hierarchical_ndd_model.pkl")
    
    # Verify loaded model
    t_pred_loaded = loaded.predict_t(seqs[0])
    print(f"Prediction from loaded model: {[f'{x:.4f}' for x in t_pred_loaded]}")
