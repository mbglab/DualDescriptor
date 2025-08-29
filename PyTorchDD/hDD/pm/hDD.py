# Copyright (C) 2005-2025, Bin-Guang Ma (mbg@mail.hzau.edu.cn); SPDX-License-Identifier: MIT
# The Hierarchical Numeric Dual Descriptor (P Matrix form) implemented with PyTorch
# With layer normalization and residual connections and generation capability
# This program is for the demonstration of methodology and not fully refined.
# Author: Bin-Guang Ma (assisted by DeepSeek); Date: 2025-8-26

import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class Layer(nn.Module):
    """Single layer of Hierarchical Dual Descriptor (with 2D P matrix)"""
    def __init__(self, in_dim, out_dim, use_residual):
        """
        Initialize a hierarchical layer with simplified P matrix
        
        Args:
            in_dim (int): Input dimension
            out_dim (int): Output dimension
            use_residual (str or None): Residual connection type. Options are:
                - 'separate': use a separate linear projection for the residual connection.
                - 'shared': share the linear transformation matrix M for the residual.
                - None or other: no residual connection, unless in_dim==out_dim, then use identity residual.
        """
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.use_residual = use_residual
        
        # Linear transformation matrix (shared for main path and residual)
        self.M = nn.Parameter(torch.empty(out_dim, in_dim))
        
        # Simplified P matrix (out_dim, out_dim)
        self.P = nn.Parameter(torch.empty(out_dim, out_dim))
        
        # Precompute periods matrix (fixed, not trainable)
        periods = torch.zeros(out_dim, out_dim)
        for i in range(out_dim):
            for j in range(out_dim):
                # Simplified period calculation
                periods[i, j] = i * out_dim + j + 2
        self.register_buffer('periods', periods)

        # Residual projection processing
        if self.use_residual == 'separate':
            self.residual_proj = nn.Linear(in_dim, out_dim, bias=False) if in_dim != out_dim else nn.Identity()
        else:
            self.residual_proj = None
        
        # Layer normalization
        self.norm = nn.LayerNorm(out_dim)
        
        # Initialize parameters        
        nn.init.uniform_(self.M, -0.5, 0.5)
        nn.init.uniform_(self.P, -0.1, 0.1)
        if isinstance(self.residual_proj, nn.Linear):            
            nn.init.uniform_(self.residual_proj.weight, -0.1, 0.1)      
    
    def forward(self, x, positions):
        """
        Forward pass for the layer with simplified P matrix
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, in_dim)
            positions (Tensor): Position indices of shape (seq_len,)
        
        Returns:
            Tensor: Output of shape (batch_size, seq_len, out_dim)
        """
        batch_size, seq_len, _ = x.shape
        
        # Linear transformation: x' = x @ M^T
        x_trans = torch.matmul(x, self.M.t())  # (batch_size, seq_len, out_dim)

        # Layer normalization
        x_trans = self.norm(x_trans)
        
        # Prepare position-dependent transformation
        k = positions.view(-1, 1, 1).float()  # (seq_len, 1, 1)
        
        # Compute basis function: phi = cos(2π * k / period)
        # periods shape: (1, out_dim, out_dim)
        periods = self.periods.unsqueeze(0)  # add dimension for broadcasting
        phi = torch.cos(2 * math.pi * k / periods)  # (seq_len, out_dim, out_dim)
        
        # Compute position-dependent transformation matrix
        M_k = self.P.unsqueeze(0) * phi  # (seq_len, out_dim, out_dim)
        
        # Apply transformation using Einstein summation
        # 'bsj,sij->bsi': for each position s, multiply input vector with transformation matrix
        Nk = torch.einsum('bsj,sij->bsi', x_trans, M_k)
        
        # Residual connection processing 
        if self.use_residual == 'separate':
            residual = self.residual_proj(x)
            out = Nk + residual            
        elif self.use_residual == 'shared':        
            residual = torch.matmul(x, self.M.t())
            out = Nk + residual
        else:
            if self.in_dim == self.out_dim and self.use_residual != None:
                residual = x
                out = Nk + residual
            else:
                out = Nk
                
        return out

class HierDDpm(nn.Module):
    """Hierarchical Numeric Dual Descriptor with PyTorch and GPU support (Simplified)"""
    def __init__(self, input_dim=2, model_dims=[2], use_residual_list=None, device='cpu'):
        """
        Initialize hierarchical HierDDpm with simplified P matrix
        
        Args:
            input_dim (int): Input vector dimension
            model_dims (list): Output dimensions for each layer
            use_residual_list (list): List of residual connection types for each layer
            device (str): Device to use ('cpu' or 'cuda')
        """
        super().__init__()
        self.input_dim = input_dim
        self.model_dims = model_dims
        if use_residual_list == None:
            self.use_residual_list = ['separate'] * len(self.model_dims)
        else:
            self.use_residual_list = use_residual_list            
        self.num_layers = len(model_dims)
        self.device = device
        self.trained = False  # Track training status
        
        # Create hierarchical layers
        layers = []
        in_dim = input_dim
        for out_dim, use_residual in zip(self.model_dims, self.use_residual_list):
            layers.append(Layer(in_dim, out_dim, use_residual))
            in_dim = out_dim  # Next layer input is current layer output
        
        self.layers = nn.ModuleList(layers)
        self.to(device)
    
    def forward(self, seq):
        """
        Forward pass through all layers
        
        Args:
            seq (Tensor): Input sequence of shape (batch_size, seq_len, input_dim)
        
        Returns:
            Tensor: Output sequence of shape (batch_size, seq_len, model_dims[-1])
        """
        # Generate position indices
        batch_size, seq_len, _ = seq.shape
        positions = torch.arange(seq_len, device=self.device)
        
        x = seq
        for layer in self.layers:
            x = layer(x, positions)
        return x        
    
    def deviation(self, seqs, t_list):
        """
        Compute mean squared deviation D across sequences
        
        Args:
            seqs (list): List of input sequences
            t_list (list): List of target vectors
        
        Returns:
            float: Mean squared deviation
        """
        total_loss = 0.0
        count = 0
        
        for seq, t in zip(seqs, t_list):
            # Convert to tensors
            seq_tensor = torch.tensor(seq, dtype=torch.float32, device=self.device)
            t_tensor = torch.tensor(t, dtype=torch.float32, device=self.device)
            
            # Predict and compute loss
            pred = self.predict_t(seq_tensor.unsqueeze(0)).squeeze(0)
            loss = torch.sum((pred - t_tensor) ** 2)
            
            total_loss += loss.item()
            count += 1
        
        return total_loss / count if count else 0.0
    
    def train_model(self, seqs, t_list, max_iters=1000, tol=1e-88, lr=0.01, 
                    decay_rate=0.999, print_every=10):
        """
        Train model using Adam optimizer
        
        Args:
            seqs (list): List of training sequences
            t_list (list): List of target vectors
            max_iters (int): Maximum training iterations
            lr (float): Learning rate
            decay_rate (float): Learning rate decay rate
            print_every (int): Print progress every N iterations
        
        Returns:
            list: Training loss history
        """
        optimizer = optim.Adam(self.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)
        history = []
        prev_loss = float('inf')
        
        for it in range(max_iters):
            total_loss = 0.0
            count = 0
            
            for seq, t in zip(seqs, t_list):
                # Convert to tensors
                seq_tensor = torch.tensor(seq, dtype=torch.float32, device=self.device)
                t_tensor = torch.tensor(t, dtype=torch.float32, device=self.device)
                
                # Forward pass
                pred = self.predict_t(seq_tensor.unsqueeze(0)).squeeze(0)
                
                # Compute loss
                loss = torch.sum((pred - t_tensor) ** 2)
                total_loss += loss.item()
                count += 1
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()             
            
            # Record and print progress
            avg_loss = total_loss / count
            history.append(avg_loss)

            # Print progress
            if it % print_every == 0 or it == max_iters - 1:
                current_lr = scheduler.get_last_lr()[0]
                print(f"Iter {it:4d}: Loss = {avg_loss:.6e}, LR = {current_lr:.6f}")

            # Update learning rate
            scheduler.step()

            # Check convergence
            if abs(prev_loss - avg_loss) < tol:
                print(f"Converged after {it+1} iterations.")
                break
                
            prev_loss = avg_loss
        
        self.trained = True
        return history

    def predict_t(self, seq):
        """
        Predict target vector for a sequence
        Returns the average of all output vectors in the sequence
        
        Args:
            seq (Tensor): Input sequence of shape (batch_size, seq_len, input_dim)
        
        Returns:
            Tensor: Predicted target vector of shape (batch_size, model_dims[-1])
        """
        outputs = self.forward(seq)
        return outputs.mean(dim=1)  # Average over sequence length

    def auto_train(self, seqs, max_iters=100, tol=1e-16, learning_rate=0.01, 
                  continued=False, auto_mode='gap', decay_rate=1.0, print_every=10):
        """
        Self-training for the hierarchical model with two modes:
          - 'gap': Predicts current input vector (self-consistency)
          - 'reg': Predicts next input vector (auto-regressive) with causal masking          
        
        Args:
            seqs: List of input sequences
            max_iters: Maximum training iterations
            tol: Convergence tolerance
            learning_rate: Initial learning rate
            continued: Whether to continue from existing parameters
            auto_mode: Training mode ('gap' or 'reg')
            decay_rate: Learning rate decay rate
            print_every: Print interval
        
        Returns:
            Training loss history
        """
        if auto_mode not in ('gap', 'reg'):
            raise ValueError("auto_mode must be either 'gap' or 'reg'")
        
        # Convert sequences to tensors
        seqs_tensor = [torch.tensor(seq, dtype=torch.float32, device=self.device) for seq in seqs]
        
        # Initialize all layers if not continuing training
        if not continued:
            for layer in self.layers:
                # Reinitialize parameters with small random values
                nn.init.uniform_(layer.M, -0.5, 0.5)
                nn.init.uniform_(layer.P, -0.1, 0.1)
                
                # Reinitialize residual components if using 'separate' mode
                if layer.use_residual == 'separate' and hasattr(layer, 'residual_proj'):
                    if isinstance(layer.residual_proj, nn.Linear):
                        nn.init.uniform_(layer.residual_proj.weight, -0.1, 0.1)
            
            # Calculate global mean vector of input sequences
            total = torch.zeros(self.input_dim, device=self.device)
            total_vectors = 0
            for seq in seqs_tensor:
                total += torch.sum(seq, dim=0)
                total_vectors += seq.size(0)
            self.mean_t = (total / total_vectors).cpu().numpy()
        
        # Calculate total training samples
        total_samples = 0
        for seq in seqs_tensor:
            if auto_mode == 'gap':
                total_samples += seq.size(0)  # All positions are samples
            else:  # 'reg' mode
                total_samples += max(0, seq.size(0) - 1)  # All except last position
        
        if total_samples == 0:
            raise ValueError("No training samples found")
        
        # Setup optimizer
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)
        
        history = []
        prev_loss = float('inf')
        
        for it in range(max_iters):
            total_loss = 0.0
            
            for seq in seqs_tensor:
                optimizer.zero_grad()
                
                # Forward pass
                batch_size, seq_len, _ = seq.unsqueeze(0).shape
                positions = torch.arange(seq_len, device=self.device)
                
                current = seq.unsqueeze(0)  # Add batch dimension
                
                for layer in self.layers:
                    current = layer(current, positions)
                
                # Remove batch dimension
                current = current.squeeze(0)
                
                # Calculate loss based on auto_mode
                loss = 0.0
                valid_positions = 0
                
                for k in range(current.size(0)):
                    # Skip last position in 'reg' mode
                    if auto_mode == 'reg' and k == seq.size(0) - 1:
                        continue
                    
                    # Determine target based on mode
                    if auto_mode == 'gap':
                        target = seq[k]
                    else:  # 'reg' mode
                        target = seq[k + 1]
                    
                    # Calculate MSE loss
                    loss += torch.sum((current[k] - target) ** 2)
                    valid_positions += 1
                
                if valid_positions > 0:
                    loss = loss / valid_positions
                    total_loss += loss.item()
                    loss.backward()
                    optimizer.step()
            
            # Average loss
            avg_loss = total_loss / len(seqs_tensor)
            history.append(avg_loss)             
            
            # Print progress
            if it % print_every == 0 or it == max_iters - 1:
                current_lr = scheduler.get_last_lr()[0]
                mode_display = "Gap" if auto_mode == 'gap' else "Reg"
                print(f"AutoTrain({mode_display}) Iter {it:3d}: loss = {avg_loss:.6f}, LR = {current_lr:.6f}")

            # Update learning rate
            if it % 5 == 0:  # Decay every 5 iterations
                scheduler.step()
            
            # Check convergence
            if prev_loss - avg_loss < tol:
                print(f"Converged after {it+1} iterations")
                break
            prev_loss = avg_loss
        
        self.trained = True
        return history
    
    def generate(self, L, tau=0.0, discrete_mode=False, vocab_size=None):
        """
        Generate sequence of vectors with temperature-controlled randomness.
        Supports both continuous and discrete generation modes.
        
        Args:
            L (int): Number of vectors to generate
            tau (float): Temperature parameter
            discrete_mode (bool): If True, use discrete sampling
            vocab_size (int): Required for discrete mode
        
        Returns:
            Generated sequence as numpy array
        """
        assert self.trained and hasattr(self, 'mean_t'), "Model must be auto-trained first"
        if tau < 0:
            raise ValueError("Temperature must be non-negative")
        if discrete_mode and vocab_size is None:
            raise ValueError("vocab_size must be specified for discrete mode")

        # Set model to evaluation mode
        self.eval()
        
        generated = []
        # Create initial sequence with proper shape and values
        seq_len = 10  # Fixed sequence length for generation
        current_seq = torch.normal(
            mean=0.0,
            std=0.1,
            size=(seq_len, self.input_dim),
            device=self.device
        )
        # Add the learned mean vector to the initial sequence
        mean_tensor = torch.tensor(self.mean_t, device=self.device, dtype=torch.float32)
        current_seq = current_seq + mean_tensor.unsqueeze(0)  # Broadcast mean to all positions
        
        with torch.no_grad():
            for _ in range(L):
                # Forward pass
                batch_input = current_seq.unsqueeze(0)  # Add batch dimension
                positions = torch.arange(seq_len, device=self.device)
                
                current = batch_input
                for layer in self.layers:
                    current = layer(current, positions)
                
                # Remove batch dimension
                current = current.squeeze(0)
                
                # Get the last output vector
                output_vector = current[-1]
                
                if discrete_mode:
                    # Discrete generation mode
                    discrete_vector = []
                    for value in output_vector:
                        # Create logits and sample
                        logits = torch.full((vocab_size,), value.item())
                        if tau > 0:
                            logits += torch.normal(0, tau, size=(vocab_size,))
                        
                        if tau == 0:
                            sampled_index = torch.argmax(logits)
                        else:
                            probs = torch.softmax(logits / tau, dim=0)
                            sampled_index = torch.multinomial(probs, 1)
                        
                        discrete_vector.append(sampled_index.item())
                    
                    output_vector = torch.tensor(discrete_vector, device=self.device).float()
                else:
                    # Continuous generation mode
                    if tau > 0:
                        noise = torch.normal(0, tau * torch.abs(output_vector) + 0.01)
                        output_vector = output_vector + noise
                
                generated.append(output_vector.cpu().numpy())
                
                # Update current sequence (shift window)
                current_seq = torch.cat([current_seq[1:], output_vector.unsqueeze(0)])
        
        return np.array(generated)
    
    def save(self, filename):
        """Save model state to file"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': {
                'input_dim': self.input_dim,
                'model_dims': self.model_dims,
                'use_residual_list': self.use_residual_list
            },
            'mean_t': self.mean_t if hasattr(self, 'mean_t') else None,
            'trained': self.trained
        }, filename)
        print(f"Model saved to {filename}")
    
    @classmethod
    def load(cls, filename, device=None):
        """Load model from file"""
        # Device configuration
        device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        
        # Load checkpoint
        checkpoint = torch.load(filename, map_location=device, weights_only=False) 
        config = checkpoint['config']
        
        # Create model instance
        model = cls(
            input_dim=config['input_dim'],
            model_dims=config['model_dims'],
            use_residual_list=config.get('use_residual_list', None),
            device=device
        )
        
        # Load state
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load additional attributes
        if 'mean_t' in checkpoint:
            model.mean_t = checkpoint['mean_t']
        if 'trained' in checkpoint:
            model.trained = checkpoint['trained']
            
        print(f"Model loaded from {filename}")
        return model
    
    def count_parameters(self):
        """Count and print learnable parameters in the model by layer and parameter type"""
        total_params = 0
        layer_params = []
        
        # Print header
        print("="*60)
        print(f"{'Layer':<10} | {'Param Type':<20} | {'Count':<15} | {'Shape':<20}")
        print("-"*60)
        
        # Iterate through each layer
        for i, layer in enumerate(self.layers):
            layer_total = 0
            layer_name = f"Layer {i}"
            
            # Count parameters in this layer
            for name, param in layer.named_parameters():
                if param.requires_grad:
                    num = param.numel()
                    shape = str(tuple(param.shape))
                    
                    # Print parameter details
                    print(f"{layer_name:<10} | {name:<20} | {num:<15} | {shape:<20}")
                    
                    layer_total += num
                    total_params += num
            
            # Store layer summary
            layer_params.append((layer_name, layer_total))
        
        # Print summary by layer
        print("-"*60)
        for name, count in layer_params:
            print(f"{name} total parameters: {count}")
        
        # Print grand total
        print("="*60)
        print(f"Total trainable parameters: {total_params}")
        print("="*60)
        
        return total_params

# === Example Usage ===
if __name__ == "__main__":

    from scipy.stats import pearsonr
    
    # Set random seeds for reproducibility
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    
    # Configuration
    input_dim = 10          # Input vector dimension
    model_dims = [8, 6, 3]  # Output dimensions for each layer
    use_residual_list = ['separate'] * 3 
    num_seqs = 100          # Number of training sequences
    min_len, max_len = 100, 200  # Sequence length range
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print(f"Model config: input_dim={input_dim}, model_dims={model_dims}, use_residual={use_residual_list}")
    
    # Generate synthetic training data(random)
    seqs = []     # List of sequences (each sequence: list of n-dim vectors)
    t_list = []   # List of target vectors (m-dim)
    
    for _ in range(num_seqs):
        L = random.randint(min_len, max_len)
        # Generate n-dimensional input sequence
        seq = [[random.uniform(-1, 1) for _ in range(input_dim)] for __ in range(L)]
        seqs.append(seq)
        # Generate m-dimensional target vector
        t_list.append([random.uniform(-1, 1) for _ in range(model_dims[-1])])    
    
    # Create model with simplified P matrix
    print("\nCreating Hierarchical HierDDpm model (with simplified P matrix)...")
    model = HierDDpm(
        input_dim=input_dim,
        model_dims=model_dims,
        use_residual_list=use_residual_list,
        device=device
    )
    model.count_parameters()
    
    # Train model
    print("\nTraining model...")
    history = model.train_model(
        seqs, 
        t_list,
        max_iters=100,
        lr=0.1,
        decay_rate=0.97,
        print_every=10
    )
    
    # Predict targets
    print("\nPredictions (first 3 sequences):")
    predictions = []
    for i, seq in enumerate(seqs[:3]):
        seq_tensor = torch.tensor(seq, dtype=torch.float32, device=device)
        t_pred = model.predict_t(seq_tensor.unsqueeze(0)).squeeze(0).detach().cpu().numpy()
        
        print(f"Seq {i+1}:")
        print(f"  Target:    {[f'{x:.4f}' for x in t_list[i]]}")
        print(f"  Predicted: {[f'{x:.4f}' for x in t_pred]}")
        predictions.append(t_pred)
    
    # Calculate prediction correlations
    print("\nCalculating prediction correlations...")
    all_preds = []
    for seq in seqs:
        seq_tensor = torch.tensor(seq, dtype=torch.float32, device=device)
        t_pred = model.predict_t(seq_tensor.unsqueeze(0)).squeeze(0).detach().cpu().numpy()
        all_preds.append(t_pred)
    
    output_dim = model_dims[-1]
    correlations = []
    for i in range(output_dim):
        actual = [t[i] for t in t_list]
        predicted = [p[i] for p in all_preds]
        corr, p_value = pearsonr(actual, predicted)
        correlations.append(corr)
        print(f"Output dim {i} correlation: {corr:.4f} (p={p_value:.4e})")
    
    avg_corr = np.mean(correlations)
    print(f"\nAverage correlation: {avg_corr:.4f}")      
    
    # Save and load model
    print("Testing save/load functionality...")
    # Save model
    model.save("test_model.pth")
    
    # Load model
    loaded_model = HierDDpm.load("test_model.pth", device=device)
    print(f"Model loaded successfully. Trained: {loaded_model.trained}")
    
    # Verify loaded model works
    test_seq = seqs[0]
    test_tensor = torch.tensor(test_seq, dtype=torch.float32, device=device)
    original_pred = model.predict_t(test_tensor.unsqueeze(0)).squeeze(0)
    loaded_pred = loaded_model.predict_t(test_tensor.unsqueeze(0)).squeeze(0)
    
    diff = torch.sum((original_pred - loaded_pred) ** 2).item()
    print(f"Prediction difference between original and loaded model: {diff:.6e}")

    # Example of using the new auto_train and generate methods
    print("\n" + "="*50)
    print("Example of using auto_train and generate methods")
    print("="*50)

    # Create a new model for auto-training
    auto_model = HierDDpm(
        input_dim=input_dim,
        model_dims=[10, 20, 10],
        use_residual_list=use_residual_list,
        device=device
    )
    
    # Auto-train the model in 'gap' mode
    print("\nAuto-training model in 'gap' mode...")
    auto_history = auto_model.auto_train(
        seqs[:10],
        max_iters=20,
        learning_rate=0.01,
        auto_mode='gap',
        print_every=5
    )
    
    # Generate new sequences
    print("\nGenerating new sequences...")
    generated_seq = auto_model.generate(L=5, tau=0.1)
    print(f"Generated sequence shape: {generated_seq.shape}")
    print("First 5 generated vectors:")
    for i, vec in enumerate(generated_seq):
        print(f"Vector {i+1}: {[f'{x:.4f}' for x in vec]}")
    
    # Save and load model
    print("\nSaving and loading model...")
    auto_model.save("auto_model.pth")
    loaded_model = HierDDpm.load("auto_model.pth", device=device)
    
    # Generate with loaded model
    loaded_generated = loaded_model.generate(L=3, tau=0.05)
    print("Generated with loaded model:")
    for i, vec in enumerate(loaded_generated):
        print(f"Vector {i+1}: {[f'{x:.4f}' for x in vec]}")    
    
    print("Training and testing completed!")
