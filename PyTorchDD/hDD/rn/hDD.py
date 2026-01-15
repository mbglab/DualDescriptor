# Copyright (C) 2005-2025, Bin-Guang Ma (mbg@mail.hzau.edu.cn); SPDX-License-Identifier: MIT
# The Hierarchical Numeric Dual Descriptor (Random AB matrix form) implemented with PyTorch
# With layer normalization and residual connections and generation capability
# Added batch processing, checkpointing, and improved parameter counting
# This program is for the demonstration of methodology and not fully refined.
# Author: Bin-Guang Ma (assisted by DeepSeek); Date: 2025-7-22 ~ 2026-1-15

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import os

class Layer(nn.Module):
    """
    Single layer of the Hierarchical Numeric Dual Descriptor with Residual Connections.
    
    Args:
        in_dim (int): Input dimension
        out_dim (int): Output dimension
        basis_dim (int): Basis dimension for this layer
        use_residual (str or None): Residual connection type. Options are:
            - 'separate': use a separate linear projection for the residual connection.
            - 'shared': share the linear transformation for the residual.
            - None or other: no residual connection, unless in_dim==out_dim, then use identity residual.
        device (str): Device to run computations on ('cuda' or 'cpu')
    """
    def __init__(self, in_dim, out_dim, basis_dim, use_residual='separate', device='cuda'):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.basis_dim = basis_dim
        self.use_residual = use_residual
        self.device = device
        
        # Linear transformation from input to output dimension
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        
        # Layer normalization for output dimension
        self.norm = nn.LayerNorm(out_dim)
        
        # Coefficient matrix: out_dim x basis_dim
        self.Acoeff = nn.Parameter(torch.Tensor(out_dim, basis_dim))
        
        # Basis matrix: basis_dim x out_dim
        self.Bbasis = nn.Parameter(torch.Tensor(basis_dim, out_dim))
        
        # Initialize parameters
        nn.init.uniform_(self.linear.weight, -0.1, 0.1)
        nn.init.uniform_(self.Acoeff, -0.1, 0.1)
        nn.init.uniform_(self.Bbasis, -0.1, 0.1)
        
        # Residual projection processing
        if self.use_residual == 'separate' and in_dim != out_dim:
            self.residual_proj = nn.Linear(in_dim, out_dim, bias=False)
            nn.init.uniform_(self.residual_proj.weight, -0.1, 0.1)
        else:
            self.residual_proj = None
            
        self.to(device)
    
    def forward(self, x):
        """
        Forward pass through the layer with residual connections
        Supports batch processing with shape (batch_size, seq_len, in_dim)
        
        Args:
            x (torch.Tensor): Input sequence of shape (batch_size, seq_len, in_dim)
        
        Returns:
            torch.Tensor: Output sequence of shape (batch_size, seq_len, out_dim)
        """
        batch_size, seq_len, _ = x.shape
        
        # Save input for residual connection
        residual = x
        
        # Apply linear transformation
        transformed = self.linear(x)
        
        # Apply layer normalization
        normalized = self.norm(transformed)
        
        # Compute basis indices: j = k % basis_dim for each position
        j_indices = torch.arange(seq_len) % self.basis_dim
        j_indices = j_indices.to(self.device)
        
        # Select basis vectors: (seq_len, out_dim)
        B_vectors = self.Bbasis[j_indices]  # (seq_len, out_dim)
        
        # Expand for batch processing
        B_vectors_exp = B_vectors.unsqueeze(0).expand(batch_size, -1, -1)  # (batch_size, seq_len, out_dim)
        
        # Compute scalars: dot product of normalized and B_vectors
        scalars = torch.einsum('bsd,bsd->bs', normalized, B_vectors_exp)  # (batch_size, seq_len)
        
        # Select coefficient vectors: (seq_len, out_dim)
        A_vectors = self.Acoeff[:, j_indices].t()  # (seq_len, out_dim)
        A_vectors_exp = A_vectors.unsqueeze(0).expand(batch_size, -1, -1)  # (batch_size, seq_len, out_dim)
        
        # Compute new features: scalars * A_vectors
        new_features = scalars.unsqueeze(2) * A_vectors_exp  # (batch_size, seq_len, out_dim)
        
        # Residual connection processing 
        if self.use_residual == 'separate':
            if self.residual_proj is not None:  # Only project if dimensions differ
                residual = self.residual_proj(residual)
            out = new_features + residual
        elif self.use_residual == 'shared':        
            # Use same linear transformation as main path
            residual = self.linear(residual)
            out = new_features + residual
        else:
            if self.in_dim == self.out_dim and self.use_residual != None:
                out = new_features + residual  # Identity residual
            else:
                out = new_features  # No residual
                
        return out

class HierDDrn(nn.Module):
    """
    Hierarchical Numeric Dual Descriptor with Residual Connections and Layer Normalization.
    Supports varying dimensions across layers while keeping sequence length unchanged.
    Optimized for batch processing.
    
    Args:
        input_dim (int): Dimension of input vectors
        model_dims (list): List of output dimensions for each layer
        basis_dims (list): List of basis dimensions for each layer
        use_residual_list (list): Residual connection type for each layer
        device (str): Device to run computations on ('cuda' or 'cpu')
    """
    def __init__(self, input_dim=4, model_dims=[4], basis_dims=[50], 
                 use_residual_list=None, device='cuda'):
        super().__init__()
        self.input_dim = input_dim
        self.model_dims = model_dims
        self.basis_dims = basis_dims
        self.num_layers = len(model_dims)
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.trained = False
        
        # Validate dimensions
        if len(basis_dims) != self.num_layers:
            raise ValueError("basis_dims length must match model_dims length")
        
        # Set default residual connection types
        if use_residual_list is None:
            self.use_residual_list = ['separate'] * self.num_layers
        else:
            self.use_residual_list = use_residual_list
            
        # Create layers
        self.layers = nn.ModuleList()
        in_dim = input_dim
        for i, out_dim in enumerate(model_dims):
            layer = Layer(
                in_dim=in_dim,
                out_dim=out_dim,
                basis_dim=basis_dims[i],
                use_residual=self.use_residual_list[i],
                device=self.device
            )
            self.layers.append(layer)
            in_dim = out_dim  # Next layer's input is this layer's output
        
        # Mean target vector for reconstruction
        self.mean_target = None
        
        # Training statistics
        self.mean_t = None
        
        self.to(self.device)
    
    def forward(self, x):
        """
        Forward pass through the entire hierarchical model
        
        Args:
            x (torch.Tensor): Input sequence of shape (batch_size, seq_len, input_dim)
        
        Returns:
            torch.Tensor: Output sequence of shape (batch_size, seq_len, model_dims[-1])
        """
        current = x
        for layer in self.layers:
            current = layer(current)
        return current
    
    def describe(self, vec_seq):
        """
        Compute descriptor vectors for each position in the sequence
        
        Args:
            vec_seq (list): List of input vectors (each vector is list of floats)
        
        Returns:
            list: List of output vectors from the final layer
        """
        # Convert to tensor and move to device
        if isinstance(vec_seq, list):
            vec_seq = torch.tensor(vec_seq, dtype=torch.float32).to(self.device)
        
        # Add batch dimension if needed
        if vec_seq.dim() == 2:
            vec_seq = vec_seq.unsqueeze(0)
        
        # Forward pass
        output_tensor = self.forward(vec_seq)
        
        # Convert back to list of vectors
        return output_tensor.cpu().detach().numpy().tolist()
    
    def deviation(self, seqs, t_list):
        """
        Compute mean squared error between descriptors and targets
        
        Args:
            seqs (list): List of input sequences
            t_list (list): List of target vectors
        
        Returns:
            float: Mean squared error
        """
        total_loss = 0.0
        total_positions = 0
        
        for seq, t in zip(seqs, t_list):
            # Convert to tensors
            input_tensor = torch.tensor(seq, dtype=torch.float32).to(self.device)
            target_tensor = torch.tensor(t, dtype=torch.float32).to(self.device)
            
            # Forward pass
            outputs = self.forward(input_tensor.unsqueeze(0)).squeeze(0)
            
            # Compute MSE for each position
            for pos in range(len(seq)):
                loss = torch.mean((outputs[pos] - target_tensor) ** 2)
                total_loss += loss.item()
                total_positions += 1
        
        return total_loss / total_positions if total_positions > 0 else 0.0

    def reg_train(self, seqs, t_list, max_iters=1000, tol=1e-8,
                    learning_rate=0.01, decay_rate=1.0, print_every=10,
                    continued=False, batch_size=256, checkpoint_file=None, 
                    checkpoint_interval=10):
        """
        Train the model using Adam optimizer with learning rate decay and early stopping
        Uses batch processing for faster training
        
        Args:
            seqs (list): List of training sequences
            t_list (list): List of target vectors (must match final layer dimension)
            max_iters (int): Maximum training iterations
            tol (float): Convergence tolerance
            learning_rate (float): Initial learning rate
            decay_rate (float): Learning rate decay factor
            print_every (int): Print progress every N iterations
            continued (bool): Whether to continue from existing parameters
            batch_size (int): Batch size for training
            checkpoint_file (str): Path to save/load checkpoint file for resuming training
            checkpoint_interval (int): Interval (in iterations) for saving checkpoints
        
        Returns:
            list: Training loss history
        """
        # Load checkpoint if continuing and checkpoint file exists
        start_iter = 0
        best_loss = float('inf')
        best_model_state = None
        
        if continued and checkpoint_file and os.path.exists(checkpoint_file):
            checkpoint = torch.load(checkpoint_file, map_location=self.device, weights_only=False)
            self.load_state_dict(checkpoint['model_state_dict'])
            optimizer_state = checkpoint['optimizer_state_dict']
            scheduler_state = checkpoint['scheduler_state_dict']
            history = checkpoint['history']
            start_iter = checkpoint['iteration'] + 1
            best_loss = checkpoint.get('best_loss', float('inf'))
            print(f"Resumed training from checkpoint at iteration {start_iter}, best loss: {best_loss:.6e}")
        else:
            if not continued:
                self.reset_parameters() 
            history = []
        
        # Convert targets to tensors
        target_tensors = torch.tensor(t_list, dtype=torch.float32, device='cpu')
        
        # Pre-process data: convert sequences to tensors on CPU
        seqs_tensors = [torch.tensor(seq, dtype=torch.float32) for seq in seqs]
        max_len = max(len(seq) for seq in seqs)
        
        # Pad sequences to same length for batch processing
        padded_seqs = []
        for seq_tensor in seqs_tensors:
            if len(seq_tensor) < max_len:
                padding = torch.zeros(max_len - len(seq_tensor), self.input_dim)
                padded = torch.cat([seq_tensor, padding], dim=0)
            else:
                padded = seq_tensor
            padded_seqs.append(padded)
        
        all_seqs_tensor = torch.stack(padded_seqs)  # [num_seqs, max_len, input_dim]
        
        # Set up optimizer
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)
        
        # Load optimizer and scheduler states if resuming
        if continued and checkpoint_file and os.path.exists(checkpoint_file):
            optimizer.load_state_dict(optimizer_state)
            scheduler.load_state_dict(scheduler_state)
        
        prev_loss = float('inf') if start_iter == 0 else history[-1] if history else float('inf')
        
        for it in range(start_iter, max_iters):
            total_loss = 0.0
            num_samples = len(seqs)
            
            # Shuffle indices
            perm = torch.randperm(num_samples)
            
            for i in range(0, num_samples, batch_size):
                batch_idx = perm[i:i+batch_size]
                batch_seqs = all_seqs_tensor[batch_idx].to(self.device)
                batch_targets = target_tensors[batch_idx].to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass for the whole batch
                outputs = self.forward(batch_seqs)  # [batch_size, max_len, output_dim]
                
                # Compute loss (mean across positions and dimensions)
                # Average over sequence dimension
                pred_targets = outputs.mean(dim=1)  # [batch_size, output_dim]
                
                # Compute MSE loss
                loss = torch.mean((pred_targets - batch_targets) ** 2)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item() * len(batch_idx)
            
            # Calculate average loss
            avg_loss = total_loss / num_samples
            history.append(avg_loss)
            
            # Update best model state if current loss is lower
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_model_state = copy.deepcopy(self.state_dict())
            
            # Print progress
            if it % print_every == 0 or it == max_iters - 1:
                lr = optimizer.param_groups[0]['lr']
                print(f"Iter {it:4d}: Loss = {avg_loss:.8f}, LR = {lr:.6f}")

            # Update learning rate
            scheduler.step()
            
            # Save checkpoint at specified intervals
            if checkpoint_file and (it % checkpoint_interval == 0 or it == max_iters - 1):
                self._save_checkpoint(
                    checkpoint_file, it, history, optimizer, scheduler, best_loss
                )
            
            # Check for improvement
            if abs(prev_loss - avg_loss) < tol:
                print(f"Converged after {it} iterations")
                # Restore the best model state before breaking
                if best_model_state is not None and avg_loss > best_loss:
                    self.load_state_dict(best_model_state)                    
                    print(f"Restored best model state with loss = {best_loss:.8f}")
                    history[-1] = best_loss
                break
            prev_loss = avg_loss
        
        # If training ended without convergence, restore best model state
        if best_model_state is not None and best_loss < history[-1]:
            self.load_state_dict(best_model_state)  
            print(f"Training ended. Restored best model state with loss = {best_loss:.8f}")
            history[-1] = best_loss
        
        # Store training statistics and compute mean target
        self._compute_training_statistics(seqs)
        self._compute_mean_target(seqs)
        self.trained = True
        
        return history
    
    def _save_checkpoint(self, checkpoint_file, iteration, history, optimizer, scheduler, best_loss):
        """
        Save training checkpoint with complete training state
        
        Args:
            checkpoint_file: Path to save checkpoint file
            iteration: Current training iteration
            history: Training loss history
            optimizer: Optimizer instance
            scheduler: Learning rate scheduler instance
            best_loss: Best loss achieved so far
        """
        # Convert mean_target to numpy for saving
        mean_target_np = self.mean_target.cpu().numpy() if self.mean_target is not None else None
        
        checkpoint = {
            'iteration': iteration,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'history': history,
            'best_loss': best_loss,
            'config': {
                'input_dim': self.input_dim,
                'model_dims': self.model_dims,
                'basis_dims': self.basis_dims,
                'use_residual_list': self.use_residual_list,
            },
            'training_stats': {
                'mean_t': self.mean_t,
                'mean_target': mean_target_np
            }
        }
        torch.save(checkpoint, checkpoint_file)
        #print(f"Checkpoint saved at iteration {iteration} to {checkpoint_file}")

    def predict_t(self, vec_seq):
        """
        Predict target vector as average of all final layer outputs
        
        Args:
            vec_seq (list): Input sequence of vectors
        
        Returns:
            list: Predicted target vector
        """
        with torch.no_grad():
            if isinstance(vec_seq, list):
                vec_seq = torch.tensor(vec_seq, dtype=torch.float32).to(self.device)
            
            if vec_seq.dim() == 2:
                vec_seq = vec_seq.unsqueeze(0)
            
            outputs = self.forward(vec_seq)
            pred = outputs.mean(dim=1).squeeze(0)
            
            return pred.cpu().numpy().tolist()
    
    def self_train(self, seqs, max_iters=100, tol=1e-6, learning_rate=0.01, 
                  continued=False, self_mode='gap', decay_rate=1.0, print_every=10,
                  batch_size=256, checkpoint_file=None, checkpoint_interval=5):
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
            self_mode: Training mode ('gap' or 'reg')
            decay_rate: Learning rate decay rate
            print_every: Print interval
            batch_size: Batch size for training
            checkpoint_file: Path to save/load checkpoint file for resuming training
            checkpoint_interval: Interval (in iterations) for saving checkpoints
        
        Returns:
            Training loss history
        """
        if self_mode not in ('gap', 'reg'):
            raise ValueError("self_mode must be either 'gap' or 'reg'")
        
        # Load checkpoint if continuing and checkpoint file exists
        start_iter = 0
        best_loss = float('inf')
        best_model_state = None
        
        if continued and checkpoint_file and os.path.exists(checkpoint_file):
            checkpoint = torch.load(checkpoint_file, map_location=self.device, weights_only=False)
            self.load_state_dict(checkpoint['model_state_dict'])
            optimizer_state = checkpoint['optimizer_state_dict']
            scheduler_state = checkpoint['scheduler_state_dict']
            history = checkpoint['history']
            start_iter = checkpoint['iteration'] + 1
            best_loss = checkpoint.get('best_loss', float('inf'))
            print(f"Resumed self-training from checkpoint at iteration {start_iter}, best loss: {best_loss:.6f}")
        else:
            if not continued:
                self.reset_parameters()   
            history = []
        
        # Convert sequences to tensors on CPU
        seqs_tensors = [torch.tensor(seq, dtype=torch.float32) for seq in seqs]
        max_len = max(len(seq) for seq in seqs)
        
        # Pad sequences to same length for batch processing
        padded_seqs = []
        for seq_tensor in seqs_tensors:
            if len(seq_tensor) < max_len:
                padding = torch.zeros(max_len - len(seq_tensor), self.input_dim)
                padded = torch.cat([seq_tensor, padding], dim=0)
            else:
                padded = seq_tensor
            padded_seqs.append(padded)
        
        all_seqs_tensor = torch.stack(padded_seqs)  # [num_seqs, max_len, input_dim]
        num_samples = len(seqs)
        
        # Set up optimizer
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)
        
        # Load optimizer and scheduler states if resuming
        if continued and checkpoint_file and os.path.exists(checkpoint_file):
            optimizer.load_state_dict(optimizer_state)
            scheduler.load_state_dict(scheduler_state)
        
        prev_loss = float('inf') if start_iter == 0 else history[-1] if history else float('inf')
        
        for it in range(start_iter, max_iters):
            total_loss = 0.0
            
            # Shuffle indices
            perm = torch.randperm(num_samples)
            
            # Process each batch
            for i in range(0, num_samples, batch_size):
                batch_idx = perm[i:i+batch_size]
                batch_seqs = all_seqs_tensor[batch_idx].to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.forward(batch_seqs)  # [batch_size, seq_len, output_dim]
                
                # Calculate loss based on self_mode
                loss = 0.0
                
                if self_mode == 'gap':
                    # ||Hier[k] - Input[k]||^2
                    diff = outputs - batch_seqs[:, :, :self.model_dims[-1]]  # Ensure dimensions match
                    loss = torch.mean(diff ** 2)
                else:  # 'reg' mode
                    # Predict next: Hier[k] vs Input[k+1]
                    if max_len > 1:
                        # Targets: Input 1 to max_len
                        targets = batch_seqs[:, 1:, :self.model_dims[-1]]
                        # Preds: Hier 0 to max_len-1
                        preds = outputs[:, :-1, :]
                        
                        diff = preds - targets
                        loss = torch.mean(diff ** 2)
                    else:
                        loss = torch.tensor(0.0, device=self.device)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item() * len(batch_idx)
            
            # Average loss
            avg_loss = total_loss / num_samples
            history.append(avg_loss)
            
            # Update best model state if current loss is lower
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_model_state = copy.deepcopy(self.state_dict())
            
            # Print progress
            if it % print_every == 0 or it == max_iters - 1:
                current_lr = scheduler.get_last_lr()[0]
                mode_display = "Gap" if self_mode == 'gap' else "Reg"
                print(f"SelfTrain({mode_display}) Iter {it:3d}: loss = {avg_loss:.6f}, LR = {current_lr:.6f}")

            # Update learning rate
            if it % 5 == 0:  # Decay every 5 iterations
                scheduler.step()
            
            # Save checkpoint at specified intervals
            if checkpoint_file and (it % checkpoint_interval == 0 or it == max_iters - 1):
                self._save_checkpoint(
                    checkpoint_file, it, history, optimizer, scheduler, best_loss
                )
            
            # Check convergence
            if prev_loss - avg_loss < tol:
                print(f"Converged after {it+1} iterations")
                # Restore the best model state before breaking
                if best_model_state is not None and avg_loss > prev_loss:
                    self.load_state_dict(best_model_state)
                    print(f"Restored best model state with loss = {best_loss:.6f}")
                    history[-1] = best_loss                    
                break
            prev_loss = avg_loss
        
        # If training ended without convergence, restore best model state
        if best_model_state is not None and best_loss < history[-1]:
            self.load_state_dict(best_model_state)
            print(f"Training ended. Restored best model state with loss = {best_loss:.6f}")            
            history[-1] = best_loss
        
        # Store training statistics and compute mean target
        self._compute_training_statistics(seqs)
        self._compute_mean_target(seqs)
        self.trained = True
        
        return history
    
    def reconstruct(self, L, tau=0.0, discrete_mode=False, vocab_size=None):
        """
        Reconstruct sequence of vectors with temperature-controlled randomness.
        Supports both continuous and discrete reconstruction modes.
        
        Args:
            L (int): Number of vectors to reconstruct
            tau (float): Temperature parameter
            discrete_mode (bool): If True, use discrete sampling
            vocab_size (int): Required for discrete mode
        
        Returns:
            Reconstructed sequence as numpy array
        """
        assert self.trained and hasattr(self, 'mean_t'), "Model must be self-trained first"
        if tau < 0:
            raise ValueError("Temperature must be non-negative")
        if discrete_mode and vocab_size is None:
            raise ValueError("vocab_size must be specified for discrete mode")

        # Set model to evaluation mode
        self.eval()
        
        reconstructed = []
        # Create initial sequence with proper shape and values
        seq_len = 10  # Fixed sequence length for reconstruction
        current_seq = torch.normal(
            mean=0.0,
            std=0.1,
            size=(1, seq_len, self.input_dim),
            device=self.device
        )
        # Add the learned mean vector to the initial sequence
        mean_tensor = torch.tensor(self.mean_t, device=self.device, dtype=torch.float32)
        current_seq = current_seq + mean_tensor.unsqueeze(0).unsqueeze(0)  # Broadcast mean to all positions
        
        with torch.no_grad():
            for _ in range(L):
                # Forward pass
                outputs = self.forward(current_seq)
                
                # Get the last output vector
                output_vector = outputs[0, -1]
                
                if discrete_mode:
                    # Discrete reconstruction mode
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
                    # Continuous reconstruction mode
                    if tau > 0:
                        noise = torch.normal(0, tau * torch.abs(output_vector) + 0.01)
                        output_vector = output_vector + noise
                
                reconstructed.append(output_vector.cpu().numpy())
                
                # Update current sequence (shift window)
                current_seq = torch.cat([current_seq[:, 1:, :], output_vector.unsqueeze(0).unsqueeze(0)], dim=1)
        
        return np.array(reconstructed)

    def _compute_training_statistics(self, seqs, batch_size=256):
        """Compute and store statistics for reconstruction"""
        # Flatten all sequences to compute mean
        all_vectors = []
        for seq in seqs:
            all_vectors.extend(seq)
        
        if all_vectors:
            all_vectors_tensor = torch.tensor(all_vectors, dtype=torch.float32, device=self.device)
            self.mean_t = all_vectors_tensor.mean(dim=0).cpu().numpy()
        else:
            self.mean_t = np.zeros(self.input_dim)

    def _compute_mean_target(self, seqs, batch_size=256):
        """
        Compute the mean target vector in the final hierarchical layer output space
        This represents the average pattern learned by the entire hierarchical model
        """
        if not seqs:
            self.mean_target = None
            return
            
        final_dim = self.model_dims[-1] if self.model_dims else self.input_dim
        total_output = torch.zeros(final_dim, device=self.device)
        total_sequences = 0
        
        # Process sequences in batches
        for i in range(0, len(seqs), batch_size):
            batch_seqs = seqs[i:i+batch_size]
            batch_tensors = []
            
            for seq in batch_seqs:
                seq_tensor = torch.tensor(seq, dtype=torch.float32, device=self.device)
                batch_tensors.append(seq_tensor)
            
            # Find max length in batch
            max_len = max(len(seq) for seq in batch_seqs)
            
            # Pad sequences in batch
            padded_batch = []
            for seq_tensor in batch_tensors:
                if len(seq_tensor) < max_len:
                    padding = torch.zeros(max_len - len(seq_tensor), self.input_dim, device=self.device)
                    padded = torch.cat([seq_tensor, padding], dim=0)
                else:
                    padded = seq_tensor
                padded_batch.append(padded.unsqueeze(0))
            
            if padded_batch:
                batch_tensor = torch.cat(padded_batch, dim=0)
                
                with torch.no_grad():
                    output = self.forward(batch_tensor)
                    
                    # Mean over sequence, Sum over batch
                    batch_means = output.mean(dim=1)
                    total_output += batch_means.sum(dim=0)
                    total_sequences += len(batch_seqs)
        
        if total_sequences > 0:
            self.mean_target = total_output / total_sequences
        else:
            self.mean_target = None
    
    def reset_parameters(self):
        """
        Reset all model parameters to initial values
        """
        for layer in self.layers:
            nn.init.uniform_(layer.linear.weight, -0.1, 0.1)
            nn.init.uniform_(layer.Acoeff, -0.1, 0.1)
            nn.init.uniform_(layer.Bbasis, -0.1, 0.1)
            if layer.residual_proj is not None:
                nn.init.uniform_(layer.residual_proj.weight, -0.1, 0.1)
        
        # Reset training state
        self.mean_target = None
        self.trained = False
        self.mean_t = None

    def count_parameters(self):
        """
        Calculate and print the number of learnable parameters in table format
        """
        total_params = 0
        
        print("="*70)
        print(f"{'Component':<15} | {'Layer/Param':<25} | {'Count':<15} | {'Shape':<20}")
        print("-"*70)
        
        # Count parameters for each layer
        for l_idx, layer in enumerate(self.layers):
            in_dim = self.input_dim if l_idx == 0 else self.model_dims[l_idx-1]
            out_dim = self.model_dims[l_idx]
            basis_dim = self.basis_dims[l_idx]
            layer_params = 0
            layer_name = f"Layer {l_idx}"
            
            # Linear parameters
            num = layer.linear.weight.numel()
            shape = str(tuple(layer.linear.weight.shape))
            print(f"{layer_name:<15} | {'linear':<25} | {num:<15} | {shape:<20}")
            layer_params += num
            total_params += num
            
            # Acoeff parameters
            num = layer.Acoeff.numel()
            shape = str(tuple(layer.Acoeff.shape))
            print(f"{layer_name:<15} | {'Acoeff':<25} | {num:<15} | {shape:<20}")
            layer_params += num
            total_params += num
            
            # Bbasis parameters
            num = layer.Bbasis.numel()
            shape = str(tuple(layer.Bbasis.shape))
            print(f"{layer_name:<15} | {'Bbasis':<25} | {num:<15} | {shape:<20}")
            layer_params += num
            total_params += num
            
            # LayerNorm parameters
            norm_params = sum(p.numel() for p in layer.norm.parameters())
            shape = f"({out_dim},)"  # LayerNorm has parameters of size out_dim
            print(f"{layer_name:<15} | {'LayerNorm':<25} | {norm_params:<15} | {shape:<20}")
            layer_params += norm_params
            total_params += norm_params
            
            # Residual projection parameters if using 'separate' mode
            if layer.use_residual == 'separate' and hasattr(layer, 'residual_proj'):
                if isinstance(layer.residual_proj, nn.Linear):
                    num = layer.residual_proj.weight.numel()
                    shape = str(tuple(layer.residual_proj.weight.shape))
                    print(f"{layer_name:<15} | {'residual_proj':<25} | {num:<15} | {shape:<20}")
                    layer_params += num
                    total_params += num
            
            print(f"{layer_name:<15} | {'TOTAL':<25} | {layer_params:<15} |")
            print("-"*70)
        
        print(f"{'TOTAL':<15} | {'ALL PARAMETERS':<25} | {total_params:<15} |")
        print("="*70)
        
        return total_params
    
    def save(self, filename):
        """Save model state to file"""
        # Convert mean_target to numpy for saving
        mean_target_np = self.mean_target.cpu().numpy() if self.mean_target is not None else None
        
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': {
                'input_dim': self.input_dim,
                'model_dims': self.model_dims,
                'basis_dims': self.basis_dims,
                'use_residual_list': self.use_residual_list
            },
            'training_stats': {
                'mean_t': self.mean_t if hasattr(self, 'mean_t') else None,
                'mean_target': mean_target_np
            }
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
            basis_dims=config['basis_dims'],
            use_residual_list=config.get('use_residual_list', None),
            device=device
        )
        
        # Load state
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load training statistics
        if 'training_stats' in checkpoint:
            stats = checkpoint['training_stats']
            if stats['mean_t'] is not None:
                model.mean_t = stats['mean_t']
            if stats['mean_target'] is not None:
                model.mean_target = torch.tensor(stats['mean_target'], device=device)
            model.trained = True
            
        print(f"Model loaded from {filename}")
        return model


# === Example Usage ===
if __name__ == "__main__":

    from statistics import correlation    
    
    # Set random seeds for reproducibility
    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)
    
    # Configuration
    input_dim = 10            # Input vector dimension
    model_dims = [8, 6, 3]    # Output dimensions for each layer
    basis_dims = [150, 100, 50] # Basis dimensions for each layer
    use_residual_list = ['separate', 'separate', 'separate']  # Residual types for each layer
    seq_count = 100            # Number of training sequences
    min_len = 100              # Minimum sequence length
    max_len = 200              # Maximum sequence length
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Generate training data
    print("\nGenerating training data...")
    print(f"Input dimension: {input_dim}, Model dims: {model_dims}, Basis dims: {basis_dims}")
    print(f"Residual types: {use_residual_list}")
    seqs = []     # List of input sequences
    t_list = []   # List of target vectors (dimension = model_dims[-1])
    
    for i in range(seq_count):
        # Random sequence length
        length = random.randint(min_len, max_len)
        # Generate vector sequence
        sequence = np.random.uniform(-1, 1, (length, input_dim)).tolist()
        seqs.append(sequence)
        
        # Generate random target vector (dimension = last model_dim)
        target = np.random.uniform(-1, 1, model_dims[-1]).tolist()
        t_list.append(target)
    
    # Create model
    print("\nCreating Hierarchical HierDDrn model...")
    model = HierDDrn(
        input_dim=input_dim,
        model_dims=model_dims,
        basis_dims=basis_dims,
        use_residual_list=use_residual_list,
        device=device
    )
    
    # Show model structure
    print("\nModel parameter counts (table format):")
    model.count_parameters()
    
    # Train model with batch processing and checkpointing
    print("\nTraining model with batch processing and checkpointing...")
    history = model.reg_train(
        seqs,
        t_list,
        learning_rate=0.05,
        max_iters=100,
        tol=1e-8,
        decay_rate=0.99,
        print_every=10,
        batch_size=32,
        checkpoint_file='hierddrn_gd_checkpoint.pth',
        checkpoint_interval=10
    )
    
    # Evaluate predictions
    print("\nEvaluating predictions...")
    pred_t_list = [model.predict_t(seq) for seq in seqs]
    
    print("Prediction correlations per dimension:")
    corrs = []
    for d in range(model_dims[-1]):
        actuals = [t[d] for t in t_list]
        preds = [p[d] for p in pred_t_list]
        corr = correlation(actuals, preds)
        corrs.append(corr)
        print(f"  Dim {d}: correlation = {corr:.4f}")
    
    print(f"Average correlation: {np.mean(corrs):.4f}")     
    
    # Save and load demonstration
    print("\nSave and load demonstration:")
    model.save("hierarchical_model_rn.pth")
    
    # Load the model
    loaded_model = HierDDrn.load("hierarchical_model_rn.pth", device=device)
    
    # Verify loaded model works
    test_seq = seqs[0][:10]  # First 10 vectors of first sequence
    original_output = model.predict_t(test_seq)
    loaded_output = loaded_model.predict_t(test_seq)
    
    print(f"Original model prediction: {original_output}")
    print(f"Loaded model prediction: {loaded_output}")
    print(f"Predictions match: {np.allclose(original_output, loaded_output, atol=1e-6)}")

    # Example of using the new self_train and reconstruct methods
    print("\n" + "="*50)
    print("Example of using self_train and reconstruct methods")
    print("="*50)

    # Create a new model for self-training
    self_model = HierDDrn(
        input_dim=input_dim,
        model_dims=[10, 20, 10],
        basis_dims=basis_dims,
        use_residual_list=use_residual_list,
        device=device
    )
    
    # Self-train the model in 'gap' mode with checkpointing
    print("\nSelf-training model in 'gap' mode with checkpointing...")
    self_history = self_model.self_train(
        seqs[:10],
        max_iters=20,
        learning_rate=0.01,
        self_mode='gap',
        print_every=5,
        batch_size=16,
        checkpoint_file='hierddrn_self_checkpoint.pth',
        checkpoint_interval=5
    )
    
    # Reconstruct new sequences
    print("\nReconstructing new sequences...")
    reconstructed_seq = self_model.reconstruct(L=5, tau=0.1)
    print(f"Reconstructed sequence shape: {reconstructed_seq.shape}")
    print("First 5 reconstructed vectors:")
    for i, vec in enumerate(reconstructed_seq):
        print(f"Vector {i+1}: {[f'{x:.4f}' for x in vec]}")
    
    # Test self-training in 'reg' mode
    print("\n" + "="*50)
    print("Testing self-training in 'reg' mode")
    print("="*50)
    
    reg_model = HierDDrn(
        input_dim=input_dim,
        model_dims=[8, 6, 4],
        basis_dims=basis_dims,
        use_residual_list=use_residual_list,
        device=device
    )
    
    # Self-train in 'reg' mode
    reg_history = reg_model.self_train(
        seqs[:5],
        max_iters=10,
        learning_rate=0.01,
        self_mode='reg',
        print_every=2,
        batch_size=8
    )
    
    # Save and load model with checkpoint resume capability
    print("\nTesting checkpoint resume capability...")
    
    # Train a model and save checkpoint
    checkpoint_model = HierDDrn(
        input_dim=input_dim,
        model_dims=[8, 6, 3],
        basis_dims=[100, 80, 60],
        use_residual_list=['separate', 'separate', 'separate'],
        device=device
    )
    
    # First training session (10 iterations)
    print("\nFirst training session (10 iterations)...")
    checkpoint_model.reg_train(
        seqs[:3],
        t_list[:3],
        max_iters=10,
        learning_rate=0.01,
        print_every=5,
        batch_size=4,
        checkpoint_file='hierddrn_resume_checkpoint.pth',
        checkpoint_interval=5
    )
    
    # Resume training from checkpoint
    print("\nResuming training from checkpoint...")
    checkpoint_model.reg_train(
        seqs[:3],
        t_list[:3],
        max_iters=20,
        learning_rate=0.01,
        continued=True,
        print_every=5,
        batch_size=4,
        checkpoint_file='hierddrn_resume_checkpoint.pth',
        checkpoint_interval=5
    )
    
    print("\nAll training and testing completed successfully!")
        
    print("\n=== Hierarchical Vector Sequence Processing with Random Basis and Batch Processing Completed ===")
