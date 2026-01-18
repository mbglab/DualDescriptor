# Copyright (C) 2005-2025, Bin-Guang Ma (mbg@mail.hzau.edu.cn); SPDX-License-Identifier: MIT
# The Hierarchical Numeric Dual Descriptor (AB matrix form) with Linker matrices in PyTorch
# With layer normalization and residual connections and generation capability
# This program is for the demonstration of methodology and not fully refined.
# Author: Bin-Guang Ma (assisted by DeepSeek); Date: 2025-8-26 ~ 2026-1-15

import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import os

class Layer(nn.Module):
    """
    A single layer of the Hierarchical Numeric Dual Descriptor with Linker
    
    Args:
        in_dim (int): Input dimension
        out_dim (int): Output dimension
        basis_dim (int): Basis dimension
        in_seq_len (int): Input sequence length
        out_seq_len (int): Output sequence length
        linker_trainable (bool): Whether linker matrix is trainable
        use_residual (str): Residual connection type ('separate', 'shared', or None)
        device (str): Device to run the layer on ('cuda' or 'cpu')
    """
    def __init__(self, in_dim, out_dim, basis_dim, in_seq_len, out_seq_len, 
                 linker_trainable, use_residual=None, device='cuda'):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.basis_dim = basis_dim
        self.in_seq_len = in_seq_len
        self.out_seq_len = out_seq_len
        self.use_residual = use_residual
        self.linker_trainable = linker_trainable
        self.device = device
        
        # Initialize transformation matrix M
        self.M = nn.Parameter(torch.empty(out_dim, in_dim, device=device))
        nn.init.uniform_(self.M, -0.1, 0.1)
        
        # Initialize coefficient matrix Acoeff
        self.Acoeff = nn.Parameter(torch.empty(out_dim, basis_dim, device=device))
        nn.init.uniform_(self.Acoeff, -0.1, 0.1)
        
        # Fixed basis matrix Bbasis (non-trainable)
        Bbasis = torch.empty(basis_dim, out_dim, device=device)
        for k in range(basis_dim):
            for i in range(out_dim):
                Bbasis[k, i] = math.cos(2 * math.pi * (k + 1) / (i + 1))
        self.register_buffer('Bbasis', Bbasis)
        
        # Initialize Linker matrix
        self.Linker = nn.Parameter(torch.empty(in_seq_len, out_seq_len, device=device))
        nn.init.uniform_(self.Linker, -0.1, 0.1)
        
        # Freeze Linker if not trainable
        if not linker_trainable:
            self.Linker.requires_grad = False
            
        # Layer normalization
        self.layer_norm = nn.LayerNorm(out_dim, device=device)
        
        # Handle different residual connection types
        if self.use_residual == 'separate':
            # Separate projection and linker for residual path
            self.residual_proj = nn.Linear(in_dim, out_dim, bias=False, device=device)
            nn.init.uniform_(self.residual_proj.weight, -0.1, 0.1)
            
            self.residual_linker = nn.Parameter(torch.empty(in_seq_len, out_seq_len, device=device))
            nn.init.uniform_(self.residual_linker, -0.1, 0.1)
            if not linker_trainable:
                self.residual_linker.requires_grad = False
        else:
            self.residual_proj = None
            self.residual_linker = None
            
        # Precompute basis indices for sequence positions
        self.register_buffer('basis_indices', 
                            torch.tensor([i % basis_dim for i in range(in_seq_len)], device=device))
        
    def forward(self, x):
        """
        Forward pass for the layer
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, in_seq_len, in_dim)
            
        Returns:
            Tensor: Output tensor of shape (batch_size, out_seq_len, out_dim)
        """
        # Main path processing
        # Apply linear transformation
        z = torch.matmul(x, self.M.T)
        
        # Apply layer normalization
        z = self.layer_norm(z)

        # Position-wise processing
        # Select basis vectors for each position: (in_seq_len, out_dim)
        basis_vectors = self.Bbasis[self.basis_indices]
        
        # Compute scalars: (batch_size, in_seq_len)
        scalars = torch.einsum('bsd,sd->bs', z, basis_vectors)
        
        # Select coefficients for each position: (in_seq_len, out_dim)
        coeffs = self.Acoeff[:, self.basis_indices].permute(1, 0)
        
        # Compute position outputs: (batch_size, in_seq_len, out_dim)
        u = coeffs * scalars.unsqueeze(-1)
        
        # Apply sequence length transformation using Linker matrix
        # (batch_size, in_seq_len, out_dim) x (in_seq_len, out_seq_len) -> (batch_size, out_dim, out_seq_len)
        v = torch.matmul(u.permute(0, 2, 1), self.Linker)
        
        # Transpose to (batch_size, out_seq_len, out_dim)
        main_output = v.permute(0, 2, 1)
        
        # Process residual connections
        if self.use_residual == 'separate':
            # Separate projection and linker for residual
            residual_feat = self.residual_proj(x)
            residual = torch.matmul(
                residual_feat.permute(0, 2, 1), 
                self.residual_linker
            ).permute(0, 2, 1)
            return main_output + residual
            
        elif self.use_residual == 'shared':
            # Shared M and Linker for residual
            residual_feat = torch.matmul(x, self.M.T)
            residual = torch.matmul(
                residual_feat.permute(0, 2, 1), 
                self.Linker
            ).permute(0, 2, 1)
            return main_output + residual
            
        elif self.use_residual is None:
            # Identity residual only if dimensions match
            if self.in_dim == self.out_dim and self.in_seq_len == self.out_seq_len:
                residual = torch.matmul(
                    x.permute(0, 2, 1), 
                    self.Linker
                ).permute(0, 2, 1)
                return main_output + residual
            else:
                return main_output
                
        return main_output

class HierDDLab(nn.Module):
    """
    Hierarchical Numeric Dual Descriptor with Linker for vector sequences
    
    Args:
        input_dim (int): Dimension of input vectors
        model_dims (list): Output dimensions for each layer
        basis_dims (list): Basis dimensions for each layer
        linker_dims (list): Output sequence lengths for each layer
        input_seq_len (int): Fixed input sequence length
        linker_trainable (bool|list): Controls if Linker matrices are trainable
        use_residual (str|list): Residual connection type ('separate', 'shared', or None)
        device (str): Device to run the model on ('cuda' or 'cpu')
    """
    def __init__(self, input_dim=4, model_dims=[4], basis_dims=[50], 
                 linker_dims=[50], input_seq_len=100, linker_trainable=False,
                 use_residual=None, device='cuda'):
        super().__init__()
        self.input_dim = input_dim
        self.model_dims = model_dims
        self.basis_dims = basis_dims
        self.linker_dims = linker_dims
        self.input_seq_len = input_seq_len
        self.num_layers = len(model_dims)
        self.trained = False
        self.device = device
        
        # Classification heads (initialized when needed)
        self.num_classes = None
        self.classifier = None
        self.num_labels = None
        self.labeller = None
        
        # Validate dimensions
        if len(basis_dims) != self.num_layers:
            raise ValueError("basis_dims length must match model_dims length")
        if len(linker_dims) != self.num_layers:
            raise ValueError("linker_dims length must match model_dims length")
        
        # Process linker_trainable parameter
        if isinstance(linker_trainable, bool):
            self.linker_trainable = [linker_trainable] * self.num_layers
        elif isinstance(linker_trainable, list):
            if len(linker_trainable) != self.num_layers:
                raise ValueError("linker_trainable list length must match number of layers")
            self.linker_trainable = linker_trainable
        else:
            raise TypeError("linker_trainable must be bool or list of bools")
        
        # Process use_residual parameter
        if use_residual is None or isinstance(use_residual, str):
            self.use_residual = [use_residual] * self.num_layers
        elif isinstance(use_residual, list):
            if len(use_residual) != self.num_layers:
                raise ValueError("use_residual list length must match number of layers")
            self.use_residual = use_residual
        else:
            raise TypeError("use_residual must be str, None, or list of str/None")
        
        # Create layers
        self.layers = nn.ModuleList()
        for i, out_dim in enumerate(model_dims):
            # Determine input dimensions for this layer
            in_dim = input_dim if i == 0 else model_dims[i-1]
            in_seq = input_seq_len if i == 0 else linker_dims[i-1]
            out_seq = linker_dims[i]
            
            layer = Layer(
                in_dim=in_dim,
                out_dim=out_dim,
                basis_dim=basis_dims[i],
                in_seq_len=in_seq,
                out_seq_len=out_seq,
                linker_trainable=self.linker_trainable[i],
                use_residual=self.use_residual[i],
                device=device
            )
            self.layers.append(layer)
    
    def forward(self, x):
        """
        Forward pass through all layers
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, input_seq_len, input_dim)
            
        Returns:
            Tensor: Output tensor of shape (batch_size, linker_dims[-1], model_dims[-1])
        """
        for layer in self.layers:
            x = layer(x)
        return x
    
    def extract_sequence_representation(self, x):
        """
        Extract sequence-level representation by averaging over sequence dimension
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, dim)
            
        Returns:
            Tensor: Sequence representation of shape (batch_size, dim)
        """
        # Forward pass through all layers
        output = self.forward(x)  # (batch_size, out_seq_len, out_dim)
        
        # Average over sequence dimension to get sequence-level representation
        seq_rep = output.mean(dim=1)  # (batch_size, out_dim)
        return seq_rep
    
    def describe(self, vec_seq):
        """
        Compute descriptor vectors for a sequence
        
        Args:
            vec_seq (list or Tensor): Sequence of vectors (seq_len, input_dim)
            
        Returns:
            list: List of output vectors (linker_dims[-1], model_dims[-1])
        """
        # Convert to tensor and add batch dimension
        if not isinstance(vec_seq, torch.Tensor):
            vec_seq = torch.tensor(vec_seq, dtype=torch.float32, device=self.device)
        else:
            vec_seq = vec_seq.to(self.device)
        x = vec_seq.unsqueeze(0)  # (1, seq_len, input_dim)
        
        # Forward pass
        with torch.no_grad():
            output = self.forward(x)
            
        # Remove batch dimension and convert to list of vectors
        return output.squeeze(0).tolist()      
    
    def deviation(self, seqs, t_list):
        """
        Compute mean squared error loss
        
        Args:
            seqs (list): List of sequences, each is (input_seq_len, input_dim)
            t_list (list): List of target vectors (model_dims[-1])
            
        Returns:
            Tensor: Mean squared error loss
        """
        # Convert to tensors and move to device
        seqs_tensor = torch.tensor(np.array(seqs), dtype=torch.float32, device=self.device)
        targets = torch.tensor(np.array(t_list), dtype=torch.float32, device=self.device)
        
        # Forward pass
        outputs = self.forward(seqs_tensor)  # (batch_size, out_seq, out_dim)
        
        # Compute average vector for each sequence
        avg_vectors = outputs.mean(dim=1)  # (batch_size, out_dim)
        
        # Compute MSE loss
        loss = nn.functional.mse_loss(avg_vectors, targets)
        return loss
    
    def reg_train(self, seqs, t_list, max_iters=1000, tol=1e-88, 
                   learning_rate=0.01, decay_rate=1.0, print_every=10,
                   continued=False, checkpoint_file=None, checkpoint_interval=10,
                   batch_size=256):
        """
        Train the model using Adam optimizer with batch processing and checkpoint support
        
        Args:
            seqs (list): Training sequences
            t_list (list): Target vectors
            max_iters (int): Maximum training iterations
            tol (float): Tolerance for convergence
            learning_rate (float): Initial learning rate
            decay_rate (float): Learning rate decay rate
            print_every (int): Print progress every n iterations
            continued (bool): Whether to continue from existing parameters
            checkpoint_file (str): File to save/load checkpoints
            checkpoint_interval (int): Save checkpoint every n iterations
            batch_size (int): Batch size for training
            
        Returns:
            list: Training loss history
        """
        # Load checkpoint if continuing
        start_iter = 0
        best_loss = float('inf')
        best_model_state = None
        history = []
        
        if continued and checkpoint_file and os.path.exists(checkpoint_file):
            checkpoint = torch.load(checkpoint_file, map_location=self.device, weights_only=False)
            self.load_state_dict(checkpoint['model_state_dict'])
            optimizer_state = checkpoint['optimizer_state_dict']
            scheduler_state = checkpoint['scheduler_state_dict']
            history = checkpoint['history']
            start_iter = checkpoint['iteration'] + 1
            best_loss = checkpoint.get('best_loss', float('inf'))
            print(f"Resumed training from checkpoint at iteration {start_iter}, best loss: {best_loss:.6e}")
        
        # Convert data to tensors
        seqs_tensor = torch.tensor(np.array(seqs), dtype=torch.float32, device=self.device)
        targets = torch.tensor(np.array(t_list), dtype=torch.float32, device=self.device)
        num_samples = len(seqs)
        
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)
        
        # Load optimizer state if continuing
        if continued and checkpoint_file and os.path.exists(checkpoint_file):
            optimizer.load_state_dict(optimizer_state)
            scheduler.load_state_dict(scheduler_state)
        
        for it in range(start_iter, max_iters):
            total_loss = 0.0
            
            # Shuffle indices
            perm = torch.randperm(num_samples)
            
            for i in range(0, num_samples, batch_size):
                # Get batch indices
                batch_idx = perm[i:i+batch_size]
                batch_seqs = seqs_tensor[batch_idx]
                batch_targets = targets[batch_idx]
                
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.forward(batch_seqs)
                avg_vectors = outputs.mean(dim=1)
                
                # Compute loss
                loss = nn.functional.mse_loss(avg_vectors, batch_targets)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                
                # Update parameters
                optimizer.step()
                
                total_loss += loss.item() * len(batch_idx)
            
            # Average loss
            avg_loss = total_loss / num_samples
            history.append(avg_loss)
            
            # Update best loss
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_model_state = copy.deepcopy(self.state_dict())
            
            # Print progress
            if it % print_every == 0 or it == max_iters - 1:
                lr = optimizer.param_groups[0]['lr']
                print(f"GD Iter {it:3d}: Loss = {avg_loss:.6e}, LR = {lr:.6e}")
            
            # Save checkpoint
            if checkpoint_file and (it % checkpoint_interval == 0 or it == max_iters - 1):
                self._save_checkpoint(checkpoint_file, it, history, optimizer, scheduler, best_loss)
            
            # Check convergence
            if len(history) > 1 and abs(history[-2] - avg_loss) < tol:
                print(f"Converged after {it+1} iterations")
                break
            
            # Decay learning rate
            scheduler.step()
        
        # Restore best model if available
        if best_model_state is not None and best_loss < history[-1]:
            self.load_state_dict(best_model_state)
            print(f"Restored best model state with loss = {best_loss:.6e}")
            history[-1] = best_loss
        
        self.trained = True
        return history
    
    def cls_train(self, seqs, labels, num_classes, max_iters=1000, tol=1e-8, learning_rate=0.01,
                  continued=False, decay_rate=1.0, print_every=10, batch_size=32,
                  checkpoint_file=None, checkpoint_interval=10):
        """
        Train the model for multi-class classification using cross-entropy loss.
        
        Args:
            seqs (list): List of vector sequences for training
            labels (list): List of integer class labels (0 to num_classes-1)
            num_classes (int): Number of classes in the classification problem
            max_iters (int): Maximum number of training iterations
            tol (float): Convergence tolerance
            learning_rate (float): Initial learning rate for optimizer
            continued (bool): Whether to continue training from existing parameters
            decay_rate (float): Learning rate decay rate
            print_every (int): Print progress every N iterations
            batch_size (int): Number of sequences to process in each batch
            checkpoint_file (str): Path to save training checkpoints
            checkpoint_interval (int): Save checkpoint every N iterations
            
        Returns:
            list: Training loss history
        """
        # Initialize classification head if not already done
        if self.classifier is None or self.num_classes != num_classes:
            self.classifier = nn.Linear(self.model_dims[-1], num_classes).to(self.device)
            self.num_classes = num_classes
        
        # Load checkpoint if continuing
        start_iter = 0
        best_loss = float('inf')
        best_model_state = None
        history = []
        
        if continued and checkpoint_file and os.path.exists(checkpoint_file):
            checkpoint = torch.load(checkpoint_file, map_location=self.device, weights_only=False)
            self.load_state_dict(checkpoint['model_state_dict'])
            optimizer_state = checkpoint['optimizer_state_dict']
            scheduler_state = checkpoint['scheduler_state_dict']
            history = checkpoint['history']
            start_iter = checkpoint['iteration'] + 1
            best_loss = checkpoint.get('best_loss', float('inf'))
            print(f"Resumed classification training from checkpoint at iteration {start_iter}, best loss: {best_loss:.6e}")
        
        # Convert data to tensors
        seqs_tensor = torch.tensor(np.array(seqs), dtype=torch.float32, device=self.device)
        labels_tensor = torch.tensor(labels, dtype=torch.long, device=self.device)
        num_samples = len(seqs)
        
        # Setup optimizer and scheduler
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)
        
        # Cross-entropy loss
        criterion = nn.CrossEntropyLoss()
        
        # Load optimizer state if continuing
        if continued and checkpoint_file and os.path.exists(checkpoint_file):
            optimizer.load_state_dict(optimizer_state)
            scheduler.load_state_dict(scheduler_state)
        
        for it in range(start_iter, max_iters):
            total_loss = 0.0
            total_correct = 0
            total_samples = 0
            
            # Shuffle indices
            perm = torch.randperm(num_samples)
            
            for i in range(0, num_samples, batch_size):
                # Get batch indices
                batch_idx = perm[i:i+batch_size]
                batch_seqs = seqs_tensor[batch_idx]
                batch_labels = labels_tensor[batch_idx]
                
                optimizer.zero_grad()
                
                # Forward pass through hierarchical layers
                outputs = self.forward(batch_seqs)  # (batch_size, out_seq_len, out_dim)
                
                # Extract sequence-level representation by averaging over sequence dimension
                seq_reps = outputs.mean(dim=1)  # (batch_size, out_dim)
                
                # Pass through classification head
                logits = self.classifier(seq_reps)  # (batch_size, num_classes)
                
                # Compute loss
                loss = criterion(logits, batch_labels)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                
                # Update parameters
                optimizer.step()
                
                # Calculate accuracy
                with torch.no_grad():
                    predictions = torch.argmax(logits, dim=1)
                    correct = (predictions == batch_labels).sum().item()
                    total_correct += correct
                    total_samples += len(batch_labels)
                    total_loss += loss.item() * len(batch_labels)
            
            # Calculate average loss and accuracy
            avg_loss = total_loss / num_samples if num_samples > 0 else 0.0
            accuracy = total_correct / total_samples if total_samples > 0 else 0.0
            history.append(avg_loss)
            
            # Update best loss
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_model_state = copy.deepcopy(self.state_dict())
            
            # Print progress
            if it % print_every == 0 or it == max_iters - 1:
                lr = optimizer.param_groups[0]['lr']
                print(f"CLS-Train Iter {it:3d}: Loss = {avg_loss:.6e}, Acc = {accuracy:.4f}, LR = {lr:.6e}")
            
            # Save checkpoint
            if checkpoint_file and (it % checkpoint_interval == 0 or it == max_iters - 1):
                checkpoint = {
                    'iteration': it,
                    'model_state_dict': self.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'history': history,
                    'best_loss': best_loss,
                    'num_classes': self.num_classes
                }
                torch.save(checkpoint, checkpoint_file)
            
            # Check convergence
            if len(history) > 1 and abs(history[-2] - avg_loss) < tol:
                print(f"Converged after {it+1} iterations")
                break
            
            # Decay learning rate
            scheduler.step()
        
        # Restore best model if available
        if best_model_state is not None and best_loss < history[-1]:
            self.load_state_dict(best_model_state)
            print(f"Restored best model state with loss = {best_loss:.6e}")
            history[-1] = best_loss
        
        self.trained = True
        return history
    
    def lbl_train(self, seqs, labels, num_labels, max_iters=1000, tol=1e-8, learning_rate=0.01, 
                  continued=False, decay_rate=1.0, print_every=10, batch_size=32,
                  checkpoint_file=None, checkpoint_interval=10, pos_weight=None):
        """
        Train the model for multi-label classification using binary cross-entropy loss.
        
        Args:
            seqs (list): List of vector sequences for training
            labels (list): List of binary label vectors (list of lists) or 2D numpy array/torch tensor
            num_labels (int): Number of labels for multi-label prediction task
            max_iters (int): Maximum number of training iterations
            tol (float): Convergence tolerance
            learning_rate (float): Initial learning rate for optimizer
            continued (bool): Whether to continue training from existing parameters
            decay_rate (float): Learning rate decay rate
            print_every (int): Print progress every N iterations
            batch_size (int): Number of sequences to process in each batch
            checkpoint_file (str): Path to save training checkpoints
            checkpoint_interval (int): Save checkpoint every N iterations
            pos_weight (list or torch.Tensor): Weight for positive class (shape [num_labels])
            
        Returns:
            tuple: (loss_history, acc_history)
        """
        # Initialize label head if not already done
        if self.labeller is None or self.num_labels != num_labels:
            self.labeller = nn.Linear(self.model_dims[-1], num_labels).to(self.device)
            self.num_labels = num_labels
        
        # Load checkpoint if continuing
        start_iter = 0
        best_loss = float('inf')
        best_model_state = None
        loss_history = []
        acc_history = []
        
        if continued and checkpoint_file and os.path.exists(checkpoint_file):
            checkpoint = torch.load(checkpoint_file, map_location=self.device, weights_only=False)
            self.load_state_dict(checkpoint['model_state_dict'])
            optimizer_state = checkpoint['optimizer_state_dict']
            scheduler_state = checkpoint['scheduler_state_dict']
            loss_history = checkpoint.get('loss_history', [])
            acc_history = checkpoint.get('acc_history', [])
            start_iter = checkpoint['iteration'] + 1
            best_loss = checkpoint.get('best_loss', float('inf'))
            print(f"Resumed multi-label training from checkpoint at iteration {start_iter}, best loss: {best_loss:.6e}")
        
        # Convert data to tensors
        seqs_tensor = torch.tensor(np.array(seqs), dtype=torch.float32, device=self.device)
        if isinstance(labels, list):
            labels_tensor = torch.tensor(labels, dtype=torch.float32, device=self.device)
        else:
            labels_tensor = torch.as_tensor(labels, dtype=torch.float32, device=self.device)
        num_samples = len(seqs)
        
        # Setup loss function with optional positive class weighting
        if pos_weight is not None:
            pos_weight_tensor = torch.tensor(pos_weight, dtype=torch.float32, device=self.device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        else:
            criterion = nn.BCEWithLogitsLoss()
        
        # Setup optimizer and scheduler
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)
        
        # Load optimizer state if continuing
        if continued and checkpoint_file and os.path.exists(checkpoint_file):
            optimizer.load_state_dict(optimizer_state)
            scheduler.load_state_dict(scheduler_state)
        
        for it in range(start_iter, max_iters):
            total_loss = 0.0
            total_correct = 0
            total_predictions = 0
            
            # Shuffle indices
            perm = torch.randperm(num_samples)
            
            for i in range(0, num_samples, batch_size):
                # Get batch indices
                batch_idx = perm[i:i+batch_size]
                batch_seqs = seqs_tensor[batch_idx]
                batch_labels = labels_tensor[batch_idx]
                
                optimizer.zero_grad()
                
                # Forward pass through hierarchical layers
                outputs = self.forward(batch_seqs)  # (batch_size, out_seq_len, out_dim)
                
                # Extract sequence-level representation by averaging over sequence dimension
                seq_reps = outputs.mean(dim=1)  # (batch_size, out_dim)
                
                # Pass through label head
                logits = self.labeller(seq_reps)  # (batch_size, num_labels)
                
                # Compute loss
                loss = criterion(logits, batch_labels)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                
                # Update parameters
                optimizer.step()
                
                # Calculate accuracy
                with torch.no_grad():
                    probs = torch.sigmoid(logits)
                    predictions = (probs > 0.5).float()
                    correct = (predictions == batch_labels).sum().item()
                    total_correct += correct
                    total_predictions += batch_labels.numel()
                    total_loss += loss.item() * len(batch_labels)
            
            # Calculate average loss and accuracy
            avg_loss = total_loss / num_samples if num_samples > 0 else 0.0
            accuracy = total_correct / total_predictions if total_predictions > 0 else 0.0
            loss_history.append(avg_loss)
            acc_history.append(accuracy)
            
            # Update best loss
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_model_state = copy.deepcopy(self.state_dict())
            
            # Print progress
            if it % print_every == 0 or it == max_iters - 1:
                lr = optimizer.param_groups[0]['lr']
                print(f"MLC-Train Iter {it:3d}: Loss = {avg_loss:.6e}, Acc = {accuracy:.4f}, LR = {lr:.6e}")
            
            # Save checkpoint
            if checkpoint_file and (it % checkpoint_interval == 0 or it == max_iters - 1):
                checkpoint = {
                    'iteration': it,
                    'model_state_dict': self.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss_history': loss_history,
                    'acc_history': acc_history,
                    'best_loss': best_loss,
                    'num_labels': self.num_labels
                }
                torch.save(checkpoint, checkpoint_file)
            
            # Check convergence
            if len(loss_history) > 1 and abs(loss_history[-2] - avg_loss) < tol:
                print(f"Converged after {it+1} iterations")
                break
            
            # Decay learning rate
            scheduler.step()
        
        # Restore best model if available
        if best_model_state is not None and best_loss < loss_history[-1]:
            self.load_state_dict(best_model_state)
            print(f"Restored best model state with loss = {best_loss:.6e}")
            loss_history[-1] = best_loss
        
        self.trained = True
        return loss_history, acc_history
    
    def predict_c(self, vec_seq):
        """
        Predict class label for a vector sequence using the classification head.
        
        Args:
            vec_seq (list or Tensor): Input vector sequence
            
        Returns:
            tuple: (predicted_class, class_probabilities)
        """
        if self.classifier is None:
            raise ValueError("Model must be trained first for classification")
        
        # Convert to tensor and add batch dimension
        if not isinstance(vec_seq, torch.Tensor):
            vec_seq = torch.tensor(vec_seq, dtype=torch.float32, device=self.device)
        else:
            vec_seq = vec_seq.to(self.device)
        x = vec_seq.unsqueeze(0)  # (1, seq_len, input_dim)
        
        with torch.no_grad():
            # Forward pass through hierarchical layers
            output = self.forward(x)  # (1, out_seq_len, out_dim)
            
            # Extract sequence-level representation by averaging over sequence dimension
            seq_rep = output.mean(dim=1)  # (1, out_dim)
            
            # Pass through classification head
            logits = self.classifier(seq_rep)  # (1, num_classes)
            
            # Get probabilities and predicted class
            probabilities = torch.softmax(logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            
        return predicted_class, probabilities[0].cpu().numpy()
    
    def predict_l(self, vec_seq, threshold=0.5):
        """
        Predict multi-label classification for a vector sequence.
        
        Args:
            vec_seq (list or Tensor): Input vector sequence
            threshold (float): Probability threshold for binary classification (default: 0.5)
            
        Returns:
            tuple: (binary_predictions, probability_scores)
        """
        if self.labeller is None:
            raise ValueError("Model must be trained first for multi-label classification")
        
        # Convert to tensor and add batch dimension
        if not isinstance(vec_seq, torch.Tensor):
            vec_seq = torch.tensor(vec_seq, dtype=torch.float32, device=self.device)
        else:
            vec_seq = vec_seq.to(self.device)
        x = vec_seq.unsqueeze(0)  # (1, seq_len, input_dim)
        
        with torch.no_grad():
            # Forward pass through hierarchical layers
            output = self.forward(x)  # (1, out_seq_len, out_dim)
            
            # Extract sequence-level representation by averaging over sequence dimension
            seq_rep = output.mean(dim=1)  # (1, out_dim)
            
            # Pass through label head
            logits = self.labeller(seq_rep)  # (1, num_labels)
            
            # Get probabilities
            probabilities = torch.sigmoid(logits).cpu().numpy()[0]
            
            # Apply threshold to get binary predictions
            binary_predictions = (probabilities > threshold).astype(np.float32)
            
        return binary_predictions, probabilities
    
    def _save_checkpoint(self, checkpoint_file, iteration, history, optimizer, scheduler, best_loss):
        """
        Save training checkpoint
        
        Args:
            checkpoint_file (str): File to save checkpoint
            iteration (int): Current iteration
            history (list): Training loss history
            optimizer: Optimizer object
            scheduler: Learning rate scheduler
            best_loss (float): Best loss so far
        """
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
                'input_seq_len': self.input_seq_len,
                'linker_dims': self.linker_dims,
                'linker_trainable': self.linker_trainable,
                'use_residual': self.use_residual,
                'device': str(self.device),
                'num_classes': self.num_classes,
                'num_labels': self.num_labels
            },
            'training_stats': {
                'mean_t': self.mean_t if hasattr(self, 'mean_t') else None
            }
        }
        torch.save(checkpoint, checkpoint_file)

    def predict_t(self, vec_seq):
        """
        Predict target vector as average of all final layer outputs
        
        Args:
            vec_seq (list or Tensor): Sequence of vectors (seq_len, input_dim)
            
        Returns:
            list: Average output vector (model_dims[-1])
        """
        outputs = self.describe(vec_seq)
        if not outputs:
            return [0.0] * self.model_dims[-1]
        
        # Compute average vector
        return np.mean(outputs, axis=0).tolist()
    
    def self_train(self, seqs, max_iters=100, tol=1e-16, learning_rate=0.01, 
                  continued=False, self_mode='gap', decay_rate=1.0, print_every=10,
                  batch_size=256, checkpoint_file=None, checkpoint_interval=5):
        """
        Self-training for the ENTIRE hierarchical model with two modes:
          - 'gap': Predicts current input vector (self-consistency)
          - 'reg': Predicts next input vector (auto-regressive) with causal masking
        
        Now trains ALL layers of the hierarchical model, not just the first layer.
        Stores statistical information for reconstruction and generation.
        
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
            checkpoint_file: File to save/load checkpoints
            checkpoint_interval: Save checkpoint every n iterations
        
        Returns:
            Training loss history
        """
        if self_mode not in ('gap', 'reg'):
            raise ValueError("self_mode must be either 'gap' or 'reg'")
        
        # Validate input sequences
        for seq in seqs:
            if len(seq) != self.input_seq_len:
                raise ValueError(f"All sequences must have length {self.input_seq_len}")
        
        # Load checkpoint if continuing
        start_iter = 0
        best_loss = float('inf')
        best_model_state = None
        history = []
        
        if continued and checkpoint_file and os.path.exists(checkpoint_file):
            checkpoint = torch.load(checkpoint_file, map_location=self.device, weights_only=False)
            self.load_state_dict(checkpoint['model_state_dict'])
            optimizer_state = checkpoint['optimizer_state_dict']
            scheduler_state = checkpoint['scheduler_state_dict']
            history = checkpoint['history']
            start_iter = checkpoint['iteration'] + 1
            best_loss = checkpoint.get('best_loss', float('inf'))
            print(f"Resumed self-training from checkpoint at iteration {start_iter}, best loss: {best_loss:.6f}")
        
        # Convert sequences to tensors
        seqs_tensor = torch.tensor(np.array(seqs), dtype=torch.float32, device=self.device)
        num_samples = len(seqs)
        
        # Initialize all layers if not continuing training
        if not continued:
            for layer in self.layers:
                # Reinitialize parameters with small random values
                nn.init.uniform_(layer.M, -0.5, 0.5)
                nn.init.uniform_(layer.Acoeff, -0.1, 0.1)
                
                if layer.Linker.requires_grad:
                    nn.init.uniform_(layer.Linker, -0.1, 0.1)
                
                # Reinitialize residual components if using 'separate' mode
                if layer.use_residual == 'separate':
                    nn.init.uniform_(layer.residual_proj.weight, -0.1, 0.1)
                    if hasattr(layer, 'residual_linker') and layer.residual_linker.requires_grad:
                        nn.init.uniform_(layer.residual_linker, -0.1, 0.1)
            
            # Calculate global mean vector of input sequences
            total = torch.zeros(self.input_dim, device=self.device)
            total_vectors = 0
            for seq in seqs_tensor:
                total += torch.sum(seq, dim=0)
                total_vectors += seq.size(0)
            self.mean_t = (total / total_vectors).cpu().numpy()            
        
        # Setup optimizer
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)
        
        # Load optimizer state if continuing
        if continued and checkpoint_file and os.path.exists(checkpoint_file):
            optimizer.load_state_dict(optimizer_state)
            scheduler.load_state_dict(scheduler_state)
        
        for it in range(start_iter, max_iters):
            total_loss = 0.0
            total_batches = 0  # Track number of processed batches
            
            # Shuffle indices
            perm = torch.randperm(num_samples)
            
            for i in range(0, num_samples, batch_size):
                # Get batch indices
                batch_idx = perm[i:i+batch_size]
                batch_seqs = seqs_tensor[batch_idx]
                
                optimizer.zero_grad()
                
                # Apply causal masking to Linker matrices in reg mode
                original_linkers = []
                original_residual_linkers = []
                
                if self_mode == 'reg':
                    for layer in self.layers:
                        # Create causal mask for main Linker
                        causal_mask = torch.tril(torch.ones_like(layer.Linker))
                        original_linker = layer.Linker.clone()
                        layer.Linker.data = layer.Linker.data * causal_mask
                        original_linkers.append(original_linker)
                        
                        # Create causal mask for residual Linker if exists
                        if hasattr(layer, 'residual_linker') and layer.residual_linker is not None:
                            causal_mask_residual = torch.tril(torch.ones_like(layer.residual_linker))
                            original_residual_linker = layer.residual_linker.clone()
                            layer.residual_linker.data = layer.residual_linker.data * causal_mask_residual
                            original_residual_linkers.append(original_residual_linker)
                        else:
                            original_residual_linkers.append(None)
                
                # Forward pass through all layers
                x = batch_seqs
                for layer in self.layers:
                    x = layer(x)
                
                # Restore original Linker matrices after forward pass in reg mode
                if self_mode == 'reg':
                    for i, layer in enumerate(self.layers):
                        layer.Linker.data = original_linkers[i]
                        if original_residual_linkers[i] is not None:
                            layer.residual_linker.data = original_residual_linkers[i]
                
                # Calculate loss based on self_mode
                loss = 0.0
                
                # Vectorized loss calculation
                if self_mode == 'gap':
                    # For gap mode, we need to compare hierarchical output with original input
                    # Check if output dimensions match input dimensions
                    if x.shape[1:] == batch_seqs.shape[1:]:
                        # Direct comparison when dimensions match
                        loss = nn.functional.mse_loss(x, batch_seqs)
                    else:
                        # When dimensions don't match, we need to handle differently
                        # For demonstration, we'll compute loss on overlapping dimensions
                        min_seq_len = min(x.shape[1], batch_seqs.shape[1])
                        min_dim = min(x.shape[2], batch_seqs.shape[2])
                        
                        if min_seq_len > 0 and min_dim > 0:
                            # Compare only overlapping parts
                            pred_slice = x[:, :min_seq_len, :min_dim]
                            target_slice = batch_seqs[:, :min_seq_len, :min_dim]
                            loss = nn.functional.mse_loss(pred_slice, target_slice)
                        else:
                            # Skip this batch if no overlap
                            continue
                else:  # 'reg' mode
                    # Predict next vector: Hier[k] vs Input[k+1]
                    # We need at least 2 positions for meaningful comparison
                    if batch_seqs.shape[1] > 1 and x.shape[1] >= 1:
                        # Targets: Input from position 1 to end
                        targets = batch_seqs[:, 1:, :]
                        
                        # Predictions: Hierarchical output from position 0 to end-1
                        # We need to ensure we have enough output positions
                        max_pred_len = min(x.shape[1], targets.shape[1])
                        
                        if max_pred_len > 0:
                            preds = x[:, :max_pred_len, :]
                            targets_slice = targets[:, :max_pred_len, :]
                            loss = nn.functional.mse_loss(preds, targets_slice)
                        else:
                            continue
                    else:
                        # Skip this batch if not enough positions
                        continue
                
                # Only proceed if we have a valid loss
                if loss.item() > 0:
                    # Backward pass and optimization
                    loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                    
                    # Update parameters
                    optimizer.step()
                    
                    total_loss += loss.item()
                    total_batches += 1
            
            # Calculate average loss only if we processed any batches
            if total_batches > 0:
                avg_loss = total_loss / total_batches
            else:
                # If no batches were processed, use a high loss to continue training
                avg_loss = 1.0  # Arbitrary high value to continue training
                print(f"Iteration {it}: No valid batches processed, using default loss")
            
            history.append(avg_loss)
            
            # Update best loss
            if avg_loss < best_loss and total_batches > 0:
                best_loss = avg_loss
                best_model_state = copy.deepcopy(self.state_dict())
            
            # Print progress
            if it % print_every == 0 or it == max_iters - 1:
                current_lr = scheduler.get_last_lr()[0]
                mode_display = "Gap" if self_mode == 'gap' else "Reg"
                print(f"SelfTrain({mode_display}) Iter {it:3d}: loss = {avg_loss:.6f}, LR = {current_lr:.6f}, Batches = {total_batches}")
            
            # Save checkpoint
            if checkpoint_file and (it % checkpoint_interval == 0 or it == max_iters - 1):
                self._save_checkpoint(checkpoint_file, it, history, optimizer, scheduler, best_loss)
            
            # Check convergence - only if we processed batches and loss is not default
            if len(history) > 1 and total_batches > 0 and avg_loss != 1.0:
                loss_change = abs(history[-2] - avg_loss)
                if loss_change < tol:
                    print(f"Converged after {it+1} iterations with loss change {loss_change:.6e}")
                    break
            
            # Update learning rate scheduler - only if optimizer.step() was called
            if total_batches > 0:
                scheduler.step()
        
        # Restore best model if available
        if best_model_state is not None and best_loss < history[-1] and total_batches > 0:
            self.load_state_dict(best_model_state)
            print(f"Restored best model state with loss = {best_loss:.6f}")
            history[-1] = best_loss
        
        self.trained = True
        return history
    
    def reconstruct(self, L, tau=0.0, discrete_mode=False, vocab_size=None):
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
        assert self.trained and hasattr(self, 'mean_t'), "Model must be self-trained first"
        if tau < 0:
            raise ValueError("Temperature must be non-negative")
        if discrete_mode and vocab_size is None:
            raise ValueError("vocab_size must be specified for discrete mode")

        # Set model to evaluation mode
        self.eval()
        
        generated = []
        
        # Create initial sequence with proper shape and values
        current_seq = torch.normal(
            mean=0.0,  # Use float mean instead of tensor
            std=0.1,
            size=(self.input_seq_len, self.input_dim),
            device=self.device
        )
        # Add the learned mean vector to the initial sequence
        mean_tensor = torch.tensor(self.mean_t, device=self.device, dtype=torch.float32)
        current_seq = current_seq + mean_tensor.unsqueeze(0)  # Broadcast mean to all positions
        
        with torch.no_grad():
            for _ in range(L):
                # Forward pass through all layers
                current = current_seq.unsqueeze(0)  # Add batch dimension
                for layer in self.layers:
                    current = layer(current)
                
                # Remove batch dimension
                current = current.squeeze(0)
                
                # Get the last output vector
                output_vector = current[-1]
                
                if discrete_mode:
                    # Discrete generation mode
                    discrete_vector = []
                    for value in output_vector:
                        # Create logits and sample
                        logits = torch.full((vocab_size,), value.item(), device=self.device)
                        if tau > 0:
                            logits += torch.normal(0, tau, size=(vocab_size,), device=self.device)
                        
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

    def count_parameters(self):
        """
        Calculate and print the number of learnable parameters in table format
        """
        total_params = 0
        trainable_params = 0
        
        print("="*70)
        print(f"{'Component':<15} | {'Layer/Param':<25} | {'Count':<15} | {'Shape':<20}")
        print("-"*70)
        
        for l_idx, layer in enumerate(self.layers):
            in_dim = self.input_dim if l_idx == 0 else self.model_dims[l_idx-1]
            out_dim = self.model_dims[l_idx]
            basis_dim = self.basis_dims[l_idx]
            in_seq = self.input_seq_len if l_idx == 0 else self.linker_dims[l_idx-1]
            out_seq = self.linker_dims[l_idx]
            
            print(f"Layer {l_idx:<13} | {'Config':<25} | {f'in={in_dim}, out={out_dim}':<15} | {f'L={basis_dim}, seq={in_seq}→{out_seq}':<20}")
            print("-"*70)
            
            layer_params = 0
            layer_trainable = 0
            
            # Count parameters for each parameter in the layer
            for name, param in layer.named_parameters():
                if param.requires_grad:
                    num = param.numel()
                    layer_trainable += num
                    print(f"{'':<15} | {name:<25} | {num:<15} | {str(tuple(param.shape)):<20}")
                layer_params += param.numel()
            
            # Count buffers (non-trainable parameters)
            for name, buffer in layer.named_buffers():
                if buffer is not None:
                    num = buffer.numel()
                    print(f"{'':<15} | {name+'(buffer)':<25} | {num:<15} | {str(tuple(buffer.shape)):<20}")
            
            total_params += layer_params
            trainable_params += layer_trainable
            
            print(f"{'':<15} | {'TOTAL':<25} | {layer_params:<15} |")
            print(f"{'':<15} | {'Trainable':<25} | {layer_trainable:<15} |")
            print("-"*70)
        
        # Count classification heads if they exist
        if self.classifier is not None:
            for name, param in self.classifier.named_parameters():
                if param.requires_grad:
                    num = param.numel()
                    trainable_params += num
                    print(f"{'Classifier':<15} | {name:<25} | {num:<15} | {str(tuple(param.shape)):<20}")
                total_params += param.numel()
            print("-"*70)
        
        if self.labeller is not None:
            for name, param in self.labeller.named_parameters():
                if param.requires_grad:
                    num = param.numel()
                    trainable_params += num
                    print(f"{'Labeller':<15} | {name:<25} | {num:<15} | {str(tuple(param.shape)):<20}")
                total_params += param.numel()
            print("-"*70)
        
        # Print summary
        print(f"{'SUMMARY':<15} | {'Total Parameters':<25} | {total_params:<15} |")
        print(f"{'':<15} | {'Trainable Parameters':<25} | {trainable_params:<15} |")
        print("="*70)
        
        return total_params, trainable_params  
    
    def save(self, filename):
        """Save model state to file"""
        # Create classifier and labeller state dicts if they exist
        classifier_state = None
        labeller_state = None
        
        if self.classifier is not None:
            classifier_state = self.classifier.state_dict()
        
        if self.labeller is not None:
            labeller_state = self.labeller.state_dict()
        
        save_dict = {
            'model_state_dict': self.state_dict(),
            'classifier_state_dict': classifier_state,
            'labeller_state_dict': labeller_state,
            'config': {
                'input_dim': self.input_dim,
                'model_dims': self.model_dims,
                'basis_dims': self.basis_dims,
                'linker_dims': self.linker_dims,
                'input_seq_len': self.input_seq_len,
                'linker_trainable': self.linker_trainable,
                'use_residual': self.use_residual,
                'device': self.device
            },
            'mean_t': self.mean_t if hasattr(self, 'mean_t') else None,
            'trained': self.trained,
            'num_classes': self.num_classes,
            'num_labels': self.num_labels
        }
        torch.save(save_dict, filename)
        print(f"Model saved to {filename}")
    
    @classmethod
    def load(cls, filename, device=None):
        """Load model from file"""
        # Device configuration
        if device is None:
            # Try to get device from saved config, default to cuda if available
            checkpoint = torch.load(filename, map_location='cpu', weights_only=False)
            device = checkpoint['config'].get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
            device = torch.device(device)
        
        # Load checkpoint
        checkpoint = torch.load(filename, map_location=device, weights_only=False)
        config = checkpoint['config']
        
        # Create model instance
        model = cls(
            input_dim=config['input_dim'],
            model_dims=config['model_dims'],
            basis_dims=config['basis_dims'],
            linker_dims=config['linker_dims'],
            input_seq_len=config['input_seq_len'],
            linker_trainable=config['linker_trainable'],
            use_residual=config['use_residual'],
            device=device
        ).to(device)
        
        # Load main model state (excluding classifier and labeller weights)
        model_state_dict = checkpoint['model_state_dict']
        
        # Filter out classifier and labeller weights from main state dict
        filtered_state_dict = {}
        for key, value in model_state_dict.items():
            if not (key.startswith('classifier.') or key.startswith('labeller.')):
                filtered_state_dict[key] = value
        
        # Load the filtered state dict
        model.load_state_dict(filtered_state_dict, strict=False)
        
        # Load additional attributes
        if 'mean_t' in checkpoint:
            model.mean_t = checkpoint['mean_t']
        
        model.trained = checkpoint.get('trained', True)
        model.num_classes = checkpoint.get('num_classes', None)
        model.num_labels = checkpoint.get('num_labels', None)
        
        # Recreate and load classifier if it exists
        if model.num_classes is not None and 'classifier_state_dict' in checkpoint:
            model.classifier = nn.Linear(model.model_dims[-1], model.num_classes).to(device)
            model.classifier.load_state_dict(checkpoint['classifier_state_dict'])
        
        # Recreate and load labeller if it exists
        if model.num_labels is not None and 'labeller_state_dict' in checkpoint:
            model.labeller = nn.Linear(model.model_dims[-1], model.num_labels).to(device)
            model.labeller.load_state_dict(checkpoint['labeller_state_dict'])
        
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
    input_dim = 10          # Input vector dimension
    input_seq_len = 300     # Fixed input sequence length
    model_dims = [8, 6, 3] # Output dimensions for each layer
    basis_dims = [150, 100, 50] # Basis dimensions for each layer
    linker_dims = [50, 20, 10] # Output sequence lengths for each layer
    seq_count = 100         # Number of training sequences
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Generate training data with fixed sequence length
    print("\nGenerating training data...")
    print(f"Input dim: {input_dim}, Seq len: {input_seq_len}")
    print(f"Layer dims: {model_dims}, Basis dims: {basis_dims}, Linker dims: {linker_dims}")
    
    seqs = []   # List of sequences
    t_list = [] # List of target vectors

    L = input_seq_len
    for i in range(seq_count):
        # Generate vector sequence with fixed length
        sequence = np.random.uniform(-1, 1, (input_seq_len, input_dim))
        seqs.append(sequence)
        
        # Generate random target vector
        target = np.random.uniform(-1, 1, model_dims[-1])
        t_list.append(target)
    
    # Test Case 1: Mixed residual strategies
    print("\n=== Test Case 1: Mixed Residual Strategies ===")
    model_mixed = HierDDLab(
        input_dim=input_dim,
        model_dims=model_dims,
        basis_dims=basis_dims,
        linker_dims=linker_dims,
        input_seq_len=input_seq_len,
        linker_trainable=[True, False, True],
        use_residual=['separate', 'shared', None],  # Different residual for each layer
        device=device
    )
    
    # Show parameter counts in table format
    print("\nModel parameter counts (table format):")
    total_params, trainable_params = model_mixed.count_parameters()
    
    # Train model with batch processing and checkpoint
    print("\nTraining model with batch processing...")
    reg_history = model_mixed.reg_train(
        seqs, 
        t_list,
        learning_rate=0.01,
        tol=1e-88,
        max_iters=100,
        decay_rate=0.99,
        print_every=10,
        batch_size=256,
        checkpoint_file='hier_ddl_gd_checkpoint.pth',
        checkpoint_interval=20
    )
    
    # Evaluate predictions
    pred_t_list = [model_mixed.predict_t(seq) for seq in seqs]
    print("\nPrediction correlations per dimension:")
    corrs = []
    for d in range(model_dims[-1]):
        actuals = [t[d] for t in t_list]
        preds = [p[d] for p in pred_t_list]
        corr = correlation(actuals, preds)
        corrs.append(corr)
        print(f"  Dim {d}: correlation = {corr:.4f}")
    
    print(f"Average correlation: {np.mean(corrs):.4f}")      
    
    # Test Case 2: Self-training and reconstruction
    print("\n=== Test Case 2: Self-training and Reconstruction ===")
    
    # Create a fresh model for self-training
    model_self = HierDDLab(
        input_dim=input_dim,
        model_dims=[10, 20, 10],
        basis_dims=basis_dims,
        linker_dims=linker_dims,
        input_seq_len=input_seq_len,
        linker_trainable=[True, False, True],
        use_residual=['separate', 'shared', None],
        device=device
    )
    
    # Self-train in gap mode with checkpoint
    print("\nSelf-training in 'gap' mode...")
    self_history = model_self.self_train(
        seqs[:10],
        max_iters=20,
        learning_rate=0.01,
        self_mode='gap',
        print_every=5,
        batch_size=32,
        checkpoint_file='hier_ddl_self_checkpoint.pth',
        checkpoint_interval=10
    )
    
    # Reconstruct new sequences
    print("\nReconstructing sequences...")
    reconstructed_seq = model_self.reconstruct(L=5, tau=0.1)
    print(f"Reconstructed sequence shape: {reconstructed_seq.shape}")
    print("First 5 reconstructed vectors:")
    for i, vec in enumerate(reconstructed_seq):
        print(f"Vector {i+1}: {[f'{x:.4f}' for x in vec]}")
    
    # Test Case 3: Continued training
    print("\n=== Test Case 3: Continued Training ===")
    
    # Continue training from checkpoint
    print("\nContinuing self-training from checkpoint...")
    continued_history = model_self.self_train(
        seqs[:10],
        max_iters=15,
        learning_rate=0.005,
        self_mode='gap',
        print_every=5,
        batch_size=32,
        continued=True,
        checkpoint_file='hier_ddl_self_checkpoint.pth',
        checkpoint_interval=10
    )
    
    print(f"Initial self-training had {len(self_history)} iterations")
    print(f"Continued training added {len(continued_history)} iterations")
    
    # Test Case 4: Classification task
    print("\n=== Test Case 4: Classification Task ===")
    
    # Generate classification data
    num_classes = 3
    class_seqs = []
    class_labels = []
    
    # Create sequences with different patterns for each class
    for class_id in range(num_classes):
        for _ in range(30):  # 30 sequences per class
            if class_id == 0:
                # Class 0: Pattern with positive bias
                seq = np.random.uniform(0.5, 1.5, (input_seq_len, input_dim))
            elif class_id == 1:
                # Class 1: Pattern with negative bias
                seq = np.random.uniform(-1.5, -0.5, (input_seq_len, input_dim))
            else:
                # Class 2: Normal distribution
                seq = np.random.uniform(-1, 1, (input_seq_len, input_dim))
            
            class_seqs.append(seq)
            class_labels.append(class_id)
    
    # Create a fresh model for classification
    model_cls = HierDDLab(
        input_dim=input_dim,
        model_dims=[8, 6, 4],  # Output dimension 4 for classification
        basis_dims=[100, 80, 60],
        linker_dims=[50, 30, 10],
        input_seq_len=input_seq_len,
        linker_trainable=[True, False, True],
        use_residual=['separate', 'shared', None],
        device=device
    )
    
    # Train for classification
    print("\nTraining for classification...")
    cls_history = model_cls.cls_train(
        class_seqs,
        class_labels,
        num_classes=num_classes,
        max_iters=50,
        learning_rate=0.005,
        decay_rate=0.99,
        print_every=5,
        batch_size=16,
        checkpoint_file='hier_ddl_cls_checkpoint.pth'
    )
    
    # Test classification predictions
    print("\nTesting classification predictions:")
    correct = 0
    for i in range(min(5, len(class_seqs))):
        pred_class, probs = model_cls.predict_c(class_seqs[i])
        true_class = class_labels[i]
        if pred_class == true_class:
            correct += 1
        print(f"Seq {i+1}: True={true_class}, Pred={pred_class}, Probs={[f'{p:.3f}' for p in probs]}")
    
    print(f"Accuracy on first 5 samples: {correct}/5 = {correct/5:.2f}")
    
    # Test Case 5: Multi-label classification task
    print("\n=== Test Case 5: Multi-Label Classification Task ===")
    
    # Generate multi-label data
    num_labels = 4
    label_seqs = []
    labels = []
    
    for _ in range(50):
        # Random sequence
        seq = np.random.uniform(-1, 1, (input_seq_len, input_dim))
        label_seqs.append(seq)
        
        # Random binary labels (multi-label classification)
        # Each sequence can have 0-4 active labels
        label_vec = [1.0 if np.random.random() > 0.7 else 0.0 for _ in range(num_labels)]
        labels.append(label_vec)
    
    # Create a fresh model for multi-label classification
    model_lbl = HierDDLab(
        input_dim=input_dim,
        model_dims=[8, 6, 4],  # Output dimension 4 for label prediction
        basis_dims=[100, 80, 60],
        linker_dims=[50, 30, 10],
        input_seq_len=input_seq_len,
        linker_trainable=[True, False, True],
        use_residual=['separate', 'shared', None],
        device=device
    )
    
    # Train for multi-label classification
    print("\nTraining for multi-label classification...")
    lbl_loss_history, lbl_acc_history = model_lbl.lbl_train(
        label_seqs,
        labels,
        num_labels=num_labels,
        max_iters=50,
        learning_rate=0.005,
        decay_rate=0.99,
        print_every=5,
        batch_size=16,
        checkpoint_file='hier_ddl_lbl_checkpoint.pth'
    )
    
    # Test multi-label predictions
    print("\nTesting multi-label predictions:")
    for i in range(min(3, len(label_seqs))):
        binary_pred, probs = model_lbl.predict_l(label_seqs[i], threshold=0.5)
        true_labels = labels[i]
        print(f"Seq {i+1}:")
        print(f"  True labels: {true_labels}")
        print(f"  Pred binary: {binary_pred}")
        print(f"  Pred probs: {[f'{p:.3f}' for p in probs]}")
        correct_all = all(binary_pred[j] == true_labels[j] for j in range(num_labels))
        print(f"  All correct: {correct_all}")
    
    # Test Case 6: Save and load model
    print("\n=== Test Case 6: Save and Load Model ===")
    
    # Save model
    model_cls.save("hier_ddl_cls_model.pth")
    
    # Load model
    loaded_model = HierDDLab.load("hier_ddl_cls_model.pth", device=device)
    
    # Verify loaded model works for classification
    test_pred_class, test_probs = loaded_model.predict_c(class_seqs[0])
    print(f"Original prediction for first sequence: Class {test_pred_class}")
    print(f"Loaded model prediction for first sequence: Class {test_pred_class}")
    
    # Test with multi-label model
    model_lbl.save("hier_ddl_lbl_model.pth")
    loaded_lbl_model = HierDDLab.load("hier_ddl_lbl_model.pth", device=device)
    
    # Verify multi-label predictions
    lbl_binary, lbl_probs = loaded_lbl_model.predict_l(label_seqs[0])
    print(f"\nOriginal multi-label prediction: {lbl_binary}")
    print(f"Loaded model multi-label prediction: {lbl_binary}")
    
    # Compare parameter counts
    print("\nOriginal classification model parameters:")
    total_orig, trainable_orig = model_cls.count_parameters()
    
    print("\nLoaded classification model parameters:")
    total_loaded, trainable_loaded = loaded_model.count_parameters()
    
    if total_orig == total_loaded and trainable_orig == trainable_loaded:
        print("✓ Parameter counts match between original and loaded classification models")
    else:
        print("✗ Parameter counts do not match")
    
    print("\n=== Hierarchical Vector Sequence Processing with Linker Matrices Demo Completed ===")
