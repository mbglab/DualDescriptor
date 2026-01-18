# Copyright (C) 2005-2025, Bin-Guang Ma (mbg@mail.hzau.edu.cn); SPDX-License-Identifier: MIT
# The Hierarchical Numeric Dual Descriptor (Tensor form) implemented with PyTorch
# With layer normalization and residual connections and generation capability
# This program is for the demonstration of methodology and not fully refined.
# Author: Bin-Guang Ma (assisted by DeepSeek); Date: 2025-7-22 ~ 2026-1-18

import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
import os

class Layer(nn.Module):
    """Single layer of Hierarchical Dual Descriptor"""
    def __init__(self, in_dim, out_dim, num_basis, use_residual):
        """
        Initialize a hierarchical layer
        
        Args:
            in_dim (int): Input dimension
            out_dim (int): Output dimension
            num_basis (int): Number of basis functions
            use_residual (str or None): Residual connection type. Options are:
                - 'separate': use a separate linear projection for the residual connection.
                - 'shared': share the linear transformation matrix M for the residual.
                - None or other: no residual connection, unless in_dim==out_dim, then use identity residual.
        """
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_basis = num_basis
        self.use_residual = use_residual
        
        # Linear transformation matrix (shared for main path and residual)
        self.M = nn.Parameter(torch.empty(out_dim, in_dim))
        
        # Tensor of basis coefficients
        self.P = nn.Parameter(torch.empty(out_dim, out_dim, num_basis))
        
        # Precompute periods tensor (fixed, not trainable)
        periods = torch.zeros(out_dim, out_dim, num_basis)
        for i in range(out_dim):
            for j in range(out_dim):
                for g in range(num_basis):
                    periods[i, j, g] = i*(out_dim*num_basis) + j*num_basis + g + 2
        self.register_buffer('periods', periods)

        # Residual projection processing
        if self.use_residual == 'separate':
            self.residual_proj = nn.Linear(in_dim, out_dim, bias=False) if in_dim != out_dim else nn.Identity()
        
        # Layer normalization
        self.norm = nn.LayerNorm(out_dim)
        
        # Initialize parameters        
        nn.init.uniform_(self.M, -0.5, 0.5)
        nn.init.uniform_(self.P, -0.1, 0.1)
        if isinstance(self.residual_proj, nn.Linear):            
            nn.init.uniform_(self.residual_proj.weight, -0.1, 0.1)      
    
    def forward(self, x, positions):
        """
        Forward pass for the layer
        
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
        # Expand tensors for broadcasting: positions (seq_len, 1, 1, 1)
        k = positions.view(-1, 1, 1, 1).float()
        
        # Compute basis functions: phi = cos(2π * k / period)
        # periods shape: (1, out_dim, out_dim, num_basis)
        periods = self.periods.unsqueeze(0)
        phi = torch.cos(2 * math.pi * k / periods)  # (seq_len, out_dim, out_dim, num_basis)
        
        # Apply position-dependent transformation
        # x_trans: (batch_size, seq_len, out_dim) -> (batch_size, seq_len, 1, out_dim, 1)
        x_exp = x_trans.unsqueeze(2).unsqueeze(4)
        
        # P: (out_dim, out_dim, num_basis) -> (1, 1, out_dim, out_dim, num_basis)
        P_exp = self.P.unsqueeze(0).unsqueeze(0)
        
        # Compute: P * x' * phi
        # Result shape: (batch_size, seq_len, out_dim, out_dim, num_basis)
        product = P_exp * x_exp * phi
        
        # Sum over j (input dim) and g (basis index)
        Nk = torch.sum(product, dim=(3, 4))  # (batch_size, seq_len, out_dim)
       
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

class HierDDts(nn.Module):
    """Hierarchical Numeric Dual Descriptor with PyTorch and GPU support"""
    def __init__(self, input_dim=2, model_dims=[2], num_basis_list=[5], use_residual_list=None, device='cpu'):
        """
        Initialize hierarchical HierDDts
        
        Args:
            input_dim (int): Input vector dimension
            model_dims (list): Output dimensions for each layer
            num_basis_list (list): Number of basis functions for each layer
            device (str): Device to use ('cpu' or 'cuda')
        """
        super().__init__()
        self.input_dim = input_dim
        self.model_dims = model_dims
        self.num_basis_list = num_basis_list
        if use_residual_list == None:
            self.use_residual_list = ['separate'] * len(self.model_dims)
        else:
            self.use_residual_list = use_residual_list            
        self.num_layers = len(model_dims)
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.trained = False
        
        # Create hierarchical layers
        layers = []
        in_dim = input_dim
        for out_dim, num_basis, use_residual in zip(self.model_dims, self.num_basis_list, self.use_residual_list):
            layers.append(Layer(in_dim, out_dim, num_basis, use_residual))
            in_dim = out_dim  # Next layer input is current layer output
        
        self.layers = nn.ModuleList(layers)
        
        # Classification head (initialized later when num_classes is known)
        self.num_classes = None
        self.classifier = None
        
        # Multi-label classification head (initialized later when num_labels is known)
        self.num_labels = None
        self.labeller = None
        
        self.to(self.device)
    
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
    
    def get_sequence_representation(self, seq):
        """
        Get sequence representation by averaging over sequence length
        
        Args:
            seq (Tensor): Input sequence of shape (batch_size, seq_len, input_dim)
        
        Returns:
            Tensor: Sequence representation of shape (batch_size, model_dims[-1])
        """
        return self.predict_t(seq)
    
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
    
    def reg_train(self, seqs, t_list, max_iters=1000, tol=1e-88, lr=0.01, 
                    decay_rate=0.999, print_every=10, continued=False,
                    checkpoint_file=None, checkpoint_interval=10):
        """
        Train model using Adam optimizer
        
        Args:
            seqs (list): List of training sequences
            t_list (list): List of target vectors
            max_iters (int): Maximum training iterations
            lr (float): Learning rate
            decay_rate (float): Learning rate decay rate
            print_every (int): Print progress every N iterations
            continued (bool): Whether to continue from existing parameters
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
                # Reinitialize parameters if not continuing
                for layer in self.layers:
                    nn.init.uniform_(layer.M, -0.5, 0.5)
                    nn.init.uniform_(layer.P, -0.1, 0.1)
                    if layer.use_residual == 'separate' and hasattr(layer, 'residual_proj'):
                        if isinstance(layer.residual_proj, nn.Linear):
                            nn.init.uniform_(layer.residual_proj.weight, -0.1, 0.1)
            history = []
        
        optimizer = optim.Adam(self.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)
        
        # Load optimizer and scheduler states if resuming
        if continued and checkpoint_file and os.path.exists(checkpoint_file):
            optimizer.load_state_dict(optimizer_state)
            scheduler.load_state_dict(scheduler_state)
        
        prev_loss = float('inf') if start_iter == 0 else history[-1] if history else float('inf')
        
        for it in range(start_iter, max_iters):
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
            
            # Update best model state if current loss is lower
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_model_state = copy.deepcopy(self.state_dict())
            
            # Print progress
            if it % print_every == 0 or it == max_iters - 1:
                current_lr = scheduler.get_last_lr()[0]
                print(f"Iter {it:4d}: Loss = {avg_loss:.6e}, LR = {current_lr:.6f}")

            # Update learning rate
            scheduler.step()
            
            # Save checkpoint at specified intervals
            if checkpoint_file and (it % checkpoint_interval == 0 or it == max_iters - 1):
                self._save_checkpoint(
                    checkpoint_file, it, history, optimizer, scheduler, best_loss
                )
            
            # Check convergence
            if abs(prev_loss - avg_loss) < tol:
                print(f"Converged after {it+1} iterations.")
                # Restore the best model state before breaking
                if best_model_state is not None and avg_loss > prev_loss:
                    self.load_state_dict(best_model_state)                    
                    print(f"Restored best model state with loss = {best_loss:.6e}")
                    history[-1] = best_loss
                break
                
            prev_loss = avg_loss
        
        # If training ended without convergence, restore best model state
        if best_model_state is not None and best_loss < history[-1]:
            self.load_state_dict(best_model_state)  
            print(f"Training ended. Restored best model state with loss = {best_loss:.6e}")
            history[-1] = best_loss
        
        return history

    def cls_train(self, seqs, labels, num_classes, max_iters=1000, tol=1e-8, learning_rate=0.01,
                  continued=False, decay_rate=1.0, print_every=10, batch_size=32,
                  checkpoint_file=None, checkpoint_interval=10):
        """
        Train the model for multi-class classification using cross-entropy loss.
        
        Args:
            seqs: List of input sequences for training
            labels: List of integer class labels (0 to num_classes-1)
            num_classes: Number of classes in the classification problem
            max_iters: Maximum number of training iterations
            tol: Convergence tolerance
            learning_rate: Initial learning rate for optimizer
            continued: Whether to continue training from existing parameters
            decay_rate: Learning rate decay rate
            print_every: Print progress every N iterations
            batch_size: Number of sequences to process in each batch
            checkpoint_file: Path to save training checkpoints
            checkpoint_interval: Save checkpoint every N iterations
        
        Returns:
            list: Training loss history
            list: Training accuracy history
        """
        # Initialize classification head if not already done
        if self.classifier is None or self.num_classes != num_classes:
            self.classifier = nn.Linear(self.model_dims[-1], num_classes).to(self.device)
            self.num_classes = num_classes
        
        if not continued:
            # Reinitialize model parameters if not continuing
            for layer in self.layers:
                nn.init.uniform_(layer.M, -0.5, 0.5)
                nn.init.uniform_(layer.P, -0.1, 0.1)
                if layer.use_residual == 'separate' and hasattr(layer, 'residual_proj'):
                    if isinstance(layer.residual_proj, nn.Linear):
                        nn.init.uniform_(layer.residual_proj.weight, -0.1, 0.1)
            # Initialize classifier
            nn.init.normal_(self.classifier.weight, 0, 0.01)
            if self.classifier.bias is not None:
                nn.init.constant_(self.classifier.bias, 0)
        
        # Load checkpoint if continuing and checkpoint file exists
        start_iter = 0
        best_loss = float('inf')
        best_model_state = None
        
        if continued and checkpoint_file and os.path.exists(checkpoint_file):
            checkpoint = torch.load(checkpoint_file, map_location=self.device, weights_only=False)
            self.load_state_dict(checkpoint['model_state_dict'])
            optimizer_state = checkpoint['optimizer_state_dict']
            scheduler_state = checkpoint['scheduler_state_dict']
            loss_history = checkpoint.get('loss_history', [])
            acc_history = checkpoint.get('acc_history', [])
            start_iter = checkpoint['iteration'] + 1
            best_loss = checkpoint.get('best_loss', float('inf'))
            print(f"Resumed classification training from checkpoint at iteration {start_iter}, best loss: {best_loss:.6e}")
        else:
            loss_history = []
            acc_history = []
        
        # Convert labels to tensor
        label_tensors = torch.tensor(labels, dtype=torch.long, device=self.device)
        
        # Setup optimizer and scheduler
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)
        
        # Cross-entropy loss
        criterion = nn.CrossEntropyLoss()
        
        # Load optimizer and scheduler states if resuming
        if continued and checkpoint_file and os.path.exists(checkpoint_file):
            optimizer.load_state_dict(optimizer_state)
            scheduler.load_state_dict(scheduler_state)
        
        prev_loss = float('inf') if start_iter == 0 else loss_history[-1] if loss_history else float('inf')
        
        for it in range(start_iter, max_iters):
            total_loss = 0.0
            total_correct = 0
            total_samples = 0
            
            # Shuffle sequences for each epoch
            indices = list(range(len(seqs)))
            random.shuffle(indices)
            
            # Process sequences in batches
            for batch_start in range(0, len(indices), batch_size):
                batch_indices = indices[batch_start:batch_start + batch_size]
                batch_seqs = [seqs[idx] for idx in batch_indices]
                batch_labels = label_tensors[batch_indices]
                
                optimizer.zero_grad()
                batch_loss = 0.0
                batch_correct = 0
                batch_logits = []
                
                # Process each sequence in the batch
                for seq in batch_seqs:
                    # Convert to tensor
                    seq_tensor = torch.tensor(seq, dtype=torch.float32, device=self.device)
                    
                    # Forward pass to get sequence representation
                    seq_representation = self.get_sequence_representation(seq_tensor.unsqueeze(0)).squeeze(0)
                    
                    # Get logits through classification head
                    logits = self.classifier(seq_representation.unsqueeze(0))
                    batch_logits.append(logits)
                
                # Stack all logits and compute loss
                if batch_logits:
                    all_logits = torch.cat(batch_logits, dim=0)
                    loss = criterion(all_logits, batch_labels)
                    loss.backward()
                    optimizer.step()
                    
                    # Calculate batch statistics
                    batch_loss = loss.item()
                    total_loss += batch_loss * len(batch_seqs)
                    
                    # Calculate accuracy
                    with torch.no_grad():
                        predictions = torch.argmax(all_logits, dim=1)
                        batch_correct = (predictions == batch_labels).sum().item()
                        total_correct += batch_correct
                        total_samples += len(batch_seqs)
            
            # Calculate average loss and accuracy for this iteration
            if total_samples > 0:
                avg_loss = total_loss / total_samples
                avg_acc = total_correct / total_samples
            else:
                avg_loss = 0.0
                avg_acc = 0.0
                
            loss_history.append(avg_loss)
            acc_history.append(avg_acc)
            
            # Update best model state
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_model_state = copy.deepcopy(self.state_dict())
            
            # Print training progress
            if it % print_every == 0 or it == max_iters - 1:
                current_lr = scheduler.get_last_lr()[0]
                print(f"CLS-Train Iter {it:3d}: Loss = {avg_loss:.6e}, Acc = {avg_acc:.4f}, LR = {current_lr:.6f}")
            
            # Save checkpoint at specified intervals
            if checkpoint_file and (it % checkpoint_interval == 0 or it == max_iters - 1):
                checkpoint = {
                    'iteration': it,
                    'model_state_dict': self.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss_history': loss_history,
                    'acc_history': acc_history,
                    'best_loss': best_loss,
                    'num_classes': self.num_classes
                }
                torch.save(checkpoint, checkpoint_file)
                print(f"Checkpoint saved at iteration {it}")
            
            # Check convergence
            if abs(prev_loss - avg_loss) < tol:
                print(f"Converged after {it+1} iterations.")
                # Restore best model state
                if best_model_state is not None:
                    self.load_state_dict(best_model_state)
                break
                
            prev_loss = avg_loss
            
            # Learning rate scheduling
            scheduler.step()
        
        # If training ended without convergence, restore best model state
        if best_model_state is not None and best_loss < loss_history[-1]:
            self.load_state_dict(best_model_state)
            print(f"Training ended. Restored best model state with loss = {best_loss:.6f}")
            loss_history[-1] = best_loss
        
        self.trained = True
        return loss_history, acc_history

    def lbl_train(self, seqs, labels, num_labels, max_iters=1000, tol=1e-8, learning_rate=0.01, 
                  continued=False, decay_rate=1.0, print_every=10, batch_size=32,
                  checkpoint_file=None, checkpoint_interval=10, pos_weight=None):
        """
        Train the model for multi-label classification using binary cross-entropy loss.
        
        Args:
            seqs: List of input sequences for training
            labels: List of binary label vectors (list of lists) or 2D numpy array/torch tensor
            num_labels: Number of labels for multi-label prediction task
            max_iters: Maximum number of training iterations
            tol: Convergence tolerance
            learning_rate: Initial learning rate for optimizer
            continued: Whether to continue training from existing parameters
            decay_rate: Learning rate decay rate
            print_every: Print progress every N iterations
            batch_size: Number of sequences to process in each batch
            checkpoint_file: Path to save training checkpoints
            checkpoint_interval: Save checkpoint every N iterations
            pos_weight: Weight for positive class (torch.Tensor of shape [num_labels])
        
        Returns:
            list: Training loss history
            list: Training accuracy history
        """
        # Initialize label head if not already done
        if self.labeller is None or self.num_labels != num_labels:
            self.labeller = nn.Linear(self.model_dims[-1], num_labels).to(self.device)
            self.num_labels = num_labels
        
        if not continued:
            # Reinitialize model parameters if not continuing
            for layer in self.layers:
                nn.init.uniform_(layer.M, -0.5, 0.5)
                nn.init.uniform_(layer.P, -0.1, 0.1)
                if layer.use_residual == 'separate' and hasattr(layer, 'residual_proj'):
                    if isinstance(layer.residual_proj, nn.Linear):
                        nn.init.uniform_(layer.residual_proj.weight, -0.1, 0.1)
            # Initialize labeller
            nn.init.xavier_uniform_(self.labeller.weight)
            nn.init.zeros_(self.labeller.bias)
        
        # Load checkpoint if continuing and checkpoint file exists
        start_iter = 0
        best_loss = float('inf')
        best_model_state = None
        
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
        else:
            loss_history = []
            acc_history = []
        
        # Convert labels to tensor
        if isinstance(labels, list):
            labels_tensor = torch.tensor(labels, dtype=torch.float32, device=self.device)
        else:
            labels_tensor = torch.as_tensor(labels, dtype=torch.float32, device=self.device)
        
        # Setup loss function with optional positive class weighting
        if pos_weight is not None:
            pos_weight_tensor = torch.tensor(pos_weight, dtype=torch.float32, device=self.device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        else:
            criterion = nn.BCEWithLogitsLoss()
        
        # Setup optimizer and scheduler
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)
        
        # Load optimizer and scheduler states if resuming
        if continued and checkpoint_file and os.path.exists(checkpoint_file):
            optimizer.load_state_dict(optimizer_state)
            scheduler.load_state_dict(scheduler_state)
        
        prev_loss = float('inf') if start_iter == 0 else loss_history[-1] if loss_history else float('inf')
        
        for it in range(start_iter, max_iters):
            total_loss = 0.0
            total_correct = 0
            total_predictions = 0
            total_samples = 0
            
            # Shuffle sequences for each epoch
            indices = list(range(len(seqs)))
            random.shuffle(indices)
            
            # Process sequences in batches
            for batch_start in range(0, len(indices), batch_size):
                batch_indices = indices[batch_start:batch_start + batch_size]
                batch_seqs = [seqs[idx] for idx in batch_indices]
                batch_labels = labels_tensor[batch_indices]
                
                optimizer.zero_grad()
                batch_loss = 0.0
                batch_correct = 0
                batch_predictions = 0
                batch_logits_list = []
                
                # Process each sequence in the batch
                for seq in batch_seqs:
                    # Convert to tensor
                    seq_tensor = torch.tensor(seq, dtype=torch.float32, device=self.device)
                    
                    # Forward pass to get sequence representation
                    seq_representation = self.get_sequence_representation(seq_tensor.unsqueeze(0)).squeeze(0)
                    
                    # Get logits through label head
                    logits = self.labeller(seq_representation.unsqueeze(0))
                    batch_logits_list.append(logits)
                
                # Stack predictions for the batch
                if batch_logits_list:
                    batch_logits = torch.cat(batch_logits_list, dim=0)
                    
                    # Calculate loss for the batch
                    batch_loss = criterion(batch_logits, batch_labels)
                    
                    # Calculate accuracy
                    with torch.no_grad():
                        # Apply sigmoid to get probabilities
                        probs = torch.sigmoid(batch_logits)
                        # Threshold at 0.5 for binary predictions
                        predictions = (probs > 0.5).float()
                        # Calculate number of correct predictions
                        batch_correct = (predictions == batch_labels).sum().item()
                        batch_predictions = batch_labels.numel()
                    
                    # Backpropagate
                    batch_loss.backward()
                    optimizer.step()
                    
                    total_loss += batch_loss.item() * len(batch_seqs)
                    total_correct += batch_correct
                    total_predictions += batch_predictions
                    total_samples += len(batch_seqs)
            
            # Calculate average loss and accuracy for this iteration
            if total_samples > 0:
                avg_loss = total_loss / total_samples
                avg_acc = total_correct / total_predictions if total_predictions > 0 else 0.0
            else:
                avg_loss = 0.0
                avg_acc = 0.0
                
            loss_history.append(avg_loss)
            acc_history.append(avg_acc)
            
            # Update best model state
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_model_state = copy.deepcopy(self.state_dict())
            
            # Print training progress
            if it % print_every == 0 or it == max_iters - 1:
                current_lr = scheduler.get_last_lr()[0]
                print(f"MLC-Train Iter {it:3d}: Loss = {avg_loss:.6e}, Acc = {avg_acc:.4f}, LR = {current_lr:.6f}")
            
            # Save checkpoint at specified intervals
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
                print(f"Checkpoint saved at iteration {it}")
            
            # Check convergence
            if abs(prev_loss - avg_loss) < tol:
                print(f"Converged after {it+1} iterations.")
                # Restore best model state
                if best_model_state is not None:
                    self.load_state_dict(best_model_state)
                break
                
            prev_loss = avg_loss
            
            # Learning rate scheduling
            scheduler.step()
        
        # If training ended without convergence, restore best model state
        if best_model_state is not None and best_loss < loss_history[-1]:
            self.load_state_dict(best_model_state)
            print(f"Training ended. Restored best model state with loss = {best_loss:.6f}")
            loss_history[-1] = best_loss
        
        self.trained = True
        return loss_history, acc_history

    def predict_c(self, seq):
        """
        Predict class label for a sequence using the classification head.
        
        Args:
            seq: Input sequence (list of vectors or numpy array)
        
        Returns:
            tuple: (predicted_class, class_probabilities)
        """
        if self.classifier is None:
            raise ValueError("Model must be trained first for classification")
        
        # Convert to tensor
        if not isinstance(seq, torch.Tensor):
            seq_tensor = torch.tensor(seq, dtype=torch.float32, device=self.device)
        else:
            seq_tensor = seq.to(self.device)
        
        # Add batch dimension if needed
        if seq_tensor.dim() == 2:
            seq_tensor = seq_tensor.unsqueeze(0)
        
        # Get sequence representation
        with torch.no_grad():
            seq_representation = self.get_sequence_representation(seq_tensor).squeeze(0)
            
            # Get logits through classification head
            logits = self.classifier(seq_representation.unsqueeze(0))
            probabilities = torch.softmax(logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            
        return predicted_class, probabilities[0].cpu().numpy()

    def predict_l(self, seq, threshold=0.5):
        """
        Predict multi-label classification for a sequence.
        
        Args:
            seq: Input sequence (list of vectors or numpy array)
            threshold: Probability threshold for binary classification (default: 0.5)
        
        Returns:
            numpy.ndarray: Binary label predictions (0 or 1 for each label)
            numpy.ndarray: Probability scores for each label
        """
        if self.labeller is None:
            raise ValueError("Model must be trained first for multi-label prediction")
        
        # Convert to tensor
        if not isinstance(seq, torch.Tensor):
            seq_tensor = torch.tensor(seq, dtype=torch.float32, device=self.device)
        else:
            seq_tensor = seq.to(self.device)
        
        # Add batch dimension if needed
        if seq_tensor.dim() == 2:
            seq_tensor = seq_tensor.unsqueeze(0)
        
        # Get sequence representation
        with torch.no_grad():
            seq_representation = self.get_sequence_representation(seq_tensor).squeeze(0)
            
            # Get logits through label head
            logits = self.labeller(seq_representation.unsqueeze(0))
            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(logits).cpu().numpy()
        
        # Apply threshold to get binary predictions
        binary_preds = (probs > threshold).astype(np.float32)
        
        return binary_preds[0], probs[0]

    def self_train(self, seqs, max_iters=100, tol=1e-6, learning_rate=0.01, 
              continued=False, self_mode='gap', decay_rate=1.0, print_every=10,
              window_size=None, checkpoint_file=None, checkpoint_interval=5):
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
            window_size: For 'reg' mode, specifies the causal window size. 
                        If None, uses full sequence length (no masking).
                        If integer, uses only previous window_size positions.
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
            self.mean_t = checkpoint.get('mean_t', None)
            self.mean_window_count = checkpoint.get('mean_window_count', None)
            print(f"Resumed self-training from checkpoint at iteration {start_iter}, best loss: {best_loss:.6f}")
        else:
            history = []
        
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
            total_windows = 0
            for seq in seqs_tensor:
                total += torch.sum(seq, dim=0)
                total_windows += seq.size(0)
            self.mean_t = (total / total_windows).cpu().numpy()
            self.mean_window_count = total_windows / len(seqs)
        
        # Calculate total training samples
        total_samples = 0
        for seq in seqs_tensor:
            if self_mode == 'gap':
                total_samples += seq.size(0)  # All positions are samples
            else:  # 'reg' mode
                if window_size is None:
                    total_samples += max(0, seq.size(0) - 1)  # All except last position
                else:
                    # For causal masking, we start from window_size position
                    total_samples += max(0, seq.size(0) - window_size)
        
        if total_samples == 0:
            raise ValueError("No training samples found")
        
        # Setup optimizer
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)
        
        # Load optimizer and scheduler states if resuming
        if continued and checkpoint_file and os.path.exists(checkpoint_file):
            optimizer.load_state_dict(optimizer_state)
            scheduler.load_state_dict(scheduler_state)
        
        prev_loss = float('inf') if start_iter == 0 else history[-1] if history else float('inf')
        
        for it in range(start_iter, max_iters):
            total_loss = 0.0
            
            for seq in seqs_tensor:
                optimizer.zero_grad()
                
                # Forward pass
                batch_size, seq_len, _ = seq.unsqueeze(0).shape
                positions = torch.arange(seq_len, device=self.device)
                
                current = seq.unsqueeze(0)  # Add batch dimension
                
                # Apply causal masking for 'reg' mode if window_size is specified
                if self_mode == 'reg' and window_size is not None:
                    # Create a mask to zero out future positions
                    mask = torch.ones(seq_len, seq_len, device=self.device)
                    mask = torch.tril(mask, diagonal=0)  # Lower triangular matrix
                    mask = mask.unsqueeze(0).unsqueeze(-1)  # Add batch and feature dimensions
                    
                    # Store original input for residual connections
                    original_input = current.clone()
                    
                    # Apply mask to input sequence
                    for i in range(seq_len):
                        # For position i, only keep the last window_size positions
                        start_idx = max(0, i - window_size + 1)
                        current[0, i, :] = original_input[0, i, :]  # Keep current position
                        # Zero out positions beyond the causal window
                        if i >= window_size:
                            current[0, :start_idx, :] = 0
                
                for layer in self.layers:
                    current = layer(current, positions)
                
                # Remove batch dimension
                current = current.squeeze(0)
                
                # Calculate loss based on self_mode
                loss = 0.0
                valid_positions = 0
                
                for k in range(current.size(0)):
                    # Skip positions based on mode and window size
                    if self_mode == 'reg':
                        if window_size is None:
                            # Original behavior: skip last position
                            if k == seq.size(0) - 1:
                                continue
                        else:
                            # Causal masking: skip first (window_size-1) positions
                            if k < window_size - 1:
                                continue
                    
                    # Determine target based on mode
                    if self_mode == 'gap':
                        target = seq[k]
                    else:  # 'reg' mode
                        target = seq[k + 1] if window_size is None else seq[k]
                    
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
            
            # Update best model state if current loss is lower
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_model_state = copy.deepcopy(self.state_dict())
            
            # Update learning rate
            if it % 5 == 0:  # Decay every 5 iterations
                scheduler.step()
            
            # Print progress
            if it % print_every == 0 or it == max_iters - 1:
                current_lr = scheduler.get_last_lr()[0]
                mode_display = "Gap" if self_mode == 'gap' else "Reg"
                window_info = f", Window: {window_size}" if self_mode == 'reg' and window_size is not None else ""
                print(f"SelfTrain({mode_display}{window_info}) Iter {it:3d}: loss = {avg_loss:.6f}, LR = {current_lr:.6f}")
            
            # Save checkpoint at specified intervals
            if checkpoint_file and (it % checkpoint_interval == 0 or it == max_iters - 1):
                self._save_checkpoint(
                    checkpoint_file, it, history, optimizer, scheduler, best_loss,
                    self_mode=self_mode, window_size=window_size
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
                current_seq = torch.cat([current_seq[1:], output_vector.unsqueeze(0)])
        
        return np.array(reconstructed)

    def _save_checkpoint(self, checkpoint_file, iteration, history, optimizer, scheduler, 
                         best_loss, **kwargs):
        """
        Save training checkpoint with complete training state
        
        Args:
            checkpoint_file: Path to save checkpoint file
            iteration: Current training iteration
            history: Training loss history
            optimizer: Optimizer instance
            scheduler: Learning rate scheduler instance
            best_loss: Best loss achieved so far
            **kwargs: Additional training parameters to save
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
                'num_basis_list': self.num_basis_list,
                'use_residual_list': self.use_residual_list,
                'num_classes': self.num_classes,
                'num_labels': self.num_labels
            },
            'training_stats': {
                'mean_t': self.mean_t if hasattr(self, 'mean_t') else None,
                'mean_window_count': self.mean_window_count if hasattr(self, 'mean_window_count') else None
            },
            'training_params': kwargs
        }
        torch.save(checkpoint, checkpoint_file)
        #print(f"Checkpoint saved at iteration {iteration} to {checkpoint_file}")

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
        
        # Count classifier parameters if exists
        if self.classifier is not None:
            layer_total = 0
            for name, param in self.classifier.named_parameters():
                if param.requires_grad:
                    num = param.numel()
                    shape = str(tuple(param.shape))
                    print(f"{'Classifier':<10} | {name:<20} | {num:<15} | {shape:<20}")
                    layer_total += num
                    total_params += num
            layer_params.append(('Classifier', layer_total))
        
        # Count labeller parameters if exists
        if self.labeller is not None:
            layer_total = 0
            for name, param in self.labeller.named_parameters():
                if param.requires_grad:
                    num = param.numel()
                    shape = str(tuple(param.shape))
                    print(f"{'Labeller':<10} | {name:<20} | {num:<15} | {shape:<20}")
                    layer_total += num
                    total_params += num
            layer_params.append(('Labeller', layer_total))
        
        # Print summary by layer
        print("-"*60)
        for name, count in layer_params:
            print(f"{name} total parameters: {count}")
        
        # Print grand total
        print("="*60)
        print(f"Total trainable parameters: {total_params}")
        print("="*60)
        
        return total_params
    
    def save(self, filename):
        """Save model state to file"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': {
                'input_dim': self.input_dim,
                'model_dims': self.model_dims,
                'num_basis_list': self.num_basis_list,
                'use_residual_list': self.use_residual_list,
                'num_classes': self.num_classes,
                'num_labels': self.num_labels
            },
            'training_stats': {
                'mean_t': self.mean_t if hasattr(self, 'mean_t') else None,
                'mean_window_count': self.mean_window_count if hasattr(self, 'mean_window_count') else None
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
            num_basis_list=config['num_basis_list'],
            use_residual_list=config.get('use_residual_list', None),
            device=device
        )
        
        # Load state
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load additional attributes
        if 'training_stats' in checkpoint:
            stats = checkpoint['training_stats']
            if stats['mean_t'] is not None:
                model.mean_t = stats['mean_t']
            if stats['mean_window_count'] is not None:
                model.mean_window_count = stats['mean_window_count']
        
        # Load classification heads if they exist in config
        if 'num_classes' in config and config['num_classes'] is not None:
            model.num_classes = config['num_classes']
            model.classifier = nn.Linear(model.model_dims[-1], model.num_classes).to(device)
        
        if 'num_labels' in config and config['num_labels'] is not None:
            model.num_labels = config['num_labels']
            model.labeller = nn.Linear(model.model_dims[-1], model.num_labels).to(device)
            
        model.trained = True
        print(f"Model loaded from {filename}")
        return model    
    

# === Example Usage ===
if __name__ == "__main__":

    from statistics import correlation
    
    # Set random seeds for reproducibility
    torch.manual_seed(2)
    random.seed(2)
    np.random.seed(2)
    
    # Configuration
    input_dim = 10          # Input vector dimension
    model_dims = [8, 6, 3]  # Output dimensions for each layer
    num_basis_list = [5, 4, 3]  # Basis functions per layer
    use_residual_list = ['separate'] * 3 
    num_seqs = 100          # Number of training sequences
    min_len, max_len = 100, 200  # Sequence length range
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print(f"Model config: input_dim={input_dim}, model_dims={model_dims}, num_basis={num_basis_list}, use_residual={use_residual_list}")
    
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

    # Create model with shared M matrix
    print("\nCreating Hierarchical HierDDts model...")
    model = HierDDts(
        input_dim=input_dim,
        model_dims=model_dims,
        num_basis_list=num_basis_list,
        use_residual_list=use_residual_list,
        device=device
    )
    model.count_parameters()
    
    # Train model
    print("\nTraining model...")
    history = model.reg_train(
        seqs, 
        t_list,
        max_iters=100,
        lr=0.1,
        decay_rate=0.97,
        print_every=10,
        continued=False,
        checkpoint_file='gd_checkpoint.pth',
        checkpoint_interval=10
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
        corr = correlation(actual, predicted)
        correlations.append(corr)
        print(f"Output dim {i} correlation: {corr:.4f}")
    
    avg_corr = np.mean(correlations)
    print(f"\nAverage correlation: {avg_corr:.4f}")
    
    # Save and load model example
    print("\n" + "="*50)
    print("MODEL SAVE AND LOAD EXAMPLE")
    print("="*50)
    
    # Save the trained model
    model.save("HierDDts_model.pth")
    
    # Load the model
    loaded_model = HierDDts.load("HierDDts_model.pth", device=device)
    
    # Verify loaded model works correctly
    print("\nTesting loaded model...")
    test_seq = seqs[0]
    test_seq_tensor = torch.tensor(test_seq, dtype=torch.float32, device=device)
    
    # Original model prediction
    original_pred = model.predict_t(test_seq_tensor.unsqueeze(0)).squeeze(0).detach().cpu().numpy()
    
    # Loaded model prediction
    loaded_pred = loaded_model.predict_t(test_seq_tensor.unsqueeze(0)).squeeze(0).detach().cpu().numpy()
    
    print(f"Original model prediction: {[f'{x:.6f}' for x in original_pred]}")
    print(f"Loaded model prediction:   {[f'{x:.6f}' for x in loaded_pred]}")
    
    # Check if predictions are identical
    prediction_diff = np.max(np.abs(original_pred - loaded_pred))
    print(f"Maximum prediction difference: {prediction_diff:.6e}")
    
    if prediction_diff < 1e-6:
        print("✓ Model save/load test PASSED - predictions are identical")
    else:
        print("✗ Model save/load test FAILED - predictions differ")
    
    # Classification task example
    print("\n" + "="*50)
    print("CLASSIFICATION TASK EXAMPLE")
    print("="*50)
    
    # Generate classification data
    num_classes = 3
    class_seqs = []
    class_labels = []
    
    # Create vector sequences with different patterns for each class
    for class_id in range(num_classes):
        for _ in range(50):  # 50 sequences per class
            L = random.randint(150, 250)
            if class_id == 0:
                # Class 0: Vectors with positive mean
                seq = [[random.uniform(0.5, 1.5) for _ in range(input_dim)] for __ in range(L)]
            elif class_id == 1:
                # Class 1: Vectors with negative mean
                seq = [[random.uniform(-1.5, -0.5) for _ in range(input_dim)] for __ in range(L)]
            else:
                # Class 2: Standard random vectors
                seq = [[random.uniform(-1, 1) for _ in range(input_dim)] for __ in range(L)]
            
            class_seqs.append(seq)
            class_labels.append(class_id)
    
    # Create a new model for classification
    cls_model = HierDDts(
        input_dim=input_dim,
        model_dims=[10, 20, 10],
        num_basis_list=num_basis_list,
        use_residual_list=use_residual_list,
        device=device
    )
    
    # Train for classification
    print("\nStarting Classification Training...")
    loss_history, acc_history = cls_model.cls_train(
        class_seqs, 
        class_labels, 
        num_classes,
        max_iters=30,
        learning_rate=0.05,
        decay_rate=0.99,
        batch_size=16,
        print_every=5
    )
    
    # Show prediction results
    print("\nClassification Prediction Results:")
    correct = 0
    for i in range(min(5, len(class_seqs))):
        pred_class, probs = cls_model.predict_c(class_seqs[i])
        true_label = class_labels[i]
        
        if pred_class == true_label:
            correct += 1
        
        print(f"Seq {i+1}: True={true_label}, Pred={pred_class}, Probs={[f'{p:.3f}' for p in probs]}")
    
    # Test on all sequences
    print("\nTesting on all sequences...")
    all_correct = 0
    for seq, true_label in zip(class_seqs, class_labels):
        pred_class, _ = cls_model.predict_c(seq)
        if pred_class == true_label:
            all_correct += 1
    
    accuracy = all_correct / len(class_seqs)
    print(f"Overall accuracy: {accuracy:.4f} ({all_correct}/{len(class_seqs)})")
    
    # Multi-label classification example
    print("\n" + "="*50)
    print("MULTI-LABEL CLASSIFICATION EXAMPLE")
    print("="*50)
    
    # Generate multi-label data
    num_labels = 4  # Example: 4 different labels
    label_seqs = []
    labels = []
    
    for _ in range(100):
        L = random.randint(150, 250)
        seq = [[random.uniform(-1, 1) for _ in range(input_dim)] for __ in range(L)]
        label_seqs.append(seq)
        
        # Create random binary labels (multi-label classification)
        # Each sequence can have 0-4 active labels
        label_vec = [random.random() > 0.7 for _ in range(num_labels)]
        labels.append([1.0 if x else 0.0 for x in label_vec])
    
    # Create a new model for multi-label classification
    lbl_model = HierDDts(
        input_dim=input_dim,
        model_dims=[10, 20, 10],
        num_basis_list=num_basis_list,
        use_residual_list=use_residual_list,
        device=device
    )
    
    # Train for multi-label classification
    print("\nStarting Multi-Label Classification Training...")
    loss_history, acc_history = lbl_model.lbl_train(
        label_seqs,
        labels,
        num_labels,
        max_iters=30,
        learning_rate=0.05,
        decay_rate=0.99,
        batch_size=16,
        print_every=5
    )
    
    # Show prediction results
    print("\nMulti-Label Prediction Results:")
    label_names = ["Label_A", "Label_B", "Label_C", "Label_D"]
    
    # Test on a few sequences
    for i in range(min(3, len(label_seqs))):
        binary_pred, probs_pred = lbl_model.predict_l(label_seqs[i], threshold=0.5)
        true_labels = labels[i]
        
        print(f"\nSequence {i+1}:")
        print(f"True labels: {true_labels}")
        print(f"Predicted binary: {binary_pred}")
        print(f"Predicted probabilities: {[f'{p:.4f}' for p in probs_pred]}")
        
        # Interpret the predictions
        print("Label interpretation:")
        for j, (true, pred, prob) in enumerate(zip(true_labels, binary_pred, probs_pred)):
            status = "ACTIVE" if pred > 0.5 else "INACTIVE"
            correct = "✓" if (pred > 0.5) == (true > 0.5) else "✗"
            print(f"  {label_names[j]}: {status} (prob: {prob:.4f}) {correct}")
    
    # Self-training example
    print("\n" + "="*50)
    print("SELF-TRAINING AND RECONSTRUCTION EXAMPLE")
    print("="*50)
    
    # Create a new model for self-training
    self_model = HierDDts(
        input_dim=input_dim,
        model_dims=[10, 20, 10],
        num_basis_list=num_basis_list,
        use_residual_list=use_residual_list,
        device=device
    )
    
    # Prepare sequences for self-training (use shorter sequences)
    self_seqs = []
    for i in range(50):  # Fewer sequences for self-training
        seq_len = 100  # Fixed sequence length for self-training
        seq = [[random.uniform(-1, 1) for _ in range(input_dim)] for __ in range(seq_len)]
        self_seqs.append(seq)
    
    # Self-train in gap mode (self-consistency)
    print("\nSelf-training in gap mode...")
    self_history = self_model.self_train(
        self_seqs,
        max_iters=50,
        learning_rate=0.01,
        self_mode='gap',
        print_every=5,
        checkpoint_file='self_checkpoint.pth',
        checkpoint_interval=10
    )
    
    # Save self-trained model
    self_model.save("self_trained_model.pth")
    
    # Load self-trained model
    loaded_self_model = HierDDts.load("self_trained_model.pth", device=device)

    # Reconstruct new sequences with both models
    print("\nReconstructing sequences with original and loaded models...")
    
    # Set random seeds to ensure identical reconstruction
    torch.manual_seed(2)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(2)
    np.random.seed(2)
    random.seed(2)
    reconstructed_original = self_model.reconstruct(L=10, tau=0.1, discrete_mode=False)

    # Reset random seeds again for loaded model
    torch.manual_seed(2)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(2)
    np.random.seed(2)
    random.seed(2)
    reconstructed_loaded = loaded_self_model.reconstruct(L=10, tau=0.1, discrete_mode=False)
    
    print(f"Original model reconstructed shape: {reconstructed_original.shape}")
    print(f"Loaded model reconstructed shape: {reconstructed_loaded.shape}")
    
    # Check if reconstructions are identical
    reconstruction_diff = np.max(np.abs(reconstructed_original - reconstructed_loaded))
    print(f"Maximum reconstruction difference: {reconstruction_diff:.6e}")
    
    if reconstruction_diff < 1e-6:
        print("✓ Self-trained model save/load test PASSED - reconstructions are identical")
    else:
        print("✗ Self-trained model save/load test FAILED - reconstructions differ")
    
    # Reconstruct using reg-trained model
    print("\nReconstructing with reg-trained model...")
    self_model_reg = HierDDts(
        input_dim=input_dim,
        model_dims=[10, 20, 10],
        num_basis_list=num_basis_list,
        use_residual_list=use_residual_list,
        device=device
    )
    
    self_history_reg = self_model_reg.self_train(
        self_seqs,
        max_iters=20,
        learning_rate=0.01,
        self_mode='reg',
        window_size=10,
        print_every=5,
        checkpoint_file='reg_checkpoint.pth',
        checkpoint_interval=5
    )
    
    # Reconstruct using reg-trained model
    reconstructed_reg = self_model_reg.reconstruct(
        L=30,
        tau=0.05,
        discrete_mode=False
    )
    
    print(f"Reg-reconstructed sequence shape: {reconstructed_reg.shape}")
    print("\nTest completed successfully!")
    print("="*50)
