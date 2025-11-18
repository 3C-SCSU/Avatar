"""
GaussianNB Naive Bayes Classifier - PyTorch Implementation
This module implements Gaussian Naive Bayes using PyTorch for brain wave classification.
"""

import torch
import torch.nn as nn
import numpy as np


class GaussianNB(nn.Module):
    """
    Gaussian Naive Bayes classifier implemented in PyTorch.
    
    This implementation calculates mean (μ) and variance (σ²) for each feature
    per class during training, then uses the Gaussian Probability Density Function
    and Bayes' theorem for prediction.
    
    No traditional backpropagation is used - parameters are calculated directly
    from the training data.
    """
    
    def __init__(self, num_features, num_classes):
        """
        Initialize the GaussianNB model.
        
        Args:
            num_features (int): Number of input features (e.g., 32 for EEG channels)
            num_classes (int): Number of output classes (e.g., 6 for drone commands)
        """
        super(GaussianNB, self).__init__()
        
        self.num_features = num_features
        self.num_classes = num_classes
        
        # Register buffers (non-trainable parameters)
        # These will store the calculated statistics
        self.register_buffer('class_priors', torch.zeros(num_classes))
        self.register_buffer('means', torch.zeros(num_classes, num_features))
        self.register_buffer('variances', torch.zeros(num_classes, num_features))
        self.register_buffer('is_fitted', torch.tensor(False))
        
    def fit(self, X, y):
        """
        Fit the Gaussian Naive Bayes model.
        
        This calculates the mean and variance for each feature within each class.
        
        Args:
            X (torch.Tensor): Training data of shape (n_samples, n_features)
            y (torch.Tensor): Target labels of shape (n_samples,)
        """
        # Ensure tensors are on the same device as the model
        X = X.to(self.means.device)
        y = y.to(self.means.device).long()
        
        n_samples = X.shape[0]
        
        # Calculate class priors P(y)
        for c in range(self.num_classes):
            class_mask = (y == c)
            self.class_priors[c] = class_mask.sum().float() / n_samples
            
            # Get samples for this class
            X_c = X[class_mask]
            
            if len(X_c) > 0:
                # Calculate mean μ for each feature
                self.means[c] = X_c.mean(dim=0)
                
                # Calculate variance σ² for each feature
                # Add small epsilon to avoid division by zero
                self.variances[c] = X_c.var(dim=0) + 1e-9
        
        self.is_fitted = torch.tensor(True)
        
    def forward(self, X):
        """
        Make predictions using the Gaussian PDF and Bayes' theorem.
        
        Args:
            X (torch.Tensor): Input data of shape (n_samples, n_features)
            
        Returns:
            torch.Tensor: Predicted class labels of shape (n_samples,)
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before making predictions")
        
        X = X.to(self.means.device)
        n_samples = X.shape[0]
        
        # Calculate log probabilities for numerical stability
        log_probs = torch.zeros(n_samples, self.num_classes, device=X.device)
        
        for c in range(self.num_classes):
            # Log of class prior: log(P(y=c))
            log_prior = torch.log(self.class_priors[c] + 1e-10)
            
            # Gaussian PDF: P(x|y=c) = (1/sqrt(2πσ²)) * exp(-(x-μ)²/(2σ²))
            # Log PDF: log(P(x|y=c)) = -0.5 * [log(2πσ²) + (x-μ)²/σ²]
            
            # Calculate (x - μ)²
            diff_sq = (X - self.means[c]).pow(2)
            
            # Calculate log likelihood: sum over all features (independence assumption)
            log_likelihood = -0.5 * torch.sum(
                torch.log(2 * np.pi * self.variances[c]) + diff_sq / self.variances[c],
                dim=1
            )
            
            # Posterior: log(P(y=c|x)) ∝ log(P(x|y=c)) + log(P(y=c))
            log_probs[:, c] = log_likelihood + log_prior
        
        # Return class with highest posterior probability
        return torch.argmax(log_probs, dim=1)
    
    def predict_proba(self, X):
        """
        Predict class probabilities.
        
        Args:
            X (torch.Tensor): Input data of shape (n_samples, n_features)
            
        Returns:
            torch.Tensor: Class probabilities of shape (n_samples, n_classes)
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before making predictions")
        
        X = X.to(self.means.device)
        n_samples = X.shape[0]
        
        log_probs = torch.zeros(n_samples, self.num_classes, device=X.device)
        
        for c in range(self.num_classes):
            log_prior = torch.log(self.class_priors[c] + 1e-10)
            diff_sq = (X - self.means[c]).pow(2)
            log_likelihood = -0.5 * torch.sum(
                torch.log(2 * np.pi * self.variances[c]) + diff_sq / self.variances[c],
                dim=1
            )
            log_probs[:, c] = log_likelihood + log_prior
        
        # Convert log probabilities to probabilities using softmax
        probs = torch.softmax(log_probs, dim=1)
        return probs