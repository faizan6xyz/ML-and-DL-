"""
Advanced Radial Basis Function Network (RBFN) Implementation

REQUIREMENTS:
1. Support multiple RBF kernels: Gaussian, Multiquadric, Inverse Multiquadric
2. Implement K-means clustering for center initialization
3. Support both regression and binary classification tasks
4. Include regularization (L1, L2) to prevent overfitting
5. Efficient batch training with configurable learning rates
6. Prediction with uncertainty estimation using softmax/sigmoid
7. Support for weighted hidden layer activation
8. Cross-validation support for hyperparameter tuning
9. Model persistence (save/load) functionality
10. Performance metrics: MSE, MAE, Accuracy, Precision, Recall, F1-Score
"""

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, 
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
from sklearn.model_selection import cross_val_score
import pickle
from typing import Tuple, Optional, Union, List
import warnings


class RBFKernel:
    """
    Radial Basis Function kernel implementations
    
    Supports multiple RBF kernels for flexibility in non-linear transformations
    """
    
    @staticmethod
    def gaussian(r: np.ndarray, sigma: float) -> np.ndarray:
        """
        Gaussian (RBF) kernel
        phi(r) = exp(-r^2 / (2*sigma^2))
        """
        return np.exp(-np.square(r) / (2 * sigma ** 2))
    
    @staticmethod
    def multiquadric(r: np.ndarray, c: float) -> np.ndarray:
        """
        Multiquadric kernel
        phi(r) = sqrt(r^2 + c^2)
        """
        return np.sqrt(np.square(r) + c ** 2)
    
    @staticmethod
    def inverse_multiquadric(r: np.ndarray, c: float) -> np.ndarray:
        """
        Inverse Multiquadric kernel
        phi(r) = 1 / sqrt(r^2 + c^2)
        """
        return 1.0 / np.sqrt(np.square(r) + c ** 2)
    
    @staticmethod
    def thin_plate_spline(r: np.ndarray) -> np.ndarray:
        """
        Thin Plate Spline kernel
        phi(r) = r^2 * log(r) for r > 0
        """
        result = np.zeros_like(r, dtype=float)
        mask = r > 0
        result[mask] = np.square(r[mask]) * np.log(r[mask])
        return result


class AdvancedRBFN:
    """
    Advanced Radial Basis Function Network for regression and classification
    
    Features:
    - Multiple RBF kernel support
    - K-means center initialization
    - Adaptive learning rate
    - L1/L2 regularization
    - Uncertainty estimation
    - Cross-validation support
    - Model serialization
    """
    
    def __init__(
        self,
        n_centers: int = 10,
        kernel: str = 'gaussian',
        sigma: float = 1.0,
        learning_rate: float = 0.01,
        regularization: str = 'l2',
        lambda_reg: float = 0.001,
        max_iterations: int = 1000,
        batch_size: int = 32,
        task: str = 'regression',
        random_state: Optional[int] = None,
        verbose: bool = False
    ):
        """
        Initialize Advanced RBFN
        
        Parameters:
        -----------
        n_centers : int
            Number of RBF centers (hidden units)
        kernel : str
            Type of RBF kernel: 'gaussian', 'multiquadric', 'inverse_multiquadric', 'thin_plate_spline'
        sigma : float
            Width parameter for Gaussian kernel
        learning_rate : float
            Initial learning rate for weight updates
        regularization : str
            Regularization type: 'l1', 'l2', or None
        lambda_reg : float
            Regularization strength
        max_iterations : int
            Maximum training iterations
        batch_size : int
            Size of mini-batches for training
        task : str
            'regression' or 'classification'
        random_state : int
            Random seed for reproducibility
        verbose : bool
            Print training progress
        """
        self.n_centers = n_centers
        self.kernel = kernel
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.lambda_reg = lambda_reg
        self.max_iterations = max_iterations
        self.batch_size = batch_size
        self.task = task
        self.random_state = random_state
        self.verbose = verbose
        
        self.centers = None
        self.weights = None
        self.bias = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.is_fitted = False
        self.training_loss = []
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def _initialize_centers(self, X: np.ndarray) -> None:
        """
        Initialize RBF centers using K-means clustering
        
        Parameters:
        -----------
        X : np.ndarray
            Input training data
        """
        kmeans = KMeans(
            n_clusters=self.n_centers,
            random_state=self.random_state,
            n_init=10
        )
        kmeans.fit(X)
        self.centers = kmeans.cluster_centers_
    
    def _compute_rbf_activation(self, X: np.ndarray) -> np.ndarray:
        """
        Compute RBF activation matrix
        
        Parameters:
        -----------
        X : np.ndarray
            Input data of shape (n_samples, n_features)
        
        Returns:
        --------
        np.ndarray
            RBF activation matrix of shape (n_samples, n_centers)
        """
        # Compute Euclidean distances from inputs to centers
        distances = cdist(X, self.centers, metric='euclidean')
        
        # Apply RBF kernel
        if self.kernel == 'gaussian':
            activations = RBFKernel.gaussian(distances, self.sigma)
        elif self.kernel == 'multiquadric':
            activations = RBFKernel.multiquadric(distances, self.sigma)
        elif self.kernel == 'inverse_multiquadric':
            activations = RBFKernel.inverse_multiquadric(distances, self.sigma)
        elif self.kernel == 'thin_plate_spline':
            activations = RBFKernel.thin_plate_spline(distances)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")
        
        # Add bias term (always 1)
        activations = np.hstack([activations, np.ones((X.shape[0], 1))])
        
        return activations
    
    def _apply_regularization(self, weights: np.ndarray) -> float:
        """
        Calculate regularization penalty
        
        Parameters:
        -----------
        weights : np.ndarray
            Weight matrix
        
        Returns:
        --------
        float
            Regularization penalty value
        """
        if self.regularization is None:
            return 0.0
        elif self.regularization == 'l1':
            return self.lambda_reg * np.sum(np.abs(weights))
        elif self.regularization == 'l2':
            return self.lambda_reg * np.sum(np.square(weights))
        else:
            raise ValueError(f"Unknown regularization: {self.regularization}")
    
    def _apply_regularization_gradient(self, weights: np.ndarray) -> np.ndarray:
        """
        Calculate regularization gradient for backpropagation
        
        Parameters:
        -----------
        weights : np.ndarray
            Weight matrix
        
        Returns:
        --------
        np.ndarray
            Regularization gradient
        """
        if self.regularization is None:
            return 0.0
        elif self.regularization == 'l1':
            return self.lambda_reg * np.sign(weights)
        elif self.regularization == 'l2':
            return 2 * self.lambda_reg * weights
        else:
            return 0.0
    
    def _forward_pass(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass through the network
        
        Parameters:
        -----------
        X : np.ndarray
            Input data
        
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            (RBF activations, network output)
        """
        activations = self._compute_rbf_activation(X)
        
        # Linear combination with weights
        output = np.dot(activations, self.weights.T)
        
        return activations, output
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'AdvancedRBFN':
        """
        Train the RBFN model
        
        Parameters:
        -----------
        X : np.ndarray
            Input training data of shape (n_samples, n_features)
        y : np.ndarray
            Target values of shape (n_samples,) or (n_samples, 1)
        
        Returns:
        --------
        AdvancedRBFN
            Fitted model
        """
        # Ensure proper shapes
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1, 1)
        
        # Standardize inputs and outputs
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y)
        
        # Initialize centers
        self._initialize_centers(X_scaled)
        
        # Compute initial RBF activations
        activations = self._compute_rbf_activation(X_scaled)
        n_features = activations.shape[1]
        
        # Initialize weights using pseudo-inverse (fast initialization)
        self.weights = np.linalg.lstsq(activations, y_scaled, rcond=None)[0].T
        self.bias = self.weights[:, -1]
        
        self.training_loss = []
        
        # Mini-batch gradient descent with adaptive learning rate
        n_samples = X_scaled.shape[0]
        n_batches = max(1, n_samples // self.batch_size)
        
        for iteration in range(self.max_iterations):
            epoch_loss = 0.0
            
            # Shuffle data
            indices = np.random.permutation(n_samples)
            
            for batch_idx in range(n_batches):
                # Get batch
                batch_indices = indices[batch_idx * self.batch_size:(batch_idx + 1) * self.batch_size]
                X_batch = X_scaled[batch_indices]
                y_batch = y_scaled[batch_indices]
                
                # Forward pass
                activations_batch, output = self._forward_pass(X_batch)
                
                # Compute loss
                error = output - y_batch
                mse = np.mean(np.square(error)) / 2
                reg_penalty = self._apply_regularization(self.weights)
                loss = mse + reg_penalty
                
                # Backward pass
                gradient = np.dot(error.T, activations_batch) / len(batch_indices)
                reg_grad = self._apply_regularization_gradient(self.weights)
                gradient += reg_grad
                
                # Update weights
                self.weights -= self.learning_rate * gradient
                
                epoch_loss += loss
            
            avg_loss = epoch_loss / n_batches
            self.training_loss.append(avg_loss)
            
            # Adaptive learning rate decay
            if iteration > 0 and iteration % 50 == 0:
                self.learning_rate *= 0.95
            
            if self.verbose and (iteration + 1) % 100 == 0:
                print(f"Iteration {iteration + 1}/{self.max_iterations}, Loss: {avg_loss:.6f}")
        
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions
        
        Parameters:
        -----------
        X : np.ndarray
            Input data of shape (n_samples, n_features)
        
        Returns:
        --------
        np.ndarray
            Predicted values
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X = np.asarray(X, dtype=np.float32)
        X_scaled = self.scaler_X.transform(X)
        
        _, output = self._forward_pass(X_scaled)
        predictions = self.scaler_y.inverse_transform(output)
        
        if self.task == 'classification':
            predictions = (predictions > 0.5).astype(int).flatten()
        
        return predictions.flatten()
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities (for classification tasks)
        
        Parameters:
        -----------
        X : np.ndarray
            Input data
        
        Returns:
        --------
        np.ndarray
            Probability estimates
        """
        if self.task != 'classification':
            raise ValueError("predict_proba only available for classification tasks")
        
        X = np.asarray(X, dtype=np.float32)
        X_scaled = self.scaler_X.transform(X)
        
        _, output = self._forward_pass(X_scaled)
        output_rescaled = self.scaler_y.inverse_transform(output)
        
        # Apply sigmoid for probability estimation
        proba = 1.0 / (1.0 + np.exp(-output_rescaled))
        
        return np.hstack([1 - proba, proba])
    
    def predict_uncertainty(self, X: np.ndarray) -> np.ndarray:
        """
        Estimate prediction uncertainty
        
        Parameters:
        -----------
        X : np.ndarray
            Input data
        
        Returns:
        --------
        np.ndarray
            Uncertainty estimates (standard deviation approximation)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before estimating uncertainty")
        
        X = np.asarray(X, dtype=np.float32)
        X_scaled = self.scaler_X.transform(X)
        
        # Use distance to nearest center as uncertainty proxy
        distances = cdist(X_scaled, self.centers, metric='euclidean')
        min_distances = np.min(distances, axis=1)
        
        # Normalize uncertainty
        uncertainty = min_distances / np.max(min_distances)
        
        return uncertainty
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Evaluate model performance
        
        Parameters:
        -----------
        X : np.ndarray
            Input data
        y : np.ndarray
            True target values
        
        Returns:
        --------
        dict
            Performance metrics
        """
        predictions = self.predict(X)
        y = np.asarray(y).flatten()
        
        metrics = {}
        
        if self.task == 'regression':
            metrics['mse'] = mean_squared_error(y, predictions)
            metrics['mae'] = mean_absolute_error(y, predictions)
            metrics['rmse'] = np.sqrt(metrics['mse'])
        else:
            metrics['accuracy'] = accuracy_score(y, predictions)
            metrics['precision'] = precision_score(y, predictions)
            metrics['recall'] = recall_score(y, predictions)
            metrics['f1'] = f1_score(y, predictions)
            metrics['confusion_matrix'] = confusion_matrix(y, predictions)
        
        return metrics
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray, cv: int = 5) -> dict:
        """
        Perform cross-validation
        
        Parameters:
        -----------
        X : np.ndarray
            Input data
        y : np.ndarray
            Target values
        cv : int
            Number of folds
        
        Returns:
        --------
        dict
            Cross-validation results
        """
        from sklearn.model_selection import cross_validate, KFold
        
        if self.task == 'regression':
            scoring = {
                'mse': 'neg_mean_squared_error',
                'mae': 'neg_mean_absolute_error'
            }
        else:
            scoring = {
                'accuracy': 'accuracy',
                'precision': 'precision',
                'recall': 'recall',
                'f1': 'f1'
            }
        
        cv_splitter = KFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        results = cross_validate(self, X, y, cv=cv_splitter, scoring=scoring, return_train_score=True)
        
        return results
    
    def save(self, filepath: str) -> None:
        """
        Save model to disk
        
        Parameters:
        -----------
        filepath : str
            Path to save the model
        """
        model_data = {
            'centers': self.centers,
            'weights': self.weights,
            'bias': self.bias,
            'scaler_X': self.scaler_X,
            'scaler_y': self.scaler_y,
            'n_centers': self.n_centers,
            'kernel': self.kernel,
            'sigma': self.sigma,
            'task': self.task,
            'is_fitted': self.is_fitted
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    @classmethod
    def load(cls, filepath: str) -> 'AdvancedRBFN':
        """
        Load model from disk
        
        Parameters:
        -----------
        filepath : str
            Path to load the model
        
        Returns:
        --------
        AdvancedRBFN
            Loaded model
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        model = cls(
            n_centers=model_data['n_centers'],
            kernel=model_data['kernel'],
            sigma=model_data['sigma'],
            task=model_data['task']
        )
        
        model.centers = model_data['centers']
        model.weights = model_data['weights']
        model.bias = model_data['bias']
        model.scaler_X = model_data['scaler_X']
        model.scaler_y = model_data['scaler_y']
        model.is_fitted = model_data['is_fitted']
        
        return model


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example 1: Regression Task
    print("="*70)
    print("EXAMPLE 1: REGRESSION TASK")
    print("="*70)
    
    # Generate synthetic regression data
    np.random.seed(42)
    X_train = np.random.uniform(-5, 5, (100, 2))
    y_train = np.sin(X_train[:, 0]) + 0.5 * X_train[:, 1] + np.random.normal(0, 0.1, 100)
    
    X_test = np.random.uniform(-5, 5, (30, 2))
    y_test = np.sin(X_test[:, 0]) + 0.5 * X_test[:, 1]
    
    # Create and train model
    rbfn_reg = AdvancedRBFN(
        n_centers=15,
        kernel='gaussian',
        sigma=1.0,
        learning_rate=0.01,
        regularization='l2',
        lambda_reg=0.001,
        max_iterations=500,
        task='regression',
        verbose=True
    )
    
    rbfn_reg.fit(X_train, y_train)
    
    # Predictions and evaluation
    y_pred = rbfn_reg.predict(X_test)
    metrics = rbfn_reg.evaluate(X_test, y_test)
    
    print(f"\nRegression Metrics:")
    print(f"MSE: {metrics['mse']:.6f}")
    print(f"MAE: {metrics['mae']:.6f}")
    print(f"RMSE: {metrics['rmse']:.6f}")
    
    # Uncertainty estimation
    uncertainty = rbfn_reg.predict_uncertainty(X_test[:5])
    print(f"\nUncertainty estimates for first 5 samples: {uncertainty[:5]}")
    
    # Example 2: Classification Task
    print("\n" + "="*70)
    print("EXAMPLE 2: CLASSIFICATION TASK")
    print("="*70)
    
    # Generate synthetic classification data
    from sklearn.datasets import make_classification
    
    X_train, y_train = make_classification(
        n_samples=150, n_features=2, n_informative=2,
        n_redundant=0, random_state=42
    )
    X_test, y_test = make_classification(
        n_samples=50, n_features=2, n_informative=2,
        n_redundant=0, random_state=123
    )
    
    # Create and train model
    rbfn_clf = AdvancedRBFN(
        n_centers=12,
        kernel='gaussian',
        sigma=0.8,
        learning_rate=0.01,
        regularization='l2',
        lambda_reg=0.001,
        max_iterations=500,
        task='classification',
        verbose=True
    )
    
    rbfn_clf.fit(X_train, y_train)
    
    # Predictions and evaluation
    y_pred = rbfn_clf.predict(X_test)
    metrics = rbfn_clf.evaluate(X_test, y_test)
    
    print(f"\nClassification Metrics:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1']:.4f}")
    print(f"Confusion Matrix:\n{metrics['confusion_matrix']}")
    
    # Probability predictions
    proba = rbfn_clf.predict_proba(X_test[:5])
    print(f"\nProbability predictions for first 5 samples:\n{proba}")
    
    # Example 3: Cross-Validation
    print("\n" + "="*70)
    print("EXAMPLE 3: CROSS-VALIDATION")
    print("="*70)
    
    rbfn_cv = AdvancedRBFN(
        n_centers=10,
        kernel='gaussian',
        sigma=0.8,
        task='classification'
    )
    
    cv_results = rbfn_cv.cross_validate(X_train, y_train, cv=5)
    print(f"\nCross-Validation Results:")
    print(f"Mean Accuracy: {-cv_results['test_accuracy'].mean():.4f} (+/- {cv_results['test_accuracy'].std():.4f})")
    print(f"Mean F1-Score: {-cv_results['test_f1'].mean():.4f} (+/- {cv_results['test_f1'].std():.4f})")
    
    # Example 4: Model Persistence
    print("\n" + "="*70)
    print("EXAMPLE 4: MODEL PERSISTENCE")
    print("="*70)
    
    rbfn_reg.save('rbfn_model.pkl')
    print("Model saved to 'rbfn_model.pkl'")
    
    loaded_model = AdvancedRBFN.load('rbfn_model.pkl')
    y_pred_loaded = loaded_model.predict(X_test[:5])
    print(f"Predictions from loaded model: {y_pred_loaded}")
