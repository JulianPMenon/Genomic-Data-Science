import numpy as np
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)
from typing import Optional, Dict, Union, Literal
import pickle
from scipy import sparse
import warnings
from sklearn.base import BaseEstimator, ClassifierMixin # for sklearn compatibility for beyesian optimization


class LinearSVMClassifier( BaseEstimator, ClassifierMixin):
    """
    Linear SVM wrapper for scRNA-seq perturbation classification.
    
    This class provides a clean interface for training LinearSVC models on 
    sparse, high-dimensional gene expression data with built-in preprocessing 
    and evaluation capabilities.
    
    Parameters
    ----------
    C : float, default=1.0
        Regularization parameter. Smaller values = stronger regularization.
        
    max_iter : int, default=2000
        Maximum number of iterations for the solver.
        
    tol : float, default=1e-4
        Tolerance for stopping criterion.
        
    class_weight : dict, 'balanced', or None, default=None
        Weights associated with classes. If 'balanced', automatically adjusts 
        weights inversely proportional to class frequencies especialy for the norman Dataset.
        
    n_components : int or None, default=None
        Number of components for dimensionality reduction via TruncatedSVD.
        If None, no dimensionality reduction is performed.
        
    use_sparse : bool, default=True
        Whether to use sparse-compatible preprocessing. Set to False if working 
        with dense arrays and want mean-centering.
        
    random_state : int, default=42
        Random seed for reproducibility.
        
    verbose : int, default=0
        Verbosity level for LinearSVC training (0=silent, 1=progress).
    """
    
    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 2000,
        tol: float = 1e-4,
        class_weight: Union[Dict, Literal['balanced'], None] = None,
        n_components: Optional[int] = None,
        use_sparse: bool = True,
        random_state: int = 42,
        verbose: int = 0
    ):
        self.C = C
        self.max_iter = max_iter
        self.tol = tol
        self.class_weight = class_weight
        self.n_components = n_components
        self.use_sparse = use_sparse
        self.random_state = random_state
        self.verbose = verbose
        
        # Initialize components
        self.model_ = None
        self.scaler_ = None
        self.svd_ = None
        self.is_fitted_ = False
        self.n_classes_ = None
        self.n_features_in_ = None
        self.n_features_reduced_ = None
    
    def _create_model(self) -> LinearSVC:
        """Create a new LinearSVC instance with current hyperparameters."""
        return LinearSVC(
            C=self.C,
            loss='squared_hinge',  
            dual=False,  
            max_iter=self.max_iter,
            tol=self.tol,
            class_weight=self.class_weight,
            random_state=self.random_state,
            verbose=self.verbose
        )
    
    def __sklearn_tags__(self):
        # Return an empty dict or custom tags as needed for sklearn compatibility
        return super().__sklearn_tags__() or {}


    def fit(self, X: Union[np.ndarray, sparse.spmatrix], y: np.ndarray) -> 'LinearSVMClassifier':
        """
        Fit the Linear SVM model.
        
        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            Training gene expression data. (AnnData sparse gene matrix)
            
        y : array-like, shape (n_samples,)
            Target perturbation labels (encoded as integers).
        
        Returns
        -------
        self : LinearSVMClassifier
            Fitted estimator.
        """
        # Store input info
        self.n_features_in_ = X.shape[1]
        self.n_classes_ = len(np.unique(y))
        
        # Initialize preprocessing pipeline
        # Use with_mean=False for sparse matrices (centering would densify them)
        self.scaler_ = StandardScaler(with_mean=not self.use_sparse)
        
        # Scale features
        X_scaled = self.scaler_.fit_transform(X)
        
        # Optional dimensionality reduction
        if self.n_components is not None:
            if self.n_components >= X_scaled.shape[1]:
                warnings.warn(
                    f"n_components ({self.n_components}) >= n_features ({X_scaled.shape[1]}). "
                    f"Skipping dimensionality reduction."
                )
                self.svd_ = None
                self.n_features_reduced_ = X_scaled.shape[1]
            else:
                self.svd_ = TruncatedSVD(
                    n_components=self.n_components,
                    random_state=self.random_state
                )
                X_scaled = self.svd_.fit_transform(X_scaled)
                self.n_features_reduced_ = self.n_components
                
                if self.verbose > 0:
                    explained_var = self.svd_.explained_variance_ratio_.sum()
                    print(f"TruncatedSVD: {self.n_features_in_} → {self.n_components} features")
                    print(f"Explained variance: {explained_var:.2%}")
        else:
            self.n_features_reduced_ = X_scaled.shape[1]
        
        # Train LinearSVC
        self.model_ = self._create_model()
        self.model_.fit(X_scaled, y)
        self.is_fitted_ = True
        
        return self
    
    def predict(self, X: Union[np.ndarray, sparse.spmatrix]) -> np.ndarray:
        """
        Predict class labels for samples in X.
        
        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            Gene expression data to predict.
        
        Returns
        -------
        y_pred : ndarray, shape (n_samples,)
            Predicted class labels.
        """
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted before making predictions. Call fit() first.")
        
        # Apply same preprocessing as training
        X_scaled = self.scaler_.transform(X)
        if self.svd_ is not None:
            X_scaled = self.svd_.transform(X_scaled)
        
        return self.model_.predict(X_scaled)
    
    def predict_decision_function(self, X: Union[np.ndarray, sparse.spmatrix]) -> np.ndarray:
        """
        Compute decision function values for samples in X.
        
        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            Gene expression data.
        
        Returns
        -------
        scores : ndarray, shape (n_samples, n_classes) or (n_samples,)
            Decision function values. For binary classification, shape is (n_samples,).
            For multi-class, shape is (n_samples, n_classes) using one-vs-rest.
        """
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted before prediction. Call fit() first.")
        
        X_scaled = self.scaler_.transform(X)
        if self.svd_ is not None:
            X_scaled = self.svd_.transform(X_scaled)
        
        return self.model_.decision_function(X_scaled)
    
    def evaluate(
        self, 
        X: Union[np.ndarray, sparse.spmatrix], 
        y: np.ndarray,
        detailed: bool = False
    ) -> Dict[str, Union[float, str, np.ndarray]]:
        """
        Evaluate model performance on test data.
        
        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            Test gene expression data.
            
        y : array-like, shape (n_samples,)
            True labels.
            
        detailed : bool, default=False
            If True, include classification report and confusion matrix in results.
        
        Returns
        -------
        results : dict
            Dictionary containing evaluation metrics:
            - accuracy: Overall accuracy
            - precision: Weighted precision
            - recall: Weighted recall
            - f1_score: Weighted F1 score
            - classification_report: Detailed per-class metrics (if detailed=True)
            - confusion_matrix: Confusion matrix (if detailed=True)
        """
        y_pred = self.predict(X)
        
        # Compute metrics
        accuracy = accuracy_score(y, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y, y_pred, average='weighted', zero_division=0
        )
        
        results = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1)
        }
        
        if detailed:
            results['classification_report'] = classification_report(
                y, y_pred, zero_division=0
            )
            results['confusion_matrix'] = confusion_matrix(y, y_pred)
        
        return results
    
    def save(self, filepath: str) -> None:
        """
        Save the fitted model to disk.
        
        Parameters
        ----------
        filepath : str
            Path where the model will be saved (should end with .pkl).
        """
        if not self.is_fitted_:
            raise RuntimeError("Cannot save an unfitted model. Call fit() first.")
        
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        
        print(f"Model saved to {filepath}")
    
    @staticmethod
    def load(filepath: str) -> 'LinearSVMClassifier':
        """
        Load a fitted model from disk.
        
        Parameters
        ----------
        filepath : str
            Path to the saved model file.
        
        Returns
        -------
        model : LinearSVMClassifier
            Loaded fitted model.
        """
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        
        if not isinstance(model, LinearSVMClassifier):
            raise TypeError(f"Loaded object is not a LinearSVMClassifier instance")
        
        print(f"Model loaded from {filepath}")
        return model
    
    def get_params(self, deep: bool = True) -> Dict:
        """
        Get parameters for this estimator (sklearn compatibility).
        This was used for hyperparameter tuning with sklearn tools.

        Parameters
        ----------
        deep : bool, default=True
            If True, return parameters for sub-objects.
        
        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        return {
            'C': self.C,
            'max_iter': self.max_iter,
            'tol': self.tol,
            'class_weight': self.class_weight,
            'n_components': self.n_components,
            'use_sparse': self.use_sparse,
            'random_state': self.random_state,
            'verbose': self.verbose
        }
    
    def set_params(self, **params) -> 'LinearSVMClassifier':
        """
        Set parameters for this estimator (sklearn compatibility).
        This was used for hyperparameter tuning with sklearn tools.

        Parameters
        ----------
        **params : dict
            Estimator parameters.
        
        Returns
        -------
        self : LinearSVMClassifier
            Estimator instance.
        """
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter {key} for estimator {type(self).__name__}")
        
        return self
    
    def __repr__(self) -> str:
        """String representation of the model."""
        params = self.get_params(deep=False)
        params_str = ', '.join(f'{k}={v}' for k, v in params.items())
        fitted_str = "fitted" if self.is_fitted_ else "not fitted"
        return f"LinearSVMClassifier({params_str}) [{fitted_str}]"


# Backward compatibility alias
SVMModel = LinearSVMClassifier