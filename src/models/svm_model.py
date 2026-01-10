# SVM Model for Norman Perturbation Data
# Support Vector Machine implementation for genomic data classification

import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from typing import Optional, Tuple, Dict, Literal
import pickle

'''
    ### SVM Model Class Wrapper SVC from sklearn ###
'''
class SVMModel:

    def __init__(
                    self, 
                    kernel: Literal['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'] = 'rbf', 
                    C: float = 0.99,
                    gamma: Literal['scale', 'auto'] | float = 'scale', 
                    pca_components: Optional[int] = None, 
                    random_state: int = 42
                ):
        
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.pca_components = pca_components
        self.random_state = random_state

        self.scaler = StandardScaler()
        self.pca = PCA(n_components=pca_components) if pca_components else None

        self.model = SVC(
            kernel=kernel,
            C=C,
            gamma=gamma,
            random_state=random_state
        )

        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SVMModel':
        X_scaled = self.scaler.fit_transform(X)
        if self.pca:
            X_scaled = self.pca.fit_transform(X_scaled)
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("The model must be fitted before prediction.")
        X_scaled = self.scaler.transform(X)
        if self.pca:
            X_scaled = self.pca.transform(X_scaled)
        return self.model.predict(X_scaled)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        y_pred = self.predict(X)
        accuracy = accuracy_score(y, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y, y_pred, average='weighted')
        return {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1)
        }
    
    def save(self, filepath: str) -> None:
        if not self.is_fitted:
            raise RuntimeError("The model must be fitted before saving.")
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"Model saved to {filepath}")

    @staticmethod
    def load(filepath: str) -> 'SVMModel':
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded from {filepath}")
        return model
        