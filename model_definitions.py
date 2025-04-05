import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class EnhancedEpilepsyModel(BaseEstimator, ClassifierMixin):
    """Enhanced epilepsy prediction model"""
    
    def __init__(self, threshold=0.00005):
        self.threshold = threshold
        # Extended to 8 features to match the input data in the UI
        self.feature_importance = np.array([0.2, 0.15, 0.12, 0.18, 0.14, 0.08, 0.13, 0.0])
        
    def predict(self, X):
        """Predict seizure occurrence based on EEG data"""
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
            
        # Apply feature importance weights - ensure dimensionality match
        weighted_X = X[:, :self.feature_importance.shape[0]] * self.feature_importance
        
        # Calculate the weighted magnitude
        weighted_magnitude = np.mean(np.abs(weighted_X), axis=1)
        
        # Make predictions based on threshold
        predictions = (weighted_magnitude > self.threshold).astype(int)
        return predictions
    
    def get_anomaly_scores(self, X):
        """Calculate anomaly scores based on deviation from normal patterns"""
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
            
        # Apply feature importance weights - ensure dimensionality match
        weighted_X = X[:, :self.feature_importance.shape[0]] * self.feature_importance
        
        # Calculate the weighted magnitude
        weighted_magnitude = np.mean(np.abs(weighted_X), axis=1)
        
        # Normalize scores to 0-1 range
        normalized_scores = weighted_magnitude / (self.threshold * 2)
        return np.clip(normalized_scores, 0, 1)  # Ensure values are between 0 and 1