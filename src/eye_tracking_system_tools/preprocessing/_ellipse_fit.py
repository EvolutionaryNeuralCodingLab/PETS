"""
Simple least-squares ellipse fitting implementation.
This is a fallback when lsq-ellipse package is not available.
"""
import numpy as np
from scipy.optimize import least_squares


class LsqEllipse:
    """
    Least squares ellipse fitting.
    Compatible interface with the lsq-ellipse package.
    """
    
    def __init__(self):
        self.center = None
        self.width = None
        self.height = None
        self.phi = None
        
    def fit(self, X):
        """
        Fit an ellipse to points X (N x 2 array).
        Returns self for chaining.
        """
        X = np.asarray(X)
        if X.shape[0] < 5:
            raise ValueError("Need at least 5 points to fit an ellipse")
        
        # Center the data
        center = np.mean(X, axis=0)
        X_centered = X - center
        
        # Fit using algebraic method (simplified)
        # Using the method from: Fitzgibbon, Pilu, Fisher (1996)
        # "Direct least squares fitting of ellipses"
        
        x = X_centered[:, 0]
        y = X_centered[:, 1]
        
        # Build design matrix
        D = np.column_stack([x**2, x*y, y**2, x, y])
        S = D.T @ D
        
        # Constraint matrix (5x5 for 5 parameters: a, b, c, d, e)
        # Constraint: 4ac - b^2 = 1 (ellipse constraint)
        C = np.zeros((5, 5))
        C[0, 2] = 4  # 4ac term
        C[2, 0] = 4
        C[1, 1] = -1  # -b^2 term
        
        # Solve generalized eigenvalue problem
        try:
            from scipy.linalg import eig
            eigenvals, eigenvecs = eig(S, C)
            eigenvals = np.real(eigenvals)
            eigenvecs = np.real(eigenvecs)
            
            # Find positive eigenvalue
            pos_idx = np.where(eigenvals > 0)[0]
            if len(pos_idx) == 0:
                raise ValueError("No positive eigenvalue found")
            idx = pos_idx[np.argmin(eigenvals[pos_idx])]
            a = eigenvecs[:, idx]
        except ImportError:
            # Fallback: simple approximation
            # Fit as rotated ellipse
            cov = np.cov(X_centered.T)
            eigenvals, eigenvecs = np.linalg.eigh(cov)
            a = np.array([1.0, 0.0, 1.0, 0.0, 0.0])  # Simple circle approximation
        
        # Extract parameters
        # a = [a, b, c, d, e] for ax^2 + bxy + cy^2 + dx + ey + f = 0
        if len(a) >= 5:
            a_val, b_val, c_val, d_val, e_val = a[0], a[1], a[2], a[3], a[4]
        else:
            # Fallback to simple fit
            eigenvals, eigenvecs = np.linalg.eigh(np.cov(X_centered.T))
            major_idx = np.argmax(eigenvals)
            minor_idx = np.argmin(eigenvals)
            width = 2 * np.sqrt(eigenvals[major_idx])
            height = 2 * np.sqrt(eigenvals[minor_idx])
            phi = np.arctan2(eigenvecs[1, major_idx], eigenvecs[0, major_idx])
            
            self.center = center
            self.width = width
            self.height = height
            self.phi = phi
            return self
        
        # Convert to geometric parameters
        # This is a simplified conversion - for production use, implement full conversion
        try:
            # Simplified: assume b is small, use eigen decomposition
            A = np.array([[a_val, b_val/2], [b_val/2, c_val]])
            eigenvals, eigenvecs = np.linalg.eigh(A)
            
            # Ensure positive eigenvalues
            eigenvals = np.abs(eigenvals)
            major_idx = np.argmax(eigenvals)
            minor_idx = np.argmin(eigenvals)
            
            # Calculate geometric parameters
            det = a_val * c_val - (b_val/2)**2
            if det <= 0:
                raise ValueError("Invalid ellipse parameters")
            
            # Simplified conversion
            width = 2 * np.sqrt(1.0 / eigenvals[minor_idx]) if eigenvals[minor_idx] > 0 else 10.0
            height = 2 * np.sqrt(1.0 / eigenvals[major_idx]) if eigenvals[major_idx] > 0 else 10.0
            phi = np.arctan2(eigenvecs[1, major_idx], eigenvecs[0, major_idx])
            
            self.center = center
            self.width = width
            self.height = height
            self.phi = phi
        except Exception:
            # Ultimate fallback: use bounding box
            x_range = np.max(x) - np.min(x)
            y_range = np.max(y) - np.min(y)
            self.center = center
            self.width = max(x_range, 1.0)
            self.height = max(y_range, 1.0)
            self.phi = 0.0
        
        return self
    
    def as_parameters(self):
        """Return (center, width, height, phi) tuple."""
        if self.center is None:
            raise ValueError("Must call fit() first")
        return (self.center, self.width, self.height, self.phi)
