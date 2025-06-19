from src.modules import *
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.extmath import fast_logdet

class KLRClassifier(BaseEstimator, ClassifierMixin):
    """
    Kernel Logistic Regression (KLR) Classifier.

    This class implements a Kernel Logistic Regression model for binary classification.
    It uses a kernel function to compute the pairwise distances between training samples
    and applies logistic regression in the kernel space.

    Attributes:
    kernel_name (str): The name of the kernel function to use (default is 'sqeuclidean').
    ridge (float): The ridge regularization parameter (default is 1e-3).
    band_factor (float): Factor to adjust the bandwidth for the kernel (default is 1).

    Methods:
    fit(X, y): Fit the KLR model to the training data.
    predict(X): Predict the labels for the input data using the trained model.
    score(X, y): Calculate the accuracy of the model on the given data.
    validate(X, y, ls_ridge, ls_band_factor): Validate the model using cross-validation to find the best hyperparameters.
    """
    def __init__(self):
        None

    def fit(self, X, y, ridge = 1e-3, band_factor = 1, kernel_name = 'sqeuclidean'):
        """
        Fit the KLR model to the training data.
        Parameters:
        X (ndarray): Training features.
        y (ndarray): Training labels (0 or 1).
        kernel_name (str): The name of the kernel function to use (default is 'sqeuclidean').
        ridge (float): The ridge regularization parameter (default is 1e-3).
        band_factor (float): Factor to adjust the bandwidth for the kernel (default is 1).
        """
        self.is_fitted_ = True

        self.X_train = X
        self.y_train = y
        self.kernel_name = kernel_name
        self.pairwise_dists =  cdist(X, X, self.kernel_name)
        
        self.band_factor = band_factor
        self.bandwidth = 2 * np.median(self.pairwise_dists[self.pairwise_dists > 0])* self.band_factor
        self.ridge = ridge

        kernel_matrix = np.exp(- self.pairwise_dists / self.bandwidth)

        self.unique_labels = np.unique(y)
        Ks = [ kernel_matrix[self.y_train == y , :] for y in self.unique_labels]
        self.ms = [K.mean(0) for K in Ks]
        self.Cs = [np.cov(K.T, bias=True) for K in Ks]


        return self
    
    def decision_function(self, X):
        """
        Calculate the log-likelihood ratios for the input data.

        Parameters:
        X (ndarray): Input features for prediction.

        Returns:
        LLRs (ndarray): Log-likelihood ratios for the predictions.
        """
        kx_matrix = np.exp(- cdist(self.X_train, X, self.kernel_name) / self.bandwidth)
        Ls = [  
            np.einsum('ij,ij->j',
                       kx_matrix - m[:, np.newaxis], 
                       np.linalg.solve(     C+ np.eye(C.shape[0])*self.ridge,
                                             kx_matrix - m[:, np.newaxis], )) 
                                             + fast_logdet(C+ np.eye(C.shape[0])*self.ridge)
                       for m, C in zip(self.ms, self.Cs)
                       ]
        return np.array(Ls)
    
    def predict(self, X):
        """
        Predict the labels for the input data using the trained model.

        Parameters:
        X (ndarray): Input features for prediction.
        Returns:

        y_pred (ndarray): Predicted labels for the input data.
        LLRs (ndarray): Log-likelihood ratios for the predictions.
        """        
        return np.argmax(-self.decision_function(X), axis=0)

    def score(self, X, y):
        """
        Calculate the accuracy of the model on the given data.

        Parameters:
        X (ndarray): Input features.
        y (ndarray): True labels.

        Returns:
        float: Accuracy of the model.
        """
        y_pred = self.predict(X)
        return np.mean(y_pred == y)  
    
    def predict_proba(self, X):
        """Converts scores to probability estimates."""
        scores = np.exp(-self.decision_function(X))
        # probs = (scores/scores.sum(0)).T
        return np.exp(-self.decision_function(X))

    def validate(self, X, y, ls_ridge = [0.1,0.01,0.001,0.0001], ls_band_factor = [5,1,0.5,0.1,0.05]):
        """
        Validate the KLR model using cross-validation to find the best hyperparameters.

        Parameters:
        X (ndarray): validation features.
        y (ndarray): validation labels (0 or 1).
        ls_ridge (list): List of ridge regularization parameters to test.
        ls_band_factor (list): List of bandwidth factors to test.

        Returns:
        tuple: Best ridge parameter, best bandwidth factor, and the best accuracy.
        """
        best_accuracy = 0
        best_ridge = 1e-3
        best_band_factor = 1

        for _ridge in ls_ridge:
            for _band_factor in ls_band_factor:
                _kernel_matrix = np.exp(- self.pairwise_dists / (self.bandwidth * _band_factor))
                _Ks = [ _kernel_matrix[self.y_train == y , :] for y in self.unique_labels]
                _ms = [K.mean(0) for K in _Ks]
                _Cs = [np.cov(K.T, bias=True) for K in _Ks]
                _kx_matrix = np.exp(- cdist(self.X_train, X, self.kernel_name) / (self.bandwidth * _band_factor))
                _Ls = [  
                    np.einsum('ij,ij->j',
                            _kx_matrix - m[:, np.newaxis], 
                                    np.linalg.solve(C+ np.eye(C.shape[0])*_ridge,
                                                _kx_matrix - m[:, np.newaxis] ))
                                        + fast_logdet(C+ np.eye(C.shape[0])*_ridge) 
                            for m, C in zip(_ms, _Cs)
                            ]
                _accuracy = np.mean(np.argmax(_Ls, axis=0) == y)
                if _accuracy > best_accuracy:
                    best_ridge = _ridge
                    best_band_factor = _band_factor
                    best_accuracy = _accuracy
        self.fit(X, y, ridge=best_ridge, band_factor=best_band_factor)
        return self





# class KLRClassifier(BaseEstimator, ClassifierMixin):
#     """
#     Kernel Logistic Regression (KLR) Classifier.

#     This class implements a Kernel Logistic Regression model for binary classification.
#     It uses a kernel function to compute the pairwise distances between training samples
#     and applies logistic regression in the kernel space.

#     Attributes:
#     kernel_name (str): The name of the kernel function to use (default is 'sqeuclidean').
#     ridge (float): The ridge regularization parameter (default is 1e-3).
#     band_factor (float): Factor to adjust the bandwidth for the kernel (default is 1).

#     Methods:
#     fit(X, y): Fit the KLR model to the training data.
#     predict(X): Predict the labels for the input data using the trained model.
#     score(X, y): Calculate the accuracy of the model on the given data.
#     validate(X, y, ls_ridge, ls_band_factor): Validate the model using cross-validation to find the best hyperparameters.
#     """
#     def __init__(self):
#         self.is_fitted_ = False

#     def fit(self, X, y, ridge = 1e-3, band_factor = 1, kernel_name = 'sqeuclidean'):
#         """
#         Fit the KLR model to the training data.
#         Parameters:
#         X (ndarray): Training features.
#         y (ndarray): Training labels (0 or 1).
#         kernel_name (str): The name of the kernel function to use (default is 'sqeuclidean').
#         ridge (float): The ridge regularization parameter (default is 1e-3).
#         band_factor (float): Factor to adjust the bandwidth for the kernel (default is 1).
#         """
#         self.X_train = X
#         self.y_train = y
#         self.kernel_name = kernel_name
#         self.pairwise_dists =  cdist(X, X, self.kernel_name)
#         self.bandwidth = 2 * np.median(self.pairwise_dists[self.pairwise_dists > 0])* band_factor
#         self.ridge = ridge
#         kernel_matrix = np.exp(- self.pairwise_dists / self.bandwidth)


#         K0 = kernel_matrix[self.y_train ==0, :]
#         K1 = kernel_matrix[self.y_train ==1, :]
#         self.m0 = K0.mean(0)
#         self.m1 = K1.mean(0)
#         self.C0 = np.cov(K0.T, bias=True)
#         self.C1 = np.cov(K1.T, bias=True) 
#         self.is_fitted_ = True

#         return self
    
#     def decision_function(self, X):
#         """
#         Calculate the log-likelihood ratios for the input data.

#         Parameters:
#         X (ndarray): Input features for prediction.

#         Returns:
#         LLRs (ndarray): Log-likelihood ratios for the predictions.
#         """
#         kx_matrix = np.exp(- cdist(self.X_train, X, self.kernel_name) / self.bandwidth)
#         diff1_matrix = kx_matrix - self.m0[:, np.newaxis]
#         diff2_matrix = kx_matrix - self.m1[:, np.newaxis]
#         C0_inv_diff1 = np.linalg.solve(self.C0 + np.eye(self.C0.shape[0])*self.ridge, diff1_matrix)
#         C1_inv_diff2 = np.linalg.solve(self.C1 + np.eye(self.C1.shape[0])*self.ridge, diff2_matrix)
#         term1 = np.einsum('ij,ij->j', diff1_matrix, C0_inv_diff1)
#         term2 = np.einsum('ij,ij->j', diff2_matrix, C1_inv_diff2)       
#         return 0.5 * (term1 - term2)
    
#     def predict(self, X):
#         """
#         Predict the labels for the input data using the trained model.

#         Parameters:
#         X (ndarray): Input features for prediction.
#         Returns:

#         y_pred (ndarray): Predicted labels for the input data.
#         LLRs (ndarray): Log-likelihood ratios for the predictions.
#         """        
#         kx_matrix = np.exp(- cdist(self.X_train, X, self.kernel_name) / self.bandwidth)
#         diff1_matrix = kx_matrix - self.m0[:, np.newaxis]
#         diff2_matrix = kx_matrix - self.m1[:, np.newaxis]
#         C0_inv_diff1 = np.linalg.solve(self.C0 + np.eye(self.C0.shape[0])*self.ridge, diff1_matrix)
#         C1_inv_diff2 = np.linalg.solve(self.C1 + np.eye(self.C1.shape[0])*self.ridge, diff2_matrix)
#         term1 = np.einsum('ij,ij->j', diff1_matrix, C0_inv_diff1)
#         term2 = np.einsum('ij,ij->j', diff2_matrix, C1_inv_diff2)
#         y_pred = np.where(0.5 * (term1 - term2) < 0, 0, 1)
#         return y_pred

#     def score(self, X, y):
#         """
#         Calculate the accuracy of the model on the given data.

#         Parameters:
#         X (ndarray): Input features.
#         y (ndarray): True labels.

#         Returns:
#         float: Accuracy of the model.
#         """
#         y_pred = self.predict(X)
#         return np.mean(y_pred == y)  
    
#     def predict_proba(self, X):
#         """Converts scores to probability estimates."""
#         kx_matrix = np.exp(- cdist(self.X_train, X, self.kernel_name) / self.bandwidth)
#         diff1_matrix = kx_matrix - self.m0[:, np.newaxis]
#         diff2_matrix = kx_matrix - self.m1[:, np.newaxis]
#         C0_inv_diff1 = np.linalg.solve(self.C0 + np.eye(self.C0.shape[0])*self.ridge, diff1_matrix)
#         C1_inv_diff2 = np.linalg.solve(self.C1 + np.eye(self.C1.shape[0])*self.ridge, diff2_matrix)
#         term1 = np.einsum('ij,ij->j', diff1_matrix, C0_inv_diff1)
#         term2 = np.einsum('ij,ij->j', diff2_matrix, C1_inv_diff2)      
#         probs = np.array([[np.exp(t2) / (np.exp(t1) + np.exp(t2)), np.exp(t1) / (np.exp(t1) + np.exp(t2))] for t1, t2 in zip(term1, term2)]); 
#         return probs
#         # # probs = 1 / (1 + np.exp(-scores))
#         # # probs = 1- np.exp( - np.exp(scores))
#         # probs = 1-np.exp(-np.exp(scores))
#         # # probs = (np.clip(scores, -1,1)+1)/2
#         # return np.column_stack([1 - probs, probs]) 


#     def validate(self, X, y, ls_ridge = [0.1,0.01,0.001,0.0001], ls_band_factor = [5,1,0.5,0.1,0.05]):
#         """
#         Validate the KLR model using cross-validation to find the best hyperparameters.

#         Parameters:
#         X (ndarray): validation features.
#         y (ndarray): validation labels (0 or 1).
#         ls_ridge (list): List of ridge regularization parameters to test.
#         ls_band_factor (list): List of bandwidth factors to test.

#         Returns:
#         tuple: Best ridge parameter, best bandwidth factor, and the best accuracy.
#         """
#         best_accuracy = 0
#         best_ridge = None
#         best_band_factor = None

#         for ridge in ls_ridge:
#             for band_factor in ls_band_factor:
#                 _kernel_matrix = np.exp(- self.pairwise_dists / (self.bandwidth * band_factor))
#                 _K0,_K1 = _kernel_matrix[self.y_train ==0, :], _kernel_matrix[self.y_train ==1, :]
#                 _m0, _m1 = _K0.mean(0), _K1.mean(0)
#                 _C0,_C1 = np.cov(_K0.T, bias=True), np.cov(_K1.T, bias=True)
#                 _C0_inv_matrix = np.linalg.inv(self.C0)
#                 _C1_inv_matrix = np.linalg.inv(self.C1)
#                 _kx_matrix = np.exp(- cdist(self.X_train, X, self.kernel_name) / self.bandwidth* band_factor)
#                 _diff1_matrix = _kx_matrix - _m0[:, np.newaxis]
#                 _diff2_matrix = _kx_matrix - _m1[:, np.newaxis]
#                 _C0_inv_diff1 = np.linalg.solve(self.C0 + np.eye(_C0.shape[0])*ridge, _diff1_matrix)
#                 _C1_inv_diff2 = np.linalg.solve(self.C1 + np.eye(_C1.shape[0])*ridge, _diff2_matrix)
#                 _term1_vals = np.einsum('ij,ij->j', _diff1_matrix, _C0_inv_diff1)
#                 _term2_vals = np.einsum('ij,ij->j', _diff2_matrix, _C1_inv_diff2)
#                 _LLRs = 0.5 * (_term2_vals - _term1_vals)
#                 _y_pred = np.where(_LLRs < 0, 0, 1)
#                 accuracy = np.mean(_y_pred == y)

#                 if accuracy > best_accuracy:
#                     best_ridge = ridge
#                     best_band_factor = band_factor
#                     best_accuracy = accuracy
#         self.fit(X, y, ridge=best_ridge, band_factor=best_band_factor)
#         return self

