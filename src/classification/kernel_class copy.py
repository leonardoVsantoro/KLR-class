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
        self.demb = self.Cs[0].shape[0]

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
    
    def predict_proba(self, X,):
        """Converts scores to probability estimates."""
        Ls = self.decision_function(X)
        preds = np.argmax(-Ls, axis=0)
        probs = np.zeros(preds.shape)
        probs[preds==1] = -Ls[1,:][preds==1]
        probs[preds==0] = Ls[0,:][preds==0]
        return probs


    def validate(self, X, y, ls_ridge = [0.1,0.01,0.001,0.0001,0.00001], ls_band_factor = [5,1,0.5,0.1,0.05,0.01]):
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




# from src.modules import *
# from sklearn.base import BaseEstimator, ClassifierMixin
# from sklearn.utils.extmath import fast_logdet

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
#         None

#     def fit(self, X, y, ridge = 1e-3, band_factor = 1, kernel_name = 'sqeuclidean', 
#             crossvalidation = False, ls_ridge = [0.1,0.01,0.001,0.0001], ls_band_factor = [5,1,0.5,0.1,0.05]):
#         """
#         Fit the KLR model to the training data.
#         Parameters:
#         X (ndarray): Training features.
#         y (ndarray): Training labels (0 or 1).
#         kernel_name (str): The name of the kernel function to use (default is 'sqeuclidean').
#         ridge (float): The ridge regularization parameter (default is 1e-3).
#         band_factor (float): Factor to adjust the bandwidth for the kernel (default is 1).
#         """
#         self.is_fitted_ = True

#         self.X_train = X
#         self.y_train = y
#         self.kernel_name = kernel_name
#         self.pairwise_dists =  cdist(X, X, self.kernel_name)
#         self.median = np.median(self.pairwise_dists[self.pairwise_dists > 0])
#         self.unique_labels = np.unique(y)

#         if not crossvalidation:
#             self.band_factor = band_factor
#             self.bandwidth = 2 * self.median * self.band_factor
#             self.ridge = ridge
#             kernel_matrix = np.exp(- self.pairwise_dists / self.bandwidth)
#             Ks = [ kernel_matrix[self.y_train == y , :] for y in self.unique_labels]
#             self.ms = [K.mean(0) for K in Ks]
#             self.Cs = [np.cov(K.T, bias=True) for K in Ks]
#             self.demb = self.Cs[0].shape[0]
#         else:
#             best_accuracy, best_ridge, best_band_factor = 0, 1e-3, 1
#             for _ridge in ls_ridge:
#                 for _band_factor in ls_band_factor:
#                     _kernel_matrix = np.exp(- self.pairwise_dists / (self.median * _band_factor))
#                     _Ks = [ _kernel_matrix[self.y_train == y , :] for y in self.unique_labels]
#                     _ms = [K.mean(0) for K in _Ks]
#                     _Cs = [np.cov(K.T, bias=True) for K in _Ks]
#                     _kx_matrix = np.exp(- cdist(self.X_train, X, self.kernel_name) / (self.median * _band_factor))
#                     _Ls = [  
#                         np.einsum('ij,ij->j',
#                                 _kx_matrix - m[:, np.newaxis], 
#                                         np.linalg.solve(C+ np.eye(C.shape[0])*_ridge,
#                                                     _kx_matrix - m[:, np.newaxis] ))
#                                             + fast_logdet(C+ np.eye(C.shape[0])*_ridge) 
#                                 for m, C in zip(_ms, _Cs)
#                                 ]
#                     _accuracy = np.mean(np.argmax(_Ls, axis=0) == y)
#                     if _accuracy > best_accuracy:
#                         best_ridge = _ridge
#                         best_band_factor = _band_factor
#                         best_accuracy = _accuracy
#             self.fit(X, y, ridge=best_ridge, band_factor=best_band_factor, crossvalidation=False, kernel_name=kernel_name)
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
#         Ls = [  
#             np.einsum('ij,ij->j',
#                        kx_matrix - m[:, np.newaxis], 
#                        np.linalg.solve(     C+ np.eye(C.shape[0])*self.ridge,
#                                              kx_matrix - m[:, np.newaxis], )) 
#                                              + fast_logdet(C+ np.eye(C.shape[0])*self.ridge)
#                        for m, C in zip(self.ms, self.Cs)
#                        ]
#         return np.array(Ls)
    
#     def predict(self, X):
#         """
#         Predict the labels for the input data using the trained model.

#         Parameters:
#         X (ndarray): Input features for prediction.
#         Returns:

#         y_pred (ndarray): Predicted labels for the input data.
#         LLRs (ndarray): Log-likelihood ratios for the predictions.
#         """        
#         return np.argmax(-self.decision_function(X), axis=0)

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
    
#     def predict_proba(self, X,):
#         """Converts scores to probability estimates."""
#         Ls = self.decision_function(X)
#         preds = np.argmax(-Ls, axis=0)
#         probs = np.zeros(preds.shape)
#         probs[preds==1] = -Ls[1,:][preds==1]
#         probs[preds==0] = Ls[0,:][preds==0]
#         return probs


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
#         best_ridge = 1e-3
#         best_band_factor = 1

#         for _ridge in ls_ridge:
#             for _band_factor in ls_band_factor:
#                 _kernel_matrix = np.exp(- self.pairwise_dists / (self.bandwidth * _band_factor))
#                 _Ks = [ _kernel_matrix[self.y_train == y , :] for y in self.unique_labels]
#                 _ms = [K.mean(0) for K in _Ks]
#                 _Cs = [np.cov(K.T, bias=True) for K in _Ks]
#                 _kx_matrix = np.exp(- cdist(self.X_train, X, self.kernel_name) / (self.bandwidth * _band_factor))
#                 _Ls = [  
#                     np.einsum('ij,ij->j',
#                             _kx_matrix - m[:, np.newaxis], 
#                                     np.linalg.solve(C+ np.eye(C.shape[0])*_ridge,
#                                                 _kx_matrix - m[:, np.newaxis] ))
#                                         + fast_logdet(C+ np.eye(C.shape[0])*_ridge) 
#                             for m, C in zip(_ms, _Cs)
#                             ]
#                 _accuracy = np.mean(np.argmax(_Ls, axis=0) == y)
#                 if _accuracy > best_accuracy:
#                     best_ridge = _ridge
#                     best_band_factor = _band_factor
#                     best_accuracy = _accuracy
#         self.fit(X, y, ridge=best_ridge, band_factor=best_band_factor)
#         return self

