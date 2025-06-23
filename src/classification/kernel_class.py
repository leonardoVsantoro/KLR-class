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

    def fit(self, X, y, ridge = 1e-3, band_factor = 1, kernel_name = 'sqeuclidean', 
            crossvalidation = False, validation_set = None, ls_ridge = [0.1,0.01,0.001,0.0001, 0.00001], ls_band_factor = [100,50,10,5,1,0.5,0.1,0.05,0.01]):
        """
        Fit the KLR model to the training data.

        Parameters:
        X (ndarray): Training features.
        y (ndarray): Training labels (0 or 1).
        kernel_name (str): The name of the kernel function to use (default is 'sqeuclidean').
        ridge (float): The ridge regularization parameter (default is 1e-3).
        band_factor (float): Factor to adjust the bandwidth for the kernel (default is 1).
        crossvalidation (bool): If True, perform cross-validation to find the best hyperparameters (default is False).
        ls_ridge (list): List of ridge regularization parameters to test if crossvalidation is True.
        ls_band_factor (list): List of bandwidth factors to test if crossvalidation is True.

        Returns:
        self: Fitted KLRClassifier instance.
        """
        self.is_fitted_ = True
        self.LabelEncoder = LabelEncoder()
        self.X_train = X
        self.y_train = self.LabelEncoder.fit_transform(y)
        self.kernel_name = kernel_name
        self.pairwise_dists =  cdist(X, X, self.kernel_name)
        self.median = np.median(self.pairwise_dists[self.pairwise_dists > 0])
        self.unique_labels = np.unique(self.y_train)

        if crossvalidation is False:
            self.band_factor = band_factor
            self.bandwidth = 2 * self.median * self.band_factor
            self.ridge = ridge
            kernel_matrix = np.exp(- self.pairwise_dists / self.bandwidth)
            Ks = [ kernel_matrix[self.y_train == lab , :] for lab in self.unique_labels]
            self.ms = [K.mean(0) for K in Ks]
            self.Cs = [np.cov(K.T, bias=True) for K in Ks]
            self.demb = self.Cs[0].shape[0]
        else:
            try:
                if validation_set is None:
                    raise ValueError("validation_set must be provided when crossvalidation is True.")
            except ValueError as e:
                print(f"{e} Running fit with crossvalidation=False.")
                return self.fit(X, y, ridge=ridge, band_factor=band_factor, crossvalidation=False, kernel_name=kernel_name)
            X_validation, y_validation = validation_set[0], self.LabelEncoder.transform(validation_set[1])
            best_accuracy, best_ridge, best_band_factor = 0, 1e-3, 1
            for _ridge in ls_ridge:
                for _band_factor in ls_band_factor:
                    _kernel_matrix = np.exp(- self.pairwise_dists / (self.median * _band_factor))
                    _Ks = [ _kernel_matrix[self.y_train == lab , :] for lab in self.unique_labels]
                    _ms = [K.mean(0) for K in _Ks]
                    _Cs = [np.cov(K.T, bias=True) for K in _Ks]
                    _kx_matrix = np.exp(- cdist(self.X_train, X_validation, self.kernel_name) / (self.median * _band_factor))
                    _Ls = np.array([  
                        np.einsum('ij,ij->j',
                                _kx_matrix - m[:, np.newaxis], 
                                        np.linalg.solve(C+ np.eye(C.shape[0])*_ridge,
                                                    _kx_matrix - m[:, np.newaxis] ))
                                            + fast_logdet(C+ np.eye(C.shape[0])*_ridge) 
                                for m, C in zip(_ms, _Cs)
                                ])
                    _accuracy = np.mean(np.argmax(-_Ls, axis=0) == y_validation)
                    if (_ridge, _band_factor) == (1e-3, 1):
                        default_accuracy = _accuracy
                    if _accuracy > best_accuracy:
                        best_ridge = _ridge
                        best_band_factor = _band_factor
                        best_accuracy = _accuracy
            print(f"Selected ridge: {best_ridge}, Selected band_factor: {best_band_factor}")
            print(f"Validation ACC: {best_accuracy:.4f} (gained: {(best_accuracy- default_accuracy):.4f})")
            self.fit(X, y, ridge=best_ridge, band_factor=best_band_factor, crossvalidation=False, kernel_name=kernel_name)
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
        return np.mean(y_pred == self.LabelEncoder.transform(y))  
    
    def predict_proba(self, X, doNotExp = True):
        """Converts scores to probability estimates."""
        if doNotExp:
            Ls = -self.decision_function(X)
        else:
            Ls = np.exp(-self.decision_function(X))
            # return (Ls / np.sum(Ls, axis=0)).T
        return Ls.T

        # preds = np.argmax(-Ls, axis=0)
        # probs = np.zeros(preds.shape)
        # for i in self.unique_labels:
        #     probs[preds==i] = -Ls[i,:][preds==i]


        # probs[preds==1] = -Ls[1,:][preds==1]
        # probs[preds==0] = Ls[0,:][preds==0]
        return probs


