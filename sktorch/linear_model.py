import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import RegressorMixin, ClassifierMixin
from .base import BaseModel, PyTorchBaseEstimator
from .utils import check_X_y, check_array, check_is_fitted, unique_labels

class LinearRegressionModel(BaseModel):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        return self.linear(x)

class LinearRegression(PyTorchBaseEstimator, RegressorMixin):
    def __init__(self, n_epochs=100, batch_size=32, lr=0.01):
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.X_ = X
        self.y_ = y

        input_dim = X.shape[1]
        model = LinearRegressionModel(input_dim)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.lr)

        super().__init__(model, criterion, optimizer, self.n_epochs, self.batch_size)
        return super().fit(X, y)

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        return super().predict(X)
    
class LogisticRegressionModel(BaseModel):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        return torch.sigmoid(self.linear(x))

class LogisticRegression(PyTorchBaseEstimator, ClassifierMixin):
    def __init__(self, n_epochs=100, batch_size=32, lr=0.01):
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr

    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        
        if len(self.classes_) != 2:
            raise ValueError("LogisticRegression supports only binary classification. "
                             f"Got classes {self.classes_}")

        self.X_ = X
        self.y_ = y

        input_dim = X.shape[1]
        model = LogsticRegressionModel(input_dim)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.lr)

        super().__init__(model, criterion, optimizer, self.n_epochs, self.batch_size)
        return super().fit(X, y)

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        return super().predict(X)