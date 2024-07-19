import torch
import torch.nn as nn
from sklearn.base import BaseEstimator

class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        raise NotImplementedError("Subclasses should implement this!")

class PyTorchBaseEstimator(BaseEstimator):
    def __init__(self, model, criterion, optimizer, n_epochs=100, batch_size=32):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.n_epochs = n_epochs
        self.batch_size = batch_size

    def fit(self, X, y):
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y).reshape(-1, 1)

        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.n_epochs):
            for X_batch, y_batch in dataloader:
                self.optimizer.zero_grad()
                y_pred = self.model(X_batch)
                loss = self.criterion(y_pred, y_batch)
                loss.backward()
                self.optimizer.step()

        return self

    def predict(self, X):
        X_tensor = torch.FloatTensor(X)
        with torch.no_grad():
            y_pred = self.model(X_tensor)
        return y_pred.numpy().flatten()