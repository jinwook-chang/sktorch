# %%
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sktorch.linear_model import RidgeRegression
# %%
# Generate synthetic data
np.random.seed(42)
X = np.random.randn(1000, 20)
true_weights = np.random.randn(20) * 0.5
y = X.dot(true_weights) + np.random.randn(1000) * 0.1

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# %%
# Initialize and train the Ridge Regression model
ridge_model = RidgeRegression(alpha=0.1, n_epochs=1000, batch_size=32, lr=0.01)
ridge_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred_train = ridge_model.predict(X_train_scaled)
y_pred_test = ridge_model.predict(X_test_scaled)

# Evaluate the model
train_mse = mean_squared_error(y_train, y_pred_train)
test_mse = mean_squared_error(y_test, y_pred_test)
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)

print("Ridge Regression Results:")
print(f"Train MSE: {train_mse:.4f}")
print(f"Test MSE: {test_mse:.4f}")
print(f"Train R2: {train_r2:.4f}")
print(f"Test R2: {test_r2:.4f}")

# Compare with true weights
model_weights = ridge_model.model.linear.weight.detach().numpy().flatten()
weight_mse = mean_squared_error(true_weights, model_weights)
print(f"\nMSE between true and learned weights: {weight_mse:.4f}")

# Print a few true weights vs learned weights
print("\nTrue weights vs Learned weights:")
for i in range(5):  # Print first 5 weights
    print(f"Feature {i}: True = {true_weights[i]:.4f}, Learned = {model_weights[i]:.4f}")
# %%
