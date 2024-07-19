# %%

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import make_regression
from sktorch.linear_model import LinearRegression
import matplotlib.pyplot as plt

# %%

# 데이터 생성
X, y = make_regression(n_samples=1000, n_features=1, noise=0.1, random_state=42)

# 학습 데이터와 테스트 데이터로 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %%

# 모델 초기화 및 학습
model = LinearRegression(n_epochs=100, batch_size=32, lr=0.1)
model.fit(X_train, y_train)

# 예측
y_pred = model.predict(X_test)

# %%

# 성능 평가
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.4f}")
print(f"R-squared Score: {r2:.4f}")

# %%

# 결과 시각화
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', alpha=0.5, label='Actual')
plt.scatter(X_test, y_pred, color='red', alpha=0.5, label='Predicted')
plt.plot(X_test, y_pred, color='green', linewidth=2, label='Regression Line')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression: Actual vs Predicted')
plt.legend()
plt.show()

# 학습 과정에서의 가중치와 편향 출력
weight = model.model.linear.weight.item()
bias = model.model.linear.bias.item()
print(f"Learned Weight: {weight:.4f}")
print(f"Learned Bias: {bias:.4f}")
# %%
