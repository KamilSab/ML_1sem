# Импорт необходимых библиотек
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_squared_error
from mpl_toolkits.mplot3d import Axes3D

data = pd.read_csv('AmesHousing.csv')

cols_to_drop = ['Order', 'PID', 'Alley', 'Pool QC', 'Misc Feature', 'Fence', 'Fireplace Qu']
data.drop(cols_to_drop, axis=1, inplace=True)

for column in data.columns:
    if data[column].dtype == 'object':
        data[column].fillna(data[column].mode()[0], inplace=True)
    else:
        data[column].fillna(data[column].median(), inplace=True)

data = pd.get_dummies(data, drop_first=True)

X = data.drop('SalePrice', axis=1)
y = data['SalePrice']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_pca[:, 0], X_pca[:, 1], y, c='b', marker='o', alpha=0.5)
ax.set_xlabel('PCA Component 1', fontsize=12)
ax.set_ylabel('PCA Component 2', fontsize=12)
ax.set_zlabel('SalePrice', fontsize=12)
plt.title('3D Visualization of Housing Data (PCA Reduced)', fontsize=14)
plt.tight_layout()
plt.show()

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42)

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
lr_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'Linear Regression RMSE: {lr_rmse:.2f}')

alphas = np.logspace(-4, 2, 100)
rmse_values = []
coefs = []

for alpha in alphas:
    lasso = Lasso(alpha=alpha, max_iter=10000)
    lasso.fit(X_train, y_train)
    y_pred = lasso.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    rmse_values.append(rmse)
    coefs.append(lasso.coef_)

plt.figure(figsize=(12, 6))
plt.semilogx(alphas, rmse_values)
plt.xlabel('Regularization Strength (alpha)', fontsize=12)
plt.ylabel('RMSE', fontsize=12)
plt.title('Lasso Regularization Performance', fontsize=14)
plt.grid(True)
plt.show()

optimal_idx = np.argmin(rmse_values)
optimal_alpha = alphas[optimal_idx]
best_lasso = Lasso(alpha=optimal_alpha, max_iter=10000)
best_lasso.fit(X_train, y_train)

feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': best_lasso.coef_,
    'Abs_Coefficient': np.abs(best_lasso.coef_)
})

top_features = feature_importance.sort_values('Abs_Coefficient', ascending=False).head(5)
print("\nTop 5 Most Important Features:")
print(top_features[['Feature', 'Coefficient']].to_string(index=False))

plt.figure(figsize=(12, 6))
plt.barh(top_features['Feature'], top_features['Abs_Coefficient'])
plt.xlabel('Absolute Coefficient Value', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.title(f'Top 5 Most Important Features (alpha={optimal_alpha:.4f})', fontsize=14)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()