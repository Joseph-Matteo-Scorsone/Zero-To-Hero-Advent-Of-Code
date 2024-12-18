import numpy as np
import pandas as pd
import plotly.graph_objects as go

df_2y = pd.read_csv('../Bond_Yields/DGS2.csv', parse_dates=['observation_date'], index_col='observation_date')
df_5y = pd.read_csv('../Bond_Yields/DGS5.csv', parse_dates=['observation_date'], index_col='observation_date')
df_10y = pd.read_csv('../Bond_Yields/DGS10.csv', parse_dates=['observation_date'], index_col='observation_date')
df_30y = pd.read_csv('../Bond_Yields/DGS30.csv', parse_dates=['observation_date'], index_col='observation_date')

yields = pd.concat([df_2y, df_5y, df_10y, df_30y], axis=1)

yields.interpolate(method='time', inplace=True)  # Interpolate missing values based on time
yields.dropna(inplace=True)  # Drop rows with remaining missing values

# Normalize the data zero mean
mean_yields = np.mean(yields, axis=0)
std_yields = np.std(yields, axis=0)
normalized_yields = (yields - mean_yields) / std_yields

# covariance matrix
cov_matrix = np.cov(normalized_yields, rowvar=False)

# Perform eigen decomposition
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Sort eigenvalues and eigenvectors
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]

# Project the data onto the principal components
n_components = 2
top_eigenvectors = eigenvectors[:, :n_components]
projected_data = np.dot(normalized_yields, top_eigenvectors)

# explained variance ratio
explained_variance_ratio = eigenvalues / np.sum(eigenvalues)

fig1 = go.Figure(data=[go.Bar(x=[f'PC{i+1}' for i in range(len(explained_variance_ratio))], 
                              y=explained_variance_ratio)])
fig1.update_layout(title='Explained Variance by Principal Components',
                   xaxis_title='Principal Component',
                   yaxis_title='Explained Variance Ratio')
fig1.show()

# projection onto the first two principal components
fig2 = go.Figure(data=go.Scatter(x=projected_data[:, 0], 
                                 y=projected_data[:, 1], 
                                 mode='markers'))
fig2.update_layout(title='Projection onto First Two Principal Components',
                   xaxis_title='Principal Component 1',
                   yaxis_title='Principal Component 2')
fig2.show()