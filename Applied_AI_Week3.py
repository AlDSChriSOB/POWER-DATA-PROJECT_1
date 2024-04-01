#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import cdist
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv('household_power_consumption.csv', sep=';')
df


# In[3]:


df.info()


# In[4]:


df_toclean = df.drop(['Date', 'Time'], axis=1)


# In[5]:


def clean_dataset(df):
    assert isinstance(df, pd.DataFrame)
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)


# In[6]:


cleaned_df = clean_dataset(df_toclean)


# In[7]:


cleaned_df.isnull().sum()


# ##### 2. K-Means Clustering

# In[8]:


from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(cleaned_df)


# In[9]:


X = cleaned_df.to_numpy()
k_means = KMeans(n_clusters=3)
labels = k_means.fit_predict(X)


# In[10]:


# create a scatter plot
fig = plt.figure(figsize=(7, 6)) # set the size of the figure
ax = Axes3D(fig) # define that you want a 3D figure
# define which data is x, y, z
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c = labels, cmap = 'magma')
# Add labels to the plot and a title
ax.set_title("3 clusters")
fig.show()


# ##### 3. How many clusters?

# In[11]:


wccs = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    wccs.append(kmeans.inertia_)

#visualize results
plt.plot(range(1, 11), sse)
plt.xticks(range(1, 11))
plt.xlabel("Number of Clusters")
plt.ylabel("wccs")
plt.show()


# ##### Correlation analysis

# In[ ]:


sns.heatmap(cleaned_df.corr(), annot=True)


# In[ ]:


from sklearn.decomposition import PCA
pca = PCA(n_components=4)
x_pca = pca.fit_transform(X_scaled)
percent_variance = np.round(pca.explained_variance_ratio_* 100, decimals =2)
print("percent_variance=", percent_variance)


# In[ ]:


k_means = KMeans(n_clusters=4)
labels = k_means.fit_predict(x_pca)

# create a scatter plot
fig = plt.figure(figsize=(7, 6)) # set the size of the figure
ax = Axes3D(fig) # define that you want a 3D figure
# define which data is x, y, z
ax.scatter(x_pca[:, 0], x_pca[:, 1], x_pca[:, 2], c = labels, cmap = 'viridis')
# Add labels to the plot and a title
ax.set_title("4 clusters")
fig.show()


# In[ ]:




