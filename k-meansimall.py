#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


# In[9]:


df=pd.read_csv("https://raw.githubusercontent.com/HarshNerpagar/ML-Lab/refs/heads/main/Mall_Customers.csv")


# In[10]:


df


# In[11]:


df.isnull().sum()


# In[12]:


df.dtypes


# In[13]:


Encoder={
    
}
for columns in df.columns:
    if df[columns].dtype == object:
        encoder=LabelEncoder()
        df[columns]=encoder.fit_transform(df[columns])
        Encoder[columns]=encoder


# In[14]:


df.dtypes


# In[18]:


X=df.drop(['CustomerID'],axis=1)


# In[19]:


scaler=StandardScaler()
X=scaler.fit_transform(df)


# In[32]:


inertia = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X)
    inertia.append(km.inertia_)

plt.plot(np.arange(1, 11), inertia, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()


# In[33]:


k=3
kmeans=KMeans(n_clusters=k,random_state=42)
kmeans.fit(X)
labels=kmeans.labels_


# In[37]:


df["Cluster"] = labels


# In[38]:


pca = PCA(n_components=2)
components = pca.fit_transform(X)
df["PCA1"] = components[:, 0]
df["PCA2"] = components[:, 1]


# In[39]:


kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)
df['Cluster'] = kmeans.labels_


# In[23]:


plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x="PCA1", y="PCA2", hue="Cluster", palette="Set2", s=100)
plt.title("K-Means Clusters (PCA-reduced)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend()
plt.grid(True)
plt.show()


# In[31]:


plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x="PCA1", y="PCA2", hue="Cluster", palette="Set2", s=100)
plt.title("K-Means Clusters (PCA-reduced)")
plt.grid(True)
plt.show()


# In[ ]:




