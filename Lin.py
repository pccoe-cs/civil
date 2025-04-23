#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# In[2]:


df=pd.read_csv("https://raw.githubusercontent.com/lazy-punk/jackofalltrades/refs/heads/main/jackofalltrades/datasets/london_house_prices.csv")


# In[3]:


df


# In[4]:


df.shape


# In[5]:


df.info()


# In[6]:


df.isnull().sum()


# In[7]:


df['size_sqft'].fillna(df['size_sqft'].mean(),inplace=True)
df['bedrooms'].fillna(df['bedrooms'].mean(),inplace=True)
df['bathrooms'].fillna(df['bathrooms'].mean(),inplace=True)
df['tenure'].fillna(df['tenure'].mode()[0],inplace=True)
df['postcode_outer'].fillna(df['postcode_outer'].mode()[0],inplace=True)


# In[8]:


df.isnull().sum()


# In[9]:


Encoder={
}

for columns in df.columns:
    if df[columns].dtype==object:
        encoder=LabelEncoder()
        df[columns]=encoder.fit_transform(df[columns])
        Encoder[columns] = encoder


# In[10]:


df.dtypes


# In[11]:


X=df.drop(['price_pounds'],axis=1)


# In[12]:


y=df['price_pounds']


# In[13]:


df


# In[14]:


scaler = StandardScaler()
X = scaler.fit_transform(X)


# In[15]:


X


# In[16]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)


# In[18]:


model=LinearRegression()
model.fit(X_train,y_train )


# In[20]:


y_pred=model.predict(X_test)


# In[22]:


r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)

print("R² Score:", r2)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("Mean Absolute Error (MAE):", mae)


# In[25]:


plt.figure()
plt.scatter(y_test, y_pred, alpha=0.7, color='b')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted Values")
plt.grid(True)
plt.show()


# In[26]:


plt.scatter(y_test, y_pred)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.show()

