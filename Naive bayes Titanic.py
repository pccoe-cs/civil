#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score,classification_report


# In[3]:


df=pd.read_csv("https://raw.githubusercontent.com/HarshNerpagar/ML-Lab/refs/heads/main/Titanic-Dataset.csv")


# In[4]:


df


# In[5]:


df.info()


# In[6]:


df.shape


# In[7]:


df.describe()


# In[8]:


df.isnull().sum()


# In[9]:


df['Age'].fillna(df['Age'].mean(),inplace=True)
df['Cabin'].fillna(df['Cabin'].mode(),inplace=True)
df['Embarked'].fillna(df['Embarked'].mode(),inplace=True)


# In[10]:


df.dtypes


# In[11]:


# Encoders={
    
# }
# for columns in df.columns:
#     if df[columns].dtype==object:
#         encoder=LabelEncoder()
#         df[columns]=encoder.fit_transform(df[columns])
#         Encoders=encoder


# In[12]:


Encoders={
    
}
for columns in df.columns:
    if df[columns].dtype==object:
        encoder=LabelEncoder()
        df[columns]=encoder.fit_transform(df[columns])
        Encoders[columns]=encoder


# In[13]:


X=df.drop(['Survived'],axis=1)
y=df['Survived']


# In[14]:


len(X), len (y)


# In[15]:


scaler=StandardScaler()
X=scaler.fit_transform(X)


# In[16]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)


# In[17]:


clf=GaussianNB()
clf.fit(X_train,y_train)


# In[18]:


y_pred=clf.predict(X_test)


# In[19]:


print("Accuracy Report: ",accuracy_score(y_pred,y_test))
print(" ",classification_report(y_pred,y_test))


# In[20]:


from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
cm=confusion_matrix(y_test,y_pred)
ConfusionMatrixDisplay(confusion_matrix=cm).plot()
plt.title("Naive Bayes Confusion Matrix")
plt.show()


# In[22]:


sns.heatmap(cm)


# In[ ]:




