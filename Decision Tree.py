#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


# In[2]:


df=pd.read_csv("https://raw.githubusercontent.com/HarshNerpagar/ML-Lab/refs/heads/main/Titanic-Dataset.csv")


# In[5]:


df.info()


# In[6]:


df.describe()


# In[8]:


df.isnull().sum()


# In[9]:


df.dtypes


# In[11]:


df['Age'].fillna(df['Age'].mean(),inplace=True)
df['Cabin'].fillna(df['Cabin'].mode(),inplace=True)
df['Embarked'].fillna(df['Embarked'].mode(),inplace=True)


# In[13]:


Encoders={
    
}

for columns in df.columns:
    if df[columns].dtype==object:
        encoder=LabelEncoder()
        df[columns]=encoder.fit_transform(df[columns])
        Encoders[columns]=encoder


# In[16]:


X=df.drop(['Survived'],axis=1)
y=df['Survived']


# In[17]:


scaler=StandardScaler()
X=scaler.fit_transform(X)


# In[18]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)


# In[19]:


clf=DecisionTreeClassifier()
clf.fit(X_train,y_train)


# In[20]:


y_pred=clf.predict(X_test)


# In[22]:


from sklearn.metrics import accuracy_score,classification_report
print("Accuracy",accuracy_score(y_test,y_pred))
print("Classification Report",classification_report(y_test,y_pred))


# In[25]:


from sklearn.tree import plot_tree
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay


# In[27]:


cm=confusion_matrix(y_test,y_pred)
ConfusionMatrixDisplay(confusion_matrix=cm).plot()
plt.show()


# In[30]:


plt.figure(figsize=(15,10))
plot_tree(clf,filled=True,feature_names=df.drop(['Survived'], axis=1).columns, class_names=True)
plt.title("Decision Tree Structure")
plt.show()


# In[ ]:




