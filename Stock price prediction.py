#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')
sns.set_style('darkgrid')


# In[28]:


dataset = pd.read_csv('C:/Users/Harsha/OneDrive/Desktop/ICICIBANK.csv')


# In[ ]:





# In[8]:


dataset.shape


# In[9]:


dataset.columns


# In[12]:


dataset.info()


# In[13]:


dataset.describe()


# In[14]:


display(dataset.head().style.hide_index())


# In[15]:


dataset.drop(dataset.columns.difference(['Date', 'Open', 'Close']), 1, inplace=True)


# In[16]:


display(dataset.head().style.hide_index())


# In[17]:


fig, ax = plt.subplots(figsize=(20, 10))
plot1 = sns.scatterplot(data=dataset.head(100), x="Open", y="Close", ax=ax)
plot1.set(title='Open v/s Close')
plt.show()


# In[18]:


dataset.hist(bins=50, figsize=(20, 6))
plt.show()


# In[19]:


from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


# In[20]:


X = dataset['Open'].values
y = dataset['Close'].values


# In[21]:


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3)
model1 = LinearRegression()
build1 = model1.fit(X_train.reshape(-1, 1), y_train)
predict1 = model1.predict(X_test.reshape(-1, 1))
print("Co-efficient: ", model1.coef_)
print("\nIntercept: ", model1.intercept_)


# In[22]:


df1 = pd.DataFrame(list(zip(y_test, predict1)), columns=["Actual Values", "Predicted Values"])
df1.head().style.hide_index()


# In[23]:


df1.head(50).plot(kind="bar", figsize=(20, 10), title='Simple Linear Regression')
plt.show()


# In[24]:


accuracy1 = r2_score(y_test, predict1)
print("Accuracy of Simple Linear Regression:", accuracy1)


# In[25]:





# In[26]:





# In[ ]:





# In[ ]:





# In[ ]:




