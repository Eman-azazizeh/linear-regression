#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import libraries
import pandas as pd 
import numpy as np




# In[2]:


#import dataset
data_df= pd.read_csv('data.csv')
data_df.head()


# In[3]:


#define x and y 
x=data_df.drop(['B'],axis=1).values
y=data_df['B'].values


# In[4]:


print(x) 


# In[5]:


print(y)


# In[6]:


#spilt the dataset in training set and test set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=2)
#it was bad model when the test size equal to 0.3 (score between t pred and y test was 0.2)but when i reduce th test size the score between t pred and y test become 0.7


# In[7]:


#train the model on training set
from sklearn.linear_model import LinearRegression
ml=LinearRegression()
ml.fit(x_train,y_train)


# In[8]:


#predict the test set result
y_pred=ml.predict(x_test)
print(y_pred)


# In[9]:


ml.predict([[51,30,39,61,92,45]])


# In[10]:


#evaluate the model
from sklearn.metrics import r2_score
r2_score(y_test,y_pred)


# In[11]:


#plot the result 
import  matplotlib.pyplot as plt

plt.figure(figsize=(10,7))
plt.scatter(y_test,y_pred)
plt.xlabel('actual')

plt.ylabel('predict')
plt.title('actual vs. predict')


# In[12]:


#predicted values
pred_y_df=pd.DataFrame({'actual value':y_test,'predicted value ':y_pred,'Difference':y_test-y_pred})
pred_y_df[0:5]


# In[13]:




