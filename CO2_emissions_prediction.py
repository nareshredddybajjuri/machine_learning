#!/usr/bin/env python
# coding: utf-8

# In[14]:


#import_libs
import pandas as pd
from sklearn import linear_model


# In[4]:


#load_data
df = pd.read_csv("")
df


# In[7]:


#define_dependent_and_independent_Variables
X = df[['Weight', 'Volume']]
Y = df['CO2']


# In[8]:


#From the sklearn module we will use the LinearRegression() method to create a linear regression object.
regr = linear_model.LinearRegression()
regr.fit(X,Y)


# In[15]:


#predict_CO2 values based on a car's weight and volume:
predictedCO2 = regr.predict([[2300, 1300]])
print(predictedCO2)


# In[21]:


#save the model to a file 
import joblib
joblib.dump(predictedCO2,'CO2_model')


# In[ ]:




