#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
import pandas as pd
import seaborn as sns


# In[2]:


sales = pd.read_csv('C:/Users/Sharon Mbanga/OneDrive/Desktop/Data/Month_Value_1.csv')
sales.head()


# Average cost = Average cost of production
# 
# The average annual payroll of the region = The average number of employees in the region per year

# In[26]:


import matplotlib.pyplot as plt
sales.plot(kind='scatter', x='Sales_quantity', y='Revenue', alpha=1)


# In[29]:


sns.lmplot(x='Sales_quantity', y='Revenue', data=sales, scatter_kws={'s':50})


# In[3]:


sales.info()


# No missing values. All columns are numerical. 

# In[4]:


sales.shape


# The data has 96 observations and 5 features

# In[5]:


sales.describe()


# In[6]:


import matplotlib.pyplot as plt
import seaborn as sns


# The Revenue feature is not normally distributed.

# In[7]:


sns.displot(sales['Revenue'])


# In[8]:


sales['Revenue'].skew()


# The Revenue feature is skewed to the right but the skewnes is low

# In[9]:


print("Minimum Revenue: \t",sales['Revenue'].min())
print("Maximum Revenue: \t",sales['Revenue'].max())
print("Average Revenue: \t",sales['Revenue'].mean())
print("Standard deviation of Revenue: \t",sales['Revenue'].std())


# In[10]:


x=sales['The_average_annual_payroll_of_the_region']
y=sales['Revenue']
plt.scatter(x,y)

plt.rcParams.update({'figure.figsize':(10,8), 'figure.dpi':100})
plt.title('Simple Scatter plot')
plt.xlabel('The average annual payroll of the region')
plt.ylabel('Revenue')
plt.show()


# There is no correlation between The average annual payroll of the region and Revenue

# ### Dealing with missing values

# In[15]:


sales= sales.dropna()
sales.shape


# ### Stationarity plot

# In[16]:


from statsmodels.tsa.ar_model import AutoReg
x = sales.values
sales.plot()


# The data is not stationary sine the mean values increases with time

# ### Check for stationarity

# In[17]:


from statsmodels.tsa.stattools import adfuller
stationary_feature = adfuller(sales['Revenue'], autolag='AIC')
print("1. ADF: ", stationary_feature[0])
print("2. P-Value: ", stationary_feature[1])
print("3. Number of Lags: ", stationary_feature[2])
print("4. Number of observations used for ADF Regression and Critical values calculation: ", stationary_feature[3])
print("5. Critical Values: ")
for key, val in stationary_feature[4].items():
    print("\t", key, ": ", val)


# P-Value is greater than 0.05 showing our data is not stationary. We need to make it stationary

# ### Making the data stationary

# In[18]:


#Using log operation method
stationary_feature_log = np.sqrt(sales['Revenue'])
stationary_feature_diff = stationary_feature_log.diff().dropna()


# In[19]:


stationary_feature = adfuller(stationary_feature_diff)
print("1. ADF: ", stationary_feature[0])
print("2. P-Value: ", stationary_feature[1])
print("3. Number of Lags: ", stationary_feature[2])
print("4. Number of observations used for ADF Regression and Critical values calculation: ", stationary_feature[3])
print("5. Critical Values: ")
for key, val in stationary_feature[4].items():
    print("\t", key, ": ", val)


# The p-value is less than 0.05 hence our hypothesis is significant meaning the data is stationary

# ### Plotting ACF nd PACF

# In[20]:


import warnings
warnings.filterwarnings("ignore")
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
pacf = plot_pacf(sales['Revenue'], lags=15)
acf = plot_acf(sales['Revenue'], lags=15)


# We will take the PACF plot, since it is the one recommended for AR model

# ### Splitting Data into training and test sets

# In[83]:


sales.shape


# In[89]:


train = sales.iloc[:45,]
test = sales.iloc[45:, ]
print(train.shape, test.shape)


# ### Fitting our model

# In[91]:


model = AutoReg(train['Revenue'], lags=15).fit()
print(model.summary())


# From the p-values, we can conclude that most variables are not significant in the prediction

# ### Make predictions on Test data

# In[93]:


starting = len(train)
ending = len(sales)-1
pred = model.predict(start=starting, end=ending, dynamic=False )
print(pred)


# In[96]:


plt.plot(pred)
plt.plot(test['Revenue'], color='green')


# ### Estimating Accuracy

# In[97]:


from math import sqrt
from sklearn.metrics import mean_squared_error
rmse = sqrt(mean_squared_error(test['Revenue'], pred))
print(rmse)


# In[98]:


test['Revenue'].mean()


# ### Estimating perentage of our rmse

# In[99]:


(7946245.166028368/43675966.77579369)*100


# Our rmse is small meaning that the model is a better predictor

# ### Making future predictions

# In[100]:


sales.tail()


# In[111]:


future_revenue = pd.date_range(start='01.04.2020', end='02.03.2020')
print(future_revenue)
pred1 = model.predict(start=len(sales), end=len(sales)+30, dynamic=False)
print("The future prediction of revenue for next month")
pred1.index = future_revenue
print(pred1)


# In[106]:


print("Number of predictions made: \t", len(pred1))


# In[108]:


len(future_revenue)


# In[ ]:




