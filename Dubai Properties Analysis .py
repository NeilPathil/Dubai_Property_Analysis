#!/usr/bin/env python
# coding: utf-8

# In[96]:


import pandas as pd 
import numpy as np 
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.preprocessing as StandardScaler
from scipy import stats
from scipy.stats import kurtosis 
from scipy.stats import skew 
from scipy.stats import norm 

get_ipython().run_line_magic('matplotlib', 'inline')


# In[97]:


data = pd.read_csv('properties_data.csv')
data.head()


# In[98]:


data.info()


# In[99]:


data.price.describe() #shows us how the the price coloumn of the data is distributed


# In[100]:


#with this data we shall find the kurtosis and the skewness of the prices 

skew = data.price.skew() #measures the symmetry in a distribution
kurt = data.price.kurt() #measures the the weight of the tails 


# In[101]:


print(skew,kurt)


# In[102]:


print('Skewness: ', format(skew))
print('Kurtosis: ', format(kurt))


# In[103]:


#data shows high skewness, let's plot this on a diagram

fig, ax = plt.subplots(figsize = (8,8))
sns.distplot(data.price)


# In[104]:


#distribution leaning to the left

data_lt5 = data[data.price < 5000000]
data_mt5 = data[data.price >= 5000000]
data_lt5.head()


# In[105]:


#in the previous step we have created two variables in order to split the prices coloumn into price points that are Less Than
#5M and price points that are more than 5m

data_lt5.price.describe()


# In[106]:


data_mt5.price.describe()


# In[107]:


#from the above analysis we can see that we have 89 results that are more than 5m while the rest are not
#this great variance in the result might affect our analysis so we will remove the data

fig, ax = plt.subplots(figsize = (8,8))
sns.distplot(data_lt5.price)


# In[108]:


# above is the distribution of house prices that are less than 5m

print('Skewness_lt5: ',format(data_lt5.price.skew()))
print('Kurtosis_lt5: ', format(data_lt5.price.kurt()))
      
      


# In[112]:


#even though the skewness is above one it is still better than the previous score of 6+
#The next section will look at the relationship between price and location

data_neighbor_price = data[['price','neighborhood']].sort_values(by=['price'], ascending = False)

plt.figure(figsize = (30, 10))
plt.bar(data_neighbor_price.neighborhood, data_neighbor_price.price, align = 'center', alpha = 0.5)
plt.xticks(rotation = 'vertical')
plt.show()


# In[113]:


#analysing sqft price and location

data_sqftprice_location = data[['price_per_sqft', 'neighborhood']].sort_values(by='price_per_sqft', 
                                                                          ascending = False)
data_sqftprice_location['neighborhood'].nunique()

plt.figure(figsize = (30,10))
plt.bar(data_sqftprice_location.neighborhood, data_sqftprice_location.price_per_sqft, align = 'center', alpha = 0.5)
plt.xticks(rotation = 'vertical')
plt.show


# In[111]:


#now we analyze price and area 

plt.figure(figsize = (10,8))
feature = 'size_in_sqft'
plt.scatter(data[feature], data['price'])
plt.xlabel('Size in SqFt')
plt.grid(True)


# In[121]:


#price vs number of bedrooms

data_price_bedrooms = data[['price', 'no_of_bedrooms']].sort_values(by = 'price', ascending = False)


plt.figure(figsize = (10,8))
plt.scatter(data['no_of_bedrooms'],data['price'])
plt.xlabel('Number of Bedrooms')
plt.ylabel('Price')
plt.ylim(ymin = 200000, ymax = 40000000)
plt.grid(True)
plt.show


# In[127]:


#Finally, finding overall correlation

fig, ax = plt.subplots(figsize = (15,10))
correlation = data.corr()
sns.heatmap(correlation,annot = False)
plt.show


# In[ ]:




