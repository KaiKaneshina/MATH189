#!/usr/bin/env python
# coding: utf-8

# # HW 1, Problem 2 : Kai Kaneshina

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# ## Part C

# In[2]:


x = np.array([0,2,3,4])
y = np.array([1,3,6,8])

m = 63/35
b = 18/35

y_hat = m*x + b


# In[3]:


plt.plot(x,y, 'o', label='points')
plt.plot(x,y_hat, label='optimal linear fit')
plt.legend()


# # Part D

# In[10]:


# generate points
x_new = np.arange(100)
# generate noise
noise = np.random.normal(0, 1, 100)
# add noise around the line
y_noise = m*x_new + b + noise


# In[11]:


# compute X matrix
X = np.array([x_new,np.ones_like(x_new)]).T
# compute theta using normal equations
theta = np.matmul(np.matmul(np.linalg.inv(np.matmul(X.T,X)), X.T), y_noise)
# calculate new line: 
m_new = theta[0] 
b_new = theta[1]
y_new = m_new*x_new + b_new

# get points for old line
y_old = m*x_new + b


# In[12]:


plt.plot(x_new,y_noise, 'o', label='line with noise')
plt.plot(x_new,y_new, label='line fit on noise data')
plt.plot(x_new, y_old, label='old line (optimal linear fit from part C)')
plt.legend()


# ## the new and old lines are close to each other. They are touching so close that you cannot even see them in the plot above

# ### look at the old vs new slopes (m)

# In[13]:


print(m_new) # new


# In[14]:


print(m) # old


# ### look at the old vs new intercepts (b)

# In[15]:


print(b_new) # new


# In[16]:


print(b) # old


# In[ ]:




