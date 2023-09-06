#!/usr/bin/env python
# coding: utf-8

# K-NN Supervised Classification Algorithm
# 
# K-Nearest-Neighbours is used to classify data based on its proximity to similar data points.
# This program takes a data set of customer information. The set includes personal information and a customer
# category 1,2,3 or 4 for the type of service that they recieve.
# Based on the personal information, the K-NN algorithm will try to guess which service should be suggested to a
# new customer. 

# In[52]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import preprocessing,metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


# Read data from CSV into a pandas Data Frame

# In[3]:


df = pd.read_csv('teleCust1000t.csv')
df.head()


# View info of the data

# In[103]:


df.describe()


# In[105]:


df.hist(column='age', bins=50)


# Now split the data into a feature set and a target set. The target set will be the customer category data 'custcat', and the features will be the data in all remaining columns.

# In[106]:


columns = df.columns[:-1]


# In[42]:


#Create feature set
features = df[columns].values
features[:5]


# In[43]:


#Create target set
targets = df[df.columns[-1]].values
targets[:5]


# To improve efficiency, standardise the data by forcing the mean and standard deviation to be equal to 1.

# In[44]:


features = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
features[0:5]


# Split the data into rows for testing and rows for training. This is done using the SKLearn module.

# In[48]:


features_train, features_test, targets_train, targets_test = train_test_split( features, targets, test_size=0.2, random_state=4)
print ('Train set:', features_train.shape,  targets_train.shape)
print ('Test set:', features_test.shape,  targets_test.shape)


# Define the number of Nearest-Neighbours to use, then train the model using the SKLearn function. This determines which data points the algorithm will look at to train the model. The higher the number of neighbours, the more data points will be used to create the classification.

# In[107]:


k = 6
neighbour_model = KNeighborsClassifier(n_neighbors = k).fit(features_train,targets_train)
neighbour_model


# Use SKLearn predict to calculate the model predictions from the testing data set.

# In[51]:


yhat = neighbour_model.predict(features_test)
yhat[0:5]


# Calculate the out of sample accuracy of the model using the accuracy_score method by comparing the predictions to the actual classification value.

# In[55]:


print("Train set Accuracy: ", metrics.accuracy_score(targets_train, neighbour_model.predict(features_train)))
print("Test set Accuracy: ", metrics.accuracy_score(targets_test, yhat))


# This process can be repeated to obtain a more optimal value of K

# In[109]:


test_accuracy = []
k = 11
for i in range(1,k):
    features_train, features_test, targets_train, targets_test = train_test_split( features, targets, test_size=0.2, random_state=4)
    neighbour_model = KNeighborsClassifier(n_neighbors = i).fit(features_train,targets_train)
    neighbour_model
    yhat = neighbour_model.predict(features_test)
    test_accuracy.append(metrics.accuracy_score(targets_test, yhat))
    print("K: ",i,"Test set Accuracy: ", test_accuracy[i-1])   


# In[91]:


plt.figure(figsize=(8,6))
plt.plot(range(1,k),test_accuracy)
plt.ylabel('Accuracy ')
plt.xlabel('Number of Neighbors (K)')
plt.show()


# In[96]:


print( "The best accuracy was with", np.asarray(test_accuracy).max(), "with k=", np.asarray(test_accuracy).argmax()+1) 

