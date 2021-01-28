#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pylab as plt 
Airlines = pd.read_csv("EastWestAirlines1.csv")


# In[2]:


#Normalization function
def norm_func(i):
    x = (i-i.mean())/(i.std())
    return (x)


# In[3]:


# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(Airlines.iloc[:,1:])


# In[4]:


from scipy.cluster.hierarchy import linkage 

import scipy.cluster.hierarchy as sch # for creating dendrogram 

type(df_norm)


# In[5]:


#p = np.array(df_norm) # converting into numpy array format 
z = linkage(df_norm, method="complete",metric="euclidean")


# In[6]:


plt.figure(figsize=(15, 5));
plt.title('Hierarchical Clustering Dendrogram');
plt.xlabel('Index');
plt.ylabel('Distance')
sch.dendrogram(
    z,
    leaf_rotation=0.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
plt.show()


# In[9]:


from sklearn.cluster import AgglomerativeClustering 
h_complete=AgglomerativeClustering(n_clusters=3,linkage='complete',affinity="euclidean").fit(df_norm)


# In[10]:


cluster_labels=pd.Series(h_complete.labels_)


# In[16]:


Airlines['clust']=cluster_labels # creating a  new column and assigning it to new column 
Airlines = Airlines.iloc[:,[0,12,1,2,3,4,5,6,7,8,9,10,11]]
Airlines.head()


# In[13]:


# getting aggregate mean of each cluster
Airlines.iloc[:,2:].groupby(Airlines.clust).median()


# ### Kmeans Clustering

# In[20]:


#elbow Curve

from scipy.spatial.distance import cdist 
from sklearn.cluster import KMeans
k = list(range(2,15))
k
TWSS = [] # variable for storing total within sum of squares for each kmeans 
for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df_norm)
    
    WSS = [] # variable for storing within sum of squares for each cluster 
    for j in range(i):
        WSS.append(sum(cdist(df_norm.iloc[kmeans.labels_==j,:],kmeans.cluster_centers_[j].reshape(1,df_norm.shape[1]),"euclidean")))
    TWSS.append(sum(WSS))



# In[21]:


plt.plot(k,TWSS, 'ro-');
plt.xlabel("No_of_Clusters");
plt.ylabel("total_within_SS");
plt.xticks(k)


# In[22]:


# Selecting 5 clusters from the above scree plot which is the optimum number of clusters 
model=KMeans(n_clusters=5) 
model.fit(df_norm)


# In[23]:


# getting the labels of clusters assigned to each row 
model.labels_ 


# In[24]:


# converting numpy array into pandas series object
md=pd.Series(model.labels_)  


# In[27]:


# creating a  new column and assigning it to new column 
Airlines['clust']=md 
df_norm.head()


# In[45]:


Airlines = Airlines.iloc[:,[7,0,1,2,3,4,5,6]]

Airlines.iloc[:,1:7].groupby(Airlines.clust).mean()


# In[46]:


Airlines.to_csv("EastWestAirlines.csv")

