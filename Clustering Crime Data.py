#!/usr/bin/env python
# coding: utf-8

# In[26]:


import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import sklearn.cluster as cluster
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist 
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn import metrics


# In[2]:


import warnings
warnings.filterwarnings('ignore')


# In[10]:


df= pd.read_csv('crime_data.csv')


# In[11]:


df.head()


# In[12]:


df.columns


# In[13]:


df1= df.rename({'Unnamed: 0':'City'},axis=1)


# In[14]:


df1.columns


# In[15]:


df2 = df1.iloc[:,1:]


# In[16]:


df2.head()


# In[17]:


df2.info()


# In[18]:


# Normalization function 
def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return (x)


# In[19]:


# Normalized data frame (considering the numerical part of data)
X = norm_func(df2.iloc[:,:])


# ##### K Means Clustering

# In[20]:


#Elbow Curve
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

plt.figure(figsize=(10, 8))
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_) #criterion based on which K-means clustering works
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# In[21]:


#Checking with the Silhouette Score for K(clusters)
for i in range(3,13):
    labels=cluster.KMeans(n_clusters=i,init="k-means++",random_state=200).fit(X).labels_
    print ("Silhouette score for k(clusters) = "+str(i)+" is "
           +str(metrics.silhouette_score(X,labels,metric="euclidean",sample_size=1000,random_state=200)))


# ###### We can conclude that optimium number of clusters is 4

# In[22]:


model=KMeans(n_clusters=4) 
model.fit(X)
model.labels_


# In[23]:


km = pd.Series(model.labels_) 
df['kclust']= km 
df.iloc[:,1:5].groupby(df.kclust).mean()


# ##### Hierarchical Clustering

# In[32]:


# create dendrogram
plt.figure(figsize=(10, 7))  
dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))


# ###### The x-axis contains the samples and y-axis represents the distance between these samples. The vertical line with maximum distance is the blue line. If we decide a threshold of 1.5 and cut the dendrogram: 

# In[33]:


plt.figure(figsize=(10, 7))  
plt.title("Dendrograms")  
dend = sch.dendrogram(sch.linkage(X, method='ward'))
plt.axhline(y=1.5, color='r', linestyle='--')
plt.show()


# In[34]:


# With Single Linkage and euclidean distance
# create clusters
hc = AgglomerativeClustering(n_clusters=4, affinity = 'euclidean', linkage = 'single')


# In[37]:


y_hc = hc.fit_predict(X)
Clusters=pd.DataFrame(y_hc,columns=['Clusters'])


# In[42]:


Clusters


# In[43]:


df['hc_clust']= Clusters
df.iloc[:,1:7].groupby(df.hc_clust).mean()


# In[46]:


plt.figure(figsize=(12,6))
sns.scatterplot(x=df['Murder'], y =df['Assault'], hue=df['kclust'])


# In[56]:


plt.figure(figsize=(12,6))
sns.scatterplot(x=df['Murder'], y =df['Assault'], hue=df['hc_clust'])


# In[66]:


# With complete Linkage and euclidean distance
# create clusters
hc = AgglomerativeClustering(n_clusters=4, affinity = 'euclidean', linkage = 'complete')
y_hc = hc.fit_predict(X)
Clusters=pd.DataFrame(y_hc,columns=['Clusters'])
Clusters


# In[67]:


df['hc_clust']= Clusters
df.iloc[:,1:7].groupby(df.hc_clust).mean()


# In[68]:


plt.figure(figsize=(12,6))
sns.scatterplot(x=df['Murder'], y =df['Assault'], hue=df['kclust'])


# In[69]:


plt.figure(figsize=(12,6))
sns.scatterplot(x=df['Murder'], y =df['Assault'], hue=df['hc_clust'])


# In[79]:


# With average Linkage and euclidean distance
# create clusters
hc = AgglomerativeClustering(n_clusters=4, affinity = 'euclidean', linkage = 'average')
y_hc = hc.fit_predict(X)
Clusters=pd.DataFrame(y_hc,columns=['Clusters'])
Clusters


# In[72]:


df['hc_clust']= Clusters
df.iloc[:,1:7].groupby(df.hc_clust).mean()


# In[73]:


plt.figure(figsize=(12,6))
sns.scatterplot(x=df['Murder'], y =df['Assault'], hue=df['kclust'])


# In[74]:


plt.figure(figsize=(12,6))
sns.scatterplot(x=df['Murder'], y =df['Assault'], hue=df['hc_clust'])


# In[80]:


# With ward Linkage and euclidean distance
# create clusters
hc = AgglomerativeClustering(n_clusters=4, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(X)
Clusters=pd.DataFrame(y_hc,columns=['Clusters'])
Clusters


# In[81]:


df['hc_clust']= Clusters
df.iloc[:,1:7].groupby(df.hc_clust).mean()


# In[82]:


plt.figure(figsize=(12,6))
sns.scatterplot(x=df['Murder'], y =df['Assault'], hue=df['kclust'])


# In[83]:


plt.figure(figsize=(12,6))
sns.scatterplot(x=df['Murder'], y =df['Assault'], hue=df['hc_clust'])


# In[ ]:




