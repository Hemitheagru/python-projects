# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 14:42:14 2025

@author: asus
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

veriler=pd.read_csv('musteriler.csv')

X=veriler.iloc[:,3:]
#KMeans
from sklearn.cluster import KMeans
kmeans=KMeans(n_clusters=3,init='k-means++')
kmeans.fit(X)

print(kmeans.cluster_centers_)
sonuclar=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++',random_state=123)
    kmeans.fit(X)
    sonuclar.append(kmeans.inertia_)#wcss değerlerini hesaplayan kod
    
plt.show(plt.plot(range(1,11),sonuclar))
#KMeans
from sklearn.cluster import KMeans
kmeans=KMeans(n_clusters=4,init='k-means++',random_state=123)
Y_tahmin=kmeans.fit_predict(X)
Y_tahmin
plt.scatter(X[Y_tahmin==0,0],X[Y_tahmin==0,1], s=100, c='red')
plt.scatter(X[Y_tahmin==1,0],X[Y_tahmin==1,1],s=100,c='blue')
plt.scatter(X[Y_tahmin==2,0],X[Y_tahmin==2,1],s=100,c='green')
plt.scatter(X[Y_tahmin==3,0],X[Y_tahmin==3,1],s=100,c='yellow')

#HC
from sklearn.cluster import AgglomerativeClustering
ac=AgglomerativeClustering(n_clusters=4,metric='euclidean',linkage='ward')
Y_tahmin=ac.fit_predict(X)
print(Y_tahmin)

#dendogram

import scipy.cluster.hierarchy as sch
dendrogram=sch.dendrogram(sch.linkage(X,method='ward'))
plt.show()



