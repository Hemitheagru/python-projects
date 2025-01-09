# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 17:03:06 2025

@author: asus
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets

iris=datasets.load_iris()
X=iris.data[:, :2]
y=iris.target
x_min,x_max=X[:, 0].min()-.5,X[:, 0].max()+.5
y_min,y_max=X[: ,1].min()-.5,X[:, 1].max()+.5

plt.figure(2, figsize=(8,6))
plt.clf()

plt.scatter(X[:, 0],X[:, 1],c=y,cmap=plt.cm.Set1,edgecolor="k")
plt.xlabel("Sepal length")
plt.ylabel("Sepal width")
plt.xlim(x_min,x_max)
plt.xlim(y_min,y_max)
plt.xticks(())
plt.yticks(())

fig=plt.figure(1,figsize=(8,6))
ax=Axes3D(fig,elev=150,azim=110)
ax.scatter(iris.data[:, 0],iris.data[:, 1],iris.data[:, 2],c=y,cmap=plt.cm.Set1,edgecolor="k",s=40)
ax.set_title("Iris verisi")
ax.set_xlabel("brinci özellik")
ax.xaxis.set_ticklabels([])
ax.set_ylabel("ikinci ozellik")
ax.yaxis.set_ticklabels([])

ax.set_zlabel("ucuncu ozellik")
ax.zaxis.set_ticklabels([])

plt.show()
import mpl_toolkits.mplot3d  # noqa: F401

from sklearn.decomposition import PCA

fig = plt.figure(1, figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d", elev=-150, azim=110)

X_reduced = PCA(n_components=3).fit_transform(iris.data)
ax.scatter(
    X_reduced[:, 0],
    X_reduced[:, 1],
    X_reduced[:, 2],
    c=iris.target,
    s=40,
)
plt.show()


