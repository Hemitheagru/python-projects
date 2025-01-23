# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 15:48:05 2025

@author: asus
"""

import pandas as pd
veriler=pd.read_csv('Wine.csv')
veriler
X=veriler.iloc[:,0:13].values
Y=veriler.iloc[:,13].values

from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(X,Y,test_size=0.33, random_state=0)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()

X_train=sc.fit_transform(x_train)
X_test=sc.fit_transform(x_test)

from sklearn.decomposition import PCA

pca=PCA(n_components=2)
X_train2=pca.fit_transform(X_train)
X_test2=pca.transform(X_test)

from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0)
classifier.fit(X_train,y_train)

classifier2=LogisticRegression(random_state=0)
classifier2.fit(X_train2,y_train)

y_pred=classifier.predict(X_test)
y_pred2=classifier2.predict(X_test2)


from sklearn.metrics import confusion_matrix
print("pca olmadan")
cm1=confusion_matrix(y_test,y_pred)
print(cm1)
print("pca li")
cm2=confusion_matrix(y_test,y_pred2)
print(cm2)
print("pca li,pca siz karşılaştırma")
cm3=confusion_matrix(y_pred,y_pred2)
print(cm3)