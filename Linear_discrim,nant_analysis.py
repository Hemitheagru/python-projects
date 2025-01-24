# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 16:32:14 2025

@author: asus
"""

#LdA gözetimli
#sınıfları birbirindan ayıran eniyi algoritmayı bulmak 
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

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda=LDA(n_components=2)
X_train_lda=lda.fit_transform(X_train,y_train)#lda sınıfları öğreniyor bu yüzden 2 parametre birden alıyor
X_test_lda=lda.transform(X_test)

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0)
classifier.fit(X_train,y_train)

classifier_lda=LogisticRegression(random_state=0)
classifier_lda.fit(X_train_lda,y_train)

y_pred=classifier.predict(X_test)
y_pred_lda=classifier_lda.predict(X_test_lda)

from sklearn.metrics import confusion_matrix
print("lda olmadan")
cm1=confusion_matrix(y_test,y_pred)
print(cm1)
print("lda ile")
cm1=confusion_matrix(y_test,y_pred_lda)
print(cm1)


