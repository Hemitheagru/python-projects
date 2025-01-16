# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 15:43:38 2025

@author: asus
"""

import numpy as np
import pandas as pd


reviews=pd.read_csv('reviews.txt',sep=';',names=['Review','Liked'])
import re
import nltk

from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

nltk.download('stopwords')
from nltk.corpus import stopwords

pl=[]
for i in range (1000):
        review=re.sub('[^a-aZ-Z]','  ',reviews['Review'][i])
        review=review.lower()
        review=review.split()
        review=[ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
        review=' '.join(review)
        pl.append(review)
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=1000)
X=cv.fit_transform(pl).toarray()#bağımsız değişken 
y=reviews.iloc[:,1].values#bağımlı değişken

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train,y_train)

y_pred = gnb.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)
