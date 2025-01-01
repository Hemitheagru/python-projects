# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
##veriönisleme
veriler=pd.read_csv('eksikveriler.csv')
veriler
##eksik verilerin tamamlanması
from sklearn.impute import SimpleImputer
imputer=SimpleImputer(missing_values=np.nan,strategy="mean")
yas=veriler.iloc[:,1:4].values
yas
imputer=imputer.fit(yas[:,1:4])
yas[:,1:4]=imputer.transform(yas[:,1:4])
yas
## ülke verilerinin numerik hale getilmesi
ulke=veriler.iloc[:,0:1].values
ulke
from sklearn import preprocessing
le=preprocessing.LabelEncoder()
ulke[:,0]=le.fit_transform(veriler.iloc[:,0])
print(ulke)
ohe=preprocessing.OneHotEncoder()
ulke=ohe.fit_transform(ulke).toarray()
ulke
sonuc=pd.DataFrame(data=ulke,index=range(22),columns=["fr","tr","us"])
sonuc
sonuc2=pd.DataFrame(data=yas,index=range(22),columns=["boy","kilo","yas"])
sonuc2
cinsiyet=veriler.iloc[:,-1].values
cinsiyet
sonuc3=pd.DataFrame(data=cinsiyet,index=range(22),columns=["cinsiyet"])
sonuc3
s=pd.concat([sonuc,sonuc2],axis=1)
s
s2=pd.concat([s,sonuc3],axis=1)
s2
## eğitim ve test kümesinin bölünmesi
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(s,sonuc3,test_size=0.33,random_state=0)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(x_train)
X_test=sc.fit_transform(x_test)
