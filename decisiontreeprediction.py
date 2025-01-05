# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 15:39:10 2025

@author: asus
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
veriler=pd.read_csv('egitimseviyesitahmin.csv')

x=veriler.iloc[:,1:2]
y=veriler.iloc[:,2:]
X = x.values
Y = y.values

from sklearn.tree import DecisionTreeRegressor
r_dt=DecisionTreeRegressor(random_state=0)
r_dt.fit(X,Y)

plt.scatter(X,Y)
plt.show(plt.plot(X,r_dt.predict(X),color='r'))

print(r_dt.predict([[6.6]]))
print(r_dt.predict([[11]]))


from  sklearn.ensemble import RandomForestRegressor
rf_reg=RandomForestRegressor(n_estimators=10,random_state=0)
rf_reg.fit(X,Y.ravel())
print(rf_reg.predict([[6.6]]))
plt.scatter(X,Y,color='red')
plt.show(plt.plot(X,rf_reg.predict(X),color='b'))

from  sklearn.ensemble import RandomForestRegressor
rf_reg=RandomForestRegressor(n_estimators=5,random_state=0)
rf_reg.fit(X,Y.ravel())
print(rf_reg.predict([[6.6]]))


plt.scatter(X,Y,color='red')
plt.show(plt.plot(X,rf_reg.predict(X),color='b'))



##r square hata payının ölçümünü gösterir 1e yaklaştıka başarı yükselir


from sklearn.metrics import r2_score

print(r2_score(Y, rf_reg.predict(X)))
