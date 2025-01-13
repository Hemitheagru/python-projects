# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 16:09:00 2025

@author: asus
"""

import numpy as np

import pandas as pd
import matplotlib.pyplot as plt

Data=pd.read_csv('Ads_CTR_Optimisation.csv')
#random selection
'''import random

N=10000
d=10
toplam=0
secilenler=[]
for n in range(0,N):
    ad=random.randrange(d)
    secilenler.append(ad)
    odul=Data.values[n,ad]#data n.satır =1 ise odul 1
    toplam=toplam+odul
    
plt.show(plt.hist(secilenler))'''

#upper confidence bound
import math
N=10000#10000 tıklama
d=10#toplam 10 ilan var
#Ri(n)
oduller=[0]*d#ilk başta bütün ilanların ödülü sıfır
#Ni(n)
tiklamalar=[0]*d#o ana kadarki tıklamalar
toplam=0#toplam odul en başta sIfır
secilenler=[]
for n in range(1,N):
    ad=0
    max_ucb=0
    for i in range(0,d):
        if(tiklamalar[i]>0):       
            ortalama=oduller[i]/tiklamalar[i]
            delta=math.sqrt(3/2* math.log(n)/tiklamalar[i])
            ucb=ortalama+delta
        else:
            ucb=N*10
        if max_ucb<ucb:#maxtan büyük bir ucb çıktı ise
            max_ucb=ucb
            ad=i
    secilenler.append(ad)
    tiklamalar[ad]=tiklamalar[ad] + 1
    odul=Data.values[n,ad]
    oduller[ad]=oduller[ad]+odul
    toplam=toplam+odul
print("toplam odul")
print(toplam)


plt.show(plt.hist(secilenler))