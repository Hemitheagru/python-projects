# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 15:09:38 2025

@author: asus
"""

import numpy as np

import pandas as pd
import matplotlib.pyplot as plt

Data=pd.read_csv('Ads_CTR_Optimisation.csv')



#thompson
import math
import random
N=10000#10000 tıklama
d=10#toplam 10 ilan var
#Ri(n)
oduller=[0]*d#ilk başta bütün ilanların ödülü sıfır
#Ni(n)
tiklamalar=[0]*d#o ana kadarki tıklamalar
toplam=0#toplam odul en başta sIfır
secilenler=[]
birler=[0]*d
sifirlar=[0]*d
for n in range(1,N):
    ad=0
    max_th=0
    for i in range(0,d):
        rasbeta=random.betavariate(birler[i]+1,sifirlar[i]+1)
        if rasbeta>max_th:  
            max_th=rasbeta
            ad=i
    secilenler.append(ad)
    odul=Data.values[n,ad]
    if odul==1:
        birler[ad]=birler[ad]+1
    else:
        sifirlar[ad]=sifirlar[ad]+1
    toplam=toplam+odul             
    
print("toplam odul")
print(toplam)


plt.show(plt.hist(secilenler))