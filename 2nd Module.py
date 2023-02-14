# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#Importing Packages
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates



#Importing Data

dktc = pd.read_csv('/home/AliAzzam/Downloads/Surge Clustring/SurgeBase.csv')

#EDA
dktc.info()
pd.set_option('display.precision',1)
dktc.describe()
dktc.info() 
cols = dktc.columns.tolist()
print(cols)
cols_n = cols[3:8] 
dksub = dktc[cols_n].copy()  #new n-only dataframe
dksub.info()                               #copy make sure changes to


# Applying Pre-Processing

mmscaler = preprocessing.MinMaxScaler()  
dksub_mm = pd.DataFrame(mmscaler.fit_transform(dksub),columns = dksub.columns)
dksub_mm.describe()


#Clustering USing ML

from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
Z = linkage(dksub_mm, metric='euclidean', method='single')

Zavg = linkage(dksub_mm, metric='euclidean', method='average')

Zward = linkage(dksub_mm, metric='euclidean', method='ward')


#Polting  Dendogram for Average Linkage method

th_avg = 0.52
dendrogram(Zavg, color_threshold=th_avg)
plt.axhline(y=th_avg, c='grey', lw=2, linestyle='dashed')
plt.text(150,th_avg, 't='+str(th_avg))

#Polting  Dendogram for ward method

th_ward = 4
dendrogram(Zward, color_threshold=th_ward)
plt.axhline(y=th_ward, c='grey', lw=2, linestyle='dashed')
plt.text(150,th_ward, 't='+str(th_ward))
plt.title("Dendrogram with Wards Method for Clustering")



#Mapping CLusters to datapoints
dktc['Wards'] = fcluster(Zward, 5, criterion='maxclust')


dktc['Average_Linkage'] = fcluster(Zavg, 5, criterion='maxclust')


#Ecporting Data
dktc.to_excel('Surge Clustring/F5C.xlsx')




