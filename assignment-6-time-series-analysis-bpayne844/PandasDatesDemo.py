# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 08:25:25 2020

@author: brandi.payne
"""

import pandas as pd
import numpy as np
from pandas import Series, DataFrame
import matplotlib.pyplot as plt

#Load and read file 
ao = np.loadtxt('monthly.ao.index.b50.current.ascii.txt')

# put values into a series array
ao[0:2]

#shape of array 
ao.shape

#covert data to time series. Bring in all data to be timestamped by frequnecy of months
dates = pd.date_range('1950-01', periods = ao.shape[0], freq = 'M')

#shape of array
dates.shape

#create time series 
AO = Series(ao[:,2], index = dates)

AO

#plot all data in 1 plot
AO.plot()
plt.xlabel('Year')
plt.ylabel('Atlantic Oscillation Value')
plt.savefig('Daily Atlantic Oscillation.pdf')

#plot 1980 to 1990
print(AO["1980" : "1990"].plot())

print(AO['1980-05': '1981-03'].plot())

#explore different ways to reference in the dataframe
print(AO[120])

print(AO['1960-01'])

print(AO['1960'])

print(AO[AO>0])


# DataFrame Next section of Tutorial 
#Load and read file, and create series
nao = np.loadtxt('norm.nao.monthly.b5001.current.ascii.txt')
dates_nao = pd.date_range('1950-01', periods=nao.shape[0], freq='M')
NAO = Series(nao[:,2], index=dates_nao)

print(NAO.index)

#create dataframe first column contains headers 
aonao = DataFrame({'AO':AO,'NAO':NAO})

#view all data in sub plots 
print(aonao.plot(subplots=True))

#view first couple rows
print(aonao.head())

#reference a column
print(aonao['NAO'])

#add column to dataframe
aonao['Diff'] = aonao['AO'] - aonao['NAO']
print(aonao.head())

#delete diff column and view last few rows 
del aonao['Diff']
print(aonao.tail())

#slicing the dataframe
print(aonao['1981-01': '1981-03'])

#indexing to plot by choosing NAO values in the 1980's for month AO is positive and NAO is negative 
import datetime
aonao.loc[(aonao.AO > 0) & (aonao.NAO < 0) 
        & (aonao.index > datetime.datetime(1980,1,1)) 
        & (aonao.index < datetime.datetime(1989,1,1)),
        'NAO'].plot(kind='barh')

#Statistics Next section in tutorial 
#default is columnwise

#average both columns
print(aonao.mean())

#max value of each column
print(aonao.max())

#min value of each column
print(aonao.min())

#average by row
print(aonao.mean(1))

#get all stats at once
print(aonao.describe())

#Resampling

#calculate annual mean
AO_mm = AO.resample('A').mean()
print(AO_mm.plot(style = 'g--'))


#calculate Median
AO_mm = AO.resample("A").median()
AO_mm.plot()
plt.xlabel('Year')
plt.ylabel('Atlantic Oscillations Values')
plt.savefig('Annual Median Values for AO.pdf')


#change method to methods for resample, freq changed to every 3 year sample
AO_mm = AO.resample("3A").apply(np.max)
AO_mm.plot()

#example using several function at once to get subplots individuals and all in one plot 
AO_mm = AO.resample("A").apply(['mean', np.min, np.max])
AO_mm['1900':'2020'].plot(subplots=True)
AO_mm['1900':'2020'].plot()

#view whole table od stats 
print(AO_mm)

#rolling statsistics

#rolling mean
aonao.rolling(window=12, center=False).mean().plot(style='-g')
plt.xlabel('Year')
plt.ylabel('Rolling Mean Values')
plt.savefig('Rolling Mean for Both AO and NAO.pdf')

#rolling correlation
aonao.AO.rolling(window=120).corr(other=aonao.NAO).plot(style='-g')

#view correlation coefficients
print(aonao.corr())



