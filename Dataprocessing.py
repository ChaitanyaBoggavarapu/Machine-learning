# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 22:30:34 2018

@author: Chaitanya
"""

#Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset = pd.read_csv('Data.csv')
#Selecting the Features Independent Variable

x = dataset.iloc[:, :-1].values

#Selecting the Features dependent Variable
y = dataset.iloc[:, 3].values

#to correct the missing values : Fiiling the missing values with the mean in columns
#Importing SCIKIT learn
#Imputer class used to fix missing data
from sklearn.preprocessing import Imputer
#
imputer = Imputer(missing_values = 'NaN',strategy = 'mean',axis = 0)
#imputer.fit is used to fit the values which has to be corrected
imputer = imputer.fit(x[:,1:3])
x[:,1:3] = imputer.transform(x[:,1:3])

#Catagorical Variables
#In Data Country and Purchased we have different catagories and also we should convert them to numbers
#Label Encoder labels the catagories with Intezers
from sklearn.preprocessing import LabelEncoder
labelencoder_x = LabelEncoder()
x[:,0] = labelencoder_x.fit_transform(x[:,0])
#After Transforming France = 0,Germany = 1,Spain = 2
#Which might assume spain > Farnce
#WE use onehotencoder class to make dummy indesxes in dataset
from sklearn.preprocessing import OneHotEncoder
onehotencoder_x = OneHotEncoder(categorical_features = [0])
x = onehotencoder_x.fit_transform(x).toarray() 

#We label our Y dependent Variable
#For Dependent Varaible numbers does not matter because it is the output
#So we need not to use OneHotEncoder
from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)






