#Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:, :-1].values
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
