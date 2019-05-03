# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 15:17:21 2019

@author: Abhilash
"""

import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from sklearn.datasets.samples_generator import make_blobs


#Training Samples = 50
#Number of features  = 1
X = 2 * np.random.rand(50,1)
y = 4 +3 * X+np.random.randn(50,1)

Cost = []
itera = []
def gradientdescent(theta,x,y,iterations,learningrate,b):
    for i in range(iterations):
        theta1 = theta
        #Caluculating the hypothesis using thate
        pred = np.dot(x,theta)
        pred = pred.reshape(50,1)
        
        #Do the transpose of X
        c = np.transpose(x)
        g = pred-y    
        Costit = 0.5*(sum(g)**2)/float(len(X))
        Cost.append(Costit)
        itera.append(i)
        b = ((sum(y-pred))/float(len(X)))
        theta = 0.5*(theta - (learningrate*np.dot(c,g)))
       
       
    
    return theta,b,Cost,itera

theta = np.array([0.5])

#theta = gradientdescent(theta,gaussian1,gaussian2,100,0.01)
theta,b,Cost,itera = gradientdescent(theta,X,y,100,0.01,0)

#ypred = np.dot(gaussian1,theta)

plt.scatter(X,y)
plt.plot(X,(b+X*theta),'-',color='red')
plt.show()


plt.plot(itera,Cost)
plt.show()




