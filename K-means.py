# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 14:03:30 2020

@author: abhil
"""
import numpy as np
from matplotlib import pyplot as plt

from sklearn.datasets.samples_generator import make_blobs

mean1 = [1,2]

sigma1 = np.array([[3,1],[1,2]])

mean2 = [-1,-2]

sigma2 = np.array([[2,0],[0,1]])

mean3 = [3,-3]

sigma3 = np.array([[1,.3],[.3,1]])


x1, y1 = np.random.multivariate_normal(mean1, sigma1, 100).T
x1 = x1.reshape(100,1)
y1 = y1.reshape(100,1)

x2, y2 = np.random.multivariate_normal(mean1, sigma1, 100).T
x2 = x2.reshape(100,1)
y2 = y2.reshape(100,1)

x3, y3 = np.random.multivariate_normal(mean1, sigma1, 200).T
x3 = x3.reshape(200,1)
y3 = y3.reshape(200,1)



#X = np.append(x1,y1,axis=1)
X = np.append(x1,y1,axis=1)

Y = np.append(x2,y2,axis=1)

Z = np.append(x3,y3,axis=1)

XX = np.append(X,Y,axis=0)

XY = np.append(XX,Z,axis=0)


##Asumming two clusters
Center = ([[-1.0,3.0],[-2.0,4.0],[2.0,1.25]])
print(len(Center))
#k = 2


##k-means algorithm


for k in range(50):
    
    firstcluster = []
    secondcluster = []
    thirdcluster = []
    for i in range(len(XY)):
        a = []
        for j in range(len(Center)):
            distancefrompointtocenteroid = np.sqrt( (XY[i][0]-Center[j][0])**2 + (XY[i][1]-Center[j][1])**2 )
            a.append(distancefrompointtocenteroid)
        minpos = a.index(min(a))          
        if (minpos==0):
            print('if')
            firstcluster.append([XY[i][0],XY[i][1]])
        elif(minpos==1):
            print('elif')
            secondcluster.append([XY[i][0],XY[i][1]])
        else:
            print('inelse')
            thirdcluster.append([XY[i][0],XY[i][1]])
    firstcluster = np.asarray(firstcluster)
    secondcluster = np.asarray(secondcluster)    
    thirdcluster = np.asarray(thirdcluster)
    newcenter = ([[np.mean(firstcluster[:][:,0]),np.mean(firstcluster[:][:,1])],[np.mean(secondcluster[:][:,0]),np.mean(secondcluster[:][:,1])],[np.mean(thirdcluster[:][:,0]),np.mean(thirdcluster[:][:,1])]])
         
    if(newcenter == Center):
        print(k)
        print('in thsi')
        break
    else:
        Center = newcenter
        

plt.scatter(firstcluster[:][:,0],firstcluster[:][:,1],s=8)
plt.scatter(secondcluster[:][:,0],secondcluster[:][:,1],c='g',s=8)
plt.scatter(thirdcluster[:][:,0],thirdcluster[:][:,1],c='r',s=8)
plt.xticks(np.arange(-3, 7, 1)) 
plt.yticks(np.arange(-3, 7, 1)) 
plt.show()


 