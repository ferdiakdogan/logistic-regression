# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 21:49:10 2020

@author: Ferdi
"""

import numpy as np
from matplotlib import pyplot as plt
import scipy.optimize as opt

file = open("ex2data1.txt", "r")

X = []
y = []

for line in file:
    data = line.split(',')
    data = [float(a) for a in data]
    a = data[0:-1]
    b = data[-1]
    X.append(a)
    y.append(b)

file.close()

X = np.array(X)
y = np.array(y) 
y = y.reshape((-1, 1)) 

def plot_data(X, y):
    pos = np.where(y == 1)
    neg = np.where(y == 0)
    plt.plot(X[pos[0], 0], X[pos[0], 1], '+', label='Passed')
    plt.plot(X[neg[0], 0], X[neg[0], 1], 'o', label='Failed')
    plt.legend()
    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')
    plt.show()
    return

plot_data(X, y)

def sigmoid(z):
    g = np.zeros(np.shape(z));
    g = 1/(1+np.exp(-z));
    return g

z = np.ones((X.shape[0], 1))
X = np.hstack((z, X))
theta = [0, 0, 0]

def cost_function(theta, X, y):
    J = 0;
    m = X.shape[0]
    receive_theta = np.array(theta)[np.newaxis]
    theta = np.transpose(receive_theta)
    z = np.dot(X,theta) 
    h = sigmoid(z) 
    J = np.sum(np.dot((-y.T),np.log(h))-np.dot((1-y).T,np.log(1-h)))/m
    grad = np.dot(X.T,(h-y))/m
    return J,grad


model = opt.fmin_tnc(func=cost_function, x0=theta, args=(X, y))
theta = model[0]

def plot_decision_boundary(theta, X, y):
    plot_x = [np.min(X[:,1]-2), np.max(X[:,2]+2)]
    plot_y = -1/theta[2]*(theta[0] 
              + np.dot(theta[1],plot_x))  
    mask = y.flatten() == 1
    adm = plt.scatter(X[mask][:,1], X[mask][:,2])
    not_adm = plt.scatter(X[~mask][:,1], X[~mask][:,2])
    decision_boun = plt.plot(plot_x, plot_y, 'g')
    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')
    plt.legend((adm, not_adm, decision_boun), ('Passed', 'Failed', 'Decision Boundary'))
    plt.show()

plot_decision_boundary(theta, X, y)























