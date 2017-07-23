'''
Created on Jul 23, 2017

@author: Francisco Dominguez
'''
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pyKalmanFilter as kf
import pyCovPlot as cp

if __name__ == '__main__':
    vpm=kf.VelocityProcessModel()
    X=np.matrix([0,0,0]).T
    u=np.matrix([1.5,2.0]).T
    lmm=kf.LandmarksMeasurementModel()
    map=[]
    map.append(np.array([10,10]))
    map.append(np.array([ 0,10]))
    map.append(np.array([10, 0]))
    lmm.m=map
    cov=np.array([[  8.87790565,   4.65839419],
                  [  4.65839419,  12.01291835]]) #[[0.4, 9],[9, 10]])
    print "At:"
    print X
    ellip=cp.plot_cov_ellipse(cov, X[:2,:], nstd=0.02, alpha=0.5, color='orange')
    for C in range(len(map)):
        lmm.setC(C)
        print lmm.eval(X)
    X=vpm.eval(X,u)
    print "At:"
    print X
    ellip=cp.plot_cov_ellipse(cov, X[:2,:], nstd=0.02, alpha=0.5, color='orange')
    for C in range(len(map)):
        lmm.setC(C)
        print lmm.eval(X)
    X=vpm.eval(X,u)
    print "At:"
    print X
    ellip=cp.plot_cov_ellipse(cov, X[:2,:], nstd=0.02, alpha=0.5, color='orange')
    for C in range(len(map)):
        lmm.setC(C)
        print lmm.eval(X)
    X=vpm.eval(X,u)
    print "At:"
    print X
    ellip=cp.plot_cov_ellipse(cov, X[:2,:], nstd=0.02, alpha=0.5, color='orange')
    for C in range(len(map)):
        lmm.setC(C)
        print lmm.eval(X)
    X=vpm.eval(X,u)
    ellip=cp.plot_cov_ellipse(cov, X[:2,:], nstd=0.02, alpha=0.5, color='orange')

    plt.show()