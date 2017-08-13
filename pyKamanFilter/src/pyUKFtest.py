'''
Created on Jul 23, 2017

@author: Francisco Dominguez
'''
from math import cos,sin,atan2
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pyKalmanFilter as kf
import pyCovPlot as cp

def getLandmark(X,z):
    alpha=z[1,0]
    theta=X[2,0]
    o=alpha+theta
    on=atan2(sin(o),cos(o))
    x=X[0,0]+z[0,0]*cos(on)
    y=X[1,0]+z[0,0]*sin(on)
    return x,y

if __name__ == '__main__':
    vpm=kf.VelocityProcessModel()
    vpm.dt=1
    X=np.matrix([0.0,0.0,0.0]).T
    u=np.matrix([1.0,0.11]).T
    lmm=kf.LandmarkMeasurementModel()
    map=[]
    #map.append(np.array([10,10]))
    map.append(np.array([ 0,10]))
    map.append(np.array([10, 0.0]))
    map.append(np.array([-15, 0.0]))
    map.append(np.array([ 0,-1]))
    map.append(np.array([ 15,15]))
    lmm.m=map
    cov=np.array([[  8.87790565,   4.65839419],
                  [  4.65839419,  12.01291835]]) #[[0.4, 9],[9, 10]])
    cov=np.array([[  0.01,  0],
                  [  0,  0.01]]) #[[0.4, 9],[9, 10]])
    lx=[x for x,y in map]
    ly=[y for x,y in map]
    print x,y
    plt.plot(lx, ly, 'ro')
    STEPS=10
    for i in range(STEPS):
        print "At:"
        print X
        cp.plot_cov_ellipse(cov, X[:2,:].copy(), nstd=3, alpha=0.5, color='orange')
        X=vpm.eval(X,u)
    #ax = plt.gca()
    #ax.set_autoscale_on(True)
    kalman=kf.pyUnscentedKalmanFilter(vpm,lmm)
    kalman.Xcov = np.matrix(np.array([[1,0,0],
                                       [0,1,0],
                                       [0,0,1]],np.float32) * 0.005)
    kalman.XNcov = np.matrix(np.array([[1,0,0],
                                       [0,1,0],
                                       [0,0,1]],np.float32) * 0.001525**2)
    #kalman.Zcov = np.matrix(np.array([[1,0,0],
    #                                   [0,1,0],
    #                                   [0,0,1]],np.float32) * 0.0525)
    kalman.ZNcov = np.matrix(np.array([[1,0,0],
                                       [0,1,0],
                                       [0,0,1]],np.float32) * 0.001525**2)
    for k in range(STEPS):
        print "k=",k
        X,Xcov=kalman.predict(u)
        sigmaPts=kalman.getSigmaPoints(X,Xcov)
        print sigmaPts[0]
        xl=[s[0,0] for s in sigmaPts]
        yl=[s[1,0] for s in sigmaPts]
        plt.plot(xl,yl,'g+')
        cp.plot_cov_ellipse(Xcov[:2,:2].copy(), X[:2,:].copy(), nstd=3, alpha=0.2, color='green')
        print "det=",np.linalg.det(Xcov)
        for j in range(1):
            print "j=",j
            print "det=",np.linalg.det(Xcov)
            #r=np.random.randint(0,100)
            #if r<80:
            #    continue
            #if np.linalg.det(Xcov)<0.002:
            #    continue
            # Choose a random landmark
            i=np.random.randint(0,len(map))
            lmm.setC(i)#observe landmark 0
            # Measure the landmark
            zd=lmm.eval(X)
            #z=zd
            # add noise to the measurent
            z=np.matrix(np.array([zd[0,0]+np.random.randn(1)[0]*0.001525,zd[1,0]+np.random.randn(1)[0]*0.001525,0])).T
            # Plot the measurement
            x,y=getLandmark(X,z)
            x,y=getLandmark(X,z)
            plt.plot([x], [y], 'b*')
            plt.plot([x,X[0,0]],[y,X[1,0]],"g",alpha=0.2)
            # Actual Kalman UPDATE
            X,Xcov=kalman.update(z)
            # Plot state
            cp.plot_cov_ellipse(Xcov[:2,:2].copy(), X[:2,:].copy(), nstd=3, alpha=0.5, color='blue')
            # Feedback state to process other measurement
            kalman.X_=X
            kalman.Xcov_=Xcov
        #cp.plot_cov_ellipse(Xcov[:2,:2].copy(), X[:2,:].copy(), nstd=3, alpha=0.5, color='red')
        kalman.X=X
        kalman.Xcov=Xcov
    plt.show()