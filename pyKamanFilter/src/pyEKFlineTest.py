'''
Created on Jul 23, 2017

@author: Francisco Dominguez
'''
from math import cos,sin,atan2
from math import pi as PI
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pyKalmanFilter as kf
import pyCovPlot as cp

def getGlobalLine(X,z):
    r=z[0,0]
    a=z[1,0]
    x=X[0,0]
    y=X[1,0]
    theta=X[2,0]
    aw=a+theta
    #Normalize angle
    awn=atan2(sin(aw),cos(aw))
    rw=r+(x*cos(awn)+y*sin(awn))
    zw=np.matrix([[rw],[awn]])
    return zw
def getCenterLine(z):
    r=z[0,0]
    alpha=z[1,0]

    xr=r*cos(alpha)
    yr=r*sin(alpha)
    return xr,yr
    
def plotLine(z,color="r"):
    r=z[0,0]
    alpha=z[1,0]
    a=cos(alpha)
    b=sin(alpha)
    c=r
    xr=r*cos(alpha)
    yr=r*sin(alpha)
    plt.plot([xr],[yr],color+'+')
    #if abs(alpha)<PI/2.0 :
    if b<1e-6:
        x0=np.sign(c)*10
        y0=10
        x1=np.sign(c)*10
        y1=-10
    else:
        x0=10
        y0=-a/b*x0+c/b
        x1=-10
        y1=-a/b*x1+c/b
#     else:
#         if a<1e-5:
#             x0=10
#             y0=10
#             x1=-10
#             y1=10
#         else:
#             y0=10
#             x0=-b/a*y0+c/a
#             y1=-10
#             x1=-b/a*y1+c/a
    plt.plot([x0,x1],[y0,y1],color,alpha=0.5)
    
if __name__ == '__main__':
    vpm=kf.VelocityProcessModel()
    vpm.dt=1
    X=np.matrix([0.0,0.0,0.0]).T
    u=np.matrix([1.0,0.11]).T
    lmm=kf.LineMeasurementModel()
    map=[]
    #map.append(np.array([ 10, 0.0]))
    map.append(np.array([ 10, PI/2.0]))
    #map.append(np.array([ 10, PI]))
    map.append(np.array([ 10,-PI/2.0]))
    map.append(np.array([ 10, PI/4.0]))
    map.append(np.array([ 10, PI*3.0/8.0]))
    #map.append(np.array([ -10, PI*1.0/8.0]))
    lmm.m=map
    cov=np.array([[  8.87790565,   4.65839419],
                  [  4.65839419,  12.01291835]]) #[[0.4, 9],[9, 10]])
    cov=np.array([[  0.01,  0],
                  [  0,  0.01]]) #[[0.4, 9],[9, 10]])
    for r,a in map:
        plotLine(np.matrix([[r],[a]]))
    STEPS=55
    for i in range(STEPS):
        print "At:"
        print X
        cp.plot_cov_ellipse(cov, X[:2,:].copy(), nstd=3, alpha=0.5, color='orange')
        X=vpm.eval(X,u)
    #ax = plt.gca()
    #ax.set_autoscale_on(True)
    kalman=kf.pyExtendedKalmanFilter(vpm,lmm)
    kalman.Xcov = np.matrix(np.array([[1,0,0],
                                       [0,1,0],
                                       [0,0,1]],np.float32) * 0.005)
    kalman.XNcov = np.matrix(np.array([[1,0,0],
                                       [0,1,0],
                                       [0,0,1]],np.float32) * 0.01525)
    #kalman.Zcov = np.matrix(np.array([[1,0,0],
    #                                   [0,1,0],
    #                                   [0,0,1]],np.float32) * 0.0525)
    kalman.ZNcov = np.matrix(np.array([[1,0],
                                       [0,1]],np.float32) * 0.01525)
    for j in range(STEPS):
        X,Xcov=kalman.predict(u)
        cp.plot_cov_ellipse(Xcov[:2,:2].copy(), X[:2,:].copy(), nstd=3, alpha=0.2, color='green')
        print "det=",np.linalg.det(Xcov)
        for j in range(1):
            print "det=",np.linalg.det(Xcov)
            #r=np.random.randint(0,100)
            #if r<80:
            #    continue
            if np.linalg.det(Xcov)<0.002:
                continue
            i=np.random.randint(0,len(map))
            #i=j
            lmm.setC(i)#observe landmark 0
            zd=lmm.eval(X)
            #z=zd
            z=np.matrix(np.array([zd[0,0]+np.random.randn(1)[0]*0.01525,zd[1,0]+np.random.randn(1)[0]*0.01525])).T
            print zd
            print z
            zw=getGlobalLine(X,z)
            plotLine(zw,"g")
            #plt.plot([x], [y], 'b*')
            x,y=getCenterLine(zw)
            plt.plot([x,X[0,0]],[y,X[1,0]],"g",alpha=0.2)
            X,Xcov=kalman.update(z)
            cp.plot_cov_ellipse(Xcov[:2,:2].copy(), X[:2,:].copy(), nstd=3, alpha=0.5, color='blue')
            kalman.X_=X
            kalman.Xcov_=Xcov
        #cp.plot_cov_ellipse(Xcov[:2,:2].copy(), X[:2,:].copy(), nstd=3, alpha=0.5, color='red')
        kalman.X=X
        kalman.Xcov=Xcov
    plt.show()