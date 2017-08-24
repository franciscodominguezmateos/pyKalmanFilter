'''
Created on Jul 30, 2017

@author: Francisco Dominguez
'''
from math import sqrt,sin,cos,atan2
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pyKalmanFilter as kf
import pyCovPlot as cp

class VelocityProcessSLAMModel(object):
    def __init__(self):
        self.dt=0.1
    def getDeltaTime(self):
        return self.dt
    def getDim(self):
        return self.dim
    def eval(self,X,u):
        #State dimension can change
        self.dim=X.shape[0]
        vt=u[0,0]
        wt=u[1,0]
        vw=vt/wt
        x=X[0,0]
        y=X[1,0]
        theta=X[2,0]
        dt=self.getDeltaTime()
        a=theta+wt*dt
        #Normalize angle
        alpha=atan2(sin(a),cos(a))
        dX0=-vw*sin(theta)+vw*sin(alpha)
        dX1= vw*cos(theta)-vw*cos(alpha)
        dX2= wt*dt
        #dX=np.matrix([[dX0],
        #              [dX1],
        #              [dX2]])
        X[0,0]+=dX0
        X[1,0]+=dX1
        #a=+wt*dt
        #Normalize angle
        #alpha=atan2(sin(a),cos(a))       
        X[2,0]=alpha
        return X
    def jacobian(self,X,u):
        #State dimension can change
        self.dim=X.shape[0]
        vt=u[0,0]
        wt=u[1,0]
        vw=vt/wt
        theta=X[2,0]
        dt=self.getDeltaTime()
        a=theta+wt*dt
        #Normalize angle
        alpha=atan2(sin(a),cos(a))
        g02=-vw*cos(theta)+vw*cos(alpha)
        g12=-vw*sin(theta)+vw*sin(alpha)
        G=np.matrix(np.array(
                     [[1, 0, g02],
                     [ 0, 1, g12],
                     [ 0, 0,   1]]))
        Gr=np.matrix(np.eye(self.dim))
        Gr[:3,:3]=G
        return Gr
class LandmarkMeasurementSLAMModel(object):
    def __init__(self):
        self.C=0 # correspondence problem dependent variable
    def setC(self,C):
        self.C=C
    def getDim(self):
        return self.dim
    def eval(self,X):
        #State dimension can change
        self.dim=X.shape[0]
        j=self.C*2+3
        #Map is in the state X
        mjx=X[j  ,0]#x landmark pos
        mjy=X[j+1,0]#y landmark pos
        x=X[0,0]
        y=X[1,0]
        theta=X[2,0]
        dx=mjx-x
        dy=mjy-y
        dx2=dx*dx
        dy2=dy*dy
        d2=dx2+dy2
        d =sqrt(d2) #distance from object/robot to landmark
        th=atan2(dy,dx)-theta #angle from object/robot to landmark
        #Normalize angle
        thn=atan2(sin(th),cos(th))
        Z_=np.matrix(np.array(
                     [[ d],
                      [thn]]))
        return Z_
    
    def jacobian(self,X):
        #State dimension can change
        self.dim=X.shape[0]
        j=self.C*2+3
        #Map is in the state X
        mjx=X[j  ,0]#x landmark pos
        mjy=X[j+1,0]#y landmark pos
        x=X[0,0]
        y=X[1,0]
        theta=X[2,0]
        dx=mjx-x
        dy=mjy-y
        dx2=dx*dx
        dy2=dy*dy
        d2=dx2+dy2
        d =sqrt(d2)#distance from measure to landmark
        #Jacobian with respect to X
        Hx=np.matrix(np.array(
                    [[-dx/d ,-dy/d , 0],
                     [ dy/d2,-dx/d2,-1]]))
        #Jacobian with respect to m=landmark j
        Hm=np.matrix(np.array(
                    [[ dx/d , dy/d ],
                     [-dy/d2, dx/d2]]))
        H=np.matrix(np.zeros((2,self.dim)))
        H[:, :3  ]=Hx
        H[:,j:j+2]=Hm
        return H
    
class LineMeasurementSLAMModel(object):
    def __init__(self):
        self.C=0 # correspondence problem dependent variable
    def setC(self,C):
        self.C=C
    def getDim(self):
        return self.dim
    def eval(self,X):
        #State dimension can change
        self.dim=X.shape[0]
        j=self.C*2+3
        #Map is in the state X
        mjr=X[j  ,0]#x landmark pos
        mja=X[j+1,0]#y landmark pos
        x=X[0,0]
        y=X[1,0]
        theta=X[2,0]
        d =mjr-(x*cos(mja)+y*sin(mja)) #distance from object/robot to line landmark
        th=mja-theta #angle from object/robot to line landmark 
         #Normalize angle
        thn=atan2(sin(th),cos(th))
        Z_=np.matrix(np.array(
                     [[ d],
                      [thn]]))
        return Z_
     
    def jacobian(self,X):
        #State dimension can change
        self.dim=X.shape[0]
        j=self.C*2+3
        #Map is in the state X
        mjr=X[j  ,0]#x landmark pos
        mja=X[j+1,0]#y landmark pos
        x=X[0,0]
        y=X[1,0]
        theta=X[2,0]
        #Jacobian with respect to X
        Hx=np.matrix(np.array(
                    [[ -cos(mja),-sin(mja),  0.0],
                     [      0.0 ,     0.0 , -1.0]]))
        #Jacobian with respect to m=landmark j 
        Hm=np.matrix(np.array(
                    [[  1.0 , x*sin(mja)-y*cos(mja) ],
                     [  0.0 ,                  1.0  ]]))
        H=np.matrix(np.zeros((2,self.dim)))
        H[:, :3  ]=Hx
        H[:,j:j+2]=Hm
        return H
    
def plotMap(X):
    x=[X[i*2+3+0,0] for i in range((X.shape[0]-3)/2)]
    y=[X[i*2+3+1,0] for i in range((X.shape[0]-3)/2)]
    plt.plot(x,y,'go')

def plotCovMap(X,Xcov):
    N=(X.shape[0]-3)/2
    for i in range(N):
        j=i*2+3
        cp.plot_cov_ellipse(Xcov[j:j+2,j:j+2].copy(), X[j:j+2,:].copy(), nstd=.3, alpha=0.5, color='orange')
    
def getLandmark(X,z):
    alpha=z[1,0]
    theta=X[2,0]
    o=alpha+theta
    on=atan2(sin(o),cos(o))
    x=X[0,0]+z[0,0]*cos(on)
    y=X[1,0]+z[0,0]*sin(on)
    return x,y

def getMeasurement(X,m):
    mjx=m[0,0]#x landmark pos
    mjy=m[1,0]#y landmark pos
    x=X[0,0]
    y=X[1,0]
    theta=X[2,0]
    dx=mjx-x
    dy=mjy-y
    dx2=dx*dx
    dy2=dy*dy
    d2=dx2+dy2
    d =sqrt(d2) #distance from object/robot to landmark
    th=atan2(dy,dx)-theta #angle from object/robot to landmark
    #Normalize angle
    thn=atan2(sin(th),cos(th))
    Z_=np.matrix(np.array(
                 [[ d],
                  [th]]))
    return Z_ 
    
if __name__ == '__main__':
    #Build a map
    map=[]
    map.append(np.array([10,10]))
    map.append(np.array([ 0,10]))
    map.append(np.array([10, 0.0]))
    map.append(np.array([-15, 0.0]))
    map.append(np.array([ 0,-5]))
    map.append(np.array([ 15,15]))
    amap=np.matrix(np.array(map).ravel()).T
    #Draw the map
    lx=[x for x,y in map]
    ly=[y for x,y in map]
    plt.plot(lx, ly, 'ro')
    #State dimension
    INF=10e12
    DIM=3+amap.shape[0]
    print DIM
    #Process model
    vpm=VelocityProcessSLAMModel()
    vpm.dim=DIM
    vpm.dt=1
    X=np.matrix(np.vstack((np.matrix([0,0,0]).T,amap)),np.float)
    u=np.matrix([1.0,0.12]).T
    #Masurement model
    lmm=LandmarkMeasurementSLAMModel()
    lmm.dim=DIM
    #Identity and zeros matrices
    eye  =np.matrix(np.eye(DIM))
    zeros=np.matrix(np.zeros((DIM,DIM)))
    cov=eye
    STEPS=60
    for i in range(30):
        #print "At:"
        #print X[:2,:]
        cp.plot_cov_ellipse(cov[:2,:2].copy(), X[:2,:].copy(), nstd=.3, alpha=0.5, color='orange')
        X=vpm.eval(X,u)
    #ax = plt.gca()
    #ax.set_autoscale_on(True)
    kalman=kf.pyExtendedKalmanFilter(vpm,lmm)
    X[:3,:]=np.zeros((3,1))
    #X=np.matrix(np.zeros((DIM,1)))
    Xini=X.copy()
    kalman.X=X
    kalman.Xcov  = eye * 10# INF
    Xc=0.00015
    kalman.Xcov[0,0]=Xc
    kalman.Xcov[1,1]=Xc
    kalman.Xcov[2,2]=Xc
    kalman.XNcov = zeros
    XN=0.0002525
    kalman.XNcov[0,0]=XN
    kalman.XNcov[1,1]=XN
    kalman.XNcov[2,2]=XN
    #kalman.Zcov = np.matrix(np.array([[1,0,0],
    #                                   [0,1,0],
    #                                   [0,0,1]],np.float32) * 0.0525)
    ZN=0.0225
    kalman.ZNcov = np.matrix(np.array([[1,0],
                                       [0,1]],np.float32) * ZN)
    for j in range(STEPS):
        X,Xcov=kalman.predict(u)
        cp.plot_cov_ellipse(Xcov[:2,:2].copy(), X[:2,:].copy(), nstd=3, alpha=0.2, color='green')
        for k in range(1):#samples for step
            print np.linalg.det(Xcov[:3,:3])
            #if np.linalg.det(Xcov[:3,:3])<0.0000001:
            #    continue
            i=np.random.randint(0,len(map))
            #simulate measurement
            m=np.matrix(map[i]).T
            zd=getMeasurement(X,m) #get landmark in order to simulate measure
            #z=zd
            z=np.matrix(np.array([zd[0,0]+np.random.randn(1)[0]*ZN/2,zd[1,0]+np.random.randn(1)[0]*ZN/2])).T
            x,y=getLandmark(X,z)
            plt.plot([x], [y], 'b*')
            plt.plot([x,X[0,0]],[y,X[1,0]],"g",alpha=0.2)
            lmm.setC(i)#observe landmark i
            X,Xcov=kalman.update(z)
            #print X[3:,0],(Xini-X)[3:,0]
            cp.plot_cov_ellipse(Xcov[:2,:2].copy(), X[:2,:].copy(), nstd=3, alpha=0.1, color='blue')
            kalman.X_=X
            kalman.Xcov_=Xcov
        cp.plot_cov_ellipse(Xcov[:2,:2].copy(), X[:2,:].copy(), nstd=3, alpha=0.5, color='red')
        kalman.X=X
        kalman.Xcov=Xcov
        plotMap(X)
    plotCovMap(X,Xcov)
    plt.show()