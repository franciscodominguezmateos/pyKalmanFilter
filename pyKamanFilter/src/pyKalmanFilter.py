'''
Created on Feb 17, 2017

@author: Francisco Dominguez
'''
from math import cos,sin,sqrt,atan2
import numpy as np
from numpy.linalg import inv,det
from numpy import log,exp,pi

class VelocityProcessModel(object):
    def __init__(self):
        self.dt=0.1
    def getDeltaTime(self):
        return self.dt
    def getDim(self):
        return 3
    def eval(self,X,u):
        vt=u[0,0]
        wt=u[1,0]
        vw=vt/wt
        x=X[0,0]
        y=X[1,0]
        theta=X[2,0]
        dt=self.getDeltaTime()
        dX0=-vw*sin(theta)+vw*sin(theta+wt*dt)
        dX1= vw*cos(theta)-vw*cos(theta+wt*dt)
        dX2=wt*dt
        dX=np.matrix(np.array(
                     [[dX0],
                      [dX1],
                      [dX2]]))
        return X+dX
    def jacobian(self,X,u):
        vt=u[0,0]
        wt=u[1,0]
        vw=vt/wt
        theta=X[2,0]
        dt=self.getDeltaTime()
        g02=-vw*cos(theta)+vw*cos(theta+wt*dt)
        g12=-vw*sin(theta)+vw*sin(theta+wt*dt)
        G=np.matrix(np.array(
                     [[1, 0, g02],
                     [0, 1, g12],
                     [0, 0,   1]]))
        return G
class LandmarksMeasurementModel(object):
    def __init__(self):
        self.m=[] #map is a list
        self.C=0 # correspondence problem dependent variable
    def setC(self,C):
        self.C=C
    def getDim(self):
        return 3
    def eval(self,X):
        j=self.C
        mjx=self.m[j][0]#x landmark pos
        mjy=self.m[j][1]#y landmark pos
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
        pass
        Z_=np.matrix(np.array(
                     [[ d],
                      [th],
                      [ 0]]))
        return Z_
    
    def jacobian(self,X):
        j=self.C
        mjx=self.m[j][0]
        mjy=self.m[j][1]
        x=X[0,0]
        y=X[1,0]
        theta=X[2,0]
        dx=mjx-x
        dy=mjy-y
        dx2=dx*dx
        dy2=dy*dy
        d2=dx2+dy2
        d =sqrt(d2)#distance from measure to landmark
        H=np.matrix(np.array(
                    [[-dx/d ,-dy/d , 0],
                     [ dy/d2,-dx/d2,-1],
                     [     0,     0, 0]]))
        return H
    
class pyUnscentedKalmanFilter(object):
    def __init__(self, processModel, measurementModel):
        '''
        Constructor
        '''
        self.processModel=processModel
        self.measurementModel=measurementModel
        Xdim=processModel.getDim()
        Zdim=measurementModel.getDim()
        # State data
        Xini =np.matrix(np.zeros(Xdim)).T
        Uini =np.matrix(np.zeros(Xdim)).T#It doesn't have to be same size
        Zini =np.matrix(np.zeros(Zdim)).T
        XX=np.matrix(np.eye(Xdim,Xdim))
        ZX=np.matrix(np.eye(Zdim,Xdim))
        ZZ=np.matrix(np.eye(Zdim,Zdim))
        
        self.X    =Xini # State
        self.XNcov=XX   # State Noise covariance
        self.Xcov =XX   # State covariance
        self.X_   =Xini # State prediction
        self.Xcov_=XX   # State covariance prediction
        self.XTran=XX   # State transformation/transition Matrix/In EKF is Jacobian
        self.U    =Uini # Control
        self.UTran=XX   # Control transformation matrix/In EKF is Jacobian
        # Measurement data
        self.Z    =Zini # Measurement
        self.Z_   =Zini # Measurement prediction
        self.ZNcov=ZZ   # Measurement Noise covariance
        self.Zcov =ZZ   # Measurement covariance
        self.ZTran=ZX   # Measurement transformation matrix/In EKF is Jacobian
        # Temporal data
        self.Innov=Zini # Innovation
        self.setParameters()
        self.setWeights(Xdim)
    def setParameters(self,a=1,b=2,k=0.1):
        n=self.Xcov.shape[1]+1
        # alpha
        self.a=a
        # beta
        self.b=b
        # kapa
        self.k=k
        # lambda
        self.l=a*a*(n+k)-n
    def getSigmaPoints(self,mean,covIn):
        cols=covIn.shape[1]
        n=cols+1
        sigmaPts=[]
        # first sigma point is the mean
        tmp=(n+self.l)*covIn
        cov=np.linalg.cholesky(tmp)
        sigmaPts.append(mean)
        for i in range(cols):
            c=cov[:,i]
            print "c=",c
            sigma=mean+c
            sigmaPts.append(sigma)
            sigma=mean-c
            sigmaPts.append(sigma)
        for sp in sigmaPts:
            print sp
        return sigmaPts
    def setWeights(self,n):
        self.wm=[]
        self.wc=[]
        l=self.l
        a=self.a
        b=self.b
        # weight for mean
        wm0=l/(n+l)
        self.wm.append(wm0)
        # weight for covariance
        wc0=l/(n+l)+1-a*a+b
        self.wc.append(wc0)
        for i in range(2*n):
            w=1/(2*(n+l))
            self.wm.append(w)
            self.wc.append(w)
    def projectProcessModel(self,sigmaPts,U):
        sigmaPts_=[]
        for s in sigmaPts:
            s_=self.processModel.eval(s,U)
            sigmaPts_.append(s_)
        return sigmaPts_
    def projectMeasurementModel(self,sigmaPts):
        zPts_=[]
        for s in sigmaPts:
            s_=self.measurementModel.eval(s)
            zPts_.append(s_)
        return zPts_
    def getMoments(self,sigmaPts_,noiseCov):
        dim=sigmaPts_[0].shape[0]
        # mean estimation
        M_=np.matrix(np.zeros(dim)).T
        for w,s in zip(self.wm,sigmaPts_):
            print s
            M_+=w*s
        # covariance estimation
        cov_=noiseCov
        for w,s in zip(self.wc,sigmaPts_):
            cov_+=w*(s-M_)*(s-M_).T
        print M_,cov_
        return M_,cov_
    def getCrossCov(self,XsigmaPts,X_,ZsigmaPts,Z_):
        dimX=X_.shape[0]
        dimZ=Z_.shape[0]
        cov=np.matrix(np.zeros((dimX,dimZ)))
        print cov.shape
        for w,x,z in zip(self.wc,XsigmaPts,ZsigmaPts):
            print x.shape
            print z.T.shape
            dX=x-X_
            dZ=z-Z_
            print dX.shape
            print dZ.T.shape
            cov+=w*dX*dZ.T
        return cov
    def predict(self,U=np.matrix('0,0,0,0').T):
        #U=np.matrix(U).T
        self.U=U
        # get data
        UTran=self.UTran
        X    =self.X
        Xcov =self.Xcov
        XNcov=self.XNcov
        # Prediction
        # sample sigmaPoints from gauss pdf with x mean and Xcov covarianace
        sigmaPts=self.getSigmaPoints(X,Xcov)
        # project points in time with process model
        # new predicted sigma points
        sigmaPts_=self.projectProcessModel(sigmaPts,U)
        # estimate new mean and covariance from sigma predicted and noise cov
        X_,Xcov_=self.getMoments(sigmaPts_, XNcov)
        # set data
        self.X_=X_
        self.Xcov_=Xcov_
        self.XsigmaPts=sigmaPts
        self.XsigmaPts_=sigmaPts_
        return (X_,Xcov_)
    def update(self,Z):
        # get data
        #Z=np.matrix(Z).T
        self.Z=Z
        X_   =self.X_
        Xcov_=self.Xcov_
        XsigmaPts_=self.getSigmaPoints(X_,Xcov_)
        ZNcov=self.ZNcov
        # Predict measurement Z_ from predicted state X_
        # project sigma points in time through measurement model
        ZsigmaPts_=self.projectMeasurementModel(XsigmaPts_)
        # estimate new mean and covariance from sigma predicted and noise cov
        Z_, Zcov= self.getMoments(ZsigmaPts_, ZNcov)
        iZcov=inv(Zcov)
        # Innovation = Actual measurement - Predicted measurement
        Innov= Z - Z_
        # estimate cross covariance from process X and measurement Z
        crossCov=self.getCrossCov(XsigmaPts_,X_,ZsigmaPts_,Z_)
        # Kalman gain
        print iZcov
        print crossCov
        K    = crossCov*iZcov
        # Update State
        print "K=",K
        print "X=",X_
        X    =X_+K*Innov
        Xcov =Xcov_-K*Zcov*K.T
        print "X=",X
        print "Xcov=",Xcov
        # set data
        self.Z_=Z_
        self.Zcov=Zcov
        self.X   =X
        self.Xcov=Xcov
        return (X,Xcov)
    def gauss(self,X,Xmean,Xcov):
        Xdim = Xmean.shape[0]
        Xdif = X-Xmean
        XdifT=Xdif.transpose()
        iXcov=inv(Xcov)
        E    = 0.5 * XdifT * iXcov * Xdif
        logP = E + 0.5 * Xdim * log(2*pi) + 0.5*log(det(Xcov))
        P=exp(-logP)
        return (P[0],logP[0])
    def MeasurementLikelihood(self,Z):   
        P,logP=self.gauss(Z,self.Z_,self.Zcov)
        
class pyExtendedKalmanFilter(object):
    '''
    classdocs
    '''
    def __init__(self, processModel, measurementModel):
        '''
        Constructor
        '''
        self.processModel=processModel
        self.measurementModel=measurementModel
        Xdim=processModel.getDim()
        Zdim=measurementModel.getDim()
        self.Xdim=Xdim
        self.Zdim=Zdim
        # State data
        Xini =np.matrix(np.zeros(Xdim)).T
        Uini =np.matrix(np.zeros(Xdim)).T#It doesn't have to be same size
        Zini =np.matrix(np.zeros(Zdim)).T
        XX=np.matrix(np.eye(Xdim,Xdim))
        ZX=np.matrix(np.eye(Zdim,Xdim))
        ZZ=np.matrix(np.eye(Zdim,Zdim))
        
        self.X    =Xini # State
        self.XNcov=XX   # State Noise covariance
        self.Xcov =XX   # State covariance
        self.X_   =Xini # State prediction
        self.Xcov_=XX   # State covariance prediction
        self.XTran=XX   # State transformation/transition Matrix/In EKF is Jacobian
        self.U    =Uini # Control
        self.UTran=XX   # Control transformation matrix/In EKF is Jacobian
        # Measurement data
        self.Z    =Zini # Measurement
        self.Z_   =Zini # Measurement prediction
        self.ZNcov=ZZ   # Measurement Noise covariance
        self.Zcov =ZZ   # Measurement covariance
        self.ZTran=ZX   # Measurement transformation matrix/In EKF is Jacobian
        # Temporal data
        self.Innov=Zini # Innovation
        
    def predict(self,U=np.matrix('0,0,0,0').T):
        #U=np.matrix(U).T
        self.U=U
        # get data
        UTran=self.UTran
        X=self.X
        XTran=self.processModel.jacobian(X,U)
        XNcov=self.XNcov
        Xcov =self.Xcov
        # Prediction
        X_   = self.processModel.eval(X,U)
        Xcov_= XTran*Xcov*XTran.T + XNcov # not noise model for U for now
        # set data
        self.X_=X_
        self.Xcov_=Xcov_
        return (X_,Xcov_)

    def measurePrediction(self):
        # get data
        X_   =self.X_
        Xcov_=self.Xcov_
        ZTran=self.measurementModel.jacobian(X_)
        ZNcov=self.ZNcov
        # Predict measurement Z_ from predicted state X_
        Z_   = self.measurementModel.eval(X_)
        Zcov = ZTran*Xcov_*ZTran.T 
        Zcov+= ZNcov
        self.Z_=Z_
        self.Zcov=Zcov
        return (Z_,Zcov ,ZTran)
        
    def update(self,Z):
        # get data
        #Z=np.matrix(Z).T
        self.Z=Z
        X_   =self.X_
        Xcov_=self.Xcov_
        ZNcov=self.ZNcov
        # Predict measurement Z from predicted state X_
        Z_,Zcov,ZTran=self.measurePrediction()
        iZcov=inv(Zcov)
        # Innovation = Actual measurement - Predicted measurement
        Innov= Z - Z_
        # Kalman gain
        K    = Xcov_*ZTran.T*iZcov
        # Update State
        correction=K*Innov
        X    =X_+correction
        Xcov =Xcov_-K*Zcov*K.T
        # set data
        self.Z_=Z_
        self.Zcov=Zcov
        self.X   =X
        self.Xcov=Xcov
        return (X,Xcov)
    
    def gauss(self,X,Xmean,Xcov):
        Xdim = Xmean.shape[0]
        Xdif = X-Xmean
        XdifT=Xdif.transpose()
        iXcov=inv(Xcov)
        E    = 0.5 * XdifT * iXcov * Xdif
        logP = E + 0.5 * Xdim * log(2*pi) + 0.5*log(det(Xcov))
        P=exp(-logP)
        return (P[0],logP[0])
    
    def MeasurementLikelihood(self,Z):   
        P,logP=self.gauss(Z,self.Z_,self.Zcov)

class pyKalmanFilter(object):
    '''
    classdocs
    '''
    def __init__(self, Xdim,Zdim):
        '''
        Constructor
        '''
        self.Xdim=Xdim
        self.Zdim=Zdim
        # State data
        Xini =np.matrix(np.zeros(Xdim)).T
        Uini =np.matrix(np.zeros(Xdim)).T#It doesn't have to be same size
        Zini =np.matrix(np.zeros(Zdim)).T
        XX=np.matrix(np.eye(Xdim,Xdim))
        ZX=np.matrix(np.eye(Zdim,Xdim))
        ZZ=np.matrix(np.eye(Zdim,Zdim))
        
        self.X    =Xini # State
        self.XNcov=XX   # State Noise covariance
        self.Xcov =XX   # State covariance
        self.X_   =Xini # State prediction
        self.Xcov_=XX   # State covariance prediction
        self.XTran=XX   # State transformation/transition Matrix
        self.U    =Uini # Control
        self.UTran=XX   # Control transformation matrix
        # Measurement data
        self.Z    =Zini # Measurement
        self.Z_   =Zini # Measurement prediction
        self.ZNcov=ZZ   # Measurement Noise covariance
        self.Zcov =ZZ   # Measurement covariance prediction
        self.ZTran=ZX   # Measurement transformation matrix
        # Temporal data
        self.Innov=Zini # Innovation
    def predict(self,U=np.matrix('0,0,0,0')):
        U=np.matrix(U).T
        self.U=U
        # get data
        UTran=self.UTran
        X    =self.X
        XTran=self.XTran
        XNcov=self.XNcov
        Xcov =self.Xcov
        # Prediction
        X_   = XTran*X + UTran*U
        Xcov_= XTran*Xcov*XTran.T + XNcov
        # set data
        self.X_=X_
        self.Xcov_=Xcov_
        return (X_,Xcov_)
    def measurePrediction(self):
        # get data
        X_   =self.X_
        Xcov_=self.Xcov_
        ZTran=self.ZTran
        ZNcov=self.ZNcov
        # Predict measurement Z_ from predicted state X_
        Z_   = ZTran*X_
        Zcov = ZTran*Xcov_*ZTran.T + ZNcov
        self.Z_=Z_
        self.Zcov=Zcov
        return (Z_,Zcov)
        
    def update(self,Z):
        # get data
        Z=np.matrix(Z).T
        self.Z=Z
        X_   =self.X_
        Xcov_=self.Xcov_
        ZTran=self.ZTran
        ZNcov=self.ZNcov
        # Predict measurement Z from predicted state X_
        Z_,Zcov=self.measurePrediction()
        iZcov=inv(Zcov)
        # Innovation = Actual measurement - Predicted measurement
        Innov= Z - Z_
        # Kalman gain
        K    = Xcov_*ZTran.T*iZcov
        # Update State
        X    =X_+K*Innov
        Xcov =Xcov_-K*Zcov*K.T
        # set data
        self.Z_=Z_
        self.Zcov=Zcov
        self.X   =X
        self.Xcov=Xcov
        return (X,Xcov)
    
    def gauss(self,X,Xmean,Xcov):
        Xdim = Xmean.shape[0]
        Xdif = X-Xmean
        XdifT=Xdif.transpose()
        iXcov=inv(Xcov)
        E    = 0.5 * XdifT * iXcov * Xdif
        logP = E + 0.5 * Xdim * log(2*pi) + 0.5*log(det(Xcov))
        P=exp(-logP)
        return (P[0],logP[0])
    def MeasurementLikelihood(self,Z):   
        P,logP=self.gauss(Z,self.Z_,self.Zcov)
        

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        