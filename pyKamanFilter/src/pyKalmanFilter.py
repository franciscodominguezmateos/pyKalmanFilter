'''
Created on Feb 17, 2017

@author: Francisco Dominguez
'''
import numpy as np
from numpy.linalg import inv,det
from numpy import log,exp,pi

class VelocityProcessModel(object):
    def __init__(self):
        pass
    def eval(self,s):
        pass
    def jacobian(self,x,u):
        pass
class LandmarksMeasurementModel(object):
    def __init__(self):
        self.m=None #map
        
        pass
    def eval(self,s):
        pass
    def jacobian(self,x,u):
        pass
    
class pyUnscentedKalmanFilter(object):
    def getSigmaPoints(self,mean,cov):
        sigmaPts=[]
        # first sigma point is the mean
        sigmaPts.append(mean)
        cols=cov.shape[1]
        n=cols+1
        for i in range(cols):
            sigma=mean+np.sqrt((n+self.l)*cov[:,i])
            sigmaPts.append(sigma)
            sigma=mean-np.sqrt((n+self.l)*cov[:,i])
            sigmaPts.append(sigma)
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
    def projectProcessModel(self,sigmaPts):
        sigmaPts_=[]
        for s in sigmaPts:
            s_=self.processModel.eval(s)
            sigmaPts_.append(s_)
        return sigmaPts_
    def projectMeasurementModel(self,sigmaPts):
        sigmaPts_=[]
        for s in sigmaPts:
            s_=self.measurementModel.eval(s)
            sigmaPts_.append(s_)
        return sigmaPts_
    def unscentedTransform(self,sigmaPts_,noiseCov):
        dim=sigmaPts_[0].shape[0]
        # mean estimation
        M_=np.matrix(np.zeros(dim)).T
        for w,s in zip(self.wm,sigmaPts_):
            M_+=w*s
        # covariance estimation
        cov_=noiseCov
        for w,s in zip(self.wc,sigmaPts_):
            cov_+=w*(s-M_)(s-M_).T
        return M_,cov_
    def getCrossCov(self):
        pass
        
    def __init__(self,Xdim,Zdim):
        self.processModel=None
        self.measurementModel=None
        pass
    def setParameters(self,a,b,k):
        n=self.Xcov.shape[1]+1
        # alpha
        self.a=a
        # beta
        self.b=b
        # kapa
        self.k=k
        # lambda
        self.l=a*a*(n+k)-n
    def predict(self,U=np.matrix('0,0,0,0')):
        U=np.matrix(U).T
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
        sigmaPts_=self.projectProcessModel(sigmaPts)
        # estimate new mean and covariance from sigma predicted and noise cov
        X_,Xcov_=self.unscentedTransform(sigmaPts_, XNcov)
        # set data
        self.X_=X_
        self.Xcov_=Xcov_
        self.XsigmaPts=sigmaPts
        self.XsigmaPts_=sigmaPts_
        return (X_,Xcov_)
    def update(self,Z):
        # get data
        Z=np.matrix(Z).T
        self.Z=Z
        X_   =self.X_
        Xcov_=self.Xcov_
        XsigmaPts_=self.XsigmaPts_
        ZNcov=self.ZNcov
        # Predict measurement Z_ from predicted state X_
        # project sigma points in time through measurement model
        ZsigmaPts_=self.projectMasurementModel(XsigmaPts_)
        # estimate new mean and covariance from sigma predicted and noise cov
        Z_, Zcov= self.unscentedTransform(ZsigmaPts_, ZNcov)
        iZcov=inv(Zcov)
        # Innovation = Actual measurement - Predicted measurement
        Innov= Z - Z_
        # estimate cross covariance from process X and measurement Z
        crossCov=self.getCrossCov()
        # Kalman gains
        K    = crossCov*iZcov
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
        
class pyExtendedKalmanFilter(object):
    '''
    classdocs
    '''
    def __init__(self, Xdim,Zdim):
        '''
        Constructor
        '''
        self.processModel=None
        self.measurementModel=None
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
    def predict(self,U=np.matrix('0,0,0,0')):
        U=np.matrix(U).T
        self.U=U
        # get data
        UTran=self.UTran
        X    =self.processModel.eval(self.X,U)
        XTran=self.processModel.jacobian(X,U)
        XNcov=self.XNcov
        Xcov =self.Xcov
        # Prediction
        X_   = XTran*X + UTran*U
        Xcov_= XTran*Xcov*XTran.T + XNcov
        # set data
        self.X_=X_
        self.Xcov_=Xcov_
        return (X_,Xcov_)
    def update(self,Z):
        # get data
        Z=np.matrix(Z).T
        self.Z=Z
        X_   =self.X_
        Xcov_=self.Xcov_
        ZTran=self.measurementModel.jacobian(X_)
        ZNcov=self.ZNcov
        # Predict measurement Z_ from predicted state X_
        Z_   = self.measurementModel.eval(X_)
        # Innovation = Actual measurement - Predicted measurement
        Innov= Z - Z_
        Zcov = ZTran*Xcov_*ZTran.T + ZNcov
        iZcov=inv(Zcov)
        # Kalman gains
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
        self.Zcov =ZZ   # Measurement covariance
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
    def update(self,Z):
        # get data
        Z=np.matrix(Z).T
        self.Z=Z
        X_   =self.X_
        Xcov_=self.Xcov_
        ZTran=self.ZTran
        ZNcov=self.ZNcov
        # Predict measurement Z_ from predicted state X_
        Z_   = ZTran*X_
        # Innovation = Actual measurement - Predicted measurement
        Innov= Z - Z_
        Zcov = ZTran*Xcov_*ZTran.T + ZNcov
        iZcov=inv(Zcov)
        # Kalman gains
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
        
import cv2

#The code below is an adaptation from a cv2.KalmanFilter example
meas=[]
pred=[]
frame = np.zeros((400,400,3), np.uint8) # drawing canvas
mp = np.array((2,1), np.float32) # measurement
tp = np.zeros((2,1), np.float32) # tracked / prediction

def onmouse(k,x,y,s,p):
    global mp,meas
    x+=np.random.randn(1)[0]*5
    y+=np.random.randn(1)[0]*5
    mp = np.array([[np.float32(x)],[np.float32(y)]])
    meas.append((int(x),int(y)))

def paint():
    global frame,meas,pred 
    for i in range(len(meas)-1): cv2.line(frame,meas[i],meas[i+1],(0,100,0))
    for i in range(len(pred)-1): cv2.line(frame,pred[i],pred[i+1],(0,0,200))

def reset():
    global meas,pred,frame
    meas=[]
    pred=[]
    frame = np.zeros((400,400,3), np.uint8)

if __name__ == '__main__':
    cv2.namedWindow("kalman")
    cv2.setMouseCallback("kalman",onmouse);
    kalman = pyKalmanFilter(4,2)
    kalman.ZTran = np.matrix(np.array([[1,0,0,0],
                                       [0,1,0,0]],np.float32))
    kalman.XTran = np.matrix(np.array([[1,0,1,0],
                                       [0,1,0,1],
                                       [0,0,1,0],
                                       [0,0,0,1]],np.float32))
    kalman.XNcov = np.matrix(np.array([[1,0,0,0],
                                       [0,1,0,0],
                                       [0,0,1,0],
                                       [0,0,0,1]],np.float32) * 0.01)
    kalman.ZNcov = np.matrix(np.array([[1,0],
                                       [0,1]],np.float32) * 5)
    while True:
        kalman.update(mp)
        tp,_ = kalman.predict()
        tp=np.ravel(np.array(tp))
        pred.append((int(tp[0]),int(tp[1])))
        paint()
        # Write some Text
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame,"%d,%d"%(tp[2],tp[3]),(10,10), font, 0.5,(255,255,255),1)
        cv2.imshow("kalman",frame)
        k = cv2.waitKey(30) &0xFF
        if k == 27: break
        if k == 32: reset()
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        