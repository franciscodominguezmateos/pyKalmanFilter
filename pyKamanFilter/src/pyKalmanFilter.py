'''
Created on Feb 17, 2017

@author: Francisco
'''
import numpy as np
from numpy.linalg import inv,det
from numpy import log,exp,pi

class pyKalmanFilter(object):
    '''
    classdocs
    '''
    def __init__(self, Xdim,Zdim):
        '''
        Constructor
        '''
        # State data
        Xini =np.matrix(np.zeros(Xdim)).T
        Uini =np.matrix(np.zeros(Xdim)).T#I doesn't have to be same size
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
    def predict(self,U):
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
        print("Z_=",Z_.shape)
        # Innovation = Actual measurement - Predicted measurement
        Innov= Z - Z_
        print("Innov=",Innov.shape)
        Zcov = ZTran*Xcov_*ZTran.T + ZNcov
        print("Zcov=",Zcov.shape)
        iZcov=inv(Zcov)
        # Kalman gains
        K    = Xcov_*ZTran.T*iZcov
        print("Zcov=",Zcov.shape)
        # Update State
        X    =X_+K*Innov
        print("X=",X.shape)
        Xcov =Xcov_-K*Zcov*K.T
        print("Xcov=",Xcov.shape)
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
                                       [0,0,0,1]],np.float32) * 1)
    kalman.ZNcov = np.matrix(np.array([[1,0],
                                       [0,1]],np.float32) * 5)
    while True:
        kalman.update(mp)
        tp,_ = kalman.predict(np.matrix('0,0,0,0'))
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
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        