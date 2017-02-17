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
    def __init__(self, params):
        '''
        Constructor
        '''
        # State data
        self.X    =None # State
        self.XNcov=None # State Noise covariance
        self.Xcov =None # State covariance
        self.X_   =None # State prediction
        self.Xcov_=None # State covariance prediction
        self.XTran=None # State transformation/transition Matrix
        self.U    =None # Control
        self.UTran=None # Control transformation matrix
        # Measurement data
        self.Z    =None # Measurement
        self.Z_   =None # Measurement prediction
        self.ZNcov=None # Measurement Noise covariance
        self.Zcov =None # Measurement covariance
        self.ZTran=None # Measurement transformation matrix
        # Temporal data
        self.Innov=None # Innovation
    def predict(self,U):
        self.U=U
        # get data
        UTran=self.UTran
        X    =self.X
        XTran=self.XTran
        XTranT=XTran.transpose()
        XNcov=self.XNcov
        Xcov =self.Xcov
        # Prediction
        X_   = XTran*X + UTran*U
        Xcov_= XTran*Xcov*XTranT + XNcov
        # set data
        self.X_=X_
        self.Xcov_=Xcov_
        return (X_,Xcov_)
    def update(self,Z):
        # get data
        self.Z=Z
        X_   =self.X_
        Xcov_=self.Xcov_
        ZTran=self.ZTran
        ZTranT=ZTran.transpose()
        ZNcov=self.ZNcov
        # Predict measurement Z_ from predicted state X_
        Z_   = ZTran*X_
        # Innovation = Actual measurement - Predicted measurement
        Innov= Z - Z_
        Zcov = ZTran*Xcov_*ZTranT + ZNcov
        iZcov=inv(Zcov)
        # Kalman gains
        K    = Xcov_*ZTranT*iZcov
        Kt   = K.transpose()
        # Update State
        X    =X_+K*Innov
        Xcov =Xcov_-K*Zcov*Kt
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
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        