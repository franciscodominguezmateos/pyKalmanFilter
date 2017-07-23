'''
Created on Jul 23, 2017

@author: Francisco Domingugez
'''
import cv2
import numpy as np
from pyKalmanFilter import pyKalmanFilter

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
                                       [0,0,0,1]],np.float32) * 1)
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