import cv2
import numpy as np

class KalmanFilter_XYHW(object):
    def __init__(self):
        self.kf_xy = KalmanFilter()
        self.kf_hw = KalmanFilter()
        
    def detect(self, c_X, c_Y, c_H, c_W):
        x, y = self.kf_xy.detect(c_X, c_Y)
        h, w = self.kf_xy.detect(c_H, c_W)
        return x,y,h,w
    
    
class KalmanFilter(object):
    def __init__(self):
        self.kf = cv2.KalmanFilter(4,2)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                            [0, 1, 0, 1],
                                            [0, 0, 1, 0],
                                            [0, 0, 0, 1]], dtype=np.float32)

        self.kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                              [0, 1, 0, 0]], dtype=np.float32)
        
    def predict(self, coord1, coord2):
        '''
            Point estimation of the next point using Kalman predict and correct
        '''
        
        measured = np.array([[np.float32(coord1)], [np.float32(coord2)]])
        self.kf.correct(measured) # first, we correct the KF with the new measurement of the bbox coordinate
        c_1, c_2 = self.kf.predict() # then, we predict the bbox coordinate for the next frame
        return c_1, c_2
        