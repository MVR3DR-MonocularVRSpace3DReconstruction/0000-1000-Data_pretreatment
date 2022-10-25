import cv2
import numpy as np
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
#import argparse
#from matplotlib import pyplot as plt
#import imutils
def nothing(x):
    pass

ptime = 0
ntime = 0
fps = 0

template = cv2.imread('imagecropped.jpg',0) #from documents folder and imagecropper code


camera = PiCamera()
resx = 640
resy = 480
# 320 240
# 640 480
# 1280 960

xbbox = 75
ybbox = 75

resxa = ((resx // 2) - (xbbox // 2))
resxb = ((resx // 2) + (xbbox // 2))
resya = ((resy // 2) - (ybbox // 2))
resyb = ((resy // 2) + (ybbox // 2))

a = resxa
B = resxb
c = resya
d = resyb

camera.resolution = (resx, resy)
camera.framerate = 60

rawCapture = PiRGBArray(camera, size=(resx, resy))

for frame in camera.capture_continuous(rawCapture, format = "yuv", use_video_port=True):

    #test for speed between hsv, bgr, bayer, and yuv figure out buffer probelm
    #imagea = frame.array[:,:,2][a:B,c:d] #only red color, and cropped for less calulations
    imagea = frame.array[:,:,0][a:B,c:d] #only Y grayscale color, and cropped for less calulations #[:,:,0]
    cv2.imshow("LOCKED",imagea)
    ret, u = cv2.threshold(imagea,225,255,cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(u,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    ap = a
    Bp = B
    cp = c
    dp = d
    if len(contours) != 0: #track phase
        c = max(contours, key = cv2.contourArea)
        M = cv2.moments(c)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            a = (ap + cY - ybbox)
            B = (ap + cY + ybbox)
            c = (cp + cX - xbbox)
            d = (cp + cX + xbbox) #add time based a1B1c1d1 and off of cv2 bounding rect to do cropping
            if (a > resy): a = resy
            if (a < 0): a = 0
            if (B > resy): B = resy
            if (B < 0): B = 0
            if (c > resx): c = resx
            if (c < 0): c = 0
            if (d > resx): d = resx
            if (d < 0): d = 0
        else:
        #aquisition phase
            a = 0
            B = resx
            c = 0
            d = resy
    key = cv2.waitKey(1)
    #rawCapture.truncate(0)
    if key == 27:
            break
cv2.destroyAllWindows()
