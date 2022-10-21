import os
import time
import subprocess
import numpy as np
from datetime import datetime

import RPi.GPIO as GPIO
import picamera

# from ffmpeg_proc import exec_progress
from utils import full_light, full_dark, light_one, dark_one

####################################################
# Settings
####################################################

raw_dir = "raw/"
out_dir = "cropped/"
frame_dir = "frames/"
GPIO.setmode(GPIO.BCM)
recording_key = 5
GPIO.setup(recording_key,GPIO.IN,GPIO.PUD_UP)

####################################################
# Camera init
####################################################
cam = picamera.PiCamera(stereo_mode='side-by-side', stereo_decimate=False)
cam.framerate = 30
cam.resolution = (1280, 480) # 1280 x 480   2560 x 720  960 x 320
# cam.exposure_mode = 'antishake' # auto

# cam.saturation = 80 # 设置图像视频的饱和度
# cam.brightness = 50 # 设置图像的亮度(50表示白平衡的状态)
# cam.shutter_speed = 6000000 # 相机快门速度
cam.iso = 800 
# ISO标准实际上就是来自胶片工业的标准称谓，ISO是衡量胶片对光线敏感程度的标准。
# 如50 ISO, 64 ISO, 100 ISO表示在曝光感应速度上要比高数值的来得慢，高数值ISO是指超过200以上的标准，如200 ISO, 400 ISO
cam.sharpness = 0 #设置图像的锐度值，默认是0，取值范围是-100~100之间

cam.vflip = True
cam.start_preview(fullscreen=False, window=(0,0,640,480))
# cam.start_preview()

####################################################
# Start
####################################################

recording = False
keyPressed = False

videoTime = ""
in_route = ""


try:     
    while True:
        ####################################################
        # Detect press
        ####################################################

        while True:
            if GPIO.input(recording_key) == 0:
                keyPressed = True
                while GPIO.input(recording_key) == 0:
                    if recording:
                        cam.wait_recording(0.01)
                    else:
                        time.sleep(0.01)
                break
        
        ####################################################
        # Recording 
        ####################################################
        if not recording and keyPressed:
            print("=> Start Recording")
            videoTime = datetime.now().strftime("%y-%m-%d_%H-%M-%S")
            in_route = raw_dir + videoTime + ".h264"
            cam.start_recording(in_route, format="h264", quality=10)
            full_light()
            recording = not recording
        elif recording and keyPressed:
            cam.stop_recording()
            print("=> Done! <{}> saved!".format(videoTime))
            full_dark()
            recording = not recording

        keyPressed = False
    
except KeyboardInterrupt:  
    # 當你按下 CTRL+C 中止程式後，所要做的動作
    print("STOP")
  
except:  
    # 其他例外發生的時候，所要做的動作
    print("Other error or exception occurred!" )
  
finally:  
    GPIO.cleanup() # 把這段程式碼放在 finally 區域，確保程式中止時能夠執行並清掉GPIO的設定！

