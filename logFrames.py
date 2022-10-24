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
useRaspiCam = False


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
                    time.sleep(0.01)
                break
        
        ####################################################
        # Recording 
        ####################################################

        if not recording and keyPressed:
            print("=> Start Recording")
            videoTime = datetime.now().strftime("%y-%m-%d_%H-%M-%S")

            os.system("rm -rf {0} && mkdir {0}".format(frame_dir+videoTime))
            os.system("mkdir {0}/left/ && mkdir {0}/right/".format(frame_dir+videoTime))
            # cam.start_recording(in_route, format="h264", quality=10)
            cmd = "raspistill -th 480,270 -e bmp -cs 1 -t 0 -tl 50 -vf -hf -o {0}/left/%5d.bmp \
                & raspistill -th 480,270 -e bmp -cs 0 -t 0 -tl 50 -vf -hf -o {0}/right/%5d.bmp".format(frame_dir)

            process = subprocess.Popen(cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=True)
            
            full_light()
            recording = not recording
        elif recording and keyPressed:
            # cam.stop_recording()
            process.kill()
            print("=> Done! <{}> saved!".format(videoTime))
            full_dark()
            time.sleep(0.5)
            recording = not recording

        keyPressed = False
    
except KeyboardInterrupt:  
    # 當你按下 CTRL+C 中止程式後，所要做的動作
    print("STOP")
  
except Exception as e:  
    # 其他例外發生的時候，所要做的動作
    print("Other error or exception occurred!" )
    print(e)
  
finally:  
    GPIO.cleanup() # 把這段程式碼放在 finally 區域，確保程式中止時能夠執行並清掉GPIO的設定！

