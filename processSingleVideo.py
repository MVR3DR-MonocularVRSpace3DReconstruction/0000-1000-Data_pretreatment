import os
import time
import subprocess
import numpy as np
from datetime import datetime
import glob

####################################################
# Settings
####################################################

raw_dir = "raw/"
out_dir = "cropped/"
frame_dir = "frames/"

####################################################
# Start
####################################################

recording = False
keyPressed = False


###################################################
# Process data
###################################################
videos = sorted(glob.glob(raw_dir+"*.h264"))

for v in videos:
    try:     
        print("=> Processing Data...")
        
        print("=> ", v)
        out = v.split("/")[-1].replace(".h264", "")
        print("=> ",out)
        if "left" in out:
            isLeftVideo = True
        else:
            isLeftVideo = False

        cmd = 'ffmpeg -i {} {}.mp4'.format(v, out_dir+out)
        subprocess.run(cmd, shell=True)
        print("=> MPEG4 saved.")

        print("=> Slice frames")
        os.system("rm -rf {0}/{1} && mkdir {0}/{1}".format(frame_dir, out))
        if isLeftVideo:
            os.system("mkdir {0}/{1}/left".format(frame_dir, out))
            cmd = "ffmpeg -i {}.mp4 {}/{}/left/%05d.png".format(out_dir+out, frame_dir, out)
        else:
            os.system("mkdir {0}/{1}/right".format(frame_dir, out))
            cmd = "ffmpeg -i {}.mp4 {}/{}/right/%05d.png".format(out_dir+out, frame_dir, out)

        subprocess.run(cmd, shell=True)
        
    except Exception as e:  
        # 其他例外發生的時候，所要做的動作
        print("Other error or exception occurred!" )
        print(e)
    
    finally:  
        os.system("mv {0} {0}.fin".format(v))
        print("=> Frames sliced.")
