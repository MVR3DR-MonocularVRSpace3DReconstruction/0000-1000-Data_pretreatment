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
    print("=> Processing Data...")
    print("=> ", v)
    out = v.split("/")[-1].replace(".h264", "")
    print("=> ",out)
    cmd = 'ffmpeg -i {} -filter:v "crop=in_w/2:in_h:0:0" -c:a copy {}-right.mp4'.format(v, out_dir+out)
    subprocess.run(cmd, shell=True)

    cmd = 'ffmpeg -i {} -filter:v "crop=in_w/2:in_h:in_w/2:0" -c:a copy {}-left.mp4'.format(v, out_dir+out)
    subprocess.run(cmd, shell=True)

    print("=> L/R sights saved.")

    print("=> Slice frames")

    os.system("rm -rf {0}/{1} && mkdir {0}/{1}".format(frame_dir, out))
    os.system("mkdir {0}/{1}/left && mkdir {0}/{1}/right".format(frame_dir, out))

    cmd = "ffmpeg -i {}-left.mp4 {}/{}/left/%05d.png".format(out_dir+out, frame_dir, out)
    subprocess.run(cmd, shell=True)
    cmd = "ffmpeg -i {}-right.mp4 {}/{}/right/%05d.png".format(out_dir+out, frame_dir, out)
    subprocess.run(cmd, shell=True)

    os.system("mv {0} {0}.fin".format(v))
    print("=> Frames sliced.")


