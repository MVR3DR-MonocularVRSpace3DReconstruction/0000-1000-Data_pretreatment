import io
import time
from datetime import datetime
import RPi.GPIO as GPIO
import picamera

import numpy as np

LED = [26,12,17,18]
GPIO.setmode(GPIO.BCM)

def ledWrite(pin, value):
	if value:
		GPIO.output(pin, GPIO.HIGH)
	else:
		GPIO.output(pin, GPIO.LOW)

for idx in LED:
    print("=> ",idx)
    GPIO.setup(idx, GPIO.OUT)
    ledWrite(idx, 0)

def full_light():
    try:
        for idx in LED:
            ledWrite(idx,1)
    except:
        print("except")
        GPIO.cleanup()

def full_dark():
  try:
    for idx in LED:
        ledWrite(idx,0)
  except:
        print("except")
        GPIO.cleanup()

def light_one(idx):
    try:
        ledWrite(LED[idx],1)
    except:
        print("except")
        GPIO.cleanup()

def dark_one(idx):
    try:
        ledWrite(LED[idx],0)
    except:
        print("except")
        GPIO.cleanup()