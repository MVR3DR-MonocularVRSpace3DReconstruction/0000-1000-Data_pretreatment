import RPi.GPIO as GPIO

import smbus
import numpy as np
import os


bus = smbus.SMBus(1)



# This is the address we setup in the Arduino Program
#Slave Address 1
address = 0x03

#Slave Address 2
address_2 = 0x04

def writeNumber(value):
    bus.write_byte(address, value)
    bus.write_byte(address_2, value)
    # bus.write_byte_data(address, 0, value)
    return -1

def readNumber(add):
    # number = bus.read_byte(address)
    number = bus.read_byte_data(add, 1)
    return number

while True:
    data = bus.read_i2c_block_data(0x68, 0x00, 32)
    print(data)

for idx in range(70):
    print("=> ",hex(idx))
    try:
        
        num = readNumber(hex(idx))
        print(num)
    except:
        continue

