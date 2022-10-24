from __future__ import print_function
import qwiic_icm20948
import time
import sys
import smbus

import i2cdriver

i2c = i2cdriver.I2CDriver("/dev/i2c-1") # gpiochip0  i2c-1

i2c.scan()
import qwiic_i2c
connectedDevices = qwiic_i2c.getI2CDriver()

print(connectedDevices)

def runExample():
    bus = smbus.SMBus(1)

    time.sleep(1) #wait here to avoid 121 IO Error
    while False:
        data = bus.read_i2c_block_data(0x68,0x08,16)
        result = 0
        for b in data:
            result = result * 256 + int(b)
        print(result)
        time.sleep(0.01)
        
    print("\nSparkFun 9DoF ICM-20948 Sensor  Example 1\n")
    # help(qwiic_icm20948.QwiicIcm20948)
    IMU = qwiic_icm20948.QwiicIcm20948(0x68,qwiic_i2c.getI2CDriver())

    if IMU.isConnected() == False:
        print("The Qwiic ICM20948 device isn't connected to the system. Please check your connection", \
            file=sys.stderr)
        return

    IMU.begin()

    while True:
        if IMU.dataReady():
            IMU.getAgmt() # read all axis and temp from sensor, note this also updates all instance variables
            print(\
             '{: 06d}'.format(IMU.axRaw)\
            , '\t', '{: 06d}'.format(IMU.ayRaw)\
            , '\t', '{: 06d}'.format(IMU.azRaw)\
            , '\t', '{: 06d}'.format(IMU.gxRaw)\
            , '\t', '{: 06d}'.format(IMU.gyRaw)\
            , '\t', '{: 06d}'.format(IMU.gzRaw)\
            , '\t', '{: 06d}'.format(IMU.mxRaw)\
            , '\t', '{: 06d}'.format(IMU.myRaw)\
            , '\t', '{: 06d}'.format(IMU.mzRaw)\
            )
            time.sleep(0.03)
        else:
            print("Waiting for data")
            time.sleep(0.5)

if __name__ == '__main__':
    try:
        runExample()
    except (KeyboardInterrupt, SystemExit) as exErr:
        print("\nEnding Example 1")
        sys.exit(0)