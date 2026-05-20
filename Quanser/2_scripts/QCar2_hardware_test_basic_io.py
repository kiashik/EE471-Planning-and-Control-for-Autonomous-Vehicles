# (c) 2026 S. Farzan, Electrical Engineering Department, Cal Poly
# EE 471 Test Script for QCar 2
'''hardware_test_basic_io.py
This example demonstrates how to use the QCar class to perform basic I/O.
Learn how to write throttle and steering, as well as LED commands to the
vehicle, and read sensor data such as battery voltage. See the QCar class
definition for other sensor buffers such as motorTach, accelometer, gyroscope
etc.
'''
import numpy as np
import time
from pal.products.qcar import QCar, IS_PHYSICAL_QCAR

#Initial Setup
sampleRate = 200
runTime = 15.0 # seconds

with QCar(readMode=1, frequency=sampleRate) as myCar:
    t0 = time.time()
    while time.time() - t0  < runTime:
        t = time.time()

        # Read from onboard sensors
        myCar.read()

        # Basic IO - write motor commands
        #TODO: Implement sinusoidal throttle and steering commands
        M_throttle = 0.075
        A_throttle = 0.25
        throttle = M_throttle * np.sin(t * A_throttle * np.pi)

        M_steering = 0.05 # rad
        A_steering = 1
        steering = M_steering * np.sin(t * A_steering * np.pi)

        LEDs = np.array([0, 0, 0, 0, 0, 0, 1, 1])
        #TODO: Implement the LED behavior
        if steering < 0:    # turn right, LED 1, 3 ON
            LEDs[1] = 1
            LEDs[3] = 1
        elif steering > 0:  # turn left, LED 0, 2 ON
            LEDs[0] = 1
            LEDs[2] = 1
        else:   # going straight, so turn all turning lndicator LEDs off
            LEDs[1] = 0
            LEDs[3] = 0
            LEDs[0] = 0
            LEDs[2] = 0
        
        if throttle < 0:
            LEDs[5] = 1
        else:
            LEDs[5] = 0




        myCar.write(throttle, steering, LEDs)

        print(
            f'time: {(t-t0):.2f}'
            + f', Battery Voltage: {myCar.batteryVoltage:.2f}'
            + f', Motor Current: {myCar.motorCurrent:.2f}'
            + f', Motor Encoder: {myCar.motorEncoder}'
            + f', Motor Tach: {myCar.motorTach:.2f}'
            + f', Accelerometer: {myCar.accelerometer}'
            + f', Gyroscope: {myCar.gyroscope}'
        )