# (c) 2026 S. Farzan, Electrical Engineering Department, Cal Poly
# EE 471 (SP26): Planning and Control for Autonomous Vehicles
"""
lab01_ee471.py

Script for EE 471 Lab 1:
IMU Sensor Interfacing, Noise Characterization, and Statistical Modeling for QCar
Please review the Lab 1 PDF document on Canvas.

Authors: Ashik Islam, William Taing
Group 7B
"""
# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

#region : File Description and Imports

import matplotlib.pyplot as plt
import numpy as np
import time

from pal.products.qcar import QCar, IS_PHYSICAL_QCAR
from pal.utilities.stream import BasicStream , Timeout
from pal.utilities.scope import MultiScope, Signal
from threading import Thread

if not IS_PHYSICAL_QCAR:
    import qlabs_setup
    qlabs_setup.setup()

Signal.maxChunks = 1500
#endregion

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

#region : SensorInterfacing Class Setup

class sensorInterfacing():
    def __init__(self, taskRate,specifiedSamples):

        # QCar DAQ configuration
        self.readRate = int(taskRate) #Hz
        self.samples  = int(specifiedSamples)
        self.mode     = "read"

        # Configure QCar properties
        self.myQCar = QCar(frequency = self.readRate, readMode = 1)

        # Variable for saving sensor readings
        self.accelerometerData = np.zeros((3))
        self.gyroscopeData     = np.zeros((3))
        self.plotData = np.zeros((self.samples,6))

        self.time = np.zeros((int(specifiedSamples)))

        # Define scopes for plotting
        txt = "Accelerometer Data Scope"
        self.accelPlot = MultiScope(title=txt, rows=1, cols=1, fps=taskRate)
        self.accelPlot.addAxis(row = 0,col= 0,
                            timeWindow=int(self.samples*(1/self.readRate)),
                            set_ylabel='m/s^2',
                            set_xlabel='time (s)')

        self.accelPlot.axes[0].attachSignal(name = 'Accel in x')
        self.accelPlot.axes[0].attachSignal(name = 'Accel in y')
        self.accelPlot.axes[0].attachSignal(name = 'Accel in z')
        self.accelPlot.axes[0].yLim = (-11, 11)

        txt = "Gyroscope Data Scope"
        self.gyroPlot = MultiScope(title=txt,rows=1, cols=1, fps=taskRate)

        self.gyroPlot.addAxis(row =0, col = 0,
                            timeWindow=int(self.samples*(1/self.readRate)),
                            set_ylabel='rad/s',
                            set_xlabel='time (s)')

        self.gyroPlot.axes[0].attachSignal(name = 'Gyro in x')
        self.gyroPlot.axes[0].attachSignal(name = 'Gyro in y')
        self.gyroPlot.axes[0].attachSignal(name = 'Gyro in z')
        self.gyroPlot.axes[0].yLim = (-2.5, 2.5)


    # Timing function used to send timestamp to sigmoid generator.
    def elapsedTime(self,startTime):
        return time.time()-startTime

    def sensor_read(self):
        t0       = time.time()
        loopTime = self.samples*(1/self.readRate) # Simulation Time
        print("Sim time: ",loopTime)
        self.timeStamp = 0

        # Run the data logging only for the length of time specified
        i = 0
        while i < self.samples:
            if self.mode == "read":
                if self.timeStamp < int(loopTime/2):
                    self.myQCar.write(throttle=0.075, steering=0.5)
                if self.timeStamp >= int(loopTime/2):
                    self.myQCar.write(throttle=0.075, steering=-0.5)


    # ============  Part A2 - Sensor Reading and Data Parsing =============
            self.timeStamp = self.elapsedTime(t0)

            # TODO: implementation for parsing out the IMU information
            # YOUR CODE HERE
            self.myQCar.read()
            self.accelerometerData = self.myQCar.accelerometer
            self.gyroscopeData = self.myQCar.gyroscope
            self.time[i] = time.perf_counter()

            # print(f"Data = {self.accelerometerData}, {self.gyroscopeData},{self.time[i]}")

            self.plotData[i] = np.hstack((self.accelerometerData, self.gyroscopeData))

            
            # Data sent to scope class for visualization
            self.accelPlot.axes[0].sample(self.timeStamp,
                                [self.accelerometerData[0],
                                self.accelerometerData[1],
                                self.accelerometerData[2]].copy())

            self.gyroPlot.axes[0].sample(self.timeStamp,
                                [self.gyroscopeData[0],
                                self.gyroscopeData[1],
                                self.gyroscopeData[2]].copy())

            i += 1

        self.myQCar.write(throttle=0, steering=0)

    def sensor_stats(self):

        # ========== Part B2 - Statistical Properties ==========
        mu = np.zeros(6)
        var = np.zeros(6)
        
        # TODO: implementation for calculating mu and variance
        # YOUR CODE HERE
        for i in range(self.plotData.shape[1]):             # for each col
            mu[i] = np.mean(self.plotData[:, i])            # get mean for each variable
            var[i] = np.var(self.plotData[:, i], ddof=1)    # calc unbiased var

        print("Mean: ", mu)
        print("Variance: ",var)

        # ========== Part B3 - Data Plotting  ==========
        #  TODO: implementation for plotting sensor characteristics
        # YOUR CODE HERE
        fig, axes = plt.subplots(3,2, layout="constrained")

        axes[0,0].plot(self.time, self.plotData[:,0])
        axes[0,0].set_xlabel("time [s]")
        axes[0,0].set_ylabel("ax [m/s^2]")

        axes[1,0].plot(self.time, self.plotData[:,1])
        axes[1,0].set_xlabel("time [s]")
        axes[1,0].set_ylabel("ay [m/s^2]")

        axes[2,0].plot(self.time, self.plotData[:,2])
        axes[2,0].set_ylabel("az [m/s^2]")
        axes[2,0].set_xlabel("time [s]")

        axes[0,1].plot(self.time, self.plotData[:,3])
        axes[0,1].set_xlabel("time [s]")
        axes[0,1].set_ylabel("wx [rad/s]")

        axes[1,1].plot(self.time, self.plotData[:,4])
        axes[1,1].set_xlabel("time [s]")
        axes[1,1].set_ylabel("wy [rad/s]")

        axes[2,1].plot(self.time, self.plotData[:,5])
        axes[2,1].set_xlabel("time [s]")
        axes[2,1].set_ylabel("wz [rad/s]")
        fig.suptitle('Time Series Plots')

        num_bins = 100
        fig, axes =plt.subplots(3,2, layout="constrained")
        axes[0,0].hist(self.plotData[:,0], bins=num_bins)
        axes[0,0].set_ylabel("samples")
        axes[0,0].set_xlabel("ax [m/s^2]")

        axes[1,0].hist(self.plotData[:,1], bins=num_bins)
        axes[1,0].set_ylabel("samples")
        axes[1,0].set_xlabel("ay [m/s^2]")

        axes[2,0].hist(self.plotData[:,2], bins=num_bins)
        axes[2,0].set_ylabel("samples")
        axes[2,0].set_xlabel("az [m/s^2]")

        axes[0,1].hist(self.plotData[:,3], bins=num_bins)
        axes[0,1].set_ylabel("samples")
        axes[0,1].set_xlabel("wx [rad/s]")

        axes[1,1].hist(self.plotData[:,4], bins=num_bins)
        axes[1,1].set_ylabel("samples")
        axes[1,1].set_xlabel("wy [rad/s]")

        axes[2,1].hist(self.plotData[:,5], bins=num_bins)
        axes[2,1].set_ylabel("samples")
        axes[2,1].set_xlabel("wz [rad/s]")
        fig.suptitle('Histogram Plots')


        plt.tight_layout()
        plt.show()
        

    def start(self, mode = "read"):

        if mode == "read":
            mainThread = Thread(target=self.sensor_read)
            mainThread.start()

            while mainThread.is_alive():
                self.accelPlot.refresh()
                self.gyroPlot.refresh()

            time.sleep(15)


        if mode == "sensor_stats":
            self.mode = mode
            mainThread = Thread(target=self.sensor_read)
            mainThread.start()

            while mainThread.is_alive():
                self.accelPlot.refresh()
                self.gyroPlot.refresh()

            self.sensor_stats()

#endregion

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

#region : Main Loop
def main():
    try:
        '''
        INPUTS
        taskRate = Int The frequency (Hz) at which the read/write operation
                will be performed.
        specifiedSamples = Int Number of samples saved for statistical analysis of
                        sensor data.
        '''
        # ========= Part A1/B1 - Student Inputs for Sensor Interfacing ===========
        sensorLab = sensorInterfacing(taskRate = 120,
                            specifiedSamples = 600)

        ''' TODO: Select the specific activity for the sensor interfacing lab
        List of current activities:
        - read (sensor interfacing)
        - sensor_stats (sensor characterization)
        '''
        # YOUR CODE HERE
        sensorMode = "sensor_stats"


        sensorLab.start(mode = sensorMode)
    finally:
        if not IS_PHYSICAL_QCAR:
            qlabs_setup.terminate()

        input('Experiment complete. Press any key to exit...')


#endregion

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

#region : Run
if __name__ == '__main__':
    main()
#endregion
