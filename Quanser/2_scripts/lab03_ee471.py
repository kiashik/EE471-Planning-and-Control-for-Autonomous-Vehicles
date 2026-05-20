# (c) 2026 S. Farzan, Electrical Engineering Department, Cal Poly
# EE 471 (SP26): Planning and Control for Autonomous Vehicles
"""
lab03_ee471.py

Script for EE 471 Lab 3:
Vehicle longitudinal and lateral controllers for QCar
Please review the Lab 3 PDF document on Canvas.
"""
# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

#region : File Description and Imports
import os
import signal
import numpy as np
from threading import Thread
import time
import cv2
import pyqtgraph as pg

from pal.products.qcar import QCar, QCarGPS, IS_PHYSICAL_QCAR
from pal.utilities.scope import MultiScope
from pal.utilities.math import wrap_to_pi
from hal.content.qcar_functions import QCarEKF
from hal.products.mats import SDCSRoadMap
import pal.resources.images as images


#================ Part A1/B1: Experiment Configuration ================
# ===== Timing Parameters
# - tf: experiment duration in seconds.
# - startDelay: delay to give filters time to settle in seconds.
# - controllerUpdateRate: control update rate in Hz. Shouldn't exceed 500
tf = 50
startDelay = 1
controllerUpdateRate = 100

# ===== Part A1: Speed Controller Parameters
# - v_ref: desired velocity in m/s
# - K_p: proportional gain for speed controller
# - K_i: integral gain for speed controller
v_ref = 0.5
K_p = 0.8
K_i = 1.4

# ===== Part A1/B1: Steering Controller Parameters
# - enableSteeringControl: whether or not to enable steering control
# - K_stanley: K gain for stanley controller
# - nodeSequence: list of nodes from roadmap. Used for trajectory generation.
enableSteeringControl = True
K_stanley = 0.5
# nodeSequence = [10, 4, 6, 0, 13, 17, 20, 10]
nodeSequence = [2, 4, 6, 8, 10, 2]

#endregion
# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

#region : Initial setup
if enableSteeringControl:
    roadmap = SDCSRoadMap(leftHandTraffic=False)
    waypointSequence = roadmap.generate_path(nodeSequence)
    initialPose = roadmap.get_node_pose(nodeSequence[0]).squeeze()
else:
    initialPose = [0, 0, 0]

if not IS_PHYSICAL_QCAR:
    import qlabs_setup
    qlabs_setup.setup(
        initialPosition=[initialPose[0] +0.2, initialPose[1]-0.5, 0],
        initialOrientation=[0, 0, initialPose[2]+0.1]
    )
    calibrate=False
else:
    calibrate =  'y' in input('do you want to recalibrate?(y/n)')

# Define the calibration pose
# Calibration pose is either [0,0,-pi/2] or [0,2,-pi/2]
# Comment out the one that is not used 
calibrationPose = [0,0,-np.pi/2]
# calibrationPose = [0,2,-np.pi/2]

# Used to enable safe keyboard triggered shutdown
global KILL_THREAD
KILL_THREAD = False
def sig_handler(*args):
    global KILL_THREAD
    KILL_THREAD = True
signal.signal(signal.SIGINT, sig_handler)
#endregion

class SpeedController:

    def __init__(self, kp=0, ki=0):
        self.maxThrottle = 0.3

        self.kp = kp
        self.ki = ki

        self.ei = 0


    # ==============  Part A2 -  Speed Control  ====================
    def update(self, v, v_ref, dt):
        # # self.kp – proportional gain Kp ,
        # self.ki – integral gain Ki ,
        # self.ei – persistent integrator state ei , initialized to 0,
        # self.maxThrottle – saturation limit umax = 0:3

        ev = v_ref - v
        self.ei += dt * ev
        u = self.kp * ev + self.ki * self.ei
        return np.clip(u, -self.maxThrottle, self.maxThrottle)

    

class SteeringController:

    def __init__(self, waypoints, k=1, cyclic=True):
        self.maxSteeringAngle = np.pi/6

        self.wp = waypoints
        self.N = len(waypoints[0, :])
        self.wpi = 0

        self.k = k
        self.cyclic = cyclic

        self.p_ref = (0, 0)
        self.psi_ref = 0

    # ==============  Part B2 -  Steering Control  ====================
    def update(self, p, psi, speed):
        '''
        # self.wp – waypoint array of shape (2; N),
        # self.N – number of waypoints,
        # self.wpi – index of the current segment start (updated inside update),
        # self.cyclic – boolean flag; if True, the path loops back to the start,
        # self.k – Stanley gain k (from K stanley),
        # self.maxSteeringAngle – saturation limit ‹max = ı=6,
        # self.p_ref, self.th_ref – closest-point and path-tangent values, stored each cycle for plotting
        '''

        wp_1 = self.wp[:, np.mod(self.wpi, self.N-1)]
        wp_2 = self.wp[:, np.mod(self.wpi+1, self.N-1)]  
   
        t_hat = (wp_2 - wp_1) / np.linalg.norm(wp_2-wp_1)   # tangent vector along segment
        theta_path = np.arctan2(t_hat[1], t_hat[0])           # theta of path

        # print(f"wp_1{wp_1.shape}: {wp_1}")
        s = np.dot(p - wp_1, t_hat)     # arclength
        # print(f"s{type(s)}: {s}")

        if abs(s) >= np.linalg.norm(wp_2 - wp_1):    # if passed wp2
            if self.cyclic:
                self.wpi = (self.wpi + 1) % (self.N - 1)    # modulus and loop back to 0 index
            else:
                self.wpi = min(self.wpi + 1, self.N - 2)    # continue until N-2 waypoint

            # # update waypoints
            # wp_1 = self.wp[:, np.mod(self.wpi, self.N - 1)]
            # wp_2 = self.wp[:, np.mod(self.wpi + 1, self.N - 1)]
             
            # # recalculate
            # t_hat = (wp_2 - wp_1) / np.linalg.norm(wp_2-wp_1)   # projection onto path segment
            # theta_path = np.arctan2(t_hat[1], t_hat[0])           # theta of path
            # s = np.dot(p - wp_1, t_hat)     # arclength
    
        p_ref = wp_1 + s * t_hat    # closest point
        e =  p_ref -p               # crosstrack error 
        ect = t_hat[0] * e[1] - t_hat[1] * e[0]
        theta_e = wrap_to_pi(theta_path - psi)    # wrap theta error
        delta = theta_e + np.arctan2(self.k * ect, speed)   # calculate stanley delta command
        delta = np.clip(delta, -self.maxSteeringAngle, self.maxSteeringAngle)   # clamp steering angle

        self.p_ref = p_ref          # update point reference
        self.th_ref = theta_path    # update theta reference
        # print(f"delta{type(delta)}: {delta.shape}, {delta}")
        # print(f"np.arctan2(self.k * ect, speed): {np.arctan2(self.k * ect, speed)}")
        return delta

def controlLoop():
    #region controlLoop setup
    global KILL_THREAD
    u = 0
    delta = 0
    # used to limit data sampling to 10hz
    countMax = controllerUpdateRate / 10
    count = 0
    #endregion

    #region Controller initialization
    speedController = SpeedController(
        kp=K_p,
        ki=K_i
    )
    if enableSteeringControl:
        steeringController = SteeringController(
            waypoints=waypointSequence,
            k=K_stanley
        )
    #endregion

    #region QCar interface setup
    qcar = QCar(readMode=1, frequency=controllerUpdateRate)
    if enableSteeringControl or calibrate:
        ekf = QCarEKF(x_0=initialPose)
        gps = QCarGPS(initialPose=calibrationPose,calibrate=calibrate)
    else:
        gps = memoryview(b'')
    #endregion

    with qcar, gps:
        t0 = time.time()
        t=0
        while (t < tf+startDelay) and (not KILL_THREAD):
            #region : Loop timing update
            tp = t
            t = time.time() - t0
            dt = t-tp
            #endregion

            #region : Read from sensors and update state estimates
            qcar.read()
            if enableSteeringControl:
                if gps.readGPS():
                    y_gps = np.array([
                        gps.position[0],
                        gps.position[1],
                        gps.orientation[2]
                    ])
                    ekf.update(
                        [qcar.motorTach, delta],
                        dt,
                        y_gps,
                        qcar.gyroscope[2],
                    )
                else:
                    ekf.update(
                        [qcar.motorTach, delta],
                        dt,
                        None,
                        qcar.gyroscope[2],
                    )

                x = ekf.x_hat[0,0]
                y = ekf.x_hat[1,0]
                psi = ekf.x_hat[2,0]
                p = ( np.array([x, y])
                    + np.array([np.cos(psi), np.sin(psi)]) * 0.2)
            v = qcar.motorTach
            #endregion

            #region : Update controllers and write to car
            if t < startDelay:
                u = 0
                delta = 0
            else:
                #region : Speed controller update
                u = speedController.update(v, v_ref, dt)
                #endregion

                #region : Steering controller update
                if enableSteeringControl:
                    delta = steeringController.update(p, psi, v)
                else:
                    delta = 0
                #endregion

            qcar.write(u, delta)
            #endregion

            #region : Update Scopes
            count += 1
            if count >= countMax and t > startDelay:
                t_plot = t - startDelay

                # Speed control scope
                speedScope.axes[0].sample(t_plot, [v, v_ref])
                speedScope.axes[1].sample(t_plot, [v_ref-v])
                speedScope.axes[2].sample(t_plot, [u])

                # Steering control scope
                if enableSteeringControl:
                    steeringScope.axes[4].sample(t_plot, [[p[0],p[1]]])

                    p[0] = ekf.x_hat[0,0]
                    p[1] = ekf.x_hat[1,0]

                    x_ref = steeringController.p_ref[0]
                    y_ref = steeringController.p_ref[1]
                    psi_ref = steeringController.psi_ref

                    x_ref = gps.position[0]
                    y_ref = gps.position[1]
                    psi_ref = gps.orientation[2]

                    steeringScope.axes[0].sample(t_plot, [p[0], x_ref])
                    steeringScope.axes[1].sample(t_plot, [p[1], y_ref])
                    steeringScope.axes[2].sample(t_plot, [psi, psi_ref])
                    steeringScope.axes[3].sample(t_plot, [delta])


                    arrow.setPos(p[0], p[1])
                    arrow.setStyle(angle=180-psi*180/np.pi)

                count = 0
            #endregion
            continue
        qcar.read_write_std(throttle= 0, steering= 0)

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

#region : Setup and run experiment
if __name__ == '__main__':

    #region : Setup scopes
    if IS_PHYSICAL_QCAR:
        fps = 10
    else:
        fps = 30
    # Scope for monitoring speed controller
    speedScope = MultiScope(
        rows=3,
        cols=1,
        title='Vehicle Speed Control',
        fps=fps
    )
    speedScope.addAxis(
        row=0,
        col=0,
        timeWindow=tf,
        yLabel='Vehicle Speed [m/s]',
        yLim=(0, 1)
    )
    speedScope.axes[0].attachSignal(name='v_meas', width=2)
    speedScope.axes[0].attachSignal(name='v_ref')

    speedScope.addAxis(
        row=1,
        col=0,
        timeWindow=tf,
        yLabel='Speed Error [m/s]',
        yLim=(-0.5, 0.5)
    )
    speedScope.axes[1].attachSignal()

    speedScope.addAxis(
        row=2,
        col=0,
        timeWindow=tf,
        xLabel='Time [s]',
        yLabel='Throttle Command [%]',
        yLim=(-0.3, 0.3)
    )
    speedScope.axes[2].attachSignal()

    # Scope for monitoring steering controller
    if enableSteeringControl:
        steeringScope = MultiScope(
            rows=4,
            cols=2,
            title='Vehicle Steering Control',
            fps=fps
        )

        steeringScope.addAxis(
            row=0,
            col=0,
            timeWindow=tf,
            yLabel='x Position [m]',
            yLim=(-2.5, 2.5)
        )
        steeringScope.axes[0].attachSignal(name='x_meas')
        steeringScope.axes[0].attachSignal(name='x_ref')

        steeringScope.addAxis(
            row=1,
            col=0,
            timeWindow=tf,
            yLabel='y Position [m]',
            yLim=(-1, 5)
        )
        steeringScope.axes[1].attachSignal(name='y_meas')
        steeringScope.axes[1].attachSignal(name='y_ref')

        steeringScope.addAxis(
            row=2,
            col=0,
            timeWindow=tf,
            yLabel='Heading Angle [rad]',
            yLim=(-3.5, 3.5)
        )
        steeringScope.axes[2].attachSignal(name='psi_meas')
        steeringScope.axes[2].attachSignal(name='psi_ref')

        steeringScope.addAxis(
            row=3,
            col=0,
            timeWindow=tf,
            yLabel='Steering Angle [rad]',
            yLim=(-0.6, 0.6)
        )
        steeringScope.axes[3].attachSignal()
        steeringScope.axes[3].xLabel = 'Time [s]'

        steeringScope.addXYAxis(
            row=0,
            col=1,
            rowSpan=4,
            xLabel='x Position [m]',
            yLabel='y Position [m]',
            xLim=(-2.5, 2.5),
            yLim=(-1, 5)
        )

        im = cv2.imread(
            images.SDCS_CITYSCAPE,
            cv2.IMREAD_GRAYSCALE
        )

        steeringScope.axes[4].attachImage(
            scale=(-0.002035, 0.002035),
            offset=(1125,2365),
            rotation=180,
            levels=(0, 255)
        )
        steeringScope.axes[4].images[0].setImage(image=im)

        referencePath = pg.PlotDataItem(
            pen={'color': (85,168,104), 'width': 2},
            name='Reference'
        )
        steeringScope.axes[4].plot.addItem(referencePath)
        referencePath.setData(waypointSequence[0, :],waypointSequence[1, :])

        steeringScope.axes[4].attachSignal(name='Estimated', width=2)

        arrow = pg.ArrowItem(
            angle=180,
            tipAngle=60,
            headLen=10,
            tailLen=10,
            tailWidth=5,
            pen={'color': 'w', 'fillColor': [196,78,82], 'width': 1},
            brush=[196,78,82]
        )
        arrow.setPos(initialPose[0], initialPose[1])
        steeringScope.axes[4].plot.addItem(arrow)
    #endregion

    #region : Setup control thread, then run experiment
    controlThread = Thread(target=controlLoop)
    controlThread.start()

    try:
        while controlThread.is_alive() and (not KILL_THREAD):
            MultiScope.refreshAll()
            time.sleep(0.01)
    finally:
        KILL_THREAD = True
    #endregion
    if not IS_PHYSICAL_QCAR:
        qlabs_setup.terminate()

    input('Experiment complete. Press any key to exit...')
#endregion