# (c) 2026 S. Farzan, Electrical Engineering Department, Cal Poly
# EE 471 (SP26): Planning and Control for Autonomous Vehicles
"""
lab05_ee471.py

Script for EE 471 Lab 5: Environment Interpretation
LiDAR Inverse Measurement Model and Occupancy Grid Mapping for the QCar 2

Students complete four sections in this file:
    Part A1 - Log-Odds Initialization
    Part A2 - Polar occupancy grid update from a single LiDAR scan
    Part A3 - Polar-to-Cartesian patch generation via bilinear interpolation
    Part B2 - Global occupancy grid update using a binary Bayes filter

All occupancy arrays (polarPatch, patch, map) store LOG-ODDS values, not
probabilities. Use the precomputed bounds self.l_low, self.l_prior,
self.l_high, self.l_min, and self.l_max defined in OccupancyGrid.__init__.
"""
# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

#region : File Description and Imports
import numpy as np
from scipy.special import logit, expit
from scipy import ndimage
from threading import Thread, Lock
import time
import pyqtgraph as pg
import signal

from pal.products.qcar import QCar, QCarGPS, IS_PHYSICAL_QCAR, QCAR_CONFIG
from pal.utilities.scope import MultiScope
from pal.utilities.math import find_overlap, wrap_to_2pi, wrap_to_pi
from hal.content.qcar_functions import QCarEKF, QCarDriveController
from hal.products.mats import SDCSRoadMap
#endregion

#================ Part A0/B1 - Experiment Configuration ================
# ===== Timing Parameters
# - tf: experiment duration in seconds.
# - startDelay: delay to give filters time to settle in seconds.
# - controllerUpdateRate: control update rate in Hz. Shouldn't exceed 500
tf = 100
startDelay = 1
controllerUpdateRate = 100

# ===== Vehicle Controller Parameters
# - enableVehicleControl: If True, the QCar will drive through the specified
#   node sequence. If False, the QCar will remain stationary.
# - v_ref: desired velocity in m/s
# - nodeSequence: list of nodes from roadmap. Used for trajectory generation.
enableVehicleControl = False
v_ref = 0.3
nodeSequence = [0, 20, 0]

# ===== Occupancy Grid Parameters
# - cellWidth: edge length for occupancy grid cells (in meters)
# - r_res: range resolution for polar grid cells (in meters)
# - r_max: maximum range of the lidar (in meters)
# - p_low: occupancy probability assigned to cells declared FREE by a scan
# - p_high: occupancy probability assigned to cells declared OCCUPIED by a scan
cellWidth = 0.02
r_res = 0.05
r_max = 10
p_low = 0.4
p_high = 0.6


#region Initial Setup
lock = Lock()

roadmap = SDCSRoadMap()
waypointSequence = roadmap.generate_path(nodeSequence)
initialPose = roadmap.get_node_pose(nodeSequence[0]).squeeze()
x_hat = initialPose
t_hat = 0

if not IS_PHYSICAL_QCAR:
    import qlabs_setup
    from qvl.qcar import QLabsQCar
    hQCar = qlabs_setup.setup(
        initialPosition=[initialPose[0], initialPose[1], 0],
        initialOrientation=[0, 0, initialPose[2]]
    )
    calibrate = False
else:
    calibrate = 'y' in input('do you want to recalibrate? (y/n) ')

# Used to enable safe keyboard triggered shutdown
KILL_THREAD = False
def sig_handler(*args):
    global KILL_THREAD
    KILL_THREAD = True
signal.signal(signal.SIGINT, sig_handler)

gps = QCarGPS(initialPose=initialPose, calibrate=calibrate)
while (not KILL_THREAD) and (gps.readGPS() or gps.readLidar()):
    pass
#endregion


class OccupancyGrid:

    def __init__(self,
            x_min=-4,
            x_max=3,
            y_min=-3,
            y_max=6,
            cellWidth=0.02,
            r_max=5,
            r_res=0.02,
            p_low=0.4,
            p_high=0.6
        ):

        #region define probabilities and their log-odds forms
        # All occupancy arrays in this class store LOG-ODDS values.
        # The five 'l_*' constants below are used as fixed reference values
        # throughout the lab when writing to self.polarPatch, self.patch,
        # and self.map.
        self.p_low = p_low
        self.p_prior = 0.5
        self.p_high = p_high
        self.p_sat = 0.001

        # ============== Part A1 - Log-Odds Initialization ====================
        # Compute the log-odds form of each probability above using the
        # scipy.special.logit function (already imported at the top of the file).
        # Assign each result to the corresponding attribute of self:
        #   self.l_low    : log-odds for cells declared FREE by a scan
        #   self.l_prior  : log-odds for unknown cells (should evaluate to 0)
        #   self.l_high   : log-odds for cells declared OCCUPIED by a scan
        #   self.l_min    : saturation lower bound, computed from self.p_sat
        #   self.l_max    : saturation upper bound, computed from (1 - self.p_sat)
        # Note: avoid passing exactly 0 or 1 to logit, since logit(0) = -inf
        # and logit(1) = +inf. Use self.p_sat and (1 - self.p_sat) instead.
        # YOUR CODE HERE
        if self.p_sat == 0:
            self.p_sat = 0.001 
        elif self.p_sat == 1:
            self.p_sat = 0.999 

        self.l_low = logit(self.p_low)
        self.l_prior = logit(self.p_prior)
        self.l_high = logit(self.p_high)
        self.l_min = logit(self.p_sat)
        self.l_max = logit(1 - self.p_sat)

        #endregion

        self.init_polar_grid(r_max, r_res)

        self.init_world_map(
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            cellWidth=cellWidth
        )
        self.init_patch()


    # ==============  Part A2 - Polar Grid ====================
    def init_polar_grid(self, r_max, r_res):
        # Configuration Parameters for polar grid (already implemented)
        fov = 2*np.pi
        self.phiRes = 1 * np.pi/180
        self.r_max = r_max
        self.r_res = r_res

        # Size of polar patch
        self.mPolarPatch = np.int_(np.ceil(fov / self.phiRes))
        self.nPolarPatch = np.int_(np.floor(self.r_max/self.r_res))

        self.polarPatch = np.zeros(
            shape = (self.mPolarPatch, self.nPolarPatch),
            dtype = np.float32
        )

    def update_polar_grid(self, r):
        # Populate self.polarPatch with the LOG-ODDS occupancy interpretation
        # of one LiDAR scan.
        # - r is a 1D array of distances (in meters) of length self.mPolarPatch
        # - r[i] corresponds to angle i * self.phiRes in the LiDAR frame
        # - r[i] == 0 indicates an invalid / out-of-range reading
        #
        # For each angular bin i with a valid reading, compute the radial
        # bin index k_i = round(r[i] / self.r_res) and assign:
        #     polarPatch[i, 0:k_i]   -> self.l_low      (free)
        #     polarPatch[i, k_i]     -> self.l_high     (obstacle hit)
        #     polarPatch[i, k_i+1:]  -> self.l_prior    (unknown beyond hit)
        # For invalid readings, leave the entire row at self.l_prior.
        #
        # YOUR CODE HERE
        # np.int_ converts index to integer number (does rounding already)
        k = np.int_(self.mPolarPatch)
        n_radial = self.polarPatch.shape[1]
        
        if len(r) == 0:
            return
        
        for i in range(k):
            # print("test")
            if r[i] > 0:
                k_i = round(r[i]/self.r_res)
                if k_i < n_radial:
                    self.polarPatch[i, 0:k_i] = self.l_low
                    self.polarPatch[i, k_i:k_i+1] = self.l_high
                    self.polarPatch[i, k_i+1:] = self.l_prior
                else:
                    self.polarPatch[i,:] = self.l_prior
                


    # ==============  Part A3 - Polar to Cartesian Interpolation ====================
    def init_patch(self):
        # Already implemented: square Cartesian patch sized to ~+/- r_max
        self.nPatch = np.int_(2*np.ceil(self.r_max/self.cellWidth) + 1)
        self.patch = np.zeros(
            shape = (self.nPatch, self.nPatch),
            dtype = np.float32
        )

    def generate_patch(self, psi):
        # Map self.polarPatch into a Cartesian patch (self.patch) centered
        # at the LiDAR and rotated by the vehicle heading th.
        #
        # Recommended approach (use scipy.ndimage.map_coordinates):
        #   1. Build a Cartesian meshgrid covering [-cx, +cx] x [-cy, +cy]
        #      where cx = cy = (self.nPatch * self.cellWidth) / 2.
        #   2. For each Cartesian cell at (xv, yv), compute fractional
        #      polar indices.
        #   3. Call ndimage.map_coordinates(
        #          input=self.polarPatch,
        #          coordinates=[phiPatch, rPatch],
        #          output=self.patch
        #      )
        #      to perform bilinear interpolation in place.
        #
        # YOUR CODE HERE
        cx = cy = (self.nPatch * self.cellWidth) / 2

        xc = np.linspace(-cx, cx, self.nPatch)
        yc = np.linspace(-cy, cy, self.nPatch)
        xv, yv = np.meshgrid(xc, yc)

        rPatch   = np.sqrt(np.square(xv) + np.square(yv)) / self.r_res
        phiPatch = wrap_to_2pi(np.arctan2(yv, xv) + psi) / self.phiRes

        ndimage.map_coordinates(
            input=self.polarPatch,
            coordinates=[phiPatch, rPatch],
            output=self.patch,
        )

    # ==============  Part B2 - Occupancy Grid Update  ====================
    def init_world_map(self,
            x_min = -4,
            x_max = 3,
            y_min = -3,
            y_max = 6,
            cellWidth=0.02
        ):
        # Already implemented: world map covering the SDCS roadmap region
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.cellWidth = cellWidth
        self.xLength = x_max - x_min
        self.yLength = y_max - y_min
        self.m = np.int_(np.ceil(self.yLength/self.cellWidth))
        self.n = np.int_(np.ceil(self.xLength/self.cellWidth))

        self.map = np.full(
            shape = (self.m, self.n),
            fill_value = self.l_prior,
            dtype = np.float32
        )

    def xy_to_ij(self, x, y):
        # Already implemented: world coordinates -> map indices
        i = np.int_(np.round( (self.y_max - y) / self.cellWidth ))
        j = np.int_(np.round( (x - self.x_min) / self.cellWidth ))
        return i, j

    def updateMap(self, x, y, psi, angles, distances):
        # Update the global log-odds map self.map by fusing the latest scan.
        #
        # Steps 1 and 2 below call the methods you completed in Parts A2 / A3:
        self.update_polar_grid(distances)
        self.generate_patch(psi)

        # Step 3 (YOUR CODE BELOW): fuse self.patch into self.map.
        # Recommended approach:
        #   a. Convert (x, y) into map indices (iy, jx) using self.xy_to_ij.
        #   b. Compute the top-left placement of the patch in the map.
        #   c. Use the helper find_overlap (already imported) to obtain the
        #      matching slice objects for the map and the patch.
        #   d. Apply the binary Bayes update in log-odds form (additive),
        #      then clip to [self.l_min, self.l_max].
        #
        # YOUR CODE HERE
        pass
        


def controlLoop():
    #region controlLoop setup
    global KILL_THREAD, x_hat, t_hat
    u = 0
    delta = 0
    # used to limit data sampling to 10hz
    countMax = controllerUpdateRate / 10
    count = 0
    #endregion

    #region Set up plot items
    arrow1 = pg.ArrowItem(
        angle=180,
        tipAngle=60,
        headLen=10,
        tailLen=10,
        tailWidth=5,
        pen={'color': 'w', 'width': 1},
        brush='r'
    )
    arrow1.setPos(0,0)
    scope.axes[1].plot.addItem(arrow1)

    arrow2 = pg.ArrowItem(
        angle=180,
        tipAngle=60,
        headLen=10,
        tailLen=10,
        tailWidth=5,
        pen={'color': 'w', 'width': 1},
        brush='r'
    )
    scope.axes[2].plot.addItem(arrow2)
    #endregion

    #region QCar interface setup
    with lock:
        ekf = QCarEKF(x_0=x_hat)
    driveController = QCarDriveController(waypointSequence, cyclic=False)

    qcar = QCar(readMode=1, frequency=controllerUpdateRate)
    #endregion

    with qcar:
        t0 = time.time()
        t = 0
        while (t < tf+startDelay) and (not KILL_THREAD):
            #region : Loop timing update
            tp = t
            t = time.time() - t0
            dt = t-tp
            #endregion

            #region : Update QCar state estimates and drive controller
            qcar.read()
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
            with lock:
                t_hat = time.time()
                x_hat = ekf.x_hat[:]

            x = ekf.x_hat[0, 0]
            y = ekf.x_hat[1, 0]
            v = qcar.motorTach
            psi = ekf.x_hat[2, 0]
            p = np.array([x, y]) + np.array([np.cos(psi), np.sin(psi)]) * 0.2

            if t < startDelay or (not enableVehicleControl):
                u = 0
                delta = 0
            else:
                u, delta = driveController.update(p, psi, v, v_ref, dt)
            qcar.write(u, delta)
            #endregion

            #region : Update Scopes
            count += 1
            if count >= countMax and t > startDelay:
                scope.axes[2].sample(t, [[p[0], p[1]]])
                arrow1.setStyle(angle=180-psi*180/np.pi)
                arrow2.setPos(p[0],p[1])
                arrow2.setStyle(angle=180-psi*180/np.pi)

                count = 0
            #endregion

            if driveController.steeringController.pathComplete:
                return
            continue
        with lock:
            print('Control thread terminated')


def mappingLoop():
    global KILL_THREAD, x_hat, t_hat

    og = OccupancyGrid(
        cellWidth=cellWidth,
        r_res=r_res,
        r_max=r_max,
        p_low=p_low,
        p_high=p_high
    )

    #region Configure Plots
    scope.axes[0].images[0].rotation = 90
    scope.axes[0].images[0].scale = (og.r_res, -og.phiRes*180/np.pi)
    scope.axes[0].images[0].offset = (0, 0)
    scope.axes[0].images[0].levels = (0, 1)

    scope.axes[1].images[0].scale = (og.r_res, -og.r_res)
    scope.axes[1].images[0].offset = (-og.nPatch/2, -og.nPatch/2)
    scope.axes[1].images[0].levels = (0, 1)

    scope.axes[2].images[0].scale = (og.cellWidth, -og.cellWidth)
    scope.axes[2].images[0].offset = (
        og.x_min/og.cellWidth,
        -og.y_max/og.cellWidth
    )
    scope.axes[2].images[0].levels = (0, 1)
    #endregion

    t0 = time.time()
    while time.time()-t0 < startDelay:
        gps.readLidar()

    while (not KILL_THREAD):
        # Get latest pose estimate from the EKF (Lab 2)
        with lock:
            t = t_hat
            x = x_hat[0,0]
            y = x_hat[1,0]
            psi = x_hat[2,0]

        # ============== Part B3 - LiDAR-to-Rear-Axle Offset ====================
        # The EKF state (x, y, psi) is referenced to the REAR AXLE, but the
        # LiDAR is mounted ~0.125 m forward of the rear axle. Shift the pose
        # to the LiDAR mounting position before updating the map.
        # (Already implemented; do not modify.)
        x += 0.125 * np.cos(psi)
        y += 0.125 * np.sin(psi)
        #endregion

        #Read from Lidar and Update Occupancy Grid
        gps.readLidar()

        if gps.scanTime < t and QCAR_CONFIG['cartype'] == 1:
            continue

        og.updateMap(x, y, psi, gps.angles, gps.distances)

        scope.axes[0].images[0].setImage(image=expit(og.polarPatch))
        scope.axes[1].images[0].setImage(image=expit(og.patch))
        scope.axes[2].images[0].setImage(image=expit(og.map))

    with lock:
        print('Mapping thread terminated')


#region : Setup and run experiment
if __name__ == '__main__':

    #region : Setup Scopes
    if IS_PHYSICAL_QCAR:
        fps = 10
    else:
        fps = 30

    scope = MultiScope(
        rows=2,
        cols=2,
        title='Environment Interpretation',
        fps=fps
    )

    # Polar Patch
    scope.addXYAxis(
        row=0,
        col=0,
        xLabel='Angle [deg]',
        yLabel='Range [m]'
    )
    scope.axes[0].attachImage()

    # Patch
    scope.addXYAxis(
        row=1,
        col=0,
        xLabel='x Position [m]',
        yLabel='y Position [m]'
    )
    scope.axes[1].attachImage()

    # Generated Map and followed trajectory
    scope.addXYAxis(
        row=0,
        col=1,
        rowSpan=2,
        xLabel='x Position [m]',
        yLabel='y Position [m]',
        xLim=(-4, 3),
        yLim=(-2, 6)
    )
    scope.axes[2].attachSignal(name='Measured', width=2, style='--.')
    scope.axes[2].attachImage()

    referencePath = pg.PlotDataItem(
        pen={'color': (85,168,104), 'width': 2},
        name='Reference'
    )
    referencePath.setData(waypointSequence[0, :], waypointSequence[1, :])
    scope.axes[2].plot.addItem(referencePath)
    #endregion

    #region : Setup threads, then run experiment
    controlThread = Thread(target=controlLoop)
    mappingThread = Thread(target=mappingLoop)

    controlThread.start()
    mappingThread.start()

    try:
        while controlThread.is_alive() and mappingThread.is_alive():
            MultiScope.refreshAll()
            time.sleep(0.01)
    finally:
        KILL_THREAD = True
        controlThread.join()
        mappingThread.join()
        gps.terminate()
    #endregion

    if not IS_PHYSICAL_QCAR:
        qlabs_setup.terminate()

    input('Experiment complete. Press any key to exit...')
#endregion
