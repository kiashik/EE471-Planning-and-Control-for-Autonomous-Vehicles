# (c) 2026 S. Farzan, Electrical Engineering Department, Cal Poly
# EE 471 (SP26): Planning and Control for Autonomous Vehicles
"""
lab06_ee471.py

Script for EE 471 Lab 6: Global Motion Planning for the QCar 2

Students implement Dijkstra's algorithm and A* search over the roadmap
graph to compute a node sequence between user-specified start and goal nodes.
The computed sequence is then used to generate a reference path for the
Lab 3 vehicle controllers (longitudinal speed + Stanley steering).
"""
# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

#region : File Description and Imports
import os
import signal
import heapq
import time
import numpy as np
from threading import Thread
import matplotlib.pyplot as plt

import cv2
import pyqtgraph as pg

from pal.products.qcar import QCar, QCarGPS, IS_PHYSICAL_QCAR
from pal.utilities.scope import MultiScope
from pal.utilities.math import wrap_to_pi
from hal.content.qcar_functions import QCarEKF
from hal.products.mats import SDCSRoadMap
import pal.resources.images as images

#endregion


#================ Part A1 / B1 / D1 - Experiment Configuration ================
# ===== Planner Parameters
# - startNode: index of the start node in the SDCS roadmap.
# - goalNode:  index of the goal node in the SDCS roadmap.
# - plannerType: which algorithm to use to compute the node sequence.
#       'dijkstra'             : uniform-cost search using edge.length
#       'astar'                : A* with Euclidean heuristic on (x, y)
#       'astar_turn_penalty'   : A* + extra cost on heading changes (Part C)
# - turnPenalty: scalar weight on |delta heading| at each node (rad^-1).
#       Only used when plannerType == 'astar_turn_penalty'.
# - leftHandTraffic: traffic-flow convention used to construct the roadmap.
#       Must match Lab 3 / Lab 5 to keep node indices consistent.
startNode = 20
goalNode = 6 # 15
plannerType = 'astar_turn_penalty'   # 'dijkstra' | 'astar' | 'astar_turn_penalty'
turnPenalty = 10.0
leftHandTraffic = False

# ===== Driving Parameters
# - enableDriving: if False, only run the planner and show the plan.
#                  if True,  also drive the QCar along the planned path.
# - tf: experiment duration in seconds (after the start delay).
# - startDelay: delay to give filters time to settle in seconds.
# - controllerUpdateRate: control update rate in Hz. Shouldn't exceed 500.
enableDriving = False
tf = 30
startDelay = 1
controllerUpdateRate = 100

# ===== Speed Controller Parameters
# - v_ref: desired velocity in m/s
# - K_p:   proportional gain for speed controller
# - K_i:   integral gain for speed controller
v_ref = 0.5
K_p = 0.08
K_i = 1.0

# ===== Steering Controller Parameters
# - K_stanley: K gain for Stanley controller
K_stanley = 1

# ===== Goal-Stop Parameter
# When the QCar finishes the last segment of the planned path AND is
# within goalStopDistance of the final waypoint, it is declared "at
# goal": the speed reference is held at zero (so the PI loop actively
# brakes) and steering is frozen at zero. The latch never resets.
# - goalStopDistance: meters from final waypoint at which to stop.
goalStopDistance = 0.1
#endregion


# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

#region : Path Planner

class PathPlanner:
    """Graph-search-based motion planner over the SDCS roadmap.

    The planner treats the roadmap as a directed graph: each node has zero
    or more `outEdges`, and each edge has a `length` attribute (in meters)
    that is used as the edge cost. The output of every planning method is a
    list of node indices that can be passed directly to
    `roadmap.generate_path(...)` to obtain a 2xN waypoint array.

    Attributes:
        roadmap (SDCSRoadMap): the roadmap instance to plan on.
        nodeIdx (dict): maps RoadMapNode -> integer index, for O(1) lookup
            during path reconstruction.
        nodesExpanded (int): number of nodes popped from the open set in
            the most recent planning call. Useful for comparing Dijkstra
            with A*.
    """

    def __init__(self, roadmap):
        self.roadmap = roadmap
        self.nodeIdx = {node: i for i, node in enumerate(roadmap.nodes)}
        self.nodesExpanded = 0

    # ============== Roadmap Inspection ====================
    # Helper utilities for inspection. These are provided to students.
    def num_nodes(self):
        return len(self.roadmap.nodes)

    def node_position(self, nodeID):
        """Return the (x, y, heading) of node `nodeID`."""
        return tuple(self.roadmap.nodes[nodeID].pose[:, 0])

    def outgoing_edges(self, nodeID):
        """Return a list of (neighborID, edgeLength) tuples for outgoing
        edges of `nodeID`.
        """
        node = self.roadmap.nodes[nodeID]
        return [(self.nodeIdx[e.toNode], e.length) for e in node.outEdges]

    # ============== Part A2 - Dijkstra's Algorithm =================
    def dijkstra(self, startID, goalID):
        """Compute the lowest-cost node sequence from `startID` to `goalID`
        using Dijkstra's algorithm on the directed roadmap graph.

        Args:
            startID (int): index of the start node.
            goalID  (int): index of the goal node.

        Returns:
            (nodeSequence, totalCost) where nodeSequence is a list of node
            indices including both endpoints, and totalCost is the sum of
            edge lengths along the path (meters). Returns (None, inf) if
            the goal is not reachable from the start.
        """
        nodes = self.roadmap.nodes
        if startID == goalID:
            self.nodesExpanded = 0
            return [startID], 0.0
        startNode, goalNode = nodes[startID], nodes[goalID]
        g = {n: float('inf') for n in nodes}
        g[startNode] = 0
        parent = {n: None for n in nodes}

        pq = []
        counter = 0
        heapq.heappush(pq, (0.0, counter, startNode)); counter += 1
        closed = set()
        expanded = []
        
        while pq:
            gu, _, u = heapq.heappop(pq)
            if u in closed:
                continue
            closed.add(u)
            expanded.append(self.nodeIdx[u])
            if u is goalNode:
                self.nodesExpanded = len(expanded)
                # self.expandedList = expanded
                return self._reconstruct(parent, goalNode), float(gu)
            for edge in u.outEdges:
                v = edge.toNode
                if v in closed:
                    continue
                tentative = gu + edge.length
                if tentative < g[v]:
                    g[v] = tentative
                    parent[v] = u
                    heapq.heappush(pq, (tentative, counter, v))
                    counter += 1

        self.nodesExpanded = len(expanded)
        # self.expandedList = expanded

        return None, float('inf')

    # ============== Part B1 - A* Heuristic =========================
    def heuristic(self, nodeA, nodeB):
        """Admissible heuristic estimate of the remaining cost from
        `nodeA` to `nodeB`. We use straight-line Euclidean distance in
        the (x, y) plane, which lower-bounds any feasible path through
        the road network.

        Args:
            nodeA, nodeB (RoadMapNode): nodes to compare.

        Returns:
            float: heuristic value h(nodeA, nodeB), in meters.
        """
        # YOUR CODE HERE
        return np.hypot(nodeA.pose[0] - nodeB.pose[0],
                        nodeA.pose[1] - nodeB.pose[1])
        
  

    # ============== Part B2 / C1 - A* Search ============================
    def astar(self, startID, goalID, turn_penalty=0.0):
        """Compute the lowest-cost node sequence using A* search with the
        Euclidean heuristic.

        Optionally adds a turn-penalty term to the per-edge cost. When
        `turn_penalty > 0`, each edge incurs an extra cost
            turn_penalty * |wrap_to_pi(edge.toNode.heading - prev_heading)|
        which biases the search toward routes with fewer / smaller
        heading changes. Setting turn_penalty = 0 recovers standard A*.

        Args:
            startID (int): index of the start node.
            goalID  (int): index of the goal node.
            turn_penalty (float): per-radian cost on heading changes.

        Returns:
            (nodeSequence, totalCost). Returns (None, inf) if unreachable.
        """
        nodes = self.roadmap.nodes
        if startID == goalID:
            self.nodesExpanded = 0
            return [startID], 0.0
        # YOUR CODE HERE (ignore turn_penalty for Part C2)
 
        startNode, goalNode = nodes[startID], nodes[goalID]
        g = {n: float('inf') for n in nodes}
        g[startNode] = 0.0
        parent = {n: None for n in nodes}
        hasParent = {n: False for n in nodes}     # for turn-penalty check

        pq = []
        counter = 0
        heapq.heappush(pq,
            (self.heuristic(startNode, goalNode), counter, startNode))
        counter += 1
        closed = set()
        expanded = []

        while pq:
            _, _, u = heapq.heappop(pq)
            if u in closed:
                continue
            closed.add(u)
            expanded.append(self.nodeIdx[u])
            if u is goalNode:
                self.nodesExpanded = len(expanded)
                # self.expandedList = expanded
                return self._reconstruct(parent, goalNode), float(g[u])
            heading_u = u.pose[2]
            for edge in u.outEdges:
                v = edge.toNode
                if v in closed:
                    continue
                step = edge.length
                # Add turn cost when leaving an interior node (not the start).
                if turn_penalty > 0.0 and hasParent[u]:
                    dpsi = wrap_to_pi(v.pose[2] - heading_u)
                    step += turn_penalty * abs(dpsi)
                tentative = g[u] + step
                if tentative < g[v]:
                    g[v] = tentative
                    parent[v] = u
                    hasParent[v] = True
                    f = tentative + self.heuristic(v, goalNode)
                    heapq.heappush(pq, (f, counter, v))
                    counter += 1

        self.nodesExpanded = len(expanded)
        # self.expandedList = expanded
        return None, float('inf')

    # ============== Helpers (provided) =============================
    def _reconstruct(self, cameFrom, goalNode):
        """Walk the cameFrom map back to the start to produce a list of
        node indices in forward order.
        """
        seq = [self.nodeIdx[goalNode]]
        node = goalNode
        while cameFrom[node] is not None:
            node = cameFrom[node]
            seq.append(self.nodeIdx[node])
        seq.reverse()
        return seq

    def plan(self, startID, goalID, plannerType='astar', turn_penalty=0.0):
        """Dispatcher: run the requested planner."""
        t0 = time.time()
        if plannerType == 'dijkstra':
            seq, cost = self.dijkstra(startID, goalID)
        elif plannerType == 'astar':
            seq, cost = self.astar(startID, goalID, turn_penalty=0.0)
        elif plannerType == 'astar_turn_penalty':
            seq, cost = self.astar(startID, goalID, turn_penalty=turn_penalty)
        else:
            raise ValueError(
                "plannerType must be 'dijkstra', 'astar', or "
                "'astar_turn_penalty', got {!r}".format(plannerType)
            )
        runtime = time.time() - t0
        return seq, cost, runtime

#endregion


#region : Visualization

def visualize_plan(roadmap, nodeSequence, title="Planned path"):
    """Overlay the planned node sequence and waypoint path on the roadmap.

    Returns the matplotlib pyplot module so the caller can decide when to
    show / save the figure.
    """
    plt_obj, ax = roadmap.display()
    ax.set_title(title)

    if nodeSequence is None or len(nodeSequence) < 2:
        print("[visualize_plan] No path to visualize.")
        return plt_obj

    # Generate the dense waypoint sequence and overlay it.
    waypoints = roadmap.generate_path(nodeSequence)
    if waypoints is not None and waypoints.size > 0:
        ax.plot(
            waypoints[0, :], waypoints[1, :],
            color='blue', linewidth=3, alpha=0.6, zorder=2,
            label='Planned path'
        )

    # Highlight the planned nodes themselves.
    for k, nid in enumerate(nodeSequence):
        x, y, _ = roadmap.nodes[nid].pose[:, 0]
        if k == 0:
            ax.plot(x, y, marker='*', markersize=18, color='lime',
                    markeredgecolor='black', zorder=4, label='Start')
        elif k == len(nodeSequence) - 1:
            ax.plot(x, y, marker='X', markersize=14, color='magenta',
                    markeredgecolor='black', zorder=4, label='Goal')
        else:
            ax.plot(x, y, marker='o', markersize=10,
                    markerfacecolor='blue', markeredgecolor='black',
                    zorder=3)

    ax.legend(loc='best')
    return plt_obj

#endregion


# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

#region : Initial setup -- build roadmap and run planner

print('=' * 60)
print('Lab 6: Global Motion Planning on the SDCS Roadmap')
print('=' * 60)

roadmap = SDCSRoadMap(leftHandTraffic=leftHandTraffic)
planner = PathPlanner(roadmap)

print('Roadmap loaded: {} nodes, {} edges.'.format(
    len(roadmap.nodes), len(roadmap.edges)
))
print('Start node: {}  ->  Goal node: {}'.format(startNode, goalNode))
print('Planner: {}'.format(plannerType))

# Run the chosen planner.
nodeSequence, totalCost, plannerRuntime = planner.plan(
    startID=startNode,
    goalID=goalNode,
    plannerType=plannerType,
    turn_penalty=turnPenalty,
)

if nodeSequence is None:
    print('\n[ERROR] No path found from {} to {}.'.format(startNode, goalNode))
    print('        Check that the goal is reachable on the directed graph.')
    raise SystemExit(1)

print('\nResult:')
print('  Node sequence : {}'.format(nodeSequence))
print('  Total cost    : {:.3f} m'.format(totalCost))
print('  Nodes expanded: {}'.format(planner.nodesExpanded))
print('  Runtime       : {:.2f} ms'.format(1000 * plannerRuntime))

# Optional: cross-check against Quanser's built-in A* in the RoadMap class.
# This is purely a verification aid; it does NOT affect what we drive.
try:
    refPath = roadmap.find_shortest_path(startNode, goalNode)
    if refPath is not None:
        print('  Built-in A* returned a path of {} waypoints '
              '(reference only).'.format(refPath.shape[1]))
except Exception as e:
    print('  Built-in A* check failed: {}'.format(e))

# Generate the dense waypoint sequence used by the Stanley controller.
waypointSequence = roadmap.generate_path(nodeSequence)
initialPose = roadmap.get_node_pose(nodeSequence[0]).squeeze()

# Visualize the plan before driving. The student can close the figure to
# proceed to the driving phase (or skip driving entirely with enableDriving).
visualize_plan(
    roadmap,
    nodeSequence,
    title='Lab 6 plan ({}): start={}, goal={}, cost={:.2f} m'.format(
        plannerType, startNode, goalNode, totalCost
    ),
)
print('\nClose the plan figure to continue.')
plt.show(block=True)

#endregion


#region : Drive setup (only if enableDriving == True)

if enableDriving:
    if not IS_PHYSICAL_QCAR:
        import qlabs_setup
        qlabs_setup.setup(
            initialPosition=[initialPose[0], initialPose[1], 0],
            initialOrientation=[0, 0, initialPose[2]]
        )
        calibrate = False
    else:
        calibrate = 'y' in input('do you want to recalibrate? (y/n) ')

    # Calibration pose for the artificial GPS. Match Lab 3 convention.
    calibrationPose = [0, 0, -np.pi/2]

global KILL_THREAD
KILL_THREAD = False
def sig_handler(*args):
    global KILL_THREAD
    KILL_THREAD = True
signal.signal(signal.SIGINT, sig_handler)

#endregion


# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

#region : Speed and Steering Controllers (reused from Lab 3)
# These are deliberately identical to vehicle_control.py so that students
# focus on the planning content, not on re-implementing controllers.

class SpeedController:
    def __init__(self, kp=0, ki=0):
        self.maxThrottle = 0.3
        self.kp = kp
        self.ki = ki
        self.ei = 0

    def update(self, v, v_ref, dt):
        e = v_ref - v
        self.ei += dt * e
        return np.clip(
            self.kp * e + self.ki * self.ei,
            -self.maxThrottle,
            self.maxThrottle,
        )


class SteeringController:
    def __init__(self, waypoints, k=1, cyclic=False):
        self.maxSteeringAngle = np.pi/6
        self.wp = waypoints
        self.N = len(waypoints[0, :])
        self.wpi = 0
        self.k = k
        self.cyclic = cyclic
        self.p_ref = (0, 0)
        self.th_ref = 0

    def update(self, p, th, speed):
        # Note: modulus by self.N (not self.N-1) so that wp[N-1] is reachable
        # as the destination of the final segment. The original Lab 3 form
        # used mod (N-1), which was harmless on closed-loop paths because
        # wp[0] == wp[N-1] there, but skipped the final segment on open
        # paths and silently routed the car back toward wp[0].
        wp_1 = self.wp[:, np.mod(self.wpi,     self.N)]
        wp_2 = self.wp[:, np.mod(self.wpi + 1, self.N)]

        v = wp_2 - wp_1
        v_mag = np.linalg.norm(v)
        try:
            v_uv = v / v_mag
        except ZeroDivisionError:
            return 0

        tangent = np.arctan2(v_uv[1], v_uv[0])
        s = np.dot(p - wp_1, v_uv)

        if s >= v_mag:
            if self.cyclic or self.wpi < self.N - 2:
                self.wpi += 1

        ep = wp_1 + v_uv * s
        ct = ep - p
        dir_ = wrap_to_pi(np.arctan2(ct[1], ct[0]) - tangent)

        ect = np.linalg.norm(ct) * np.sign(dir_)
        psi = wrap_to_pi(tangent - th)

        self.p_ref = ep
        self.th_ref = tangent

        return np.clip(
            wrap_to_pi(psi + np.arctan2(self.k * ect, speed)),
            -self.maxSteeringAngle,
            self.maxSteeringAngle,
        )

#endregion


#region : Control loop (reused / lightly adapted from Lab 3)

def controlLoop():
    global KILL_THREAD
    u = 0
    delta = 0
    countMax = controllerUpdateRate / 10
    count = 0

    speedController    = SpeedController(kp=K_p, ki=K_i)
    steeringController = SteeringController(
        waypoints=waypointSequence, k=K_stanley, cyclic=False,
    )

    qcar = QCar(readMode=1, frequency=controllerUpdateRate)
    ekf  = QCarEKF(x_0=initialPose)
    gps  = QCarGPS(initialPose=calibrationPose, calibrate=calibrate)

    # Goal-stop state. goalPos is the (x, y) of the final waypoint that the
    # steering controller is tracking; goalReached latches True once the
    # QCar reaches the last segment AND is physically close to goalPos, and
    # never resets so the car holds position even if the EKF later drifts.
    # Using wpi (path progress) rather than raw distance avoids false
    # triggers when the map geometry takes the path physically near the
    # goal point during earlier segments.
    N_wp = waypointSequence.shape[1]
    last_segment_index = N_wp - 2
    goalPos = np.asarray(waypointSequence[:, -1], dtype=float)
    goalReached = False

    with qcar, gps:
        t0 = time.time()
        t = 0
        while (t < tf + startDelay) and (not KILL_THREAD):
            tp = t
            t  = time.time() - t0
            dt = t - tp

            # --- sense + estimate ---
            qcar.read()
            if gps.readGPS():
                y_gps = np.array([
                    gps.position[0],
                    gps.position[1],
                    gps.orientation[2],
                ])
                ekf.update([qcar.motorTach, delta], dt, y_gps,
                           qcar.gyroscope[2])
            else:
                ekf.update([qcar.motorTach, delta], dt, None,
                           qcar.gyroscope[2])

            x  = ekf.x_hat[0, 0]
            y  = ekf.x_hat[1, 0]
            th = ekf.x_hat[2, 0]
            p = (np.array([x, y])
                 + np.array([np.cos(th), np.sin(th)]) * 0.2)
            v = qcar.motorTach

            # --- detect goal: on last path segment AND close to wp[-1] ---
            on_last_segment = (steeringController.wpi >= last_segment_index)
            distToGoal = float(np.linalg.norm(p - goalPos))
            if (not goalReached) and on_last_segment \
                    and (distToGoal < goalStopDistance):
                goalReached = True
                print('Goal reached at t = {:.2f} s '
                      '(d = {:.3f} m). Stopping QCar.'
                      .format(t - startDelay, distToGoal))

            # --- act ---
            if t < startDelay:
                u, delta = 0, 0
            elif goalReached:
                # Active braking via PI; freeze steering at zero.
                u = speedController.update(v, 0.0, dt)
                delta = 0
            else:
                # Normal driving: constant v_ref. No ramp -- a slow ramp
                # would expose the Stanley low-speed singularity
                # (arctan2(k*ect, v) saturates as v -> 0) and cause the
                # vehicle to oscillate before reaching the goal.
                u = speedController.update(v, v_ref, dt)
                delta = steeringController.update(p, th, v)

            qcar.write(u, delta)

            # --- scopes ---
            count += 1
            if count >= countMax and t > startDelay:
                t_plot = t - startDelay
                speedScope.axes[0].sample(t_plot, [v, v_ref])
                speedScope.axes[1].sample(t_plot, [v_ref - v])
                speedScope.axes[2].sample(t_plot, [u])

                steeringScope.axes[4].sample(t_plot, [[p[0], p[1]]])
                p[0] = ekf.x_hat[0, 0]
                p[1] = ekf.x_hat[1, 0]
                x_ref = gps.position[0]
                y_ref = gps.position[1]
                th_ref = gps.orientation[2]
                steeringScope.axes[0].sample(t_plot, [p[0], x_ref])
                steeringScope.axes[1].sample(t_plot, [p[1], y_ref])
                steeringScope.axes[2].sample(t_plot, [th, th_ref])
                steeringScope.axes[3].sample(t_plot, [delta])
                arrow.setPos(p[0], p[1])
                arrow.setStyle(angle=180 - th * 180 / np.pi)
                count = 0

#endregion


#region : Main entry: scope setup and run

if __name__ == '__main__':
    if not enableDriving:
        print('\nenableDriving = False  ->  planning-only mode. Exiting.')
        raise SystemExit(0)

    fps = 10 if IS_PHYSICAL_QCAR else 30

    # Speed-control scope
    speedScope = MultiScope(rows=3, cols=1, title='Speed Control', fps=fps)
    speedScope.addAxis(row=0, col=0, timeWindow=tf,
                       yLabel='Vehicle Speed [m/s]', yLim=(0, 1))
    speedScope.axes[0].attachSignal(name='v_meas', width=2)
    speedScope.axes[0].attachSignal(name='v_ref')
    speedScope.addAxis(row=1, col=0, timeWindow=tf,
                       yLabel='Speed Error [m/s]', yLim=(-0.5, 0.5))
    speedScope.axes[1].attachSignal()
    speedScope.addAxis(row=2, col=0, timeWindow=tf,
                       xLabel='Time [s]', yLabel='Throttle [%]',
                       yLim=(-0.3, 0.3))
    speedScope.axes[2].attachSignal()

    # Steering-control scope (with planned path overlay on the SDCS map)
    steeringScope = MultiScope(rows=4, cols=2, title='Steering Control',
                               fps=fps)
    steeringScope.addAxis(row=0, col=0, timeWindow=tf,
                          yLabel='x [m]', yLim=(-2.5, 2.5))
    steeringScope.axes[0].attachSignal(name='x_meas')
    steeringScope.axes[0].attachSignal(name='x_ref')
    steeringScope.addAxis(row=1, col=0, timeWindow=tf,
                          yLabel='y [m]', yLim=(-1, 5))
    steeringScope.axes[1].attachSignal(name='y_meas')
    steeringScope.axes[1].attachSignal(name='y_ref')
    steeringScope.addAxis(row=2, col=0, timeWindow=tf,
                          yLabel='Heading [rad]', yLim=(-3.5, 3.5))
    steeringScope.axes[2].attachSignal(name='th_meas')
    steeringScope.axes[2].attachSignal(name='th_ref')
    steeringScope.addAxis(row=3, col=0, timeWindow=tf,
                          yLabel='Steering [rad]', yLim=(-0.6, 0.6))
    steeringScope.axes[3].attachSignal()
    steeringScope.axes[3].xLabel = 'Time [s]'

    steeringScope.addXYAxis(row=0, col=1, rowSpan=4,
                            xLabel='x [m]', yLabel='y [m]',
                            xLim=(-2.5, 2.5), yLim=(-1, 5))
    im = cv2.imread(images.SDCS_CITYSCAPE, cv2.IMREAD_GRAYSCALE)
    steeringScope.axes[4].attachImage(scale=(-0.002035, 0.002035),
                                      offset=(1125, 2365),
                                      rotation=180, levels=(0, 255))
    steeringScope.axes[4].images[0].setImage(image=im)

    referencePath = pg.PlotDataItem(
        pen={'color': (85, 168, 104), 'width': 2}, name='Reference')
    steeringScope.axes[4].plot.addItem(referencePath)
    referencePath.setData(waypointSequence[0, :], waypointSequence[1, :])
    steeringScope.axes[4].attachSignal(name='Estimated', width=2)

    arrow = pg.ArrowItem(
        angle=180, tipAngle=60, headLen=10, tailLen=10, tailWidth=5,
        pen={'color': 'w', 'fillColor': [196, 78, 82], 'width': 1},
        brush=[196, 78, 82],
    )
    arrow.setPos(initialPose[0], initialPose[1])
    steeringScope.axes[4].plot.addItem(arrow)

    # Run the control loop in a thread so MultiScope can refresh in main.
    controlThread = Thread(target=controlLoop)
    controlThread.start()
    try:
        while controlThread.is_alive() and not KILL_THREAD:
            MultiScope.refreshAll()
            time.sleep(0.01)
    finally:
        KILL_THREAD = True

    if not IS_PHYSICAL_QCAR:
        qlabs_setup.terminate()
    input('Experiment complete. Press any key to exit...')

#endregion
