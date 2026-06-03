# (c) 2026 S. Farzan, Electrical Engineering Department, Cal Poly
# EE 471 (SP26): Planning and Control for Autonomous Vehicles
"""
lab07_ee471.py

Script for EE 471 Final Lab: Perception-Aware Driving Under Traffic Law.

This lab extends Lab 6's global motion planner with a perception
pipeline that detects stop signs and traffic lights in the front camera
feed, and a behavior FSM that gates the speed reference so the vehicle
respects traffic policies along the planned path.

Students implement the following pieces and integrate them with the
existing Lab 6 control loop:
    Part A1 : HSV color thresholding              (Perception.color_threshold)
    Part A2 : Apply a mask to an image            (Perception.mask_img)
    Part B1 : Detect stop sign and traffic light  (Perception.detect)
    Part B2 : Distance estimation from depth      (Perception.find_distance)
    Part C1 : Traffic-light state classification  (Perception.classify_light)
    Part D1 : Stop-sign behavior FSM              (StopSignFSM.update)
    Part D2 : Traffic-light behavior FSM          (TrafficLightFSM.update)
    Part D3 : Apply FSM output to v_ref           (in controlLoop)

The Lab 6 PathPlanner (Dijkstra, A*) and Lab 3 Stanley + PI controllers
are reused without modification.

This is the STUDENT STARTER file. Each section you must implement is
marked with a "# YOUR CODE HERE (Part X)" comment and a placeholder
return value. Replace the placeholder with your implementation;
the function signatures, docstrings, and all provided helper code
must stay as they are. See the lab manual for the algorithm for
each part.
"""
# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

#region : File Description and Imports
import os
import signal
import heapq
import time
import threading
import numpy as np
from threading import Thread, Lock
import matplotlib.pyplot as plt

import cv2
import pyqtgraph as pg

from pal.products.qcar import QCar, QCarGPS, QCarRealSense, IS_PHYSICAL_QCAR
from pal.utilities.scope import MultiScope
from pal.utilities.math import wrap_to_pi
from hal.content.qcar_functions import QCarEKF
from hal.products.mats import SDCSRoadMap
from pit.YOLO.utils import QCar2DepthAligned
import pal.resources.images as images

#endregion


# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

#region : Experiment Configuration

# ===== Run Mode
# - mode: 'plan_only'       -> just plan and show the figure (Lab 6 behavior)
#         'perception_only' -> run camera + perception + display window only
#                              (useful for Part A / B / C tuning, no driving)
#         'drive'           -> full integration: plan + perception + driving
mode = 'perception_only'

# ===== Planner Parameters (Lab 6)
# - startNode, goalNode: roadmap node indices for the planned route.
# - plannerType: 'dijkstra' | 'astar' | 'astar_turn_penalty'.
# - leftHandTraffic: must match Lab 3 / Lab 5 / Lab 6 (False for SDCS).
startNode = 0
goalNode = 10
plannerType = 'astar'
turnPenalty = 0.0
leftHandTraffic = False

# ===== Driving Parameters (Lab 6 defaults)
tf = 60
startDelay = 1
controllerUpdateRate = 100
v_ref = 0.51 # 0.75
K_p = 0.08
K_i = 1.1
K_stanley = 0.8
goalStopDistance = 0.5

#================ Part A1 - Perception Configuration ================
# HSV bounds for the three colors of interest. Each bound is a 3-tuple
# (H, S, V) with H in [0, 180] and S, V in [0, 255] (OpenCV convention).
#
# Red wraps around in HSV (it appears near H=0 AND H=180), so we use
# TWO ranges for red and OR the resulting masks together inside
# `Perception.detect`.
#
# Students will tune these values in `perception_only` mode using the
# trackbars window. Replace the placeholders below with the values you
# record. The reference values that follow are tuned for the virtual
# SDCS Cityscape; values for the physical car will differ.
# VIRTUAL:
# RED_LOWER_1 = (0, 90, 70)   # red, low-hue half
# RED_UPPER_1 = (12, 255, 255)
# RED_LOWER_2 = (150, 90, 70)   # red, high-hue half
# RED_UPPER_2 = (180, 255, 255)

# YELLOW_LOWER = (18, 55, 80)  # yellow (traffic-light body & yellow light)
# YELLOW_UPPER = (40, 255, 255)

# GREEN_LOWER = (40, 70, 60)   # green (traffic light "go" lamp)
# GREEN_UPPER = (95, 255, 255)

# PHSYICAL:
RED_LOWER_1 = (0, 170, 80) 
RED_UPPER_1 = (6, 255, 255) 
RED_LOWER_2 = (165, 170, 80) 
RED_UPPER_2 = (180, 255, 255) 

YELLOW_LOWER = (2, 100, 50) 
YELLOW_UPPER =(32, 255, 255) 

GREEN_LOWER = (47, 80, 80) 
GREEN_UPPER = (64, 255, 255) 

 

# Minimum contour area (in pixels) for a detection to count. Filters out
# noise blobs from background clutter.
MIN_BLOB_AREA = 200

# Distance Scaling
SCALE = 1 if IS_PHYSICAL_QCAR else 10

#================ Part D - Behavior Configuration ================
# - stopTriggerDistance: stop-sign detection within this distance (m)
#       engages the stop-sign FSM.
# - stopHoldDuration: how long (s) the QCar must remain stopped at a
#       stop sign before it is allowed to proceed.
# - signCooldownDuration: how long (s) the sign must be absent from the
#       camera before the FSM is allowed to re-trigger on a new sign.
# - lightTriggerDistance: traffic-light detection within this distance
#       (m) engages the traffic-light FSM.
# - approachSpeed: speed (m/s) commanded while approaching a stop sign
#       before the car is fully stopped. Set equal to v_ref to disable
#       the approach phase.
stopTriggerDistance   = 1.0*SCALE
stopHoldDuration      = 3.0
signCooldownDuration  = 2.0
lightTriggerDistance  = 1.25*SCALE
approachSpeed         = 0.025

# Perception loop rate (Hz). 10-15 Hz is plenty for tabletop speeds.
perceptionRate = 15.0

#endregion


# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

#region : Path Planner (REUSED FROM LAB 6 -- NO STUDENT WORK HERE)

class PathPlanner:
    """Graph-search-based motion planner over the SDCS roadmap. Identical
    to the Lab 6 solution: Dijkstra and A* with an Euclidean heuristic.
    """

    def __init__(self, roadmap):
        self.roadmap = roadmap
        self.nodeIdx = {node: i for i, node in enumerate(roadmap.nodes)}
        self.nodesExpanded = 0

    def num_nodes(self):
        return len(self.roadmap.nodes)

    def node_position(self, nodeID):
        return tuple(self.roadmap.nodes[nodeID].pose[:, 0])

    def outgoing_edges(self, nodeID):
        node = self.roadmap.nodes[nodeID]
        return [(self.nodeIdx[e.toNode], e.length) for e in node.outEdges]

    def dijkstra(self, startID, goalID):
        nodes = self.roadmap.nodes
        if startID == goalID:
            self.nodesExpanded = 0
            return [startID], 0.0
        startN, goalN = nodes[startID], nodes[goalID]
        gScore = {n: float('inf') for n in nodes}
        gScore[startN] = 0.0
        cameFrom = {n: None for n in nodes}
        openSet, counter, closed, expanded = [], 0, set(), 0
        heapq.heappush(openSet, (0.0, counter, startN)); counter += 1
        while openSet:
            g, _, current = heapq.heappop(openSet)
            if current in closed:
                continue
            closed.add(current); expanded += 1
            if current is goalN:
                self.nodesExpanded = expanded
                return self._reconstruct(cameFrom, goalN), float(g)
            for edge in current.outEdges:
                neighbor = edge.toNode
                if neighbor in closed or edge.length is None:
                    continue
                tentative = g + edge.length
                if tentative < gScore[neighbor]:
                    gScore[neighbor] = tentative
                    cameFrom[neighbor] = current
                    heapq.heappush(openSet, (tentative, counter, neighbor))
                    counter += 1
        self.nodesExpanded = expanded
        return None, float('inf')

    def heuristic(self, nodeA, nodeB):
        return float(np.linalg.norm(nodeA.pose[:2, 0] - nodeB.pose[:2, 0]))

    def astar(self, startID, goalID, turn_penalty=0.0):
        nodes = self.roadmap.nodes
        if startID == goalID:
            self.nodesExpanded = 0
            return [startID], 0.0
        startN, goalN = nodes[startID], nodes[goalID]
        gScore = {n: float('inf') for n in nodes}
        gScore[startN] = 0.0
        cameFrom = {n: None for n in nodes}
        cameFromEdge = {n: None for n in nodes}
        openSet, counter, closed, expanded = [], 0, set(), 0
        h0 = self.heuristic(startN, goalN)
        heapq.heappush(openSet, (h0, counter, startN)); counter += 1
        while openSet:
            _, _, current = heapq.heappop(openSet)
            if current in closed:
                continue
            closed.add(current); expanded += 1
            if current is goalN:
                self.nodesExpanded = expanded
                return self._reconstruct(cameFrom, goalN), float(gScore[goalN])
            for edge in current.outEdges:
                neighbor = edge.toNode
                if neighbor in closed or edge.length is None:
                    continue
                stepCost = edge.length
                if turn_penalty > 0.0 and cameFromEdge[current] is not None:
                    prevEdge = cameFromEdge[current]
                    dPsi = wrap_to_pi(edge.toNode.pose[2, 0]
                                      - prevEdge.toNode.pose[2, 0])
                    stepCost += turn_penalty * abs(dPsi)
                tentative = gScore[current] + stepCost
                if tentative < gScore[neighbor]:
                    gScore[neighbor] = tentative
                    cameFrom[neighbor] = current
                    cameFromEdge[neighbor] = edge
                    f = tentative + self.heuristic(neighbor, goalN)
                    heapq.heappush(openSet, (f, counter, neighbor))
                    counter += 1
        self.nodesExpanded = expanded
        return None, float('inf')

    def _reconstruct(self, cameFrom, goalNode):
        seq = [self.nodeIdx[goalNode]]
        node = goalNode
        while cameFrom[node] is not None:
            node = cameFrom[node]
            seq.append(self.nodeIdx[node])
        seq.reverse()
        return seq

    def plan(self, startID, goalID, plannerType='astar', turn_penalty=0.0):
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
        return seq, cost, time.time() - t0

#endregion


# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

#region : Perception

class Perception:
    """Classical color-based perception for stop signs and traffic lights.

    The detector consumes an aligned RGB + depth frame from
    QCar2DepthAligned and returns, for each known object class, the
    largest matching color blob along with its bounding box, binary
    mask, estimated distance, and (for traffic lights) the lamp color
    that is currently lit.

    All methods that students implement are clearly marked with a
    `# YOUR CODE HERE (Part X)` comment.
    """

    # Class names returned by `detect`.
    STOP_SIGN     = 'stop_sign'
    TRAFFIC_LIGHT = 'traffic_light'

    def __init__(self):
        # Tunable HSV bounds, pulled from the top-of-file configuration
        # so students can adjust them in one place.
        self.red_lo_1 = np.array(RED_LOWER_1, dtype=np.uint8)
        self.red_hi_1 = np.array(RED_UPPER_1, dtype=np.uint8)
        self.red_lo_2 = np.array(RED_LOWER_2, dtype=np.uint8)
        self.red_hi_2 = np.array(RED_UPPER_2, dtype=np.uint8)
        self.yel_lo   = np.array(YELLOW_LOWER, dtype=np.uint8)
        self.yel_hi   = np.array(YELLOW_UPPER, dtype=np.uint8)
        self.grn_lo   = np.array(GREEN_LOWER,  dtype=np.uint8)
        self.grn_hi   = np.array(GREEN_UPPER,  dtype=np.uint8)
        self.min_area = MIN_BLOB_AREA

    # =================== Part A1 - HSV Thresholding =====================
    def color_threshold(self, img_hsv, lower, upper):
        """Produce a binary mask of pixels that fall inside the inclusive
        HSV box [lower, upper].

        Args:
            img_hsv (ndarray): HxWx3 image in HSV.
            lower (array-like): length-3 lower bound (H, S, V).
            upper (array-like): length-3 upper bound (H, S, V).

        Returns:
            ndarray: HxW uint8 binary mask with values in {0, 255}.

        Implementation hint: a single call to `cv2.inRange` is enough.
        """
        # YOUR CODE HERE (Part A1)
        return cv2.inRange(img_hsv, lower, upper)

    # ===================== Part A2 - Mask Image ========================
    def mask_img(self, img, mask):
        """Apply a binary mask to a BGR image (used in the tuning UI).

        Args:
            img (ndarray): HxWx3 BGR image.
            mask (ndarray): HxW uint8 mask.

        Returns:
            ndarray: HxWx3 image with non-mask pixels set to 0.

        Implementation hint: see `cv2.bitwise_and(img, img, mask=mask)`.
        """
        # YOUR CODE HERE (Part A2)
        return cv2.bitwise_and(img, img, mask=mask)

    # ============== Helper: morphological clean-up (provided) ===========
    def _clean_mask(self, mask):
        """Standard open-then-close to remove speckle and seal small
        holes. Provided so students don't burn time on it.
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask

    # ============== Helper: largest contour (provided) =================
    def _largest_blob(self, mask, min_aspect=None, max_y_frac=None):
        """Return (bbox, blob_mask) for the largest contour in `mask`
        that ALSO passes the optional shape/position filters, or
        (None, None) if nothing qualifies.

        Args:
            mask (ndarray):    HxW uint8 binary mask.
            min_aspect (float, optional): require height/width >= this.
                Set to ~1.2 for traffic lights to reject wide horizontal
                blobs (yellow lane stripes appear wider than tall in
                the camera image).
            max_y_frac (float, optional): require bbox center_y / image
                height <= this. Set to ~0.7 for traffic lights to
                reject blobs in the lower portion of the frame
                (anything on the ground, including lane lines).

        Without the optional filters this behaves exactly as before:
        return the single largest contour above `self.min_area`.
        """
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None, None

        H = mask.shape[0]

        def passes(c):
            if cv2.contourArea(c) < self.min_area:
                return False
            x, y, w, h = cv2.boundingRect(c)
            if min_aspect is not None and h / max(1, w) < min_aspect:
                return False
            if max_y_frac is not None and (y + 0.5 * h) / H > max_y_frac:
                return False
            return True

        valid = [c for c in contours if passes(c)]
        if not valid:
            return None, None
        c = max(valid, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        blob_mask = np.zeros(mask.shape[:2], dtype=np.uint8)
        cv2.drawContours(blob_mask, [c], -1, 255, thickness=-1)
        return (int(x), int(y), int(w), int(h)), blob_mask

    # ====================== Part B1 - Detection ========================
    def detect(self, img_bgr):
        """Detect a stop sign and a traffic light in `img_bgr`.

        Approach:
            1. Convert to HSV.
            2. Build a RED mask = OR of the two red HSV ranges (red
               wraps around in hue).
            3. Build a YELLOW mask for the traffic-light body. Use the
               yellow body for the bbox so the detection works regardless
               of which lamp is currently lit. (The yellow lamp inside
               the housing is part of this same yellow region.)
            4. Morphologically clean both masks (use `_clean_mask`).
            5. Extract the largest blob from each mask (use
               `_largest_blob`).
            6. Return a dict with keys 'stop_sign' and 'traffic_light'.
               Each value is either None (not detected) or a dict
               {'bbox': (x, y, w, h), 'mask': HxW uint8}. Distance and
               (for the light) color are filled in by callers later --
               keep detection cheap and stateless.

        Args:
            img_bgr (ndarray): HxWx3 BGR image from QCar2DepthAligned.rgb.

        Returns:
            dict: {'stop_sign': dict_or_None, 'traffic_light': dict_or_None}
        """
        # YOUR CODE HERE (Part B1)
        detect_dict = {self.STOP_SIGN: None, self.TRAFFIC_LIGHT: None}

        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

        red_mask =  cv2.bitwise_or(self.color_threshold(hsv, RED_LOWER_1, RED_UPPER_1), self.color_threshold(hsv, RED_LOWER_2, RED_UPPER_2),)
        yellow_mask = self.color_threshold(hsv, YELLOW_LOWER, YELLOW_UPPER)

        red_mask_clean = self._clean_mask(red_mask)
        yellow_mask_clean = self._clean_mask(yellow_mask)
        red_bbox, red_blob = self._largest_blob(red_mask_clean)
        yellow_bbox, yellow_blob = self._largest_blob(yellow_mask_clean, min_aspect=1.2, max_y_frac=0.7)
        
        if red_bbox is not None:
            detect_dict[self.STOP_SIGN] = {'bbox': red_bbox, 'mask': red_blob}
        if yellow_bbox is not None:
            detect_dict[self.TRAFFIC_LIGHT] = {'bbox': yellow_bbox, 'mask': yellow_blob}

        # print(detect_dict)
        return detect_dict

    # ================= Part B2 - Distance Estimation ===================
    def find_distance(self, depth, mask):
        """Estimate the distance to an object given a depth image and
        the object's binary mask.

        Depth alignment is handled upstream: on the physical QCar the
        depth image is warped into the RGB frame (homography) before it
        reaches this function; in the virtual setup QCar2DepthAligned
        returns depth already aligned. So this function can assume the
        mask and depth are registered. The centroid-patch sampling
        below is kept as defense-in-depth against residual calibration
        error -- it samples only the most reliable interior region.

        Units: depth values are in whatever scale the active setup
        uses -- physical meters on hardware, virtual-world units in
        simulation (10x physical). All distance comparisons downstream
        use the same scale via the module-level SCALE constant, and the
        valid-range filter below scales the same way.

        Algorithm:
            1. Compute the mask CENTROID -- a point reliably on the
               object regardless of any residual misalignment.
            2. Sample depth in a small patch centered on the centroid,
               sized proportionally to the mask.
            3. Drop invalid depth values (NaN, ~0, and absurdly large).
            4. Return the MEDIAN of the patch.
            5. If no valid depth pixels remain, return float('inf').

        Args:
            depth (ndarray): HxW depth image (meters * SCALE).
            mask  (ndarray): HxW uint8 mask, non-zero on the object.

        Returns:
            float: distance (meters * SCALE), or inf if no valid pixels.
        """
        # YOUR CODE HERE (Part B2)
        if mask is None or cv2.countNonZero(mask) == 0:
            return float('inf')

        # Centroid of the mask (one robust point inside the object).
        M = cv2.moments(mask)
        if M['m00'] == 0:
            return float('inf')
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        # Size the sample patch relative to the mask's extent. About
        # 1/6 of the smaller bounding dimension on each side -> the
        # patch is ~1/3 the size of the mask, well inside the object.
        ys, xs = np.where(mask > 0)
        bbox_w = int(xs.max() - xs.min() + 1)
        bbox_h = int(ys.max() - ys.min() + 1)
        half_patch = max(5, min(30, min(bbox_w, bbox_h) // 6))

        H, W = depth.shape[:2]
        x0 = max(0, cx - half_patch)
        x1 = min(W, cx + half_patch + 1)
        y0 = max(0, cy - half_patch)
        y1 = min(H, cy + half_patch + 1)
        patch = depth[y0:y1, x0:x1]

        # Valid-range filter. The upper bound scales with SCALE so it is
        # a true sensor-range limit (10 m physical / 100 in virtual) and
        # never silently caps a trigger distance.
        vals = patch[np.isfinite(patch)]
        vals = vals[(vals > 0.05) & (vals < 10.0 * SCALE)]
        if vals.size == 0:
            return float('inf')
        return float(np.median(vals))
		# return float('inf')

    # ============== Part C1 - Traffic Light Classification =============
    def classify_light(self, img_bgr, bbox):
        """Decide which lamp (red, yellow, green, or none) is lit inside
        the given traffic-light bounding box.

        Approach: a traffic light's body is a tall yellow rectangle with
        three circular lamps stacked vertically -- red on top, yellow in
        the middle, green on the bottom. We don't know in advance which
        lamp is lit, but we know WHERE each lamp lives within the bbox.
        So sample HSV values in three vertical thirds of the bbox and
        report the third whose color matches its expected hue range AND
        is the brightest. If no lamp scores above a brightness threshold,
        return 'unknown'.

        Args:
            img_bgr (ndarray): HxWx3 BGR image.
            bbox (tuple): (x, y, w, h) bounding box of the traffic light.

        Returns:
            str: one of 'red', 'yellow', 'green', or 'unknown'.

        Implementation hints:
            - Slice three sub-images, one per third of the bbox height.
            - Convert each sub-image to HSV.
            - Build a mask using the expected color range for that third
              and compute the average V (value/brightness) of the masked
              pixels. Unlit lamps will have very low mean V.
            - Pick the third with the highest masked-V if that V exceeds
              ~100 (out of 255).
        """
        # YOUR CODE HERE (Part C1)

        if bbox is None:
            return 'unknown'

        x, y, w, h = bbox
        if w < 20 or h < 50:    # TUNEABLE: if the bbox is too small, it's probably a false detection.
            return 'unknown'

        min_on_brightness = 70
        traffic_light_roi = img_bgr[y:y+h, x:x+w]
        traffic_light_hsv = cv2.cvtColor(traffic_light_roi, cv2.COLOR_BGR2HSV)
        top_third = traffic_light_hsv[0:h//3, :]
        middle_third = traffic_light_hsv[h//3:2*h//3, :]
        bottom_third = traffic_light_hsv[2*h//3:h, :]

        thirds = {
            'red': top_third,
            'yellow': middle_third,
            'green': bottom_third,
        }

        # Each lamp region is scored using only its expected color mask.
        lamp_ranges = {
            'red': [
                (RED_LOWER_1, RED_UPPER_1),
                (RED_LOWER_2, RED_UPPER_2),
            ],
            'yellow': [(YELLOW_LOWER, YELLOW_UPPER)],
            'green': [(GREEN_LOWER, GREEN_UPPER)],
        }

        mean_brightness = {}
        for color, hsv_third in thirds.items():
            ranges = lamp_ranges[color]
            first_lower, first_upper = ranges[0]
            mask = self.color_threshold(hsv_third, first_lower, first_upper)
            for lower, upper in ranges[1:]:
                current_mask = self.color_threshold(hsv_third, lower, upper)
                mask = cv2.bitwise_or(mask, current_mask)

            if cv2.countNonZero(mask) == 0:
                mean_brightness[color] = 0.0
                continue

            value_channel = hsv_third[:, :, 2]
            mean_brightness[color] = cv2.mean(value_channel, mask=mask)[0]

        brightest_color, highest_brightness = max(
            mean_brightness.items(),
            key=lambda item: item[1],
        )
        if highest_brightness < min_on_brightness:
            return 'unknown'

        return brightest_color

    # ============== Helper: annotate frame (provided) =================
    def annotate(self, img_bgr, detections, distances, light_color):
        """Draw bounding boxes, labels, and distance/color text on a copy
        of the input image. Provided so the perception display window
        looks the same for every student.
        """
        out = img_bgr.copy()
        for name, det in detections.items():
            if det is None:
                continue
            x, y, w, h = det['bbox']
            d = distances.get(name, float('inf'))
            label = name
            if name == self.TRAFFIC_LIGHT and light_color != 'unknown':
                label = '{} ({})'.format(name, light_color)
            color = (0, 255, 255) if name == self.TRAFFIC_LIGHT else (0, 0, 255)
            cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
            txt = '{} {:.2f}m'.format(label, d) if np.isfinite(d) else label
            cv2.putText(out, txt, (x, max(0, y - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        return out

#endregion


# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

#region : Thread-safe perception state (PROVIDED)

class PerceptionState:
    """Shared state between the perception thread (producer) and the
    control loop (consumer). All access is guarded by a lock.

    Keys:
        stop_sign     : None or {'distance': float, 'bbox': tuple, 't': float}
        traffic_light : None or {'distance': float, 'bbox': tuple,
                                 'color': str, 't': float}
        last_frame    : latest annotated BGR frame, or None
    """

    def __init__(self):
        self._lock = Lock()
        self.stop_sign = None
        self.traffic_light = None
        self.last_frame = None
        self.frame_count = 0

    def update(self, stop_sign, traffic_light, frame):
        with self._lock:
            self.stop_sign = stop_sign
            self.traffic_light = traffic_light
            self.last_frame = frame
            self.frame_count += 1

    def snapshot(self):
        """Return a (stop_sign, traffic_light) tuple captured under the
        lock. Callers should NOT hold references to the live dicts.
        """
        with self._lock:
            return (
                dict(self.stop_sign) if self.stop_sign else None,
                dict(self.traffic_light) if self.traffic_light else None,
            )

    def get_frame(self):
        with self._lock:
            return None if self.last_frame is None else self.last_frame.copy()

#endregion


# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

#region : Perception thread (PROVIDED)

def perceptionLoop(state, stop_event):
    """Producer thread: continuously grab aligned RGB + depth frames,
    run the perception pipeline, and publish the latest result to
    `state`. Runs at `perceptionRate` Hz.

    Students should not need to edit this. The actual perception logic
    lives inside the methods of `Perception`, which is what they
    implement.
    """
    if IS_PHYSICAL_QCAR:
        cam = QCarRealSense(
            mode='RGB, Depth',
            frameWidthRGB=640,
            frameHeightRGB=480,
            frameRateRGB=30,
            frameWidthDepth=640,
            frameHeightDepth=480,
            frameRateDepth=15,
        )
    else:
        cam = QCar2DepthAligned()
    detector = Perception()
    dt_target = 1.0 / perceptionRate

    # depth_scale: empirical conversion from the raw 'PX' depth buffer
    # (uint8, 0-255) to meters on the PHYSICAL QCar. The PAL 'M' dataMode
    # relies on the RealSense SDK's get_meters(), which does not return
    # usable data on this camera interface, so we read 'PX' and convert
    # by hand. 5.5 was measured against known tabletop distances --
    # re-measure it if the camera or its firmware changes. Note the 'PX'
    # buffer is only 8-bit, so physical depth resolution is coarse
    # (~0.18 m per level).
    depth_scale = 5.5

    # depthToRgbH: 3x3 homography that warps the depth image into the
    # RGB image frame on the PHYSICAL QCar. The RealSense depth and RGB
    # sensors have different fields of view and a small baseline, so an
    # RGB-derived mask does not line up with raw depth pixels; this
    # transform re-aligns them. Calibrated once against a known target
    # for this QCar's camera pair -- recalibrate if a camera is swapped.
    depthToRgbH = np.array(
        [[ 1.440749898, -6.45417E-16, -126.2303115],
         [-0.000294167,  1.445138872, -106.3509378],
         [-2.24559E-06, -1.62662E-18,  1.0         ]], dtype=np.float32)

    try:
        while not stop_event.is_set():
            t0 = time.time()

            if IS_PHYSICAL_QCAR:
                cam.read_RGB()
                img = cam.imageBufferRGB
                # The PAL 'M' (meters) dataMode is not functional on this
                # QCar's RealSense interface, so we read the raw 'PX'
                # buffer (uint8, 0-255) and convert to meters with the
                # empirically calibrated `depth_scale` above.
                cam.read_depth(dataMode='PX')
                depth = cam.imageBufferDepthPX / depth_scale
                # Warp the depth image into the RGB frame so the
                # RGB-derived object masks index the correct depth
                # pixels. INTER_NEAREST (not LINEAR) is important here:
                # linear interpolation across an object/background edge
                # fabricates phantom intermediate depths.
                depth = cv2.warpPerspective(
                    depth,
                    depthToRgbH,
                    (640, 480),
                    flags=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                )
            else:
                cam.read()
                img = cam.rgb
                depth = cam.depth
            if img is None:
                time.sleep(dt_target)
                continue

            detections = detector.detect(img)
            distances = {}
            light_color = 'unknown'
            stop_pub, light_pub = None, None
            now = time.time()

            stop_det = detections.get(Perception.STOP_SIGN)
            if stop_det is not None:
                d = detector.find_distance(depth, stop_det['mask'])
                distances[Perception.STOP_SIGN] = d
                stop_pub = {'distance': d, 'bbox': stop_det['bbox'], 't': now}

            light_det = detections.get(Perception.TRAFFIC_LIGHT)
            if light_det is not None:
                d = detector.find_distance(depth, light_det['mask'])
                light_color = detector.classify_light(img, light_det['bbox'])
                distances[Perception.TRAFFIC_LIGHT] = d
                light_pub = {'distance': d, 'bbox': light_det['bbox'],
                             'color': light_color, 't': now}

            annotated = detector.annotate(img, detections, distances,
                                          light_color)
            state.update(stop_pub, light_pub, annotated)

            # Sleep to maintain target rate.
            elapsed = time.time() - t0
            if elapsed < dt_target:
                time.sleep(dt_target - elapsed)
    finally:
        try:
            cam.terminate()
        except Exception:
            pass

#endregion


# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

#region : Behavior FSMs

class StopSignFSM:
    """Finite-state machine that gates v_ref based on stop-sign
    detections.

    States:
        IDLE      : no relevant stop sign in view; no override.
        APPROACH  : a stop sign is within `trigger_distance`; output
                    `approach_speed` so the car slows down before
                    reaching it.
        STOP      : the car has reached the sign and is being held at
                    v_ref = 0. The stop timer counts up here.
        COOLDOWN  : the hold timer elapsed; the car is allowed to
                    proceed past the sign, but the FSM ignores
                    stop-sign detections until the sign has been absent
                    for `cooldown_duration` seconds (so we don't
                    immediately re-trigger on the same sign as we drive
                    past it).

    `update` should be called once per control tick with the latest
    perception state and the current vehicle speed. It returns a
    speed-reference override:
        None  -- no override, use the nominal v_ref.
        float -- the FSM is requesting this v_ref instead.
    """

    IDLE, APPROACH, STOP, COOLDOWN = 'IDLE', 'APPROACH', 'STOP', 'COOLDOWN'

    def __init__(self, trigger_distance, hold_duration, cooldown_duration,
                 approach_speed):
        self.trigger_distance  = trigger_distance
        self.hold_duration     = hold_duration
        self.cooldown_duration = cooldown_duration
        self.approach_speed    = approach_speed

        self.state = self.IDLE
        # Wall-clock timestamps used for the hold and cooldown timers.
        self._t_stop_begin   = None
        self._t_sign_lastseen = None

    def update(self, detection, current_t, v):
        """
        Args:
            detection (dict or None): the stop_sign entry from
                PerceptionState.snapshot(). Keys: 'distance', 'bbox',
                't'. None means no stop sign was detected this tick.
            current_t (float): current wall-clock time (seconds).
            v (float): current vehicle speed estimate (m/s).

        Returns:
            float or None: requested v_ref (m/s), or None for no override.
        """
        # YOUR CODE HERE (Part D1)
        sign_visible = ( detection is not None and np.isfinite ( detection.get ( 'distance' , np.inf )))
        if sign_visible :
            self._t_sign_lastseen = current_t
        if self.state == self.IDLE :
            if sign_visible and detection ['distance'] < self.trigger_distance :
                self.state = self.APPROACH
                return self.approach_speed
            return None
        if self.state == self.APPROACH :
            if v < 0.05: # stopped -> latch
                self.state = self.STOP
                self._t_stop_begin = current_t
                return 0.0
            return self.approach_speed
        if self.state == self.STOP :
            if current_t - self._t_stop_begin >= self.hold_duration :
                self.state = self.COOLDOWN
                return None
            return 0.0
        if self.state == self.COOLDOWN :
            if current_t - self._t_sign_lastseen >= self.cooldown_duration :
                self.state = self.IDLE
                return None


class TrafficLightFSM:
    """Finite-state machine that gates v_ref based on traffic-light
    detections. Unlike StopSignFSM this is purely reactive (no timed
    latch): if the closest visible light is red or yellow and within
    `trigger_distance`, command v_ref = 0; if green, no override.
    """

    def __init__(self, trigger_distance):
        self.trigger_distance = trigger_distance
        # Hysteresis: once we have decided to stop, keep stopping until
        # we see green at least `_clear_required` times in a row, so a
        # single noisy 'green' classification on a red light doesn't
        # release the brake.
        self._clear_required = 10
        self._clear_streak   = 0
        self._engaged        = False

    def update(self, detection, current_t, v):
        """
        Args:
            detection (dict or None): the traffic_light entry from
                PerceptionState.snapshot(). Keys: 'distance', 'bbox',
                'color', 't'. None means no light was detected this tick.
            current_t (float): unused here, kept for API symmetry.
            v (float): unused here.

        Returns:
            float or None: requested v_ref (m/s), or None for no override.
        """
        # YOUR CODE HERE (Part D2)

        # if traffic light detected
        light_visible = ( detection is not None and np.isfinite ( detection.get ( 'distance' , np.inf )))

        if light_visible and detection ['distance'] < self.trigger_distance:  # if within trigger distance
            # update hysteresis
            if detection ['color'] == 'green':  # if green then begin disengage
                if self._clear_streak >= self._clear_required:  # if required streak met
                    self._engaged = False   # disengage
                else:   # if streak not met
                    self._clear_streak += 1 # increase streak
            else:   # if yellow or red light
                self._clear_streak = 0  # reset streak
                self._engaged = True
        else:   # if outside trigger distance
            self._engaged = False   # disengage
            self._clear_streak = 0  # reset streak

        # return velocity
        if self._engaged is True:   #if engaged(red or yellow)
            return 0.0      # stop
        else:   # if disengaged(green)
            return None    # continue

#endregion


# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

#region : Visualization helper (provided, from Lab 6)

def visualize_plan(roadmap, nodeSequence, title="Planned path"):
    plt_obj, ax = roadmap.display()
    ax.set_title(title)
    if nodeSequence is None or len(nodeSequence) < 2:
        return plt_obj
    waypoints = roadmap.generate_path(nodeSequence)
    if waypoints is not None and waypoints.size > 0:
        ax.plot(waypoints[0, :], waypoints[1, :],
                color='blue', linewidth=3, alpha=0.6, zorder=2,
                label='Planned path')
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
                    markerfacecolor='blue', markeredgecolor='black', zorder=3)
    ax.legend(loc='best')
    return plt_obj

#endregion


# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

#region : Initial setup -- build roadmap and run planner

print('=' * 64)
print('Final Lab: Perception-Aware Driving Under Traffic Law')
print('=' * 64)

roadmap = SDCSRoadMap(leftHandTraffic=leftHandTraffic)
planner = PathPlanner(roadmap)

print('Roadmap loaded: {} nodes, {} edges.'.format(
    len(roadmap.nodes), len(roadmap.edges)))
print('Mode: {}'.format(mode))

if mode in ('plan_only', 'drive'):
    print('Start node: {}  ->  Goal node: {}'.format(startNode, goalNode))
    print('Planner: {}'.format(plannerType))

    nodeSequence, totalCost, plannerRuntime = planner.plan(
        startID=startNode, goalID=goalNode,
        plannerType=plannerType, turn_penalty=turnPenalty,
    )
    if nodeSequence is None:
        print('\n[ERROR] No path from {} to {}.'.format(startNode, goalNode))
        raise SystemExit(1)

    print('\nResult:')
    print('  Node sequence : {}'.format(nodeSequence))
    print('  Total cost    : {:.3f} m'.format(totalCost))
    print('  Nodes expanded: {}'.format(planner.nodesExpanded))
    print('  Runtime       : {:.2f} ms'.format(1000 * plannerRuntime))

    waypointSequence = roadmap.generate_path(nodeSequence)
    initialPose = roadmap.get_node_pose(nodeSequence[0]).squeeze()

    visualize_plan(roadmap, nodeSequence,
                   title='Final Lab plan ({}): start={}, goal={}, cost={:.2f} m'
                         .format(plannerType, startNode, goalNode, totalCost))
    print('\nClose the plan figure to continue.')
    plt.show(block=True)

    if mode == 'plan_only':
        print('\nmode = plan_only  ->  exiting.')
        raise SystemExit(0)

#endregion


#region : QLabs / signal setup

if mode == 'drive':
    if not IS_PHYSICAL_QCAR:
        import qlabs_setup_signage
        qlabs_setup_signage.setup(
            initialPosition=[initialPose[0], initialPose[1], 0],
            initialOrientation=[0, 0, initialPose[2]],
        )
        calibrate = False
    else:
        calibrate = 'y' in input('do you want to recalibrate? (y/n) ')

    calibrationPose = [0, 0, -np.pi/2]

# Shared state and stop event used by every thread we spawn below.
KILL_THREAD = False
def sig_handler(*args):
    global KILL_THREAD
    KILL_THREAD = True
signal.signal(signal.SIGINT, sig_handler)

#endregion


# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

#region : Speed and Steering Controllers (reused from Lab 3 / Lab 6)

class SpeedController:
    def __init__(self, kp=0, ki=0):
        self.maxThrottle = 0.3
        self.kp = kp
        self.ki = ki
        self.ei = 0

    def update(self, v, v_ref, dt):
        e = v_ref - v
        self.ei += dt * e
        return np.clip(self.kp * e + self.ki * self.ei,
                       -self.maxThrottle, self.maxThrottle)


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
        wp_1 = self.wp[:, np.mod(self.wpi, self.N)]
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
            -self.maxSteeringAngle, self.maxSteeringAngle,
        )

#endregion


# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

#region : Control loop with perception integration

def controlLoop(perception_state):
    """Lab 6 control loop, modified to gate v_ref based on the latest
    perception state. The only NEW logic relative to Lab 6 lives in
    Part D3 below; everything else is identical to Lab 6.
    """
    global KILL_THREAD
    u, delta = 0, 0
    countMax = controllerUpdateRate / 10
    count = 0

    speedController    = SpeedController(kp=K_p, ki=K_i)
    steeringController = SteeringController(
        waypoints=waypointSequence, k=K_stanley, cyclic=False,
    )

    # Behavior FSMs that turn detections into v_ref overrides.
    stopFSM  = StopSignFSM(
        trigger_distance=stopTriggerDistance,
        hold_duration=stopHoldDuration,
        cooldown_duration=signCooldownDuration,
        approach_speed=approachSpeed,
    )
    lightFSM = TrafficLightFSM(trigger_distance=lightTriggerDistance)

    qcar = QCar(readMode=1, frequency=controllerUpdateRate)
    ekf  = QCarEKF(x_0=initialPose)
    gps  = QCarGPS(initialPose=calibrationPose, calibrate=calibrate)

    # Goal-stop state (reused from Lab 6).
    N_wp = waypointSequence.shape[1]
    last_segment_index = N_wp - 2
    goalPos = np.asarray(waypointSequence[:, -1], dtype=float)
    goalReached = False

    # Track the previous v_ref decision to log transitions cleanly.
    prev_reason = 'startup'

    with qcar, gps:
        t0 = time.time()
        t = 0
        while (t < tf + startDelay) and (not KILL_THREAD):
            tp = t
            t  = time.time() - t0
            dt = t - tp

            # --- sense + estimate (unchanged from Lab 6) ---
            qcar.read()
            if gps.readGPS():
                y_gps = np.array([gps.position[0], gps.position[1],
                                  gps.orientation[2]])
                ekf.update([qcar.motorTach, delta], dt, y_gps,
                           qcar.gyroscope[2])
            else:
                ekf.update([qcar.motorTach, delta], dt, None,
                           qcar.gyroscope[2])
            x  = ekf.x_hat[0, 0]
            y  = ekf.x_hat[1, 0]
            th = ekf.x_hat[2, 0]
            # Reference point shifted forward of the rear axle (Lab 3 conv.).
            p = (np.array([x, y])
                 + np.array([np.cos(th), np.sin(th)]) * 0.2)
            v = qcar.motorTach

            # --- goal-stop detection (unchanged from Lab 6) ---
            on_last_segment = (steeringController.wpi >= last_segment_index)
            distToGoal = float(np.linalg.norm(p - goalPos))
            if (not goalReached) and on_last_segment \
                    and (distToGoal < goalStopDistance):
                goalReached = True
                print('Goal reached at t = {:.2f} s (d = {:.3f} m).'
                      .format(t - startDelay, distToGoal))

            # ============== Part D3 - Apply Perception to v_ref ==========
            # The two FSMs each return either None (no opinion) or a
            # target v_ref. We start from the nominal v_ref and let
            # each FSM lower it; we never let an FSM raise it. We also
            # log whichever reason currently owns the speed reference
            # so the console shows when traffic policies engage.
            #
            # The effective speed is the MINIMUM of all proposals: the most
            # conservative request wins. goalReached forces 0.0 (the lowest
            # possible), so it always dominates. stop-sign and traffic-light
            # are NOT strictly ranked: whichever proposes the lower speed wins.
            stop_det, light_det = perception_state.snapshot()
            stop_ovr  = stopFSM.update(stop_det,  t, v)
            light_ovr = lightFSM.update(light_det, t, v)

            # YOUR CODE HERE (Part D3)
            # Combine stop_ovr, light_ovr, and goalReached into a
            # single effective speed reference (v_ref_effective)
            # and a reason string. See the lab guide, Part D3, for
            # the required arbitration rule. The two lines below
            # are a placeholder (nominal cruise, no overrides) so
            # the file runs before you implement this part.
            v_ref_effective = v_ref
            reason = 'nominal'

            if stop_ovr is not None and stop_ovr < v_ref_effective:
                # v_ref_effective = stop_ovr
                reason = f'stop_sign{stopFSM.state}'

            elif light_ovr is not None and light_ovr < v_ref_effective:
                # v_ref_effective = light_ovr
                reason = f'traffic_light{lightFSM._engaged}'
            
            if goalReached is True:
                # v_ref_effective = 0.0
                reason = 'goal reached'
            
            v_ref_effective = min(v_ref, stop_ovr, light_ovr, 0 if goalReached else None, key=lambda x: float('inf') if x is None else x)


            if reason != prev_reason:
                print('  [t={:6.2f}] v_ref -> {:.2f} m/s  ({})'.format(
                    max(0.0, t - startDelay), v_ref_effective, reason))
                prev_reason = reason

            # --- act ---
            if t < startDelay:
                u, delta = 0, 0
            elif goalReached:
                u = speedController.update(v, 0.0, dt)
                delta = 0
            else:
                u = speedController.update(v, v_ref_effective, dt)
                # Freeze steering at zero while fully stopped to avoid
                # wheel-slip artifacts at v ~ 0.
                if v_ref_effective <= 0.01 and v < 0.05:
                    delta = 0
                else:
                    delta = steeringController.update(p, th, v)

            qcar.write(u, delta)

            # --- scopes ---
            count += 1
            if count >= countMax and t > startDelay:
                t_plot = t - startDelay
                speedScope.axes[0].sample(t_plot, [v, v_ref_effective])
                speedScope.axes[1].sample(t_plot, [v_ref_effective - v])
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


# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

#region : Main entry: scope setup and thread orchestration

if __name__ == '__main__':

    # ------------------------------------------------------------------
    # perception_only mode: run camera + Perception in the main thread
    # so students can tune HSV bounds without bringing up QLabs / scopes.
    # ------------------------------------------------------------------
    if mode == 'perception_only':
        print('\nmode = perception_only  ->  starting camera. '
              'Press q in the Image window to quit.')

        # In virtual mode, spawn the QCar and signs so students have
        # something to look at while tuning Parts A, B, C. The QCar is
        # placed in front of the first sign in SIGN_PLAN. On the
        # physical car, the camera is already live and no QLabs setup
        # is needed.
        if not IS_PHYSICAL_QCAR:
            import qlabs_setup_signage
            pos, ori = qlabs_setup_signage.default_perception_pose()
            qlabs_setup_signage.setup(initialPosition=pos,
                                      initialOrientation=ori)

        perception_state = PerceptionState()
        stop_event = threading.Event()
        pThread = Thread(target=perceptionLoop,
                         args=(perception_state, stop_event), daemon=True)
        pThread.start()
        try:
            while not KILL_THREAD:
                frame = perception_state.get_frame()
                if frame is not None:
                    cv2.imshow('Image', frame)
                if cv2.waitKey(30) & 0xFF == ord('q'):
                    break
        finally:
            stop_event.set()
            pThread.join(timeout=2.0)
            cv2.destroyAllWindows()
            if not IS_PHYSICAL_QCAR:
                try:
                    import qlabs_setup_signage
                    qlabs_setup_signage.terminate()
                except Exception:
                    pass
        raise SystemExit(0)

    # ------------------------------------------------------------------
    # drive mode: scope setup + perception thread + control thread.
    # ------------------------------------------------------------------
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

    # Steering-control scope with planned-path overlay on the SDCS map
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

    # Spawn perception + control threads.
    perception_state = PerceptionState()
    stop_event = threading.Event()

    perceptionThread = Thread(
        target=perceptionLoop,
        args=(perception_state, stop_event),
        daemon=True,
    )
    controlThread = Thread(
        target=controlLoop,
        args=(perception_state,),
        daemon=False,
    )

    perceptionThread.start()
    controlThread.start()

    try:
        while controlThread.is_alive() and not KILL_THREAD:
            MultiScope.refreshAll()
            frame = perception_state.get_frame()
            if frame is not None:
                cv2.imshow('Image', frame)
                cv2.waitKey(1)
            time.sleep(0.01)
    finally:
        KILL_THREAD = True
        stop_event.set()
        controlThread.join(timeout=2.0)
        perceptionThread.join(timeout=2.0)
        cv2.destroyAllWindows()

    if not IS_PHYSICAL_QCAR:
        import qlabs_setup_signage
        qlabs_setup_signage.terminate()
    input('Experiment complete. Press any key to exit...')

#endregion
