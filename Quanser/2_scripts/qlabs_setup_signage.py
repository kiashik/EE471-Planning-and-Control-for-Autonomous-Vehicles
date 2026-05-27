# (c) 2026 S. Farzan, Electrical Engineering Department, Cal Poly
# EE 471 (SP26): Planning and Control for Autonomous Vehicles
"""
qlabs_setup_signage.py

Companion setup script for the EE 471 final lab. Identical to the
Lab 6 `qlabs_setup.py`, but ALSO spawns a stop sign and a traffic light
along the SDCS roadmap so the planned path drives the QCar past them.

This file is provided to students. They do not need to edit it.

Sign placements are expressed relative to roadmap NODE INDICES rather
than world coordinates so the locations stay valid if Quanser updates
the SDCS map. Each entry specifies:
    node    : SDCS roadmap node index to anchor the sign to
    offset  : (dx, dy) shift in meters from the node, in roadmap-frame
              coordinates. Used to push the sign onto the shoulder of
              the road so the QCar does not collide with it.
    yaw     : additional yaw rotation applied to the sign actor (rad).
              The sign is first aligned to face oncoming traffic at the
              anchor node, then rotated by this offset.

If you want to design their own scenario, modify the SIGN_PLAN
list and pick a different (startNode, goalNode) pair in
lab07_ee471.py.
"""
import numpy as np
import time

from qvl.qlabs import QuanserInteractiveLabs
from qvl.qcar2 import QLabsQCar2
from qvl.qcar import QLabsQCar
from qvl.free_camera import QLabsFreeCamera
from qvl.real_time import QLabsRealTime
from qvl.traffic_light import QLabsTrafficLight
from qvl.stop_sign import QLabsStopSign

import pal.resources.rtmodels as rtmodels
from pal.products.qcar import QCAR_CONFIG
from hal.products.mats import SDCSRoadMap


# ----------------------------------------------------------------------
# Scenario definition. The default scenario assumes a planned path from
# node 2 to node 20, which crosses through nodes that are reasonable
# anchors for a stop sign and a traffic light. Adjust to taste.
# ----------------------------------------------------------------------
SIGN_PLAN = [
    {
        'kind':   'stop_sign',
        'node':    8,                 # anchor node on the SDCS map
        'offset': (0.15,  0.10),       # m, in roadmap coordinates
        # 'node':    15,                 # anchor node on the SDCS map
        # 'offset': (0.0, 0.3),
        'yaw':     0.0,                # rad
    },
    {
        'kind':   'traffic_light',
        'node':   15,
        'offset': (-0.2, -0.2),
        'yaw':     np.pi/2,
        # Initial lamp color: 'red' | 'yellow' | 'green'.
        'color':  'green',
        # If non-zero, automatically cycle the light through
        # red -> green -> red ... with this period in seconds.
        # Set to 0 to leave the lamp fixed at the initial color.
        'cycle_period': 0,
    },
]


def _color_constant(name):
    """Map 'red'/'yellow'/'green' to QLabsTrafficLight color constants."""
    return {
        'red':    QLabsTrafficLight.COLOR_RED,
        'yellow': QLabsTrafficLight.COLOR_YELLOW,
        'green':  QLabsTrafficLight.COLOR_GREEN,
    }.get(name.lower(), QLabsTrafficLight.COLOR_RED)


# ----------------------------------------------------------------------
# Module-level handles -- populated by setup(), used by terminate()
# and (optionally) by a background thread that cycles the light.
# ----------------------------------------------------------------------
_qlabs = None
_handles = {}


def default_perception_pose(setback=1.0):
    """Compute a sensible spawn pose for perception tuning: `setback`
    meters behind the first sign in SIGN_PLAN, facing the sign so it
    sits squarely in the front camera. Used by `perception_only` mode
    and by the standalone `tune_hsv.py` utility so students do not
    have to bring up the scene themselves.

    Returns:
        (position, orientation) -- both 3-lists suitable for passing
        to `setup()`.
    """
    roadmap = SDCSRoadMap(leftHandTraffic=False)
    anchor_node = SIGN_PLAN[1]['node']
    x, y, psi = roadmap.nodes[anchor_node].pose[:, 0]
    return (
        [float(x - setback * np.cos(psi)),
         float(y - setback * np.sin(psi)),
         0.0],
        [0.0, 0.0, float(psi)],
    )


def setup(initialPosition=[0, 0, 0],
          initialOrientation=[0, 0, 0],
          rtModel=rtmodels.QCAR2):
    """Connect to QLabs, spawn the QCar at `initialPosition`, and spawn
    the signs listed in SIGN_PLAN at their roadmap-anchored locations.
    """
    global _qlabs, _handles

    qlabs = QuanserInteractiveLabs()
    print('Connecting to QLabs...')
    try:
        qlabs.open('localhost')
    except Exception:
        print('Unable to connect to QLabs')
        raise SystemExit(1)
    print('Connected to QLabs')

    # Clear state and stop any RT model that may still be running.
    qlabs.destroy_all_spawned_actors()
    QLabsRealTime().terminate_all_real_time_models()
    time.sleep(0.5)

    # Spawn the QCar.
    if QCAR_CONFIG['cartype'] == 1:
        hqcar = QLabsQCar(qlabs)
        rtModel = rtmodels.QCAR
    else:
        hqcar = QLabsQCar2(qlabs)
        rtModel = rtmodels.QCAR2

    hqcar.spawn_id(
        actorNumber=0,
        location=[p*10 for p in initialPosition],
        rotation=initialOrientation,
        waitForConfirmation=True,
    )

    # Free camera anchored above the roadmap. Hand-tuned for SDCS.
    hcamera = QLabsFreeCamera(qlabs)
    hcamera.spawn([8.484, 1.973, 12.209], [-0, 0.748, 0.792])
    hqcar.possess()

    # ------------------------------------------------------------------
    # Spawn signs at roadmap-anchored locations.
    # ------------------------------------------------------------------
    roadmap = SDCSRoadMap(leftHandTraffic=False)

    for i, plan in enumerate(SIGN_PLAN):
        node = roadmap.nodes[plan['node']]
        x, y, psi = node.pose[:, 0]

        # Push the sign onto the shoulder of the road.
        dx, dy = plan['offset']
        sx = x + dx
        sy = y + dy
        sz = 0.005  # sit just above ground plane

        # Face the sign toward oncoming traffic at the anchor node.
        # `psi` is the direction of travel at that node, so a sign that
        # the driver sees head-on is rotated by psi + pi.
        sign_yaw = psi + np.pi + plan.get('yaw', 0.0)

        if plan['kind'] == 'stop_sign':
            ss = QLabsStopSign(qlabs)
            ss.spawn(location=[sx*10, sy*10, sz],
                     rotation=[0, 0, sign_yaw],
                     scale=[1, 1, 1],
                     waitForConfirmation=True)
            _handles['stop_sign_{}'.format(i)] = ss
            print('  spawned stop sign at node {} -> world ({:.2f},{:.2f})'
                  .format(plan['node'], sx, sy))

        elif plan['kind'] == 'traffic_light':
            tl = QLabsTrafficLight(qlabs)
            tl.spawn(location=[sx*10, sy*10, sz],
                     rotation=[0, 0, sign_yaw],
                     scale=[1, 1, 1],
                     configuration=0,
                     waitForConfirmation=True)
            initial_color = _color_constant(plan.get('color', 'red'))
            tl.set_color(initial_color, waitForConfirmation=True)
            _handles['traffic_light_{}'.format(i)] = tl
            print('  spawned traffic light at node {} -> world ({:.2f},{:.2f}) '
                  '[initial {}]'.format(plan['node'], sx, sy, plan.get('color')))

            # Optional: start a daemon thread that cycles the light.
            period = float(plan.get('cycle_period', 0.0))
            if period > 0.0:
                import threading
                stop_evt = threading.Event()

                def _cycle(handle=tl, T=period, stop=stop_evt):
                    seq = [QLabsTrafficLight.COLOR_RED,
                           QLabsTrafficLight.COLOR_GREEN]
                    k = 0
                    while not stop.is_set():
                        handle.set_color(seq[k % len(seq)],
                                         waitForConfirmation=False)
                        k += 1
                        stop.wait(T)

                t = threading.Thread(target=_cycle, daemon=True)
                t.start()
                _handles['traffic_light_{}__stop'.format(i)] = stop_evt

    QLabsRealTime().start_real_time_model(rtModel)
    _qlabs = qlabs

    print('QLabs setup complete. {} signs spawned.'
          .format(sum(1 for k in _handles if not k.endswith('__stop'))))


def terminate():
    """Tear down the RT model and any background cycling threads.
    Safe to call multiple times.
    """
    global _qlabs, _handles
    for k, h in list(_handles.items()):
        if k.endswith('__stop'):
            try:
                h.set()
            except Exception:
                pass
    _handles = {}
    try:
        QLabsRealTime().terminate_real_time_model('QCar_Workspace')
        QLabsRealTime().terminate_real_time_model('QCar2_Workspace')
    except Exception:
        pass
    if _qlabs is not None:
        try:
            _qlabs.close()
        except Exception:
            pass
        _qlabs = None
