# (c) 2026 S. Farzan, Electrical Engineering Department, Cal Poly
# EE 471 (SP26): Planning and Control for Autonomous Vehicles
"""
tune_hsv.py

Helper utility for Part A1 of the final lab.

Opens the QCar 2 aligned camera and an OpenCV trackbar window so you
can tune HSV bounds for the three colors of interest (red, yellow,
green). When you find a set of bounds that cleanly isolates a target
color, write the values into the corresponding *_LOWER / *_UPPER
constants at the top of lab07_ee471.py.

Run from a terminal:
    python tune_hsv.py

In VIRTUAL mode this script automatically spawns the QCar 2 and the
default scene from qlabs_setup_signage so the camera has something
to look at; in PHYSICAL mode it uses the live camera directly.

Press 'q' in the Image window to quit. Press 'p' to print the current
bounds to the console. Press 'r' / 'R' / 'y' / 'g' to switch between
red-low / red-high / yellow / green presets (starting bounds; you
will tighten them with the trackbars).
"""
import cv2
import numpy as np

from pit.YOLO.utils import QCar2DepthAligned
from pal.products.qcar import IS_PHYSICAL_QCAR

# Reasonable starting presets. These are intentionally wide -- students
# tighten them on the trackbars and copy the final values into
# lab07_ee471.py.
PRESETS = {
    'r': ('red (low-hue range)',  (  0, 90,  70), ( 12, 255, 255)),
    'R': ('red (high-hue range)', ( 170, 90, 70), (180, 255, 255)),
    'y': ('yellow',               ( 18, 55, 80), ( 40, 255, 255)),
    'g': ('green',                ( 40, 70, 60), ( 95, 255, 255)),
}

WINDOW = 'Image'


def _nothing(_):
    pass


def _make_trackbars(initial):
    h_lo, s_lo, v_lo = initial[0]
    h_hi, s_hi, v_hi = initial[1]
    cv2.createTrackbar('hue low',          WINDOW,  h_lo, 180, _nothing)
    cv2.createTrackbar('hue high',         WINDOW,  h_hi, 180, _nothing)
    cv2.createTrackbar('saturation low',   WINDOW,  s_lo, 255, _nothing)
    cv2.createTrackbar('saturation high',  WINDOW,  s_hi, 255, _nothing)
    cv2.createTrackbar('value low',        WINDOW,  v_lo, 255, _nothing)
    cv2.createTrackbar('value high',       WINDOW,  v_hi, 255, _nothing)


def _read_trackbars():
    return (
        (cv2.getTrackbarPos('hue low',         WINDOW),
         cv2.getTrackbarPos('saturation low',  WINDOW),
         cv2.getTrackbarPos('value low',       WINDOW)),
        (cv2.getTrackbarPos('hue high',        WINDOW),
         cv2.getTrackbarPos('saturation high', WINDOW),
         cv2.getTrackbarPos('value high',      WINDOW)),
    )


def _set_trackbars(lo, hi):
    cv2.setTrackbarPos('hue low',         WINDOW, lo[0])
    cv2.setTrackbarPos('hue high',        WINDOW, hi[0])
    cv2.setTrackbarPos('saturation low',  WINDOW, lo[1])
    cv2.setTrackbarPos('saturation high', WINDOW, hi[1])
    cv2.setTrackbarPos('value low',       WINDOW, lo[2])
    cv2.setTrackbarPos('value high',      WINDOW, hi[2])


def main():
    # In virtual mode, bring up the same scene the lab drives in
    # so the camera has signs to look at. In physical mode, use the
    # live camera directly.
    setup_module = None
    if not IS_PHYSICAL_QCAR:
        import qlabs_setup_signage as setup_module
        pos, ori = setup_module.default_perception_pose()
        setup_module.setup(initialPosition=pos, initialOrientation=ori)

    cam = QCar2DepthAligned()

    cv2.namedWindow(WINDOW)
    _make_trackbars(PRESETS['r'][1:])
    current_preset = 'r'

    print('HSV tuning utility')
    print('  r / R / y / g  : load preset for red-lo / red-hi / yellow / green')
    print('  p              : print current bounds')
    print('  q              : quit')

    try:
        while True:
            cam.read()
            img = cam.rgb
            if img is None:
                continue
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            lo, hi = _read_trackbars()
            mask = cv2.inRange(hsv, np.array(lo), np.array(hi))
            masked = cv2.bitwise_and(img, img, mask=mask)

            display = np.hstack([img, masked])
            cv2.imshow(WINDOW, display)
            k = cv2.waitKey(30) & 0xFF
            if k == ord('q'):
                break
            elif k == ord('p'):
                name = PRESETS[current_preset][0]
                print('  [{}]  lower = {}  upper = {}'.format(name, lo, hi))
            elif chr(k) in PRESETS:
                current_preset = chr(k)
                name, lo0, hi0 = PRESETS[current_preset]
                print('  switched to preset: {}'.format(name))
                _set_trackbars(lo0, hi0)
    finally:
        try:
            cam.terminate()
        except Exception:
            pass
        cv2.destroyAllWindows()
        if setup_module is not None:
            try:
                setup_module.terminate()
            except Exception:
                pass


if __name__ == '__main__':
    main()
