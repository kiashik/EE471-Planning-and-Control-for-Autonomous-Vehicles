# (c) 2026 S. Farzan, Electrical Engineering Department, Cal Poly
# EE 471 (SP26): Planning and Control for Autonomous Vehicles
"""
prelab07_ee471.py

Pre-Lab 7 starter file: Color-Based Perception Primitives.

In this pre-lab you will implement four perception primitives that you
will reuse, unchanged, inside the `Perception` class in Lab 7. The
function signatures here match the Lab 7 file exactly, so when you
move on to Lab 7 you can copy your working code straight in.

Run me with:
    python prelab07_ee471.py

The bottom of this file is a self-test driver that loads the three
sample images, runs your pipeline, and saves annotated outputs to
`images/_out/`. Inspect those outputs visually to confirm your
implementations are correct.

Required packages: numpy, opencv-python
    pip install numpy opencv-python
"""
import os
import cv2
import numpy as np


# ============================================================================
# Configuration: HSV bounds (Part 1 deliverable)
# ----------------------------------------------------------------------------
# Replace the placeholder values below with the bounds you tuned in Part 1.
# H is in [0, 180], S and V are in [0, 255] (OpenCV convention).
#
# Red wraps around in HSV (it appears near H=0 AND near H=180), so we use
# TWO ranges for red and OR the resulting masks together in the test driver.
# ============================================================================
RED_LOWER_1 = (  0,  90,  70)   # tuned values
RED_UPPER_1 = ( 12, 255, 255)
RED_LOWER_2 = (170,  90,  70)
RED_UPPER_2 = (180, 255, 255)

YELLOW_LOWER = ( 18,  55,  80)
YELLOW_UPPER = ( 40, 255, 255)

GREEN_LOWER = ( 40,  70,  60)
GREEN_UPPER = ( 95, 255, 255)

# Minimum contour area (in pixels) for a detection to count.
MIN_BLOB_AREA = 200


# ============================================================================
# Part 1: HSV Thresholding
# ============================================================================
def color_threshold(img_hsv, lower, upper):
    """Return a binary mask of pixels that fall inside the inclusive HSV box.

    Args:
        img_hsv (ndarray): HxWx3 image in HSV.
        lower (array-like): length-3 lower bound (H, S, V).
        upper (array-like): length-3 upper bound (H, S, V).

    Returns:
        ndarray: HxW uint8 binary mask with values in {0, 255}.

    Hint: a single call to `cv2.inRange` is enough.
    """
    return cv2.inRange(img_hsv, lower, upper)


def mask_img(img, mask):
    """Return the BGR image with all non-mask pixels set to zero.

    Args:
        img (ndarray):  HxWx3 BGR image.
        mask (ndarray): HxW uint8 mask.

    Returns:
        ndarray: HxWx3 BGR image, masked.

    Hint: see `cv2.bitwise_and(img, img, mask=mask)`.
    """
    return cv2.bitwise_and(img, img, mask=mask)


# ============================================================================
# Part 2: Blob Detection
# ============================================================================
def _clean_mask(mask):
    """Morphological open + close to remove speckle and seal small holes.
    PROVIDED -- do not modify.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def detect_largest_blob(mask, min_area=MIN_BLOB_AREA):
    """Find the largest connected component in a binary mask.

    Args:
        mask (ndarray): HxW uint8 binary mask.
        min_area (int): minimum contour area in pixels. Smaller blobs
                        are rejected. Returns (None, None) if no
                        contour clears this threshold.

    Returns:
        ((x, y, w, h), blob_mask)  if a blob is found, otherwise
        (None, None).

        - (x, y, w, h) is the axis-aligned bounding box in pixel
          coordinates: x = left, y = top, w = width, h = height.
        - blob_mask is a HxW uint8 mask containing ONLY that single
          blob filled in (not the original full mask).

    Implementation outline:
      1. `cv2.findContours` with RETR_EXTERNAL + CHAIN_APPROX_SIMPLE.
      2. If no contours found -> return (None, None).
      3. Pick the contour with the largest `cv2.contourArea`.
      4. If that area < min_area -> return (None, None).
      5. Use `cv2.boundingRect` for the bbox and `cv2.drawContours`
         (with thickness=-1) to fill the contour into a fresh mask.
    """
    # find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None
    # find largest contour
    largest_contour = max(contours, key=cv2.contourArea)

    if cv2.contourArea(largest_contour) < min_area:
        return None, None
    
    bbox = cv2.boundingRect(largest_contour)
    blob_mask = np.zeros_like(mask)
    cv2.drawContours(blob_mask, [largest_contour], -1, 255, thickness=-1)
    return bbox, blob_mask


# ============================================================================
# Part 3: Traffic-Light State Classification
# ============================================================================
def classify_light(img_bgr, bbox):
    """Identify which lamp of a traffic light is currently lit.

    The traffic-light housing is a tall yellow rectangle with three
    circular lamps stacked vertically: red on top, yellow in the middle,
    green on the bottom. Even though we know WHERE each lamp lives within
    the bbox, we don't know which one is ON. So sample the appropriate
    sub-region of the bbox and report whichever third has the strongest
    color signal in its expected hue range.

    Args:
        img_bgr (ndarray): HxWx3 BGR image.
        bbox (tuple): (x, y, w, h) bounding box of the traffic light,
                      or None.

    Returns:
        str: one of 'red', 'yellow', 'green', or 'unknown'.

    Implementation outline:
      1. If bbox is None or too small -> return 'unknown'.
      2. Slice three vertical thirds of the bbox region. Top third =
         where the red lamp lives; middle = yellow; bottom = green.
      3. For each third, convert to HSV and build an in-range mask
         using the bounds for THAT lamp's color (the global RED_*,
         YELLOW_*, GREEN_* constants at the top of this file).
         Remember red needs the OR of its two ranges.
      4. Compute the mean V (brightness) of the masked pixels in each
         third. An unlit lamp produces a very low mean V; a lit lamp
         produces a high mean V.
      5. Return the third whose mean V is highest IF that V exceeds a
         brightness threshold (e.g. 100/255). Otherwise return
         'unknown' -- if no lamp is clearly lit, we fail safe.
    """
    if bbox is None:
        return 'unknown'

    x, y, w, h = bbox
    if w < 20 or h < 50:    # TUNEABLE: if the bbox is too small, it's probably a false detection.
        return 'unknown'

    min_on_brightness = 100
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
        mask = color_threshold(hsv_third, first_lower, first_upper)
        for lower, upper in ranges[1:]:
            current_mask = color_threshold(hsv_third, lower, upper)
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

# ============================================================================
# Self-test driver (PROVIDED -- do not modify)
# ============================================================================
def _red_mask(hsv):
    """Helper: OR of the two red HSV ranges."""
    return cv2.bitwise_or(
        color_threshold(hsv, RED_LOWER_1, RED_UPPER_1),
        color_threshold(hsv, RED_LOWER_2, RED_UPPER_2),
    )


def _yellow_mask(hsv):
    return color_threshold(hsv, YELLOW_LOWER, YELLOW_UPPER)


def _annotate(img, label, bbox, color=(0, 0, 255)):
    out = img.copy()
    if bbox is None:
        return out
    x, y, w, h = bbox
    cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
    cv2.putText(out, label, (x, max(0, y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return out


def run_self_test(image_dir='images', out_dir='images/_out'):
    os.makedirs(out_dir, exist_ok=True)

    # ---- Image 1: stop sign only ----
    print('\n[test 01]  stop sign')
    img = cv2.imread(os.path.join(image_dir, 'test_01_stopsign.jpg'))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = _clean_mask(_red_mask(hsv))
    bbox, _ = detect_largest_blob(mask)
    print('  bbox:', bbox)
    cv2.imwrite(os.path.join(out_dir, '01_mask.jpg'),
                mask_img(img, mask))
    cv2.imwrite(os.path.join(out_dir, '01_annotated.jpg'),
                _annotate(img, 'stop_sign', bbox, (0, 0, 255)))

    # ---- Image 2: three traffic lights side by side ----
    print('\n[test 02]  three traffic lights')
    img = cv2.imread(os.path.join(image_dir, 'test_02_lights.jpg'))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # We expect to find three yellow-body blobs; iterate over them.
    yel = _clean_mask(_yellow_mask(hsv))
    contours, _ = cv2.findContours(yel, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    out = img.copy()
    labels = []
    for c in sorted(contours, key=lambda c: cv2.boundingRect(c)[0]):
        if cv2.contourArea(c) < MIN_BLOB_AREA:
            continue
        x, y, w, h = cv2.boundingRect(c)
        color = classify_light(img, (x, y, w, h))
        labels.append(color)
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 200, 200), 2)
        cv2.putText(out, color, (x, max(0, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 200), 1,
                    cv2.LINE_AA)
    print('  classified lamps (left -> right):', labels)
    print('  expected:                          [red, yellow, green]')
    cv2.imwrite(os.path.join(out_dir, '02_annotated.jpg'), out)

    # ---- Image 3: stop sign + traffic light + distractor ----
    print('\n[test 03]  scene')
    img = cv2.imread(os.path.join(image_dir, 'test_03_scene.jpg'))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    out = img.copy()

    red_m = _clean_mask(_red_mask(hsv))
    bbox_stop, _ = detect_largest_blob(red_m)
    out = _annotate(out, 'stop_sign', bbox_stop, (0, 0, 255))

    yel_m = _clean_mask(_yellow_mask(hsv))
    bbox_light, _ = detect_largest_blob(yel_m)
    if bbox_light is not None:
        color = classify_light(img, bbox_light)
        out = _annotate(out, 'traffic_light ({})'.format(color),
                        bbox_light, (0, 200, 200))

    print('  stop bbox:    ', bbox_stop)
    print('  light bbox:   ', bbox_light)
    cv2.imwrite(os.path.join(out_dir, '03_annotated.jpg'), out)

    print('\nAnnotated outputs written to', out_dir)


if __name__ == '__main__':
    run_self_test()
