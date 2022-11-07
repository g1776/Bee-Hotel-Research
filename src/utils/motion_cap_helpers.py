import cv2
from typing import List, Tuple
from math import dist
import numpy as np
from .logging import log_it


def get_tube_hives_coords(frame, log=None) -> np.ndarray:
    """
    Get the coordinates of the tube hives in the frame. Returns a numpy array of the coordinates.
    """
    tube_hives = cv2.HoughCircles(
        image=frame,
        method=cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=50,
        param1=50,
        param2=30,
        minRadius=5,
        maxRadius=60,
    )

    tube_hives: np.ndarray = np.around(tube_hives).astype(int)[0]

    # deal with logging
    tube_hives_msg = "Coordinates of Tube Hives detected, along with associated Bee ID:\n"
    tube_hives_msg += "\n".join(
        [f"Bee ID={i} Tube Hive Coords: {circle[:2]}" for i, circle in enumerate(tube_hives)]
    )
    tube_hives_msg += "\n\n"
    print(tube_hives_msg)
    if log:
        log_it(log, tube_hives_msg)

    return tube_hives


def get_contour_center(c) -> Tuple[int, int]:
    M = cv2.moments(c)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    return cX, cY


def find_closest_circle(circles, point):
    """
    Find the closest circle to a point. Returns the index of the circle in the list of circles.
    """
    # detect all circles in the frame

    # find circle closest to middle of contour
    closest_circle_dist = None
    index_of_closest_circle = None
    for i, circle in enumerate(circles):
        circle_dist = dist(np.array(circle)[:2], point)
        if closest_circle_dist is None or circle_dist < closest_circle_dist:
            closest_circle_dist = circle_dist
            index_of_closest_circle = i

    return index_of_closest_circle


def process_contours_window(contours_window: List[List[dict]]) -> List[dict]:
    pass
