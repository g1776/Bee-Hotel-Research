import cv2
from typing import List, Tuple
from math import dist
import numpy as np
from .logging import log_it
from collections import namedtuple

RECT_NAMEDTUPLE = namedtuple("RECT_NAMEDTUPLE", "x1 x2 y1 y2")


def overlap(rec1, rec2):
    if (rec2.x2 > rec1.x1 and rec2.x2 < rec1.x2) or (rec2.x1 > rec1.x1 and rec2.x1 < rec1.x2):
        x_match = True
    else:
        x_match = False
    if (rec2.y2 > rec1.y1 and rec2.y2 < rec1.y2) or (rec2.y1 > rec1.y1 and rec2.y1 < rec1.y2):
        y_match = True
    else:
        y_match = False
    if x_match and y_match:
        return True
    else:
        return False


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
    if len(contours_window) == 0:
        return []

    detected_bees_info = []

    last_frame = contours_window[-1]
    for contour_info in last_frame:
        overlap_found = False
        for older_frame in contours_window[: len(contours_window) - 1]:

            # if we found an overlap, we don''t need to keep checking this contour against old frames
            if overlap_found:
                break

            for older_contour_info in older_frame:
                if overlap(
                    RECT_NAMEDTUPLE(
                        contour_info["bb"][0],
                        contour_info["bb"][0] + contour_info["bb"][2],
                        contour_info["bb"][1],
                        contour_info["bb"][1] + contour_info["bb"][3],
                    ),
                    RECT_NAMEDTUPLE(
                        older_contour_info["bb"][0],
                        older_contour_info["bb"][0] + older_contour_info["bb"][2],
                        older_contour_info["bb"][1],
                        older_contour_info["bb"][1] + older_contour_info["bb"][3],
                    ),
                ):
                    overlap_found = True
                    break

        if not overlap_found:
            detected_bees_info.append(contour_info)

    return detected_bees_info
