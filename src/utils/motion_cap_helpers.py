import cv2
from typing import List, Tuple
from math import dist
import numpy as np
from .logging import log_it, generate_log_message
from collections import namedtuple

RECT_NAMEDTUPLE = namedtuple("RECT_NAMEDTUPLE", "x1 x2 y1 y2")


def overlap(rec1, rec2) -> bool:
    """Check if two rectangles overlap.

    Args:
        rec1 (RECT_NAMEDTUPLE): First rectangle
        rec2 (RECT_NAMEDTUPLE): Second rectangle

    Returns:
        bool: True if the rectangles overlap, False otherwise
    """
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


def process_contours_window(contours_window: List[List[dict]], TOTAL_FRAMES, config) -> None:
    """Process the contour window to determine when a bee leaves the frame.
        The idea here is that when a bee finally disappears, it is at the end of its
        flight/movement path and it has most disappeared from the frame, into the tube hive.
        This makes sure we only process the final frame the bee is visible to determine its ID.

    Args:
        contours_window (List[List[dict]]): The list of contours for each frame in the window.

    Returns:
        None
    """

    if len(contours_window) == 0:
        return []

    # process the contours window to determine when a bee leaves the frame
    contours_to_log = []
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
            contours_to_log.append(contour_info)

    # log the contours that made it through the window
    for contour_info in contours_to_log:

        # extract elements from contour info
        frame = contour_info["frame"]
        bee_id = contour_info["bee_id"]
        timestamp_text = contour_info["timestamp"]

        log_msg = generate_log_message(frame, TOTAL_FRAMES, timestamp_text, bee_id)
        print(log_msg)
        if config.LOG:
            log_it(config.LOG, log_msg)
