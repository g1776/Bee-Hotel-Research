import cv2
import numpy as np


def individual_detect(frame):
    pass


def find_closest_circle(circles, point):
    """
    Find the closest circle to a point. Returns the circle and the index of the circle in the list of circles.
    """
    # detect all circles in the frame

    # find circle closest to middle of contour
    closest_circle = None
    closest_circle_dist = None
    index_of_closest_circle = None
    for i, circle in enumerate(circles):
        circle_dist = np.linalg.norm(np.array(circle)[:2] - np.array(point))
        if closest_circle_dist is None or circle_dist < closest_circle_dist:
            closest_circle = circle
            closest_circle_dist = circle_dist
            index_of_closest_circle = i

    return closest_circle, index_of_closest_circle
