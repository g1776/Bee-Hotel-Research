"""This file is testing built-in object tracker in OpenCV"""
import argparse
from dotenv import load_dotenv
from src.config import MotionCapConfig
import cv2
import datetime
import cv2
from .utils.logging import *
from .utils.motion_cap_helpers import overlap, RECT_NAMEDTUPLE, get_bb_center, find_closest_circle
import imutils
from imutils.video import FPS
from typing import List
from uuid import uuid4


def obj_tracker(config: MotionCapConfig):
    print(
        f"--- Motion detection session started at {datetime.datetime.now()} for file"
        f" {config.VIDEO} ---"
    )

    if config.LOG:
        init_logging_session(config.LOG, config.VIDEO)

    cap = cv2.VideoCapture(config.VIDEO)
    tracker = cv2.TrackerCSRT_create()
    initBB = None
    fps = None

    while True:
        frame = cap.read()
        frame = frame[1]
        if frame is None:
            break

        (H, W) = frame.shape[:2]

        # check to see if we are currently tracking an object
        if initBB is not None:
            # grab the new bounding box coordinates of the object
            frame = imutils.resize(frame, width=500)
            (success, box) = tracker.update(frame)
            # check to see if the tracking was a success
            if success:
                (x, y, w, h) = [int(v) for v in box]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # update the FPS counter
            fps.update()
            fps.stop()
            # initialize the set of information we'll be displaying on
            # the frame
            info = [
                ("Tracker", tracker),
                ("Success", "Yes" if success else "No"),
                ("FPS", "{:.2f}".format(fps.fps())),
            ]
            # loop over the info tuples and draw them on our frame
            for (i, (k, v)) in enumerate(info):
                text = "{}: {}".format(k, v)
                cv2.putText(
                    frame,
                    text,
                    (10, H - ((i * 20) + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )

        # show the output frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        # if the 's' key is selected, we are going to "select" a bounding
        # box to track
        if key == ord("s"):
            # select the bounding box of the object we want to track (make
            # sure you press ENTER or SPACE after selecting the ROI)
            initBB = cv2.selectROI("Frame", frame, fromCenter=False, showCrosshair=True)
            # start OpenCV object tracker using the supplied bounding box
            # coordinates, then start the FPS throughput estimator as well
            tracker.init(frame, initBB)
            fps = FPS().start()

        # if the `q` key was pressed, break from the loop
        elif key == ord("q"):
            break

    cap.release()


class ObjTracker:
    ID = 0

    def __init__(self, config: MotionCapConfig, init_bb, init_frame):
        self.config = config
        self.tracker = cv2.TrackerKCF_create()
        self.tracker.init(init_frame, init_bb)
        self.current_bb = init_bb
        self.object_exists = False

        # unique id for each tracker
        self.id = ObjTracker.ID
        ObjTracker.ID += 1

    def update(self, frame):
        # grab the new bounding box coordinates of the object
        (success, box) = self.tracker.update(frame)
        self.current_bb = box
        # check to see if the tracking was a success
        if success:
            (x, y, w, h) = [int(v) for v in box]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            self.object_exists = True
        else:
            self.object_exists = False

    def get_current_bee_id(self, tube_hives, config) -> int:
        """Get the id of the bee that is currently tracked by this tracker."""
        dropped_bee_bb = self.current_bb
        middle = get_bb_center(dropped_bee_bb)
        bee_id = find_closest_circle(tube_hives, middle, config)

        return bee_id

    def draw(self, frame):
        (x, y, w, h) = [int(v) for v in self.current_bb]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    def overlaps_with_bb(self, contour_bb):
        c_rnt = RECT_NAMEDTUPLE(*contour_bb)
        i_rnt = RECT_NAMEDTUPLE(*self.current_bb)
        return overlap(c_rnt, i_rnt)


class Multitracker:
    def __init__(self, max_trackers=10):
        self.trackers: List[ObjTracker] = []
        self.max_trackers: int = max_trackers

    def add_tracker(self, tracker):
        if self.max_trackers > len(self.trackers):
            self.trackers.append(tracker)

    def remove_tracker(self, tracker):
        self.trackers.remove(tracker)

    def update(self, frame, detected_contours) -> List[ObjTracker]:
        """Update all trackers and return the ones that have been dropped.
            Gets dropped if the object was lost by the tracker, or if the object was lost by the contour detection.

        Args:
            frame (np.array): The frame to update the trackers with

        Returns:
            List[ObjTracker]: The trackers that have been dropped.
        """
        dropped_trackers = []
        for tracker in self.trackers:
            tracker.update(frame)

            # we drop the tracker if there is not a corresponding contour
            drop = False
            for contour in detected_contours:
                if tracker.overlaps_with_bb(contour["bb"]):
                    self.trackers.remove(tracker)
                    dropped_trackers.append(tracker)
                    drop = True
                    break
            if drop:
                continue

            if tracker.object_exists:
                tracker.draw(frame)
            else:
                # we also drop the tracker if the object was lost by the tracker
                self.trackers.remove(tracker)
                dropped_trackers.append(tracker)
        return dropped_trackers


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        help="Path to .env file. Default is .env",
        default=".env",
    )
    args = vars(parser.parse_args())

    # Load the .env file
    load_dotenv(args["config"])

    obj_tracker(config=MotionCapConfig())
