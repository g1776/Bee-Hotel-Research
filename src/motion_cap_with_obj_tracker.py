import datetime
from typing import List
import cv2
from .utils.motion_cap_helpers import *
from .utils.logging import *
from src.config import MotionCapConfig
from src.obj_tracker import ObjTracker, Multitracker
import argparse
from dotenv import load_dotenv


def motion_detector(
    config: MotionCapConfig,
):
    """Detect motion in a video"""

    # region init

    print(
        f"--- Motion detection session started at {datetime.datetime.now()} for file"
        f" {config.VIDEO} ---"
    )

    if config.LOG:
        init_logging_session(config.LOG, config.VIDEO)

    cap = cv2.VideoCapture(config.VIDEO)

    if config.MOTION_GRANULARITY is None:
        config.MOTION_GRANULARITY = int(cap.get(cv2.CAP_PROP_FPS))

    TOTAL_FRAMES = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    frame_count = 0
    previous_frame = None
    tube_hives = []
    obj_trackers: Multitracker = Multitracker(5)

    # endregion

    while True:

        # ignore the first `BUFFER_FRAMES` frames to allow the camera to adjust to the environment
        if frame_count < config.BUFFER_FRAMES:
            frame_count += 1
            continue

        # read and preprocess the frame
        success, frame = cap.read()
        if not success:
            break
        preprocessed = preprocess_frame(frame, config)

        # Once we have reached the `BUFFER_FRAMES`th frame, we grab the tube hive coordinates and assign them to Bee IDs
        if frame_count == config.BUFFER_FRAMES:
            tube_hives = get_tube_hives_coords(preprocessed, config.LOG)

        # determine motion on every `DETECTION_RATE-th frame
        frame_count += 1
        if (frame_count % config.DETECTION_RATE) == 0:
            # 3. Set previous frame and continue if there is None
            if previous_frame is None:
                # First frame; there is no previous one yet
                previous_frame = preprocessed
                continue

            # Detect motion and grab the contours that represent this motion
            contours, previous_frame = detect_contours_of_motion(
                preprocessed, previous_frame, config
            )

            # filter out contours on size and distance to tubes.
            # This will also assign bee_ids to the contours
            # NOTE: This is where the bee_ids are assigned
            contours_to_track = filter_contours(contours, tube_hives, config)

            # First, check contours without trackers and start trackers for them
            for contour in contours_to_track:

                # check if contour overlaps with any existing trackers. If so, don't start a new tracker
                ignore_contour = False
                for obj_tracker in obj_trackers.trackers:
                    if obj_tracker.overlaps_with_bb(contour["bb"]):
                        ignore_contour = True
                        break
                if ignore_contour:
                    continue

                print(f"Frame {frame_count}: Tracking bee starting at {contour['bb']}")
                obj_tracker = ObjTracker(config, contour["bb"], frame)
                obj_trackers.add_tracker(obj_tracker)

            # Then, update all trackers.
            print(f"Frame {frame_count}: Updating {len(obj_trackers.trackers)} trackers")
            dropped_trackers = obj_trackers.update(frame)

            # TODO: Generate the logs from the dropped trackers

        if config.SHOW:
            cv2.imshow("🐝🏨 motion detector", frame)

        # check for quit operation
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Exiting by user input.")
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nFinished processing {config.VIDEO}")


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

    motion_detector(config=MotionCapConfig())
