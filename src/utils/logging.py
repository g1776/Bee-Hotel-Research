import datetime
from ..config import MotionCapConfig


def log_bee_enter(
    fp: str, frame_count: int, total_frames: int, timestamp: str, bee_id: int, tracker_id: int
):
    msg = (
        f"(Tracker {tracker_id} start) Bee ID={bee_id} enter at frame"
        f" {frame_count}/{total_frames}, Timestamp: {timestamp}"
    )
    log_it(fp, msg)


def log_bee_exit(
    fp: str,
    frame_count: int,
    total_frames: int,
    timestamp: str,
    bee_id: int,
    tracker_id: int,
):
    msg = (
        f"(Tracker {tracker_id} end) Bee ID={bee_id} exit at frame {frame_count}/{total_frames},"
        f" Timestamp: {timestamp}"
    )
    log_it(fp, msg)


def log_it(fp: str, msg: str):
    print(msg)
    with open(fp, "a") as f:
        f.writelines(f"{msg}\n")


def init_logging_session(config: MotionCapConfig):
    fp = config.LOG
    video_path = config.VIDEO
    log_it(
        fp, f"--- Logging session started at {datetime.datetime.now()} for file {video_path} ---"
    )
    log_it(fp, config)
