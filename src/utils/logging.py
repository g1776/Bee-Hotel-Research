import datetime
from typing import Callable


def generate_log_message(frame_count: int, total_frames: int, timestamp: str, bee_id: int) -> str:
    return (
        f"Bee ID={bee_id} detected at frame {frame_count}/{total_frames}, Timestamp: {timestamp}"
    )


def log_it(fp: str, msg: str, logging_callback: Callable = None):
    with open(fp, "a") as f:
        f.writelines(f"{msg}\n")
        if logging_callback:
            logging_callback(msg)


def init_logging_session(fp, video_path, logging_callback: Callable = None):
    with open(fp, "a") as f:
        f.writelines(
            f"--- Logging session started at {datetime.datetime.now()} for file {video_path} ---\n"
        )

    if logging_callback:
        logging_callback(
            f"--- Logging session started at {datetime.datetime.now()} for file {video_path} ---"
        )
