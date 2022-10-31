import datetime


def generate_log_message(frame: int, total_frames: int, timestamp: str) -> str:
    return f"Motion detected at frame {frame}/{total_frames}, Timestamp: {timestamp}"


def log_it(fp: str, msg: str):
    with open(fp, "a") as f:
        f.writelines(f"{msg}\n")


def init_logging_session(fp, video_path):
    with open(fp, "a") as f:
        f.writelines(
            f"--- Logging session started at {datetime.datetime.now()} for file {video_path} ---\n"
        )
