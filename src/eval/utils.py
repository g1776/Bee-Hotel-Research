import pandas as pd
from datetime import datetime


def timestamp_to_seconds(timestamp: str) -> float:
    """
    Convert a timestamp to seconds.

    Args:
        timestamp (str): The timestamp to convert.

    Returns:
        float: The timestamp in seconds.
    """
    # split the timestamp into hours, minutes, seconds, and milliseconds
    hours, minutes, seconds = timestamp.split(":")

    # convert the timestamp to seconds
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds)


def load_log(log, verbose=True) -> pd.DataFrame:

    with open(log, "r") as f:
        lines = f.readlines()

    logs = []

    for line in lines:
        if "detected at" not in line:
            continue

        bee_id = int(line.split("=")[1].split(" ")[0])
        timestamp_to_parse = line.split(" ")[-1].strip()

        ####################
        # clean timestamps #
        ####################

        if len(timestamp_to_parse) < 5:
            # there just isn't enough to parse
            if verbose:
                print("Dropping row with insufficient timestamp:", timestamp_to_parse)
            continue

        edited_timestamp = False

        if ":" not in timestamp_to_parse:
            # add missing colons
            timestamp_to_parse = (
                timestamp_to_parse[:2]
                + ":"
                + timestamp_to_parse[2:4]
                + ":"
                + timestamp_to_parse[4:]
            )
            edited_timestamp = True

        if timestamp_to_parse.count(":") == 1:
            edited_timestamp = True
            chars_after_colon = timestamp_to_parse.split(":")[1]
            if len(chars_after_colon) == 3:
                # add zero
                chars_after_colon = chars_after_colon[:2] + ":" + chars_after_colon[2] + "0"
            elif len(chars_after_colon) == 2:
                # add missing seconds
                chars_after_colon = chars_after_colon + ":00"
            elif len(chars_after_colon) == 1:
                # add missing seconds and minutes
                chars_after_colon = "0:00" + chars_after_colon
            else:
                chars_after_colon = chars_after_colon[:2] + ":" + chars_after_colon[2:]

            timestamp_to_parse = timestamp_to_parse.split(":")[0] + ":" + chars_after_colon

        # parse
        timestamp_to_parse = timestamp_to_parse[:8]
        try:
            timestamp = datetime.strptime(timestamp_to_parse, "%H:%M:%S")
        except ValueError:
            if verbose:
                print("Unable to parse timestamp:", timestamp_to_parse)
            continue

        logs.append(
            {"bee_id": bee_id, "timestamp": timestamp, "timestamp was edited": edited_timestamp}
        )

    df = pd.DataFrame(logs)
    df["timestamp"] = df["timestamp"]
    return df
