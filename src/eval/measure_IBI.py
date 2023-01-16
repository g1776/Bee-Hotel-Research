import pandas as pd
from utils import load_log
import sys


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


def measure_IBI_accuracy(
    captured: pd.DataFrame, ground_truth: pd.DataFrame, buffer=10, check_id=True, verbose=True
) -> float:
    """
    Measure the accuracy of a a set of IBI events captured by the model vs. the ground truth measured by a human watching the video.

    Args:
        captured (pd.DataFrame): A dataframe containing the IBI events captured by the model.
        ground_truth (pd.DataFrame): A dataframe containing the IBI events measured by a human.
        buffer (int): The number of seconds to use as a buffer.
        check_id (bool): Whether to check the bee ID of the captured events.
        verbose (bool): Whether to print the score as it is calculated.
    """

    # True and False Positive counts for each Bee ID
    TPs = {}
    FPs = {}

    for c_i, c_row in captured.iterrows():
        for g_i, g_row in ground_truth.iterrows():

            if g_row["Bee ID"] not in TPs.keys():
                TPs[g_row["Bee ID"]] = 0
                FPs[g_row["Bee ID"]] = 0

            # get the start and end of the ground truth event
            gt_start = timestamp_to_seconds(g_row["Start Timestamp"])
            gt_end = timestamp_to_seconds(g_row["End Timestamp"])

            # get the time of the captured event
            cap = timestamp_to_seconds(c_row["timestamp"].strftime("%H:%M:%S"))

            # if id is not checked, then we do not assume the bee IDs match, so we will log the combo of both.
            if not check_id:
                key = f"GT: {g_row['Bee ID']}, CAP: {c_row['bee_id']}"
            else:
                key = c_row["bee_id"]

            # create key if it does not exist
            if key not in TPs.keys():
                TPs[key] = 0
            if key not in FPs.keys():
                FPs[key] = 0

            # check if the captured event is within `buffer`seconds of the ground truth event, or inside it
            if bool(cap >= gt_start - buffer) and bool(cap <= gt_end + buffer):

                if check_id:
                    # check if the bee IDs match
                    if g_row["Bee ID"] != c_row["bee_id"]:
                        FPs[key] += 1
                    else:
                        TPs[key] += 1
                else:
                    TPs[key] += 1
            else:
                FPs[key] += 1

    # print the captured bee ID counts, sorted by the number of times they were captured, descending
    print("--- REPORT ---")
    print("Captured Events: ", len(captured))
    print("Ground Truth Events: ", len(ground_truth))
    print("Total Captured True Positives:", sum(TPs.values()))
    print("Total Captured False Positives:", sum(FPs.values()))

    print("\n---SNAPSHOT---")

    print("Top 5 Captured True Positives by Bee ID:")
    top_5_TPs = dict(sorted(TPs.items(), key=lambda item: item[1], reverse=True)[:5])
    top_5_TPs_df = pd.DataFrame(top_5_TPs.items(), columns=["Bee ID", "Count"])
    print(top_5_TPs_df)

    print("Top 5 Captured False Positives by Bee ID:")
    top_5_FPs = dict(sorted(FPs.items(), key=lambda item: item[1], reverse=True)[:5])
    top_5_FPs_df = pd.DataFrame(top_5_FPs.items(), columns=["Bee ID", "Count"])
    print(top_5_FPs_df)

    # calculate the accuracy
    return sum(TPs.values()) / (sum(TPs.values()) + sum(FPs.values()))


if __name__ == "__main__":
    # parse the first argument as the path to the ground truth data, and the second as the path to the captured data
    ground_truth = pd.read_csv(sys.argv[1])
    captured = load_log(sys.argv[2], verbose=False)

    # measure the accuracy of the captured data
    acc = measure_IBI_accuracy(captured, ground_truth)
    print(f"---\nAccuracy: {acc}")
