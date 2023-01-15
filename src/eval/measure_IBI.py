import pandas as pd
from utils import load_log
import sys
from pprint import pprint


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


def measure_IBI(
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

    captured_bee_id_counts = {}

    # for each ground truth event, check if there is a captured event within `buffer` seconds of it, or inside it
    # if there is, add 1 to the score
    # if there isn't, add 0 to the score
    # divide the score by the number of ground truth events
    score = 0
    for index, row in ground_truth.iterrows():
        # get the start and end of the ground truth event
        gt_start = timestamp_to_seconds(row["Start Timestamp"])
        gt_end = timestamp_to_seconds(row["End Timestamp"])

        # check if there is a captured event within `buffer` seconds of it, or inside it
        for index2, row2 in captured.iterrows():
            # get the start and end of the captured event
            cap = timestamp_to_seconds(row2["timestamp"].strftime("%H:%M:%S"))

            # to speed up processing, if the captured event is 10 or more seconds after the ground truth event,
            # we can assume that there are no more captured events that could match the ground truth event
            if cap > gt_end + 10:
                if verbose:
                    print(
                        f"Ground truth event {index} has no matching captured event. Score:"
                        f" {score}/{len(ground_truth)}"
                    )
                break

            # check if the captured event is within `buffer`seconds of the ground truth event, or inside it
            if bool(cap >= gt_start - buffer) and bool(cap <= gt_end + buffer):

                # check if the bee IDs match
                if check_id and row["Bee ID"] != row2["bee_id"]:
                    if verbose:
                        print(
                            f"Ground truth event {index} has a matching captured event, but the"
                            f" bee IDs do not match. Score: {score}/{len(ground_truth)}"
                        )
                    continue

                score += 1
                if verbose:
                    print(
                        f"Ground truth event {index} has a matching captured event. Score:"
                        f" {score}/{len(ground_truth)}"
                    )

                # add the bee ID to the captured bee ID counts.
                # If check_id is False, the bee ID can be assumed the same for both the ground truth and captured events
                if check_id:
                    key = str(row["Bee ID"])
                else:
                    key = f"GT: {row['Bee ID']}, CAP: {row2['bee_id']}"

                if key in captured_bee_id_counts:
                    captured_bee_id_counts[key] += 1
                else:
                    captured_bee_id_counts[key] = 1

                break

    # print the captured bee ID counts, sorted by the number of times they were captured, descending
    print("\n# of events matched for each Bee ID (or combo of Bee IDs if check_id=False):")
    pprint(dict(sorted(captured_bee_id_counts.items(), key=lambda x: x[1], reverse=True)))

    # worst 0, best 1
    return score / len(ground_truth)


if __name__ == "__main__":
    # parse the first argument as the path to the ground truth data, and the second as the path to the captured data
    ground_truth = pd.read_csv(sys.argv[1])
    captured = load_log(sys.argv[2], verbose=False)

    # measure the accuracy of the captured data
    score = measure_IBI(captured, ground_truth)
    print(f"---\nScore: {score}")
