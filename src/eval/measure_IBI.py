import pandas as pd
from utils import load_log, timestamp_to_seconds
import sys


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

        current_bee_id = str(c_row["bee_id"])
        # get the time of the captured event
        cap = timestamp_to_seconds(c_row["timestamp"].strftime("%H:%M:%S"))

        # if we haven't seen this bee ID before, initialize the TPs and FPs to 0
        if current_bee_id not in TPs:
            TPs[current_bee_id] = 0
            FPs[current_bee_id] = 0

        match_found = False
        for g_i, g_row in ground_truth.iterrows():

            # get the start and end of the ground truth event
            gt_start = timestamp_to_seconds(g_row["Start Timestamp"])
            gt_end = timestamp_to_seconds(g_row["End Timestamp"])

            # if we have already found a match, then we can skip the rest of the ground truth events.
            # We can also do any early escape if we've gone 5 seconds past the buffer.
            if match_found or cap > gt_end + buffer + 5:
                break

            # check if the captured event is within `buffer`seconds of the ground truth event, or inside it
            if bool(cap >= gt_start - buffer) and bool(cap <= gt_end + buffer):
                if check_id:
                    # check if the bee IDs match
                    if g_row["Bee ID"] != c_row["bee_id"]:
                        if verbose:
                            print(f"... Bee ID mismatch: {c_row['bee_id']} != {g_row['Bee ID']}")
                    else:

                        if verbose:
                            print(
                                "SUCCESS: Match for captured event:"
                                f" {c_row['bee_id']}=={g_row['Bee ID']} at {cap}"
                            )
                        TPs[current_bee_id] += 1
                        match_found = True
                else:
                    if verbose:
                        print(f"SUCCESS: Match for captured event: {c_row['bee_id']} at {cap}")
                    TPs[current_bee_id] += 1
                    match_found = True
            else:
                if verbose:
                    print(f"... No match for captured event: {c_row['bee_id']} at {cap}")

        # we were unable to find a match for the captured event in the ground truth
        if not match_found:
            if verbose:
                print(f"FAILURE: No match for captured event: {c_row['bee_id']} at {cap}")
            FPs[current_bee_id] += 1

    # region report

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

    # endregion

    # calculate the accuracy
    return sum(TPs.values()) / (sum(TPs.values()) + sum(FPs.values()))


if __name__ == "__main__":
    # parse the first argument as the path to the ground truth data, and the second as the path to the captured data
    ground_truth = pd.read_csv(sys.argv[1])
    captured = load_log(sys.argv[2], verbose=False)

    # measure the accuracy of the captured data
    acc = measure_IBI_accuracy(captured, ground_truth, verbose=False)
    print(f"---\nAccuracy: {acc}")
