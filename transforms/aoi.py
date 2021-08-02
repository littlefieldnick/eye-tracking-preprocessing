import numpy as np
import pandas as pd


def find_first_glance_bottom(participant, stim_data, stim_name):
    """
    Return the time location the subject looks at the bottom box for a given stimulus

    :param data: data frame containing a single subject and timestamps and AOI for bottom box (AOI hit [Box:Bottom])
    for each presented stimulus.

    :return all_stims: List of starting times for each stimulus, timestamp subject first looks at AOI, and the diff
    between the start and time of first look at AOI

    """

    stim_start = stim_data.loc[0, "recording_timestamp"]
    cond = (stim_data["presented_stimulus_name"] == stim_name) & (stim_data["aoi_hit_[box:bottom]"] == 1)

    first_glance = stim_data[cond]

    # Some stimuli have no records indicating first glace... Check with Dana and Jeanne.
    if first_glance.shape[0] == 0:
        return [participant, stim_data.iloc[0]["recording_name"], stim_name, np.nan, np.nan, np.nan]

    first_glance_time = first_glance.iloc[0]["recording_timestamp"]
    diff = first_glance_time - stim_start

    stim_info = [participant, first_glance.iloc[0]["recording_name"], first_glance.iloc[0]["presented_stimulus_name"], \
                 stim_start, first_glance_time, diff]

    return stim_info


def track_aoi_change(stim_data, encoding):
    """
    Keep track of changes in AOI as subject goes back and forth between [Box:Bottom] and [Box:Top]

    :param data: data frame containing all data records
    :param subject: subject for filtering the data
    :param simulus: stimulus to filter for
    :param encoding: numerical encoding for AOI targets. Required encodings are needed for:
        - AOI hit [Box:Bottom],
        - AOI hit [Box:Top],
        - Neither (for when both AOI targets are 0s)

    :return changes: list of changes between AOI targets and the time spent before changing
    """

    # Figure out where we are staring at when when we start
    start_aoi_data = stim_data.iloc[0]
    if start_aoi_data["aoi_hit_[box:bottom]"] == 0 and start_aoi_data["aoi_hit_[box:top]"] == 1:
        aoi = encoding["aoi_hit_[box:top]"]
    elif start_aoi_data["aoi_hit_[box:bottom]"] == 1 and start_aoi_data["aoi_hit_[box:top]"] == 0:
        aoi = encoding["aoi_hit_[box:bottom]"]
    else:
        aoi = encoding["neither"]
    changes = []

    # Get initial starting time
    start = start_aoi_data["recording_timestamp"]

    # Loop through data and track AOI changes
    for i, row in stim_data.iterrows():
        prev_aoi = aoi
        if row["aoi_hit_[box:bottom]"] == 0 and row["aoi_hit_[box:top]"] == 1:
            aoi = encoding["aoi_hit_[box:top]"]
        elif row["aoi_hit_[box:bottom]"] == 1 and row["aoi_hit_[box:top]"] == 0:
            aoi = encoding["aoi_hit_[box:bottom]"]
        else:
            aoi = encoding["neither"]

        if prev_aoi is not aoi:
            new_start = row["recording_timestamp"]
            stare_time = new_start - start

            changes.append([row["participant_name"], row["recording_name"], row["presented_stimulus_name"],
                            prev_aoi, start, stare_time])

            start = new_start

    return changes