import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class AOITracking(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.aoi_encoding = {
                "aoi_hit_[box:top]": "top",
                "aoi_hit_[box:bottom]": "bottom",
                "neither": "neither"
            }

    def _track_aoi_change(self, stim_data):
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
            aoi = self.aoi_encoding["aoi_hit_[box:top]"]
        elif start_aoi_data["aoi_hit_[box:bottom]"] == 1 and start_aoi_data["aoi_hit_[box:top]"] == 0:
            aoi = self.aoi_encoding["aoi_hit_[box:bottom]"]
        else:
            aoi = self.aoi_encoding["neither"]
        changes = []

        # Get initial starting time
        start = start_aoi_data["recording_timestamp"]

        # Loop through data and track AOI changes
        for i, row in stim_data.iterrows():
            prev_aoi = aoi
            if row["aoi_hit_[box:bottom]"] == 0 and row["aoi_hit_[box:top]"] == 1:
                aoi = self.aoi_encoding["aoi_hit_[box:top]"]
            elif row["aoi_hit_[box:bottom]"] == 1 and row["aoi_hit_[box:top]"] == 0:
                aoi = self.aoi_encoding["aoi_hit_[box:bottom]"]
            else:
                aoi = self.aoi_encoding["neither"]

            if prev_aoi is not aoi:
                new_start = row["recording_timestamp"]
                stare_time = new_start - start

                changes.append([row["participant_name"], row["recording_name"], row["presented_stimulus_name"],
                                prev_aoi, start, stare_time])

                start = new_start

        return changes

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        aoi_tracks = []
        for sub in pd.unique(X["participant_name"]):
            for stim in pd.unique(X["presented_stimulus_name"]):
                # Filter for the desired stimulus for the given subject
                stim_data = X[(X["participant_name"] == sub) \
                              & (X["presented_stimulus_name"] == stim)].reset_index(drop=True)

                if len(stim_data) == 0:
                    continue

                changes = self._track_aoi_change(stim_data)

                aoi_tracks.extend(changes)

        aoi_track = pd.DataFrame(aoi_tracks, columns=["participant_name", "recording_name", "presented_stimulus_name",
                                                      "aoi_targ", "start_timestamp", "time_on_aoi"])

        return aoi_track

class FirstGlanceFinder(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def _find_first_glance_bottom(self, participant, stim_data, stim_name):
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

        stim_info = [participant, first_glance.iloc[0]["recording_name"],
                     first_glance.iloc[0]["presented_stimulus_name"], \
                     stim_start, first_glance_time, diff]

        return stim_info

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        first_glances = []
        for sub in pd.unique(X["participant_name"]):
            for stim in pd.unique(X["presented_stimulus_name"]):
                # Filter for the desired stimulus for the given subject
                stim_data = X[(X["participant_name"] == sub) \
                              & (X["presented_stimulus_name"] == stim)].reset_index(drop=True)

                # Find the gaze changes for the given stimulus

                # Find the first glance for the given stimulus

                if len(stim_data) == 0:
                    print(sub, ":", "Subject never views AOI hit[Box:Bottom] for", stim, "...")
                    nas = [sub, np.nan, stim, np.nan, np.nan, np.nan]
                    first_glances.append(nas)
                else:
                    glance = self._find_first_glance_bottom(sub, stim_data, stim)
                    first_glances.append(glance)

        glances_df = pd.DataFrame(first_glances,
                                  columns=["participant_name", "recording_name", "presented_stimulus_name",
                                           "start_timestamp", "first_glance", "time_till_first_glance"])

        return glances_df

