import argparse
import re
import numpy as np

from sklearn_pandas import DataFrameMapper, gen_features

from utils.csv import *
from utils.config import Config
from transforms.base import MissingValTransformer, NumericalFeatureEngineer
from transforms.aoi import find_first_glance_bottom, track_aoi_change
from transforms.pupil import PupilOperation, PupilSignificance
from transforms.feature_selector import BaseFeatureSelector, FeatureSelector

def get_parser():
    parser = argparse.ArgumentParser(description="Preprocess eye tracking data.")
    parser.add_argument("--config", dest="config_file",
                        help="Directory where the configuration file for preprocessing is stored.")
    return parser

def run_base_clean_step(config, data):

    regex = re.compile(config.get_config_setting("stimulusNameFilter"))
    inval_mvmnts = config.get_config_setting("invalidEyeMovements")

    feat_selector = BaseFeatureSelector(config.get_config_setting("columnsToKeep"), regex, inval_mvmnts)
    feats = feat_selector.fit_transform(data)

    # Build preprocessing steps
    no_preprocessing = gen_features(columns=["participant_name", "recording_name", "recording_timestamp",
                                             "presented_stimulus_name"], classes=None)
    feat_engineer = [(["participant_name","recording_timestamp", "presented_stimulus_name"], NumericalFeatureEngineer(), {"alias": "time_since_stimulus_appeared"})]
    missing_features = gen_features(columns=["pupil_diameter_left", "pupil_diameter_right"],
                                    classes=[MissingValTransformer])
    eye_movements = [("eye_movement_type", None)]
    aoi_features = gen_features(columns=["aoi_hit_[box:bottom]", "aoi_hit_[box:top]"], classes=None)

    # Construct array of all feature preprocessing
    all_features = no_preprocessing + feat_engineer + eye_movements + missing_features + aoi_features

    mapper = DataFrameMapper(all_features, input_df=True, df_out=True)
    transformed = mapper.fit_transform(feats)

    return transformed

def run_pupil_diameter_step(config, data):
    feat_selector = FeatureSelector(config.get_config_setting("columnsForPupil"), "pupil", dropna=True,
                                    convert_lower=True, space_to_underscore=True)
    feats = feat_selector.transform(data)

    no_preprocessing = gen_features(columns=["participant_name", "recording_name", "recording_timestamp",
                                             "presented_stimulus_name", "pupil_diameter_left", "pupil_diameter_right"], classes=None)

    pupil_diff_features = gen_features(columns=["pupil_diameter_left", "pupil_diameter_right"],
                                       classes=[{"class": PupilOperation, "op": "difference"}], prefix="diff_")

    pupil_avg_features = [(["pupil_diameter_left", "pupil_diameter_right"], [PupilOperation("average")], {"alias": "avg_pupil_diameter"}),
                          (["pupil_diameter_left", "pupil_diameter_right"], [PupilOperation("average"),
                                                                             PupilOperation("difference")], {"alias": "avg_diff_pupil_diameter"})]

    thresh_ops = [(["pupil_diameter_left", "pupil_diameter_right"], PupilSignificance(thresh=config.get_config_setting("pupilThreshold")),
                   {"alias": "sign_diff_in_pupil_size"})]

    all_steps = no_preprocessing + pupil_diff_features + pupil_avg_features + thresh_ops
    mapper = DataFrameMapper(all_steps, input_df=True, df_out=True)
    transformed = mapper.fit_transform(feats)
    transformed.to_csv("data/pupil_feats.csv", index=False)
    return transformed

def run_aoi_step(config, data):
    aoi_encoding = {
        "aoi_hit_[box:top]": "top",
        "aoi_hit_[box:bottom]": "bottom",
        "neither": "neither"
    }

    feat_selector = FeatureSelector(config.get_config_setting("columnsForAOI"), "aoi", dropna=True,
                                    convert_lower=True, space_to_underscore=True)
    feats = feat_selector.transform(data)

    aoi_tracks = []
    first_glances = []
    for sub in pd.unique(feats["participant_name"]):
        for stim in pd.unique(feats["presented_stimulus_name"]):
            # Filter for the desired stimulus for the given subject
            stim_data = feats[(feats["participant_name"] == sub) \
                             & (feats["presented_stimulus_name"] == stim)].reset_index(drop=True)

            # Find the gaze changes for the given stimulus

            # Find the first glance for the given stimulus

            if len(stim_data) == 0:
                print(sub, ":", "Subject never views AOI hit[Box:Bottom] for", stim, "...")
                nas = [sub, np.nan, stim, np.nan, np.nan, np.nan]
                first_glances.append(nas)
            else:
                glance = find_first_glance_bottom(sub, stim_data, stim)
                changes = track_aoi_change(stim_data, aoi_encoding)

                aoi_tracks.extend(changes)
                first_glances.append(glance)

    aoi_track_df = pd.DataFrame(aoi_tracks, columns=["participant_name", "recording_name", "presented_stimulus_name",
                                                    "aoi_targ", "start_timestamp", "time_on_aoi"])
    glances_df = pd.DataFrame(first_glances, columns=["participant_name", "recording_name", "presented_stimulus_name",
                                                      "start_timestamp", "first_glance", "time_till_first_glance"])

    aoi_track_df.to_csv("data/aoi_track.csv", index=False)
    glances_df.to_csv("data/glances.csv", index=False)

    return aoi_track_df, glances_df

def main():
    parser = get_parser()
    args = parser.parse_args()

    config = None
    if args.config_file is None:
        print("No configuration file was provided. Exiting...")
        exit(0)
    else:
        config = Config(config_file_pth=args.config_file)

    if not config.check_param_length(param_name="filesToProcess", min_length=1):
        print("No CSV files were provided for processing. Exiting...")
        exit(0)

    data = load_csv("data/cleaned_test.csv", low_memory=False)
    # run_base_clean_step(config, data)
    # pupil = run_pupil_diameter_step(config, data)
    aoi, glances = run_aoi_step(config, data)

if __name__ == "__main__":
    main()
