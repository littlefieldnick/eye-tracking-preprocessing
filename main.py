import argparse
import re
import os
from sklearn_pandas import DataFrameMapper, gen_features

from utils.csv import *
from utils.config import Config
from transforms.base import MissingValTransformer, NumericalFeatureEngineer
from transforms.aoi import AOITracking, FirstGlanceFinder
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
    return transformed

def run_aoi_step(config, data):
    """
    Feature engineering for AOI data: First Glances and AOI Gaze Tracking

    :param config: configuration settings
    :param data: data to transform and process
    :return:
    """
    feat_selector = FeatureSelector(config.get_config_setting("columnsForAOI"), "aoi", dropna=True,
                                    convert_lower=True, space_to_underscore=True)
    feats = feat_selector.transform(data)

    aoi_tracker = AOITracking()
    first_glance_finder = FirstGlanceFinder()

    return aoi_tracker.transform(feats), first_glance_finder.transform(feats)

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

    for file in config.get_config_setting("filesToProcess"):
        data = load_csv(file, low_memory=False)
        # Preprocess the data
        base = run_base_clean_step(config, data)
        pupil = run_pupil_diameter_step(config, base)
        aoi, glances = run_aoi_step(config, base)

        # Strip off root directory of data csv and extract only file name
        file_base = file.split("/")[-1].replace(" ", "_")

        # Remove the .csv ending to append to the end of file name
        if ".tsv" in file_base:
            file_base = file_base.split(".tsv")[0]
        else:
            file_base = file_base.split(".csv")[0]

        # Save preprocessed data to csv files
        df_to_csv(base, os.path.join(config.get_config_setting("outdir"), file_base + "_cleaned.csv"))
        df_to_csv(pupil, os.path.join(config.get_config_setting("outdir"), file_base + "_pupil_data.csv"))
        df_to_csv(aoi, os.path.join(config.get_config_setting("outdir"), file_base + "_aoi_tracking.csv"))
        df_to_csv(glances, os.path.join(config.get_config_setting("outdir"), file_base + "_first_glances.csv"))


if __name__ == "__main__":
    main()
