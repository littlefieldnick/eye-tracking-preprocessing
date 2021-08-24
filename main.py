import argparse
import re
import os

from sklearn.pipeline import Pipeline
from tqdm import tqdm

from utils.csv import *
from utils.config import Config
from transforms.base import DropByValues, RegexFilter, MissingValueReplacer, BaseNumericalFeatureEngineer
from transforms.aoi import FirstGlanceFinder, AOITrackingFeatures
from transforms.pupil import PupilAggregator, BinarizerWrapper
from transforms.feature_selector import  FeatureSelector

def get_parser():
    parser = argparse.ArgumentParser(description="Preprocess eye tracking data.")
    parser.add_argument("--config", dest="config_file",
                        help="Directory where the configuration file for preprocessing is stored.")
    return parser

def get_base_pipeline(config):
    cols_to_keep = cols_to_lower(config.get_config_setting("columnsToKeep"))
    feat_selector = FeatureSelector(cols_to_keep)
    regex = re.compile(config.get_config_setting("stimulusNameFilter"))

    pipeline = Pipeline([("feat_sel", feat_selector),
                         ("inval_mvmnts", DropByValues(["eye_movement_type"], {
                             "eye_movement_type": config.get_config_setting("invalidEyeMovements")
                         })),
                         ("stim_filter", RegexFilter(regex, ["presented_stimulus_name"])),
                         ("miss_val", MissingValueReplacer(["pupil_diameter_left", "pupil_diameter_right"], {
                             "pupil_diameter_left": -1,
                             "pupil_diameter_right": -1
                         }, {
                              "pupil_diameter_left": "NaN",
                               "pupil_diameter_right": "NaN"
                          })),
                         ("feat_eng", BaseNumericalFeatureEngineer())])
    return pipeline

def get_pupil_pipeline(config):
    pup_cols_to_keep = cols_to_lower(config.get_config_setting("columnsForPupil"))

    pup_pipeline = Pipeline([
        ("pupil_feat", FeatureSelector(pup_cols_to_keep)),
        ("drop_na", DropByValues(["pupil_diameter_left", "pupil_diameter_right"], {
            "pupil_diameter_right": ["NaN"],
            "pupil_diameter_left": ["NaN"]
        })),
        ("aggregator", PupilAggregator()),
        ("signif_change", BinarizerWrapper("diff_avg_pupil_diameter", config.get_config_setting("pupilThreshold")))
    ])

    return pup_pipeline

def get_aoi_pipelines(config):
    cols_to_keep = cols_to_lower(config.get_config_setting("columnsForAOI"))
    aoi_cols = cols_to_lower(config.get_config_setting("aoiColumns"))
    aoi_enc = config.get_config_setting("aoiEncodings")

    tracking_pipeline = Pipeline([
        ("aoi_feat", FeatureSelector(cols_to_keep)),
        ("first_glance", AOITrackingFeatures(aoi_cols, aoi_enc))
    ])

    glance_pipeline = Pipeline([
        ("aoi_feat", FeatureSelector(cols_to_keep)),
        ("first_glance", FirstGlanceFinder())
    ])
    return tracking_pipeline, glance_pipeline

def cols_to_lower(columns):
    return [col.lower().replace(" ", "_") for col in columns]

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

    for file in tqdm(config.get_config_setting("filesToProcess")):
        data = load_csv(file, low_memory=False)
        data.columns = cols_to_lower(data.columns)

        # Preprocess the data
        base = get_base_pipeline(config)
        pupil = get_pupil_pipeline(config)
        tracking, glances = get_aoi_pipelines(config)

        base_data = base.fit_transform(data)
        pupil_data = pupil.fit_transform(base_data)
        tracking_data = tracking.fit_transform(base_data)
        glance_data = glances.fit_transform(base_data)

        # Strip off root directory of data csv and extract only file name
        file_base = file.split("/")[-1].replace(" ", "_")

        # Remove the .csv ending to append to the end of file name
        if ".tsv" in file_base:
            file_base = file_base.split(".tsv")[0]
        else:
            file_base = file_base.split(".csv")[0]

        # Save preprocessed data to csv files
        df_to_csv(base_data, os.path.join(config.get_config_setting("outdir"), file_base + "_cleaned.csv"))
        df_to_csv(pupil_data, os.path.join(config.get_config_setting("outdir"), file_base + "_pupil_data.csv"))
        df_to_csv(tracking_data, os.path.join(config.get_config_setting("outdir"), file_base + "_aoi_tracking.csv"))
        df_to_csv(glance_data, os.path.join(config.get_config_setting("outdir"), file_base + "_first_glances.csv"))


if __name__ == "__main__":
    main()
