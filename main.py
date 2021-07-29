import argparse
import re
from sklearn.preprocessing import Binarizer, LabelBinarizer, OneHotEncoder
from sklearn.pipeline import Pipeline
from utils.csv import *
from utils.config import Config
from transforms.base_transformer import BaseTransformer
from transforms.feature_selector import FeatureSelector

def get_parser():
    parser = argparse.ArgumentParser(description="Preprocess eye tracking data.")
    parser.add_argument("--config", dest="config_file",
                        help="Directory where the configuration file for preprocessing is stored.")
    parser.add_argument("--preprocess", dest="pre_all", help="Perform all preprocessing steps")
    parser.add_argument("--step",  dest="pre_step",
                        help="Perform a specific preprocessing step: Cleaning, Pupil Diameters, or AOI Gaze/Tracking")
    return parser

def run_clean_step():
    return

def run_pupil_diameter_step():
    return

def run_aoi_step():
    return

def search_column_names(columns, colname):
    """
    Search a list of dataset columns for a specific column

    :param columns: Columns in dataset
    :param colname: Specific column to find
    :return: i (index for column found) or -1 (column not found)
    """

    for i, col in enumerate(columns):
        if col == colname:
            return i

    return -1

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

    for csv in config.get_config_setting("filesToProcess"):
        data = load_csv(csv, low_memory=False)

        stim_regex = re.compile(config.get_config_setting("stimulusNameFilter"))

        pre_pipeline = Pipeline(steps=[
            ("feat_selector", FeatureSelector(config.get_config_setting("columnsToKeep"))),
            ("base_trans", BaseTransformer(stim_regex,
                                           config.get_config_setting("invalidEyeMovements"),
                                           time_since=True))
        ])

        cleaned = pre_pipeline.transform(data)
        cleaned.to_csv("data/cleaned.csv", index=False)
if __name__ == "__main__":
    main()