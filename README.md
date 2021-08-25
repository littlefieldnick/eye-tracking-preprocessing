# Eye Tracking Preprocessing Pipeline

### Installation and Setup
1. Download and install (if needed) Python 3.7 or higher. 
2. Clone or download the preprocessing code
3. Open terminal (command prompt on Windows) and `cd` into the code directory
4. Run `pip3 install -r requirements.txt` to install the requirements to run the code

### Running the Preprocessing Pipeline   
1. Open `config.yml` check the configurations and add the files you want to preprocess under `filesToProcess`
2. Modify any configurations that need to be changed. 
3. In the terminal run the following command listed below. This will run the preprocessing pipeline and process any files listed in the config file under `filesToProcess`

```
$ python3 main.py --config "config/config.yml"
```

### Preprocessed Output
The preprocessed files will be saved in the `data/` directory (unless `outdir` is changed in `config.yml`. 
The following files will be created:
- `<data_file>_cleaned.csv` (or .tsv depending on file format) -- The cleaned data after removing NAs and filtering out unwanted stimulus names
- `<data_file>_pupil.csv` (or tsv) -- Pupil data extracted from the original data file as well as the difference between pupil sizes and average pupil size
- `<data_file>_first_glances.csv` (or tsv) -- Extracts AOI data and finds the first time an AOI target is glanced at
- `<data_file>_aoi_tracking.csv` (or tsv) -- Extracts AOI data and tracks the changes between AOI targets and the time spent on the target
