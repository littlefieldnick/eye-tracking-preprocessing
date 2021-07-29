import yaml
import os

class Config():
    def __init__(self, config_file_pth):
        self.configurations = self.load_config_file(config_file_pth)
        self.__create_config_output_dir_if_not_exists()

    def load_config_file(self, config_pth):
        with open(config_pth) as config:
            configurations = yaml.full_load(config)

        if config is None:
            print("No configurations were provided. Exiting...")
            exit(0)

        return configurations

    def get_config_setting(self, setting_name):
        return self.configurations.get(setting_name, None)

    def check_param_length(self, param_name, min_length):
        return len(self.configurations[param_name]) >= min_length

    def __create_config_output_dir_if_not_exists(self):
        out_dir = self.configurations.get("outdir", None)
        if not os.path.exists(out_dir):
            try:
                os.mkdir(out_dir)
            except Exception as e:
                print(e.message)
                exit(0)
