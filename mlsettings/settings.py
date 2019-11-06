import os
import sys
import json

APP_ML_PATH = "ML_PATH"
APP_ML_DATA_PATH = "ML_DATASOURCE"

DEFAULT_FILENAME = r"D:\MachineLearning\mlsettings\mlconfig.json"
DEFAULT_APP_ENV = "DEV"

CONFIG_KEYS = ['ML_DATASOURCE', 'ML_PATH']
import logging
logger = logging.getLogger()
fhandler = logging.FileHandler(filename='mylog.log', mode='a')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
fhandler.setFormatter(formatter)
logger.addHandler(fhandler)
logger.setLevel(logging.DEBUG)


def load_app_config(filename=DEFAULT_FILENAME, app_env=DEFAULT_APP_ENV):
    config_dict = load_config_json(filename)

    for config_value in config_dict[app_env]:

        if config_value in os.environ:
            sys.path.append(os.environ.get(config_value))
            logger.info("Adding {0} to system path".format(os.environ.get(config_value)))
        else:
            logger.info("{0} not found in system path setting from config file".format(config_value))
            sys.path.append(config_dict[app_env][config_value])



def load_config_json(filename):
    if not os.path.exists(filename):
        raise ValueError(" config {0}  file not found".format(filename))
    with open(filename, 'r') as logging_configuration_file:
        config_dict = json.load(logging_configuration_file)
    return config_dict


def get_datafolder_path(data_path=APP_ML_DATA_PATH):
    if data_path in os.environ:
        return os.environ.get(data_path)
    else:
        return os.environ(APP_ML_DATA_PATH)


def load_ml_path():
    if APP_ML_PATH in os.environ:
        sys.path.append(os.environ.get('ML_PATH'))
        logger.info("Adding {0}  to system path".format(os.environ.get('ML_PATH')))
        return True
    else:
        logger.info("{0} not found".format(APP_ML_PATH))
        sys.exit(-1)
        return False
