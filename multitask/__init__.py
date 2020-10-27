__version__ = '0.1.0'


def relativePath(abspath, degree):
    abspath = str(abspath)
    plist = abspath.split('/')

    rtn = ''
    for blk in (plist[-degree:]):
        rtn += blk
        rtn += '/'
    return rtn[:-1]

import requests
import os,logger
from get_data import download
# download dataset
download()
logger.logger.info("finish downloading datasets......")

# create reports folder
reports_path = os.path.join(os.path.abspath(__file__ + "/../"), 'reports')
reports_path = relativePath(reports_path,1)
os.makedirs(reports_path, exist_ok=True)

logger.logger.info("create a folder for reports")
# create output models' file
models_path = os.path.join(os.path.abspath(__file__ + "/../"), '/models/model_out')
models_path = (relativePath(models_path,2))

os.makedirs(models_path, exist_ok=True)
logger.logger.info("create a folder for trained models")



