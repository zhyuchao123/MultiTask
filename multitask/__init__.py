__version__ = '0.1.0'



import requests
import os,logger
from get_data import download
# download dataset
download()
logger.logger.info("finish downloading datasets......")

# create reports folder
reports_path = os.path.join(os.path.abspath(__file__ + "/../"), 'reports')
os.makedirs(reports_path, exist_ok=True)

logger.logger.info("create a folder for reports")
# create output models' file
models_path = os.path.join(os.path.abspath(__file__ + "/../"), '/models/model_out')

os.makedirs(models_path, exist_ok=True)
logger.logger.info("create a folder for trained models")



