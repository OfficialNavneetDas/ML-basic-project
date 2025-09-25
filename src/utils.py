import os
import sys
import numpy as np
import pandas as pd
import dill

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# Add the project root to the system path
sys.path.insert(0, project_root)

from src.exception import CustomException
from src.logger import logging

def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open (file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)
        logging.info("preprocessing pkl object saved")

    except Exception as e:
        raise CustomException(e,sys)