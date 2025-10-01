import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# Add the project root to the system path
sys.path.insert(0, project_root)

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from imblearn.over_sampling import SMOTE

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts","preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    
    def get_data_transformer_object(self):
        try:
            numerical_columns = ["IQ","Prev_Sem_Result",
                                 "CGPA","Academic_Performance",
                                 "Extra_Curricular_Score",
                                 "Communication_Skills",
                                 "Projects_Completed"
                                 ]
            
            categorical_columns = ["Internship_Experience",]
            
            cat_pipeline = Pipeline(
                steps=[
                    ("OneHotEncoder",OneHotEncoder(sparse_output=False)),
                ]
            )
            num_pipeline = Pipeline(
                steps=[
                    ("StandardScaler",StandardScaler())
                ]
            )

            logging.info("encoding completed")
            preprocessor=ColumnTransformer(
                [
                    ("cat_pipeline",cat_pipeline,categorical_columns),
                    ("num_pipeline",num_pipeline,numerical_columns)
                ],
                remainder='passthrough'
            )
            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
    
    def get_data_load_balance_object(self):
        try:
            logging.info("load balancing object initiate")
            smote = SMOTE(random_state=42)
            return smote
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read Train and test Data Completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_object()
            loadBalancing_obj=self.get_data_load_balance_object()
            target_column_name="Placement"

            x_train=train_df.drop(columns=[target_column_name],axis=1)
            x_test=test_df.drop(columns=[target_column_name],axis=1)

            y_train=train_df[target_column_name]
            y_test=test_df[target_column_name]
            logging.info("split data into xtrain, xtest, ytrain, ytest")

            logging.info("Applying preprocessing object on training and testing data frame")
            x_train_transformed_arr=preprocessing_obj.fit_transform(x_train)
            x_test_transformed_arr=preprocessing_obj.transform(x_test)

            logging.info("Applying loadbalancing object on training and testing data frame")
            x_train_TransformedAndBalanced_arr, y_train_balanced = loadBalancing_obj.fit_resample(x_train_transformed_arr,np.array(y_train))

            train_arr=np.c_[x_train_TransformedAndBalanced_arr,np.array(y_train_balanced)]
            test_arr=np.c_[x_test_transformed_arr,np.array(y_test)]




            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
                )
            logging.info("saved preprocesing object")

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e,sys)