import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# Add the project root to the system path
sys.path.insert(0, project_root)

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object,evaluate_models
from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("split training and testing data")
            xtrain,xtest,ytrain,ytest=(
                train_array[:,:-1],
                test_array[:,:-1],
                train_array[:,-1],
                test_array[:,-1]
            )

            models={
                "SVM":SVC(),
                "KNN":KNeighborsClassifier(n_neighbors = 3, metric = 'minkowski'),
                "Navebias":GaussianNB(),
            }

            logging.info("getting the best model")
            model_report:dict=evaluate_models(xtrain=xtrain,xtest=xtest,ytrain=ytrain,ytest=ytest,models=models)

            # best model score
            best_model_score=max(sorted(model_report.values()))
            # best model name
            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model=models[best_model_name]
            # logging.info("got it")

            if best_model_score<0.6:
                raise CustomException("NO best model found")
            logging.info("best model found on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(xtest)
            report=classification_report(ytest,predicted,labels=["No","Yes"])
            return report


        except Exception as e:
            raise CustomException(e,sys)


