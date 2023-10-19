import sys
import os
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
from src.utils import save_obj,evaluate_model
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation

from sklearn.tree import  DecisionTreeClassifier
from sklearn.naive_bayes import  GaussianNB
from sklearn.linear_model import  LogisticRegression
from sklearn.ensemble import  RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self,train_arr,test_arr):
        try:
            logging.info('Spliting the data into train and test')

            X_train,X_test,y_train,y_test = (train_arr[:,:-1],test_arr[:,:-1],
            train_arr[:,-1],test_arr[:,-1])

            models = {
                'LogisticRegression':LogisticRegression(),
                'DecisionTree':DecisionTreeClassifier(),
                'SVM':SVC(),
                'NaiveBayes':GaussianNB(),
                'KNN':KNeighborsClassifier(),
                'RandomForest':RandomForestClassifier()
                }

            logging.info('Checking for model report')

            model_reports = evaluate_model(models, X_train, X_test, y_train, y_test)

            logging.info('Model accuracy report obtained')
            logging.info(model_reports)

            best_model_score = max(list(model_reports.values()))

            best_model_name = list(model_reports.keys())[list(model_reports.values()).index(best_model_score)]
            print(best_model_name,best_model_score)



            best_model = models[best_model_name]
        

            logging.info(f'Best model found: {best_model_name}, Best model accuracy:{best_model_score}')

            print(f'Best model found: {best_model_name}, Best model accuracy:{best_model_score}')

            save_obj(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

            logging.info("Best Model obj is saved")



        except Exception as e:
            logging.info('This is some issue at initiate model training')
            raise CustomException(e,sys)
        

