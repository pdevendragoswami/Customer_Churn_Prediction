import os
import sys
import pickle
from src.exception import CustomException
from src.logger import logging
from sklearn.metrics import confusion_matrix

def save_obj(file_path:str,obj:str):
    try:
        logging.info('Saving Object is initiated')
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,'wb') as file_obj:
            pickle.dump(obj,file_obj)

        logging.info('Object is saved')

    except Exception as e:
        logging.info('There is some issue at Save Object')
        raise CustomException(e,sys)
        


def evaluate_model(models,X_train,X_test,y_train,y_test):
    try:
        reports = {}
        for i in range(len(models)):
            model = list(models.values())[i]
            model.fit(X_train,y_train)
            y_pred = model.predict(X_test)

            confusion_met = confusion_matrix(y_test,y_pred)

            true_positive = confusion_met[0][0]
            false_positive = confusion_met[0][1]
            false_negative = confusion_met[1][0]
            true_negative = confusion_met[1][1]

            accuracy = (true_positive + true_negative) / (true_positive +false_positive + false_negative + true_negative)
            
            reports[list(models.keys())[i]] = accuracy

        return reports
            
    except Exception as e:
        logging.info('There is some issue at evaluation metrics')
        raise CustomException(e,sys)

def load_obj(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        logging.info('There is some issue at load_obj')
        raise CustomException(e,sys)