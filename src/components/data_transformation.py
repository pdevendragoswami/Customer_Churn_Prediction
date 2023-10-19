import os
import sys
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OrdinalEncoder
from sklearn.compose import ColumnTransformer
from src.utils import save_obj
from sklearn.pipeline import Pipeline
from src.components.data_ingestion import DataIngestion

@dataclass
class DataTransfomationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transfomation_config = DataTransfomationConfig()

    def get_data_transformation_obj(self):
        try:
            #segregating numerical columns
            categorical_cols = ['Gender', 'Location']
            numerical_cols = ['Age','Subscription_Length_Months','Monthly_Bill','Total_Usage_GB']


            logging.info('Pipeline creation is initiated')

            numerical_pipeline = Pipeline(
                steps = [
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())
                ]
            )

            categorical_pipeline = Pipeline(
                steps = [
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('ordinalencoder',OrdinalEncoder()),
                    ('scaler',StandardScaler())
                ]
            )

            preprocessor = ColumnTransformer([
                ('numerical_pipeline',numerical_pipeline,numerical_cols),
                ('categorical_pipeline',categorical_pipeline,categorical_cols)
            ])

            logging.info('Pipeline creation is done successfully')
            return preprocessor
                    
        except Exception as e:
            logging.info('There is some issue at get data transformation part')
            raise CustomException(e,sys)

    def initiate_data_transformation(self,train_data_path,test_data_path):
        try:
            logging.info('Data transformation is initiated')

            logging.info('Reading testing and training data')
            df_train = pd.read_csv(train_data_path)
            df_test = pd.read_csv(test_data_path)

            logging.info('Data reading completed successfully')

            target_feature = 'Churn'
            drop_features = ['CustomerID','Name',target_feature]

            df_train_input_feature = df_train.drop(drop_features,axis=1)
            df_train_target_feature = df_train[target_feature]

            df_test_input_feature = df_test.drop(drop_features,axis=1)
            df_test_target_feature = df_test[target_feature]

            preprocessor_obj = self.get_data_transformation_obj()

            input_feature_train_arr = preprocessor_obj.fit_transform(df_train_input_feature)
            input_feature_test_arr = preprocessor_obj.transform(df_test_input_feature)

            train_arr = np.c_[input_feature_train_arr,df_train_target_feature]
            test_arr = np.c_[input_feature_test_arr,df_test_target_feature]

            logging.info('Data transformation is completed')

            save_obj(
                file_path = self.data_transfomation_config.preprocessor_obj_file_path,
                obj = preprocessor_obj)

            logging.info('Pickle file saved.')

            return(
                train_arr,
                test_arr,
                self.data_transfomation_config.preprocessor_obj_file_path
            )





        except Exception as e :
            logging.info('There is some issue in initiate data transformation')
            raise CustomException(e,sys)
        




    