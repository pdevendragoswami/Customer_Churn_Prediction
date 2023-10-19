import os
import sys
from src.exception import CustomException
from src.logger import logging
from src.utils import load_obj
import pandas as pd

class PredictPipeline:
    def __init__(self):
        pass

    def predict_value(self,features):
        try:
            preprocessor_path = os.path.join('artifacts','preprocessor.pkl')

            model_path = os.path.join('artifacts','model.pkl')

            preprocessor = load_obj(preprocessor_path)
            
            model = load_obj(model_path)

            scaled_data = preprocessor.transform(features)

            predicted_value = model.predict(scaled_data)

            return predicted_value

        
        except Exception as e:
            logging.info('There is some issue at predict values')
            raise CustomException(e,sys)

class CustomData:
    def __init__(self,
                 age:int,
                 gender:str,
                 location:str,
                 subscription_length_months:int,
                 monthly_bill:float,
                 total_usage_gb:int):
        
        self.age=age
        self.gender=gender
        self.location=location
        self.subscription_length_months=subscription_length_months
        self.monthly_bill=monthly_bill

    
    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict =  {
                'Age':[self.age],
                'Gender':[self.gender],
                'Location':[self.location],
                'Subscription_Length_Months':[self.subscription_length_months],
                'Monthly_Bill':[self.monthly_bill],
                'Total_Usage_GB':[self.pay_1],
        
            }

            df = pd.DataFrame(custom_data_input_dict)

            logging.info('Data converted into DF')

            return df


        except Exception as e:
            logging.info('There is some issue at get data as dataframe')
            raise CustomException(e, sys)
        