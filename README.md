'Customer Churn Prediction'

Problem Statement:

To build a classification model to determine whether a person  will churn or not. 


Data Description:

The data contains 100000 instances with the following attributes:
Features:
1.	Age - continuous: the age of customer.
2.	Gender - categorical: the gender of customer Male and Female.
3.	Location - categorical. the city of customer, Los Angeles,New York,Miami,Chicago and Houston.
4.	Subscription_Length_Months - continuous: No of months of subcription
5.	Monthly_Bill - continuous: Monthly bill of customer
6.	Total_Usage_GB - continuous: No of total usage(in gbs)

Target Label:
Whether a customer will chunk or not.
7.	Churn:  Yes = 1, No = 0.



 
Model Training :
1) Data Preprocessing   
   a) Drop columns not useful for training the model. Such columns were selected while doing the EDA(CustomerId and Name).
   b) Check for null values in the columns. If present, impute the null values using the Simple imputer.
   c) Pipeline has been created for categorical and numerical features.
   d) Categorical columns are converted into numerical columns using Ordinal Encoder.
   e) Scale the training and test data separately using Standard Scaler .


2) Model Selection - We calculate the Accuacy scores for different different models and select the model with the best score and we get KNN algorithms 

Prediction Data Description:
 
Client will directly post the data in on application. Data will contain 6 features.


Deployment :

We will be upload the model to the github. 










