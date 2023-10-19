from flask import Flask,render_template,request,jsonify
from src.pipelines.prediction_pipeline import CustomData,PredictPipeline
from src.exception import CustomException
from src.logger import logging

application = Flask(__name__)

app = application

@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/predict',methods= ['GET','POST'])
def predict_datapoint():
    try:
        if request.method == 'GET':
            return render_template('form.html')

        else:
            data = CustomData(
                age=int(request.form.get('age')),
                gender = request.form.get('gender'),
                location = request.form.get('location'),
                subscription_length_months = int(request.form.get('subscription_length_months')),
                monthly_bill = float(request.form.get('monthly_bill')),
                total_usage_gb = int(request.form.get('total_usage_gb')))

            final_data = data.get_data_as_dataframe()
            predict_pipeline_obj = PredictPipeline()
            pred_value = predict_pipeline_obj.predict_value(final_data)

            if pred_value[0] == 0:
                results = ' not '
            elif pred_value[0]  == 1:
                results = ' '

            return render_template('results.html',final_result = results)

    except  Exception as e:
        logging.info('There is some issue at predict_datapoint')
        raise CustomException(e,sys)



if __name__ == "__main__":
    app.run(host='0.0.0.0')