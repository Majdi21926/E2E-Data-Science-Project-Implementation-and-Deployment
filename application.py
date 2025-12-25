## This script is for flask app^
from doctest import debug
from email.mime import application
import re
from flask import Flask, jsonify, request, render_template
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
import pandas as pd
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

app = Flask(__name__)
application = app

## Route for home page
@app.route('/')
def index():
    return render_template('index.html')

## Route for prediction
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:

        dataObject = CustomData(
            gender = request.form.get('gender'),
            race_ethnicity = request.form.get('race_ethnicity'),
            parental_level_of_education = request.form.get('parental_level_of_education'),
            lunch = request.form.get('lunch'),
            test_preparation_course = request.form.get('test_preparation_course'),
            reading_score = float(request.form.get('reading_score')),
            writing_score = float(request.form.get('writing_score'))
        )
        pred_pd = dataObject.get_data_as_data_frame()
        print(pred_pd)
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_pd)
        return render_template('home.html', results=results[0])
    

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)