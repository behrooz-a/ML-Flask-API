#!/usr/bin/env python

import xgboost as xgb
from flask import Flask, render_template, request
import numpy as np
import pickle

model = pickle.load(open('xbg_model.sav', 'rb'))
app = Flask(__name__)

@app.route("/")
def web():
    return render_template("web.html")

@app.route(
    "/Prediction", methods=["POST"])  # this line will be activated when Submit button has been
def Prediction():

    height_strings = request.form.get("height")
    height_floats = list(map(int,height_strings.rstrip().split()))

    predicted_weight=model.predict([height_floats])
    if predicted_weight==0:
        BMI='severely underweight'
    elif predicted_weight==1:
        BMI='Weak'
    elif predicted_weight==2:
        BMI='Normal'
    elif predicted_weight==3:
        BMI='Overweight'
    elif predicted_weight==4:
        BMI='Obesity'
    else:
        BMI='Extreme obesity'

    return render_template(
        "web.html", result="Your BMI Index is {} and your weight status is {}.".format(predicted_weight, BMI)
    )


if __name__ == "__main__":
    app.run(debug=True)
