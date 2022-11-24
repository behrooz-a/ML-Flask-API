#!/usr/bin/env python

from flask import Flask, render_template, request
import numpy as np
import pickle

model = pickle.load(open("./ML_model.pkl", "rb"))
app = Flask(__name__)


@app.route("/")
def web():
    return render_template("web.html")


@app.route(
    "/Prediction", methods=["POST"]
)  # this line will be activated when Submit button has been
def Prediction():

    height_strings = request.form.get("height").split(" ")

    # Option 1: Use map() to generate a list of floats
    # height_floats = map(float, height_strings)

    # Option 2: Use foreach to generate a list of floats
    height_floats = []
    for height_string in height_strings:
        height_floats.append(float(height_string))

    x = np.array(list(height_floats))
    predicted_weights = model.predict(x.reshape(1, -1))

    return render_template(
        "web.html", result="You entered heights of {}. The predicted weights are {}kg.".format(height_floats, predicted_weights)
    )


if __name__ == "__main__":
    app.run(debug=True)
