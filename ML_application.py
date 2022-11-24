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
    height_floats = map(float, height_strings)
    print(
        f"height_strings is a {type(height_strings)}; height_floats is a {type(height_floats)}."
    )
    x = np.array(list(height_floats))
    predicted_weight = model.predict(x.reshape(1, -1))
    print(f"predicted_weight is a {type(predicted_weight)}")

    return render_template(
        "web.html", result="The predicted weights are {}kg".format(predicted_weight)
    )


if __name__ == "__main__":
    app.run(debug=True)
