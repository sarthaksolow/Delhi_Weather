import numpy as np
from flask import Flask,request,jsonify,render_template
import pickle

#Create app
app = Flask(__name__)

#Load model
model = pickle.load(open("model.pkl","rb"))

@app.route("/")
def Home():
    return render_template("index.html")

@app.route("/predict", methods = ["POST"])
def predict():
    input_features = [int(x) for x in request.form.values()]
    features = [np.array(input_features)]
    prediction = model.predict(features)

    return render_template("index.html",prediction_text = "Will it rain ? : {}".format(bool(prediction)))

if __name__ == "__main__":
    app.run(debug=True)
