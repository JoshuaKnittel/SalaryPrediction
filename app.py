import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import locale
locale.setlocale(locale.LC_ALL, '')

# Create flask app
flask_app = Flask(__name__)

# Load preprocessor and model
preprocessor = pickle.load(open("model_and_preprocessor\preprocessing_steps.pkl", "rb"))
model = pickle.load(open("model_and_preprocessor\model.pkl", "rb"))


@flask_app.route("/")
def Home():
    return render_template("index.html")


@flask_app.route("/predict", methods = ["POST"])
def predict():
    features = [x for x in request.form.values()] # save the inputs of the user in a list
    features[0] = float(features[0]) # convert first value (years of experience) to float
    features[1] = float(features[1]) # convert cluster assignment to float
    # Create a dataframe with the input
    df_pred = pd.DataFrame(columns = ['YearsCodePro', 'Cluster', 'EdLevel', 'OpSys', 'Age', 'Country_reduced', 'OrgSize'])
    df_pred.loc[0] = features
    # Apply preprocessing on input
    preprocessed_features = preprocessor.transform(df_pred)
    # Make prediction for input
    prediction = model.predict(preprocessed_features)

    # Display prediction of the model in HTML
    return render_template("index.html", prediction_text = f"The predicted salary is ${int(prediction[0]):n}")

if __name__ == "__main__":
    flask_app.run(debug=True)