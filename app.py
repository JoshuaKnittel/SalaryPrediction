import numpy as np
from flask import Flask, request, render_template
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

    # Prepare dataframe for returning inputs to user
    # Rename columns
    df_pred.rename(columns={
        "YearsCodePro": "Years of working Experience",
        "Cluster": "Job Title",
        "EdLevel": "Education Level",
        "OpSys": "Operating System",
        "Country_reduced": "Country",
        "OrgSize": "Organization Size"
    }, inplace=True)
    # Map cluster assignment to job title
    job_titles = {
        0: "Java Developer",
        1: "Ruby Developer",
        2: "Web Developer",
        3: "Data Scientist",
        4: ".NET Developer",
        5: "C Developer",
        6: "Full-Stack Developer"
        }  
    df_pred.replace({"Job Title": job_titles}, inplace=True)
    # Save working experience as float and then transform it to string with " years" added. 
    df_pred["Years of working Experience"] = df_pred["Years of working Experience"].astype(int)
    df_pred["Years of working Experience"] = df_pred["Years of working Experience"].apply(lambda x: str(x) + " year(s)")
    # Save input in dictionary
    data = df_pred.loc[0].to_dict()

    # Display prediction of the model and input in HTML template
    return render_template("index.html", prediction_text = f"The predicted salary is ${int(prediction[0]):n}", data = data, title_query = f"Chosen Values:")

if __name__ == "__main__":
    flask_app.run(debug=True)