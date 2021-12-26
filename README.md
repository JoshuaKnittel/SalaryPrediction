# Salary prediction
A machine learning model to predict the salaries of software developers. Deployed with Flask.
 
## Machine learning model
A detailed variant of the model is included in the [jupyter notebook](https://github.com/JoshuaKnittel/SalaryPrediction/blob/main/jupyter_notebook/detailed_model.ipynb). This notebook contains comments, visualizations and an error analysis. 

The [model.py](https://github.com/JoshuaKnittel/SalaryPrediction/blob/main/model.py) contains the same model in a shortened form. Based on this file the Flask application is created. 
 
## Deployment of machine learning model with Flask
Simple Flask application, which looks like this:
![app_low](https://user-images.githubusercontent.com/70914456/146842824-5e612972-a453-4f10-99f5-494b8eb5c52d.gif)

To try it out for yourself, follow the steps below:
1. `pip install virtualenv`
2. Create virtual environment: `virtualenv venv`
3. Activate virtual environment: `venv\Scripts\activate`
4. Install the packages from the requirements.txt: `pip install -r requirements.txt`
5. Run the Flask app: `python app.py`
The trained model and preprocessor are already saved in .pkl files. If you want to run or adjust the model or the preprocessor on your own, you can run/edit the [model.py](https://github.com/JoshuaKnittel/SalaryPrediction/blob/main/model.py) file.  

