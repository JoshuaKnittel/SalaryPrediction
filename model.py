# Import packages
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
from sklearn.neural_network import MLPRegressor

# Read data
df = pd.read_csv("survey_results_public.csv",index_col=0) # dataset


# 1) Clean Data

df = df.loc[df["ConvertedCompYearly"].notnull(),:] # remove null values for target
df = df.loc[(df["ConvertedCompYearly"]<=df["ConvertedCompYearly"].quantile(0.95)) & (df["ConvertedCompYearly"]>=df["ConvertedCompYearly"].quantile(0.01)),:] # remove outliers
# Encode "Less than 1 year" with 0
df.loc[df["YearsCodePro"]=="Less than 1 year","YearsCodePro"] = 0
# Encode "More than 50 years" with 50
df.loc[df["YearsCodePro"]=="More than 50 years","YearsCodePro"] = 50
# Change type to float
df["YearsCodePro"]= df["YearsCodePro"].astype(float)


# 2 ) Feature engineering

# Function to reduce a feature to X most frequent categories (number_of_categories in function).
def reduce_feature(df, feature, number_of_categories):
    
    # Get the names of the X most frequent categories
    categories_to_keep = list(df.groupby(feature).size().sort_values(ascending=False).head(number_of_categories).index)
    
    # Create a function which checks, if the value is in the most frequent categories.
    def reduce(row):
        if row in categories_to_keep:
            return row
        else:
            return "Other" # Encode everything else as "other"
    
    # Apply the function on the dataframe
    df[feature+"_reduced"] = df[feature].apply(reduce)
    
    return df

# Country
# Reduce "Country" to 15 most frequent categories. 
df = reduce_feature(df, "Country", 15)

# Tech stack
# Apply Clustering on this variable. Firs we create tfidf matrix. Then we apply kmeans on this matrix. 
used_tools = [
    "LanguageHaveWorkedWith",
    "DatabaseHaveWorkedWith",
    "MiscTechHaveWorkedWith", 
    "WebframeHaveWorkedWith",
    "ToolsTechHaveWorkedWith", 
    "PlatformHaveWorkedWith"]
# Fill NAs with empty strings
df[used_tools] = df[used_tools].fillna("")
# Create new variable, which contains all languages, databases,... for a developer. 
df["all_tools"] = df["LanguageHaveWorkedWith"] + ";" + df["DatabaseHaveWorkedWith"] + ";" + df["MiscTechHaveWorkedWith"] + ";" + df["WebframeHaveWorkedWith"] + ";" + df["ToolsTechHaveWorkedWith"] + ";" + df["PlatformHaveWorkedWith"] 
df["all_tools"] = df["all_tools"].str.replace(r"[;]+", ";", regex=True)
df["all_tools"] = df["all_tools"].str.replace(" ", "")
# Replace the semicolons with spaces
stack = list(df["all_tools"].str.replace(";"," "))
# Create tfidf-matrix
tfidfvectorizer = TfidfVectorizer()
vector = tfidfvectorizer.fit_transform(stack)
tfidf_matrix = pd.DataFrame(vector.toarray(), columns = tfidfvectorizer.get_feature_names())
kmeans = KMeans(n_clusters = 7, random_state=1).fit(tfidf_matrix)
clusters = kmeans.fit_predict(tfidf_matrix)
# Put cluster assignment into dataframe
df["Cluster"] = clusters


# 3) Data preprocessing

# Choose relevant features
categorical_features = [
    'Cluster',
    'EdLevel',
    'OpSys',
    'Age',
    'Country_reduced',
    "OrgSize"
]
numerical_features = ['YearsCodePro']
features = numerical_features + categorical_features
X = df[features]
y = df["ConvertedCompYearly"]

# Create a train and a test set.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Define pipelines for categorical and numerical features
categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')), # first, impute the na's as "missing"
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))]) # then, one-hot encode the categorical features
numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer()), # first, impute missing value with mean (this is the default)
        ('scaler', StandardScaler())]) # then, standardize the features. 

# Define column transformer
preprocessor = ColumnTransformer(
        transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)])

# Prepare data
X_train = X_train[features]
X_test = X_test[features]

# Preprocess the data from the train set
X_train = preprocessor.fit_transform(X_train)

# Preprocess the data from the test set
X_test = preprocessor.transform(X_test)
pickle.dump(preprocessor, open("model_and_preprocessor\preprocessing_steps.pkl", "wb")) # dump the preprocessor into pkl-file (will be loaded in app.py-file)


# 4) Regression Analysis

# Train a simple neural network with 4 hidden layers
neural_network = MLPRegressor(random_state=1, hidden_layer_sizes=(25,25,25,25), max_iter=200, verbose = True).fit(X_train, y_train) # fit the model
pickle.dump(neural_network, open("model_and_preprocessor\model.pkl", "wb")) # dump the trained model into pkl-file (will be loaded in app.py-file)

