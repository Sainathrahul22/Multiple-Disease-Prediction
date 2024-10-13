import numpy as np 
import pickle
import pandas as pd 

# Load the dataset
df = pd.read_csv(r"D:\\Data_science_Projects\\Multipel-Disease-Prediction-Using-Flask\\Multiple-Disease-Prediction-Using-Flask\\Dataset\\diabetes.csv")

# Rename the DiabetesPredictionFunction column as DPF for easier reference
df = df.rename(columns={'DiabetesPredictionFunction': 'DPF'})

# Replacing the 0 values from ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI'] with NaN (Not a number)
df_copy = df.copy(deep=True)
df_copy[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = df_copy[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.NaN)

# Replacing NaN values by mean or median depending on distribution
df_copy['Glucose'].fillna(df_copy['Glucose'].mean(), inplace=True)
df_copy['BloodPressure'].fillna(df_copy['BloodPressure'].mean(), inplace=True)
df_copy['SkinThickness'].fillna(df_copy['SkinThickness'].median(), inplace=True)
df_copy['Insulin'].fillna(df_copy['Insulin'].median(), inplace=True)
df_copy['BMI'].fillna(df_copy['BMI'].median(), inplace=True)

# Model Building
from sklearn.model_selection import train_test_split

# Define features and target variable
X = df_copy.drop(columns='Outcome')  # Use df_copy here instead of df
y = df_copy['Outcome']                # Use df_copy here instead of df

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Create Random Forest Model
from sklearn.ensemble import RandomForestClassifier

# Initialize the Random Forest Classifier
classifier = RandomForestClassifier(n_estimators=20, random_state=0)  # Added random_state for reproducibility
classifier.fit(X_train, y_train)

# Creating a pickle file for the classifier
filename = 'diabetes-prediction-rfc-model.pkl'
with open(filename, 'wb') as file:
    pickle.dump(classifier, file)  # Using 'with' to ensure proper file closure

print("Successfully saved the model!")
