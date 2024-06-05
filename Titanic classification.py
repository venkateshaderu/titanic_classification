#Import Libraries:
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
#Load Data:
# Assuming you have a CSV file with the Titanic dataset
df = pd.read_csv(r'C:\Users\User\Desktop\titanic.csv')
df
#Data Preprocessing:
# Handle missing values
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df['Fare'].fillna(df['Fare'].median(), inplace=True)

# Convert categorical features to numerical
df = pd.get_dummies(df, columns=['Sex', 'Embarked', 'Pclass'], drop_first=True)
df
#Feature Selection:
features = df[['Pclass_2', 'Pclass_3', 'Sex_male', 'Age', 'SibSp', 'Parch', 'Fare']]
#Split Data:
X_train, X_test, y_train, y_test = train_test_split(features, df['Survived'], test_size=0.2, random_state=42)
#Build and Train Model:
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
#Make Predictions:
predictions = model.predict(X_test)
#Evaluate Model:
accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy: {accuracy}')
print('Classification Report:')
print(classification_report(y_test, predictions))