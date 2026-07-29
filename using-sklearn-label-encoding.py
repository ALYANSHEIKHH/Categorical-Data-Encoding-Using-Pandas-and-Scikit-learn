import pandas as pd 

dataset = pd.read_csv("Salary_Dataset.csv")

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

# Label Encoding
dataset['Purchased'] = le.fit_transform(dataset['Purchased'])

print(dataset.head())











