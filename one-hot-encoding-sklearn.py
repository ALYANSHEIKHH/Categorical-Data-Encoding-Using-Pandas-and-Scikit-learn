from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

dataset_1 = pd.read_csv("Salary_Dataset.csv")
ct = ColumnTransformer(transformers = [('encoding', OneHotEncoder(), [0])], remainder='passthrough')

dataset_1 = pd.DataFrame(ct.fit_transform(dataset_1))

# print(dataset_1)

dataset_1.columns = ['Australia', 'Canada','Dubai', 'USA','YearsExperience','Salary','Purchased']

print(dataset_1.head(10))