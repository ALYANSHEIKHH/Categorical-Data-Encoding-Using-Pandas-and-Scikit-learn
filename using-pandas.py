import pandas as pd 
import numpy as np


dataset = pd.read_csv("Salary_Dataset.csv")

#One Hot Encoding using pandas
country_dummies = pd.get_dummies(dataset['country'])

print(country_dummies.head())


dataset = pd.concat([dataset, country_dummies], axis = 1)

print(dataset.head())

dataset.drop('country', axis = 1, inplace=True)
dataset = dataset[['Australia', 'Canada','Dubai', 'USA','YearsExperience','Salary','Purchased']]


#In pandas we do label incude using properties like .cat.codes
dataset['Purchased'] = dataset['Purchased'].astype('category').cat.codes

print(dataset.head())
