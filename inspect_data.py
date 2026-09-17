import pandas as pd

data = pd.read_csv("data/IMDB Dataset.csv")

print(data.head())
print(data.shape)
print(data.columns)
print(data["sentiment"].value_counts())
print(data.isnull().sum())