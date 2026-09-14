import csv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

reviews = []
sentiments = []

with open("data/reviews.csv", "r") as file:
    reader = csv.DictReader(file)
    
    
    for row in reader:
        reviews.append(row["review"])
        sentiments.append(row["sentiment"])

vectorizer = CountVectorizer()

X = vectorizer.fit_transform(reviews)

model = LogisticRegression()

model.fit(X, sentiments)

joblib.dump(model, "sentiment_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")



