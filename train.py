import csv
from sklearn.feature_extraction.text import CountVectorizer

reviews = []
sentiments = []

with open("data/reviews.csv", "r") as file:
    reader = csv.DictReader(file)
    
    
    for row in reader:
        reviews.append(row["review"])
        sentiments.append(row["sentiment"])

vectorizer = CountVectorizer()

X = vectorizer.fit_transform(reviews)

print(vectorizer.get_feature_names_out())
print(X.toarray())

