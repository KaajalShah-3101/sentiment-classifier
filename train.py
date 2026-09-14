import csv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
from sklearn.model_selection import train_test_split

reviews = []
sentiments = []

with open("data/reviews.csv", "r") as file:
    reader = csv.DictReader(file)
    
    
    for row in reader:
        reviews.append(row["review"])
        sentiments.append(row["sentiment"])


reviews_train, reviews_test, y_train, y_test = train_test_split(
    reviews, 
    sentiments, 
    test_size=0.25, 
    random_state=42
)

vectorizer = CountVectorizer()

X_train = vectorizer.fit_transform(reviews_train)
X_test = vectorizer.transform(reviews_test)


model = LogisticRegression()

model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)

print("Accuracy:", accuracy)

predictions = model.predict(X_test)

for review, prediction, actual in zip(reviews_test, predictions, y_test):
    print()
    print("Review:", review)
    print("Prediction:", prediction)
    print("Actual:", actual)

joblib.dump(model, "sentiment_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")



