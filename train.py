import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
from sklearn.model_selection import train_test_split

data = pd.read_csv("data/IMDB Dataset.csv")

reviews = data["review"]
sentiments = data["sentiment"]


reviews_train, reviews_test, y_train, y_test = train_test_split(
    reviews, 
    sentiments, 
    test_size=0.25, 
    random_state=42
)

vectorizer = CountVectorizer()

X_train = vectorizer.fit_transform(reviews_train)
X_test = vectorizer.transform(reviews_test)


model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)

print("Accuracy:", accuracy)


joblib.dump(model, "sentiment_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")



