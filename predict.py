import joblib

model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

review = ["I hated this movie but the acting was amazing"]

X = vectorizer.transform(review)

prediction = model.predict(X)

print(prediction)