
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report


# STEP 1: Load the external dataset
df = pd.read_csv("spam.csv", encoding="latin-1")

# Kaggle dataset commonly uses v1 for labels and v2 for messages
df = df[["v1", "v2"]]
df.columns = ["label", "message"]

df = df.dropna()
df["label"] = df["label"].str.strip().str.lower()
df["message"] = df["message"].astype(str).str.strip()

df = df[df["label"].isin(["spam", "ham"])]
df = df.drop_duplicates(subset=["message"])

print("Dataset loaded successfully!")
print("Total messages:", len(df))
print(df["label"].value_counts())


# STEP 2: Separate input messages and target labels
X = df["message"]
y = df["label"]


# STEP 3: Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# STEP 4: Create the ML pipeline
model = Pipeline([
    ("vectorizer", CountVectorizer()),
    ("classifier", MultinomialNB())
])


# STEP 5: Train using the external dataset
model.fit(X_train, y_train)

print("\nModel training completed!")


# STEP 6: Evaluate on unseen test messages
predictions = model.predict(X_test)

print("\nModel accuracy:",
      round(accuracy_score(y_test, predictions) * 100, 2), "%")

print("\nClassification report:")
print(classification_report(y_test, predictions))


# STEP 7: Save the trained model
joblib.dump(model, "spam_model.joblib")

print("Trained model saved as spam_model.joblib")


# STEP 8: Load the saved trained model
trained_model = joblib.load("spam_model.joblib")


# STEP 9: Predict new messages
while True:
    message = input(
        "\nEnter a message (or type 'exit' to quit): "
    ).strip()

    if message.lower() == "exit":
        print("Application closed.")
        break

    if not message:
        print("Please enter a message.")
        continue

    prediction = trained_model.predict([message])[0]
    probabilities = trained_model.predict_proba([message])[0]
    classes = trained_model.classes_

    confidence = probabilities[list(classes).index(prediction)] * 100

    if prediction == "spam":
        print("Prediction: SPAM")
    else:
        print("Prediction: NOT SPAM (HAM)")

    print(f"Model confidence: {confidence:.2f}%")
