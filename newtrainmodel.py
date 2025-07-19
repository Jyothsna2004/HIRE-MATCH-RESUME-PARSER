# import pandas as pd
# import re
# import pickle
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.preprocessing import LabelEncoder
# from sklearn.svm import SVC
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report, accuracy_score
# from sentence_transformers import SentenceTransformer

# # Load dataset
# df = pd.read_csv("Resume/Resume.csv")

# # Step 1: Clean text
# def clean_text(text):
#     text = re.sub(r"http\S+", " ", text)
#     text = re.sub(r"[^a-zA-Z]", " ", text)
#     text = re.sub(r"\s+", " ", text)
#     return text.strip().lower()

# df["cleaned_text"] = df["Resume_str"].apply(clean_text)

# # Step 2: Encode labels
# le = LabelEncoder()
# df["target"] = le.fit_transform(df["Category"])

# # # Step 3: TF-IDF
# # tfidf = TfidfVectorizer(max_features=1000)
# # X = tfidf.fit_transform(df["cleaned_text"]).toarray()
# # y = df["target"]

# model = SentenceTransformer('all-MiniLM-L6-v2')
# embeddings = model.encode(df["cleaned_text"].tolist(), show_progress_bar=True)
# # # Step 4: Train/test split
# # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# X_train, X_test, y_train, y_test = train_test_split(embeddings, df["target"], test_size=0.2, random_state=42)

# # Step 5: Model training
# clf = SVC(kernel='linear', probability=True)
# clf.fit(X_train, y_train)

# # Step 6: Evaluation
# y_pred = clf.predict(X_test)
# print("Accuracy:", accuracy_score(y_test, y_pred))
# print(classification_report(y_test, y_pred, target_names=le.classes_))

# # Step 7: Save model
# # pickle.dump(model, open("clf.pkl", "wb"))
# # pickle.dump(tfidf, open("tfidf.pkl", "wb"))
# # pickle.dump(le, open("encoder.pkl", "wb"))
# pickle.dump(clf, open("bert_clf.pkl", "wb"))
# pickle.dump(le, open("encoder.pkl", "wb"))
# pickle.dump(model, open("bert_embedder.pkl", "wb"))
import pandas as pd
import re
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sentence_transformers import SentenceTransformer

# Load dataset
df = pd.read_csv("Resume/Resume.txt")
print(df.columns)

# Step 1: Clean text
def clean_text(text):
    # Remove URLs, special characters, and unnecessary spaces
    text = re.sub(r"http\S+", " ", text)  # Remove URLs
    text = re.sub(r"[^a-zA-Z]", " ", text)  # Keep only alphabets
    text = re.sub(r"\s+", " ", text)  # Replace multiple spaces with single space
    return text.strip().lower()  # Remove leading/trailing spaces

df["cleaned_text"] = df["Resume_str"].apply(clean_text)  # Apply cleaning to each resume text

# Step 2: Encode target labels (categories)
le = LabelEncoder()
df["target"] = le.fit_transform(df["Category"])

# Step 3: Generate sentence embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(df["cleaned_text"].tolist(), show_progress_bar=True)

# Step 4: TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=1000)
tfidf_features = tfidf.fit_transform(df["cleaned_text"]).toarray()

# Combine TF-IDF features and Sentence embeddings
import numpy as np
X = np.hstack((embeddings, tfidf_features))

# Step 5: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, df["target"], test_size=0.2, random_state=42)

# Step 6: Model training - Random Forest
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Step 7: Evaluate the model
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Step 8: Cross-validation for better performance estimation
cv_scores = cross_val_score(clf, X, df["target"], cv=5)  # 5-fold cross-validation
print("Cross-validation scores:", cv_scores)
print("Average cross-validation score:", cv_scores.mean())

# Step 9: Save the model and other artifacts
pickle.dump(clf, open("rf_clf.pkl", "wb"))  # Save classifier
pickle.dump(le, open("encoder.pkl", "wb"))  # Save label encoder
pickle.dump(model, open("bert_embedder.pkl", "wb"))  # Save sentence transformer model
pickle.dump(tfidf, open("tfidf.pkl", "wb"))  # Save TF-IDF vectorizer

