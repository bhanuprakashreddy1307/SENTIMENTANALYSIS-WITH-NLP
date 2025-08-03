from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Sample customer reviews and their sentiments (1 = Positive, 0 = Negative)
reviews = [
    "I loved the product! It was amazing.",
    "Horrible experience. Never buying again.",
    "Great quality and fast shipping.",
    "Terrible, the item broke on the first day.",
    "Absolutely fantastic! Highly recommend.",
    "Worst purchase ever.",
    "The service was outstanding.",
    "Very disappointed with this product.",
    "Really satisfied with my order.",
    "It was a waste of money.",
    "Excellent customer support.",
    "Bad packaging and poor quality.",
    "The product is okay, not great.",
    "I’m very happy with my purchase.",
    "This is not worth the price.",
    "Love it! Will buy again.",
    "It stopped working after a week.",
    "Perfect condition and quick delivery.",
    "Product didn't match the description.",
    "Very good quality for the price."
]

labels = [
    1, 0, 1, 0, 1,
    0, 1, 0, 1, 0,
    1, 0, 0, 1, 0,
    1, 0, 1, 0, 1
]

# 1. Split the data
X_train, X_test, y_train, y_test = train_test_split(reviews, labels, test_size=0.2, random_state=42)

# 2. Vectorize the text using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 3. Train Logistic Regression
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# 4. Predict and evaluate
y_pred = model.predict(X_test_tfidf)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 5. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
