# KARTHIK-R
NLP mini project
import pandas as pd
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.metrics.pairwise import cosine_similarity

# =========================
# DOWNLOAD NLTK DATA
# =========================
nltk.download('punkt')
nltk.download('stopwords')

# =========================
# SAMPLE DATASET
# =========================
data = {
    'Category': [
        'Electronics', 'Electronics',
        'Fashion', 'Fashion',
        'Books', 'Books',
        'Home Appliances', 'Home Appliances'
    ],

    'Review': [
        'Excellent mobile phone with great battery life',
        'Camera quality is very poor and disappointing',

        'The shirt fabric is soft and comfortable',
        'Size fitting is terrible and cloth quality is bad',

        'Amazing story and very interesting book',
        'Book pages are damaged and printing is poor',

        'Washing machine works perfectly and quietly',
        'Mixer grinder stopped working within one week'
    ],

    'Sentiment': [
        'Positive',
        'Negative',

        'Positive',
        'Negative',

        'Positive',
        'Negative',

        'Positive',
        'Negative'
    ]
}

# CREATE DATAFRAME
df = pd.DataFrame(data)

# =========================
# TEXT PREPROCESSING
# =========================
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess_text(text):

    # LOWERCASE
    text = text.lower()

    # REMOVE PUNCTUATION
    text = text.translate(str.maketrans('', '', string.punctuation))

    # TOKENIZATION
    tokens = word_tokenize(text)

    # REMOVE STOPWORDS + STEMMING
    filtered_tokens = []

    for word in tokens:
        if word not in stop_words:
            stemmed_word = stemmer.stem(word)
            filtered_tokens.append(stemmed_word)

    return " ".join(filtered_tokens)

# APPLY PREPROCESSING
df['Processed_Review'] = df['Review'].apply(preprocess_text)

# =========================
# FEATURE EXTRACTION
# =========================
vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(df['Processed_Review'])

y = df['Sentiment']

# =========================
# TRAIN TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# =========================
# TRAIN NAIVE BAYES MODEL
# =========================
model = MultinomialNB()

model.fit(X_train, y_train)

# =========================
# MODEL TESTING
# =========================
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n===================================")
print("MODEL ACCURACY")
print("===================================")

print("Accuracy :", round(accuracy * 100, 2), "%")

# =========================
# USER INPUT REVIEW
# =========================
print("\n===================================")
print("MULTI-CATEGORY REVIEW ANALYZER")
print("===================================")

user_review = input("Enter a product review: ")

# PREPROCESS USER REVIEW
processed_input = preprocess_text(user_review)

# VECTORIZE INPUT
input_vector = vectorizer.transform([processed_input])

# =========================
# SENTIMENT PREDICTION
# =========================
prediction = model.predict(input_vector)

print("\n===================================")
print("PREDICTED SENTIMENT")
print("===================================")

print("Sentiment :", prediction[0])

# =========================
# SIMILAR REVIEW RETRIEVAL
# =========================
similarity_scores = cosine_similarity(input_vector, X)

# GET MOST SIMILAR REVIEW INDEX
similar_index = similarity_scores.argmax()

print("\n===================================")
print("MOST SIMILAR PRODUCT REVIEW")
print("===================================")

print("Category :", df.iloc[similar_index]['Category'])
print("Review    :", df.iloc[similar_index]['Review'])
print("Sentiment :", df.iloc[similar_index]['Sentiment'])

# =========================
# TOP 3 SIMILAR REVIEWS
# =========================
print("\n===================================")
print("TOP 3 SIMILAR REVIEWS")
print("===================================")

scores = list(enumerate(similarity_scores[0]))

# SORT BY SIMILARITY
sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)

for i in sorted_scores[:3]:

    index = i[0]
    score = i[1]

    print("\nReview :", df.iloc[index]['Review'])
    print("Category :", df.iloc[index]['Category'])
    print("Sentiment :", df.iloc[index]['Sentiment'])
    print("Similarity Score :", round(score, 2))

# =========================
# END OF PROJECT
# =========================
print("\n===================================")
print("PROJECT COMPLETED SUCCESSFULLY")
print("===================================")
