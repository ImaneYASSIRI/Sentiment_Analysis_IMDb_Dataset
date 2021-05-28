"""
question 1 :
ce fichier app.py contient les tâches du preprocessing ajouteés + CountVectorizer + Naive Bayes
"""
import pandas as pd
import re
from flask import Flask, render_template, url_for, request
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
# from sklearn.externals import joblib

app = Flask(__name__)

# Machine Learning code goes here
@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    df = pd.read_csv("Data/IMDB Dataset.csv")
    df = df[:500]
    df_data = df[['review', 'sentiment']]
    # Features and Labels
    df_x = df_data['review']
    df_y = df_data.sentiment
    # Extract the features with countVectorizer
    corpus = df_x

    # -> Preprocessing
    processed_reviews_list = []
    for review in corpus:
        processed_review = review.lower()
        processed_review = re.sub('[^a-zA-Z]', ' ', processed_review)
        processed_review = re.sub(r'\s+', ' ', processed_review)
        processed_reviews_list.append(processed_review)
    all_reviews_words = []
    lemmmatizer = WordNetLemmatizer()
    for text in processed_reviews_list:
        data = text
        words = word_tokenize(data)
        words = [lemmmatizer.lemmatize(word.lower()) for word in words if
                 (not word in set(stopwords.words('english')) and word.isalpha())]
        all_reviews_words.append(words)
    #all_reviews_words

    # -> Extract the features with countVectorizer

    # fit.transform() accept comment argument : series. On doit donc changer le type de all_texts_words
    series = pd.Series(all_reviews_words).astype(str)
    cv = CountVectorizer()
    X = cv.fit_transform(series)
    X_train, X_test, y_train, y_test = train_test_split(X, df_y, test_size=0.33, random_state=42)

    # -> Naive Bayes

    clf = MultinomialNB()
    clf.fit(X_train, y_train)
    clf.score(X_test, y_test)

    if request.method == 'POST':
        comment = request.form['comment']
        data = [comment]
        vect = cv.transform(data).toarray()
        my_prediction = clf.predict(vect)
    return render_template('result.html', prediction=my_prediction)


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5555, debug=True)
