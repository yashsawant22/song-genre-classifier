from flask import Flask, request, render_template
import pickle
import nltk
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.utils import shuffle
import pandas as pd
import numpy as np
from text_utils import tokenizer

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')


app = Flask(__name__)

#stop = list(set(stopwords.words('english'))) # stopwords
#wnl = WordNetLemmatizer() # lemmatizer

#def tokenizer(x): # custom tokenizer
#    return (
#        wnl.lemmatize(w) 
#        for w in word_tokenize(x) 
#        if len(w) > 2 and w.isalnum() # only words that are > 2 characters
#    )  

import pickle
with open('genrepredict.pkl', 'rb') as f:
    classifier = pickle.load(f)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        lyrics = request.form['lyrics']
        genre = predict_genre(lyrics)  # Assume this function will handle your ML model
        return render_template('index.html', genre=genre[0])
    return render_template('index.html')


def predict_genre(lyrics):
    result = classifier.predict([lyrics])
    return result


if __name__ == '__main__':
    app.run(debug=True)
