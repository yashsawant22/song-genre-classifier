from flask import Flask, request, render_template
import pickle
import nltk
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

app = Flask(__name__)

stop = list(set(stopwords.words('english'))) # stopwords
wnl = WordNetLemmatizer() # lemmatizer

def tokenizer(x): # custom tokenizer
    return (
        wnl.lemmatize(w) 
        for w in word_tokenize(x) 
        if len(w) > 2 and w.isalnum() # only words that are > 2 characters
    )  

import pickle
with open('genrepredict.pkl', 'rb') as f:
    classifier = pickle.load(f)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        lyrics = request.form['lyrics']
        genre = predict_genre(lyrics)  # Assume this function will handle your ML model
        return render_template('index.html', genre=genre)
    return render_template('index.html')


def predict_genre(lyrics):
    result = classifier.predict([lyrics])
    return result


if __name__ == '__main__':
    app.run(debug=True)
