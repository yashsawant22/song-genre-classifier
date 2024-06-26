
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer

wnl = WordNetLemmatizer()

def tokenizer(x): # custom tokenizer
    return (
        wnl.lemmatize(w) 
        for w in word_tokenize(x) 
        if len(w) > 2 and w.isalnum() # only words that are > 2 characters
    ) 
