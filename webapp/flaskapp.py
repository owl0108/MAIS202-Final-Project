from flask import Flask, render_template, request
from wtforms import Form, TextAreaField, validators
import re
import numpy as np
import joblib
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 

# loading pickle files
clf = joblib.load("pkl_objects/multi_NB.pkl")
model_vocab = joblib.load("pkl_objects/model_vocab.pkl")

################################
# List of helper functions
def clean(review):
    '''(str)->str
    remove punctuation, numbers
    '''
    review = review.lower() # lowercase
    review = re.sub("[^A-Za-z ]", '', review) #remove non-alphabetic characters but preserve apostrophe
    return review

def convert_tag(tag):
    '''(str)->str
    convert a tag
    '''
    if tag.startswith('V'): string = 'v'
    elif tag.startswith('J'): string = 'a'
    elif tag.startswith('R'): string = 'r'
    else: string = 'n'
    return string

def tokenize(review):
    '''(str)->list
    tokenization and clean each token
    '''
    tokenized = word_tokenize(review)
    tagged = nltk.pos_tag(tokenized)
    new_tokenized = list()
    for word in tokenized:
        new_tokenized.append(clean(word))
    return new_tokenized

def rmv_stopwd(tokenized):
    '''(list) -> list
    Remove stop words
    '''
    stop_list = stopwords.words('english')
    return [word for word in tokenized if word not in stop_list]

def lemmatize(tokenized):
    '''(list)->list
    Lemmatize
    '''
    tagged = nltk.pos_tag(tokenized)
    lemmatizer = WordNetLemmatizer() 
    lmt = list()
    for word, tag in tagged:
        new_tag = convert_tag(tag)
        lemmatized = lemmatizer.lemmatize(word, pos=new_tag)
        lmt.append(lemmatized)
    return lmt

def get_vocab(review):
    tokenized = tokenize(review)
    tokenized = rmv_stopwd(tokenized)
    tokenized = list(filter(None, tokenized))
    tokenized = lemmatize(tokenized)
    return tokenized

def get_bow(review_vocab, model_vocab):
  bow_list = []
  for word in model_vocab:
    if word in review_vocab:
      bow_list.append(1)
    else:
      bow_list.append(0)
  return bow_list

def get_BoW(review):
  '''(str) -> np.array
  Takes string as input and returns a BoW expression
  '''
  vocab = get_vocab(review)
  return np.array(get_bow(vocab, model_vocab)).reshape(1,-1)

def classify(review):
    label = {0:'negative', 1:'somewhat negative', 2:'neutral', 3:'somewhat positive', 4:'positive'}
    BoW = get_BoW(review)
    prediction = clf.predict(BoW)[0]
    return label[prediction]
##############################

# starting the app
app = Flask(__name__)

# class for input validation
class ReviewForm(Form):
    moviereview = TextAreaField('', [validators.DataRequired(), validators.length(min=15)])

@app.route('/')
def index():
    form = ReviewForm(request.form)
    return render_template('reviewform.html', form=form)

@app.route('/results', methods=['POST'])
def results():
    form = ReviewForm(request.form)
    if request.method == "POST" and form.validate():
        review = request.form["moviereview"]
        pred_txt = classify(review)

        return render_template("results.html",content=review,\
         prediction=pred_txt)
    
    return render_template('reviewform.html', form=form)

if __name__ == '__main__':
    app.run(debug=True)
    

