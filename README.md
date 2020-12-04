# Sentiment Predictor for Movie Reviews
This repository stores all the deliverables and codes for the final project of [MAIS 202](https://www.mcgillai.com/mais202) (Fall 2020).

Training data was retrieved from [Kaggle](https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews).

## Project description
The project aims to build a webapp that can predict a sentiment of a given movie review on a scale of 0 to 4:

0 - Negative
1 - Somewhat negative
2 - Neutral
3 - Somewhat positive
4 - Positive

The classifier adopts the multinomial Naive Bayes from scikit-learn and several text preprocesssing steps such as tokenization and lemmatization were impementend using Natural Language Toolkit.
The accuracy of prediction is around 55%.

## Running the app
1. Install all packages in requirements.txt
2. Navigate to the webapp directory
3. Run download.py to download nltk files
4. Run flask.py and open http://localhost:5000/ in your browser.

## Repository organization
1. Deliverables/
    * Deliverables submitted over the course of MAIS202

2. data/
    * Training data retrieved from Kaggle

3. pics/
    * The confusion matrix for the model and screenshots of the app running.

4. webapp/
    * Contains everything necessary to run the webapp.

5. getPredictionModel.ipynb
    * Jupyter Notebook to train the model



