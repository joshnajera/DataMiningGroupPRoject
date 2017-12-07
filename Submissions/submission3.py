from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

''' KAGGLE FILES '''
TRAIN_FILE = 'train.csv'
df = pd.read_csv(TRAIN_FILE)
KAGGLE_TEST_FILE = 'test.csv'
kaggle_test_df = pd.read_csv(KAGGLE_TEST_FILE)

''' Renaming Labels '''
df.loc[df['author']=='EAP', 'author'] = 0
df.loc[df['author']=='HPL', 'author'] = 1
df.loc[df['author']=='MWS', 'author'] = 2

''' Extracting Necessary Things '''
X = df['text']
X_kaggle = kaggle_test_df['text']
y = df['author'].astype('int')

''' Transforming Data For Use '''
tfidf = TfidfVectorizer(stop_words='english')
X = tfidf.fit_transform(X)
X_kaggle = tfidf.transform(X_kaggle)

clf = MultinomialNB(fit_prior=True, alpha=.01)
clf.fit(X, y)
kaggle_results = clf.predict_proba(X_kaggle)
print(kaggle_results)

ids = kaggle_test_df['id']
predict_data = pd.DataFrame(kaggle_results,  columns=['EAP', 'HPL', 'MWS'])
conc = pd.concat([ids, predict_data], axis=1)
conc.to_csv('mySubmission.csv', index=False)