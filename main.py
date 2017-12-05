from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import pandas as pd

FILE = 'train.csv'
df = pd.read_csv(FILE)
print(df.head())
# print(df[df.author =='EAP'].head())

df.loc[df['author']=='EAP', 'author'] = 0
df.loc[df['author']=='HPL', 'author'] = 1
df.loc[df['author']=='MWS', 'author'] = 2

X = df['text']
y = df['author'].astype('int')

# Classifier
mnb = MultinomialNB()
# 'Term Frequency–Inverse Document Frequency' Vectorizer

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3)

#### Normal Vectorizer Test ####
cv = CountVectorizer()
X_train = cv.fit_transform(X_train)
X_test = cv.transform(X_test)
# print(cv.inverse_transform(X_test))

#### TFIDF Vectorizer Test ####
tfidf = TfidfVectorizer(min_df=1, stop_words=None)
# X_train = tfidf.fit_transform(X_train)
# X_test = tfidf.transform(X_test)
# print(tfidf.inverse_transform(X_test))

# print(y_test)

mnb.fit(X_train, y_train)
# pred = mnb.predict(X_test)
# print(pred)

score = mnb.score(X_test, y_test)
print(score)