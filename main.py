from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.feature_selection import RFECV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
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
mnb = MultinomialNB(fit_prior=True, class_prior=None)
# bnb = BernoulliNB(alpha=1.0,binarize=.0,fit_prior=True,class_prior=None)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .30, random_state=0)

#### Normal Vectorizer Test ####
cv = CountVectorizer()
# X_train = cv.fit_transform(X_train)
# X_test = cv.transform(X_test)
# print(cv.inverse_transform(X_test))

#### TFIDF Vectorizer Test ####
tfidf = TfidfVectorizer(stop_words=None)
X_train = tfidf.fit_transform(X_train)
X_test = tfidf.transform(X_test)
# print(tfidf.inverse_transform(X_test))

# print(y_test)

# mnb.fit(X_train, y_train)
mnb_ovr = OneVsRestClassifier(mnb, n_jobs=1).fit(X_train, y_train)
print("mnb trained")
# pred = mnb.predict(X_test)
# print(pred)
# mlp = MLPClassifier(hidden_layer_sizes=(30,15),activation="logistic",solver='adam',alpha=0.0001,batch_size='auto',learning_rate="constant",learning_rate_init=0.001,\
#     power_t=0.5, max_iter=25, shuffle=True, random_state=None, tol=1e-4, verbose=True, warm_start=False, momentum=0.9, nesterovs_momentum=True, \
#     early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
mlp = MLPClassifier(hidden_layer_sizes=(30, 20),activation="relu",solver='adam',alpha=0.0001,batch_size='auto',learning_rate="adaptive",learning_rate_init=0.0003,\
    power_t=0.5, max_iter=5, shuffle=True, random_state=None, tol=1e-4, verbose=True, warm_start=True, momentum=0.9, nesterovs_momentum=True, \
    early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
mlp.fit(X_train, y_train)
print("mlp trained")

# mnb_score = mnb_ovr.score(X_test, y_test)
mlp_score = mlp.score(X_test, y_test)
# print('mnb:', mnb_score)
print('mlp score:', mlp_score)

# mlp = MLPClassifier(hidden_layer_sizes=(30, 15),activation="relu",solver='adam',alpha=0.0001,batch_size='auto',learning_rate="adaptive",learning_rate_init=0.001,\
#     power_t=0.5, max_iter=50, shuffle=True, random_state=None, tol=1e-4, verbose=False, warm_start=True, momentum=0.9, nesterovs_momentum=True, \
#     early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
# mlp.fit(X_train, y_train)
# print("mlp50 trained")

# mlp_score = mlp.score(X_test, y_test)
# print('mlp 50 iter:', mlp_score)


# mlp = MLPClassifier(hidden_layer_sizes=(30, 15),activation="relu",solver='adam',alpha=0.0001,batch_size='auto',learning_rate="adaptive",learning_rate_init=0.001,\
#     power_t=0.5, max_iter=100, shuffle=True, random_state=None, tol=1e-4, verbose=False, warm_start=True, momentum=0.9, nesterovs_momentum=True, \
#     early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
# mlp.fit(X_train, y_train)
# print("mlp50 trained")

# mlp_score = mlp.score(X_test, y_test)
# print('mlp 100 iter:', mlp_score)

# mlp = MLPClassifier(hidden_layer_sizes=(30, 15),activation="relu",solver='adam',alpha=0.0001,batch_size='auto',learning_rate="adaptive",learning_rate_init=0.001,\
#     power_t=0.5, max_iter=200, shuffle=True, random_state=None, tol=1e-4, verbose=False, warm_start=True, momentum=0.9, nesterovs_momentum=True, \
#     early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
# mlp.fit(X_train, y_train)
# print("mlp50 trained")

# mlp_score = mlp.score(X_test, y_test)
# print('mlp 200 iter:', mlp_score)