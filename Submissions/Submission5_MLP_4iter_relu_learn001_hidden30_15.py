from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

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

''' Multi Layer Perceptron '''
# mlp = MLPClassifier(hidden_layer_sizes=(30,15),activation="logistic",solver='adam',alpha=0.0001,batch_size='auto',learning_rate="constant",learning_rate_init=0.001,\
#     power_t=0.5, max_iter=25, shuffle=True, random_state=None, tol=1e-4, verbose=True, warm_start=False, momentum=0.9, nesterovs_momentum=True, \
#     early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-8)

mlp = MLPClassifier(hidden_layer_sizes=(30, 15, 15, 15),activation="relu",solver='adam',alpha=0.0001,batch_size='auto',learning_rate="constant",learning_rate_init=0.001,\
    power_t=0.5, max_iter=4, shuffle=True, random_state=None, tol=1e-4, verbose=True, warm_start=True, momentum=0.9, nesterovs_momentum=True, \
    early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
mlp.fit(X, y)
print(mlp.score(X, y))


kaggle_results = mlp.predict_proba(X_kaggle)
print(kaggle_results)

ids = kaggle_test_df['id']
predict_data = pd.DataFrame(kaggle_results,  columns=['EAP', 'HPL', 'MWS'])
conc = pd.concat([ids, predict_data], axis=1)
conc.to_csv('mySubmission.csv', index=False)


"""
''' Grid Search '''
clf = MultinomialNB(fit_prior=True, alpha=.01)
alpha_range = [i for i in np.arange(0.000,1.0,.015)]
prior = [True, False] # Taken out after initial grid search so we wcan graph easier
grid_clf = GridSearchCV(clf, {'alpha': alpha_range},scoring='accuracy', n_jobs=1, cv=100)
grid_clf.fit(X, y)

''' Processing Results '''
mean_scores = [result.mean_validation_score for result in grid_clf.grid_scores_]
plt.plot(alpha_range, mean_scores)
plt.ylabel('Mean Score (CV=100)')
plt.xlabel('Alpha')
plt.show()
print(grid_clf.best_score_, grid_clf.best_params_)


''' One Vs Rest Classification from Grid Search Result '''
mnb_ovr = OneVsRestClassifier(grid_clf.best_estimator_, n_jobs=1)
mnb_ovr.fit(X, y)
kaggle_results = mnb_ovr.predict_proba(X_kaggle)
print(kaggle_results)
"""