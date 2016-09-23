# in order to run tests put in into the directiry prior to LogisticRegression
# directory

import pandas as pd
from LogisticRegression import LogisticRegression

df = pd.read_csv('iris.csv')
df[df.columns[4]] = df[df.columns[4]].map({'Iris-setosa': 0,
                               'Iris-versicolor': 1, 'Iris-virginica': 2})
print(df.head())

clf = LogisticRegression.LogisticRegression(n_iter=500)
print('\n Fitting...\n')
clf.fit(df[df.columns[:-1]].values, df[df.columns[-1]].values)
print('Predicting... \n\n')
prd = clf.predict(df[df.columns[:-1]].values)

for r, p in zip(df[df.columns[-1]].values, prd):
    print(r, p)
