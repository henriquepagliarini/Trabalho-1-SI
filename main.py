import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

DATA_PATH = './datasets/vict/10000v/data.csv'

df = pd.read_csv(DATA_PATH)

x = df[['idade', 'fc', 'fr', 'pas', 'spo2', 'temp', 'pr', 'sg', 'fx', 'queim']]
y = df['tri']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42, shuffle=True, stratify=y)

parameters = [
    {
        'criterion': ['entropy'],
        'max_depth': [3],
        'min_samples_leaf': [50]
    },
    {
        'criterion': ['entropy'],
        'max_depth': [10],
        'min_samples_leaf': [20]
    },
    {
        'criterion': ['entropy'],
        'max_depth': [30],
        'min_samples_leaf': [1]
    }
]

model = DecisionTreeClassifier(random_state=42)

n_folds = 5;

clf = GridSearchCV(model, parameters, cv=n_folds, scoring='f1_macro', verbose=4)

clf.fit(x_train, y_train)

all_models = []

res = clf.cv_results_