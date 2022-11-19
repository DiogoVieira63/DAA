import math
import sklearn as skl
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score


from sklearn.linear_model import LinearRegression

from sklearn import metrics
from sklearn.svm import SVC

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder



df = pd.read_csv('music_genre.csv')


df = df.drop(['instance_id','obtained_date','artist_name','track_name'],axis=1)
print(df.nunique())



print(df['tempo'].value_counts())

lb_make = LabelEncoder()

df['tempo'] = df['tempo'].apply(lambda x : 0 if x == '?' else float(x))
df['mode'] = df['mode'].apply(lambda x : 1 if x == 'Major' else 0)
mean = df['tempo'].mean()
df['tempo'] = df['tempo'].apply(lambda x : mean if x == 0 else float(x))

#print(df.isna().sum())
df = df.dropna(axis=0)

df['music_genre'] = lb_make.fit_transform(df['music_genre'])
df['key'] = lb_make.fit_transform(df['key'])

#df = df.drop(['key','music_genre'],axis=1)

print(df.info())

x = df.drop(['music_genre'], axis=1)
y = df['music_genre'].to_frame()


clf = DecisionTreeRegressor(random_state=2022)

#scores = cross_val_score(clf,x,y,cv=10)
scores = cross_val_score(clf, x, y, scoring='neg_mean_squared_error', cv=10)

print( np.sqrt(np.abs(scores.mean())))

X_train,X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=2021)

clf.fit(X_train,y_train)


#print(scores.mean())

predictions = clf.predict(X_test)

file = open("tentativa.csv","w+")


print('MAE:', mean_absolute_error(y_test, predictions))
print('MSE:', mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(mean_squared_error(y_test, predictions)))

