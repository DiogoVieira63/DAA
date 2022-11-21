import math
import sklearn as skl
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
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


df = df.dropna(axis=0)

# length of track name 
df['track_name_length'] = df['track_name'].apply(lambda x: len(x))

# check if remixz in track name
#df['isRemix'] = df['track_name'].apply(lambda x: 1 if 'remix' in x.lower() else 0)
#print(df['isRemix'].value_counts())


df = df.drop(['instance_id','obtained_date','artist_name','track_name'],axis=1)



lb_make = LabelEncoder()

# handle missing values on tempo
df['tempo'] = df['tempo'].apply(lambda x : 0 if x == '?' else float(x))
mean = df['tempo'].mean()

#Label encoding mode
df['mode'] = df['mode'].apply(lambda x : 1 if x == 'Major' else 0)
df['tempo'] = df['tempo'].apply(lambda x : mean if x == 0 else float(x))

#Join genre Rap and Hip-Hop
df['music_genre'] = df['music_genre'].apply(lambda x : 'Rap' if x == 'Hip-Hop' else x)

# handle missing values on duration
mean = df['duration_ms'].mean()
df['duration_ms'] = df['duration_ms'].apply(lambda x : mean if x == -1 else x)


#print(df.isna().sum())

#df['music_genre'] = lb_make.fit_transform(df['music_genre'])
df['key'] = lb_make.fit_transform(df['key'])

#df = df.drop(['key','music_genre'],axis=1)
print(df['duration_ms'].value_counts())



def correlation(df):
    corr_matrix = df.corr() 
    f, ax = plt.subplots(figsize=(12, 16))
    sns.heatmap(corr_matrix, vmin=-1, vmax=1, square=True, annot=True)
    plt.show()




x = df.drop(['music_genre'], axis=1)
y = df['music_genre'].to_frame()

def decisionTree(x,y):
    clf = DecisionTreeClassifier(random_state=2022)

    scores = cross_val_score(clf,x,y,cv=10)
    # scores = cross_val_score(clf, x, y,  cv=10)
    print(scores.mean())
    print(scores)

    X_train,X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=2021)
    clf.fit(X_train,y_train)

    predictions = clf.predict(X_test)
    print(accuracy_score(y_test, predictions))
    #conf = confusion_matrix(y_test, predictions)
    #df_cm = pd.DataFrame(conf, range(9), range(9))
    ## plt.figure(figsize=(10,7))
    #sns.heatmap(df_cm, annot=True) # font size
    #plt.show()

    #print(scores.mean())

def randomForest(x,y):
    clf = RandomForestClassifier(random_state=2022)

    scores = cross_val_score(clf,x,y.values.ravel(),cv=5)
    print(scores.mean())
    print(scores)

    X_train,X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=2021)
    clf.fit(X_train,y_train.values.ravel())

    predictions = clf.predict(X_test)
    print(accuracy_score(y_test, predictions))

    #print(scores.mean())

#boxplot = df.boxplot(column=['loudness','tempo','time_signature','key','mode','duration','acousticness','danceability','energy','instrumentalness','liveness','speechiness','valence'])
#sns.catplot(x="music_genre", y="loudness", kind="box", data=df)

def vectorMachine(x,y):
    clf = SVC(random_state=2022)
    #scores = cross_val_score(clf,x,y.values.ravel(),cv=5)
    #print(scores.mean())
    #print(scores)

    X_train,X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=2021)
    clf.fit(X_train,y_train.values.ravel())
    predictions = clf.predict(X_test)
    print(accuracy_score(y_test, predictions))

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def logisticRegression(x,y):
    clf = LogisticRegression(random_state=2022,solver='liblinear',max_iter=1000)
    scores = cross_val_score(clf,x,y.values.ravel(),cv=10)

    print(scores.mean())
    
    #X_train,X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=2021)
    #clf.fit(X_train,y_train.values.ravel())
    #predictions = clf.predict(X_test)
    #print(predictions.shape)
    #cm = confusion_matrix(y_test, predictions)
    #print(cm)
#
    #cmd = ConfusionMatrixDisplay(cm, display_labels=df['music_genre'].unique())
    #cmd.plot()
    #plt.show()


#logisticRegression(x,y)

decisionTree(x,y)

randomForest(x,y)

#vectorMachine(x,y)
