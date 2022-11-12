import math
import sklearn as skl
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn import metrics
from sklearn.svm import SVC

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn. datasets import make_blobs

from sklearn.cluster import KMeans





df = pd.read_csv("training_data.csv")

test = pd.read_csv("test_data.csv")

# Info about columns and their types
#print(df.info())

# Check unique values on columns
print(df.nunique(axis=0))

print(df['luminosity'].value_counts())

# city_name e avg_precipitation só com um valor possível, por isso vamos retirar
df=df.drop(    ['city_name', 'avg_precipitation','affected_roads'], axis=1)
test=test.drop(['city_name', 'avg_precipitation','affected_roads'], axis=1)


# check for missing values
#print(df.isna().sum())

# Make the values with a ',' NaN

def parse_roads(x):
    if type(x)!=str:
        return 0
    if x==','.strip():
        return 0
    return x.count(',')

#df['affected_roads'] = df["affected_roads"].apply(parse_roads)
#test['affected_roads'] = test["affected_roads"].apply(parse_roads)






def parse_delay(x):
    if x == 'UNDEFINED':
       return 0
    elif x== 'MODERATE':
       return 1
    else:
       return 2


df['magnitude_of_delay'] = df['magnitude_of_delay'].apply(parse_delay)
test['magnitude_of_delay'] = test['magnitude_of_delay'].apply(parse_delay)


def parse_rain(x):
    if x == 'Sem Chuva':
       return 0
    elif x== 'chuva fraca':
       return 1
    elif x== 'chuva moderada':
       return 2
    else:
        return 3
        

df['avg_rain'] = df['avg_rain'].apply(parse_rain)
test['avg_rain'] = test['avg_rain'].apply(parse_rain)



def parse_luminosity(x):
    if x == 'DARK':
       return 0
    elif x== 'LOW_LIGHT':
       return 1
    else:   
       return 2

        

df['luminosity'] = df['luminosity'].apply(parse_luminosity)
test['luminosity'] = test['luminosity'].apply(parse_luminosity)


# Check correlations
"""
corr_matrix = df.corr() 
f, ax = plt.subplots(figsize=(12, 16))
sns.heatmap(corr_matrix, vmin=-1, vmax=1, square=True, annot=True);
plt.show()
"""


df['record_date'] = pd.to_datetime(df['record_date'], format = '%Y-%m-%d %H:%', errors='coerce')
test['record_date'] = pd.to_datetime(test['record_date'], format = '%Y-%m-%d %H:%', errors='coerce')

#df['record_date_year'] = df['record_date'].dt.year
df['record_date_month'] = df['record_date'].dt.month
df['record_date_day'] = df['record_date'].dt.day
df['record_date_hour'] = df['record_date'].dt.hour
#df['record_date_minute'] = df['record_date'].dt.minute

#test['record_date_year'] =  test['record_date'].dt.year
test['record_date_month'] = test['record_date'].dt.month
test['record_date_day'] =   test['record_date'].dt.day
test['record_date_hour'] =  test['record_date'].dt.hour
#test['record_date_minute'] =  test['record_date'].dt.minute


df=df.drop(['record_date'], axis=1)
test=test.drop(['record_date'], axis=1)
1
#print(df.info())


print(df.nunique(axis=0))

print("Duplicated:",df.duplicated().sum())
#print(df.drop_duplicated)

X = df.drop(['incidents'],axis=1)
y = df['incidents'].to_frame()

X_train,X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=2021)


#clf = DecisionTreeClassifier(random_state=2022)

clf = RandomForestClassifier(n_estimators=100)

clf.fit(X,y.values.ravel())

predictions = clf.predict(test)

#clf.fit(X_train,y_train)

scores = cross_val_score(clf,X,y.values.ravel(),cv=10)
print(scores.mean())
file = open("tentativa.csv","w+")

file.write("RowId,Incidents\n")

i = 1
for num in predictions:
    file.write(str(i) + "," +predictions[i-1] +"\n")
    i+=1
#conf = confusion_matrix(y,predictions)
#scores = clf.score(X_test,y_test)
#print("hold-out",scores)
#
#scores = cross_val_score(clf,X,y,cv=10)
#
#print("Cross",scores.mean())
#

#df1 = df.drop(df.columns[[0, 1]], axis=1)

#print(df1.info())
# X=df1.values
# Y=df['Private']

# kmeans = KMeans(n_clusters=2,random_state=2022)
# kmeans.fit(X)

# Y = Y.apply(lambda x : 1 if x =='No' else 0)


# y_pred = kmeans.predict(X)
# print(confusion_matrix(Y,y_pred))
# print(classification_report(Y,y_pred))


"""
kmeans = KMeans (n_clusters=2,random_state=2022)
kmeans.fit(X)
kmeans.cluster_centers_
kmeans.labels_

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(10,6))
ax1.set_title('k Means')
ax1.scatter(X[:,0], X[:,1], c=kmeans.labels_, cmap='rainbow')
ax2.set_title("Original")
ax2.scatter(X[:,0],X[:,1],c=Y,cmap='rainbow')
for i, txt in enumerate(Y):
     if i%5 == 0:
        plt.annotate(txt, (X[i,0],X[i,1]))

plt.show()

"""