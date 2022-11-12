import math
import sklearn as skl
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn import metrics
from sklearn.svm import SVC

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder



dfMath = pd.read_csv("Maths.csv")

dfPort = pd.read_csv("Portuguese.csv")


dfPort['subject']='Portuguese'
dfMath['subject']='Maths'


df = pd.concat([dfPort, dfMath])

df.to_csv("all.csv",index=False)

df = pd.read_csv("all.csv")
print(df.info())
""""
 0   school      1044 non-null   object
 1   sex         1044 non-null   object
 2   age         1044 non-null   int64 
 3   address     1044 non-null   object
 4   famsize     1044 non-null   object
 5   Pstatus     1044 non-null   object
 6   Medu        1044 non-null   int64 
 7   Fedu        1044 non-null   int64 
 8   Mjob        1044 non-null   object
 9   Fjob        1044 non-null   object
 10  reason      1044 non-null   object
 11  guardian    1044 non-null   object
 12  traveltime  1044 non-null   int64 
 13  studytime   1044 non-null   int64 
 14  failures    1044 non-null   int64 
 15  schoolsup   1044 non-null   object
 16  famsup      1044 non-null   object
 17  paid        1044 non-null   object
 18  activities  1044 non-null   object
 19  nursery     1044 non-null   object
 20  higher      1044 non-null   object
 21  internet    1044 non-null   object
 22  romantic    1044 non-null   object
 23  famrel      1044 non-null   int64 
 24  freetime    1044 non-null   int64 
 25  goout       1044 non-null   int64 
 26  Dalc        1044 non-null   int64 
 27  Walc        1044 non-null   int64 
 28  health      1044 non-null   int64 
 29  absences    1044 non-null   int64 
 30  G1          1044 non-null   int64 
 31  G2          1044 non-null   int64 
 32  G3          1044 non-null   int64 
 33  subject     1044 non-null   object
"""

x = df.drop(['G1', 'G2', 'G3'], axis=1)
y = df['G3'].to_frame()

print(df.nunique(axis=0))
print(df['Mjob'].value_counts())
lb_make = LabelEncoder()

#print(df['Fjob'].value_counts())
#print(df['reason'].value_counts())
#print(df['guardian'].value_counts())



clf = DecisionTreeClassifier(random_state=2022)
scores = cross_val_score(clf,x,y,cv=10)

print(scores.mean())


#print('MAE:', metrics.mean_absolute_error(y_test, predictions))
#print('MSE:', metrics.mean_squared_error(y_test, predictions))
#print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))