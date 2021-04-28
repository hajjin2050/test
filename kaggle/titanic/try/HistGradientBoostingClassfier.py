import numpy as np
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')

df_train = pd.read_csv('C:\workspace/titanic\data/train.csv')
df_test = pd.read_csv('C:\workspace/titanic\data/test.csv')

df_train['Fare']=df_train['Fare'].astype('float')
df_test['Fare']=df_test['Fare'].astype('float')

df_train.info()
df_train.isna().apply(pd.value_counts).T
df_test.isna().apply(pd.value_counts).T

df_train['Age'].fillna(df_train['Age'].median(),inplace=True)
df_test['Age'].fillna(df_test['Age'].median(),inplace=True)
pd.set_option('display.max_rows',None)

df_train['Fare'].fillna(df_test['Fare'].median,inplace=True)
df_test['Fare'].fillna(df_test['Fare'].median,inplace=True)

df_train['Ticket'].fillna('X',inplace=True)
df_test['Ticket'].fillna('X',inplace=True)

df_test['Cabin'].fillna('X',inplace=True)
df_train['Cabin'].fillna('X',inplace=True)
df_train['SibSp'].nunique()

print(df_train['Embarked'].value_counts())

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
#print(df_train['Embarked'])

df_train['Embarked'] = encoder.fit_transform(df_train['Embarked'].astype('str'))
df_test['Embarked'] = encoder.fit_transform(df_test['Embarked'].astype('str'))
df_test['Embarked'].fillna(df_test['Embarked'].median(),inplace=True)
df_train['Embarked'].fillna(df_train['Embarked'].median(),inplace=True)


df_train['Sex'] = encoder.fit_transform(df_train['Sex']).astype('float')
df_test['Sex'] = encoder.fit_transform(df_test['Sex']).astype('float')
df_train['Ticket'] = encoder.fit_transform(df_train['Ticket'])
df_test['Ticket'] = encoder.fit_transform(df_test['Ticket'])
df_train['Cabin'] = encoder.fit_transform(df_train['Cabin'])
df_test['Cabin'] = encoder.fit_transform(df_test['Cabin'])
df_train.drop('Fare',axis=1,inplace=True)
df_test.drop('Fare',axis=1,inplace=True)


df_train['Name'] = df_train['Name'].map(lambda x: x.split(',')[0])
df_test['Name'] = df_test['Name'].map(lambda x: x.split(',')[0])
df_train['Name'] = encoder.fit_transform(df_train['Name']).astype('float')
df_test['Name'] = encoder.fit_transform(df_test['Name']).astype('float')
X_train = df_train.drop(['PassengerId','Survived'],axis=1)
X_test = df_test.drop(['PassengerId'],axis=1)
y_train = df_train['Survived']

from sklearn.preprocessing import StandardScaler
std = StandardScaler()
X_test.head()

X_train = std.fit_transform(X_train)
X_test = std.fit_transform(X_test)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
rf = RandomForestClassifier(n_estimators=500,criterion='gini',max_depth=7)
rf.fit(X_train, y_train)
scores = cross_val_score(rf, X_train, y_train, cv=5)
scores.mean()

#from xgboost import XGBClassifier
#from sklearn.experimental import enable_hist_gradient_boosting
#from sklearn.ensemble import HistGradientBoostingClassifier

#model = HistGradientBoostingClassifier()

#model.fit(X_train,y_train)
#scores_xg = cross_val_score(model,X_train,y_train,cv=5)
#scores_model.mean()

sample_submission = pd.read_csv('C:\workspace/titanic\data\sample_submission.csv')
predictions = rf.predict(X_test)
sample_submission['Survived'] = predictions
sample_submission.to_csv('submission.csv',index=False)
predictions_model = model.predict(X_test)
sample_submission['Survived'] = predictions_model
sample_submission.to_csv('submission_model.csv',index=False)