# -*- coding: utf-8 -*-
# -*- Python -*-
"""
Created  on Thu 2018-03-29 11:27:01 
Update   on 
@author: Qi
Github:  
"""

import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import cross_val_score


train=pd.read_csv('C:/Users/A/Desktop/datasets/Titanic/train.csv')
test=pd.read_csv('C:/Users/A/Desktop/datasets/Titanic/test.csv')

# 人工选取特征
selected_features=['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']
x_train=train[selected_features]
x_test=test[selected_features]
y_train=train['Survived']

# 补充缺失值
x_train['Embarked'].fillna('S',inplace=True)
x_test['Embarked'].fillna('S',inplace=True)
x_train['Age'].fillna(x_train['Age'].mean(),inplace=True)
x_test['Age'].fillna(x_test['Age'].mean(),inplace=True)
x_test['Fare'].fillna(x_test['Fare'].mean(),inplace=True)

# 特征向量化
dict_vec=DictVectorizer(sparse=True)         # DictVectorizer()
X_train=dict_vec.fit_transform(x_train.to_dict(orient='record')) # to_dict()
X_test=dict_vec.fit_transform(x_test.to_dict(orient='record'))

# 随机森林
rfc=RandomForestClassifier()
rfc.fit(X_train,y_train)
params={'n_estimators':[10,30,50]}
rfc_best=RandomForestClassifier()
gs=GridSearchCV(rfc_best,params,n_jobs=-1,verbose=1)   # GridSearchCV()
gs.fit(X_train,y_train)
print(gs.best_params_)
rfc_y=gs.predict(X_test)
rfc_submission=pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':rfc_y})  # pd.DataFrame()
rfc_submission.to_csv('C:/Users/A/Desktop/datasets/Titanic/submission.csv',index=False)
