import pandas as pd
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
print(train.head(10))
print(train.isnull().sum())
train['Age'].fillna(train['Age'].median(), inplace=True)
train['Embarked'].fillna(train['Embarked'].mode()[0], inplace=True)
train.drop(['Cabin'],axis=1,inplace=True)
train.columns
train['Family_size'] = train['SibSp']+train['Parch']+1
train.drop(['SibSp', 'Parch'], inplace=True, axis=1)
train.columns
test.info()
test['Age'].fillna(test['Age'].median(), inplace=True)
test['Fare'].fillna(test['Fare'].median(), inplace=True)
test['Family_size'] = test['SibSp']+test['Parch']+1
test.drop(['SibSp','Parch','Cabin'],inplace=True, axis=1)
train.columns
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
train['Sex'] = le.fit_transform(train['Sex'])
test['Sex'] = le.fit_transform(test['Sex'])
train['Ticket'] = le.fit_transform(train['Ticket'])
test['Ticket'] = le.fit_transform(test['Ticket'])
train.info()
train.drop(['Name'], inplace=True, axis=1)
test.drop(['Name'],inplace=True,axis=1)
train['Embarked'] = le.fit_transform(train['Embarked'])
test['Embarked'] = le.fit_transform(test['Embarked'])
test.info()
X = train[['PassengerId', 'Pclass', 'Sex', 'Age', 'Ticket', 'Fare', 'Embarked', 'Family_size']]
y = train['Survived']
from xgboost import XGBClassifier
xgb = XGBClassifier()
xgb.fit(X,y)
predict = xgb.predict(test)
import numpy as np
predict = np.array(predict)
predict.reshape(-1,1)
data = {'PassengerId': test['PassengerId'], 'Survived': predict}
submission = pd.DataFrame(data)
submission.to_csv('xgboost_submission',index=False)
