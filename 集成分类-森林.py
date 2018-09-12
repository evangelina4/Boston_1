import pandas as pd
titanic=pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')
#人工选取pclass，age，sex作为判别乘客是否能够生还的特征
X=titanic[['pclass','age','sex']]
y=titanic['survived']

#对于缺失的年龄信息，我们使用全体乘客的平均年龄代替，这为了尽可能的不影响预测任务
#??????????
X['age'].fillna(X['age'].mean(),inplace=True)
#?????????

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=33)

#对类别性特征进行转化形成特征向量
from sklearn.feature_extraction import DictVectorizer
vec=DictVectorizer(sparse=False)
X_train=vec.fit_transform(X_train.to_dict(orient='record'))
X_test=vec.transform(X_test.to_dict(orient='record'))

#使用单一决策树
from sklearn.tree import DecisionTreeClassifier
dtc=DecisionTreeClassifier()
dtc.fit(X_train,y_train)
dtc_y_pred=dtc.predict(X_test)

#使用森林
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier()
rfc.fit(X_train,y_train)
rfc_y_pred=rfc.predict(X_test)

#使用梯度
from sklearn.ensemble import GradientBoostingClassifier
gbc=GradientBoostingClassifier()
gbc.fit(X_train,y_train)
gbc_y_pred=gbc.predict(X_test)

from sklearn.metrics import classification_report
print('The accuracy of decision tree is:',dtc.score(X_test,y_test))
print(classification_report(dtc_y_pred,y_test))

print('The accuracy of randomforestclassifier is:',rfc.score(X_test,y_test))
print(classification_report(rfc_y_pred,y_test))

print('The accuracy of gradientboostingclassifier is:',gbc.score(X_test,y_test))
print(classification_report(gbc_y_pred,y_test))
