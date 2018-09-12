import pandas as pd
titanic=pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')
print(titanic.head())#查看文件的前n行
titanic.info()#查看索引、内存、数据信息

#特征的选择
X=titanic[['pclass','age','sex']]#以dataframe的形式返回多列
y=titanic['survived']

X.info()
X['age'].fillna(X['age'].mean,inplace=True)
X.info()

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=33)

#使用特征转换器
from sklearn.feature_extraction import DictVectorizer
vec=DictVectorizer(sparse=False)
X_train=vec.fit_transform(X_train.to_dict(orient='record'))
print(vec.feature_names_)

X_test=vec.transform(X_test.to_dict(orient='record'))
from sklearn.tree import DecisionTreeClassifier
dtc=DecisionTreeClassifier()
#使用分割到的训练数据进行模型学习
dtc.fit(X_train,y_train)
#用训练好的决策树模型对测试特征数据进行预测
y_predict=dtc.predict(X_test)

#性能测评
from sklearn.metrics import classification_report
print(dtc.score(X_test,y_test))
print(classification_report(y_predict,y_test,target_names=['died','survived']))
