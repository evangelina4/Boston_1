#1
#导入手写体数字加载器
from sklearn.datasets import load_digits
digits=load_digits()
print(digits.data.shape)

#2
#数据分割
from sklearn.model_selection import train_test_split
#选取75%的训练样本及25%的测试样本
X_train,X_test,y_train,y_test=train_test_split(digits.data,digits.target,test_size=0.25,random_state=33)
print(y_train.shape)
print(y_test.shape)

#3
#支持向量机
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
ss=StandardScaler()
X_train=ss.fit_transform(X_train)
X_test=ss.fit_transform(X_test)
lsvc=LinearSVC()
lsvc.fit(X_train,y_train)
y_predict=lsvc.predict(X_test)

#4
#使用模型自带的评估函数进行准确性测评
print('The Accuracy of LinearSVC is:',lsvc.score(X_test,y_test))

from sklearn.metrics import classification_report
print(classification_report(y_test,y_predict,target_names=digits.target_names.astype(str)))

