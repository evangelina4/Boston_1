#1
import pandas as pd
import numpy as np

column_names=['Sample code number','Clump Thickness','Uniformity of Cell Size','Uniformity of Cell Shape','Marginal Adhesion','Single Epithelial Cell Size','Bare Nuclei','Bland Chromatin','Normal Nucleoli','Mitoses','Class']
data=pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data',names=column_names)

data=data.replace(to_replace='?',value=np.nan)
data=data.dropna(how='any')
print(data.shape)


#2
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(data[column_names[1:10]],data[column_names[10]],test_size=0.25,random_state=33)
print(y_train.value_counts())
print(y_test.value_counts())


#3
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import stochastic_gradient

ss=StandardScaler()
X_train=ss.fit_transform(X_train)
X_test=ss.transform(X_test)

lr=LogisticRegression()
sgdc=stochastic_gradient.SGDClassifier()

lr.fit(X_train,y_train)
lr_y_predict=lr.predict(X_test)

sgdc.fit(X_train,y_train)
sgdc_y_predict=sgdc.predict(X_test)

from sklearn.metrics import classification_report
print('accuracy of LRClassificer:',lr.score(X_test,y_test))
print(classification_report(y_test,lr_y_predict,target_names=['Benign','Malignant']))
print('accuarcy of SGDClassifier:',sgdc.score(X_test,y_test))
print(classification_report(y_test,sgdc_y_predict,target_names=['Bengin','Maligant']))

