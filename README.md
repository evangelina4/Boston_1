from sklearn.datasets import load_boston
boston=load_boston()
print(boston.DESCR)

from sklearn.model_selection import train_test_split
import numpy as np
X=boston.data
y=boston.target

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=33,test_size=0.25)
print("The max target value is:",np.max(boston.target))
print("The min target value is:",np.min(boston.target))
print("The average target is:",np.mean(boston.target))

from sklearn.preprocessing import StandardScaler
ss_X=StandardScaler()
ss_y=StandardScaler()
X_train=ss_X.fit_transform(X_train)
X_test=ss_X.transform(X_test)
y_train=ss_y.fit_transform(y_train.reshape(-1,1))
y_test=ss_y.transform(y_test.reshape(-1,1))

from sklearn.linear_model import LinearRegression

lr=LinearRegression()
lr.fit(X_train,y_train)
lr_y_predict=lr.predict(X_test)

from sklearn.linear_model import SGDRegressor
sgdr=SGDRegressor()
sgdr.fit(X_train,y_train)
sgdr_y_predict=sgdr.predict(X_test)

print("the value of default measurement of LinearRegression is:",lr.score(X_test,y_test))

#使用inverse_transform函数是将归一化后的预测值再次转回原来的目标预测值
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
print("the value of R_squared of LinearRegression is:",r2_score(y_test,lr_y_predict))
print("the value of MSE:",mean_squared_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(lr_y_predict)))
print("the value of MAE is:",mean_absolute_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(lr_y_predict)))
print(' ')
print("the value of default measurement of LinearRegression is:",sgdr.score(X_test,y_test))
print("the value of R_squared of SGDRegression is :",r2_score(y_test,sgdr_y_predict))
print("the value of MSE is:",mean_squared_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(sgdr_y_predict)))
print("the value of MAE is:",mean_absolute_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(sgdr_y_predict)))
print(' ')

#svr=support vector regression支持向量回归机 kernel=核函数
from sklearn.svm import SVR
linear_svr=SVR(kernel='linear')
linear_svr.fit(X_train,y_train)
linear_svr_y_predict=linear_svr.predict(X_test)

poly_svr=SVR(kernel='poly')
poly_svr.fit(X_train,y_train)
poly_svr_y_predict=poly_svr.predict(X_test)

rbf_svr=SVR(kernel='rbf')
rbf_svr.fit(X_train,y_train)
rbf_svr_y_predict=rbf_svr.predict(X_test)

print("R_squared value of linear SVR is:",linear_svr.score(X_test,y_test))
print("MSE value of linear SVR is:",mean_squared_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(linear_svr_y_predict)))
print("MAE value of linear SVR is:",mean_absolute_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(linear_svr_y_predict)))
print(' ')

print("R_squared value of poly SVR is:",poly_svr.score(X_test,y_test))
print("MSE value of poly SVR is:",mean_squared_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(poly_svr_y_predict)))
print("MAE value of poly SVR is:",mean_absolute_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(poly_svr_y_predict)))
print(' ')

print("R_squared value of rbf SVR is:",rbf_svr.score(X_test,y_test))
print("MSE value of rbf SVR is:",mean_squared_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(rbf_svr_y_predict)))
print("MAE value of rbf SVR is:",mean_absolute_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(rbf_svr_y_predict)))
print(' ')


from sklearn.neighbors import KNeighborsRegressor
uni_knr=KNeighborsRegressor(weights='uniform')#平均回归
uni_knr.fit(X_train,y_train)
uni_knr_y_predict=uni_knr.predict(X_test)

dis_knr=KNeighborsRegressor(weights='distance')#距离加权回归
dis_knr.fit(X_train,y_train)
dis_knr_y_predict=dis_knr.predict(X_test)

print("R-squared value of uniform-weighted KNeighborRegression is:",uni_knr.score(X_test,y_test))
print("The MSE of uniform-weighted KNeighborRegression is:",mean_squared_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(uni_knr_y_predict)))
print("The MAE of uniform-weighted KNeighborRegression is:",mean_absolute_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(uni_knr_y_predict)))
print(' ')
print("R-squared value of distance-weighted KNeighborRegression is:",dis_knr.score(X_test,y_test))
print("The MSE of distance-weighted KNeighborRegression is:",mean_squared_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(dis_knr_y_predict)))
print("The MAE of distance-weighted KNeighborRegression is:",mean_absolute_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(dis_knr_y_predict)))
print(' ')

from sklearn.tree import DecisionTreeRegressor
dtr=DecisionTreeRegressor()
dtr.fit(X_train,y_train)
dtr_y_predict=dtr.predict(X_test)
print("R-squared value of DecisionTreeRegressor is:",dtr.score(X_test,y_test))
print("The MSE of DecisionTreeRegressor is:",mean_squared_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(dtr_y_predict)))
print("The MAE of DecisionTreeRegressor is:",mean_absolute_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(dtr_y_predict)))
print(" ")


from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor,GradientBoostingRegressor
rfr=RandomForestRegressor()
rfr.fit(X_train,y_train)
rfr_y_predict=rfr.predict(X_test)
print("R-squared value of RandomForestRegressor is:",rfr.score(X_test,y_test))
print("The MSE value of RandomForestRegressor is:",mean_squared_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(rfr_y_predict)))
print("The MAE value of RandomForestRegressor is:",mean_absolute_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(rfr_y_predict)))
print(" ")

etr=ExtraTreesRegressor()
etr.fit(X_train,y_train)
etr_y_predict=etr.predict(X_test)
print("R-squared value of ExtraTreesRegressor is:",etr.score(X_test,y_test))
print("The MSE value of ExtraTreesRegressor is:",mean_squared_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(etr_y_predict)))
print("The MAE value of ExtraTreesRegressor is:",mean_absolute_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(etr_y_predict)))
#print(np.sort(zip(etr.feature_importances_,boston.feature_names),axis=0))
print(" ")

#DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().




gbr=GradientBoostingRegressor()
gbr.fit(X_train,y_train)
gbr_y_predict=gbr.predict(X_test)
print("R-squared value of GradientBoostingRegressor is:",gbr.score(X_test,y_test))
print("The MSE value of GradientBoostingRegressor is:",mean_squared_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(gbr_y_predict)))
print("The MAE value of GradientBoostingRegressor is:",mean_absolute_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(gbr_y_predict)))
print(" ")
