# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: gokul sachin k
RegisterNumber: 212223220025
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```

## Output:
dataset
![dataset](https://github.com/vksachin2018/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/149366019/669914fe-9c2a-4b75-8596-b37866d2536e)
Head values
![head](https://github.com/vksachin2018/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/149366019/68111540-e4ff-443a-b6b0-cb3ff30d879f)
Tail values
![tail](https://github.com/vksachin2018/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/149366019/3f2aaf68-fb96-4c79-9d53-ec77fc04be27)
X and Y values
![xyvalues](https://github.com/vksachin2018/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/149366019/13513b22-b2cc-46ba-b21c-a71ef9f51d14)
Predication values of X and Y
![predict ](https://github.com/vksachin2018/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/149366019/aa9be2bb-0d4a-40b5-9f5f-88591b7b801c)
MSE,MAE and RMSE
![values](https://github.com/vksachin2018/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/149366019/fd6365fc-e2d4-472d-9ca4-d7a0a8bea510)
Training set
![train](https://github.com/vksachin2018/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/149366019/d58a5b66-1484-4378-8ab6-a559f6c36104)
Training set
![test](https://github.com/vksachin2018/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/149366019/773c638a-7d8a-43db-b9e7-589620f196af)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
