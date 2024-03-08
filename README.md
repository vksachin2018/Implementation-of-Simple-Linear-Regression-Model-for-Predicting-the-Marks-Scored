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
![dataset](https://github.com/vksachin2018/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/149366019/bd791a20-f976-4746-9376-d3c304861258)
![head](https://github.com/vksachin2018/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/149366019/b2a197c5-b596-46d4-8836-9800510dd708)
![tail](https://github.com/vksachin2018/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/149366019/bfa669ed-6dd2-487c-9cae-a98064677956)
![xyvalues](https://github.com/vksachin2018/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/149366019/c8c63382-7695-43f9-9b71-380a7fac9512)
![predict ](https://github.com/vksachin2018/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/149366019/af563636-2963-4371-bb1c-15df4a917900)
![values](https://github.com/vksachin2018/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/149366019/21375649-a3a9-4d78-8cc5-3d5861277749)
![train](https://github.com/vksachin2018/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/149366019/fd61f5a2-b6fa-4c21-b13f-beebdf2c65e3)
![test](https://github.com/vksachin2018/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/149366019/9489e2c1-7711-4973-a755-739c41c1de46)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
