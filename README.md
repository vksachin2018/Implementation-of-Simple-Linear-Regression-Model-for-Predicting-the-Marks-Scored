# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook
   

## Algorithm
STEP 1 START

STEP 2: Import the standard libraries.

STEP 3: Set variable for assingning dataset values.

STEP 4: Import linear regression from sklearn.

STEP 5: Assign the points for representing in the graph.

STEP 6: Predict the regression for marks by using the representation of the graph.

STEP 7: Compare the graphs and hence we obtained the linear regression for the given datas.

STEP 8 STOP   
## Program:
```
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
## df.head()
![229978451-2b6bdc4f-522e-473e-ae2f-84ec824344c5](https://github.com/vksachin2018/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/149366019/2f2c3c9a-a38f-46bf-a96b-b2e9599e62ed)

## df.tail()
![229978854-6af7d9e9-537f-4820-a10b-ab537f3d0683](https://github.com/vksachin2018/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/149366019/0dfc9ed4-b8be-4195-8adb-ded0d9b66cd8)

## Array value of X
![229978918-707c006d-0a30-4833-bf77-edd37e8849bb](https://github.com/vksachin2018/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/149366019/9af83f21-0201-439a-9ad3-2a1d59231e29)

## Array value of Y
![229978994-b0d2c87c-bef9-4efe-bba2-0bc57d292d20](https://github.com/vksachin2018/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/149366019/3ef48b0c-6b5a-43a7-b8a5-c27187d3d235)

## Array values of Y test
![229979114-3667c4b7-7610-4175-9532-5538b83957ac](https://github.com/vksachin2018/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/149366019/a827a6f8-2db8-491f-af2d-7c97a64e97be)

## Predication values of Y
![229979053-f32194cb-7ed4-4326-8a39-fe8186079b63](https://github.com/vksachin2018/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/149366019/84bbd927-7c1e-4012-9134-2913a23912d4)

## MSE,MAE and RMSE
![229979276-bb9ffc68-25f8-42fe-9f2a-d7187753aa1c](https://github.com/vksachin2018/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/149366019/926ccb2d-e6c3-4d2b-b788-d74d9d6a5d66)

## Training set graph
![image](https://github.com/vksachin2018/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/149366019/5c9f5544-22fc-480c-8722-2731ee245bba)

## Training set graph
![image](https://github.com/vksachin2018/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/149366019/691a7f1b-1422-4b7a-b3b0-bb75a15a60b5)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
