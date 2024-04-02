# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the libraries and read the data frame using pandas.
2. Calculate the null values present in the dataset and apply label encoder.
3. Determine test and training data set and apply decison tree regression in dataset.
4. calculate Mean square error,data prediction and r2. 


## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: H Vishinu
RegisterNumber:  212223220124
*/
import pandas as pd
data=pd.read_csv("/content/Salary.csv")
data.head()

data.info()

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

data["Position"]=le.fit_transform(data["Position"])
data.head()

x=data[["Position","Level"]]
x.head()

y=data[["Salary"]]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
mse=metrics.mean_squared_error(y_test, y_pred)
mse

r2=metrics.r2_score(y_test,y_pred)
r2

dt.predict([[5,6]])
```

## Output:
### Initial dataset:

![output1](https://github.com/VisHinu24/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/144244396/a1f8d597-707f-481e-bb99-5314ce1b5355)


### Data Info:
![output2](https://github.com/VisHinu24/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/144244396/5c840522-2bc0-4b69-8f6c-38941c60d64c)


### Optimization of null values:
![output3](https://github.com/VisHinu24/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/144244396/b1902b10-a4fc-4a88-9fa8-8857684a3988)



### Converting string literals to numericl values using label encoder:
![output4](https://github.com/VisHinu24/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/144244396/8af68dfd-588f-43bc-ba50-ba9bdf6327a4)


### Assigning x and y values:
![output5](https://github.com/VisHinu24/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/144244396/3b7dfc59-4281-4767-8b5a-c8d0f33d0e81)


### Mean Squared Error:

![output6](https://github.com/VisHinu24/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/144244396/c57db343-5fd5-436c-a027-9d5732a2fed1)


### R2 (variance):

![output7](https://github.com/VisHinu24/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/144244396/eb429648-8813-4253-a8e4-ec652d686372)

### Prediction:

![output8](https://github.com/VisHinu24/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/144244396/f8967ec9-c0e9-4333-839e-c9ef6861f9eb)

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
