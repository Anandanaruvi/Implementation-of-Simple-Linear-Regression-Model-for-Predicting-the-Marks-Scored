# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import necessary libraries (e.g., pandas, numpy,matplotlib).
2. Load the dataset and then split the dataset into training and testing sets using sklearn library.
3. Create a Linear Regression model and train the model using the training data (study hours as input, marks scored as output).
4. Use the trained model to predict marks based on study hours in the test dataset.
5.Plot the regression line on a scatter plot to visualize the relationship between study hours and marks scored. 

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: A.ARUVI.
RegisterNumber:  212222230014.
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv("/content/student_scores.csv")
df.head()

x=df.iloc[:,:-1].values
x

y=df.iloc[:,1].values
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)

y_train
y_pred

plt.scatter(x_train,y_train,color="yellow")
plt.plot(x_train,regressor.predict(x_train),color="purple")
plt.title("Hours VS Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(x_test,y_test,color="yellow")
plt.plot(x_test,regressor.predict(x_test),color="purple")
plt.title("Hours vs scores(Test Data Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_squared_error(y_test,y_pred)
print("MSE= ",mse)
mae=mean_absolute_error(y_test,y_pred)
print("MAE = ",mae)
rmse=np.sqrt(mse)
print("RMSE = ",rmse)
```

## Output:
![image](https://github.com/Anandanaruvi/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/120443233/742a3fe1-9919-42c0-963e-da801be3f267)
### Values of x:
![image](https://github.com/Anandanaruvi/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/120443233/926ed654-dd74-434a-9ded-746b0d2e4b46)
### Values of y:
![image](https://github.com/Anandanaruvi/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/120443233/b7d93d93-9478-40d5-8898-69ba7e9d3e1f)
### Values of y_train:
![image](https://github.com/Anandanaruvi/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/120443233/25902326-3c53-45cd-aeb3-a1dcb8aef9ee)
### Values of y_pred:
![image](https://github.com/Anandanaruvi/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/120443233/bb838b72-5de7-49df-9ae5-8e16eda68328)
### Training Set Graph:
![image](https://github.com/Anandanaruvi/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/120443233/ee34961f-54c1-4eb0-a4a5-e19c9a0764d7)
### Test set Graph:
![image](https://github.com/Anandanaruvi/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/120443233/10f4ba60-bb82-4b20-ae97-0b12a4015b10)

## Result:

Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.

