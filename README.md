# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Import the standard Libraries.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Assign the points for representing in the graph.
5.Predict the regression for marks by using the representation of the graph.
6.Compare the graphs and hence we obtained the linear regression for the given datas.
 ```

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: DHARSHINI S
RegisterNumber: 212224040074
*/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Step 1: Create DataFrame from given values
data = pd.DataFrame({
    'X': [12, 18, 24, 32, 46, 49, 85, 71, 29],
    'Y': [1245, 2345, 3214, 3615, 4256, 5621, 9216, 7214, 3624]
})

# Step 2: Display head and tail
print("---- HEAD ----")
print(data.head())
print("\n---- TAIL ----")
print(data.tail())

# Step 3: Split features & target
X = data[['X']]
y = data['Y']

print("\nX Values:\n", X.values)
print("\nY Values:\n", y.values)

# Step 4: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 5: Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 6: Predictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

print("\nPredicted values for Training Set:\n", y_pred_train)
print("\nPredicted values for Testing Set:\n", y_pred_test)

# Step 7: Plot Training Set
plt.figure(figsize=(12,5))
plt.subplot(1, 2, 1)
plt.scatter(X_train, y_train, color='blue', label="Actual (Train)")
plt.plot(X_train, y_pred_train, color='red', label="Regression Line")
plt.title("Training Set")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()

# Step 8: Plot Testing Set
plt.subplot(1, 2, 2)
plt.scatter(X_test, y_test, color='green', label="Actual (Test)")
plt.plot(X_train, y_pred_train, color='red', label="Regression Line")
plt.title("Testing Set")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()

plt.tight_layout()
plt.show()

# Step 9: Error Metrics
mae = mean_absolute_error(y_test, y_pred_test)
mse = mean_squared_error(y_test, y_pred_test)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred_test)

print("\n---- Error Metrics ----")
print(f"MAE  : {mae:.2f}")
print(f"MSE  : {mse:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.4f}")

```

## Output:
## Head and Tail files,X and Y values,Predicted Values

<img width="1273" height="951" alt="Screenshot 2025-08-18 141616" src="https://github.com/user-attachments/assets/e18d9d93-f8c1-4829-8282-cf847e81b054" /> 


## Graph For Traning  and Testing Set

<img width="1478" height="713" alt="Screenshot 2025-08-18 141553" src="https://github.com/user-attachments/assets/a2d2c916-c1c8-46ed-8206-e81890fe4475" />



## Error values


<img width="265" height="118" alt="Screenshot 2025-08-13 111726" src="https://github.com/user-attachments/assets/3ed218f8-ae28-4605-a519-efa9ab244c27" />


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
