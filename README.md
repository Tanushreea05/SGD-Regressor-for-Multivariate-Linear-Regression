# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import necessary libraries: `numpy`, `pandas`, `StandardScaler`, `SGDRegressor`, `MultiOutputRegressor`, and `train_test_split` from `sklearn`.
2. Load the California housing dataset using `fetch_california_housing`.
3. Extract input features `x` (using the first three columns) and create `y` as a multi-output target containing the house price and the number of occupants (using the 7th column).
4. Split the data into training and testing sets using an 80-20 split and set `random_state` for reproducibility.
5. Initialize `StandardScaler` for scaling both `x` (features) and `y` (targets).
6. Fit the scaler on `x_train` and transform `x_train` and `x_test`.
7. Fit the scaler on `y_train` and transform `y_train` and `y_test`.
8. Create an instance of `SGDRegressor` for stochastic gradient descent regression with a maximum of 1000 iterations and a tolerance of 1e-3.
9. Wrap the `SGDRegressor` with `MultiOutputRegressor` to handle multi-output regression and fit it on the scaled `x_train` and `y_train`.
10. Predict the target variables on `x_test` using the trained multi-output model.
11. Inverse-transform the predictions and `y_test` to bring them back to the original scale.
12. Calculate the mean squared error (MSE) between the predicted and actual values.
13. Print the mean squared error.
14. Print the first five predictions for verification.


## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: Tanushree
RegisterNumber:212223100057
*/
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
data = fetch_california_housing()
X = data.data[:, :3]
Y = np.column_stack((data.target, data.data[:, 6]))
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
scaler_X = StandardScaler()
scaler_Y = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)
Y_train = scaler_Y.fit_transform(Y_train)
Y_test = scaler_Y.transform(Y_test)
sgd = SGDRegressor(max_iter=1000, tol=1e-3)
multi_output_sgd = MultiOutputRegressor(sgd)
multi_output_sgd.fit(X_train, Y_train)
Y_pred = multi_output_sgd.predict(X_test)
Y_pred = scaler_Y.inverse_transform(Y_pred)
Y_test = scaler_Y.inverse_transform(Y_test)
mse = mean_squared_error(Y_test, Y_pred)
print("Mean Square Error:", mse)
print("\nPredictions:\n", Y_pred[:5])
```

## Output:

## mean square error
![image](https://github.com/user-attachments/assets/ec53c6bb-7d14-454e-8416-195aefbf36a1)

## predictions
![image](https://github.com/user-attachments/assets/e4c21086-36b8-43db-95e7-9ec3959d1cdc)



## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
