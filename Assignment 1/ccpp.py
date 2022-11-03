import pandas as pd
import numpy as np
from annfuncs import *
from config_ccpp import *

# Read dataset
df = pd.read_csv("ccpp.csv")

# Scale the data between -0.9 and 0.9
df=((df-df.min())/(df.max()-df.min())) * (0.9 - (-0.9)) + (-0.9)

# Remove any nan values
df = df.dropna().reset_index(drop=True)

# Split data into train, val and test sets
# Last 10% is put in test set
test_size = round(df.shape[0]/10)
test = df[-test_size:]

rem_size = (df.shape[0] - test.shape[0])
rem = df[:rem_size]

train = rem[rem.index % 5 != 0]  # Excludes every 5th row
val = rem[rem.index % 5 == 0]  # Selects every 5th

# Separate inputs and outputs
x_train = train.iloc[:, :-1].values
y_train = train.iloc[:, -1].values
x_test = test.iloc[:, :-1].values
y_test = test.iloc[:, -1].values
x_val = val.iloc[:, :-1].values
y_val = val.iloc[:, -1].values

# Putting the data in the right shape
x_train = np.transpose(x_train)
y_train = np.transpose(y_train)
y_train = y_train.reshape(1, -1)
x_test = np.transpose(x_test)
y_test = np.transpose(y_test)
y_test = y_test.reshape(1, -1)
x_val = np.transpose(x_val)
y_val = np.transpose(y_val)
y_val = y_val.reshape(1, -1)

print("No. of features: ", x_train.shape[0])
print("No. of samples in training set: ", x_train.shape[1])
print("No. of samples in testing set: ", x_test.shape[1])
print("No. of samples in validation set: ", x_val.shape[1])

# Train the model
parameters = L_layer_model_minib(x_train, y_train, layers_dims,
                valid = True, valid_x = x_val, valid_y = y_val, num_iterations = num_iterations,
                he_init = True, mini_batch_size = mini_batch_size, learning_rate = learning_rate, print_cost = True,
                regularisation = regularisation, lambd = lambd,
                optimizer = optimizer, beta = beta, beta1 = beta1, beta2 = beta2, epsilon = epsilon,
                activation = activation, cost_func = cost_func)

# MAPE
pred_val = predictvals(x_val, parameters, activation=activation)
mape_val = mape_cost(y_val, pred_val)
print("Validation mape : ", mape_val)

pred_test = predictvals(x_test, parameters, activation=activation)
mape_test = mape_cost(y_test, pred_test)
print("Test mape : ", mape_test)