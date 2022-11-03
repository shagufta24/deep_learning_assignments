import math
import numpy as np
from numpy import arange
import numpy as np
from annfuncs import *
from config_sine import *

def generate_sine_data():
    train_data = []
    for i in arange(lower, upper, step):
        train_data.append([i, math.sin(i)])

    train_data = np.array(train_data)
    np.random.shuffle(train_data)

    x_train = train_data[:, :1]
    y_train = train_data[:, 1:2]

    x_train = np.transpose(x_train)
    y_train = np.transpose(y_train)

    x_val = np.random.uniform(low = lower, high = upper, size = 300)
    y_val = np.array([math.sin(x) for x in x_val])

    x_val = x_val.reshape((1, x_val.shape[0]))
    y_val = y_val.reshape((1, y_val.shape[0]))

    return x_train, y_train, x_val, y_val

if __name__ == "__main__":

    lower = -2*math.pi
    upper = 2*math.pi
    interval = upper - lower
    count = 1000
    step = interval/count

    x_train, y_train, x_val, y_val = generate_sine_data()
    print("No. of training samples: ", x_train.shape[1])
    print("No. of validation samples: ", x_val.shape[1])

    # Train the model
    parameters = L_layer_model_minib(x_train, y_train, layers_dims,
                    valid = True, valid_x = x_val, valid_y = y_val, num_iterations = num_iterations,
                    he_init = True, mini_batch_size = mini_batch_size, learning_rate = learning_rate, print_cost = True,
                    regularisation = regularisation, lambd = lambd,
                    optimizer = optimizer, beta = beta, beta1 = beta1, beta2 = beta2, epsilon = epsilon,
                    activation = activation, cost_func = cost_func)
                    
    preds, cache = L_model_forward(x_val, parameters, activation='tanh')
    plt.scatter(x_val[0],y_val[0],color='blue')
    plt.scatter(x_val[0],preds[0],color='red')
    plt.legend(['train','test'])
    plt.show()

    # Forward pass
    preds, cache = L_model_forward(x_val, parameters, activation=activation)
    plt.scatter(x_train[0],y_train[0],color='blue')
    plt.scatter(x_val[0],preds[0],color='red')
    plt.legend(['train','test'])
    plt.show()

    # MAPE
    pred_val = predictvals(x_val, parameters, activation=activation)
    mape_val = mape_cost(y_val, pred_val)
    print("Validation mape : ", mape_val)