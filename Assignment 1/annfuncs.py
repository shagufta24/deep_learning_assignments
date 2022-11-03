import numpy as np
import matplotlib.pyplot as plt
import math

from activations import *

np.random.seed(1)

def initialize_parameters(n_x, n_h, n_y):
    W1 = np.random.randn(n_h, n_x)*0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h)*0.01
    b2 = np.zeros((n_y, 1))
    
    assert(W1.shape == (n_h, n_x))
    assert(b1.shape == (n_h, 1))
    assert(W2.shape == (n_y, n_h))
    assert(b2.shape == (n_y, 1))
    
    parameters = {"W1": W1,"b1": b1,"W2": W2,"b2": b2}
    return parameters     

def initialize_parameters_deep(layer_dims):
    parameters = {}
    L = len(layer_dims) # number of layers in the network

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l],layer_dims[l-1]) * 0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l],1))
        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))

    return parameters

def initialize_parameters_deep_he(layers_dims):
    parameters = {}
    L = len(layers_dims) - 1 # integer representing the number of layers
     
    for l in range(1, L + 1):
        parameters['W' + str(l)] = np.random.randn(layers_dims[l],layers_dims[l-1])*np.sqrt(2/layers_dims[l-1])
        parameters['b' + str(l)] = np.zeros((layers_dims[l],1))
         
    return parameters

def linear_forward(A, W, b):
    Z = W.dot(A) + b
    
    assert(Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)
    
    return Z, cache

def linear_activation_forward(A_prev, W, b, activation):
    if activation == "sigmoid":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    
    elif activation == "relu":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
    elif activation == "tanh":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = tanh(Z)
    
    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)

    return A, cache

def L_model_forward(X, parameters,activation):
    caches = []
    A = X
    L = len(parameters) // 2  # number of layers in the neural network
    
    # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
    for l in range(1, L):
        A_prev = A 
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], activation = "relu")
        caches.append(cache)
    
    # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], activation)
    caches.append(cache)
    
    assert(AL.shape == (1,X.shape[1]))      
    return AL, caches

def compute_cost(AL,Y,parameters,lambd,regularisation,cost_func='mse'):

    m = Y.shape[1] # no of samples
    L = len(parameters) // 2     
    epsilon=0.001
    # Compute loss from aL and y.
    if (cost_func=='log'):
        cost = (1./m) * np.sum(-np.dot(Y,np.log(AL+epsilon).T) - np.dot(1-Y, np.log(1-AL+epsilon).T))
    if (cost_func=='mape'):
        cost=np.mean(np.abs((Y-AL)/(Y+epsilon)))*100
    if (cost_func=='mse'):
        cost=np.mean(np.square(AL-Y))*0.5

    # L2 Regularisation cost
    if(regularisation=='L2'):
        sumw=0 #sum of weights
        for l in range(1, L + 1):
            # parameters['W' + str(l)] = np.random.randn(layers_dims[l],layers_dims[l-1])*np.sqrt(2/layers_dims[l-1])
            sumw=sumw+np.sum(np.square(parameters['W' + str(l)]))
        L2_regularization_cost = (1/m)*(lambd/2)*(sumw)
    
    cost = np.squeeze(cost)      # To make sure the cost's shape is what is expected (e.g. turns [[17]] into 17).
    assert(cost.shape == ())
    if(regularisation=='L2'):
        cost=cost+L2_regularization_cost
    return cost

def mape_cost(Y,AL):
    epsilon=0.001
    cost=np.mean(np.abs((Y-AL)/(Y+epsilon)))*100
    return cost

def linear_backward(dZ, cache,regularisation,lambd):
    # Here cache is "linear_cache" containing (A_prev, W, b) coming from the forward propagation in the current layer
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = 1./m * np.dot(dZ,A_prev.T)
    if(regularisation=='L2'):
        dW=dW + lambd*W*(1/m)

    db = 1./m * np.sum(dZ, axis = 1, keepdims = True)
    dA_prev = np.dot(W.T,dZ)
    
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    
    return dA_prev, dW, db

def linear_activation_backward(dA, cache,regularisation,lambd,activation):
    linear_cache, activation_cache = cache
    
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)

    elif activation == "tanh":
        dZ = tanh_backward(dA, activation_cache)
    
    dA_prev, dW, db = linear_backward(dZ, linear_cache,regularisation,lambd)
    return dA_prev, dW, db

def L_model_backward(AL, Y, caches,activation,regularisation,lambd,cost_func='log'):
    
    grads = {}
    L = len(caches) # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) # Y is now the same shape as AL
    
    # Initializing the backpropagation
    if(cost_func=='log'):
        dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    if(cost_func=='mse'):
        dAL=(AL-Y)
    
    # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "AL, Y, caches". Outputs: "grads["dAL"], grads["dWL"], grads["dbL"]
    current_cache = caches[L-1]
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache,regularisation,lambd, activation)
    
    for l in reversed(range(L-1)):
        # lth layer: (RELU -> LINEAR) gradients.
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 1)], current_cache,regularisation,lambd,activation = "relu")
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads

def update_parameters(parameters, grads, learning_rate):

    L = len(parameters) // 2 # number of layers in the neural network

    # Update rule for each parameter. Use a for loop.
    for l in range(L):
        # if(regularisation=="L2"):
        #     parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * (grads["dW" + str(l+1)]+lambd*parameters["W" + str(l+1)]*(1/m))
        #     parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]
        # else:
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]
        
    return parameters


def predict(X, y, parameters):
    
    m = X.shape[1]
    n = len(parameters) // 2 # number of layers in the neural network
    p = np.zeros((1,m))
    
    # Forward propagation
    probas, caches = L_model_forward(X, parameters)

    
    # convert probas to 0/1 predictions
    for i in range(0, probas.shape[1]):
        if probas[0,i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0
    
    #print results
    #print ("predictions: " + str(p))
    #print ("true labels: " + str(y))
    print("Accuracy: "  + str(np.sum((p == y)/m)))
        
    return p

def predictvals(x, parameters,activation):
    probas, caches = L_model_forward(x, parameters,activation)
    return probas

def predicterr(x,y,parameters,lambd,activation,regularisation,cost_func):
    probas, caches = L_model_forward(x, parameters,activation)
    err=compute_cost(probas,y,parameters=parameters,lambd=lambd,regularisation=regularisation,cost_func=cost_func)
    return err
 
def initialize_adam(parameters) :
    L = len(parameters) // 2 # number of layers in the neural networks
    v = {}
    s = {}
    
    # Initialize v, s. Input: "parameters". Outputs: "v, s".
    for l in range(L):
        v["dW" + str(l+1)] = np.zeros(parameters["W"+str(l+1)].shape)
        v["db" + str(l+1)] = np.zeros(parameters["b"+str(l+1)].shape)
        s["dW" + str(l+1)] = np.zeros(parameters["W"+str(l+1)].shape)
        s["db" + str(l+1)] = np.zeros(parameters["b"+str(l+1)].shape)
    
    return v, s

def initialize_velocity(parameters):
    L = len(parameters) // 2 # number of layers in the neural networks
    v = {}
    
    # Initialize velocity
    for l in range(L):
        v["dW" + str(l+1)] = np.zeros(parameters["W"+str(l+1)].shape)
        v["db" + str(l+1)] = np.zeros(parameters["b"+str(l+1)].shape)
        
    return v

def update_parameters_with_adam(parameters, grads, v, s, t, learning_rate = 0.01,
                                beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8):
    
    L = len(parameters) // 2                 # number of layers in the neural networks
    v_corrected = {}                         # Initializing first moment estimate, python dictionary
    s_corrected = {}                         # Initializing second moment estimate, python dictionary
    
    # Perform Adam update on all parameters
    for l in range(L):
        # Moving average of the gradients. Inputs: "v, grads, beta1". Output: "v".
        v["dW" + str(l+1)] = beta1*v["dW" + str(l+1)] +(1-beta1)*grads["dW" + str(l+1)]
        v["db" + str(l+1)] = beta1*v["db" + str(l+1)] +(1-beta1)*grads["db" + str(l+1)]

        # Compute bias-corrected first moment estimate. Inputs: "v, beta1, t". Output: "v_corrected".
        v_corrected["dW" + str(l+1)] =  v["dW" + str(l+1)]/(1-math.pow(beta1,t))
        v_corrected["db" + str(l+1)] =  v["db" + str(l+1)]/(1-math.pow(beta1,t))

        # Moving average of the squared gradients. Inputs: "s, grads, beta2". Output: "s".
        s["dW" + str(l+1)] = beta2*s["dW" + str(l+1)] +(1-beta2)*grads["dW" + str(l+1)]*grads["dW" + str(l+1)]
        s["db" + str(l+1)] = beta2*s["db" + str(l+1)] +(1-beta2)*grads["db" + str(l+1)]*grads["db" + str(l+1)]

        # Compute bias-corrected second raw moment estimate. Inputs: "s, beta2, t". Output: "s_corrected".
        s_corrected["dW" + str(l+1)] = s["dW" + str(l+1)]/(1-math.pow(beta2,t))
        s_corrected["db" + str(l+1)] = s["db" + str(l+1)]/(1-math.pow(beta2,t))

        # Update parameters. Inputs: "parameters, learning_rate, v_corrected, s_corrected, epsilon". Output: "parameters".
        parameters["W" + str(l+1)] =  parameters["W" + str(l+1)]- learning_rate*v_corrected["dW" + str(l+1)]/(np.sqrt(s_corrected["dW" + str(l+1)]+epsilon))
        parameters["b" + str(l+1)] =  parameters["b" + str(l+1)]- learning_rate*v_corrected["db" + str(l+1)]/(np.sqrt(s_corrected["db" + str(l+1)]+epsilon))

    return parameters, v, s

def update_parameters_with_momentum(parameters, grads, v, beta, learning_rate):
    L = len(parameters) // 2 # number of layers in the neural networks
    
    # Momentum update for each parameter
    for l in range(L):
        
        # compute velocities
        v["dW" + str(l+1)] = beta*v["dW" + str(l+1)] +(1-beta)*grads["dW" + str(l+1)]
        v["db" + str(l+1)] = v["db" + str(l+1)]*beta +(1-beta)*grads["db" + str(l+1)]
        # update parameters
        parameters["W" + str(l+1)] =  parameters["W" + str(l+1)]-learning_rate*(v["dW" + str(l+1)])
        parameters["b" + str(l+1)] =  parameters["b" + str(l+1)]-learning_rate*(v["db" + str(l+1)])
        
    return parameters, v

def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
   
    np.random.seed(seed)            # To make your "random" minibatches the same as ours
    m = X.shape[1]                  # number of training examples
    mini_batches = []
        
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((1,m))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k*mini_batch_size : (k+1)*mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k*mini_batch_size : (k+1)*mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches*mini_batch_size : ]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches*mini_batch_size : ]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches


def L_layer_model_minib(X, Y,layers_dims,valid=False,valid_x=None,valid_y=None, optimizer='none', learning_rate = 0.0007,he_init=False, 
                        mini_batch_size = 64, beta = 0.9, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8, num_iterations = 10000,
                        activation='sigmoid',regularisation='none',print_cost = True,lambd=0.1, cost_func='mse'):

    L = len(layers_dims)             # number of layers in the neural networks
    costs = []  
    validcosts=[]                     # to keep track of the cost
    t = 0                            # initializing the counter required for Adam update
    seed = 10                        # For grading purposes, so that your "random" minibatches are the same as ours
    m = X.shape[1]
    batches=m//mini_batch_size                   # number of training examples
    
    # Initialize parameters
    # parameters = initialize_parameters(layers_dims)
    if(he_init):
        parameters=initialize_parameters_deep_he(layers_dims)
    else:
        parameters=initialize_parameters_deep(layers_dims)

    # Initialize the optimizer
    if optimizer == "gd":
        pass # no initialization required for gradient descent
    elif optimizer == "momentum":
        v = initialize_velocity(parameters)
    elif optimizer == "adam":
        v, s = initialize_adam(parameters)
    
    # Optimization loop
    for i in range(num_iterations):
        
        # Define the random minibatches. We increment the seed to reshuffle differently the dataset after each epoch
        seed = seed + 1
        minibatches = random_mini_batches(X, Y, mini_batch_size, seed)
        cost_total = 0
        
        for minibatch in minibatches:

            # Select a minibatch
            (minibatch_X, minibatch_Y) = minibatch

            # Forward propagation
            a3, caches =  L_model_forward(X=minibatch_X,parameters= parameters,activation= activation)

            # Compute cost and add to the cost total
            cost_total += compute_cost(AL= a3, Y= minibatch_Y,parameters= parameters,lambd= lambd,regularisation= regularisation,cost_func= cost_func)

            # Backward propagation
            grads =  L_model_backward(AL= a3,Y= minibatch_Y,caches= caches,activation= activation,regularisation= regularisation,lambd=lambd,cost_func= cost_func)

            # Update parameters
            if optimizer == "gd" or optimizer=='none':
                parameters = update_parameters(parameters, grads, learning_rate)
            elif optimizer == "momentum":
                parameters, v = update_parameters_with_momentum(parameters, grads, v, beta, learning_rate)
            elif optimizer == "adam":
                t = t + 1 # Adam counter
                parameters, v, s = update_parameters_with_adam(parameters, grads, v, s,
                                                               t, learning_rate, beta1, beta2,  epsilon)
        cost_avg = cost_total / batches
        
        # Print the cost every 1000 epoch
        if print_cost and i % 1 == 0:
            if(valid==True):
                valid_err=predicterr(valid_x,valid_y,parameters=parameters,lambd=lambd,activation=activation,regularisation='none',cost_func=cost_func)
                validcosts.append(valid_err)
                print ("Cost after epoch %i: %f,  Valid err: %f" %(i, cost_avg,valid_err))
            else:
                print ("Cost after epoch %i: %f" %(i, cost_avg))
            costs.append(cost_avg)
                
    # plot the cost
    plt.plot(costs)
    if(valid==True):
        plt.plot(validcosts)
        plt.legend(["train", "validation"], loc ="upper right") 
    plt.ylabel('cost')
    plt.xlabel('epochs')
    plt.title("Convergence history")
    plt.show()

    return parameters