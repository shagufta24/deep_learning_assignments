# Configuration for ANN

# Set hyperparameters
mini_batch_size = 64
learning_rate = 0.001
num_iterations = 4000
activation = "tanh"
cost_func = "mse"

# Regularization
regularisation = "L2" # none or L2
lambd = 0.1

# Optimizer
optimizer = "adam" # none or momentum or adam
beta = 0.9
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8

# Define model
layers_dims = [1, 20, 20, 1]