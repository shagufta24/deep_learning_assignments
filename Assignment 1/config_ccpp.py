# Configuration for ANN

# Set hyperparameters
mini_batch_size = 64
learning_rate = 0.001
num_iterations = 100
activation = "tanh"
cost_func = "mse"

# Regularization
regularisation = "none" # none or L2
lambd = 0.1

# Optimizer
optimizer = "none" # none or momentum or adam
beta = 0.9
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8

# Define model
layers_dims = [4,10,10,1]