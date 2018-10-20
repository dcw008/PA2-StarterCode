import numpy as np
import pickle

config = {}
config['layer_specs'] = [784, 100, 100, 10]  # The length of list denotes number of hidden layers; each element denotes number of neurons in that layer; first element is the size of input layer, last element is the size of output layer.
config['activation'] = 'sigmoid' # Takes values 'sigmoid', 'tanh' or 'ReLU'; denotes activation function for hidden layers
config['batch_size'] = 1000  # Number of training samples per batch to be passed to network
config['epochs'] = 50  # Number of epochs to train the model
config['early_stop'] = True  # Implement early stopping or not
config['early_stop_epoch'] = 5  # Number of epochs for which validation loss increases to be counted as overfitting
config['L2_penalty'] = 0  # Regularization constant
config['momentum'] = False  # Denotes if momentum is to be applied or not
config['momentum_gamma'] = 0.9  # Denotes the constant 'gamma' in momentum expression
config['learning_rate'] = 0.0001 # Learning rate of gradient descent algorithm


def softmax(x):
  """
  Write the code for softmax activation function that takes in a numpy array and returns a numpy array.
  """
  total = np.sum(np.exp(x))
  return np.true_divide(x, total)


def load_data(fname):
  """
  Write code to read the data and return it as 2 numpy arrays.
  Make sure to convert labels to one hot encoded format.
  """
  f = open(fname, 'rb')
  data = pickle.load(f, encoding='bytes')
  f.close() 
  X = []
  Y = []
  for d in data:
    X.append(d[:784])
    Y.append(d[784])
  return np.array(X), np.array(Y)


class Activation:
  def __init__(self, activation_type = "sigmoid"):
    self.activation_type = activation_type
    self.x = None # Save the input 'x' for sigmoid or tanh or ReLU to this variable since it will be used later for computing gradients.
  
  def forward_pass(self, a):
    if self.activation_type == "sigmoid":
      return self.sigmoid(a)
    
    elif self.activation_type == "tanh":
      return self.tanh(a)
    
    elif self.activation_type == "ReLU":
      return self.relu(a)
  
  def backward_pass(self, delta):
    if self.activation_type == "sigmoid":
      grad = self.grad_sigmoid()
    
    elif self.activation_type == "tanh":
      grad = self.grad_tanh()
    
    elif self.activation_type == "ReLU":
      grad = self.grad_ReLU()
    
    return grad * delta
      
  def sigmoid(self, x):
    """
    Write the code for sigmoid activation function that takes in a numpy array and returns a numpy array.
    """
    self.x = x
    return np.array([np.true_divide(1, np.add(1, np.exp(-e))) for e in x])

  def tanh(self, x):
    """
    Write the code for tanh activation function that takes in a numpy array and returns a numpy array.
    """
    self.x = x
    return np.array([np.tanh(e) for e in x])

  def ReLU(self, x):
    """
    Write the code for ReLU activation function that takes in a numpy array and returns a numpy array.
    """
    self.x = x
    return np.array([max(0, e) for e in x])

  def grad_sigmoid(self):
    """
    Write the code for gradient through sigmoid activation function that takes in a numpy array and returns a numpy array.
    """
    return np.array([self.sigmoid(e) * (1 - self.sigmoid(e)) for e in self.x])

  def grad_tanh(self):
    """
    Write the code for gradient through tanh activation function that takes in a numpy array and returns a numpy array.
    """
    return np.array([1 - np.power(self.tanh(e),2) for e in self.x])

  def grad_ReLU(self):
    """
    Write the code for gradient through ReLU activation function that takes in a numpy array and returns a numpy array.
    """
    return np.array([((e > 0) * 1) for e in self.x])


class Layer():
  def __init__(self, in_units, out_units):
    np.random.seed(42)
    self.in_units = in_units
    self.out_units = out_units
    #since we are passing in entire batches, we will initialize in forward pass
    self.w = None
    self.b = None
    self.x = None  # Save the input to forward_pass in this
    self.a = None  # Save the output of forward pass in this (without activation)
    self.d_x = None  # Save the gradient w.r.t x in this
    self.d_w = None  # Save the gradient w.r.t w in this
    self.d_b = None  # Save the gradient w.r.t b in this

  def forward_pass(self, x):
    """
    Write the code for forward pass through a layer. Do not apply activation function here.
    """
    self.x = x
    sampleSize = len(self.x)
    # Weight matrix. weights for all data inputs in the batch
    self.w = [np.random.randn(self.in_units, self.out_units) for i in range(sampleSize)]  
    # bias matrix for all data inputs
    self.b = np.zeros(sampleSize, self.out_units).astype(np.float32)  
    # Compute over all units in the layer. 
    self.a = np.add(np.matmul(self.w, x), self.b) # Matrix multiply + bias (as in writeup)
    return self.a
  
  def backward_pass(self, delta):
    """
    Write the code for backward pass. This takes in gradient from its next layer as input,
    computes gradient for its weights and the delta to pass to its previous layers.
    """
    #gradient w.r.t w is curr_delta times input of current layer
    self.d_w = delta * self.x
    #gradient w.r.t x is curr_delta times w
    self.d_x = np.matmul(delta, self.w) 
    #gradient w.r.t b is alpha times detla
    self.d_b = delta

    return self.d_x

      
class Neuralnetwork():
  def __init__(self, config):
    self.layers = []
    self.x = None  # Save the input to forward_pass in this
    self.y = None  # Save the output vector of model in this
    self.targets = None  # Save the targets in forward_pass in this variable
    for i in range(len(config['layer_specs']) - 1):
      self.layers.append( Layer(config['layer_specs'][i], config['layer_specs'][i+1]))
      if i < len(config['layer_specs']) - 2:
        self.layers.append(Activation(config['activation']))  
    
  def forward_pass(self, x, targets=None):
    """
    Write the code for forward pass through all layers of the model and return loss and predictions.
    If targets == None, loss should be None. If not, then return the loss computed.
    """
    self.x = x # x contains all the samples in the batch, N * 784
    size = len(self.layers)
    weighted_sums = None
    i = 0
    while i < size:
      # this should result with an arary of weighted_sum of this layer
      weighted_sums = self.layers[i].forward_pass(self.x)     
      i+=1
      if i == size: break
      a_obj = self.layers[i]
      #update input for next layer
      self.x = a_obj.forward_pass(weighted_sums)
      i+=1

    self.y = softmax(weighted_sums) # N * 10

    loss = None if not targets else self.loss_func(self.y, self.targets)
    return loss, self.y


  # calculate the total of loss, then I
  # targets are an array that contains tha target for all the samples, and each target is one-hot encoding format
  def loss_func(self, logits, targets):
    '''
    find cross entropy loss between logits and targets
    '''

    return sum(np.matmul(targets, np.log(logits)))
    
  def backward_pass(self):
    '''
    implement the backward pass for the whole network. 
    hint - use previously built functions.
    '''
    curr_delta = np.subtract(self.t, self.y)
    for i in range(len(self.layers) - 1, 0, -1):
      layer = self.layears[i]
      act_obj = self.layers[i-1]
      d_x = layer.back_pass(curr_delta)
      act_grad = act_obj.back_pass(delta) #TODO gives g'(a) * delta?? y tho.
      curr_delta = np.multiply(d_x, act_grad)

    #learning rate
    alpha = config['learning_rate']
    #simultaneous update of weights and biases after training on all examples
    for layer in self.layers:
      if isisntance(layer, Layer): 
        layer.w += layer.w + (np.multiply(alpha, layer.d_w))
        layer.b += layer.b + (np.multiply(alpha, layer.d_b))
      

def trainer(model, X_train, y_train, X_valid, y_valid, config):
  """
  Write the code to train the network. Use values from config to set parameters
  such as L2 penalty, number of epochs, momentum, etc.
  """
  batch_size = config['batch_size']
  train_size = len(X_train)
  epoch_count = config['epoch']

  nn = Neuralnetwork(config)

  loss = None
  outputs = None
  for epoch in epoch_count:
    #iterate through each batch size
    for i in range(0, train_size, batch_size):
      #forward pass
      loss, outputs = nn.forward_pass(X_train[i:batch_size], y_train[i:batch_size])
      #back pass and update weights and biases
      nn.backward_pass() 
  
def test(model, X_test, y_test, config):
  """
  Write code to run the model on the data passed as input and return accuracy.
  """
  return accuracy
      

if __name__ == "__main__":
  train_data_fname = 'MNIST_train.pkl'
  valid_data_fname = 'MNIST_valid.pkl'
  test_data_fname = 'MNIST_test.pkl'
  
  ### Train the network ###
  model = Neuralnetwork(config)
  X_train, y_train = load_data(train_data_fname)
  X_valid, y_valid = load_data(valid_data_fname)
  X_test, y_test = load_data(test_data_fname)
  trainer(model, X_train, y_train, X_valid, y_valid, config)
  test_acc = test(model, X_test, y_test, config)

