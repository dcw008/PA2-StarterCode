import numpy as np
import pickle
import time

import matplotlib.pyplot as plt

config = {}
config['layer_specs'] = [784, 50, 50, 10]  # The length of list denotes number of hidden layers; each element denotes number of neurons in that layer; first element is the size of input layer, last element is the size of output layer.
config['activation'] = 'sigmoid' # Takes values 'sigmoid', 'tanh' or 'ReLU'; denotes activation function for hidden layers
config['batch_size'] = 1000  # Number of training samples per batch to be passed to network
config['epochs'] = 50  # Number of epochs to train the model
config['early_stop'] = True  # Implement early stopping or not
config['early_stop_epoch'] = 5  # Number of epochs for which validation loss increases to be counted as overfitting
config['L2_penalty'] = 0  # Regularization constant
config['momentum'] = True  # Denotes if momentum is to be applied or not
config['momentum_gamma'] = 0.9  # Denotes the constant 'gamma' in momentum expression
config['learning_rate'] = 0.009 # Learning rate of gradient descent algorithm #used for tanh and sigmoid
# config['learning_rate'] = 0.000000001 # Learning rate of gradient descent algorithm


def softmax(x):
  """
  Write the code for softmax activation function that takes in a numpy array and returns a numpy array.
  """

  total = np.sum(np.exp(x), axis=1).reshape((len(x), 1))
  return np.true_divide(np.exp(x), total)


def load_data(fname):
  """
  Write code to read the data and return it as 2 numpy arrays.
  Make sure to convert labels to one hot encoded format.
  """
  f = open(fname, 'rb')
  data = pickle.load(f, encoding='latin1')
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
      return self.ReLU(a)
  
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
    return np.true_divide(1, np.add(1, np.exp(-x)))

  def tanh(self, x):
    """
    Write the code for tanh activation function that takes in a numpy array and returns a numpy array.
    """
    self.x = x
    return np.tanh(x)


  def ReLu_v(self, e):
    return max(0, e)


  def ReLU(self, x):
    """
    Write the code for ReLU activation function that takes in a numpy array and returns a numpy array.
    """
    self.x = x
    rVec = np.vectorize(self.ReLu_v)
    return rVec(self.x)

  def grad_sigmoid(self):
    """
    Write the code for gradient through sigmoid activation function that takes in a numpy array and returns a numpy array.
    """
    return self.sigmoid(self.x) * (np.subtract(1 , self.sigmoid(self.x)))

  def grad_tanh(self):
    """
    Write the code for gradient through tanh activation function that takes in a numpy array and returns a numpy array.
    """
    return np.subtract(1, np.power(self.tanh(self.x),2))


  def grad_ReLU_v(self, e):
    return (e > 0) * 1

  def grad_ReLU(self):
    """
    Write the code for gradient through ReLU activation function that takes in a numpy array and returns a numpy array.
    """
    rgVec = np.vectorize(self.grad_ReLU_v)
    return rgVec(self.x)

class Layer():
  def __init__(self, in_units, out_units):
    np.random.seed(42)
    self.in_units = in_units
    self.out_units = out_units
    #since we are passing in entire batches, we will initialize in forward pass
    self.x = None  # Save the input to forward_pass in this
    self.a = None  # Save the output of forward pass in this (without activation)
    self.d_x = None  # Save the gradient w.r.t x in this
    self.d_w = None  # Save the gradient w.r.t w in this
    self.d_b = None  # Save the gradient w.r.t b in this
    self.w = np.random.randn(self.in_units, self.out_units) # initialize the weight
    self.b = np.zeros((1, self.out_units)).astype(np.float32)
    #below are used for momentum
    if config['momentum']:
      self.v_dw = np.zeros((self.in_units, self.out_units))
      self.v_db = np.zeros((1, self.out_units))

  def forward_pass(self, x):
    """
    Write the code for forward pass through a layer. Do not apply activation function here.
    """
    self.x = x # N * 784
    # Compute over all units in the layer.
    self.a = np.add(np.dot(self.x, self.w), self.b)
    return self.a
  
  def backward_pass(self, delta):
    """
    Write the code for backward pass. This takes in gradient from its next layer as input,
    computes gradient for its weights and the delta to pass to its previous layers.
    """
    #gradient w.r.t w is curr_delta times input of current layer
    # print(self.x.shape)
    # print(delta.shape)
    self.d_w = np.dot(np.transpose(self.x), delta)
    #gradient w.r.t x is curr_delta times w
    self.d_x = np.dot(delta, np.transpose(self.w))
    #gradient w.r.t b is alpha times delta
    self.d_b = np.mean(delta, axis=0)
    if config['momentum']:
      beta = config['momentum_gamma']
      one_minus_beta = np.subtract(1, beta)
      #v_dw = B * v_dw + (1-B)*d_w
      self.v_dw = np.add(np.multiply(beta, self.v_dw), np.multiply(one_minus_beta, self.d_w))
      #v_db = B * v_db + (1-B)*d_b
      self.v_db = np.add(np.multiply(beta, self.v_db), np.multiply(one_minus_beta, self.d_b))

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
    self.targets = targets # targets is a matrix N * 10
    size = len(self.layers)
    weighted_sums = None
    i = 0
    while i < size:
      # this should result with an arary of weighted_sum of this layer
      weighted_sums = self.layers[i].forward_pass(self.x)  # N * output_unit
      i+=1
      if i == size:
        break
      a_obj = self.layers[i]
      #update input for next layer
      self.x = a_obj.forward_pass(weighted_sums) # N * output_unit
      i+=1

    #softmax activation on the last layer
    self.y = softmax(weighted_sums) # N * 10
    #calculate loss
    loss = None if targets is None else self.loss_func(self.y, self.targets)
    return loss, self.y


  # calculate the total of loss, then I
  # targets are an array that contains tha target for all the samples, and each target is one-hot encoding format
  def loss_func(self, logits, targets):
    '''
    find cross entropy loss between logits and targets
    '''
    return -np.sum(np.multiply(np.log(logits + 0.00001), targets))


  def backward_pass(self):
    '''
    implement the backward pass for the whole network. 
    hint - use previously built functions.
    '''
    #calculate delta for the output layer
    curr_delta = np.subtract(self.targets, self.y)
    #backprop from outter hidden layer
    for i in range(len(self.layers) - 1, -1, -2):
      layer = self.layers[i]
      d_x = layer.backward_pass(curr_delta) #gives the summation part of the equation
      if i == 0: break
      act_obj = self.layers[i-1]
      curr_delta = act_obj.backward_pass(d_x) #gives gradient multiplied with d_x


def getOneHot(targets):
  labels = np.zeros((len(targets), 10))
  for i,t in enumerate(targets):
    labels[i][int(t)] = 1
  return labels


def experimentNumericalWeightApproxHelper(model, X_train, y_train, i, j, k):

  layer = model.layers[i]
  e = 0.01
  print("select w[" + str(j) + "][" + str(k) + "] from " + ("hidden" if i == -1 else str(i) )+ " layer")
  layer.w[j][k] += e
  loss_inc, outputs = model.forward_pass(X_train, y_train)
  print("loss after increasing e: ", loss_inc)
  layer.w[j][k] -= 2 * e
  loss_dec, outputs = model.forward_pass(X_train, y_train)
  print("loss after decreasing e: ", loss_dec)
  approx = -(loss_inc - loss_dec) / (2 * e)
  print("the approximation is ", approx)
  model.backward_pass()
  backprop_dw = layer.d_w[j][k]
  print("dw from back prop is ", backprop_dw)
  diff = abs(backprop_dw - approx) - abs(e * e)

  if diff <= 0:
    print("the approximation " + str(diff) + " are within o(e^2)")
  else:
    print("These approximation is not within o(e^2)")
  print("------------------------------------------")


def experimentNumericalBiasWeightApproxHelper(model, X_train, y_train, i):
  layer = model.layers[i]

  e = 0.01
  print("select " + ("hidden " if i == -1 else "output ") + "bias w")
  layer.b[0][0] += e
  print(layer.b.shape)
  loss_inc, outputs = model.forward_pass(X_train, y_train)
  print("loss after increasing e: ", loss_inc)
  layer.b[0][0] -= 2 * e
  loss_dec, outputs = model.forward_pass(X_train, y_train)
  print("loss after decreasing e: ", loss_dec)
  approx = -(loss_inc - loss_dec) / (2 * e)
  print("the approximation is ", approx)
  model.backward_pass()
  backprop_dw = layer.d_b[0]
  print("d_b from back prop is ", backprop_dw)
  diff = abs(backprop_dw - approx) - abs(e * e)

  if diff <= 0:
    print("the approximation " + str(diff) + " are within o(e^2)")
  else:
    print("These approximation is not within o(e^2)")
  print("------------------------------------------")


# For each selected weight w, first increment the weight by small value ε,
# do a forward pass for one training example, and compute the loss. This value is E(w + ε)
def experimentNumericalApproximation(model, X_train, y_train):
  experimentNumericalWeightApproxHelper(model, X_train, y_train, -1, 1, 2)
  experimentNumericalWeightApproxHelper(model, X_train, y_train, -1, 2, 3)

  experimentNumericalWeightApproxHelper(model, X_train, y_train, 0, 1, 2)
  experimentNumericalWeightApproxHelper(model, X_train, y_train, 0, 2, 3)

  experimentNumericalBiasWeightApproxHelper(model, X_train, y_train, -1)
  experimentNumericalBiasWeightApproxHelper(model, X_train, y_train, 0)


def trainer(model, X_train, y_train, X_valid, y_valid, X_test, y_test, config):
  """
  Write the code to train the network. Use values from config to set parameters
  such as L2 penalty, number of epochs, momentum, etc.
  """
  old_y_train = y_train[:]
  old_y_test = y_test[:]
  #one hot encode the labels
  y_train = getOneHot(y_train)
  y_valid = getOneHot(y_valid)
  y_test = getOneHot(y_test)

  batch_size = config['batch_size']
  train_size = len(X_train)
  epoch_count = config['epochs']
  alpha = config['learning_rate']

  min_loss = (1<<31) - 1
  best_layers = None
  all_loss = []
  best_epoch = 0

  train_acc_list = []
  test_acc_list = []

  # experimentNumericalApproximation(model, X_train[:1], y_train[:1]) #numerical approximation
  #train over epochs
  for i in range(epoch_count):
    print(i)
    #iterate through each batch size
    for j in range(0, train_size, batch_size):
      #forward pass
      loss, outputs = model.forward_pass(X_train[j:j+batch_size], y_train[j:j+batch_size])
      #back pass
      model.backward_pass()

      #stochastic gradient descent: simultaneous update of weights and biases after training on all examples
      for layer in model.layers:
        if isinstance(layer, Layer):
          if config['momentum']:
            layer.w = np.add(layer.w, np.multiply(alpha, layer.v_dw))
            layer.b = np.add(layer.b, np.multiply(alpha, layer.v_db))
          else:
            # without momentum
            layer.w = np.add(np.multiply(alpha, layer.d_w), layer.w)
            layer.b = np.add(layer.b,np.multiply(alpha,layer.d_b))
      #cross_validation
      if config['early_stop']:
        v_loss, v_outputs =  model.forward_pass(X_valid, y_valid)
    #cross-validation early-stopping, best weights are maintained by the model's layers
    if config['early_stop']:
      if v_loss <= min_loss:
        min_loss = v_loss
        best_layers = model.layers
        best_epoch = i
      else:
        model.layers = best_layers
    if config['early_stop']: all_loss.append(v_loss)

    # train_acc = test(model, X_train, old_y_train,config)
    # train_acc_list.append(train_acc)
    #
    # test_acc = test(model, X_test, old_y_test ,config)
    #
    # test_acc_list.append(test_acc)


  # X = [i+1 for i in range(epoch_count)]
  # train_acc_list = [100 * x for x in train_acc_list]
  # test_acc_list = [100 * x for x in test_acc_list]
  # plt.plot(X, train_acc_list, label='train_accuracy')
  # plt.plot(X, test_acc_list, label='test_accuracy')
  # plt.ylabel("% Accuracy")

  # print('best epoch: ',best_epoch)
  # plt.xlabel("Epoch")
  # plt.ylabel("Loss")
  # plt.title("Loss vs. Epoch")
  # plt.plot(X, all_loss)
  # plt.legend()
  # plt.show()


def test(model, X_test, y_test, config):
  """
  Write code to run the model on the data passed as input and return accuracy.
  """
  y_test = getOneHot(y_test)
  #assuming that model is the nn. forward pass to build the network with weights
  loss, outputs = model.forward_pass(X_test, y_test) #can directly pass all the inputs into forward pass
  predictions = predict(outputs)
  count = 0
  for y,p in zip(y_test,predictions):
    if np.array_equal(y,p): count += 1
  accuracy = count / len(predictions)
  # print(accuracy)
  return accuracy

      
#make predictions from the probability distribution from the neural network
def predict(probabilities):
  predictions = np.zeros((len(probabilities), 10))
  indices = np.argmax(probabilities, axis=1)
  for i, p in zip(indices, predictions):
    p[i] = 1
  return predictions




if __name__ == "__main__":
  train_data_fname = 'MNIST_train.pkl'
  valid_data_fname = 'MNIST_valid.pkl'
  test_data_fname = 'MNIST_test.pkl'
  
  # ### Train the network ###
  # config['momentum'] = False
  # model = Neuralnetwork(config)
  X_train, y_train = load_data(train_data_fname)
  #
  X_valid, y_valid = load_data(valid_data_fname)
  X_test, y_test = load_data(test_data_fname)
  # start = time.time()
  # trainer(model, X_train, y_train, X_valid, y_valid, X_test[:1000], y_test[:1000], config)
  # end = time.time()
  # print('without momentum time: ', (end-start))
  # test_acc = test(model, X_test, y_test, config)
  # train_acc = test(model, X_train, y_train, config)

  ### Train the network ###
  config['momentum'] = True
  model = Neuralnetwork(config)
  start = time.time()
  trainer(model, X_train, y_train, X_valid, y_valid, X_test, y_test, config)
  end = time.time()
  print('with momentum time: ', (end-start))
  test_acc = test(model, X_test, y_test, config)
  print(test_acc)
