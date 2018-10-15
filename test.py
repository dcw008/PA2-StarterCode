from neuralnet_starter import load_data

X, Y = load_data('./data/MNIST_train.pkl')
print(Y)
print(len(Y))