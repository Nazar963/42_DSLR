import numpy as np
import sys
import os
import pickle
from describe import get_data
learning_rate = 0.01
epochs = 2000
lambda_= 0.001

class SelfMadeStandardScaler:
    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
    
    def transform(self, X):
        return (X - self.mean_) / self.scale_
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

def initialize_parameters(n_features):
    weights = np.zeros((n_features, 1))
    bias = 0
    return weights, bias

def compute_loss(y_pred, y_true, weights):
    m = y_true.shape[0]
    cost = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    reg_cost = (lambda_ / (2 * m)) * np.sum(np.square(weights))
    return cost + reg_cost

def update_parameters(weights, bias, dw, db, m):
    weights -= learning_rate * (dw + (lambda_ * weights) / m)
    bias -= learning_rate * db
    return weights, bias

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def forward_pass(X, weights, bias):
    z = np.dot(X, weights) + bias
    return sigmoid(z)

def gradient_descent(X, y_true, y_pred):
    m = y_true.shape[0]
    dw = np.dot(X.T, (y_pred - y_true)) / m
    db = np.mean(y_pred - y_true)
    return dw, db

def train_model(X_train, y_train, house):
    n_features = X_train.shape[1]
    weights, bias = initialize_parameters(n_features)
    scaler = SelfMadeStandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    y_train = (y_train == house).astype(int).reshape(-1, 1)
    for i in range(epochs):
        y_pred = forward_pass(X_train_scaled, weights, bias)
        loss = compute_loss(y_pred, y_train, weights)
        dw, db = gradient_descent(X_train_scaled, y_train, y_pred)
        weights, bias = update_parameters(weights, bias, dw, db, y_train.shape[0])
        # if i % 100 == 0:
        #     print(f"Iteration {i}, Loss: {loss}")
    return weights, bias, scaler

if __name__ == '__main__':
	if len(sys.argv) != 2:
		print("Please provide the training file.")
		sys.exit(1)
	dataset_path = sys.argv[1]
	if not os.path.isfile(dataset_path):
		print(f"The provided path '{dataset_path}' does not exist or is not a file.")
		sys.exit(1)

	data = get_data(dataset_path)[0][1:, :]
	data = data[:, [1, 8, 9, 17, 18]]
	data = np.where(data == '', np.nan, data)
	X = data[:, 1:]
	X = X.astype(float)
	X = np.nan_to_num(X)
	y = data[:, 0]
	houses = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']

	if not os.path.exists('models'):
		os.mkdir('models')

	for house in houses:
		weights, bias, scaler = train_model(X, y, house)
		# print(f"Training for {house} complete")
		# print(weights, bias)
		np.save(f'models/{house}_weights.npy', weights)
		np.save(f'models/{house}_bias.npy', bias)
	with open('scaler.pkl', 'wb') as f:
		pickle.dump(scaler, f)