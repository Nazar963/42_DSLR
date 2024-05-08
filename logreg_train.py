###################################################
#        Versione senza limiti di subject         #
###################################################
# import pandas as pd
# import pickle
# import sys
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split

# #* sag vuol dire Stochastic Average Gradient descent 
# if __name__ == '__main__':
# 	if len(sys.argv) != 2:
# 		print("EHhH che ne dici di mettere il file train giusto?")
# 		sys.exit(1)
# 	dataset_path = sys.argv[1]
# 	data = pd.read_csv(dataset_path)
# 	data = data[['Defense Against the Dark Arts', 'Astronomy', 'Hogwarts House']].dropna()
# 	X = data[['Defense Against the Dark Arts', 'Astronomy']]
# 	y = data['Hogwarts House']
# 	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 	model = LogisticRegression(solver='sag', multi_class='multinomial', max_iter=1000)
# 	model.fit(X_train, y_train)
# 	with open('hogwarts_model.pkl', 'wb') as file:
# 		pickle.dump(model, file)


###################################################
#        Versione con limiti di subject            #
###################################################

import numpy as np
from describe import get_data
learning_rate = 0.01
epochs = 1000

# Initialize parameters
def initialize_parameters(n_features):
		# Initialize weights and intercept to zeros
		weights = np.zeros((n_features, 1))
		bias = 0
		return weights, bias

# Loss function (binary cross-entropy) -> log(likelihood)
def compute_loss(y_pred, y_true):
		return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# Update parameters
def update_parameters(weights, bias, dw, db):
		weights -= learning_rate * dw
		bias -= learning_rate * db
		return weights, bias

# Sigmoid function -> p(x) = 1 / e^-(b_0 + sum_n(b_n * x_n))
def sigmoid(z):
		return 1 / (1 + np.exp(-z))

# Forward pass
def forward_pass(X, weights, bias):
		z = np.dot(X, weights) + bias
		return sigmoid(z)

# Gradient descent
def gradient_descent(X:np.ndarray, y_true:np.ndarray, y_pred:np.ndarray):
		dw = np.dot(X.T, (y_pred - y_true)) / len(y_true)
		db = np.mean(y_pred - y_true)
		return dw, db

# Model training
def train_model(X_train, y_train, house):
		n_features = X_train.shape[1]
		weights, bias = initialize_parameters(n_features)
		#Convert y_train to binary
		y_train = (y_train == house).astype(int)
		y_train = y_train.reshape(-1, 1)
		for i in range(epochs):
				y_pred = forward_pass(X_train, weights, bias)
				loss = compute_loss(y_pred, y_train)
				dw, db = gradient_descent(X_train, y_train, y_pred)
				weights, bias = update_parameters(weights, bias, dw, db)
				if i % 100 == 0:
						print(f"Iteration {i}, Loss: {loss}")
		return weights, bias

if __name__ == '__main__':
	# Load the data
	data = get_data('./datasets/dataset_train.csv')[0]
	data = data[1:, :]

	# Take only the classes we consider, in this case "Defense Against the Dark Arts" and "Herbology"
	data = data[:, [1, 8, 9]]
	# Clear
	data = np.where(data == '', np.nan, data)

	# Split the data into features and target
	X = data[:, 1:]
	y = data[:, 0]
	X = X.astype(float)
	X = np.nan_to_num(X)

	houses = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']

	for house in houses:
		# Train the models
		weights, bias = train_model(X, y, house)
		print(f"Training for {house} complete")
		print(weights, bias)
		# Save the models
		np.save(f'models/{house}_weights.npy', weights)
		np.save(f'models/{house}_bias.npy', bias)



# # Model evaluation
# def evaluate_model(X_test, y_test, weight, bias):
# 		y_pred = predict(X_test, weight, bias)
# 		accuracy = np.mean(y_pred == y_test)
# 		return accuracy
# 		return (y_pred > 0.5).astype(int)
