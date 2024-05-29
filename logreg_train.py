###################################################
#        Versione senza limiti di subject         #
###################################################
# import pandas as pd
# import sys
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score
# from sklearn.model_selection import train_test_split
# from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
# from sklearn.preprocessing import LabelEncoder

# if __name__ == '__main__':
# 	data = pd.read_csv('datasets/dataset_train.csv')

# 	data_clean = data.dropna()
# 	C = data_clean.drop(['Hogwarts House', 'First Name', 'Last Name', 'Birthday', 'Best Hand'], axis=1)
# 	u = data_clean['Hogwarts House']
# 	encoder = LabelEncoder()
# 	u_encoded = encoder.fit_transform(u)
# 	selector = SelectKBest(mutual_info_classif, k=2)
# 	X_new = selector.fit_transform(C, u_encoded)
# 	mask = selector.get_support()
# 	best_features = C.columns[mask]
# 	print(f"Best features: {best_features}")

# 	data = data[['Defense Against the Dark Arts', 'Herbology', 'Hogwarts House']].dropna()
# 	X = data[['Defense Against the Dark Arts', 'Herbology']]
# 	y = data['Hogwarts House']
# 	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 	model = LogisticRegression(solver='sag', multi_class='multinomial', max_iter=1000)
# 	model.fit(X_train, y_train)
# 	test_data = pd.read_csv('datasets/dataset_test.csv')
# 	test_data = test_data.dropna(subset=['Defense Against the Dark Arts', 'Herbology'])
# 	features = test_data[['Defense Against the Dark Arts', 'Herbology']]
# 	predictions = model.predict(features)
# 	y_pred = model.predict(X_test)
# 	accuracy = accuracy_score(y_test, y_pred)
# 	print(f'Accuracy: {accuracy * 100}%')
# 	output = pd.DataFrame({
# 		'Index': test_data.index,
# 		'Hogwarts House': predictions
# 	})
# 	output.to_csv('houses.csv', index=False)

###################################################
#        Versione con limiti di subject           #
###################################################

import numpy as np
import pandas as pd
import sys
import os
import pickle
from describe import get_data
from sklearn.preprocessing import StandardScaler
learning_rate = 0.1
epochs = 1000

def initialize_parameters(n_features):
    weights = np.zeros((n_features, 1))
    bias = 0
    return weights, bias

def compute_loss(y_pred, y_true, weights, lambda_=0.01):
    m = y_true.shape[0]
    cost = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    reg_cost = (lambda_ / (2 * m)) * np.sum(np.square(weights))
    return cost + reg_cost

def update_parameters(weights, bias, dw, db, learning_rate=0.01, lambda_=0.01):
    weights -= learning_rate * (dw + lambda_ * weights)
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

def train_model(X_train, y_train, house, learning_rate=1, epochs=1000, lambda_=0.01):
    n_features = X_train.shape[1]
    weights, bias = initialize_parameters(n_features)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    y_train = (y_train == house).astype(int).reshape(-1, 1)
    
    for i in range(epochs):
        y_pred = forward_pass(X_train_scaled, weights, bias)
        loss = compute_loss(y_pred, y_train, weights, lambda_)
        dw, db = gradient_descent(X_train_scaled, y_train, y_pred)
        weights, bias = update_parameters(weights, bias, dw, db, learning_rate, lambda_)
        
        if i % 100 == 0:
            print(f"Iteration {i}, Loss: {loss}")
    
    return weights, bias, scaler

if __name__ == '__main__':
	if len(sys.argv) != 2:
		print("EHhH che ne dici di mettere il file train giusto?")
		sys.exit(1)
	dataset_path = sys.argv[1]
	if not os.path.isfile(dataset_path):
		print(f"The provided path '{dataset_path}' does not exist or is not a file.")
		sys.exit(1)

	data = get_data(dataset_path)[0][1:, :]
	data = data[:, [1, 8, 9]]
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