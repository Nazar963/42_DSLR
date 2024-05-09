
###################################################
#        Versione senza limiti di subject         #
###################################################
# import pandas as pd
# import pickle
# import sys


# def load_model(path_to_model):
#     with open(path_to_model, 'rb') as file:
#         model = pickle.load(file)
#     return model

# def predict_and_save(input_csv, model_path, output_csv):
#     model = load_model(model_path)
#     test_data = pd.read_csv(input_csv)
#     test_data = test_data.dropna(subset=['Defense Against the Dark Arts', 'Astronomy'])
#     features = test_data[['Defense Against the Dark Arts', 'Astronomy']]
#     predictions = model.predict(features)
#     output = pd.DataFrame({
#         'Index': test_data.index,
#         'Hogwarts House': predictions
#     })
#     output.to_csv(output_csv, index=False)

# if __name__ == '__main__':
# 	if len(sys.argv) != 2:
# 		print("EHhH che ne dici di mettere il file test giusto?")
# 		sys.exit(1)
# 	dataset_path = sys.argv[1]
# 	predict_and_save(dataset_path, 'hogwarts_model.pkl', 'houses.csv')

###################################################
#        Versione con limiti di subject            #
###################################################

import numpy as np
import sys
import os
import csv
from logreg_train import forward_pass
from describe import get_data

# Prediction
def predict(X, weights, biases):
	y_pred = np.zeros((X.shape[0], 4), dtype=float)
	for k in range(X.shape[0]):
		for i in range(len(weights)):
			weight = weights[i]
			bias = biases[i]
			y_pred[k][i] = forward_pass(X[k], weight, bias)
	max_index = np.argmax(y_pred, axis=1)
	houses = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']
	predicted_houses = [houses[i] for i in max_index]
	return predicted_houses


if __name__ == '__main__':

	if len(sys.argv) != 2:
		print("Usage: python3 logreg_predict.py <filename>")
		exit(1)
	filename = sys.argv[1]
	if not os.path.isfile(filename):
		print("File not found.")
		exit(1)
	data = get_data(filename)[0]
	data = data[1:, :]
	data = data[:, [1, 8, 9]]
	data = np.where(data == '', np.nan, data)
	X = data[:, 1:]
	X = X.astype(float)
	X = np.nan_to_num(X)
	# Load the models

	try:
		Gryffindor_weights = np.load('models/Gryffindor_weights.npy')
		Gryffindor_bias = np.load('models/Gryffindor_bias.npy')
		Hufflepuff_weights = np.load('models/Hufflepuff_weights.npy')
		Hufflepuff_bias = np.load('models/Hufflepuff_bias.npy')
		Ravenclaw_weights = np.load('models/Ravenclaw_weights.npy')
		Ravenclaw_bias = np.load('models/Ravenclaw_bias.npy')
		Slytherin_weights = np.load('models/Slytherin_weights.npy')
		Slytherin_bias = np.load('models/Slytherin_bias.npy')
	except FileNotFoundError:
		print("Models not found. Please train the models first.")
		exit(1)

	weights = [Gryffindor_weights, Hufflepuff_weights, Ravenclaw_weights, Slytherin_weights]
	biases = [Gryffindor_bias, Hufflepuff_bias, Ravenclaw_bias, Slytherin_bias]
	# Predict
	print("Predicting...")
	pred = predict(X, weights, biases)
	print("Predictions complete, writing to CSV file...")
	file_name = 'houses.csv'
	# Write the data to the CSV file
	with open(file_name, mode='w', newline='') as file:
		writer = csv.writer(file)
		writer.writerow(['Index', 'Hogwarts House'])  # Write the header
		for i, house in enumerate(pred):
			writer.writerow([i, house])
