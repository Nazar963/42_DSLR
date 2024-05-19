import numpy as np
import sys
import os
import csv
from logreg_train import forward_pass
from describe import get_data

def predict(X, weights, biases):
	y_pred = np.zeros((X.shape[0], len(weights)), dtype=float)
	for i, (weight, bias) in enumerate(zip(weights, biases)):
		for k in range(X.shape[0]):
			y_pred[k][i] = forward_pass(X[k], weight, bias)
	max_index = np.argmax(y_pred, axis=1)
	houses = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']
	predicted_houses = [houses[index] for index in max_index]
	return predicted_houses

def load_models(houses):
	weights, biases = [], []
	for house in houses:
		w = np.load(f'models/{house}_weights.npy')
		b = np.load(f'models/{house}_bias.npy')
		weights.append(w)
		biases.append(b)
	return weights, biases

def main(fileName):
	if not os.path.isfile(fileName):
		print("File not found.")
		return

	data = get_data(fileName)[0][1:, [1, 8, 9]]
	data = np.where(data == '', np.nan, data)
	X = np.nan_to_num(data[:, 1:].astype(float))
	houses = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']

	try:
		weights, biases = load_models(houses)
	except FileNotFoundError:
		print("Models not found. Please train the models first.")
		return

	print("Predicting...")
	predicted_houses = predict(X, weights, biases)
	print("Predictions complete.")
	return predicted_houses

if __name__ == '__main__':
	if len(sys.argv) != 2:
		print("Usage: python3 logreg_predict.py <filename>")
	else:
		predictions = main(sys.argv[1])
		if predictions:
			with open('houses.csv', mode='w', newline='') as file:
				writer = csv.writer(file)
				writer.writerow(['Index', 'Hogwarts House'])
				for i, house in enumerate(predictions):
					writer.writerow([i, house])