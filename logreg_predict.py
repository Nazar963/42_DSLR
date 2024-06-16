import numpy as np
import sys
import os
import csv
import pickle
from logreg_train import forward_pass
from describe import get_data
from logreg_train import SelfMadeStandardScaler

def calculate_accuracy(true_labels, predicted_labels):
	correct_predictions = true_labels == predicted_labels
	accuracy = accuracy = np.mean(correct_predictions)
	return accuracy

def predict(X, weights, biases, scaler):
	y_pred = np.zeros((X.shape[0], len(weights)), dtype=float)
	X_test_scaled = scaler.transform(X)
	for i, (weight, bias) in enumerate(zip(weights, biases)):
		for k in range(X_test_scaled.shape[0]):
			y_pred[k][i] = forward_pass(X_test_scaled[k], weight, bias)[0]
	max_index = np.argmax(y_pred, axis=1)
	houses = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']
	predicted_houses = [houses[index] for index in max_index]
	return predicted_houses, max_index

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

	data = get_data(fileName)[0][:, [1, 8, 9, 17, 18]]
	data = np.where(data == '', np.nan, data)
	X = np.nan_to_num(data[:, 1:].astype(float))
	houses = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']

	try:
		weights, biases = load_models(houses)
		with open('scaler.pkl', 'rb') as f:
			scaler = pickle.load(f)
	except FileNotFoundError:
		print("Models not found. Please train the models first.")
		return

	print("Predicting...")
	predicted_houses, p = predict(X, weights, biases, scaler)
	print("Predictions complete.")
	return predicted_houses, p

if __name__ == '__main__':
	if len(sys.argv) != 2:
		print("Usage: python3 logreg_predict.py <filename>")
	else:
		predictions, p = main(sys.argv[1])

		# data_new = get_data('datasets-for-accuracy/true_train.csv')[0]
		# true_labels = data_new[:,1:2]
		# pred = np.reshape(predictions, (-1, 1))
		# accuracy = calculate_accuracy(true_labels, pred)
		# print(f"Accuracy: {accuracy * 100}%")
		# print("Predictions complete, writing to CSV file...")

		if predictions:
			with open('houses.csv', mode='w', newline='') as file:
				writer = csv.writer(file)
				writer.writerow(['Index', 'Hogwarts House'])
				for i, house in enumerate(predictions):
					writer.writerow([i, house])