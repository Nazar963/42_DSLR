# Import necessary libraries
import pandas as pd
import pickle
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

#* sag vuol dire Stochastic Average Gradient descent 
if __name__ == '__main__':
	if len(sys.argv) != 2:
		print("EHhH che ne dici di mettere il file train giusto?")
		sys.exit(1)
	dataset_path = sys.argv[1]
	data = pd.read_csv(dataset_path)
	data = data[['Defense Against the Dark Arts', 'Astronomy', 'Hogwarts House']].dropna()
	X = data[['Defense Against the Dark Arts', 'Astronomy']]
	y = data['Hogwarts House']
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
	model = LogisticRegression(solver='sag', multi_class='multinomial', max_iter=1000)
	model.fit(X_train, y_train)
	with open('hogwarts_model.pkl', 'wb') as file:
		pickle.dump(model, file)