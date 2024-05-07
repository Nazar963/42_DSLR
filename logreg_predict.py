import pandas as pd
import pickle
import sys

def load_model(path_to_model):
    with open(path_to_model, 'rb') as file:
        model = pickle.load(file)
    return model

def predict_and_save(input_csv, model_path, output_csv):
    model = load_model(model_path)
    test_data = pd.read_csv(input_csv)
    test_data = test_data.dropna(subset=['Defense Against the Dark Arts', 'Astronomy'])
    features = test_data[['Defense Against the Dark Arts', 'Astronomy']]
    predictions = model.predict(features)
    output = pd.DataFrame({
        'Index': test_data.index,
        'Hogwarts House': predictions
    })
    output.to_csv(output_csv, index=False)

if __name__ == '__main__':
	if len(sys.argv) != 2:
		print("EHhH che ne dici di mettere il file test giusto?")
		sys.exit(1)
	dataset_path = sys.argv[1]
	predict_and_save(dataset_path, 'hogwarts_model.pkl', 'houses.csv')