import numpy as np
import pandas as pd
import csv
import os
import pickle
from describe import get_data
from sklearn.preprocessing import StandardScaler

if __name__ == '__main__':
    new_str = ""
    arr = []
    with open('datasets-for-accuracy/dataset_test.csv', mode='r') as file:
        for i, line in enumerate(file):
            new_str = str(i) + line[line.find(","):]
            arr.append(new_str)

    with open('prova.csv', mode='w', newline='') as outfile:
        for row in arr:
            outfile.write(row) 