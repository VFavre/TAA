import numpy as np
np.set_printoptions(threshold=10000,suppress=True)
import pandas as pd
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

data = pd.read_csv('villes.csv', sep=';')
X = data.iloc[:, 1:13].values
labels = data.iloc[:, 0].values