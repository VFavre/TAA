#Favre Victor p1806797

import numpy as np
np.set_printoptions(threshold=10000,suppress=True)
import pandas as pd
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib


data = pd.read_csv('villes.csv', sep=';')
X = data.iloc[:, 1:13].values
labels = data.iloc[:, 0].values

x = StandardScaler().fit_transform(X)

pca = PCA()
principalComponents = pca.fit_transform(x)
somme = 0
ite = 0
for i in pca.explained_variance_ratio_:
    if(somme >= 0.90):
        break
    else:
        somme += i
        ite += 1
print("il faut ",ite," principale compoenent pour avoir 90% de l'info")

X_pca = principalComponents[:,:2]
        
plt.scatter(X_pca[:, 0], X_pca[:, 1])
for label, x, y in zip(labels, X_pca[:, 0], X_pca[:, 1]):
 plt.annotate(label, xy=(x, y), xytext=(-0.2, 0.2), textcoords='offset points')
plt.title("PCA villes")
plt.show()

########### Crimes ###########

data = pd.read_csv('crimes.csv', sep=';')
X = data.iloc[:, 1:8].values
labels = data.iloc[:, 0].values

x = StandardScaler().fit_transform(X)

pca = PCA()
principalComponents = pca.fit_transform(x)
somme = 0
ite = 0
for i in pca.explained_variance_ratio_:
    if(somme >= 0.90):
        break
    else:
        somme += i
        ite += 1
print("il faut ",ite," principale compoenent pour avoir 90% de l'info")

X_pca = principalComponents[:,:2]
        
import matplotlib
plt.scatter(X_pca[:, 0], X_pca[:, 1])
for label, x, y in zip(labels, X_pca[:, 0], X_pca[:, 1]):
 plt.annotate(label, xy=(x, y), xytext=(-0.2, 0.2), textcoords='offset points')
plt.title("PCA crimes")
plt.show()

######   Startups #########

data = pd.read_csv('50_Startups.csv', sep=';')
X = data.iloc[:, 1:].values
labels = data.iloc[:, 0].values

x = StandardScaler().fit_transform(X)

pca = PCA()
principalComponents = pca.fit_transform(x)
somme = 0
ite = 0
for i in pca.explained_variance_ratio_:
    if(somme >= 0.90):
        break
    else:
        somme += i
        ite += 1
print("il faut ",ite," principale component pour avoir 90% de l'information")

X_pca = principalComponents[:,:2]
        
import matplotlib
plt.scatter(X_pca[:, 0], X_pca[:, 1])
for label, x, y in zip(labels, X_pca[:, 0], X_pca[:, 1]):
 plt.annotate(label, xy=(x, y), xytext=(-0.2, 0.2), textcoords='offset points')
plt.title("PCA startups")
plt.show()
